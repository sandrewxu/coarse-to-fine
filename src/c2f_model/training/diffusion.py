"""Masked discrete diffusion (MDLM) baseline.

Bit-faithful port of the continuous-time SUBS variant of MDLM
(Sahoo et al. 2024, ``kuleshov-group/mdlm`` master branch). Specifically:

- ``LogLinearNoise``     ↔ ``mdlm/noise_schedule.py:126-151``
- ``q_xt``               ↔ ``mdlm/diffusion.py:575-586``
- ``subs_parameterization`` ↔ ``mdlm/diffusion.py:261-277``
- ``sample_t``           ↔ ``mdlm/diffusion.py:800-808``
- ``mdlm_loss``          ↔ ``mdlm/diffusion.py:847-894`` (continuous-time, SUBS)

Intentional deviations from upstream (also documented in
``plans/please-thoroughly-examine-my-kind-sunset.md``):

1. The MASK index is a dedicated ``[MASK]`` token (id=4) added to the space
   tokenizer's reserved specials — it never appears in real docs, so
   ``x0 == mask_id`` is impossible by construction. This matches upstream
   MDLM. The AR/C2F checkpoints share the same extended vocab for
   ``check_vocab_consistency`` (``src/eval/common.py``); they simply never
   emit [MASK] at training time, treating it as an unused output class.
2. Backbone is ``Qwen3ForCausalLM`` (matched to AR/C2F) with bidirectional
   attention enabled by passing a 4D additive mask. The MDLM math is
   backbone-agnostic.
3. ``time_conditioning`` is hard-coded ``False`` — the upstream code
   supports both, the MDLM paper recommends ``False``, and skipping ``t``
   conditioning lets us use stock Qwen3 unmodified.
"""

import torch
from transformers import Trainer

NEG_INF = -1e6


class LogLinearNoise:
    """``σ(t) = -log1p(-(1-ε)·t)``, ``dσ/dt = (1-ε)/(1-(1-ε)·t)``.

    Port of ``mdlm/noise_schedule.py:126-151``. Combined with the SUBS
    parameterization, the per-token NELBO weight ``dσ/(eᵟ-1)`` simplifies
    to ``1/t`` (modulo the ``ε`` correction that bounds gradients near
    ``t→0``).
    """

    def __init__(self, eps: float = 1e-3) -> None:
        self.eps = eps

    def total_noise(self, t: torch.Tensor) -> torch.Tensor:
        return -torch.log1p(-(1.0 - self.eps) * t)

    def rate_noise(self, t: torch.Tensor) -> torch.Tensor:
        return (1.0 - self.eps) / (1.0 - (1.0 - self.eps) * t)


def make_bidirectional_4d_mask(
    input_ids: torch.Tensor, pad_id: int, dtype: torch.dtype
) -> torch.Tensor:
    """Build a 4D additive attention mask for bidirectional attention.

    Shape ``(B, 1, S, S)`` — passes through Qwen3's ``_update_causal_mask``
    untouched (HF returns 4D masks as-is) and is added directly to attention
    scores in the eager attention kernel. Zero where attention is allowed,
    ``finfo.min`` to block attention to padding (key dimension only).
    """
    bsz, seq_len = input_ids.shape
    not_pad = (input_ids != pad_id).to(dtype)  # (B, S)
    # Allow any query to attend to any non-pad key; broadcast over query dim.
    additive = (1.0 - not_pad)[:, None, None, :] * torch.finfo(dtype).min
    return additive.expand(bsz, 1, seq_len, seq_len).contiguous()


def q_xt(x0: torch.Tensor, move_chance: torch.Tensor, mask_id: int) -> torch.Tensor:
    """Independently mask each token with prob ``move_chance``.

    ``move_chance`` is per-example, shape ``(B, 1)``. Port of
    ``mdlm/diffusion.py:575-586``.
    """
    move = torch.rand(x0.shape, device=x0.device) < move_chance
    return torch.where(move, torch.full_like(x0, mask_id), x0)


def subs_parameterization(logits: torch.Tensor, xt: torch.Tensor, mask_id: int) -> torch.Tensor:
    """SUBS parameterization with carry-over unmasking.

    Port of ``mdlm/diffusion.py:261-277``. Returns log-probabilities. After
    this transform: log-prob at MASK is ``-inf`` everywhere; for *unmasked*
    positions, ``log_p[i, j, x[i,j]] == 0`` and all other entries are
    ``-inf`` (model is forced to keep the observed token); for *masked*
    positions, the model's softmax over the non-MASK vocab is unchanged.

    Modifies ``logits`` in-place (matches upstream).
    """
    logits = logits.clone()
    logits[..., mask_id] = NEG_INF
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    # Re-pin MASK to NEG_INF exactly: the finite sentinel leaks through
    # log-softmax as `NEG_INF - logsumexp(row)`, which is slightly more
    # negative than NEG_INF and breaks the "MASK is -inf everywhere" invariant.
    log_probs[..., mask_id] = NEG_INF
    unmasked = xt != mask_id
    log_probs[unmasked] = NEG_INF
    log_probs[unmasked, xt[unmasked]] = 0.0
    return log_probs


def sample_t(
    bsz: int, device: torch.device, eps_t: float = 1e-3, antithetic: bool = True
) -> torch.Tensor:
    """``t ~ (1-ε_t)·U(0,1) + ε_t``, optionally antithetic.

    Port of ``mdlm/diffusion.py:800-808``. Antithetic sampling halves
    variance at no compute cost.
    """
    u = torch.rand(bsz, device=device)
    if antithetic:
        offset = torch.arange(bsz, device=device, dtype=u.dtype) / bsz
        u = (u / bsz + offset) % 1.0
    return (1.0 - eps_t) * u + eps_t


def mdlm_loss(
    model,
    x0: torch.Tensor,
    loss_mask: torch.Tensor,
    mask_id: int,
    pad_id: int,
    schedule: LogLinearNoise,
    *,
    eps_t: float = 1e-3,
    antithetic: bool = True,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute one Monte-Carlo sample of the continuous-time SUBS NELBO.

    Args:
        model: ``Qwen3ForCausalLM``-style module returning ``.logits``.
        x0: clean input ids ``(B, S)``.
        loss_mask: 1.0 at positions to score, 0.0 elsewhere ``(B, S)``.
            Typically excludes BOS and padding.
        mask_id: vocab id of the absorbing/MASK token (``[UNK]`` here).
        pad_id: vocab id of padding (used to build the attention mask).
        schedule: ``LogLinearNoise`` (or any object with ``total_noise``,
            ``rate_noise``).
        eps_t: lower bound on sampled ``t`` to keep ``1/t`` weight bounded.
        antithetic: enable antithetic ``t`` sampling.
        reduction: ``"mean"`` returns a scalar (mean over loss_mask=1
            positions). ``"per_doc"`` returns ``(B,)`` summed per document
            (used by the eval path for per-doc bootstrap).

    Returns:
        Scalar loss (``"mean"``) or per-document NLL ``(B,)`` (``"per_doc"``).
    """
    bsz, _ = x0.shape
    device = x0.device

    t = sample_t(bsz, device, eps_t=eps_t, antithetic=antithetic)
    sigma = schedule.total_noise(t)
    dsigma = schedule.rate_noise(t)
    move_chance = (1.0 - torch.exp(-sigma)).unsqueeze(-1)  # (B, 1)

    xt = q_xt(x0, move_chance, mask_id)

    attn_4d = make_bidirectional_4d_mask(x0, pad_id, dtype=torch.float32)
    out = model(input_ids=xt, attention_mask=attn_4d)
    log_probs = subs_parameterization(out.logits.float(), xt, mask_id)

    log_p_x0 = log_probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)  # (B, S)
    weight = (dsigma / torch.expm1(sigma)).unsqueeze(-1)  # (B, 1)
    weighted_nll = -log_p_x0 * weight  # (B, S)
    weighted_nll = weighted_nll * loss_mask

    if reduction == "per_doc":
        return weighted_nll.sum(dim=-1)
    denom = loss_mask.sum().clamp(min=1.0)
    return weighted_nll.sum() / denom


class DiffusionTrainer(Trainer):
    """HF ``Trainer`` with the MDLM ``compute_loss`` override.

    Reuses ``ARDataset`` items unchanged: ``input_ids`` is ``x0``, ``labels``
    encodes the loss mask (``-100`` for BOS / padding, real id elsewhere).
    """

    def __init__(
        self,
        *args,
        mask_id: int,
        pad_id: int,
        eps_t: float = 1e-3,
        noise_eps: float = 1e-3,
        antithetic: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._mask_id = mask_id
        self._pad_id = pad_id
        self._eps_t = eps_t
        self._antithetic = antithetic
        self._schedule = LogLinearNoise(eps=noise_eps)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        x0 = inputs["input_ids"]
        labels = inputs["labels"]
        loss_mask = (labels != -100).to(x0.device, dtype=torch.float32)
        loss = mdlm_loss(
            model,
            x0,
            loss_mask,
            mask_id=self._mask_id,
            pad_id=self._pad_id,
            schedule=self._schedule,
            eps_t=self._eps_t,
            antithetic=self._antithetic,
        )
        if return_outputs:
            # HF Trainer's prediction_step does ``outputs[1:]`` and then
            # iterates the result, so we must return *something* indexable.
            # An empty dict satisfies the contract without leaking the
            # logits we already discarded in ``mdlm_loss``.
            return loss, {}
        return loss


__all__ = [
    "NEG_INF",
    "DiffusionTrainer",
    "LogLinearNoise",
    "make_bidirectional_4d_mask",
    "mdlm_loss",
    "q_xt",
    "sample_t",
    "subs_parameterization",
]
