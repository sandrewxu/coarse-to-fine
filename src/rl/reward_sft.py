"""Reward manager for Phase A (GRPO on ``q_φ``, ``p_θ`` frozen).

ELBO setup — two interchangeable paths
--------------------------------------

The ELBO we want to optimize, per the project's simplified derivation, is

    L(θ, φ; x) = E_{z ~ q_φ(z|x)}[ log p_θ(x, z) - log q_φ(z|x) ]     (I)
               = E_{z ~ q_φ(z|x)}[ log p_θ(x, z) ] + H(q_φ)

There are two mathematically equivalent ways to implement (I) in veRL. Each
has a knob in the experiment YAML.

**Option B (preferred, default)** — use veRL's built-in entropy bonus:

    reward(z)           = log p_θ(x, z)  + format_bonus
    actor.entropy_coeff = rl.sft_rl.entropy_coeff  (default 1.0)

``dp_actor.py:650`` subtracts ``entropy_coeff · H(q_φ)`` from the policy loss,
equivalently adding it to the maximization objective. At ``entropy_coeff = 1``
this *is* ``L`` — using the full-distribution per-token entropy (lower
variance than a single-sample MC). No extra model load, no extra forward.

**Option A (opt-in debug path)** — explicit reward-side ``-log q_ref(z|x)``:

    reward(z)           = log p_θ(x, z) - α · log q_ref(z|x) + format_bonus
    actor.kl_loss_coef  = β · KL(q_φ ∥ q_ref)

With ``α = β = 1`` this is also ``L`` (the ``log q_ref`` terms cancel), but
uses a *stale* ``q_ref`` (the initial SFT) for the entropy term and costs an
extra frozen 4B model in the reward manager. Enable by setting
``rl.sft_rl.ref_nll_coef > 0``; the reference model is loaded lazily.
Defaults to OFF (``ref_nll_coef = 0``) so Option B is the single active path
unless you explicitly turn Option A on for an ablation.

**Why both**: Option B is cheaper and lower variance; Option A exposes
``log q_ref`` in ``reward_extra_info`` for analysis, and lets you decouple
``α`` from ``β`` to study deviations from the exact ELBO. For production
training, leave Option A off.

Deviations from the ELBO
------------------------
* ``format_bonus``: **reward shaping**, not part of the ELBO. It biases
  ``q_φ`` toward producing the ``z_n:`` layered format and is constant on
  valid responses. Once ``q_φ`` is emitting valid format reliably, the
  bonus is a constant offset (gradient w.r.t. φ is zero) and doesn't
  perturb the ELBO direction. In early training it *does* perturb the
  gradient; treat it as a warm-start convenience.

* ``/num_tokens`` normalization: **removed**. The PDF derivation uses the
  un-normalized sequence log-prob ``log p(x, z)``. Per-token mean would
  reweight docs by ``1/num_tokens``, which matters when ``num_tokens``
  varies. In this project the response length is roughly constant
  (strictly-verified z_n layout), but the un-normalized form exactly
  matches the math.

Reference model (Option A only)
-------------------------------
``q_ref`` is the initial SFT checkpoint (the actor's starting point for RL).
That's the natural reference for the KL-to-reference term and, by the
decomposition above, also the natural q_ref for the reward's -log q_ref term.
Loaded in eval mode, bf16, frozen; one extra forward per reward call.
Skipped entirely when ``ref_nll_coef == 0`` (the default).
"""

from typing import Any

import torch
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase

from src.common.logging import get_logger
from src.rl.common import (
    build_c2f_input,
    load_c2f_components,
    load_exp_config,
    parse_layers,
)
from src.verification import verify as verify_layers

log = get_logger(__name__)


class C2FRewardManager(RewardManagerBase):
    """GRPO reward manager implementing the ELBO reward for Phase A."""

    def __init__(
        self,
        config,
        tokenizer,
        compute_score=None,
        **kwargs,
    ) -> None:
        super().__init__(config, tokenizer, compute_score)

        exp_config = load_exp_config()
        self.components = load_c2f_components(
            exp_config, sft_tokenizer=tokenizer, c2f_section_key="sft_rl"
        )

        # ── Freeze the C2F decoder ──────────────────────────────────────────
        c2f_model = self.components.c2f_model
        c2f_model.eval()
        for p in c2f_model.parameters():
            p.requires_grad_(False)
        self.device = next(c2f_model.parameters()).device

        # ── Phase-specific config ───────────────────────────────────────────
        rl_cfg = exp_config.get("rl", {}).get("sft_rl", {})
        self.format_bonus_weight: float = float(rl_cfg.get("format_bonus_weight", 0.1))
        self.ref_nll_coef: float = float(rl_cfg.get("ref_nll_coef", 1.0))

        # ── Reference model for -log q_ref(z|x) term ────────────────────────
        self.ref_model = None
        if self.ref_nll_coef != 0.0:
            ref_path = rl_cfg.get("ref_model_path") or rl_cfg.get("model_path")
            if not ref_path:
                raise RuntimeError(
                    "ref_nll_coef != 0 but neither rl.sft_rl.ref_model_path nor "
                    "rl.sft_rl.model_path is set in the experiment YAML. The "
                    "reward can't compute -log q_ref(z|x) without a reference model."
                )
            from transformers import AutoModelForCausalLM

            log.info(
                "Loading frozen reference SFT model from %s for ELBO reward (ref_nll_coef=%.3f)...",
                ref_path,
                self.ref_nll_coef,
            )
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                str(ref_path), trust_remote_code=True, dtype=torch.bfloat16
            )
            self.ref_model.to(self.device)
            self.ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad_(False)
        else:
            log.info(
                "ref_nll_coef=0.0 — skipping reference-model load. Reward degenerates "
                "to log p_θ(x,z) + format_bonus (pre-ELBO behavior)."
            )

    # ── Reward components ────────────────────────────────────────────────────

    def _format_bonus(self, response: str) -> float:
        """Shaping term — NOT part of the ELBO. See module docstring."""
        from src.rl.common import strip_think

        cleaned = strip_think(response)
        result = verify_layers(
            cleaned,
            self.components.word_count_constraints,
            strict_word_count=self.components.strict_word_count,
        )
        if result.passed:
            return self.format_bonus_weight

        if result.layers:
            correct = sum(
                1
                for layer in result.layers
                if layer.word_count
                == self.components.word_count_constraints.get(layer.layer_name, -1)
            )
            return self.format_bonus_weight * (correct / len(result.layers))
        return 0.0

    @torch.no_grad()
    def _log_p_c2f(self, layer_contents: list[str], prompt: str) -> float:
        """Compute ``log p_θ(x, z)`` — the un-normalized sequence log-prob."""
        c = self.components
        input_ids, labels = build_c2f_input(
            layer_contents,
            prompt,
            scale_lengths=c.scale_lengths,
            word_boundaries=c.word_boundaries,
            space_tokenizer=c.space_tokenizer,
            bos_id=c.bos_id,
            pad_id=c.pad_id,
            seq_len=c.seq_len,
            label_strategy="sft",
        )
        input_ids = input_ids.unsqueeze(0).to(self.device)
        labels = labels.unsqueeze(0).to(self.device)

        outputs = c.c2f_model(input_ids=input_ids, labels=labels)
        loss: float = outputs.loss.item()
        num_unmasked: int = int((labels != -100).sum().item())
        # HF CE returns mean loss; multiply back by num_unmasked to get the
        # summed log-prob as required by the ELBO (see module docstring).
        return -loss * num_unmasked

    @torch.no_grad()
    def _log_q_ref_single(self, response: str, prompt: str) -> float:
        """``log q_ref(z | x)`` summed over response tokens only.

        Teacher-forces the chat-templated ``[user=x, assistant=z]`` sequence
        through the frozen reference SFT model and sums log-probs of the
        assistant-response tokens (prompt tokens masked out via labels=-100),
        matching ``_sft_nll_per_doc`` in ``src/eval/bound.py``.
        """
        if self.ref_model is None:
            return 0.0

        tok = self.components.sft_tokenizer
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        full_text = tok.apply_chat_template(messages, tokenize=False)
        full_ids = tok(full_text, truncation=True, max_length=4096)["input_ids"]

        prompt_text = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = tok(prompt_text, truncation=True, max_length=4096)["input_ids"]

        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]

        input_ids_t = torch.tensor([full_ids], dtype=torch.long, device=self.device)
        labels_t = torch.tensor([labels], dtype=torch.long, device=self.device)
        attn = torch.ones_like(input_ids_t)

        out = self.ref_model(input_ids=input_ids_t, attention_mask=attn)
        shift_logits = out.logits[:, :-1, :]
        shift_labels = labels_t[:, 1:]
        per_tok = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).view(shift_labels.shape)
        # per_tok is -log q_ref at each response position; zero at prompt (via
        # ignore_index). Summed → NLL_ref(z|x) (non-negative). Return log-prob
        # (negative) by negating.
        nll_ref = per_tok.sum().item()
        return -nll_ref

    # ── veRL interface ───────────────────────────────────────────────────────

    def __call__(self, data: Any, return_dict: bool = False) -> Any:
        """Score a batch of GRPO rollouts.

        Sets ``data.batch['token_level_scores']`` to a tensor of shape
        ``(B, T_resp)`` with the scalar reward placed at the last non-pad
        response token of each sample.
        """
        c = self.components
        batch_size: int = data.batch["responses"].shape[0]
        response_len: int = data.batch["responses"].shape[1]

        response_strs: list[str] = c.sft_tokenizer.batch_decode(
            data.batch["responses"], skip_special_tokens=True
        )
        ground_truths = data.non_tensor_batch.get("ground_truth", [""] * batch_size)
        if hasattr(ground_truths, "tolist"):
            ground_truths = ground_truths.tolist()

        reward_tensor = torch.zeros(batch_size, response_len, dtype=torch.float32)
        pad_id = getattr(c.sft_tokenizer, "pad_token_id", None)

        for i, (response, prompt) in enumerate(zip(response_strs, ground_truths, strict=False)):
            layer_contents = parse_layers(
                response, c.word_count_constraints, strict=c.strict_word_count
            )
            if layer_contents is None:
                continue  # malformed → zero reward

            log_p = self._log_p_c2f(layer_contents, str(prompt))
            log_q_ref = self._log_q_ref_single(response, str(prompt))
            bonus = self._format_bonus(response)

            # ELBO reward: log p(x,z) - α·log q_ref(z|x) + format_bonus.
            # α=1 (default) with kl_loss_coef=1 in veRL gives the exact ELBO
            # gradient on φ. See module docstring.
            reward = log_p - self.ref_nll_coef * log_q_ref + bonus

            response_ids = data.batch["responses"][i]
            if pad_id is not None:
                non_pad = (response_ids != pad_id).nonzero(as_tuple=True)[0]
            else:
                non_pad = torch.arange(response_len, device=response_ids.device)
            last_pos = int(non_pad[-1].item()) if len(non_pad) > 0 else response_len - 1
            reward_tensor[i, last_pos] = reward

        data.batch["token_level_scores"] = reward_tensor

        if return_dict:
            return {"reward_tensor": reward_tensor}
        return data

    async def run_single(self, data) -> dict:
        c = self.components
        data_item = data[0]
        response_ids = data_item.batch["responses"]
        response_str = c.sft_tokenizer.decode(response_ids.tolist(), skip_special_tokens=True)

        nt = data_item.non_tensor_batch
        rm = nt.get("reward_model")
        gt = rm.get("ground_truth", "") if isinstance(rm, dict) else nt.get("ground_truth", "")

        if not getattr(type(self), "_debug_dumped", False):
            type(self)._debug_dumped = True
            log.debug(
                "first sample: response (len=%d) %r | ground_truth (len=%d) %r",
                len(response_str),
                response_str[:800],
                len(str(gt)),
                str(gt)[:300],
            )

        layer_contents = parse_layers(
            response_str, c.word_count_constraints, strict=c.strict_word_count
        )
        if layer_contents is None:
            return {
                "reward_score": 0.0,
                "reward_extra_info": {
                    "log_p": 0.0,
                    "log_q_ref": 0.0,
                    "format_bonus": 0.0,
                    "malformed": 1.0,
                },
            }

        log_p = self._log_p_c2f(layer_contents, str(gt))
        log_q_ref = self._log_q_ref_single(response_str, str(gt))
        bonus = self._format_bonus(response_str)
        reward = log_p - self.ref_nll_coef * log_q_ref + bonus
        return {
            "reward_score": float(reward),
            "reward_extra_info": {
                "log_p": float(log_p),
                "log_q_ref": float(log_q_ref),
                "format_bonus": float(bonus),
                "malformed": 0.0,
            },
        }
