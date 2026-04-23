"""ELBO / IWAE-K upper bound on ``-log p(x)`` for C2F, using the SFT as ``q_φ``.

The C2F decoder models ``p(x, z) = p(z_4) p(z_3|z_4) p(z_2|z_3) p(z_1|z_2) p(x|z_1)``.
The SFT model was trained to approximate ``q(z|x)``: it takes the text as a
user message and emits the multi-scale summary as the assistant response.
Together they give an IWAE-K upper bound on ``-log p(x)``:

    log p(x) >= E_{z_1..K ~ q}[ log (1/K) sum_k p(x, z_k) / q(z_k | x) ]

Equivalently, as an upper bound on ``-log p(x)`` per text word:

    -log p(x) / T  <=  (log K - logsumexp_k(log p(x, z_k) - log q(z_k|x))) / T

where ``T = text_word_count`` (32 by default). This is directly comparable to
the AR baseline's exact NLL per text word — AR reports ``-log p(x) / T``
without latents, and a tight C2F bound below AR would mean the C2F model
assigns higher probability to ``x``.

K modes
-------
* ``K == 1`` — gold-z fast path. The ``z`` in the test parquet was itself
  sampled from ``q_φ`` at generation time, so it's a valid single-sample ELBO.
  No online sampling required.
* ``K > 1`` — draws ``K`` fresh ``z`` samples per doc from ``q_φ`` via
  :mod:`src.eval.q_sampler`, rejection-filters on the per-scale word-count
  constraints, then combines via ``logsumexp``. The bound is monotonically
  tighter in ``K`` (Burda et al. 2016).

Notes
-----
* The two models use different tokenizations (space tokenizer for C2F, Qwen3
  BPE for SFT). Both assign a well-defined probability to the *text* of the
  ``z`` sample; the ELBO is over strings, so mixed tokenization is fine.
* SFT's NLL is summed only over the assistant-response tokens (prompt tokens
  are ignored via ``-100`` labels, matching the training-time loss mask).
* C2F's joint NLL is summed over all content tokens (z_4 + z_3 + z_2 + z_1 +
  text) exactly as :func:`src.eval.c2f.eval_c2f` does.
* Rejection sampling introduces a correction term ``log q(valid | x)`` that we
  drop as negligible when the validity rate is >= 90 %. The sampler warns
  when that assumption breaks; see :mod:`src.eval.q_sampler` for details.
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.common.logging import get_logger
from src.eval.common import bootstrap_ci, check_vocab_consistency, load_space_tokenizer

log = get_logger(__name__)


def _reconstruct_prompt_response_from_c2f(
    texts: list[str],
    *,
    word_count_constraints: dict[str, int],
    text_word_count: int,
    layer_names: tuple[str, ...],
    compute_word_boundaries,
) -> tuple[list[str], list[str]]:
    """Invert ``flatten_for_c2f`` so we can score ``q(z|x)`` on a c2f-format parquet.

    Each row's ``text`` is the whitespace concatenation of the four latent
    scales (``z_4 z_3 z_2 z_1``) followed by the ``text_word_count``-word
    document. The word_count_constraints fix the per-scale widths exactly,
    so we can split by position and re-emit the standard SFT ``z_n:`` format
    string that the SFT model was trained to produce.

    Word content is recovered losslessly; exact whitespace of the original
    SFT response is not (we emit newline-separated ``z_n: <words>`` lines).
    This is acceptable because the SFT model is robust to whitespace-level
    variation at a BPE level and because the ELBO is valid for any ``q`` as
    long as we score the same string we condition on.
    """
    boundaries = compute_word_boundaries(word_count_constraints, text_word_count)
    *latent_boundaries, (text_start, text_end) = boundaries
    prompts: list[str] = []
    responses: list[str] = []
    for text in texts:
        words = text.split()
        prompts.append(" ".join(words[text_start:text_end]))
        lines = [
            f"{name}: {' '.join(words[start:end])}"
            for name, (start, end) in zip(layer_names, latent_boundaries, strict=True)
        ]
        responses.append("\n".join(lines))
    return prompts, responses


def _scale_ranges(scale_lengths: list[int]) -> list[tuple[int, int]]:
    """``(start, end)`` token slices per scale, relative to the BOS-prefixed sequence."""
    ranges: list[tuple[int, int]] = []
    pos = 1
    for length in scale_lengths:
        ranges.append((pos, pos + length))
        pos += length
    return ranges


@torch.no_grad()
def _c2f_joint_nll_per_doc(
    model,
    dataset,
    device: torch.device,
    batch_size: int,
    *,
    scale_lengths: list[int] | None = None,
    scale_names: tuple[str, ...] = ("z_4", "z_3", "z_2", "z_1", "text"),
) -> tuple[np.ndarray, dict[str, np.ndarray] | None]:
    """Return ``(joint_nll_per_doc, per_scale_nll_per_doc)`` under the C2F model.

    Mirrors the per-doc accounting in :func:`src.eval.c2f.eval_c2f`. If
    ``scale_lengths`` is provided, also returns per-scale per-doc NLL for
    diagnostics; otherwise the second return value is ``None``.
    """
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    mask_type = model.config.mask_type

    per_doc_nll: list[float] = []
    per_scale_nll: dict[str, list[float]] | None = None
    scale_slices: list[tuple[str, slice]] | None = None
    if scale_lengths is not None:
        per_scale_nll = {name: [] for name in scale_names}
        ranges = _scale_ranges(scale_lengths)
        scale_slices = []
        for name, (start, end) in zip(scale_names, ranges, strict=False):
            s = slice(start - 1, end - 1) if mask_type == "causal" else slice(start, end)
            scale_slices.append((name, s))

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        out = model(input_ids=input_ids)
        logits = out.logits
        if mask_type == "causal":
            logits = logits[:, :-1, :]
            lab = labels[:, 1:]
        else:
            lab = labels
        B, T, V = logits.shape
        per_tok = loss_fn(logits.reshape(-1, V), lab.reshape(-1)).view(B, T)
        mask = (lab != -100).float()
        doc_nll = (per_tok * mask).sum(dim=1).cpu().numpy()
        per_doc_nll.extend(doc_nll.tolist())
        if scale_slices is not None:
            assert per_scale_nll is not None
            for name, s in scale_slices:
                seg_nll = (per_tok[:, s] * mask[:, s]).sum(dim=1).cpu().numpy()
                per_scale_nll[name].extend(seg_nll.tolist())

    joint = np.asarray(per_doc_nll, dtype=np.float64)
    if per_scale_nll is None:
        return joint, None
    per_scale_arr = {
        name: np.asarray(vals, dtype=np.float64) for name, vals in per_scale_nll.items()
    }
    return joint, per_scale_arr


@torch.no_grad()
def _sft_nll_per_doc(
    model,
    tokenizer,
    prompts: list[str],
    responses: list[str],
    device: torch.device,
    batch_size: int,
    max_length: int = 4096,
) -> np.ndarray:
    """Return ``-log q(z|x)`` per document under the SFT model.

    Applies the Qwen3 chat template exactly as ``src.sft.train._tokenize`` does:
    tokenize ``[user=x, assistant=z]`` as the input, use ``add_generation_prompt``
    to find the prompt-boundary, mask prompt tokens with ``-100`` so CE only
    scores the response.
    """
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    # Pre-tokenize all docs (pure CPU work, fast). Keep per-doc lengths so we can
    # right-pad per batch.
    all_input_ids: list[list[int]] = []
    all_labels: list[list[int]] = []
    for prompt, response in zip(prompts, responses, strict=True):
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        full_text = tokenizer.apply_chat_template(messages, tokenize=False)
        full_ids = tokenizer(full_text, truncation=True, max_length=max_length)["input_ids"]

        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = tokenizer(prompt_text, truncation=True, max_length=max_length)["input_ids"]

        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
        # If truncation clipped the response entirely (rare), labels can end up
        # all -100; loss_fn handles that correctly (contributes 0).
        all_input_ids.append(full_ids)
        all_labels.append(labels)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    per_doc_nll = np.empty(len(prompts), dtype=np.float64)

    for start in range(0, len(prompts), batch_size):
        end = min(start + batch_size, len(prompts))
        B = end - start
        max_len = max(len(all_input_ids[i]) for i in range(start, end))
        input_ids = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
        attn_mask = torch.zeros((B, max_len), dtype=torch.long, device=device)
        labels = torch.full((B, max_len), -100, dtype=torch.long, device=device)
        for b, doc_idx in enumerate(range(start, end)):
            ids = all_input_ids[doc_idx]
            lab = all_labels[doc_idx]
            input_ids[b, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
            attn_mask[b, : len(ids)] = 1
            labels[b, : len(lab)] = torch.tensor(lab, dtype=torch.long, device=device)

        out = model(input_ids=input_ids, attention_mask=attn_mask)
        # HF causal LMs predict input_ids[i+1] from logits[i]; shift both sides.
        shift_logits = out.logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        _, T, V = shift_logits.shape
        per_tok = loss_fn(shift_logits.reshape(-1, V), shift_labels.reshape(-1)).view(B, T)
        # ignore_index zeros out -100 contributions; plain sum gives per-doc NLL.
        doc_nll = per_tok.sum(dim=1).cpu().numpy()
        per_doc_nll[start:end] = doc_nll
    return per_doc_nll


def _load_prompts_and_responses(
    test: Path,
    config: dict[str, Any],
    text_word_count: int,
) -> tuple[str, list[str], list[str]]:
    """Load ``(dataset_format, prompts, responses)`` for either parquet format."""
    import pyarrow.parquet as pq

    from src.rl.common import LayerNames, compute_word_boundaries

    cols = pq.read_schema(str(test)).names
    if "prompt" in cols and "response" in cols:
        table = pq.read_table(str(test), columns=["prompt", "response"])
        return "sft", table.column("prompt").to_pylist(), table.column("response").to_pylist()
    if "text" in cols:
        texts = pq.read_table(str(test), columns=["text"]).column("text").to_pylist()
        prompts, responses = _reconstruct_prompt_response_from_c2f(
            texts,
            word_count_constraints=config["word_count_constraints"],
            text_word_count=text_word_count,
            layer_names=LayerNames,
            compute_word_boundaries=compute_word_boundaries,
        )
        return "c2f", prompts, responses
    raise ValueError(
        f"eval_c2f_bound requires a parquet with either (prompt + response) "
        f"or (text) columns; got {cols} at {test}."
    )


class _SampledC2FInputs(torch.utils.data.Dataset):
    """Tensor-backed dataset mirroring :class:`C2FDataset`'s output shape.

    Used by the IWAE-K path to score C2F joint NLL on programmatically
    built ``(prompt, layers)`` samples — ``_c2f_joint_nll_per_doc`` iterates
    a ``DataLoader`` over this, so the item dict matches C2FDataset.
    """

    def __init__(self, input_ids: list[torch.Tensor], labels: list[torch.Tensor]) -> None:
        self.input_ids = input_ids
        self.labels = labels

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"input_ids": self.input_ids[idx], "labels": self.labels[idx]}


def _build_sampled_c2f_dataset(
    space_tokenizer,
    config: dict[str, Any],
    prompts: list[str],
    layers_per_item: list[list[str]],
    text_word_count: int,
) -> _SampledC2FInputs:
    """Wrap ``(prompts, layers)`` pairs into a C2F-dataset-shaped tensor dataset.

    ``layers_per_item[i]`` is ``[z_4, z_3, z_2, z_1]`` content strings.
    Uses ``build_c2f_input`` with ``label_strategy="joint"`` to match
    ``C2FDataset._build_labels`` exactly (so joint NLL matches the training
    objective for sampled z's too).
    """
    import math as _math

    from src.rl.common import build_c2f_input, compute_word_boundaries

    scale_lengths = config["scale_lengths"]
    wcc = config["word_count_constraints"]
    word_boundaries = compute_word_boundaries(wcc, text_word_count)
    seq_len = 2 ** _math.ceil(_math.log2(1 + sum(scale_lengths)))
    bos_id = space_tokenizer.bos_token_id or space_tokenizer.eos_token_id
    pad_id = space_tokenizer.pad_token_id or space_tokenizer.eos_token_id

    all_input_ids: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    for prompt, layers in zip(prompts, layers_per_item, strict=True):
        input_ids, labels = build_c2f_input(
            layer_contents=layers,
            prompt=prompt,
            scale_lengths=scale_lengths,
            word_boundaries=word_boundaries,
            space_tokenizer=space_tokenizer,
            bos_id=bos_id,
            pad_id=pad_id,
            seq_len=seq_len,
            label_strategy="joint",
        )
        all_input_ids.append(input_ids)
        all_labels.append(labels)

    return _SampledC2FInputs(all_input_ids, all_labels)


def _iwae_logsumexp_bound(
    joint_nll_p: np.ndarray,  # (B, K)
    nll_q: np.ndarray,  # (B, K)
    K: int,
) -> np.ndarray:
    """Combine per-sample ``(joint_nll_p, nll_q)`` into the IWAE-K per-doc bound.

    log w_k   = log p(x, z_k) - log q(z_k|x) = nll_q - joint_nll_p
    log p^K   = logsumexp_k(log w_k) - log K
    bound(x)  = -log p^K                     (upper bound on -log p(x))

    Done in float64 via scipy-style logsumexp (we implement in numpy to
    avoid pulling in scipy just for this).
    """
    log_w = nll_q.astype(np.float64) - joint_nll_p.astype(np.float64)  # (B, K)
    m = log_w.max(axis=1, keepdims=True)
    log_phat_K = m.squeeze(-1) + np.log(np.exp(log_w - m).sum(axis=1)) - np.log(K)
    return -log_phat_K


@torch.no_grad()
def eval_c2f_bound(
    *,
    c2f_ckpt: Path,
    sft_ckpt: Path,
    test: Path,
    config: dict[str, Any],
    limit: int | None = None,
    batch_size: int = 8,
    sft_batch_size: int = 4,
    tokenizer_dir: Path | None = None,
    text_word_count: int | None = None,
    K: int = 1,
    q_temperature: float = 1.0,
    q_max_new_tokens: int = 128,
) -> dict[str, Any]:
    """IWAE-K ELBO upper bound on ``-log p(x) / text_word_count`` for C2F.

    Args:
        c2f_ckpt: C2F decoder checkpoint directory.
        sft_ckpt: SFT checkpoint directory (used as ``q_φ``).
        test: SFT-format parquet with ``prompt`` (text) and ``response``
            (multi-scale z_n lines) columns, OR a c2f-flat parquet with a
            ``text`` column. For K=1, the gold ``z`` in the parquet is used
            as the single sample from ``q_φ``. For K>1, prompts are re-read
            and fresh samples are drawn from ``q_φ``.
        config: Experiment config (for scale_lengths, word_count_constraints,
            text_word_count, tokenizer_dir).
        limit: Cap number of docs scored.
        batch_size: C2F forward-pass batch size.
        sft_batch_size: SFT forward-pass batch size (default lower because SFT
            is Qwen3-4B and sequences are longer after BPE tokenization).
        tokenizer_dir: Override the C2F (space) tokenizer dir.
        text_word_count: Denominator for the per-text-word bound. Defaults to
            ``config["text_word_count"]`` (32 in the default setup).
        K: IWAE sample count. K=1 uses the gold ``z`` in the parquet for a
            single-sample ELBO; K>1 draws K fresh samples per doc from
            ``q_φ``. Larger K gives a monotonically tighter bound.
        q_temperature: Sampling temperature for ``q_φ``. Must stay at 1.0 for
            a mathematically valid bound (T≠1 samples from ``q^{1/T}``).
        q_max_new_tokens: Max generation length for ``q_φ`` responses. The
            default (128) comfortably fits a z_4..z_1 block for the standard
            2/4/8/16 layout; raise for larger scales.

    Returns:
        Dict with ``model_kind="c2f_bound"``, ``K``, plus per-doc bound
        rows, per-scale p-side contributions, and a bootstrap CI.
    """
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")

    if K == 1:
        return _eval_bound_gold_z(
            c2f_ckpt=c2f_ckpt,
            sft_ckpt=sft_ckpt,
            test=test,
            config=config,
            limit=limit,
            batch_size=batch_size,
            sft_batch_size=sft_batch_size,
            tokenizer_dir=tokenizer_dir,
            text_word_count=text_word_count,
        )

    return _eval_bound_iwae_k(
        c2f_ckpt=c2f_ckpt,
        sft_ckpt=sft_ckpt,
        test=test,
        config=config,
        limit=limit,
        batch_size=batch_size,
        sft_batch_size=sft_batch_size,
        tokenizer_dir=tokenizer_dir,
        text_word_count=text_word_count,
        K=K,
        q_temperature=q_temperature,
        q_max_new_tokens=q_max_new_tokens,
    )


@torch.no_grad()
def _eval_bound_gold_z(
    *,
    c2f_ckpt: Path,
    sft_ckpt: Path,
    test: Path,
    config: dict[str, Any],
    limit: int | None,
    batch_size: int,
    sft_batch_size: int,
    tokenizer_dir: Path | None,
    text_word_count: int | None,
) -> dict[str, Any]:
    """K=1 fast path: score the gold ``z`` from the parquet under both p and q."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.c2f_model.configuration import C2FConfig
    from src.c2f_model.modeling import C2FForCausalLM
    from src.c2f_model.training.dataset import C2FDataset
    from src.rl.common import load_c2f_weights

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if text_word_count is None:
        text_word_count = config.get("text_word_count", config["scale_lengths"][-1])

    # --- Load C2F side ---
    space_tokenizer = load_space_tokenizer(config, tokenizer_dir)
    log.info("Loading C2F model from %s...", c2f_ckpt)
    c2f_model_config = C2FConfig.from_pretrained(str(c2f_ckpt))
    c2f_model = C2FForCausalLM(c2f_model_config)
    c2f_model = load_c2f_weights(c2f_model, c2f_ckpt)
    c2f_model.to(device)
    c2f_model.eval()
    check_vocab_consistency(c2f_model.config.vocab_size, space_tokenizer.vocab_size)

    dataset_format, prompts, responses = _load_prompts_and_responses(test, config, text_word_count)

    c2f_dataset = C2FDataset(
        data_dir=str(test.parent),
        tokenizer=space_tokenizer,
        scale_lengths=config["scale_lengths"],
        word_count_constraints=config["word_count_constraints"],
        text_word_count=text_word_count,
        parquet_filename=test.name,
        dataset_format=dataset_format,
    )
    if limit is not None:
        from torch.utils.data import Subset

        c2f_dataset = Subset(c2f_dataset, range(min(limit, len(c2f_dataset))))
    n_docs = len(c2f_dataset)
    log.info("Scoring %d documents for ELBO bound (K=1, gold z)...", n_docs)

    prompts = prompts[:n_docs]
    responses = responses[:n_docs]
    if len(prompts) != n_docs or len(responses) != n_docs:
        raise RuntimeError(
            f"Parquet row count mismatch: expected {n_docs}, got "
            f"{len(prompts)} prompts / {len(responses)} responses."
        )

    # --- C2F joint NLL with per-scale breakdown ---
    joint_nll_p, per_scale_p = _c2f_joint_nll_per_doc(
        c2f_model,
        c2f_dataset,
        device,
        batch_size,
        scale_lengths=config["scale_lengths"],
    )
    c2f_model.to("cpu")
    del c2f_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Load SFT side ---
    log.info("Loading SFT (q_φ) model from %s...", sft_ckpt)
    sft_tokenizer = AutoTokenizer.from_pretrained(str(sft_ckpt), trust_remote_code=True)
    if sft_tokenizer.pad_token is None:
        sft_tokenizer.pad_token = sft_tokenizer.eos_token
    sft_model = AutoModelForCausalLM.from_pretrained(
        str(sft_ckpt), trust_remote_code=True, dtype=torch.bfloat16
    )
    sft_model.to(device)
    sft_model.eval()

    nll_q = _sft_nll_per_doc(sft_model, sft_tokenizer, prompts, responses, device, sft_batch_size)

    bound_per_doc = joint_nll_p - nll_q
    per_doc_tokens = np.full(n_docs, text_word_count, dtype=np.int64)
    point, lo, hi = bootstrap_ci(bound_per_doc, per_doc_tokens)

    rows = [
        {
            "idx": i,
            "joint_nll_p": float(joint_nll_p[i]),
            "nll_q": float(nll_q[i]),
            "bound_nats": float(bound_per_doc[i]),
            "text_word_count": int(text_word_count),
        }
        for i in range(n_docs)
    ]

    per_scale_bound: dict[str, float] = {}
    if per_scale_p is not None:
        # q-side per-scale is deferred (needs BPE scale-boundary detection);
        # for now we report the p-side contribution per text word so users
        # can see where the joint budget goes.
        for name, vals in per_scale_p.items():
            per_scale_bound[name] = float(vals.mean()) / text_word_count

    return {
        "model_kind": "c2f_bound",
        "ckpt": str(c2f_ckpt),
        "sft_ckpt": str(sft_ckpt),
        "mask_type": c2f_model_config.mask_type,
        "K": 1,
        "num_docs": n_docs,
        "total_tokens": int(per_doc_tokens.sum()),
        "nats_per_word": point,
        "nats_per_word_ci95": [lo, hi],
        "denominator": "text_word",
        "comparable_to": "neg_log_p_x_per_text_word",
        "mean_joint_nll_p": float(joint_nll_p.mean()),
        "mean_nll_q": float(nll_q.mean()),
        "text_word_count": int(text_word_count),
        "per_scale_joint_nll_per_text_word": per_scale_bound,
        "per_doc_rows": rows,
    }


@torch.no_grad()
def _eval_bound_iwae_k(  # noqa: PLR0915  orchestration; splitting harms readability
    *,
    c2f_ckpt: Path,
    sft_ckpt: Path,
    test: Path,
    config: dict[str, Any],
    limit: int | None,
    batch_size: int,
    sft_batch_size: int,
    tokenizer_dir: Path | None,
    text_word_count: int | None,
    K: int,
    q_temperature: float,
    q_max_new_tokens: int,
) -> dict[str, Any]:
    """K>=2 path: sample K fresh ``z`` per doc from ``q_φ``, aggregate via logsumexp.

    Order of operations (memory-driven): load SFT first, sample + score q,
    free SFT, load C2F, score p. Reverse of the K=1 path because we need
    fresh samples from q before we know which (x, z) pairs to score under p.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.c2f_model.configuration import C2FConfig
    from src.c2f_model.modeling import C2FForCausalLM
    from src.eval.q_sampler import sample_k_valid_responses
    from src.rl.common import load_c2f_weights

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if text_word_count is None:
        text_word_count = config.get("text_word_count", config["scale_lengths"][-1])

    # --- Load prompts only (we sample fresh z, so parquet responses are unused) ---
    dataset_format, prompts, _responses = _load_prompts_and_responses(test, config, text_word_count)
    if limit is not None:
        prompts = prompts[:limit]
    n_docs = len(prompts)
    log.info(
        "Scoring %d docs for IWAE-K bound (K=%d, sampling from q_φ, T=%.2f)...",
        n_docs,
        K,
        q_temperature,
    )
    if dataset_format not in ("sft", "c2f"):  # pragma: no cover — guarded earlier
        raise ValueError(dataset_format)

    # --- Load SFT (q_φ) for sampling + scoring q ---
    log.info("Loading SFT (q_φ) model from %s...", sft_ckpt)
    sft_tokenizer = AutoTokenizer.from_pretrained(str(sft_ckpt), trust_remote_code=True)
    if sft_tokenizer.pad_token is None:
        sft_tokenizer.pad_token = sft_tokenizer.eos_token
    sft_model = AutoModelForCausalLM.from_pretrained(
        str(sft_ckpt), trust_remote_code=True, dtype=torch.bfloat16
    )
    sft_model.to(device)
    sft_model.eval()

    samples = sample_k_valid_responses(
        sft_model,
        sft_tokenizer,
        prompts,
        K=K,
        word_count_constraints=config["word_count_constraints"],
        temperature=q_temperature,
        max_new_tokens=q_max_new_tokens,
        device=device,
    )
    # Flatten (B, K) → (B*K,) pairs, preserving (b, k) order.
    flat_prompts: list[str] = []
    flat_responses: list[str] = []
    flat_layers: list[list[str]] = []
    for b, per_prompt in enumerate(samples):
        for qs in per_prompt:
            flat_prompts.append(prompts[b])
            flat_responses.append(qs.response)
            flat_layers.append(qs.layers)

    log.info("Scoring -log q(z_k|x) for %d (x, z_k) pairs...", len(flat_prompts))
    flat_nll_q = _sft_nll_per_doc(
        sft_model, sft_tokenizer, flat_prompts, flat_responses, device, sft_batch_size
    )

    sft_model.to("cpu")
    del sft_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Load C2F for scoring p(x, z_k) ---
    space_tokenizer = load_space_tokenizer(config, tokenizer_dir)
    log.info("Loading C2F model from %s...", c2f_ckpt)
    c2f_model_config = C2FConfig.from_pretrained(str(c2f_ckpt))
    c2f_model = C2FForCausalLM(c2f_model_config)
    c2f_model = load_c2f_weights(c2f_model, c2f_ckpt)
    c2f_model.to(device)
    c2f_model.eval()
    check_vocab_consistency(c2f_model.config.vocab_size, space_tokenizer.vocab_size)

    sampled_ds = _build_sampled_c2f_dataset(
        space_tokenizer, config, flat_prompts, flat_layers, text_word_count
    )
    log.info("Scoring -log p(x, z_k) for %d (x, z_k) pairs...", len(sampled_ds))
    flat_joint_nll_p, flat_per_scale_p = _c2f_joint_nll_per_doc(
        c2f_model,
        sampled_ds,
        device,
        batch_size,
        scale_lengths=config["scale_lengths"],
    )

    # --- Combine: reshape (B*K,) → (B, K), logsumexp over K ---
    joint_nll_p = flat_joint_nll_p.reshape(n_docs, K)
    nll_q = flat_nll_q.reshape(n_docs, K)
    bound_per_doc = _iwae_logsumexp_bound(joint_nll_p, nll_q, K)

    per_doc_tokens = np.full(n_docs, text_word_count, dtype=np.int64)
    point, lo, hi = bootstrap_ci(bound_per_doc, per_doc_tokens)

    rows = [
        {
            "idx": i,
            "joint_nll_p_samples": joint_nll_p[i].tolist(),
            "nll_q_samples": nll_q[i].tolist(),
            "log_w_samples": (nll_q[i] - joint_nll_p[i]).tolist(),
            "bound_nats": float(bound_per_doc[i]),
            "text_word_count": int(text_word_count),
        }
        for i in range(n_docs)
    ]

    per_scale_bound: dict[str, float] = {}
    if flat_per_scale_p is not None:
        for name, vals in flat_per_scale_p.items():
            per_scale_bound[name] = float(vals.mean()) / text_word_count

    return {
        "model_kind": "c2f_bound",
        "ckpt": str(c2f_ckpt),
        "sft_ckpt": str(sft_ckpt),
        "mask_type": c2f_model_config.mask_type,
        "K": K,
        "q_temperature": q_temperature,
        "num_docs": n_docs,
        "total_tokens": int(per_doc_tokens.sum()),
        "nats_per_word": point,
        "nats_per_word_ci95": [lo, hi],
        "denominator": "text_word",
        "comparable_to": "neg_log_p_x_per_text_word",
        "mean_joint_nll_p": float(joint_nll_p.mean()),
        "mean_nll_q": float(nll_q.mean()),
        "text_word_count": int(text_word_count),
        "per_scale_joint_nll_per_text_word": per_scale_bound,
        "per_doc_rows": rows,
    }
