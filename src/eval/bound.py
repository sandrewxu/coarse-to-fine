"""ELBO upper bound on ``-log p(x)`` for C2F, using the SFT model as ``q_φ``.

The C2F decoder models ``p(x, z) = p(z_4) p(z_3|z_4) p(z_2|z_3) p(z_1|z_2) p(x|z_1)``.
The SFT model was trained to model ``q(z|x)``: it takes the text as a user
message and emits the multi-scale summary as the assistant response. Together
they give a valid ELBO:

    log p(x) ≥ log p(x, z) - log q(z|x)      for z ~ q(z|x)

Equivalently, as an upper bound on ``-log p(x)`` per text word:

    -log p(x) / T  ≤  (JointNLL_p(x, z) - NLL_q(z|x)) / T

where ``T = text_word_count`` (32 by default). This is directly comparable to
the AR baseline's exact NLL per text word — AR reports ``-log p(x) / T``
without latents, and a tight C2F bound below AR would mean the C2F model
assigns higher probability to ``x``.

We use K=1 with the gold ``z`` already stored in the test parquet (which was
sampled from ``q_φ`` at generation time). This gives an unbiased single-sample
ELBO estimate — in expectation over ``z ~ q``, the RHS is an upper bound on
``-log p(x)``. IWAE-K>1 (tighter bound) requires fresh samples from ``q_φ``
and is deferred.

Notes
-----
* The two models use different tokenizations (space tokenizer for C2F, Qwen3
  BPE for SFT). Both assign a well-defined probability to the *text* of the
  ``z`` sample; the ELBO is valid over text.
* SFT's NLL is summed only over the assistant-response tokens (prompt tokens
  are ignored via ``-100`` labels, matching the training-time loss mask).
* C2F's joint NLL is summed over all content tokens (z_4 + z_3 + z_2 + z_1 +
  text) exactly as :func:`src.eval.c2f.eval_c2f` does.
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


@torch.no_grad()
def _c2f_joint_nll_per_doc(
    model,
    dataset,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Return the joint NLL ``-log p(x, z)`` per document under the C2F model.

    Mirrors the per-doc accounting in :func:`src.eval.c2f.eval_c2f` without the
    per-scale breakdown — we only need the joint for the ELBO combination.
    """
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    mask_type = model.config.mask_type

    per_doc_nll: list[float] = []
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
    return np.asarray(per_doc_nll, dtype=np.float64)


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
) -> dict[str, Any]:
    """IWAE-1 ELBO upper bound on ``-log p(x) / text_word_count`` for C2F.

    Args:
        c2f_ckpt: C2F decoder checkpoint directory.
        sft_ckpt: SFT checkpoint directory (used as ``q_φ``).
        test: SFT-format parquet with ``prompt`` (text) and ``response``
            (multi-scale z_n lines) columns. The gold ``z`` in the parquet
            is used as the single K=1 sample from ``q_φ``.
        config: Experiment config (for scale_lengths, word_count_constraints,
            text_word_count, tokenizer_dir).
        limit: Cap number of docs scored.
        batch_size: C2F forward-pass batch size.
        sft_batch_size: SFT forward-pass batch size (default lower because SFT
            is Qwen3-4B and sequences are longer after BPE tokenization).
        tokenizer_dir: Override the C2F (space) tokenizer dir.
        text_word_count: Denominator for the per-text-word bound. Defaults to
            ``config["text_word_count"]`` (32 in the default setup).

    Returns:
        Dict with ``model_kind="c2f_bound"``, ``K=1``, plus per-doc bound
        rows and a bootstrap CI on ``nats_per_text_word_bound``.
    """
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

    # Detect parquet format (SFT vs c2f-flat); reconstruct prompt+response in
    # the c2f case so q(z|x) has concrete strings to score. Reconstruction is
    # word-lossless because flatten_for_c2f is the inverse operation.
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

    # HF ``load_dataset`` preserves parquet row order, so the first n_docs
    # strings align with C2FDataset[:n_docs]. Test covered in
    # ``tests/test_eval_bound.py::test_bound_parquet_row_order_preserved``.
    prompts = prompts[:n_docs]
    responses = responses[:n_docs]
    if len(prompts) != n_docs or len(responses) != n_docs:
        raise RuntimeError(
            f"Parquet row count mismatch: expected {n_docs}, got "
            f"{len(prompts)} prompts / {len(responses)} responses."
        )

    # --- C2F joint NLL ---
    joint_nll_p = _c2f_joint_nll_per_doc(c2f_model, c2f_dataset, device, batch_size)
    # Free C2F GPU memory before loading SFT.
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

    # --- Combine: bound_per_doc = joint_nll_p - nll_q; per text word ---
    bound_per_doc = joint_nll_p - nll_q
    per_doc_tokens = np.full(n_docs, text_word_count, dtype=np.int64)

    point, lo, hi = bootstrap_ci(bound_per_doc, per_doc_tokens)
    # The AR baseline reports -log p(x) / text_word_count exactly; here the
    # same quantity is an upper bound (larger than the true value if the ELBO
    # is loose). The comparison is: if this bound < AR's exact NLL, the C2F
    # model is strictly better at modeling p(x).

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
        "per_doc_rows": rows,
    }
