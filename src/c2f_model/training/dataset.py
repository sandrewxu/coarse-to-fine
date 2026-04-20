"""
C2F tokenization pipeline: converts latent+text sequences into
fixed-layout token tensors for the Coarse-to-Fine model.

Supports two input formats controlled by ``dataset_format``:

- ``"c2f"`` (default): A single ``text`` column containing flat word sequences
  (latent words + prompt words concatenated).  Produced by ``flatten_for_c2f``
  in step 5.

- ``"sft"``: Two columns — ``prompt`` (original 32-word text) and ``response``
  (raw ``z_n:`` format latent layers).  This is the veRL-compatible SFT parquet
  from step 3.  The dataset parses z_n labels and flattens on the fly.

Output: PyTorch tensors with layout [BOS | scale_0 | scale_1 | ... | padding],
        where each scale occupies exactly scale_lengths[k] token positions.
"""

import math
import re
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import Dataset as TorchDataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

_LAYER_PATTERN = re.compile(r"^z_\d+:\s*(.*)$")


class C2FDataset(TorchDataset):
    """
    Dataset that tokenizes word sequences into fixed-position token slots
    for the C2F model.

    Supports two input formats (``dataset_format``):

    - ``"c2f"``: reads a ``text`` column of flat word sequences from
      ``c2f_train.parquet``.
    - ``"sft"``: reads ``prompt`` + ``response`` columns from ``train.parquet``
      (veRL SFT format) and flattens them on the fly.

    The flat text is split by word count (the strict verification guarantees
    exact counts):
        words[0:2]   -> scale 0 (z_4)  -> scale_lengths[0] tokens
        words[2:6]   -> scale 1 (z_3)  -> scale_lengths[1] tokens
        words[6:14]  -> scale 2 (z_2)  -> scale_lengths[2] tokens
        words[14:30] -> scale 3 (z_1)  -> scale_lengths[3] tokens
        words[30:62] -> scale 4 (text) -> scale_lengths[4] tokens
    """

    VALID_FORMATS = ("c2f", "sft")

    def __init__(
        self,
        data_dir: str | Path,
        tokenizer_name_or_path: str = "",
        scale_lengths: list[int] | None = None,
        word_count_constraints: dict[str, int] | None = None,
        text_word_count: int = 32,
        parquet_filename: str | None = None,
        dataset_format: str = "c2f",
        tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> None:
        """
        Args:
            data_dir: Directory containing the training parquet.
            tokenizer_name_or_path: HF tokenizer name or local checkpoint path.
                Ignored when ``tokenizer`` is provided.
            scale_lengths: Token positions per scale, e.g. [2, 4, 8, 16, 32].
            word_count_constraints: Dict like {"z_4": 2, "z_3": 4, "z_2": 8, "z_1": 16}.
            text_word_count: Number of words in the text scale (original document).
            parquet_filename: Name of the parquet file in data_dir.  Defaults to
                ``"c2f_train.parquet"`` for ``"c2f"`` format and
                ``"train.parquet"`` for ``"sft"`` format.
            dataset_format: ``"c2f"`` for flat text column, ``"sft"`` for
                prompt+response columns (veRL format).
            tokenizer: Optional pre-built tokenizer.  When provided,
                ``tokenizer_name_or_path`` is ignored.  Use this to pass a
                space-based tokenizer trained by
                :func:`src.c2f_model.training.tokenizer.load_or_train_space_tokenizer`.
        """
        if dataset_format not in self.VALID_FORMATS:
            raise ValueError(
                f"dataset_format must be one of {self.VALID_FORMATS}, got {dataset_format!r}"
            )

        if scale_lengths is None or word_count_constraints is None:
            raise ValueError(
                "C2FDataset requires both scale_lengths and word_count_constraints "
                "(typically loaded from config/latent_generation.yaml)."
            )

        self.dataset_format = dataset_format
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name_or_path, trust_remote_code=True
            )
        self.scale_lengths = scale_lengths
        self.seq_len = 2 ** math.ceil(math.log2(1 + sum(scale_lengths)))

        layer_names = ["z_4", "z_3", "z_2", "z_1"]
        self.word_counts = [word_count_constraints[n] for n in layer_names]
        self.word_counts.append(text_word_count)

        self.word_boundaries = []
        pos = 0
        for wc in self.word_counts:
            self.word_boundaries.append((pos, pos + wc))
            pos += wc

        if parquet_filename is None:
            parquet_filename = "c2f_train.parquet" if dataset_format == "c2f" else "train.parquet"

        data_dir = Path(data_dir)
        parquet_path = data_dir / parquet_filename
        self.dataset = load_dataset("parquet", data_files=str(parquet_path), split="train")

        # Qwen3 has no explicit BOS; use eos as fallback.
        self.bos_id = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        self.pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

    def __len__(self) -> int:
        return len(self.dataset)

    def _row_to_words(self, row: dict) -> list[str]:
        """Extract a flat word list from a dataset row, handling both formats."""
        if self.dataset_format == "c2f":
            return row["text"].split()

        # SFT format: parse z_n: labels from response, append prompt words
        layer_contents = _parse_sft_response(row["response"])
        flat_text = " ".join([*layer_contents, row["prompt"]])
        return flat_text.split()

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        words = self._row_to_words(self.dataset[idx])
        input_ids = self._build_token_sequence(words)
        labels = self._build_labels(input_ids)
        return {"input_ids": input_ids, "labels": labels}

    def _build_token_sequence(self, words: list[str]) -> torch.LongTensor:
        """Tokenize per-scale word segments and assemble the fixed-layout sequence."""
        tokens = [self.bos_id]

        for (start, end), length in zip(
            self.word_boundaries, self.scale_lengths, strict=False
        ):
            segment_text = " ".join(words[start:end])
            encoded = self.tokenizer.encode(segment_text, add_special_tokens=False)
            if len(encoded) >= length:
                encoded = encoded[:length]
            else:
                encoded = encoded + [self.pad_id] * (length - len(encoded))
            tokens.extend(encoded)

        while len(tokens) < self.seq_len:
            tokens.append(self.pad_id)

        return torch.tensor(tokens[: self.seq_len], dtype=torch.long)

    def _build_labels(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        """Labels for unshifted cross-entropy: -100 at BOS and post-content padding."""
        labels = input_ids.clone()
        labels[0] = -100
        # Position-based (not token-id based): real content tokens whose id
        # happens to equal pad_id must still be scored.
        content_len = 1 + sum(self.scale_lengths)
        labels[content_len:] = -100
        return labels

    def train_test_split(self, test_size: float = 0.05, seed: int = 42) -> dict[str, "C2FDataset"]:
        """
        Split into train and test sets.

        Returns:
            Dict with 'train' and 'test' keys, each a C2FDataset-like object.
        """
        split = self.dataset.train_test_split(test_size=test_size, seed=seed)

        train_ds = _C2FDatasetView(self, split["train"])
        test_ds = _C2FDatasetView(self, split["test"])

        return {"train": train_ds, "test": test_ds}


class _C2FDatasetView(TorchDataset):
    """Lightweight view over a C2FDataset with a different underlying HF dataset split."""

    def __init__(self, parent: C2FDataset, hf_dataset) -> None:
        self.parent = parent
        self.dataset = hf_dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        words = self.parent._row_to_words(self.dataset[idx])
        input_ids = self.parent._build_token_sequence(words)
        labels = self.parent._build_labels(input_ids)
        return {"input_ids": input_ids, "labels": labels}


def _parse_sft_response(response: str) -> list[str]:
    """
    Parse z_n: labels from an SFT response and return content strings per layer.

    Args:
        response: Raw response with ``z_4: ...\\nz_3: ...`` format.

    Returns:
        List of content strings (one per layer, in z_4 -> z_1 order).
    """
    contents = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        match = _LAYER_PATTERN.match(line)
        if match:
            contents.append(match.group(1).strip())
    return contents


class ARDataset(TorchDataset):
    """Text-only dataset for the no-latents AR baseline.

    Reads the same parquet as :class:`C2FDataset` but emits a sequence of
    ``[BOS, w_0, ..., w_{N-1}]`` (length ``text_word_count + 1``) consisting
    only of the document's text tokens. Latents are dropped.

    Labels mirror ``input_ids`` with padding masked to ``-100``; the model's
    standard shifted-CE loss head consumes them. Used by
    ``scripts/06b_train_ar_baseline.py``.
    """

    VALID_FORMATS = ("c2f", "sft")

    def __init__(
        self,
        data_dir: str | Path,
        scale_lengths: list[int] | None = None,
        text_word_count: int = 32,
        parquet_filename: str | None = None,
        dataset_format: str = "c2f",
        tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> None:
        """
        Args:
            data_dir: Directory containing the training parquet.
            scale_lengths: Same list as C2FDataset uses, only ``scale_lengths[-1]``
                (the text scale) is read; ignored otherwise. Accepted for
                config-call symmetry with C2FDataset.
            text_word_count: Words per document (overridden by
                ``scale_lengths[-1]`` when both are provided).
            parquet_filename: Defaults to ``"c2f_train.parquet"`` for ``"c2f"``
                format, ``"train.parquet"`` for ``"sft"`` format.
            dataset_format: ``"c2f"`` (single ``text`` column with concatenated
                latents+text) or ``"sft"`` (``prompt`` column has the raw doc).
            tokenizer: Pre-built space tokenizer.
        """
        if dataset_format not in self.VALID_FORMATS:
            raise ValueError(
                f"dataset_format must be one of {self.VALID_FORMATS}, got {dataset_format!r}"
            )
        if tokenizer is None:
            raise ValueError("ARDataset requires a pre-built tokenizer (the space tokenizer).")

        self.dataset_format = dataset_format
        self.tokenizer = tokenizer
        if scale_lengths is not None:
            text_word_count = scale_lengths[-1]
        self.text_word_count = text_word_count
        # Sequence layout: [BOS, w_0, ..., w_{N-1}] -> N+1 positions.
        self.seq_len = text_word_count + 1

        if parquet_filename is None:
            parquet_filename = "c2f_train.parquet" if dataset_format == "c2f" else "train.parquet"

        data_dir = Path(data_dir)
        parquet_path = data_dir / parquet_filename
        self.dataset = load_dataset("parquet", data_files=str(parquet_path), split="train")

        self.bos_id = self.tokenizer.bos_token_id
        if self.bos_id is None:
            self.bos_id = self.tokenizer.eos_token_id
        self.pad_id = self.tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = self.tokenizer.eos_token_id

    def __len__(self) -> int:
        return len(self.dataset)

    def _row_to_text(self, row: dict) -> str:
        """Extract the original document text (no latents) from a row."""
        if self.dataset_format == "sft":
            return row["prompt"]
        # c2f format: ``text`` column has [latent words, text words] concatenated.
        # The last text_word_count words are the original document.
        words = row["text"].split()
        return " ".join(words[-self.text_word_count :])

    def _encode_row(self, row: dict) -> dict[str, torch.Tensor]:
        """Tokenize one row into ``{input_ids, labels}`` tensors."""
        text = self._row_to_text(row)
        encoded = self.tokenizer.encode(text, add_special_tokens=False)
        if len(encoded) >= self.text_word_count:
            encoded = encoded[: self.text_word_count]
            content_words = self.text_word_count
        else:
            content_words = len(encoded)
            encoded = encoded + [self.pad_id] * (self.text_word_count - content_words)

        tokens = [self.bos_id, *encoded]
        input_ids = torch.tensor(tokens, dtype=torch.long)

        # labels[i] is the target for the token at position i in shifted-CE
        # (logits[:-1] predicts labels[1:]). Mask the padding region from loss.
        labels = input_ids.clone()
        content_end = 1 + content_words
        if content_end < self.seq_len:
            labels[content_end:] = -100
        return {"input_ids": input_ids, "labels": labels}

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self._encode_row(self.dataset[idx])

    def train_test_split(self, test_size: float = 0.05, seed: int = 42) -> dict[str, "ARDataset"]:
        """Same 95/5 deterministic split as C2FDataset (use the same seed for parity)."""
        split = self.dataset.train_test_split(test_size=test_size, seed=seed)
        return {
            "train": _ARDatasetView(self, split["train"]),
            "test": _ARDatasetView(self, split["test"]),
        }


class _ARDatasetView(TorchDataset):
    """View over an ARDataset with a different underlying HF dataset split."""

    def __init__(self, parent: ARDataset, hf_dataset) -> None:
        self.parent = parent
        self.dataset = hf_dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.parent._encode_row(self.dataset[idx])
