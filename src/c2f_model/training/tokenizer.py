"""
Space-based (word-level) tokenizer for C2F training.

Builds a vocabulary from the training data where each space-separated word is
a single token.  This guarantees a 1:1 mapping between words and token
positions, which is ideal for the C2F model's fixed-layout sequence structure:
every ``scale_lengths[k]`` words become exactly ``scale_lengths[k]`` tokens
with zero truncation or intra-scale padding.

The tokenizer is saved in HuggingFace format so it can be loaded later via
``AutoTokenizer.from_pretrained`` or ``PreTrainedTokenizerFast.from_pretrained``.
"""

import re
from collections.abc import Iterator
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.trainers import WordLevelTrainer
from transformers import PreTrainedTokenizerFast

from src.common.logging import get_logger

log = get_logger(__name__)

_LAYER_PATTERN = re.compile(r"^z_\d+:\s*(.*)$")

# Special tokens — indices 0-3 are reserved.
PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
BOS_TOKEN = "[BOS]"
EOS_TOKEN = "[EOS]"


# ── Text iterators (one line per scale segment) ────────────────────────────


def _iter_texts_sft(data_dir: Path, parquet_filename: str = "train.parquet") -> Iterator[str]:
    """Yield text lines from an SFT parquet (prompt + parsed response layers)."""
    ds = load_dataset("parquet", data_files=str(data_dir / parquet_filename), split="train")
    for row in ds:
        # Prompt: the original 32-word document
        yield row["prompt"]
        # Response: z_4: ...\nz_3: ... format — yield each layer's content
        for line in row["response"].strip().split("\n"):
            line = line.strip()
            match = _LAYER_PATTERN.match(line)
            if match:
                yield match.group(1).strip()


def _iter_texts_c2f(data_dir: Path, parquet_filename: str = "c2f_train.parquet") -> Iterator[str]:
    """Yield text lines from a C2F parquet (single flat ``text`` column)."""
    ds = load_dataset("parquet", data_files=str(data_dir / parquet_filename), split="train")
    for row in ds:
        yield row["text"]


# ── Public API ──────────────────────────────────────────────────────────────


def train_space_tokenizer(
    data_dir: str | Path,
    dataset_format: str = "sft",
    parquet_filename: str | None = None,
    min_frequency: int = 1,
) -> PreTrainedTokenizerFast:
    """
    Train a word-level tokenizer from the training data.

    Each space-separated word becomes exactly one token, giving a perfect
    1:1 mapping between word count constraints and token positions.

    Args:
        data_dir: Directory containing the training parquet.
        dataset_format: ``"sft"`` or ``"c2f"``.
        parquet_filename: Override parquet filename.
        min_frequency: Minimum word frequency to include in vocab.

    Returns:
        HuggingFace-compatible word-level tokenizer.
    """
    data_dir = Path(data_dir)
    if parquet_filename is None:
        parquet_filename = "train.parquet" if dataset_format == "sft" else "c2f_train.parquet"

    # Collect text lines from the dataset
    if dataset_format == "sft":
        texts = list(_iter_texts_sft(data_dir, parquet_filename))
    else:
        texts = list(_iter_texts_c2f(data_dir, parquet_filename))

    # Build a word-level tokenizer (splits on whitespace, one token per word)
    tokenizer = Tokenizer(WordLevel(unk_token=UNK_TOKEN))
    tokenizer.pre_tokenizer = WhitespaceSplit()

    trainer = WordLevelTrainer(
        special_tokens=[PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN],
        min_frequency=min_frequency,
    )
    tokenizer.train_from_iterator(texts, trainer)

    # Wrap as HuggingFace PreTrainedTokenizerFast for Trainer compatibility
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
        pad_token=PAD_TOKEN,
        unk_token=UNK_TOKEN,
    )
    return hf_tokenizer


def load_or_train_space_tokenizer(
    tokenizer_dir: str | Path,
    data_dir: str | Path,
    dataset_format: str = "sft",
    parquet_filename: str | None = None,
    min_frequency: int = 1,
) -> PreTrainedTokenizerFast:
    """
    Load a previously saved space tokenizer, or train and save a new one.

    Args:
        tokenizer_dir: Directory to save/load the tokenizer.
        data_dir: Directory containing the training parquet (used if training).
        dataset_format: ``"sft"`` or ``"c2f"``.
        parquet_filename: Override parquet filename.
        min_frequency: Minimum word frequency to include in vocab.

    Returns:
        HuggingFace-compatible word-level tokenizer.
    """
    tokenizer_dir = Path(tokenizer_dir)

    if (tokenizer_dir / "tokenizer.json").exists():
        log.info(f"Loading existing space tokenizer from {tokenizer_dir}")
        return PreTrainedTokenizerFast.from_pretrained(str(tokenizer_dir))

    log.info(f"Training new space tokenizer from {data_dir}...")
    tokenizer = train_space_tokenizer(
        data_dir=data_dir,
        dataset_format=dataset_format,
        parquet_filename=parquet_filename,
        min_frequency=min_frequency,
    )

    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(tokenizer_dir))
    log.info(f"  Vocab size: {tokenizer.vocab_size}")
    log.info(f"  Saved to {tokenizer_dir}")

    return tokenizer
