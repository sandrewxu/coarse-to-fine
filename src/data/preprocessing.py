"""Reusable data preprocessing functions."""

import glob
from pathlib import Path
from src.common.logging import get_logger

log = get_logger(__name__)


def preprocess_tinystories(src_dir: str | Path, words_per_chunk: int = 32) -> None:
    """Split TinyStories text files into fixed-length word chunks.

    Each ``.txt`` file under *src_dir* is processed in-place:

    1. Split on ``<|endoftext|>`` to recover individual stories.
    2. Chunk each story into segments of exactly *words_per_chunk* words.
    3. Discard the last incomplete chunk of each story.
    4. Overwrite the file with one chunk per line.

    Args:
        src_dir: Directory containing ``.txt`` files to process.
        words_per_chunk: Number of words per output chunk.
    """
    src_dir = str(src_dir)
    log.info(f"Preprocessing TinyStories: splitting into {words_per_chunk}-word chunks...")

    txt_files = glob.glob(f"{src_dir}/**/*.txt", recursive=True)

    for txt_file in txt_files:
        log.info(f"Processing {txt_file}...")
        with open(txt_file, "r", encoding="utf-8") as f:
            content = f.read()

        stories = content.split("<|endoftext|>")
        stories = [s.strip() for s in stories if s.strip()]

        all_chunks: list[str] = []
        for story in stories:
            words = story.split()
            for i in range(0, len(words), words_per_chunk):
                chunk = words[i : i + words_per_chunk]
                if len(chunk) == words_per_chunk:
                    all_chunks.append(" ".join(chunk))

        with open(txt_file, "w", encoding="utf-8") as f:
            for chunk in all_chunks:
                f.write(chunk + "\n")

        log.info(f"  Created {len(all_chunks)} chunks from {len(stories)} stories")

    log.info("Preprocessing complete!")
