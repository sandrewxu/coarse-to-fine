"""Create OpenAI Batch API request JSONL files."""

import json
from pathlib import Path
from typing import Any


def load_documents(
    path: Path,
    start: int = 0,
    end: int | None = None,
) -> list[str]:
    """Load documents from a text or JSONL file and return the specified slice.

    Each non-empty line is read.  If the line is valid JSON with a ``text``
    field, that field is used; otherwise the raw stripped line is used.
    Negative indices are supported (e.g. ``start=-10000``).
    """
    docs: list[str] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                docs.append(data.get("text", line))
            except json.JSONDecodeError:
                docs.append(line)

    if end is None:
        return docs[start:]
    return docs[start:end]


def load_few_shot_examples(path: Path) -> list[dict[str, str]]:
    """Load few-shot examples from a JSONL file.

    Each line must have ``user`` and ``assistant`` keys.
    Returns a flat list of alternating user/assistant message dicts.
    """
    messages: list[dict[str, str]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            example = json.loads(line)
            messages.append({"role": "user", "content": example["user"]})
            messages.append({"role": "assistant", "content": example["assistant"]})
    return messages


def create_batch_requests(
    documents: list[str],
    system_prompt: str,
    user_prompt_template: str,
    few_shot_messages: list[dict[str, str]],
    model: str = "gpt-5-nano-2025-08-07",
    reasoning_effort: str = "low",
    verbosity: str = "medium",
) -> list[dict[str, Any]]:
    """Build OpenAI Batch API request objects for a list of documents.

    Args:
        documents: Document texts to process.
        system_prompt: System prompt content.
        user_prompt_template: Template with ``{doc}`` placeholder.
        few_shot_messages: Few-shot example messages (from :func:`load_few_shot_examples`).
        model: OpenAI model name.
        reasoning_effort: Reasoning effort level.
        verbosity: Verbosity level.

    Returns:
        List of request dicts ready to be written as JSONL.
    """
    requests: list[dict[str, Any]] = []
    for i, doc in enumerate(documents):
        messages = [{"role": "developer", "content": system_prompt}]
        messages.extend(few_shot_messages)
        messages.append({"role": "user", "content": user_prompt_template.format(doc=doc)})

        requests.append({
            "custom_id": f"request-{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": messages,
                "reasoning_effort": reasoning_effort,
                "verbosity": verbosity,
            },
        })
    return requests


def save_batch_requests(requests: list[dict], output_path: Path) -> Path:
    """Write batch API requests to a JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")
    return output_path
