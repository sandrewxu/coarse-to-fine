"""Submit, monitor, and download OpenAI Batch API jobs."""

import json
import os
import time
from pathlib import Path

from openai import OpenAI


def upload_file(client: OpenAI, file_path: Path) -> str:
    """Upload a JSONL file to OpenAI for batch processing. Returns file ID."""
    with open(file_path, "rb") as f:
        response = client.files.create(file=f, purpose="batch")
    return response.id


def submit_batch(
    client: OpenAI,
    file_path: Path,
    metadata: dict[str, str] | None = None,
) -> str:
    """Upload file and submit a batch processing job.

    Returns the batch ID.
    """
    print(f"Uploading {file_path}...")
    file_id = upload_file(client, file_path)
    print(f"  File ID: {file_id}")

    print("Submitting batch request...")
    batch = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata=metadata or {},
    )

    print(f"\nBatch submitted successfully!")
    print(f"  Batch ID: {batch.id}")
    print(f"  Status: {batch.status}")
    if metadata:
        print(f"  Metadata: {json.dumps(metadata, indent=2)}")

    return batch.id


# ---------------------------------------------------------------------------
# Monitoring helpers
# ---------------------------------------------------------------------------


def _print_batch_info(batch) -> None:
    """Print detailed information about a batch job."""
    print(f"\n{'=' * 50}")
    print(f"Batch ID: {batch.id}")
    print(f"Status: {batch.status}")

    if batch.created_at:
        print(f"Created: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(batch.created_at))}")
    if batch.in_progress_at:
        print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(batch.in_progress_at))}")
    if batch.completed_at:
        print(f"Completed: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(batch.completed_at))}")

    if hasattr(batch, "request_counts") and batch.request_counts:
        rc = batch.request_counts
        print(f"Requests: {rc.completed}/{rc.total} ({rc.failed} failed)")

    if batch.metadata:
        print(f"Metadata: {batch.metadata}")
    print(f"{'=' * 50}")


def _download_file(client: OpenAI, file_id: str, output_path: Path) -> Path:
    """Download a file from OpenAI and save it locally."""
    os.makedirs(output_path.parent, exist_ok=True)
    response = client.files.content(file_id)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(response.text)
    print(f"  Saved: {output_path}")
    return output_path


def _get_output_path(batch, output_dir: Path, file_type: str = "output") -> Path:
    """Generate output path encoding batch metadata in the directory structure."""
    path = output_dir
    for key, value in sorted(batch.metadata.items()):
        path = path / f"{key}__{value}"
    path = path / batch.id / f"{file_type}.jsonl"
    return path


# ---------------------------------------------------------------------------
# Public monitoring / download API
# ---------------------------------------------------------------------------

_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled", "expired"})


def monitor_batch(
    client: OpenAI,
    batch_id: str,
    output_dir: Path,
    poll_interval: int = 30,
) -> Path | None:
    """Poll a single batch until completion and download results.

    Returns the output file path on success, or ``None`` on failure.
    """
    while True:
        batch = client.batches.retrieve(batch_id)
        _print_batch_info(batch)

        if batch.status in _TERMINAL_STATUSES:
            if batch.status == "completed" and batch.output_file_id:
                out_path = _get_output_path(batch, output_dir, "output")
                _download_file(client, batch.output_file_id, out_path)

                if batch.error_file_id:
                    err_path = _get_output_path(batch, output_dir, "errors")
                    _download_file(client, batch.error_file_id, err_path)

                return out_path

            print(f"Batch ended with status: {batch.status}")
            return None

        print(f"Polling again in {poll_interval}s...")
        time.sleep(poll_interval)


def monitor_all(
    client: OpenAI,
    output_dir: Path,
    poll_interval: int = 30,
) -> list[Path]:
    """Monitor all active batches and download results as they complete."""
    all_batches = list(client.batches.list(limit=100))
    active = {
        b.id: b for b in all_batches
        if b.status not in _TERMINAL_STATUSES
    }

    if not active:
        print("No active batches found.")
        return []

    print(f"Monitoring {len(active)} active batches...")
    downloaded: list[Path] = []

    while active:
        for batch_id in list(active):
            batch = client.batches.retrieve(batch_id)
            _print_batch_info(batch)

            if batch.status in _TERMINAL_STATUSES:
                if batch.status == "completed" and batch.output_file_id:
                    out_path = _get_output_path(batch, output_dir, "output")
                    _download_file(client, batch.output_file_id, out_path)
                    downloaded.append(out_path)

                    if batch.error_file_id:
                        err_path = _get_output_path(batch, output_dir, "errors")
                        _download_file(client, batch.error_file_id, err_path)

                active.pop(batch_id)

        if active:
            print(f"\n{len(active)} batches still active. Polling in {poll_interval}s...")
            time.sleep(poll_interval)

    return downloaded


def download_completed(
    client: OpenAI,
    output_dir: Path,
    run_tag: str | None = None,
    limit: int = 100,
) -> list[Path]:
    """Download results for all completed batches, optionally filtered by run_tag."""
    all_batches = list(client.batches.list(limit=limit))
    downloaded: list[Path] = []

    for batch in all_batches:
        if batch.status != "completed":
            continue
        if run_tag and (not batch.metadata or batch.metadata.get("run_tag") != run_tag):
            continue

        print(f"\nBatch {batch.id} (completed)")

        if batch.output_file_id:
            out_path = _get_output_path(batch, output_dir, "output")
            if not out_path.exists():
                _download_file(client, batch.output_file_id, out_path)
                downloaded.append(out_path)
            else:
                print(f"  Already exists: {out_path}")

        if batch.error_file_id:
            err_path = _get_output_path(batch, output_dir, "errors")
            if not err_path.exists():
                _download_file(client, batch.error_file_id, err_path)

    if not downloaded:
        print("No new completed batches to download.")
    return downloaded
