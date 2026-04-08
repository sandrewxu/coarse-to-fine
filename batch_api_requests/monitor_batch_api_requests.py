import os
import time
from pathlib import Path
from typing import Optional

import httpx
import typer
from openai import OpenAI

from utils.common_utils import jload


# Load OpenAI API key from file and initialize client
oai_certs = jload("../certs/openai.json")
api_key = oai_certs["OPENAI_API_KEY"]

# Optional: organization and project (only pass if available)
client_kwargs = {
    "api_key": api_key,
    "http_client": httpx.Client(verify=False)
}
if "OPENAI_ORGANIZATION" in oai_certs:
    client_kwargs["organization"] = oai_certs["OPENAI_ORGANIZATION"]
if "OPENAI_PROJECT" in oai_certs:
    client_kwargs["project"] = oai_certs["OPENAI_PROJECT"]

client = OpenAI(**client_kwargs)

# Output directory for the JSONL files
OUTPUT_DIR = "../data/sft_data_out/v1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = typer.Typer(help="Monitor and download OpenAI batch jobs")


def get_output_filename(batch, output_dir, file_type="output"):
    """
    Generate a filename that includes metadata information.

    Args:
        batch: The batch object
        file_type: Either 'output' or 'errors'
    """
    output_dir = Path(output_dir)
    for key, value in sorted(batch.metadata.items()):
        output_dir /= f"{key}__{value}"

    output_dir /= f"{batch.id}"
    output_dir /= f"{file_type}.jsonl"
    return output_dir


def fetch_completed_sft_batches(limit=100, run_tag=None):
    """
    Fetch all completed batch jobs, optionally filtering by run_tag in metadata.

    Args:
        limit (int): Maximum number of batches to retrieve
        run_tag (str): Optional run_tag to filter by (e.g., "active_reading_v1")
    """
    try:
        batches = client.batches.list(limit=limit)
        matching_batches = []

        print(
            f"\nLooking for completed batches{' with run_tag: ' + run_tag if run_tag else ''}..."
        )
        for batch in batches:
            # Skip if we're filtering by run_tag and this batch doesn't match
            if run_tag:
                if not batch.metadata or batch.metadata.get("run_tag") != run_tag:
                    continue

            # Check if this is a completed batch
            if batch.status == "completed":
                matching_batches.append(batch)
                print(f"\nFound completed batch: {batch.id}")
                if batch.metadata:
                    print(f"  Model: {batch.metadata.get('model', 'unknown')}")
                    print(f"  User: {batch.metadata.get('user', 'unknown')}")
                    print(f"  Run Tag: {batch.metadata.get('run_tag', 'unknown')}")

        print(f"\nFound {len(matching_batches)} completed matching batches.")
        return matching_batches
    except Exception as e:
        print(f"Error fetching batches: {e}")
        return []


def download_completed_sft_batches(output_dir, run_tag=None):
    """
    Download results for all completed batches matching the run_tag.

    Args:
        run_tag (str, optional): Filter batches by this run_tag
    """
    matching_batches = fetch_completed_sft_batches(run_tag=run_tag)

    if not matching_batches:
        print("No matching completed batches found.")
        return

    for batch in matching_batches:
        print(f"\nProcessing batch {batch.id}...")

        # Print batch metadata
        if batch.metadata:
            print(f"  Model: {batch.metadata.get('model', 'unknown')}")
            print(f"  User: {batch.metadata.get('user', 'unknown')}")
            print(f"  Run Tag: {batch.metadata.get('run_tag', 'unknown')}")

        # Download output file
        if batch.output_file_id:
            output_file = get_output_filename(batch, output_dir, "output")
            if not os.path.exists(output_file):
                print(f"Downloading output file to {output_file}")
                download_batch_results(batch.output_file_id, output_file)
            else:
                print(f"Output file already exists: {output_file}")

        # Download error file if it exists
        if batch.error_file_id:
            error_file = get_output_filename(batch, output_dir, "errors")
            if not os.path.exists(error_file):
                print(f"Downloading error file to {error_file}")
                download_batch_results(batch.error_file_id, error_file)
            else:
                print(f"Error file already exists: {error_file}")


def print_batch_info(batch):
    """
    Print detailed information about a batch job.
    """
    print("\n" + "=" * 50)
    print(f"Batch ID: {batch.id}")
    print(f"Status: {batch.status}")
    print(f"Endpoint: {batch.endpoint}")
    print(
        f"Created at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(batch.created_at))}"
    )

    if batch.in_progress_at:
        print(
            f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(batch.in_progress_at))}"
        )

    if batch.completed_at:
        print(
            f"Completed at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(batch.completed_at))}"
        )

    print(
        f"Expires at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(batch.expires_at))}"
    )

    if hasattr(batch, "request_counts"):
        print("\nRequest Counts:")
        print(f"  Total: {batch.request_counts.total}")
        print(f"  Completed: {batch.request_counts.completed}")
        print(f"  Failed: {batch.request_counts.failed}")

    if batch.metadata:
        print("\nMetadata:", batch.metadata)

    print("=" * 50)


def fetch_active_batches(limit=100):
    """
    Fetch all active batch jobs from the OpenAI API.
    """
    try:
        batches = client.batches.list(limit=limit)
        all_batches = list(batches)  # Convert iterator to list

        active_batches = [
            batch
            for batch in all_batches
            if batch.status not in ["completed", "failed", "cancelled", "expired"]
        ]

        print(
            f"\nFound {len(all_batches)} total batches, {len(active_batches)} active."
        )

        # Print status summary
        status_counts = {}
        for batch in all_batches:
            status_counts[batch.status] = status_counts.get(batch.status, 0) + 1

        print("\nStatus Summary:")
        for status, count in status_counts.items():
            print(f"  {status}: {count}")

        # Print detailed info for active batches
        if active_batches:
            print("\nActive Batch Details:")
            for batch in active_batches:
                print_batch_info(batch)

        return active_batches
    except Exception as e:
        print(f"Error fetching batches: {e}")
        return []


def check_batch_status(batch_id):
    """
    Check the status of a batch job.
    """
    try:
        batch = client.batches.retrieve(batch_id)
        return batch.status, batch
    except Exception as e:
        print(f"Error checking batch {batch_id}: {e}")
        return None, None


def download_batch_results(file_id, output_file):
    """
    Download batch results and save them locally.
    """
    try:
        response = client.files.content(file_id)

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(response.text)
        print(f"Results saved to {output_file}")
    except Exception as e:
        print(f"Error downloading results for file {file_id}: {e}")


def monitor_batches(output_dir, poll_interval=30):
    """
    Monitor batch jobs and download outputs when available.
    """
    remaining_batches = {batch.id: batch for batch in fetch_active_batches()}

    while remaining_batches:
        print(f"\nMonitoring {len(remaining_batches)} pending batches...")
        for batch_id in list(remaining_batches):
            status, batch = check_batch_status(batch_id)

            if batch:
                print_batch_info(batch)
            else:
                print(f"Could not retrieve info for batch {batch_id}")

            if status in ["completed", "failed", "cancelled", "expired"]:
                if status == "completed" and batch.output_file_id:
                    print(f"Batch {batch_id} is complete. Downloading results...")
                    output_file = os.path.join(output_dir, f"{batch_id}_output.jsonl")
                    download_batch_results(batch.output_file_id, output_file)

                    if batch.error_file_id:
                        error_file = os.path.join(
                            output_dir, f"{batch_id}_errors.jsonl"
                        )
                        download_batch_results(batch.error_file_id, error_file)

                remaining_batches.pop(batch_id)

        if remaining_batches:
            print(f"\nSleeping for {poll_interval} seconds...")
            time.sleep(poll_interval)


@app.command("download-sft")
def cli_download_sft(
    run_tag: Optional[str] = typer.Option(
        None, "--run-tag", help='Filter batches by run_tag (e.g., "active_reading_v1")'
    ),
    output_dir: Path = typer.Option(
        Path(OUTPUT_DIR),
        "--output-dir",
        dir_okay=True,
        file_okay=False,
        help="Directory to save output and error JSONL files",
    ),
):
    """
    Download results for all completed batches, optionally filtered by --run-tag.
    """
    os.makedirs(output_dir, exist_ok=True)
    download_completed_sft_batches(output_dir=output_dir, run_tag=run_tag)


@app.command("monitor")
def cli_monitor(
    output_dir: Path = typer.Option(
        Path(OUTPUT_DIR),
        "--output-dir",
        dir_okay=True,
        file_okay=False,
        help="Directory to save output and error JSONL files",
    ),
    poll_interval: int = typer.Option(30, "--poll-interval", help="Polling interval in seconds"),
):
    """
    Monitor active batch jobs and download outputs/errors when they complete.
    """
    os.makedirs(output_dir, exist_ok=True)
    monitor_batches(output_dir=str(output_dir), poll_interval=poll_interval)


if __name__ == "__main__":
    app()