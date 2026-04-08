import json
from pathlib import Path
from typing import Optional

import httpx
import typer
from openai import OpenAI

from utils.common_utils import jload


def upload_file(client, file_path):
    """Upload a file to OpenAI for batch processing."""
    try:
        with open(file_path, "rb") as f:
            response = client.files.create(file=f, purpose="batch")

        typer.echo(f"File uploaded successfully. File ID: {response.id}")
        return response.id
    except Exception as e:
        typer.secho(f"Error uploading file: {e}", fg=typer.colors.RED)
        raise typer.Exit(1)


def submit_batch_request(client, input_file_id, metadata):
    """Submit a batch processing request."""
    try:
        batch = client.batches.create(
            input_file_id=input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata=metadata,
        )
        return batch
    except Exception as e:
        typer.secho(f"Error submitting batch request: {e}", fg=typer.colors.RED)
        raise typer.Exit(1)


app = typer.Typer(help="Submit a batch processing job to OpenAI.")


@app.command()
def submit(
    file_path: Path = typer.Option(
        "../data/prompt_data/v1/gpt-5-nano-2025-08-07/docs_0000010000/gemini-3-pro-7/latent-generation/sft.jsonl",
        help="Path to the JSONL file containing the requests.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    model: str = typer.Option("gpt-5-nano-2025-08-07", help="Model to use."),
    user: str = typer.Option("ax46", help="User identifier."),
    run_tag: str = typer.Option("latent_generation_10k_v1", "--run-tag", help="Run tag for the batch."),
    extra_metadata: Optional[str] = typer.Option(
        """{"reasoning_effort": "low", "verbosity": "medium", "user_prompt_name": "gemini-3-pro-7", "system_prompt_name": "latent-generation"}""", 
        help="Additional metadata as JSON string."
    ),
):
    """Submit a batch processing job to OpenAI."""
    # Load API key and initialize client
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

    # Prepare metadata
    metadata = {"model": model, "user": user, "run_tag": run_tag}

    # Add any extra metadata if provided
    if extra_metadata:
        try:
            extra = json.loads(extra_metadata)
            metadata.update(extra)
        except json.JSONDecodeError:
            typer.secho(
                "Error: --extra-metadata must be a valid JSON string.",
                fg=typer.colors.RED,
            )
            raise typer.Exit(1)

    # Upload file
    typer.echo(f"Uploading file: {file_path}")
    file_id = upload_file(client, file_path)

    # Submit batch request
    typer.echo("Submitting batch request...")
    batch = submit_batch_request(client, file_id, metadata)

    # Print results
    typer.echo("\nBatch submitted successfully!")
    typer.echo(f"Batch ID: {batch.id}")
    typer.echo(f"Status: {batch.status}")
    typer.echo(f"Input File ID: {batch.input_file_id}")
    typer.echo(f"Metadata: {json.dumps(metadata, indent=2)}")


if __name__ == "__main__":
    app()
