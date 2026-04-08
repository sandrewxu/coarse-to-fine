import json
from pathlib import Path

import typer
from datasets import Dataset, DatasetDict
from huggingface_hub import create_repo


def prepare_hf_dataset(input_file, output_file):
    """
    Extract the document and completion from batch API files and create a HuggingFace dataset.
    """
    # Read input and output files
    inputs = []
    with open(input_file, "r") as f:
        for line in f:
            data = json.loads(line)
            messages = data["body"]["messages"]
            user_message = messages[-1]["content"]
            inputs.append(user_message)

    outputs = []
    with open(output_file, "r") as f:
        for line in f:
            data = json.loads(line)
            completion = data["response"]["body"]["choices"][0]["message"]["content"]
            outputs.append(completion)

    assert len(inputs) == len(outputs), "Number of inputs and outputs must match"

    return Dataset.from_dict({"input": inputs, "output": outputs})


def upload_to_huggingface(dataset, repo_name):
    """
    Upload the dataset to HuggingFace.
    """
    # Create repository if it doesn't exist
    try:
        create_repo(repo_name, repo_type="dataset", private=True)
    except Exception as e:
        print(f"Repository might already exist: {e}")

    # Create a DatasetDict with a single split
    dataset_dict = DatasetDict({"train": dataset})

    # Push to hub
    dataset_dict.push_to_hub(repo_name, private=True)


app = typer.Typer(help="Upload batch API results to HuggingFace as a dataset.")


@app.command()
def upload(
    input_file: Path = typer.Argument(
        "../data/prompt_data/v1/gpt-5-nano-2025-08-07/docs_0000010000/gemini-3-pro-7/latent-generation/sft.jsonl",
        help="Path to the original input JSONL file.",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
    output_file: Path = typer.Argument(
        "../data/sft_data_out/v1/model__gpt-5-nano-2025-08-07/reasoning_effort__low/run_tag__latent_generation_10k_v1/system_prompt_name__latent-generation/user__ax46/user_prompt_name__gemini-3-pro-7/verbosity__medium/batch_6971db0456e081908c39483dd5230333/output.jsonl",
        help="Path to the batch API output file.",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
    repo_name: str = typer.Argument(
        "sandrewxu/tinystories_latent_generation_10k_v1", help='HuggingFace repository name (e.g., "username/dataset-name").'
    ),
):
    if not input_file.exists():
        typer.secho(f"Input file not found: {input_file}", fg=typer.colors.RED)
        raise typer.Exit(1)
    if not output_file.exists():
        typer.secho(f"Output file not found: {output_file}", fg=typer.colors.RED)
        raise typer.Exit(1)

    typer.echo("Preparing dataset...")
    dataset = prepare_hf_dataset(input_file, output_file)

    typer.echo("\nDataset preview:")
    typer.echo(str(dataset))

    typer.echo(f"\nUploading to HuggingFace repository: {repo_name}")
    upload_to_huggingface(dataset, repo_name)

    typer.echo("\nUpload complete! You can now access your dataset on HuggingFace.")


if __name__ == "__main__":
    app()
