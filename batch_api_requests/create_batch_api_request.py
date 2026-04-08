"""Create requests to submit to the OpenAI Batch API for SFT data."""

import json
import os
import random
from pathlib import Path
from typing import Optional

import typer

from utils.common_utils import jlload, read, load_text_lines


def load_prompts_for_batch_api(
    doc_jsonl_path: str = "/gpfs/radev/home/ax46/project/diffusion-rl/data/tinystoriesv2_shuffled/tinystoriesv2.prompt.jsonl",
    doc_start: int = -10000,
    doc_end: Optional[int] = None,
    user_prompt_name: str = "gemini-3-pro-7",
    system_prompt_name: str = "latent-generation",
    few_shot_examples_name: str = "latent-generation",
    output_dir: str = "../data/prompt_data/v1",
    seed: int = 0,
    batch_api_model: str = "gpt-5-nano-2025-08-07",
    reasoning_effort: str = "low",
    verbosity: str = "medium",
) -> None:
    """
    Load prompts for batch API and save to file.

    Args:
        doc_jsonl_path: str, the path to the document JSONL file.
        doc_start: int, the start document index.
        doc_end: int, the end document index.
        user_prompt_name: str, the name of the user prompt file.
        system_prompt_name: str, the name of the system prompt file.
        output_dir: str, the base output directory.
        seed: int, the random seed.
        batch_api_model: str, the batch API model to use.
        reasoning_effort: str, the reasoning effort.
        verbosity: str, the verbosity for the model.
    """
    # Set seed
    random.seed(seed)

    # Load docs
    docs = load_text_lines(doc_jsonl_path)

    if doc_end is None:
        docs = docs[doc_start:]
    else:
        docs = docs[doc_start:doc_end]

    print(
        f"Loaded {len(docs)} docs from {doc_jsonl_path} from {doc_start} to {doc_end if doc_end is not None else 'end'}."
    )

    system_prompt_filename = f"{system_prompt_name}.txt"
    user_prompt_filename = f"{user_prompt_name}.txt"
    system_prompt_path = os.path.join(
        "prompts/system_prompts", system_prompt_filename
    )
    user_prompt_path = os.path.join("prompts/user_prompts", user_prompt_filename)
    system_prompt = read(system_prompt_path)
    user_prompt = read(user_prompt_path)
    prompts = [user_prompt.format(doc=doc["text"]) for doc in docs]
    few_shot_messages = []
    if few_shot_examples_name:
        few_shot_examples_filename = f"{few_shot_examples_name}.jsonl"
        few_shot_examples_path = os.path.join(
            "prompts/few_shot_examples", few_shot_examples_filename
        )
        few_shot_examples = jlload(few_shot_examples_path)
        for example in few_shot_examples:
            few_shot_messages.append({"role": "user", "content": example["user"]})
            few_shot_messages.append({"role": "assistant", "content": example["assistant"]})

    output_path = (
        Path(output_dir)
        / batch_api_model
        / f"docs_{str(len(docs)).zfill(10)}"
        / user_prompt_name
        / system_prompt_name
        / "sft.jsonl"
    )
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)

    print(f"Saving {len(prompts)} prompts to {output_path}")
    request_id = 0
    with open(output_path, "w") as outfile:
        # Create API request objects
        for prompt in prompts:
            messages = [{"role": "developer", "content": system_prompt}]
            messages.extend(few_shot_messages)
            messages.append({"role": "user", "content": prompt})
            request_obj = {
                "custom_id": f"request-{request_id}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": batch_api_model,
                    "messages": messages,
                    "reasoning_effort": reasoning_effort,
                    "verbosity": verbosity,
                },
            }
            json_line = json.dumps(request_obj)
            outfile.write(json_line + "\n")
            request_id += 1

    print(f"Saved {request_id} prompts to {output_path}")


app = typer.Typer(help="Create requests for the OpenAI Batch API for SFT data.")


@app.command()
def create(
    doc_jsonl_path: str = typer.Option(
        "/gpfs/radev/home/ax46/project/diffusion-rl/data/tinystoriesv2_shuffled/tinystoriesv2.prompt.jsonl",
        help="Path to the chunk assignments JSON file.",
    ),
    doc_start: int = typer.Option(-10000, help="Start document index."),
    doc_end: Optional[int] = typer.Option(
        None, help="End document index. None means slice until the end."
    ),
    user_prompt_name: str = typer.Option(
        "gemini-3-pro-7", help="User prompt base filename."
    ),
    system_prompt_name: str = typer.Option(
        "latent-generation",
        help="System prompt base filename.",
    ),
    few_shot_examples_name: Optional[str] = typer.Option(
       "latent-generation", help="Few-shot examples filename (without .jsonl)."
   ),
    output_dir: str = typer.Option("../data/prompt_data/v1", help="Base output directory."),
    seed: int = typer.Option(0, help="Random seed."),
    batch_api_model: str = typer.Option(
        "gpt-5-nano-2025-08-07", help="Batch API model to use."
    ),
    reasoning_effort: str = typer.Option("low", help="Reasoning effort."),
    verbosity: str = typer.Option("medium", help="Verbosity for the model."),
):
    """Generate and write batch API request JSONL for the specified docs."""
    load_prompts_for_batch_api(
        doc_jsonl_path=doc_jsonl_path,
        doc_start=doc_start,
        doc_end=doc_end,
        user_prompt_name=user_prompt_name,
        system_prompt_name=system_prompt_name,
        few_shot_examples_name=few_shot_examples_name,
        output_dir=output_dir,
        seed=seed,
        batch_api_model=batch_api_model,
        reasoning_effort=reasoning_effort,
        verbosity=verbosity,
    )


if __name__ == "__main__":
    app()
