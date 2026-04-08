"""Analyze batch API output for token usage and cost estimation."""

import json
from pathlib import Path
from typing import Optional

import typer


def analyze_batch_output(
    output_file: Path,
    input_price_per_m: float = 0.025,
    cached_price_per_m: float = 0.0025,
    output_price_per_m: float = 0.20,
):
    """
    Analyze a batch API output file for token usage and cost.
    
    Args:
        output_file: Path to the batch API output JSONL file
        input_price_per_m: Price per million regular input tokens (default: $0.025)
        cached_price_per_m: Price per million cached input tokens (default: $0.0025)
        output_price_per_m: Price per million output tokens (default: $0.20)
    """
    # Statistics
    total_requests = 0
    requests_with_cache = 0
    
    total_prompt_tokens = 0
    total_cached_tokens = 0
    total_completion_tokens = 0
    total_reasoning_tokens = 0
    
    completion_token_list = []
    
    # Read the output file
    with open(output_file, "r") as f:
        for line in f:
            data = json.loads(line)
            
            # Check if request was successful
            if data.get("error"):
                continue
                
            total_requests += 1
            
            # Extract usage data
            usage = data["response"]["body"]["usage"]
            prompt_tokens = usage["prompt_tokens"]
            completion_tokens = usage["completion_tokens"]
            cached_tokens = usage["prompt_tokens_details"]["cached_tokens"]
            reasoning_tokens = usage["completion_tokens_details"]["reasoning_tokens"]
            
            # Accumulate statistics
            total_prompt_tokens += prompt_tokens
            total_cached_tokens += cached_tokens
            total_completion_tokens += completion_tokens
            total_reasoning_tokens += reasoning_tokens
            
            completion_token_list.append(completion_tokens)
            
            if cached_tokens > 0:
                requests_with_cache += 1
    
    # Calculate derived metrics
    total_uncached_tokens = total_prompt_tokens - total_cached_tokens
    avg_completion_tokens = sum(completion_token_list) / len(completion_token_list) if completion_token_list else 0
    
    # Calculate costs (per million tokens)
    cost_uncached_input = (total_uncached_tokens / 1_000_000) * input_price_per_m
    cost_cached_input = (total_cached_tokens / 1_000_000) * cached_price_per_m
    cost_output = (total_completion_tokens / 1_000_000) * output_price_per_m
    total_cost = cost_uncached_input + cost_cached_input + cost_output
    
    # Print results
    print("=" * 80)
    print("BATCH API COST ANALYSIS")
    print("=" * 80)
    print(f"\nFile: {output_file}")
    print(f"\n{'USAGE STATISTICS':-^80}")
    print(f"Total Requests:                  {total_requests:,}")
    print(f"Requests with Cached Tokens:     {requests_with_cache:,} ({100 * requests_with_cache / total_requests:.1f}%)")
    print(f"Requests without Cache:          {total_requests - requests_with_cache:,} ({100 * (total_requests - requests_with_cache) / total_requests:.1f}%)")
    
    print(f"\n{'TOKEN COUNTS':-^80}")
    print(f"Total Prompt Tokens:             {total_prompt_tokens:,}")
    print(f"  - Uncached:                    {total_uncached_tokens:,} ({100 * total_uncached_tokens / total_prompt_tokens:.1f}%)")
    print(f"  - Cached:                      {total_cached_tokens:,} ({100 * total_cached_tokens / total_prompt_tokens:.1f}%)")
    print(f"Total Completion Tokens:         {total_completion_tokens:,}")
    print(f"  - Regular Output:              {total_completion_tokens - total_reasoning_tokens:,}")
    print(f"  - Reasoning Tokens:            {total_reasoning_tokens:,}")
    print(f"Average Completion Tokens:       {avg_completion_tokens:.1f}")
    print(f"Total Tokens (Prompt + Output):  {total_prompt_tokens + total_completion_tokens:,}")
    
    print(f"\n{'COST BREAKDOWN':-^80}")
    print(f"Pricing:")
    print(f"  - Regular Input:  ${input_price_per_m:.4f} / 1M tokens")
    print(f"  - Cached Input:   ${cached_price_per_m:.4f} / 1M tokens")
    print(f"  - Output:         ${output_price_per_m:.4f} / 1M tokens")
    print()
    print(f"Costs:")
    print(f"  - Uncached Input:  ${cost_uncached_input:>10.4f}  ({total_uncached_tokens:,} tokens)")
    print(f"  - Cached Input:    ${cost_cached_input:>10.4f}  ({total_cached_tokens:,} tokens)")
    print(f"  - Output:          ${cost_output:>10.4f}  ({total_completion_tokens:,} tokens)")
    print(f"  {'─' * 40}")
    print(f"  - TOTAL COST:      ${total_cost:>10.4f}")
    
    # Calculate savings from caching
    cost_without_cache = ((total_prompt_tokens) / 1_000_000) * input_price_per_m + cost_output
    savings = cost_without_cache - total_cost
    savings_pct = (savings / cost_without_cache * 100) if cost_without_cache > 0 else 0
    
    print(f"\n{'CACHING BENEFITS':-^80}")
    print(f"Cost without caching:            ${cost_without_cache:.4f}")
    print(f"Cost with caching:               ${total_cost:.4f}")
    print(f"Savings from caching:            ${savings:.4f} ({savings_pct:.1f}%)")
    print("=" * 80)
    
    return {
        "total_requests": total_requests,
        "requests_with_cache": requests_with_cache,
        "total_prompt_tokens": total_prompt_tokens,
        "total_cached_tokens": total_cached_tokens,
        "total_uncached_tokens": total_uncached_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_reasoning_tokens": total_reasoning_tokens,
        "avg_completion_tokens": avg_completion_tokens,
        "cost_uncached_input": cost_uncached_input,
        "cost_cached_input": cost_cached_input,
        "cost_output": cost_output,
        "total_cost": total_cost,
        "savings": savings,
    }


app = typer.Typer(help="Analyze batch API output for token usage and cost.")


@app.command()
def analyze(
    output_file: Path = typer.Argument(
        "../data/sft_data_out/v1/model__gpt-5-nano-2025-08-07/reasoning_effort__low/run_tag__latent_generation_10k_v1/system_prompt_name__latent-generation/user__ax46/user_prompt_name__gemini-3-pro-7/verbosity__medium/batch_6971db0456e081908c39483dd5230333/output.jsonl",
        help="Path to the batch API output JSONL file.",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
    input_price: float = typer.Option(0.025, "--input-price", help="Price per 1M regular input tokens ($)."),
    cached_price: float = typer.Option(0.0025, "--cached-price", help="Price per 1M cached input tokens ($)."),
    output_price: float = typer.Option(0.20, "--output-price", help="Price per 1M output tokens ($)."),
    save_json: Optional[Path] = typer.Option(None, "--save-json", help="Save statistics to JSON file."),
):
    """Analyze batch API output for token usage and cost estimation."""
    stats = analyze_batch_output(
        output_file=output_file,
        input_price_per_m=input_price,
        cached_price_per_m=cached_price,
        output_price_per_m=output_price,
    )
    
    if save_json:
        save_json.parent.mkdir(parents=True, exist_ok=True)
        with open(save_json, "w") as f:
            json.dump(stats, f, indent=2)
        typer.secho(f"\nStatistics saved to {save_json}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
