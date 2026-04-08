"""Analyze batch API output for token usage and cost estimation."""

import json
from pathlib import Path
from typing import Any


def analyze_batch_output(
    output_file: Path,
    input_price_per_m: float = 0.025,
    cached_price_per_m: float = 0.0025,
    output_price_per_m: float = 0.20,
) -> dict[str, Any]:
    """Analyze a batch API output JSONL file for token usage and cost.

    Args:
        output_file: Path to the batch API output JSONL file.
        input_price_per_m: Price per million regular input tokens.
        cached_price_per_m: Price per million cached input tokens.
        output_price_per_m: Price per million output tokens.

    Returns:
        Dict with token counts and cost breakdown.
    """
    total_requests = 0
    requests_with_cache = 0
    total_prompt_tokens = 0
    total_cached_tokens = 0
    total_completion_tokens = 0
    total_reasoning_tokens = 0
    completion_token_list: list[int] = []

    with open(output_file) as f:
        for line in f:
            data = json.loads(line)
            if data.get("error"):
                continue

            total_requests += 1
            usage = data["response"]["body"]["usage"]
            prompt_tokens = usage["prompt_tokens"]
            completion_tokens = usage["completion_tokens"]
            cached_tokens = usage["prompt_tokens_details"]["cached_tokens"]
            reasoning_tokens = usage["completion_tokens_details"]["reasoning_tokens"]

            total_prompt_tokens += prompt_tokens
            total_cached_tokens += cached_tokens
            total_completion_tokens += completion_tokens
            total_reasoning_tokens += reasoning_tokens
            completion_token_list.append(completion_tokens)

            if cached_tokens > 0:
                requests_with_cache += 1

    total_uncached_tokens = total_prompt_tokens - total_cached_tokens
    avg_completion = sum(completion_token_list) / len(completion_token_list) if completion_token_list else 0

    cost_uncached = (total_uncached_tokens / 1_000_000) * input_price_per_m
    cost_cached = (total_cached_tokens / 1_000_000) * cached_price_per_m
    cost_output = (total_completion_tokens / 1_000_000) * output_price_per_m
    total_cost = cost_uncached + cost_cached + cost_output

    cost_no_cache = (total_prompt_tokens / 1_000_000) * input_price_per_m + cost_output
    savings = cost_no_cache - total_cost

    # Print report
    print(f"{'=' * 60}")
    print("BATCH API COST ANALYSIS")
    print(f"{'=' * 60}")
    print(f"\nFile: {output_file}")
    print(f"Total Requests: {total_requests:,}")
    if total_requests > 0:
        print(f"Requests with Cache: {requests_with_cache:,} ({100 * requests_with_cache / total_requests:.1f}%)")
    print(f"\nPrompt Tokens: {total_prompt_tokens:,} (uncached: {total_uncached_tokens:,}, cached: {total_cached_tokens:,})")
    print(f"Completion Tokens: {total_completion_tokens:,} (reasoning: {total_reasoning_tokens:,})")
    print(f"Avg Completion: {avg_completion:.1f}")
    print(f"\nCost: ${total_cost:.4f} (savings from cache: ${savings:.4f})")
    print(f"{'=' * 60}")

    return {
        "total_requests": total_requests,
        "requests_with_cache": requests_with_cache,
        "total_prompt_tokens": total_prompt_tokens,
        "total_cached_tokens": total_cached_tokens,
        "total_uncached_tokens": total_uncached_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_reasoning_tokens": total_reasoning_tokens,
        "avg_completion_tokens": avg_completion,
        "cost_uncached_input": cost_uncached,
        "cost_cached_input": cost_cached,
        "cost_output": cost_output,
        "total_cost": total_cost,
        "savings": savings,
    }
