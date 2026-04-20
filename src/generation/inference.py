"""
Autoregressive generation from an SFT model.

Two backends: vLLM (fast, batched) and HuggingFace (simple, no extra dependency).
Both apply the model's chat template before generating.
"""

from src.common.logging import get_logger

log = get_logger(__name__)


def _apply_chat_template(tokenizer, prompts: list[str]) -> list[str]:
    """Format each prompt as a chat message matching the SFT training format."""
    formatted = []
    for text in prompts:
        messages = [{"role": "user", "content": text}]
        formatted.append(
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        )
    return formatted


def _vllm_worker(
    gpu_id: int, model_path: str, prompts: list[str], sampling_kwargs: dict, seed: int, result_queue
) -> None:
    """Run vLLM on a single GPU. Puts (gpu_id, results) on the queue."""
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from vllm import LLM, SamplingParams

    from src.common.constants import VLLM_MAX_MODEL_LEN, VLLM_MAX_NUM_SEQS

    llm = LLM(
        model=model_path,
        seed=seed,
        trust_remote_code=True,
        max_model_len=VLLM_MAX_MODEL_LEN,
        max_num_seqs=VLLM_MAX_NUM_SEQS,
        gpu_memory_utilization=0.9,
    )
    sampling_params = SamplingParams(**sampling_kwargs)
    tokenizer = llm.get_tokenizer()
    formatted = _apply_chat_template(tokenizer, prompts)
    outputs = llm.generate(formatted, sampling_params)
    result_queue.put((gpu_id, [o.outputs[0].text for o in outputs]))


def generate_vllm(
    model_path: str,
    prompts: list[str],
    *,
    num_gpus: int = 1,
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = -1,
    repetition_penalty: float = 1.0,
    seed: int = 42,
) -> list[str]:
    """Generate using vLLM (high-throughput batched inference)."""
    sampling_kwargs = dict(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )

    if num_gpus <= 1:
        import multiprocessing as mp

        mp.set_start_method("spawn", force=True)
        queue = mp.Queue()
        _vllm_worker(0, model_path, prompts, sampling_kwargs, seed, queue)
        _, results = queue.get()
        return results

    # Data parallelism: split prompts across GPUs using non-daemon Process
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)

    chunk_size = (len(prompts) + num_gpus - 1) // num_gpus
    prompt_chunks = [prompts[i : i + chunk_size] for i in range(0, len(prompts), chunk_size)]

    log.info(f"  Data-parallel: splitting {len(prompts)} prompts across {len(prompt_chunks)} GPUs")
    result_queue = mp.Queue()
    procs = []
    for gpu_id, chunk in enumerate(prompt_chunks):
        p = mp.Process(
            target=_vllm_worker,
            args=(gpu_id, model_path, chunk, sampling_kwargs, seed, result_queue),
        )
        p.start()
        procs.append(p)

    # Collect results in gpu_id order
    collected = {}
    for _ in procs:
        gpu_id, texts = result_queue.get()
        collected[gpu_id] = texts

    for p in procs:
        p.join()

    return [text for gpu_id in sorted(collected) for text in collected[gpu_id]]


def generate_hf(
    model_path: str,
    prompts: list[str],
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.0,
    batch_size: int = 8,
) -> list[str]:
    """Generate using HuggingFace transformers (simple, no vLLM dependency)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        .to(device)
        .eval()
    )

    formatted = _apply_chat_template(tokenizer, prompts)
    results = []

    for i in range(0, len(formatted), batch_size):
        batch = formatted[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to(device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=temperature > 0,
            )

        for ids in output_ids:
            text = tokenizer.decode(ids[input_len:], skip_special_tokens=True)
            results.append(text)

        if (i // batch_size + 1) % 10 == 0:
            log.info(f"  Generated {min(i + batch_size, len(formatted))}/{len(formatted)}")

    return results


def generate(
    backend: str,
    model_path: str,
    prompts: list[str],
    **kwargs,
) -> list[str]:
    """
    Generate from an SFT model using the specified backend.

    Args:
        backend: "vllm" or "hf"
        model_path: Path to the SFT model checkpoint
        prompts: List of text prompts
        **kwargs: Passed to the backend function (max_tokens, temperature, etc.)

    Returns:
        List of generated text strings
    """
    if backend == "vllm":
        return generate_vllm(model_path, prompts, **kwargs)
    elif backend == "hf":
        # Translate vLLM-style kwargs to HF-style
        if "max_tokens" in kwargs:
            kwargs["max_new_tokens"] = kwargs.pop("max_tokens")
        if kwargs.get("top_k", 0) <= 0:
            kwargs.pop("top_k", None)
        kwargs.pop("num_gpus", None)
        kwargs.pop("seed", None)
        return generate_hf(model_path, prompts, **kwargs)
    else:
        raise ValueError(f"Unknown backend '{backend}'. Use 'vllm' or 'hf'.")
