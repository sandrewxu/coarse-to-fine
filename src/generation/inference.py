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
    dp_rank: int,
    dp_size: int,
    dp_master_ip: str,
    dp_master_port: int,
    model_path: str,
    prompts: list[str],
    sampling_kwargs: dict,
    seed: int,
    result_queue,
) -> None:
    """Run a single vLLM engine as DP rank `dp_rank` of `dp_size`.

    Follows vllm/examples/offline_inference/data_parallel.py: each rank sets
    VLLM_DP_* env vars before LLM init, and vLLM coordinates CUDA device
    assignment + collective init across ranks via the master IP/port rendezvous.
    For dp_size==1 we skip DP env vars and pin to GPU 0 manually.
    """
    import os
    import time

    if dp_size > 1:
        os.environ["VLLM_DP_RANK"] = str(dp_rank)
        os.environ["VLLM_DP_RANK_LOCAL"] = str(dp_rank)
        os.environ["VLLM_DP_SIZE"] = str(dp_size)
        os.environ["VLLM_DP_MASTER_IP"] = dp_master_ip
        os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    import sys
    import traceback

    texts: list[str] = []
    try:
        from vllm import LLM, SamplingParams

        from src.common.constants import VLLM_MAX_MODEL_LEN, VLLM_MAX_NUM_SEQS

        llm = LLM(
            model=model_path,
            seed=seed,
            trust_remote_code=True,
            tensor_parallel_size=1,
            max_model_len=VLLM_MAX_MODEL_LEN,
            max_num_seqs=VLLM_MAX_NUM_SEQS,
            gpu_memory_utilization=0.9,
        )
        sampling_params = SamplingParams(**sampling_kwargs)
        tokenizer = llm.get_tokenizer()
        # vLLM rejects any prompt longer than max_model_len at input validation.
        # Drop over-length prompts so the batch survives; return "" for their
        # positions to keep the output list 1-1 with the input.
        max_prompt_tokens = VLLM_MAX_MODEL_LEN - sampling_kwargs["max_tokens"]
        formatted_all = _apply_chat_template(tokenizer, prompts)
        keep_mask = [
            len(tokenizer(t, add_special_tokens=False)["input_ids"]) <= max_prompt_tokens
            for t in formatted_all
        ]
        dropped = sum(1 for k in keep_mask if not k)
        if dropped:
            print(
                f"[DP rank {dp_rank}] skipping {dropped}/{len(prompts)} prompts "
                f"over {max_prompt_tokens} tokens",
                flush=True,
            )
        formatted = [t for t, k in zip(formatted_all, keep_mask, strict=True) if k]
        outputs = llm.generate(formatted, sampling_params)
        gen_texts = iter(o.outputs[0].text for o in outputs)
        texts = [next(gen_texts) if k else "" for k in keep_mask]
    except BaseException:
        # Print with rank prefix so the user can tell which engine failed.
        print(f"[DP rank {dp_rank}] worker crashed:", file=sys.stderr, flush=True)
        traceback.print_exc()
    finally:
        # Always unblock the parent's queue.get(), even on failure.
        try:
            result_queue.put((dp_rank, texts))
        except BaseException:
            traceback.print_exc()
        # vLLM docs: give engines a moment to pause their loops before exit.
        time.sleep(1)
        # Kill the EngineCore subprocess so the fork-inherited sentinel FD
        # closes — otherwise main's p.join() hangs forever on select().
        import contextlib
        import signal
        from pathlib import Path

        for path in Path(f"/proc/{os.getpid()}/task").glob("*/children"):
            with contextlib.suppress(OSError), path.open() as f:
                for pid in f.read().split():
                    with contextlib.suppress(OSError, ValueError):
                        os.kill(int(pid), signal.SIGKILL)
        # Skip Python atexit to avoid vLLM V1 teardown deadlock on _MPClient join.
        os._exit(0)


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
    """Generate using vLLM with native data parallelism across ``num_gpus`` GPUs.

    Spawns one engine subprocess per DP rank. Ranks share a master IP/port so
    vLLM can coordinate init and collective ops — the failure mode we hit
    before (silent ZMQ-drop on one rank) came from running 8 *uncoordinated*
    LLM instances, which vLLM doesn't support. Placeholder prompts are fed to
    ranks whose slice of ``prompts`` is empty, because every DP rank must call
    ``.generate()`` or the group hangs.
    """
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)

    sampling_kwargs = dict(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )

    if num_gpus > 1:
        from vllm.utils import get_open_port

        dp_master_ip = "127.0.0.1"
        dp_master_port = get_open_port()
    else:
        dp_master_ip = ""
        dp_master_port = 0

    per_rank = (len(prompts) + num_gpus - 1) // num_gpus
    prompt_chunks: list[list[str]] = []
    for rank in range(num_gpus):
        chunk = prompts[rank * per_rank : (rank + 1) * per_rank]
        if not chunk:
            chunk = ["placeholder"]
        prompt_chunks.append(chunk)

    log.info(
        f"  vLLM DP={num_gpus}: {len(prompts)} prompts"
        + (f" (master {dp_master_ip}:{dp_master_port})" if num_gpus > 1 else "")
    )

    result_queue = mp.Queue()
    procs = []
    for rank, chunk in enumerate(prompt_chunks):
        p = mp.Process(
            target=_vllm_worker,
            args=(
                rank,
                num_gpus,
                dp_master_ip,
                dp_master_port,
                model_path,
                chunk,
                sampling_kwargs,
                seed,
                result_queue,
            ),
        )
        p.start()
        procs.append(p)

    collected: dict[int, list[str]] = {}
    for _ in procs:
        rank, texts = result_queue.get()
        collected[rank] = texts

    for p in procs:
        p.join(timeout=300)
        if p.exitcode is None:
            log.warning(f"  Killing non-exiting DP worker pid={p.pid}")
            p.kill()

    # Drop placeholder outputs from ranks whose original slice was empty.
    out: list[str] = []
    for rank in range(num_gpus):
        start = rank * per_rank
        if start >= len(prompts):
            continue
        out.extend(collected[rank])
    return out


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
