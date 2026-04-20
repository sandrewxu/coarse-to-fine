"""Reward manager for joint ELBO training (posterior + decoder, simultaneous).

Like :class:`C2FRewardManager` but ``p_θ`` is **trainable**. Concurrent
``run_single`` coroutines (spawned by veRL's agent_loop, one per rollout sample)
enqueue their C2F inputs onto a shared queue. A single flusher coroutine
coalesces the queue on a small time window and runs **one** batched
forward+backward+optimizer step per flush, scoring all samples at the same
``θ`` and taking one update on the averaged gradient. Validation samples
(``is_validation == True``) are scored without a grad step.

Batching is controlled by ``rl.joint.c2f_batch_window`` (seconds). Set to
``0.0`` to fall back to per-sample updates (legacy behaviour).

The math delta vs. per-sample updates:
  - REINFORCE gradient remains unbiased (rewards are ``.detach()``ed).
  - Rewards have lower variance because all samples in a flush see the same
    ``θ``, so the reward baseline is consistent.
  - The p_θ trajectory differs: N AdamW steps at bs=1 ≠ 1 AdamW step at bs=N.
    Functionally equivalent at small lr / long horizon; users may need to
    re-sweep ``c2f_lr`` when switching from the legacy per-sample path.
"""

import asyncio
import gc
import os
import shutil
from pathlib import Path

import torch
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase

try:
    # Stdlib on POSIX; absent on Windows. Only used for a DEBUG-level RSS log
    # per flush, so a missing module is a no-op rather than a failure.
    import resource as _resource
except ImportError:  # pragma: no cover - Windows path, not our target
    _resource = None  # type: ignore[assignment]

from src.common.logging import get_logger
from src.rl.common import (
    build_c2f_input,
    load_c2f_components,
    load_exp_config,
    parse_layers,
)

log = get_logger(__name__)


class JointC2FRewardManager(RewardManagerBase):
    """Reward = ``-CE_loss`` from a *trainable* C2F; updates p_θ in place."""

    def __init__(
        self,
        config,
        tokenizer,
        compute_score=None,
        reward_router_address=None,
        reward_model_tokenizer=None,
        **kwargs,
    ) -> None:
        super().__init__(config, tokenizer)
        self.compute_score = compute_score

        exp_config = load_exp_config()
        joint_cfg = exp_config.get("rl", {}).get("joint", {})
        c2f_mask_type = joint_cfg.get("c2f_mask_type", "causal")

        self.components = load_c2f_components(
            exp_config,
            sft_tokenizer=tokenizer,
            c2f_section_key="joint",
            c2f_mask_type=c2f_mask_type,
        )

        # ── Move C2F to CUDA and put in train mode ─────────────────────────
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        c2f_model = self.components.c2f_model
        c2f_model.to(self.device)
        c2f_model.train()

        # ── Phase-specific config + optimiser + checkpoint state ────────────
        self.malformed_reward: float = float(joint_cfg.get("malformed_reward", -10.0))
        c2f_lr: float = float(joint_cfg.get("c2f_lr", 1e-4))
        c2f_wd: float = float(joint_cfg.get("c2f_weight_decay", 0.01))
        self._save_steps: int = int(joint_cfg.get("c2f_save_steps", 100))
        self._keep_last_n: int = int(joint_cfg.get("c2f_keep_last_n", 3))
        self._micro_bs: int = int(joint_cfg.get("c2f_micro_batch_size", 32))

        save_dir_base = Path(joint_cfg.get("c2f_save_dir", "checkpoints/rl/joint/c2f"))
        self._save_dir = save_dir_base / f"worker_{os.getpid()}"
        self._save_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = torch.optim.AdamW(c2f_model.parameters(), lr=c2f_lr, weight_decay=c2f_wd)
        self._step = 0
        self._last_save_step = 0

        # ── Dynamic batching state ─────────────────────────────────────────
        # Concurrent ``run_single`` coroutines enqueue onto ``_queue`` and
        # await a per-sample ``future``. One flusher coroutine drains the
        # queue on a ``_batch_window``-second cadence and runs a single
        # batched fwd+bwd+step, resolving all futures together.
        self._batch_window: float = float(joint_cfg.get("c2f_batch_window", 0.05))
        self._gc_every: int = int(joint_cfg.get("c2f_gc_every", 50))
        self._queue: list[dict] = []
        self._queue_lock = asyncio.Lock()
        self._flusher_running: bool = False
        self._flusher_task: asyncio.Task | None = None
        self._flush_count: int = 0

        log.info(
            "JointC2FRewardManager ready. p trainable on %s, lr=%s, "
            "malformed_reward=%s, batch_window=%ss, micro_bs=%d, gc_every=%d",
            self.device,
            c2f_lr,
            self.malformed_reward,
            self._batch_window,
            self._micro_bs,
            self._gc_every,
        )

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _build_input(
        self, layer_contents: list[str], prompt: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        c = self.components
        return build_c2f_input(
            layer_contents,
            prompt,
            scale_lengths=c.scale_lengths,
            word_boundaries=c.word_boundaries,
            space_tokenizer=c.space_tokenizer,
            bos_id=c.bos_id,
            pad_id=c.pad_id,
            seq_len=c.seq_len,
            label_strategy="joint",
        )

    def _ce_per_sample(self, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Cross-entropy per sample, accounting for the configured mask_type.

        Returns a 1-D tensor of shape ``[B]`` with mean per-token loss per sample.
        """
        c2f_model = self.components.c2f_model
        outputs = c2f_model(input_ids=input_ids)
        logits = outputs.logits

        if c2f_model.config.mask_type == "causal":
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
        else:
            shift_logits = logits
            shift_labels = labels

        per_token_loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view(shift_labels.shape)

        mask = shift_labels != -100
        num_tokens = mask.sum(dim=1).clamp(min=1)
        return (per_token_loss * mask).sum(dim=1) / num_tokens

    def _save_checkpoint(self) -> None:
        save_path = self._save_dir / f"step_{self._step}"
        try:
            save_path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            log.warning("checkpoint dir create failed at %s: %r", save_path, e)
            return

        # Save model weights (safetensors via HF). Treat this as the primary
        # artifact — if it fails we abandon the whole step.
        # Catch broadly: safetensors raises its own SafetensorError on disk
        # quota / I/O failures, and torch raises RuntimeError on
        # serialization corruption. Either way we want to keep training, not
        # crash the whole RL run on a checkpoint failure.
        try:
            self.components.c2f_model.save_pretrained(str(save_path))
        except Exception as e:
            log.warning(
                "model save failed at %s: %r. Cleaning up partial save and continuing.",
                save_path,
                e,
            )
            shutil.rmtree(save_path, ignore_errors=True)
            return

        # Save optimizer state separately. Use legacy (non-zip) serialization
        # to dodge a torch>=2.9 zip-container bug that fires on Adam states
        # after the moments are populated. If this still fails, log and keep
        # the model weights — optimizer state is recoverable from the saved
        # weights (with a warm restart cost).
        opt_path = save_path / "optimizer.pt"
        try:
            torch.save(
                self.optimizer.state_dict(),
                opt_path,
                _use_new_zipfile_serialization=False,
            )
        except Exception as e:
            log.warning(
                "optimizer save failed at %s: %r. Keeping model weights, "
                "optimizer will be reinitialised on resume.",
                opt_path,
                e,
            )
            opt_path.unlink(missing_ok=True)

        log.info("Saved p checkpoint: %s", save_path)

        # Prune older checkpoints, keep last N.
        step_dirs = sorted(
            (
                p
                for p in self._save_dir.iterdir()
                if p.is_dir()
                and p.name.startswith("step_")
                and p.name.removeprefix("step_").isdigit()
            ),
            key=lambda p: int(p.name.removeprefix("step_")),
        )
        for old_dir in step_dirs[: -self._keep_last_n]:
            shutil.rmtree(old_dir, ignore_errors=True)

    # ── veRL interface ──────────────────────────────────────────────────────

    def __call__(self, data, return_dict: bool = False):
        c = self.components
        batch_size: int = data.batch["responses"].shape[0]
        response_len: int = data.batch["responses"].shape[1]

        response_strs: list[str] = c.sft_tokenizer.batch_decode(
            data.batch["responses"], skip_special_tokens=True
        )
        ground_truths = data.non_tensor_batch.get("ground_truth", [""] * batch_size)
        if hasattr(ground_truths, "tolist"):
            ground_truths = ground_truths.tolist()

        # ── Phase 1: parse responses, collect valid C2F inputs ─────────────
        valid_indices: list[int] = []
        all_input_ids: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []

        for i, (response, prompt) in enumerate(zip(response_strs, ground_truths, strict=False)):
            layer_contents = parse_layers(
                response, c.word_count_constraints, strict=c.strict_word_count
            )
            if layer_contents is None:
                continue
            input_ids, labels = self._build_input(layer_contents, str(prompt))
            valid_indices.append(i)
            all_input_ids.append(input_ids)
            all_labels.append(labels)

        reward_tensor = torch.zeros(batch_size, response_len, dtype=torch.float32)
        num_valid = len(valid_indices)
        num_malformed = batch_size - num_valid

        # ── Phase 2: micro-batched forward + backward on p ─────────────────
        # Accumulate grads per micro-batch (each scaled by 1/num_valid) so
        # peak memory holds one micro-batch's activations, not all of them.
        # Mathematically equivalent to a single ``.mean().backward()`` call
        # over the full concatenated loss.
        per_sample_reward: torch.Tensor | None = None
        avg_loss: float = 0.0
        if num_valid > 0:
            batch_ids = torch.stack(all_input_ids).to(self.device)
            batch_labels = torch.stack(all_labels).to(self.device)

            self.optimizer.zero_grad()
            all_per_sample_loss: list[torch.Tensor] = []
            for mb_start in range(0, num_valid, self._micro_bs):
                mb_loss = self._ce_per_sample(
                    batch_ids[mb_start : mb_start + self._micro_bs],
                    batch_labels[mb_start : mb_start + self._micro_bs],
                )
                all_per_sample_loss.append(mb_loss.detach())
                (mb_loss.sum() / num_valid).backward()
            self.optimizer.step()

            per_sample_loss = torch.cat(all_per_sample_loss)
            per_sample_reward = -per_sample_loss.cpu()
            avg_loss = per_sample_loss.mean().item()

            pad_id = getattr(c.sft_tokenizer, "pad_token_id", None)
            for j, idx in enumerate(valid_indices):
                response_ids = data.batch["responses"][idx]
                if pad_id is not None:
                    non_pad = (response_ids != pad_id).nonzero(as_tuple=True)[0]
                else:
                    non_pad = torch.arange(response_len)
                last_pos = int(non_pad[-1].item()) if len(non_pad) > 0 else response_len - 1
                reward_tensor[idx, last_pos] = per_sample_reward[j]

        # ── Phase 3: malformed samples get the malformed-reward floor ──────
        if num_malformed > 0:
            malformed_indices = set(range(batch_size)) - set(valid_indices)
            pad_id = getattr(c.sft_tokenizer, "pad_token_id", None)
            for idx in malformed_indices:
                response_ids = data.batch["responses"][idx]
                if pad_id is not None:
                    non_pad = (response_ids != pad_id).nonzero(as_tuple=True)[0]
                else:
                    non_pad = torch.arange(response_len)
                last_pos = int(non_pad[-1].item()) if len(non_pad) > 0 else response_len - 1
                reward_tensor[idx, last_pos] = self.malformed_reward

        data.batch["token_level_scores"] = reward_tensor

        # ── Logging + checkpointing ────────────────────────────────────────
        self._step += 1
        avg_reward = per_sample_reward.mean().item() if per_sample_reward is not None else 0.0
        log.info(
            "[Joint] step=%d valid=%d/%d p_loss=%.4f reward=%.4f",
            self._step,
            num_valid,
            batch_size,
            avg_loss,
            avg_reward,
        )
        if self._step - self._last_save_step >= self._save_steps:
            self._last_save_step = self._step
            self._save_checkpoint()

        if return_dict:
            return {"reward_tensor": reward_tensor}
        return data

    # ── Dynamic batching: flusher + processor ───────────────────────────────

    def _process_batch(self, batch_items: list[dict]) -> None:
        """Run one batched fwd (+ bwd + step for training samples) on ``p_θ``.

        Called from :meth:`_flusher` on the reward-worker event loop; the body
        is synchronous CUDA work that blocks the loop while it runs. Resolves
        each item's ``future`` with ``{"reward": float, "p_loss": float}``.

        Train vs. validation samples are processed separately: training
        samples contribute to a single gradient-accumulated optimizer step;
        validation samples get a no-grad forward only. Validation output
        doesn't update ``_step`` or trigger a checkpoint.
        """
        if not batch_items:
            return

        c = self.components
        c2f_model = c.c2f_model

        train_items = [it for it in batch_items if not it["is_validate"]]
        val_items = [it for it in batch_items if it["is_validate"]]

        # ── TRAIN path: one optimizer step on averaged gradient ──────────
        if train_items:
            c2f_model.train()
            n_train = len(train_items)
            train_ids = torch.stack([it["input_ids"] for it in train_items]).to(self.device)
            train_labels = torch.stack([it["labels"] for it in train_items]).to(self.device)

            self.optimizer.zero_grad()
            train_losses: list[torch.Tensor] = []
            # Gradient accumulation across micro-batches is equivalent to
            # computing per_sample_loss.mean().backward() on the full batch
            # (sum of per-sample grads scaled by 1/n_train), but caps peak
            # activation memory to one micro-batch.
            for mb_start in range(0, n_train, self._micro_bs):
                mb_end = mb_start + self._micro_bs
                mb_loss = self._ce_per_sample(
                    train_ids[mb_start:mb_end],
                    train_labels[mb_start:mb_end],
                )
                train_losses.append(mb_loss.detach())
                (mb_loss.sum() / n_train).backward()
            self.optimizer.step()

            train_loss_cpu = torch.cat(train_losses).cpu()
            for i, it in enumerate(train_items):
                loss_i = float(train_loss_cpu[i].item())
                if not it["future"].done():
                    it["future"].set_result({"reward": -loss_i, "p_loss": loss_i})

            self._step += n_train
            avg_train_loss = float(train_loss_cpu.mean().item())
            log.info(
                "[Joint] step=%d train_batch=%d p_loss=%.4f reward=%.4f",
                self._step,
                n_train,
                avg_train_loss,
                -avg_train_loss,
            )
            if self._step - self._last_save_step >= self._save_steps:
                self._last_save_step = self._step
                self._save_checkpoint()

            # Drop references to per-flush tensors so they're collectable
            # immediately — closes any lifetime ambiguity that could keep the
            # autograd graph or CPU loss tensor alive past ``step()``.
            del train_ids, train_labels, train_losses, train_loss_cpu

        # ── VALIDATION path: forward only, no grads, no optimizer step ────
        if val_items:
            c2f_model.eval()
            n_val = len(val_items)
            val_ids = torch.stack([it["input_ids"] for it in val_items]).to(self.device)
            val_labels = torch.stack([it["labels"] for it in val_items]).to(self.device)

            val_losses: list[torch.Tensor] = []
            with torch.no_grad():
                for mb_start in range(0, n_val, self._micro_bs):
                    mb_end = mb_start + self._micro_bs
                    val_losses.append(
                        self._ce_per_sample(val_ids[mb_start:mb_end], val_labels[mb_start:mb_end])
                    )

            val_loss_cpu = torch.cat(val_losses).cpu()
            for i, it in enumerate(val_items):
                loss_i = float(val_loss_cpu[i].item())
                if not it["future"].done():
                    it["future"].set_result({"reward": -loss_i, "p_loss": loss_i})

            del val_ids, val_labels, val_losses, val_loss_cpu

    # Wait used for iterations *after* the first one — just enough to catch
    # stragglers that arrived while we were processing and to detect
    # quiescence before exiting. First iteration still uses the full
    # ``c2f_batch_window`` to gather the bulk of the rollout.
    _STRAGGLER_WINDOW_S: float = 0.1

    async def _flusher(self) -> None:
        """Drain the queue in coalesced batches until empty, then exit.

        Invariant: at most one flusher is active at a time, tracked by
        ``_flusher_running`` under ``_queue_lock``. Exactly one of the
        ``run_single`` calls that found ``_flusher_running=False`` is the
        owner; it spawns this task. Subsequent enqueues while the flusher is
        live just append and await their future.

        Timing: first iteration sleeps ``c2f_batch_window`` to let a full
        rollout's worth of samples accumulate. After each productive flush,
        subsequent iterations sleep only ``_STRAGGLER_WINDOW_S`` so we exit
        promptly once the rollout is drained, instead of burning another
        ``c2f_batch_window`` on an empty check.
        """
        first_iteration = True
        try:
            while True:
                # Batching window: yield to let more concurrent ``run_single``
                # calls enqueue before we drain. First iter covers the full
                # rollout; later iters are just a straggler grace period.
                await asyncio.sleep(
                    self._batch_window if first_iteration else self._STRAGGLER_WINDOW_S
                )
                first_iteration = False

                async with self._queue_lock:
                    batch_items = self._queue
                    self._queue = []
                    if not batch_items:
                        # Empty under the lock — safe to exit; any enqueue
                        # that lands after this point will see
                        # ``_flusher_running=False`` and spawn a new flusher.
                        self._flusher_running = False
                        # Drop the coroutine-frame reference so captured
                        # ``batch_items`` from prior iterations don't linger
                        # until the next flush overwrites this slot.
                        self._flusher_task = None
                        return

                try:
                    self._process_batch(batch_items)
                except Exception as exc:
                    for it in batch_items:
                        if not it["future"].done():
                            it["future"].set_exception(exc)
                    raise
                finally:
                    # Drop the local ref regardless — don't let the loop
                    # iteration hold onto the processed items.
                    del batch_items

                self._flush_count += 1
                if self._gc_every > 0 and self._flush_count % self._gc_every == 0:
                    # Force collection of reference cycles accumulated in
                    # Ray's async worker. Safe: no tensors or futures live
                    # across this call.
                    gc.collect()

                if _resource is not None:
                    # ru_maxrss is bytes on Linux, KiB on macOS — fine as a
                    # delta signal between flushes.
                    log.debug(
                        "[Joint] flush=%d rss_maxrss=%d",
                        self._flush_count,
                        _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss,
                    )
        except Exception:
            # Make sure nothing is left awaiting a future that will never
            # resolve — propagate the exception to any stranded items too.
            async with self._queue_lock:
                stranded = self._queue
                self._queue = []
                self._flusher_running = False
                self._flusher_task = None
            for it in stranded:
                if not it["future"].done():
                    it["future"].set_exception(
                        RuntimeError("JointC2FRewardManager flusher crashed")
                    )
            raise

    async def run_single(self, data) -> dict:
        c = self.components
        data_item = data[0]
        # veRL's agent_loop._compute_score strips meta_info when building the
        # per-sample DataProto, so the trainer's ``validate`` flag doesn't reach
        # us that way. We source it from the ``is_validation`` parquet column
        # instead (see src/generation/dataset.py:build_rl_parquet); meta_info is
        # a fallback for legacy parquets that lack the column.
        nt = data_item.non_tensor_batch
        is_validate = bool(nt.get("is_validation", data.meta_info.get("validate", False)))
        response_ids = data_item.batch["responses"]
        response_str = c.sft_tokenizer.decode(response_ids.tolist(), skip_special_tokens=True)

        rm = nt.get("reward_model")
        gt = rm.get("ground_truth", "") if isinstance(rm, dict) else nt.get("ground_truth", "")

        if not getattr(type(self), "_debug_dumped", False):
            type(self)._debug_dumped = True
            log.debug(
                "first sample (validate=%s): response (len=%d) %r | ground_truth (len=%d) %r",
                is_validate,
                len(response_str),
                response_str[:800],
                len(str(gt)),
                str(gt)[:300],
            )

        layer_contents = parse_layers(
            response_str, c.word_count_constraints, strict=c.strict_word_count
        )
        if layer_contents is None:
            return {
                "reward_score": float(self.malformed_reward),
                "reward_extra_info": {
                    "p_loss": 0.0,
                    "malformed": 1.0,
                    "validate": 1.0 if is_validate else 0.0,
                },
            }

        input_ids, labels = self._build_input(layer_contents, str(gt))
        # Stay on CPU until the flusher stacks + moves the batch — keeps
        # peak GPU memory from scaling with in-flight ``run_single`` count.

        future: asyncio.Future = asyncio.get_event_loop().create_future()
        async with self._queue_lock:
            self._queue.append(
                {
                    "input_ids": input_ids,
                    "labels": labels,
                    "is_validate": is_validate,
                    "future": future,
                }
            )
            start_flusher = not self._flusher_running
            if start_flusher:
                self._flusher_running = True

        if start_flusher:
            # Hold a reference so the task isn't GC'd mid-flight. A previous
            # flusher may still hold this slot briefly after clearing
            # ``_flusher_running``; overwriting is safe since finished tasks
            # don't need to be awaited.
            self._flusher_task = asyncio.create_task(self._flusher())

        result = await future
        return {
            "reward_score": float(result["reward"]),
            "reward_extra_info": {
                "p_loss": float(result["p_loss"]),
                "malformed": 0.0,
                "validate": 1.0 if is_validate else 0.0,
            },
        }
