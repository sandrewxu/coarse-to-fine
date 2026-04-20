"""Reward manager for joint ELBO training (posterior + decoder, simultaneous).

Like :class:`C2FRewardManager` but ``p_θ`` is **trainable**: per rollout sample
the manager computes ``reward = -CE_loss`` and takes a per-sample optimizer step
on ``p_θ`` (scoring and training share one forward+backward). Validation calls
(``data.meta_info['validate'] == True``) skip the optimizer step.

All ``p_θ`` updates are serialised through ``self._c2f_lock`` so concurrent
``run_single`` coroutines don't interleave CUDA ops on the shared model.
"""

import asyncio
import os
import shutil
from pathlib import Path

import torch
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase

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

        self._c2f_lock = asyncio.Lock()

        log.info(
            "JointC2FRewardManager ready. p trainable on %s, lr=%s, malformed_reward=%s",
            self.device,
            c2f_lr,
            self.malformed_reward,
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
        per_sample_reward: torch.Tensor | None = None
        p_loss: torch.Tensor | None = None
        if num_valid > 0:
            batch_ids = torch.stack(all_input_ids).to(self.device)
            batch_labels = torch.stack(all_labels).to(self.device)

            all_per_sample_loss: list[torch.Tensor] = []
            for mb_start in range(0, num_valid, self._micro_bs):
                mb_loss = self._ce_per_sample(
                    batch_ids[mb_start : mb_start + self._micro_bs],
                    batch_labels[mb_start : mb_start + self._micro_bs],
                )
                all_per_sample_loss.append(mb_loss)

            per_sample_loss = torch.cat(all_per_sample_loss)
            per_sample_reward = -per_sample_loss.detach().cpu()

            p_loss = per_sample_loss.mean()
            self.optimizer.zero_grad()
            p_loss.backward()
            self.optimizer.step()

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
        avg_loss = p_loss.item() if p_loss is not None else 0.0
        log.info(
            "[Joint] step=%d valid=%d/%d p_loss=%.4f reward=%.4f",
            self._step,
            num_valid,
            batch_size,
            avg_loss,
            avg_reward,
        )
        if self._step % self._save_steps == 0:
            self._save_checkpoint()

        if return_dict:
            return {"reward_tensor": reward_tensor}
        return data

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
        input_ids = input_ids.unsqueeze(0).to(self.device)
        labels = labels.unsqueeze(0).to(self.device)

        async with self._c2f_lock:
            c.c2f_model.train(mode=not is_validate)
            with torch.set_grad_enabled(not is_validate):
                sample_loss = self._ce_per_sample(input_ids, labels).squeeze(0)

            reward = -sample_loss.detach().cpu().item()

            if not is_validate:
                self.optimizer.zero_grad()
                sample_loss.backward()
                self.optimizer.step()
                self._step += 1
                if self._step % self._save_steps == 0:
                    self._save_checkpoint()

        return {
            "reward_score": float(reward),
            "reward_extra_info": {
                "p_loss": float(sample_loss.detach().cpu().item()),
                "malformed": 0.0,
                "validate": 1.0 if is_validate else 0.0,
            },
        }
