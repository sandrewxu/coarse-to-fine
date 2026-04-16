"""
C2F reward manager for GRPO training (Phase A).

Reward = log p_θ(x, z) / num_tokens  (C2F log-joint, token-normalized)
       + format_bonus                 (format_bonus_weight when all 4 layers
                                       have the correct word count; partial
                                       credit proportional to passing layers)

Implements veRL's class-based reward manager interface:

    class MyRewardManager:
        def __init__(self, tokenizer, num_examine=0, ...): ...
        def __call__(self, data: DataProto, return_dict=False): ...

veRL provides the SFT model tokenizer at init so we can decode response token
IDs back to strings.  The C2F model and space tokenizer are loaded from paths
stored in the experiment YAML (``config/latent_generation.yaml``),
which is located via the ``C2F_CONFIG_PATH`` environment variable (set by the
SLURM/launch script before torchrun) or defaults to the repo-relative path.
"""
import asyncio
import math
import os
import re
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase

from src.c2f_training.tokenizer import load_or_train_space_tokenizer
from src.qwen3_joint.configuration import C2FConfig
from src.qwen3_joint.modeling import C2FForCausalLM
from src.verification import verify as verify_layers

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

# Default config path (relative to repo root), overridden by C2F_CONFIG_PATH env var.
_DEFAULT_CONFIG_PATH = "config/latent_generation.yaml"


def _load_c2f_weights(model: C2FForCausalLM, checkpoint_path: Path) -> C2FForCausalLM:
    """Load weights from a saved checkpoint (safetensors or pytorch_model.bin)."""
    sf_path = checkpoint_path / "model.safetensors"
    pt_path = checkpoint_path / "pytorch_model.bin"

    if sf_path.exists():
        from safetensors.torch import load_file

        state_dict = load_file(str(sf_path))
    elif pt_path.exists():
        state_dict = torch.load(str(pt_path), map_location="cpu", weights_only=True)
    else:
        raise FileNotFoundError(
            f"No model weights found at {checkpoint_path}. "
            "Expected model.safetensors or pytorch_model.bin."
        )
    model.load_state_dict(state_dict)
    return model


class C2FRewardManager(RewardManagerBase):
    """
    Reward manager for GRPO training on the SFT model q_φ.

    Reward = log p_θ(x, z) / num_tokens + format_bonus  (C2F is frozen).

    Conforms to verl's experimental reward_loop interface: subclasses
    ``RewardManagerBase`` and implements ``async run_single`` for per-sample
    scoring. The experiment YAML is located via the ``C2F_CONFIG_PATH`` env
    var (set by the launch script), or falls back to ``config/latent_generation.yaml``.
    """

    def __init__(
        self,
        config,
        tokenizer,
        compute_score=None,
        **kwargs,
    ):
        super().__init__(config, tokenizer, compute_score)

        from src.config import load_config

        config_path = os.environ.get("C2F_CONFIG_PATH", _DEFAULT_CONFIG_PATH)
        self.exp_config = load_config(config_path)

        self.scale_lengths: list[int] = self.exp_config["scale_lengths"]
        self.word_count_constraints: dict[str, int] = self.exp_config["word_count_constraints"]
        self.text_word_count: int = self.exp_config.get("text_word_count", 32)

        rl_cfg = self.exp_config.get("rl", {}).get("sft_rl", {})
        self.format_bonus_weight: float = float(rl_cfg.get("format_bonus_weight", 0.1))

        # ── SFT tokenizer (provided by veRL) ────────────────────────────────
        self.sft_tokenizer = tokenizer

        # ── Frozen C2F model ─────────────────────────────────────────────────
        c2f_checkpoint = rl_cfg.get("c2f_model_path", "checkpoints/decoder")
        c2f_checkpoint = Path(c2f_checkpoint)
        print(f"[C2FRewardManager] Loading frozen C2F from {c2f_checkpoint}...")
        model_config = C2FConfig.from_pretrained(str(c2f_checkpoint))
        self.c2f_model = C2FForCausalLM(model_config)
        self.c2f_model = _load_c2f_weights(self.c2f_model, c2f_checkpoint)
        self.c2f_model.eval()
        for p in self.c2f_model.parameters():
            p.requires_grad_(False)
        self.device = next(self.c2f_model.parameters()).device

        # ── Space tokenizer (word-level, 1 word = 1 token) ──────────────────
        c2f_train_cfg = self.exp_config.get("c2f_training", {})
        tokenizer_dir = Path(
            c2f_train_cfg.get("tokenizer_dir", "checkpoints/decoder/tokenizer")
        )
        print(f"[C2FRewardManager] Loading space tokenizer from {tokenizer_dir}...")
        self.space_tokenizer = load_or_train_space_tokenizer(
            tokenizer_dir=tokenizer_dir,
            data_dir=c2f_train_cfg.get("dataset_dir", "data/sft_dataset"),
            dataset_format=c2f_train_cfg.get("dataset_format", "sft"),
        )

        # ── Verification config ─────────────────────────────────────────────
        self.strict_word_count = self.exp_config.get("verification", {}).get("strict_word_count", True)

        # ── Token ID shortcuts ───────────────────────────────────────────────
        self.bos_id = (
            self.space_tokenizer.bos_token_id or self.space_tokenizer.eos_token_id
        )
        self.pad_id = (
            self.space_tokenizer.pad_token_id or self.space_tokenizer.eos_token_id
        )
        self.seq_len = 2 ** math.ceil(math.log2(1 + sum(self.scale_lengths)))

        # ── Word boundaries for flat sequence segmentation ───────────────────
        # Layout: [z_4 words | z_3 words | z_2 words | z_1 words | text words]
        layer_names = ["z_4", "z_3", "z_2", "z_1"]
        word_counts = [self.word_count_constraints[n] for n in layer_names]
        word_counts.append(self.text_word_count)
        self.word_boundaries: list[tuple[int, int]] = []
        pos = 0
        for wc in word_counts:
            self.word_boundaries.append((pos, pos + wc))
            pos += wc

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _strip_think(self, text: str) -> str:
        """Strip <think>...</think> blocks from SFT output."""
        return _THINK_RE.sub("", text).strip()

    def _parse_layers(self, response: str) -> list[str] | None:
        """
        Parse z_n: lines from SFT response (after stripping think blocks).

        Returns layer content strings in [z_4, z_3, z_2, z_1] order, or None
        if parsing or verification fails.
        """
        cleaned = self._strip_think(response)
        result = verify_layers(
            cleaned, self.word_count_constraints,
            strict_word_count=self.strict_word_count,
        )
        if not result.passed:
            return None
        return [layer.content for layer in result.layers]

    def _format_bonus(self, response: str) -> float:
        """
        Compute format bonus.

        Full bonus (``format_bonus_weight``) when all 4 layers pass; partial
        credit proportional to the number of correctly-sized layers otherwise.
        """
        cleaned = self._strip_think(response)
        result = verify_layers(
            cleaned, self.word_count_constraints,
            strict_word_count=self.strict_word_count,
        )
        if result.passed:
            return self.format_bonus_weight

        # Partial credit even when overall verification fails
        if result.layers:
            expected_layers = ["z_4", "z_3", "z_2", "z_1"]
            correct = sum(
                1
                for layer in result.layers
                if layer.word_count == self.word_count_constraints.get(layer.layer_name, -1)
            )
            return self.format_bonus_weight * (correct / len(expected_layers))
        return 0.0

    def _build_c2f_input(
        self, layer_contents: list[str], prompt: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build C2F input_ids and labels from layer content strings + prompt.

        Replicates the tokenisation logic from
        ``C2FDataset._build_token_sequence`` / ``_build_labels``.

        Returns:
            (input_ids, labels) tensors of shape [seq_len].
        """
        flat_parts = layer_contents + [prompt]
        words = " ".join(flat_parts).split()

        tokens = [self.bos_id]
        for (start, end), length in zip(self.word_boundaries, self.scale_lengths):
            segment_text = " ".join(words[start:end])
            encoded = self.space_tokenizer.encode(segment_text, add_special_tokens=False)
            if len(encoded) >= length:
                encoded = encoded[:length]
            else:
                encoded = encoded + [self.pad_id] * (length - len(encoded))
            tokens.extend(encoded)

        while len(tokens) < self.seq_len:
            tokens.append(self.pad_id)

        input_ids = torch.tensor(tokens[: self.seq_len], dtype=torch.long)

        # Labels: mask BOS and padding (unshifted loss, same as C2FDataset)
        labels = input_ids.clone()
        labels[0] = -100
        labels[input_ids == self.pad_id] = -100

        return input_ids, labels

    @torch.no_grad()
    def _log_p_c2f(self, layer_contents: list[str], prompt: str) -> tuple[float, int]:
        """
        Compute log p_θ(x, z) via a forward pass of the frozen C2F model.

        Returns:
            (log_prob, num_unmasked_tokens)
            log_prob = -CE_loss × num_unmasked  (unnormalised log-likelihood)
        """
        input_ids, labels = self._build_c2f_input(layer_contents, prompt)
        input_ids = input_ids.unsqueeze(0).to(self.device)
        labels = labels.unsqueeze(0).to(self.device)

        outputs = self.c2f_model(input_ids=input_ids, labels=labels)
        loss: float = outputs.loss.item()
        num_unmasked: int = int((labels != -100).sum().item())
        log_p = -loss * num_unmasked
        return log_p, num_unmasked

    # ── veRL interface ───────────────────────────────────────────────────────

    def __call__(self, data: Any, return_dict: bool = False) -> Any:
        """
        Compute rewards for a batch of GRPO rollouts.

        Expects:
            data.batch['responses']          – response token IDs (B × T_resp)
            data.non_tensor_batch['ground_truth'] – original document per sample

        Sets:
            data.batch['token_level_scores'] – reward placed at last real token
                                               position, zeros elsewhere (B × T_resp)

        Returns:
            DataProto with token_level_scores populated (or a dict if
            ``return_dict=True``).
        """
        batch_size: int = data.batch["responses"].shape[0]
        response_len: int = data.batch["responses"].shape[1]

        # Decode response token IDs → strings
        response_strs: list[str] = self.sft_tokenizer.batch_decode(
            data.batch["responses"], skip_special_tokens=True
        )

        # Original documents (set as ground_truth during RL parquet prep)
        ground_truths = data.non_tensor_batch.get("ground_truth", [""] * batch_size)
        if hasattr(ground_truths, "tolist"):
            ground_truths = ground_truths.tolist()

        reward_tensor = torch.zeros(batch_size, response_len, dtype=torch.float32)
        pad_id = getattr(self.sft_tokenizer, "pad_token_id", None)

        for i, (response, prompt) in enumerate(zip(response_strs, ground_truths)):
            # 1. Parse latent layers from SFT output
            layer_contents = self._parse_layers(response)
            if layer_contents is None:
                # Malformed output: zero reward
                continue

            # 2. log p_θ(x, z) from frozen C2F
            log_p, num_tokens = self._log_p_c2f(layer_contents, str(prompt))
            log_p_normalized = log_p / max(num_tokens, 1)

            # 3. Format bonus
            bonus = self._format_bonus(response)

            reward = log_p_normalized + bonus

            # 4. Place reward at last non-padding response token
            response_ids = data.batch["responses"][i]
            if pad_id is not None:
                non_pad = (response_ids != pad_id).nonzero(as_tuple=True)[0]
            else:
                non_pad = torch.arange(response_len, device=response_ids.device)
            last_pos = int(non_pad[-1].item()) if len(non_pad) > 0 else response_len - 1
            reward_tensor[i, last_pos] = reward

        data.batch["token_level_scores"] = reward_tensor

        if return_dict:
            return {"reward_tensor": reward_tensor}
        return data

    async def run_single(self, data) -> dict:
        data_item = data[0]
        response_ids = data_item.batch["responses"]
        response_str = self.sft_tokenizer.decode(
            response_ids.tolist(), skip_special_tokens=True,
        )

        nt = data_item.non_tensor_batch
        rm = nt.get("reward_model")
        if isinstance(rm, dict):
            gt = rm.get("ground_truth", "")
        else:
            gt = nt.get("ground_truth", "")

        if not getattr(type(self), "_debug_dumped", False):
            type(self)._debug_dumped = True
            print(
                f"[C2FRewardManager DEBUG] first sample:\n"
                f"  response (len={len(response_str)}): {response_str[:800]!r}\n"
                f"  ground_truth (len={len(str(gt))}): {str(gt)[:300]!r}",
                flush=True,
            )

        layer_contents = self._parse_layers(response_str)
        if layer_contents is None:
            return {"reward_score": 0.0, "reward_extra_info": {"malformed": 1.0}}

        log_p, num_tokens = self._log_p_c2f(layer_contents, str(gt))
        log_p_normalized = log_p / max(num_tokens, 1)
        bonus = self._format_bonus(response_str)
        return {
            "reward_score": float(log_p_normalized + bonus),
            "reward_extra_info": {
                "log_p_normalized": float(log_p_normalized),
                "format_bonus": float(bonus),
            },
        }


# ── Joint reward manager ────────────────────────────────────────────────────


class JointC2FRewardManager(RewardManagerBase):
    """
    Reward manager for joint ELBO training (posterior collapse experiment).

    Like :class:`C2FRewardManager` but p_θ is **trainable**: per rollout sample
    the manager computes ``reward = -CE_loss`` and takes a per-sample optimizer
    step on p_θ (scoring and training share one forward+backward). Validation
    calls (``data.meta_info['validate'] == True``) skip the optimizer step.

    All p_θ updates are serialized through ``self._c2f_lock`` so concurrent
    ``run_single`` coroutines don't interleave CUDA ops on the shared model.
    """

    def __init__(
        self,
        config,
        tokenizer,
        compute_score=None,
        **kwargs,
    ):
        super().__init__(config, tokenizer, compute_score)

        from src.config import load_config

        config_path = os.environ.get("C2F_CONFIG_PATH", _DEFAULT_CONFIG_PATH)
        self.exp_config = load_config(config_path)

        self.scale_lengths: list[int] = self.exp_config["scale_lengths"]
        self.word_count_constraints: dict[str, int] = self.exp_config["word_count_constraints"]
        self.text_word_count: int = self.exp_config.get("text_word_count", 32)

        joint_cfg = self.exp_config.get("rl", {}).get("joint", {})
        self.malformed_reward: float = float(joint_cfg.get("malformed_reward", -10.0))
        c2f_lr: float = float(joint_cfg.get("c2f_lr", 1e-4))
        c2f_wd: float = float(joint_cfg.get("c2f_weight_decay", 0.01))
        self._save_steps: int = int(joint_cfg.get("c2f_save_steps", 100))
        self._save_dir = Path(joint_cfg.get("c2f_save_dir", "checkpoints/rl/joint/c2f"))
        self._save_dir.mkdir(parents=True, exist_ok=True)

        # ── SFT tokenizer (provided by veRL) ────────────────────────────────
        self.sft_tokenizer = tokenizer

        # ── Trainable C2F model ─────────────────────────────────────────────
        c2f_checkpoint = Path(joint_cfg.get("c2f_model_path", "checkpoints/decoder"))
        mask_type = joint_cfg.get("c2f_mask_type", "causal")
        print(f"[JointC2FRewardManager] Loading C2F from {c2f_checkpoint} (mask_type={mask_type})...")
        model_config = C2FConfig.from_pretrained(str(c2f_checkpoint))
        model_config.mask_type = mask_type
        self.c2f_model = C2FForCausalLM(model_config)
        self.c2f_model = _load_c2f_weights(self.c2f_model, c2f_checkpoint)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.c2f_model.to(self.device)
        self.c2f_model.train()

        self.optimizer = torch.optim.AdamW(
            self.c2f_model.parameters(), lr=c2f_lr, weight_decay=c2f_wd,
        )
        self._step = 0

        # ── Space tokenizer ─────────────────────────────────────────────────
        c2f_train_cfg = self.exp_config.get("c2f_training", {})
        tokenizer_dir = Path(
            c2f_train_cfg.get("tokenizer_dir", "checkpoints/decoder/tokenizer")
        )
        print(f"[JointC2FRewardManager] Loading space tokenizer from {tokenizer_dir}...")
        self.space_tokenizer = load_or_train_space_tokenizer(
            tokenizer_dir=tokenizer_dir,
            data_dir=c2f_train_cfg.get("dataset_dir", "data/sft_dataset"),
            dataset_format=c2f_train_cfg.get("dataset_format", "sft"),
        )

        # ── Verification config ─────────────────────────────────────────────
        self.strict_word_count = self.exp_config.get("verification", {}).get(
            "strict_word_count", True,
        )

        # ── Token ID shortcuts ───────────────────────────────────────────────
        self.bos_id = (
            self.space_tokenizer.bos_token_id or self.space_tokenizer.eos_token_id
        )
        self.pad_id = (
            self.space_tokenizer.pad_token_id or self.space_tokenizer.eos_token_id
        )
        self.seq_len = 2 ** math.ceil(math.log2(1 + sum(self.scale_lengths)))

        # ── Word boundaries ─────────────────────────────────────────────────
        layer_names = ["z_4", "z_3", "z_2", "z_1"]
        word_counts = [self.word_count_constraints[n] for n in layer_names]
        word_counts.append(self.text_word_count)
        self.word_boundaries: list[tuple[int, int]] = []
        pos = 0
        for wc in word_counts:
            self.word_boundaries.append((pos, pos + wc))
            pos += wc

        self._c2f_lock = asyncio.Lock()

        print(f"[JointC2FRewardManager] Ready. p trainable on {self.device}, "
              f"lr={c2f_lr}, malformed_reward={self.malformed_reward}")

    # ── Helpers (same logic as C2FRewardManager) ────────────────────────────

    def _strip_think(self, text: str) -> str:
        return _THINK_RE.sub("", text).strip()

    def _parse_layers(self, response: str) -> list[str] | None:
        cleaned = self._strip_think(response)
        result = verify_layers(
            cleaned, self.word_count_constraints,
            strict_word_count=self.strict_word_count,
        )
        if not result.passed:
            return None
        return [layer.content for layer in result.layers]

    def _build_c2f_input(
        self, layer_contents: list[str], prompt: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flat_parts = layer_contents + [prompt]
        words = " ".join(flat_parts).split()

        tokens = [self.bos_id]
        for (start, end), length in zip(self.word_boundaries, self.scale_lengths):
            segment_text = " ".join(words[start:end])
            encoded = self.space_tokenizer.encode(segment_text, add_special_tokens=False)
            if len(encoded) >= length:
                encoded = encoded[:length]
            else:
                encoded = encoded + [self.pad_id] * (length - len(encoded))
            tokens.extend(encoded)

        while len(tokens) < self.seq_len:
            tokens.append(self.pad_id)

        input_ids = torch.tensor(tokens[: self.seq_len], dtype=torch.long)
        labels = input_ids.clone()
        labels[0] = -100
        content_len = 1 + sum(self.scale_lengths)
        labels[content_len:] = -100
        return input_ids, labels

    def _save_checkpoint(self) -> None:
        save_path = self._save_dir / f"step_{self._step}"
        save_path.mkdir(parents=True, exist_ok=True)
        self.c2f_model.save_pretrained(str(save_path))
        torch.save(self.optimizer.state_dict(), save_path / "optimizer.pt")
        print(f"[JointC2FRewardManager] Saved p checkpoint: {save_path}")

    # ── veRL interface ──────────────────────────────────────────────────────

    def __call__(self, data, return_dict: bool = False):
        batch_size: int = data.batch["responses"].shape[0]
        response_len: int = data.batch["responses"].shape[1]

        # Decode response token IDs → strings
        response_strs: list[str] = self.sft_tokenizer.batch_decode(
            data.batch["responses"], skip_special_tokens=True,
        )
        ground_truths = data.non_tensor_batch.get("ground_truth", [""] * batch_size)
        if hasattr(ground_truths, "tolist"):
            ground_truths = ground_truths.tolist()

        # ── Phase 1: parse responses, collect valid C2F inputs ──────────────
        valid_indices: list[int] = []
        all_input_ids: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []

        for i, (response, prompt) in enumerate(zip(response_strs, ground_truths)):
            layer_contents = self._parse_layers(response)
            if layer_contents is None:
                continue
            input_ids, labels = self._build_c2f_input(layer_contents, str(prompt))
            valid_indices.append(i)
            all_input_ids.append(input_ids)
            all_labels.append(labels)

        reward_tensor = torch.full(
            (batch_size, response_len), 0.0, dtype=torch.float32,
        )
        num_valid = len(valid_indices)
        num_malformed = batch_size - num_valid

        # ── Phase 2: micro-batched forward + backward on p ──────────────────
        if num_valid > 0:
            batch_ids = torch.stack(all_input_ids).to(self.device)    # [N, seq_len]
            batch_labels = torch.stack(all_labels).to(self.device)    # [N, seq_len]

            # Micro-batch the C2F forward to avoid OOM when N is large
            c2f_micro_bs = 32
            all_per_sample_loss: list[torch.Tensor] = []

            for mb_start in range(0, num_valid, c2f_micro_bs):
                mb_ids = batch_ids[mb_start:mb_start + c2f_micro_bs]
                mb_labels = batch_labels[mb_start:mb_start + c2f_micro_bs]

                # Forward without internal loss (avoid retaining unused graph)
                mb_outputs = self.c2f_model(input_ids=mb_ids)

                mb_logits = mb_outputs.logits
                if self.c2f_model.config.mask_type == "causal":
                    shift_logits = mb_logits[:, :-1, :].contiguous()
                    shift_labels = mb_labels[:, 1:].contiguous()
                else:
                    shift_logits = mb_logits
                    shift_labels = mb_labels

                per_token_loss = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                    reduction="none",
                ).view(shift_labels.shape)  # [mb, T]

                mask = shift_labels != -100  # [mb, T]
                num_tokens_per_sample = mask.sum(dim=1).clamp(min=1)  # [mb]
                mb_per_sample_loss = (per_token_loss * mask).sum(dim=1) / num_tokens_per_sample
                all_per_sample_loss.append(mb_per_sample_loss)

            per_sample_loss = torch.cat(all_per_sample_loss)  # [N]

            # Rewards = log p / num_tokens = -loss (detached, for veRL)
            per_sample_reward = -per_sample_loss.detach().cpu()

            # Update p: backward on mean loss
            p_loss = per_sample_loss.mean()
            self.optimizer.zero_grad()
            p_loss.backward()
            self.optimizer.step()

            # Place rewards
            pad_id = getattr(self.sft_tokenizer, "pad_token_id", None)
            for j, idx in enumerate(valid_indices):
                response_ids = data.batch["responses"][idx]
                if pad_id is not None:
                    non_pad = (response_ids != pad_id).nonzero(as_tuple=True)[0]
                else:
                    non_pad = torch.arange(response_len)
                last_pos = int(non_pad[-1].item()) if len(non_pad) > 0 else response_len - 1
                reward_tensor[idx, last_pos] = per_sample_reward[j]

        # ── Phase 3: malformed samples get negative reward ──────────────────
        if num_malformed > 0:
            malformed_indices = set(range(batch_size)) - set(valid_indices)
            pad_id = getattr(self.sft_tokenizer, "pad_token_id", None)
            for idx in malformed_indices:
                response_ids = data.batch["responses"][idx]
                if pad_id is not None:
                    non_pad = (response_ids != pad_id).nonzero(as_tuple=True)[0]
                else:
                    non_pad = torch.arange(response_len)
                last_pos = int(non_pad[-1].item()) if len(non_pad) > 0 else response_len - 1
                reward_tensor[idx, last_pos] = self.malformed_reward

        data.batch["token_level_scores"] = reward_tensor

        # ── Logging + checkpointing ─────────────────────────────────────────
        self._step += 1
        avg_reward = per_sample_reward.mean().item() if num_valid > 0 else 0.0
        avg_loss = p_loss.item() if num_valid > 0 else 0.0
        print(
            f"[Joint] step={self._step}  valid={num_valid}/{batch_size}  "
            f"p_loss={avg_loss:.4f}  reward={avg_reward:.4f}"
        )
        if self._step % self._save_steps == 0:
            self._save_checkpoint()

        if return_dict:
            return {"reward_tensor": reward_tensor}
        return data

    async def run_single(self, data) -> dict:
        is_validate = bool(data.meta_info.get("validate", False))
        data_item = data[0]
        response_ids = data_item.batch["responses"]
        response_str = self.sft_tokenizer.decode(
            response_ids.tolist(), skip_special_tokens=True,
        )

        nt = data_item.non_tensor_batch
        rm = nt.get("reward_model")
        if isinstance(rm, dict):
            gt = rm.get("ground_truth", "")
        else:
            gt = nt.get("ground_truth", "")

        if not getattr(type(self), "_debug_dumped", False):
            type(self)._debug_dumped = True
            print(
                f"[JointC2FRewardManager DEBUG] first sample (validate={is_validate}):\n"
                f"  response (len={len(response_str)}): {response_str[:800]!r}\n"
                f"  ground_truth (len={len(str(gt))}): {str(gt)[:300]!r}",
                flush=True,
            )

        layer_contents = self._parse_layers(response_str)
        if layer_contents is None:
            return {
                "reward_score": float(self.malformed_reward),
                "reward_extra_info": {"malformed": 1.0},
            }

        input_ids, labels = self._build_c2f_input(layer_contents, str(gt))
        input_ids = input_ids.unsqueeze(0).to(self.device)
        labels = labels.unsqueeze(0).to(self.device)

        async with self._c2f_lock:
            self.c2f_model.train(mode=not is_validate)
            with torch.set_grad_enabled(not is_validate):
                outputs = self.c2f_model(input_ids=input_ids)
                logits = outputs.logits

                if self.c2f_model.config.mask_type == "causal":
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
                num_tokens = mask.sum().clamp(min=1)
                sample_loss = (per_token_loss * mask).sum() / num_tokens

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
