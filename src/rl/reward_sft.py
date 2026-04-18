"""Reward manager for Phase A (GRPO on q_φ, p_θ frozen).

Reward = ``log p_θ(x, z) / num_tokens + format_bonus``.

Implements veRL's class-based reward manager interface:

    class MyRewardManager:
        def __init__(self, tokenizer, num_examine=0, ...): ...
        def __call__(self, data: DataProto, return_dict=False): ...

The C2F model and space tokenizer are loaded via the ``C2F_CONFIG_PATH`` env
var (set by the SLURM/launch script), or fall back to the repo-relative path.
"""

from typing import Any

import torch
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase

from src.common.logging import get_logger
from src.rl.common import (
    build_c2f_input,
    load_c2f_components,
    load_exp_config,
    parse_layers,
)
from src.verification import verify as verify_layers

log = get_logger(__name__)


class C2FRewardManager(RewardManagerBase):
    """GRPO reward manager: scores SFT rollouts against a frozen C2F decoder."""

    def __init__(
        self,
        config,
        tokenizer,
        compute_score=None,
        **kwargs,
    ):
        super().__init__(config, tokenizer, compute_score)

        exp_config = load_exp_config()
        self.components = load_c2f_components(
            exp_config, sft_tokenizer=tokenizer, c2f_section_key="sft_rl"
        )

        # ── Freeze the C2F decoder ──────────────────────────────────────────
        c2f_model = self.components.c2f_model
        c2f_model.eval()
        for p in c2f_model.parameters():
            p.requires_grad_(False)
        self.device = next(c2f_model.parameters()).device

        # ── Phase-specific config ───────────────────────────────────────────
        rl_cfg = exp_config.get("rl", {}).get("sft_rl", {})
        self.format_bonus_weight: float = float(rl_cfg.get("format_bonus_weight", 0.1))

    # ── Reward components ────────────────────────────────────────────────────

    def _format_bonus(self, response: str) -> float:
        """Full bonus when all 4 layers pass; partial credit otherwise."""
        from src.rl.common import strip_think

        cleaned = strip_think(response)
        result = verify_layers(
            cleaned,
            self.components.word_count_constraints,
            strict_word_count=self.components.strict_word_count,
        )
        if result.passed:
            return self.format_bonus_weight

        if result.layers:
            correct = sum(
                1
                for layer in result.layers
                if layer.word_count
                == self.components.word_count_constraints.get(layer.layer_name, -1)
            )
            return self.format_bonus_weight * (correct / len(result.layers))
        return 0.0

    @torch.no_grad()
    def _log_p_c2f(self, layer_contents: list[str], prompt: str) -> tuple[float, int]:
        """Compute ``log p_θ(x, z)`` and the unmasked-token count."""
        c = self.components
        input_ids, labels = build_c2f_input(
            layer_contents,
            prompt,
            scale_lengths=c.scale_lengths,
            word_boundaries=c.word_boundaries,
            space_tokenizer=c.space_tokenizer,
            bos_id=c.bos_id,
            pad_id=c.pad_id,
            seq_len=c.seq_len,
            label_strategy="sft",
        )
        input_ids = input_ids.unsqueeze(0).to(self.device)
        labels = labels.unsqueeze(0).to(self.device)

        outputs = c.c2f_model(input_ids=input_ids, labels=labels)
        loss: float = outputs.loss.item()
        num_unmasked: int = int((labels != -100).sum().item())
        return -loss * num_unmasked, num_unmasked

    # ── veRL interface ───────────────────────────────────────────────────────

    def __call__(self, data: Any, return_dict: bool = False) -> Any:
        """Score a batch of GRPO rollouts.

        Sets ``data.batch['token_level_scores']`` to a tensor of shape
        ``(B, T_resp)`` with the scalar reward placed at the last non-pad
        response token of each sample.
        """
        c = self.components
        batch_size: int = data.batch["responses"].shape[0]
        response_len: int = data.batch["responses"].shape[1]

        response_strs: list[str] = c.sft_tokenizer.batch_decode(
            data.batch["responses"], skip_special_tokens=True
        )
        ground_truths = data.non_tensor_batch.get("ground_truth", [""] * batch_size)
        if hasattr(ground_truths, "tolist"):
            ground_truths = ground_truths.tolist()

        reward_tensor = torch.zeros(batch_size, response_len, dtype=torch.float32)
        pad_id = getattr(c.sft_tokenizer, "pad_token_id", None)

        for i, (response, prompt) in enumerate(zip(response_strs, ground_truths, strict=False)):
            layer_contents = parse_layers(
                response, c.word_count_constraints, strict=c.strict_word_count
            )
            if layer_contents is None:
                continue  # malformed → zero reward

            log_p, num_tokens = self._log_p_c2f(layer_contents, str(prompt))
            log_p_normalized = log_p / max(num_tokens, 1)
            reward = log_p_normalized + self._format_bonus(response)

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
        c = self.components
        data_item = data[0]
        response_ids = data_item.batch["responses"]
        response_str = c.sft_tokenizer.decode(response_ids.tolist(), skip_special_tokens=True)

        nt = data_item.non_tensor_batch
        rm = nt.get("reward_model")
        gt = rm.get("ground_truth", "") if isinstance(rm, dict) else nt.get("ground_truth", "")

        if not getattr(type(self), "_debug_dumped", False):
            type(self)._debug_dumped = True
            log.debug(
                "first sample: response (len=%d) %r | ground_truth (len=%d) %r",
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
                "reward_score": 0.0,
                "reward_extra_info": {
                    "log_p_normalized": 0.0,
                    "format_bonus": 0.0,
                    "malformed": 1.0,
                },
            }

        log_p, num_tokens = self._log_p_c2f(layer_contents, str(gt))
        log_p_normalized = log_p / max(num_tokens, 1)
        bonus = self._format_bonus(response_str)
        return {
            "reward_score": float(log_p_normalized + bonus),
            "reward_extra_info": {
                "log_p_normalized": float(log_p_normalized),
                "format_bonus": float(bonus),
                "malformed": 0.0,
            },
        }
