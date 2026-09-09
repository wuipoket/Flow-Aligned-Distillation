#!/usr/bin/env python3
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from thesis_geometry_redundancy_final_gemma2 import (
    collate_tokenized_batch,
    tokenize_records,
)


COMPATIBILITY_MODE = "decision_token_ce_without_candidate_channel"


def tokenize_decision_records(
    tokenizer,
    records: Sequence[Dict[str, Any]],
    cutoff_len: int,
):
    # Preserve the released FAD prompt/response representation.
    # No candidate tensors are emitted, so the main pipeline uses its
    # decision-token CE fallback.
    return tokenize_records(
        tokenizer,
        records,
        cutoff_len=int(cutoff_len),
    )


def collate_decision_batch(
    batch: Sequence[Dict[str, Any]],
    pad_token_id: int,
):
    return collate_tokenized_batch(
        batch,
        pad_token_id=int(pad_token_id),
    )


def shifted_target_mask(
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lens: Optional[torch.Tensor],
    scope: str,
    pad_token_id: int,
    eos_token_id: Optional[int],
    exclude_eos: bool,
) -> torch.Tensor:
    if input_ids.dim() != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError("input_ids and attention_mask must have shape [B, T]")

    batch_size, seq_len = input_ids.shape
    if seq_len < 2:
        return torch.zeros(
            (batch_size, 0),
            dtype=torch.bool,
            device=input_ids.device,
        )

    targets = input_ids[:, 1:]
    valid_targets = attention_mask[:, 1:].to(dtype=torch.bool)
    valid_targets &= targets.ne(int(pad_token_id))

    if bool(exclude_eos) and eos_token_id is not None:
        valid_targets &= targets.ne(int(eos_token_id))

    normalized_scope = str(scope or "all").strip().lower()

    if normalized_scope in {
        "all",
        "full",
        "sequence",
        "global",
        "token",
        "tokens",
    }:
        return valid_targets

    result = torch.zeros_like(valid_targets, dtype=torch.bool)
    prompt_values = (
        prompt_lens.to(device=input_ids.device, dtype=torch.long).view(-1)
        if prompt_lens is not None
        else None
    )

    for row in range(batch_size):
        valid_input_positions = torch.nonzero(
            attention_mask[row].to(dtype=torch.bool),
            as_tuple=False,
        ).view(-1)

        if valid_input_positions.numel() == 0:
            continue

        first_valid = int(valid_input_positions[0].item())
        valid_count = int(valid_input_positions.numel())

        if prompt_values is None:
            response_start = first_valid
        else:
            prompt_len = max(
                0,
                min(int(prompt_values[row].item()), valid_count),
            )
            response_start = first_valid + prompt_len

        # Shifted mask position j supervises input token j+1.
        target_positions = torch.arange(
            1,
            seq_len,
            device=input_ids.device,
            dtype=torch.long,
        )

        response_mask = valid_targets[row] & (
            target_positions >= int(response_start)
        )

        if normalized_scope in {
            "response",
            "answer",
            "completion",
            "response_all",
        }:
            result[row] = response_mask
            continue

        if normalized_scope in {
            "prompt",
            "instruction",
            "input",
        }:
            result[row] = valid_targets[row] & (
                target_positions < int(response_start)
            )
            continue

        if normalized_scope in {
            "decision",
            "last",
            "last_token",
            "decision_token",
        }:
            eligible = torch.nonzero(
                response_mask,
                as_tuple=False,
            ).view(-1)

            if eligible.numel() == 0:
                eligible = torch.nonzero(
                    valid_targets[row],
                    as_tuple=False,
                ).view(-1)

            if eligible.numel() > 0:
                result[row, int(eligible[-1].item())] = True
            continue

        raise ValueError(f"unsupported loss scope: {scope!r}")

    return result


def masked_next_token_cross_entropy(
    *,
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    token_mask: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if logits.dim() != 3:
        raise ValueError("logits must have shape [B, T, V]")
    if input_ids.dim() != 2:
        raise ValueError("input_ids must have shape [B, T]")

    shift_logits = logits[:, :-1, :].contiguous()
    shift_targets = input_ids[:, 1:].contiguous()

    mask = token_mask.to(
        device=shift_logits.device,
        dtype=torch.bool,
    )

    if mask.shape == input_ids.shape:
        mask = mask[:, 1:]
    if mask.shape != shift_targets.shape:
        raise ValueError(
            f"token_mask shape {tuple(mask.shape)} does not match "
            f"shifted targets {tuple(shift_targets.shape)}"
        )

    per_token = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_targets.reshape(-1),
        reduction="none",
    ).view_as(shift_targets)

    weights = mask.to(dtype=per_token.dtype)
    token_count = weights.sum()

    if float(token_count.detach().item()) <= 0.0:
        loss = shift_logits.sum() * 0.0
    else:
        loss = (per_token * weights).sum() / token_count

    stats = {
        "token_count": token_count.detach(),
        "loss": loss.detach(),
    }
    return loss, stats


def _unsupported_missing_release_source(name: str):
    raise RuntimeError(
        f"{name} is unavailable because the released repository omitted "
        "the original candidate-channel/SAGE implementation. "
        "Use distill_mode=ce and lambda_kd=0 for this Gemma port."
    )


def candidate_decision_logits(*args, **kwargs):
    return _unsupported_missing_release_source(
        "candidate_decision_logits"
    )


def candidate_decision_cross_entropy(*args, **kwargs):
    return _unsupported_missing_release_source(
        "candidate_decision_cross_entropy"
    )


def sage_information_gain_js(*args, **kwargs):
    return _unsupported_missing_release_source(
        "sage_information_gain_js"
    )


def sage_candidate_information_gain_js(*args, **kwargs):
    return _unsupported_missing_release_source(
        "sage_candidate_information_gain_js"
    )


def sage_rate_at_step(
    *,
    step: int,
    total_steps: int,
    max_rate: float,
    warmup_ratio: float,
    decay_start_ratio: float,
    min_rate_ratio: float,
) -> float:
    total = max(1, int(total_steps))
    progress = min(1.0, max(0.0, float(step) / float(total)))
    warmup = min(1.0, max(0.0, float(warmup_ratio)))
    decay_start = min(
        1.0,
        max(warmup, float(decay_start_ratio)),
    )
    max_value = max(0.0, float(max_rate))
    minimum = max_value * max(
        0.0,
        min(1.0, float(min_rate_ratio)),
    )

    if warmup > 0.0 and progress < warmup:
        return max_value * progress / warmup

    if progress <= decay_start or decay_start >= 1.0:
        return max_value

    fraction = (progress - decay_start) / (1.0 - decay_start)
    return max_value + fraction * (minimum - max_value)
