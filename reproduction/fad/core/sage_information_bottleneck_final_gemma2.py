#!/usr/bin/env python3
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from thesis_geometry_redundancy_final_gemma2 import (
    collate_tokenized_batch,
    last_pred_indices,
    tokenize_records,
)
from thesis_common_final_llama import (
    render_phase1_prompt_prefix,
)


COMPATIBILITY_MODE = "reconstructed_decision_candidate_channel_v1"

_FORMAT_RE = re.compile(
    r"answer\s*format\s*:\s*([^\r\n]+)",
    flags=re.IGNORECASE,
)
_LABEL_RE = re.compile(
    r"\b(?:answer|ending|option|solution)\d+\b|\b(?:true|false)\b",
    flags=re.IGNORECASE,
)
_NUMBERED_LABEL_RE = re.compile(
    r"^(answer|ending|option|solution)(\d+)$",
    flags=re.IGNORECASE,
)


def _unique_in_order(values: Sequence[str]) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values:
        normalized = str(value).strip().lower()
        if not normalized or normalized in seen:
            continue
        result.append(normalized)
        seen.add(normalized)
    return result


def decision_candidate_labels(record: Dict[str, Any]) -> List[str]:
    """Return the declared decision labels for one commonsense record.

    The released dataset carries an explicit Answer format contract. Parsing
    that contract keeps three-, four-, and five-way answerN examples distinct
    instead of guessing a global candidate count.
    """

    instruction = str(record.get("instruction", "") or "")
    input_text = str(record.get("input", "") or "")
    gold = str(record.get("answer", "") or "").strip().lower()
    searchable = "\n".join((instruction, input_text))

    labels: List[str] = []
    match = _FORMAT_RE.search(searchable)
    if match is not None:
        labels = _unique_in_order(_LABEL_RE.findall(match.group(1)))

    if not labels and gold in {"true", "false"}:
        labels = ["true", "false"]

    numbered_gold = _NUMBERED_LABEL_RE.fullmatch(gold)
    if not labels and numbered_gold is not None:
        family = numbered_gold.group(1).lower()
        observed = []
        for candidate in _LABEL_RE.findall(searchable):
            parsed = _NUMBERED_LABEL_RE.fullmatch(candidate)
            if parsed is not None and parsed.group(1).lower() == family:
                observed.append(candidate.lower())
        labels = sorted(
            _unique_in_order(observed),
            key=lambda value: int(_NUMBERED_LABEL_RE.fullmatch(value).group(2)),
        )

    if not labels:
        raise ValueError(
            "cannot determine decision candidates from Answer format; "
            f"gold={gold!r}"
        )
    if gold not in labels:
        raise ValueError(
            f"gold decision label {gold!r} is absent from candidates {labels!r}"
        )
    if len(labels) < 2:
        raise ValueError(f"at least two decision candidates are required: {labels!r}")
    return labels


def _candidate_last_token_id(tokenizer, label: str) -> int:
    encoded = tokenizer(
        " " + str(label).strip(),
        add_special_tokens=False,
    )
    token_ids = encoded.get("input_ids", [])
    if torch.is_tensor(token_ids):
        token_ids = token_ids.detach().cpu().tolist()
    if token_ids and isinstance(token_ids[0], list):
        if len(token_ids) != 1:
            raise ValueError("candidate tokenizer unexpectedly returned a batch")
        token_ids = token_ids[0]
    if not token_ids:
        raise ValueError(f"candidate label produced no tokens: {label!r}")
    return int(token_ids[-1])


def _flat_token_ids(encoded) -> List[int]:
    token_ids = encoded.get("input_ids", [])

    if torch.is_tensor(token_ids):
        token_ids = token_ids.detach().cpu().tolist()

    if token_ids and isinstance(token_ids[0], list):
        if len(token_ids) != 1:
            raise ValueError(
                "tokenizer unexpectedly returned a batch"
            )
        token_ids = token_ids[0]

    return [int(token_id) for token_id in token_ids]


def _decision_item_preserving_response(
    tokenizer,
    record: Dict[str, Any],
    cutoff_len: int,
) -> Dict[str, Any]:
    """Tokenize a decision record without truncating its response.

    Long commonsense prompts can exceed cutoff_len before the response begins.
    For decision-aligned training, preserve the complete response and truncate
    only the oldest prompt tokens. Keep BOS when the tokenizer supplies one.
    """

    cutoff = max(2, int(cutoff_len))
    prompt_text = render_phase1_prompt_prefix(record)
    response_text = str(
        record.get("output", "") or ""
    ).strip()

    if not response_text:
        raise ValueError(
            "decision-aligned record has an empty output"
        )

    prompt_ids = _flat_token_ids(
        tokenizer(
            prompt_text,
            truncation=False,
            padding=False,
            add_special_tokens=True,
        )
    )
    response_ids = _flat_token_ids(
        tokenizer(
            response_text,
            truncation=False,
            padding=False,
            add_special_tokens=False,
        )
    )

    if not response_ids:
        raise ValueError(
            "decision response produced no tokens"
        )

    eos_token_id = tokenizer.eos_token_id
    suffix_ids = list(response_ids)

    if (
        eos_token_id is not None
        and suffix_ids[-1] != int(eos_token_id)
    ):
        suffix_ids.append(int(eos_token_id))

    if len(suffix_ids) >= cutoff:
        raise ValueError(
            "decision response alone exceeds cutoff_len; "
            f"response_tokens={len(suffix_ids)} "
            f"cutoff_len={cutoff}"
        )

    prompt_budget = cutoff - len(suffix_ids)

    if len(prompt_ids) > prompt_budget:
        bos_token_id = getattr(
            tokenizer,
            "bos_token_id",
            None,
        )

        if (
            bos_token_id is not None
            and prompt_ids
            and prompt_ids[0] == int(bos_token_id)
        ):
            if prompt_budget == 1:
                prompt_ids = [int(bos_token_id)]
            else:
                prompt_ids = (
                    [int(bos_token_id)]
                    + prompt_ids[-(prompt_budget - 1):]
                )
        else:
            prompt_ids = prompt_ids[-prompt_budget:]

    input_ids = prompt_ids + suffix_ids

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "prompt_len": len(prompt_ids),
        "response_preserved_after_truncation": True,
    }


def _last_non_eos_token_id(
    item: Dict[str, Any],
    eos_token_id: Optional[int],
) -> Optional[int]:
    content_ids = [
        int(token_id)
        for token_id in item["input_ids"]
        if (
            eos_token_id is None
            or int(token_id) != int(eos_token_id)
        )
    ]

    if not content_ids:
        return None

    return int(content_ids[-1])


def tokenize_decision_records(
    tokenizer,
    records: Sequence[Dict[str, Any]],
    cutoff_len: int,
):
    tokenized: List[Dict[str, Any]] = []

    # Process records independently so candidate metadata cannot shift when a
    # record needs output-preserving prompt truncation.
    for record in records:
        labels = decision_candidate_labels(record)
        candidate_ids = [
            _candidate_last_token_id(tokenizer, label)
            for label in labels
        ]

        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError(
                "decision labels must have distinct final token IDs; "
                f"labels={labels!r} ids={candidate_ids!r}"
            )

        gold = str(
            record.get("answer", "") or ""
        ).strip().lower()
        gold_index = int(labels.index(gold))
        expected_gold_id = int(candidate_ids[gold_index])
        eos_token_id = tokenizer.eos_token_id

        # Preserve the existing SFT tokenization for normal-length examples.
        base_items = tokenize_records(
            tokenizer,
            [record],
            cutoff_len=int(cutoff_len),
        )

        item = (
            dict(base_items[0])
            if base_items
            else None
        )

        actual_gold_id = (
            _last_non_eos_token_id(
                item,
                eos_token_id,
            )
            if item is not None
            else None
        )

        # A right-truncated prompt can remove the response entirely or leave a
        # non-gold token at the sequence end. Retry by truncating only the
        # oldest prompt tokens while retaining the complete response.
        if actual_gold_id != expected_gold_id:
            item = _decision_item_preserving_response(
                tokenizer,
                record,
                cutoff_len=int(cutoff_len),
            )
            actual_gold_id = _last_non_eos_token_id(
                item,
                eos_token_id,
            )

        if actual_gold_id != expected_gold_id:
            raise ValueError(
                "gold candidate final token does not match the "
                "output-preserving tokenized response; "
                f"gold={gold!r} expected={expected_gold_id} "
                f"actual={actual_gold_id}"
            )

        item["candidate_token_ids"] = candidate_ids
        item["candidate_mask"] = [
            True
        ] * len(candidate_ids)
        item["gold_candidate_index"] = gold_index
        item["candidate_labels"] = labels
        tokenized.append(item)

    return tokenized


def collate_decision_batch(
    batch: Sequence[Dict[str, Any]],
    pad_token_id: int,
):
    result = collate_tokenized_batch(
        batch,
        pad_token_id=int(pad_token_id),
    )

    has_candidate_metadata = [
        all(
            key in item
            for key in (
                "candidate_token_ids",
                "candidate_mask",
                "gold_candidate_index",
            )
        )
        for item in batch
    ]
    if not any(has_candidate_metadata):
        return result
    if not all(has_candidate_metadata):
        raise ValueError("mixed candidate and non-candidate records in one batch")

    width = max(len(item["candidate_token_ids"]) for item in batch)
    candidate_rows: List[List[int]] = []
    mask_rows: List[List[bool]] = []
    gold_indices: List[int] = []

    for item in batch:
        ids = [int(value) for value in item["candidate_token_ids"]]
        mask = [bool(value) for value in item["candidate_mask"]]
        if len(ids) != len(mask) or not ids:
            raise ValueError("invalid candidate_token_ids/candidate_mask metadata")

        gold_index = int(item["gold_candidate_index"])
        if not 0 <= gold_index < len(ids) or not mask[gold_index]:
            raise ValueError("gold_candidate_index does not select a valid candidate")

        padding = width - len(ids)
        candidate_rows.append(ids + [int(pad_token_id)] * padding)
        mask_rows.append(mask + [False] * padding)
        gold_indices.append(gold_index)

    result.update(
        {
            "candidate_token_ids": torch.tensor(
                candidate_rows,
                dtype=torch.long,
            ),
            "candidate_mask": torch.tensor(
                mask_rows,
                dtype=torch.bool,
            ),
            "gold_candidate_index": torch.tensor(
                gold_indices,
                dtype=torch.long,
            ),
        }
    )
    return result


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


def _validate_candidate_tensors(
    *,
    candidate_token_ids: torch.Tensor,
    candidate_mask: torch.Tensor,
    gold_candidate_index: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    if candidate_token_ids.dim() != 2:
        raise ValueError("candidate_token_ids must have shape [B, C]")
    if candidate_mask.shape != candidate_token_ids.shape:
        raise ValueError("candidate_mask must match candidate_token_ids")

    ids = candidate_token_ids.to(dtype=torch.long)
    mask = candidate_mask.to(device=ids.device, dtype=torch.bool)
    if bool((mask.sum(dim=1) < 2).any().item()):
        raise ValueError("every example must contain at least two candidates")

    gold = None
    if gold_candidate_index is not None:
        gold = gold_candidate_index.to(
            device=ids.device,
            dtype=torch.long,
        ).view(-1)
        if gold.numel() != ids.size(0):
            raise ValueError("gold_candidate_index must have shape [B]")
        if bool(((gold < 0) | (gold >= ids.size(1))).any().item()):
            raise ValueError("gold_candidate_index is out of range")
        if not bool(mask.gather(1, gold[:, None]).all().item()):
            raise ValueError("gold_candidate_index selects a masked candidate")
    return ids, mask, gold


def candidate_decision_logits(
    *,
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    candidate_token_ids: torch.Tensor,
    candidate_mask: torch.Tensor,
    eos_token_id: Optional[int],
) -> torch.Tensor:
    if logits.dim() != 3:
        raise ValueError("logits must have shape [B, T, V]")
    if input_ids.dim() != 2 or input_ids.shape != attention_mask.shape:
        raise ValueError("input_ids and attention_mask must have shape [B, T]")
    if logits.shape[:2] != input_ids.shape:
        raise ValueError("logits sequence shape must match input_ids")

    ids, mask, _ = _validate_candidate_tensors(
        candidate_token_ids=candidate_token_ids,
        candidate_mask=candidate_mask,
    )
    ids = ids.to(device=logits.device)
    mask = mask.to(device=logits.device)
    if ids.size(0) != logits.size(0):
        raise ValueError("candidate batch size does not match logits")
    if bool(((ids[mask] < 0) | (ids[mask] >= logits.size(-1))).any().item()):
        raise ValueError("candidate token ID is outside the vocabulary")

    prediction_indices = last_pred_indices(
        attention_mask=attention_mask,
        input_ids=input_ids,
        eos_token_id=eos_token_id,
        device=logits.device,
    )
    batch_indices = torch.arange(
        logits.size(0),
        device=logits.device,
        dtype=torch.long,
    )
    decision_vocab_logits = logits[batch_indices, prediction_indices, :]
    result = decision_vocab_logits.gather(1, ids)
    return result.masked_fill(~mask, float("-inf"))


def candidate_decision_cross_entropy(
    *,
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    candidate_token_ids: torch.Tensor,
    candidate_mask: torch.Tensor,
    gold_candidate_index: torch.Tensor,
    eos_token_id: Optional[int],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    ids, mask, gold = _validate_candidate_tensors(
        candidate_token_ids=candidate_token_ids,
        candidate_mask=candidate_mask,
        gold_candidate_index=gold_candidate_index,
    )
    assert gold is not None
    candidate_logits = candidate_decision_logits(
        logits=logits,
        input_ids=input_ids,
        attention_mask=attention_mask,
        candidate_token_ids=ids,
        candidate_mask=mask,
        eos_token_id=eos_token_id,
    )
    gold = gold.to(device=candidate_logits.device)
    per_example = F.cross_entropy(
        candidate_logits.float(),
        gold,
        reduction="none",
    )
    loss = per_example.mean()
    accuracy = candidate_logits.argmax(dim=-1).eq(gold).float().mean()
    return loss, {
        "candidate_logits": candidate_logits,
        "per_example_loss": per_example.detach(),
        "accuracy": accuracy.detach(),
        "sample_count": torch.tensor(
            float(candidate_logits.size(0)),
            device=candidate_logits.device,
        ),
        "loss": loss.detach(),
    }


def _sage_from_candidate_logits(
    *,
    student_candidate_logits: torch.Tensor,
    teacher_candidate_logits: torch.Tensor,
    candidate_mask: torch.Tensor,
    gold_candidate_index: torch.Tensor,
    temperature: float,
    gain_margin: float,
    gain_temperature: float,
    confidence_margin: float,
    confidence_temperature: float,
    confidence_power: float,
    require_teacher_correct: bool,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if student_candidate_logits.shape != teacher_candidate_logits.shape:
        raise ValueError("student and teacher candidate logits must match")
    if student_candidate_logits.dim() != 2:
        raise ValueError("candidate logits must have shape [N, C]")

    dummy_ids = torch.zeros_like(candidate_mask, dtype=torch.long)
    _, mask, gold = _validate_candidate_tensors(
        candidate_token_ids=dummy_ids,
        candidate_mask=candidate_mask,
        gold_candidate_index=gold_candidate_index,
    )
    assert gold is not None
    device = student_candidate_logits.device
    mask = mask.to(device=device)
    gold = gold.to(device=device)

    tau = max(float(temperature), 1e-6)
    mask_value = torch.finfo(torch.float32).min
    student_scaled = (student_candidate_logits.float() / tau).masked_fill(
        ~mask,
        mask_value,
    )
    teacher_scaled = (teacher_candidate_logits.float() / tau).masked_fill(
        ~mask,
        mask_value,
    )

    student_log_prob = F.log_softmax(student_scaled, dim=-1)
    teacher_log_prob = F.log_softmax(teacher_scaled, dim=-1)
    student_prob = student_log_prob.exp()
    teacher_prob = teacher_log_prob.exp()
    mixture = 0.5 * (student_prob + teacher_prob)
    log_mixture = mixture.clamp_min(1e-12).log()

    student_js_term = torch.where(
        mask,
        student_prob * (student_log_prob - log_mixture),
        torch.zeros_like(student_prob),
    )
    teacher_js_term = torch.where(
        mask,
        teacher_prob * (teacher_log_prob - log_mixture),
        torch.zeros_like(teacher_prob),
    )
    js = 0.5 * (
        student_js_term.sum(dim=-1)
        + teacher_js_term.sum(dim=-1)
    )

    student_gold_nll = -student_log_prob.gather(1, gold[:, None]).squeeze(1)
    teacher_gold_nll = -teacher_log_prob.gather(1, gold[:, None]).squeeze(1)
    information_gain = student_gold_nll - teacher_gold_nll

    teacher_gold_logits = teacher_scaled.gather(
        1,
        gold[:, None],
    ).squeeze(1)
    non_gold_mask = mask.clone()
    non_gold_mask.scatter_(1, gold[:, None], False)
    strongest_other = teacher_scaled.masked_fill(
        ~non_gold_mask,
        mask_value,
    ).max(dim=-1).values
    teacher_margin = teacher_gold_logits - strongest_other
    teacher_correct = teacher_scaled.argmax(dim=-1).eq(gold)

    gain_tau = max(float(gain_temperature), 1e-6)
    confidence_tau = max(float(confidence_temperature), 1e-6)
    gain_gate = torch.sigmoid(
        (information_gain.detach() - float(gain_margin)) / gain_tau
    )
    confidence_gate = torch.sigmoid(
        (teacher_margin.detach() - float(confidence_margin))
        / confidence_tau
    )
    confidence_gate = confidence_gate.pow(max(0.0, float(confidence_power)))
    gate = gain_gate * confidence_gate
    if bool(require_teacher_correct):
        gate = gate * teacher_correct.to(dtype=gate.dtype)
    gate = gate.detach()

    gate_sum = gate.sum()
    if float(gate_sum.item()) <= 0.0:
        loss = student_candidate_logits.sum() * 0.0
    else:
        loss = (js * gate).sum() / gate_sum.clamp_min(1e-12)

    stats = {
        "gate_mean": gate.mean().detach(),
        "active_fraction": gate.gt(0.5).float().mean().detach(),
        "teacher_correct_fraction": teacher_correct.float().mean().detach(),
        "information_gain_mean": information_gain.mean().detach(),
        "teacher_margin_mean": teacher_margin.mean().detach(),
        "js_mean": js.mean().detach(),
        "weighted_js": loss.detach(),
    }
    return loss, stats


def sage_candidate_information_gain_js(
    *,
    student_candidate_logits: torch.Tensor,
    teacher_candidate_logits: torch.Tensor,
    candidate_mask: torch.Tensor,
    gold_candidate_index: torch.Tensor,
    temperature: float,
    gain_margin: float,
    gain_temperature: float,
    confidence_margin: float,
    confidence_temperature: float,
    confidence_power: float,
    require_teacher_correct: bool,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    return _sage_from_candidate_logits(
        student_candidate_logits=student_candidate_logits,
        teacher_candidate_logits=teacher_candidate_logits,
        candidate_mask=candidate_mask,
        gold_candidate_index=gold_candidate_index,
        temperature=temperature,
        gain_margin=gain_margin,
        gain_temperature=gain_temperature,
        confidence_margin=confidence_margin,
        confidence_temperature=confidence_temperature,
        confidence_power=confidence_power,
        require_teacher_correct=require_teacher_correct,
    )


def sage_information_gain_js(
    *,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    input_ids: torch.Tensor,
    token_mask: torch.Tensor,
    temperature: float,
    topk: int,
    gain_margin: float,
    gain_temperature: float,
    confidence_margin: float,
    confidence_temperature: float,
    confidence_power: float,
    require_teacher_correct: bool,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if student_logits.shape != teacher_logits.shape or student_logits.dim() != 3:
        raise ValueError("student and teacher logits must match [B, T, V]")
    if input_ids.shape != student_logits.shape[:2]:
        raise ValueError("input_ids must match logits sequence dimensions")

    mask = token_mask.to(
        device=student_logits.device,
        dtype=torch.bool,
    )
    if mask.shape == input_ids.shape:
        mask = mask[:, 1:]
    expected = (input_ids.size(0), input_ids.size(1) - 1)
    if tuple(mask.shape) != expected:
        raise ValueError(
            f"token_mask shape {tuple(mask.shape)} does not match {expected}"
        )

    selected = torch.nonzero(mask, as_tuple=False)
    if selected.numel() == 0:
        zero = student_logits.sum() * 0.0
        detached = zero.detach()
        return zero, {
            "gate_mean": detached,
            "active_fraction": detached,
            "teacher_correct_fraction": detached,
            "information_gain_mean": detached,
            "teacher_margin_mean": detached,
            "js_mean": detached,
            "weighted_js": detached,
            "token_count": detached,
        }

    vocab_size = int(student_logits.size(-1))
    keep = min(max(1, int(topk)), vocab_size)
    student_rows: List[torch.Tensor] = []
    teacher_rows: List[torch.Tensor] = []
    gold_indices: List[int] = []

    for batch_index, shifted_index in selected.tolist():
        student_row = student_logits[batch_index, shifted_index].float()
        teacher_row = teacher_logits[batch_index, shifted_index].float()
        gold_token = int(input_ids[batch_index, shifted_index + 1].item())

        candidate_ids = torch.unique(
            torch.cat(
                (
                    torch.topk(student_row.detach(), k=keep).indices,
                    torch.topk(teacher_row.detach(), k=keep).indices,
                    torch.tensor(
                        [gold_token],
                        device=student_row.device,
                        dtype=torch.long,
                    ),
                )
            ),
            sorted=False,
        )
        student_rows.append(student_row.index_select(0, candidate_ids))
        teacher_rows.append(teacher_row.index_select(0, candidate_ids))
        gold_position = torch.nonzero(
            candidate_ids.eq(gold_token),
            as_tuple=False,
        ).view(-1)
        if gold_position.numel() != 1:
            raise RuntimeError("failed to place gold token in SAGE candidate set")
        gold_indices.append(int(gold_position.item()))

    width = max(int(row.numel()) for row in student_rows)
    padded_student = torch.stack(
        [F.pad(row, (0, width - int(row.numel()))) for row in student_rows]
    )
    padded_teacher = torch.stack(
        [F.pad(row, (0, width - int(row.numel()))) for row in teacher_rows]
    )
    candidate_mask = torch.stack(
        [
            torch.arange(width, device=row.device) < int(row.numel())
            for row in student_rows
        ]
    )
    gold = torch.tensor(
        gold_indices,
        device=student_logits.device,
        dtype=torch.long,
    )

    loss, stats = _sage_from_candidate_logits(
        student_candidate_logits=padded_student,
        teacher_candidate_logits=padded_teacher,
        candidate_mask=candidate_mask,
        gold_candidate_index=gold,
        temperature=temperature,
        gain_margin=gain_margin,
        gain_temperature=gain_temperature,
        confidence_margin=confidence_margin,
        confidence_temperature=confidence_temperature,
        confidence_power=confidence_power,
        require_teacher_correct=require_teacher_correct,
    )
    stats["token_count"] = torch.tensor(
        float(selected.size(0)),
        device=student_logits.device,
    )
    return loss, stats


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
