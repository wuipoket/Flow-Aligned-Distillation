#!/usr/bin/env python3
from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from thesis_common_final_llama import (
    load_json_records,
    normalize_record_for_sft,
    render_phase1_prompt,
    render_phase1_prompt_prefix,
)


def build_frozen_teacher(
    teacher_ckpt_dir: str,
    base_model: str = "",
    teacher_loader: str = "auto",
    target_dtype: torch.dtype = torch.bfloat16,
    trust_remote_code: bool = False,
):
    loader = str(teacher_loader or "auto").strip().lower()
    if loader not in {"auto", "native"}:
        raise ValueError(
            "Gemma helper supports teacher_loader=auto/native only; "
            f"received {teacher_loader!r}"
        )

    source = str(teacher_ckpt_dir or base_model).strip()
    if not source:
        raise ValueError("teacher_ckpt_dir or base_model is required")

    model = AutoModelForCausalLM.from_pretrained(
        source,
        dtype=target_dtype,
        trust_remote_code=bool(trust_remote_code),
        low_cpu_mem_usage=True,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def prepare_records(
    data_path: str,
    max_records: int = 0,
    seed: int = 42,
    shuffle_records: bool = False,
) -> List[Dict[str, Any]]:
    raw_records = load_json_records(data_path)
    source_name = Path(data_path).stem
    records: List[Dict[str, Any]] = []

    for raw in raw_records:
        original = dict(raw) if isinstance(raw, dict) else {"text": str(raw)}
        normalized = normalize_record_for_sft(
            original,
            source_name=source_name,
        )
        original.update(normalized)
        if original.get("instruction") or original.get("output"):
            records.append(original)

    if shuffle_records:
        rng = random.Random(int(seed))
        rng.shuffle(records)

    if int(max_records) > 0:
        records = records[: int(max_records)]

    return records


def tokenize_records(
    tokenizer,
    records: Sequence[Dict[str, Any]],
    cutoff_len: int,
) -> List[Dict[str, Any]]:
    tokenized: List[Dict[str, Any]] = []
    cutoff = max(2, int(cutoff_len))

    for record in records:
        prompt_text = render_phase1_prompt_prefix(record)
        full_text = render_phase1_prompt(record)

        encoded = tokenizer(
            full_text,
            truncation=True,
            max_length=cutoff,
            padding=False,
            add_special_tokens=True,
        )
        prompt_encoded = tokenizer(
            prompt_text,
            truncation=True,
            max_length=cutoff,
            padding=False,
            add_special_tokens=True,
        )

        input_ids = list(encoded.get("input_ids", []))
        attention_mask = list(
            encoded.get("attention_mask", [1] * len(input_ids))
        )

        eos_token_id = tokenizer.eos_token_id
        if (
            eos_token_id is not None
            and input_ids
            and input_ids[-1] != int(eos_token_id)
            and len(input_ids) < cutoff
        ):
            input_ids.append(int(eos_token_id))
            attention_mask.append(1)

        if len(input_ids) < 2:
            continue

        prompt_len = min(
            len(prompt_encoded.get("input_ids", [])),
            len(input_ids),
        )
        if prompt_len >= len(input_ids):
            continue

        tokenized.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "prompt_len": int(prompt_len),
            }
        )

    return tokenized


def collate_tokenized_batch(
    batch: Sequence[Dict[str, Any]],
    pad_token_id: int,
) -> Dict[str, torch.Tensor]:
    if not batch:
        raise ValueError("empty batch")

    width = max(len(item["input_ids"]) for item in batch)
    input_rows = []
    mask_rows = []
    prompt_lens = []

    for item in batch:
        ids = list(item["input_ids"])
        mask = list(item.get("attention_mask", [1] * len(ids)))
        pad = width - len(ids)

        input_rows.append([int(pad_token_id)] * pad + ids)
        mask_rows.append([0] * pad + mask)
        prompt_lens.append(int(item.get("prompt_len", 0)))

    return {
        "input_ids": torch.tensor(input_rows, dtype=torch.long),
        "attention_mask": torch.tensor(mask_rows, dtype=torch.long),
        "prompt_lens": torch.tensor(prompt_lens, dtype=torch.long),
    }


def last_content_indices(
    attention_mask: torch.Tensor,
    input_ids: torch.Tensor,
    eos_token_id: Optional[int],
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    target_device = device or input_ids.device
    results = []

    for row in range(int(input_ids.size(0))):
        valid = torch.nonzero(
            attention_mask[row].to(dtype=torch.bool),
            as_tuple=False,
        ).view(-1)

        if valid.numel() == 0:
            results.append(0)
            continue

        if eos_token_id is not None:
            non_eos = valid[
                input_ids[row].index_select(0, valid)
                != int(eos_token_id)
            ]
            chosen = non_eos[-1] if non_eos.numel() else valid[-1]
        else:
            chosen = valid[-1]

        results.append(int(chosen.item()))

    return torch.tensor(results, dtype=torch.long, device=target_device)


def last_pred_indices(
    attention_mask: torch.Tensor,
    input_ids: torch.Tensor,
    eos_token_id: Optional[int],
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    target_device = device or input_ids.device
    content = last_content_indices(
        attention_mask,
        input_ids,
        eos_token_id,
        input_ids.device,
    )
    results = []

    for row, content_index in enumerate(content.tolist()):
        valid = torch.nonzero(
            attention_mask[row].to(dtype=torch.bool),
            as_tuple=False,
        ).view(-1)
        previous = valid[valid < int(content_index)]
        chosen = previous[-1] if previous.numel() else content_index
        results.append(int(chosen))

    return torch.tensor(results, dtype=torch.long, device=target_device)


def build_last_k_eval_indices(
    centers: torch.Tensor,
    window_size: int,
    attention_mask: torch.Tensor,
    sample_mode: str = "all",
    random_pick_min: int = 1,
    random_pick_max: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = attention_mask.device
    size = max(1, int(window_size))
    mode = str(sample_mode or "all").strip().lower()
    batch_indices = []
    token_indices = []

    for row, center in enumerate(centers.tolist()):
        valid = torch.nonzero(
            attention_mask[row].to(dtype=torch.bool),
            as_tuple=False,
        ).view(-1)
        candidates = valid[valid <= int(center)]
        if candidates.numel() == 0:
            continue

        candidates = candidates[-size:]

        if mode == "random" and candidates.numel() > 1:
            low = max(1, int(random_pick_min))
            high = max(low, int(random_pick_max))
            count = random.randint(low, high)
            count = min(count, int(candidates.numel()))
            order = torch.randperm(
                int(candidates.numel()),
                device=candidates.device,
            )[:count]
            candidates = candidates.index_select(0, order).sort().values

        batch_indices.append(
            torch.full(
                (int(candidates.numel()),),
                row,
                dtype=torch.long,
                device=device,
            )
        )
        token_indices.append(candidates.to(device=device, dtype=torch.long))

    if not batch_indices:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty

    return torch.cat(batch_indices), torch.cat(token_indices)


@torch.no_grad()
def capture_hidden_states(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
):
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True,
        use_cache=False,
    )
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None:
        raise RuntimeError("model did not return hidden_states")
    return list(hidden_states)
