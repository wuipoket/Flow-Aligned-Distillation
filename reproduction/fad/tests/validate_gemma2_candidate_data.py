#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--cutoff-len", type=int, default=384)
    args = parser.parse_args()

    core = Path(__file__).resolve().parents[1] / "core"
    sys.path.insert(0, str(core))

    from sage_information_bottleneck_final_gemma2 import (
        collate_decision_batch,
        decision_candidate_labels,
        tokenize_decision_records,
    )

    records = json.loads(
        Path(args.data).read_text(encoding="utf-8")
    )
    first_by_label = {}
    for record in records:
        label = str(record.get("answer", "")).strip().lower()
        first_by_label.setdefault(label, record)

    expected = {
        "answer1", "answer2", "answer3", "answer4", "answer5",
        "ending1", "ending2", "ending3", "ending4",
        "option1", "option2", "solution1", "solution2",
        "true", "false",
    }
    if set(first_by_label) != expected:
        raise AssertionError(
            f"unexpected decision labels: {sorted(first_by_label)}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    selected = [
        first_by_label[label]
        for label in sorted(first_by_label)
    ]
    tokenized = tokenize_decision_records(
        tokenizer,
        selected,
        cutoff_len=int(args.cutoff_len),
    )
    if len(tokenized) != len(selected):
        raise AssertionError(
            f"tokenized {len(tokenized)} of {len(selected)} label probes"
        )

    for record, item in zip(selected, tokenized):
        labels = decision_candidate_labels(record)
        gold = str(record["answer"]).strip().lower()
        gold_index = int(item["gold_candidate_index"])
        if labels[gold_index] != gold:
            raise AssertionError((gold, labels, gold_index))

        eos = tokenizer.eos_token_id
        content_ids = [
            int(token_id)
            for token_id in item["input_ids"]
            if eos is None or int(token_id) != int(eos)
        ]
        gold_token_id = int(item["candidate_token_ids"][gold_index])
        if content_ids[-1] != gold_token_id:
            raise AssertionError(
                (gold, content_ids[-1], gold_token_id)
            )
        print(
            f"{gold:10s} candidates={labels} "
            f"gold_token_id={gold_token_id}"
        )

    batch = collate_decision_batch(
        tokenized,
        pad_token_id=int(tokenizer.pad_token_id),
    )
    required = {
        "input_ids",
        "attention_mask",
        "prompt_lens",
        "candidate_token_ids",
        "candidate_mask",
        "gold_candidate_index",
    }
    if not required.issubset(batch):
        raise AssertionError(
            f"missing batch keys: {sorted(required - set(batch))}"
        )
    if batch["candidate_token_ids"].shape != batch["candidate_mask"].shape:
        raise AssertionError("candidate tensor shapes do not match")
    if batch["candidate_token_ids"].size(0) != len(selected):
        raise AssertionError("candidate batch size mismatch")
    if batch["candidate_token_ids"].size(1) != 5:
        raise AssertionError("five-way answer candidate was not preserved")
    if batch["gold_candidate_index"].dtype != torch.long:
        raise AssertionError("gold_candidate_index must be torch.long")

    print("Gemma decision candidate data contract: PASS")


if __name__ == "__main__":
    main()
