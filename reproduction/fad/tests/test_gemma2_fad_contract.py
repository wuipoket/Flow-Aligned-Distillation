#!/usr/bin/env python3
from __future__ import annotations

import math
import sys
import types
import unittest
from pathlib import Path

import torch


CORE = Path(__file__).resolve().parents[1] / "core"
sys.path.insert(0, str(CORE))

# Keep the CPU numerical tests self-contained. The real geometry helper and
# Gemma model integration are exercised by the separate preflight command.
geometry_stub = types.ModuleType(
    "thesis_geometry_redundancy_final_gemma2"
)


def _stub_collate(batch, pad_token_id):
    width = max(len(item["input_ids"]) for item in batch)
    ids = []
    masks = []
    prompt_lens = []
    for item in batch:
        padding = width - len(item["input_ids"])
        ids.append([pad_token_id] * padding + list(item["input_ids"]))
        masks.append([0] * padding + list(item["attention_mask"]))
        prompt_lens.append(int(item["prompt_len"]))
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "attention_mask": torch.tensor(masks, dtype=torch.long),
        "prompt_lens": torch.tensor(prompt_lens, dtype=torch.long),
    }


def _stub_last_pred_indices(
    attention_mask,
    input_ids,
    eos_token_id,
    device=None,
):
    result = []
    for row in range(input_ids.size(0)):
        valid = torch.nonzero(
            attention_mask[row].bool(),
            as_tuple=False,
        ).view(-1)
        content = valid
        if eos_token_id is not None:
            content = valid[
                input_ids[row].index_select(0, valid) != int(eos_token_id)
            ]
        last_content = int(
            (content[-1] if content.numel() else valid[-1]).item()
        )
        previous = valid[valid < last_content]
        result.append(
            int(previous[-1].item()) if previous.numel() else last_content
        )
    return torch.tensor(
        result,
        dtype=torch.long,
        device=device or input_ids.device,
    )


geometry_stub.collate_tokenized_batch = _stub_collate
geometry_stub.last_pred_indices = _stub_last_pred_indices
geometry_stub.tokenize_records = lambda *args, **kwargs: []
sys.modules[geometry_stub.__name__] = geometry_stub

from sage_information_bottleneck_final_gemma2 import (  # noqa: E402
    COMPATIBILITY_MODE,
    candidate_decision_cross_entropy,
    candidate_decision_logits,
    collate_decision_batch,
    decision_candidate_labels,
    sage_candidate_information_gain_js,
    sage_information_gain_js,
    sage_rate_at_step,
    shifted_target_mask,
)


class GemmaFADContractTest(unittest.TestCase):
    def test_candidate_label_contracts(self):
        cases = [
            (
                {
                    "instruction": "Question\nAnswer1: a Answer2: b Answer3: c\n"
                    "Answer format: answer1/answer2/answer3",
                    "answer": "answer2",
                },
                ["answer1", "answer2", "answer3"],
            ),
            (
                {
                    "instruction": "Question\nAnswer format: answer1/answer2/"
                    "answer3/answer4/answer5",
                    "answer": "answer5",
                },
                ["answer1", "answer2", "answer3", "answer4", "answer5"],
            ),
            (
                {
                    "instruction": "Question\nAnswer format: true/false",
                    "answer": "false",
                },
                ["true", "false"],
            ),
            (
                {
                    "instruction": "Question Option1: x Option2: y",
                    "answer": "option1",
                },
                ["option1", "option2"],
            ),
        ]
        for record, expected in cases:
            self.assertEqual(decision_candidate_labels(record), expected)

    def test_candidate_collator_padding(self):
        batch = [
            {
                "input_ids": [1, 10, 11, 2],
                "attention_mask": [1, 1, 1, 1],
                "prompt_len": 2,
                "candidate_token_ids": [11, 12, 13],
                "candidate_mask": [True, True, True],
                "gold_candidate_index": 0,
            },
            {
                "input_ids": [1, 20, 21, 2],
                "attention_mask": [1, 1, 1, 1],
                "prompt_len": 2,
                "candidate_token_ids": [21, 22],
                "candidate_mask": [True, True],
                "gold_candidate_index": 1,
            },
        ]
        result = collate_decision_batch(batch, pad_token_id=0)
        self.assertEqual(tuple(result["candidate_token_ids"].shape), (2, 3))
        self.assertEqual(result["candidate_mask"].tolist(), [[True] * 3, [True, True, False]])
        self.assertEqual(result["gold_candidate_index"].tolist(), [0, 1])

    def test_shifted_masks_with_left_padding(self):
        input_ids = torch.tensor(
            [
                [0, 0, 1, 10, 11, 2],
                [0, 1, 20, 21, 22, 2],
            ]
        )
        attention_mask = torch.tensor(
            [
                [0, 0, 1, 1, 1, 1],
                [0, 1, 1, 1, 1, 1],
            ]
        )
        prompt_lens = torch.tensor([2, 2])
        response = shifted_target_mask(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_lens=prompt_lens,
            scope="response",
            pad_token_id=0,
            eos_token_id=2,
            exclude_eos=True,
        )
        decision = shifted_target_mask(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_lens=prompt_lens,
            scope="decision",
            pad_token_id=0,
            eos_token_id=2,
            exclude_eos=True,
        )
        self.assertEqual(response.sum(dim=1).tolist(), [1, 2])
        self.assertEqual(decision.sum(dim=1).tolist(), [1, 1])

    def test_candidate_ce_position_mask_and_gradient(self):
        input_ids = torch.tensor(
            [
                [1, 8, 11, 2],
                [0, 1, 9, 22],
            ]
        )
        attention_mask = torch.tensor(
            [
                [1, 1, 1, 1],
                [0, 1, 1, 1],
            ]
        )
        candidate_ids = torch.tensor(
            [
                [10, 11, 12],
                [21, 22, 0],
            ]
        )
        candidate_mask = torch.tensor(
            [
                [True, True, True],
                [True, True, False],
            ]
        )
        gold = torch.tensor([1, 1])
        logits = torch.zeros(2, 4, 32, requires_grad=True)
        with torch.no_grad():
            logits[0, 1, 11] = 5.0
            logits[1, 2, 22] = 5.0

        selected = candidate_decision_logits(
            logits=logits,
            input_ids=input_ids,
            attention_mask=attention_mask,
            candidate_token_ids=candidate_ids,
            candidate_mask=candidate_mask,
            eos_token_id=2,
        )
        self.assertEqual(tuple(selected.shape), (2, 3))
        self.assertTrue(torch.isneginf(selected[1, 2]))

        loss, stats = candidate_decision_cross_entropy(
            logits=logits,
            input_ids=input_ids,
            attention_mask=attention_mask,
            candidate_token_ids=candidate_ids,
            candidate_mask=candidate_mask,
            gold_candidate_index=gold,
            eos_token_id=2,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertLess(float(loss.item()), 0.03)
        self.assertEqual(float(stats["accuracy"].item()), 1.0)
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_candidate_sage_is_finite_and_differentiable(self):
        student = torch.tensor(
            [[4.0, 1.0, 0.0], [0.0, 3.0, float("-inf")]],
            requires_grad=True,
        )
        teacher = torch.tensor(
            [[0.0, 5.0, 1.0], [0.0, 5.0, float("-inf")]]
        )
        mask = torch.tensor(
            [[True, True, True], [True, True, False]]
        )
        gold = torch.tensor([1, 1])

        loss, stats = sage_candidate_information_gain_js(
            student_candidate_logits=student,
            teacher_candidate_logits=teacher,
            candidate_mask=mask,
            gold_candidate_index=gold,
            temperature=1.0,
            gain_margin=0.0,
            gain_temperature=0.25,
            confidence_margin=0.0,
            confidence_temperature=1.0,
            confidence_power=1.0,
            require_teacher_correct=True,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(loss.item()), 0.0)
        self.assertGreater(float(stats["gate_mean"].item()), 0.0)
        self.assertEqual(float(stats["teacher_correct_fraction"].item()), 1.0)
        loss.backward()
        self.assertIsNotNone(student.grad)
        self.assertTrue(torch.isfinite(student.grad).all())

    def test_full_vocab_sage(self):
        input_ids = torch.tensor([[1, 5, 7, 2]])
        mask = torch.tensor([[False, True, False]])
        student = torch.zeros(1, 4, 16, requires_grad=True)
        teacher = torch.zeros(1, 4, 16)
        with torch.no_grad():
            student[0, 1, 3] = 5.0
            teacher[0, 1, 7] = 5.0

        loss, stats = sage_information_gain_js(
            student_logits=student,
            teacher_logits=teacher,
            input_ids=input_ids,
            token_mask=mask,
            temperature=1.0,
            topk=4,
            gain_margin=0.0,
            gain_temperature=0.25,
            confidence_margin=0.0,
            confidence_temperature=1.0,
            confidence_power=1.0,
            require_teacher_correct=True,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(stats["token_count"].item()), 1.0)
        loss.backward()
        self.assertTrue(torch.isfinite(student.grad).all())

    def test_rate_schedule_and_mode_marker(self):
        self.assertEqual(
            COMPATIBILITY_MODE,
            "reconstructed_decision_candidate_channel_v1",
        )
        self.assertTrue(
            math.isclose(
                sage_rate_at_step(
                    step=0,
                    total_steps=100,
                    max_rate=1.0,
                    warmup_ratio=0.1,
                    decay_start_ratio=0.8,
                    min_rate_ratio=0.2,
                ),
                0.0,
            )
        )
        self.assertTrue(
            math.isclose(
                sage_rate_at_step(
                    step=10,
                    total_steps=100,
                    max_rate=1.0,
                    warmup_ratio=0.1,
                    decay_start_ratio=0.8,
                    min_rate_ratio=0.2,
                ),
                1.0,
            )
        )
        self.assertTrue(
            math.isclose(
                sage_rate_at_step(
                    step=100,
                    total_steps=100,
                    max_rate=1.0,
                    warmup_ratio=0.1,
                    decay_start_ratio=0.8,
                    min_rate_ratio=0.2,
                ),
                0.2,
            )
        )

    def test_gemma_hook_contract_is_present(self):
        pipeline = CORE / "newthesis_pipeline_final_gemma2.py"
        if not pipeline.is_file():
            self.skipTest("pipeline source is not included in the patch archive")
        source = pipeline.read_text(encoding="utf-8")
        self.assertIn('SUPPORTED_DECODER_MODEL_TYPES = {"llama", "qwen2", "gemma2"}', source)
        self.assertIn('"num_hidden_layers": 42', source)
        self.assertIn('"pre_feedforward_layernorm"', source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
