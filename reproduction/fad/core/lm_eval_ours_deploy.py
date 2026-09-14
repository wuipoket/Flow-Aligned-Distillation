#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from lm_eval import evaluator
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from tqdm import tqdm


DEFAULT_PIPELINE = str(Path(__file__).with_name("newthesis_pipeline_final_llama.py"))


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid bool: {value!r}")


def dtype_from_name(name: str) -> torch.dtype:
    text = str(name).strip().lower()
    if text in {"auto", ""}:
        if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
            return torch.bfloat16
        if torch.cuda.is_available():
            return torch.float16
        return torch.float32
    if text in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if text in {"fp16", "float16", "half"}:
        return torch.float16
    if text in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unknown dtype: {name}")


def load_pipeline_module(path: str):
    spec = importlib.util.spec_from_file_location("newthesis_pipeline_for_lm_eval", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import pipeline from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sanitize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): sanitize_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


class OursDeployLM(LM):
    def __init__(
        self,
        *,
        deploy_bundle: str,
        pipeline_py: str = DEFAULT_PIPELINE,
        base_model_name_or_path: str = "",
        tokenizer_name_or_path: str = "",
        dtype: str = "auto",
        device: str = "cuda:0",
        stage_on_cpu: bool = False,
        batch_size: int | str = 4,
        max_length: int | None = None,
        max_gen_toks: int = 256,
        trust_remote_code: bool = True,
        use_quant_bank_int4: bool = False,
    ) -> None:
        super().__init__()
        self.deploy_bundle = str(deploy_bundle)
        self.pipeline_py = str(pipeline_py)
        self.base_model_name_or_path = str(
            base_model_name_or_path
        ).strip()
        self.stage_on_cpu = bool(stage_on_cpu)
        self._device = torch.device(
            device
            if torch.cuda.is_available() or str(device) == "cpu"
            else "cpu"
        )
        self._dtype = dtype_from_name(dtype)
        self._batch_size = int(batch_size) if str(batch_size).strip().lower() != "auto" else 4
        self._max_length_override = int(max_length) if max_length else None
        self._max_gen_toks = int(max_gen_toks)
        self.trust_remote_code = bool(trust_remote_code)
        self.use_quant_bank_int4 = bool(use_quant_bank_int4)

        pipe = load_pipeline_module(self.pipeline_py)
        bundle = torch.load(
            self.deploy_bundle,
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )

        embedded_base_model = str(bundle["base_model"])
        self.base_model = (
            self.base_model_name_or_path
            or embedded_base_model
        )

        build_device = (
            torch.device("cpu")
            if (
                self.stage_on_cpu
                and self._device.type == "cuda"
            )
            else self._device
        )

        print(
            "[Ours-lm-eval] model build device="
            f"{build_device} target={self._device}"
        )

        self.model, self.quant_eval = (
            pipe._build_shared_model_for_eval(
                base_model=self.base_model,
                atlas_payload=bundle.get("atlas", {}),
                shared_payload=bundle["shared_student"],
                quant_bank_int4=bundle.get(
                    "quant_bank_int4"
                ),
                use_quant_bank_int4=(
                    self.use_quant_bank_int4
                ),
                device=build_device,
                dtype=self._dtype,
                trust_remote_code=self.trust_remote_code,
            )
        )

        del bundle
        gc.collect()

        if build_device != self._device:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(
                "[Ours-lm-eval] moving compressed "
                f"student to {self._device}"
            )

            self.model.to(
                device=self._device,
                dtype=self._dtype,
            )

        tokenizer_name = (
            str(tokenizer_name_or_path).strip()
            or self.base_model
        )
        self.tokenizer = pipe.load_tokenizer(tokenizer_name, trust_remote_code=self.trust_remote_code)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()
        self.model_name = Path(self.deploy_bundle).parent.parent.name

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def eot_token_id(self) -> int:
        return int(self.tokenizer.eos_token_id)

    @property
    def prefix_token_id(self) -> int:
        if self.tokenizer.bos_token_id is not None:
            return int(self.tokenizer.bos_token_id)
        return int(self.tokenizer.eos_token_id)

    @property
    def max_length(self) -> int:
        if self._max_length_override is not None:
            return self._max_length_override
        config = getattr(self.model, "config", None)
        for attr in ("n_positions", "max_position_embeddings", "n_ctx"):
            if config is not None and hasattr(config, attr):
                value = int(getattr(config, attr))
                if value > 0:
                    return value
        value = int(getattr(self.tokenizer, "model_max_length", 4096))
        if value > 1_000_000:
            return 4096
        return value

    @property
    def max_gen_toks(self) -> int:
        return self._max_gen_toks

    @property
    def tokenizer_name(self) -> str:
        return str(getattr(self.tokenizer, "name_or_path", self.base_model)).replace("/", "__")

    def get_model_info(self) -> dict[str, Any]:
        return {
            "model": "ours_deploy",
            "deploy_bundle": self.deploy_bundle,
            "pipeline_py": self.pipeline_py,
            "base_model": self.base_model,
            "dtype": str(self._dtype).replace("torch.", ""),
            "batch_size": self.batch_size,
            "quant_eval": self.quant_eval,
        }

    def tok_encode(self, text: str, *, add_special_tokens: bool | None = None) -> list[int]:
        kwargs: dict[str, Any] = {}
        if add_special_tokens is not None:
            kwargs["add_special_tokens"] = add_special_tokens
        return list(self.tokenizer.encode(text, **kwargs))

    def tok_decode(self, tokens: list[int] | torch.Tensor, *, skip_special_tokens: bool = True) -> str:
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.detach().cpu().tolist()
        return str(self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens))

    def _encode_pair(self, context: str, continuation: str) -> tuple[list[int], list[int]]:
        if context == "":
            return [self.prefix_token_id], self.tok_encode(continuation, add_special_tokens=False)

        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tok_encode(context + continuation)
        context_enc = self.tok_encode(context)
        continuation_enc = whole_enc[len(context_enc) :]
        if not continuation_enc:
            continuation_enc = self.tok_encode(continuation, add_special_tokens=False)
        return context_enc, continuation_enc

    @torch.no_grad()
    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        encoded: list[tuple[int, tuple[str, str], list[int], list[int]]] = []
        for idx, req in enumerate(requests):
            context, continuation = req.args
            context_enc, continuation_enc = self._encode_pair(str(context), str(continuation))
            if not continuation_enc:
                encoded.append((idx, (str(context), str(continuation)), context_enc, [self.eot_token_id]))
            else:
                encoded.append((idx, (str(context), str(continuation)), context_enc, continuation_enc))

        encoded.sort(key=lambda item: -(len(item[2]) + len(item[3])))
        ordered_results: list[tuple[int, tuple[float, bool], tuple[str, str]]] = []
        pbar = tqdm(total=len(encoded), disable=(self.rank != 0), desc="Running Ours loglikelihood")
        for start in range(0, len(encoded), self.batch_size):
            chunk = encoded[start : start + self.batch_size]
            seqs: list[list[int]] = []
            conts: list[list[int]] = []
            in_lens: list[int] = []
            for _, _, context_enc, continuation_enc in chunk:
                tokens = (context_enc + continuation_enc)[-(self.max_length + 1) :]
                inp = tokens[:-1]
                if not inp:
                    inp = [self.prefix_token_id]
                seqs.append(inp)
                conts.append(continuation_enc)
                in_lens.append(len(inp))

            max_len = max(len(seq) for seq in seqs)
            pad_id = int(self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.eot_token_id)
            input_ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long, device=self.device)
            attention_mask = torch.zeros((len(seqs), max_len), dtype=torch.long, device=self.device)
            for row, seq in enumerate(seqs):
                input_ids[row, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=self.device)
                attention_mask[row, : len(seq)] = 1

            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )

            # Only convert continuation-token logits to FP32. Converting the
            # complete [batch, sequence, vocabulary] tensor can require several
            # additional GiB for Gemma 2's large vocabulary.
            for row, item in enumerate(chunk):
                original_idx, request_str, _, continuation_enc = item
                cont_len = len(continuation_enc)
                score_logits = out.logits[
                    row,
                    in_lens[row] - cont_len : in_lens[row],
                    :,
                ]
                continuation_tensor = torch.tensor(
                    continuation_enc,
                    dtype=torch.long,
                    device=self.device,
                )
                token_log_probs = F.log_softmax(
                    score_logits.float(),
                    dim=-1,
                ).gather(
                    1,
                    continuation_tensor[:, None],
                ).squeeze(1)
                greedy = bool(
                    torch.equal(
                        score_logits.argmax(dim=-1),
                        continuation_tensor,
                    )
                )
                answer = (
                    float(token_log_probs.sum().item()),
                    greedy,
                )
                ordered_results.append(
                    (original_idx, answer, request_str)
                )
                self.cache_hook.add_partial(
                    "loglikelihood",
                    request_str,
                    answer,
                )
                pbar.update(1)

            del score_logits
            del continuation_tensor
            del token_log_probs
            del out
        pbar.close()

        ordered_results.sort(key=lambda item: item[0])
        return [answer for _, answer, _ in ordered_results]

    @torch.no_grad()
    def loglikelihood_rolling(self, requests: list[Instance]) -> list[float]:
        outputs: list[float] = []
        for req in tqdm(requests, disable=(self.rank != 0), desc="Running Ours rolling loglikelihood"):
            (text,) = req.args
            ids = self.tok_encode(str(text), add_special_tokens=False)
            if not ids:
                outputs.append(0.0)
                continue
            total = 0.0
            stride = max(1, self.max_length)
            prefix = [self.prefix_token_id]
            for start in range(0, len(ids), stride):
                chunk = ids[start : start + stride]
                context = prefix if start == 0 else ids[max(0, start - 1) : start]
                score, _ = self.loglikelihood(
                    [
                        Instance(
                            request_type="loglikelihood",
                            doc={},
                            arguments=(self.tok_decode(context, skip_special_tokens=False), self.tok_decode(chunk, skip_special_tokens=False)),
                            idx=0,
                        )
                    ]
                )[0]
                total += score
            outputs.append(float(total))
            self.cache_hook.add_partial("loglikelihood_rolling", (text,), float(total))
        return outputs

    @torch.no_grad()
    def generate_until(self, requests: list[Instance]) -> list[str]:
        results: list[str] = []
        for start in tqdm(range(0, len(requests), self.batch_size), disable=(self.rank != 0), desc="Running Ours generate_until"):
            chunk = requests[start : start + self.batch_size]
            contexts: list[str] = []
            gen_kwargs: dict[str, Any] = {}
            for req in chunk:
                context, kwargs = req.args
                contexts.append(str(context))
                if isinstance(kwargs, dict):
                    gen_kwargs = kwargs

            until = gen_kwargs.get("until", [])
            if isinstance(until, str):
                until = [until]
            max_new = int(gen_kwargs.get("max_gen_toks", gen_kwargs.get("max_new_tokens", self.max_gen_toks)))
            enc = self.tokenizer(
                contexts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max(1, self.max_length - max_new),
            ).to(self.device)
            generated = self.model.generate(
                **enc,
                do_sample=bool(gen_kwargs.get("do_sample", False)),
                temperature=float(gen_kwargs.get("temperature", 1.0)),
                max_new_tokens=max_new,
                pad_token_id=self.tokenizer.pad_token_id or self.eot_token_id,
                eos_token_id=self.eot_token_id,
            )
            for req, output_ids in zip(chunk, generated, strict=True):
                prompt_len = int(enc["input_ids"].shape[1])
                text = self.tok_decode(output_ids[prompt_len:])
                for stop in until:
                    if stop:
                        text = text.split(str(stop))[0]
                context, kwargs = req.args
                results.append(text)
                self.cache_hook.add_partial("generate_until", (context, kwargs), text)
        return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lm-eval tasks on an Ours deploy_bundle.pt model.")
    parser.add_argument("--deploy_bundle", required=True)
    parser.add_argument("--pipeline_py", default=DEFAULT_PIPELINE)
    parser.add_argument(
        "--base_model_name_or_path",
        default="",
    )
    parser.add_argument("--tokenizer_name_or_path", default="")
    parser.add_argument("--tasks", default="mmlu")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", default="4")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--stage_on_cpu",
        type=str2bool,
        default=False,
    )
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--limit", default="")
    parser.add_argument("--bootstrap_iters", type=int, default=100000)
    parser.add_argument("--log_samples", type=str2bool, default=False)
    parser.add_argument("--trust_remote_code", type=str2bool, default=True)
    parser.add_argument("--use_quant_bank_int4", type=str2bool, default=False)
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--numpy_random_seed", type=int, default=1234)
    parser.add_argument("--torch_random_seed", type=int, default=1234)
    parser.add_argument("--fewshot_random_seed", type=int, default=1234)
    args = parser.parse_args()

    limit: int | float | None
    if str(args.limit).strip() == "":
        limit = None
    elif "." in str(args.limit):
        limit = float(args.limit)
    else:
        limit = int(args.limit)

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    lm = OursDeployLM(
        deploy_bundle=args.deploy_bundle,
        pipeline_py=args.pipeline_py,
        base_model_name_or_path=(
            args.base_model_name_or_path
        ),
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        dtype=args.dtype,
        device=args.device,
        stage_on_cpu=args.stage_on_cpu,
        batch_size=args.batch_size,
        trust_remote_code=args.trust_remote_code,
        use_quant_bank_int4=args.use_quant_bank_int4,
    )
    tasks = [task.strip() for task in str(args.tasks).split(",") if task.strip()]
    started = time.strftime("%Y-%m-%dT%H-%M-%S")
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=int(args.num_fewshot),
        batch_size=lm.batch_size,
        device=str(lm.device),
        limit=limit,
        bootstrap_iters=int(args.bootstrap_iters),
        log_samples=bool(args.log_samples),
        random_seed=int(args.random_seed),
        numpy_random_seed=int(args.numpy_random_seed),
        torch_random_seed=int(args.torch_random_seed),
        fewshot_random_seed=int(args.fewshot_random_seed),
    )
    if results is None:
        raise RuntimeError(
            "lm-eval returned no results on this rank"
        )

    if isinstance(results, dict):
        results["custom_metadata"] = {
            "deploy_bundle": args.deploy_bundle,
            "pipeline_py": args.pipeline_py,
            "base_model": lm.base_model,
            "runner": "lm_eval_ours_deploy.py",
            "dtype": args.dtype,
            "stage_on_cpu": bool(
                args.stage_on_cpu
            ),
            "use_quant_bank_int4": bool(
                args.use_quant_bank_int4
            ),
        }

    out_file = output_path / f"results_{started}.json"
    with out_file.open("w", encoding="utf-8") as handle:
        json.dump(sanitize_json(results), handle, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"[Ours-lm-eval] wrote {out_file}")


if __name__ == "__main__":
    main()
