#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import glob
import importlib.util
import inspect
import json
import math
import os
import random
import re
import shutil
import sys
import time
import warnings
from decimal import Decimal, InvalidOperation
from dataclasses import dataclass
from datetime import datetime
from contextlib import ExitStack, contextmanager
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

warnings.filterwarnings(
    "ignore",
    message=r".*urllib3 .*charset_normalizer.*doesn't match a supported version.*",
)
warnings.filterwarnings(
    "ignore",
    message=r"`torch_dtype` is deprecated! Use `dtype` instead!",
)

from transformers import AutoModelForCausalLM, AutoTokenizer

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
WORKSPACE_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, ".."))
DEFAULT_BASE_MODEL_LLAMA32_3B = "meta-llama/Llama-3.2-3B"
DEFAULT_BASE_MODEL_LLAMA31_8B = "meta-llama/Llama-3.1-8B"
DEFAULT_BASE_MODEL = DEFAULT_BASE_MODEL_LLAMA32_3B
DEFAULT_MERGED_TEACHER_CKPT = os.path.join(
    WORKSPACE_ROOT,
    "out",
    "lora_checkpoint_eval_20260324_161732",
    "merged_from_checkpoint",
)
DEFAULT_DATA_PATH = os.path.join(WORKSPACE_ROOT, "data", "datasets", "commonsense_170k.json")
SUPPORTED_LLAMA_MODEL_TYPE = "llama"
SUPPORTED_DECODER_MODEL_TYPES = {"llama", "qwen2", "gemma2"}
SUPPORTED_LLAMA_PROFILES = {
    "llama32_3b": {
        "display_name": "Llama-3.2-3B",
        "model_type": "llama",
        "hidden_size": 3072,
        "intermediate_size": 8192,
        "num_hidden_layers": 28,
        "num_attention_heads": 24,
        "num_key_value_heads": 8,
        "head_dim": 128,
    },
    "llama31_8b": {
        "display_name": "Llama-3.1-8B",
        "model_type": "llama",
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
    },
    "gemma2_9b": {
        "display_name": "Gemma-2-9B",
        "model_type": "gemma2",
        "hidden_size": 3584,
        "intermediate_size": 14336,
        "num_hidden_layers": 42,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 256,
    },
    "qwen25_math_7b": {
        "display_name": "Qwen2.5-Math-7B",
        "model_type": "qwen2",
        "hidden_size": 3584,
        "intermediate_size": 18944,
        "num_hidden_layers": 28,
        "num_attention_heads": 28,
        "num_key_value_heads": 4,
        "head_dim": 128,
    },
}


def _load_local_module(module_filename: str, module_alias: str):
    module_path = os.path.join(THIS_DIR, module_filename)
    spec = importlib.util.spec_from_file_location(module_alias, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load local module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_alias] = module
    spec.loader.exec_module(module)
    return module


_geometry_module = _load_local_module(
    "thesis_geometry_redundancy_final_gemma2.py",
    "thesis_geometry_redundancy_final_llama",
)
_sage_module = _load_local_module(
    "sage_information_bottleneck_final_gemma2.py",
    "sage_information_bottleneck_final_llama",
)

build_frozen_teacher = _geometry_module.build_frozen_teacher
build_last_k_eval_indices = _geometry_module.build_last_k_eval_indices
capture_hidden_states = _geometry_module.capture_hidden_states
collate_tokenized_batch = _geometry_module.collate_tokenized_batch
last_content_indices = _geometry_module.last_content_indices
last_pred_indices = _geometry_module.last_pred_indices
prepare_records = _geometry_module.prepare_records
tokenize_records = _geometry_module.tokenize_records
tokenize_decision_records = _sage_module.tokenize_decision_records
collate_decision_batch = _sage_module.collate_decision_batch
shifted_target_mask = _sage_module.shifted_target_mask
masked_next_token_cross_entropy = _sage_module.masked_next_token_cross_entropy
candidate_decision_logits = _sage_module.candidate_decision_logits
candidate_decision_cross_entropy = _sage_module.candidate_decision_cross_entropy
sage_information_gain_js = _sage_module.sage_information_gain_js
sage_candidate_information_gain_js = _sage_module.sage_candidate_information_gain_js
sage_rate_at_step = _sage_module.sage_rate_at_step

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
except Exception:
    plt = None
    LinearSegmentedColormap = None

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


HASH_LEARNING_MODE = "learned_codebook_kmeans"
HASH_IS_E2E_NEURAL = False
PREDICTOR_CTX_SOURCE = "B_T_h_l"



def str2bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid bool: {value!r}")


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, payload: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(os.path.abspath(path)) or ".")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def set_seed(seed: int) -> None:
    strict_determinism = str(os.environ.get("FAD_STRICT_DETERMINISM", "0")).strip().lower() in {
        "1", "true", "yes", "y", "on",
    }
    if strict_determinism:
        # CUBLAS_WORKSPACE_CONFIG must be exported before Python starts.  The
        # strict reproduction sbatch does that; fail early if it was omitted so
        # that a nominally "strict" run cannot silently become nondeterministic.
        cublas_workspace = str(os.environ.get("CUBLAS_WORKSPACE_CONFIG", "")).strip()
        if cublas_workspace not in {":4096:8", ":16:8"}:
            raise RuntimeError(
                "FAD_STRICT_DETERMINISM requires "
                "CUBLAS_WORKSPACE_CONFIG=:4096:8 (or :16:8) before Python starts."
            )
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        # Deterministic algorithms mode will select/validate a deterministic
        # SDPA implementation.  Disabling the fused kernels makes that choice
        # explicit and stable across H200 allocations.
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(False)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)
        if int(os.environ.get("RANK", "0")) == 0:
            print(
                "[Determinism] strict=True algorithms=True cudnn_benchmark=False "
                f"cublas_workspace={cublas_workspace} flash_sdp=False mem_efficient_sdp=False",
                flush=True,
            )
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def iter_progress(iterable, *, total: Optional[int], desc: str):
    if tqdm is None:
        return iterable
    ctx = get_distributed_context()
    disable = ctx.enabled and not is_main_process(ctx)
    return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True, leave=True, disable=disable)


def step_progress(*, total: int, desc: str, miniters: Optional[int] = None):
    if tqdm is None:
        return None
    ctx = get_distributed_context()
    disable = ctx.enabled and not is_main_process(ctx)
    kwargs: Dict[str, Any] = {
        "total": max(0, int(total)),
        "desc": desc,
        "dynamic_ncols": True,
        "leave": True,
        "disable": disable,
    }
    if miniters is not None and int(miniters) > 0:
        kwargs["miniters"] = int(miniters)
    return tqdm(**kwargs)


class _TeeStream:
    def __init__(self, primary, secondary):
        self.primary = primary
        self.secondary = secondary

    def write(self, data):
        self.primary.write(data)
        self.secondary.write(data)
        return len(data)

    def flush(self):
        self.primary.flush()
        self.secondary.flush()

    def isatty(self):
        return bool(getattr(self.primary, "isatty", lambda: False)())


@contextmanager
def tee_output_to_file(log_path: str):
    ensure_dir(os.path.dirname(os.path.abspath(log_path)) or ".")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with open(log_path, "a", encoding="utf-8", buffering=1) as handle:
        sys.stdout = _TeeStream(original_stdout, handle)
        sys.stderr = _TeeStream(original_stderr, handle)
        try:
            yield
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


@dataclass
class DistributedContext:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int


def unwrap_model(model: nn.Module) -> nn.Module:
    current = model
    while hasattr(current, "module"):
        current = current.module  # type: ignore[assignment]
    return current


def get_distributed_context() -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return DistributedContext(
        enabled=world_size > 1,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
    )


def init_distributed(device: str) -> DistributedContext:
    ctx = get_distributed_context()
    if not ctx.enabled:
        return ctx
    use_cuda = torch.cuda.is_available() and str(device).strip().lower() != "cpu"
    if use_cuda:
        torch.cuda.set_device(ctx.local_rank)
    if not dist.is_initialized():
        backend = "nccl" if use_cuda else "gloo"
        init_kwargs: Dict[str, Any] = {"backend": backend, "init_method": "env://"}
        if use_cuda and "device_id" in inspect.signature(dist.init_process_group).parameters:
            init_kwargs["device_id"] = torch.device("cuda", int(ctx.local_rank))
        dist.init_process_group(**init_kwargs)
    return ctx


def finalize_distributed(ctx: DistributedContext) -> None:
    if ctx.enabled and dist.is_initialized():
        dist_barrier(ctx)
        dist.destroy_process_group()


def is_main_process(ctx: DistributedContext) -> bool:
    return (not ctx.enabled) or int(ctx.rank) == 0


def dist_barrier(ctx: DistributedContext) -> None:
    if ctx.enabled and dist.is_initialized():
        if torch.cuda.is_available():
            dist.barrier(device_ids=[int(ctx.local_rank)])
        else:
            dist.barrier()


def dist_mean(value: float, device: torch.device, ctx: DistributedContext) -> float:
    if not ctx.enabled:
        return float(value)
    tensor = torch.tensor(float(value), device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= float(ctx.world_size)
    return float(tensor.item())


def dist_max(value: float, device: torch.device, ctx: DistributedContext) -> float:
    if not ctx.enabled:
        return float(value)
    tensor = torch.tensor(float(value), device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def resolve_device(device: str, local_rank: int = -1) -> torch.device:
    req = str(device).strip().lower()
    if req == "cpu":
        return torch.device("cpu")
    if req == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda", local_rank) if int(local_rank) >= 0 else torch.device("cuda")
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda", local_rank) if int(local_rank) >= 0 else torch.device("cuda")
    return torch.device("cpu")


def get_target_dtype(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    return torch.bfloat16


def load_tokenizer(tokenizer_name_or_path: str, *, trust_remote_code: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        use_fast=True,
        trust_remote_code=bool(trust_remote_code),
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
    tokenizer.padding_side = "left"
    return tokenizer


def _set_gradient_checkpointing(model: nn.Module, *, enabled: bool, require_input_grads: bool) -> None:
    model = unwrap_model(model)
    config = getattr(model, "config", None)
    if config is not None and hasattr(config, "use_cache"):
        config.use_cache = False
    if not enabled:
        disable_fn = getattr(model, "gradient_checkpointing_disable", None)
        if callable(disable_fn):
            disable_fn()
        return
    enable_fn = getattr(model, "gradient_checkpointing_enable", None)
    if callable(enable_fn):
        # This pipeline captures intermediate tensors via forward hooks and uses
        # them in losses outside the checkpointed decoder-layer call. With
        # non-reentrant checkpointing, PyTorch may otherwise early-stop the
        # recomputation once gradients have enough internal tensors, pruning the
        # hook side effects and triggering CheckpointError tensor-count
        # mismatches during backward.
        enable_fn(gradient_checkpointing_kwargs={"use_reentrant": False, "early_stop": False})
    if require_input_grads:
        input_grad_fn = getattr(model, "enable_input_require_grads", None)
        if callable(input_grad_fn):
            input_grad_fn()
        else:
            get_embeddings = getattr(model, "get_input_embeddings", None)
            if callable(get_embeddings):
                embeddings = get_embeddings()
                if embeddings is not None and hasattr(embeddings, "weight"):
                    embeddings.weight.requires_grad_(True)


def _infer_supported_llama_profile(model_or_config: Any) -> Optional[str]:
    config = getattr(model_or_config, "config", model_or_config)
    model_type = str(getattr(config, "model_type", "") or "").strip().lower()
    if model_type and model_type not in SUPPORTED_DECODER_MODEL_TYPES:
        return None
    for profile_name, expected in SUPPORTED_LLAMA_PROFILES.items():
        expected_model_type = str(
            expected.get("model_type", SUPPORTED_LLAMA_MODEL_TYPE)
        ).strip().lower()
        if model_type and model_type != expected_model_type:
            continue
        matched = True
        for key, expected_value in expected.items():
            if key in {"display_name", "model_type"}:
                continue
            actual = getattr(config, key, None)
            if actual is None and key == "head_dim":
                hidden_size = getattr(config, "hidden_size", None)
                num_heads = getattr(config, "num_attention_heads", None)
                if hidden_size is not None and num_heads:
                    actual = int(hidden_size) // int(num_heads)
            if actual is None or int(actual) != int(expected_value):
                matched = False
                break
        if matched:
            return profile_name
    return None


def _validate_supported_llama_config(model_or_config: Any, *, label: str) -> str:
    config = getattr(model_or_config, "config", model_or_config)
    model_type = str(getattr(config, "model_type", "") or "").strip().lower()
    if model_type and model_type not in SUPPORTED_DECODER_MODEL_TYPES:
        raise ValueError(
            f"{label} is not a supported decoder model: model_type={model_type}"
        )
    resolved_profile = _infer_supported_llama_profile(config)
    if resolved_profile is not None:
        return resolved_profile

    details: List[str] = []
    for key in ["hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim"]:
        actual = getattr(config, key, None)
        if actual is None and key == "head_dim":
            hidden_size = getattr(config, "hidden_size", None)
            num_heads = getattr(config, "num_attention_heads", None)
            if hidden_size is not None and num_heads:
                actual = int(hidden_size) // int(num_heads)
        details.append(f"{key}={actual}")
    supported_text = ", ".join(
        f"{item['display_name']}({name}, model_type={item.get('model_type', 'llama')})"
        for name, item in SUPPORTED_LLAMA_PROFILES.items()
    )
    raise ValueError(
        f"{label} is not a supported decoder profile. "
        f"Supported profiles: {supported_text}. "
        f"Observed: model_type={model_type or 'unknown'}, " + ", ".join(details)
    )


def load_tokenized_loader(
    *,
    tokenizer,
    data_path: str,
    max_records: int,
    cutoff_len: int,
    batch_size: int,
    seed: int,
    shuffle_records: bool,
    distributed_num_replicas: int = 1,
    distributed_rank: int = 0,
    prompt_mode: str = "legacy_sft",
) -> Tuple[DataLoader, int]:
    records = prepare_records(
        data_path,
        max_records=int(max_records),
        seed=int(seed),
        shuffle_records=bool(shuffle_records),
    )
    normalized_prompt_mode = str(prompt_mode).strip().lower()
    if normalized_prompt_mode == "decision_aligned":
        tokenized = tokenize_decision_records(tokenizer, records, cutoff_len=int(cutoff_len))
    elif normalized_prompt_mode == "math_reasoning":
        tokenized = []
        system_prompt = (
            "You are a careful mathematical reasoning assistant. Solve the problem step by step, "
            "show the reasoning needed to verify the result, and finish with a clearly marked final answer. "
            "For arithmetic answers use 'The answer is <number>.'; for symbolic answers use \\\\boxed{...}."
        )
        eos_token = tokenizer.eos_token or ""
        for sample in records:
            user_text = str(sample.get("instruction", "")).strip()
            input_text = str(sample.get("input", "")).strip()
            if input_text:
                user_text = f"{user_text}\n\n{input_text}".strip()
            output_text = str(sample.get("output", "")).strip()
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ]
            if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
                prompt_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                full_text = tokenizer.apply_chat_template(
                    messages + [{"role": "assistant", "content": output_text}],
                    tokenize=False,
                    add_generation_prompt=False,
                )
            else:
                prompt_text = (
                    f"{system_prompt}\n\n### Problem:\n{user_text}\n\n### Solution:\n"
                )
                full_text = f"{prompt_text}{output_text}"
            if eos_token and not full_text.endswith(eos_token):
                full_text = f"{full_text}{eos_token}"
            encoded = tokenizer(
                full_text,
                truncation=True,
                max_length=int(cutoff_len),
                padding=False,
            )
            prompt_encoded = tokenizer(
                prompt_text,
                truncation=True,
                max_length=int(cutoff_len),
                padding=False,
            )
            input_ids = list(encoded.get("input_ids", []))
            if len(input_ids) < 2:
                continue
            prompt_len = min(len(prompt_encoded.get("input_ids", [])), len(input_ids))
            if prompt_len >= len(input_ids):
                continue
            tokenized.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": list(encoded.get("attention_mask", [])),
                    "prompt_len": int(prompt_len),
                }
            )
    elif normalized_prompt_mode == "legacy_sft":
        tokenized = tokenize_records(tokenizer, records, cutoff_len=int(cutoff_len))
    else:
        raise ValueError(
            f"Unsupported prompt_mode={prompt_mode!r}; use legacy_sft, decision_aligned, or math_reasoning"
        )
    if not tokenized:
        raise RuntimeError("No tokenized samples.")
    sampler = None
    if int(distributed_num_replicas) > 1:
        sampler = DistributedSampler(
            tokenized,
            num_replicas=int(distributed_num_replicas),
            rank=int(distributed_rank),
            shuffle=bool(shuffle_records),
            seed=int(seed),
            drop_last=False,
        )
    collate_fn = (
        (lambda batch: collate_decision_batch(batch, pad_token_id=int(tokenizer.pad_token_id)))
        if normalized_prompt_mode == "decision_aligned"
        else (lambda batch: collate_tokenized_batch(batch, pad_token_id=int(tokenizer.pad_token_id)))
    )
    loader = DataLoader(
        tokenized,
        batch_size=int(batch_size),
        shuffle=False if sampler is not None else bool(shuffle_records),
        sampler=sampler,
        drop_last=False,
        collate_fn=collate_fn,
    )
    return loader, len(tokenized)


def select_token_positions(
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    eos_token_id: Optional[int],
    token_rule: str,
    window_size: int,
    window_sample_mode: str,
    window_random_pick_min: int,
    window_random_pick_max: int,
    prompt_lens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = input_ids.device
    normalized = str(token_rule).strip().lower()
    if normalized in {"response_pred", "response_all", "all_response", "all_pred"} and prompt_lens is not None:
        anchor_indices = last_pred_indices(attention_mask, input_ids, eos_token_id, device)
        prompt_lens_device = prompt_lens.to(device=device, dtype=torch.long).view(-1)
        batch_parts: List[torch.Tensor] = []
        token_parts: List[torch.Tensor] = []
        max_window = int(window_size)
        for batch_idx in range(int(input_ids.size(0))):
            valid_length = int(attention_mask[batch_idx].to(dtype=torch.long).sum().item())
            if valid_length <= 0:
                continue
            anchor = max(0, min(int(anchor_indices[batch_idx].item()), valid_length - 1))
            start = max(0, min(int(prompt_lens_device[batch_idx].item()) - 1, anchor))
            tokens = torch.arange(start, anchor + 1, device=device, dtype=torch.long)
            if max_window > 0 and int(tokens.numel()) > max_window:
                tokens = tokens[-max_window:]
            if tokens.numel() <= 0:
                continue
            keep = attention_mask[batch_idx].to(dtype=torch.bool).gather(0, tokens)
            tokens = tokens[keep]
            if tokens.numel() <= 0:
                continue
            batch_parts.append(torch.full((int(tokens.numel()),), int(batch_idx), device=device, dtype=torch.long))
            token_parts.append(tokens)
        if batch_parts:
            return torch.cat(batch_parts, dim=0), torch.cat(token_parts, dim=0)
    if normalized == "last_pred":
        centers = last_pred_indices(attention_mask, input_ids, eos_token_id, device)
    else:
        centers = last_content_indices(attention_mask, input_ids, eos_token_id, device)
    return build_last_k_eval_indices(
        centers,
        window_size=int(window_size),
        attention_mask=attention_mask,
        sample_mode=str(window_sample_mode),
        random_pick_min=int(window_random_pick_min),
        random_pick_max=int(window_random_pick_max),
    )


class Reservoir:
    def __init__(self, max_size: int, dim: int, seed: int):
        self.max_size = int(max_size)
        self.dim = int(dim)
        self.rng = random.Random(int(seed))
        self.buffer = torch.empty((self.max_size, self.dim), dtype=torch.float32) if self.max_size > 0 else None
        self.fill = 0
        self.seen = 0

    def add(self, values: torch.Tensor) -> None:
        if self.max_size <= 0 or self.buffer is None:
            return
        values_cpu = values.detach().to(dtype=torch.float32).cpu()
        for idx in range(int(values_cpu.size(0))):
            self.seen += 1
            if self.fill < self.max_size:
                self.buffer[self.fill].copy_(values_cpu[idx])
                self.fill += 1
            else:
                replacement = self.rng.randrange(self.seen)
                if replacement < self.max_size:
                    self.buffer[replacement].copy_(values_cpu[idx])

    def tensor(self) -> torch.Tensor:
        if self.buffer is None or self.fill <= 0:
            return torch.empty((0, self.dim), dtype=torch.float32)
        # Atlas only reads these samples. Returning a view avoids cloning all
        # three 42-layer reservoirs and temporarily doubling host RAM usage.
        return self.buffer[: self.fill]


def fit_kmeans(
    x: torch.Tensor,
    *,
    num_clusters: int,
    iters: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return fit_kmeans_with_mode(
        x,
        num_clusters=num_clusters,
        iters=iters,
        seed=seed,
        mode="full",
        batch_size=4096,
        warmup_size=50000,
        warmup_iters=max(1, int(iters)),
        refine_iters=0,
        assign_chunk_size=8192,
        device="cpu",
    )


def _iter_tensor_rows(x: torch.Tensor, chunk_size: int):
    n = int(x.size(0))
    if n <= 0:
        return
    if int(chunk_size) <= 0 or int(chunk_size) >= n:
        yield x
        return
    step = int(chunk_size)
    for start in range(0, n, step):
        end = min(n, start + step)
        yield x[start:end]


def _squared_l2_distance(x: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    x2 = (x * x).sum(dim=1, keepdim=True)
    c2 = (centers * centers).sum(dim=1).view(1, -1)
    dist2 = x2 + c2 - 2.0 * (x @ centers.t())
    return torch.clamp(dist2, min=0.0)


def _argmin_codes_chunked(x: torch.Tensor, centers: torch.Tensor, chunk_size: int) -> torch.Tensor:
    if x.dim() != 2 or centers.dim() != 2:
        raise ValueError("x and centers must be 2-D.")
    if int(x.size(1)) != int(centers.size(1)):
        raise ValueError("x and centers dimension mismatch.")
    n = int(x.size(0))
    if n <= 0:
        return torch.empty((0,), dtype=torch.long, device=x.device)
    if int(chunk_size) <= 0 or n <= int(chunk_size):
        return _squared_l2_distance(x, centers).argmin(dim=1)
    out = torch.empty((n,), dtype=torch.long, device=x.device)
    step = int(chunk_size)
    for start in range(0, n, step):
        end = min(n, start + step)
        out[start:end] = _squared_l2_distance(x[start:end], centers).argmin(dim=1)
    return out


def nearest_code(
    z: torch.Tensor,
    centers: torch.Tensor,
    *,
    chunk_size: int = 8192,
    device: str = "auto",
) -> torch.Tensor:
    if z.dim() != 2 or centers.dim() != 2:
        raise ValueError("z and centers must be 2-D.")
    if int(z.size(1)) != int(centers.size(1)):
        raise ValueError("z and centers dimension mismatch.")
    n = int(z.size(0))
    if n <= 0:
        return torch.empty((0,), dtype=torch.long)
    work_device = resolve_device(device)
    centers_work = centers.detach().to(device=work_device, dtype=torch.float32)
    assign_chunks: List[torch.Tensor] = []
    step = max(1, int(chunk_size))
    for start in range(0, n, step):
        end = min(n, start + step)
        x_chunk = z[start:end].to(device=work_device, dtype=torch.float32, non_blocking=True)
        assign = _argmin_codes_chunked(x_chunk, centers_work, chunk_size=0)
        assign_chunks.append(assign.cpu())
    return torch.cat(assign_chunks, dim=0)


def _kmeanspp_init(
    x: torch.Tensor,
    *,
    k: int,
    seed: int,
    device: torch.device,
    assign_chunk_size: int,
) -> torch.Tensor:
    n = int(x.size(0))
    if n <= 0:
        raise ValueError("Empty data for kmeans++ init.")
    k = max(1, min(int(k), n))
    rng_device = device if device.type != "cpu" else torch.device("cpu")
    generator = torch.Generator(device=rng_device)
    generator.manual_seed(int(seed))
    x_work = x.to(device=device, dtype=torch.float32, non_blocking=True)
    first_idx = int(torch.randint(0, n, (1,), generator=generator, device=rng_device).item())
    centers = torch.empty((k, int(x.size(1))), dtype=torch.float32, device=device)
    centers[0] = x_work[first_idx]
    min_dist2 = _squared_l2_distance(x_work, centers[0:1]).squeeze(1)
    for idx in range(1, k):
        prob = min_dist2 / min_dist2.sum().clamp(min=1e-12)
        next_idx = int(torch.multinomial(prob, num_samples=1, replacement=False, generator=generator).item())
        centers[idx] = x_work[next_idx]
        next_dist2 = _squared_l2_distance(x_work, centers[idx : idx + 1]).squeeze(1)
        min_dist2 = torch.minimum(min_dist2, next_dist2)
    return centers


def _mini_batch_refine_centers(
    *,
    x: torch.Tensor,
    centers: torch.Tensor,
    counts: torch.Tensor,
    iters: int,
    batch_size: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if int(iters) <= 0 or int(x.size(0)) <= 0:
        return centers, counts
    n = int(x.size(0))
    batch_size = max(1, int(batch_size))
    generator = torch.Generator(device=torch.device("cpu"))
    generator.manual_seed(int(seed))
    for it in range(max(1, int(iters))):
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            batch = x.index_select(0, idx).to(device=centers.device, dtype=torch.float32, non_blocking=True)
            assign = _argmin_codes_chunked(batch, centers, chunk_size=0)
            present = assign.unique(sorted=False)
            for code_id_t in present:
                code_id = int(code_id_t.item())
                mask = assign == code_id
                points = batch[mask]
                m = int(points.size(0))
                if m <= 0:
                    continue
                batch_mean = points.mean(dim=0)
                old_count = float(counts[code_id].item())
                if old_count <= 0.0:
                    centers[code_id] = batch_mean
                    counts[code_id] = float(m)
                    continue
                new_count = old_count + float(m)
                eta = float(m) / new_count
                centers[code_id] = centers[code_id] + eta * (batch_mean - centers[code_id])
                counts[code_id] = new_count
    return centers, counts


def fit_kmeans_with_mode(
    x: torch.Tensor,
    *,
    num_clusters: int,
    iters: int,
    seed: int,
    mode: str,
    batch_size: int,
    warmup_size: int,
    warmup_iters: int,
    refine_iters: int,
    assign_chunk_size: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if x.dim() != 2:
        raise ValueError("x must be 2-D for kmeans.")
    x_cpu = x.detach().to(dtype=torch.float32, device="cpu").contiguous()
    n, d = int(x_cpu.size(0)), int(x_cpu.size(1))
    if n <= 0:
        raise ValueError("Empty data for kmeans.")
    k = max(1, min(int(num_clusters), n))
    normalized_mode = str(mode).strip().lower()
    if normalized_mode == "full":
        work_device = resolve_device(device)
        x_work = x_cpu.to(device=work_device, dtype=torch.float32, non_blocking=True)
        generator = torch.Generator(device=work_device if work_device.type != "cpu" else torch.device("cpu"))
        generator.manual_seed(int(seed))
        init_idx = torch.randperm(n, generator=generator, device=work_device)[:k]
        centers = x_work.index_select(0, init_idx).clone()
        assign = torch.zeros((n,), dtype=torch.long, device=work_device)
        for _ in range(max(1, int(iters))):
            assign = _argmin_codes_chunked(x_work, centers, chunk_size=max(1, int(assign_chunk_size)))
            for cluster in range(k):
                mask = assign == cluster
                if bool(mask.any().item()):
                    centers[cluster] = x_work[mask].mean(dim=0)
        return centers.cpu(), assign.cpu()

    if normalized_mode != "minibatch":
        raise ValueError(f"Unsupported kmeans mode: {mode}")

    work_device = resolve_device(device)
    warmup_take = int(warmup_size)
    if warmup_take <= 0:
        warmup_take = n
    warmup_take = max(k, min(n, warmup_take))
    generator = torch.Generator(device=torch.device("cpu"))
    generator.manual_seed(int(seed) + 97)
    if warmup_take < n:
        warmup_idx = torch.randperm(n, generator=generator)[:warmup_take]
        warmup_x = x_cpu.index_select(0, warmup_idx).contiguous()
    else:
        warmup_x = x_cpu

    centers = _kmeanspp_init(
        warmup_x,
        k=k,
        seed=int(seed),
        device=work_device,
        assign_chunk_size=int(assign_chunk_size),
    )
    counts = torch.zeros((k,), dtype=torch.float32, device=work_device)
    centers, counts = _mini_batch_refine_centers(
        x=warmup_x,
        centers=centers,
        counts=counts,
        iters=max(1, int(warmup_iters)),
        batch_size=int(batch_size),
        seed=int(seed) + 101,
    )
    centers, counts = _mini_batch_refine_centers(
        x=x_cpu,
        centers=centers,
        counts=counts,
        iters=max(0, int(refine_iters)),
        batch_size=int(batch_size),
        seed=int(seed) + 1009,
    )
    centers_cpu = centers.detach().cpu().to(dtype=torch.float32)
    assign = nearest_code(
        x_cpu,
        centers_cpu,
        chunk_size=int(assign_chunk_size),
        device=str(device),
    )
    return centers_cpu, assign


def fit_streaming_sketch_pca(
    samples: Sequence[torch.Tensor],
    *,
    rank: int,
    oversample: int,
    power_iter: int,
    chunk_size: int,
    seed: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    non_empty = [x for x in samples if x is not None and int(x.numel()) > 0]
    if not non_empty:
        raise RuntimeError("No samples for PCA.")
    hidden_size = int(non_empty[0].size(1))
    for x in non_empty:
        if int(x.size(1)) != hidden_size:
            raise RuntimeError("Inconsistent hidden size in PCA samples.")
    count = 0
    sum_vec = torch.zeros((hidden_size,), dtype=torch.float64)
    for sample in non_empty:
        for chunk in _iter_tensor_rows(sample, int(chunk_size)):
            if int(chunk.size(0)) <= 0:
                continue
            x = chunk.to(dtype=torch.float64, device="cpu")
            sum_vec += x.sum(dim=0)
            count += int(x.size(0))
    if count < 2:
        raise RuntimeError("Too few samples for PCA.")
    mean = (sum_vec / float(count)).to(dtype=torch.float32)

    target_rank = max(1, min(int(rank), hidden_size))
    q = max(target_rank, target_rank + max(0, int(oversample)))
    q = min(hidden_size, q)
    work_device = resolve_device(device)
    mean_work = mean.to(device=work_device, dtype=torch.float32)
    rng_device = work_device if work_device.type != "cpu" else torch.device("cpu")
    generator = torch.Generator(device=rng_device)
    generator.manual_seed(int(seed))
    omega = torch.randn((hidden_size, q), generator=generator, device=work_device, dtype=torch.float32)
    denom = float(max(1, int(count - 1)))

    def cov_mul(mat: torch.Tensor) -> torch.Tensor:
        out = torch.zeros((hidden_size, int(mat.size(1))), dtype=torch.float32, device=work_device)
        for sample in non_empty:
            for chunk in _iter_tensor_rows(sample, int(chunk_size)):
                if int(chunk.size(0)) <= 0:
                    continue
                x = chunk.to(device=work_device, dtype=torch.float32, non_blocking=True)
                xc = x - mean_work.view(1, -1)
                out += xc.t() @ (xc @ mat)
        return out / denom

    y = cov_mul(omega)
    for _ in range(max(0, int(power_iter))):
        q_mat, _ = torch.linalg.qr(y, mode="reduced")
        y = cov_mul(q_mat)
    q_mat, _ = torch.linalg.qr(y, mode="reduced")

    b_small = torch.zeros((int(q_mat.size(1)), int(q_mat.size(1))), dtype=torch.float32, device=work_device)
    for sample in non_empty:
        for chunk in _iter_tensor_rows(sample, int(chunk_size)):
            if int(chunk.size(0)) <= 0:
                continue
            x = chunk.to(device=work_device, dtype=torch.float32, non_blocking=True)
            y_proj = (x - mean_work.view(1, -1)) @ q_mat
            b_small += y_proj.t() @ y_proj
    b_small = b_small / denom
    evals, evecs = torch.linalg.eigh(b_small)
    order = torch.argsort(evals, descending=True)
    top = order[:target_rank]
    basis = (q_mat @ evecs[:, top]).contiguous().cpu().to(dtype=torch.float32)
    stats = {
        "count": int(count),
        "q": int(q),
        "device": str(work_device),
    }
    return basis, mean.cpu(), stats


def _random_orthogonal_basis(
    *,
    hidden_size: int,
    rank: int,
    seed: int,
) -> torch.Tensor:
    hidden_size = int(hidden_size)
    rank = int(rank)
    if hidden_size <= 0 or rank <= 0 or rank > hidden_size:
        raise ValueError(f"Invalid random basis shape hidden_size={hidden_size} rank={rank}.")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    matrix = torch.randn((hidden_size, rank), generator=generator, dtype=torch.float32)
    q_mat, _ = torch.linalg.qr(matrix, mode="reduced")
    basis = q_mat[:, :rank].contiguous().to(dtype=torch.float32)
    eye = torch.eye(rank, dtype=torch.float32)
    orthogonality_error = float(torch.linalg.norm(basis.transpose(0, 1) @ basis - eye).item())
    if not math.isfinite(orthogonality_error) or orthogonality_error > 1e-3:
        raise RuntimeError(f"Random basis orthogonality check failed: {orthogonality_error:.6g}")
    return basis


def _fit_projection_basis(
    *,
    samples: Sequence[torch.Tensor],
    source: str,
    rank: int,
    hidden_size: int,
    pca_mode: str,
    oversample: int,
    power_iter: int,
    chunk_size: int,
    seed: int,
    random_basis_seed: int,
    device: str,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    source_norm = str(source).strip().lower()
    if source_norm in {"velocity", "velocity_pca", "teacher_velocity_pca", "delta", "delta_pca"}:
        source_norm = "velocity_pca"
    elif source_norm in {"hidden", "hidden_pca", "teacher_hidden_pca", "hidden_state_pca", "pre_ffn_pca"}:
        source_norm = "hidden_pca"
    elif source_norm in {"random", "random_orthogonal", "random_projection"}:
        source_norm = "random_orthogonal"
    else:
        raise ValueError(
            f"Unsupported projection_basis_source={source!r}. "
            "Use velocity_pca, hidden_pca, or random_orthogonal."
        )

    non_empty = [x for x in samples if torch.is_tensor(x) and int(x.numel()) > 0]
    total_samples = sum(int(x.size(0)) for x in non_empty)
    if source_norm == "random_orthogonal":
        basis = _random_orthogonal_basis(
            hidden_size=int(hidden_size),
            rank=int(rank),
            seed=int(random_basis_seed),
        )
        return basis, {
            "mode": "random_orthogonal",
            "source": source_norm,
            "count": int(total_samples),
            "q": int(rank),
            "device": "cpu",
            "random_basis_seed": int(random_basis_seed),
        }

    if total_samples < 8:
        raise RuntimeError(f"Too few samples for {source_norm} atlas basis.")
    pca_mode_norm = str(pca_mode).strip().lower()
    if pca_mode_norm == "lowrank":
        pooled = torch.cat(non_empty, dim=0)
        pooled_centered = pooled - pooled.mean(dim=0, keepdim=True)
        q_rank = min(
            int(pooled_centered.size(0)),
            int(hidden_size),
            int(rank + max(8, int(oversample))),
        )
        _, _, basis_v = torch.pca_lowrank(
            pooled_centered,
            q=int(q_rank),
            center=False,
            niter=max(1, int(power_iter)),
        )
        basis = basis_v[:, : int(rank)].contiguous().cpu().to(dtype=torch.float32)
        return basis, {
            "mode": "lowrank",
            "source": source_norm,
            "count": int(total_samples),
            "q": int(q_rank),
            "device": "cpu",
        }
    if pca_mode_norm == "stream_sketch":
        basis, _, sketch_stats = fit_streaming_sketch_pca(
            samples,
            rank=int(rank),
            oversample=max(8, int(oversample)),
            power_iter=max(0, int(power_iter)),
            chunk_size=int(chunk_size),
            seed=int(seed),
            device=str(device),
        )
        return basis, {
            "mode": "stream_sketch",
            "source": source_norm,
            "count": int(sketch_stats["count"]),
            "q": int(sketch_stats["q"]),
            "device": str(sketch_stats["device"]),
        }
    raise ValueError(f"Unsupported pca_mode: {pca_mode_norm}")


def build_alpha_layer_topk_mixture(
    *,
    tau_diag: torch.Tensor,
    layer_hist: torch.Tensor,
    tau_topk: int,
    tau_eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if tau_diag.dim() != 2:
        raise ValueError("tau_diag must be 2-D.")
    if layer_hist.dim() != 2:
        raise ValueError("layer_hist must be 2-D.")
    code_count = int(tau_diag.size(0))
    rank = int(tau_diag.size(1))
    if int(layer_hist.size(1)) != code_count:
        raise ValueError("layer_hist code dimension mismatch.")
    # tau_diag stores per-dimension variance; whitening scale is inverse std.
    eps = max(1e-12, float(tau_eps))
    tau_var = torch.clamp(tau_diag.to(dtype=torch.float32), min=eps)
    alpha_code = 1.0 / torch.sqrt(tau_var)
    tau_global = tau_var.mean(dim=0)
    alpha_global = 1.0 / torch.sqrt(torch.clamp(tau_global, min=eps))
    topk = max(1, min(int(tau_topk), code_count))
    alpha_layer = torch.empty((int(layer_hist.size(0)), rank), dtype=torch.float32)
    for layer_id in range(int(layer_hist.size(0))):
        hist = layer_hist[layer_id].to(dtype=torch.float32)
        hist_sum = float(hist.sum().item())
        if not math.isfinite(hist_sum) or hist_sum <= 0.0:
            alpha_layer[layer_id] = alpha_global
            continue
        values, indices = torch.topk(hist, k=topk, largest=True, sorted=False)
        denom = float(values.sum().item())
        if not math.isfinite(denom) or denom <= 0.0:
            alpha_layer[layer_id] = alpha_global
            continue
        weights = values / denom
        mixed = (alpha_code.index_select(0, indices) * weights.view(-1, 1)).sum(dim=0)
        alpha_layer[layer_id] = mixed
    return alpha_layer, alpha_code, alpha_global


def compute_layer_hist_entropy(layer_hist: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    if layer_hist.dim() != 2:
        raise ValueError("layer_hist must be 2-D.")
    probs = torch.clamp(layer_hist.to(dtype=torch.float32), min=0.0)
    norm = probs.sum(dim=1, keepdim=True).clamp(min=max(1e-12, float(eps)))
    probs = probs / norm
    safe = torch.clamp(probs, min=max(1e-12, float(eps)))
    entropy = -(probs * safe.log()).sum(dim=1)
    return entropy


def _cosine_similarity_matrix(x: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    if x.dim() != 2:
        raise ValueError("expected [N, D] tensor")
    x_f = x.to(dtype=torch.float32)
    denom = x_f.norm(dim=1, keepdim=True).clamp(min=float(eps))
    normalized = x_f / denom
    return normalized @ normalized.transpose(0, 1)


def _build_regime_labels(layer_count: int) -> List[str]:
    if layer_count <= 0:
        return []
    base = int(layer_count) // 3
    rem = int(layer_count) % 3
    sizes = [base + (1 if idx < rem else 0) for idx in range(3)]
    names = ["llama_early", "llama_mid", "llama_late"]
    labels: List[str] = []
    for name, size in zip(names, sizes):
        labels.extend([name] * int(size))
    return labels[: int(layer_count)]


def _fit_regime_basis_map(
    *,
    basis_samples: Sequence[torch.Tensor],
    regime_labels: Sequence[str],
    rank: int,
    hidden_size: int,
    basis_source: str,
    oversample: int,
    power_iter: int,
    chunk_size: int,
    seed: int,
    random_basis_seed: int,
    pca_mode: str,
    device: str,
    fallback_basis: torch.Tensor,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Dict[str, Any]]]:
    regime_names = list(dict.fromkeys([str(item).strip() or "llama_late" for item in regime_labels])) or ["llama_late"]
    fallback = fallback_basis.detach().to(dtype=torch.float32, device="cpu").contiguous()
    basis_map: Dict[str, torch.Tensor] = {}
    stats_map: Dict[str, Dict[str, Any]] = {}
    for offset, regime in enumerate(regime_names):
        regime_samples = [
            basis_samples[layer_id]
            for layer_id in range(min(len(basis_samples), len(regime_labels)))
            if str(regime_labels[layer_id]) == regime and torch.is_tensor(basis_samples[layer_id]) and int(basis_samples[layer_id].numel()) > 0
        ]
        if regime_samples:
            try:
                regime_basis, regime_stats = _fit_projection_basis(
                    samples=regime_samples,
                    source=str(basis_source),
                    rank=int(rank),
                    hidden_size=int(hidden_size),
                    pca_mode=str(pca_mode),
                    oversample=int(oversample),
                    power_iter=int(power_iter),
                    chunk_size=int(chunk_size),
                    seed=int(seed) + 1000 + int(offset),
                    random_basis_seed=int(random_basis_seed) + 1000 + int(offset),
                    device=str(device),
                )
                basis_map[regime] = regime_basis.detach().to(dtype=torch.float32, device="cpu").contiguous()
                stats_map[regime] = {
                    "sample_count": int(regime_stats["count"]),
                    "q": int(regime_stats["q"]),
                    "device": str(regime_stats["device"]),
                    "mode": str(regime_stats["mode"]),
                    "source": str(regime_stats["source"]),
                    "random_basis_seed": int(regime_stats.get("random_basis_seed", int(random_basis_seed) + 1000 + int(offset))),
                    "fallback": False,
                }
                continue
            except Exception as exc:
                stats_map[regime] = {
                    "sample_count": int(sum(int(x.size(0)) for x in regime_samples)),
                    "q": int(fallback.size(1)),
                    "device": "cpu",
                    "fallback": True,
                    "error": str(exc),
                }
        basis_map[regime] = fallback
        stats_map.setdefault(
            regime,
            {
                "sample_count": 0,
                "q": int(fallback.size(1)),
                "device": "cpu",
                "fallback": True,
            },
        )
    return basis_map, stats_map


def _compute_layer_reliability(
    *,
    layer_hist_entropy: torch.Tensor,
    layer_error_norm: torch.Tensor,
) -> torch.Tensor:
    entropy = layer_hist_entropy.to(dtype=torch.float32)
    error_norm = layer_error_norm.to(dtype=torch.float32)
    entropy_scale = entropy.max().clamp(min=1e-6)
    error_scale = error_norm.max().clamp(min=1e-6)
    entropy_conf = 1.0 - torch.clamp(entropy / entropy_scale, min=0.0, max=1.0)
    error_conf = 1.0 - torch.clamp(error_norm / error_scale, min=0.0, max=1.0)
    reliability = 0.5 * entropy_conf + 0.5 * error_conf
    return torch.clamp(reliability, min=0.1, max=1.0)


def _connected_components(adjacency: torch.Tensor, node_ids: Sequence[int]) -> List[List[int]]:
    components: List[List[int]] = []
    visited: set[int] = set()
    nodes = [int(x) for x in node_ids]
    for node in nodes:
        if node in visited:
            continue
        stack = [int(node)]
        visited.add(int(node))
        component: List[int] = []
        while stack:
            current = int(stack.pop())
            component.append(current)
            neighbors = torch.nonzero(adjacency[current], as_tuple=False).view(-1).tolist()
            for neighbor in neighbors:
                n = int(neighbor)
                if n in visited or n not in nodes:
                    continue
                visited.add(n)
                stack.append(n)
        components.append(sorted(component))
    return components


def _infer_prototype_count(layer_count: int) -> int:
    if int(layer_count) <= 0:
        return 1
    # Keep the historical 16-layer -> 9 prototype recommendation while scaling with depth.
    return max(1, min(int(layer_count), int(math.ceil(float(layer_count) * (9.0 / 16.0)))))


def _build_sharing_policy(
    *,
    upstream_mean: torch.Tensor,
    attention_delta_mean: torch.Tensor,
    h2_mean: torch.Tensor,
    h3_mean: torch.Tensor,
    regime_labels: Sequence[str],
    layer_reliability: torch.Tensor,
    sharing_policy_mode: str = "upstream_only",
    upstream_threshold: float = 0.95,
) -> Dict[str, Any]:
    mode = str(sharing_policy_mode).strip().lower()
    if mode != "upstream_only":
        raise ValueError(f"Unsupported sharing_policy_mode: {sharing_policy_mode}")
    layer_count = int(upstream_mean.size(0))
    upstream_sim = _cosine_similarity_matrix(upstream_mean)
    attention_sim = _cosine_similarity_matrix(attention_delta_mean)
    h2_sim = _cosine_similarity_matrix(h2_mean)
    h3_sim = _cosine_similarity_matrix(h3_mean)
    adjacency = torch.zeros((layer_count, layer_count), dtype=torch.bool)
    for i in range(layer_count):
        adjacency[i, i] = True
        for j in range(i + 1, layer_count):
            if str(regime_labels[j]) != str(regime_labels[i]):
                continue
            if float(upstream_sim[i, j].item()) >= float(upstream_threshold):
                adjacency[i, j] = True
                adjacency[j, i] = True
    raw_groups = _connected_components(adjacency, list(range(layer_count)))
    groups: List[Dict[str, Any]] = []
    layer_entries: List[Dict[str, Any]] = []
    assigned_group: Dict[int, int] = {}
    group_id = 0
    for members in raw_groups:
        if len(members) <= 1:
            continue
        pair_count = max(1, sum(1 for idx_i, _ in enumerate(members) for __ in members[idx_i + 1 :]))
        mean_attn = sum(float(attention_sim[layer_i, layer_j].item()) for idx_i, layer_i in enumerate(members) for layer_j in members[idx_i + 1 :]) / float(pair_count)
        mean_horizon_h2 = sum(float(h2_sim[layer_i, layer_j].item()) for idx_i, layer_i in enumerate(members) for layer_j in members[idx_i + 1 :]) / float(pair_count)
        mean_horizon_h3 = sum(float(h3_sim[layer_i, layer_j].item()) for idx_i, layer_i in enumerate(members) for layer_j in members[idx_i + 1 :]) / float(pair_count)
        mean_horizon = 0.5 * mean_horizon_h2 + 0.5 * mean_horizon_h3
        reliability_mean = float(layer_reliability[members].mean().item())
        groups.append(
            {
                "group_id": int(group_id),
                "layers": [int(x) for x in members],
                "regime": str(regime_labels[members[0]]),
                "private_core": False,
                "mean_upstream_similarity": float(
                    sum(float(upstream_sim[i, j].item()) for idx_i, i in enumerate(members) for j in members[idx_i + 1 :])
                    / float(pair_count)
                ),
                "mean_attention_similarity": float(mean_attn),
                "mean_horizon_h2_similarity": float(mean_horizon_h2),
                "mean_horizon_h3_similarity": float(mean_horizon_h3),
                "mean_horizon_similarity": float(mean_horizon),
                "reliability_mean": reliability_mean,
            }
        )
        for layer_id in members:
            assigned_group[int(layer_id)] = int(group_id)
        group_id += 1
    for layer_id in range(layer_count):
        private_core = int(layer_id) not in assigned_group
        layer_entries.append(
            {
                "layer_id": int(layer_id),
                "regime": str(regime_labels[layer_id]),
                "group_id": int(assigned_group[layer_id]) if int(layer_id) in assigned_group else -1,
                "private_core": bool(private_core),
                "reliability": float(layer_reliability[layer_id].item()),
            }
        )
    return {
        "source_view": "upstream_ffn_input",
        "sharing_policy_mode": str(mode),
        "sharing_group_count": int(len(groups)),
        "upstream_similarity_threshold": float(upstream_threshold),
        "groups": groups,
        "layers": layer_entries,
        "pairwise_similarity": {
            "upstream": upstream_sim.tolist(),
            "attention_delta": attention_sim.tolist(),
            "short_horizon_h2": h2_sim.tolist(),
            "short_horizon_h3": h3_sim.tolist(),
        },
    }


def _layer_groups_from_sharing_policy(
    sharing_policy: Optional[Dict[str, Any]],
    *,
    layer_count: int,
) -> Tuple[List[int], List[str]]:
    if not isinstance(sharing_policy, dict):
        return [int(x) for x in range(layer_count)], _build_regime_labels(layer_count)
    layers_payload = sharing_policy.get("layers", [])
    if not isinstance(layers_payload, list) or len(layers_payload) != layer_count:
        return [int(x) for x in range(layer_count)], _build_regime_labels(layer_count)
    layer_to_group = list(range(layer_count))
    regime_labels = _build_regime_labels(layer_count)
    for item in layers_payload:
        if not isinstance(item, dict):
            continue
        layer_id = int(item.get("layer_id", -1))
        if layer_id < 0 or layer_id >= layer_count:
            continue
        group_id = int(item.get("group_id", -1))
        if group_id >= 0:
            layer_to_group[layer_id] = int(group_id)
        else:
            layer_to_group[layer_id] = int(layer_count + layer_id)
        regime_labels[layer_id] = str(item.get("regime", regime_labels[layer_id]))
    return layer_to_group, regime_labels


def _policy_medoid_seed_mapping(
    sharing_policy: Optional[Dict[str, Any]],
    *,
    layer_to_proto: Sequence[int],
) -> Dict[int, int]:
    """Resolve one explicitly measured medoid for every shared or private core."""
    if not isinstance(sharing_policy, dict):
        raise ValueError("policy_medoid requires a loaded sharing policy")
    proto_ids = {int(item) for item in layer_to_proto}
    mapping: Dict[int, int] = {}
    groups = sharing_policy.get("groups", [])
    if not isinstance(groups, list):
        raise ValueError("sharing policy groups must be a list")
    for group in groups:
        if not isinstance(group, dict):
            raise ValueError("sharing policy group must be an object")
        members = sorted({int(item) for item in group.get("layers", [])})
        medoid = int(group.get("medoid_layer", -1))
        if not members or medoid not in members:
            raise ValueError(f"invalid policy medoid={medoid} for members={members}")
        member_proto_ids = {int(layer_to_proto[layer_id]) for layer_id in members}
        if len(member_proto_ids) != 1:
            raise ValueError(f"policy group crosses resolved prototypes: members={members}")
        proto_id = next(iter(member_proto_ids))
        if proto_id in mapping and mapping[proto_id] != medoid:
            raise ValueError(f"duplicate medoids for proto={proto_id}")
        mapping[proto_id] = medoid
    for layer_id, proto_id_raw in enumerate(layer_to_proto):
        proto_id = int(proto_id_raw)
        if proto_id not in mapping:
            mapping[proto_id] = int(layer_id)
    if set(mapping) != proto_ids:
        raise ValueError("policy medoid mapping does not cover every resolved prototype")
    return mapping


class SharedMLPAdapter(nn.Module):
    def __init__(
        self,
        *,
        proto_id: int,
        bank_refs: Sequence[nn.Module],
        hidden_size: int,
        lora_rank: int,
        lora_alpha: Optional[float] = None,
        sharing_parameterization: str = "full_parallel",
        original_mlp: Optional[nn.Module] = None,
        intermediate_size: int = 0,
        use_layer_scalar: bool = True,
    ):
        super().__init__()
        self.proto_id = int(proto_id)
        self._bank_refs: Sequence[nn.Module] = bank_refs
        self.sharing_parameterization = str(sharing_parameterization).strip().lower()
        if self.sharing_parameterization not in {
            "full_parallel",
            "down_only_parallel",
            "internal_weight_delta",
        }:
            raise ValueError(
                f"unsupported sharing_parameterization={sharing_parameterization!r}"
            )
        # Legacy experiments learned one scalar per layer.  The ICLR current
        # method explicitly removes that degree of freedom, so current-paper
        # runs pass use_layer_scalar=False and have no scalar parameter at all.
        self.use_layer_scalar = bool(use_layer_scalar)
        self.scale = (
            nn.Parameter(torch.ones((1,), dtype=torch.float32))
            if self.use_layer_scalar
            else None
        )
        self.internal_rank = (
            int(max(0, lora_rank))
            if self.sharing_parameterization == "internal_weight_delta"
            else 0
        )
        self.lora_rank = (
            int(max(0, lora_rank))
            if self.sharing_parameterization != "internal_weight_delta"
            else 0
        )
        resolved_alpha = float(self.lora_rank if lora_alpha is None else lora_alpha)
        if self.lora_rank > 0 and resolved_alpha <= 0.0:
            raise ValueError(f"lora_alpha must be positive when lora_rank>0, got {resolved_alpha}")
        self.lora_alpha = resolved_alpha if self.lora_rank > 0 else 0.0
        self.lora_scaling = self.lora_alpha / float(self.lora_rank) if self.lora_rank > 0 else 0.0
        if self.lora_rank > 0:
            self.lora_up = nn.Linear(int(hidden_size), self.lora_rank, bias=False)
            self.lora_down = nn.Linear(self.lora_rank, int(hidden_size), bias=False)
            nn.init.zeros_(self.lora_down.weight)
        else:
            self.lora_up = None
            self.lora_down = None
        self.private_gate_proj = None
        self.private_up_proj = None
        self.act_fn = None
        if self.sharing_parameterization == "down_only_parallel":
            if original_mlp is None:
                raise ValueError("down_only_parallel requires original_mlp")
            self.private_gate_proj = copy.deepcopy(original_mlp.gate_proj)
            self.private_up_proj = copy.deepcopy(original_mlp.up_proj)
            self.act_fn = original_mlp.act_fn

        self.delta_gate_a = None
        self.delta_gate_b = None
        self.delta_up_a = None
        self.delta_up_b = None
        self.delta_down_a = None
        self.delta_down_b = None
        if self.internal_rank > 0:
            width = int(intermediate_size)
            if width <= 0:
                raise ValueError("internal_weight_delta requires intermediate_size")
            rank = int(self.internal_rank)
            self.delta_gate_a = nn.Linear(int(hidden_size), rank, bias=False)
            self.delta_gate_b = nn.Linear(rank, width, bias=False)
            self.delta_up_a = nn.Linear(int(hidden_size), rank, bias=False)
            self.delta_up_b = nn.Linear(rank, width, bias=False)
            self.delta_down_a = nn.Linear(width, rank, bias=False)
            self.delta_down_b = nn.Linear(rank, int(hidden_size), bias=False)
            for module in (
                self.delta_gate_b,
                self.delta_up_b,
                self.delta_down_b,
            ):
                nn.init.zeros_(module.weight)

    def forward_base(self, hidden_states: torch.Tensor) -> torch.Tensor:
        proto = self._bank_refs[self.proto_id]
        if self.sharing_parameterization == "down_only_parallel":
            gate = self.private_gate_proj(hidden_states)
            up = self.private_up_proj(hidden_states)
            return proto(self.act_fn(gate) * up)
        if self.sharing_parameterization == "internal_weight_delta":
            gate = proto.gate_proj(hidden_states)
            up = proto.up_proj(hidden_states)
            if self.internal_rank > 0:
                gate = gate + self.delta_gate_b(self.delta_gate_a(hidden_states))
                up = up + self.delta_up_b(self.delta_up_a(hidden_states))
            intermediate = proto.act_fn(gate) * up
            out = proto.down_proj(intermediate)
            if self.internal_rank > 0:
                out = out + self.delta_down_b(self.delta_down_a(intermediate))
            return out
        return proto(hidden_states)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        base_out = self.forward_base(hidden_states)
        out = base_out * self.scale if self.scale is not None else base_out
        if self.lora_up is not None and self.lora_down is not None:
            out = out + self.lora_down(self.lora_up(hidden_states)) * self.lora_scaling
        return out

    def export_state(self) -> Dict[str, torch.Tensor]:
        return {
            key: value.detach().cpu()
            for key, value in self.state_dict().items()
        }

    def load_exported_state(self, state: Dict[str, torch.Tensor]) -> None:
        current = self.state_dict()
        converted = {
            key: value.to(dtype=current[key].dtype, device=current[key].device)
            for key, value in state.items()
            if key in current
        }
        missing, unexpected = self.load_state_dict(converted, strict=False)
        if unexpected:
            raise RuntimeError(f"unexpected adapter-state keys: {unexpected}")


class LayerMixtureVariationalTransport(nn.Module):
    """Training-only variational transport over the layer behaviors sharing a core.

    For a shared prototype k, the conditional teacher target is modeled as a
    mixture whose components correspond to the original layers assigned to k:

        q(z_T | z_S, k) = sum_l pi_l(z_S) N(z_T; z_S + delta_l, Sigma_T^l).

    Sigma_T is fixed from the teacher structure prior.  The input-conditioned
    assignment network and layer transports are learned, but deliberately not
    exported, so this objective adds no deployment parameters or latency.
    """

    def __init__(
        self,
        *,
        layer_to_proto: Sequence[int],
        layer_covariance_diag: torch.Tensor,
        layer_reliability: torch.Tensor,
        gate_hidden: int,
        assignment_temperature: float,
        entropy_tau: float,
        covariance_eps: float,
        covariance_trace_normalize: bool,
        delta_l2: float,
    ) -> None:
        super().__init__()
        if layer_covariance_diag.dim() != 2:
            raise ValueError("layer_covariance_diag must be [layers, rank]")
        layer_count, rank = (int(layer_covariance_diag.size(0)), int(layer_covariance_diag.size(1)))
        if len(layer_to_proto) != layer_count:
            raise ValueError("layer_to_proto and layer covariance layer count mismatch")
        reliability = layer_reliability.detach().to(dtype=torch.float32).view(-1)
        if int(reliability.numel()) != layer_count:
            raise ValueError("layer reliability count mismatch")

        covariance = layer_covariance_diag.detach().to(dtype=torch.float32).clamp(min=float(covariance_eps))
        if bool(covariance_trace_normalize):
            covariance = covariance / covariance.mean(dim=1, keepdim=True).clamp(min=float(covariance_eps))
        self.register_buffer("layer_covariance_diag", covariance.contiguous())
        self.register_buffer("layer_reliability", reliability.clamp(min=1e-6).contiguous())
        self.rank = rank
        self.assignment_temperature = max(1e-4, float(assignment_temperature))
        self.entropy_tau = max(0.0, float(entropy_tau))
        self.covariance_eps = max(1e-12, float(covariance_eps))
        self.covariance_trace_normalize = bool(covariance_trace_normalize)
        self.delta_l2 = max(0.0, float(delta_l2))

        grouped: Dict[int, List[int]] = {}
        for layer_id, proto_id in enumerate(layer_to_proto):
            grouped.setdefault(int(proto_id), []).append(int(layer_id))
        self.shared_groups: Dict[str, Tuple[int, ...]] = {
            f"proto_{proto_id}": tuple(layers)
            for proto_id, layers in sorted(grouped.items())
            if len(layers) > 1
        }
        if not self.shared_groups:
            raise ValueError("layer-mixture transport requires at least one genuinely shared core")

        hidden = max(1, int(gate_hidden))
        self.assignment_gates = nn.ModuleDict()
        self.layer_transport_delta = nn.ParameterDict()
        for key, layers in self.shared_groups.items():
            gate = nn.Sequential(
                nn.Linear(rank, hidden, bias=True),
                nn.SiLU(),
                nn.Linear(hidden, len(layers), bias=True),
            )
            # Begin at the maximum-entropy uniform assignment.  Component
            # covariances and transport gradients then break symmetry.
            nn.init.zeros_(gate[-1].weight)
            nn.init.zeros_(gate[-1].bias)
            self.assignment_gates[key] = gate
            self.layer_transport_delta[key] = nn.Parameter(torch.zeros((len(layers), rank), dtype=torch.float32))

    def config_dict(self) -> Dict[str, Any]:
        return {
            "family": "layer_mixture_variational_transport",
            "shared_groups": {key: [int(x) for x in layers] for key, layers in self.shared_groups.items()},
            "projection_rank": int(self.rank),
            "assignment_temperature": float(self.assignment_temperature),
            "entropy_tau": float(self.entropy_tau),
            "covariance_eps": float(self.covariance_eps),
            "covariance_trace_normalize": bool(self.covariance_trace_normalize),
            "delta_l2": float(self.delta_l2),
            "dimension_normalized_gaussian_nll": True,
            "deployment_parameters": 0,
        }

    def forward(
        self,
        z_student_by_layer: Dict[int, torch.Tensor],
        z_teacher_by_layer: Dict[int, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        weighted_nll = self.layer_covariance_diag.new_zeros(())
        weighted_gate_entropy = self.layer_covariance_diag.new_zeros(())
        weighted_posterior_entropy = self.layer_covariance_diag.new_zeros(())
        weighted_max_probability = self.layer_covariance_diag.new_zeros(())
        weight_sum = self.layer_covariance_diag.new_zeros(())
        target_count = 0

        for key, component_layers_tuple in self.shared_groups.items():
            component_layers = torch.tensor(
                component_layers_tuple,
                dtype=torch.long,
                device=self.layer_covariance_diag.device,
            )
            component_covariance = self.layer_covariance_diag.index_select(0, component_layers).unsqueeze(0)
            component_log_covariance = component_covariance.log()
            delta = self.layer_transport_delta[key].to(dtype=torch.float32).unsqueeze(0)
            gate = self.assignment_gates[key]
            for target_layer in component_layers_tuple:
                z_s = z_student_by_layer.get(int(target_layer))
                z_t = z_teacher_by_layer.get(int(target_layer))
                if z_s is None or z_t is None or z_s.shape != z_t.shape or z_s.dim() != 2:
                    continue
                z_s_f = z_s.to(dtype=torch.float32)
                z_t_f = z_t.to(dtype=torch.float32)
                gate_input = F.layer_norm(z_s_f, (int(self.rank),))
                logits = gate(gate_input) / float(self.assignment_temperature)
                log_pi = F.log_softmax(logits, dim=-1)
                pi = log_pi.exp()

                residual = z_t_f.unsqueeze(1) - z_s_f.unsqueeze(1) - delta
                # Averaging the diagonal Gaussian log density over projection
                # dimensions keeps its scale comparable across atlas ranks.
                component_nll = 0.5 * (
                    residual.pow(2) / component_covariance
                    + component_log_covariance
                    + math.log(2.0 * math.pi)
                ).mean(dim=-1)
                mixture_nll = -torch.logsumexp(log_pi - component_nll, dim=-1).mean()
                gate_entropy = -(pi * log_pi).sum(dim=-1).mean()
                posterior_log_prob = F.log_softmax(log_pi - component_nll, dim=-1)
                posterior_prob = posterior_log_prob.exp()
                posterior_entropy = -(posterior_prob * posterior_log_prob).sum(dim=-1).mean()
                max_probability = pi.max(dim=-1).values.mean()

                layer_weight = self.layer_reliability[int(target_layer)]
                weighted_nll = weighted_nll + layer_weight * mixture_nll
                weighted_gate_entropy = weighted_gate_entropy + layer_weight * gate_entropy
                weighted_posterior_entropy = weighted_posterior_entropy + layer_weight * posterior_entropy
                weighted_max_probability = weighted_max_probability + layer_weight * max_probability
                weight_sum = weight_sum + layer_weight
                target_count += 1

        denom = weight_sum.clamp(min=1e-6)
        mean_nll = weighted_nll / denom
        mean_gate_entropy = weighted_gate_entropy / denom
        mean_posterior_entropy = weighted_posterior_entropy / denom
        mean_max_probability = weighted_max_probability / denom
        delta_l2_loss = torch.stack(
            [parameter.pow(2).mean() for parameter in self.layer_transport_delta.values()]
        ).mean()
        loss = mean_nll - float(self.entropy_tau) * mean_gate_entropy + float(self.delta_l2) * delta_l2_loss
        return {
            "loss": loss,
            "nll": mean_nll,
            "gate_entropy": mean_gate_entropy,
            "posterior_entropy": mean_posterior_entropy,
            "effective_components": mean_gate_entropy.exp(),
            "max_probability": mean_max_probability,
            "delta_l2": delta_l2_loss,
            "target_layers": self.layer_covariance_diag.new_tensor(float(target_count)),
        }


class PhaseAdaptiveProjectorBank(nn.Module):
    """Training-only teacher/student charts in the 256-D FFN atlas space.

    Teacher PCA charts are frozen.  Student charts start from the same bases
    and learn coordinate corrections, so matching does not assume that the
    compressed model retains the Teacher's exact hidden coordinates.  This
    module is never installed in the exported inference model.
    """

    def __init__(self, state: Dict[str, Any], *, mode: str) -> None:
        super().__init__()
        normalized = str(mode).strip().lower()
        if normalized not in {"fixed", "layer", "phase", "soft"}:
            raise ValueError(f"unsupported phase projector mode={mode!r}")
        self.mode = normalized
        self.layers = tuple(int(value) for value in state["layers"])
        self.layer_to_slot = {layer: slot for slot, layer in enumerate(self.layers)}
        self.rank = int(state["rank"])
        self.input_dim = int(state["input_dim"])
        self.register_buffer("global_mean", state["global_mean"].float().contiguous())
        self.register_buffer("global_basis", state["global_basis"].float().contiguous())
        selected = torch.tensor(self.layers, dtype=torch.long)
        self.register_buffer(
            "layer_mean", state["layer_mean"].float().index_select(0, selected).contiguous()
        )
        self.register_buffer(
            "layer_basis", state["layer_basis"].float().index_select(0, selected).contiguous()
        )
        self.register_buffer(
            "phase_mean", state["phase_mean"].float().index_select(0, selected).contiguous()
        )
        self.register_buffer(
            "phase_basis", state["phase_basis"].float().index_select(0, selected).contiguous()
        )
        if normalized == "fixed":
            shape = (1, 1, self.input_dim, self.rank)
        elif normalized == "layer":
            shape = (len(self.layers), 1, self.input_dim, self.rank)
        else:
            shape = (len(self.layers), 4, self.input_dim, self.rank)
        self.student_delta = nn.Parameter(torch.zeros(shape, dtype=torch.float32))

    def config_dict(self) -> Dict[str, Any]:
        return {
            "family": "phase_adaptive_teacher_student_projector",
            "mode": self.mode,
            "layers": list(self.layers),
            "input_dim": self.input_dim,
            "rank": self.rank,
            "trainable_parameters": int(self.student_delta.numel()),
            "deployment_parameters": 0,
        }

    def _hard_chart(
        self, z: torch.Tensor, layer_id: int, phase_ids: torch.Tensor, *, student: bool
    ) -> torch.Tensor:
        slot = self.layer_to_slot[int(layer_id)]
        output = z.new_zeros((z.size(0), self.rank), dtype=torch.float32)
        if self.mode == "fixed":
            basis = self.global_basis + (self.student_delta[0, 0] if student else 0.0)
            return (z.float() - self.global_mean.view(1, -1)) @ basis
        if self.mode == "layer":
            basis = self.layer_basis[slot] + (self.student_delta[slot, 0] if student else 0.0)
            return (z.float() - self.layer_mean[slot].view(1, -1)) @ basis
        for phase in range(4):
            mask = phase_ids.eq(phase)
            if not bool(mask.any().item()):
                continue
            basis = self.phase_basis[slot, phase]
            if student:
                basis = basis + self.student_delta[slot, phase]
            output[mask] = (
                z[mask].float() - self.phase_mean[slot, phase].view(1, -1)
            ) @ basis
        return output

    def project(
        self,
        z: torch.Tensor,
        layer_id: int,
        phase_ids: torch.Tensor,
        progress: torch.Tensor,
        *,
        student: bool,
    ) -> torch.Tensor:
        if self.mode != "soft":
            return self._hard_chart(z, layer_id, phase_ids, student=student)
        slot = self.layer_to_slot[int(layer_id)]
        centers = progress.new_tensor([1.0 / 6.0, 0.5, 5.0 / 6.0, 1.0])
        weights = torch.softmax(-((progress.view(-1, 1) - centers.view(1, -1)) / 0.16).pow(2), dim=1)
        outputs = []
        for phase in range(4):
            basis = self.phase_basis[slot, phase]
            if student:
                basis = basis + self.student_delta[slot, phase]
            outputs.append(
                (z.float() - self.phase_mean[slot, phase].view(1, -1)) @ basis
            )
        stacked = torch.stack(outputs, dim=1)
        return (stacked * weights.unsqueeze(-1)).sum(dim=1)

    def forward(
        self,
        z_student: torch.Tensor,
        z_teacher: torch.Tensor,
        layer_id: int,
        phase_ids: torch.Tensor,
        progress: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        student = self.project(
            z_student, layer_id, phase_ids, progress, student=True
        )
        teacher = self.project(
            z_teacher.detach(), layer_id, phase_ids, progress, student=False
        ).detach()
        student_unit = F.normalize(student, dim=-1, eps=1e-6)
        teacher_unit = F.normalize(teacher, dim=-1, eps=1e-6)
        cosine_loss = (1.0 - (student_unit * teacher_unit).sum(dim=-1)).mean()
        teacher_scale = teacher.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-4)
        scale_loss = ((student - teacher) / teacher_scale).pow(2).mean()
        if int(student.size(0)) >= 2:
            relation_loss = (
                (student_unit @ student_unit.T) - (teacher_unit @ teacher_unit.T)
            ).pow(2).mean()
        else:
            relation_loss = cosine_loss.new_zeros(())
        slot = self.layer_to_slot[int(layer_id)]
        if self.mode == "fixed":
            effective_bases = (self.global_basis + self.student_delta[0, 0]).unsqueeze(0)
        elif self.mode == "layer":
            effective_bases = (self.layer_basis[slot] + self.student_delta[slot, 0]).unsqueeze(0)
        else:
            effective_bases = self.phase_basis[slot] + self.student_delta[slot]
        gram = effective_bases.transpose(-1, -2) @ effective_bases
        identity = torch.eye(self.rank, device=gram.device, dtype=gram.dtype)
        orthogonality_loss = (gram - identity).pow(2).mean()
        delta_l2 = self.student_delta.pow(2).mean()
        loss = (
            cosine_loss
            + 0.10 * scale_loss
            + 0.10 * relation_loss
            + 0.01 * orthogonality_loss
            + 0.001 * delta_l2
        )
        return {
            "loss": loss,
            "cosine": 1.0 - cosine_loss,
            "scale_loss": scale_loss,
            "relation_loss": relation_loss,
            "orthogonality_loss": orthogonality_loss,
            "delta_l2": delta_l2,
        }


def _resolve_layers(model: nn.Module) -> Sequence[nn.Module]:
    model = unwrap_model(model)
    root = getattr(model, "model", model)
    layers = getattr(root, "layers", None)
    if layers is None:
        raise RuntimeError("Cannot resolve decoder layers.")
    return layers




def _as_hook_tensor(hook_output: Any) -> Optional[torch.Tensor]:
    if isinstance(hook_output, torch.Tensor):
        return hook_output
    if isinstance(hook_output, (tuple, list)) and hook_output:
        first_item = hook_output[0]
        if isinstance(first_item, torch.Tensor):
            return first_item
    return None


def _resolve_embedding_layer(model: nn.Module) -> Optional[nn.Module]:
    model = unwrap_model(model)
    root = getattr(model, "model", model)
    embedding_layer = getattr(root, "embed_tokens", None)
    if embedding_layer is not None:
        return embedding_layer
    getter = getattr(root, "get_input_embeddings", None)
    if getter is not None:
        try:
            return getter()
        except Exception:
            return None
    return None


def _gather_selected_token_vectors(
    tensor: Optional[torch.Tensor],
    *,
    batch_indices: torch.Tensor,
    token_indices: torch.Tensor,
) -> Optional[torch.Tensor]:
    if tensor is None or tensor.dim() != 3:
        return None
    if batch_indices.numel() <= 0 or token_indices.numel() <= 0:
        return None
    return tensor[batch_indices, token_indices, :]


def _forward_with_selected_capture(
    *,
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    batch_indices: torch.Tensor,
    token_indices: torch.Tensor,
    capture_hidden: bool,
    capture_mlp_layer_ids: Optional[Sequence[int]],
    capture_pre_ffn_input_layer_ids: Optional[Sequence[int]] = None,
    capture_residual_output_layer_ids: Optional[Sequence[int]] = None,
    keep_hook_handles: Optional[List[Any]] = None,
) -> Tuple[Any, Optional[List[Optional[torch.Tensor]]], Dict[int, torch.Tensor], Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    layers = _resolve_layers(model)
    layer_count = int(len(layers))
    selected_layer_ids: Optional[set] = None
    if capture_mlp_layer_ids is not None:
        selected_layer_ids = {
            int(layer_id)
            for layer_id in capture_mlp_layer_ids
            if 0 <= int(layer_id) < layer_count
        }

    hidden_selected: Optional[List[Optional[torch.Tensor]]] = [None] * (layer_count + 1) if bool(capture_hidden) else None
    mlp_selected: Dict[int, torch.Tensor] = {}
    residual_selected: Dict[int, torch.Tensor] = {}
    hook_handles = []

    if bool(capture_hidden):
        embedding_layer = _resolve_embedding_layer(model)
        if embedding_layer is not None:
            def _embed_hook(_, __, output):
                gathered = _gather_selected_token_vectors(
                    _as_hook_tensor(output),
                    batch_indices=batch_indices,
                    token_indices=token_indices,
                )
                if hidden_selected is not None:
                    hidden_selected[0] = gathered

            hook_handles.append(embedding_layer.register_forward_hook(_embed_hook))

        for layer_idx, layer in enumerate(layers):
            def _make_layer_hook(idx: int):
                def _layer_hook(_, __, output):
                    gathered = _gather_selected_token_vectors(
                        _as_hook_tensor(output),
                        batch_indices=batch_indices,
                        token_indices=token_indices,
                    )
                    if hidden_selected is not None:
                        hidden_selected[idx + 1] = gathered
                return _layer_hook

            hook_handles.append(layer.register_forward_hook(_make_layer_hook(layer_idx)))

    for layer_idx, layer in enumerate(layers):
        if selected_layer_ids is not None and int(layer_idx) not in selected_layer_ids:
            continue
        mlp_module = getattr(layer, "mlp", None)
        if mlp_module is None:
            continue

        def _make_mlp_hook(idx: int):
            def _mlp_hook(_, __, output):
                gathered = _gather_selected_token_vectors(
                    _as_hook_tensor(output),
                    batch_indices=batch_indices,
                    token_indices=token_indices,
                )
                if gathered is not None:
                    mlp_selected[int(idx)] = gathered
            return _mlp_hook

        hook_handles.append(mlp_module.register_forward_hook(_make_mlp_hook(layer_idx)))

    residual_selected_ids: Optional[set] = None
    if capture_residual_output_layer_ids is not None:
        residual_selected_ids = {
            int(layer_id)
            for layer_id in capture_residual_output_layer_ids
            if 0 <= int(layer_id) < layer_count
        }
    if residual_selected_ids:
        for layer_idx, layer in enumerate(layers):
            if int(layer_idx) not in residual_selected_ids:
                continue
            mlp_module = getattr(layer, "mlp", None)
            residual_module = getattr(mlp_module, "residual_adapter", None)
            if residual_module is None:
                continue

            def _make_residual_hook(idx: int):
                def _residual_hook(_, __, output):
                    gathered = _gather_selected_token_vectors(
                        _as_hook_tensor(output),
                        batch_indices=batch_indices,
                        token_indices=token_indices,
                    )
                    if gathered is not None:
                        residual_selected[int(idx)] = gathered
                return _residual_hook

            hook_handles.append(residual_module.register_forward_hook(_make_residual_hook(layer_idx)))

    # Capture pre-FFN LN input (u_0 = post-attn residual) for Rectified FM.
    # Gemma 2 uses a pre-hook on pre_feedforward_layernorm to capture the post-attention residual.
    pre_ffn_input_selected: Dict[int, torch.Tensor] = {}
    if capture_pre_ffn_input_layer_ids is not None:
        pre_ffn_set = {int(lid) for lid in capture_pre_ffn_input_layer_ids if 0 <= int(lid) < layer_count}
        for layer_idx, layer in enumerate(layers):
            if int(layer_idx) not in pre_ffn_set:
                continue
            ln_module = getattr(layer, "pre_feedforward_layernorm", None)
            if ln_module is None:
                continue

            def _make_pre_ffn_hook(idx: int):
                def _pre_ffn_hook(_, args_tuple):
                    inp = args_tuple[0] if isinstance(args_tuple, tuple) and len(args_tuple) > 0 else None
                    if inp is not None and isinstance(inp, torch.Tensor) and inp.dim() == 3:
                        gathered = _gather_selected_token_vectors(
                            inp,
                            batch_indices=batch_indices,
                            token_indices=token_indices,
                        )
                        if gathered is not None:
                            pre_ffn_input_selected[int(idx)] = gathered
                return _pre_ffn_hook

            hook_handles.append(ln_module.register_forward_pre_hook(_make_pre_ffn_hook(layer_idx)))

    outputs = None
    try:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            use_cache=False,
            return_dict=True,
        )
    except TypeError:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
    finally:
        if keep_hook_handles is not None:
            keep_hook_handles.extend(hook_handles)
        else:
            for handle in hook_handles:
                try:
                    handle.remove()
                except Exception:
                    pass

    if bool(capture_hidden) and hidden_selected is not None and hidden_selected[0] is None:
        embedding_layer = _resolve_embedding_layer(model)
        if embedding_layer is not None:
            try:
                emb = embedding_layer(input_ids)
                hidden_selected[0] = _gather_selected_token_vectors(
                    emb,
                    batch_indices=batch_indices,
                    token_indices=token_indices,
                )
            except Exception:
                hidden_selected[0] = None
        if hidden_selected[0] is None and layer_count > 0 and hidden_selected[1] is not None:
            hidden_selected[0] = hidden_selected[1]

    return outputs, hidden_selected, mlp_selected, pre_ffn_input_selected, residual_selected


def _extract_logits_from_model_output(model_output: Any) -> torch.Tensor:
    if hasattr(model_output, "logits"):
        logits = getattr(model_output, "logits")
        if isinstance(logits, torch.Tensor):
            return logits
    if isinstance(model_output, (tuple, list)) and model_output:
        first_item = model_output[0]
        if isinstance(first_item, torch.Tensor):
            return first_item
    raise RuntimeError("Cannot extract logits from model output.")


def _masked_nll_per_sample(
    *,
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    token_mask: torch.Tensor,
) -> torch.Tensor:
    labels = input_ids[:, 1:]
    per_token = F.cross_entropy(
        logits[:, :-1, :].float().reshape(-1, int(logits.size(-1))),
        labels.reshape(-1),
        reduction="none",
    ).view_as(labels)
    weights = token_mask.to(dtype=per_token.dtype)
    return (per_token * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)


@torch.no_grad()
def _generate_student_on_policy_rollouts(
    *,
    student: nn.Module,
    input_ids: torch.Tensor,
    prompt_lens: torch.Tensor,
    selected_rows: torch.Tensor,
    pad_token_id: int,
    eos_token_id: Optional[int],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate Student continuations and return right-padded prompt+rollout tensors."""
    prompts: List[torch.Tensor] = []
    for row_tensor in selected_rows:
        row = int(row_tensor.item())
        prompt_len = max(1, int(prompt_lens[row].item()))
        prompts.append(input_ids[row, :prompt_len].detach())
    if not prompts:
        raise RuntimeError("on-policy rollout requested with no selected prompts")

    max_prompt = max(int(item.numel()) for item in prompts)
    prompt_batch = torch.full(
        (len(prompts), max_prompt),
        int(pad_token_id),
        dtype=torch.long,
        device=input_ids.device,
    )
    prompt_mask = torch.zeros_like(prompt_batch)
    for row, prompt in enumerate(prompts):
        width = int(prompt.numel())
        prompt_batch[row, max_prompt - width :] = prompt
        prompt_mask[row, max_prompt - width :] = 1

    model = unwrap_model(student)
    was_training = bool(model.training)
    model.eval()
    sample = float(temperature) > 0.0
    generated = model.generate(
        input_ids=prompt_batch,
        attention_mask=prompt_mask,
        max_new_tokens=max(1, int(max_new_tokens)),
        do_sample=sample,
        temperature=max(1e-5, float(temperature)) if sample else None,
        top_p=min(1.0, max(1e-5, float(top_p))) if sample else None,
        num_beams=1,
        pad_token_id=int(pad_token_id),
        eos_token_id=eos_token_id,
        use_cache=True,
    )
    if was_training:
        model.train()

    continuations: List[torch.Tensor] = []
    for row in range(int(generated.size(0))):
        continuation = generated[row, max_prompt:]
        if eos_token_id is not None:
            eos_hits = continuation.eq(int(eos_token_id)).nonzero(as_tuple=False)
            if int(eos_hits.numel()) > 0:
                continuation = continuation[: int(eos_hits[0, 0].item()) + 1]
        continuations.append(continuation)

    full_sequences = [
        torch.cat([prompt, continuation], dim=0)
        for prompt, continuation in zip(prompts, continuations)
    ]
    max_length = max(int(item.numel()) for item in full_sequences)
    rollout_ids = torch.full(
        (len(full_sequences), max_length),
        int(pad_token_id),
        dtype=torch.long,
        device=input_ids.device,
    )
    rollout_mask = torch.zeros_like(rollout_ids)
    rollout_prompt_lens = torch.zeros((len(full_sequences),), dtype=torch.long, device=input_ids.device)
    for row, (sequence, prompt) in enumerate(zip(full_sequences, prompts)):
        width = int(sequence.numel())
        rollout_ids[row, :width] = sequence
        rollout_mask[row, :width] = 1
        rollout_prompt_lens[row] = int(prompt.numel())
    return rollout_ids, rollout_mask, rollout_prompt_lens


def _on_policy_divergence(
    *,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    token_mask: torch.Tensor,
    divergence: str,
    temperature: float,
) -> torch.Tensor:
    temp = max(1e-5, float(temperature))
    s_logp = F.log_softmax(student_logits[:, :-1, :].float() / temp, dim=-1)
    t_logp = F.log_softmax(teacher_logits[:, :-1, :].float() / temp, dim=-1)
    mode = str(divergence).strip().lower()
    if mode == "reverse_kl":
        s_prob = s_logp.exp()
        per_token = (s_prob * (s_logp - t_logp)).sum(dim=-1)
    elif mode == "forward_kl":
        t_prob = t_logp.exp()
        per_token = (t_prob * (t_logp - s_logp)).sum(dim=-1)
    elif mode in {"js", "jensen_shannon"}:
        s_prob = s_logp.exp()
        t_prob = t_logp.exp()
        mixture = (0.5 * (s_prob + t_prob)).clamp(min=1e-8)
        log_mixture = mixture.log()
        per_token = 0.5 * (s_prob * (s_logp - log_mixture)).sum(dim=-1)
        per_token = per_token + 0.5 * (t_prob * (t_logp - log_mixture)).sum(dim=-1)
    else:
        raise ValueError(f"unsupported on-policy divergence: {divergence!r}")
    weights = token_mask.to(dtype=per_token.dtype)
    return (per_token * weights).sum() / weights.sum().clamp(min=1.0) * (temp * temp)


def _adaptive_token_flow_alignment(
    *,
    student_z: torch.Tensor,
    teacher_z: torch.Tensor,
    batch_indices: torch.Tensor,
    token_indices: torch.Tensor,
    metric_scale: torch.Tensor,
    energy_fraction: float,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, int, float]:
    """Match informative local secants and turns without assuming a smooth path.

    The Teacher tangent support is recomputed for every trajectory segment from
    the smallest coordinate set that explains ``energy_fraction`` of the
    metric-scaled Teacher displacement.  Consequently the Student is not tied
    to one global Teacher chart.  The second term matches observed turns rather
    than penalizing curvature: jagged but useful Teacher paths remain valid.
    """
    zero = student_z.new_zeros((), dtype=torch.float32)
    if (
        student_z.dim() != 2
        or teacher_z.dim() != 2
        or student_z.shape != teacher_z.shape
        or int(student_z.size(0)) < 2
    ):
        return zero, zero, 0, 0.0
    fraction = min(1.0, max(0.05, float(energy_fraction)))
    scale = metric_scale.to(device=student_z.device, dtype=torch.float32).view(1, -1)
    z_s = student_z.float() * scale
    z_t = teacher_z.float() * scale
    segment_student: List[torch.Tensor] = []
    segment_teacher: List[torch.Tensor] = []
    active_dimensions: List[int] = []
    segment_groups: List[List[int]] = []
    unique_batches = torch.unique(batch_indices.to(dtype=torch.long), sorted=True)
    for batch_id_tensor in unique_batches:
        members = (
            batch_indices.to(dtype=torch.long)
            .eq(int(batch_id_tensor.item()))
            .nonzero(as_tuple=False)
            .view(-1)
        )
        if int(members.numel()) < 2:
            continue
        order = torch.argsort(token_indices.index_select(0, members))
        members = members.index_select(0, order)
        local_segment_ids: List[int] = []
        for left_pos, right_pos in zip(members[:-1], members[1:]):
            left = int(left_pos.item())
            right = int(right_pos.item())
            if int(token_indices[right].item()) <= int(token_indices[left].item()):
                continue
            delta_t = z_t[right] - z_t[left]
            delta_s = z_s[right] - z_s[left]
            energy = delta_t.square()
            total = energy.sum().clamp(min=eps)
            sorted_energy, sorted_idx = torch.sort(energy, descending=True)
            cumulative = torch.cumsum(sorted_energy, dim=0)
            needed = int(
                torch.searchsorted(
                    cumulative,
                    total * fraction,
                    right=False,
                ).item()
            ) + 1
            needed = max(1, min(needed, int(delta_t.numel())))
            mask = torch.zeros_like(delta_t)
            mask.scatter_(0, sorted_idx[:needed], 1.0)
            segment_teacher.append(delta_t * mask)
            segment_student.append(delta_s * mask)
            active_dimensions.append(needed)
            local_segment_ids.append(len(segment_teacher) - 1)
        if local_segment_ids:
            segment_groups.append(local_segment_ids)
    if not segment_teacher:
        return zero, zero, 0, 0.0
    teacher_flow = torch.stack(segment_teacher, dim=0)
    student_flow = torch.stack(segment_student, dim=0)
    teacher_unit = F.normalize(teacher_flow, p=2, dim=-1, eps=eps)
    student_unit = F.normalize(student_flow, p=2, dim=-1, eps=eps)
    direction = 1.0 - (teacher_unit * student_unit).sum(dim=-1)
    teacher_norm = teacher_flow.norm(dim=-1).clamp(min=eps)
    student_norm = student_flow.norm(dim=-1).clamp(min=eps)
    scale_loss = (student_norm.log() - teacher_norm.log()).square()
    flow_loss = (direction + 0.10 * scale_loss).mean()

    turning_terms: List[torch.Tensor] = []
    for ids in segment_groups:
        if len(ids) < 2:
            continue
        for left_id, right_id in zip(ids[:-1], ids[1:]):
            teacher_turn = (teacher_unit[left_id] * teacher_unit[right_id]).sum()
            student_turn = (student_unit[left_id] * student_unit[right_id]).sum()
            turning_terms.append((student_turn - teacher_turn).square())
    turning_loss = torch.stack(turning_terms).mean() if turning_terms else zero
    mean_dimensions = float(sum(active_dimensions) / max(1, len(active_dimensions)))
    return flow_loss, turning_loss, len(segment_teacher), mean_dimensions


def resolve_proto_seed_layers(
    *,
    layer_to_proto: Sequence[int],
    proto_seed_strategy: str,
    layer_signatures: Optional[torch.Tensor],
    policy_medoid_seed_layers: Optional[Dict[int, int]] = None,
) -> Tuple[Dict[int, int], str]:
    proto_ids = sorted({int(x) for x in layer_to_proto})
    first_layer_for_proto: Dict[int, int] = {}
    for layer_idx, proto_id in enumerate(layer_to_proto):
        proto_id_int = int(proto_id)
        if proto_id_int not in first_layer_for_proto:
            first_layer_for_proto[proto_id_int] = int(layer_idx)

    strategy = str(proto_seed_strategy).strip().lower()
    if strategy == "first":
        return first_layer_for_proto, "first"
    if strategy == "policy_medoid":
        if not isinstance(policy_medoid_seed_layers, dict):
            raise ValueError(
                "proto_seed_strategy=policy_medoid requires every sharing-policy group "
                "to provide a valid medoid_layer"
            )
        resolved = {int(k): int(v) for k, v in policy_medoid_seed_layers.items()}
        expected = set(proto_ids)
        if set(resolved) != expected:
            missing = sorted(expected - set(resolved))
            extra = sorted(set(resolved) - expected)
            raise ValueError(
                "incomplete policy-medoid seed mapping: "
                f"missing_proto_ids={missing} extra_proto_ids={extra}"
            )
        for proto_id, layer_id in resolved.items():
            if layer_id < 0 or layer_id >= len(layer_to_proto):
                raise ValueError(f"policy medoid layer out of range: proto={proto_id} layer={layer_id}")
            if int(layer_to_proto[layer_id]) != int(proto_id):
                raise ValueError(
                    "policy medoid does not belong to its prototype: "
                    f"proto={proto_id} layer={layer_id} mapped_proto={layer_to_proto[layer_id]}"
                )
        return resolved, "policy_medoid"
    if strategy != "medoid":
        raise ValueError(f"Unsupported proto_seed_strategy: {proto_seed_strategy}")

    layer_count = len(layer_to_proto)
    if layer_signatures is None:
        return first_layer_for_proto, "first_fallback_no_signature"
    if not torch.is_tensor(layer_signatures):
        return first_layer_for_proto, "first_fallback_invalid_signature"
    if int(layer_signatures.dim()) != 2 or int(layer_signatures.size(0)) != int(layer_count):
        return first_layer_for_proto, "first_fallback_signature_shape"

    sig = layer_signatures.detach().to(dtype=torch.float32, device="cpu")
    medoid_layer_for_proto: Dict[int, int] = {}
    for proto_id in proto_ids:
        member_layers = [idx for idx, item in enumerate(layer_to_proto) if int(item) == int(proto_id)]
        if not member_layers:
            continue
        if len(member_layers) == 1:
            medoid_layer_for_proto[int(proto_id)] = int(member_layers[0])
            continue
        idx_tensor = torch.tensor(member_layers, dtype=torch.long)
        member_sig = sig.index_select(0, idx_tensor)
        center = member_sig.mean(dim=0, keepdim=True)
        dist2 = ((member_sig - center) ** 2).sum(dim=1)
        best_local = int(torch.argmin(dist2).item())
        medoid_layer_for_proto[int(proto_id)] = int(member_layers[best_local])
    if len(medoid_layer_for_proto) != len(proto_ids):
        return first_layer_for_proto, "first_fallback_medoid_incomplete"
    return medoid_layer_for_proto, "medoid"


def apply_shared_ffn(
    model: nn.Module,
    *,
    layer_to_proto: Sequence[int],
    lora_rank: int,
    lora_alpha: Optional[float] = None,
    proto_seed_strategy: str = "first",
    layer_signatures: Optional[torch.Tensor] = None,
    policy_medoid_seed_layers: Optional[Dict[int, int]] = None,
    sharing_parameterization: str = "full_parallel",
    use_layer_scalar: bool = True,
    adapter_every_layer: bool = False,
) -> None:
    model = unwrap_model(model)
    layers = _resolve_layers(model)
    if len(layers) != len(layer_to_proto):
        raise ValueError(f"layer_to_proto len={len(layer_to_proto)} does not match layers={len(layers)}")
    proto_ids = sorted({int(x) for x in layer_to_proto})
    proto_counts: Dict[int, int] = {}
    for proto_id_raw in layer_to_proto:
        proto_id = int(proto_id_raw)
        proto_counts[proto_id] = proto_counts.get(proto_id, 0) + 1
    seed_layer_for_proto, resolved_strategy = resolve_proto_seed_layers(
        layer_to_proto=layer_to_proto,
        proto_seed_strategy=str(proto_seed_strategy),
        layer_signatures=layer_signatures,
        policy_medoid_seed_layers=policy_medoid_seed_layers,
    )
    if resolved_strategy != str(proto_seed_strategy).strip().lower():
        print(
            f"[SharedFFN] proto_seed_strategy={proto_seed_strategy} -> {resolved_strategy}",
            flush=True,
        )

    ref_param = next(model.parameters())
    ref_device = ref_param.device
    ref_dtype = ref_param.dtype
    parameterization = str(sharing_parameterization).strip().lower()
    if parameterization not in {
        "full_parallel",
        "down_only_parallel",
        "internal_weight_delta",
    }:
        raise ValueError(f"unsupported sharing_parameterization={sharing_parameterization!r}")
    original_mlps = [layer.mlp for layer in layers]

    bank = nn.ModuleList()
    proto_to_bank_index: Dict[int, int] = {}
    for proto_id in proto_ids:
        source_layer_idx = int(seed_layer_for_proto[proto_id])
        source_mlp = original_mlps[source_layer_idx]
        copied = copy.deepcopy(
            source_mlp.down_proj
            if parameterization == "down_only_parallel"
            else source_mlp
        )
        copied = copied.to(device=ref_device, dtype=ref_dtype)
        bank.append(copied)
        proto_to_bank_index[proto_id] = int(len(bank) - 1)

    root = getattr(model, "model", model)
    setattr(root, "shared_mlp_bank", bank)
    setattr(root, "shared_mlp_parameterization", parameterization)
    bank_refs: List[nn.Module] = list(bank)
    hidden_size = int(getattr(model.config, "hidden_size", 0) or 0)
    intermediate_size = int(getattr(model.config, "intermediate_size", 0) or 0)
    if hidden_size <= 0:
        raise RuntimeError("Invalid hidden_size in config.")

    for layer_idx, layer in enumerate(layers):
        proto_id = int(layer_to_proto[layer_idx])
        bank_idx = int(proto_to_bank_index[proto_id])
        # A singleton prototype is an unchanged, layer-private FFN.  Giving it
        # a residual adapter wastes parameters and lets training perturb an
        # operator that compression did not replace.  Reserve private capacity
        # only for genuinely shared groups.
        layer_lora_rank = (
            int(lora_rank)
            if bool(adapter_every_layer)
            or (
                proto_counts[proto_id] > 1
                and int(layer_idx) != int(seed_layer_for_proto[proto_id])
            )
            else 0
        )
        layer.mlp = SharedMLPAdapter(
            proto_id=bank_idx,
            bank_refs=bank_refs,
            hidden_size=hidden_size,
            lora_rank=layer_lora_rank,
            lora_alpha=lora_alpha,
            sharing_parameterization=parameterization,
            original_mlp=original_mlps[layer_idx],
            intermediate_size=intermediate_size,
            use_layer_scalar=bool(use_layer_scalar),
        )
        layer.mlp = layer.mlp.to(device=ref_device, dtype=ref_dtype)


def extract_shared_state(model: nn.Module, layer_to_proto: Sequence[int]) -> Dict[str, Any]:
    model = unwrap_model(model)
    root = getattr(model, "model", model)
    bank = getattr(root, "shared_mlp_bank", None)
    if bank is None:
        raise RuntimeError("shared_mlp_bank not found in model.")
    layers = _resolve_layers(model)
    adapter_state: Dict[str, Dict[str, torch.Tensor]] = {}
    for layer_idx, layer in enumerate(layers):
        adapter = layer.mlp
        if not isinstance(adapter, SharedMLPAdapter):
            raise RuntimeError(f"Layer {layer_idx} has non-shared MLP.")
        adapter_state[str(layer_idx)] = adapter.export_state()
    return {
        "layer_to_proto": [int(x) for x in layer_to_proto],
        "use_layer_scalar": bool(
            any(
                isinstance(layer.mlp, SharedMLPAdapter)
                and layer.mlp.use_layer_scalar
                for layer in layers
            )
        ),
        "adapter_every_layer": bool(
            all(
                isinstance(layer.mlp, SharedMLPAdapter)
                and layer.mlp.lora_rank > 0
                for layer in layers
            )
        ),
        "sharing_parameterization": str(
            getattr(root, "shared_mlp_parameterization", "full_parallel")
        ),
        "bank_state": bank.state_dict(),
        "adapter_state": adapter_state,
    }


def load_shared_state(
    model: nn.Module,
    payload: Dict[str, Any],
    lora_rank: int,
    lora_alpha: Optional[float] = None,
) -> List[int]:
    model = unwrap_model(model)
    layer_to_proto = [int(x) for x in payload["layer_to_proto"]]
    apply_shared_ffn(
        model,
        layer_to_proto=layer_to_proto,
        lora_rank=int(lora_rank),
        lora_alpha=lora_alpha,
        sharing_parameterization=str(
            payload.get("sharing_parameterization", "full_parallel")
        ),
        use_layer_scalar=bool(payload.get("use_layer_scalar", True)),
        adapter_every_layer=bool(payload.get("adapter_every_layer", False)),
    )
    root = getattr(model, "model", model)
    bank = getattr(root, "shared_mlp_bank")
    bank.load_state_dict(payload["bank_state"], strict=True)
    layers = _resolve_layers(model)
    for layer_idx, layer in enumerate(layers):
        adapter = layer.mlp
        state = payload["adapter_state"][str(layer_idx)]
        adapter.load_exported_state(state)
    return layer_to_proto


def _metric_right_transform(
    values: torch.Tensor,
    *,
    basis: torch.Tensor,
    precision_diag: torch.Tensor,
    complement_floor: float,
    inverse: bool,
) -> torch.Tensor:
    """Apply T or T^-1 where T^2 is a subspace precision with isotropic floor."""
    floor = max(1e-4, float(complement_floor))
    basis = basis.to(device=values.device, dtype=values.dtype)
    precision = precision_diag.to(device=values.device, dtype=values.dtype).clamp(min=1e-6)
    base = math.sqrt(floor)
    if inverse:
        base_scale = 1.0 / base
        subspace_scale = precision.rsqrt() - base_scale
    else:
        base_scale = base
        subspace_scale = precision.sqrt() - base_scale
    projected = torch.matmul(values, basis)
    return values * base_scale + torch.matmul(projected * subspace_scale.view(1, -1), basis.transpose(0, 1))


@torch.no_grad()
def _select_reasoning_residual_positions(
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lens: torch.Tensor,
    teacher_logits: torch.Tensor,
    captured_outputs: Dict[int, torch.Tensor],
    flow_layer_ids: Sequence[int],
    rows_to_take: int,
    tokens_per_record: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select phase anchors plus uncertainty/high-flow reasoning positions."""
    selected_rows: List[int] = []
    selected_tokens: List[int] = []
    per_record = max(6, int(tokens_per_record))
    uncertainty_count = 2
    flow_count = 2
    phase_count = max(2, per_record - uncertainty_count - flow_count)
    for row in range(min(int(rows_to_take), int(input_ids.size(0)))):
        valid_len = int(attention_mask[row].sum().item())
        start = max(0, min(int(prompt_lens[row].item()) - 1, valid_len - 2))
        end = max(start, valid_len - 2)
        phase_positions = (
            torch.linspace(start, end, steps=phase_count, device=input_ids.device)
            .round()
            .to(dtype=torch.long)
        )
        candidate_count = min(64, end - start + 1)
        candidate_positions = (
            torch.linspace(start, end, steps=max(1, candidate_count), device=input_ids.device)
            .round()
            .to(dtype=torch.long)
        )
        candidate_logits = teacher_logits[row].index_select(0, candidate_positions).float()
        labels = input_ids[row].index_select(0, candidate_positions + 1)
        next_token_nll = (
            torch.logsumexp(candidate_logits, dim=-1)
            - candidate_logits.gather(1, labels.view(-1, 1)).squeeze(1)
        )
        uncertainty_positions = candidate_positions.index_select(
            0,
            torch.topk(
                next_token_nll,
                k=min(uncertainty_count, int(next_token_nll.numel())),
            ).indices,
        )

        flow_score = torch.zeros(
            (int(candidate_positions.numel()),),
            dtype=torch.float32,
            device=input_ids.device,
        )
        contributing_layers = 0
        for layer_id in flow_layer_ids:
            values = captured_outputs.get(int(layer_id))
            if values is None:
                continue
            sampled = values[row].index_select(0, candidate_positions).float()
            if int(sampled.size(0)) > 1:
                delta = (sampled[1:] - sampled[:-1]).pow(2).mean(dim=-1)
                flow_score[1:] += delta
                contributing_layers += 1
        if contributing_layers > 0:
            flow_score /= float(contributing_layers)
        flow_positions = candidate_positions.index_select(
            0,
            torch.topk(
                flow_score,
                k=min(flow_count, int(flow_score.numel())),
            ).indices,
        )
        combined = torch.cat(
            [phase_positions, uncertainty_positions, flow_positions], dim=0
        )
        if int(combined.numel()) < per_record:
            padding = combined[-1:].expand(per_record - int(combined.numel()))
            combined = torch.cat([combined, padding], dim=0)
        combined = combined[:per_record]
        selected_rows.extend([row] * per_record)
        selected_tokens.extend(int(item) for item in combined.tolist())
    return (
        torch.tensor(selected_rows, dtype=torch.long, device=input_ids.device),
        torch.tensor(selected_tokens, dtype=torch.long, device=input_ids.device),
    )


@torch.no_grad()
def initialize_shared_ffn_functional_residual_svd(
    *,
    student: nn.Module,
    teacher: nn.Module,
    loader: DataLoader,
    layer_to_proto: Sequence[int],
    seed_layer_for_proto: Dict[int, int],
    mode: str,
    global_records: int,
    tokens_per_record: int,
    max_fit_tokens: int,
    ridge: float,
    oversample: int,
    metric_complement_floor: float,
    regime_labels: Sequence[str],
    regime_basis_cpu_map: Dict[str, torch.Tensor],
    layer_metric_diag_cpu: torch.Tensor,
    core_metric_diag_mode: str,
    core_metric_trace_normalize: bool,
    eos_token_id: Optional[int],
    seed: int,
    device: torch.device,
) -> Dict[str, Any]:
    """Initialize layer-private input/output LoRA from FFN functional residuals.

    The shared bank is first copied from the corresponding original teacher medoid.
    For each non-medoid layer, reduced-rank ridge regression fits
      F_teacher,l(x) - scale_l F_teacher,medoid(x) ~= gamma B_l A_l x.
    In task_metric mode, output residuals are whitened by the supplied task precision
    before rank truncation and mapped back afterward.
    """
    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in {"functional", "task_metric"}:
        raise ValueError(f"unsupported residual SVD init mode: {mode!r}")
    model = unwrap_model(student)
    teacher_model = unwrap_model(teacher)
    student_layers = _resolve_layers(model)
    teacher_layers = _resolve_layers(teacher_model)
    root = getattr(model, "model", model)
    bank = getattr(root, "shared_mlp_bank", None)
    if bank is None:
        raise RuntimeError("shared_mlp_bank missing before residual SVD initialization")
    proto_ids = sorted({int(value) for value in layer_to_proto})
    if len(bank) != len(proto_ids):
        raise RuntimeError(f"bank/prototype mismatch: bank={len(bank)} protos={len(proto_ids)}")

    # Faithful replacement initialization: the tied operator is the teacher
    # medoid's native core.  A deploy-bundle teacher exposes SharedMLPAdapter at
    # each layer, so unwrap its referenced core instead of attempting to load
    # adapter scale/LoRA tensors into a native LlamaMLP.
    teacher_is_shared = False
    for bank_index, proto_id in enumerate(proto_ids):
        source_layer = int(seed_layer_for_proto[int(proto_id)])
        source_mlp = teacher_layers[source_layer].mlp
        if isinstance(source_mlp, SharedMLPAdapter):
            teacher_is_shared = True
            source_bank = getattr(source_mlp, "_bank_refs", None)
            if source_bank is None:
                raise RuntimeError(f"shared teacher layer {source_layer} has no bank references")
            source_core = source_bank[int(source_mlp.proto_id)]
        else:
            source_core = source_mlp
        target_core = bank[bank_index]
        if isinstance(target_core, nn.Linear):
            if isinstance(source_core, nn.Linear):
                source_projection = source_core
            else:
                source_projection = getattr(source_core, "down_proj", None)
            if not isinstance(source_projection, nn.Linear):
                raise RuntimeError("down-only bank source has no down_proj")
            target_core.load_state_dict(source_projection.state_dict(), strict=True)
        else:
            target_core.load_state_dict(source_core.state_dict(), strict=True)

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    requested = max(world_size, int(global_records))
    local_target = int(math.ceil(float(requested) / float(world_size)))
    effective_global_records = local_target * world_size
    layer_inputs: List[List[torch.Tensor]] = [[] for _ in student_layers]
    layer_targets: List[List[torch.Tensor]] = [[] for _ in student_layers]
    layer_shared: List[List[torch.Tensor]] = [[] for _ in student_layers]
    captured_inputs: Dict[int, torch.Tensor] = {}
    captured_outputs: Dict[int, torch.Tensor] = {}
    handles: List[Any] = []
    proto_counts: Dict[int, int] = {}
    for proto_id_raw in layer_to_proto:
        proto_id = int(proto_id_raw)
        proto_counts[proto_id] = proto_counts.get(proto_id, 0) + 1
    shared_layer_ids = [
        layer_id
        for layer_id, proto_id_raw in enumerate(layer_to_proto)
        if proto_counts[int(proto_id_raw)] > 1
    ]

    for layer_id, layer in enumerate(teacher_layers):
        if layer_id not in shared_layer_ids:
            continue
        def make_pre_hook(index: int):
            def hook(_module, args_tuple):
                value = args_tuple[0] if isinstance(args_tuple, tuple) and args_tuple else None
                if torch.is_tensor(value):
                    captured_inputs[index] = value
            return hook

        def make_output_hook(index: int):
            def hook(_module, _args, output):
                tensor = _as_hook_tensor(output)
                if tensor is not None:
                    captured_outputs[index] = tensor
            return hook

        handles.append(layer.mlp.register_forward_pre_hook(make_pre_hook(layer_id)))
        handles.append(layer.mlp.register_forward_hook(make_output_hook(layer_id)))

    collected = 0
    try:
        for batch in loader:
            if collected >= local_target:
                break
            captured_inputs.clear()
            captured_outputs.clear()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            teacher_outputs = teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            take = min(local_target - collected, int(input_ids.size(0)))
            prompt_lens = batch.get("prompt_lens", None)
            if prompt_lens is None:
                raise RuntimeError("trajectory residual initialization requires prompt_lens")
            selected_rows, selected_tokens = _select_reasoning_residual_positions(
                input_ids=input_ids,
                attention_mask=attention_mask,
                prompt_lens=prompt_lens.to(device),
                teacher_logits=teacher_outputs.logits,
                captured_outputs=captured_outputs,
                flow_layer_ids=shared_layer_ids,
                rows_to_take=take,
                tokens_per_record=int(tokens_per_record),
            )
            if len(captured_inputs) != len(shared_layer_ids) or len(captured_outputs) != len(shared_layer_ids):
                raise RuntimeError(
                    f"residual SVD hooks captured inputs={len(captured_inputs)} outputs={len(captured_outputs)} "
                    f"expected={len(shared_layer_ids)}"
                )
            for layer_id in shared_layer_ids:
                student_layer = student_layers[layer_id]
                adapter = student_layer.mlp
                if not isinstance(adapter, SharedMLPAdapter):
                    raise RuntimeError(f"layer {layer_id} is not SharedMLPAdapter")
                x = captured_inputs[layer_id][selected_rows, selected_tokens, :]
                target = captured_outputs[layer_id][selected_rows, selected_tokens, :]
                shared = adapter.forward_base(x)
                layer_inputs[layer_id].append(x.detach().to(dtype=torch.float32, device="cpu"))
                layer_targets[layer_id].append(target.detach().to(dtype=torch.float32, device="cpu"))
                layer_shared[layer_id].append(shared.detach().to(dtype=torch.float32, device="cpu"))
            collected += take
            del teacher_outputs
    finally:
        for handle in handles:
            handle.remove()
    if collected != local_target:
        raise RuntimeError(f"residual SVD calibration exhausted: collected={collected} target={local_target}")

    layer_stats: Dict[str, Any] = {}
    for layer_id, student_layer in enumerate(student_layers):
        adapter = student_layer.mlp
        proto_id = int(layer_to_proto[layer_id])
        source_layer = int(seed_layer_for_proto[proto_id])
        if proto_counts[proto_id] <= 1:
            if rank == 0:
                adapter.scale.data.fill_(1.0)
                layer_stats[str(layer_id)] = {
                    "proto_id": int(proto_id),
                    "source_layer": int(source_layer),
                    "scale": 1.0,
                    "residual_rms": 0.0,
                    "explained_energy": 1.0,
                    "medoid_zero_residual": True,
                    "private_singleton": True,
                }
            if dist.is_initialized():
                dist.broadcast(adapter.scale.data, src=0)
            continue
        x_local = torch.cat(layer_inputs[layer_id], dim=0).to(device=device)
        y_local = torch.cat(layer_targets[layer_id], dim=0).to(device=device)
        shared_local = torch.cat(layer_shared[layer_id], dim=0).to(device=device)
        if dist.is_initialized():
            # NCCL all_gather is consistently available across the cluster's
            # PyTorch builds, whereas gather support varies by build/version.
            x_parts = [torch.empty_like(x_local) for _ in range(world_size)]
            y_parts = [torch.empty_like(y_local) for _ in range(world_size)]
            shared_parts = [torch.empty_like(shared_local) for _ in range(world_size)]
            dist.all_gather(x_parts, x_local)
            dist.all_gather(y_parts, y_local)
            dist.all_gather(shared_parts, shared_local)
            if rank == 0:
                x_all = torch.cat(x_parts, dim=0)
                y_all = torch.cat(y_parts, dim=0)
                shared_all = torch.cat(shared_parts, dim=0)
        else:
            x_all, y_all, shared_all = x_local, y_local, shared_local

        if rank == 0 and int(max_fit_tokens) > 0 and int(x_all.size(0)) > int(max_fit_tokens):
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed) + 7919)
            keep = torch.randperm(
                int(x_all.size(0)), generator=generator, device=device
            )[: int(max_fit_tokens)]
            x_all = x_all.index_select(0, keep)
            y_all = y_all.index_select(0, keep)
            shared_all = shared_all.index_select(0, keep)

        if rank == 0:
            if int(adapter.lora_rank) <= 0:
                adapter.scale.data.fill_(1.0)
                layer_stats[str(layer_id)] = {
                    "proto_id": int(proto_id),
                    "source_layer": int(source_layer),
                    "scale": 1.0,
                    "residual_rms": 0.0,
                    "explained_energy": 1.0,
                    "medoid_zero_residual": True,
                    "private_singleton": False,
                }
            else:
                gamma = float(adapter.lora_scaling)
                if gamma <= 0.0:
                    raise RuntimeError("residual SVD initialization requires positive LoRA scaling")
                if int(layer_id) == source_layer:
                    scale_value = 1.0
                    down_weight = torch.zeros_like(adapter.lora_down.weight, dtype=torch.float32, device=device)
                    up_weight = torch.zeros_like(adapter.lora_up.weight, dtype=torch.float32, device=device)
                    explained = 1.0
                    residual_rms = 0.0
                else:
                    numerator = (shared_all * y_all).sum()
                    denominator = shared_all.square().sum().clamp(min=1e-12)
                    scale_value = float((numerator / denominator).item())
                    residual = y_all - shared_all * scale_value
                    residual_rms = float(residual.square().mean().sqrt().item())
                    transformed = residual
                    basis = regime_basis_cpu_map[str(regime_labels[layer_id])]
                    precision = layer_metric_diag_cpu[layer_id].float().clamp(min=1e-6)
                    if str(core_metric_diag_mode).strip().lower() == "covariance":
                        precision = precision.reciprocal()
                    if bool(core_metric_trace_normalize):
                        precision = precision / precision.mean().clamp(min=1e-6)
                    if normalized_mode == "task_metric":
                        transformed = _metric_right_transform(
                            residual,
                            basis=basis,
                            precision_diag=precision,
                            complement_floor=float(metric_complement_floor),
                            inverse=False,
                        )

                    gram = torch.matmul(x_all, x_all.transpose(0, 1))
                    ridge_absolute = max(1e-8, float(ridge)) * gram.diagonal().mean().clamp(min=1e-8)
                    regularized = gram + torch.eye(int(gram.size(0)), device=device, dtype=gram.dtype) * ridge_absolute
                    dual = torch.linalg.solve(regularized, transformed)
                    rank_target = min(int(adapter.lora_rank), int(x_all.size(0)), int(x_all.size(1)))
                    q = min(int(x_all.size(1)), rank_target + max(2, int(oversample)))
                    generator = torch.Generator(device=device)
                    generator.manual_seed(int(seed) + 1009 * int(layer_id) + 17)
                    omega = torch.randn((int(x_all.size(1)), q), generator=generator, device=device, dtype=x_all.dtype)
                    range_matrix = torch.matmul(dual.transpose(0, 1), torch.matmul(x_all, omega))
                    q_basis = torch.linalg.qr(range_matrix, mode="reduced").Q
                    small = torch.matmul(torch.matmul(dual, q_basis).transpose(0, 1), x_all)
                    u_small, singular_values, vh = torch.linalg.svd(small, full_matrices=False)
                    singular_values = singular_values[:rank_target].clamp(min=0.0)
                    u = torch.matmul(q_basis, u_small[:, :rank_target])
                    vh = vh[:rank_target, :]
                    if normalized_mode == "task_metric":
                        u = _metric_right_transform(
                            u.transpose(0, 1),
                            basis=basis,
                            precision_diag=precision,
                            complement_floor=float(metric_complement_floor),
                            inverse=True,
                        ).transpose(0, 1)
                    sqrt_s = torch.sqrt(singular_values / gamma).clamp(min=0.0)
                    down_weight = u * sqrt_s.view(1, -1)
                    up_weight = sqrt_s.view(-1, 1) * vh
                    total_energy = (gram * torch.matmul(dual, dual.transpose(0, 1))).sum().clamp(min=1e-12)
                    explained = float((singular_values.square().sum() / total_energy).clamp(min=0.0, max=1.0).item())

                adapter.scale.data.fill_(scale_value)
                adapter.lora_down.weight.data.copy_(down_weight.to(dtype=adapter.lora_down.weight.dtype))
                adapter.lora_up.weight.data.copy_(up_weight.to(dtype=adapter.lora_up.weight.dtype))
                layer_stats[str(layer_id)] = {
                    "proto_id": int(proto_id),
                    "source_layer": int(source_layer),
                    "scale": float(scale_value),
                    "residual_rms": float(residual_rms),
                    "explained_energy": float(explained),
                    "medoid_zero_residual": bool(int(layer_id) == int(source_layer)),
                    "private_singleton": False,
                }

        if dist.is_initialized():
            dist.broadcast(adapter.scale.data, src=0)
            if adapter.lora_down is not None and adapter.lora_up is not None:
                dist.broadcast(adapter.lora_down.weight.data, src=0)
                dist.broadcast(adapter.lora_up.weight.data, src=0)

    if dist.is_initialized():
        payload: List[Any] = [layer_stats if rank == 0 else None]
        dist.broadcast_object_list(payload, src=0)
        layer_stats = payload[0]
        dist.barrier()
    summary = {
        "enabled": True,
        "mode": normalized_mode,
        "global_records": int(effective_global_records),
        "tokens_per_record": int(tokens_per_record),
        "max_fit_tokens": int(max_fit_tokens),
        "token_strategy": "phase+uncertainty+teacher_flow",
        "rank": max(
            int(layer.mlp.lora_rank)
            for layer in student_layers
            if isinstance(layer.mlp, SharedMLPAdapter)
        ),
        "ridge": float(ridge),
        "oversample": int(oversample),
        "metric_complement_floor": float(metric_complement_floor),
        "teacher_bank_copy": True,
        "teacher_source": "shared_teacher_core" if teacher_is_shared else "native_teacher_layer",
        "layers": layer_stats,
    }
    if rank == 0:
        non_medoid = [row for row in layer_stats.values() if not row["medoid_zero_residual"]]
        mean_explained = sum(float(row["explained_energy"]) for row in non_medoid) / max(1, len(non_medoid))
        print(
            f"[ResidualSVD] mode={normalized_mode} records={effective_global_records} "
            f"non_medoid_layers={len(non_medoid)} mean_explained_energy={mean_explained:.6f}",
            flush=True,
        )
    return summary


@torch.no_grad()
def initialize_internal_weight_delta_svd(
    *,
    student: nn.Module,
    teacher: nn.Module,
    layer_to_proto: Sequence[int],
    seed_layer_for_proto: Dict[int, int],
    oversample: int,
    niter: int,
    seed: int,
) -> Dict[str, Any]:
    """Initialize private gate/up/down branches from projection-weight deltas."""
    model = unwrap_model(student)
    teacher_model = unwrap_model(teacher)
    student_layers = _resolve_layers(model)
    teacher_layers = _resolve_layers(teacher_model)
    rank_id = dist.get_rank() if dist.is_initialized() else 0
    stats: Dict[str, Any] = {}

    def factor_delta(
        delta: torch.Tensor,
        a_module: nn.Linear,
        b_module: nn.Linear,
        *,
        local_seed: int,
    ) -> float:
        target_rank = min(
            int(a_module.out_features),
            int(delta.size(0)),
            int(delta.size(1)),
        )
        if target_rank <= 0 or float(delta.float().square().sum().item()) <= 1e-20:
            a_module.weight.zero_()
            b_module.weight.zero_()
            return 1.0
        q = min(
            int(delta.size(0)),
            int(delta.size(1)),
            target_rank + max(2, int(oversample)),
        )
        torch.manual_seed(int(local_seed))
        if delta.device.type == "cuda":
            torch.cuda.manual_seed_all(int(local_seed))
        u, singular, v = torch.svd_lowrank(
            delta.float(),
            q=q,
            niter=max(1, int(niter)),
        )
        singular = singular[:target_rank].clamp(min=0.0)
        u = u[:, :target_rank]
        v = v[:, :target_rank]
        sqrt_s = singular.sqrt()
        b_weight = u * sqrt_s.view(1, -1)
        a_weight = sqrt_s.view(-1, 1) * v.transpose(0, 1)
        b_module.weight.copy_(b_weight.to(dtype=b_module.weight.dtype))
        a_module.weight.copy_(a_weight.to(dtype=a_module.weight.dtype))
        total = delta.float().square().sum().clamp(min=1e-12)
        return float((singular.square().sum() / total).clamp(max=1.0).item())

    for layer_id, student_layer in enumerate(student_layers):
        adapter = student_layer.mlp
        if not isinstance(adapter, SharedMLPAdapter):
            raise RuntimeError(f"layer {layer_id} is not SharedMLPAdapter")
        if adapter.sharing_parameterization != "internal_weight_delta":
            continue
        proto_id = int(layer_to_proto[layer_id])
        source_layer = int(seed_layer_for_proto[proto_id])
        if adapter.internal_rank <= 0:
            stats[str(layer_id)] = {
                "source_layer": source_layer,
                "private_singleton": True,
                "rank": 0,
            }
            continue
        if rank_id == 0:
            source_mlp = teacher_layers[source_layer].mlp
            target_mlp = teacher_layers[layer_id].mlp
            projection_stats = {}
            for offset, (name, a_module, b_module) in enumerate(
                (
                    ("gate_proj", adapter.delta_gate_a, adapter.delta_gate_b),
                    ("up_proj", adapter.delta_up_a, adapter.delta_up_b),
                    ("down_proj", adapter.delta_down_a, adapter.delta_down_b),
                )
            ):
                delta = (
                    getattr(target_mlp, name).weight.detach()
                    - getattr(source_mlp, name).weight.detach()
                )
                projection_stats[name] = factor_delta(
                    delta,
                    a_module,
                    b_module,
                    local_seed=int(seed) + 1009 * layer_id + 97 * offset,
                )
            if adapter.scale is not None:
                adapter.scale.fill_(1.0)
            stats[str(layer_id)] = {
                "source_layer": source_layer,
                "private_singleton": False,
                "medoid_zero_delta": layer_id == source_layer,
                "rank": int(adapter.internal_rank),
                "explained_weight_energy": projection_stats,
            }
        if dist.is_initialized():
            for parameter in adapter.parameters():
                dist.broadcast(parameter.data, src=0)
    if dist.is_initialized():
        payload: List[Any] = [stats if rank_id == 0 else None]
        dist.broadcast_object_list(payload, src=0)
        stats = payload[0]
        dist.barrier()
    return {
        "enabled": True,
        "mode": "internal_weight_delta",
        "oversample": int(oversample),
        "niter": int(niter),
        "layers": stats,
    }


def freeze_backbone_for_shared_train(model: nn.Module) -> None:
    model = unwrap_model(model)
    for _, param in model.named_parameters():
        param.requires_grad_(False)
    root = getattr(model, "model", model)
    bank = getattr(root, "shared_mlp_bank", None)
    if bank is not None:
        for param in bank.parameters():
            param.requires_grad_(True)
    for layer in _resolve_layers(model):
        if isinstance(layer.mlp, SharedMLPAdapter):
            for name, param in layer.mlp.named_parameters():
                # down-only sharing keeps the layer-native gate/up routing
                # exactly private; those large matrices are saved but not
                # optimized during recovery.
                trainable = not (
                    name.startswith("private_gate_proj.")
                    or name.startswith("private_up_proj.")
                )
                param.requires_grad_(trainable)


def set_shared_bank_trainable(model: nn.Module, enabled: bool) -> None:
    model = unwrap_model(model)
    root = getattr(model, "model", model)
    bank = getattr(root, "shared_mlp_bank", None)
    if bank is not None:
        for param in bank.parameters():
            param.requires_grad_(bool(enabled))


def gather_layer_samples(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    eos_token_id: Optional[int],
    token_rule: str,
    window_size: int,
    window_sample_mode: str,
    window_random_pick_min: int,
    window_random_pick_max: int,
    max_batches: int,
    reservoir_size: int,
    seed: int,
    checkpoint_every_batches: int,
    checkpoint_dir: Optional[str],
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], Dict[str, torch.Tensor]]:
    layers = _resolve_layers(model)
    layer_count = int(len(layers))
    hidden_size = int(getattr(model.config, "hidden_size", 0) or 0)
    if hidden_size <= 0:
        raise RuntimeError("Invalid hidden_size.")
    delta_reservoirs = [Reservoir(reservoir_size, hidden_size, seed + 104729 * (i + 1)) for i in range(layer_count)]
    ctx_reservoirs = [Reservoir(reservoir_size, hidden_size, seed + 130363 * (i + 1)) for i in range(layer_count)]
    pre_ffn_reservoirs = [Reservoir(reservoir_size, hidden_size, seed + 150013 * (i + 1)) for i in range(layer_count)]
    hidden_sum = torch.zeros((layer_count, hidden_size), dtype=torch.float64)
    pre_ffn_sum = torch.zeros((layer_count, hidden_size), dtype=torch.float64)
    attention_delta_sum = torch.zeros((layer_count, hidden_size), dtype=torch.float64)
    count_sum = torch.zeros((layer_count,), dtype=torch.float64)
    h2_sum = torch.zeros((layer_count, hidden_size), dtype=torch.float64)
    h3_sum = torch.zeros((layer_count, hidden_size), dtype=torch.float64)
    h2_count = torch.zeros((layer_count,), dtype=torch.float64)
    h3_count = torch.zeros((layer_count,), dtype=torch.float64)

    total_batches = len(loader)
    if int(max_batches) > 0:
        total_batches = min(int(total_batches), int(max_batches))
    loader_iter = iter_progress(
        loader,
        total=total_batches,
        desc="[Atlas] collect Δh/ctx",
    )
    if checkpoint_dir:
        ensure_dir(checkpoint_dir)
    for batch_idx, batch in enumerate(loader_iter):
        if int(max_batches) > 0 and int(batch_idx) >= int(max_batches):
            break
        step = int(batch_idx) + 1
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        batch_idx_tensor, token_idx_tensor = select_token_positions(
            input_ids=input_ids,
            attention_mask=attention_mask,
            eos_token_id=eos_token_id,
            token_rule=token_rule,
            window_size=window_size,
            window_sample_mode=window_sample_mode,
            window_random_pick_min=window_random_pick_min,
            window_random_pick_max=window_random_pick_max,
            prompt_lens=batch.get("prompt_lens", None),
        )
        if batch_idx_tensor.numel() <= 0:
            continue
        _, hidden_selected, mlp_selected, pre_ffn_selected, _ = _forward_with_selected_capture(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            batch_indices=batch_idx_tensor,
            token_indices=token_idx_tensor,
            capture_hidden=True,
            capture_mlp_layer_ids=list(range(layer_count)),
            capture_pre_ffn_input_layer_ids=list(range(layer_count)),
        )
        if hidden_selected is None or len(hidden_selected) < layer_count + 1:
            continue
        for layer_id in range(layer_count):
            h_l = hidden_selected[layer_id]
            delta_ffn = mlp_selected.get(int(layer_id))
            pre_ffn = pre_ffn_selected.get(int(layer_id))
            if h_l is None or delta_ffn is None or pre_ffn is None:
                continue
            ctx_reservoirs[layer_id].add(h_l.detach().to(dtype=torch.float32))
            delta_reservoirs[layer_id].add(delta_ffn.detach().to(dtype=torch.float32))
            pre_ffn_reservoirs[layer_id].add(pre_ffn.detach().to(dtype=torch.float32))
            hidden_sum[layer_id] += h_l.detach().to(dtype=torch.float64).sum(dim=0).cpu()
            pre_ffn_sum[layer_id] += pre_ffn.detach().to(dtype=torch.float64).sum(dim=0).cpu()
            attention_delta_sum[layer_id] += (pre_ffn.detach().to(dtype=torch.float64) - h_l.detach().to(dtype=torch.float64)).sum(dim=0).cpu()
            count_sum[layer_id] += float(h_l.size(0))
            if layer_id + 2 <= layer_count and hidden_selected[layer_id + 2] is not None:
                h2_delta = hidden_selected[layer_id + 2].detach().to(dtype=torch.float64) - h_l.detach().to(dtype=torch.float64)
                h2_sum[layer_id] += h2_delta.sum(dim=0).cpu()
                h2_count[layer_id] += float(h2_delta.size(0))
            if layer_id + 3 <= layer_count and hidden_selected[layer_id + 3] is not None:
                h3_delta = hidden_selected[layer_id + 3].detach().to(dtype=torch.float64) - h_l.detach().to(dtype=torch.float64)
                h3_sum[layer_id] += h3_delta.sum(dim=0).cpu()
                h3_count[layer_id] += float(h3_delta.size(0))
        if (
            checkpoint_dir
            and int(checkpoint_every_batches) > 0
            and step % int(checkpoint_every_batches) == 0
        ):
            checkpoint_path = os.path.join(checkpoint_dir, f"atlas_collect_step_{step}.pt")
            layer_stats = []
            for layer_id in range(layer_count):
                layer_stats.append(
                    {
                        "layer_id": int(layer_id),
                        "delta_fill": int(delta_reservoirs[layer_id].fill),
                        "delta_seen": int(delta_reservoirs[layer_id].seen),
                        "ctx_fill": int(ctx_reservoirs[layer_id].fill),
                        "ctx_seen": int(ctx_reservoirs[layer_id].seen),
                        "pre_ffn_fill": int(pre_ffn_reservoirs[layer_id].fill),
                        "pre_ffn_seen": int(pre_ffn_reservoirs[layer_id].seen),
                    }
                )
            torch.save(
                {
                    "phase": "atlas_collect",
                    "step": int(step),
                    "layer_count": int(layer_count),
                    "reservoir_size": int(reservoir_size),
                    "layer_stats": layer_stats,
                },
                checkpoint_path,
            )
            print(f"[Atlas] checkpoint saved {checkpoint_path}", flush=True)

    delta_samples = [item.tensor() for item in delta_reservoirs]
    ctx_samples = [item.tensor() for item in ctx_reservoirs]
    pre_ffn_samples = [item.tensor() for item in pre_ffn_reservoirs]

    def _safe_mean(sum_tensor: torch.Tensor, count_tensor: torch.Tensor) -> torch.Tensor:
        denom = count_tensor.clamp(min=1.0).view(-1, 1)
        return (sum_tensor / denom).to(dtype=torch.float32)

    aux = {
        "hidden_mean": _safe_mean(hidden_sum, count_sum),
        "pre_ffn_mean": _safe_mean(pre_ffn_sum, count_sum),
        "attention_delta_mean": _safe_mean(attention_delta_sum, count_sum),
        "sample_count": count_sum.to(dtype=torch.int64),
        "h2_mean": _safe_mean(h2_sum, h2_count),
        "h3_mean": _safe_mean(h3_sum, h3_count),
        "h2_count": h2_count.to(dtype=torch.int64),
        "h3_count": h3_count.to(dtype=torch.int64),
    }
    return delta_samples, ctx_samples, pre_ffn_samples, aux


def _sample_rows_for_merge(tensor: torch.Tensor, max_rows: int, seed: int) -> torch.Tensor:
    tensor = tensor.detach().to(dtype=torch.float32, device="cpu").contiguous()
    n_rows = int(tensor.size(0)) if tensor.dim() > 0 else 0
    if n_rows <= max(0, int(max_rows)):
        return tensor
    generator = torch.Generator(device=torch.device("cpu"))
    generator.manual_seed(int(seed))
    keep = torch.randperm(n_rows, generator=generator)[: int(max_rows)]
    keep, _ = torch.sort(keep)
    return tensor.index_select(0, keep).contiguous()


def _merge_mean_from_rank_payloads(
    payloads: Sequence[Dict[str, Any]],
    *,
    mean_key: str,
    count_key: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    first_aux = payloads[0]["aux_stats"]
    first_mean = first_aux[mean_key].to(dtype=torch.float64, device="cpu")
    total_sum = torch.zeros_like(first_mean, dtype=torch.float64)
    total_count = torch.zeros_like(first_aux[count_key].to(dtype=torch.float64, device="cpu"))
    for payload in payloads:
        aux = payload["aux_stats"]
        count = aux[count_key].to(dtype=torch.float64, device="cpu")
        mean = aux[mean_key].to(dtype=torch.float64, device="cpu")
        total_sum += mean * count.clamp(min=0.0).view(-1, 1)
        total_count += count
    merged_mean = total_sum / total_count.clamp(min=1.0).view(-1, 1)
    return merged_mean.to(dtype=torch.float32), total_count.to(dtype=torch.int64)


def _merge_atlas_rank_payloads(
    payloads: Sequence[Dict[str, Any]],
    *,
    reservoir_size: int,
    seed: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], Dict[str, torch.Tensor], int]:
    if not payloads:
        raise RuntimeError("No atlas rank payloads to merge.")
    layer_count = len(payloads[0]["delta_samples"])
    merged: Dict[str, List[torch.Tensor]] = {
        "delta_samples": [],
        "ctx_samples": [],
        "pre_ffn_samples": [],
    }
    key_seed_offset = {
        "delta_samples": 104729,
        "ctx_samples": 130363,
        "pre_ffn_samples": 150013,
    }
    for key in ("delta_samples", "ctx_samples", "pre_ffn_samples"):
        for layer_id in range(layer_count):
            chunks = [
                payload[key][layer_id].detach().to(dtype=torch.float32, device="cpu")
                for payload in payloads
                if torch.is_tensor(payload[key][layer_id]) and int(payload[key][layer_id].numel()) > 0
            ]
            if chunks:
                combined = torch.cat(chunks, dim=0).contiguous()
            else:
                hidden_size = int(payloads[0].get("hidden_size", 0))
                combined = torch.empty((0, hidden_size), dtype=torch.float32)
            merged[key].append(
                _sample_rows_for_merge(
                    combined,
                    max_rows=int(reservoir_size),
                    seed=int(seed) + int(key_seed_offset[key]) * (int(layer_id) + 1),
                )
            )

    hidden_mean, sample_count = _merge_mean_from_rank_payloads(
        payloads,
        mean_key="hidden_mean",
        count_key="sample_count",
    )
    pre_ffn_mean, _ = _merge_mean_from_rank_payloads(
        payloads,
        mean_key="pre_ffn_mean",
        count_key="sample_count",
    )
    attention_delta_mean, _ = _merge_mean_from_rank_payloads(
        payloads,
        mean_key="attention_delta_mean",
        count_key="sample_count",
    )
    h2_mean, h2_count = _merge_mean_from_rank_payloads(
        payloads,
        mean_key="h2_mean",
        count_key="h2_count",
    )
    h3_mean, h3_count = _merge_mean_from_rank_payloads(
        payloads,
        mean_key="h3_mean",
        count_key="h3_count",
    )
    aux = {
        "hidden_mean": hidden_mean,
        "pre_ffn_mean": pre_ffn_mean,
        "attention_delta_mean": attention_delta_mean,
        "sample_count": sample_count,
        "h2_mean": h2_mean,
        "h3_mean": h3_mean,
        "h2_count": h2_count,
        "h3_count": h3_count,
    }
    total_tokenized = max(int(payload.get("total_tokenized", 0)) for payload in payloads)
    return (
        merged["delta_samples"],
        merged["ctx_samples"],
        merged["pre_ffn_samples"],
        aux,
        int(total_tokenized),
    )

def stage_atlas(args: argparse.Namespace) -> Dict[str, Any]:
    set_seed(int(args.seed))
    dist_ctx = init_distributed(str(args.device))
    output_dir = str(args.output_dir or f"./out/newthesis_atlas_{now_tag()}")
    if is_main_process(dist_ctx):
        ensure_dir(output_dir)
    dist_barrier(dist_ctx)
    device = resolve_device(args.device, dist_ctx.local_rank if dist_ctx.enabled else -1)
    dtype = get_target_dtype(device)

    teacher_deploy_bundle = str(getattr(args, "teacher_deploy_bundle", "")).strip()
    if teacher_deploy_bundle:
        teacher_deploy_bundle = os.path.abspath(teacher_deploy_bundle)
        teacher_bundle = torch.load(teacher_deploy_bundle, map_location="cpu", weights_only=False)
        if not isinstance(teacher_bundle, dict) or "shared_student" not in teacher_bundle or "base_model" not in teacher_bundle:
            raise ValueError("--teacher_deploy_bundle is not a valid Phase-1.5 deploy bundle")
        teacher, _ = _build_shared_model_for_eval(
            base_model=str(teacher_bundle["base_model"]),
            atlas_payload=teacher_bundle.get("atlas", {}),
            shared_payload=teacher_bundle["shared_student"],
            quant_bank_int4=teacher_bundle.get("quant_bank_int4"),
            use_quant_bank_int4=False,
            device=device,
            dtype=dtype,
            trust_remote_code=bool(getattr(args, "trust_remote_code", False)),
        )
        print(f"[Atlas] teacher_loader=shared_deploy_bundle path={teacher_deploy_bundle}", flush=True)
    else:
        teacher = build_frozen_teacher(
            teacher_ckpt_dir=args.teacher_ckpt,
            base_model=args.base_model,
            teacher_loader=args.teacher_loader,
            target_dtype=dtype,
            trust_remote_code=bool(getattr(args, "trust_remote_code", False)),
        )
        teacher.to(device)
    _set_gradient_checkpointing(
        teacher,
        enabled=bool(getattr(args, "teacher_gradient_checkpointing", False)),
        require_input_grads=False,
    )
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    _validate_supported_llama_config(teacher, label="atlas teacher")

    tokenizer_src = str(args.tokenizer_name_or_path).strip() or str(args.base_model).strip() or str(args.teacher_ckpt).strip()
    tokenizer = load_tokenizer(
        tokenizer_src,
        trust_remote_code=bool(getattr(args, "trust_remote_code", False)),
    )
    loader, total_tokenized = load_tokenized_loader(
        tokenizer=tokenizer,
        data_path=args.data_path,
        max_records=int(args.max_records),
        cutoff_len=int(args.cutoff_len),
        batch_size=int(args.batch_size),
        seed=int(args.seed),
        shuffle_records=bool(args.shuffle_records),
        distributed_num_replicas=dist_ctx.world_size if dist_ctx.enabled else 1,
        distributed_rank=dist_ctx.rank if dist_ctx.enabled else 0,
        prompt_mode=str(getattr(args, "training_prompt_mode", "legacy_sft")),
    )
    if is_main_process(dist_ctx):
        print(
            f"[Atlas] tokenized={total_tokenized} batch_size_per_rank={int(args.batch_size)} "
            f"world_size={int(dist_ctx.world_size)} max_batches={int(args.max_batches)}",
            flush=True,
        )

    eos_token_id = tokenizer.eos_token_id
    if is_main_process(dist_ctx):
        print("[Atlas] collecting hidden states...", flush=True)
    atlas_ckpt_dir = os.path.join(output_dir, "checkpoints")
    rank_atlas_ckpt_dir = (
        os.path.join(atlas_ckpt_dir, f"rank{int(dist_ctx.rank)}")
        if dist_ctx.enabled
        else atlas_ckpt_dir
    )
    per_rank_reservoir_size = int(args.reservoir_size)
    if dist_ctx.enabled:
        per_rank_reservoir_size = max(1, int(math.ceil(float(args.reservoir_size) / float(dist_ctx.world_size))))
    per_rank_max_batches = int(args.max_batches)
    if dist_ctx.enabled and int(args.max_batches) > 0:
        per_rank_max_batches = max(1, int(math.ceil(float(args.max_batches) / float(dist_ctx.world_size))))
    delta_samples, ctx_samples, pre_ffn_samples, aux_stats = gather_layer_samples(
        model=teacher,
        loader=loader,
        device=device,
        eos_token_id=eos_token_id,
        token_rule=str(args.token_rule),
        window_size=int(args.window_size),
        window_sample_mode=str(args.window_sample_mode),
        window_random_pick_min=int(args.window_random_pick_min),
        window_random_pick_max=int(args.window_random_pick_max),
        max_batches=per_rank_max_batches,
        reservoir_size=per_rank_reservoir_size,
        seed=int(args.seed) + int(dist_ctx.rank) * 1000003,
        checkpoint_every_batches=int(args.ckpt_every_batches),
        checkpoint_dir=rank_atlas_ckpt_dir,
    )
    if dist_ctx.enabled:
        rank_payload_dir = os.path.join(output_dir, "rank_samples")
        if is_main_process(dist_ctx):
            ensure_dir(rank_payload_dir)
        dist_barrier(dist_ctx)
        rank_payload_path = os.path.join(rank_payload_dir, f"rank_{int(dist_ctx.rank)}.pt")
        torch.save(
            {
                "rank": int(dist_ctx.rank),
                "world_size": int(dist_ctx.world_size),
                "hidden_size": int(getattr(teacher.config, "hidden_size", 0) or 0),
                "total_tokenized": int(total_tokenized),
                "delta_samples": [tensor.cpu() for tensor in delta_samples],
                "ctx_samples": [tensor.cpu() for tensor in ctx_samples],
                "pre_ffn_samples": [tensor.cpu() for tensor in pre_ffn_samples],
                "aux_stats": {key: value.cpu() for key, value in aux_stats.items()},
            },
            rank_payload_path,
        )
        dist_barrier(dist_ctx)
        if not is_main_process(dist_ctx):
            dist_barrier(dist_ctx)
            finalize_distributed(dist_ctx)
            return {}
        payloads = [
            torch.load(os.path.join(rank_payload_dir, f"rank_{rank}.pt"), map_location="cpu")
            for rank in range(int(dist_ctx.world_size))
        ]
        delta_samples, ctx_samples, pre_ffn_samples, aux_stats, total_tokenized = _merge_atlas_rank_payloads(
            payloads,
            reservoir_size=int(args.reservoir_size),
            seed=int(args.seed),
        )
        print(
            f"[Atlas] merged rank samples world_size={int(dist_ctx.world_size)} "
            f"per_rank_reservoir={int(per_rank_reservoir_size)} target_reservoir={int(args.reservoir_size)}",
            flush=True,
        )
        shutil.rmtree(rank_payload_dir, ignore_errors=True)
    print("[Atlas] fitting shared chart + learned codebook...", flush=True)

    layer_count = len(delta_samples)
    hidden_size = int(getattr(teacher.config, "hidden_size", 0) or 0)
    if hidden_size <= 0:
        raise RuntimeError("Invalid hidden_size.")
    non_empty_delta = [x for x in delta_samples if x.numel() > 0]
    total_delta = sum(int(x.size(0)) for x in non_empty_delta)
    if total_delta < 8:
        raise RuntimeError("Too few delta samples for atlas.")
    rank = min(int(args.analysis_rank), int(total_delta - 1), int(hidden_size))
    if rank <= 0:
        raise RuntimeError("Invalid analysis_rank for current sample count.")
    pca_mode = str(getattr(args, "pca_mode", "stream_sketch")).strip().lower()
    projection_basis_source = str(getattr(args, "projection_basis_source", "velocity_pca")).strip().lower()
    if projection_basis_source in {"velocity", "teacher_velocity_pca", "delta", "delta_pca"}:
        projection_basis_source = "velocity_pca"
    elif projection_basis_source in {"hidden", "teacher_hidden_pca", "hidden_state_pca", "pre_ffn_pca"}:
        projection_basis_source = "hidden_pca"
    elif projection_basis_source in {"random", "random_projection"}:
        projection_basis_source = "random_orthogonal"
    projection_basis_samples = pre_ffn_samples if projection_basis_source == "hidden_pca" else delta_samples
    random_basis_seed = int(getattr(args, "random_basis_seed", 0))
    pca_t0 = time.perf_counter()
    basis, pca_stats = _fit_projection_basis(
        samples=projection_basis_samples,
        source=projection_basis_source,
        rank=int(rank),
        hidden_size=int(hidden_size),
        pca_mode=pca_mode,
        oversample=max(8, int(args.pca_oversample)),
        power_iter=max(0, int(args.pca_niter)),
        chunk_size=int(args.pca_stream_chunk_size),
        seed=int(args.seed),
        random_basis_seed=random_basis_seed,
        device=str(args.pca_device),
    )
    pca_elapsed = time.perf_counter() - pca_t0
    print(
        f"[Atlas] projection_basis_source={projection_basis_source} PCA mode={pca_stats['mode']} "
        f"rank={rank} q={pca_stats['q']} device={pca_stats['device']} "
        f"samples={pca_stats['count']} time={pca_elapsed:.1f}s",
        flush=True,
    )

    predictor_ctx_source = str(PREDICTOR_CTX_SOURCE)
    if predictor_ctx_source != "B_T_h_l":
        raise ValueError(f"Unsupported predictor context source: {predictor_ctx_source}")

    layer_hist = torch.zeros((layer_count, int(args.num_codes)), dtype=torch.float32)
    all_z_ctx: List[torch.Tensor] = []
    all_z_tgt: List[torch.Tensor] = []
    all_layer: List[torch.Tensor] = []

    for layer_id in range(layer_count):
        if delta_samples[layer_id].numel() <= 0:
            continue
        z_delta = torch.matmul(delta_samples[layer_id], basis)
        z_ctx = torch.matmul(ctx_samples[layer_id], basis)
        all_z_tgt.append(z_delta)
        all_z_ctx.append(z_ctx)
        all_layer.append(torch.full((z_delta.size(0),), layer_id, dtype=torch.long))

    z_tgt_all = torch.cat(all_z_tgt, dim=0)
    z_ctx_all = torch.cat(all_z_ctx, dim=0)
    layer_ids_all = torch.cat(all_layer, dim=0)
    print(
        f"[Atlas] hash_mode={HASH_LEARNING_MODE} neural_hash={HASH_IS_E2E_NEURAL} z_ctx_source={predictor_ctx_source}",
        flush=True,
    )
    kmeans_mode = str(getattr(args, "kmeans_mode", "minibatch")).strip().lower()
    kmeans_t0 = time.perf_counter()
    centers, codes = fit_kmeans_with_mode(
        z_tgt_all,
        num_clusters=int(args.num_codes),
        iters=int(args.kmeans_iters),
        seed=int(args.seed),
        mode=kmeans_mode,
        batch_size=int(args.kmeans_batch_size),
        warmup_size=int(args.kmeans_warmup_size),
        warmup_iters=int(args.kmeans_warmup_iters),
        refine_iters=int(args.kmeans_refine_iters),
        assign_chunk_size=int(args.kmeans_assign_chunk_size),
        device=str(args.kmeans_device),
    )
    kmeans_elapsed = time.perf_counter() - kmeans_t0
    print(
        f"[Atlas] kmeans mode={kmeans_mode} centers={int(centers.size(0))} "
        f"warmup={int(args.kmeans_warmup_size)} warmup_iters={int(args.kmeans_warmup_iters)} "
        f"refine_iters={int(args.kmeans_refine_iters)} batch={int(args.kmeans_batch_size)} "
        f"time={kmeans_elapsed:.1f}s",
        flush=True,
    )
    code_count = int(centers.size(0))
    tau_diag = torch.zeros((code_count, rank), dtype=torch.float32)
    code_usage = torch.zeros((code_count,), dtype=torch.long)
    tau_nmin = max(1, int(getattr(args, "tau_nmin", 50)))
    tau_shrink_lambda = max(0.0, float(getattr(args, "tau_shrink_lambda", 0.1)))
    tau_eps = max(1e-12, float(getattr(args, "tau_eps", 1e-5)))
    tau_min_var = max(0.0, float(args.tau_min_var))
    global_var = torch.clamp(z_tgt_all.var(dim=0, unbiased=False), min=tau_min_var) + tau_eps
    for code_id in range(code_count):
        mask = codes == code_id
        count = int(mask.sum().item())
        code_usage[code_id] = count
        if bool(mask.any().item()):
            chunk = z_tgt_all[mask]
            diff = chunk - centers[code_id].view(1, -1)
            var_code = (diff * diff).mean(dim=0)
        else:
            var_code = global_var
        if count < tau_nmin:
            stabilized = global_var
        else:
            shrink_ratio = min(1.0, tau_shrink_lambda * (float(tau_nmin) / float(max(1, count))))
            stabilized = (1.0 - shrink_ratio) * var_code + shrink_ratio * global_var
        tau_diag[code_id] = torch.clamp(stabilized, min=tau_min_var) + tau_eps

    code_centers_for_samples = centers.index_select(0, codes)
    z_code_residual = z_tgt_all - code_centers_for_samples
    layer_signature_distribution = torch.zeros((layer_count, rank + code_count), dtype=torch.float32)
    layer_signature_error = torch.zeros((layer_count, 1), dtype=torch.float32)
    layer_signature = torch.zeros((layer_count, rank + code_count + 1), dtype=torch.float32)
    layer_error_norm = torch.zeros((layer_count,), dtype=torch.float32)
    for layer_id in range(layer_count):
        mask_layer = layer_ids_all == layer_id
        if not bool(mask_layer.any().item()):
            continue
        layer_codes = codes[mask_layer]
        hist = torch.bincount(layer_codes, minlength=code_count).to(dtype=torch.float32)
        hist = hist / max(1.0, float(hist.sum().item()))
        layer_hist[layer_id, :code_count] = hist
        layer_mean_ctx = z_ctx_all[mask_layer].mean(dim=0)
        layer_signature_distribution[layer_id, :rank] = layer_mean_ctx
        layer_signature_distribution[layer_id, rank : rank + code_count] = hist
        layer_residual = z_code_residual[mask_layer]
        layer_error_norm[layer_id] = torch.sqrt((layer_residual * layer_residual).mean(dim=1).mean().clamp(min=1e-12))
        layer_signature_error[layer_id, 0] = layer_error_norm[layer_id]
        layer_signature[layer_id, : rank + code_count] = layer_signature_distribution[layer_id]
        layer_signature[layer_id, rank + code_count :] = layer_signature_error[layer_id]

    alpha_layer, alpha_code, alpha_global = build_alpha_layer_topk_mixture(
        tau_diag=tau_diag,
        layer_hist=layer_hist[:, :code_count],
        tau_topk=int(getattr(args, "tau_topk", 3)),
        tau_eps=tau_eps,
    )
    layer_hist_entropy = compute_layer_hist_entropy(layer_hist[:, :code_count], eps=tau_eps)
    layer_reliability = _compute_layer_reliability(
        layer_hist_entropy=layer_hist_entropy,
        layer_error_norm=layer_error_norm,
    )
    usage_ratio = code_usage.to(dtype=torch.float32) / float(max(1, int(code_usage.sum().item())))
    effective_code_count = int((usage_ratio > 0.005).sum().item())

    regime_labels = _build_regime_labels(layer_count)
    regime_basis_map, regime_basis_stats = _fit_regime_basis_map(
        basis_samples=projection_basis_samples,
        regime_labels=regime_labels,
        rank=int(rank),
        hidden_size=int(hidden_size),
        basis_source=projection_basis_source,
        oversample=max(8, int(args.pca_oversample)),
        power_iter=max(0, int(args.pca_niter)),
        chunk_size=int(args.pca_stream_chunk_size),
        seed=int(args.seed),
        random_basis_seed=random_basis_seed,
        pca_mode=pca_mode,
        device=str(args.pca_device),
        fallback_basis=basis,
    )
    regime_metric_global: Dict[str, torch.Tensor] = {}
    for regime_name, regime_basis in regime_basis_map.items():
        pooled = [
            torch.matmul(delta_samples[layer_id], regime_basis)
            for layer_id in range(layer_count)
            if str(regime_labels[layer_id]) == regime_name and int(delta_samples[layer_id].numel()) > 0
        ]
        if pooled:
            pooled_cat = torch.cat(pooled, dim=0)
            regime_metric_global[regime_name] = torch.clamp(
                pooled_cat.var(dim=0, unbiased=False),
                min=tau_eps,
            ) + tau_eps
        else:
            regime_metric_global[regime_name] = torch.ones((rank,), dtype=torch.float32)
    layer_metric_diag = torch.zeros((layer_count, rank), dtype=torch.float32)
    for layer_id in range(layer_count):
        regime_name = str(regime_labels[layer_id])
        regime_basis = regime_basis_map.get(regime_name, basis)
        if int(delta_samples[layer_id].numel()) > 0 and int(delta_samples[layer_id].size(0)) >= 2:
            z_layer = torch.matmul(delta_samples[layer_id], regime_basis)
            layer_metric_diag[layer_id] = torch.clamp(
                z_layer.var(dim=0, unbiased=False),
                min=tau_eps,
            ) + tau_eps
        else:
            layer_metric_diag[layer_id] = regime_metric_global.get(regime_name, torch.ones((rank,), dtype=torch.float32))

    upstream_basis, _, upstream_stats = fit_streaming_sketch_pca(
        pre_ffn_samples,
        rank=int(rank),
        oversample=max(8, int(args.pca_oversample)),
        power_iter=max(0, int(args.pca_niter)),
        chunk_size=int(args.pca_stream_chunk_size),
        seed=int(args.seed) + 123,
        device=str(args.pca_device),
    )
    upstream_mean = aux_stats["pre_ffn_mean"].to(dtype=torch.float32)
    attention_delta_mean = aux_stats["attention_delta_mean"].to(dtype=torch.float32)
    h2_mean = aux_stats["h2_mean"].to(dtype=torch.float32)
    h3_mean = aux_stats["h3_mean"].to(dtype=torch.float32)
    sharing_policy = _build_sharing_policy(
        upstream_mean=upstream_mean,
        attention_delta_mean=attention_delta_mean,
        h2_mean=h2_mean,
        h3_mean=h3_mean,
        regime_labels=regime_labels,
        layer_reliability=layer_reliability,
        sharing_policy_mode=str(getattr(args, "sharing_policy_mode", "upstream_only")),
        upstream_threshold=float(getattr(args, "upstream_similarity_threshold", 0.95)),
    )
    sharing_policy["regime_labels"] = list(regime_labels)
    sharing_policy["upstream_basis_rank"] = int(upstream_basis.size(1))
    sharing_policy["upstream_basis_sample_count"] = int(upstream_stats["count"])
    sharing_policy_path = os.path.join(output_dir, "sharing_policy.json")
    save_json(sharing_policy_path, sharing_policy)

    final_structure_prior = {
        "basis": basis.cpu(),
        "basis_source": projection_basis_source,
        "basis_stats": dict(pca_stats),
        "regime_basis": {name: tensor.cpu() for name, tensor in regime_basis_map.items()},
        "regime_basis_stats": regime_basis_stats,
        "layer_metric_diag": layer_metric_diag.cpu(),
        "layer_inv_std": alpha_layer.cpu(),
        "layer_reliability": layer_reliability.cpu(),
        "layer_to_regime": list(regime_labels),
        "tau_diag": tau_diag.cpu(),
        "tau_eps": float(tau_eps),
        "sharing_policy_path": sharing_policy_path,
    }
    final_structure_prior_path = os.path.join(output_dir, "final_structure_prior.pt")
    torch.save(final_structure_prior, final_structure_prior_path)

    prototype_count = _infer_prototype_count(layer_count)
    proto_centers, layer_proto = fit_kmeans_with_mode(
        layer_signature,
        num_clusters=int(prototype_count),
        iters=max(5, int(args.kmeans_iters)),
        seed=int(args.seed) + 17,
        mode="full",
        batch_size=4096,
        warmup_size=0,
        warmup_iters=max(5, int(args.kmeans_iters)),
        refine_iters=0,
        assign_chunk_size=max(256, int(args.kmeans_assign_chunk_size)),
        device="cpu",
    )

    cache_path = os.path.join(output_dir, "atlas_train_cache.pt")
    torch.save(
        {
            "z_ctx": z_ctx_all.cpu(),
            "z_tgt": z_tgt_all.cpu(),
            "code_ids": codes.cpu(),
            "layer_ids": layer_ids_all.cpu(),
            "z_ctx_source": predictor_ctx_source,
            "hash_learning_mode": HASH_LEARNING_MODE,
        },
        cache_path,
    )
    atlas_state = {
        "config": {
            "base_model": str(args.base_model),
            "teacher_ckpt": str(args.teacher_ckpt),
            "teacher_loader": str(args.teacher_loader),
            "data_path": str(args.data_path),
            "delta_type": "ffn_mlp_output",
            "token_rule": str(args.token_rule),
            "window_size": int(args.window_size),
            "window_sample_mode": str(args.window_sample_mode),
            "window_random_pick_min": int(args.window_random_pick_min),
            "window_random_pick_max": int(args.window_random_pick_max),
            "analysis_rank": int(rank),
            "num_codes": int(code_count),
            "prototype_count": int(proto_centers.size(0)),
            "reservoir_size": int(args.reservoir_size),
            "ckpt_every_batches": int(args.ckpt_every_batches),
            "pca_mode": str(pca_mode),
            "projection_basis_source": projection_basis_source,
            "random_basis_seed": int(random_basis_seed),
            "pca_oversample": int(args.pca_oversample),
            "pca_niter": int(args.pca_niter),
            "pca_stream_chunk_size": int(args.pca_stream_chunk_size),
            "pca_device": str(args.pca_device),
            "kmeans_mode": str(kmeans_mode),
            "kmeans_iters": int(args.kmeans_iters),
            "kmeans_batch_size": int(args.kmeans_batch_size),
            "kmeans_warmup_size": int(args.kmeans_warmup_size),
            "kmeans_warmup_iters": int(args.kmeans_warmup_iters),
            "kmeans_refine_iters": int(args.kmeans_refine_iters),
            "kmeans_assign_chunk_size": int(args.kmeans_assign_chunk_size),
            "kmeans_device": str(args.kmeans_device),
            "tau_topk": int(getattr(args, "tau_topk", 3)),
            "tau_eps": float(tau_eps),
            "tau_nmin": int(tau_nmin),
            "tau_shrink_lambda": float(tau_shrink_lambda),
            "tau_min_var": float(tau_min_var),
            "tau_diag_semantics": "variance",
            "alpha_mode": "inv_std",
            "hash_learning_mode": HASH_LEARNING_MODE,
            "hash_is_e2e_neural": bool(HASH_IS_E2E_NEURAL),
            "predictor_ctx_source": predictor_ctx_source,
        },
        "basis": basis,
        "basis_var": basis,
        "basis_source": projection_basis_source,
        "basis_stats": dict(pca_stats),
        "code_centers": centers.cpu(),
        "tau_diag": tau_diag.cpu(),
        "alpha_layer": alpha_layer.cpu(),
        "alpha_code": alpha_code.cpu(),
        "alpha_global": alpha_global.cpu(),
        "tau_global": global_var.cpu(),
        "layer_code_hist": layer_hist[:, :code_count].cpu(),
        "layer_signature_distribution": layer_signature_distribution.cpu(),
        "layer_signature_error": layer_signature_error.cpu(),
        "layer_signature": layer_signature.cpu(),
        "layer_error_norm": layer_error_norm.cpu(),
        "layer_to_proto": layer_proto.cpu(),
        "proto_centers": proto_centers.cpu(),
        "cache_path": cache_path,
    }
    atlas_path = os.path.join(output_dir, "atlas_state.pt")
    torch.save(atlas_state, atlas_path)
    report = {
        "atlas_path": atlas_path,
        "final_structure_prior_path": final_structure_prior_path,
        "sharing_policy_path": sharing_policy_path,
        "cache_path": cache_path,
        "total_tokenized": int(total_tokenized),
        "layer_count": int(layer_count),
        "hidden_size": int(hidden_size),
        "analysis_rank": int(rank),
        "num_codes": int(code_count),
        "prototype_count": int(proto_centers.size(0)),
        "ckpt_every_batches": int(args.ckpt_every_batches),
        "checkpoint_dir": atlas_ckpt_dir,
        "pca_mode": str(pca_mode),
        "pca_time_sec": float(pca_elapsed),
        "pca_q": int(pca_stats["q"]),
        "pca_device": str(pca_stats["device"]),
        "kmeans_mode": str(kmeans_mode),
        "kmeans_time_sec": float(kmeans_elapsed),
        "kmeans_batch_size": int(args.kmeans_batch_size),
        "kmeans_warmup_size": int(args.kmeans_warmup_size),
        "kmeans_warmup_iters": int(args.kmeans_warmup_iters),
        "kmeans_refine_iters": int(args.kmeans_refine_iters),
        "kmeans_assign_chunk_size": int(args.kmeans_assign_chunk_size),
        "kmeans_device": str(args.kmeans_device),
        "delta_type": "ffn_mlp_output",
        "regime_basis_stats": regime_basis_stats,
        "tau_topk": int(getattr(args, "tau_topk", 3)),
        "tau_eps": float(tau_eps),
        "tau_nmin": int(tau_nmin),
        "tau_shrink_lambda": float(tau_shrink_lambda),
        "tau_min_var": float(tau_min_var),
        "hash_learning_mode": HASH_LEARNING_MODE,
        "hash_is_e2e_neural": bool(HASH_IS_E2E_NEURAL),
        "predictor_ctx_source": predictor_ctx_source,
        "code_usage": [int(x) for x in code_usage.tolist()],
        "code_usage_ratio": [float(x) for x in usage_ratio.tolist()],
        "effective_code_count_gt_0p5pct": int(effective_code_count),
        "layer_hist_entropy": [float(x) for x in layer_hist_entropy.tolist()],
        "layer_hist_entropy_mean": float(layer_hist_entropy.mean().item()),
        "layer_hist_entropy_min": float(layer_hist_entropy.min().item()),
        "layer_hist_entropy_max": float(layer_hist_entropy.max().item()),
        "layer_error_norm_mean": float(layer_error_norm.mean().item()),
        "layer_error_norm_min": float(layer_error_norm.min().item()),
        "layer_error_norm_max": float(layer_error_norm.max().item()),
        "layer_reliability_mean": float(layer_reliability.mean().item()),
        "layer_reliability_min": float(layer_reliability.min().item()),
        "layer_reliability_max": float(layer_reliability.max().item()),
        "layer_to_proto": [int(x) for x in layer_proto.tolist()],
        "layer_to_regime": list(regime_labels),
    }
    report_path = os.path.join(output_dir, "atlas_report.json")
    save_json(report_path, report)
    if plt is not None:
        fig = plt.figure(figsize=(10, 3))
        ax = fig.add_subplot(1, 1, 1)
        ax.bar(list(range(code_count)), [int(x) for x in code_usage.tolist()])
        ax.set_title("Regime code usage")
        ax.set_xlabel("Code")
        ax.set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "code_usage.png"), dpi=160)
        plt.close(fig)
    print(f"[Atlas] saved {atlas_path}")
    print(f"[Atlas] saved {report_path}")
    if dist_ctx.enabled:
        dist_barrier(dist_ctx)
        finalize_distributed(dist_ctx)
    return report


def _ce_shift_loss(logits: torch.Tensor, input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=int(pad_token_id),
    )


def _normalize_distill_mode(value: str) -> str:
    text = str(value or "ce").strip().lower().replace("-", "_")
    text = text.replace("+", "_").replace("/", "_")
    if text in {"ce", "ce_only", "none"}:
        return "ce"
    if text in {"ce_kd", "kd_ce", "cekd", "kd"}:
        return "ce_kd"
    if text in {"sage_ib", "sage", "information_bottleneck", "selective_js", "ig_js"}:
        return "sage_ib"
    if text in {"ce_hidden_mse", "hidden_mse", "hidden", "ce_mse", "mse"}:
        return "ce_hidden_mse"
    raise ValueError(f"Unsupported distill mode: {value!r}. Use ce, ce_kd, sage_ib, or ce_hidden_mse.")


def _pretty_distill_mode(value: str) -> str:
    mode = _normalize_distill_mode(value)
    if mode == "ce_kd":
        return "CE+KD"
    if mode == "ce_hidden_mse":
        return "CE+hidden-MSE"
    if mode == "sage_ib":
        return "SAGE-IB"
    return "CE"


def _kd_shift_masked_token_mean(
    *,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    attention_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    temp = max(1e-6, float(temperature))
    s_shift = student_logits[:, :-1, :].contiguous()
    t_shift = teacher_logits[:, :-1, :].contiguous()
    if int(s_shift.size(-1)) != int(t_shift.size(-1)):
        vocab = min(int(s_shift.size(-1)), int(t_shift.size(-1)))
        s_shift = s_shift[..., :vocab].contiguous()
        t_shift = t_shift[..., :vocab].contiguous()
    valid = attention_mask[:, 1:].to(dtype=s_shift.dtype)
    s_log_prob = F.log_softmax(s_shift / temp, dim=-1)
    t_prob = F.softmax(t_shift / temp, dim=-1)
    kl_per_token = F.kl_div(s_log_prob, t_prob, reduction="none").sum(dim=-1)
    denom = valid.sum().clamp(min=1.0)
    kd = (kl_per_token * valid).sum() / denom
    return kd * (temp * temp)


def _ce_shift_per_example_loss(logits: torch.Tensor, input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    per_token = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=int(pad_token_id),
        reduction="none",
    ).view(shift_labels.size(0), shift_labels.size(1))
    valid = (shift_labels != int(pad_token_id)).to(dtype=per_token.dtype)
    denom = valid.sum(dim=1).clamp(min=1.0)
    return (per_token * valid).sum(dim=1) / denom

def _compute_layer_proto_weights(
    *,
    layer_to_proto: Sequence[int],
    layer_signatures: Optional[torch.Tensor],
    proto_centers: Optional[torch.Tensor],
    soft_assignment: bool,
    soft_assignment_temp: float,
    soft_assignment_topk: int = 0,
) -> torch.Tensor:
    layer_count = int(len(layer_to_proto))
    proto_count = max(1, len(sorted({int(x) for x in layer_to_proto})))
    hard = torch.zeros((layer_count, proto_count), dtype=torch.float32)
    for layer_idx, proto_id in enumerate(layer_to_proto):
        if 0 <= int(proto_id) < proto_count:
            hard[int(layer_idx), int(proto_id)] = 1.0
    if not bool(soft_assignment):
        return hard
    if not torch.is_tensor(layer_signatures) or not torch.is_tensor(proto_centers):
        return hard
    if layer_signatures.dim() != 2 or proto_centers.dim() != 2:
        return hard
    if int(layer_signatures.size(0)) != layer_count:
        return hard
    if int(proto_centers.size(0)) != proto_count:
        return hard
    temp = max(1e-6, float(soft_assignment_temp))
    dists = torch.cdist(
        layer_signatures.to(dtype=torch.float32),
        proto_centers.to(dtype=torch.float32),
        p=2.0,
    )
    weights = F.softmax(-dists / temp, dim=1)
    topk = max(0, int(soft_assignment_topk))
    if topk > 0 and topk < int(weights.size(1)):
        topk_idx = torch.topk(weights, k=topk, dim=1).indices
        mask = torch.zeros_like(weights)
        mask.scatter_(1, topk_idx, 1.0)
        weights = weights * mask
        weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
    return weights.to(dtype=torch.float32)


def _load_shared_payload_from_ckpt(path: str) -> Dict[str, Any]:
    loaded = torch.load(path, map_location="cpu")
    if isinstance(loaded, dict) and "shared_state" in loaded:
        payload = loaded.get("shared_state") or {}
        if isinstance(payload, dict):
            if "meta" not in payload and isinstance(loaded.get("meta"), dict):
                payload = dict(payload)
                payload["meta"] = loaded["meta"]
            if "step" not in payload and "step" in loaded:
                payload = dict(payload)
                payload["step"] = loaded.get("step")
            return payload
    if not isinstance(loaded, dict):
        raise ValueError(f"Invalid shared checkpoint: {path}")
    return loaded


def _collect_trainable_groups(model: nn.Module) -> Dict[str, List[nn.Parameter]]:
    model = unwrap_model(model)
    bank_params: List[nn.Parameter] = []
    adapter_params: List[nn.Parameter] = []
    root = getattr(model, "model", model)
    bank = getattr(root, "shared_mlp_bank", None)
    if bank is not None:
        for param in bank.parameters():
            if param.requires_grad:
                bank_params.append(param)
    for layer in _resolve_layers(model):
        if isinstance(layer.mlp, SharedMLPAdapter):
            for param in layer.mlp.parameters():
                if param.requires_grad:
                    adapter_params.append(param)
    return {
        "bank": bank_params,
        "adapter": adapter_params,
    }


def _first_param_group_lr(optimizer: torch.optim.Optimizer, name: str) -> float:
    for group in optimizer.param_groups:
        if str(group.get("name", "")).strip().lower() == str(name).strip().lower():
            return float(group["lr"])
    return 0.0


def _build_lr_scheduler(
    *,
    optimizer: torch.optim.Optimizer,
    schedule: str,
    total_steps: int,
    warmup_steps: int,
    warmup_ratio: float,
    min_lr_ratio: float,
) -> Tuple[Optional[torch.optim.lr_scheduler.LambdaLR], Dict[str, Any]]:
    schedule_name = str(schedule).strip().lower()
    total_steps = max(1, int(total_steps))
    warmup_steps = max(0, int(warmup_steps))
    if warmup_steps <= 0:
        warmup_steps = int(round(float(max(0.0, warmup_ratio)) * float(total_steps)))
    warmup_steps = min(total_steps, warmup_steps)
    min_lr_ratio = float(max(0.0, min(1.0, float(min_lr_ratio))))
    if schedule_name == "none":
        return None, {
            "schedule": "none",
            "total_steps": int(total_steps),
            "warmup_steps": int(warmup_steps),
            "warmup_ratio": float(warmup_ratio),
            "min_lr_ratio": float(min_lr_ratio),
        }
    if schedule_name != "warmup_cosine":
        raise ValueError(f"Unsupported lr_schedule: {schedule}")

    decay_steps = max(1, int(total_steps - warmup_steps))

    def lr_lambda(step_idx: int) -> float:
        step_num = int(step_idx) + 1
        if warmup_steps > 0 and step_num <= warmup_steps:
            return float(step_num) / float(max(1, warmup_steps))
        decay_idx = max(0, step_num - warmup_steps)
        progress = min(1.0, float(decay_idx) / float(decay_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(min_lr_ratio) + (1.0 - float(min_lr_ratio)) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return scheduler, {
        "schedule": "warmup_cosine",
        "total_steps": int(total_steps),
        "warmup_steps": int(warmup_steps),
        "warmup_ratio": float(warmup_ratio),
        "min_lr_ratio": float(min_lr_ratio),
    }


def _select_single_token_positions(
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    eos_token_id: Optional[int],
    token_rule: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    normalized = str(token_rule).strip().lower()
    if normalized == "last_pred":
        token_indices = last_pred_indices(attention_mask, input_ids, eos_token_id, input_ids.device)
    else:
        token_indices = last_content_indices(attention_mask, input_ids, eos_token_id, input_ids.device)
    batch_indices = torch.arange(int(input_ids.size(0)), device=input_ids.device, dtype=torch.long)
    return batch_indices, token_indices.to(dtype=torch.long)


def _select_core_token_positions(
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    eos_token_id: Optional[int],
    token_rule: str,
    selection_mode: str,
    candidate_tokens: int,
    prompt_lens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return flattened token positions for core/FAD loss, plus sample groups."""
    single_batch, single_token = _select_single_token_positions(
        input_ids=input_ids,
        attention_mask=attention_mask,
        eos_token_id=eos_token_id,
        token_rule=token_rule,
    )
    mode = str(selection_mode or "last_pred").strip().lower()
    max_tokens = max(1, int(candidate_tokens))
    if mode in {"response_all", "all_response", "response_pred", "all_pred", "all_tokens", "phase_response"} and prompt_lens is not None:
        prompt_lens_device = prompt_lens.to(device=input_ids.device, dtype=torch.long).view(-1)
        batch_out: List[torch.Tensor] = []
        token_out: List[torch.Tensor] = []
        group_out: List[torch.Tensor] = []
        anchor_out: List[torch.Tensor] = []
        for batch_idx in range(int(input_ids.size(0))):
            valid_length = int(attention_mask[batch_idx].to(dtype=torch.long).sum().item())
            if valid_length <= 0:
                continue
            anchor = max(0, min(int(single_token[batch_idx].item()), valid_length - 1))
            start = max(0, min(int(prompt_lens_device[batch_idx].item()) - 1, anchor))
            tokens = torch.arange(start, anchor + 1, device=input_ids.device, dtype=torch.long)
            if int(tokens.numel()) > max_tokens:
                if mode == "phase_response":
                    select = torch.linspace(
                        0, int(tokens.numel()) - 1, steps=max_tokens, device=tokens.device
                    ).round().long().unique(sorted=True)
                    tokens = tokens.index_select(0, select)
                else:
                    tokens = tokens[-max_tokens:]
            if tokens.numel() <= 0:
                continue
            keep = attention_mask[batch_idx].to(dtype=torch.bool).gather(0, tokens)
            tokens = tokens[keep]
            if eos_token_id is not None and tokens.numel() > 1:
                not_eos = input_ids[batch_idx, tokens] != int(eos_token_id)
                tokens = tokens[not_eos | (tokens == int(anchor))]
            if tokens.numel() <= 0:
                tokens = single_token[batch_idx].view(1)
            count = int(tokens.numel())
            batch_out.append(torch.full((count,), int(batch_idx), device=input_ids.device, dtype=torch.long))
            token_out.append(tokens.to(device=input_ids.device, dtype=torch.long))
            group_out.append(torch.full((count,), int(batch_idx), device=input_ids.device, dtype=torch.long))
            anchor_out.append((tokens.to(device=input_ids.device, dtype=torch.long) == int(anchor)).to(dtype=torch.bool))
        if batch_out:
            return (
                torch.cat(batch_out, dim=0),
                torch.cat(token_out, dim=0),
                torch.cat(group_out, dim=0),
                torch.cat(anchor_out, dim=0),
            )
    if mode in {"", "last_pred", "single", "anchor"} or max_tokens <= 1:
        group_ids = torch.arange(int(input_ids.size(0)), device=input_ids.device, dtype=torch.long)
        anchor_mask = torch.ones_like(group_ids, dtype=torch.bool)
        return single_batch, single_token, group_ids, anchor_mask

    batch_out: List[torch.Tensor] = []
    token_out: List[torch.Tensor] = []
    group_out: List[torch.Tensor] = []
    anchor_out: List[torch.Tensor] = []
    batch_size = int(input_ids.size(0))
    for batch_idx in range(batch_size):
        anchor = int(single_token[batch_idx].item())
        valid = torch.nonzero(attention_mask[batch_idx].to(dtype=torch.bool), as_tuple=False).view(-1)
        if valid.numel() <= 0:
            valid = single_token[batch_idx].view(1)
        valid = valid[valid <= int(anchor)]
        if eos_token_id is not None and valid.numel() > 1:
            not_eos = input_ids[batch_idx, valid] != int(eos_token_id)
            keep = not_eos | (valid == int(anchor))
            valid = valid[keep]
        if valid.numel() <= 0:
            valid = single_token[batch_idx].view(1)

        non_anchor = valid[valid != int(anchor)]
        slots = max(0, max_tokens - 1)
        if non_anchor.numel() > slots and slots > 0:
            sample_idx = torch.linspace(
                0,
                int(non_anchor.numel()) - 1,
                steps=int(slots),
                device=non_anchor.device,
            ).round().to(dtype=torch.long)
            non_anchor = non_anchor.index_select(0, sample_idx).unique(sorted=True)
        elif slots <= 0:
            non_anchor = non_anchor[:0]

        tokens = torch.cat([non_anchor, single_token[batch_idx].view(1)], dim=0).unique(sorted=True)
        if tokens.numel() > max_tokens:
            tokens = torch.cat([tokens[: max_tokens - 1], single_token[batch_idx].view(1)], dim=0).unique(sorted=True)
        count = int(tokens.numel())
        batch_out.append(torch.full((count,), int(batch_idx), device=input_ids.device, dtype=torch.long))
        token_out.append(tokens.to(device=input_ids.device, dtype=torch.long))
        group_out.append(torch.full((count,), int(batch_idx), device=input_ids.device, dtype=torch.long))
        anchor_out.append((tokens.to(device=input_ids.device, dtype=torch.long) == int(anchor)).to(dtype=torch.bool))

    return (
        torch.cat(batch_out, dim=0),
        torch.cat(token_out, dim=0),
        torch.cat(group_out, dim=0),
        torch.cat(anchor_out, dim=0),
    )


def _phase_progress_for_selected_tokens(
    *,
    token_batch_indices: torch.Tensor,
    token_indices: torch.Tensor,
    prompt_lens: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    phase_ids = torch.zeros_like(token_indices, dtype=torch.long)
    progress = torch.zeros_like(token_indices, dtype=torch.float32)
    prompt = prompt_lens.to(device=token_indices.device, dtype=torch.long).view(-1)
    for row in range(int(token_indices.numel())):
        batch = int(token_batch_indices[row].item())
        start = max(0, int(prompt[batch].item()) - 1)
        valid = int(attention_mask[batch].to(dtype=torch.long).sum().item())
        end = max(start, valid - 2)
        value = float(int(token_indices[row].item()) - start) / float(max(1, end - start))
        value = min(1.0, max(0.0, value))
        progress[row] = value
        if int(token_indices[row].item()) >= end:
            phase_ids[row] = 3
        elif value <= 1.0 / 3.0:
            phase_ids[row] = 0
        elif value <= 2.0 / 3.0:
            phase_ids[row] = 1
        else:
            phase_ids[row] = 2
    return phase_ids, progress


def _metric_inverse_sqrt(
    metric_diag: torch.Tensor,
    *,
    eps: float,
    trace_normalize: bool,
    input_is_precision: bool = False,
) -> torch.Tensor:
    """Return sqrt(M), accepting either covariance diag or direct precision diag."""
    safe_diag = metric_diag.to(dtype=torch.float32).clamp(min=float(eps))
    precision_diag = safe_diag if bool(input_is_precision) else safe_diag.reciprocal()
    if bool(trace_normalize):
        dimension = max(1, int(precision_diag.numel()))
        scale = float(dimension) / precision_diag.sum().clamp(min=float(eps))
        precision_diag = precision_diag * scale
    return torch.sqrt(precision_diag)


def _compute_core_token_weights(
    *,
    z_teacher_by_layer: Dict[int, torch.Tensor],
    tau_layer_ids: Sequence[int],
    layer_metric_diag_device: torch.Tensor,
    layer_reliability_device: torch.Tensor,
    token_batch_indices: Optional[torch.Tensor],
    token_indices: Optional[torch.Tensor],
    group_ids: Optional[torch.Tensor],
    anchor_mask: Optional[torch.Tensor],
    teacher_logits: Optional[torch.Tensor],
    selection_mode: str,
    temperature: float,
    anchor_boost: float,
    energy_alpha: float,
    entropy_beta: float,
    hard_topk: int,
    metric_eps: float,
    use_metric_whitening: bool,
    metric_trace_normalize: bool,
    metric_diag_is_precision: bool,
    use_reliability_weighting: bool,
) -> Optional[torch.Tensor]:
    mode = str(selection_mode or "last_pred").strip().lower()
    if mode not in {"iets", "iets_softmax", "iets_topk", "energy", "energy_softmax"}:
        return None
    if group_ids is None or group_ids.numel() <= 0:
        return None
    device = group_ids.device
    count = int(group_ids.numel())
    score = torch.zeros((count,), dtype=torch.float32, device=device)
    weight_sum = 0.0
    for layer_id in tau_layer_ids:
        z_t = z_teacher_by_layer.get(int(layer_id))
        if z_t is None or z_t.dim() != 2 or int(z_t.size(0)) != count:
            continue
        if bool(use_metric_whitening):
            inv_std = _metric_inverse_sqrt(
                layer_metric_diag_device[int(layer_id)].view(1, -1),
                eps=float(metric_eps),
                trace_normalize=bool(metric_trace_normalize),
                input_is_precision=bool(metric_diag_is_precision),
            )
            centered = (z_t.to(dtype=torch.float32) - z_t.to(dtype=torch.float32).mean(dim=0, keepdim=True)) * inv_std
        else:
            centered = z_t.to(dtype=torch.float32) - z_t.to(dtype=torch.float32).mean(dim=0, keepdim=True)
        energy = 0.5 * centered.pow(2).mean(dim=1)
        layer_weight = float(layer_reliability_device[int(layer_id)].item()) if bool(use_reliability_weighting) else 1.0
        score = score + float(layer_weight) * energy
        weight_sum += float(layer_weight)
    if weight_sum > 0.0:
        score = score / float(weight_sum)
    score = float(energy_alpha) * score

    if (
        abs(float(entropy_beta)) > 0.0
        and teacher_logits is not None
        and token_batch_indices is not None
        and token_indices is not None
        and int(token_batch_indices.numel()) == count
        and int(token_indices.numel()) == count
    ):
        selected_logits = teacher_logits[token_batch_indices, token_indices, :].to(dtype=torch.float32)
        log_probs = F.log_softmax(selected_logits, dim=-1)
        entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
        entropy = (entropy - entropy.mean()) / entropy.std(unbiased=False).clamp(min=1e-6)
        score = score + float(entropy_beta) * entropy

    if anchor_mask is not None and int(anchor_mask.numel()) == count and abs(float(anchor_boost)) > 0.0:
        score = score + anchor_mask.to(device=device, dtype=torch.float32) * float(anchor_boost)

    token_weights = torch.zeros_like(score)
    temp = max(1e-4, float(temperature))
    unique_groups = torch.unique(group_ids)
    for group_id in unique_groups.tolist():
        mask = group_ids == int(group_id)
        local_score = score[mask]
        if local_score.numel() <= 0:
            continue
        if mode == "iets_topk":
            keep = max(1, min(int(hard_topk), int(local_score.numel())))
            top_idx = torch.topk(local_score, k=keep, dim=0).indices
            local_weight = torch.zeros_like(local_score)
            local_weight[top_idx] = 1.0 / float(keep)
        else:
            local_weight = F.softmax(local_score / temp, dim=0)
        token_weights[mask] = local_weight
    return token_weights.detach()


def _diffusion_manifold_embed(
    *,
    z: torch.Tensor,
    anchor_z: torch.Tensor,
    anchor_phi: torch.Tensor,
    metric_diag: torch.Tensor,
    sigma2: torch.Tensor,
    temperature: float,
    eps: float,
) -> torch.Tensor:
    if z.dim() != 2 or anchor_z.dim() != 2 or anchor_phi.dim() != 2:
        raise ValueError("diffusion manifold tensors must be rank-2.")
    if int(anchor_z.size(0)) != int(anchor_phi.size(0)):
        raise ValueError("diffusion manifold anchor_z/anchor_phi count mismatch.")
    if int(z.size(1)) != int(anchor_z.size(1)) or int(z.size(1)) != int(metric_diag.numel()):
        raise ValueError("diffusion manifold rank mismatch.")
    temp = max(1e-4, float(temperature))
    denom = sigma2.to(device=z.device, dtype=torch.float32).view(()).clamp(min=float(eps)) * temp
    inv_std = torch.rsqrt(metric_diag.to(device=z.device, dtype=torch.float32).view(1, 1, -1).clamp(min=float(eps)))
    delta = (z.to(dtype=torch.float32).unsqueeze(1) - anchor_z.to(device=z.device, dtype=torch.float32).unsqueeze(0)) * inv_std
    dist2 = delta.pow(2).sum(dim=-1)
    weights = F.softmax(-dist2 / denom, dim=1)
    return weights.matmul(anchor_phi.to(device=z.device, dtype=torch.float32))


def _diffusion_manifold_expected_risk(
    *,
    z: torch.Tensor,
    anchor_z: torch.Tensor,
    anchor_risk: torch.Tensor,
    metric_diag: torch.Tensor,
    sigma2: torch.Tensor,
    temperature: float,
    eps: float,
) -> torch.Tensor:
    if z.dim() != 2 or anchor_z.dim() != 2:
        raise ValueError("diffusion manifold risk tensors must be rank-2.")
    if int(anchor_z.size(0)) != int(anchor_risk.numel()):
        raise ValueError("diffusion manifold anchor_z/anchor_risk count mismatch.")
    if int(z.size(1)) != int(anchor_z.size(1)) or int(z.size(1)) != int(metric_diag.numel()):
        raise ValueError("diffusion manifold risk rank mismatch.")
    temp = max(1e-4, float(temperature))
    denom = sigma2.to(device=z.device, dtype=torch.float32).view(()).clamp(min=float(eps)) * temp
    inv_std = torch.rsqrt(metric_diag.to(device=z.device, dtype=torch.float32).view(1, 1, -1).clamp(min=float(eps)))
    delta = (z.to(dtype=torch.float32).unsqueeze(1) - anchor_z.to(device=z.device, dtype=torch.float32).unsqueeze(0)) * inv_std
    dist2 = delta.pow(2).sum(dim=-1)
    weights = F.softmax(-dist2 / denom, dim=1)
    return weights.matmul(anchor_risk.to(device=z.device, dtype=torch.float32).view(-1, 1)).view(-1)


def _resolve_tau_layer_ids(
    *,
    mode: str,
    layer_count: int,
    seed_layer_for_proto: Dict[int, int],
    layer_to_proto: Optional[Sequence[int]] = None,
    layer_priority: Optional[torch.Tensor] = None,
    extra_topk: int = 0,
) -> List[int]:
    normalized = str(mode).strip().lower()
    if normalized.startswith("layers:"):
        raw = normalized.split(":", 1)[1].strip()
        if not raw:
            raise ValueError("Explicit core layer set must contain at least one zero-based layer id.")
        try:
            selected = sorted({int(value.strip()) for value in raw.split(",") if value.strip()})
        except ValueError as exc:
            raise ValueError(f"Invalid explicit core layer set: {mode!r}") from exc
        invalid = [value for value in selected if value < 0 or value >= int(layer_count)]
        if invalid:
            raise ValueError(
                f"Explicit core layer ids must be in 0..{int(layer_count) - 1}; got {invalid}."
            )
        if not selected:
            raise ValueError("Explicit core layer set must contain at least one layer id.")
        return selected
    if normalized == "all":
        return list(range(int(layer_count)))
    if normalized == "all_shared_layers":
        if layer_to_proto is None:
            return list(range(int(layer_count)))
        counts: Dict[int, int] = {}
        for proto_id_raw in layer_to_proto:
            proto_id = int(proto_id_raw)
            counts[proto_id] = counts.get(proto_id, 0) + 1
        return [
            layer_id
            for layer_id, proto_id_raw in enumerate(layer_to_proto)
            if counts[int(proto_id_raw)] > 1
        ]
    if normalized == "proto_seed_layers":
        unique = sorted({int(v) for v in seed_layer_for_proto.values() if 0 <= int(v) < int(layer_count)})
        return unique
    if normalized == "proto_seed_plus_topk_error":
        selected = {
            int(v)
            for v in seed_layer_for_proto.values()
            if 0 <= int(v) < int(layer_count)
        }
        topk = max(0, int(extra_topk))
        if topk > 0 and torch.is_tensor(layer_priority) and layer_priority.dim() == 1 and int(layer_priority.numel()) == int(layer_count):
            top_idx = torch.topk(layer_priority.to(dtype=torch.float32), k=min(topk, int(layer_priority.numel()))).indices.tolist()
            for layer_id in top_idx:
                if 0 <= int(layer_id) < int(layer_count):
                    selected.add(int(layer_id))
        return sorted(selected)
    raise ValueError(
        f"Unsupported tau_layers={mode!r}. Use all/all_shared_layers/proto_seed_layers/"
        "proto_seed_plus_topk_error or layers:<zero-based comma-separated ids>."
    )


def _core_lambda_at_step(
    *,
    base_lambda: float,
    schedule: str,
    step: int,
    total_steps: int,
    warmup_ratio: float,
    cutoff_ratio: float,
) -> float:
    """Resolve the FAD coefficient for one optimizer step (step is one-based)."""
    base = max(0.0, float(base_lambda))
    mode = str(schedule).strip().lower()
    current = max(1, int(step))
    total = max(1, int(total_steps))
    if mode == "constant":
        return base
    if mode == "warmup":
        warmup_steps = max(1, int(round(total * min(1.0, max(0.0, float(warmup_ratio))))))
        return base * min(1.0, float(current) / float(warmup_steps))
    if mode == "linear_decay":
        if total <= 1:
            return 0.0
        return base * max(0.0, 1.0 - float(current - 1) / float(total - 1))
    if mode in {"early_only", "early_then_ce"}:
        cutoff_steps = max(1, int(round(total * min(1.0, max(0.0, float(cutoff_ratio))))))
        return base if current <= cutoff_steps else 0.0
    raise ValueError(
        f"Unsupported core_lambda_schedule={schedule!r}; use constant/warmup/linear_decay/early_only."
    )


def _adapter_param_regularization(params: Sequence[nn.Parameter]) -> torch.Tensor:
    reg = None
    count = 0
    for param in params:
        if not param.requires_grad:
            continue
        value = param.float().pow(2).mean()
        reg = value if reg is None else reg + value
        count += 1
    if reg is None or count <= 0:
        return torch.zeros((), dtype=torch.float32)
    return reg / float(count)


def _parse_int_list(text: Any) -> List[int]:
    if text is None:
        return []
    flat = str(text).replace(",", " ")
    out: List[int] = []
    seen = set()
    for token in flat.split():
        try:
            value = int(token)
        except Exception:
            continue
        if value in seen:
            continue
        out.append(value)
        seen.add(value)
    return out


def _build_window_specs(
    *,
    ordered_layer_ids: Sequence[int],
    requested_start_layers: Any,
    window_size: int,
    weighting: str,
) -> List[Dict[str, Any]]:
    ordered = [int(x) for x in ordered_layer_ids]
    win_size = max(2, int(window_size))
    if len(ordered) < win_size:
        return []

    requested = set(_parse_int_list(requested_start_layers))
    specs: List[Dict[str, Any]] = []
    for start_idx in range(len(ordered) - win_size + 1):
        start_layer = int(ordered[start_idx])
        if requested and start_layer not in requested:
            continue
        window_layers = ordered[start_idx : start_idx + win_size]
        specs.append(
            {
                "start_idx": int(start_idx),
                "start_layer": int(start_layer),
                "layer_ids": [int(x) for x in window_layers],
            }
        )
    if not specs:
        return []

    weighting_mode = str(weighting or "uniform").strip().lower()
    raw_weights: List[float] = []
    if weighting_mode == "late_focus":
        raw_weights = [float(idx + 1) for idx in range(len(specs))]
    else:
        raw_weights = [1.0] * len(specs)
    mean_weight = sum(raw_weights) / max(1, len(raw_weights))
    if mean_weight <= 0.0:
        mean_weight = 1.0
    for spec, raw_weight in zip(specs, raw_weights):
        spec["weight"] = float(raw_weight / mean_weight)
    return specs


def _fit_proto_residual_basis(
    *,
    cov_sum: torch.Tensor,
    weight_sum: torch.Tensor,
    residual_rank: int,
) -> Dict[str, torch.Tensor]:
    if cov_sum.dim() < 3 or weight_sum.dim() < 1:
        raise ValueError("Invalid covariance/weight shapes for proto residual basis.")
    proto_count = int(cov_sum.size(-3))
    atlas_rank = int(cov_sum.size(-1))
    rank = max(1, min(int(residual_rank), atlas_rank))
    lead_shape = tuple(int(x) for x in cov_sum.shape[:-3])
    flat_groups = int(math.prod(lead_shape)) if lead_shape else 1

    cov_flat = cov_sum.reshape(flat_groups, proto_count, atlas_rank, atlas_rank)
    weight_flat = weight_sum.reshape(flat_groups, proto_count)
    basis_flat = torch.zeros((flat_groups, proto_count, atlas_rank, rank), dtype=torch.float32)
    explained_flat = torch.zeros((flat_groups, proto_count), dtype=torch.float32)
    eff90_flat = torch.zeros((flat_groups, proto_count), dtype=torch.long)
    eff95_flat = torch.zeros((flat_groups, proto_count), dtype=torch.long)
    covariance_flat = torch.zeros((flat_groups, proto_count, atlas_rank, atlas_rank), dtype=torch.float32)
    eigenvalues_flat = torch.zeros((flat_groups, proto_count, atlas_rank), dtype=torch.float32)

    for group_id in range(flat_groups):
        for proto_id in range(proto_count):
            denom = float(weight_flat[group_id, proto_id].item())
            if denom <= 0.0:
                continue
            cov = cov_flat[group_id, proto_id] / denom
            cov = 0.5 * (cov + cov.transpose(0, 1))
            cov_f32 = cov.to(dtype=torch.float32)
            covariance_flat[group_id, proto_id] = cov_f32
            evals, evecs = torch.linalg.eigh(cov_f32)
            order = torch.argsort(evals, descending=True)
            evals = evals.index_select(0, order)
            evecs = evecs.index_select(1, order)
            eigenvalues_flat[group_id, proto_id] = evals
            basis_flat[group_id, proto_id] = evecs[:, :rank].contiguous()
            total_energy = float(torch.clamp(evals.sum(), min=1e-12).item())
            explained_flat[group_id, proto_id] = float(evals[:rank].sum().item() / total_energy)
            cumulative = torch.cumsum(evals, dim=0) / total_energy
            eff90_flat[group_id, proto_id] = int(torch.searchsorted(cumulative, torch.tensor(0.90)).item()) + 1
            eff95_flat[group_id, proto_id] = int(torch.searchsorted(cumulative, torch.tensor(0.95)).item()) + 1

    if lead_shape:
        basis_out = basis_flat.reshape(*lead_shape, proto_count, atlas_rank, rank)
        explained_out = explained_flat.reshape(*lead_shape, proto_count)
        eff90_out = eff90_flat.reshape(*lead_shape, proto_count)
        eff95_out = eff95_flat.reshape(*lead_shape, proto_count)
        covariance_out = covariance_flat.reshape(*lead_shape, proto_count, atlas_rank, atlas_rank)
        eigenvalues_out = eigenvalues_flat.reshape(*lead_shape, proto_count, atlas_rank)
    else:
        basis_out = basis_flat.reshape(proto_count, atlas_rank, rank)
        explained_out = explained_flat.reshape(proto_count)
        eff90_out = eff90_flat.reshape(proto_count)
        eff95_out = eff95_flat.reshape(proto_count)
        covariance_out = covariance_flat.reshape(proto_count, atlas_rank, atlas_rank)
        eigenvalues_out = eigenvalues_flat.reshape(proto_count, atlas_rank)

    return {
        "basis": basis_out,
        "explained_ratio": explained_out,
        "effective_rank_90": eff90_out,
        "effective_rank_95": eff95_out,
        "covariance": covariance_out,
        "eigenvalues": eigenvalues_out,
    }


def _lambda_tau_at_step(
    *,
    step: int,
    total_steps: int,
    lambda_tau_max: float,
    warmup_start_ratio: float,
    warmup_end_ratio: float,
) -> float:
    lam_max = max(0.0, float(lambda_tau_max))
    if lam_max <= 0.0:
        return 0.0
    total = max(1, int(total_steps))
    progress = float(max(1, int(step))) / float(total)
    start = min(1.0, max(0.0, float(warmup_start_ratio)))
    end = min(1.0, max(start + 1e-8, float(warmup_end_ratio)))
    if progress <= start:
        return 0.0
    if progress >= end:
        return lam_max
    ratio = (progress - start) / max(1e-8, (end - start))
    return lam_max * ratio


def _run_compress_validation(
    *,
    student: nn.Module,
    teacher: Optional[nn.Module],
    loader: DataLoader,
    device: torch.device,
    pad_token_id: int,
    eos_token_id: Optional[int],
    lambda_ce: float,
    lambda_kd: float,
    kd_temperature: float,
    distill_mode: str,
    max_batches: int,
    loss_scope: str = "all",
    loss_exclude_eos: bool = True,
    sage_config: Optional[Dict[str, Any]] = None,
    subspace_viz_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    prev_mode = bool(student.training)
    student.eval()
    if teacher is not None:
        teacher.eval()
    total_loss = 0.0
    total_ce = 0.0
    total_kd = 0.0
    total_response_ce = 0.0
    total_decision_ce = 0.0
    total_sage_gate = 0.0
    total_sage_active = 0.0
    total_teacher_correct = 0.0
    batch_count = 0
    sample_count = 0
    subspace_report: Optional[Dict[str, Any]] = None
    subspace_enabled = bool(isinstance(subspace_viz_config, dict) and subspace_viz_config.get("enabled", False))
    subspace_layer_ids: List[int] = []
    subspace_regime_labels: List[str] = []
    subspace_regime_basis_device_map: Dict[str, torch.Tensor] = {}
    subspace_token_rule = "last_pred"
    subspace_step = -1
    subspace_output_dir = ""
    subspace_max_points_per_regime = 256
    subspace_save_plot = True
    subspace_plot_prefix = ""
    subspace_layer_metric_diag: Optional[torch.Tensor] = None
    if subspace_enabled:
        if teacher is None:
            raise ValueError("validation subspace visualization requires a teacher model")
        subspace_layer_ids = [int(x) for x in subspace_viz_config.get("layer_ids", [])]
        subspace_regime_labels = [str(x) for x in subspace_viz_config.get("regime_labels", [])]
        subspace_regime_basis_device_map = {
            str(key): value
            for key, value in dict(subspace_viz_config.get("regime_basis_device_map", {})).items()
            if torch.is_tensor(value)
        }
        subspace_token_rule = str(subspace_viz_config.get("token_rule", "last_pred")).strip().lower()
        subspace_step = int(subspace_viz_config.get("step", -1))
        subspace_output_dir = str(subspace_viz_config.get("output_dir", "")).strip()
        subspace_max_points_per_regime = max(0, int(subspace_viz_config.get("max_points_per_regime", 256)))
        subspace_save_plot = bool(subspace_viz_config.get("save_plot", True))
        subspace_plot_prefix = str(subspace_viz_config.get("plot_prefix", "")).strip()
        candidate_metric = subspace_viz_config.get("layer_metric_diag", None)
        if torch.is_tensor(candidate_metric) and candidate_metric.dim() == 2:
            subspace_layer_metric_diag = candidate_metric.detach().float().cpu().contiguous()
        if not subspace_layer_ids or not subspace_regime_labels or not subspace_regime_basis_device_map or not subspace_output_dir:
            subspace_enabled = False

    subspace_teacher_points: Dict[str, List[torch.Tensor]] = {}
    subspace_student_points: Dict[str, List[torch.Tensor]] = {}
    subspace_points_by_regime: Dict[str, int] = {}
    subspace_regime_gap_sum: Dict[str, float] = {}
    subspace_regime_gap_count: Dict[str, int] = {}
    subspace_layer_gap_sum: Dict[int, float] = {}
    subspace_layer_gap_count: Dict[int, int] = {}
    subspace_regime_cos_sum: Dict[str, float] = {}
    subspace_regime_cos_count: Dict[str, int] = {}
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if int(max_batches) > 0 and int(batch_idx) >= int(max_batches):
                break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            prompt_lens = batch.get("prompt_lens", None)
            if prompt_lens is not None:
                prompt_lens = prompt_lens.to(device)
            candidate_token_ids = batch.get("candidate_token_ids", None)
            candidate_mask = batch.get("candidate_mask", None)
            gold_candidate_index = batch.get("gold_candidate_index", None)
            if candidate_token_ids is not None:
                candidate_token_ids = candidate_token_ids.to(device)
            if candidate_mask is not None:
                candidate_mask = candidate_mask.to(device)
            if gold_candidate_index is not None:
                gold_candidate_index = gold_candidate_index.to(device)
            teacher_mlp_selected: Dict[int, torch.Tensor] = {}
            student_mlp_selected: Dict[int, torch.Tensor] = {}
            t_out = None
            if subspace_enabled:
                assert teacher is not None
                batch_indices, token_indices = _select_single_token_positions(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    eos_token_id=eos_token_id,
                    token_rule=subspace_token_rule,
                )
                t_out, _, teacher_mlp_selected, _, _ = _forward_with_selected_capture(
                    model=teacher,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    batch_indices=batch_indices,
                    token_indices=token_indices,
                    capture_hidden=False,
                    capture_mlp_layer_ids=subspace_layer_ids,
                    capture_pre_ffn_input_layer_ids=None,
                    capture_residual_output_layer_ids=None,
                )
            elif teacher is not None:
                t_out = teacher(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                    use_cache=False,
                )
            if subspace_enabled:
                s_out, _, student_mlp_selected, _, _ = _forward_with_selected_capture(
                    model=student,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    batch_indices=batch_indices,
                    token_indices=token_indices,
                    capture_hidden=False,
                    capture_mlp_layer_ids=subspace_layer_ids,
                    capture_pre_ffn_input_layer_ids=None,
                    capture_residual_output_layer_ids=None,
                )
            else:
                s_out = student(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                    use_cache=False,
                )
            s_logits = s_out.logits.float()
            t_logits = _extract_logits_from_model_output(t_out).float() if t_out is not None else None
            scope_mask = shifted_target_mask(
                input_ids=input_ids,
                attention_mask=attention_mask,
                prompt_lens=prompt_lens,
                scope=str(loss_scope),
                pad_token_id=int(pad_token_id),
                eos_token_id=eos_token_id,
                exclude_eos=bool(loss_exclude_eos),
            )
            response_mask = shifted_target_mask(
                input_ids=input_ids,
                attention_mask=attention_mask,
                prompt_lens=prompt_lens,
                scope="response",
                pad_token_id=int(pad_token_id),
                eos_token_id=eos_token_id,
                exclude_eos=bool(loss_exclude_eos),
            )
            decision_mask = shifted_target_mask(
                input_ids=input_ids,
                attention_mask=attention_mask,
                prompt_lens=prompt_lens,
                scope="decision",
                pad_token_id=int(pad_token_id),
                eos_token_id=eos_token_id,
                exclude_eos=True,
            )
            ce_loss, _ = masked_next_token_cross_entropy(
                logits=s_logits,
                input_ids=input_ids,
                token_mask=scope_mask,
            )
            response_ce_loss, _ = masked_next_token_cross_entropy(
                logits=s_logits,
                input_ids=input_ids,
                token_mask=response_mask,
            )
            has_candidate_channel = candidate_token_ids is not None and candidate_mask is not None and gold_candidate_index is not None
            student_candidate_logits: Optional[torch.Tensor] = None
            teacher_candidate_logits: Optional[torch.Tensor] = None
            if has_candidate_channel:
                decision_ce_loss, decision_stats = candidate_decision_cross_entropy(
                    logits=s_logits,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    candidate_token_ids=candidate_token_ids,
                    candidate_mask=candidate_mask,
                    gold_candidate_index=gold_candidate_index,
                    eos_token_id=eos_token_id,
                )
                student_candidate_logits = decision_stats["candidate_logits"]
                if t_logits is not None:
                    teacher_candidate_logits = candidate_decision_logits(
                        logits=t_logits,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        candidate_token_ids=candidate_token_ids,
                        candidate_mask=candidate_mask,
                        eos_token_id=eos_token_id,
                    )
                if str(loss_scope) == "decision":
                    ce_loss = decision_ce_loss
            else:
                decision_ce_loss, _ = masked_next_token_cross_entropy(
                    logits=s_logits,
                    input_ids=input_ids,
                    token_mask=decision_mask,
                )
            sage_stats: Dict[str, torch.Tensor] = {}
            if str(distill_mode) == "ce_kd" and float(lambda_kd) > 0.0:
                if t_logits is None:
                    raise RuntimeError("CE+KD validation requires teacher logits")
                kd_loss = _kd_shift_masked_token_mean(
                    student_logits=s_logits,
                    teacher_logits=t_logits,
                    attention_mask=attention_mask,
                    temperature=float(kd_temperature),
                )
            elif str(distill_mode) == "sage_ib" and float(lambda_kd) > 0.0:
                cfg = dict(sage_config or {})
                if has_candidate_channel and student_candidate_logits is not None and teacher_candidate_logits is not None:
                    kd_loss, sage_stats = sage_candidate_information_gain_js(
                        student_candidate_logits=student_candidate_logits,
                        teacher_candidate_logits=teacher_candidate_logits,
                        candidate_mask=candidate_mask,
                        gold_candidate_index=gold_candidate_index,
                        temperature=float(kd_temperature),
                        gain_margin=float(cfg.get("gain_margin", 0.0)),
                        gain_temperature=float(cfg.get("gain_temperature", 0.25)),
                        confidence_margin=float(cfg.get("confidence_margin", 0.0)),
                        confidence_temperature=float(cfg.get("confidence_temperature", 1.0)),
                        confidence_power=float(cfg.get("confidence_power", 1.0)),
                        require_teacher_correct=bool(cfg.get("require_teacher_correct", True)),
                    )
                else:
                    kd_loss, sage_stats = sage_information_gain_js(
                        student_logits=s_logits,
                        teacher_logits=t_logits,
                        input_ids=input_ids,
                        token_mask=decision_mask,
                        temperature=float(kd_temperature),
                        topk=int(cfg.get("topk", 32)),
                        gain_margin=float(cfg.get("gain_margin", 0.0)),
                        gain_temperature=float(cfg.get("gain_temperature", 0.25)),
                        confidence_margin=float(cfg.get("confidence_margin", 0.0)),
                        confidence_temperature=float(cfg.get("confidence_temperature", 1.0)),
                        confidence_power=float(cfg.get("confidence_power", 1.0)),
                        require_teacher_correct=bool(cfg.get("require_teacher_correct", True)),
                    )
            else:
                kd_loss = torch.zeros((), dtype=torch.float32, device=device)
            loss = float(lambda_ce) * ce_loss + float(lambda_kd) * kd_loss
            total_loss += float(loss.item())
            total_ce += float(ce_loss.item())
            total_kd += float(kd_loss.item())
            total_response_ce += float(response_ce_loss.item())
            total_decision_ce += float(decision_ce_loss.item())
            total_sage_gate += float(sage_stats.get("gate_mean", torch.zeros(())).item())
            total_sage_active += float(sage_stats.get("active_fraction", torch.zeros(())).item())
            total_teacher_correct += float(sage_stats.get("teacher_correct_fraction", torch.zeros(())).item())
            batch_count += 1
            sample_count += int(input_ids.size(0))
            if subspace_enabled:
                for layer_id in subspace_layer_ids:
                    if not (0 <= int(layer_id) < len(subspace_regime_labels)):
                        continue
                    teacher_delta = teacher_mlp_selected.get(int(layer_id))
                    student_delta = student_mlp_selected.get(int(layer_id))
                    if teacher_delta is None or student_delta is None:
                        continue
                    regime_name = str(subspace_regime_labels[int(layer_id)]).strip() or "llama_late"
                    regime_basis = subspace_regime_basis_device_map.get(regime_name, None)
                    if regime_basis is None:
                        continue
                    z_teacher = torch.matmul(teacher_delta.to(dtype=torch.float32), regime_basis)
                    z_student = torch.matmul(student_delta.to(dtype=torch.float32), regime_basis)
                    pair_gap = torch.linalg.norm(z_student - z_teacher, dim=1)
                    pair_cos = F.cosine_similarity(z_student, z_teacher, dim=1, eps=1e-8)
                    subspace_regime_gap_sum[regime_name] = float(subspace_regime_gap_sum.get(regime_name, 0.0) + float(pair_gap.sum().item()))
                    subspace_regime_gap_count[regime_name] = int(subspace_regime_gap_count.get(regime_name, 0) + int(pair_gap.numel()))
                    subspace_regime_cos_sum[regime_name] = float(subspace_regime_cos_sum.get(regime_name, 0.0) + float(pair_cos.sum().item()))
                    subspace_regime_cos_count[regime_name] = int(subspace_regime_cos_count.get(regime_name, 0) + int(pair_cos.numel()))
                    subspace_layer_gap_sum[int(layer_id)] = float(subspace_layer_gap_sum.get(int(layer_id), 0.0) + float(pair_gap.sum().item()))
                    subspace_layer_gap_count[int(layer_id)] = int(subspace_layer_gap_count.get(int(layer_id), 0) + int(pair_gap.numel()))

                    current_points = int(subspace_points_by_regime.get(regime_name, 0))
                    remaining = max(0, int(subspace_max_points_per_regime) - current_points)
                    if remaining <= 0:
                        continue
                    keep = min(int(remaining), int(z_teacher.size(0)))
                    if keep <= 0:
                        continue
                    subspace_teacher_points.setdefault(regime_name, []).append(
                        z_teacher[:keep].detach().float().cpu().contiguous()
                    )
                    subspace_student_points.setdefault(regime_name, []).append(
                        z_student[:keep].detach().float().cpu().contiguous()
                    )
                    subspace_points_by_regime[regime_name] = int(current_points + keep)
    if prev_mode:
        student.train()
    denom = float(max(1, int(batch_count)))
    result: Dict[str, Any] = {
        "loss": float(total_loss / denom),
        "ce": float(total_ce / denom),
        "kd": float(total_kd / denom),
        "response_ce": float(total_response_ce / denom),
        "decision_ce": float(total_decision_ce / denom),
        "sage_gate_mean": float(total_sage_gate / denom),
        "sage_active_fraction": float(total_sage_active / denom),
        "sage_teacher_correct_fraction": float(total_teacher_correct / denom),
        "batches": float(batch_count),
        "samples": float(sample_count),
    }
    if subspace_enabled:
        regime_order = list(
            dict.fromkeys(
                [
                    str(subspace_regime_labels[int(layer_id)]).strip() or "llama_late"
                    for layer_id in subspace_layer_ids
                    if 0 <= int(layer_id) < len(subspace_regime_labels)
                ]
            )
        )
        regime_gap_mean = {
            regime: float(subspace_regime_gap_sum.get(regime, 0.0) / max(1, int(subspace_regime_gap_count.get(regime, 0))))
            for regime in regime_order
            if int(subspace_regime_gap_count.get(regime, 0)) > 0
        }
        regime_cos_mean = {
            regime: float(subspace_regime_cos_sum.get(regime, 0.0) / max(1, int(subspace_regime_cos_count.get(regime, 0))))
            for regime in regime_order
            if int(subspace_regime_cos_count.get(regime, 0)) > 0
        }
        total_gap_sum = float(sum(subspace_regime_gap_sum.values()))
        total_gap_count = int(sum(subspace_regime_gap_count.values()))
        total_cos_sum = float(sum(subspace_regime_cos_sum.values()))
        total_cos_count = int(sum(subspace_regime_cos_count.values()))
        layer_gap_mean = {
            str(int(layer_id)): float(subspace_layer_gap_sum.get(int(layer_id), 0.0) / max(1, int(subspace_layer_gap_count.get(int(layer_id), 0))))
            for layer_id in subspace_layer_ids
            if int(subspace_layer_gap_count.get(int(layer_id), 0)) > 0
        }
        energy_metric_diag_by_regime: Dict[str, torch.Tensor] = {}
        if torch.is_tensor(subspace_layer_metric_diag):
            for regime in regime_order:
                metric_rows: List[torch.Tensor] = []
                for layer_id in subspace_layer_ids:
                    if not (0 <= int(layer_id) < len(subspace_regime_labels)):
                        continue
                    if str(subspace_regime_labels[int(layer_id)]).strip() != str(regime):
                        continue
                    if 0 <= int(layer_id) < int(subspace_layer_metric_diag.size(0)):
                        metric_rows.append(subspace_layer_metric_diag[int(layer_id)].float().view(-1))
                if metric_rows:
                    energy_metric_diag_by_regime[str(regime)] = torch.stack(metric_rows, dim=0).mean(dim=0).clamp(min=1e-12)
        report = _build_validation_subspace_snapshot(
            teacher_points_by_regime=subspace_teacher_points,
            student_points_by_regime=subspace_student_points,
            regime_gap_mean=regime_gap_mean,
            regime_cos_mean=regime_cos_mean,
            layer_gap_mean=layer_gap_mean,
            energy_metric_diag_by_regime=energy_metric_diag_by_regime,
            regime_order=regime_order,
            step=int(subspace_step),
            output_dir=subspace_output_dir,
            save_plot=bool(subspace_save_plot),
            plot_prefix=subspace_plot_prefix,
        )
        report["mean_pair_l2"] = float(total_gap_sum / max(1, total_gap_count))
        report["mean_pair_cosine"] = float(total_cos_sum / max(1, total_cos_count)) if total_cos_count > 0 else None
        report["mean_pair_l2_by_regime"] = regime_gap_mean
        report["mean_pair_cosine_by_regime"] = regime_cos_mean
        report["mean_pair_l2_by_layer"] = layer_gap_mean
        report["samples_by_regime"] = {str(key): int(value) for key, value in subspace_points_by_regime.items()}
        save_json(str(report["output_json"]), report)
        subspace_report = report
        result["subspace_viz"] = report
    return result


def _load_sidecar_prior_from_atlas(
    atlas_path: str,
    *,
    sharing_policy_path: str = "",
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    atlas_dir = os.path.dirname(os.path.abspath(str(atlas_path)))
    prior_path = os.path.join(atlas_dir, "final_structure_prior.pt")
    fallback_policy_path = os.path.join(atlas_dir, "sharing_policy.json")
    policy_path = str(sharing_policy_path).strip() or fallback_policy_path
    prior_state = None
    policy_state = None
    if os.path.isfile(prior_path):
        prior_state = torch.load(prior_path, map_location="cpu")
    if not os.path.isfile(policy_path) and policy_path != fallback_policy_path:
        policy_path = fallback_policy_path
    if os.path.isfile(policy_path):
        with open(policy_path, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if isinstance(loaded, dict):
            policy_state = loaded
    return prior_state, policy_state



def stage_compress(args: argparse.Namespace) -> Dict[str, Any]:
    set_seed(int(args.seed))
    dist_ctx = init_distributed(str(args.device))
    output_dir = str(args.output_dir or f"./out/newthesis_compress_{now_tag()}")
    if is_main_process(dist_ctx):
        ensure_dir(output_dir)
    atlas = torch.load(args.atlas_path, map_location="cpu")
    sharing_policy_override = str(getattr(args, "sharing_policy_path", "")).strip()
    prior_state, sharing_policy = _load_sidecar_prior_from_atlas(
        str(args.atlas_path),
        sharing_policy_path=sharing_policy_override,
    )
    raw_layer_to_proto = [int(x) for x in atlas["layer_to_proto"].tolist()]
    layer_count = int(len(raw_layer_to_proto))
    layer_to_proto, regime_labels = _layer_groups_from_sharing_policy(
        sharing_policy,
        layer_count=layer_count,
    )
    layer_count = int(len(layer_to_proto))
    atlas_layer_signature = atlas.get("layer_signature", None)
    proto_seed_strategy = str(getattr(args, "proto_seed_strategy", "medoid")).strip().lower()
    policy_medoid_seed_layers = (
        _policy_medoid_seed_mapping(sharing_policy, layer_to_proto=layer_to_proto)
        if proto_seed_strategy == "policy_medoid"
        else None
    )
    seed_layer_for_proto, resolved_seed_strategy = resolve_proto_seed_layers(
        layer_to_proto=layer_to_proto,
        proto_seed_strategy=proto_seed_strategy,
        layer_signatures=atlas_layer_signature,
        policy_medoid_seed_layers=policy_medoid_seed_layers,
    )

    basis = prior_state.get("basis", None) if isinstance(prior_state, dict) else atlas.get("basis", None)
    if not torch.is_tensor(basis) or basis.dim() != 2:
        raise RuntimeError("atlas_state missing valid basis [D, r].")
    basis_cpu = basis.detach().to(dtype=torch.float32, device="cpu").contiguous()
    core_basis_mode = str(getattr(args, "core_basis_mode", "regime")).strip().lower()
    if core_basis_mode not in {"global", "regime"}:
        raise ValueError(f"Unsupported core_basis_mode={core_basis_mode!r}")
    regime_basis_cpu_map: Dict[str, torch.Tensor] = {
        name: basis_cpu for name in (list(dict.fromkeys([str(item).strip() or "llama_late" for item in regime_labels])) or ["llama_late"])
    }
    if isinstance(prior_state, dict) and core_basis_mode == "regime":
        prior_regime_basis = prior_state.get("regime_basis", None)
        if isinstance(prior_regime_basis, dict):
            for regime_name in list(regime_basis_cpu_map.keys()):
                candidate = prior_regime_basis.get(regime_name, None)
                if torch.is_tensor(candidate) and candidate.dim() == 2 and candidate.shape == basis_cpu.shape:
                    regime_basis_cpu_map[regime_name] = candidate.detach().to(dtype=torch.float32, device="cpu").contiguous()
    atlas_rank = int(basis_cpu.size(1))
    tau_eps = max(1e-12, float(getattr(args, "tau_eps", 1e-5)))
    prior_metric = prior_state.get("layer_metric_diag", None) if isinstance(prior_state, dict) else None
    if not torch.is_tensor(prior_metric):
        raise RuntimeError(
            "pass1 requires final_structure_prior.pt with a valid layer_metric_diag tensor; "
            "the tau_diag/layer_hist alpha fallback path has been removed."
        )
    if tuple(prior_metric.shape) != (int(layer_count), int(atlas_rank)):
        raise RuntimeError(
            "final_structure_prior layer_metric_diag shape mismatch for pass1: "
            f"expected {(int(layer_count), int(atlas_rank))}, got {tuple(prior_metric.shape)}."
        )
    layer_metric_diag_cpu = prior_metric.detach().to(dtype=torch.float32, device="cpu").contiguous()
    core_metric_diag_path = str(getattr(args, "core_metric_diag_path", "")).strip()
    core_metric_diag_mode = str(getattr(args, "core_metric_diag_mode", "covariance")).strip().lower()
    if core_metric_diag_mode not in {"covariance", "precision"}:
        raise ValueError(f"Unsupported core_metric_diag_mode={core_metric_diag_mode!r}")
    if core_metric_diag_path:
        external_metric = torch.load(core_metric_diag_path, map_location="cpu", weights_only=False)
        if isinstance(external_metric, dict):
            for key in ("layer_metric_precision_diag", "layer_metric_diag", "metric_diag"):
                candidate = external_metric.get(key, None)
                if torch.is_tensor(candidate):
                    external_metric = candidate
                    break
        if not torch.is_tensor(external_metric) or tuple(external_metric.shape) != (int(layer_count), int(atlas_rank)):
            observed = tuple(external_metric.shape) if torch.is_tensor(external_metric) else type(external_metric).__name__
            raise RuntimeError(
                f"external core metric shape mismatch: expected {(int(layer_count), int(atlas_rank))}, got {observed}"
            )
        layer_metric_diag_cpu = external_metric.detach().to(dtype=torch.float32, device="cpu").contiguous()
    core_metric_source = (
        "external_diagonal_precision"
        if core_metric_diag_path and core_metric_diag_mode == "precision"
        else "external_diagonal_covariance"
        if core_metric_diag_path
        else "regime_layer_metric_diag_from_final_structure_prior"
    )
    layer_reliability_cpu = torch.ones((layer_count,), dtype=torch.float32)
    if isinstance(prior_state, dict):
        prior_reliability = prior_state.get("layer_reliability", None)
        if torch.is_tensor(prior_reliability) and int(prior_reliability.numel()) == int(layer_count):
            layer_reliability_cpu = prior_reliability.detach().to(dtype=torch.float32, device="cpu").view(-1)
    sharing_policy_path = sharing_policy_override or os.path.join(os.path.dirname(os.path.abspath(str(args.atlas_path))), "sharing_policy.json")
    training_stage = str(getattr(args, "training_stage", "compress")).strip().lower()
    lambda_core = max(0.0, float(getattr(args, "lambda_core", getattr(args, "lambda_tau_max", 0.0))))
    core_lambda_schedule = str(getattr(args, "core_lambda_schedule", "constant")).strip().lower()
    core_lambda_warmup_ratio = min(
        1.0, max(0.0, float(getattr(args, "core_lambda_warmup_ratio", 0.1)))
    )
    core_lambda_cutoff_ratio = min(
        1.0, max(0.0, float(getattr(args, "core_lambda_cutoff_ratio", 0.5)))
    )
    _core_lambda_at_step(
        base_lambda=lambda_core,
        schedule=core_lambda_schedule,
        step=1,
        total_steps=max(1, int(getattr(args, "steps", 1))),
        warmup_ratio=core_lambda_warmup_ratio,
        cutoff_ratio=core_lambda_cutoff_ratio,
    )
    core_coordinate_mode = str(
        getattr(args, "core_coordinate_mode", "projected")
    ).strip().lower()
    if core_coordinate_mode not in {"projected", "ambient"}:
        raise ValueError(
            f"Unsupported core_coordinate_mode={core_coordinate_mode!r}; "
            "use projected or ambient."
        )
    lambda_hidden_mse = max(0.0, float(getattr(args, "lambda_hidden_mse", 1.0)))
    pass1_mode = training_stage == "pass1"
    if not pass1_mode:
        training_stage = "compress"
    layer_error_priority = atlas.get("layer_error_norm", None)
    if not torch.is_tensor(layer_error_priority) or layer_error_priority.dim() != 1 or int(layer_error_priority.numel()) != int(layer_count):
        layer_error_priority = torch.zeros((layer_count,), dtype=torch.float32)
    tau_layers_mode = str(getattr(args, "tau_layers", "all_shared_layers")).strip().lower()
    tau_layer_ids = _resolve_tau_layer_ids(
        mode=tau_layers_mode,
        layer_count=layer_count,
        seed_layer_for_proto=seed_layer_for_proto,
        layer_to_proto=layer_to_proto,
        layer_priority=layer_error_priority,
        extra_topk=int(getattr(args, "tau_extra_topk", 0)),
    )
    tau_token_rule = str(getattr(args, "tau_token_rule", "last_pred")).strip().lower()
    core_token_selection = str(getattr(args, "core_token_selection", "last_pred")).strip().lower()
    core_candidate_tokens = max(1, int(getattr(args, "core_candidate_tokens", 1)))
    core_iets_temperature = max(1e-4, float(getattr(args, "core_iets_temperature", 1.0)))
    core_iets_anchor_boost = float(getattr(args, "core_iets_anchor_boost", 0.0))
    core_iets_energy_alpha = float(getattr(args, "core_iets_energy_alpha", 1.0))
    core_iets_entropy_beta = float(getattr(args, "core_iets_entropy_beta", 0.0))
    core_iets_topk = max(1, int(getattr(args, "core_iets_topk", 1)))
    core_use_information_weighting = bool(getattr(args, "core_use_information_weighting", False))
    core_information_power = max(0.0, float(getattr(args, "core_information_power", 1.0)))
    lambda_geodesic_core = max(0.0, float(getattr(args, "lambda_geodesic_core", 0.0)))
    geodesic_core_max_layer_gap = max(1, int(getattr(args, "geodesic_core_max_layer_gap", 1)))
    lambda_relational_core = max(0.0, float(getattr(args, "lambda_relational_core", 0.0)))
    lambda_variance_core = max(0.0, float(getattr(args, "lambda_variance_core", 0.0)))
    variance_core_floor_ratio = max(0.0, float(getattr(args, "variance_core_floor_ratio", 0.70)))
    lambda_token_flow_core = max(0.0, float(getattr(args, "lambda_token_flow_core", 0.0)))
    lambda_token_turning_core = max(0.0, float(getattr(args, "lambda_token_turning_core", 0.0)))
    token_flow_energy_fraction = min(
        1.0, max(0.05, float(getattr(args, "token_flow_energy_fraction", 0.95)))
    )
    lambda_manifold_core = max(0.0, float(getattr(args, "lambda_manifold_core", 0.0)))
    manifold_core_temperature = max(1e-4, float(getattr(args, "manifold_core_temperature", 1.0)))
    lambda_delta_manifold_core = max(0.0, float(getattr(args, "lambda_delta_manifold_core", 0.0)))
    delta_manifold_core_temperature = max(1e-4, float(getattr(args, "delta_manifold_core_temperature", 1.0)))
    delta_manifold_risk_weight = max(0.0, float(getattr(args, "delta_manifold_risk_weight", 0.25)))
    on_policy_enable = bool(getattr(args, "on_policy_enable", False))
    on_policy_start_step = max(1, int(getattr(args, "on_policy_start_step", 1500)))
    on_policy_interval = max(1, int(getattr(args, "on_policy_interval", 5)))
    on_policy_batch_size = max(1, int(getattr(args, "on_policy_batch_size", 1)))
    on_policy_max_new_tokens = max(1, int(getattr(args, "on_policy_max_new_tokens", 192)))
    on_policy_generation_temperature = max(
        0.0, float(getattr(args, "on_policy_generation_temperature", 0.7))
    )
    on_policy_top_p = min(1.0, max(1e-5, float(getattr(args, "on_policy_top_p", 0.95))))
    on_policy_divergence = str(getattr(args, "on_policy_divergence", "js")).strip().lower()
    if on_policy_divergence not in {"js", "jensen_shannon", "forward_kl", "reverse_kl"}:
        raise ValueError(f"unsupported on_policy_divergence={on_policy_divergence!r}")
    on_policy_temperature = max(1e-4, float(getattr(args, "on_policy_temperature", 1.0)))
    on_policy_lambda_kd = max(0.0, float(getattr(args, "on_policy_lambda_kd", 1.0)))
    on_policy_lambda_core = max(0.0, float(getattr(args, "on_policy_lambda_core", 0.25)))
    on_policy_lambda_flow = max(0.0, float(getattr(args, "on_policy_lambda_flow", 0.0)))
    on_policy_lambda_turning = max(
        0.0, float(getattr(args, "on_policy_lambda_turning", 0.0))
    )
    on_policy_teacher_advantage_margin = float(
        getattr(args, "on_policy_teacher_advantage_margin", 0.0)
    )
    on_policy_final_answer_weight = max(
        1.0, float(getattr(args, "on_policy_final_answer_weight", 2.0))
    )
    on_policy_core_tokens = max(1, int(getattr(args, "on_policy_core_tokens", 8)))
    on_policy_ramp_steps = max(1, int(getattr(args, "on_policy_ramp_steps", 1000)))
    on_policy_enabled_resolved = bool(
        on_policy_enable
        and (
            on_policy_lambda_kd > 0.0
            or on_policy_lambda_core > 0.0
            or on_policy_lambda_flow > 0.0
            or on_policy_lambda_turning > 0.0
        )
    )
    lambda_layer_mixture = max(0.0, float(getattr(args, "lambda_layer_mixture", 0.0)))
    layer_mixture_entropy_tau = max(0.0, float(getattr(args, "layer_mixture_entropy_tau", 0.0)))
    layer_mixture_assignment_temperature = max(
        1e-4,
        float(getattr(args, "layer_mixture_assignment_temperature", 1.0)),
    )
    layer_mixture_gate_hidden = max(1, int(getattr(args, "layer_mixture_gate_hidden", 64)))
    layer_mixture_covariance_trace_normalize = bool(
        getattr(args, "layer_mixture_covariance_trace_normalize", False)
    )
    layer_mixture_delta_l2 = max(0.0, float(getattr(args, "layer_mixture_delta_l2", 0.0)))
    layer_mixture_lr = float(getattr(args, "layer_mixture_lr", 0.0))
    if layer_mixture_lr <= 0.0:
        layer_mixture_lr = float(getattr(args, "lr_adapter", float(args.lr)))
    layer_mixture_enabled = float(lambda_layer_mixture) > 0.0
    lambda_phase_adaptive_core = max(
        0.0, float(getattr(args, "lambda_phase_adaptive_core", 0.0))
    )
    phase_projector_bank_path = str(
        getattr(args, "phase_projector_bank_path", "")
    ).strip()
    phase_projector_mode = str(
        getattr(args, "phase_projector_mode", "phase")
    ).strip().lower()
    phase_projector_lr = float(getattr(args, "phase_projector_lr", 0.0))
    if phase_projector_lr <= 0.0:
        phase_projector_lr = float(getattr(args, "lr_adapter", float(args.lr)))
    phase_projector_enabled = lambda_phase_adaptive_core > 0.0
    if phase_projector_enabled and not phase_projector_bank_path:
        raise ValueError("--lambda_phase_adaptive_core>0 requires --phase_projector_bank_path")
    # Do not install/capture 28 layers of MLP hooks when the entire core term is
    # multiplied by zero. This was a large avoidable cost in SAGE job 259386.
    tau_enabled = (
        float(lambda_core) > 0.0
        or layer_mixture_enabled
        or phase_projector_enabled
    ) and len(tau_layer_ids) > 0
    hidden_mse_enabled = _normalize_distill_mode(getattr(args, "distill_mode", "ce")) == "ce_hidden_mse" and lambda_hidden_mse > 0.0
    teacher_free_ce = bool(getattr(args, "teacher_free_ce", False))
    requested_distill_mode = _normalize_distill_mode(getattr(args, "distill_mode", "ce"))
    if teacher_free_ce and (
        requested_distill_mode != "ce"
        or float(getattr(args, "lambda_kd", 0.0)) != 0.0
        or float(lambda_hidden_mse) != 0.0
        or bool(tau_enabled)
        or bool(on_policy_enabled_resolved)
    ):
        raise ValueError(
            "--teacher_free_ce requires distill_mode=ce, lambda_kd=0, "
            "lambda_hidden_mse=0, lambda_core=0, and on-policy disabled"
        )
    hidden_mse_layer_ids = list(tau_layer_ids)
    if hidden_mse_enabled and not hidden_mse_layer_ids:
        hidden_mse_layer_ids = list(range(layer_count))
    init_shared_student_ckpt = str(getattr(args, "init_shared_student_ckpt", "")).strip()
    residual_svd_init_mode = str(getattr(args, "residual_svd_init_mode", "none")).strip().lower()
    if residual_svd_init_mode not in {"none", "functional", "task_metric"}:
        raise ValueError(f"unsupported residual_svd_init_mode={residual_svd_init_mode!r}")
    if init_shared_student_ckpt and residual_svd_init_mode != "none":
        raise ValueError("residual SVD initialization cannot be combined with an explicit shared checkpoint")
    lr_bank = float(getattr(args, "lr_bank", float(args.lr)))
    lr_adapter = float(getattr(args, "lr_adapter", float(args.lr)))
    bank_freeze_steps = 0

    device = resolve_device(args.device, dist_ctx.local_rank if dist_ctx.enabled else -1)
    dtype = get_target_dtype(device)
    basis_device = basis_cpu.to(device=device, dtype=torch.float32)
    regime_basis_device_map: Dict[str, torch.Tensor] = {
        name: tensor.to(device=device, dtype=torch.float32)
        for name, tensor in regime_basis_cpu_map.items()
    }
    layer_metric_diag_device = layer_metric_diag_cpu.to(device=device, dtype=torch.float32)
    layer_reliability_device = layer_reliability_cpu.to(device=device, dtype=torch.float32)
    layer_information_weight_cpu = torch.ones_like(layer_metric_diag_cpu)
    if isinstance(prior_state, dict):
        prior_information_weight = prior_state.get("layer_information_weight", None)
        if torch.is_tensor(prior_information_weight) and tuple(prior_information_weight.shape) == tuple(layer_metric_diag_cpu.shape):
            layer_information_weight_cpu = prior_information_weight.detach().to(dtype=torch.float32, device="cpu").contiguous()
    layer_information_weight_cpu = layer_information_weight_cpu.clamp(min=1e-6)
    layer_information_weight_cpu = layer_information_weight_cpu / layer_information_weight_cpu.mean(dim=1, keepdim=True).clamp(min=1e-6)
    layer_information_weight_device = layer_information_weight_cpu.to(device=device, dtype=torch.float32)
    manifold_anchor_z_device: Optional[torch.Tensor] = None
    manifold_anchor_phi_device: Optional[torch.Tensor] = None
    manifold_sigma2_device: Optional[torch.Tensor] = None
    manifold_core_enabled = False
    manifold_anchor_count = 0
    manifold_dim = 0
    if lambda_manifold_core > 0.0 and isinstance(prior_state, dict):
        diffusion_state = prior_state.get("teacher_diffusion_manifold", None)
        if isinstance(diffusion_state, dict):
            anchor_z = diffusion_state.get("layer_anchor_z", None)
            anchor_phi = diffusion_state.get("layer_anchor_phi", None)
            sigma2 = diffusion_state.get("layer_kernel_sigma2", None)
            if (
                torch.is_tensor(anchor_z)
                and torch.is_tensor(anchor_phi)
                and torch.is_tensor(sigma2)
                and anchor_z.dim() == 3
                and anchor_phi.dim() == 3
                and sigma2.dim() == 1
                and int(anchor_z.size(0)) == int(layer_count)
                and int(anchor_phi.size(0)) == int(layer_count)
                and int(sigma2.numel()) == int(layer_count)
                and int(anchor_z.size(1)) == int(anchor_phi.size(1))
                and int(anchor_z.size(2)) == int(layer_metric_diag_cpu.size(1))
            ):
                manifold_anchor_z_device = anchor_z.detach().to(device=device, dtype=torch.float32).contiguous()
                manifold_anchor_phi_device = anchor_phi.detach().to(device=device, dtype=torch.float32).contiguous()
                manifold_sigma2_device = sigma2.detach().to(device=device, dtype=torch.float32).contiguous().clamp(min=float(tau_eps))
                manifold_anchor_count = int(anchor_z.size(1))
                manifold_dim = int(anchor_phi.size(2))
                manifold_core_enabled = manifold_anchor_count > 0 and manifold_dim > 0
    if lambda_manifold_core > 0.0 and not manifold_core_enabled:
        print(
            "[Compress][Warn] lambda_manifold_core>0 but final_structure_prior.pt has no valid teacher_diffusion_manifold; "
            "manifold_core will be disabled.",
            flush=True,
        )
        lambda_manifold_core = 0.0
    delta_manifold_anchor_device: Optional[torch.Tensor] = None
    delta_manifold_phi_device: Optional[torch.Tensor] = None
    delta_manifold_origin_phi_device: Optional[torch.Tensor] = None
    delta_manifold_sigma2_device: Optional[torch.Tensor] = None
    delta_manifold_risk_device: Optional[torch.Tensor] = None
    delta_manifold_core_enabled = False
    delta_manifold_anchor_count = 0
    delta_manifold_dim = 0
    if lambda_delta_manifold_core > 0.0 and isinstance(prior_state, dict):
        delta_state = prior_state.get("delta_h_error_manifold", None)
        if isinstance(delta_state, dict):
            anchor_e = delta_state.get("layer_anchor_e", None)
            anchor_phi = delta_state.get("layer_anchor_phi", None)
            origin_phi = delta_state.get("layer_origin_phi", None)
            sigma2 = delta_state.get("layer_kernel_sigma2", None)
            anchor_risk = delta_state.get("layer_anchor_risk", None)
            if (
                torch.is_tensor(anchor_e)
                and torch.is_tensor(anchor_phi)
                and torch.is_tensor(origin_phi)
                and torch.is_tensor(sigma2)
                and torch.is_tensor(anchor_risk)
                and anchor_e.dim() == 3
                and anchor_phi.dim() == 3
                and origin_phi.dim() == 2
                and sigma2.dim() == 1
                and anchor_risk.dim() == 2
                and int(anchor_e.size(0)) == int(layer_count)
                and int(anchor_phi.size(0)) == int(layer_count)
                and int(origin_phi.size(0)) == int(layer_count)
                and int(sigma2.numel()) == int(layer_count)
                and int(anchor_risk.size(0)) == int(layer_count)
                and int(anchor_e.size(1)) == int(anchor_phi.size(1))
                and int(anchor_risk.size(1)) == int(anchor_e.size(1))
                and int(anchor_e.size(2)) == int(layer_metric_diag_cpu.size(1))
                and int(origin_phi.size(1)) == int(anchor_phi.size(2))
            ):
                delta_manifold_anchor_device = anchor_e.detach().to(device=device, dtype=torch.float32).contiguous()
                delta_manifold_phi_device = anchor_phi.detach().to(device=device, dtype=torch.float32).contiguous()
                delta_manifold_origin_phi_device = origin_phi.detach().to(device=device, dtype=torch.float32).contiguous()
                delta_manifold_sigma2_device = sigma2.detach().to(device=device, dtype=torch.float32).contiguous().clamp(min=float(tau_eps))
                delta_manifold_risk_device = anchor_risk.detach().to(device=device, dtype=torch.float32).contiguous().clamp(min=0.0)
                delta_manifold_anchor_count = int(anchor_e.size(1))
                delta_manifold_dim = int(anchor_phi.size(2))
                delta_manifold_core_enabled = delta_manifold_anchor_count > 0 and delta_manifold_dim > 0
    if lambda_delta_manifold_core > 0.0 and not delta_manifold_core_enabled:
        print(
            "[Compress][Warn] lambda_delta_manifold_core>0 but final_structure_prior.pt has no valid "
            "delta_h_error_manifold; delta_manifold_core will be disabled.",
            flush=True,
        )
        lambda_delta_manifold_core = 0.0
    atlas_hidden_size = int(basis_cpu.size(0))

    teacher_deploy_bundle = str(getattr(args, "teacher_deploy_bundle", "")).strip()
    teacher_bundle: Optional[Dict[str, Any]] = None
    if teacher_deploy_bundle:
        teacher_deploy_bundle = os.path.abspath(teacher_deploy_bundle)
        if not os.path.isfile(teacher_deploy_bundle):
            raise FileNotFoundError(f"teacher deploy bundle not found: {teacher_deploy_bundle}")
        teacher_bundle = torch.load(teacher_deploy_bundle, map_location="cpu", weights_only=False)
        if not isinstance(teacher_bundle, dict) or "shared_student" not in teacher_bundle or "base_model" not in teacher_bundle:
            raise ValueError(
                "--teacher_deploy_bundle must be a Phase-1.5 deploy bundle containing "
                "'base_model' and 'shared_student'."
            )
    teacher_model_path = (
        str(teacher_bundle["base_model"]).strip()
        if teacher_bundle is not None
        else str(args.teacher_ckpt).strip() or str(args.base_model).strip()
    )
    student_model_path = str(args.base_model).strip() or teacher_model_path
    if teacher_free_ce:
        teacher_model_path = ""
    if not teacher_model_path and not teacher_free_ce:
        raise ValueError("teacher model path is required (set --teacher_ckpt or --base_model).")
    if not student_model_path:
        raise ValueError("student model path is required (set --base_model or --teacher_ckpt).")

    setup_pbar = step_progress(total=5, desc="[Compress] setup")
    print("[Compress] loading tokenizer...", flush=True)
    tokenizer_source = str(args.tokenizer_name_or_path).strip() or student_model_path or teacher_model_path
    tokenizer = load_tokenizer(
        tokenizer_source,
        trust_remote_code=bool(getattr(args, "trust_remote_code", False)),
    )
    eos_token_id = tokenizer.eos_token_id
    if setup_pbar is not None:
        setup_pbar.update(1)

    teacher: Optional[nn.Module] = None
    if teacher_free_ce:
        if teacher_bundle is not None:
            raise ValueError("--teacher_free_ce cannot be combined with --teacher_deploy_bundle")
        print("[Compress] teacher-free CE enabled: teacher loading and forward are skipped", flush=True)
        teacher_quant_report = {"requested": False, "enabled": False, "skipped_teacher_free_ce": True}
    elif teacher_bundle is not None:
        print(f"[Compress] loading shared deploy-bundle teacher: {teacher_deploy_bundle}", flush=True)
        teacher, teacher_quant_report = _build_shared_model_for_eval(
            base_model=teacher_model_path,
            atlas_payload=teacher_bundle.get("atlas", {}),
            shared_payload=teacher_bundle["shared_student"],
            quant_bank_int4=teacher_bundle.get("quant_bank_int4"),
            use_quant_bank_int4=False,
            device=device,
            dtype=dtype,
            trust_remote_code=bool(getattr(args, "trust_remote_code", False)),
        )
    else:
        print("[Compress] loading native teacher...", flush=True)
        teacher = AutoModelForCausalLM.from_pretrained(
            teacher_model_path,
            torch_dtype=dtype,
            trust_remote_code=bool(getattr(args, "trust_remote_code", False)),
        ).to(device)
        teacher_quant_report = {"requested": False, "enabled": False}
    if teacher is not None:
        _set_gradient_checkpointing(
            teacher,
            enabled=bool(getattr(args, "teacher_gradient_checkpointing", False)),
            require_input_grads=False,
        )
        _validate_supported_llama_config(teacher, label="compress teacher")
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad_(False)
    if setup_pbar is not None:
        setup_pbar.update(1)

    print("[Compress] loading student...", flush=True)
    student = AutoModelForCausalLM.from_pretrained(
        student_model_path,
        torch_dtype=dtype,
        trust_remote_code=bool(getattr(args, "trust_remote_code", False)),
    ).to(device)
    _set_gradient_checkpointing(
        student,
        enabled=bool(getattr(args, "student_gradient_checkpointing", False)),
        require_input_grads=True,
    )
    print(
        f"[Compress] gradient_checkpointing student={bool(getattr(args, 'student_gradient_checkpointing', False))} "
        f"teacher={bool(getattr(args, 'teacher_gradient_checkpointing', False))}",
        flush=True,
    )
    _validate_supported_llama_config(student, label="compress student/base")
    teacher_layers = int(len(_resolve_layers(teacher))) if teacher is not None else int(len(_resolve_layers(student)))
    student_layers = int(len(_resolve_layers(student)))
    student_hidden = int(getattr(student.config, "hidden_size", 0) or 0)
    teacher_hidden = int(getattr(teacher.config, "hidden_size", 0) or 0) if teacher is not None else student_hidden
    if teacher_layers != layer_count or student_layers != layer_count or teacher_hidden != atlas_hidden_size or student_hidden != atlas_hidden_size:
        raise ValueError(
            "Model architecture mismatch for Phase-1.5 shared-FFN compression.\n"
            f"- atlas expects layers={layer_count}, hidden_size={atlas_hidden_size}\n"
            f"- teacher({teacher_model_path}) layers={teacher_layers}, hidden_size={teacher_hidden}\n"
            f"- student/base_model({student_model_path}) layers={student_layers}, hidden_size={student_hidden}\n"
            "This pipeline does not support depth/width compression; set BASE_MODEL to match TEACHER_CKPT (or rebuild atlas with a matching teacher)."
        )
    init_shared = False
    if init_shared_student_ckpt:
        print(f"[Compress] loading shared init checkpoint: {init_shared_student_ckpt}", flush=True)
        init_payload = _load_shared_payload_from_ckpt(init_shared_student_ckpt)
        loaded_layer_to_proto = load_shared_state(
            student,
            init_payload,
            lora_rank=int(args.lora_rank),
            lora_alpha=float(args.lora_alpha),
        )
        if [int(x) for x in loaded_layer_to_proto] != [int(x) for x in layer_to_proto]:
            raise ValueError("init_shared_student_ckpt layer_to_proto does not match atlas layer_to_proto.")
        init_shared = True
    else:
        apply_shared_ffn(
            student,
            layer_to_proto=layer_to_proto,
            lora_rank=int(args.lora_rank),
            lora_alpha=float(args.lora_alpha),
            proto_seed_strategy=proto_seed_strategy,
            layer_signatures=atlas_layer_signature if torch.is_tensor(atlas_layer_signature) else None,
            policy_medoid_seed_layers=policy_medoid_seed_layers,
            sharing_parameterization=str(
                getattr(args, "sharing_parameterization", "full_parallel")
            ),
            use_layer_scalar=bool(getattr(args, "use_layer_scalar", True)),
            adapter_every_layer=bool(getattr(args, "adapter_every_layer", False)),
        )
    print(
        f"[Compress] proto seed strategy={proto_seed_strategy} resolved={resolved_seed_strategy} "
        f"seed_layers={seed_layer_for_proto}",
        flush=True,
    )
    internal_weight_delta_init_report: Dict[str, Any] = {
        "enabled": False,
        "mode": str(getattr(args, "sharing_parameterization", "full_parallel")),
    }
    if (
        str(getattr(args, "sharing_parameterization", "full_parallel")).strip().lower()
        == "internal_weight_delta"
        and not init_shared
    ):
        if teacher is None:
            raise ValueError("internal_weight_delta initialization requires a teacher")
        internal_weight_delta_init_report = initialize_internal_weight_delta_svd(
            student=student,
            teacher=teacher,
            layer_to_proto=layer_to_proto,
            seed_layer_for_proto=seed_layer_for_proto,
            oversample=int(getattr(args, "internal_delta_svd_oversample", 16)),
            niter=int(getattr(args, "internal_delta_svd_niter", 2)),
            seed=int(args.seed),
        )
    freeze_backbone_for_shared_train(student)
    student.train()
    layer_mixture_transport: Optional[nn.Module] = None
    if layer_mixture_enabled:
        mixture_covariance = (
            layer_metric_diag_device.reciprocal().clamp(min=tau_eps)
            if core_metric_diag_mode == "precision"
            else layer_metric_diag_device
        )
        layer_mixture_transport = LayerMixtureVariationalTransport(
            layer_to_proto=layer_to_proto,
            layer_covariance_diag=mixture_covariance,
            layer_reliability=layer_reliability_device,
            gate_hidden=layer_mixture_gate_hidden,
            assignment_temperature=layer_mixture_assignment_temperature,
            entropy_tau=layer_mixture_entropy_tau,
            covariance_eps=tau_eps,
            covariance_trace_normalize=layer_mixture_covariance_trace_normalize,
            delta_l2=layer_mixture_delta_l2,
        ).to(device=device, dtype=torch.float32)
        layer_mixture_transport.train()
    phase_projector_bank: Optional[nn.Module] = None
    if phase_projector_enabled:
        projector_state = torch.load(
            phase_projector_bank_path, map_location="cpu", weights_only=False
        )
        phase_projector_bank = PhaseAdaptiveProjectorBank(
            projector_state, mode=phase_projector_mode
        ).to(device=device, dtype=torch.float32)
        missing_layers = sorted(
            set(int(value) for value in tau_layer_ids)
            - set(unwrap_model(phase_projector_bank).layers)
        )
        if missing_layers:
            raise ValueError(
                f"phase projector bank misses supervised layers {missing_layers}"
            )
        phase_projector_bank.train()
    if dist_ctx.enabled:
        ddp_device_ids = [dist_ctx.local_rank] if device.type == "cuda" else None
        student = DDP(
            student,
            device_ids=ddp_device_ids,
            output_device=dist_ctx.local_rank if device.type == "cuda" else None,
            find_unused_parameters=False,
        )
        if layer_mixture_transport is not None:
            layer_mixture_transport = DDP(
                layer_mixture_transport,
                device_ids=ddp_device_ids,
                output_device=dist_ctx.local_rank if device.type == "cuda" else None,
                find_unused_parameters=False,
            )
        if phase_projector_bank is not None:
            phase_projector_bank = DDP(
                phase_projector_bank,
                device_ids=ddp_device_ids,
                output_device=dist_ctx.local_rank if device.type == "cuda" else None,
                find_unused_parameters=True,
            )
    if setup_pbar is not None:
        setup_pbar.update(1)

    print("[Compress] tokenizing training data...", flush=True)
    loader, tokenized_count = load_tokenized_loader(
        tokenizer=tokenizer,
        data_path=args.data_path,
        max_records=int(args.max_records),
        cutoff_len=int(args.cutoff_len),
        batch_size=int(args.batch_size),
        seed=int(args.seed),
        shuffle_records=True,
        distributed_num_replicas=dist_ctx.world_size if dist_ctx.enabled else 1,
        distributed_rank=dist_ctx.rank if dist_ctx.enabled else 0,
        prompt_mode=str(getattr(args, "training_prompt_mode", "legacy_sft")),
    )
    print(f"[Compress] tokenized={int(tokenized_count)} batch_size={int(args.batch_size)}", flush=True)
    if setup_pbar is not None:
        setup_pbar.update(1)

    generic_replay_data_path = str(getattr(args, "generic_replay_data_path", "")).strip()
    lambda_generic_replay = max(0.0, float(getattr(args, "lambda_generic_replay", 0.0)))
    generic_replay_interval = max(0, int(getattr(args, "generic_replay_interval", 0)))
    generic_replay_batch_size = int(getattr(args, "generic_replay_batch_size", 0))
    if generic_replay_batch_size <= 0:
        generic_replay_batch_size = int(args.batch_size)
    generic_replay_max_records = max(0, int(getattr(args, "generic_replay_max_records", 0)))
    generic_replay_loader: Optional[DataLoader] = None
    generic_replay_iterator = None
    generic_replay_tokenized_count = 0
    generic_replay_enabled = bool(
        generic_replay_data_path
        and lambda_generic_replay > 0.0
        and generic_replay_interval > 0
    )
    if generic_replay_enabled:
        print("[Compress] tokenizing generic replay data...", flush=True)
        generic_replay_loader, generic_replay_tokenized_count = load_tokenized_loader(
            tokenizer=tokenizer,
            data_path=generic_replay_data_path,
            max_records=generic_replay_max_records,
            cutoff_len=int(args.cutoff_len),
            batch_size=generic_replay_batch_size,
            seed=int(args.seed) + 1701,
            shuffle_records=True,
            distributed_num_replicas=dist_ctx.world_size if dist_ctx.enabled else 1,
            distributed_rank=dist_ctx.rank if dist_ctx.enabled else 0,
            prompt_mode="legacy_sft",
        )
        generic_replay_iterator = iter(generic_replay_loader)
        print(
            f"[Compress] generic replay enabled interval={generic_replay_interval} "
            f"lambda={lambda_generic_replay:.4f} tokenized={generic_replay_tokenized_count} "
            f"batch_size={generic_replay_batch_size}",
            flush=True,
        )
    elif is_main_process(dist_ctx):
        print("[Compress] generic replay disabled", flush=True)

    residual_svd_init_report: Dict[str, Any] = {
        "enabled": False,
        "mode": residual_svd_init_mode,
    }
    if residual_svd_init_mode != "none":
        if teacher is None:
            raise ValueError("residual SVD initialization requires a teacher")
        residual_svd_init_report = initialize_shared_ffn_functional_residual_svd(
            student=student,
            teacher=teacher,
            loader=loader,
            layer_to_proto=layer_to_proto,
            seed_layer_for_proto=seed_layer_for_proto,
            mode=residual_svd_init_mode,
            global_records=int(getattr(args, "residual_svd_init_records", 512)),
            tokens_per_record=int(
                getattr(args, "residual_svd_tokens_per_record", 10)
            ),
            max_fit_tokens=int(getattr(args, "residual_svd_max_fit_tokens", 1024)),
            ridge=float(getattr(args, "residual_svd_init_ridge", 1e-3)),
            oversample=int(getattr(args, "residual_svd_init_oversample", 16)),
            metric_complement_floor=float(getattr(args, "residual_svd_metric_complement_floor", 0.1)),
            regime_labels=regime_labels,
            regime_basis_cpu_map=regime_basis_cpu_map,
            layer_metric_diag_cpu=layer_metric_diag_cpu,
            core_metric_diag_mode=core_metric_diag_mode,
            core_metric_trace_normalize=bool(getattr(args, "core_metric_trace_normalize", False)),
            eos_token_id=eos_token_id,
            seed=int(args.seed),
            device=device,
        )

    val_loader: Optional[DataLoader] = None
    val_data_path = str(getattr(args, "val_data_path", "")).strip()
    val_every = int(getattr(args, "val_every", 0))
    val_max_records = int(getattr(args, "val_max_records", 0))
    val_max_batches = int(getattr(args, "val_max_batches", 0))
    val_seed = int(getattr(args, "val_seed", int(args.seed)))
    val_batch_size = int(getattr(args, "val_batch_size", 0))
    if val_batch_size <= 0:
        val_batch_size = int(args.batch_size)
    val_tokenized_count = 0
    if val_data_path and val_every > 0 and is_main_process(dist_ctx):
        print("[Compress] tokenizing validation data...", flush=True)
        val_loader, val_tokenized_count = load_tokenized_loader(
            tokenizer=tokenizer,
            data_path=val_data_path,
            max_records=val_max_records,
            cutoff_len=int(args.cutoff_len),
            batch_size=int(val_batch_size),
            seed=val_seed,
            shuffle_records=False,
            prompt_mode=str(getattr(args, "training_prompt_mode", "legacy_sft")),
        )
        print(
            f"[Compress] validation enabled val_every={val_every} "
            f"val_tokenized={int(val_tokenized_count)} val_batch_size={val_batch_size} "
            f"val_max_batches={val_max_batches}",
            flush=True,
        )
    elif is_main_process(dist_ctx):
        print("[Compress] validation disabled (set --val_data_path and --val_every>0 to enable)", flush=True)

    param_groups = _collect_trainable_groups(student)
    bank_params = param_groups["bank"]
    adapter_params = param_groups["adapter"]
    mixture_params = (
        [parameter for parameter in unwrap_model(layer_mixture_transport).parameters() if parameter.requires_grad]
        if layer_mixture_transport is not None
        else []
    )
    phase_projector_params = (
        [
            parameter
            for parameter in unwrap_model(phase_projector_bank).parameters()
            if parameter.requires_grad
        ]
        if phase_projector_bank is not None
        else []
    )
    optimizer_groups = []
    if bank_params:
        optimizer_groups.append({"params": bank_params, "lr": float(lr_bank), "name": "bank"})
    if adapter_params:
        optimizer_groups.append({"params": adapter_params, "lr": float(lr_adapter), "name": "adapter"})
    if mixture_params:
        optimizer_groups.append({"params": mixture_params, "lr": float(layer_mixture_lr), "name": "mixture"})
    if phase_projector_params:
        optimizer_groups.append(
            {
                "params": phase_projector_params,
                "lr": float(phase_projector_lr),
                "name": "phase_projector",
            }
        )
    if not optimizer_groups:
        raise RuntimeError("No trainable parameters found for compression.")
    optimizer = torch.optim.AdamW(optimizer_groups, lr=float(args.lr), weight_decay=float(args.weight_decay))
    total_steps = int(args.steps)
    grad_accum_steps = max(1, int(getattr(args, "grad_accum_steps", 1)))
    scheduler, scheduler_meta = _build_lr_scheduler(
        optimizer=optimizer,
        schedule=str(getattr(args, "lr_schedule", "warmup_cosine")),
        total_steps=int(total_steps),
        warmup_steps=int(getattr(args, "lr_warmup_steps", 0)),
        warmup_ratio=float(getattr(args, "lr_warmup_ratio", 0.1)),
        min_lr_ratio=float(getattr(args, "lr_min_ratio", 0.1)),
    )
    print(
        f"[Compress] trainable bank_params={sum(int(p.numel()) for p in bank_params)} "
        f"adapter_params={sum(int(p.numel()) for p in adapter_params)} "
        f"mixture_params={sum(int(p.numel()) for p in mixture_params)} "
        f"phase_projector_params={sum(int(p.numel()) for p in phase_projector_params)} "
        f"init_shared={init_shared}",
        flush=True,
    )
    print(
        f"[Compress] batch per_rank={int(args.batch_size)} world_size={int(dist_ctx.world_size)} "
        f"grad_accum_steps={grad_accum_steps} "
        f"effective_global_batch={int(args.batch_size) * int(dist_ctx.world_size) * grad_accum_steps}",
        flush=True,
    )
    print(
        f"[Compress] private_adapter rank={int(args.lora_rank)} "
        f"alpha={float(args.lora_alpha):.6g} scaling={float(args.lora_alpha) / max(1, int(args.lora_rank)):.6g}",
        flush=True,
    )
    if layer_mixture_transport is not None:
        mixture_config = unwrap_model(layer_mixture_transport).config_dict()
        print(
            f"[Compress] layer_mixture enabled=True lambda={lambda_layer_mixture:.6g} "
            f"entropy_tau={layer_mixture_entropy_tau:.6g} "
            f"assignment_temperature={layer_mixture_assignment_temperature:.6g} "
            f"gate_hidden={layer_mixture_gate_hidden} lr={layer_mixture_lr:.6g} "
            f"covariance_trace_normalize={layer_mixture_covariance_trace_normalize} "
            f"groups={mixture_config['shared_groups']}",
            flush=True,
        )
    if phase_projector_bank is not None:
        print(
            f"[Compress] phase_projector enabled=True "
            f"lambda={lambda_phase_adaptive_core:.6g} "
            f"lr={phase_projector_lr:.6g} "
            f"config={unwrap_model(phase_projector_bank).config_dict()}",
            flush=True,
        )
    print(
        f"[Compress] lr schedule={scheduler_meta['schedule']} base_lr={float(args.lr):.6g} "
        f"lr_bank={float(lr_bank):.6g} lr_adapter={float(lr_adapter):.6g} "
        f"warmup_steps={int(scheduler_meta['warmup_steps'])} min_lr_ratio={float(scheduler_meta['min_lr_ratio']):.4f}",
        flush=True,
    )
    print(
        f"[Compress] stage={training_stage} core enabled={tau_enabled} layers={tau_layers_mode} "
        f"count={len(tau_layer_ids)} lambda_core={lambda_core:.4f} "
        f"metric_source=regime_layer_metric_diag_from_final_structure_prior "
        f"metric_whitening={bool(getattr(args, 'core_use_metric_whitening', True))} "
        f"metric_trace_normalize={bool(getattr(args, 'core_metric_trace_normalize', False))} "
        f"reliability_weighting={bool(getattr(args, 'core_use_reliability_weighting', True))}",
        flush=True,
    )
    print(
        f"[Compress] manifold_core enabled={manifold_core_enabled} "
        f"lambda_manifold_core={float(lambda_manifold_core):.4f} "
        f"temperature={float(manifold_core_temperature):.4f} "
        f"anchors={int(manifold_anchor_count)} manifold_dim={int(manifold_dim)} "
        f"lambda_geodesic_core={float(lambda_geodesic_core):.4f}",
        flush=True,
    )
    print(
        f"[Compress] delta_manifold_core enabled={delta_manifold_core_enabled} "
        f"lambda_delta_manifold_core={float(lambda_delta_manifold_core):.4f} "
        f"temperature={float(delta_manifold_core_temperature):.4f} "
        f"risk_weight={float(delta_manifold_risk_weight):.4f} "
        f"anchors={int(delta_manifold_anchor_count)} manifold_dim={int(delta_manifold_dim)}",
        flush=True,
    )
    print(
        f"[Compress] core_token_selection={core_token_selection} token_rule={tau_token_rule} "
        f"candidate_tokens={core_candidate_tokens} iets_temperature={core_iets_temperature:.4f} "
        f"anchor_boost={core_iets_anchor_boost:.4f} energy_alpha={core_iets_energy_alpha:.4f} "
        f"entropy_beta={core_iets_entropy_beta:.4f} iets_topk={core_iets_topk}",
        flush=True,
    )
    print(
        f"[Compress] objective={_pretty_distill_mode(getattr(args, 'distill_mode', 'ce'))} "
        f"lambda_ce={float(getattr(args, 'lambda_ce', 1.0)):.4f} "
        f"lambda_kd={float(getattr(args, 'lambda_kd', 0.0)):.4f} "
        f"lambda_hidden_mse={float(lambda_hidden_mse):.4f} "
        f"kd_temperature={float(getattr(args, 'kd_temperature', 2.0)):.4f}",
        flush=True,
    )
    print(f"[Compress] sharing_policy={sharing_policy_path}", flush=True)

    ckpt_dir = os.path.join(output_dir, "checkpoints")
    ensure_dir(ckpt_dir)
    if setup_pbar is not None:
        setup_pbar.update(1)
        setup_pbar.close()

    distill_mode = _normalize_distill_mode(getattr(args, "distill_mode", "ce"))
    ce_coeff = float(args.lambda_ce)
    kd_coeff = float(getattr(args, "lambda_kd", 0.0))
    kd_temperature = float(getattr(args, "kd_temperature", 2.0))
    loss_scope = str(getattr(args, "loss_scope", "all")).strip().lower()
    loss_exclude_eos = bool(getattr(args, "loss_exclude_eos", True))
    sage_config: Dict[str, Any] = {
        "topk": max(1, int(getattr(args, "sage_topk", 32))),
        "gain_margin": float(getattr(args, "sage_gain_margin", 0.0)),
        "gain_temperature": max(1e-6, float(getattr(args, "sage_gain_temperature", 0.25))),
        "confidence_margin": float(getattr(args, "sage_confidence_margin", 0.0)),
        "confidence_temperature": max(1e-6, float(getattr(args, "sage_confidence_temperature", 1.0))),
        "confidence_power": max(0.0, float(getattr(args, "sage_confidence_power", 1.0))),
        "require_teacher_correct": bool(getattr(args, "sage_require_teacher_correct", True)),
        "rate_warmup_ratio": min(1.0, max(0.0, float(getattr(args, "sage_rate_warmup_ratio", 0.10)))),
        "rate_decay_start_ratio": min(1.0, max(0.0, float(getattr(args, "sage_rate_decay_start_ratio", 0.55)))),
        "rate_min_ratio": min(1.0, max(0.0, float(getattr(args, "sage_rate_min_ratio", 0.10)))),
    }
    val_selection_metric = str(getattr(args, "val_selection_metric", "loss")).strip().lower()
    val_include_step0_candidate = bool(getattr(args, "val_include_step0_candidate", False))
    val_min_improvement = max(0.0, float(getattr(args, "val_min_improvement", 0.0)))
    if distill_mode == "ce":
        kd_coeff = 0.0
    elif kd_coeff <= 0.0:
        print(f"[Compress][Warn] distill_mode={distill_mode} but lambda_kd<=0; teacher channel will be disabled.", flush=True)
    if distill_mode == "ce_hidden_mse":
        kd_coeff = 0.0
    step = 0
    running = {
        "loss": 0.0,
        "ce": 0.0,
        "response_ce": 0.0,
        "decision_ce": 0.0,
        "kd": 0.0,
        "sage_gate": 0.0,
        "sage_active": 0.0,
        "sage_teacher_correct": 0.0,
        "sage_rate": 0.0,
        "hidden": 0.0,
        "core": 0.0,
        "point_core": 0.0,
        "geodesic_core": 0.0,
        "relational_core": 0.0,
        "variance_core": 0.0,
        "manifold_core": 0.0,
        "delta_manifold_core": 0.0,
        "generic_replay_ce": 0.0,
        "generic_replay_active": 0.0,
        "on_policy_loss": 0.0,
        "on_policy_kd": 0.0,
        "on_policy_core": 0.0,
        "on_policy_active": 0.0,
        "on_policy_gate": 0.0,
        "on_policy_tokens": 0.0,
        "layer_mixture": 0.0,
        "layer_mixture_nll": 0.0,
        "layer_mixture_entropy": 0.0,
        "layer_mixture_posterior_entropy": 0.0,
        "layer_mixture_effective_components": 0.0,
        "layer_mixture_max_probability": 0.0,
        "layer_mixture_delta_l2": 0.0,
        "phase_adaptive_core": 0.0,
        "phase_adaptive_cosine": 0.0,
        "core_lambda": 0.0,
        "lr": 0.0,
        "lr_bank": 0.0,
        "lr_adapter": 0.0,
        "lr_mixture": 0.0,
        "lr_phase_projector": 0.0,
    }
    log_every_steps = max(1, int(args.log_every))
    val_history: List[Dict[str, Any]] = []
    last_core_loss = 0.0
    last_point_core_loss = 0.0
    last_geodesic_core_loss = 0.0
    last_relational_core_loss = 0.0
    last_variance_core_loss = 0.0
    last_manifold_core_loss = 0.0
    last_delta_manifold_core_loss = 0.0
    last_generic_replay_ce_loss = 0.0
    last_on_policy_loss = 0.0
    last_on_policy_kd_loss = 0.0
    last_on_policy_core_loss = 0.0
    last_on_policy_gate = 0.0
    on_policy_activation_count = 0
    on_policy_activation_token_count = 0.0
    on_policy_activation_gate_sum = 0.0
    last_core_lambda = 0.0
    last_response_ce_loss = 0.0
    last_decision_ce_loss = 0.0
    last_sage_gate_mean = 0.0
    last_sage_active_fraction = 0.0
    last_layer_mixture_loss = 0.0
    last_layer_mixture_nll = 0.0
    last_layer_mixture_entropy = 0.0
    last_layer_mixture_posterior_entropy = 0.0
    last_layer_mixture_effective_components = 0.0
    last_layer_mixture_max_probability = 0.0
    last_layer_mixture_delta_l2 = 0.0
    last_phase_adaptive_core_loss = 0.0
    last_phase_adaptive_cosine = 0.0
    best_val_loss = float("inf")
    best_val_step = -1
    best_val_ckpt_path = os.path.join(ckpt_dir, "compress_best_val.pt")
    val_subspace_viz_dir = os.path.join(output_dir, "compress_val_subspace")
    train_sampler = loader.sampler if isinstance(getattr(loader, "sampler", None), DistributedSampler) else None
    epoch_idx = 0
    pbar = step_progress(
        total=total_steps,
        desc="[Compress] train",
        miniters=log_every_steps,
    ) if is_main_process(dist_ctx) else None
    pbar_pending = 0
    core_use_metric_whitening = bool(getattr(args, "core_use_metric_whitening", True))
    core_metric_trace_normalize = bool(getattr(args, "core_metric_trace_normalize", False))
    core_metric_diag_is_precision = str(getattr(args, "core_metric_diag_mode", "covariance")).strip().lower() == "precision"
    core_use_reliability_weighting = bool(getattr(args, "core_use_reliability_weighting", True))
    print(
        f"[Compress] information_channel prompt_mode={str(getattr(args, 'training_prompt_mode', 'legacy_sft'))} "
        f"loss_scope={loss_scope} exclude_eos={loss_exclude_eos} val_selection={val_selection_metric} "
        f"include_step0={val_include_step0_candidate} min_improvement={val_min_improvement:.6g}",
        flush=True,
    )
    if distill_mode == "sage_ib":
        print(
            f"[Compress] SAGE-IB topk={sage_config['topk']} gain_margin={sage_config['gain_margin']:.4f} "
            f"gain_temp={sage_config['gain_temperature']:.4f} confidence_margin={sage_config['confidence_margin']:.4f} "
            f"confidence_temp={sage_config['confidence_temperature']:.4f} "
            f"require_teacher_correct={sage_config['require_teacher_correct']} "
            f"rate_warmup={sage_config['rate_warmup_ratio']:.3f} "
            f"rate_decay_start={sage_config['rate_decay_start_ratio']:.3f} "
            f"rate_min_ratio={sage_config['rate_min_ratio']:.3f}",
            flush=True,
        )

    def _build_compress_meta(
        *,
        actual_final_step: int,
        best_step: int,
        best_loss: float,
        used_best_val_ckpt: bool,
    ) -> Dict[str, Any]:
        return {
            "training_stage": str(training_stage),
            "base_model": student_model_path,
            "teacher_model": teacher_model_path,
            "teacher_free_ce": bool(teacher_free_ce),
            "teacher_deploy_bundle": str(teacher_deploy_bundle),
            "teacher_loader_resolved": (
                "skipped_teacher_free_ce"
                if teacher_free_ce
                else ("shared_deploy_bundle" if teacher_bundle is not None else "native")
            ),
            "lora_rank": int(args.lora_rank),
            "lora_alpha": float(args.lora_alpha),
            "init_shared_student_ckpt": str(init_shared_student_ckpt),
            "init_shared": bool(init_shared),
            "proto_seed_strategy": str(proto_seed_strategy),
            "proto_seed_strategy_resolved": str(resolved_seed_strategy),
            "proto_seed_layers": {str(k): int(v) for k, v in seed_layer_for_proto.items()},
            "sharing_policy_path": str(sharing_policy_path),
            "regime_labels": list(regime_labels),
            "lr_schedule": str(scheduler_meta["schedule"]),
            "lr_bank": float(lr_bank),
            "lr_adapter": float(lr_adapter),
            "lr_warmup_steps": int(scheduler_meta["warmup_steps"]),
            "lr_warmup_ratio": float(scheduler_meta["warmup_ratio"]),
            "lr_min_ratio": float(scheduler_meta["min_lr_ratio"]),
            "steps": int(args.steps),
            "actual_final_step": int(actual_final_step),
            "best_val_step": int(best_step),
            "best_val_loss": float(best_loss) if best_step >= 0 else None,
            "used_best_val_ckpt": bool(used_best_val_ckpt),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "lambda_ce": float(args.lambda_ce),
            "lambda_kd": float(kd_coeff),
            "lambda_hidden_mse": float(lambda_hidden_mse),
            "hidden_mse_layer_ids": [int(x) for x in hidden_mse_layer_ids],
            "kd_temperature": float(kd_temperature),
            "distill_mode": str(distill_mode),
            "training_prompt_mode": str(getattr(args, "training_prompt_mode", "legacy_sft")),
            "loss_scope": str(loss_scope),
            "loss_exclude_eos": bool(loss_exclude_eos),
            "sage_config": dict(sage_config),
            "val_selection_metric": str(val_selection_metric),
            "val_include_step0_candidate": bool(val_include_step0_candidate),
            "val_min_improvement": float(val_min_improvement),
            "lambda_core": float(lambda_core),
            "core_lambda_schedule": str(core_lambda_schedule),
            "core_lambda_warmup_ratio": float(core_lambda_warmup_ratio),
            "core_lambda_cutoff_ratio": float(core_lambda_cutoff_ratio),
            "core_metric_source": str(core_metric_source),
            "core_metric_diag_path": str(core_metric_diag_path),
            "core_metric_diag_mode": str(core_metric_diag_mode),
            "core_layer_ids": [int(x) for x in tau_layer_ids],
            "core_metric_eps": float(tau_eps),
            "core_use_metric_whitening": bool(core_use_metric_whitening),
            "core_metric_trace_normalize": bool(core_metric_trace_normalize),
            "core_use_reliability_weighting": bool(core_use_reliability_weighting),
            "core_token_selection": str(core_token_selection),
            "core_candidate_tokens": int(core_candidate_tokens),
            "core_iets_temperature": float(core_iets_temperature),
            "core_iets_anchor_boost": float(core_iets_anchor_boost),
            "core_iets_energy_alpha": float(core_iets_energy_alpha),
            "core_iets_entropy_beta": float(core_iets_entropy_beta),
            "core_iets_topk": int(core_iets_topk),
            "core_use_information_weighting": bool(core_use_information_weighting),
            "core_information_power": float(core_information_power),
            "lambda_layer_mixture": float(lambda_layer_mixture),
            "layer_mixture_enabled": bool(layer_mixture_transport is not None),
            "layer_mixture_lr": float(layer_mixture_lr),
            "layer_mixture_config": (
                unwrap_model(layer_mixture_transport).config_dict()
                if layer_mixture_transport is not None
                else {}
            ),
            "lambda_phase_adaptive_core": float(lambda_phase_adaptive_core),
            "phase_projector_enabled": bool(phase_projector_bank is not None),
            "phase_projector_bank_path": str(phase_projector_bank_path),
            "phase_projector_mode": str(phase_projector_mode),
            "phase_projector_lr": float(phase_projector_lr),
            "phase_projector_config": (
                unwrap_model(phase_projector_bank).config_dict()
                if phase_projector_bank is not None
                else {}
            ),
            "lambda_geodesic_core": float(lambda_geodesic_core),
            "geodesic_core_max_layer_gap": int(geodesic_core_max_layer_gap),
            "lambda_manifold_core": float(lambda_manifold_core),
            "manifold_core_temperature": float(manifold_core_temperature),
            "manifold_core_enabled": bool(manifold_core_enabled),
            "manifold_anchor_count": int(manifold_anchor_count),
            "manifold_dim": int(manifold_dim),
            "layer_reliability": [float(x) for x in layer_reliability_cpu.tolist()],
            "world_size": int(dist_ctx.world_size),
            "per_rank_batch_size": int(args.batch_size),
            "global_batch_size": int(args.batch_size) * int(dist_ctx.world_size),
            "gradient_accumulation_steps": int(grad_accum_steps),
            "effective_global_batch_size": int(args.batch_size) * int(dist_ctx.world_size) * int(grad_accum_steps),
            "checkpoint_policy": "best_val_only",
            "best_val_ckpt_path": str(best_val_ckpt_path),
        }

    def _run_and_record_validation(step_for_val: int, *, allow_best: bool) -> None:
        nonlocal best_val_loss, best_val_step
        if val_loader is not None and val_every > 0:
            # Validation is intentionally rank-0 only. Running a DDP wrapper on
            # one rank can enqueue collectives that other ranks never enter.
            val_student = unwrap_model(student) if dist_ctx.enabled else student
            val_stats = _run_compress_validation(
                student=val_student,
                teacher=teacher,
                loader=val_loader,
                device=device,
                pad_token_id=int(tokenizer.pad_token_id),
                eos_token_id=eos_token_id,
                lambda_ce=ce_coeff,
                lambda_kd=kd_coeff,
                kd_temperature=kd_temperature,
                distill_mode=distill_mode,
                max_batches=val_max_batches,
                loss_scope=loss_scope,
                loss_exclude_eos=loss_exclude_eos,
                sage_config=sage_config,
                subspace_viz_config={
                    "enabled": bool(getattr(args, "val_subspace_viz_enable", False)),
                    "layer_ids": [int(x) for x in tau_layer_ids],
                    "regime_labels": list(regime_labels),
                    "regime_basis_device_map": regime_basis_device_map,
                    "layer_metric_diag": layer_metric_diag_cpu,
                    "token_rule": tau_token_rule,
                    "step": int(step_for_val),
                    "output_dir": val_subspace_viz_dir,
                    "max_points_per_regime": int(getattr(args, "val_subspace_viz_max_points_per_regime", 256)),
                    "save_plot": bool(int(step_for_val) == 0),
                    "plot_prefix": _validation_step_tag(int(step_for_val)) if int(step_for_val) == 0 else "",
                },
            )
            val_entry = {
                "step": int(step_for_val),
                "loss": float(val_stats["loss"]),
                "ce": float(val_stats["ce"]),
                "kd": float(val_stats.get("kd", 0.0)),
                "response_ce": float(val_stats.get("response_ce", val_stats["ce"])),
                "decision_ce": float(val_stats.get("decision_ce", val_stats.get("response_ce", val_stats["ce"]))),
                "sage_gate_mean": float(val_stats.get("sage_gate_mean", 0.0)),
                "sage_active_fraction": float(val_stats.get("sage_active_fraction", 0.0)),
                "sage_teacher_correct_fraction": float(val_stats.get("sage_teacher_correct_fraction", 0.0)),
                "batches": int(val_stats["batches"]),
                "samples": int(val_stats["samples"]),
            }
            subspace_viz = val_stats.get("subspace_viz")
            if isinstance(subspace_viz, dict):
                val_entry["subspace_gap_l2"] = float(subspace_viz.get("mean_pair_l2", 0.0))
                val_entry["subspace_gap_cosine"] = (
                    float(subspace_viz["mean_pair_cosine"])
                    if subspace_viz.get("mean_pair_cosine") is not None
                    else None
                )
                val_entry["subspace_gap_l2_by_regime"] = {
                    str(k): float(v)
                    for k, v in dict(subspace_viz.get("mean_pair_l2_by_regime", {})).items()
                }
                val_entry["subspace_gap_cosine_by_regime"] = {
                    str(k): float(v)
                    for k, v in dict(subspace_viz.get("mean_pair_cosine_by_regime", {})).items()
                }
                val_entry["subspace_snapshot_png"] = str(subspace_viz.get("output_png", ""))
                val_entry["subspace_snapshot_json"] = str(subspace_viz.get("output_json", ""))
                val_entry["subspace_plot_data_pt"] = str(subspace_viz.get("output_data_pt", ""))
                val_entry["subspace_snapshot_pngs"] = dict(subspace_viz.get("output_pngs", {}))
            val_history.append(val_entry)
            if val_selection_metric not in val_entry:
                raise ValueError(
                    f"val_selection_metric={val_selection_metric!r} is unavailable; "
                    f"available={sorted(val_entry.keys())}"
                )
            cur_val_loss = float(val_entry[val_selection_metric])
            improved = bool(allow_best) and cur_val_loss < (best_val_loss - float(val_min_improvement))
            if improved:
                best_val_loss = cur_val_loss
                best_val_step = int(step_for_val)
                best_val_meta = _build_compress_meta(
                    actual_final_step=int(step_for_val),
                    best_step=int(best_val_step),
                    best_loss=float(best_val_loss),
                    used_best_val_ckpt=True,
                )
                best_val_shared_state = extract_shared_state(student, layer_to_proto=layer_to_proto)
                best_val_shared_state["meta"] = dict(best_val_meta)
                best_val_mixture_state = (
                    {
                        key: value.detach().cpu()
                        for key, value in unwrap_model(layer_mixture_transport).state_dict().items()
                    }
                    if layer_mixture_transport is not None
                    else {}
                )
                best_val_phase_projector_state = (
                    {
                        key: value.detach().cpu()
                        for key, value in unwrap_model(phase_projector_bank).state_dict().items()
                    }
                    if phase_projector_bank is not None
                    else {}
                )
                torch.save(
                    {
                        "phase": "compress_best_val",
                        "step": int(step_for_val),
                        "val_loss": float(val_entry["loss"]),
                        "val_selection_metric": str(val_selection_metric),
                        "val_selection_score": float(best_val_loss),
                        "meta": dict(best_val_meta),
                        "shared_state": best_val_shared_state,
                        "layer_mixture_transport_state": best_val_mixture_state,
                        "phase_projector_state": best_val_phase_projector_state,
                    },
                    best_val_ckpt_path,
                )
                if isinstance(subspace_viz, dict):
                    render_report = _render_validation_subspace_plots_from_pt(
                        plot_data_pt=str(subspace_viz.get("output_data_pt", "")),
                        output_dir=val_subspace_viz_dir,
                        file_prefix="bestval",
                    )
                    previous_plot = dict(subspace_viz.get("plot", {}))
                    merged_plot = {**previous_plot, **dict(render_report)}
                    if "regimes" in previous_plot and "regimes" not in merged_plot:
                        merged_plot["regimes"] = previous_plot["regimes"]
                    subspace_viz["plot"] = merged_plot
                    subspace_viz["best_val_step"] = int(best_val_step)
                    subspace_viz["best_val_loss"] = float(best_val_loss)
                    if render_report.get("output_png"):
                        subspace_viz["output_png"] = str(render_report.get("output_png", ""))
                        val_entry["subspace_snapshot_png"] = str(render_report.get("output_png", ""))
                    if isinstance(render_report.get("output_pngs"), dict):
                        subspace_viz["output_pngs"] = dict(render_report.get("output_pngs", {}))
                        val_entry["subspace_snapshot_pngs"] = dict(render_report.get("output_pngs", {}))
                    val_entry["subspace_snapshot_json"] = str(subspace_viz.get("output_json", ""))
                    val_entry["subspace_plot_data_pt"] = str(subspace_viz.get("output_data_pt", ""))
                    bestval_data_pt = os.path.join(val_subspace_viz_dir, "bestval_subspace_plot_data.pt")
                    source_data_pt = str(subspace_viz.get("output_data_pt", ""))
                    if source_data_pt and os.path.isfile(source_data_pt):
                        shutil.copyfile(source_data_pt, bestval_data_pt)
                        subspace_viz["bestval_output_data_pt"] = bestval_data_pt
                        val_entry["bestval_subspace_plot_data_pt"] = bestval_data_pt
                    bestval_json = os.path.join(val_subspace_viz_dir, "bestval_subspace_snapshot.json")
                    subspace_viz["bestval_output_json"] = bestval_json
                    save_json(str(subspace_viz.get("output_json", "")), subspace_viz)
                    save_json(bestval_json, subspace_viz)
            best_summary = (
                f"{best_val_loss:.4f}@step{best_val_step}"
                if best_val_step >= 0
                else "n/a"
            )
            subspace_gap_text = (
                f" subspace_gap={float(val_entry['subspace_gap_l2']):.4f}"
                if "subspace_gap_l2" in val_entry
                else ""
            )
            print(
                f"[Compress][val@{step_for_val}] "
                f"loss={val_entry['loss']:.4f} ce={val_entry['ce']:.4f} "
                f"response_ce={val_entry['response_ce']:.4f} kd={val_entry['kd']:.4f} "
                f"decision_ce={val_entry['decision_ce']:.4f} "
                f"sage_gate={val_entry['sage_gate_mean']:.4f} "
                f"batches={val_entry['batches']} samples={val_entry['samples']}"
                f"{subspace_gap_text}"
                f"{' *best*' if improved else ''}"
                f" (best={best_summary})",
                flush=True,
            )
            if isinstance(subspace_viz, dict):
                trend_report = _save_validation_subspace_trend(
                    val_history=val_history,
                    output_png=os.path.join(val_subspace_viz_dir, "subspace_gap_trend.png"),
                    output_json=os.path.join(val_subspace_viz_dir, "subspace_gap_trend.json"),
                    title="Validation teacher-student gap in regime-specific subspace",
                )
                if isinstance(trend_report, dict):
                    val_entry["subspace_trend_png"] = str(trend_report.get("output_png", ""))
                    val_entry["subspace_trend_json"] = str(trend_report.get("output_json", ""))
            if dist_ctx.enabled:
                dist_barrier(dist_ctx)
        elif dist_ctx.enabled and val_every > 0 and step_for_val % int(val_every) == 0:
            dist_barrier(dist_ctx)

    if val_every > 0:
        _run_and_record_validation(step_for_val=0, allow_best=val_include_step0_candidate)

    micro_in_accum = 0
    optimizer.zero_grad(set_to_none=True)
    train_wall_t0 = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    while step < total_steps:
        if train_sampler is not None:
            train_sampler.set_epoch(int(args.seed) + epoch_idx)
            epoch_idx += 1
        for batch in loader:
            if step >= total_steps:
                break
            pending_step = int(step) + 1
            if bank_freeze_steps > 0 and pending_step == bank_freeze_steps + 1:
                set_shared_bank_trainable(student, True)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            prompt_lens = batch.get("prompt_lens", None)
            if prompt_lens is not None:
                prompt_lens = prompt_lens.to(device)
            candidate_token_ids = batch.get("candidate_token_ids", None)
            candidate_mask = batch.get("candidate_mask", None)
            gold_candidate_index = batch.get("gold_candidate_index", None)
            if candidate_token_ids is not None:
                candidate_token_ids = candidate_token_ids.to(device)
            if candidate_mask is not None:
                candidate_mask = candidate_mask.to(device)
            if gold_candidate_index is not None:
                gold_candidate_index = gold_candidate_index.to(device)
            tau_batch_indices: Optional[torch.Tensor] = None
            tau_token_indices: Optional[torch.Tensor] = None
            tau_group_ids: Optional[torch.Tensor] = None
            tau_anchor_mask: Optional[torch.Tensor] = None
            tau_phase_ids: Optional[torch.Tensor] = None
            tau_progress: Optional[torch.Tensor] = None
            need_token_select = tau_enabled or hidden_mse_enabled
            if need_token_select:
                tau_batch_indices, tau_token_indices, tau_group_ids, tau_anchor_mask = _select_core_token_positions(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    eos_token_id=eos_token_id,
                    token_rule=tau_token_rule,
                    selection_mode=core_token_selection if tau_enabled else "last_pred",
                    candidate_tokens=core_candidate_tokens if tau_enabled else 1,
                    prompt_lens=prompt_lens,
                )
                if phase_projector_bank is not None:
                    if prompt_lens is None:
                        raise RuntimeError(
                            "phase-adaptive projector requires response prompt lengths"
                        )
                    tau_phase_ids, tau_progress = _phase_progress_for_selected_tokens(
                        token_batch_indices=tau_batch_indices,
                        token_indices=tau_token_indices,
                        prompt_lens=prompt_lens,
                        attention_mask=attention_mask,
                    )

            need_capture = tau_enabled or hidden_mse_enabled
            capture_mlp_ids = sorted(set(tau_layer_ids)) if need_capture else None

            with torch.no_grad():
                teacher_mlp_selected: Dict[int, torch.Tensor] = {}
                teacher_hidden_selected: Optional[List[Optional[torch.Tensor]]] = None
                t_out = None
                if teacher is not None and need_capture and tau_batch_indices is not None and tau_token_indices is not None:
                    t_out, teacher_hidden_selected, teacher_mlp_selected, _, _ = _forward_with_selected_capture(
                        model=teacher,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        batch_indices=tau_batch_indices,
                        token_indices=tau_token_indices,
                        capture_hidden=hidden_mse_enabled,
                        capture_mlp_layer_ids=capture_mlp_ids if tau_enabled else None,
                        capture_pre_ffn_input_layer_ids=None,
                        capture_residual_output_layer_ids=None,
                    )
                elif teacher is not None:
                    t_out = teacher(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_dict=True,
                        use_cache=False,
                    )

            student_mlp_selected: Dict[int, torch.Tensor] = {}
            student_hidden_selected: Optional[List[Optional[torch.Tensor]]] = None
            student_capture_handles: Optional[List[Any]] = [] if bool(getattr(args, "student_gradient_checkpointing", False)) and need_capture else None
            if need_capture and tau_batch_indices is not None and tau_token_indices is not None:
                s_out, student_hidden_selected, student_mlp_selected, _, _ = _forward_with_selected_capture(
                    model=student,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    batch_indices=tau_batch_indices,
                    token_indices=tau_token_indices,
                    capture_hidden=hidden_mse_enabled,
                    capture_mlp_layer_ids=capture_mlp_ids if tau_enabled else None,
                    capture_pre_ffn_input_layer_ids=None,
                    capture_residual_output_layer_ids=None,
                    keep_hook_handles=student_capture_handles,
                )
            else:
                s_out = student(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                    use_cache=False,
                )
            s_logits = _extract_logits_from_model_output(s_out).float()
            t_logits = _extract_logits_from_model_output(t_out).float() if t_out is not None else None

            scope_mask = shifted_target_mask(
                input_ids=input_ids,
                attention_mask=attention_mask,
                prompt_lens=prompt_lens,
                scope=loss_scope,
                pad_token_id=int(tokenizer.pad_token_id),
                eos_token_id=eos_token_id,
                exclude_eos=loss_exclude_eos,
            )
            decision_mask = shifted_target_mask(
                input_ids=input_ids,
                attention_mask=attention_mask,
                prompt_lens=prompt_lens,
                scope="decision",
                pad_token_id=int(tokenizer.pad_token_id),
                eos_token_id=eos_token_id,
                exclude_eos=True,
            )
            response_mask = shifted_target_mask(
                input_ids=input_ids,
                attention_mask=attention_mask,
                prompt_lens=prompt_lens,
                scope="response",
                pad_token_id=int(tokenizer.pad_token_id),
                eos_token_id=eos_token_id,
                exclude_eos=loss_exclude_eos,
            )
            ce_loss, _ = masked_next_token_cross_entropy(
                logits=s_logits,
                input_ids=input_ids,
                token_mask=scope_mask,
            )
            response_ce_loss, _ = masked_next_token_cross_entropy(
                logits=s_logits,
                input_ids=input_ids,
                token_mask=response_mask,
            )
            has_candidate_channel = candidate_token_ids is not None and candidate_mask is not None and gold_candidate_index is not None
            student_candidate_logits: Optional[torch.Tensor] = None
            teacher_candidate_logits: Optional[torch.Tensor] = None
            if has_candidate_channel:
                decision_ce_loss, decision_stats = candidate_decision_cross_entropy(
                    logits=s_logits,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    candidate_token_ids=candidate_token_ids,
                    candidate_mask=candidate_mask,
                    gold_candidate_index=gold_candidate_index,
                    eos_token_id=eos_token_id,
                )
                student_candidate_logits = decision_stats["candidate_logits"]
                if t_logits is not None:
                    teacher_candidate_logits = candidate_decision_logits(
                        logits=t_logits,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        candidate_token_ids=candidate_token_ids,
                        candidate_mask=candidate_mask,
                        eos_token_id=eos_token_id,
                    )
                if loss_scope == "decision":
                    ce_loss = decision_ce_loss
            else:
                decision_ce_loss, _ = masked_next_token_cross_entropy(
                    logits=s_logits,
                    input_ids=input_ids,
                    token_mask=decision_mask,
                )
            sage_stats: Dict[str, torch.Tensor] = {}
            teacher_coeff_now = float(kd_coeff)
            if distill_mode == "ce_kd" and kd_coeff > 0.0:
                if t_logits is None:
                    raise RuntimeError("CE+KD training requires teacher logits")
                kd_loss = _kd_shift_masked_token_mean(
                    student_logits=s_logits,
                    teacher_logits=t_logits,
                    attention_mask=attention_mask,
                    temperature=kd_temperature,
                )
            elif distill_mode == "sage_ib" and kd_coeff > 0.0:
                teacher_coeff_now = sage_rate_at_step(
                    step=int(pending_step),
                    total_steps=int(total_steps),
                    max_rate=float(kd_coeff),
                    warmup_ratio=float(sage_config["rate_warmup_ratio"]),
                    decay_start_ratio=float(sage_config["rate_decay_start_ratio"]),
                    min_rate_ratio=float(sage_config["rate_min_ratio"]),
                )
                if has_candidate_channel and student_candidate_logits is not None and teacher_candidate_logits is not None:
                    kd_loss, sage_stats = sage_candidate_information_gain_js(
                        student_candidate_logits=student_candidate_logits,
                        teacher_candidate_logits=teacher_candidate_logits,
                        candidate_mask=candidate_mask,
                        gold_candidate_index=gold_candidate_index,
                        temperature=kd_temperature,
                        gain_margin=float(sage_config["gain_margin"]),
                        gain_temperature=float(sage_config["gain_temperature"]),
                        confidence_margin=float(sage_config["confidence_margin"]),
                        confidence_temperature=float(sage_config["confidence_temperature"]),
                        confidence_power=float(sage_config["confidence_power"]),
                        require_teacher_correct=bool(sage_config["require_teacher_correct"]),
                    )
                else:
                    kd_loss, sage_stats = sage_information_gain_js(
                        student_logits=s_logits,
                        teacher_logits=t_logits,
                        input_ids=input_ids,
                        token_mask=decision_mask,
                        temperature=kd_temperature,
                        topk=int(sage_config["topk"]),
                        gain_margin=float(sage_config["gain_margin"]),
                        gain_temperature=float(sage_config["gain_temperature"]),
                        confidence_margin=float(sage_config["confidence_margin"]),
                        confidence_temperature=float(sage_config["confidence_temperature"]),
                        confidence_power=float(sage_config["confidence_power"]),
                        require_teacher_correct=bool(sage_config["require_teacher_correct"]),
                    )
            else:
                kd_loss = torch.zeros((), dtype=torch.float32, device=device)
                teacher_coeff_now = 0.0
            hidden_loss = torch.zeros((), dtype=torch.float32, device=device)
            hidden_layers_used = 0
            if hidden_mse_enabled and teacher_hidden_selected is not None and student_hidden_selected is not None:
                for layer_id in hidden_mse_layer_ids:
                    hidden_idx = int(layer_id) + 1
                    if hidden_idx < 0 or hidden_idx >= len(teacher_hidden_selected) or hidden_idx >= len(student_hidden_selected):
                        continue
                    h_t = teacher_hidden_selected[hidden_idx]
                    h_s = student_hidden_selected[hidden_idx]
                    if h_t is None or h_s is None:
                        continue
                    hidden_loss = hidden_loss + (h_s.to(dtype=torch.float32) - h_t.to(dtype=torch.float32)).pow(2).mean()
                    hidden_layers_used += 1
                if hidden_layers_used > 0:
                    hidden_loss = hidden_loss / float(hidden_layers_used)
            z_student_by_layer: Dict[int, torch.Tensor] = {}
            z_teacher_by_layer: Dict[int, torch.Tensor] = {}
            core_token_weights: Optional[torch.Tensor] = None
            if tau_enabled:
                for layer_id in capture_mlp_ids or []:
                    t_ffn = teacher_mlp_selected.get(int(layer_id))
                    if t_ffn is None:
                        continue
                    s_ffn = student_mlp_selected.get(int(layer_id))
                    if s_ffn is None:
                        continue
                    layer_regime = str(regime_labels[int(layer_id)]) if int(layer_id) < len(regime_labels) else "llama_late"
                    layer_basis = regime_basis_device_map.get(layer_regime, basis_device)
                    if core_coordinate_mode == "ambient":
                        z_student_by_layer[int(layer_id)] = s_ffn.to(dtype=torch.float32)
                        z_teacher_by_layer[int(layer_id)] = t_ffn.to(dtype=torch.float32)
                    else:
                        z_student_by_layer[int(layer_id)] = torch.matmul(s_ffn.to(dtype=torch.float32), layer_basis)
                        z_teacher_by_layer[int(layer_id)] = torch.matmul(t_ffn.to(dtype=torch.float32), layer_basis)
                core_token_weights = _compute_core_token_weights(
                    z_teacher_by_layer=z_teacher_by_layer,
                    tau_layer_ids=tau_layer_ids,
                    layer_metric_diag_device=layer_metric_diag_device,
                    layer_reliability_device=layer_reliability_device,
                    token_batch_indices=tau_batch_indices,
                    token_indices=tau_token_indices,
                    group_ids=tau_group_ids,
                    anchor_mask=tau_anchor_mask,
                    teacher_logits=t_logits if abs(float(core_iets_entropy_beta)) > 0.0 else None,
                    selection_mode=core_token_selection,
                    temperature=core_iets_temperature,
                    anchor_boost=core_iets_anchor_boost,
                    energy_alpha=core_iets_energy_alpha,
                    entropy_beta=core_iets_entropy_beta,
                    hard_topk=core_iets_topk,
                    metric_eps=tau_eps,
                    use_metric_whitening=core_use_metric_whitening,
                    metric_trace_normalize=core_metric_trace_normalize,
                    metric_diag_is_precision=core_metric_diag_is_precision,
                    use_reliability_weighting=core_use_reliability_weighting,
                )

            mixture_zero = torch.zeros((), dtype=torch.float32, device=device)
            layer_mixture_stats: Dict[str, torch.Tensor] = {
                "loss": mixture_zero,
                "nll": mixture_zero,
                "gate_entropy": mixture_zero,
                "posterior_entropy": mixture_zero,
                "effective_components": mixture_zero,
                "max_probability": mixture_zero,
                "delta_l2": mixture_zero,
                "target_layers": mixture_zero,
            }
            if layer_mixture_transport is not None:
                layer_mixture_stats = layer_mixture_transport(
                    z_student_by_layer,
                    z_teacher_by_layer,
                )
                if int(round(float(layer_mixture_stats["target_layers"].detach().item()))) <= 0:
                    raise RuntimeError("layer-mixture transport received no shared-layer teacher/student captures")
            layer_mixture_loss = layer_mixture_stats["loss"]

            core_loss = torch.zeros((), dtype=torch.float32, device=device)
            phase_adaptive_core_loss = torch.zeros((), dtype=torch.float32, device=device)
            phase_adaptive_cosine = torch.zeros((), dtype=torch.float32, device=device)
            phase_adaptive_layers_used = 0
            point_core_loss = torch.zeros((), dtype=torch.float32, device=device)
            geodesic_core_loss = torch.zeros((), dtype=torch.float32, device=device)
            relational_core_loss = torch.zeros((), dtype=torch.float32, device=device)
            variance_core_loss = torch.zeros((), dtype=torch.float32, device=device)
            token_flow_core_loss = torch.zeros((), dtype=torch.float32, device=device)
            token_turning_core_loss = torch.zeros((), dtype=torch.float32, device=device)
            manifold_core_loss = torch.zeros((), dtype=torch.float32, device=device)
            delta_manifold_core_loss = torch.zeros((), dtype=torch.float32, device=device)
            core_layers_used = 0
            core_layer_weight_sum = 0.0
            relational_layers_used = 0
            relational_weight_sum = 0.0
            variance_layers_used = 0
            variance_weight_sum = 0.0
            token_flow_layers_used = 0
            token_flow_weight_sum = 0.0
            if (
                phase_projector_bank is not None
                and tau_phase_ids is not None
                and tau_progress is not None
            ):
                for layer_id in tau_layer_ids:
                    z_s = z_student_by_layer.get(int(layer_id))
                    z_t = z_teacher_by_layer.get(int(layer_id))
                    if z_s is None or z_t is None:
                        continue
                    phase_stats = phase_projector_bank(
                        z_s,
                        z_t,
                        int(layer_id),
                        tau_phase_ids,
                        tau_progress,
                    )
                    phase_adaptive_core_loss = (
                        phase_adaptive_core_loss + phase_stats["loss"]
                    )
                    phase_adaptive_cosine = (
                        phase_adaptive_cosine + phase_stats["cosine"]
                    )
                    phase_adaptive_layers_used += 1
                if phase_adaptive_layers_used > 0:
                    phase_adaptive_core_loss = phase_adaptive_core_loss / float(
                        phase_adaptive_layers_used
                    )
                    phase_adaptive_cosine = phase_adaptive_cosine / float(
                        phase_adaptive_layers_used
                    )
            if tau_enabled:
                for layer_id in tau_layer_ids:
                    z_s = z_student_by_layer.get(int(layer_id))
                    z_t = z_teacher_by_layer.get(int(layer_id))
                    if z_s is None or z_t is None:
                        continue
                    if core_use_metric_whitening and core_coordinate_mode == "ambient":
                        raise RuntimeError(
                            "ambient velocity MSE must use an isotropic ambient metric; "
                            "set core_use_metric_whitening=False"
                        )
                    if core_use_metric_whitening:
                        metric_inv_std = _metric_inverse_sqrt(
                            layer_metric_diag_device[int(layer_id)].view(1, -1),
                            eps=tau_eps,
                            trace_normalize=core_metric_trace_normalize,
                            input_is_precision=core_metric_diag_is_precision,
                        )
                    else:
                        metric_inv_std = torch.ones((1, z_s.shape[-1]), dtype=z_s.dtype, device=z_s.device)
                    diff = (z_s - z_t) * metric_inv_std
                    if core_use_reliability_weighting:
                        layer_weight = float(layer_reliability_device[int(layer_id)].item())
                    else:
                        layer_weight = 1.0
                    per_dim_loss = diff * diff
                    if core_use_information_weighting and core_information_power > 0.0:
                        info_weight = layer_information_weight_device[int(layer_id)].view(1, -1).clamp(min=1e-6)
                        info_weight = info_weight.pow(float(core_information_power))
                        info_weight = info_weight / info_weight.mean().clamp(min=1e-6)
                        per_dim_loss = per_dim_loss * info_weight
                    per_token_loss = per_dim_loss.mean(dim=1)
                    if core_token_weights is not None and int(core_token_weights.numel()) == int(per_token_loss.numel()):
                        layer_core = (per_token_loss * core_token_weights.to(device=per_token_loss.device)).sum() / core_token_weights.sum().clamp(min=1e-6)
                    else:
                        layer_core = per_token_loss.mean()
                    point_core_loss = point_core_loss + layer_weight * layer_core
                    core_layers_used += 1
                    core_layer_weight_sum += layer_weight
                    if (
                        lambda_relational_core > 0.0
                        and z_s.dim() == 2
                        and z_t.dim() == 2
                        and z_s.shape == z_t.shape
                        and int(z_s.size(0)) >= 2
                    ):
                        z_s_metric = z_s * metric_inv_std
                        z_t_metric = z_t * metric_inv_std
                        z_s_centered = z_s_metric - z_s_metric.mean(dim=0, keepdim=True)
                        z_t_centered = z_t_metric - z_t_metric.mean(dim=0, keepdim=True)
                        z_s_unit = torch.nn.functional.normalize(z_s_centered, p=2, dim=1, eps=tau_eps)
                        z_t_unit = torch.nn.functional.normalize(z_t_centered, p=2, dim=1, eps=tau_eps)
                        pair_mask = torch.triu(
                            torch.ones(
                                (int(z_s.size(0)), int(z_s.size(0))),
                                dtype=torch.bool,
                                device=z_s.device,
                            ),
                            diagonal=1,
                        )
                        relational_layer = (
                            (z_s_unit @ z_s_unit.transpose(0, 1))
                            - (z_t_unit @ z_t_unit.transpose(0, 1))
                        ).pow(2)[pair_mask].mean()
                        relational_core_loss = relational_core_loss + layer_weight * relational_layer
                        relational_layers_used += 1
                        relational_weight_sum += layer_weight
                    if (
                        lambda_variance_core > 0.0
                        and z_s.dim() == 2
                        and z_t.dim() == 2
                        and z_s.shape == z_t.shape
                        and int(z_s.size(0)) >= 2
                    ):
                        z_s_metric = z_s * metric_inv_std
                        z_t_metric = z_t * metric_inv_std
                        var_s = z_s_metric.var(dim=0, unbiased=False)
                        var_t = z_t_metric.var(dim=0, unbiased=False).detach()
                        floor = float(variance_core_floor_ratio) * var_t
                        variance_layer = (
                            torch.relu(floor - var_s) / var_t.clamp(min=tau_eps)
                        ).pow(2).mean()
                        variance_core_loss = variance_core_loss + layer_weight * variance_layer
                        variance_layers_used += 1
                        variance_weight_sum += layer_weight
                    if (
                        (lambda_token_flow_core > 0.0 or lambda_token_turning_core > 0.0)
                        and tau_batch_indices is not None
                        and tau_token_indices is not None
                    ):
                        flow_layer, turning_layer, segment_count, _ = (
                            _adaptive_token_flow_alignment(
                                student_z=z_s,
                                teacher_z=z_t,
                                batch_indices=tau_batch_indices,
                                token_indices=tau_token_indices,
                                metric_scale=metric_inv_std,
                                energy_fraction=token_flow_energy_fraction,
                                eps=tau_eps,
                            )
                        )
                        if segment_count > 0:
                            token_flow_core_loss = (
                                token_flow_core_loss + layer_weight * flow_layer
                            )
                            token_turning_core_loss = (
                                token_turning_core_loss + layer_weight * turning_layer
                            )
                            token_flow_layers_used += 1
                            token_flow_weight_sum += layer_weight
                if core_layers_used > 0:
                    point_core_loss = point_core_loss / float(max(core_layer_weight_sum, 1e-6))
                if relational_layers_used > 0:
                    relational_core_loss = relational_core_loss / float(max(relational_weight_sum, 1e-6))
                if variance_layers_used > 0:
                    variance_core_loss = variance_core_loss / float(max(variance_weight_sum, 1e-6))
                if token_flow_layers_used > 0:
                    token_flow_core_loss = token_flow_core_loss / float(
                        max(token_flow_weight_sum, 1e-6)
                    )
                    token_turning_core_loss = token_turning_core_loss / float(
                        max(token_flow_weight_sum, 1e-6)
                    )
                geodesic_pairs_used = 0
                geodesic_weight_sum = 0.0
                if lambda_geodesic_core > 0.0:
                    available_layers = sorted(
                        int(layer_id)
                        for layer_id in tau_layer_ids
                        if int(layer_id) in z_student_by_layer and int(layer_id) in z_teacher_by_layer
                    )
                    for left, right in zip(available_layers[:-1], available_layers[1:]):
                        if int(right) - int(left) > int(geodesic_core_max_layer_gap):
                            continue
                        z_s_left = z_student_by_layer.get(int(left))
                        z_s_right = z_student_by_layer.get(int(right))
                        z_t_left = z_teacher_by_layer.get(int(left))
                        z_t_right = z_teacher_by_layer.get(int(right))
                        if z_s_left is None or z_s_right is None or z_t_left is None or z_t_right is None:
                            continue
                        if z_s_left.shape != z_s_right.shape or z_t_left.shape != z_t_right.shape:
                            continue
                        velocity_diff = (z_s_right - z_s_left) - (z_t_right - z_t_left)
                        if core_use_metric_whitening:
                            pair_metric = torch.sqrt(
                                layer_metric_diag_device[int(left)].clamp(min=tau_eps)
                                * layer_metric_diag_device[int(right)].clamp(min=tau_eps)
                            )
                            pair_inv_std = _metric_inverse_sqrt(
                                pair_metric.view(1, -1),
                                eps=tau_eps,
                                trace_normalize=core_metric_trace_normalize,
                                input_is_precision=core_metric_diag_is_precision,
                            )
                        else:
                            pair_inv_std = torch.ones((1, velocity_diff.shape[-1]), dtype=velocity_diff.dtype, device=velocity_diff.device)
                        geo_dim_loss = (velocity_diff * pair_inv_std).pow(2)
                        if core_use_information_weighting and core_information_power > 0.0:
                            pair_info = 0.5 * (
                                layer_information_weight_device[int(left)] + layer_information_weight_device[int(right)]
                            )
                            pair_info = pair_info.view(1, -1).clamp(min=1e-6).pow(float(core_information_power))
                            pair_info = pair_info / pair_info.mean().clamp(min=1e-6)
                            geo_dim_loss = geo_dim_loss * pair_info
                        geo_token_loss = geo_dim_loss.mean(dim=1)
                        if core_token_weights is not None and int(core_token_weights.numel()) == int(geo_token_loss.numel()):
                            pair_loss = (geo_token_loss * core_token_weights.to(device=geo_token_loss.device)).sum() / core_token_weights.sum().clamp(min=1e-6)
                        else:
                            pair_loss = geo_token_loss.mean()
                        if core_use_reliability_weighting:
                            pair_weight = 0.5 * (
                                float(layer_reliability_device[int(left)].item())
                                + float(layer_reliability_device[int(right)].item())
                            )
                        else:
                            pair_weight = 1.0
                        geodesic_core_loss = geodesic_core_loss + pair_weight * pair_loss
                        geodesic_pairs_used += 1
                        geodesic_weight_sum += pair_weight
                    if geodesic_pairs_used > 0:
                        geodesic_core_loss = geodesic_core_loss / float(max(geodesic_weight_sum, 1e-6))
                manifold_layers_used = 0
                manifold_weight_sum = 0.0
                if (
                    lambda_manifold_core > 0.0
                    and manifold_core_enabled
                    and manifold_anchor_z_device is not None
                    and manifold_anchor_phi_device is not None
                    and manifold_sigma2_device is not None
                ):
                    for layer_id in tau_layer_ids:
                        z_s = z_student_by_layer.get(int(layer_id))
                        z_t = z_teacher_by_layer.get(int(layer_id))
                        if z_s is None or z_t is None or z_s.dim() != 2 or z_t.dim() != 2 or z_s.shape != z_t.shape:
                            continue
                        anchor_z = manifold_anchor_z_device[int(layer_id)]
                        anchor_phi = manifold_anchor_phi_device[int(layer_id)]
                        sigma2 = manifold_sigma2_device[int(layer_id)]
                        phi_s = _diffusion_manifold_embed(
                            z=z_s,
                            anchor_z=anchor_z,
                            anchor_phi=anchor_phi,
                            metric_diag=layer_metric_diag_device[int(layer_id)],
                            sigma2=sigma2,
                            temperature=manifold_core_temperature,
                            eps=tau_eps,
                        )
                        phi_t = _diffusion_manifold_embed(
                            z=z_t,
                            anchor_z=anchor_z,
                            anchor_phi=anchor_phi,
                            metric_diag=layer_metric_diag_device[int(layer_id)],
                            sigma2=sigma2,
                            temperature=manifold_core_temperature,
                            eps=tau_eps,
                        )
                        manifold_token_loss = (phi_s - phi_t).pow(2).mean(dim=1)
                        if core_token_weights is not None and int(core_token_weights.numel()) == int(manifold_token_loss.numel()):
                            layer_manifold = (manifold_token_loss * core_token_weights.to(device=manifold_token_loss.device)).sum() / core_token_weights.sum().clamp(min=1e-6)
                        else:
                            layer_manifold = manifold_token_loss.mean()
                        if core_use_reliability_weighting:
                            manifold_weight = float(layer_reliability_device[int(layer_id)].item())
                        else:
                            manifold_weight = 1.0
                        manifold_core_loss = manifold_core_loss + manifold_weight * layer_manifold
                        manifold_layers_used += 1
                        manifold_weight_sum += manifold_weight
                    if manifold_layers_used > 0:
                        manifold_core_loss = manifold_core_loss / float(max(manifold_weight_sum, 1e-6))
                delta_manifold_layers_used = 0
                delta_manifold_weight_sum = 0.0
                if (
                    lambda_delta_manifold_core > 0.0
                    and delta_manifold_core_enabled
                    and delta_manifold_anchor_device is not None
                    and delta_manifold_phi_device is not None
                    and delta_manifold_origin_phi_device is not None
                    and delta_manifold_sigma2_device is not None
                    and delta_manifold_risk_device is not None
                ):
                    for layer_id in tau_layer_ids:
                        z_s = z_student_by_layer.get(int(layer_id))
                        z_t = z_teacher_by_layer.get(int(layer_id))
                        if z_s is None or z_t is None or z_s.dim() != 2 or z_t.dim() != 2 or z_s.shape != z_t.shape:
                            continue
                        error_z = z_s - z_t
                        anchor_e = delta_manifold_anchor_device[int(layer_id)]
                        anchor_phi = delta_manifold_phi_device[int(layer_id)]
                        origin_phi = delta_manifold_origin_phi_device[int(layer_id)].view(1, -1)
                        sigma2 = delta_manifold_sigma2_device[int(layer_id)]
                        phi_e = _diffusion_manifold_embed(
                            z=error_z,
                            anchor_z=anchor_e,
                            anchor_phi=anchor_phi,
                            metric_diag=layer_metric_diag_device[int(layer_id)],
                            sigma2=sigma2,
                            temperature=delta_manifold_core_temperature,
                            eps=tau_eps,
                        )
                        delta_phi_loss = (phi_e - origin_phi.to(device=phi_e.device, dtype=phi_e.dtype)).pow(2).mean(dim=1)
                        if float(delta_manifold_risk_weight) > 0.0:
                            risk_loss = _diffusion_manifold_expected_risk(
                                z=error_z,
                                anchor_z=anchor_e,
                                anchor_risk=delta_manifold_risk_device[int(layer_id)],
                                metric_diag=layer_metric_diag_device[int(layer_id)],
                                sigma2=sigma2,
                                temperature=delta_manifold_core_temperature,
                                eps=tau_eps,
                            )
                            delta_token_loss = delta_phi_loss + float(delta_manifold_risk_weight) * risk_loss
                        else:
                            delta_token_loss = delta_phi_loss
                        if core_token_weights is not None and int(core_token_weights.numel()) == int(delta_token_loss.numel()):
                            layer_delta_manifold = (delta_token_loss * core_token_weights.to(device=delta_token_loss.device)).sum() / core_token_weights.sum().clamp(min=1e-6)
                        else:
                            layer_delta_manifold = delta_token_loss.mean()
                        if core_use_reliability_weighting:
                            delta_weight = float(layer_reliability_device[int(layer_id)].item())
                        else:
                            delta_weight = 1.0
                        delta_manifold_core_loss = delta_manifold_core_loss + delta_weight * layer_delta_manifold
                        delta_manifold_layers_used += 1
                        delta_manifold_weight_sum += delta_weight
                    if delta_manifold_layers_used > 0:
                        delta_manifold_core_loss = delta_manifold_core_loss / float(max(delta_manifold_weight_sum, 1e-6))
                core_loss = (
                    point_core_loss
                    + float(lambda_geodesic_core) * geodesic_core_loss
                    + float(lambda_relational_core) * relational_core_loss
                    + float(lambda_variance_core) * variance_core_loss
                    + float(lambda_token_flow_core) * token_flow_core_loss
                    + float(lambda_token_turning_core) * token_turning_core_loss
                    + float(lambda_manifold_core) * manifold_core_loss
                    + float(lambda_delta_manifold_core) * delta_manifold_core_loss
                )
            lambda_core_now = _core_lambda_at_step(
                base_lambda=lambda_core,
                schedule=core_lambda_schedule,
                step=pending_step,
                total_steps=total_steps,
                warmup_ratio=core_lambda_warmup_ratio,
                cutoff_ratio=core_lambda_cutoff_ratio,
            )
            loss = (
                ce_coeff * ce_loss
                + teacher_coeff_now * kd_loss
                + float(lambda_hidden_mse) * hidden_loss
                + float(lambda_core_now) * core_loss
                + float(lambda_layer_mixture) * layer_mixture_loss
                + float(lambda_phase_adaptive_core) * phase_adaptive_core_loss
            )

            should_optimizer_step = (micro_in_accum + 1) >= grad_accum_steps
            try:
                with ExitStack() as backward_stack:
                    if dist_ctx.enabled and not should_optimizer_step:
                        if isinstance(student, DDP):
                            backward_stack.enter_context(student.no_sync())
                        if isinstance(layer_mixture_transport, DDP):
                            backward_stack.enter_context(layer_mixture_transport.no_sync())
                        if isinstance(phase_projector_bank, DDP):
                            backward_stack.enter_context(phase_projector_bank.no_sync())
                    (loss / float(grad_accum_steps)).backward()
            finally:
                if student_capture_handles:
                    for handle in student_capture_handles:
                        try:
                            handle.remove()
                        except Exception:
                            pass

            on_policy_loss = torch.zeros((), dtype=torch.float32, device=device)
            on_policy_kd_loss = torch.zeros((), dtype=torch.float32, device=device)
            on_policy_core_loss = torch.zeros((), dtype=torch.float32, device=device)
            on_policy_flow_loss = torch.zeros((), dtype=torch.float32, device=device)
            on_policy_turning_loss = torch.zeros((), dtype=torch.float32, device=device)
            on_policy_active = False
            on_policy_gate_mean = 0.0
            on_policy_token_count = 0.0
            if (
                on_policy_enabled_resolved
                and should_optimizer_step
                and pending_step >= on_policy_start_step
                and pending_step % on_policy_interval == 0
                and prompt_lens is not None
            ):
                with torch.no_grad():
                    student_gold_nll = _masked_nll_per_sample(
                        logits=s_logits,
                        input_ids=input_ids,
                        token_mask=response_mask,
                    )
                    teacher_gold_nll = _masked_nll_per_sample(
                        logits=t_logits,
                        input_ids=input_ids,
                        token_mask=response_mask,
                    )
                    teacher_advantage = student_gold_nll - teacher_gold_nll
                    active_rows = teacher_advantage.gt(
                        float(on_policy_teacher_advantage_margin)
                    ).nonzero(as_tuple=False).view(-1)
                    on_policy_gate_mean = float(
                        teacher_advantage.gt(float(on_policy_teacher_advantage_margin))
                        .float()
                        .mean()
                        .item()
                    )
                    active_rows = active_rows[: min(on_policy_batch_size, int(active_rows.numel()))]
                    # Every DDP rank must enter (or skip) the additional rollout
                    # backward together.  A per-rank hard gate would otherwise
                    # deadlock when only a subset of ranks has a Teacher-advantage
                    # example in its local micro-batch.
                    if dist_ctx.enabled:
                        gate_sync = torch.tensor(
                            [1 if int(active_rows.numel()) > 0 else 0],
                            dtype=torch.int32,
                            device=device,
                        )
                        dist.all_reduce(gate_sync, op=dist.ReduceOp.MIN)
                        if int(gate_sync.item()) == 0:
                            active_rows = active_rows[:0]

                if int(active_rows.numel()) > 0:
                    rollout_ids, rollout_attention_mask, rollout_prompt_lens = (
                        _generate_student_on_policy_rollouts(
                            student=student,
                            input_ids=input_ids,
                            prompt_lens=prompt_lens,
                            selected_rows=active_rows,
                            pad_token_id=int(tokenizer.pad_token_id),
                            eos_token_id=eos_token_id,
                            max_new_tokens=on_policy_max_new_tokens,
                            temperature=on_policy_generation_temperature,
                            top_p=on_policy_top_p,
                        )
                    )
                    rollout_batch_parts: List[torch.Tensor] = []
                    rollout_token_parts: List[torch.Tensor] = []
                    for rollout_row in range(int(rollout_ids.size(0))):
                        valid_len = int(rollout_attention_mask[rollout_row].sum().item())
                        prompt_len = int(rollout_prompt_lens[rollout_row].item())
                        start_index = max(0, prompt_len - 1)
                        end_index = max(start_index, valid_len - 2)
                        count = min(on_policy_core_tokens, end_index - start_index + 1)
                        positions = torch.linspace(
                            start_index,
                            end_index,
                            steps=max(1, count),
                            device=device,
                        ).round().to(dtype=torch.long).unique(sorted=True)
                        rollout_batch_parts.append(
                            torch.full(
                                (int(positions.numel()),),
                                rollout_row,
                                dtype=torch.long,
                                device=device,
                            )
                        )
                        rollout_token_parts.append(positions)
                    rollout_batch_indices = torch.cat(rollout_batch_parts, dim=0)
                    rollout_token_indices = torch.cat(rollout_token_parts, dim=0)
                    on_policy_capture_ids = sorted(set(int(x) for x in tau_layer_ids))

                    with torch.no_grad():
                        (
                            rollout_teacher_out,
                            _,
                            rollout_teacher_mlp,
                            _,
                            _,
                        ) = _forward_with_selected_capture(
                            model=teacher,
                            input_ids=rollout_ids,
                            attention_mask=rollout_attention_mask,
                            batch_indices=rollout_batch_indices,
                            token_indices=rollout_token_indices,
                            capture_hidden=False,
                            capture_mlp_layer_ids=on_policy_capture_ids,
                            capture_pre_ffn_input_layer_ids=None,
                            capture_residual_output_layer_ids=None,
                        )

                    rollout_student_handles: List[Any] = []
                    try:
                        (
                            rollout_student_out,
                            _,
                            rollout_student_mlp,
                            _,
                            _,
                        ) = _forward_with_selected_capture(
                            model=student,
                            input_ids=rollout_ids,
                            attention_mask=rollout_attention_mask,
                            batch_indices=rollout_batch_indices,
                            token_indices=rollout_token_indices,
                            capture_hidden=False,
                            capture_mlp_layer_ids=on_policy_capture_ids,
                            capture_pre_ffn_input_layer_ids=None,
                            capture_residual_output_layer_ids=None,
                            keep_hook_handles=rollout_student_handles,
                        )
                        rollout_student_logits = _extract_logits_from_model_output(
                            rollout_student_out
                        )
                        rollout_teacher_logits = _extract_logits_from_model_output(
                            rollout_teacher_out
                        )
                        rollout_response_mask = shifted_target_mask(
                            input_ids=rollout_ids,
                            attention_mask=rollout_attention_mask,
                            prompt_lens=rollout_prompt_lens,
                            scope="response",
                            pad_token_id=int(tokenizer.pad_token_id),
                            eos_token_id=eos_token_id,
                            exclude_eos=False,
                        )
                        on_policy_token_count = float(rollout_response_mask.sum().item())
                        rollout_distill_weights = rollout_response_mask.float()
                        if on_policy_final_answer_weight > 1.0:
                            for rollout_row in range(int(rollout_distill_weights.size(0))):
                                valid_positions = (
                                    rollout_response_mask[rollout_row]
                                    .nonzero(as_tuple=False)
                                    .view(-1)
                                )
                                if int(valid_positions.numel()) > 0:
                                    tail_count = max(
                                        1, int(math.ceil(0.25 * int(valid_positions.numel())))
                                    )
                                    rollout_distill_weights[
                                        rollout_row, valid_positions[-tail_count:]
                                    ] *= float(on_policy_final_answer_weight)
                        if on_policy_lambda_kd > 0.0:
                            on_policy_kd_loss = _on_policy_divergence(
                                student_logits=rollout_student_logits,
                                teacher_logits=rollout_teacher_logits,
                                token_mask=rollout_distill_weights,
                                divergence=on_policy_divergence,
                                temperature=on_policy_temperature,
                            )

                        if on_policy_lambda_core > 0.0 and on_policy_capture_ids:
                            core_sum = torch.zeros((), dtype=torch.float32, device=device)
                            core_weight_sum = 0.0
                            for layer_id in on_policy_capture_ids:
                                student_ffn = rollout_student_mlp.get(int(layer_id))
                                teacher_ffn = rollout_teacher_mlp.get(int(layer_id))
                                if student_ffn is None or teacher_ffn is None:
                                    continue
                                layer_regime = (
                                    str(regime_labels[int(layer_id)])
                                    if int(layer_id) < len(regime_labels)
                                    else "llama_late"
                                )
                                layer_basis = regime_basis_device_map.get(
                                    layer_regime, basis_device
                                )
                                rollout_z_student = torch.matmul(
                                    student_ffn.to(dtype=torch.float32), layer_basis
                                )
                                rollout_z_teacher = torch.matmul(
                                    teacher_ffn.to(dtype=torch.float32), layer_basis
                                )
                                if core_use_metric_whitening:
                                    metric_scale = _metric_inverse_sqrt(
                                        layer_metric_diag_device[int(layer_id)].view(1, -1),
                                        eps=tau_eps,
                                        trace_normalize=core_metric_trace_normalize,
                                        input_is_precision=core_metric_diag_is_precision,
                                    )
                                else:
                                    metric_scale = torch.ones(
                                        (1, int(rollout_z_student.size(-1))),
                                        dtype=rollout_z_student.dtype,
                                        device=device,
                                    )
                                layer_loss = (
                                    (rollout_z_student - rollout_z_teacher) * metric_scale
                                ).pow(2).mean()
                                layer_weight = (
                                    float(layer_reliability_device[int(layer_id)].item())
                                    if core_use_reliability_weighting
                                    else 1.0
                                )
                                core_sum = core_sum + layer_weight * layer_loss
                                core_weight_sum += layer_weight
                            if core_weight_sum > 0.0:
                                on_policy_core_loss = core_sum / float(core_weight_sum)

                        if (
                            (on_policy_lambda_flow > 0.0 or on_policy_lambda_turning > 0.0)
                            and on_policy_capture_ids
                        ):
                            flow_sum = torch.zeros((), dtype=torch.float32, device=device)
                            turning_sum = torch.zeros((), dtype=torch.float32, device=device)
                            flow_weight_sum = 0.0
                            for layer_id in on_policy_capture_ids:
                                student_ffn = rollout_student_mlp.get(int(layer_id))
                                teacher_ffn = rollout_teacher_mlp.get(int(layer_id))
                                if student_ffn is None or teacher_ffn is None:
                                    continue
                                layer_regime = (
                                    str(regime_labels[int(layer_id)])
                                    if int(layer_id) < len(regime_labels)
                                    else "llama_late"
                                )
                                layer_basis = regime_basis_device_map.get(
                                    layer_regime, basis_device
                                )
                                rollout_z_student = torch.matmul(
                                    student_ffn.to(dtype=torch.float32), layer_basis
                                )
                                rollout_z_teacher = torch.matmul(
                                    teacher_ffn.to(dtype=torch.float32), layer_basis
                                )
                                if core_use_metric_whitening:
                                    flow_metric_scale = _metric_inverse_sqrt(
                                        layer_metric_diag_device[int(layer_id)].view(1, -1),
                                        eps=tau_eps,
                                        trace_normalize=core_metric_trace_normalize,
                                        input_is_precision=core_metric_diag_is_precision,
                                    )
                                else:
                                    flow_metric_scale = torch.ones(
                                        (1, int(rollout_z_student.size(-1))),
                                        dtype=rollout_z_student.dtype,
                                        device=device,
                                    )
                                flow_layer, turning_layer, segment_count, _ = (
                                    _adaptive_token_flow_alignment(
                                        student_z=rollout_z_student,
                                        teacher_z=rollout_z_teacher,
                                        batch_indices=rollout_batch_indices,
                                        token_indices=rollout_token_indices,
                                        metric_scale=flow_metric_scale,
                                        energy_fraction=token_flow_energy_fraction,
                                        eps=tau_eps,
                                    )
                                )
                                if segment_count <= 0:
                                    continue
                                layer_weight = (
                                    float(layer_reliability_device[int(layer_id)].item())
                                    if core_use_reliability_weighting
                                    else 1.0
                                )
                                flow_sum = flow_sum + layer_weight * flow_layer
                                turning_sum = turning_sum + layer_weight * turning_layer
                                flow_weight_sum += layer_weight
                            if flow_weight_sum > 0.0:
                                on_policy_flow_loss = flow_sum / float(flow_weight_sum)
                                on_policy_turning_loss = turning_sum / float(flow_weight_sum)

                        ramp = min(
                            1.0,
                            max(
                                0.0,
                                float(pending_step - on_policy_start_step + 1)
                                / float(on_policy_ramp_steps),
                            ),
                        )
                        on_policy_loss = ramp * (
                            float(on_policy_lambda_kd) * on_policy_kd_loss
                            + float(on_policy_lambda_core) * on_policy_core_loss
                            + float(on_policy_lambda_flow) * on_policy_flow_loss
                            + float(on_policy_lambda_turning) * on_policy_turning_loss
                        )
                        (on_policy_loss / float(grad_accum_steps)).backward()
                        on_policy_active = True
                        loss = loss + on_policy_loss.detach()
                    finally:
                        for handle in rollout_student_handles:
                            try:
                                handle.remove()
                            except Exception:
                                pass

            if on_policy_active:
                on_policy_activation_count += 1
                on_policy_activation_token_count += float(on_policy_token_count)
                on_policy_activation_gate_sum += float(on_policy_gate_mean)

            generic_replay_ce_loss = torch.zeros((), dtype=torch.float32, device=device)
            generic_replay_active = False
            if (
                generic_replay_enabled
                and generic_replay_loader is not None
                and pending_step % generic_replay_interval == 0
            ):
                try:
                    generic_batch = next(generic_replay_iterator)
                except StopIteration:
                    generic_replay_iterator = iter(generic_replay_loader)
                    generic_batch = next(generic_replay_iterator)
                generic_input_ids = generic_batch["input_ids"].to(device)
                generic_attention_mask = generic_batch["attention_mask"].to(device)
                generic_prompt_lens = generic_batch.get("prompt_lens", None)
                if generic_prompt_lens is not None:
                    generic_prompt_lens = generic_prompt_lens.to(device)
                with ExitStack() as generic_backward_stack:
                    if dist_ctx.enabled and not should_optimizer_step and isinstance(student, DDP):
                        generic_backward_stack.enter_context(student.no_sync())
                    generic_output = student(
                        input_ids=generic_input_ids,
                        attention_mask=generic_attention_mask,
                        return_dict=True,
                        use_cache=False,
                    )
                    generic_logits = _extract_logits_from_model_output(generic_output).float()
                    generic_mask = shifted_target_mask(
                        input_ids=generic_input_ids,
                        attention_mask=generic_attention_mask,
                        prompt_lens=generic_prompt_lens,
                        scope="response",
                        pad_token_id=int(tokenizer.pad_token_id),
                        eos_token_id=eos_token_id,
                        exclude_eos=loss_exclude_eos,
                    )
                    generic_replay_ce_loss, _ = masked_next_token_cross_entropy(
                        logits=generic_logits,
                        input_ids=generic_input_ids,
                        token_mask=generic_mask,
                    )
                    (
                        float(lambda_generic_replay)
                        * generic_replay_ce_loss
                        / float(grad_accum_steps)
                    ).backward()
                generic_replay_active = True
                loss = loss + float(lambda_generic_replay) * generic_replay_ce_loss.detach()
            micro_in_accum += 1
            should_optimizer_step = micro_in_accum >= grad_accum_steps
            if should_optimizer_step:
                trainable_for_clip = (
                    bank_params + adapter_params + mixture_params + phase_projector_params
                )
                torch.nn.utils.clip_grad_norm_(trainable_for_clip, float(args.grad_clip))
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                micro_in_accum = 0
                step = int(pending_step)
            lr_now = float(optimizer.param_groups[0]["lr"])
            if pbar is not None and should_optimizer_step:
                pbar_pending += 1
                if step % log_every_steps == 0 or step >= total_steps:
                    pbar.update(pbar_pending)
                    pbar_pending = 0

            loss_item = dist_mean(float(loss.item()), device, dist_ctx)
            ce_item = dist_mean(float(ce_loss.item()), device, dist_ctx)
            response_ce_item = dist_mean(float(response_ce_loss.item()), device, dist_ctx)
            decision_ce_item = dist_mean(float(decision_ce_loss.item()), device, dist_ctx)
            kd_item = dist_mean(float(kd_loss.item()), device, dist_ctx)
            sage_gate_item = dist_mean(float(sage_stats.get("gate_mean", torch.zeros(())).item()), device, dist_ctx)
            sage_active_item = dist_mean(float(sage_stats.get("active_fraction", torch.zeros(())).item()), device, dist_ctx)
            sage_teacher_correct_item = dist_mean(float(sage_stats.get("teacher_correct_fraction", torch.zeros(())).item()), device, dist_ctx)
            hidden_item = dist_mean(float(hidden_loss.item()), device, dist_ctx)
            core_item = dist_mean(float(core_loss.item()), device, dist_ctx)
            point_core_item = dist_mean(float(point_core_loss.item()), device, dist_ctx)
            geodesic_core_item = dist_mean(float(geodesic_core_loss.item()), device, dist_ctx)
            relational_core_item = dist_mean(float(relational_core_loss.item()), device, dist_ctx)
            variance_core_item = dist_mean(float(variance_core_loss.item()), device, dist_ctx)
            manifold_core_item = dist_mean(float(manifold_core_loss.item()), device, dist_ctx)
            delta_manifold_core_item = dist_mean(float(delta_manifold_core_loss.item()), device, dist_ctx)
            generic_replay_ce_item = dist_mean(float(generic_replay_ce_loss.item()), device, dist_ctx)
            on_policy_loss_item = dist_mean(float(on_policy_loss.item()), device, dist_ctx)
            on_policy_kd_item = dist_mean(float(on_policy_kd_loss.item()), device, dist_ctx)
            on_policy_core_item = dist_mean(float(on_policy_core_loss.item()), device, dist_ctx)
            on_policy_gate_item = dist_mean(float(on_policy_gate_mean), device, dist_ctx)
            on_policy_tokens_item = dist_mean(float(on_policy_token_count), device, dist_ctx)
            layer_mixture_item = dist_mean(float(layer_mixture_loss.item()), device, dist_ctx)
            layer_mixture_nll_item = dist_mean(float(layer_mixture_stats["nll"].item()), device, dist_ctx)
            layer_mixture_entropy_item = dist_mean(float(layer_mixture_stats["gate_entropy"].item()), device, dist_ctx)
            layer_mixture_posterior_entropy_item = dist_mean(float(layer_mixture_stats["posterior_entropy"].item()), device, dist_ctx)
            layer_mixture_effective_components_item = dist_mean(float(layer_mixture_stats["effective_components"].item()), device, dist_ctx)
            layer_mixture_max_probability_item = dist_mean(float(layer_mixture_stats["max_probability"].item()), device, dist_ctx)
            layer_mixture_delta_l2_item = dist_mean(float(layer_mixture_stats["delta_l2"].item()), device, dist_ctx)
            phase_adaptive_core_item = dist_mean(
                float(phase_adaptive_core_loss.item()), device, dist_ctx
            )
            phase_adaptive_cosine_item = dist_mean(
                float(phase_adaptive_cosine.item()), device, dist_ctx
            )
            metric_scale = 1.0 / float(grad_accum_steps)
            running["loss"] += loss_item * metric_scale
            running["ce"] += ce_item * metric_scale
            running["response_ce"] += response_ce_item * metric_scale
            running["decision_ce"] += decision_ce_item * metric_scale
            running["kd"] += kd_item * metric_scale
            running["sage_gate"] += sage_gate_item * metric_scale
            running["sage_active"] += sage_active_item * metric_scale
            running["sage_teacher_correct"] += sage_teacher_correct_item * metric_scale
            running["sage_rate"] += float(teacher_coeff_now) * metric_scale
            running["hidden"] += hidden_item * metric_scale
            running["core"] += core_item * metric_scale
            running["point_core"] += point_core_item * metric_scale
            running["geodesic_core"] += geodesic_core_item * metric_scale
            running["relational_core"] += relational_core_item * metric_scale
            running["variance_core"] += variance_core_item * metric_scale
            running["manifold_core"] += manifold_core_item * metric_scale
            running["delta_manifold_core"] += delta_manifold_core_item * metric_scale
            running["generic_replay_ce"] += generic_replay_ce_item * metric_scale
            running["generic_replay_active"] += float(generic_replay_active) * metric_scale
            running["on_policy_loss"] += on_policy_loss_item * metric_scale
            running["on_policy_kd"] += on_policy_kd_item * metric_scale
            running["on_policy_core"] += on_policy_core_item * metric_scale
            running["on_policy_active"] += float(on_policy_active) * metric_scale
            running["on_policy_gate"] += on_policy_gate_item * metric_scale
            running["on_policy_tokens"] += on_policy_tokens_item * metric_scale
            running["layer_mixture"] += layer_mixture_item * metric_scale
            running["layer_mixture_nll"] += layer_mixture_nll_item * metric_scale
            running["layer_mixture_entropy"] += layer_mixture_entropy_item * metric_scale
            running["layer_mixture_posterior_entropy"] += layer_mixture_posterior_entropy_item * metric_scale
            running["layer_mixture_effective_components"] += layer_mixture_effective_components_item * metric_scale
            running["layer_mixture_max_probability"] += layer_mixture_max_probability_item * metric_scale
            running["layer_mixture_delta_l2"] += layer_mixture_delta_l2_item * metric_scale
            running["phase_adaptive_core"] += phase_adaptive_core_item * metric_scale
            running["phase_adaptive_cosine"] += phase_adaptive_cosine_item * metric_scale
            running["core_lambda"] += float(lambda_core_now) * metric_scale
            running["lr"] += float(lr_now) * metric_scale
            running["lr_bank"] += _first_param_group_lr(optimizer, "bank") * metric_scale
            running["lr_adapter"] += _first_param_group_lr(optimizer, "adapter") * metric_scale
            running["lr_mixture"] += _first_param_group_lr(optimizer, "mixture") * metric_scale
            running["lr_phase_projector"] += _first_param_group_lr(optimizer, "phase_projector") * metric_scale
            last_core_loss = core_item
            last_point_core_loss = point_core_item
            last_geodesic_core_loss = geodesic_core_item
            last_relational_core_loss = relational_core_item
            last_variance_core_loss = variance_core_item
            last_manifold_core_loss = manifold_core_item
            last_delta_manifold_core_loss = delta_manifold_core_item
            last_generic_replay_ce_loss = generic_replay_ce_item
            last_on_policy_loss = on_policy_loss_item
            last_on_policy_kd_loss = on_policy_kd_item
            last_on_policy_core_loss = on_policy_core_item
            last_on_policy_gate = on_policy_gate_item
            last_core_lambda = float(lambda_core_now)
            last_response_ce_loss = response_ce_item
            last_decision_ce_loss = decision_ce_item
            last_sage_gate_mean = sage_gate_item
            last_sage_active_fraction = sage_active_item
            last_layer_mixture_loss = layer_mixture_item
            last_layer_mixture_nll = layer_mixture_nll_item
            last_layer_mixture_entropy = layer_mixture_entropy_item
            last_layer_mixture_posterior_entropy = layer_mixture_posterior_entropy_item
            last_layer_mixture_effective_components = layer_mixture_effective_components_item
            last_layer_mixture_max_probability = layer_mixture_max_probability_item
            last_layer_mixture_delta_l2 = layer_mixture_delta_l2_item
            last_phase_adaptive_core_loss = phase_adaptive_core_item
            last_phase_adaptive_cosine = phase_adaptive_cosine_item

            if not should_optimizer_step:
                continue

            if step % log_every_steps == 0:
                denom = float(log_every_steps)
                avg_loss = running["loss"] / denom
                avg_ce = running["ce"] / denom
                avg_response_ce = running["response_ce"] / denom
                avg_decision_ce = running["decision_ce"] / denom
                avg_kd = running["kd"] / denom
                avg_sage_gate = running["sage_gate"] / denom
                avg_sage_active = running["sage_active"] / denom
                avg_sage_teacher_correct = running["sage_teacher_correct"] / denom
                avg_sage_rate = running["sage_rate"] / denom
                avg_hidden = running["hidden"] / denom
                avg_core = running["core"] / denom
                avg_point_core = running["point_core"] / denom
                avg_geodesic_core = running["geodesic_core"] / denom
                avg_relational_core = running["relational_core"] / denom
                avg_variance_core = running["variance_core"] / denom
                avg_manifold_core = running["manifold_core"] / denom
                avg_delta_manifold_core = running["delta_manifold_core"] / denom
                avg_generic_replay_ce = running["generic_replay_ce"] / denom
                avg_generic_replay_active = running["generic_replay_active"] / denom
                avg_on_policy_loss = running["on_policy_loss"] / denom
                avg_on_policy_kd = running["on_policy_kd"] / denom
                avg_on_policy_core = running["on_policy_core"] / denom
                avg_on_policy_active = running["on_policy_active"] / denom
                avg_on_policy_gate = running["on_policy_gate"] / denom
                avg_on_policy_tokens = running["on_policy_tokens"] / denom
                avg_layer_mixture = running["layer_mixture"] / denom
                avg_layer_mixture_nll = running["layer_mixture_nll"] / denom
                avg_layer_mixture_entropy = running["layer_mixture_entropy"] / denom
                avg_layer_mixture_posterior_entropy = running["layer_mixture_posterior_entropy"] / denom
                avg_layer_mixture_effective_components = running["layer_mixture_effective_components"] / denom
                avg_layer_mixture_max_probability = running["layer_mixture_max_probability"] / denom
                avg_layer_mixture_delta_l2 = running["layer_mixture_delta_l2"] / denom
                avg_phase_adaptive_core = running["phase_adaptive_core"] / denom
                avg_phase_adaptive_cosine = running["phase_adaptive_cosine"] / denom
                avg_core_lambda = running["core_lambda"] / denom
                avg_lr = running["lr"] / denom
                avg_lr_bank = running["lr_bank"] / denom
                avg_lr_adapter = running["lr_adapter"] / denom
                avg_lr_mixture = running["lr_mixture"] / denom
                avg_lr_phase_projector = running["lr_phase_projector"] / denom
                if pbar is not None:
                    postfix = {
                        "loss": f"{avg_loss:.4f}",
                        "ce": f"{avg_ce:.4f}",
                        "rce": f"{avg_response_ce:.4f}",
                        "dce": f"{avg_decision_ce:.4f}",
                        "kd": f"{avg_kd:.4f}",
                        "hidden": f"{avg_hidden:.4f}",
                        "core": f"{avg_core:.4f}",
                        "point": f"{avg_point_core:.4f}",
                        "geo": f"{avg_geodesic_core:.4f}",
                        "rel": f"{avg_relational_core:.4f}",
                        "var": f"{avg_variance_core:.4f}",
                        "replay": f"{avg_generic_replay_ce:.4f}",
                        "op": f"{avg_on_policy_loss:.4f}",
                        "op_gate": f"{avg_on_policy_gate:.2f}",
                        "mani": f"{avg_manifold_core:.4f}",
                        "dmani": f"{avg_delta_manifold_core:.2e}",
                        "mix": f"{avg_layer_mixture:.3f}",
                        "Hmix": f"{avg_layer_mixture_entropy:.2f}",
                        "phase": f"{avg_phase_adaptive_core:.3f}",
                        "l_core": f"{avg_core_lambda:.3f}",
                        "lr": f"{avg_lr:.2e}",
                    }
                    pbar.set_postfix(**postfix)
                if is_main_process(dist_ctx):
                    print(
                        f"[Compress][{training_stage}][{step}/{int(args.steps)}] "
                        f"loss={avg_loss:.4f} ce={avg_ce:.4f} kd={avg_kd:.4f} "
                        f"response_ce={avg_response_ce:.4f} sage_gate={avg_sage_gate:.4f} "
                        f"decision_ce={avg_decision_ce:.4f} "
                        f"sage_active={avg_sage_active:.4f} teacher_correct={avg_sage_teacher_correct:.4f} "
                        f"teacher_rate={avg_sage_rate:.4f} "
                        f"hidden={avg_hidden:.4f} "
                        f"core={avg_core:.4f} point_core={avg_point_core:.4f} "
                        f"geodesic_core={avg_geodesic_core:.4f} "
                        f"relational_core={avg_relational_core:.4f} "
                        f"variance_core={avg_variance_core:.4f} "
                        f"manifold_core={avg_manifold_core:.4f} "
                        f"delta_manifold_core={avg_delta_manifold_core:.6e} "
                        f"generic_replay_ce={avg_generic_replay_ce:.4f} "
                        f"generic_replay_active={avg_generic_replay_active:.4f} "
                        f"on_policy_loss={avg_on_policy_loss:.4f} "
                        f"on_policy_kd={avg_on_policy_kd:.4f} "
                        f"on_policy_core={avg_on_policy_core:.4f} "
                        f"on_policy_active={avg_on_policy_active:.4f} "
                        f"on_policy_gate={avg_on_policy_gate:.4f} "
                        f"on_policy_tokens={avg_on_policy_tokens:.2f} "
                        f"layer_mixture={avg_layer_mixture:.6f} "
                        f"mixture_nll={avg_layer_mixture_nll:.6f} "
                        f"mixture_entropy={avg_layer_mixture_entropy:.6f} "
                        f"mixture_posterior_entropy={avg_layer_mixture_posterior_entropy:.6f} "
                        f"mixture_effective_components={avg_layer_mixture_effective_components:.4f} "
                        f"mixture_max_probability={avg_layer_mixture_max_probability:.4f} "
                        f"mixture_delta_l2={avg_layer_mixture_delta_l2:.6e} "
                        f"phase_adaptive_core={avg_phase_adaptive_core:.6f} "
                        f"phase_adaptive_cosine={avg_phase_adaptive_cosine:.6f} "
                        f"lambda_phase_adaptive_core={float(lambda_phase_adaptive_core):.6f} "
                        f"lambda_layer_mixture={float(lambda_layer_mixture):.6f} "
                        f"lambda_geodesic_core={float(lambda_geodesic_core):.4f} "
                        f"lambda_relational_core={float(lambda_relational_core):.4f} "
                        f"lambda_variance_core={float(lambda_variance_core):.4f} "
                        f"lambda_generic_replay={float(lambda_generic_replay):.4f} "
                        f"lambda_manifold_core={float(lambda_manifold_core):.4f} "
                        f"lambda_delta_manifold_core={float(lambda_delta_manifold_core):.4f} "
                        f"lambda_core={avg_core_lambda:.4f} "
                        f"lr={avg_lr:.2e} "
                        f"lr_bank={avg_lr_bank:.2e} lr_adapter={avg_lr_adapter:.2e} "
                        f"lr_mixture={avg_lr_mixture:.2e} "
                        f"lr_phase_projector={avg_lr_phase_projector:.2e}",
                        flush=True,
                    )
                running = {
                    "loss": 0.0,
                    "ce": 0.0,
                    "response_ce": 0.0,
                    "decision_ce": 0.0,
                    "kd": 0.0,
                    "sage_gate": 0.0,
                    "sage_active": 0.0,
                    "sage_teacher_correct": 0.0,
                    "sage_rate": 0.0,
                    "hidden": 0.0,
                    "core": 0.0,
                    "point_core": 0.0,
                    "geodesic_core": 0.0,
                    "relational_core": 0.0,
                    "variance_core": 0.0,
                    "manifold_core": 0.0,
                    "delta_manifold_core": 0.0,
                    "generic_replay_ce": 0.0,
                    "generic_replay_active": 0.0,
                    "on_policy_loss": 0.0,
                    "on_policy_kd": 0.0,
                    "on_policy_core": 0.0,
                    "on_policy_active": 0.0,
                    "on_policy_gate": 0.0,
                    "on_policy_tokens": 0.0,
                    "layer_mixture": 0.0,
                    "layer_mixture_nll": 0.0,
                    "layer_mixture_entropy": 0.0,
                    "layer_mixture_posterior_entropy": 0.0,
                    "layer_mixture_effective_components": 0.0,
                    "layer_mixture_max_probability": 0.0,
                    "layer_mixture_delta_l2": 0.0,
                    "phase_adaptive_core": 0.0,
                    "phase_adaptive_cosine": 0.0,
                    "core_lambda": 0.0,
                    "lr": 0.0,
                    "lr_bank": 0.0,
                    "lr_adapter": 0.0,
                    "lr_mixture": 0.0,
                    "lr_phase_projector": 0.0,
                }

            if val_every > 0 and step % int(val_every) == 0:
                _run_and_record_validation(step_for_val=int(step), allow_best=True)

    if pbar is not None:
        if pbar_pending > 0:
            pbar.update(pbar_pending)
        pbar.close()

    # Decide whether to use best-val checkpoint or last-step state
    use_best_val = best_val_step >= 0 and os.path.isfile(best_val_ckpt_path)
    if use_best_val:
        best_ckpt = torch.load(best_val_ckpt_path, map_location="cpu")
        payload = best_ckpt["shared_state"]
        best_mixture_state = best_ckpt.get("layer_mixture_transport_state", {})
        if layer_mixture_transport is not None and isinstance(best_mixture_state, dict) and best_mixture_state:
            unwrap_model(layer_mixture_transport).load_state_dict(best_mixture_state, strict=True)
        best_phase_projector_state = best_ckpt.get("phase_projector_state", {})
        if (
            phase_projector_bank is not None
            and isinstance(best_phase_projector_state, dict)
            and best_phase_projector_state
        ):
            unwrap_model(phase_projector_bank).load_state_dict(
                best_phase_projector_state, strict=True
            )
        print(
            f"[Compress] using best-val checkpoint: step={best_val_step} "
            f"{val_selection_metric}={best_val_loss:.4f} "
            f"(final step={step})",
            flush=True,
        )
    else:
        payload = extract_shared_state(student, layer_to_proto=layer_to_proto)
        if best_val_step < 0:
            print("[Compress] no validation was run; using final step state.", flush=True)
        else:
            print("[Compress] best-val checkpoint not found; using final step state.", flush=True)

    payload["meta"] = _build_compress_meta(
        actual_final_step=int(step),
        best_step=int(best_val_step),
        best_loss=float(best_val_loss),
        used_best_val_ckpt=bool(use_best_val),
    )

    train_wall_elapsed_sec = max(1e-9, time.perf_counter() - train_wall_t0)
    peak_gpu_memory_bytes = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    effective_global_batch = int(args.batch_size) * int(dist_ctx.world_size) * int(grad_accum_steps)
    measured_optimizer_steps = max(1, int(step))
    ckpt_path = os.path.join(output_dir, "shared_student.pt")
    report = {
        "training_stage": str(training_stage),
        "shared_student_ckpt": ckpt_path,
        "checkpoint_dir": ckpt_dir,
        "checkpoint_policy": "best_val_only",
        "best_val_ckpt": best_val_ckpt_path if os.path.isfile(best_val_ckpt_path) else "",
        "base_model": student_model_path,
        "teacher_model": teacher_model_path,
        "teacher_free_ce": bool(teacher_free_ce),
        "teacher_deploy_bundle": str(teacher_deploy_bundle),
        "teacher_loader_resolved": (
            "skipped_teacher_free_ce"
            if teacher_free_ce
            else ("shared_deploy_bundle" if teacher_bundle is not None else "native")
        ),
        "teacher_quant_report": dict(teacher_quant_report),
        "lora_rank": int(args.lora_rank),
        "lora_alpha": float(args.lora_alpha),
        "use_layer_scalar": bool(getattr(args, "use_layer_scalar", True)),
        "adapter_every_layer": bool(getattr(args, "adapter_every_layer", False)),
        "sharing_parameterization": str(
            getattr(args, "sharing_parameterization", "full_parallel")
        ),
        "init_shared_student_ckpt": str(init_shared_student_ckpt),
        "init_shared": bool(init_shared),
        "residual_svd_initialization": residual_svd_init_report,
        "internal_weight_delta_initialization": internal_weight_delta_init_report,
        "proto_seed_strategy": str(proto_seed_strategy),
        "proto_seed_strategy_resolved": str(resolved_seed_strategy),
        "proto_seed_layers": {str(k): int(v) for k, v in seed_layer_for_proto.items()},
        "sharing_policy_path": str(sharing_policy_path),
        "share_group_summary": sharing_policy.get("groups", []) if isinstance(sharing_policy, dict) else [],
        "layer_to_regime": list(regime_labels),
        "lr_schedule": str(scheduler_meta["schedule"]),
        "lr_warmup_steps": int(scheduler_meta["warmup_steps"]),
        "lr_warmup_ratio": float(scheduler_meta["warmup_ratio"]),
        "lr_min_ratio": float(scheduler_meta["min_lr_ratio"]),
        "layer_to_proto": [int(x) for x in layer_to_proto],
        "distill_mode": str(distill_mode),
        "training_prompt_mode": str(getattr(args, "training_prompt_mode", "legacy_sft")),
        "loss_scope": str(loss_scope),
        "loss_exclude_eos": bool(loss_exclude_eos),
        "sage_config": dict(sage_config),
        "val_selection_metric": str(val_selection_metric),
        "val_include_step0_candidate": bool(val_include_step0_candidate),
        "val_min_improvement": float(val_min_improvement),
        "lambda_ce": float(ce_coeff),
        "lambda_kd": float(kd_coeff),
        "kd_temperature": float(kd_temperature),
        "lambda_core": float(lambda_core),
        "core_lambda_schedule": str(core_lambda_schedule),
        "core_lambda_warmup_ratio": float(core_lambda_warmup_ratio),
        "core_lambda_cutoff_ratio": float(core_lambda_cutoff_ratio),
        "core_metric_source": str(core_metric_source),
        "core_coordinate_mode": str(core_coordinate_mode),
        "core_basis_mode": str(core_basis_mode),
        "core_metric_diag_path": str(core_metric_diag_path),
        "core_metric_diag_mode": str(core_metric_diag_mode),
        "core_layer_ids": [int(x) for x in tau_layer_ids],
        "core_metric_eps": float(tau_eps),
        "core_use_metric_whitening": bool(core_use_metric_whitening),
        "core_metric_trace_normalize": bool(core_metric_trace_normalize),
        "core_use_reliability_weighting": bool(core_use_reliability_weighting),
        "core_token_selection": str(core_token_selection),
        "core_candidate_tokens": int(core_candidate_tokens),
        "core_iets_temperature": float(core_iets_temperature),
        "core_iets_anchor_boost": float(core_iets_anchor_boost),
        "core_iets_energy_alpha": float(core_iets_energy_alpha),
        "core_iets_entropy_beta": float(core_iets_entropy_beta),
        "core_iets_topk": int(core_iets_topk),
        "core_use_information_weighting": bool(core_use_information_weighting),
        "core_information_power": float(core_information_power),
        "lambda_layer_mixture": float(lambda_layer_mixture),
        "layer_mixture_enabled": bool(layer_mixture_transport is not None),
        "layer_mixture_lr": float(layer_mixture_lr),
        "layer_mixture_config": (
            unwrap_model(layer_mixture_transport).config_dict()
            if layer_mixture_transport is not None
            else {}
        ),
        "lambda_phase_adaptive_core": float(lambda_phase_adaptive_core),
        "phase_projector_enabled": bool(phase_projector_bank is not None),
        "phase_projector_bank_path": str(phase_projector_bank_path),
        "phase_projector_mode": str(phase_projector_mode),
        "phase_projector_lr": float(phase_projector_lr),
        "phase_projector_config": (
            unwrap_model(phase_projector_bank).config_dict()
            if phase_projector_bank is not None
            else {}
        ),
        "lambda_geodesic_core": float(lambda_geodesic_core),
        "geodesic_core_max_layer_gap": int(geodesic_core_max_layer_gap),
        "lambda_relational_core": float(lambda_relational_core),
        "lambda_variance_core": float(lambda_variance_core),
        "variance_core_floor_ratio": float(variance_core_floor_ratio),
        "lambda_token_flow_core": float(lambda_token_flow_core),
        "lambda_token_turning_core": float(lambda_token_turning_core),
        "token_flow_energy_fraction": float(token_flow_energy_fraction),
        "generic_replay_enabled": bool(generic_replay_enabled),
        "generic_replay_data_path": str(generic_replay_data_path),
        "generic_replay_interval": int(generic_replay_interval),
        "generic_replay_batch_size": int(generic_replay_batch_size),
        "generic_replay_max_records": int(generic_replay_max_records),
        "generic_replay_tokenized_count": int(generic_replay_tokenized_count),
        "lambda_generic_replay": float(lambda_generic_replay),
        "on_policy_enabled": bool(on_policy_enabled_resolved),
        "on_policy_start_step": int(on_policy_start_step),
        "on_policy_interval": int(on_policy_interval),
        "on_policy_batch_size": int(on_policy_batch_size),
        "on_policy_max_new_tokens": int(on_policy_max_new_tokens),
        "on_policy_generation_temperature": float(on_policy_generation_temperature),
        "on_policy_top_p": float(on_policy_top_p),
        "on_policy_divergence": str(on_policy_divergence),
        "on_policy_temperature": float(on_policy_temperature),
        "on_policy_lambda_kd": float(on_policy_lambda_kd),
        "on_policy_lambda_core": float(on_policy_lambda_core),
        "on_policy_lambda_flow": float(on_policy_lambda_flow),
        "on_policy_lambda_turning": float(on_policy_lambda_turning),
        "on_policy_teacher_advantage_margin": float(on_policy_teacher_advantage_margin),
        "on_policy_final_answer_weight": float(on_policy_final_answer_weight),
        "on_policy_core_tokens": int(on_policy_core_tokens),
        "on_policy_ramp_steps": int(on_policy_ramp_steps),
        "on_policy_activation_count": int(on_policy_activation_count),
        "on_policy_activation_token_count": float(on_policy_activation_token_count),
        "on_policy_activation_gate_mean": (
            float(on_policy_activation_gate_sum) / float(on_policy_activation_count)
            if on_policy_activation_count > 0
            else 0.0
        ),
        "lambda_manifold_core": float(lambda_manifold_core),
        "manifold_core_temperature": float(manifold_core_temperature),
        "manifold_core_enabled": bool(manifold_core_enabled),
        "manifold_anchor_count": int(manifold_anchor_count),
        "manifold_dim": int(manifold_dim),
        "lambda_delta_manifold_core": float(lambda_delta_manifold_core),
        "delta_manifold_core_temperature": float(delta_manifold_core_temperature),
        "delta_manifold_risk_weight": float(delta_manifold_risk_weight),
        "delta_manifold_core_enabled": bool(delta_manifold_core_enabled),
        "delta_manifold_anchor_count": int(delta_manifold_anchor_count),
        "delta_manifold_dim": int(delta_manifold_dim),
        "layer_reliability": [float(x) for x in layer_reliability_cpu.tolist()],
        "last_train_core_loss": float(last_core_loss),
        "last_train_response_ce": float(last_response_ce_loss),
        "last_train_decision_ce": float(last_decision_ce_loss),
        "last_train_sage_gate_mean": float(last_sage_gate_mean),
        "last_train_sage_active_fraction": float(last_sage_active_fraction),
        "last_train_point_core_loss": float(last_point_core_loss),
        "last_train_geodesic_core_loss": float(last_geodesic_core_loss),
        "last_train_relational_core_loss": float(last_relational_core_loss),
        "last_train_variance_core_loss": float(last_variance_core_loss),
        "last_train_manifold_core_loss": float(last_manifold_core_loss),
        "last_train_delta_manifold_core_loss": float(last_delta_manifold_core_loss),
        "last_train_generic_replay_ce_loss": float(last_generic_replay_ce_loss),
        "last_train_on_policy_loss": float(last_on_policy_loss),
        "last_train_on_policy_kd_loss": float(last_on_policy_kd_loss),
        "last_train_on_policy_core_loss": float(last_on_policy_core_loss),
        "last_train_on_policy_gate": float(last_on_policy_gate),
        "last_train_layer_mixture_loss": float(last_layer_mixture_loss),
        "last_train_layer_mixture_nll": float(last_layer_mixture_nll),
        "last_train_layer_mixture_entropy": float(last_layer_mixture_entropy),
        "last_train_layer_mixture_posterior_entropy": float(last_layer_mixture_posterior_entropy),
        "last_train_layer_mixture_effective_components": float(last_layer_mixture_effective_components),
        "last_train_layer_mixture_max_probability": float(last_layer_mixture_max_probability),
        "last_train_layer_mixture_delta_l2": float(last_layer_mixture_delta_l2),
        "last_train_phase_adaptive_core_loss": float(last_phase_adaptive_core_loss),
        "last_train_phase_adaptive_cosine": float(last_phase_adaptive_cosine),
        "last_train_lambda_core": float(last_core_lambda),
        "weight_decay": float(args.weight_decay),
        "val_data_path": val_data_path,
        "val_every": int(val_every),
        "val_max_records": int(val_max_records),
        "val_max_batches": int(val_max_batches),
        "val_subspace_viz_enable": bool(getattr(args, "val_subspace_viz_enable", False)),
        "val_subspace_viz_max_points_per_regime": int(getattr(args, "val_subspace_viz_max_points_per_regime", 256)),
        "val_subspace_viz_dir": val_subspace_viz_dir if bool(getattr(args, "val_subspace_viz_enable", False)) else "",
        "val_tokenized": int(val_tokenized_count),
        "best_val_step": int(best_val_step),
        "best_val_loss": float(best_val_loss) if best_val_step >= 0 else None,
        "used_best_val_ckpt": bool(use_best_val),
        "actual_final_step": int(step),
        "training_wall_elapsed_sec": float(train_wall_elapsed_sec),
        "training_ms_per_optimizer_step": float(1000.0 * train_wall_elapsed_sec / measured_optimizer_steps),
        "training_examples_per_sec": float(effective_global_batch * measured_optimizer_steps / train_wall_elapsed_sec),
        "peak_gpu_memory_bytes_local_rank": int(peak_gpu_memory_bytes),
        "world_size": int(dist_ctx.world_size),
        "per_rank_batch_size": int(args.batch_size),
        "global_batch_size": int(args.batch_size) * int(dist_ctx.world_size),
        "gradient_accumulation_steps": int(grad_accum_steps),
        "effective_global_batch_size": int(args.batch_size) * int(dist_ctx.world_size) * int(grad_accum_steps),
    }
    if val_history:
        val_history_path = os.path.join(output_dir, "compress_val_history.json")
        report["val_history_path"] = val_history_path
        report["val_last"] = val_history[-1]
    report_path = os.path.join(output_dir, "compress_report.json")
    if bool(getattr(args, "on_policy_require_activation", False)) and int(on_policy_activation_count) <= 0:
        raise RuntimeError(
            "on-policy activation was required, but no training step activated the on-policy channel"
        )
    if is_main_process(dist_ctx):
        torch.save(payload, ckpt_path)
        if layer_mixture_transport is not None:
            mixture_state_path = os.path.join(output_dir, "layer_mixture_transport.pt")
            torch.save(
                {
                    "training_only": True,
                    "exported_for_inference": False,
                    "selected_step": int(best_val_step if use_best_val else step),
                    "config": unwrap_model(layer_mixture_transport).config_dict(),
                    "state_dict": {
                        key: value.detach().cpu()
                        for key, value in unwrap_model(layer_mixture_transport).state_dict().items()
                    },
                },
                mixture_state_path,
            )
            report["layer_mixture_state_path"] = mixture_state_path
        if phase_projector_bank is not None:
            phase_projector_state_path = os.path.join(
                output_dir, "phase_projector_student.pt"
            )
            torch.save(
                {
                    "training_only": True,
                    "exported_for_inference": False,
                    "selected_step": int(best_val_step if use_best_val else step),
                    "config": unwrap_model(phase_projector_bank).config_dict(),
                    "state_dict": {
                        key: value.detach().cpu()
                        for key, value in unwrap_model(phase_projector_bank).state_dict().items()
                    },
                },
                phase_projector_state_path,
            )
            report["phase_projector_state_path"] = phase_projector_state_path
        if val_history:
            save_json(val_history_path, {"history": val_history})
        save_json(report_path, report)
        print(f"[Compress] saved {ckpt_path}")
        print(f"[Compress] saved {report_path}")
    if dist_ctx.enabled:
        dist_barrier(dist_ctx)
    finalize_distributed(dist_ctx)
    return report


def _quantize_rowwise_int4(weight: torch.Tensor) -> Dict[str, torch.Tensor]:
    if weight.dim() != 2:
        raise ValueError("Only 2D weights are supported for int4 quantization.")
    w = weight.detach().cpu().float()
    scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 7.0
    q = torch.clamp(torch.round(w / scale), min=-8, max=7).to(torch.int16) + 8
    if q.size(1) % 2 == 1:
        q = torch.cat([q, torch.zeros((q.size(0), 1), dtype=q.dtype)], dim=1)
    q_u8 = q.to(torch.uint8)
    lo = q_u8[:, 0::2]
    hi = q_u8[:, 1::2]
    packed = lo | (hi << 4)
    return {
        "packed": packed.contiguous(),
        "scale": scale.squeeze(1).contiguous(),
        "in_features": torch.tensor(int(weight.size(1)), dtype=torch.int32),
        "out_features": torch.tensor(int(weight.size(0)), dtype=torch.int32),
    }


def _try_load_meta_from_neighbor_compress_steps(shared_student_ckpt: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    ckpt_path = os.path.abspath(str(shared_student_ckpt))
    ckpt_dir = os.path.dirname(ckpt_path)
    candidates = glob.glob(os.path.join(ckpt_dir, "compress_step_*.pt"))
    if not candidates:
        return None, None

    def _step_num(path: str) -> int:
        base = os.path.basename(path)
        prefix = "compress_step_"
        suffix = ".pt"
        if not (base.startswith(prefix) and base.endswith(suffix)):
            return -1
        raw = base[len(prefix) : -len(suffix)]
        try:
            return int(raw)
        except Exception:
            return -1

    for path in sorted(candidates, key=_step_num, reverse=True):
        try:
            payload = torch.load(path, map_location="cpu")
        except Exception:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("meta"), dict):
            return payload["meta"], path
    return None, None


def stage_export(args: argparse.Namespace) -> Dict[str, Any]:
    dist_ctx = init_distributed("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = str(args.output_dir or f"./out/newthesis_export_{now_tag()}")
    if dist_ctx.enabled and not is_main_process(dist_ctx):
        dist_barrier(dist_ctx)
        finalize_distributed(dist_ctx)
        return {}
    ensure_dir(output_dir)
    atlas = torch.load(args.atlas_path, map_location="cpu")
    loaded = torch.load(args.shared_student_ckpt, map_location="cpu")
    shared = loaded
    compress_checkpoint: Optional[str] = None
    step: Optional[int] = None
    if isinstance(loaded, dict) and "shared_state" in loaded:
        # Accept phase1.5 compress checkpoints, including best-val-only payloads.
        shared = loaded.get("shared_state") or {}
        compress_checkpoint = str(args.shared_student_ckpt)
        try:
            step = int(loaded.get("step"))
        except Exception:
            step = None
        # Attach meta if it only exists at top-level.
        if isinstance(shared, dict) and "meta" not in shared and isinstance(loaded.get("meta"), dict):
            shared = dict(shared)
            shared["meta"] = loaded["meta"]
        if isinstance(shared, dict) and step is not None and "step" not in shared:
            shared = dict(shared)
            shared["step"] = step
    if not isinstance(shared, dict) or "bank_state" not in shared:
        raise ValueError("Invalid --shared_student_ckpt payload: expected dict with key bank_state (or compress checkpoint with shared_state.bank_state).")

    shared_meta = shared.get("meta", {})
    if not isinstance(shared_meta, dict):
        shared_meta = {}
    shared_meta = dict(shared_meta)
    meta_source = "shared_meta"
    # Older compress_best_val.pt payloads may miss provenance; recover it from legacy step checkpoints if available.
    required_meta_keys = ("base_model", "lora_rank")
    missing_required = [k for k in required_meta_keys if k not in shared_meta]
    if missing_required:
        neighbor_meta, neighbor_ckpt = _try_load_meta_from_neighbor_compress_steps(str(args.shared_student_ckpt))
        if isinstance(neighbor_meta, dict):
            # Fill all missing keys from neighbor meta (not only required ones) for better provenance.
            for key, value in neighbor_meta.items():
                if key not in shared_meta:
                    shared_meta[key] = value
            meta_source = f"neighbor_ckpt:{os.path.basename(str(neighbor_ckpt))}"
    if "base_model" not in shared_meta:
        raise ValueError(
            "Cannot resolve shared meta.base_model from --shared_student_ckpt. "
            "Use phase1_5_compress/shared_student.pt, a self-contained compress_best_val.pt, "
            "or a legacy compress_step_*.pt checkpoint."
        )
    shared = dict(shared)
    shared["meta"] = shared_meta

    quant_bank: Dict[str, Dict[str, torch.Tensor]] = {}
    for name, tensor in shared["bank_state"].items():
        if name.endswith("weight") and torch.is_tensor(tensor) and tensor.dim() == 2 and int(args.quant_bits) == 4:
            quant_bank[name] = _quantize_rowwise_int4(tensor)
    atlas_bundle = dict(atlas) if isinstance(atlas, dict) else atlas
    bundle = {
        "base_model": str(shared_meta["base_model"]),
        "atlas": atlas_bundle,
        "shared_student": shared,
        "quant_bits": int(args.quant_bits),
        "quant_bank_int4": quant_bank,
    }
    bundle_path = os.path.join(output_dir, "deploy_bundle.pt")
    torch.save(bundle, bundle_path)
    report = {
        "deploy_bundle": bundle_path,
        "atlas_path": str(args.atlas_path),
        "shared_student_ckpt": str(args.shared_student_ckpt),
        "compress_checkpoint": compress_checkpoint,
        "step": int(step) if step is not None else None,
        "meta_source": str(meta_source),
        "quant_bits": int(args.quant_bits),
        "quantized_weight_count": int(len(quant_bank)),
    }
    save_json(os.path.join(output_dir, "deploy_report.json"), report)
    print(f"[Export] saved {bundle_path}")
    if dist_ctx.enabled:
        dist_barrier(dist_ctx)
        finalize_distributed(dist_ctx)
    return report


def _dequantize_rowwise_int4(quant_payload: Dict[str, torch.Tensor]) -> torch.Tensor:
    packed = quant_payload["packed"].to(dtype=torch.uint8, device="cpu")
    scale = quant_payload["scale"].to(dtype=torch.float32, device="cpu")
    in_features = int(quant_payload["in_features"].item()) if torch.is_tensor(quant_payload["in_features"]) else int(quant_payload["in_features"])
    out_features = int(quant_payload["out_features"].item()) if torch.is_tensor(quant_payload["out_features"]) else int(quant_payload["out_features"])
    lo = (packed & 0x0F).to(torch.int16)
    hi = ((packed >> 4) & 0x0F).to(torch.int16)
    q = torch.empty((int(packed.size(0)), int(packed.size(1) * 2)), dtype=torch.int16)
    q[:, 0::2] = lo
    q[:, 1::2] = hi
    q = q[:, :in_features]
    w = (q.to(torch.float32) - 8.0) * scale.view(-1, 1)
    if int(w.size(0)) != out_features:
        raise ValueError(f"Invalid quant payload shape: expected out={out_features}, got {int(w.size(0))}")
    return w.contiguous()


def _apply_quant_bank_int4(
    *,
    model: nn.Module,
    quant_bank_int4: Dict[str, Dict[str, torch.Tensor]],
) -> Dict[str, Any]:
    root = getattr(model, "model", model)
    bank = getattr(root, "shared_mlp_bank", None)
    if bank is None:
        raise RuntimeError("shared_mlp_bank not found for quantized eval.")
    param_map = dict(bank.named_parameters())
    missing: List[str] = []
    applied = 0
    for name, payload in quant_bank_int4.items():
        param = param_map.get(str(name))
        if param is None:
            missing.append(str(name))
            continue
        deq = _dequantize_rowwise_int4(payload).to(device=param.device, dtype=param.dtype)
        if tuple(deq.shape) != tuple(param.data.shape):
            raise ValueError(f"Quantized weight shape mismatch for {name}: {tuple(deq.shape)} vs {tuple(param.data.shape)}")
        param.data.copy_(deq)
        applied += 1
    return {
        "available_count": int(len(quant_bank_int4)),
        "applied_count": int(applied),
        "missing_count": int(len(missing)),
        "missing_keys": missing,
    }


def _build_shared_model_for_eval(
    *,
    base_model: str,
    atlas_payload: Dict[str, Any],
    shared_payload: Dict[str, Any],
    quant_bank_int4: Optional[Dict[str, Dict[str, torch.Tensor]]],
    use_quant_bank_int4: bool,
    device: torch.device,
    dtype: torch.dtype,
    trust_remote_code: bool,
) -> Tuple[nn.Module, Dict[str, Any]]:
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        trust_remote_code=bool(trust_remote_code),
    ).to(device)
    meta = shared_payload.get("meta", {})
    lora_rank = int(meta.get("lora_rank", 0))
    lora_alpha = float(meta.get("lora_alpha", lora_rank))
    atlas_basis = atlas_payload.get("basis", None) if isinstance(atlas_payload, dict) else None
    load_shared_state(
        model,
        shared_payload,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
    )
    quant_report: Dict[str, Any] = {
        "requested": bool(use_quant_bank_int4),
        "enabled": False,
        "available_count": int(len(quant_bank_int4 or {})),
        "applied_count": 0,
        "missing_count": 0,
        "missing_keys": [],
    }
    if bool(use_quant_bank_int4):
        if not quant_bank_int4:
            raise ValueError("--use_quant_bank_int4=True but deploy bundle has no quant_bank_int4")
        apply_report = _apply_quant_bank_int4(model=model, quant_bank_int4=quant_bank_int4)
        quant_report.update(apply_report)
        quant_report["enabled"] = True
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, quant_report


def _collect_shared_model_diagnostics(model: nn.Module) -> Dict[str, Any]:
    per_layer: List[Dict[str, Any]] = []
    for layer_idx, layer in enumerate(_resolve_layers(model)):
        adapter = getattr(layer, "mlp", None)
        if not isinstance(adapter, SharedMLPAdapter):
            continue
        per_layer.append({
            "layer": int(layer_idx),
            "proto_id": int(getattr(adapter, "proto_id", -1)),
            "lora_rank": int(getattr(adapter, "lora_rank", 0)),
            "lora_alpha": float(getattr(adapter, "lora_alpha", 0.0)),
            "lora_scaling": float(getattr(adapter, "lora_scaling", 0.0)),
        })
    return {
        "layer_count": int(len(per_layer)),
        "per_layer": per_layer,
    }

def _candidates_for_dataset(dataset: str) -> List[str]:
    name = str(dataset).strip()
    if name.lower() == "gsm8k":
        return ["numeric"]
    if name == "boolq":
        return ["true", "false"]
    if name == "piqa":
        return ["solution1", "solution2"]
    if name == "winogrande":
        return ["option1", "option2"]
    if name == "hellaswag":
        return ["ending1", "ending2", "ending3", "ending4"]
    if name in {"ARC-Challenge", "ARC-Easy", "openbookqa"}:
        return ["answer1", "answer2", "answer3", "answer4"]
    if name == "social_i_qa":
        return ["answer1", "answer2", "answer3"]
    if name == "csqa":
        return ["answer1", "answer2", "answer3", "answer4", "answer5"]
    return ["true", "false"]


def _hint_for_dataset(dataset: str) -> str:
    name = str(dataset).strip()
    if name.lower() == "gsm8k":
        return "Solve the problem step by step and finish with 'The answer is <number>'."
    if name == "boolq":
        return "Answer with exactly one word: true or false."
    if name == "piqa":
        return "Answer with exactly one token: solution1 or solution2."
    if name == "winogrande":
        return "Answer with exactly one token: option1 or option2."
    if name == "hellaswag":
        return "Answer with exactly one token: ending1, ending2, ending3, or ending4."
    if name in {"ARC-Challenge", "ARC-Easy", "openbookqa"}:
        return "Answer with exactly one token: answer1, answer2, answer3, or answer4."
    if name == "social_i_qa":
        return "Answer with exactly one token: answer1, answer2, or answer3."
    if name == "csqa":
        return "Answer with exactly one token: answer1, answer2, answer3, answer4, or answer5."
    return "Answer concisely."


def _build_prompt(item: Dict[str, Any], dataset: str) -> str:
    if str(dataset).strip().lower() == "gsm8k":
        return _build_gsm8k_prompt(item)
    instruction = str(item.get("instruction", item.get("question", ""))).strip()
    input_text = str(item.get("input", item.get("context", ""))).strip()
    hint = _hint_for_dataset(dataset)
    if input_text:
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Input:\n"
            f"{input_text}\n\n"
            f"{hint}\n\n"
            "### Response:\n"
            "Answer:"
        )
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n"
        f"{instruction}\n\n"
        f"{hint}\n\n"
        "### Response:\n"
        "Answer:"
    )


def _build_gsm8k_prompt(item: Dict[str, Any]) -> str:
    instruction = str(item.get("instruction", item.get("question", ""))).strip()
    input_text = str(item.get("input", item.get("context", ""))).strip()
    if input_text:
        return (
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Input:\n"
            f"{input_text}\n\n"
            "### Response:\n"
        )
    return (
        "### Instruction:\n"
        f"{instruction}\n\n"
        "### Response:\n"
    )


def _build_math_reasoning_chat_prompt(tokenizer, item: Dict[str, Any]) -> str:
    instruction = str(item.get("instruction", item.get("question", ""))).strip()
    input_text = str(item.get("input", item.get("context", ""))).strip()
    if input_text:
        instruction = f"{instruction}\n\n{input_text}"
    messages = [
        {
            "role": "system",
            "content": (
                "You are a careful mathematical reasoning assistant. Solve the problem step by step, "
                "show the reasoning needed to verify the result, and finish with a clearly marked final answer. "
                "For arithmetic answers use 'The answer is <number>.'; for symbolic answers use \\\\boxed{...}."
            ),
        },
        {"role": "user", "content": instruction},
    ]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return (
        f"{messages[0]['content']}\n\n### Problem:\n{instruction}\n\n### Solution:\n"
    )


_NUMBER_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")


def _normalize_gsm8k_number(text: Any) -> str:
    raw = str(text).strip()
    if not raw:
        return ""
    raw = raw.replace(",", "")
    try:
        value = Decimal(raw)
    except InvalidOperation:
        return raw
    normalized = format(value.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    if normalized == "-0":
        normalized = "0"
    return normalized


def _extract_gsm8k_number(text: Any) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    marker = "####"
    if marker in raw:
        tail = raw.rsplit(marker, 1)[-1]
        matches = _NUMBER_RE.findall(tail)
        if matches:
            return _normalize_gsm8k_number(matches[-1])
    answer_match = re.search(r"the answer is\s*([-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)", raw, flags=re.IGNORECASE)
    if answer_match:
        return _normalize_gsm8k_number(answer_match.group(1))
    matches = _NUMBER_RE.findall(raw)
    return _normalize_gsm8k_number(matches[-1]) if matches else ""


def _gsm8k_gold_answer(item: Dict[str, Any]) -> str:
    for key in ("answer", "raw_answer", "output", "label"):
        if key in item and str(item.get(key, "")).strip():
            extracted = _extract_gsm8k_number(item.get(key))
            if extracted:
                return extracted
    return ""


@torch.no_grad()
def score_candidates_logprob(
    *,
    model: nn.Module,
    tokenizer,
    prompt: str,
    candidates: Sequence[str],
    device: torch.device,
    length_norm: str = "none",
) -> Tuple[str, Dict[str, float], Dict[str, int]]:
    enc = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")
    prompt_ids = enc["input_ids"][0].to(device)
    base_len = int(prompt_ids.shape[0])
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = 0
    cand_ids_list = [
        tokenizer.encode(" " + str(cand), add_special_tokens=False)
        or tokenizer.encode(str(cand), add_special_tokens=False)
        for cand in candidates
    ]
    full_ids: List[torch.Tensor] = []
    lengths: List[int] = []
    for ids in cand_ids_list:
        ids_t = torch.tensor(ids, dtype=torch.long, device=device)
        seq = torch.cat([prompt_ids, ids_t], dim=0)
        full_ids.append(seq)
        lengths.append(int(seq.numel()))
    max_len = max(lengths) if lengths else int(prompt_ids.numel())
    input_ids = torch.full(
        (len(candidates), max_len),
        int(pad_token_id),
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros((len(candidates), max_len), dtype=torch.long, device=device)
    for row_id, seq in enumerate(full_ids):
        seq_len = int(seq.numel())
        input_ids[row_id, :seq_len] = seq
        attention_mask[row_id, :seq_len] = 1
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
    logits = out.logits
    logp = torch.log_softmax(logits.float()[:, :-1, :], dim=-1)
    scores: Dict[str, float] = {}
    for idx, ids in enumerate(cand_ids_list):
        ids_t = torch.tensor(ids, dtype=torch.long, device=device)
        if ids_t.numel() <= 0:
            scores[str(candidates[idx])] = float("-inf")
            continue
        start = base_len - 1
        token_lp = logp[idx, start : start + ids_t.numel(), :].gather(1, ids_t[:, None]).squeeze(1)
        score_t = token_lp.sum()
        if str(length_norm).strip().lower() == "avg":
            score_t = score_t / max(1, int(ids_t.numel()))
        score = float(score_t.item())
        if not math.isfinite(score):
            score = -1e30
        scores[str(candidates[idx])] = score
    pred = max(scores.items(), key=lambda kv: kv[1])[0]
    stats = {
        "forward_tokens": int(attention_mask.sum().item()),
        "forward_calls": 1,
    }
    return pred, scores, stats


def _sync_if_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def predict_by_generate(
    *,
    model: nn.Module,
    tokenizer,
    prompt: str,
    candidates: Sequence[str],
    device: torch.device,
    max_new_tokens: int = 16,
) -> Tuple[str, Dict[str, str], Dict[str, int]]:
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_tokens = int(enc["input_ids"].numel())
    generated = model.generate(
        **enc,
        max_new_tokens=int(max_new_tokens),
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    output_text = tokenizer.decode(generated[0][enc["input_ids"].shape[1] :], skip_special_tokens=True)
    token_text = output_text.strip().lower().split()[0] if output_text.strip() else ""
    pred = str(candidates[0]) if candidates else ""
    for cand in candidates:
        cand_text = str(cand).strip().lower()
        if cand_text and token_text.startswith(cand_text):
            pred = str(cand)
            break
    generated_tokens = max(0, int(generated.shape[1] - enc["input_ids"].shape[1]))
    stats = {
        "forward_tokens": int(prompt_tokens + generated_tokens),
        "forward_calls": 1,
        "generated_tokens": int(generated_tokens),
    }
    return pred, {"raw": output_text}, stats


@torch.no_grad()
def predict_gsm8k_generate(
    *,
    model: nn.Module,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 256,
) -> Tuple[str, Dict[str, str], Dict[str, int]]:
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_tokens = int(enc["input_ids"].numel())
    generated = model.generate(
        **enc,
        max_new_tokens=int(max_new_tokens),
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    output_text = tokenizer.decode(generated[0][enc["input_ids"].shape[1] :], skip_special_tokens=True)
    generated_tokens = max(0, int(generated.shape[1] - enc["input_ids"].shape[1]))
    stats = {
        "forward_tokens": int(prompt_tokens + generated_tokens),
        "forward_calls": 1,
        "generated_tokens": int(generated_tokens),
    }
    return _extract_gsm8k_number(output_text), {"raw": output_text}, stats


def _select_eval_token_index(
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    eos_token_id: Optional[int],
    token_rule: str,
) -> int:
    normalized_rule = str(token_rule).strip().lower()
    if normalized_rule == "last_pred":
        idx_tensor = last_pred_indices(attention_mask, input_ids, eos_token_id, input_ids.device)
    else:
        idx_tensor = last_content_indices(attention_mask, input_ids, eos_token_id, input_ids.device)
    if idx_tensor.numel() <= 0:
        return max(0, int(input_ids.size(1)) - 1)
    return int(idx_tensor.view(-1)[0].item())


@torch.no_grad()
def _extract_last_token_trajectory(
    *,
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    token_index: int,
) -> Optional[torch.Tensor]:
    hidden_list = capture_hidden_states(model, input_ids=input_ids, attention_mask=attention_mask)
    if not hidden_list:
        return None
    points: List[torch.Tensor] = []
    for hidden in hidden_list:
        if hidden is None or hidden.dim() != 3:
            continue
        seq_len = int(hidden.size(1))
        if seq_len <= 0:
            continue
        safe_index = max(0, min(int(token_index), seq_len - 1))
        points.append(hidden[0, safe_index, :].detach().float().cpu())
    if not points:
        return None
    return torch.stack(points, dim=0)


@torch.no_grad()
def _collect_trajectory_pair_for_prompt(
    *,
    student_model: nn.Module,
    teacher_model: nn.Module,
    tokenizer,
    prompt: str,
    device: torch.device,
    token_rule: str,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], int]:
    encoded = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, device=device, dtype=torch.long)
    else:
        attention_mask = attention_mask.to(device)
    token_index = _select_eval_token_index(
        input_ids=input_ids,
        attention_mask=attention_mask,
        eos_token_id=tokenizer.eos_token_id,
        token_rule=token_rule,
    )
    student_traj = _extract_last_token_trajectory(
        model=student_model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_index=token_index,
    )
    teacher_traj = _extract_last_token_trajectory(
        model=teacher_model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_index=token_index,
    )
    return student_traj, teacher_traj, int(token_index)


def _project_to_3d(points: torch.Tensor) -> Tuple[torch.Tensor, List[float], int]:
    if points.dim() != 2:
        raise ValueError("points must be 2D for 3D projection.")
    if int(points.size(0)) <= 0:
        return torch.zeros((0, 3), dtype=torch.float32), [0.0, 0.0, 0.0], 0
    centered = points.float() - points.float().mean(dim=0, keepdim=True)
    try:
        _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    except Exception:
        coords = torch.zeros((int(points.size(0)), 3), dtype=torch.float32)
        return coords, [0.0, 0.0, 0.0], 0
    rank = int((singular_values > 1e-8).sum().item())
    comp = min(3, int(vh.size(0)))
    if comp <= 0:
        coords = torch.zeros((int(points.size(0)), 3), dtype=torch.float32)
    else:
        basis = vh[:comp, :].transpose(0, 1)
        coords = centered @ basis
        if comp < 3:
            coords = torch.cat([coords, torch.zeros((int(points.size(0)), 3 - comp), dtype=coords.dtype)], dim=1)
    denom = float((singular_values**2).sum().item())
    explained_ratio: List[float] = []
    for idx in range(3):
        if idx < int(singular_values.numel()) and denom > 0.0:
            explained_ratio.append(float((singular_values[idx] ** 2).item() / denom))
        else:
            explained_ratio.append(0.0)
    return coords.float(), explained_ratio, rank


SUBSPACE_TEACHER_COLOR = "#204bd8"
SUBSPACE_STUDENT_COLOR = "#b80f2a"
SUBSPACE_MEAN_EDGE_COLOR = "#ffffff"
SUBSPACE_ARROW_COLOR = "#4a4a4a"
SUBSPACE_ENERGY_CMAP = (
    LinearSegmentedColormap.from_list(
        "fad_energy_reference",
        [
            (0.00, "#061fb2"),
            (0.13, "#1e63e6"),
            (0.28, "#76b6ff"),
            (0.43, "#d9edff"),
            (0.53, "#fff9d9"),
            (0.66, "#ffd27a"),
            (0.79, "#ff8440"),
            (0.91, "#e5312e"),
            (1.00, "#9a0018"),
        ],
    )
    if LinearSegmentedColormap is not None
    else "RdYlBu_r"
)
SUBSPACE_REGIME_COLORS = {
    "llama_early": "#204bd8",
    "llama_mid": "#f0a33a",
    "llama_late": "#b80f2a",
}


def _validation_step_tag(step: int) -> str:
    return f"step_{int(step):07d}" if int(step) >= 0 else f"step_{now_tag()}"


def _safe_plot_slug(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(name).strip())
    return safe.strip("_") or "regime"


def _pretty_regime_name(name: str) -> str:
    text = str(name).strip()
    if text.startswith("llama_"):
        return text.replace("llama_", "").replace("_", " ").title()
    return text.replace("_", " ").title()


def _pca_elbow_dim(cumulative_ratio: Sequence[float]) -> int:
    values = [float(x) for x in cumulative_ratio if math.isfinite(float(x))]
    if not values:
        return 0
    if len(values) <= 2:
        return int(len(values))
    x0, y0 = 1.0, float(values[0])
    x1, y1 = float(len(values)), float(values[-1])
    dx = x1 - x0
    dy = y1 - y0
    denom = math.sqrt(dx * dx + dy * dy)
    if denom <= 1e-12:
        return 1
    best_idx = 0
    best_dist = -1.0
    for idx, value in enumerate(values):
        x = float(idx + 1)
        y = float(value)
        dist = abs(dy * x - dx * y + x1 * y0 - y1 * x0) / denom
        if dist > best_dist:
            best_dist = dist
            best_idx = idx
    return int(best_idx + 1)


def _project_to_dim12_with_elbow(
    points: torch.Tensor,
    *,
    max_elbow_components: int = 12,
) -> Tuple[torch.Tensor, torch.Tensor, List[float], List[float], int, int]:
    if points.dim() != 2:
        raise ValueError("points must be 2D for 2D projection.")
    if int(points.size(0)) <= 0:
        return torch.zeros((0, 2), dtype=torch.float32), torch.zeros((int(points.size(1)), 2), dtype=torch.float32), [], [], 0, 0
    centered = points.float() - points.float().mean(dim=0, keepdim=True)
    try:
        _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    except Exception:
        return torch.zeros((int(points.size(0)), 2), dtype=torch.float32), torch.zeros((int(points.size(1)), 2), dtype=torch.float32), [], [], 0, 0

    rank = int((singular_values > 1e-8).sum().item())
    comp = min(2, int(vh.size(0)))
    if comp <= 0:
        coords = torch.zeros((int(points.size(0)), 2), dtype=torch.float32)
        basis_2d = torch.zeros((int(points.size(1)), 2), dtype=torch.float32)
    else:
        basis = vh[:comp, :].transpose(0, 1)
        basis_2d = basis
        coords = centered @ basis
        if comp < 2:
            coords = torch.cat([coords, torch.zeros((int(points.size(0)), 2 - comp), dtype=coords.dtype)], dim=1)
            basis_2d = torch.cat([basis_2d, torch.zeros((int(points.size(1)), 2 - comp), dtype=basis_2d.dtype)], dim=1)

    denom = float((singular_values**2).sum().item())
    max_components = min(max(1, int(max_elbow_components)), int(singular_values.numel()))
    explained_ratio: List[float] = []
    cumulative_ratio: List[float] = []
    running = 0.0
    for idx in range(max_components):
        ratio = float((singular_values[idx] ** 2).item() / denom) if denom > 0.0 else 0.0
        running += ratio
        explained_ratio.append(ratio)
        cumulative_ratio.append(float(min(1.0, running)))
    elbow_dim = _pca_elbow_dim(cumulative_ratio)
    return coords.float(), basis_2d.float(), explained_ratio, cumulative_ratio, rank, int(elbow_dim)


def _build_bar_colors(regime_names: Sequence[str]) -> List[str]:
    return [SUBSPACE_REGIME_COLORS.get(str(name), "#6f7f92") for name in regime_names]


def _draw_fad_energy_background(
    ax: Any,
    *,
    teacher_coords: torch.Tensor,
    student_coords: torch.Tensor,
    teacher_mean: torch.Tensor,
    pca_basis_dim12: Optional[torch.Tensor],
    energy_metric_diag: Optional[torch.Tensor],
) -> Optional[Any]:
    if pca_basis_dim12 is None or energy_metric_diag is None:
        return None
    if not torch.is_tensor(pca_basis_dim12) or not torch.is_tensor(energy_metric_diag):
        return None
    if pca_basis_dim12.dim() != 2 or int(pca_basis_dim12.size(1)) < 2:
        return None
    if int(energy_metric_diag.numel()) != int(pca_basis_dim12.size(0)):
        return None

    coords = torch.cat([teacher_coords[:, :2], student_coords[:, :2], teacher_mean.view(1, 2)], dim=0).float()
    x_min = float(coords[:, 0].min().item())
    x_max = float(coords[:, 0].max().item())
    y_min = float(coords[:, 1].min().item())
    y_max = float(coords[:, 1].max().item())
    x_span = max(1e-6, x_max - x_min)
    y_span = max(1e-6, y_max - y_min)
    x_pad = 0.18 * x_span
    y_pad = 0.18 * y_span
    x_values = torch.linspace(x_min - x_pad, x_max + x_pad, steps=120)
    y_values = torch.linspace(y_min - y_pad, y_max + y_pad, steps=120)
    yy, xx = torch.meshgrid(y_values, x_values, indexing="ij")
    grid_delta = torch.stack(
        [
            xx - float(teacher_mean[0].item()),
            yy - float(teacher_mean[1].item()),
        ],
        dim=-1,
    )

    basis_2d = pca_basis_dim12[:, :2].float().cpu()
    metric_inv = 1.0 / energy_metric_diag.float().cpu().clamp(min=1e-12)
    delta_z = torch.matmul(grid_delta.reshape(-1, 2), basis_2d.transpose(0, 1))
    energy = 0.5 * (delta_z.pow(2) * metric_inv.view(1, -1)).sum(dim=1)
    energy_grid = energy.reshape(xx.shape)
    finite_energy = energy_grid[torch.isfinite(energy_grid)]
    if int(finite_energy.numel()) <= 0:
        return None
    high = float(torch.quantile(finite_energy, 0.96).item())
    if not math.isfinite(high) or high <= 0.0:
        return None
    quantile_points = torch.linspace(0.0, 0.96, steps=16)
    levels = torch.quantile(finite_energy, quantile_points).detach().cpu().tolist()
    levels[0] = 0.0
    unique_levels: List[float] = []
    for value in levels:
        value_f = float(value)
        if not unique_levels or value_f > unique_levels[-1] + 1e-10:
            unique_levels.append(value_f)
    if len(unique_levels) < 3:
        unique_levels = torch.linspace(0.0, high, steps=8).tolist()
    filled = ax.contourf(
        xx.numpy(),
        yy.numpy(),
        energy_grid.numpy(),
        levels=unique_levels,
        cmap=SUBSPACE_ENERGY_CMAP,
        alpha=0.82,
        antialiased=True,
        zorder=0,
        extend="max",
    )
    ax.contour(
        xx.numpy(),
        yy.numpy(),
        energy_grid.numpy(),
        levels=unique_levels[1:],
        colors="#9a3c35",
        alpha=0.23,
        linewidths=0.65,
        zorder=1,
    )
    low_levels = unique_levels[1 : min(len(unique_levels), 4)]
    if low_levels:
        ax.contour(
            xx.numpy(),
            yy.numpy(),
            energy_grid.numpy(),
            levels=low_levels,
            colors=SUBSPACE_TEACHER_COLOR,
            alpha=0.78,
            linewidths=1.05,
            linestyles="--",
            zorder=2,
        )
    return filled


def _render_validation_subspace_plots(
    *,
    plot_data: Dict[str, Any],
    output_dir: str,
    file_prefix: str,
) -> Dict[str, Any]:
    ensure_dir(output_dir)
    report: Dict[str, Any] = {
        "saved": False,
        "reason": "no_plot_generated",
        "output_png": "",
        "output_pngs": {
            "regimes": {},
            "mean_pair_l2": "",
            "pca_elbow": "",
        },
    }
    if plt is None:
        report["reason"] = "matplotlib_unavailable"
        return report

    regime_order = [str(x) for x in plot_data.get("regime_order", [])]
    regimes = dict(plot_data.get("regimes", {}))
    plotted_regimes = [name for name in regime_order if isinstance(regimes.get(name), dict)]
    if not plotted_regimes:
        report["reason"] = "no_projected_points"
        return report

    prefix = _safe_plot_slug(file_prefix)
    regime_pngs: Dict[str, str] = {}
    for regime_name in plotted_regimes:
        regime_payload = dict(regimes[regime_name])
        teacher_coords = regime_payload.get("teacher_dim12")
        student_coords = regime_payload.get("student_dim12")
        if not torch.is_tensor(teacher_coords) or not torch.is_tensor(student_coords):
            continue
        if int(teacher_coords.numel()) <= 0 or int(student_coords.numel()) <= 0:
            continue

        teacher_coords = teacher_coords.float().cpu()
        student_coords = student_coords.float().cpu()
        teacher_mean = regime_payload.get("teacher_mean_dim12")
        student_mean = regime_payload.get("student_mean_dim12")
        if not torch.is_tensor(teacher_mean):
            teacher_mean = teacher_coords.mean(dim=0)
        if not torch.is_tensor(student_mean):
            student_mean = student_coords.mean(dim=0)
        teacher_mean = teacher_mean.float().cpu().view(-1)
        student_mean = student_mean.float().cpu().view(-1)

        fig, ax = plt.subplots(figsize=(6.4, 4.9))
        energy_artist = _draw_fad_energy_background(
            ax,
            teacher_coords=teacher_coords,
            student_coords=student_coords,
            teacher_mean=teacher_mean[:2],
            pca_basis_dim12=regime_payload.get("pca_basis_dim12", None),
            energy_metric_diag=regime_payload.get("energy_metric_diag", None),
        )
        if energy_artist is not None:
            cbar = fig.colorbar(energy_artist, ax=ax, fraction=0.048, pad=0.025)
            cbar.set_label("Deviation energy", fontsize=9)
            ticks = cbar.get_ticks()
            if len(ticks) >= 2:
                cbar.set_ticks([ticks[0], ticks[-1]])
                cbar.set_ticklabels(["Low", "High"])
        ax.scatter(
            teacher_coords[:, 0].tolist(),
            teacher_coords[:, 1].tolist(),
            color=SUBSPACE_TEACHER_COLOR,
            alpha=0.42,
            s=18,
            linewidths=0.0,
            label="teacher z(t)",
            zorder=3,
        )
        ax.scatter(
            student_coords[:, 0].tolist(),
            student_coords[:, 1].tolist(),
            color=SUBSPACE_STUDENT_COLOR,
            alpha=0.42,
            s=18,
            linewidths=0.0,
            label="student z(s)",
            zorder=3,
        )
        if int(teacher_mean.numel()) >= 2 and int(student_mean.numel()) >= 2:
            ax.annotate(
                "",
                xy=(float(teacher_mean[0].item()), float(teacher_mean[1].item())),
                xytext=(float(student_mean[0].item()), float(student_mean[1].item())),
                arrowprops={
                    "arrowstyle": "->",
                    "color": SUBSPACE_ARROW_COLOR,
                    "lw": 1.7,
                    "alpha": 0.86,
                    "shrinkA": 5,
                    "shrinkB": 5,
                },
            )
            ax.scatter(
                [float(teacher_mean[0].item())],
                [float(teacher_mean[1].item())],
                color=SUBSPACE_TEACHER_COLOR,
                marker="o",
                s=170,
                edgecolors=SUBSPACE_MEAN_EDGE_COLOR,
                linewidths=2.3,
                zorder=6,
                label="teacher center",
            )
            ax.scatter(
                [float(student_mean[0].item())],
                [float(student_mean[1].item())],
                color=SUBSPACE_STUDENT_COLOR,
                marker="D",
                s=108,
                edgecolors=SUBSPACE_MEAN_EDGE_COLOR,
                linewidths=1.7,
                zorder=7,
                label="student mean",
            )
            ax.scatter(
                [float(teacher_mean[0].item())],
                [float(teacher_mean[1].item())],
                facecolors="none",
                edgecolors=SUBSPACE_TEACHER_COLOR,
                s=250,
                linewidths=1.8,
                zorder=8,
            )
            ax.annotate(
                "low-energy teacher basin",
                xy=(float(teacher_mean[0].item()), float(teacher_mean[1].item())),
                xytext=(0.04, 0.12),
                textcoords="axes fraction",
                ha="left",
                va="center",
                fontsize=8.5,
                color=SUBSPACE_TEACHER_COLOR,
                arrowprops={
                    "arrowstyle": "->",
                    "color": SUBSPACE_TEACHER_COLOR,
                    "lw": 1.1,
                    "alpha": 0.85,
                },
                bbox={"boxstyle": "round,pad=0.22", "facecolor": "#ffffff", "edgecolor": "#b8c4ff", "alpha": 0.88},
                zorder=9,
            )

        regime_gap = regime_payload.get("mean_pair_l2", None)
        regime_cos = regime_payload.get("mean_pair_cosine", None)
        metric_parts: List[str] = []
        if regime_gap is not None:
            metric_parts.append(f"mean L2={float(regime_gap):.4f}")
        if regime_cos is not None:
            metric_parts.append(f"cos={float(regime_cos):.3f}")
        if metric_parts:
            ax.text(
                0.02,
                0.98,
                "  ".join(metric_parts),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                color="#202020",
                bbox={"boxstyle": "round,pad=0.24", "facecolor": "#ffffff", "edgecolor": "#d8d8d8", "alpha": 0.88},
            )
        ax.set_title(str(regime_name), fontsize=13, pad=8)
        ax.set_xlabel("Dim1")
        ax.set_ylabel("Dim2")
        ax.grid(alpha=0.18, linewidth=0.55)
        ax.legend(loc="upper right", fontsize=7.5, frameon=True)
        fig.tight_layout()
        regime_png = os.path.join(output_dir, f"{prefix}_{_safe_plot_slug(regime_name)}_subspace.png")
        fig.savefig(regime_png, dpi=220)
        plt.close(fig)
        regime_pngs[str(regime_name)] = regime_png

    bar_names = list(plotted_regimes)
    bar_values = [float(dict(plot_data.get("mean_pair_l2_by_regime", {})).get(name, 0.0)) for name in bar_names]
    bar_png = os.path.join(output_dir, f"{prefix}_mean_pair_l2.png")
    fig, ax = plt.subplots(figsize=(5.2, 3.7))
    x_positions = list(range(len(bar_names)))
    bars = ax.bar(
        x_positions,
        bar_values,
        width=0.48,
        color=_build_bar_colors(bar_names),
        alpha=0.92,
    )
    ax.set_title("Mean Pair L2", fontsize=12, pad=8)
    ax.set_ylabel("L2 gap")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([_pretty_regime_name(name) for name in bar_names], rotation=0)
    ax.grid(axis="y", alpha=0.18, linewidth=0.55)
    top = max(bar_values) if bar_values else 0.0
    ax.set_ylim(0.0, top * 1.18 if top > 0.0 else 1.0)
    for bar, value in zip(bars, bar_values):
        ax.text(
            float(bar.get_x() + bar.get_width() / 2.0),
            float(bar.get_height()),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#222222",
        )
    fig.tight_layout()
    fig.savefig(bar_png, dpi=220)
    plt.close(fig)

    elbow_png = os.path.join(output_dir, f"{prefix}_pca_elbow.png")
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    elbow_series_count = 0
    for regime_name in plotted_regimes:
        regime_payload = dict(regimes[regime_name])
        cumulative = [float(x) for x in regime_payload.get("pca_cumulative_explained_ratio", [])]
        if not cumulative:
            continue
        xs = list(range(1, len(cumulative) + 1))
        color = SUBSPACE_REGIME_COLORS.get(regime_name, None)
        ax.plot(
            xs,
            cumulative,
            marker="o",
            linewidth=1.8,
            markersize=4.0,
            color=color,
            label=f"{_pretty_regime_name(regime_name)}",
        )
        elbow_dim = int(regime_payload.get("pca_elbow_dim", 0) or 0)
        if 1 <= elbow_dim <= len(cumulative):
            ax.scatter(
                [elbow_dim],
                [float(cumulative[elbow_dim - 1])],
                color=color,
                edgecolors="#111111",
                linewidths=0.7,
                s=46,
                zorder=4,
            )
        elbow_series_count += 1
    ax.set_title("PCA Elbow", fontsize=12, pad=8)
    ax.set_xlabel("Components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.18, linewidth=0.55)
    if elbow_series_count > 0:
        ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(elbow_png, dpi=220)
    plt.close(fig)

    output_pngs = {
        "regimes": regime_pngs,
        "mean_pair_l2": bar_png,
        "pca_elbow": elbow_png,
    }
    output_png = bar_png
    report.update(
        {
            "saved": bool(regime_pngs or bar_names or elbow_series_count),
            "reason": "",
            "output_png": output_png,
            "output_pngs": output_pngs,
            "plot_count": int(len(regime_pngs) + 1 + (1 if elbow_series_count > 0 else 0)),
        }
    )
    return report


def _render_validation_subspace_plots_from_pt(
    *,
    plot_data_pt: str,
    output_dir: str,
    file_prefix: str,
) -> Dict[str, Any]:
    path = str(plot_data_pt).strip()
    if not path or not os.path.isfile(path):
        return {
            "saved": False,
            "reason": "plot_data_pt_missing",
            "output_png": "",
            "output_pngs": {"regimes": {}, "mean_pair_l2": "", "pca_elbow": ""},
        }
    try:
        plot_data = torch.load(path, map_location="cpu")
    except Exception as exc:
        return {
            "saved": False,
            "reason": f"plot_data_load_failed: {exc}",
            "output_png": "",
            "output_pngs": {"regimes": {}, "mean_pair_l2": "", "pca_elbow": ""},
        }
    if not isinstance(plot_data, dict):
        return {
            "saved": False,
            "reason": "plot_data_not_dict",
            "output_png": "",
            "output_pngs": {"regimes": {}, "mean_pair_l2": "", "pca_elbow": ""},
        }
    return _render_validation_subspace_plots(
        plot_data=plot_data,
        output_dir=output_dir,
        file_prefix=file_prefix,
    )


def _plot_trajectory_3d(
    *,
    teacher_coords: torch.Tensor,
    student_coords: torch.Tensor,
    output_png: str,
    title: str,
    max_lines: int,
) -> Dict[str, Any]:
    if plt is None:
        return {"saved": False, "reason": "matplotlib_unavailable", "output_png": output_png}
    if teacher_coords.dim() != 3 or student_coords.dim() != 3:
        return {"saved": False, "reason": "invalid_shape", "output_png": output_png}
    if int(teacher_coords.size(0)) <= 0:
        return {"saved": False, "reason": "no_samples", "output_png": output_png}

    ensure_dir(os.path.dirname(os.path.abspath(output_png)) or ".")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    layer_count = min(int(teacher_coords.size(1)), int(student_coords.size(1)))
    if layer_count <= 0:
        return {"saved": False, "reason": "no_layers", "output_png": output_png}
    depth_values = torch.linspace(0.30, 0.95, steps=layer_count).tolist()
    teacher_cmap = plt.cm.Blues
    student_cmap = plt.cm.Reds
    line_count = min(int(max_lines), int(teacher_coords.size(0)))
    line_iter = iter_progress(
        range(line_count),
        total=line_count,
        desc="[Eval][Trajectory] plot3d",
    )
    for idx in line_iter:
        teacher_line = teacher_coords[idx]
        student_line = student_coords[idx]
        if layer_count == 1:
            ax.scatter(
                [float(teacher_line[0, 0])],
                [float(teacher_line[0, 1])],
                [float(teacher_line[0, 2])],
                color=teacher_cmap(depth_values[0]),
                alpha=0.25,
                s=10,
            )
            ax.scatter(
                [float(student_line[0, 0])],
                [float(student_line[0, 1])],
                [float(student_line[0, 2])],
                color=student_cmap(depth_values[0]),
                alpha=0.25,
                s=10,
            )
            continue
        for layer_id in range(layer_count - 1):
            teacher_color = teacher_cmap(depth_values[layer_id])
            student_color = student_cmap(depth_values[layer_id])
            ax.plot(
                teacher_line[layer_id : layer_id + 2, 0].tolist(),
                teacher_line[layer_id : layer_id + 2, 1].tolist(),
                teacher_line[layer_id : layer_id + 2, 2].tolist(),
                color=teacher_color,
                alpha=0.20,
                linewidth=0.9,
            )
            ax.plot(
                student_line[layer_id : layer_id + 2, 0].tolist(),
                student_line[layer_id : layer_id + 2, 1].tolist(),
                student_line[layer_id : layer_id + 2, 2].tolist(),
                color=student_color,
                alpha=0.20,
                linewidth=0.9,
            )

    teacher_mean = teacher_coords.mean(dim=0)
    student_mean = student_coords.mean(dim=0)
    if layer_count == 1:
        ax.scatter([float(teacher_mean[0, 0])], [float(teacher_mean[0, 1])], [float(teacher_mean[0, 2])], color=teacher_cmap(depth_values[0]), s=50, marker="o")
        ax.scatter([float(student_mean[0, 0])], [float(student_mean[0, 1])], [float(student_mean[0, 2])], color=student_cmap(depth_values[0]), s=50, marker="o")
    else:
        for layer_id in range(layer_count - 1):
            teacher_color = teacher_cmap(depth_values[layer_id])
            student_color = student_cmap(depth_values[layer_id])
            ax.plot(
                teacher_mean[layer_id : layer_id + 2, 0].tolist(),
                teacher_mean[layer_id : layer_id + 2, 1].tolist(),
                teacher_mean[layer_id : layer_id + 2, 2].tolist(),
                color=teacher_color,
                alpha=1.0,
                linewidth=3.0,
            )
            ax.plot(
                student_mean[layer_id : layer_id + 2, 0].tolist(),
                student_mean[layer_id : layer_id + 2, 1].tolist(),
                student_mean[layer_id : layer_id + 2, 2].tolist(),
                color=student_color,
                alpha=1.0,
                linewidth=3.0,
            )
        ax.scatter([float(teacher_mean[0, 0])], [float(teacher_mean[0, 1])], [float(teacher_mean[0, 2])], color=teacher_cmap(depth_values[0]), s=50, marker="o")
        ax.scatter([float(student_mean[0, 0])], [float(student_mean[0, 1])], [float(student_mean[0, 2])], color=student_cmap(depth_values[0]), s=50, marker="o")
        ax.scatter([float(teacher_mean[-1, 0])], [float(teacher_mean[-1, 1])], [float(teacher_mean[-1, 2])], color=teacher_cmap(depth_values[-1]), s=65, marker="x")
        ax.scatter([float(student_mean[-1, 0])], [float(student_mean[-1, 1])], [float(student_mean[-1, 2])], color=student_cmap(depth_values[-1]), s=65, marker="x")

    ax.plot([], [], [], color=teacher_cmap(0.85), linewidth=3.0, label="merged_teacher mean (light->dark by layer)")
    ax.plot([], [], [], color=student_cmap(0.85), linewidth=3.0, label="phase1.5 mean (light->dark by layer)")
    ax.set_xlabel("Dim1")
    ax.set_ylabel("Dim2")
    ax.set_zlabel("Dim3")
    ax.set_title(title)
    ax.text2D(0.01, 0.01, "Layer depth: light -> dark", transform=ax.transAxes, fontsize=9)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_png, dpi=220)
    plt.close(fig)
    return {
        "saved": True,
        "output_png": output_png,
        "overlay_lines": int(line_count),
        "layer_count": int(layer_count),
        "color_depth_encoding": "light_to_dark_by_layer",
    }


def _build_trajectory_outputs(
    *,
    teacher_trajs: List[torch.Tensor],
    student_trajs: List[torch.Tensor],
    sample_indices: List[int],
    output_png: str,
    output_json: str,
    title: str,
    max_plot_lines: int,
) -> Dict[str, Any]:
    if not teacher_trajs or not student_trajs:
        report = {
            "enabled": True,
            "paired_samples": 0,
            "reason": "no_trajectory_pairs_collected",
        }
        save_json(output_json, report)
        return report

    post_bar = step_progress(total=4, desc="[Eval][Trajectory] postprocess")
    t0 = time.perf_counter()
    pair_count = min(len(teacher_trajs), len(student_trajs))
    teacher_stack = torch.stack(teacher_trajs[:pair_count], dim=0).float()
    student_stack = torch.stack(student_trajs[:pair_count], dim=0).float()
    layer_count = min(int(teacher_stack.size(1)), int(student_stack.size(1)))
    teacher_stack = teacher_stack[:, :layer_count, :]
    student_stack = student_stack[:, :layer_count, :]
    print(
        f"[Eval][Trajectory] postprocess start pairs={pair_count} layers={layer_count}",
        flush=True,
    )
    if post_bar is not None:
        post_bar.update(1)

    teacher_flat = teacher_stack.reshape(pair_count * layer_count, -1)
    student_flat = student_stack.reshape(pair_count * layer_count, -1)
    merged = torch.cat([teacher_flat, student_flat], dim=0)
    pca_t0 = time.perf_counter()
    coords, explained_ratio, pca_rank = _project_to_3d(merged)
    print(
        f"[Eval][Trajectory] pca3d done rank={pca_rank} time={time.perf_counter() - pca_t0:.2f}s",
        flush=True,
    )
    if post_bar is not None:
        post_bar.update(1)

    split = pair_count * layer_count
    teacher_coords = coords[:split, :].reshape(pair_count, layer_count, 3)
    student_coords = coords[split:, :].reshape(pair_count, layer_count, 3)
    teacher_mean = teacher_coords.mean(dim=0)
    student_mean = student_coords.mean(dim=0)
    layer_gap = torch.linalg.norm(student_mean - teacher_mean, dim=1)
    if post_bar is not None:
        post_bar.update(1)

    plot_t0 = time.perf_counter()
    print(f"[Eval][Trajectory] saving 3d plot -> {output_png}", flush=True)
    plot_report = _plot_trajectory_3d(
        teacher_coords=teacher_coords,
        student_coords=student_coords,
        output_png=output_png,
        title=title,
        max_lines=max_plot_lines,
    )
    print(
        f"[Eval][Trajectory] plot saved={bool(plot_report.get('saved', False))} time={time.perf_counter() - plot_t0:.2f}s",
        flush=True,
    )
    if post_bar is not None:
        post_bar.update(1)
        post_bar.close()

    report = {
        "enabled": True,
        "paired_samples": int(pair_count),
        "layer_count": int(layer_count),
        "sample_indices": [int(x) for x in sample_indices[:pair_count]],
        "pca_rank": int(pca_rank),
        "pca_explained_ratio_top3": [float(x) for x in explained_ratio],
        "mean_layer_gap_l2": float(layer_gap.mean().item()) if int(layer_gap.numel()) > 0 else 0.0,
        "mean_layer_gap_l2_by_layer": [float(x) for x in layer_gap.tolist()],
        "teacher_mean_traj_3d": [[float(v) for v in row] for row in teacher_mean.tolist()],
        "student_mean_traj_3d": [[float(v) for v in row] for row in student_mean.tolist()],
        "plot": plot_report,
        "output_png": output_png,
        "output_json": output_json,
        "postprocess_time_sec": float(time.perf_counter() - t0),
    }
    save_json(output_json, report)
    return report


def _build_validation_subspace_snapshot(
    *,
    teacher_points_by_regime: Dict[str, List[torch.Tensor]],
    student_points_by_regime: Dict[str, List[torch.Tensor]],
    regime_gap_mean: Dict[str, float],
    regime_cos_mean: Dict[str, float],
    layer_gap_mean: Dict[str, float],
    energy_metric_diag_by_regime: Optional[Dict[str, torch.Tensor]] = None,
    regime_order: Sequence[str],
    step: int,
    output_dir: str,
    save_plot: bool = True,
    plot_prefix: str = "",
) -> Dict[str, Any]:
    ensure_dir(output_dir)
    step_tag = _validation_step_tag(int(step))
    prefix = str(plot_prefix).strip() or step_tag
    planned_output_png = os.path.join(output_dir, f"{_safe_plot_slug(prefix)}_mean_pair_l2.png")
    output_json = os.path.join(output_dir, f"{step_tag}_subspace_snapshot.json")
    output_data_pt = os.path.join(output_dir, f"{step_tag}_subspace_plot_data.pt")
    report: Dict[str, Any] = {
        "enabled": True,
        "step": int(step),
        "output_png": "",
        "output_json": output_json,
        "output_data_pt": output_data_pt,
        "output_pngs": {
            "regimes": {},
            "mean_pair_l2": "",
            "pca_elbow": "",
        },
        "plot": {
            "saved": False,
            "reason": "no_plot_generated",
            "output_png": "",
        },
    }

    plotted_regimes = [
        str(regime)
        for regime in regime_order
        if teacher_points_by_regime.get(str(regime)) and student_points_by_regime.get(str(regime))
    ]
    if not plotted_regimes:
        report["plot"] = {
            "saved": False,
            "reason": "no_projected_points",
            "output_png": "",
        }
        return report

    regime_plot_meta: Dict[str, Dict[str, Any]] = {}
    plot_data: Dict[str, Any] = {
        "version": 2,
        "step": int(step),
        "axis_labels": ["Dim1", "Dim2"],
        "colors": {
            "teacher": SUBSPACE_TEACHER_COLOR,
            "student": SUBSPACE_STUDENT_COLOR,
        },
        "regime_order": list(plotted_regimes),
        "regimes": {},
        "mean_pair_l2_by_regime": {
            str(name): float(regime_gap_mean.get(str(name), 0.0))
            for name in plotted_regimes
        },
        "mean_pair_cosine_by_regime": {
            str(name): float(regime_cos_mean[str(name)])
            for name in plotted_regimes
            if regime_cos_mean.get(str(name), None) is not None
        },
        "mean_pair_l2_by_layer": {
            str(layer_id): float(value)
            for layer_id, value in dict(layer_gap_mean).items()
        },
    }
    for regime_name in plotted_regimes:
        teacher_points = torch.cat(teacher_points_by_regime[str(regime_name)], dim=0).float()
        student_points = torch.cat(student_points_by_regime[str(regime_name)], dim=0).float()
        merged = torch.cat([teacher_points, student_points], dim=0)
        coords_2d, pca_basis_2d, explained_ratio, cumulative_ratio, pca_rank, elbow_dim = _project_to_dim12_with_elbow(merged)
        teacher_count = int(teacher_points.size(0))
        teacher_coords = coords_2d[:teacher_count, :2].contiguous()
        student_coords = coords_2d[teacher_count:, :2].contiguous()
        teacher_mean = teacher_coords.mean(dim=0)
        student_mean = student_coords.mean(dim=0)
        regime_gap = float(regime_gap_mean.get(str(regime_name), 0.0))
        regime_cos = regime_cos_mean.get(str(regime_name), None)
        metric_diag = None
        if isinstance(energy_metric_diag_by_regime, dict):
            candidate = energy_metric_diag_by_regime.get(str(regime_name), None)
            if torch.is_tensor(candidate) and int(candidate.numel()) == int(teacher_points.size(1)):
                metric_diag = candidate.detach().float().cpu().view(-1).contiguous()
        regime_plot_meta[str(regime_name)] = {
            "teacher_points": int(teacher_points.size(0)),
            "student_points": int(student_points.size(0)),
            "pca_rank": int(pca_rank),
            "pca_explained_ratio": [float(x) for x in explained_ratio],
            "pca_cumulative_explained_ratio": [float(x) for x in cumulative_ratio],
            "pca_elbow_dim": int(elbow_dim),
            "pca_explained_ratio_top2": [
                float(explained_ratio[0]) if len(explained_ratio) > 0 else 0.0,
                float(explained_ratio[1]) if len(explained_ratio) > 1 else 0.0,
            ],
            "teacher_mean_xy": [float(teacher_mean[0].item()), float(teacher_mean[1].item())],
            "student_mean_xy": [float(student_mean[0].item()), float(student_mean[1].item())],
            "mean_pair_l2": float(regime_gap),
            "mean_pair_cosine": float(regime_cos) if regime_cos is not None else None,
            "energy_metric": "fad_deviation_energy_avg_layer_metric" if metric_diag is not None else "",
        }
        plot_data["regimes"][str(regime_name)] = {
            "teacher_z": teacher_points.detach().cpu().contiguous(),
            "student_z": student_points.detach().cpu().contiguous(),
            "teacher_dim12": teacher_coords.detach().cpu().contiguous(),
            "student_dim12": student_coords.detach().cpu().contiguous(),
            "teacher_mean_dim12": teacher_mean.detach().cpu().contiguous(),
            "student_mean_dim12": student_mean.detach().cpu().contiguous(),
            "pca_basis_dim12": pca_basis_2d.detach().cpu().contiguous(),
            "energy_metric_diag": metric_diag,
            "energy_definition": "E_dev = 0.5 * (z_S - z_T)^T D^{-1} (z_S - z_T)",
            "mean_pair_l2": float(regime_gap),
            "mean_pair_cosine": float(regime_cos) if regime_cos is not None else None,
            "pca_rank": int(pca_rank),
            "pca_explained_ratio": [float(x) for x in explained_ratio],
            "pca_cumulative_explained_ratio": [float(x) for x in cumulative_ratio],
            "pca_elbow_dim": int(elbow_dim),
        }

    torch.save(plot_data, output_data_pt)
    if bool(save_plot):
        plot_report = _render_validation_subspace_plots(
            plot_data=plot_data,
            output_dir=output_dir,
            file_prefix=prefix,
        )
        report["plot"] = plot_report
        if bool(plot_report.get("saved", False)) and plot_report.get("output_png"):
            report["output_png"] = str(plot_report.get("output_png", planned_output_png))
        report["output_pngs"] = dict(plot_report.get("output_pngs", report["output_pngs"]))
    report["plot"] = {
        **dict(report.get("plot", {})),
        "regimes": regime_plot_meta,
        "layer_gap_count": int(len(layer_gap_mean)),
        "data_saved": True,
        "output_data_pt": output_data_pt,
    }
    return report


def _save_validation_subspace_trend(
    *,
    val_history: Sequence[Dict[str, Any]],
    output_png: str,
    output_json: str,
    title: str,
) -> Dict[str, Any]:
    tracked = [entry for entry in val_history if isinstance(entry, dict) and entry.get("subspace_gap_l2") is not None]
    report: Dict[str, Any] = {
        "enabled": True,
        "output_png": output_png,
        "output_json": output_json,
        "points": int(len(tracked)),
        "plot": {
            "saved": False,
            "reason": "no_tracked_points",
            "output_png": output_png,
        },
    }
    if not tracked:
        save_json(output_json, report)
        return report

    regime_names: List[str] = []
    seen_regimes: set[str] = set()
    for entry in tracked:
        for regime_name in dict(entry.get("subspace_gap_l2_by_regime", {})).keys():
            norm_name = str(regime_name)
            if norm_name not in seen_regimes:
                seen_regimes.add(norm_name)
                regime_names.append(norm_name)

    report["steps"] = [int(entry.get("step", 0)) for entry in tracked]
    report["overall_gap_l2"] = [float(entry.get("subspace_gap_l2", 0.0)) for entry in tracked]
    report["gap_l2_by_regime"] = {
        regime_name: [
            float(dict(entry.get("subspace_gap_l2_by_regime", {})).get(regime_name))
            if dict(entry.get("subspace_gap_l2_by_regime", {})).get(regime_name) is not None
            else None
            for entry in tracked
        ]
        for regime_name in regime_names
    }

    if plt is None:
        report["plot"] = {
            "saved": False,
            "reason": "matplotlib_unavailable",
            "output_png": output_png,
        }
        save_json(output_json, report)
        return report

    ensure_dir(os.path.dirname(os.path.abspath(output_png)) or ".")
    fig, ax = plt.subplots(figsize=(9, 5.2))
    steps = report["steps"]
    ax.plot(
        steps,
        report["overall_gap_l2"],
        color="#111111",
        linewidth=2.4,
        marker="o",
        label="overall",
    )
    regime_colors = {
        "llama_early": "#1f77b4",
        "llama_mid": "#2ca02c",
        "llama_late": "#d62728",
    }
    for regime_name in regime_names:
        values = report["gap_l2_by_regime"][regime_name]
        usable_steps: List[int] = []
        usable_values: List[float] = []
        for step_value, maybe_gap in zip(steps, values):
            if maybe_gap is None:
                continue
            usable_steps.append(int(step_value))
            usable_values.append(float(maybe_gap))
        if not usable_steps:
            continue
        ax.plot(
            usable_steps,
            usable_values,
            linewidth=1.7,
            marker="o",
            alpha=0.90,
            color=regime_colors.get(regime_name, None),
            label=regime_name,
        )
    ax.set_xlabel("Validation step")
    ax.set_ylabel("Mean pair L2 gap in z-space")
    ax.set_title(title)
    ax.grid(alpha=0.20, linewidth=0.5)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_png, dpi=220)
    plt.close(fig)
    report["plot"] = {
        "saved": True,
        "output_png": output_png,
        "series_count": int(1 + len(regime_names)),
    }
    save_json(output_json, report)
    return report


def stage_eval(args: argparse.Namespace) -> Dict[str, Any]:
    set_seed(int(args.seed))
    dist_ctx = init_distributed(str(args.device))
    device = resolve_device(args.device, dist_ctx.local_rank if dist_ctx.enabled else -1)
    dtype = get_target_dtype(device)
    dataset = str(args.dataset)
    candidates = _candidates_for_dataset(dataset)
    data_file = os.path.join(str(args.test_data_root), dataset, "test.json")
    if not os.path.isfile(data_file):
        raise FileNotFoundError(data_file)
    with open(data_file, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError("test.json must be a list.")
    records = payload[: int(args.max_samples)] if int(args.max_samples) > 0 else payload
    total_records_before_shard = len(records)
    if dist_ctx.enabled:
        records = records[int(dist_ctx.rank) :: int(dist_ctx.world_size)]
    variant = str(args.model_variant).strip().lower()
    bundle: Dict[str, Any] = {}
    if variant == "baseline":
        model_path = str(args.base_model)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=bool(getattr(args, "trust_remote_code", False)),
        ).to(device)
        phase_modules_applied: Dict[str, bool] = {"phase1_5": False}
        quant_eval: Dict[str, Any] = {"requested": bool(args.use_quant_bank_int4), "enabled": False}
    elif variant == "phase1_5":
        deploy_bundle = str(args.deploy_bundle).strip()
        if not deploy_bundle:
            raise ValueError(f"--deploy_bundle is required for model_variant={variant}")
        bundle = torch.load(deploy_bundle, map_location="cpu")
        model_path = str(bundle["base_model"])
        model, quant_eval = _build_shared_model_for_eval(
            base_model=model_path,
            atlas_payload=bundle.get("atlas", {}),
            shared_payload=bundle["shared_student"],
            quant_bank_int4=bundle.get("quant_bank_int4"),
            use_quant_bank_int4=bool(args.use_quant_bank_int4),
            device=device,
            dtype=dtype,
            trust_remote_code=bool(getattr(args, "trust_remote_code", False)),
        )
        phase_modules_applied = {"phase1_5": True}
    else:
        raise ValueError(f"Unsupported --model_variant={variant}. Use baseline or phase1_5.")
    model.eval()
    shared_diagnostics = _collect_shared_model_diagnostics(model) if variant == "phase1_5" else {}
    tokenizer = load_tokenizer(
        str(args.tokenizer_name_or_path).strip() or model_path,
        trust_remote_code=bool(getattr(args, "trust_remote_code", False)),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    trajectory_enabled = bool(getattr(args, "traj_enable", False)) and is_main_process(dist_ctx)
    trajectory_teacher = None
    trajectory_teacher_path = ""
    trajectory_teacher_loader = str(getattr(args, "traj_teacher_loader", "auto")).strip().lower()
    trajectory_teacher_base_model = str(getattr(args, "traj_teacher_base_model", "")).strip()
    trajectory_max_samples = max(1, int(getattr(args, "traj_max_samples", 32)))
    trajectory_token_rule = str(getattr(args, "traj_token_rule", "last_content")).strip().lower()
    trajectory_student_samples: List[torch.Tensor] = []
    trajectory_teacher_samples: List[torch.Tensor] = []
    trajectory_sample_indices: List[int] = []
    trajectory_collect_bar = None
    if trajectory_enabled:
        trajectory_teacher_path = str(getattr(args, "traj_teacher_ckpt", "")).strip()
        if not trajectory_teacher_path and variant == "phase1_5":
            shared_meta = bundle.get("shared_student", {}).get("meta", {}) if isinstance(bundle, dict) else {}
            trajectory_teacher_path = str(shared_meta.get("teacher_model", "")).strip()
        if not trajectory_teacher_path:
            raise ValueError("--traj_enable=True requires --traj_teacher_ckpt, or deploy bundle meta.teacher_model.")
        if trajectory_teacher_loader not in {"auto", "native", "stateft"}:
            raise ValueError(f"Unsupported --traj_teacher_loader={trajectory_teacher_loader!r}. Use auto/native/stateft.")
        if not trajectory_teacher_base_model:
            trajectory_teacher_base_model = str(args.base_model).strip() or str(model_path).strip()
        trajectory_teacher = build_frozen_teacher(
            teacher_ckpt_dir=trajectory_teacher_path,
            base_model=trajectory_teacher_base_model,
            teacher_loader=trajectory_teacher_loader,
            target_dtype=dtype,
            trust_remote_code=bool(getattr(args, "trust_remote_code", False)),
        )
        trajectory_teacher.to(device)
        trajectory_teacher.eval()
        for parameter in trajectory_teacher.parameters():
            parameter.requires_grad_(False)
        print(
            f"[Eval][Trajectory] enabled teacher={trajectory_teacher_path} loader={trajectory_teacher_loader} "
            f"max_samples={trajectory_max_samples} token_rule={trajectory_token_rule}",
            flush=True,
        )
        trajectory_collect_bar = step_progress(
            total=int(trajectory_max_samples),
            desc="[Eval][Trajectory] collect",
        )

    total = 0
    correct = 0
    is_gsm8k = dataset.strip().lower() == "gsm8k"
    pred_count = {cand: 0 for cand in candidates}
    eval_mode = str(getattr(args, "eval_mode", "logprob")).strip().lower()
    warmup_left = max(0, int(getattr(args, "throughput_warmup_samples", 0)))
    measured_samples = 0
    total_forward_tokens = 0
    total_forward_calls = 0
    start_time: Optional[float] = None
    if warmup_left == 0:
        _sync_if_cuda()
        start_time = time.perf_counter()

    batch_size = max(1, int(getattr(args, "batch_size", 16)))
    total_batches = (len(records) + batch_size - 1) // batch_size
    batch_iter = iter_progress(
        range(total_batches),
        total=total_batches,
        desc=f"[Eval] {args.model_variant}|{dataset}" + (f"|rank{dist_ctx.rank}" if dist_ctx.enabled else ""),
    )
    for batch_id in batch_iter:
        begin = int(batch_id) * batch_size
        end = min(len(records), begin + batch_size)
        batch = records[begin:end]
        for offset, item in enumerate(batch):
            if is_gsm8k and bool(getattr(args, "reasoning_chat_prompt", False)):
                prompt = _build_math_reasoning_chat_prompt(tokenizer, item)
            else:
                prompt = _build_prompt(item, dataset)
            answer_raw: Any = item.get("answer", item.get("label", ""))
            if is_gsm8k:
                pred, scores, stats = predict_gsm8k_generate(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    device=device,
                    max_new_tokens=int(getattr(args, "max_new_tokens", 256)),
                )
            elif eval_mode == "logprob":
                pred, scores, stats = score_candidates_logprob(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    candidates=candidates,
                    device=device,
                    length_norm=str(args.length_norm),
                )
            elif eval_mode == "generate":
                pred, scores, stats = predict_by_generate(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    candidates=candidates,
                    device=device,
                    max_new_tokens=int(getattr(args, "max_new_tokens", 16)),
                )
            else:
                raise ValueError(f"Unsupported --eval_mode={eval_mode}. Use logprob or generate.")
            if bool(getattr(args, "print_scores", False)):
                print(pred, scores, flush=True)

            if warmup_left > 0:
                warmup_left -= 1
                if warmup_left == 0:
                    _sync_if_cuda()
                    start_time = time.perf_counter()
            else:
                measured_samples += 1
                total_forward_tokens += int(stats.get("forward_tokens", 0))
                total_forward_calls += int(stats.get("forward_calls", 0))

            if is_gsm8k:
                answer = _gsm8k_gold_answer(item)
            elif isinstance(answer_raw, int):
                answer = str(candidates[answer_raw]) if 0 <= int(answer_raw) < len(candidates) else str(answer_raw)
            else:
                answer = str(answer_raw).strip()
            if answer.isdigit():
                idx = int(answer)
                if 0 <= idx < len(candidates):
                    answer = str(candidates[idx])

            pred_text = str(pred).strip()
            total += 1
            if pred_text == answer:
                correct += 1
            if pred_text in pred_count:
                pred_count[pred_text] += 1

            if trajectory_teacher is not None and len(trajectory_student_samples) < trajectory_max_samples:
                student_traj, teacher_traj, _ = _collect_trajectory_pair_for_prompt(
                    student_model=model,
                    teacher_model=trajectory_teacher,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    device=device,
                    token_rule=trajectory_token_rule,
                )
                if student_traj is not None and teacher_traj is not None:
                    common_layer_count = min(int(student_traj.size(0)), int(teacher_traj.size(0)))
                    if common_layer_count > 0:
                        trajectory_student_samples.append(student_traj[:common_layer_count, :].contiguous())
                        trajectory_teacher_samples.append(teacher_traj[:common_layer_count, :].contiguous())
                        global_index = int(begin + offset)
                        if dist_ctx.enabled:
                            global_index = int(dist_ctx.rank) + int(global_index) * int(dist_ctx.world_size)
                        trajectory_sample_indices.append(global_index)
                        if trajectory_collect_bar is not None:
                            trajectory_collect_bar.update(1)

    if trajectory_collect_bar is not None:
        trajectory_collect_bar.close()

    _sync_if_cuda()
    elapsed = max(1e-9, time.perf_counter() - start_time) if start_time is not None else 0.0
    if dist_ctx.enabled:
        stats_tensor = torch.tensor(
            [
                float(total),
                float(correct),
                float(measured_samples),
                float(total_forward_tokens),
                float(total_forward_calls),
            ],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)
        elapsed = dist_max(float(elapsed), device, dist_ctx)
        pred_tensor = torch.tensor(
            [float(pred_count.get(cand, 0)) for cand in candidates],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(pred_tensor, op=dist.ReduceOp.SUM)
        total = int(stats_tensor[0].item())
        correct = int(stats_tensor[1].item())
        measured_samples = int(stats_tensor[2].item())
        total_forward_tokens = int(stats_tensor[3].item())
        total_forward_calls = int(stats_tensor[4].item())
        pred_count = {cand: int(pred_tensor[idx].item()) for idx, cand in enumerate(candidates)}
    acc = float(correct / max(1, total))
    samples_per_s = (float(measured_samples) / elapsed) if measured_samples > 0 and elapsed > 0 else None
    throughput_toks_per_s = (float(total_forward_tokens) / elapsed) if total_forward_tokens > 0 and elapsed > 0 else None
    output_path = str(args.output_json or f"./results/newthesis_eval_{args.model_variant}_{dataset}.json")
    if dist_ctx.enabled and not is_main_process(dist_ctx):
        if trajectory_teacher is not None:
            del trajectory_teacher
        dist_barrier(dist_ctx)
        finalize_distributed(dist_ctx)
        return {}
    report = {
        "model_variant": str(args.model_variant),
        "phase_modules_applied": phase_modules_applied,
        "quant_eval": quant_eval,
        "shared_diagnostics": shared_diagnostics,
        "dataset": dataset,
        "eval_mode": eval_mode,
        "length_norm": str(getattr(args, "length_norm", "none")),
        "samples": int(total),
        "source_samples": int(total_records_before_shard),
        "world_size": int(dist_ctx.world_size),
        "accuracy": float(acc),
        "elapsed_sec": float(elapsed),
        "measured_samples": int(measured_samples),
        "throughput_warmup_samples": int(getattr(args, "throughput_warmup_samples", 0)),
        "samples_per_s": float(samples_per_s) if samples_per_s is not None else None,
        "throughput_toks_per_s": float(throughput_toks_per_s) if throughput_toks_per_s is not None else None,
        "forward_tokens": int(total_forward_tokens),
        "forward_calls": int(total_forward_calls),
        "pred_dist": pred_count,
    }
    if trajectory_teacher is not None:
        traj_output_png = str(getattr(args, "traj_output_png", "")).strip() or f"{os.path.splitext(output_path)[0]}_trajectory_3d.png"
        traj_output_json = str(getattr(args, "traj_output_json", "")).strip() or f"{os.path.splitext(output_path)[0]}_trajectory.json"
        traj_title = f"{dataset} last-token hidden trajectory: merged teacher vs {str(args.model_variant)}"
        trajectory_report = _build_trajectory_outputs(
            teacher_trajs=trajectory_teacher_samples,
            student_trajs=trajectory_student_samples,
            sample_indices=trajectory_sample_indices,
            output_png=traj_output_png,
            output_json=traj_output_json,
            title=traj_title,
            max_plot_lines=max(1, int(getattr(args, "traj_plot_max_lines", 16))),
        )
        trajectory_report["teacher_ckpt"] = trajectory_teacher_path
        trajectory_report["teacher_loader"] = trajectory_teacher_loader
        trajectory_report["teacher_base_model"] = trajectory_teacher_base_model
        report["trajectory"] = trajectory_report

    if trajectory_teacher is not None:
        del trajectory_teacher

    save_json(output_path, report)
    print(
        f"[Eval] {args.model_variant}|{dataset} mode={eval_mode} quant={quant_eval.get('enabled', False)} "
        f"acc={acc:.4f} samples={total} "
        f"samples/s={samples_per_s if samples_per_s is not None else 'NA'} "
        f"toks/s={throughput_toks_per_s if throughput_toks_per_s is not None else 'NA'} "
        f"elapsed={elapsed:.2f}s",
        flush=True,
    )
    if "trajectory" in report:
        plot_path = str(report["trajectory"].get("output_png", ""))
        pair_count = int(report["trajectory"].get("paired_samples", 0))
        print(f"[Eval][Trajectory] pairs={pair_count} plot={plot_path}", flush=True)
    print(f"[Eval] saved {output_path}")
    if dist_ctx.enabled:
        dist_barrier(dist_ctx)
        finalize_distributed(dist_ctx)
    return report


def stage_all(args: argparse.Namespace) -> Dict[str, Any]:
    return stage_all_final(args)


def _add_common_model_data_final(p: argparse.ArgumentParser) -> None:
    p.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL)
    p.add_argument("--teacher_ckpt", type=str, default="")
    p.add_argument(
        "--teacher_deploy_bundle",
        type=str,
        default="",
        help="Optional Phase-1.5 deploy_bundle.pt used as the frozen teacher during compression.",
    )
    p.add_argument("--teacher_loader", type=str, default="native", choices=["auto", "native", "stateft"])
    p.add_argument("--tokenizer_name_or_path", type=str, default="")
    p.add_argument("--trust_remote_code", type=str2bool, default=True)
    p.add_argument("--student_gradient_checkpointing", type=str2bool, default=False)
    p.add_argument("--teacher_gradient_checkpointing", type=str2bool, default=False)
    p.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--cutoff_len", type=int, default=384)
    p.add_argument("--max_records", type=int, default=0)
    p.add_argument("--max_batches", type=int, default=0)
    p.add_argument("--shuffle_records", type=str2bool, default=True)
    p.add_argument(
        "--training_prompt_mode",
        type=str,
        default="legacy_sft",
        choices=["legacy_sft", "decision_aligned", "math_reasoning"],
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda", choices=["auto", "cuda", "cpu"])
    p.add_argument("--num_gpus", type=int, default=1)
    p.add_argument("--master_addr", type=str, default="127.0.0.1")
    p.add_argument("--master_port", type=int, default=29501)


def _add_atlas_args_final(p: argparse.ArgumentParser) -> None:
    p.add_argument("--output_dir", type=str, default="")
    p.add_argument("--analysis_rank", type=int, default=256)
    p.add_argument(
        "--projection_basis_source",
        type=str,
        default="velocity_pca",
        choices=["velocity_pca", "teacher_velocity_pca", "hidden_pca", "teacher_hidden_pca", "hidden_state_pca", "random_orthogonal", "random_projection"],
    )
    p.add_argument("--random_basis_seed", type=int, default=0)
    p.add_argument("--pca_mode", type=str, default="stream_sketch", choices=["stream_sketch", "lowrank"])
    p.add_argument("--pca_device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--pca_stream_chunk_size", type=int, default=4096)
    p.add_argument("--pca_oversample", type=int, default=32)
    p.add_argument("--pca_niter", type=int, default=1)
    p.add_argument("--reservoir_size", type=int, default=1024)
    p.add_argument("--num_codes", type=int, default=64)
    p.add_argument("--kmeans_mode", type=str, default="minibatch", choices=["minibatch", "full"])
    p.add_argument("--kmeans_iters", type=int, default=25)
    p.add_argument("--kmeans_batch_size", type=int, default=4096)
    p.add_argument("--kmeans_warmup_size", type=int, default=50000)
    p.add_argument("--kmeans_warmup_iters", type=int, default=12)
    p.add_argument("--kmeans_refine_iters", type=int, default=4)
    p.add_argument("--kmeans_assign_chunk_size", type=int, default=8192)
    p.add_argument("--kmeans_device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--tau_min_var", type=float, default=1e-4)
    p.add_argument("--tau_topk", type=int, default=3)
    p.add_argument("--tau_eps", type=float, default=1e-5)
    p.add_argument("--tau_nmin", type=int, default=50)
    p.add_argument("--tau_shrink_lambda", type=float, default=0.1)
    p.add_argument("--token_rule", type=str, default="last_pred", choices=["last_content", "last_pred", "response_pred", "response_all", "all_response", "all_pred"])
    p.add_argument("--sharing_policy_mode", type=str, default="upstream_only")
    p.add_argument("--upstream_similarity_threshold", type=float, default=0.95)
    p.add_argument("--window_size", type=int, default=1)
    p.add_argument("--window_sample_mode", type=str, default="all", choices=["all", "random"])
    p.add_argument("--window_random_pick_min", type=int, default=1)
    p.add_argument("--window_random_pick_max", type=int, default=2)
    p.add_argument("--ckpt_every_batches", type=int, default=5000)


def _add_training_args_final(p: argparse.ArgumentParser) -> None:
    p.add_argument("--atlas_path", type=str, required=True)
    p.add_argument("--sharing_policy_path", type=str, default="")
    p.add_argument("--output_dir", type=str, default="")
    p.add_argument("--private_down_rank", type=int, default=64)
    p.add_argument("--use_layer_scalar", type=str2bool, default=True)
    p.add_argument("--adapter_every_layer", type=str2bool, default=False)
    p.add_argument(
        "--sharing_parameterization",
        type=str,
        default="full_parallel",
        choices=[
            "full_parallel",
            "down_only_parallel",
            "internal_weight_delta",
        ],
    )
    p.add_argument(
        "--private_down_alpha",
        type=float,
        default=0.0,
        help="LoRA-style alpha for the private FFN adapter; <=0 uses private_down_rank for backward-compatible scaling=1.",
    )
    p.add_argument(
        "--proto_seed_strategy",
        type=str,
        default="medoid",
        choices=["medoid", "policy_medoid", "first"],
    )
    p.add_argument("--steps", type=int, default=12000)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--lr_bank", type=float, default=4e-5)
    p.add_argument("--lr_adapter", type=float, default=1e-4)
    p.add_argument("--lr_schedule", type=str, default="warmup_cosine", choices=["none", "warmup_cosine"])
    p.add_argument("--lr_warmup_steps", type=int, default=0)
    p.add_argument("--lr_warmup_ratio", type=float, default=0.1)
    p.add_argument("--lr_min_ratio", type=float, default=0.01)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--distill_mode", type=str, default="ce_kd", choices=["ce", "ce_kd", "ce+kd", "sage_ib", "sage-ib", "ce_hidden_mse", "ce-hidden-mse", "hidden_mse"])
    p.add_argument("--teacher_free_ce", type=str2bool, default=False)
    p.add_argument("--lambda_ce", type=float, default=1.0)
    p.add_argument("--lambda_kd", type=float, default=0.3)
    p.add_argument("--kd_temperature", type=float, default=2.0)
    p.add_argument("--loss_scope", type=str, default="all", choices=["all", "response", "decision"])
    p.add_argument("--loss_exclude_eos", type=str2bool, default=True)
    p.add_argument("--sage_topk", type=int, default=32)
    p.add_argument("--sage_gain_margin", type=float, default=0.0)
    p.add_argument("--sage_gain_temperature", type=float, default=0.25)
    p.add_argument("--sage_confidence_margin", type=float, default=0.0)
    p.add_argument("--sage_confidence_temperature", type=float, default=1.0)
    p.add_argument("--sage_confidence_power", type=float, default=1.0)
    p.add_argument("--sage_require_teacher_correct", type=str2bool, default=True)
    p.add_argument("--sage_rate_warmup_ratio", type=float, default=0.10)
    p.add_argument("--sage_rate_decay_start_ratio", type=float, default=0.55)
    p.add_argument("--sage_rate_min_ratio", type=float, default=0.10)
    p.add_argument("--lambda_hidden_mse", type=float, default=1.0)
    p.add_argument("--lambda_core", type=float, default=0.12)
    p.add_argument("--core_lambda_schedule", type=str, default="constant", choices=["constant", "warmup", "linear_decay", "early_only", "early_then_ce"])
    p.add_argument("--core_lambda_warmup_ratio", type=float, default=0.1)
    p.add_argument("--core_lambda_cutoff_ratio", type=float, default=0.5)
    p.add_argument("--core_layers", type=str, default="all_shared_layers")
    p.add_argument("--core_metric_eps", type=float, default=1e-5)
    p.add_argument("--core_use_metric_whitening", type=str2bool, default=True)
    p.add_argument(
        "--core_coordinate_mode",
        type=str,
        default="projected",
        choices=["projected", "ambient"],
    )
    p.add_argument("--core_basis_mode", choices=["global", "regime"], default="regime")
    p.add_argument("--core_metric_trace_normalize", type=str2bool, default=False)
    p.add_argument("--core_metric_diag_path", type=str, default="")
    p.add_argument("--core_metric_diag_mode", type=str, default="covariance", choices=["covariance", "precision"])
    p.add_argument("--core_use_reliability_weighting", type=str2bool, default=True)
    p.add_argument("--core_token_selection", type=str, default="last_pred", choices=["last_pred", "single", "anchor", "response_all", "all_response", "response_pred", "all_pred", "all_tokens", "phase_response", "iets", "iets_softmax", "iets_topk", "energy", "energy_softmax"])
    p.add_argument("--core_candidate_tokens", type=int, default=1)
    p.add_argument("--core_iets_temperature", type=float, default=1.0)
    p.add_argument("--core_iets_anchor_boost", type=float, default=0.0)
    p.add_argument("--core_iets_energy_alpha", type=float, default=1.0)
    p.add_argument("--core_iets_entropy_beta", type=float, default=0.0)
    p.add_argument("--core_iets_topk", type=int, default=1)
    p.add_argument("--core_use_information_weighting", type=str2bool, default=False)
    p.add_argument("--core_information_power", type=float, default=1.0)
    p.add_argument("--lambda_layer_mixture", type=float, default=0.0)
    p.add_argument("--layer_mixture_entropy_tau", type=float, default=0.0)
    p.add_argument("--layer_mixture_assignment_temperature", type=float, default=1.0)
    p.add_argument("--layer_mixture_gate_hidden", type=int, default=64)
    p.add_argument("--layer_mixture_lr", type=float, default=0.0)
    p.add_argument("--layer_mixture_covariance_trace_normalize", type=str2bool, default=False)
    p.add_argument("--layer_mixture_delta_l2", type=float, default=0.0)
    p.add_argument("--lambda_phase_adaptive_core", type=float, default=0.0)
    p.add_argument("--phase_projector_bank_path", type=str, default="")
    p.add_argument(
        "--phase_projector_mode",
        type=str,
        default="phase",
        choices=["fixed", "layer", "phase", "soft"],
    )
    p.add_argument("--phase_projector_lr", type=float, default=0.0)
    p.add_argument("--lambda_geodesic_core", type=float, default=0.0)
    p.add_argument("--geodesic_core_max_layer_gap", type=int, default=1)
    p.add_argument("--lambda_relational_core", type=float, default=0.0)
    p.add_argument("--lambda_variance_core", type=float, default=0.0)
    p.add_argument("--variance_core_floor_ratio", type=float, default=0.70)
    p.add_argument("--lambda_token_flow_core", type=float, default=0.0)
    p.add_argument("--lambda_token_turning_core", type=float, default=0.0)
    p.add_argument("--token_flow_energy_fraction", type=float, default=0.95)
    p.add_argument("--generic_replay_data_path", type=str, default="")
    p.add_argument("--generic_replay_interval", type=int, default=0)
    p.add_argument("--generic_replay_batch_size", type=int, default=0)
    p.add_argument("--generic_replay_max_records", type=int, default=0)
    p.add_argument("--lambda_generic_replay", type=float, default=0.0)
    p.add_argument("--on_policy_enable", type=str2bool, default=False)
    p.add_argument("--on_policy_start_step", type=int, default=1500)
    p.add_argument("--on_policy_interval", type=int, default=5)
    p.add_argument("--on_policy_batch_size", type=int, default=1)
    p.add_argument("--on_policy_max_new_tokens", type=int, default=192)
    p.add_argument("--on_policy_generation_temperature", type=float, default=0.7)
    p.add_argument("--on_policy_top_p", type=float, default=0.95)
    p.add_argument(
        "--on_policy_divergence",
        type=str,
        default="js",
        choices=["js", "forward_kl", "reverse_kl"],
    )
    p.add_argument("--on_policy_temperature", type=float, default=1.0)
    p.add_argument("--on_policy_lambda_kd", type=float, default=1.0)
    p.add_argument("--on_policy_lambda_core", type=float, default=0.25)
    p.add_argument("--on_policy_lambda_flow", type=float, default=0.0)
    p.add_argument("--on_policy_lambda_turning", type=float, default=0.0)
    p.add_argument("--on_policy_teacher_advantage_margin", type=float, default=0.0)
    p.add_argument("--on_policy_final_answer_weight", type=float, default=2.0)
    p.add_argument("--on_policy_core_tokens", type=int, default=8)
    p.add_argument("--on_policy_ramp_steps", type=int, default=1000)
    p.add_argument(
        "--on_policy_require_activation",
        type=str2bool,
        default=False,
        help="Fail the job if on-policy is enabled but its activation count remains zero.",
    )
    p.add_argument("--lambda_manifold_core", type=float, default=0.0)
    p.add_argument("--manifold_core_temperature", type=float, default=1.0)
    p.add_argument("--lambda_delta_manifold_core", type=float, default=0.0)
    p.add_argument("--delta_manifold_core_temperature", type=float, default=1.0)
    p.add_argument("--delta_manifold_risk_weight", type=float, default=0.25)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--log_every", type=int, default=200)
    p.add_argument("--val_data_path", type=str, default="")
    p.add_argument("--val_batch_size", type=int, default=0)
    p.add_argument("--val_every", type=int, default=500)
    p.add_argument("--val_max_records", type=int, default=2048)
    p.add_argument("--val_max_batches", type=int, default=0)
    p.add_argument("--val_seed", type=int, default=42)
    p.add_argument("--val_selection_metric", type=str, default="loss", choices=["loss", "ce", "response_ce", "decision_ce"])
    p.add_argument("--val_include_step0_candidate", type=str2bool, default=False)
    p.add_argument("--val_min_improvement", type=float, default=0.0)
    p.add_argument("--val_subspace_viz_enable", type=str2bool, default=False)
    p.add_argument("--val_subspace_viz_max_points_per_regime", type=int, default=256)
    p.add_argument("--init_shared_student_ckpt", type=str, default="")
    p.add_argument(
        "--residual_svd_init_mode",
        type=str,
        default="none",
        choices=["none", "functional", "task_metric"],
    )
    p.add_argument("--residual_svd_init_records", type=int, default=512)
    p.add_argument("--residual_svd_tokens_per_record", type=int, default=10)
    p.add_argument("--residual_svd_max_fit_tokens", type=int, default=1024)
    p.add_argument("--residual_svd_init_ridge", type=float, default=1e-3)
    p.add_argument("--residual_svd_init_oversample", type=int, default=16)
    p.add_argument("--residual_svd_metric_complement_floor", type=float, default=0.1)
    p.add_argument("--internal_delta_svd_oversample", type=int, default=16)
    p.add_argument("--internal_delta_svd_niter", type=int, default=2)


def _normalize_pass_args_final(args: argparse.Namespace, training_stage: str) -> argparse.Namespace:
    args.training_stage = str(training_stage)
    args.lora_rank = int(getattr(args, "private_down_rank", getattr(args, "lora_rank", 64)))
    private_down_alpha = float(getattr(args, "private_down_alpha", 0.0))
    args.lora_alpha = private_down_alpha if private_down_alpha > 0.0 else float(args.lora_rank)
    args.tau_layers = str(getattr(args, "core_layers", "all_shared_layers"))
    args.tau_extra_topk = 0
    args.tau_topk = int(getattr(args, "tau_topk", 3))
    args.tau_eps = float(getattr(args, "core_metric_eps", getattr(args, "tau_eps", 1e-5)))
    args.lambda_tau_max = float(getattr(args, "lambda_core", getattr(args, "lambda_tau_max", 0.0)))
    args.distill_mode = _normalize_distill_mode(getattr(args, "distill_mode", "ce"))
    args.training_prompt_mode = str(getattr(args, "training_prompt_mode", "legacy_sft")).strip().lower()
    args.loss_scope = str(getattr(args, "loss_scope", "all")).strip().lower()
    args.loss_exclude_eos = bool(getattr(args, "loss_exclude_eos", True))
    args.core_use_metric_whitening = bool(getattr(args, "core_use_metric_whitening", True))
    args.core_coordinate_mode = str(
        getattr(args, "core_coordinate_mode", "projected")
    ).strip().lower()
    args.core_basis_mode = str(getattr(args, "core_basis_mode", "regime")).strip().lower()
    args.core_metric_trace_normalize = bool(getattr(args, "core_metric_trace_normalize", False))
    args.core_metric_diag_path = str(getattr(args, "core_metric_diag_path", "")).strip()
    args.core_metric_diag_mode = str(getattr(args, "core_metric_diag_mode", "covariance")).strip().lower()
    args.core_use_reliability_weighting = bool(getattr(args, "core_use_reliability_weighting", True))
    args.core_token_selection = str(getattr(args, "core_token_selection", "last_pred")).strip().lower()
    args.core_candidate_tokens = max(1, int(getattr(args, "core_candidate_tokens", 1)))
    args.core_iets_temperature = max(1e-4, float(getattr(args, "core_iets_temperature", 1.0)))
    args.core_iets_anchor_boost = float(getattr(args, "core_iets_anchor_boost", 0.0))
    args.core_iets_energy_alpha = float(getattr(args, "core_iets_energy_alpha", 1.0))
    args.core_iets_entropy_beta = float(getattr(args, "core_iets_entropy_beta", 0.0))
    args.core_iets_topk = max(1, int(getattr(args, "core_iets_topk", 1)))
    args.core_use_information_weighting = bool(getattr(args, "core_use_information_weighting", False))
    args.core_information_power = max(0.0, float(getattr(args, "core_information_power", 1.0)))
    args.lambda_geodesic_core = max(0.0, float(getattr(args, "lambda_geodesic_core", 0.0)))
    args.geodesic_core_max_layer_gap = max(1, int(getattr(args, "geodesic_core_max_layer_gap", 1)))
    args.lambda_manifold_core = max(0.0, float(getattr(args, "lambda_manifold_core", 0.0)))
    args.manifold_core_temperature = max(1e-4, float(getattr(args, "manifold_core_temperature", 1.0)))
    args.lambda_delta_manifold_core = max(0.0, float(getattr(args, "lambda_delta_manifold_core", 0.0)))
    args.delta_manifold_core_temperature = max(1e-4, float(getattr(args, "delta_manifold_core_temperature", 1.0)))
    args.delta_manifold_risk_weight = max(0.0, float(getattr(args, "delta_manifold_risk_weight", 0.25)))
    args.tau_token_rule = "last_pred"
    return args


def _build_pass_namespace_final(
    args: argparse.Namespace,
    *,
    atlas_path: str,
    output_dir: str,
) -> argparse.Namespace:
    stage_args = argparse.Namespace(**vars(args))
    prefix = "pass1"
    stage_args.atlas_path = str(atlas_path)
    stage_args.output_dir = str(output_dir)
    stage_args.init_shared_student_ckpt = ""
    stage_args.steps = int(getattr(args, f"{prefix}_steps"))
    stage_args.lr = float(getattr(args, f"{prefix}_lr"))
    stage_args.lr_bank = float(getattr(args, f"{prefix}_lr_bank"))
    stage_args.lr_adapter = float(getattr(args, f"{prefix}_lr_adapter"))
    stage_args.lr_schedule = str(getattr(args, f"{prefix}_lr_schedule"))
    stage_args.lr_warmup_steps = int(getattr(args, f"{prefix}_lr_warmup_steps"))
    stage_args.lr_warmup_ratio = float(getattr(args, f"{prefix}_lr_warmup_ratio"))
    stage_args.lr_min_ratio = float(getattr(args, f"{prefix}_lr_min_ratio"))
    stage_args.weight_decay = float(getattr(args, f"{prefix}_weight_decay"))
    stage_args.distill_mode = _normalize_distill_mode(getattr(args, f"{prefix}_distill_mode"))
    stage_args.lambda_ce = float(getattr(args, f"{prefix}_lambda_ce"))
    stage_args.lambda_kd = float(getattr(args, f"{prefix}_lambda_kd"))
    stage_args.kd_temperature = float(getattr(args, f"{prefix}_kd_temperature"))
    stage_args.lambda_hidden_mse = float(getattr(args, f"{prefix}_lambda_hidden_mse", getattr(args, "lambda_hidden_mse", 1.0)))
    stage_args.lambda_core = float(getattr(args, f"{prefix}_lambda_core"))
    stage_args.core_lambda_schedule = str(getattr(args, f"{prefix}_core_lambda_schedule", "constant"))
    stage_args.core_lambda_warmup_ratio = float(getattr(args, f"{prefix}_core_lambda_warmup_ratio", 0.1))
    stage_args.core_lambda_cutoff_ratio = float(getattr(args, f"{prefix}_core_lambda_cutoff_ratio", 0.5))
    stage_args.core_layers = str(getattr(args, f"{prefix}_core_layers"))
    stage_args.core_use_metric_whitening = bool(getattr(args, f"{prefix}_core_use_metric_whitening"))
    stage_args.core_coordinate_mode = str(
        getattr(args, f"{prefix}_core_coordinate_mode", "projected")
    )
    stage_args.core_basis_mode = str(getattr(args, f"{prefix}_core_basis_mode", "regime"))
    stage_args.core_metric_trace_normalize = bool(getattr(args, f"{prefix}_core_metric_trace_normalize"))
    stage_args.core_metric_diag_path = str(getattr(args, f"{prefix}_core_metric_diag_path", ""))
    stage_args.core_metric_diag_mode = str(getattr(args, f"{prefix}_core_metric_diag_mode", "covariance"))
    stage_args.core_use_reliability_weighting = bool(getattr(args, f"{prefix}_core_use_reliability_weighting"))
    stage_args.core_token_selection = str(getattr(args, f"{prefix}_core_token_selection", getattr(args, "core_token_selection", "last_pred")))
    stage_args.core_candidate_tokens = int(getattr(args, f"{prefix}_core_candidate_tokens", getattr(args, "core_candidate_tokens", 1)))
    stage_args.core_iets_temperature = float(getattr(args, f"{prefix}_core_iets_temperature", getattr(args, "core_iets_temperature", 1.0)))
    stage_args.core_iets_anchor_boost = float(getattr(args, f"{prefix}_core_iets_anchor_boost", getattr(args, "core_iets_anchor_boost", 0.0)))
    stage_args.core_iets_energy_alpha = float(getattr(args, f"{prefix}_core_iets_energy_alpha", getattr(args, "core_iets_energy_alpha", 1.0)))
    stage_args.core_iets_entropy_beta = float(getattr(args, f"{prefix}_core_iets_entropy_beta", getattr(args, "core_iets_entropy_beta", 0.0)))
    stage_args.core_iets_topk = int(getattr(args, f"{prefix}_core_iets_topk", getattr(args, "core_iets_topk", 1)))
    stage_args.core_use_information_weighting = bool(getattr(args, f"{prefix}_core_use_information_weighting", getattr(args, "core_use_information_weighting", False)))
    stage_args.core_information_power = float(getattr(args, f"{prefix}_core_information_power", getattr(args, "core_information_power", 1.0)))
    stage_args.lambda_geodesic_core = float(getattr(args, f"{prefix}_lambda_geodesic_core", getattr(args, "lambda_geodesic_core", 0.0)))
    stage_args.geodesic_core_max_layer_gap = int(getattr(args, f"{prefix}_geodesic_core_max_layer_gap", getattr(args, "geodesic_core_max_layer_gap", 1)))
    stage_args.lambda_manifold_core = float(getattr(args, f"{prefix}_lambda_manifold_core", getattr(args, "lambda_manifold_core", 0.0)))
    stage_args.manifold_core_temperature = float(getattr(args, f"{prefix}_manifold_core_temperature", getattr(args, "manifold_core_temperature", 1.0)))
    stage_args.lambda_delta_manifold_core = float(getattr(args, f"{prefix}_lambda_delta_manifold_core", getattr(args, "lambda_delta_manifold_core", 0.0)))
    stage_args.delta_manifold_core_temperature = float(getattr(args, f"{prefix}_delta_manifold_core_temperature", getattr(args, "delta_manifold_core_temperature", 1.0)))
    stage_args.delta_manifold_risk_weight = float(getattr(args, f"{prefix}_delta_manifold_risk_weight", getattr(args, "delta_manifold_risk_weight", 0.25)))
    stage_args.grad_clip = float(getattr(args, f"{prefix}_grad_clip"))
    stage_args.log_every = int(getattr(args, f"{prefix}_log_every"))
    stage_args.val_data_path = str(getattr(args, f"{prefix}_val_data_path"))
    stage_args.val_every = int(getattr(args, f"{prefix}_val_every"))
    stage_args.val_max_records = int(getattr(args, f"{prefix}_val_max_records"))
    stage_args.val_max_batches = int(getattr(args, f"{prefix}_val_max_batches"))
    stage_args.val_seed = int(getattr(args, f"{prefix}_val_seed"))
    stage_args.val_subspace_viz_enable = bool(getattr(args, f"{prefix}_val_subspace_viz_enable"))
    stage_args.val_subspace_viz_max_points_per_regime = int(getattr(args, f"{prefix}_val_subspace_viz_max_points_per_regime"))
    return _normalize_pass_args_final(stage_args, "pass1")


def stage_all_final(args: argparse.Namespace) -> Dict[str, Any]:
    root = str(args.output_root or f"./out/newthesis_final_llama_{now_tag()}")
    ensure_dir(root)
    atlas_dir = os.path.join(root, "phase1_atlas")
    pass1_dir = os.path.join(root, "phase1_5_pass1")
    export_dir = os.path.join(root, "phase4_export")

    def run_stage(name: str, fn, stage_args: argparse.Namespace):
        stage_dir = str(getattr(stage_args, "output_dir", "") or root)
        ensure_dir(stage_dir)
        stage_log_path = os.path.join(stage_dir, "train.log")
        with tee_output_to_file(stage_log_path):
            stage_t0 = time.perf_counter()
            print(f"[All] >>> {name} start", flush=True)
            result = fn(stage_args)
            stage_elapsed = time.perf_counter() - stage_t0
            print(f"[All] <<< {name} done ({stage_elapsed:.1f}s)", flush=True)
        if isinstance(result, dict):
            result.setdefault("train_log", stage_log_path)
        return result

    atlas_args = argparse.Namespace(**vars(args))
    atlas_args.output_dir = atlas_dir
    atlas_report = run_stage("atlas", stage_atlas, atlas_args)

    pass1_args = _build_pass_namespace_final(args, atlas_path=str(atlas_report["atlas_path"]), output_dir=pass1_dir)
    pass1_report = run_stage("pass1", stage_compress, pass1_args)

    export_args = argparse.Namespace(**vars(args))
    export_args.atlas_path = atlas_report["atlas_path"]
    export_args.shared_student_ckpt = pass1_report["shared_student_ckpt"]
    export_args.output_dir = export_dir
    export_report = run_stage("export", stage_export, export_args)

    all_report = {
        "output_root": root,
        "atlas": atlas_report,
        "pass1": pass1_report,
        "compress": pass1_report,
        "export": export_report,
    }
    report_path = os.path.join(root, "newthesis_all_report.json")
    save_json(report_path, all_report)
    print(f"[All] saved {report_path}")
    return all_report


def build_final_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NEWTHESIS final pipeline: performance-first structure prior compression.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    atlas = sub.add_parser("atlas", help="Phase 1 atlas and structure-prior extraction")
    _add_common_model_data_final(atlas)
    _add_atlas_args_final(atlas)

    pass1 = sub.add_parser("pass1", help="Train the shared core with task + core losses")
    _add_common_model_data_final(pass1)
    _add_training_args_final(pass1)

    export = sub.add_parser("export", help="Phase 4 export deploy bundle + quantized bank")
    export.add_argument("--atlas_path", type=str, required=True)
    export.add_argument("--shared_student_ckpt", type=str, required=True)
    export.add_argument("--output_dir", type=str, default="")
    export.add_argument("--quant_bits", type=int, default=4, choices=[4, 8, 16])

    eval_p = sub.add_parser("eval", help="Evaluate baseline or exported bundle")
    eval_p.add_argument("--model_variant", type=str, default="phase1_5", choices=["baseline", "phase1_5"])
    eval_p.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL)
    eval_p.add_argument("--deploy_bundle", type=str, default="")
    eval_p.add_argument("--tokenizer_name_or_path", type=str, default="")
    eval_p.add_argument("--trust_remote_code", type=str2bool, default=True)
    eval_p.add_argument("--test_data_root", type=str, default=os.path.join(WORKSPACE_ROOT, "data", "datasets"))
    eval_p.add_argument("--dataset", type=str, default="piqa")
    eval_p.add_argument("--batch_size", type=int, default=16)
    eval_p.add_argument("--max_samples", type=int, default=0)
    eval_p.add_argument("--eval_mode", choices=["logprob", "generate"], default="logprob")
    eval_p.add_argument("--length_norm", choices=["none", "avg"], default="none")
    eval_p.add_argument("--print_scores", action="store_true")
    eval_p.add_argument("--max_new_tokens", type=int, default=16)
    eval_p.add_argument(
        "--reasoning_chat_prompt",
        type=str2bool,
        default=False,
        help="Use the same full-reasoning chat prompt as math Teacher/Student training (GSM8K).",
    )
    eval_p.add_argument("--throughput_warmup_samples", type=int, default=0)
    eval_p.add_argument("--use_quant_bank_int4", type=str2bool, default=False)
    eval_p.add_argument("--device", type=str, default="cuda", choices=["auto", "cuda", "cpu"])
    eval_p.add_argument("--seed", type=int, default=42)
    eval_p.add_argument("--output_json", type=str, default="")
    eval_p.add_argument("--traj_enable", type=str2bool, default=False)
    eval_p.add_argument("--traj_teacher_ckpt", type=str, default="")
    eval_p.add_argument("--traj_teacher_loader", type=str, default="auto", choices=["auto", "native", "stateft"])
    eval_p.add_argument("--traj_teacher_base_model", type=str, default="")
    eval_p.add_argument("--traj_max_samples", type=int, default=32)
    eval_p.add_argument("--traj_token_rule", type=str, default="last_content", choices=["last_content", "last_pred"])
    eval_p.add_argument("--traj_output_png", type=str, default="")
    eval_p.add_argument("--traj_output_json", type=str, default="")
    eval_p.add_argument("--traj_plot_max_lines", type=int, default=16)

    all_p = sub.add_parser("all", help="Run atlas -> pass1 -> export")
    _add_common_model_data_final(all_p)
    _add_atlas_args_final(all_p)
    all_p.add_argument("--output_root", type=str, default="")
    all_p.add_argument("--sharing_policy_path", type=str, default="")
    all_p.add_argument("--private_down_rank", type=int, default=64)
    all_p.add_argument("--use_layer_scalar", type=str2bool, default=True)
    all_p.add_argument("--adapter_every_layer", type=str2bool, default=False)
    all_p.add_argument(
        "--sharing_parameterization",
        type=str,
        default="full_parallel",
        choices=[
            "full_parallel",
            "down_only_parallel",
            "internal_weight_delta",
        ],
    )
    all_p.add_argument(
        "--private_down_alpha",
        type=float,
        default=0.0,
        help="LoRA-style alpha for the private FFN adapter; <=0 uses private_down_rank for backward-compatible scaling=1.",
    )
    all_p.add_argument(
        "--proto_seed_strategy",
        type=str,
        default="medoid",
        choices=["medoid", "policy_medoid", "first"],
    )
    all_p.add_argument("--pass1_steps", type=int, default=12000)
    all_p.add_argument("--pass1_lr", type=float, default=5e-5)
    all_p.add_argument("--pass1_lr_bank", type=float, default=4e-5)
    all_p.add_argument("--pass1_lr_adapter", type=float, default=1e-4)
    all_p.add_argument("--pass1_lr_schedule", type=str, default="warmup_cosine", choices=["none", "warmup_cosine"])
    all_p.add_argument("--pass1_lr_warmup_steps", type=int, default=0)
    all_p.add_argument("--pass1_lr_warmup_ratio", type=float, default=0.1)
    all_p.add_argument("--pass1_lr_min_ratio", type=float, default=0.01)
    all_p.add_argument("--pass1_weight_decay", type=float, default=0.01)
    all_p.add_argument("--pass1_distill_mode", type=str, default="ce_kd", choices=["ce", "ce_kd", "ce+kd", "ce_hidden_mse", "ce-hidden-mse", "hidden_mse"])
    all_p.add_argument("--pass1_lambda_ce", type=float, default=1.0)
    all_p.add_argument("--pass1_lambda_kd", type=float, default=0.3)
    all_p.add_argument("--pass1_kd_temperature", type=float, default=2.0)
    all_p.add_argument("--pass1_lambda_hidden_mse", type=float, default=1.0)
    all_p.add_argument("--pass1_lambda_core", type=float, default=0.12)
    all_p.add_argument("--pass1_core_lambda_schedule", type=str, default="constant", choices=["constant", "warmup", "linear_decay", "early_only", "early_then_ce"])
    all_p.add_argument("--pass1_core_lambda_warmup_ratio", type=float, default=0.1)
    all_p.add_argument("--pass1_core_lambda_cutoff_ratio", type=float, default=0.5)
    all_p.add_argument("--pass1_core_layers", type=str, default="all_shared_layers")
    all_p.add_argument("--pass1_core_use_metric_whitening", type=str2bool, default=True)
    all_p.add_argument(
        "--pass1_core_coordinate_mode",
        type=str,
        default="projected",
        choices=["projected", "ambient"],
    )
    all_p.add_argument("--pass1_core_basis_mode", choices=["global", "regime"], default="regime")
    all_p.add_argument("--pass1_core_metric_trace_normalize", type=str2bool, default=False)
    all_p.add_argument("--pass1_core_metric_diag_path", type=str, default="")
    all_p.add_argument("--pass1_core_metric_diag_mode", type=str, default="covariance", choices=["covariance", "precision"])
    all_p.add_argument("--pass1_core_use_reliability_weighting", type=str2bool, default=True)
    all_p.add_argument("--pass1_core_token_selection", type=str, default="last_pred", choices=["last_pred", "single", "anchor", "response_all", "all_response", "response_pred", "all_pred", "all_tokens", "phase_response", "iets", "iets_softmax", "iets_topk", "energy", "energy_softmax"])
    all_p.add_argument("--pass1_core_candidate_tokens", type=int, default=1)
    all_p.add_argument("--pass1_core_iets_temperature", type=float, default=1.0)
    all_p.add_argument("--pass1_core_iets_anchor_boost", type=float, default=0.0)
    all_p.add_argument("--pass1_core_iets_energy_alpha", type=float, default=1.0)
    all_p.add_argument("--pass1_core_iets_entropy_beta", type=float, default=0.0)
    all_p.add_argument("--pass1_core_iets_topk", type=int, default=1)
    all_p.add_argument("--pass1_core_use_information_weighting", type=str2bool, default=False)
    all_p.add_argument("--pass1_core_information_power", type=float, default=1.0)
    all_p.add_argument("--pass1_lambda_geodesic_core", type=float, default=0.0)
    all_p.add_argument("--pass1_geodesic_core_max_layer_gap", type=int, default=1)
    all_p.add_argument("--pass1_lambda_manifold_core", type=float, default=0.0)
    all_p.add_argument("--pass1_manifold_core_temperature", type=float, default=1.0)
    all_p.add_argument("--pass1_lambda_delta_manifold_core", type=float, default=0.0)
    all_p.add_argument("--pass1_delta_manifold_core_temperature", type=float, default=1.0)
    all_p.add_argument("--pass1_delta_manifold_risk_weight", type=float, default=0.25)
    all_p.add_argument("--pass1_grad_clip", type=float, default=1.0)
    all_p.add_argument("--pass1_log_every", type=int, default=200)
    all_p.add_argument("--pass1_val_data_path", type=str, default="")
    all_p.add_argument("--pass1_val_every", type=int, default=500)
    all_p.add_argument("--pass1_val_max_records", type=int, default=2048)
    all_p.add_argument("--pass1_val_max_batches", type=int, default=0)
    all_p.add_argument("--pass1_val_seed", type=int, default=42)
    all_p.add_argument("--pass1_val_subspace_viz_enable", type=str2bool, default=False)
    all_p.add_argument("--pass1_val_subspace_viz_max_points_per_regime", type=int, default=256)
    all_p.add_argument("--quant_bits", type=int, default=4, choices=[4, 8, 16])

    return parser


def build_parser() -> argparse.ArgumentParser:
    return build_final_parser()


def main() -> None:
    parser = build_final_parser()
    args = parser.parse_args()
    if args.cmd == "atlas":
        stage_atlas(args)
        return
    if args.cmd == "pass1":
        stage_compress(_normalize_pass_args_final(args, "pass1"))
        return
    if args.cmd == "export":
        stage_export(args)
        return
    if args.cmd == "eval":
        stage_eval(args)
        return
    if args.cmd == "all":
        stage_all_final(args)
        return
    raise ValueError(f"Unsupported cmd: {args.cmd}")


if __name__ == "__main__":
    main()
