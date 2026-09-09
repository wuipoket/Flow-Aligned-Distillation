#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid bool: {value!r}")


def ensure_dir(path: str | os.PathLike[str]) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(path: str | os.PathLike[str], payload: Dict[str, Any]) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)


def setup_distributed() -> Tuple[int, int, int, torch.device, bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    distributed = world_size > 1
    if distributed and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if use_cuda else "gloo")
    return rank, local_rank, world_size, device, distributed


def cleanup_distributed(distributed: bool) -> None:
    if distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def barrier(distributed: bool) -> None:
    if distributed and dist.is_initialized():
        dist.barrier()


def dtype_from_name(name: str, device: torch.device) -> torch.dtype:
    text = str(name).strip().lower()
    if text in {"auto", ""}:
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    if text in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if text in {"fp16", "float16", "half"}:
        return torch.float16
    if text in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unknown dtype: {name}")


def load_json_records(path: str, max_records: int, seed: int) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"expected list JSON: {path}")
    records = [row for row in payload if isinstance(row, dict)]
    rng = random.Random(int(seed))
    rng.shuffle(records)
    if max_records > 0:
        records = records[: int(max_records)]
    return records


def render_commonsense_record(row: Dict[str, Any]) -> str:
    instruction = str(row.get("instruction", "")).strip()
    input_text = str(row.get("input", "")).strip()
    output = str(row.get("output", "")).strip()
    parts = []
    if instruction:
        parts.append("Instruction:\n" + instruction)
    if input_text:
        parts.append("Input:\n" + input_text)
    if output:
        parts.append("Answer:\n" + output)
    return "\n\n".join(parts).strip()


def load_wikitext_texts(max_records: int) -> List[str]:
    from datasets import load_dataset

    texts: List[str] = []
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    for row in ds:
        text = str(row.get("text", "")).strip()
        if text:
            texts.append(text)
        if max_records > 0 and len(texts) >= max_records:
            break
    return texts


def build_lm_blocks(
    tokenizer: Any,
    *,
    data_path: str,
    data_kind: str,
    max_records: int,
    block_size: int,
    max_blocks: int,
    seed: int,
) -> torch.Tensor:
    if data_kind == "commonsense_json":
        texts = [render_commonsense_record(row) for row in load_json_records(data_path, max_records=max_records, seed=seed)]
    elif data_kind == "wikitext2":
        texts = load_wikitext_texts(max_records=max_records)
    else:
        raise ValueError(f"unsupported data_kind: {data_kind}")
    texts = [text for text in texts if text]
    if not texts:
        raise RuntimeError("no texts loaded")
    joined = "\n\n".join(texts)
    encoded = tokenizer(joined, return_tensors="pt", add_special_tokens=True)
    ids = encoded["input_ids"][0].to(torch.long)
    usable = (int(ids.numel()) // int(block_size)) * int(block_size)
    if usable < int(block_size):
        raise RuntimeError(f"not enough tokens for block_size={block_size}: {int(ids.numel())}")
    blocks = ids[:usable].view(-1, int(block_size)).contiguous()
    if max_blocks > 0:
        blocks = blocks[: int(max_blocks)]
    return blocks


def get_decoder_layers(model: torch.nn.Module) -> Sequence[torch.nn.Module]:
    root = getattr(model, "model", model)
    layers = getattr(root, "layers", None)
    if layers is None:
        raise RuntimeError("cannot find model.layers")
    return layers


def get_mlp_param_count(model: torch.nn.Module) -> int:
    layers = get_decoder_layers(model)
    if not layers:
        return 0
    return int(sum(p.numel() for p in layers[0].mlp.parameters()))


def choose_positions(total_tokens: int, max_positions: int, generator: torch.Generator, device: torch.device) -> torch.Tensor:
    if total_tokens <= max_positions:
        return torch.arange(total_tokens, device=device, dtype=torch.long)
    return torch.randperm(total_tokens, generator=generator, device=device)[: int(max_positions)]


@torch.no_grad()
def collect_feature_shard(
    model: torch.nn.Module,
    blocks: torch.Tensor,
    *,
    device: torch.device,
    rank: int,
    world_size: int,
    output_dir: Path,
    batch_size: int,
    max_batches: int,
    max_positions_per_batch: int,
    max_samples_per_layer: int,
    projection_dim: int,
    seed: int,
) -> None:
    layers = get_decoder_layers(model)
    layer_count = len(layers)
    hidden_size = int(getattr(model.config, "hidden_size"))
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed) + 1009 * int(rank))
    projection = torch.randn(
        hidden_size,
        int(projection_dim),
        generator=torch.Generator().manual_seed(int(seed) + 17),
        dtype=torch.float32,
    )
    if int(hidden_size) >= int(projection_dim):
        projection = torch.linalg.qr(projection, mode="reduced").Q
        projection_kind = "seeded_orthonormal_random_subspace"
    else:
        projection = F.normalize(projection, dim=0)
        projection_kind = "seeded_column_normalized_random_projection"
    projection = projection.to(device=device, dtype=torch.float32)
    input_features: List[List[torch.Tensor]] = [[] for _ in range(layer_count)]
    post_attn_residual_features: List[List[torch.Tensor]] = [[] for _ in range(layer_count)]
    delta_features: List[List[torch.Tensor]] = [[] for _ in range(layer_count)]
    block_delta_features: List[List[torch.Tensor]] = [[] for _ in range(layer_count)]
    # One set of positions is selected before each forward and reused by every
    # hook.  This is required for paired-token metrics across layers/views.
    current_positions: Optional[torch.Tensor] = None
    sample_count = 0
    handles = []
    post_attn_hook_count = 0

    def make_post_attn_pre_hook(layer_id: int):
        def hook(_module, args):
            if current_positions is None:
                return
            hidden = args[0] if isinstance(args, (tuple, list)) and args else None
            if hidden is None or not torch.is_tensor(hidden):
                return
            flat = hidden.detach().reshape(-1, hidden.shape[-1])
            pos = current_positions
            if pos.numel() <= 0:
                return
            projected = flat.index_select(0, pos).to(dtype=torch.float32) @ projection
            post_attn_residual_features[layer_id].append(projected.cpu().to(dtype=torch.float16))

        return hook

    def make_mlp_hook(layer_id: int):
        def hook(_module, args, output):
            if current_positions is None:
                return
            hidden = args[0] if isinstance(args, (tuple, list)) and args else None
            if hidden is None or not torch.is_tensor(hidden) or not torch.is_tensor(output):
                return
            flat_in = hidden.detach().reshape(-1, hidden.shape[-1])
            flat_out = output.detach().reshape(-1, output.shape[-1])
            pos = current_positions
            if pos.numel() <= 0:
                return
            x = flat_in.index_select(0, pos).to(dtype=torch.float32) @ projection
            y = flat_out.index_select(0, pos).to(dtype=torch.float32) @ projection
            input_features[layer_id].append(x.cpu().to(dtype=torch.float16))
            delta_features[layer_id].append(y.cpu().to(dtype=torch.float16))

        return hook

    def make_layer_hook(layer_id: int):
        def hook(_module, args, output):
            if current_positions is None:
                return
            hidden = args[0] if isinstance(args, (tuple, list)) and args else None
            layer_output = output[0] if isinstance(output, (tuple, list)) and output else output
            if hidden is None or not torch.is_tensor(hidden) or not torch.is_tensor(layer_output):
                return
            flat_in = hidden.detach().reshape(-1, hidden.shape[-1])
            flat_out = layer_output.detach().reshape(-1, layer_output.shape[-1])
            pos = current_positions
            if pos.numel() <= 0:
                return
            block_delta = flat_out.index_select(0, pos) - flat_in.index_select(0, pos)
            z_block_delta = block_delta.to(dtype=torch.float32) @ projection
            block_delta_features[layer_id].append(z_block_delta.cpu().to(dtype=torch.float16))

        return hook

    for layer_id, layer in enumerate(layers):
        handles.append(layer.mlp.register_forward_hook(make_mlp_hook(layer_id)))
        handles.append(layer.register_forward_hook(make_layer_hook(layer_id)))
        pre_ffn_norm = getattr(layer, "pre_feedforward_layernorm", None)
        if pre_ffn_norm is not None:
            handles.append(pre_ffn_norm.register_forward_pre_hook(make_post_attn_pre_hook(layer_id)))
            post_attn_hook_count += 1
    try:
        # DDP ranks must observe disjoint activation blocks.  Pair interventions
        # are partitioned elsewhere, but feature collection used to duplicate
        # the same blocks on every rank.
        local_blocks = blocks[int(rank) :: int(world_size)]
        selected = local_blocks[: int(max_batches) if max_batches > 0 else len(local_blocks)]
        for start in range(0, len(selected), int(batch_size)):
            batch = selected[start : start + int(batch_size)].to(device)
            remaining = int(max_samples_per_layer) - int(sample_count)
            if remaining <= 0:
                break
            position_limit = int(max_positions_per_batch)
            if position_limit <= 0:
                position_limit = int(batch.numel())
            take_n = min(position_limit, int(batch.numel()), remaining)
            current_positions = choose_positions(int(batch.numel()), take_n, generator, batch.device)
            _ = model(input_ids=batch, use_cache=False)
            sample_count += int(take_n)
            current_positions = None
    finally:
        current_positions = None
        for handle in handles:
            handle.remove()

    shard = {
        "rank": int(rank),
        "world_size": int(world_size),
        "input": [torch.cat(items, dim=0) if items else torch.empty(0, projection_dim, dtype=torch.float16) for items in input_features],
        "delta": [torch.cat(items, dim=0) if items else torch.empty(0, projection_dim, dtype=torch.float16) for items in delta_features],
        "block_delta": [
            torch.cat(items, dim=0) if items else torch.empty(0, projection_dim, dtype=torch.float16)
            for items in block_delta_features
        ],
        "sample_count": int(sample_count),
        "feature_pairing": "same token indices for every layer and representation view",
        "projection_kind": projection_kind,
    }
    if post_attn_hook_count == layer_count:
        shard["post_attn_residual"] = [
            torch.cat(items, dim=0) if items else torch.empty(0, projection_dim, dtype=torch.float16)
            for items in post_attn_residual_features
        ]
    torch.save(shard, output_dir / "shards" / f"features_rank{rank}.pt")


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    n = min(int(x.size(0)), int(y.size(0)))
    if n < 2:
        return float("nan")
    x = x[:n].to(dtype=torch.float32)
    y = y[:n].to(dtype=torch.float32)
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    xty = x.T @ y
    xtx = x.T @ x
    yty = y.T @ y
    denom = torch.linalg.matrix_norm(xtx) * torch.linalg.matrix_norm(yty)
    if float(denom.item()) <= 0.0:
        return float("nan")
    return float((torch.linalg.matrix_norm(xty) ** 2 / denom).item())


def cosine_mean(x: torch.Tensor, y: torch.Tensor) -> float:
    n = min(int(x.size(0)), int(y.size(0)))
    if n < 1:
        return float("nan")
    x = F.normalize(x[:n].to(dtype=torch.float32), dim=-1)
    y = F.normalize(y[:n].to(dtype=torch.float32), dim=-1)
    return float((x * y).sum(dim=-1).mean().item())


def write_matrix_csv(path: Path, matrix: torch.Tensor) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        layer_count = int(matrix.size(0))
        writer.writerow(["layer", *[str(i) for i in range(layer_count)]])
        for i in range(layer_count):
            writer.writerow([i, *[f"{float(matrix[i, j]):.8g}" for j in range(layer_count)]])


def plot_heatmap(path: Path, matrix: torch.Tensor, title: str, cmap: str = "viridis") -> None:
    if plt is None:
        return
    ensure_dir(path.parent)
    arr = matrix.detach().cpu().float().numpy()
    fig, ax = plt.subplots(figsize=(8, 7), dpi=160)
    image = ax.imshow(arr, interpolation="nearest", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("source layer")
    ax.set_ylabel("target layer")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def summarize_features(output_dir: Path, world_size: int, layer_count: int) -> Dict[str, Any]:
    shards = [torch.load(output_dir / "shards" / f"features_rank{rank}.pt", map_location="cpu") for rank in range(world_size)]
    feature_kinds = ["input", "delta"]
    if all(
        "post_attn_residual" in shard
        and all(int(item.size(0)) > 0 for item in shard["post_attn_residual"])
        for shard in shards
    ):
        feature_kinds.append("post_attn_residual")
    if all("block_delta" in shard for shard in shards):
        feature_kinds.append("block_delta")
    merged: Dict[str, List[torch.Tensor]] = {kind: [] for kind in feature_kinds}
    for kind in feature_kinds:
        for layer_id in range(layer_count):
            merged[kind].append(torch.cat([shard[kind][layer_id] for shard in shards], dim=0))
    matrices: Dict[str, torch.Tensor] = {}
    for kind in feature_kinds:
        cka = torch.empty(layer_count, layer_count)
        paired_cosine = torch.empty(layer_count, layer_count)
        for i in range(layer_count):
            for j in range(layer_count):
                cka[i, j] = linear_cka(merged[kind][i], merged[kind][j])
                paired_cosine[i, j] = cosine_mean(merged[kind][i], merged[kind][j])
        prefix = "delta" if kind == "delta" else kind
        matrices[f"{prefix}_cka"] = cka
        matrices[f"{prefix}_cosine"] = paired_cosine
    for name, matrix in matrices.items():
        torch.save(matrix, output_dir / f"{name}.pt")
        write_matrix_csv(output_dir / f"{name}.csv", matrix)
        plot_heatmap(output_dir / f"{name}.png", matrix, name)
    return {
        "feature_samples_per_layer": {
            str(i): int(merged["input"][i].size(0))
            for i in range(layer_count)
        },
        "feature_pairing": "same token indices for every layer and representation view",
        "cosine_aggregation": "mean of paired-token cosine similarities",
        "representation_definitions": {
            "post_attn_residual": "input to post_attention_layernorm; the legacy atlas grouping view",
            "input": "normalized tensor received by layer.mlp",
            "delta": "layer.mlp output = FFN residual update",
            "block_delta": "decoder-layer output minus decoder-layer input",
        },
        "matrices": {name: str(output_dir / f"{name}.csv") for name in matrices},
    }


def shift_ce_nll(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="mean",
    )


def shift_kl_mean(reference_logits: torch.Tensor, candidate_logits: torch.Tensor, chunk_tokens: int = 64) -> torch.Tensor:
    ref = reference_logits[:, :-1, :].reshape(-1, reference_logits.size(-1))
    cand = candidate_logits[:, :-1, :].reshape(-1, candidate_logits.size(-1))
    total = torch.zeros((), device=ref.device, dtype=torch.float64)
    count = 0
    for start in range(0, int(ref.size(0)), int(chunk_tokens)):
        r = ref[start : start + int(chunk_tokens)].to(dtype=torch.float32)
        c = cand[start : start + int(chunk_tokens)].to(dtype=torch.float32)
        log_p = F.log_softmax(r, dim=-1)
        log_q = F.log_softmax(c, dim=-1)
        p = log_p.exp()
        total = total + (p * (log_p - log_q)).sum(dim=-1).to(dtype=torch.float64).sum()
        count += int(r.size(0))
    return (total / max(1, count)).to(dtype=torch.float32)


@torch.no_grad()
def evaluate_pair_intervention(
    model: torch.nn.Module,
    blocks: torch.Tensor,
    *,
    device: torch.device,
    source_layer: int,
    target_layer: int,
    batch_size: int,
    max_batches: int,
    kl_chunk_tokens: int,
) -> Dict[str, Any]:
    layers = get_decoder_layers(model)
    source_mlp = layers[int(source_layer)].mlp
    original_target_mlp = layers[int(target_layer)].mlp
    selected = blocks[: int(max_batches) if max_batches > 0 else len(blocks)]
    baseline_nll_sum = 0.0
    intervention_nll_sum = 0.0
    kl_sum = 0.0
    batches = 0
    tokens = 0
    for start in range(0, len(selected), int(batch_size)):
        input_ids = selected[start : start + int(batch_size)].to(device)
        baseline = model(input_ids=input_ids, use_cache=False).logits
        baseline_nll = shift_ce_nll(baseline, input_ids)
        layers[int(target_layer)].mlp = source_mlp
        try:
            candidate = model(input_ids=input_ids, use_cache=False).logits
        finally:
            layers[int(target_layer)].mlp = original_target_mlp
        candidate_nll = shift_ce_nll(candidate, input_ids)
        kl = shift_kl_mean(baseline, candidate, chunk_tokens=int(kl_chunk_tokens))
        token_count = int(input_ids.size(0)) * max(0, int(input_ids.size(1)) - 1)
        baseline_nll_sum += float(baseline_nll.item()) * token_count
        intervention_nll_sum += float(candidate_nll.item()) * token_count
        kl_sum += float(kl.item()) * token_count
        tokens += token_count
        batches += 1
        del baseline, candidate
    baseline_nll = baseline_nll_sum / max(1, tokens)
    intervention_nll = intervention_nll_sum / max(1, tokens)
    return {
        "source_layer": int(source_layer),
        "target_layer": int(target_layer),
        "baseline_nll": float(baseline_nll),
        "intervention_nll": float(intervention_nll),
        "delta_nll": float(intervention_nll - baseline_nll),
        "kl_baseline_to_intervention": float(kl_sum / max(1, tokens)),
        "tokens": int(tokens),
        "batches": int(batches),
    }


def directed_pairs(layer_count: int, window: int) -> List[Tuple[int, int]]:
    pairs = []
    for target in range(layer_count):
        for source in range(layer_count):
            if source == target:
                continue
            if window > 0 and abs(int(source) - int(target)) > int(window):
                continue
            pairs.append((source, target))
    return pairs


def write_pair_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    fields = [
        "source_layer",
        "target_layer",
        "baseline_nll",
        "intervention_nll",
        "delta_nll",
        "kl_baseline_to_intervention",
        "merge_cost",
        "tokens",
        "batches",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def knee_count(costs: List[float]) -> int:
    if len(costs) <= 2:
        return len(costs)
    xs = torch.linspace(0.0, 1.0, len(costs))
    ys = torch.tensor(costs, dtype=torch.float32)
    y_min, y_max = float(ys.min().item()), float(ys.max().item())
    if y_max <= y_min:
        return 0
    ys = (ys - y_min) / (y_max - y_min)
    # For a sorted increasing curve, the knee is the largest distance below the diagonal.
    distances = xs - ys
    idx = int(torch.argmax(distances).item())
    return max(0, idx + 1)


def summarize_interventions(output_dir: Path, world_size: int, layer_count: int, kl_weight: float, mlp_params: int, total_params: int) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for rank in range(world_size):
        shard_path = output_dir / "shards" / f"interventions_rank{rank}.json"
        if not shard_path.exists():
            continue
        with shard_path.open("r", encoding="utf-8") as handle:
            rows.extend(json.load(handle))
    for row in rows:
        row["merge_cost"] = float(row["delta_nll"]) + float(kl_weight) * float(row["kl_baseline_to_intervention"])
    rows.sort(key=lambda item: (float(item["merge_cost"]), int(item["target_layer"]), int(item["source_layer"])))
    write_pair_rows(output_dir / "intervention_pairs_sorted.csv", rows)
    delta_matrix = torch.full((layer_count, layer_count), float("nan"))
    kl_matrix = torch.full((layer_count, layer_count), float("nan"))
    cost_matrix = torch.full((layer_count, layer_count), float("nan"))
    for row in rows:
        s = int(row["source_layer"])
        t = int(row["target_layer"])
        delta_matrix[t, s] = float(row["delta_nll"])
        kl_matrix[t, s] = float(row["kl_baseline_to_intervention"])
        cost_matrix[t, s] = float(row["merge_cost"])
    for name, matrix in (("delta_nll_intervention", delta_matrix), ("kl_intervention", kl_matrix), ("merge_cost", cost_matrix)):
        torch.save(matrix, output_dir / f"{name}.pt")
        write_matrix_csv(output_dir / f"{name}.csv", matrix)
        plot_heatmap(output_dir / f"{name}.png", matrix, name, cmap="magma")

    best_by_target: List[Dict[str, Any]] = []
    for target in range(layer_count):
        candidates = [row for row in rows if int(row["target_layer"]) == target]
        if candidates:
            best_by_target.append(candidates[0])
    best_by_target.sort(key=lambda item: float(item["merge_cost"]))
    k = knee_count([float(row["merge_cost"]) for row in best_by_target])
    suggested = best_by_target[:k]
    write_pair_rows(output_dir / "suggested_merges_knee.csv", suggested)
    saved_params = int(len(suggested) * int(mlp_params))
    natural_compression = float(saved_params / max(1, int(total_params)))
    return {
        "num_pairs": len(rows),
        "best_per_target_count": len(best_by_target),
        "suggested_merge_count_knee": len(suggested),
        "suggested_merge_cost_max": float(max([row["merge_cost"] for row in suggested], default=float("nan"))),
        "mlp_params_per_layer": int(mlp_params),
        "estimated_saved_params_from_suggested_merges": int(saved_params),
        "estimated_natural_compression_from_suggested_merges": natural_compression,
        "outputs": {
            "sorted_pairs": str(output_dir / "intervention_pairs_sorted.csv"),
            "suggested_merges_knee": str(output_dir / "suggested_merges_knee.csv"),
            "merge_cost_matrix": str(output_dir / "merge_cost.csv"),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Observation-driven FFN functional redundancy analysis.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--tokenizer_name_or_path", default="")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--data_kind", choices=["commonsense_json", "wikitext2"], default="commonsense_json")
    parser.add_argument("--feature_data_path", default="", help="Optional calibration source; defaults to --data_path")
    parser.add_argument("--intervention_data_path", default="", help="Optional held-out diagnostic source; defaults to --data_path")
    parser.add_argument("--feature_data_kind", default="", help="Defaults to --data_kind")
    parser.add_argument("--intervention_data_kind", default="", help="Defaults to --data_kind")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_records", type=int, default=100000)
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--max_blocks", type=int, default=64)
    parser.add_argument("--activation_batches", type=int, default=12)
    parser.add_argument("--intervention_batches", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_positions_per_batch", type=int, default=512)
    parser.add_argument("--max_samples_per_layer", type=int, default=4096)
    parser.add_argument("--projection_dim", type=int, default=256)
    parser.add_argument("--pair_window", type=int, default=0, help="0 evaluates all directed pairs; positive limits abs(source-target).")
    parser.add_argument("--kl_weight", type=float, default=1.0)
    parser.add_argument("--kl_chunk_tokens", type=int, default=32)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--trust_remote_code", type=str2bool, default=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    rank, local_rank, world_size, device, distributed = setup_distributed()
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir / "shards")
    try:
        random.seed(int(args.seed) + rank)
        torch.manual_seed(int(args.seed) + rank)
        if rank == 0:
            print(
                f"[FFN-Redundancy] world_size={world_size} model={args.model_name_or_path} "
                f"data_kind={args.data_kind} output={output_dir}",
                flush=True,
            )
        dtype = dtype_from_name(args.dtype, device)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=dtype,
            trust_remote_code=bool(args.trust_remote_code),
            low_cpu_mem_usage=True,
        ).to(device)
        model.eval()
        model.config.use_cache = False
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name_or_path or args.model_name_or_path,
            trust_remote_code=bool(args.trust_remote_code),
            use_fast=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        feature_data_path = str(args.feature_data_path).strip() or str(args.data_path)
        intervention_data_path = str(args.intervention_data_path).strip() or str(args.data_path)
        feature_data_kind = str(args.feature_data_kind).strip() or str(args.data_kind)
        intervention_data_kind = str(args.intervention_data_kind).strip() or str(args.data_kind)
        valid_data_kinds = {"commonsense_json", "wikitext2"}
        if feature_data_kind not in valid_data_kinds or intervention_data_kind not in valid_data_kinds:
            raise ValueError(
                f"data kinds must be in {sorted(valid_data_kinds)}: "
                f"feature={feature_data_kind!r}, intervention={intervention_data_kind!r}"
            )
        same_source = (
            Path(feature_data_path).expanduser().resolve() == Path(intervention_data_path).expanduser().resolve()
            and feature_data_kind == intervention_data_kind
        )
        if same_source:
            requested_total = 2 * int(args.max_blocks) if int(args.max_blocks) > 0 else 0
            all_blocks = build_lm_blocks(
                tokenizer,
                data_path=feature_data_path,
                data_kind=feature_data_kind,
                max_records=int(args.max_records),
                block_size=int(args.block_size),
                max_blocks=requested_total,
                seed=int(args.seed),
            )
            split_at = int(all_blocks.size(0)) // 2
            feature_blocks = all_blocks[:split_at].contiguous()
            intervention_blocks = all_blocks[split_at:].contiguous()
            data_split_mode = "disjoint_halves_from_same_source"
        else:
            feature_blocks = build_lm_blocks(
                tokenizer,
                data_path=feature_data_path,
                data_kind=feature_data_kind,
                max_records=int(args.max_records),
                block_size=int(args.block_size),
                max_blocks=int(args.max_blocks),
                seed=int(args.seed),
            )
            intervention_blocks = build_lm_blocks(
                tokenizer,
                data_path=intervention_data_path,
                data_kind=intervention_data_kind,
                max_records=int(args.max_records),
                block_size=int(args.block_size),
                max_blocks=int(args.max_blocks),
                seed=int(args.seed) + 1,
            )
            data_split_mode = "separate_sources"
        if int(feature_blocks.size(0)) <= 0 or int(intervention_blocks.size(0)) <= 0:
            raise RuntimeError("feature/intervention split produced an empty block set")
        layers = get_decoder_layers(model)
        layer_count = len(layers)
        total_params = int(sum(p.numel() for p in model.parameters()))
        mlp_params = get_mlp_param_count(model)
        if rank == 0:
            save_json(
                output_dir / "run_config.json",
                {
                    **vars(args),
                    "world_size": int(world_size),
                    "layer_count": int(layer_count),
                    "total_params": int(total_params),
                    "mlp_params_per_layer": int(mlp_params),
                    "feature_data_path_resolved": feature_data_path,
                    "intervention_data_path_resolved": intervention_data_path,
                    "feature_data_kind_resolved": feature_data_kind,
                    "intervention_data_kind_resolved": intervention_data_kind,
                    "data_split_mode": data_split_mode,
                    "feature_num_blocks": int(feature_blocks.size(0)),
                    "intervention_num_blocks": int(intervention_blocks.size(0)),
                    "feature_num_tokens": int(feature_blocks.numel()),
                    "intervention_num_tokens": int(intervention_blocks.numel()),
                },
            )
            print(
                f"[FFN-Redundancy] feature_blocks={int(feature_blocks.size(0))} "
                f"intervention_blocks={int(intervention_blocks.size(0))} "
                f"block_size={int(feature_blocks.size(1))} split={data_split_mode} "
                f"layers={layer_count} total_params={total_params}",
                flush=True,
            )
        collect_feature_shard(
            model,
            feature_blocks,
            device=device,
            rank=rank,
            world_size=world_size,
            output_dir=output_dir,
            batch_size=int(args.batch_size),
            max_batches=int(args.activation_batches),
            max_positions_per_batch=int(args.max_positions_per_batch),
            max_samples_per_layer=int(args.max_samples_per_layer),
            projection_dim=int(args.projection_dim),
            seed=int(args.seed),
        )
        barrier(distributed)
        feature_summary: Dict[str, Any] = {}
        if rank == 0:
            print("[FFN-Redundancy] summarizing feature similarities", flush=True)
            feature_summary = summarize_features(output_dir, world_size, layer_count)
            save_json(output_dir / "feature_summary.json", feature_summary)
        barrier(distributed)

        pairs = directed_pairs(layer_count, int(args.pair_window))
        local_pairs = [pair for idx, pair in enumerate(pairs) if idx % int(world_size) == int(rank)]
        local_rows: List[Dict[str, Any]] = []
        if rank == 0:
            print(f"[FFN-Redundancy] intervention pairs={len(pairs)} local_per_rank~{math.ceil(len(pairs)/world_size)}", flush=True)
        for idx, (source, target) in enumerate(local_pairs):
            row = evaluate_pair_intervention(
                model,
                intervention_blocks,
                device=device,
                source_layer=source,
                target_layer=target,
                batch_size=int(args.batch_size),
                max_batches=int(args.intervention_batches),
                kl_chunk_tokens=int(args.kl_chunk_tokens),
            )
            local_rows.append(row)
            if idx % 25 == 0:
                print(
                    f"[FFN-Redundancy][rank{rank}] {idx + 1}/{len(local_pairs)} "
                    f"src={source} tgt={target} dNLL={row['delta_nll']:.5f} KL={row['kl_baseline_to_intervention']:.5f}",
                    flush=True,
                )
        with (output_dir / "shards" / f"interventions_rank{rank}.json").open("w", encoding="utf-8") as handle:
            json.dump(local_rows, handle, ensure_ascii=False, indent=2)
        barrier(distributed)
        if rank == 0:
            intervention_summary = summarize_interventions(
                output_dir,
                world_size,
                layer_count,
                float(args.kl_weight),
                mlp_params,
                total_params,
            )
            summary = {
                "elapsed_s": float(time.perf_counter() - started),
                "feature_summary": feature_summary,
                "intervention_summary": intervention_summary,
            }
            save_json(output_dir / "summary.json", summary)
            print(f"[FFN-Redundancy] saved summary -> {output_dir / 'summary.json'}", flush=True)
    finally:
        cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
