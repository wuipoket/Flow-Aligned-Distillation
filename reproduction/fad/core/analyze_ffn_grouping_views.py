#!/usr/bin/env python3
"""Compare FFN-input, FFN-output, and functional grouping signals.

This script consumes the artifacts emitted by ``ffn_functional_redundancy_ddp.py``.
It deliberately separates three questions:

1. input-domain compatibility (the tensor received by the MLP),
2. native FFN transport similarity (MLP output = FFN residual update), and
3. counterfactual interchangeability (replace target MLP by source MLP).

It also emits matched-prototype sharing policies using constrained complete-link
agglomeration.  Those policies can be passed to the existing final-llama
compression pipeline through ``--sharing_policy_path``.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


EPS = 1e-12


def load_torch(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=False)
        handle.write("\n")


def write_matrix_csv(path: Path, matrix: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = matrix.detach().cpu().to(dtype=torch.float64)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["layer", *range(int(matrix.size(0)))])
        for i in range(int(matrix.size(0))):
            writer.writerow(
                [
                    i,
                    *[
                        "nan" if not math.isfinite(float(matrix[i, j])) else f"{float(matrix[i, j]):.10g}"
                        for j in range(int(matrix.size(1)))
                    ],
                ]
            )


def write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def plot_heatmap(path: Path, matrix: torch.Tensor, title: str, cmap: str = "viridis") -> None:
    if plt is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    values = matrix.detach().cpu().to(dtype=torch.float32).numpy()
    fig, ax = plt.subplots(figsize=(7.4, 6.4), dpi=160)
    image = ax.imshow(values, interpolation="nearest", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("layer j")
    ax.set_ylabel("layer i")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def load_matrix(obs_dir: Path, stem: str) -> Optional[torch.Tensor]:
    path = obs_dir / f"{stem}.pt"
    if not path.exists():
        return None
    value = load_torch(path)
    if not torch.is_tensor(value) or value.dim() != 2 or value.size(0) != value.size(1):
        raise ValueError(f"expected square tensor: {path}")
    return value.detach().cpu().to(dtype=torch.float64)


def symmetrize_cost(directed: torch.Tensor) -> torch.Tensor:
    if directed.dim() != 2 or directed.size(0) != directed.size(1):
        raise ValueError("directed intervention cost must be square")
    a = directed.detach().cpu().to(dtype=torch.float64)
    b = a.T
    both = torch.maximum(torch.nan_to_num(a, nan=-float("inf")), torch.nan_to_num(b, nan=-float("inf")))
    missing = torch.isnan(a) & torch.isnan(b)
    both[missing] = float("nan")
    both.fill_diagonal_(0.0)
    return both


def default_regimes(layer_count: int) -> List[str]:
    base, remainder = divmod(int(layer_count), 3)
    sizes = [base + (1 if idx < remainder else 0) for idx in range(3)]
    labels: List[str] = []
    for name, size in zip(("llama_early", "llama_mid", "llama_late"), sizes):
        labels.extend([name] * size)
    return labels[:layer_count]


def regimes_from_policy(policy: Mapping[str, Any], layer_count: int) -> List[str]:
    labels = default_regimes(layer_count)
    entries = policy.get("layers", [])
    if not isinstance(entries, list):
        return labels
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        layer_id = int(entry.get("layer_id", -1))
        if 0 <= layer_id < layer_count:
            labels[layer_id] = str(entry.get("regime", labels[layer_id]))
    return labels


def validate_similarity(matrix: torch.Tensor, layer_count: int, name: str) -> torch.Tensor:
    if tuple(matrix.shape) != (layer_count, layer_count):
        raise ValueError(f"{name}: expected {(layer_count, layer_count)}, got {tuple(matrix.shape)}")
    matrix = 0.5 * (matrix + matrix.T)
    matrix.fill_diagonal_(1.0)
    if not bool(torch.isfinite(matrix).all().item()):
        raise ValueError(f"{name}: similarity matrix contains non-finite values")
    return matrix


def rankdata(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.tensor(list(values), dtype=torch.float64)
    order = torch.argsort(tensor, stable=True)
    ranks = torch.empty_like(tensor)
    start = 0
    while start < int(tensor.numel()):
        end = start + 1
        current = float(tensor[order[start]])
        while end < int(tensor.numel()) and float(tensor[order[end]]) == current:
            end += 1
        average_rank = 0.5 * float(start + end - 1) + 1.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    tx = torch.tensor(list(x), dtype=torch.float64)
    ty = torch.tensor(list(y), dtype=torch.float64)
    if tx.numel() < 2:
        return float("nan")
    tx = tx - tx.mean()
    ty = ty - ty.mean()
    denom = torch.linalg.vector_norm(tx) * torch.linalg.vector_norm(ty)
    if float(denom) <= EPS:
        return float("nan")
    return float(torch.dot(tx, ty) / denom)


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) < 2:
        return float("nan")
    return pearson(rankdata(x).tolist(), rankdata(y).tolist())


def kendall_tau_b(x: Sequence[float], y: Sequence[float]) -> float:
    concordant = discordant = ties_x = ties_y = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx == 0.0 and dy == 0.0:
                ties_x += 1
                ties_y += 1
            elif dx == 0.0:
                ties_x += 1
            elif dy == 0.0:
                ties_y += 1
            elif dx * dy > 0.0:
                concordant += 1
            else:
                discordant += 1
    numerator = float(concordant - discordant)
    denominator = math.sqrt(
        float(concordant + discordant + ties_x) * float(concordant + discordant + ties_y)
    )
    return numerator / denominator if denominator > 0.0 else float("nan")


def auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    positive = [float(score) for score, label in zip(scores, labels) if int(label) == 1]
    negative = [float(score) for score, label in zip(scores, labels) if int(label) == 0]
    if not positive or not negative:
        return float("nan")
    wins = 0.0
    for pos in positive:
        for neg in negative:
            wins += 1.0 if pos > neg else 0.5 if pos == neg else 0.0
    return wins / float(len(positive) * len(negative))


def average_precision(scores: Sequence[float], labels: Sequence[int]) -> float:
    order = sorted(range(len(scores)), key=lambda idx: (-float(scores[idx]), idx))
    positives = sum(int(labels[idx]) for idx in order)
    if positives <= 0:
        return float("nan")
    hit = 0
    precision_sum = 0.0
    for rank, idx in enumerate(order, start=1):
        if int(labels[idx]) == 1:
            hit += 1
            precision_sum += float(hit) / float(rank)
    return precision_sum / float(positives)


def pair_indices(layer_count: int, regimes: Sequence[str], scope: str) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    for i in range(layer_count):
        for j in range(i + 1, layer_count):
            same_regime = str(regimes[i]) == str(regimes[j])
            adjacent = abs(i - j) == 1
            if scope == "all":
                keep = True
            elif scope == "same_regime":
                keep = same_regime
            elif scope == "adjacent":
                keep = adjacent
            elif scope == "same_regime_adjacent":
                keep = same_regime and adjacent
            else:
                raise ValueError(f"unknown scope: {scope}")
            if keep:
                pairs.append((i, j))
    return pairs


def rank_normalize_matrix(matrix: torch.Tensor, regimes: Sequence[str]) -> torch.Tensor:
    layer_count = int(matrix.size(0))
    pairs = pair_indices(layer_count, regimes, "same_regime")
    values = [float(matrix[i, j]) for i, j in pairs]
    ranks = rankdata(values)
    denom = max(1.0, float(len(values) - 1))
    normalized = torch.zeros_like(matrix, dtype=torch.float64)
    normalized.fill_diagonal_(1.0)
    for (i, j), rank in zip(pairs, ranks.tolist()):
        value = (float(rank) - 1.0) / denom
        normalized[i, j] = value
        normalized[j, i] = value
    return normalized


def qap_p_value(
    similarity: torch.Tensor,
    cost: torch.Tensor,
    regimes: Sequence[str],
    permutations: int,
    seed: int,
) -> Tuple[float, float]:
    pairs = pair_indices(int(similarity.size(0)), regimes, "same_regime")
    observed = spearman([float(similarity[i, j]) for i, j in pairs], [float(cost[i, j]) for i, j in pairs])
    if permutations <= 0 or not math.isfinite(observed):
        return observed, float("nan")
    rng = random.Random(int(seed))
    regime_to_layers: Dict[str, List[int]] = {}
    for layer_id, regime in enumerate(regimes):
        regime_to_layers.setdefault(str(regime), []).append(layer_id)
    at_least_as_good = 0
    for _ in range(int(permutations)):
        mapping = list(range(len(regimes)))
        for members in regime_to_layers.values():
            shuffled = list(members)
            rng.shuffle(shuffled)
            for original, replacement in zip(members, shuffled):
                mapping[original] = replacement
        permuted_values = [float(similarity[mapping[i], mapping[j]]) for i, j in pairs]
        rho = spearman(permuted_values, [float(cost[i, j]) for i, j in pairs])
        if math.isfinite(rho) and rho <= observed:
            at_least_as_good += 1
    return observed, float(at_least_as_good + 1) / float(permutations + 1)


def predictive_row(
    name: str,
    similarity: torch.Tensor,
    cost: torch.Tensor,
    regimes: Sequence[str],
    scope: str,
    safest_count: int,
    qap_permutations: int,
    seed: int,
) -> Dict[str, Any]:
    pairs = [
        (i, j)
        for i, j in pair_indices(int(similarity.size(0)), regimes, scope)
        if math.isfinite(float(cost[i, j])) and math.isfinite(float(similarity[i, j]))
    ]
    sim = [float(similarity[i, j]) for i, j in pairs]
    target = [float(cost[i, j]) for i, j in pairs]
    count = min(max(1, int(safest_count)), len(pairs))
    gold = set(sorted(range(len(pairs)), key=lambda idx: (target[idx], pairs[idx]))[:count])
    predicted = set(sorted(range(len(pairs)), key=lambda idx: (-sim[idx], pairs[idx]))[:count])
    labels = [1 if idx in gold else 0 for idx in range(len(pairs))]
    loo_values: List[float] = []
    for held_layer in range(int(similarity.size(0))):
        kept = [idx for idx, (i, j) in enumerate(pairs) if held_layer not in (i, j)]
        if len(kept) >= 3:
            loo_values.append(spearman([sim[idx] for idx in kept], [target[idx] for idx in kept]))
    qap_rho = float("nan")
    qap_p = float("nan")
    if scope == "same_regime":
        qap_rho, qap_p = qap_p_value(
            similarity,
            cost,
            regimes,
            permutations=int(qap_permutations),
            seed=int(seed),
        )
    return {
        "view": name,
        "scope": scope,
        "pair_count": len(pairs),
        "spearman_similarity_vs_cost": spearman(sim, target),
        "kendall_tau_b_similarity_vs_cost": kendall_tau_b(sim, target),
        "safe_pair_count": count,
        "safe_pair_precision_at_m": float(len(gold & predicted)) / float(count),
        "safe_pair_recall_at_m": float(len(gold & predicted)) / float(count),
        "safe_pair_auroc": auroc(sim, labels),
        "safe_pair_average_precision": average_precision(sim, labels),
        "leave_one_layer_out_rho_min": min(loo_values) if loo_values else float("nan"),
        "leave_one_layer_out_rho_mean": sum(loo_values) / len(loo_values) if loo_values else float("nan"),
        "leave_one_layer_out_rho_max": max(loo_values) if loo_values else float("nan"),
        "qap_rho": qap_rho,
        "qap_one_sided_p": qap_p,
    }


def complete_link_clusters(
    similarity: torch.Tensor,
    regimes: Sequence[str],
    target_clusters: int,
    max_group_size: int,
    max_layer_span: int,
    pinned_layers: Sequence[int],
    gate_similarity: Optional[torch.Tensor] = None,
    gate_threshold: Optional[float] = None,
    require_contiguous_groups: bool = False,
) -> List[List[int]]:
    layer_count = int(similarity.size(0))
    if target_clusters <= 0 or target_clusters > layer_count:
        raise ValueError(f"target_clusters must be in [1,{layer_count}]")
    pinned = {int(item) for item in pinned_layers}
    clusters: List[List[int]] = [[layer_id] for layer_id in range(layer_count)]
    while len(clusters) > int(target_clusters):
        candidates: List[Tuple[float, float, int, int, int]] = []
        for idx_a, a in enumerate(clusters):
            for idx_b in range(idx_a + 1, len(clusters)):
                b = clusters[idx_b]
                if any(layer_id in pinned for layer_id in a) or any(layer_id in pinned for layer_id in b):
                    continue
                if str(regimes[a[0]]) != str(regimes[b[0]]):
                    continue
                merged_size = len(a) + len(b)
                if max_group_size > 0 and merged_size > int(max_group_size):
                    continue
                combined = sorted(a + b)
                if max_layer_span > 0 and max(combined) - min(combined) > int(max_layer_span):
                    continue
                if (
                    require_contiguous_groups
                    and combined != list(range(min(combined), max(combined) + 1))
                ):
                    continue
                if gate_similarity is not None:
                    if gate_threshold is None:
                        raise ValueError("gate_threshold is required when gate_similarity is provided")
                    gate_cross = [float(gate_similarity[i, j]) for i in a for j in b]
                    if (
                        not gate_cross
                        or not all(math.isfinite(value) for value in gate_cross)
                        or min(gate_cross) < float(gate_threshold)
                    ):
                        continue
                cross = [float(similarity[i, j]) for i in a for j in b]
                if not cross or not all(math.isfinite(value) for value in cross):
                    continue
                # Complete-link: maximize the worst pair introduced by a merge.
                candidates.append((min(cross), sum(cross) / len(cross), -merged_size, idx_a, idx_b))
        if not candidates:
            gate_note = (
                f" with hard input gate >= {float(gate_threshold):.8g}"
                if gate_similarity is not None and gate_threshold is not None
                else ""
            )
            raise RuntimeError(
                f"cannot reach target_clusters={target_clusters}{gate_note}; "
                "relax only a predeclared constraint and rerun policy calibration"
            )
        _minimum, _mean, _negative_size, idx_a, idx_b = max(candidates)
        merged = sorted(clusters[idx_a] + clusters[idx_b])
        clusters[idx_a] = merged
        del clusters[idx_b]
    return sorted((sorted(cluster) for cluster in clusters), key=lambda item: (min(item), len(item)))


def medoid_for_group(group: Sequence[int], directed_cost: torch.Tensor) -> Tuple[int, float, float]:
    if len(group) <= 1:
        return int(group[0]), 0.0, 0.0
    candidates: List[Tuple[float, float, int]] = []
    for source in group:
        values = [float(directed_cost[target, source]) for target in group if int(target) != int(source)]
        finite = [value for value in values if math.isfinite(value)]
        if len(finite) != len(values):
            candidates.append((float("inf"), float("inf"), int(source)))
        else:
            candidates.append((max(finite), sum(finite) / len(finite), int(source)))
    max_cost, mean_cost, source = min(candidates)
    return source, mean_cost, max_cost


def summarize_clusters(
    name: str,
    clusters: Sequence[Sequence[int]],
    similarity: torch.Tensor,
    directed_cost: torch.Tensor,
    symmetric_cost: torch.Tensor,
    regimes: Sequence[str],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    group_rows: List[Dict[str, Any]] = []
    replacement_costs: List[float] = []
    pair_costs: List[float] = []
    pair_similarities: List[float] = []
    for cluster in clusters:
        members = [int(item) for item in cluster]
        medoid, medoid_mean, medoid_max = medoid_for_group(members, directed_cost)
        within_pairs = [(a, b) for idx, a in enumerate(members) for b in members[idx + 1 :]]
        similarities = [float(similarity[a, b]) for a, b in within_pairs]
        costs = [float(symmetric_cost[a, b]) for a, b in within_pairs]
        for target in members:
            if target != medoid:
                value = float(directed_cost[target, medoid])
                if math.isfinite(value):
                    replacement_costs.append(value)
        pair_costs.extend(value for value in costs if math.isfinite(value))
        pair_similarities.extend(value for value in similarities if math.isfinite(value))
        group_rows.append(
            {
                "view": name,
                "layers": ",".join(str(item) for item in members),
                "size": len(members),
                "regime": str(regimes[members[0]]),
                "medoid_layer": medoid,
                "medoid_to_member_cost_mean": medoid_mean,
                "medoid_to_member_cost_max": medoid_max,
                "within_similarity_min": min(similarities) if similarities else 1.0,
                "within_similarity_mean": sum(similarities) / len(similarities) if similarities else 1.0,
                "within_similarity_max": max(similarities) if similarities else 1.0,
                "within_symmetric_cost_mean": sum(costs) / len(costs) if costs else 0.0,
                "within_symmetric_cost_max": max(costs) if costs else 0.0,
            }
        )
    shared = [cluster for cluster in clusters if len(cluster) > 1]
    return (
        {
            "view": name,
            "prototype_count": len(clusters),
            "saved_mlp_count": sum(len(cluster) - 1 for cluster in clusters),
            "shared_group_count": len(shared),
            "shared_layer_count": sum(len(cluster) for cluster in shared),
            "within_similarity_min": min(pair_similarities) if pair_similarities else 1.0,
            "within_similarity_mean": sum(pair_similarities) / len(pair_similarities) if pair_similarities else 1.0,
            "within_symmetric_cost_mean": sum(pair_costs) / len(pair_costs) if pair_costs else 0.0,
            "within_symmetric_cost_max": max(pair_costs) if pair_costs else 0.0,
            "medoid_replacement_cost_mean": (
                sum(replacement_costs) / len(replacement_costs) if replacement_costs else 0.0
            ),
            "medoid_replacement_cost_max": max(replacement_costs) if replacement_costs else 0.0,
            "groups": [list(map(int, cluster)) for cluster in shared],
        },
        group_rows,
    )


def clusters_from_policy(policy: Mapping[str, Any], layer_count: int) -> List[List[int]]:
    assigned: set[int] = set()
    clusters: List[List[int]] = []
    groups = policy.get("groups", [])
    if isinstance(groups, list):
        for group in groups:
            if not isinstance(group, dict):
                continue
            members = sorted({int(item) for item in group.get("layers", [])})
            if members:
                clusters.append(members)
                assigned.update(members)
    for layer_id in range(layer_count):
        if layer_id not in assigned:
            clusters.append([layer_id])
    return sorted(clusters, key=lambda item: (min(item), len(item)))


def build_policy(
    base_policy: Mapping[str, Any],
    name: str,
    clusters: Sequence[Sequence[int]],
    similarity: torch.Tensor,
    directed_cost: torch.Tensor,
    symmetric_cost: torch.Tensor,
    regimes: Sequence[str],
    target_prototypes: int,
    max_group_size: int,
    max_layer_span: int,
    pinned_layers: Sequence[int],
) -> Dict[str, Any]:
    policy = copy.deepcopy(dict(base_policy))
    layer_count = int(similarity.size(0))
    old_entries = policy.get("layers", []) if isinstance(policy.get("layers", []), list) else []
    by_layer = {
        int(item.get("layer_id")): copy.deepcopy(item)
        for item in old_entries
        if isinstance(item, dict) and "layer_id" in item
    }
    groups: List[Dict[str, Any]] = []
    layer_to_group: Dict[int, int] = {}
    for cluster in clusters:
        members = [int(item) for item in cluster]
        if len(members) <= 1:
            continue
        group_id = len(groups)
        medoid, medoid_mean, medoid_max = medoid_for_group(members, directed_cost)
        pairs = [(a, b) for idx, a in enumerate(members) for b in members[idx + 1 :]]
        sim_values = [float(similarity[a, b]) for a, b in pairs]
        cost_values = [float(symmetric_cost[a, b]) for a, b in pairs]
        groups.append(
            {
                "group_id": group_id,
                "layers": members,
                "regime": str(regimes[members[0]]),
                "private_core": False,
                "medoid_layer": int(medoid),
                "mean_view_similarity": sum(sim_values) / len(sim_values),
                "min_view_similarity": min(sim_values),
                "mean_symmetric_intervention_cost": sum(cost_values) / len(cost_values),
                "max_symmetric_intervention_cost": max(cost_values),
                "medoid_to_member_cost_mean": float(medoid_mean),
                "medoid_to_member_cost_max": float(medoid_max),
            }
        )
        for layer_id in members:
            layer_to_group[layer_id] = group_id
    entries: List[Dict[str, Any]] = []
    for layer_id in range(layer_count):
        entry = by_layer.get(layer_id, {"layer_id": layer_id})
        entry["layer_id"] = layer_id
        entry["regime"] = str(regimes[layer_id])
        entry["group_id"] = int(layer_to_group.get(layer_id, -1))
        entry["private_core"] = layer_id not in layer_to_group
        entries.append(entry)
    policy.update(
        {
            "source_view": name,
            "sharing_policy_mode": "matched_budget_complete_linkage",
            "sharing_group_count": len(groups),
            "groups": groups,
            "layers": entries,
            "pairwise_similarity": {name: similarity.tolist()},
            "regime_labels": list(regimes),
            "target_proto_count": int(target_prototypes),
            "actual_proto_count": int(len(clusters)),
            "retargeted_by": Path(__file__).name,
            "grouping_analysis": {
                "algorithm": "constrained_complete_link_agglomeration",
                "target_prototypes": int(target_prototypes),
                "max_group_size": int(max_group_size),
                "max_layer_span": int(max_layer_span),
                "pinned_private_layers": [int(item) for item in pinned_layers],
                "same_regime_required": True,
                "medoid_note": (
                    "medoid_layer minimizes measured directed intervention cost, but the current compression "
                    "pipeline uses its own proto_seed_strategy unless explicitly extended to consume this field"
                ),
            },
        }
    )
    return policy


def validate_policy(policy: Mapping[str, Any], layer_count: int, target_prototypes: int) -> None:
    entries = policy.get("layers", [])
    if not isinstance(entries, list) or len(entries) != int(layer_count):
        raise ValueError(f"policy must contain exactly {layer_count} layer entries")
    ids = [int(item.get("layer_id", -1)) for item in entries if isinstance(item, dict)]
    if sorted(ids) != list(range(layer_count)):
        raise ValueError("policy layer IDs must be unique and cover every decoder layer")
    groups = policy.get("groups", [])
    if not isinstance(groups, list):
        raise ValueError("policy groups must be a list")
    group_members: Dict[int, set[int]] = {}
    for expected_group_id, group in enumerate(groups):
        if not isinstance(group, dict):
            raise ValueError("every policy group must be an object")
        group_id = int(group.get("group_id", -1))
        if group_id != expected_group_id:
            raise ValueError("policy group IDs must be contiguous from zero")
        members = {int(item) for item in group.get("layers", [])}
        if len(members) < 2:
            raise ValueError("shared groups must contain at least two layers")
        group_members[group_id] = members
    core_ids: set[Tuple[str, int]] = set()
    for entry in entries:
        layer_id = int(entry["layer_id"])
        group_id = int(entry.get("group_id", -1))
        private = bool(entry.get("private_core", group_id < 0))
        if group_id >= 0:
            if private or layer_id not in group_members.get(group_id, set()):
                raise ValueError(f"inconsistent group mapping for layer {layer_id}")
            core_ids.add(("shared", group_id))
        else:
            if not private:
                raise ValueError(f"private layer {layer_id} has private_core=False")
            core_ids.add(("private", layer_id))
    if len(core_ids) != int(target_prototypes):
        raise ValueError(f"policy has {len(core_ids)} unique cores, expected {target_prototypes}")


def load_feature_shards(obs_dir: Path, run_config: Mapping[str, Any]) -> Dict[str, List[torch.Tensor]]:
    world_size = int(run_config.get("world_size", 1))
    shards = []
    for rank in range(world_size):
        path = obs_dir / "shards" / f"features_rank{rank}.pt"
        if not path.exists():
            raise FileNotFoundError(path)
        shards.append(load_torch(path))
    kinds = [kind for kind in ("input", "delta", "block_delta") if all(kind in shard for shard in shards)]
    layer_count = int(run_config["layer_count"])
    merged: Dict[str, List[torch.Tensor]] = {kind: [] for kind in kinds}
    for kind in kinds:
        for layer_id in range(layer_count):
            merged[kind].append(torch.cat([shard[kind][layer_id] for shard in shards], dim=0).float())
    return merged


def feature_moment_views(features: Mapping[str, Sequence[torch.Tensor]]) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, Any]]]:
    views: Dict[str, torch.Tensor] = {}
    resultant_rows: List[Dict[str, Any]] = []
    kind_name = {
        "post_attn_residual": "legacy_post_attn_residual",
        "input": "input",
        "delta": "ffn_output",
        "block_delta": "block_delta",
    }
    for kind, layers in features.items():
        prefix = kind_name[kind]
        means = torch.stack([item.mean(dim=0) for item in layers])
        mean_cosine = F.normalize(means, dim=-1) @ F.normalize(means, dim=-1).T
        views[f"{prefix}_mean_cosine"] = mean_cosine.to(dtype=torch.float64)
        for layer_id, item in enumerate(layers):
            mean_norm = float(torch.linalg.vector_norm(item.mean(dim=0)))
            average_norm = float(torch.linalg.vector_norm(item, dim=-1).mean())
            resultant_rows.append(
                {
                    "view": prefix,
                    "layer": layer_id,
                    "mean_resultant_length": mean_norm / max(EPS, average_norm),
                    "sample_count": int(item.size(0)),
                }
            )
        layer_count = len(layers)
        nmse_similarity = torch.empty(layer_count, layer_count, dtype=torch.float64)
        aligned_similarity = torch.empty(layer_count, layer_count, dtype=torch.float64)
        for i in range(layer_count):
            for j in range(layer_count):
                n = min(int(layers[i].size(0)), int(layers[j].size(0)))
                x = layers[i][:n]
                y = layers[j][:n]
                x2 = (x * x).sum(dim=-1).mean()
                y2 = (y * y).sum(dim=-1).mean()
                mse = ((x - y) ** 2).sum(dim=-1).mean()
                nmse = mse / (0.5 * (x2 + y2) + EPS)
                xy = (x * y).sum()
                alpha_xy = xy / ((x * x).sum() + EPS)
                alpha_yx = xy / ((y * y).sum() + EPS)
                error_xy = (((alpha_xy * x - y) ** 2).sum(dim=-1).mean()) / (y2 + EPS)
                error_yx = (((alpha_yx * y - x) ** 2).sum(dim=-1).mean()) / (x2 + EPS)
                nmse_similarity[i, j] = -float(nmse)
                aligned_similarity[i, j] = -0.5 * float(error_xy + error_yx)
        views[f"{prefix}_paired_nmse_similarity"] = nmse_similarity
        views[f"{prefix}_scale_aligned_nmse_similarity"] = aligned_similarity
    return views, resultant_rows


def fmt(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def render_report(
    output_path: Path,
    obs_dir: Path,
    base_policy_path: Path,
    run_config: Mapping[str, Any],
    predictive_rows: Sequence[Mapping[str, Any]],
    cluster_rows: Sequence[Mapping[str, Any]],
    policy_paths: Mapping[str, str],
    legacy_warning: bool,
    pinned_layers: Sequence[int],
    input_gate_quantile: Optional[float],
    input_gate_threshold: Optional[float],
) -> None:
    same = {str(row["view"]): row for row in predictive_rows if row["scope"] == "same_regime"}
    cluster = {str(row["view"]): row for row in cluster_rows}
    primary_views = [
        "input_paired_cosine",
        "ffn_output_paired_cosine",
        "input_cka",
        "ffn_output_cka",
        "joint_input_output_cosine",
        "input_functional_hybrid",
        "functional_oracle",
    ]
    held_out = str(run_config.get("data_split_mode", "")) in {
        "separate_sources",
        "disjoint_halves_from_same_source",
    }
    lines = [
        "# FFN input vs. FFN output 分群分析",
        "",
        "## 結論先行",
        "",
        "FFN input 不能被說成「資訊量」。它描述的是 shared FFN 會被呼叫的 input domain；真正的 information/transport 變數是 FFN output，也就是 `Δh_FFN`。Native output 相似度則仍不等於可互換性，因為它比較的是 `F_i(x_i)` 與 `F_j(x_j)`，而共享真正要求的是 `F_i(x_j)` 與 `F_j(x_j)` 接近。",
        "",
        "因此最完整的設計不是硬選 input 或 output，而是：",
        "",
        "> input = domain-compatibility gate；cross-FFN intervention/FAD error = functional criterion；`Δh_FFN` = 訓練時要保存的 transport。",
        "",
        "## 定義",
        "",
        "- `u_l = h_l + Attn_l(Norm1(h_l))`",
        "- `x_l = Norm2_l(u_l)`：真正的 FFN input",
        "- `Δh_FFN,l = F_l(x_l)`：FFN output；兩者是同一個 tensor，不是兩個 ablation",
        "- `Δh_block,l = h_(l+1) - h_l`：attention + FFN 的完整 block update",
        "",
        "## 現有 observation 的預測能力",
        "",
        "數值是 similarity 對雙向最壞 intervention cost 的 Spearman 相關；越負代表越能找出安全替換。所有 policy 仍限制在相同 depth regime。",
        "",
        "| view | same-regime ρ | Kendall τ | safe-pair AUROC | precision@M | QAP p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in primary_views:
        row = same.get(name)
        if row is None:
            continue
        lines.append(
            "| " + name + " | "
            + " | ".join(
                [
                    fmt(row["spearman_similarity_vs_cost"]),
                    fmt(row["kendall_tau_b_similarity_vs_cost"]),
                    fmt(row["safe_pair_auroc"]),
                    fmt(row["safe_pair_precision_at_m"]),
                    fmt(row["qap_one_sided_p"]),
                ]
            )
            + " |"
        )
    input_cosine_row = same.get("input_paired_cosine", {})
    output_cosine_row = same.get("ffn_output_paired_cosine", {})
    input_cka_row = same.get("input_cka", {})
    output_cka_row = same.get("ffn_output_cka", {})
    lines.extend(
        [
            "",
            "### 如何解讀",
            "",
            f"- 這份 pilot 的 raw cosine 結果是 input ρ={fmt(input_cosine_row.get('spearman_similarity_vs_cost'))}、FFN output ρ={fmt(output_cosine_row.get('spearman_similarity_vs_cost'))}；目前證據支持 input cosine，而不支持 output cosine。",
            f"- 但 CKA 結果是 input ρ={fmt(input_cka_row.get('spearman_similarity_vs_cost'))}、FFN output ρ={fmt(output_cka_row.get('spearman_similarity_vs_cost'))}；output 的 distributional geometry 並沒有失效，甚至在 same-regime scope 略強。",
            "- 若 input paired cosine 明顯比 output paired cosine 更負，只能支持「input cosine 較能預測 operator swap risk」，不能推出 input 含有較多 information。",
            "- 若 output CKA 優於 output cosine，差異可能來自 aggregation/metric，而不是 input/output representation 本身。",
            "- functional oracle 使用 intervention label 本身，只是診斷上界，不能與 deployable heuristic 當成同等證據。",
            "",
            "## 固定壓縮預算的 complete-link 分群",
            "",
            "所有方法使用相同 prototype count；complete-link 每次最大化新群組最差 pair similarity，避免 threshold connected-components 的 chaining 問題。Medoid cost 是把該 medoid MLP 放到群內其他 layer 的單層 intervention 診斷。",
            f"本次所有 views 共用的 structural private pins：`{list(pinned_layers)}`。空集合表示未固定任何層；這個選擇必須事先指定，不能依結果替單一 view 調整。",
            "",
            "| view | prototypes | saved MLPs | groups | max pair cost | mean medoid replacement cost | max medoid replacement cost |",
            "|---|---:|---:|---|---:|---:|---:|",
        ]
    )
    cluster_view_order = ["legacy_current_policy", *primary_views]
    if input_gate_threshold is not None:
        cluster_view_order.append("input_gate_functional")
    for name in cluster_view_order:
        row = cluster.get(name)
        if row is None:
            continue
        lines.append(
            f"| {name} | {row['prototype_count']} | {row['saved_mlp_count']} | "
            f"`{row['groups']}` | {fmt(row['within_symmetric_cost_max'])} | "
            f"{fmt(row['medoid_replacement_cost_mean'])} | {fmt(row['medoid_replacement_cost_max'])} |"
        )
    input_cluster = cluster.get("input_paired_cosine", {})
    output_cka_cluster = cluster.get("ffn_output_cka", {})
    if not pinned_layers and input_cluster and float(input_cluster.get("within_symmetric_cost_max", 0.0)) > 1.0:
        lines.extend(
            [
                "",
                f"未固定最後一層時，input policy 的群內最壞替換成本達 {fmt(input_cluster.get('within_symmetric_cost_max'))}；主要反例是高 input similarity 的 layers 26–27。這直接證明 input-only 不是 safety certificate。",
            ]
        )
    if pinned_layers and input_cluster and output_cka_cluster:
        lines.extend(
            [
                "",
                f"固定層 `{list(pinned_layers)}` 後，input policy 的最壞 pair cost 為 {fmt(input_cluster.get('within_symmetric_cost_max'))}，output-CKA 為 {fmt(output_cka_cluster.get('within_symmetric_cost_max'))}。這個敏感度差異表示 structural constraints 與 representation choice 必須分開報告。",
            ]
        )
    lines.extend(
        [
            "",
            "## 為什麼 input 有合理性",
            "",
            "共享的是 operator，不只是一批輸出向量。同一個 core 之後會在多層的 `x_l` 上被查詢；input similarity 控制的是 covariate/domain shift。Output-only 可能出現兩個不同函數在各自輸入上碰巧產生相同輸出，但互換後失敗。形式上，對代表點 `x_i, x_j`：",
            "",
            "`||F_i(x_j)-F_j(x_j)|| <= L_i ||x_j-x_i|| + ||F_i(x_i)-F_j(x_j)||`",
            "",
            "第一項需要 input compatibility，第二項才是 native transport similarity；兩者都不是完整 replacement error。",
            "",
            "## 為什麼不能只靠 input",
            "",
            "若 `x_i=x_j`，但 `F_i(x)=x`、`F_j(x)=-x`，input 完全相同仍不可共享。因此 input-only 最多是 gate，不是 functional equivalence proof。真正直接的目標是：",
            "",
            "`D_func(i,j)=0.5 E_{x~P_i} d(F_i(x),F_j(x)) + 0.5 E_{x~P_j} d(F_i(x),F_j(x))`",
            "",
            "其中 `d` 最好採用你的 FAD-whitened `Δh_FFN` error，便可直接接回 information-energy 敘事。",
            "",
            "## 實作與證據限制",
            "",
            f"- Observation: `{obs_dir}`",
            f"- Base policy: `{base_policy_path}`",
            f"- activation batches: {run_config.get('activation_batches', 'NA')}；intervention batches: {run_config.get('intervention_batches', 'NA')}；seed: {run_config.get('seed', 'NA')}",
            (
                "- 這份 observation 的 feature 與 intervention 使用互斥 calibration subsets；policy selection 沒有使用 validation 或 downstream test。"
                if held_out
                else "- 這份 observation 無法確認 feature/intervention 為 held-out split，因此只能視為 pilot evidence。"
            ),
            "- 單層 intervention cost 不等於同時合併整群後的真實 cost；正式實驗仍需 zero-shot multi-merge 與 recovery training。",
        ]
    )
    if input_gate_threshold is not None:
        lines.extend(
            [
                f"- `input_gate_functional` 使用 hard gate：同 regime FFN-input cosine 必須 >= {fmt(input_gate_threshold, 6)}（same-regime pair 的第 {fmt(100.0 * float(input_gate_quantile or 0.0), 1)} percentile），通過後才依 held-out intervention cost 做 complete-link 合併。",
                "- gate 若無法在其他約束下精確達到 19 cores，分析程式會直接失敗，不會自動放寬 threshold。",
            ]
        )
    if legacy_warning:
        lines.extend(
            [
                "- 此 observation 由舊 collector 產生：多 GPU ranks 重複相同 activation blocks；表面 sample count 被放大，但 point correlation 因重複相同行通常不變。舊程式在 subsampling 時也可能讓各層 token index 不一致。當前這次 `max_positions_per_batch == batch_size * block_size`，因此每層實際用了完整 token 順序，paired 指標仍對齊。",
                "- collector 已修正為跨層共用 token positions、跨 rank 使用 disjoint blocks，並新增 full-block delta。正式重跑後應以新 artifact 取代此 pilot。",
            ]
        )
    lines.extend(
        [
            "- 主 pipeline 論文寫 normalized FFN input `x_l`，但目前 atlas hook 抓的是 `post_attention_layernorm` 之前的 `u_l`。在正式 ablation 前必須改 hook 或改論文名稱。",
            "- 目前 pipeline 用 threshold graph 的 connected components；同組端點可能低於 threshold。此腳本輸出的 matched policies 使用 complete-link，移除此 confound。",
            "",
            "## 重現這份離線分析",
            "",
            "```bash",
            "python reproduction/fad/core/analyze_ffn_grouping_views.py \\",
            f"  --obs-dir {obs_dir} \\",
            f"  --base-policy {base_policy_path} \\",
            f"  --output-dir {output_path.parent} \\",
            "  --target-prototypes 19 --max-group-size 8 \\",
            ("  --pinned-layers " + ",".join(str(item) for item in pinned_layers) + " \\") if pinned_layers else "  --pinned-layers '' \\",
            (
                f"  --input-gate-quantile {float(input_gate_quantile):.6g} \\"
                if input_gate_quantile is not None
                else ""
            ),
            "  --qap-permutations 1000 --seed 44",
            "```",
            "",
            "## 決策準則",
            "",
            "1. 先用 held-out counterfactual cost 比較 input/output/joint 的 predictive correlation 與 safe-merge retrieval。",
            "2. 固定 28→19（再加 15%/25% budgets）、相同 adapter rank、參數量、資料順序與 3–5 seeds。",
            "3. 同時報告 merge 後訓練前的 ΔNLL/KL/FAD error，以及相同 recovery 後的 macro accuracy。",
            "4. 若 output 在至少兩個 budgets、zero-shot 與 recovery 指標都穩定勝出，就改 output-aware grouping。若 input 與 output 各有增量預測力，採 hybrid。",
            "5. 若 input non-inferior，論文應說它是 domain prior；information preservation 仍歸因於 `Δh_FFN`/FAD，不能把 input cosine 稱為 information measure。",
            "",
            "## 產生的 matched-budget policies",
            "",
        ]
    )
    for name, path in policy_paths.items():
        lines.append(f"- `{name}`: `{path}`")
    lines.extend(
        [
            "",
            "使用方式：將任一 policy 傳給 compression stage 的 `--sharing_policy_path`，或設定 runner 的 `SHARING_POLICY_PATH`；每個 policy 必須使用不同 `OUT_ROOT`，其餘超參數完全相同。",
            "",
            "注意：policy 中的 `medoid_layer` 是分析結果；目前 compression code 仍依 `proto_seed_strategy`/atlas signature 選 seed，尚不會直接消費這個欄位。若要做嚴格 grouping-only ablation，應固定所有 views 共用同一個 held-out functional medoid rule。",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Matched-budget FFN grouping-view analysis")
    parser.add_argument("--obs-dir", required=True, help="Output directory of ffn_functional_redundancy_ddp.py")
    parser.add_argument("--base-policy", required=True, help="Existing sharing_policy.json used as schema/regime source")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-prototypes", type=int, default=19)
    parser.add_argument("--max-group-size", type=int, default=8, help="0 disables the size constraint")
    parser.add_argument("--max-layer-span", type=int, default=0, help="0 disables the span constraint")
    parser.add_argument(
        "--require-contiguous-groups",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require every shared group to be a contiguous layer interval.",
    )
    parser.add_argument(
        "--input-gate-quantile",
        type=float,
        default=None,
        help=(
            "If set in [0,1], additionally emit input_gate_functional: every cross-pair in a merge "
            "must pass this same-regime FFN-input cosine quantile, then complete-link ranks merges "
            "only by held-out functional intervention similarity. The gate is never auto-relaxed."
        ),
    )
    parser.add_argument(
        "--pinned-layers",
        default="",
        help="Comma-separated layers forced private; accepts 'first' and 'last' (for sensitivity analysis)",
    )
    parser.add_argument("--qap-permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--recompute-feature-moments",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Load feature shards and add mean-cosine/NMSE views; uses substantially more RAM/time",
    )
    return parser.parse_args()


def parse_pinned_layers(specification: str, layer_count: int) -> List[int]:
    pinned: set[int] = set()
    for raw_item in str(specification).split(","):
        item = raw_item.strip().lower()
        if not item:
            continue
        if item == "first":
            value = 0
        elif item == "last":
            value = int(layer_count) - 1
        else:
            value = int(item)
        if value < 0:
            value += int(layer_count)
        if value < 0 or value >= int(layer_count):
            raise ValueError(f"pinned layer out of range: {raw_item}")
        pinned.add(value)
    return sorted(pinned)


def main() -> None:
    args = parse_args()
    obs_dir = Path(args.obs_dir).expanduser().resolve()
    base_policy_path = Path(args.base_policy).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base_policy = load_json(base_policy_path)
    run_config_path = obs_dir / "run_config.json"
    run_config = load_json(run_config_path) if run_config_path.exists() else {}

    directed_cost = load_matrix(obs_dir, "merge_cost")
    if directed_cost is None:
        raise FileNotFoundError(obs_dir / "merge_cost.pt")
    layer_count = int(directed_cost.size(0))
    pinned_layers = parse_pinned_layers(str(args.pinned_layers), layer_count)
    regimes = regimes_from_policy(base_policy, layer_count)
    symmetric_cost = symmetrize_cost(directed_cost)
    views: Dict[str, torch.Tensor] = {}
    artifact_mapping = {
        "legacy_post_attn_residual_paired_cosine": "post_attn_residual_cosine",
        "legacy_post_attn_residual_cka": "post_attn_residual_cka",
        "input_paired_cosine": "input_cosine",
        "ffn_output_paired_cosine": "delta_cosine",
        "input_cka": "input_cka",
        "ffn_output_cka": "delta_cka",
        "block_delta_paired_cosine": "block_delta_cosine",
        "block_delta_cka": "block_delta_cka",
    }
    for view_name, artifact_name in artifact_mapping.items():
        matrix = load_matrix(obs_dir, artifact_name)
        if matrix is not None:
            views[view_name] = validate_similarity(matrix, layer_count, view_name)

    resultant_rows: List[Dict[str, Any]] = []
    if bool(args.recompute_feature_moments):
        if not run_config:
            raise RuntimeError("run_config.json is required to load feature shards")
        feature_views, resultant_rows = feature_moment_views(load_feature_shards(obs_dir, run_config))
        for name, matrix in feature_views.items():
            views[name] = validate_similarity(matrix, layer_count, name)
        write_rows(output_dir / "mean_resultant_length.csv", resultant_rows)

    if "input_paired_cosine" not in views or "ffn_output_paired_cosine" not in views:
        raise RuntimeError("input_cosine.pt and delta_cosine.pt are required")
    functional_similarity = -symmetric_cost
    off_diagonal = ~torch.eye(layer_count, dtype=torch.bool)
    finite_functional = functional_similarity[off_diagonal & torch.isfinite(functional_similarity)]
    if int(finite_functional.numel()) <= 0:
        raise RuntimeError("no finite functional intervention similarities were found")
    functional_similarity.fill_diagonal_(float(finite_functional.max()))
    views["functional_oracle"] = functional_similarity
    input_rank = rank_normalize_matrix(views["input_paired_cosine"], regimes)
    output_rank = rank_normalize_matrix(views["ffn_output_paired_cosine"], regimes)
    functional_rank = rank_normalize_matrix(functional_similarity, regimes)
    views["joint_input_output_cosine"] = torch.minimum(input_rank, output_rank)
    views["input_functional_hybrid"] = torch.minimum(input_rank, functional_rank)

    input_gate_quantile: Optional[float] = None
    input_gate_threshold: Optional[float] = None
    input_gate_mask: Optional[torch.Tensor] = None
    if args.input_gate_quantile is not None:
        input_gate_quantile = float(args.input_gate_quantile)
        if not 0.0 <= input_gate_quantile <= 1.0:
            raise ValueError("--input-gate-quantile must be in [0,1]")
        gate_pairs = pair_indices(layer_count, regimes, "same_regime")
        gate_values = torch.tensor(
            [float(views["input_paired_cosine"][i, j]) for i, j in gate_pairs],
            dtype=torch.float64,
        )
        gate_values = gate_values[torch.isfinite(gate_values)]
        if int(gate_values.numel()) <= 0:
            raise RuntimeError("no finite same-regime FFN-input cosine values for the hard gate")
        input_gate_threshold = float(torch.quantile(gate_values, input_gate_quantile))
        input_gate_mask = (views["input_paired_cosine"] >= input_gate_threshold).to(dtype=torch.float64)
        input_gate_mask.fill_diagonal_(1.0)

    matrices_dir = output_dir / "matrices"
    matrices_dir.mkdir(parents=True, exist_ok=True)
    for name, matrix in views.items():
        torch.save(matrix.to(dtype=torch.float32), matrices_dir / f"{name}.pt")
        write_matrix_csv(matrices_dir / f"{name}.csv", matrix)
        plot_heatmap(
            matrices_dir / f"{name}.png",
            matrix,
            name,
            cmap="magma_r" if name == "functional_oracle" else "viridis",
        )
    if input_gate_mask is not None and input_gate_threshold is not None:
        torch.save(input_gate_mask.to(dtype=torch.bool), matrices_dir / "input_gate_mask.pt")
        write_matrix_csv(matrices_dir / "input_gate_mask.csv", input_gate_mask)
        save_json(
            matrices_dir / "input_gate.json",
            {
                "source_view": "input_paired_cosine",
                "scope": "same_regime",
                "quantile": input_gate_quantile,
                "threshold": input_gate_threshold,
                "comparison": "cosine >= threshold",
                "auto_relax": False,
            },
        )
    torch.save(directed_cost.to(dtype=torch.float32), matrices_dir / "directed_intervention_cost.pt")
    torch.save(symmetric_cost.to(dtype=torch.float32), matrices_dir / "symmetric_max_intervention_cost.pt")
    write_matrix_csv(matrices_dir / "symmetric_max_intervention_cost.csv", symmetric_cost)
    plot_heatmap(matrices_dir / "symmetric_max_intervention_cost.png", symmetric_cost, "symmetric max intervention cost", "magma")

    safest_count = int(layer_count - int(args.target_prototypes))
    predictive_rows: List[Dict[str, Any]] = []
    scopes = ("all", "same_regime", "adjacent", "same_regime_adjacent")
    for view_offset, (name, matrix) in enumerate(views.items()):
        for scope in scopes:
            predictive_rows.append(
                predictive_row(
                    name,
                    matrix,
                    symmetric_cost,
                    regimes,
                    scope,
                    safest_count=safest_count,
                    qap_permutations=int(args.qap_permutations),
                    seed=int(args.seed) + 1009 * view_offset,
                )
            )
    write_rows(output_dir / "predictive_metrics.csv", predictive_rows)

    pair_rows: List[Dict[str, Any]] = []
    for i, j in pair_indices(layer_count, regimes, "all"):
        row: Dict[str, Any] = {
            "layer_i": i,
            "layer_j": j,
            "regime_i": regimes[i],
            "regime_j": regimes[j],
            "same_regime": regimes[i] == regimes[j],
            "depth_gap": abs(i - j),
            "symmetric_max_intervention_cost": float(symmetric_cost[i, j]),
            "cost_i_source_to_j_target": float(directed_cost[j, i]),
            "cost_j_source_to_i_target": float(directed_cost[i, j]),
        }
        for name, matrix in views.items():
            row[name] = float(matrix[i, j])
        pair_rows.append(row)
    write_rows(output_dir / "layer_pair_metrics.csv", pair_rows)

    cluster_summaries: List[Dict[str, Any]] = []
    group_rows: List[Dict[str, Any]] = []
    legacy_clusters = clusters_from_policy(base_policy, layer_count)
    legacy_summary, legacy_groups = summarize_clusters(
        "legacy_current_policy",
        legacy_clusters,
        views["input_paired_cosine"],
        directed_cost,
        symmetric_cost,
        regimes,
    )
    cluster_summaries.append(legacy_summary)
    group_rows.extend(legacy_groups)

    policy_paths: Dict[str, str] = {}
    policies_dir = output_dir / "policies"
    cluster_specs: List[Tuple[str, torch.Tensor, Optional[torch.Tensor], Optional[float]]] = [
        (name, matrix, None, None) for name, matrix in views.items()
    ]
    if input_gate_threshold is not None:
        cluster_specs.append(
            (
                "input_gate_functional",
                functional_similarity,
                views["input_paired_cosine"],
                input_gate_threshold,
            )
        )
    for name, matrix, gate_matrix, gate_threshold in cluster_specs:
        clusters = complete_link_clusters(
            matrix,
            regimes,
            target_clusters=int(args.target_prototypes),
            max_group_size=int(args.max_group_size),
            max_layer_span=int(args.max_layer_span),
            pinned_layers=pinned_layers,
            gate_similarity=gate_matrix,
            gate_threshold=gate_threshold,
            require_contiguous_groups=bool(args.require_contiguous_groups),
        )
        summary, rows = summarize_clusters(name, clusters, matrix, directed_cost, symmetric_cost, regimes)
        cluster_summaries.append(summary)
        group_rows.extend(rows)
        policy = build_policy(
            base_policy,
            name,
            clusters,
            matrix,
            directed_cost,
            symmetric_cost,
            regimes,
            target_prototypes=int(args.target_prototypes),
            max_group_size=int(args.max_group_size),
            max_layer_span=int(args.max_layer_span),
            pinned_layers=pinned_layers,
        )
        policy["grouping_analysis"]["contiguous_groups_required"] = bool(
            args.require_contiguous_groups
        )
        if gate_matrix is not None and gate_threshold is not None:
            policy["input_domain_gate"] = {
                "hard_gate": True,
                "source_view": "normalized_ffn_input_paired_cosine",
                "scope": "same_regime_complete_link_cross_pairs",
                "quantile": input_gate_quantile,
                "threshold": float(gate_threshold),
                "comparison": "cosine >= threshold",
                "auto_relax": False,
            }
            policy["functional_criterion"] = {
                "source": "held_out_cross_mlp_intervention",
                "directed_cost": "delta_nll + kl_weight * KL(baseline || intervention)",
                "bidirectional_reduction": "maximum",
                "group_linkage": "complete_link",
            }
            policy["pairwise_similarity"]["input_paired_cosine_gate"] = gate_matrix.tolist()
            policy["grouping_analysis"].update(
                {
                    "input_gate_quantile": input_gate_quantile,
                    "input_gate_threshold": float(gate_threshold),
                    "input_gate_auto_relax": False,
                    "ranking_after_gate": "negative_symmetric_max_intervention_cost",
                }
            )
            for group in policy.get("groups", []):
                members = [int(item) for item in group.get("layers", [])]
                pairs = [(a, b) for idx, a in enumerate(members) for b in members[idx + 1 :]]
                gate_values = [float(gate_matrix[a, b]) for a, b in pairs]
                group["min_input_gate_cosine"] = min(gate_values)
                group["mean_input_gate_cosine"] = sum(gate_values) / len(gate_values)
                if min(gate_values) < float(gate_threshold):
                    raise AssertionError(f"group {members} violates the hard FFN-input gate")
        validate_policy(policy, layer_count, int(args.target_prototypes))
        policy_path = policies_dir / f"sharing_policy_{name}.json"
        save_json(policy_path, policy)
        policy_paths[name] = str(policy_path)
    write_rows(output_dir / "cluster_summary.csv", cluster_summaries)
    write_rows(output_dir / "group_details.csv", group_rows)

    manifest = {
        "instruction": (
            "Run every policy with a distinct OUT_ROOT while keeping model, data order, adapter rank, "
            "optimizer, steps, checkpoint rule, and seeds identical. Pass policy via SHARING_POLICY_PATH "
            "or --sharing_policy_path. functional_oracle, input_functional_hybrid, and "
            "input_gate_functional use held-out intervention labels and are calibrated policies/diagnostics, "
            "not deployable unsupervised baselines."
        ),
        "target_prototypes": int(args.target_prototypes),
        "pinned_private_layers": pinned_layers,
        "saved_mlp_count": safest_count,
        "runs": [
            {
                "view": name,
                "sharing_policy_path": path,
                "suggested_environment": {
                    "OUT_ROOT": f"<OUT_ROOT>/{name}_seed<SEED>",
                    "SHARING_POLICY_PATH": path,
                    "FUNCTIONAL_POLICY_ENABLE": "False",
                },
                "seeds": [42, 43, 44],
            }
            for name, path in policy_paths.items()
        ],
    }
    save_json(output_dir / "training_manifest.json", manifest)

    feature_summary = load_json(obs_dir / "feature_summary.json") if (obs_dir / "feature_summary.json").exists() else {}
    legacy_warning = "feature_pairing" not in feature_summary and int(run_config.get("world_size", 1)) > 1
    report_path = output_dir / "REPORT.md"
    render_report(
        report_path,
        obs_dir,
        base_policy_path,
        run_config,
        predictive_rows,
        cluster_summaries,
        policy_paths,
        legacy_warning,
        pinned_layers,
        input_gate_quantile,
        input_gate_threshold,
    )
    held_out = str(run_config.get("data_split_mode", "")) in {
        "separate_sources",
        "disjoint_halves_from_same_source",
    }
    summary_payload = {
        "obs_dir": str(obs_dir),
        "base_policy": str(base_policy_path),
        "layer_count": layer_count,
        "regimes": list(regimes),
        "target_prototypes": int(args.target_prototypes),
        "pinned_private_layers": pinned_layers,
        "views": list(views),
        "input_domain_gate": (
            {
                "source_view": "input_paired_cosine",
                "quantile": input_gate_quantile,
                "threshold": input_gate_threshold,
                "auto_relax": False,
            }
            if input_gate_threshold is not None
            else None
        ),
        "predictive_metrics": predictive_rows,
        "cluster_summaries": cluster_summaries,
        "policy_paths": policy_paths,
        "report": str(report_path),
        "pilot_evidence_warnings": {
            "legacy_feature_collector": legacy_warning,
            "feature_and_intervention_not_held_out": not held_out,
            "single_run_only": True,
            "block_delta_available": "block_delta_paired_cosine" in views,
        },
    }
    save_json(output_dir / "analysis_summary.json", summary_payload)
    print(f"[FFN-Grouping] report: {report_path}")
    print(f"[FFN-Grouping] policies: {policies_dir}")


if __name__ == "__main__":
    main()
