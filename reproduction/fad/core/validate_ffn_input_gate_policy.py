#!/usr/bin/env python3
"""Fail-fast contract checks for the calibrated FFN input-gate policy."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping


def load_object(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def resolved(value: Any) -> str:
    return str(Path(str(value)).expanduser().resolve())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--run-config", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--expected-teacher", required=True)
    parser.add_argument(
        "--expected-tokenizer",
        default="",
        help="Expected tokenizer path; defaults to expected teacher for compatibility.",
    )
    parser.add_argument("--expected-feature-data", required=True)
    parser.add_argument("--expected-intervention-data", required=True)
    parser.add_argument("--target-prototypes", type=int, default=19)
    parser.add_argument("--expected-layer-count", type=int, default=28)
    parser.add_argument("--pinned-layer", type=int, default=27)
    parser.add_argument(
        "--pinned-layers",
        default="",
        help="Optional comma-separated private layers; overrides --pinned-layer",
    )
    parser.add_argument("--expected-shared-layer-min", type=int, default=-1)
    parser.add_argument("--expected-shared-layer-max", type=int, default=-1)
    parser.add_argument("--require-contiguous-groups", action="store_true")
    parser.add_argument("--expected-gate-quantile", type=float, default=0.70)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    policy_path = Path(args.policy).expanduser().resolve()
    run_config_path = Path(args.run_config).expanduser().resolve()
    split_manifest_path = Path(args.split_manifest).expanduser().resolve()
    policy = load_object(policy_path)
    config = load_object(run_config_path)
    split = load_object(split_manifest_path)
    checks: Dict[str, bool] = {}

    expected_tokenizer = (
        args.expected_tokenizer
        if str(args.expected_tokenizer).strip()
        else args.expected_teacher
    )
    checks["exact_teacher"] = resolved(config.get("model_name_or_path")) == resolved(args.expected_teacher)
    checks["exact_teacher_tokenizer"] = resolved(config.get("tokenizer_name_or_path")) == resolved(
        expected_tokenizer
    )
    checks["separate_sources"] = str(config.get("data_split_mode")) == "separate_sources"
    checks["feature_path"] = resolved(config.get("feature_data_path_resolved")) == resolved(
        args.expected_feature_data
    )
    checks["intervention_path"] = resolved(config.get("intervention_data_path_resolved")) == resolved(
        args.expected_intervention_data
    )
    checks["record_hash_disjoint"] = int(split.get("feature_intervention_hash_overlap_count", -1)) == 0
    checks["split_assertion"] = bool(
        split.get("assertions", {}).get("feature_intervention_rendered_text_disjoint", False)
    )
    checks["collector_layer_count"] = int(config.get("layer_count", -1)) == int(args.expected_layer_count)

    entries = policy.get("layers", [])
    groups = policy.get("groups", [])
    require(isinstance(entries, list), "policy.layers must be a list")
    require(isinstance(groups, list), "policy.groups must be a list")
    checks["layer_entry_count"] = len(entries) == int(args.expected_layer_count)
    layer_ids = [int(item.get("layer_id", -1)) for item in entries if isinstance(item, dict)]
    checks["layer_ids_complete"] = sorted(layer_ids) == list(range(int(args.expected_layer_count)))
    checks["group_ids_contiguous"] = [int(group.get("group_id", -1)) for group in groups] == list(
        range(len(groups))
    )

    member_by_group: Dict[int, set[int]] = {}
    same_regime = True
    medoid_valid = True
    finite_costs = True
    gate_respected = True
    gate = policy.get("input_domain_gate", {})
    gate_threshold = float(gate.get("threshold", float("nan")))
    groups_contiguous = True
    shared_layers_in_range = True
    for group in groups:
        group_id = int(group.get("group_id", -1))
        members = {int(item) for item in group.get("layers", [])}
        member_by_group[group_id] = members
        ordered_members = sorted(members)
        groups_contiguous = groups_contiguous and ordered_members == list(
            range(min(ordered_members), max(ordered_members) + 1)
        )
        if int(args.expected_shared_layer_min) >= 0:
            shared_layers_in_range = shared_layers_in_range and all(
                layer_id >= int(args.expected_shared_layer_min) for layer_id in members
            )
        if int(args.expected_shared_layer_max) >= 0:
            shared_layers_in_range = shared_layers_in_range and all(
                layer_id <= int(args.expected_shared_layer_max) for layer_id in members
            )
        regimes = {
            str(entries[layer_id].get("regime"))
            for layer_id in members
            if 0 <= layer_id < len(entries)
        }
        same_regime = same_regime and len(regimes) == 1
        medoid_valid = medoid_valid and int(group.get("medoid_layer", -1)) in members
        for key in (
            "mean_symmetric_intervention_cost",
            "max_symmetric_intervention_cost",
            "medoid_to_member_cost_mean",
            "medoid_to_member_cost_max",
            "min_input_gate_cosine",
        ):
            finite_costs = finite_costs and math.isfinite(float(group.get(key, float("nan"))))
        gate_respected = gate_respected and float(group.get("min_input_gate_cosine", -float("inf"))) >= gate_threshold

    unique_cores: set[tuple[str, int]] = set()
    mapping_consistent = True
    for entry in entries:
        layer_id = int(entry.get("layer_id", -1))
        group_id = int(entry.get("group_id", -1))
        private = bool(entry.get("private_core", group_id < 0))
        if group_id >= 0:
            mapping_consistent = mapping_consistent and not private and layer_id in member_by_group.get(group_id, set())
            unique_cores.add(("group", group_id))
        else:
            mapping_consistent = mapping_consistent and private
            unique_cores.add(("private", layer_id))

    checks["source_view"] = str(policy.get("source_view")) == "input_gate_functional"
    checks["hard_gate"] = bool(gate.get("hard_gate", False)) and not bool(gate.get("auto_relax", True))
    checks["gate_quantile"] = math.isclose(
        float(gate.get("quantile", float("nan"))),
        float(args.expected_gate_quantile),
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    checks["gate_threshold_finite"] = math.isfinite(gate_threshold)
    checks["gate_respected"] = gate_respected
    checks["same_regime_groups"] = same_regime
    checks["medoids_in_group"] = medoid_valid
    checks["finite_group_diagnostics"] = finite_costs
    checks["mapping_consistent"] = mapping_consistent
    checks["prototype_count"] = len(unique_cores) == int(args.target_prototypes)
    checks["saved_mlp_count"] = (
        int(args.expected_layer_count) - len(unique_cores)
        == int(args.expected_layer_count) - int(args.target_prototypes)
    )
    checks["reported_proto_count"] = int(policy.get("actual_proto_count", -1)) == int(args.target_prototypes)
    if str(args.pinned_layers).strip():
        pinned_layers: List[int] = sorted(
            {int(item.strip()) for item in str(args.pinned_layers).split(",") if item.strip()}
        )
    else:
        pinned_layers = [int(args.pinned_layer)]
    checks["pinned_layers_private"] = all(
        0 <= layer_id < len(entries)
        and int(entries[layer_id].get("layer_id", -1)) == layer_id
        and int(entries[layer_id].get("group_id", 0)) < 0
        and bool(entries[layer_id].get("private_core", False))
        for layer_id in pinned_layers
    )
    checks["shared_layers_in_expected_range"] = shared_layers_in_range
    if bool(args.require_contiguous_groups):
        checks["groups_contiguous"] = groups_contiguous

    failed = [name for name, passed in checks.items() if not passed]
    report = {
        "status": "PASS" if not failed else "FAIL",
        "policy": str(policy_path),
        "run_config": str(run_config_path),
        "split_manifest": str(split_manifest_path),
        "target_prototypes": int(args.target_prototypes),
        "actual_prototypes": len(unique_cores),
        "saved_mlp_count": int(args.expected_layer_count) - len(unique_cores),
        "input_gate_quantile": gate.get("quantile"),
        "input_gate_threshold": gate.get("threshold"),
        "groups": [group.get("layers", []) for group in groups],
        "pinned_layers": pinned_layers,
        "checks": checks,
        "failed_checks": failed,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    if failed:
        raise AssertionError(f"policy contract failed: {', '.join(failed)}; report={output}")
    print(f"[FFN-Policy-Validate] PASS policy={policy_path}")
    print(
        f"[FFN-Policy-Validate] cores={len(unique_cores)} "
        f"saved={int(args.expected_layer_count) - len(unique_cores)} "
        f"gate={gate_threshold:.8g}"
    )
    print(f"[FFN-Policy-Validate] report={output}")


if __name__ == "__main__":
    main()
