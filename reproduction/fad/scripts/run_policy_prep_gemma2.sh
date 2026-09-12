#!/bin/bash

set -euo pipefail

BACKBONE="${1:?usage: run_policy_prep_gemma2.sh BACKBONE SUITE_ID}"
SUITE_ID="${2:?usage: run_policy_prep_gemma2.sh BACKBONE SUITE_ID}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
WORKSPACE_ROOT="${FAD_WORKSPACE_ROOT:?set FAD_WORKSPACE_ROOT to a writable experiment workspace}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$(command -v torchrun)}"
NUM_GPUS="${NUM_GPUS:-1}"
SUITE_ROOT="${WORKSPACE_ROOT}/results/fad_paper_rerun_${SUITE_ID}"
POLICY_ROOT="${SUITE_ROOT}/policies/${BACKBONE}"
SPLIT_DIR="${POLICY_ROOT}/calibration_split"
OBS_DIR="${POLICY_ROOT}/functional_observation"

SPLIT_PY="${PROJECT_ROOT}/core/make_ffn_policy_calibration_split.py"
COLLECTOR_PY="${PROJECT_ROOT}/core/ffn_functional_redundancy_ddp_gemma2.py"
ANALYZER_PY="${PROJECT_ROOT}/core/analyze_ffn_grouping_views.py"
VALIDATOR_PY="${PROJECT_ROOT}/core/validate_ffn_input_gate_policy.py"
SELECTOR_PY="${PROJECT_ROOT}/core/select_ffn_zero_shot_seed_strategy.py"
PIPELINE_PY="${PROJECT_ROOT}/core/newthesis_pipeline_final_gemma2.py"

case "${BACKBONE}" in
  gemma2_9b)
    BASE_MODEL="${FAD_BASE_MODEL:-/home/ljkj/models/gemma-2-9b}"
    TEACHER="${FAD_TEACHER_CKPT:?set FAD_TEACHER_CKPT to the merged Gemma-2-9B commonsense teacher}"
    CANONICAL_ROOT="${FAD_CANONICAL_ROOT:?set FAD_CANONICAL_ROOT to the Gemma atlas/split root}"
    TRAIN_SPLIT="${FAD_TRAIN_SPLIT:-${CANONICAL_ROOT}/shared_split/train.json}"
    VAL_SPLIT="${FAD_VAL_SPLIT:-${CANONICAL_ROOT}/shared_split/val.json}"

    : "${FAD_GEMMA2_MAX_GROUP_SIZE:?set FAD_GEMMA2_MAX_GROUP_SIZE after Gemma policy analysis}"
    : "${FAD_GEMMA2_PROTOS:?set space-separated Gemma prototype counts}"
    : "${FAD_GEMMA2_SHARED_MINS:?set space-separated Gemma shared-layer minima}"
    : "${FAD_GEMMA2_GATE_QUANTILES:?set space-separated Gemma gate quantiles}"

    LAYER_COUNT=42
    FINAL_LAYER=41
    MAX_GROUP_SIZE="${FAD_GEMMA2_MAX_GROUP_SIZE}"
    read -r -a PROTOS <<< "${FAD_GEMMA2_PROTOS}"
    read -r -a SHARED_MINS <<< "${FAD_GEMMA2_SHARED_MINS}"
    read -r -a GATE_QUANTILES <<< "${FAD_GEMMA2_GATE_QUANTILES}"
    BATCH_SIZE="${FAD_GEMMA2_POLICY_BATCH_SIZE:-1}"
    ;;
  llama32_3b)
    BASE_MODEL="meta-llama/Llama-3.2-3B"
    TEACHER="${FAD_TEACHER_CKPT:?set FAD_TEACHER_CKPT to the merged Llama-3.2-3B teacher}"
    CANONICAL_ROOT="${FAD_CANONICAL_ROOT:?set FAD_CANONICAL_ROOT to the atlas/split root}"
    TRAIN_SPLIT="${FAD_TRAIN_SPLIT:-${CANONICAL_ROOT}/shared_split/train.json}"
    VAL_SPLIT="${FAD_VAL_SPLIT:-${CANONICAL_ROOT}/shared_split/val.json}"
    LAYER_COUNT=28
    FINAL_LAYER=27
    MAX_GROUP_SIZE=8
    PROTOS=(25 22 19 17 15)
    SHARED_MINS=(22 19 16 14 12)
    # Budget-specific hard gate for the most aggressive operating point.
    # Starting at q0.40 and scanning the fixed 0.01 grid downward, q0.24 is
    # the largest feasible quantile for K=15 (q0.25 is infeasible).
    GATE_QUANTILES=(0.40 0.40 0.40 0.40 0.24)
    BATCH_SIZE=8
    ;;
  llama31_8b)
    BASE_MODEL="meta-llama/Llama-3.1-8B"
    TEACHER="${FAD_TEACHER_CKPT:?set FAD_TEACHER_CKPT to the merged Llama-3.1-8B teacher}"
    CANONICAL_ROOT="${FAD_CANONICAL_ROOT:?set FAD_CANONICAL_ROOT to the atlas/split root}"
    TRAIN_SPLIT="${FAD_TRAIN_SPLIT:-${CANONICAL_ROOT}/shared_split/train.json}"
    VAL_SPLIT="${FAD_VAL_SPLIT:-${CANONICAL_ROOT}/shared_split/val.json}"
    LAYER_COUNT=32
    FINAL_LAYER=31
    MAX_GROUP_SIZE=9
    PROTOS=(27 26 25 23 20 18 16)
    SHARED_MINS=(24 23 22 20 17 15 13)
    # Deterministic 0.01-grid feasibility scan on the fixed held-out
    # observation: K=18 supports at most q0.36 (q0.37 is infeasible), while
    # K=16 supports at most q0.20 (q0.21 is infeasible).
    GATE_QUANTILES=(0.40 0.40 0.40 0.40 0.40 0.36 0.20)
    BATCH_SIZE=2
    ;;
  *)
    echo "[Policy-Prep][Error] unsupported backbone=${BACKBONE}" >&2
    exit 1
    ;;
esac

TOKENIZER_NAME_OR_PATH="${FAD_TOKENIZER_NAME_OR_PATH:-${BASE_MODEL}}"

if [ "${#PROTOS[@]}" -ne "${#SHARED_MINS[@]}" ] ||
   [ "${#PROTOS[@]}" -ne "${#GATE_QUANTILES[@]}" ]; then
  echo "[Policy-Prep][Error] PROTOS, SHARED_MINS and GATE_QUANTILES must have equal lengths" >&2
  exit 1
fi

ATLAS_DIR="${FAD_ATLAS_DIR_OVERRIDE:-${CANONICAL_ROOT}/phase1_atlas}"
BASE_POLICY="${ATLAS_DIR}/sharing_policy.json"
FEATURE_DATA="${SPLIT_DIR}/feature_records.json"
INTERVENTION_DATA="${SPLIT_DIR}/intervention_records.json"
SPLIT_MANIFEST="${SPLIT_DIR}/split_manifest.json"

for path in \
  "${TEACHER}/config.json" \
  "${TRAIN_SPLIT}" \
  "${VAL_SPLIT}" \
  "${ATLAS_DIR}/atlas_state.pt" \
  "${ATLAS_DIR}/final_structure_prior.pt" \
  "${BASE_POLICY}" \
  "${SPLIT_PY}" \
  "${COLLECTOR_PY}" \
  "${ANALYZER_PY}" \
  "${VALIDATOR_PY}" \
  "${SELECTOR_PY}" \
  "${PIPELINE_PY}"; do
  if [ ! -f "${path}" ]; then
    echo "[Policy-Prep][Error] required file missing: ${path}" >&2
    exit 1
  fi
done

mkdir -p "${POLICY_ROOT}" "${SPLIT_DIR}" "${OBS_DIR}" "${POLICY_ROOT}/.matplotlib"
export MPLCONFIGDIR="${POLICY_ROOT}/.matplotlib"

echo "[Policy-Prep] suite=${SUITE_ID} backbone=${BACKBONE}"
echo "[Policy-Prep] method=budget-aware hard input gate + held-out functional complete-link + simultaneous seed gate"

if [ ! -f "${SPLIT_MANIFEST}" ]; then
  "${PYTHON_BIN}" "${SPLIT_PY}" \
    --source "${TRAIN_SPLIT}" \
    --output-dir "${SPLIT_DIR}" \
    --feature-records 4096 \
    --intervention-records 4096 \
    --seed 44
fi

if [ ! -f "${OBS_DIR}/run_config.json" ]; then
  "${TORCHRUN_BIN}" \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${NUM_GPUS}" \
    "${COLLECTOR_PY}" \
    --model_name_or_path "${TEACHER}" \
    --tokenizer_name_or_path "${TOKENIZER_NAME_OR_PATH}" \
    --data_path "${FEATURE_DATA}" \
    --data_kind commonsense_json \
    --feature_data_path "${FEATURE_DATA}" \
    --intervention_data_path "${INTERVENTION_DATA}" \
    --feature_data_kind commonsense_json \
    --intervention_data_kind commonsense_json \
    --output_dir "${OBS_DIR}" \
    --max_records 0 \
    --block_size 512 \
    --max_blocks 64 \
    --activation_batches 12 \
    --intervention_batches 8 \
    --batch_size 1 \
    --max_positions_per_batch 512 \
    --max_samples_per_layer 4096 \
    --projection_dim 256 \
    --pair_window 0 \
    --kl_weight 1.0 \
    --kl_chunk_tokens 32 \
    --dtype auto \
    --trust_remote_code False \
    --seed 44
fi

for idx in "${!PROTOS[@]}"; do
  proto="${PROTOS[$idx]}"
  shared_min="${SHARED_MINS[$idx]}"
  gate_quantile="${GATE_QUANTILES[$idx]}"
  proto_root="${POLICY_ROOT}/proto${proto}"
  analysis_dir="${proto_root}/grouping_analysis"
  selected_policy="${proto_root}/selected_policy.json"
  validation_report="${proto_root}/policy_validation.json"
  zero_shot_root="${proto_root}/zero_shot_seed_gate"
  strategy_txt="${zero_shot_root}/selected_seed_strategy.txt"
  generated_policy="${analysis_dir}/policies/sharing_policy_input_gate_functional.json"

  pinned=""
  for ((layer=0; layer<shared_min; layer++)); do
    pinned+="${layer},"
  done
  pinned+="${FINAL_LAYER}"

  mkdir -p "${proto_root}" "${analysis_dir}" "${zero_shot_root}"
  echo "[Policy-Prep] proto=${proto} gate_quantile=${gate_quantile} shared_range=${shared_min}-$((FINAL_LAYER - 1)) pinned=${pinned}"

  if [ ! -f "${selected_policy}" ]; then
    "${PYTHON_BIN}" "${ANALYZER_PY}" \
      --obs-dir "${OBS_DIR}" \
      --base-policy "${BASE_POLICY}" \
      --output-dir "${analysis_dir}" \
      --target-prototypes "${proto}" \
      --max-group-size "${MAX_GROUP_SIZE}" \
      --max-layer-span "$((MAX_GROUP_SIZE - 1))" \
      --pinned-layers "${pinned}" \
      --input-gate-quantile "${gate_quantile}" \
      --require-contiguous-groups \
      --qap-permutations 1000 \
      --seed 44
    if [ ! -f "${generated_policy}" ]; then
      echo "[Policy-Prep][Error] missing generated policy ${generated_policy}" >&2
      exit 1
    fi
    cp "${generated_policy}" "${selected_policy}"
  fi

  "${PYTHON_BIN}" "${VALIDATOR_PY}" \
    --policy "${selected_policy}" \
    --run-config "${OBS_DIR}/run_config.json" \
    --split-manifest "${SPLIT_MANIFEST}" \
    --expected-teacher "${TEACHER}" \
    --expected-feature-data "${FEATURE_DATA}" \
    --expected-intervention-data "${INTERVENTION_DATA}" \
    --target-prototypes "${proto}" \
    --expected-layer-count "${LAYER_COUNT}" \
    --pinned-layers "${pinned}" \
    --expected-shared-layer-min "${shared_min}" \
    --expected-shared-layer-max "$((FINAL_LAYER - 1))" \
    --require-contiguous-groups \
    --expected-gate-quantile "${gate_quantile}" \
    --output "${validation_report}"

  if [ ! -s "${strategy_txt}" ]; then
    for strategy in policy_medoid medoid; do
      trial_dir="${zero_shot_root}/${strategy}"
      "${TORCHRUN_BIN}" \
        --standalone \
        --nnodes=1 \
        --nproc_per_node="${NUM_GPUS}" \
        "${PIPELINE_PY}" pass1 \
        --atlas_path "${ATLAS_DIR}/atlas_state.pt" \
        --sharing_policy_path "${selected_policy}" \
        --output_dir "${trial_dir}" \
        --base_model "${BASE_MODEL}" \
        --teacher_ckpt "${TEACHER}" \
        --teacher_loader native \
        --tokenizer_name_or_path "${TOKENIZER_NAME_OR_PATH}" \
        --trust_remote_code False \
        --student_gradient_checkpointing False \
        --teacher_gradient_checkpointing False \
        --data_path "${VAL_SPLIT}" \
        --training_prompt_mode decision_aligned \
        --batch_size "${BATCH_SIZE}" \
        --cutoff_len 384 \
        --max_records 2048 \
        --shuffle_records False \
        --seed 44 \
        --device cuda \
        --private_down_rank 128 \
        --private_down_alpha 128 \
        --proto_seed_strategy "${strategy}" \
        --steps 0 \
        --distill_mode ce \
        --lambda_ce 1.0 \
        --lambda_kd 0.0 \
        --lambda_hidden_mse 0.0 \
        --lambda_core 1.0 \
        --loss_scope decision \
        --loss_exclude_eos True \
        --core_layers all_shared_layers \
        --core_token_selection last_pred \
        --val_data_path "${VAL_SPLIT}" \
        --val_every 1 \
        --val_max_records 2048 \
        --val_max_batches 0 \
        --val_seed 44 \
        --val_selection_metric decision_ce \
        --val_include_step0_candidate True \
        --val_subspace_viz_enable False
    done

    "${PYTHON_BIN}" "${SELECTOR_PY}" \
      --policy-medoid-report "${zero_shot_root}/policy_medoid/compress_report.json" \
      --atlas-medoid-report "${zero_shot_root}/medoid/compress_report.json" \
      --policy "${selected_policy}" \
      --output-json "${zero_shot_root}/seed_strategy_selection.json" \
      --output-strategy "${strategy_txt}"
  fi

  sha256sum "${selected_policy}" > "${proto_root}/selected_policy.sha256"
  touch "${proto_root}/POLICY_READY"
done

touch "${POLICY_ROOT}/ALL_POLICIES_READY"
echo "[Policy-Prep] complete backbone=${BACKBONE} root=${POLICY_ROOT}"
