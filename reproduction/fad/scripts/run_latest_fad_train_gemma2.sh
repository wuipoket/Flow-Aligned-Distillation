#!/bin/bash

set -euo pipefail

BACKBONE="${1:?usage: run_latest_fad_train_gemma2.sh BACKBONE BUDGET PROTO SEED VARIANT ATLAS RUN_NAME SUITE_ID}"
BUDGET="${2:?}"
PROTO="${3:?}"
SEED="${4:?}"
VARIANT="${5:?}"
ATLAS_KIND="${6:?}"
RUN_NAME="${7:?}"
SUITE_ID="${8:?}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
WORKSPACE_ROOT="${FAD_WORKSPACE_ROOT:?set FAD_WORKSPACE_ROOT to a writable experiment workspace}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
NUM_GPUS="${NUM_GPUS:-1}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29501}"
RUNNER="${PROJECT_ROOT}/scripts/run_try_to_recovery_final_gemma2.sh"
PIPELINE_PY="${PROJECT_ROOT}/core/newthesis_pipeline_final_gemma2.py"
SUITE_OUT_ROOT="${WORKSPACE_ROOT}/out/fad_paper_rerun_${SUITE_ID}"
SUITE_RESULTS_ROOT="${WORKSPACE_ROOT}/results/fad_paper_rerun_${SUITE_ID}"
OUT_ROOT="${SUITE_OUT_ROOT}/${RUN_NAME}"
RESULTS_DIR="${SUITE_RESULTS_ROOT}/runs/${RUN_NAME}"
POLICY_ROOT="${SUITE_RESULTS_ROOT}/policies/${BACKBONE}/proto${PROTO}"
POLICY="${POLICY_ROOT}/selected_policy.json"
STRATEGY_FILE="${POLICY_ROOT}/zero_shot_seed_gate/selected_seed_strategy.txt"

case "${BACKBONE}" in
  gemma2_9b)
    BASE_MODEL="${FAD_BASE_MODEL:-/home/ljkj/models/gemma-2-9b}"
    TEACHER="${FAD_TEACHER_CKPT:?set FAD_TEACHER_CKPT to the merged Gemma-2-9B commonsense teacher}"
    CANONICAL_ROOT="${FAD_CANONICAL_ROOT:?set FAD_CANONICAL_ROOT to the Gemma atlas/split root}"
    TRAIN_SPLIT="${FAD_TRAIN_SPLIT:-${CANONICAL_ROOT}/shared_split/train.json}"
    VAL_SPLIT="${FAD_VAL_SPLIT:-${CANONICAL_ROOT}/shared_split/val.json}"
    BATCH_SIZE=1
    EVAL_BATCH_SIZE=1
    ;;
  llama32_3b)
    BASE_MODEL="meta-llama/Llama-3.2-3B"
    TEACHER="${FAD_TEACHER_CKPT:?set FAD_TEACHER_CKPT to the merged Llama-3.2-3B teacher}"
    CANONICAL_ROOT="${FAD_CANONICAL_ROOT:?set FAD_CANONICAL_ROOT to the atlas/split root}"
    TRAIN_SPLIT="${FAD_TRAIN_SPLIT:-${CANONICAL_ROOT}/shared_split/train.json}"
    VAL_SPLIT="${FAD_VAL_SPLIT:-${CANONICAL_ROOT}/shared_split/val.json}"
    BATCH_SIZE=8
    EVAL_BATCH_SIZE=8
    ;;
  llama31_8b)
    BASE_MODEL="meta-llama/Llama-3.1-8B"
    TEACHER="${FAD_TEACHER_CKPT:?set FAD_TEACHER_CKPT to the merged Llama-3.1-8B teacher}"
    CANONICAL_ROOT="${FAD_CANONICAL_ROOT:?set FAD_CANONICAL_ROOT to the atlas/split root}"
    TRAIN_SPLIT="${FAD_TRAIN_SPLIT:-${CANONICAL_ROOT}/shared_split/train.json}"
    VAL_SPLIT="${FAD_VAL_SPLIT:-${CANONICAL_ROOT}/shared_split/val.json}"
    BATCH_SIZE=2
    EVAL_BATCH_SIZE=4
    ;;
  *)
    echo "[Train][Error] unsupported backbone=${BACKBONE}" >&2
    exit 1
    ;;
esac

case "${ATLAS_KIND}" in
  velocity_pca)
    ATLAS_DIR="${CANONICAL_ROOT}/phase1_atlas"
    ;;
  hidden_pca|random_projection)
    ATLAS_DIR="${SUITE_OUT_ROOT}/projection_atlas_gemma2_9b_${ATLAS_KIND}/phase1_atlas"
    ;;
  *)
    echo "[Train][Error] unsupported atlas=${ATLAS_KIND}" >&2
    exit 1
    ;;
esac

for path in \
  "${RUNNER}" \
  "${PIPELINE_PY}" \
  "${TEACHER}/config.json" \
  "${TRAIN_SPLIT}" \
  "${VAL_SPLIT}" \
  "${ATLAS_DIR}/atlas_state.pt" \
  "${ATLAS_DIR}/final_structure_prior.pt" \
  "${POLICY}" \
  "${STRATEGY_FILE}" \
  "${POLICY_ROOT}/POLICY_READY"; do
  if [ ! -f "${path}" ]; then
    echo "[Train][Error] required file missing: ${path}" >&2
    exit 1
  fi
done

PROTO_SEED_STRATEGY="$(tr -d '[:space:]' < "${STRATEGY_FILE}")"
case "${PROTO_SEED_STRATEGY}" in
  policy_medoid|medoid) ;;
  *)
    echo "[Train][Error] invalid proto seed strategy=${PROTO_SEED_STRATEGY}" >&2
    exit 1
    ;;
esac

DISTILL_MODE=ce
LAMBDA_KD=0.0
LAMBDA_HIDDEN=0.0
LAMBDA_CORE=1.0
WHITENING=True
case "${VARIANT}" in
  fad) ;;
  ce_only)
    LAMBDA_CORE=0.0
    ;;
  kl)
    DISTILL_MODE=ce_kd
    LAMBDA_KD=1.0
    LAMBDA_CORE=0.0
    ;;
  hidden)
    LAMBDA_HIDDEN=1.0
    LAMBDA_CORE=0.0
    ;;
  isotropic)
    WHITENING=False
    ;;
  *)
    echo "[Train][Error] unsupported variant=${VARIANT}" >&2
    exit 1
    ;;
esac

VIZ=False
if [ "${RUN_NAME}" = "3b_b20_k19_s44_fad" ]; then
  VIZ=True
fi
VIZ="${FAD_TRAIN_VIZ:-${VIZ}}"
case "$(echo "${VIZ}" | tr '[:upper:]' '[:lower:]')" in
  true|false|1|0|yes|no|on|off) ;;
  *)
    echo "[Train][Error] invalid FAD_TRAIN_VIZ=${VIZ}" >&2
    exit 1
    ;;
esac

mkdir -p "${OUT_ROOT}/provenance" "${RESULTS_DIR}"
cp "${POLICY}" "${OUT_ROOT}/provenance/selected_policy.json"
POLICY_COPY="${OUT_ROOT}/provenance/selected_policy.json"
sha256sum \
  "${RUNNER}" \
  "${PIPELINE_PY}" \
  "${ATLAS_DIR}/atlas_state.pt" \
  "${ATLAS_DIR}/final_structure_prior.pt" \
  "${TRAIN_SPLIT}" \
  "${VAL_SPLIT}" \
  "${POLICY_COPY}" \
  > "${OUT_ROOT}/provenance/input_hashes.sha256"

"${PYTHON_BIN}" - \
  "${OUT_ROOT}/provenance/run_contract.json" \
  "${BACKBONE}" "${BUDGET}" "${PROTO}" "${SEED}" "${VARIANT}" "${ATLAS_KIND}" \
  "${PROTO_SEED_STRATEGY}" "${DISTILL_MODE}" "${LAMBDA_KD}" "${LAMBDA_HIDDEN}" \
  "${LAMBDA_CORE}" "${WHITENING}" <<'PY'
import json
import os
import pathlib
import sys

keys = [
    "backbone", "budget", "proto_count", "seed", "variant", "atlas_kind",
    "proto_seed_strategy", "distill_mode", "lambda_kd", "lambda_hidden_mse",
    "lambda_core", "metric_whitening",
]
vals = sys.argv[2:]
payload = dict(zip(keys, vals))
payload.update({
    "training_prompt_mode": "decision_aligned",
    "loss_scope": "decision",
    "checkpoint_metric": "decision_ce",
    "core_token_selection": "last_pred",
    "core_candidate_tokens": 1,
    "metric_reliability_weighting": True,
    "cutoff_len": 384,
    "steps": 7500,
    "qwen_excluded": True,
    "training_time_subspace_viz": str(os.environ.get("FAD_TRAIN_VIZ", "auto")),
    "strict_determinism": str(os.environ.get("FAD_STRICT_DETERMINISM", "0")),
    "cublas_workspace_config": str(os.environ.get("CUBLAS_WORKSPACE_CONFIG", "")),
    "nccl_algo": str(os.environ.get("NCCL_ALGO", "")),
    "nccl_proto": str(os.environ.get("NCCL_PROTO", "")),
})
pathlib.Path(sys.argv[1]).write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY

echo "[Train] suite=${SUITE_ID} run=${RUN_NAME} backbone=${BACKBONE} budget=${BUDGET} proto=${PROTO}"
echo "[Train] latest_recipe=restricted-late functional-medoid decision-CE + point-CORE"
echo "[Train] variant=${VARIANT} atlas=${ATLAS_KIND} proto_seed=${PROTO_SEED_STRATEGY}"

RUN_TAG="${RUN_NAME}" \
OUT_ROOT="${OUT_ROOT}" \
RESULTS_DIR="${RESULTS_DIR}" \
WORKSPACE_ROOT="${WORKSPACE_ROOT}" \
PROJECT_ROOT="${PROJECT_ROOT}" \
ORIGINAL_SCRIPT_PATH="${RUNNER}" \
PIPELINE_PY="${PIPELINE_PY}" \
MODEL_PRESET="${BACKBONE}" \
BASE_MODEL="${BASE_MODEL}" \
TEACHER_CKPT="${TEACHER}" \
TEACHER_LOADER=native \
TEACHER_USE_MERGED=False \
TOKENIZER_NAME_OR_PATH="${FAD_TOKENIZER_NAME_OR_PATH:-${BASE_MODEL}}" \
TRUST_REMOTE_CODE=False \
SOURCE_DATA_PATH="${TRAIN_SPLIT}" \
TEST_DATA_ROOT="${WORKSPACE_ROOT}/data/datasets" \
DATA_SPLIT_ENABLE=False \
DATA_PATH="${TRAIN_SPLIT}" \
PASS1_VAL_DATA_PATH="${VAL_SPLIT}" \
EXTERNAL_ATLAS_DIR="${ATLAS_DIR}" \
SHARING_POLICY_PATH="${POLICY_COPY}" \
FUNCTIONAL_POLICY_ENABLE=False \
TARGET_PROTO_COUNT=0 \
FLAP_ENABLE=False \
DIRECT_TRAIN_ENABLE=False \
SEED="${SEED}" \
NUM_GPUS="${NUM_GPUS}" \
MASTER_ADDR="${MASTER_ADDR}" \
MASTER_PORT="${MASTER_PORT}" \
BATCH_SIZE="${BATCH_SIZE}" \
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
CUTOFF_LEN=384 \
PRIVATE_DOWN_RANK=128 \
PROTO_SEED_STRATEGY="${PROTO_SEED_STRATEGY}" \
STUDENT_GRADIENT_CHECKPOINTING=True \
TEACHER_GRADIENT_CHECKPOINTING=False \
DEVICE=cuda \
TRAINING_PROMPT_MODE=decision_aligned \
LOSS_SCOPE=decision \
LOSS_EXCLUDE_EOS=True \
PASS1_DISTILL_MODE="${DISTILL_MODE}" \
LAMBDA_CE=1.0 \
LAMBDA_KD="${LAMBDA_KD}" \
KD_TEMPERATURE=2.0 \
LAMBDA_HIDDEN_MSE="${LAMBDA_HIDDEN}" \
LAMBDA_CORE="${LAMBDA_CORE}" \
CORE_USE_METRIC_WHITENING="${WHITENING}" \
CORE_USE_RELIABILITY_WEIGHTING=True \
CORE_TOKEN_SELECTION=last_pred \
CORE_CANDIDATE_TOKENS=1 \
LAMBDA_GEODESIC_CORE=0.0 \
LAMBDA_MANIFOLD_CORE=0.0 \
LAMBDA_DELTA_MANIFOLD_CORE=0.0 \
PASS1_STEPS="${FAD_PASS1_STEPS:-7500}" \
PASS1_LR="${FAD_PASS1_LR:-5e-5}" \
PASS1_LR_BANK="${FAD_PASS1_LR_BANK:-3e-5}" \
PASS1_LR_ADAPTER="${FAD_PASS1_LR_ADAPTER:-8e-5}" \
PASS1_LR_SCHEDULE=warmup_cosine \
PASS1_LR_WARMUP_STEPS="${FAD_PASS1_LR_WARMUP_STEPS:-750}" \
PASS1_LR_WARMUP_RATIO=0.1 \
PASS1_LR_MIN_RATIO="${FAD_PASS1_LR_MIN_RATIO:-0.01}" \
PASS1_WEIGHT_DECAY=0.01 \
PASS1_LOG_EVERY=50 \
PASS1_VAL_EVERY=500 \
PASS1_VAL_MAX_RECORDS=2048 \
PASS1_VAL_MAX_BATCHES=0 \
PASS1_VAL_SEED="${SEED}" \
PASS1_VAL_SELECTION_METRIC=decision_ce \
PASS1_VAL_INCLUDE_STEP0_CANDIDATE="${FAD_PASS1_VAL_INCLUDE_STEP0_CANDIDATE:-False}" \
PASS1_VAL_MIN_IMPROVEMENT=0.000001 \
PASS1_VAL_SUBSPACE_VIZ_ENABLE="${VIZ}" \
PASS1_VAL_SUBSPACE_VIZ_MAX_POINTS_PER_REGIME=768 \
QUANT_BITS=16 \
EVAL_ENABLE="${FAD_EVAL_ENABLE:-True}" \
EVAL_BASELINE_ENABLE=False \
EVAL_DATASETS="piqa social_i_qa hellaswag winogrande ARC-Challenge ARC-Easy openbookqa" \
EVAL_MODE=logprob \
EVAL_LENGTH_NORM=none \
EVAL_MAX_NEW_TOKENS=16 \
EVAL_USE_QUANT_BANK_INT4=False \
bash "${RUNNER}"

if [ ! -f "${OUT_ROOT}/phase4_export/deploy_bundle.pt" ]; then
  echo "[Train][Error] deploy bundle missing after run" >&2
  exit 1
fi
touch "${RESULTS_DIR}/EXPERIMENT_COMPLETE"
echo "[Train] complete run=${RUN_NAME} results=${RESULTS_DIR}"
