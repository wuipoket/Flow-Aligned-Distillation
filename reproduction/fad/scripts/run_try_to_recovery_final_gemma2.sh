#!/bin/bash
set -euo pipefail

if [ -n "${PYTHON_BIN:-}" ]; then
  PYTHON_BIN="${PYTHON_BIN}"
elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
  PYTHON_BIN="${CONDA_PREFIX}/bin/python"
else
  PYTHON_BIN="$(command -v python || true)"
fi
if [ -z "${PYTHON_BIN}" ]; then
  echo "[FINAL][Error] python interpreter not found. Set PYTHON_BIN explicitly." >&2
  exit 1
fi

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
SCRIPT_SOURCE_PATH="${ORIGINAL_SCRIPT_PATH:-${BASH_SOURCE[0]}}"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_SOURCE_PATH}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-${FAD_WORKSPACE_ROOT:-$(cd "${PROJECT_ROOT}/../.." && pwd)}}"
PIPELINE_PY="${PIPELINE_PY:-${PROJECT_ROOT}/core/newthesis_pipeline_final_gemma2.py}"
DIRECT_TRAIN_PY="${DIRECT_TRAIN_PY:-${PROJECT_ROOT}/core/flap_ce_kd_train_final_llama.py}"

OUT_ROOT="${OUT_ROOT:-${WORKSPACE_ROOT}/out/newthesis_final_gemma2_${RUN_TAG}}"
RESULTS_DIR="${RESULTS_DIR:-${WORKSPACE_ROOT}/results/newthesis_final_gemma2_${RUN_TAG}}"
mkdir -p "${OUT_ROOT}" "${RESULTS_DIR}"

MODEL_PRESET="${MODEL_PRESET:-gemma2_9b}"
MODEL_PRESET_LC="$(echo "${MODEL_PRESET}" | tr '[:upper:]' '[:lower:]')"
DEFAULT_BASE_MODEL_GEMMA2_9B="${DEFAULT_BASE_MODEL_GEMMA2_9B:-/home/ljkj/models/gemma-2-9b}"
DEFAULT_TEACHER_MERGED_CKPT_GEMMA2_9B="${DEFAULT_TEACHER_MERGED_CKPT_GEMMA2_9B:-${WORKSPACE_ROOT}/teachers/gemma2_9b_commonsense/merged}"
DEFAULT_BASE_MODEL_LLAMA32_3B="${DEFAULT_BASE_MODEL_LLAMA32_3B:-meta-llama/Llama-3.2-3B}"
DEFAULT_BASE_MODEL_LLAMA31_8B="${DEFAULT_BASE_MODEL_LLAMA31_8B:-meta-llama/Llama-3.1-8B}"
DEFAULT_TEACHER_MERGED_CKPT_LLAMA32_3B="${DEFAULT_TEACHER_MERGED_CKPT_LLAMA32_3B:-${WORKSPACE_ROOT}/out/lora_latest/merged_teacher}"
DEFAULT_TEACHER_MERGED_CKPT_LLAMA31_8B="${DEFAULT_TEACHER_MERGED_CKPT_LLAMA31_8B:-}"

case "${MODEL_PRESET_LC}" in
  gemma2_9b)
    BASE_MODEL_DEFAULT="${DEFAULT_BASE_MODEL_GEMMA2_9B}"
    TEACHER_CKPT_DEFAULT="${DEFAULT_TEACHER_MERGED_CKPT_GEMMA2_9B}"
    ;;
  llama32_3b)
    BASE_MODEL_DEFAULT="${DEFAULT_BASE_MODEL_LLAMA32_3B}"
    TEACHER_CKPT_DEFAULT="${DEFAULT_TEACHER_MERGED_CKPT_LLAMA32_3B}"
    ;;
  llama31_8b)
    BASE_MODEL_DEFAULT="${DEFAULT_BASE_MODEL_LLAMA31_8B}"
    TEACHER_CKPT_DEFAULT="${DEFAULT_TEACHER_MERGED_CKPT_LLAMA31_8B}"
    ;;
  custom)
    BASE_MODEL_DEFAULT="${DEFAULT_BASE_MODEL_GEMMA2_9B}"
    TEACHER_CKPT_DEFAULT=""
    ;;
  *)
    echo "[FINAL][Error] Unsupported MODEL_PRESET=${MODEL_PRESET}. Use gemma2_9b or custom." >&2
    exit 1
    ;;
esac

BASE_MODEL="${BASE_MODEL:-${BASE_MODEL_DEFAULT}}"
ORIGINAL_BASE_MODEL="${BASE_MODEL}"
if [ -z "${TEACHER_CKPT_DEFAULT}" ]; then
  TEACHER_CKPT_DEFAULT="${BASE_MODEL}"
fi
TEACHER_CKPT="${TEACHER_CKPT:-${TEACHER_CKPT_DEFAULT}}"
TEACHER_LOADER="${TEACHER_LOADER:-native}"
TEACHER_USE_MERGED="${TEACHER_USE_MERGED:-auto}" # auto|true|false
TEACHER_MERGED_CKPT="${TEACHER_MERGED_CKPT:-}"
TOKENIZER_NAME_OR_PATH="${TOKENIZER_NAME_OR_PATH:-${BASE_MODEL}}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-False}"

SOURCE_DATA_PATH="${SOURCE_DATA_PATH:-${WORKSPACE_ROOT}/data/datasets/commonsense_170k.json}"
SOURCE_DATA_FALLBACK="${SOURCE_DATA_FALLBACK:-${WORKSPACE_ROOT}/data/datasets/commonsense_170k.json}"
DATA_SPLIT_ENABLE="${DATA_SPLIT_ENABLE:-True}"
DATA_SPLIT_TRAIN_RATIO="${DATA_SPLIT_TRAIN_RATIO:-0.95}"
SHARED_SPLIT_DIR="${SHARED_SPLIT_DIR:-${OUT_ROOT}/shared_split}"
DATA_SPLIT_TRAIN_PATH="${DATA_SPLIT_TRAIN_PATH:-${SHARED_SPLIT_DIR}/train_${RUN_TAG}.json}"
DATA_SPLIT_VAL_PATH="${DATA_SPLIT_VAL_PATH:-${SHARED_SPLIT_DIR}/val_${RUN_TAG}.json}"
DATA_SPLIT_MANIFEST_PATH="${DATA_SPLIT_MANIFEST_PATH:-${SHARED_SPLIT_DIR}/split_${RUN_TAG}.manifest.json}"
DATA_PATH="${DATA_PATH:-}"
TEST_DATA_ROOT="${TEST_DATA_ROOT:-${WORKSPACE_ROOT}/data/datasets}"

FLAP_ENABLE="${FLAP_ENABLE:-False}"
FLAP_ROOT="${FLAP_ROOT:-${WORKSPACE_ROOT}/FLAP}"
FLAP_MAIN_PY="${FLAP_MAIN_PY:-${FLAP_ROOT}/main.py}"
FLAP_MODEL="${FLAP_MODEL:-}"
FLAP_OUTPUT_DIR="${FLAP_OUTPUT_DIR:-${OUT_ROOT}/phase0_flap/flap_gemma2_9b_base}"
FLAP_REUSE_IF_EXISTS="${FLAP_REUSE_IF_EXISTS:-True}"
FLAP_PRUNING_RATIO="${FLAP_PRUNING_RATIO:-0.10}"
FLAP_REMOVE_HEADS="${FLAP_REMOVE_HEADS:-0}"
FLAP_METRICS="${FLAP_METRICS:-WIFV}"
FLAP_STRUCTURE="${FLAP_STRUCTURE:-AL-AM}"
FLAP_PRUNE_SCOPE="${FLAP_PRUNE_SCOPE:-all}"
FLAP_CALIB_DATASET="${FLAP_CALIB_DATASET:-json}"
FLAP_CALIB_DATA_PATH="${FLAP_CALIB_DATA_PATH:-${SOURCE_DATA_PATH}}"
FLAP_NSAMPLES="${FLAP_NSAMPLES:-256}"
FLAP_SEQLEN="${FLAP_SEQLEN:-128}"
FLAP_MODEL_LOADER="${FLAP_MODEL_LOADER:-auto}"
FLAP_UNSTR="${FLAP_UNSTR:-True}"
FLAP_CACHE_DIR="${FLAP_CACHE_DIR:-${WORKSPACE_ROOT}/llm_weights}"

DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
NUM_GPUS="${NUM_GPUS:-1}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29501}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$(command -v torchrun || true)}"

ANALYSIS_RANK="${ANALYSIS_RANK:-256}"
ATLAS_PROJECTION_BASIS_SOURCE="${ATLAS_PROJECTION_BASIS_SOURCE:-velocity_pca}"
RANDOM_BASIS_SEED="${RANDOM_BASIS_SEED:-0}"
PCA_MODE="${PCA_MODE:-stream_sketch}"
PCA_DEVICE="${PCA_DEVICE:-cuda}"
PCA_STREAM_CHUNK_SIZE="${PCA_STREAM_CHUNK_SIZE:-4096}"
RESERVOIR_SIZE="${RESERVOIR_SIZE:-51200}"
NUM_CODES="${NUM_CODES:-64}"
KMEANS_MODE="${KMEANS_MODE:-minibatch}"
KMEANS_ITERS="${KMEANS_ITERS:-50}"
KMEANS_BATCH_SIZE="${KMEANS_BATCH_SIZE:-4096}"
KMEANS_WARMUP_SIZE="${KMEANS_WARMUP_SIZE:-50000}"
KMEANS_WARMUP_ITERS="${KMEANS_WARMUP_ITERS:-50}"
KMEANS_REFINE_ITERS="${KMEANS_REFINE_ITERS:-8}"
KMEANS_ASSIGN_CHUNK_SIZE="${KMEANS_ASSIGN_CHUNK_SIZE:-8192}"
TOKEN_RULE="${TOKEN_RULE:-last_pred}"
TAU_EPS="${TAU_EPS:-1e-5}"
TAU_NMIN="${TAU_NMIN:-200}"
TAU_SHRINK_LAMBDA="${TAU_SHRINK_LAMBDA:-0.1}"
SHARING_POLICY_MODE="${SHARING_POLICY_MODE:-upstream_only}"
UPSTREAM_SIMILARITY_THRESHOLD="${UPSTREAM_SIMILARITY_THRESHOLD:-0.95}"
TARGET_PROTO_COUNT="${TARGET_PROTO_COUNT:-0}"
SHARING_POLICY_PATH="${SHARING_POLICY_PATH:-${OUT_ROOT}/phase1_atlas/sharing_policy.json}"
FUNCTIONAL_POLICY_ENABLE="${FUNCTIONAL_POLICY_ENABLE:-False}"
FUNCTIONAL_POLICY_PY="${FUNCTIONAL_POLICY_PY:-${PROJECT_ROOT}/core/retarget_sharing_policy_by_functional_cost_final_llama.py}"
FUNCTIONAL_OBS_DIR="${FUNCTIONAL_OBS_DIR:-}"
FUNCTIONAL_POLICY_ELIGIBLE_LAYERS="${FUNCTIONAL_POLICY_ELIGIBLE_LAYERS:-}"
FUNCTIONAL_POLICY_MAX_BIDIRECTIONAL_COST="${FUNCTIONAL_POLICY_MAX_BIDIRECTIONAL_COST:-0.32}"
FUNCTIONAL_POLICY_MAX_LAYER_GAP="${FUNCTIONAL_POLICY_MAX_LAYER_GAP:-1}"
FUNCTIONAL_POLICY_TARGET_SAVED_MLPS="${FUNCTIONAL_POLICY_TARGET_SAVED_MLPS:-0}"
FUNCTIONAL_POLICY_TARGET_COMPRESSION_RATIO="${FUNCTIONAL_POLICY_TARGET_COMPRESSION_RATIO:-0.0}"
FUNCTIONAL_POLICY_SAVED_MLP_WHOLE_MODEL_RATIO="${FUNCTIONAL_POLICY_SAVED_MLP_WHOLE_MODEL_RATIO:-0.0223333333}"
FUNCTIONAL_POLICY_MAX_GROUP_SIZE="${FUNCTIONAL_POLICY_MAX_GROUP_SIZE:-0}"
FUNCTIONAL_POLICY_REQUIRE_SAME_REGIME="${FUNCTIONAL_POLICY_REQUIRE_SAME_REGIME:-True}"
ATLAS_CKPT_EVERY_BATCHES="${ATLAS_CKPT_EVERY_BATCHES:-5000}"

BATCH_SIZE="${BATCH_SIZE:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
PASS1_GRAD_ACCUM_STEPS="${PASS1_GRAD_ACCUM_STEPS:-1}"
CUTOFF_LEN="${CUTOFF_LEN:-256}"
MAX_RECORDS="${MAX_RECORDS:-0}"
MAX_BATCHES="${MAX_BATCHES:-0}"
SHUFFLE_RECORDS="${SHUFFLE_RECORDS:-True}"

PRIVATE_DOWN_RANK="${PRIVATE_DOWN_RANK:-64}"
PRIVATE_DOWN_ALPHA="${PRIVATE_DOWN_ALPHA:-0}"
PROTO_SEED_STRATEGY="${PROTO_SEED_STRATEGY:-medoid}"
STUDENT_GRADIENT_CHECKPOINTING="${STUDENT_GRADIENT_CHECKPOINTING:-False}"
TEACHER_GRADIENT_CHECKPOINTING="${TEACHER_GRADIENT_CHECKPOINTING:-False}"
TRAINING_PROMPT_MODE="${TRAINING_PROMPT_MODE:-legacy_sft}"
LOSS_SCOPE="${LOSS_SCOPE:-all}"
LOSS_EXCLUDE_EOS="${LOSS_EXCLUDE_EOS:-True}"

LAMBDA_CE="${LAMBDA_CE:-1.0}"
LAMBDA_KD="${LAMBDA_KD:-1.0}"
KD_TEMPERATURE="${KD_TEMPERATURE:-2.0}"
LAMBDA_HIDDEN_MSE="${LAMBDA_HIDDEN_MSE:-1.0}"
PASS1_DISTILL_MODE="${PASS1_DISTILL_MODE:-ce+kd}"
TEACHER_DEPLOY_BUNDLE="${TEACHER_DEPLOY_BUNDLE:-}"
LAMBDA_CORE="${LAMBDA_CORE:-0.0}"
CORE_USE_METRIC_WHITENING="${CORE_USE_METRIC_WHITENING:-True}"
CORE_METRIC_TRACE_NORMALIZE="${CORE_METRIC_TRACE_NORMALIZE:-False}"
CORE_METRIC_DIAG_PATH="${CORE_METRIC_DIAG_PATH:-}"
CORE_METRIC_DIAG_MODE="${CORE_METRIC_DIAG_MODE:-covariance}"
CORE_USE_RELIABILITY_WEIGHTING="${CORE_USE_RELIABILITY_WEIGHTING:-True}"
CORE_TOKEN_SELECTION="${CORE_TOKEN_SELECTION:-last_pred}"
CORE_CANDIDATE_TOKENS="${CORE_CANDIDATE_TOKENS:-1}"
CORE_IETS_TEMPERATURE="${CORE_IETS_TEMPERATURE:-1.0}"
CORE_IETS_ANCHOR_BOOST="${CORE_IETS_ANCHOR_BOOST:-0.0}"
CORE_IETS_ENERGY_ALPHA="${CORE_IETS_ENERGY_ALPHA:-1.0}"
CORE_IETS_ENTROPY_BETA="${CORE_IETS_ENTROPY_BETA:-0.0}"
CORE_IETS_TOPK="${CORE_IETS_TOPK:-1}"
LAMBDA_RELATIONAL_CORE="${LAMBDA_RELATIONAL_CORE:-0.0}"
LAMBDA_VARIANCE_CORE="${LAMBDA_VARIANCE_CORE:-0.0}"
VARIANCE_CORE_FLOOR_RATIO="${VARIANCE_CORE_FLOOR_RATIO:-0.70}"
GENERIC_REPLAY_DATA_PATH="${GENERIC_REPLAY_DATA_PATH:-}"
GENERIC_REPLAY_INTERVAL="${GENERIC_REPLAY_INTERVAL:-0}"
GENERIC_REPLAY_BATCH_SIZE="${GENERIC_REPLAY_BATCH_SIZE:-0}"
GENERIC_REPLAY_MAX_RECORDS="${GENERIC_REPLAY_MAX_RECORDS:-0}"
LAMBDA_GENERIC_REPLAY="${LAMBDA_GENERIC_REPLAY:-0.0}"
LAMBDA_LAYER_MIXTURE="${LAMBDA_LAYER_MIXTURE:-0.0}"
LAYER_MIXTURE_ENTROPY_TAU="${LAYER_MIXTURE_ENTROPY_TAU:-0.0}"
LAYER_MIXTURE_ASSIGNMENT_TEMPERATURE="${LAYER_MIXTURE_ASSIGNMENT_TEMPERATURE:-1.0}"
LAYER_MIXTURE_GATE_HIDDEN="${LAYER_MIXTURE_GATE_HIDDEN:-64}"
LAYER_MIXTURE_LR="${LAYER_MIXTURE_LR:-0.0}"
LAYER_MIXTURE_COVARIANCE_TRACE_NORMALIZE="${LAYER_MIXTURE_COVARIANCE_TRACE_NORMALIZE:-False}"
LAYER_MIXTURE_DELTA_L2="${LAYER_MIXTURE_DELTA_L2:-0.0}"
PASS1_STEPS="${PASS1_STEPS:-12000}"
PASS1_LR="${PASS1_LR:-5e-5}"
PASS1_LR_BANK="${PASS1_LR_BANK:-4e-5}"
PASS1_LR_ADAPTER="${PASS1_LR_ADAPTER:-1e-4}"
PASS1_LR_SCHEDULE="${PASS1_LR_SCHEDULE:-warmup_cosine}"
PASS1_LR_WARMUP_STEPS="${PASS1_LR_WARMUP_STEPS:-0}"
PASS1_LR_WARMUP_RATIO="${PASS1_LR_WARMUP_RATIO:-0.1}"
PASS1_LR_MIN_RATIO="${PASS1_LR_MIN_RATIO:-0.01}"
PASS1_WEIGHT_DECAY="${PASS1_WEIGHT_DECAY:-0.01}"
PASS1_LOG_EVERY="${PASS1_LOG_EVERY:-200}"
PASS1_VAL_DATA_PATH="${PASS1_VAL_DATA_PATH:-${DATA_SPLIT_VAL_PATH}}"
PASS1_VAL_BATCH_SIZE="${PASS1_VAL_BATCH_SIZE:-0}"
PASS1_VAL_EVERY="${PASS1_VAL_EVERY:-500}"
PASS1_VAL_MAX_RECORDS="${PASS1_VAL_MAX_RECORDS:-2048}"
PASS1_VAL_MAX_BATCHES="${PASS1_VAL_MAX_BATCHES:-0}"
PASS1_VAL_SEED="${PASS1_VAL_SEED:-${SEED}}"
PASS1_VAL_SELECTION_METRIC="${PASS1_VAL_SELECTION_METRIC:-loss}"
PASS1_VAL_INCLUDE_STEP0_CANDIDATE="${PASS1_VAL_INCLUDE_STEP0_CANDIDATE:-False}"
PASS1_VAL_MIN_IMPROVEMENT="${PASS1_VAL_MIN_IMPROVEMENT:-0.0}"
PASS1_VAL_SUBSPACE_VIZ_ENABLE="${PASS1_VAL_SUBSPACE_VIZ_ENABLE:-False}"
PASS1_VAL_SUBSPACE_VIZ_MAX_POINTS_PER_REGIME="${PASS1_VAL_SUBSPACE_VIZ_MAX_POINTS_PER_REGIME:-256}"
INIT_SHARED_STUDENT_CKPT="${INIT_SHARED_STUDENT_CKPT:-}"
RESUME_TRAINING_CHECKPOINT="${RESUME_TRAINING_CHECKPOINT:-}"
RESUME_DATA_STEP="${RESUME_DATA_STEP:-0}"
PASS1_LATEST_CHECKPOINT_EVERY="${PASS1_LATEST_CHECKPOINT_EVERY:-0}"
PASS1_LATEST_CHECKPOINT_INCLUDE_OPTIMIZER="${PASS1_LATEST_CHECKPOINT_INCLUDE_OPTIMIZER:-False}"
RESIDUAL_SVD_INIT_MODE="${RESIDUAL_SVD_INIT_MODE:-none}"
RESIDUAL_SVD_INIT_RECORDS="${RESIDUAL_SVD_INIT_RECORDS:-512}"
RESIDUAL_SVD_INIT_RIDGE="${RESIDUAL_SVD_INIT_RIDGE:-1e-3}"
RESIDUAL_SVD_INIT_OVERSAMPLE="${RESIDUAL_SVD_INIT_OVERSAMPLE:-16}"
RESIDUAL_SVD_METRIC_COMPLEMENT_FLOOR="${RESIDUAL_SVD_METRIC_COMPLEMENT_FLOOR:-0.1}"

DIRECT_TRAIN_ENABLE="${DIRECT_TRAIN_ENABLE:-False}"
DIRECT_TRAIN_OUTPUT_DIR="${DIRECT_TRAIN_OUTPUT_DIR:-${OUT_ROOT}/phase1_direct_ce_kd}"
DIRECT_TRAIN_STEPS="${DIRECT_TRAIN_STEPS:-${PASS1_STEPS}}"
DIRECT_TRAIN_LR="${DIRECT_TRAIN_LR:-${PASS1_LR}}"
DIRECT_TRAIN_WEIGHT_DECAY="${DIRECT_TRAIN_WEIGHT_DECAY:-${PASS1_WEIGHT_DECAY}}"
DIRECT_TRAIN_WARMUP_STEPS="${DIRECT_TRAIN_WARMUP_STEPS:-${PASS1_LR_WARMUP_STEPS}}"
DIRECT_TRAIN_WARMUP_RATIO="${DIRECT_TRAIN_WARMUP_RATIO:-${PASS1_LR_WARMUP_RATIO}}"
DIRECT_TRAIN_LOG_EVERY="${DIRECT_TRAIN_LOG_EVERY:-${PASS1_LOG_EVERY}}"
DIRECT_TRAIN_SAVE_EVERY="${DIRECT_TRAIN_SAVE_EVERY:-0}"
DIRECT_LAMBDA_CE="${DIRECT_LAMBDA_CE:-1.0}"
DIRECT_LAMBDA_KD="${DIRECT_LAMBDA_KD:-1.0}"
DIRECT_KD_TEMPERATURE="${DIRECT_KD_TEMPERATURE:-${KD_TEMPERATURE}}"
DIRECT_GRAD_CLIP="${DIRECT_GRAD_CLIP:-1.0}"
DIRECT_GRADIENT_CHECKPOINTING="${DIRECT_GRADIENT_CHECKPOINTING:-${STUDENT_GRADIENT_CHECKPOINTING}}"
DIRECT_PRESERVE_ZERO_MASK="${DIRECT_PRESERVE_ZERO_MASK:-True}"
DIRECT_DTYPE="${DIRECT_DTYPE:-auto}"

QUANT_BITS="${QUANT_BITS:-16}"
EVAL_ENABLE="${EVAL_ENABLE:-True}"
EVAL_BASELINE_ENABLE="${EVAL_BASELINE_ENABLE:-False}"
EVAL_DATASETS="${EVAL_DATASETS:-piqa}"
EVAL_MODE="${EVAL_MODE:-logprob}"
EVAL_LENGTH_NORM="${EVAL_LENGTH_NORM:-none}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-16}"
EVAL_PRINT_SCORES="${EVAL_PRINT_SCORES:-False}"
EVAL_USE_QUANT_BANK_INT4="${EVAL_USE_QUANT_BANK_INT4:-False}"
EVAL_TRAJECTORY_ENABLE="${EVAL_TRAJECTORY_ENABLE:-False}"
EVAL_TRAJECTORY_MAX_SAMPLES="${EVAL_TRAJECTORY_MAX_SAMPLES:-32}"
EVAL_TRAJECTORY_TOKEN_RULE="${EVAL_TRAJECTORY_TOKEN_RULE:-last_content}"
EVAL_TRAJECTORY_PLOT_MAX_LINES="${EVAL_TRAJECTORY_PLOT_MAX_LINES:-16}"
EVAL_TRAJECTORY_OUTPUT_DIR="${EVAL_TRAJECTORY_OUTPUT_DIR:-${RESULTS_DIR}/trajectory}"
EXTERNAL_ATLAS_DIR="${EXTERNAL_ATLAS_DIR:-}"
ENABLE_TIME_VERBOSE="${ENABLE_TIME_VERBOSE:-True}"
TIME_BIN="${TIME_BIN:-/usr/bin/time}"

normalize_bool() {
  local v="$1"
  echo "${v}" | tr '[:upper:]' '[:lower:]'
}

numeric_is_zero() {
  local v="$1"
  awk -v v="${v}" 'BEGIN { exit ((v + 0.0) == 0.0 ? 0 : 1) }'
}

run_with_time() {
  if [ "$(normalize_bool "${ENABLE_TIME_VERBOSE}")" = "true" ] && [ -x "${TIME_BIN}" ]; then
    "${TIME_BIN}" -v "$@"
  else
    "$@"
  fi
}

find_latest_merged_teacher() {
  if [ -n "${TEACHER_MERGED_CKPT}" ] && [ -f "${TEACHER_MERGED_CKPT}/config.json" ]; then
    echo "${TEACHER_MERGED_CKPT}"
    return 0
  fi
  if [ -n "${TEACHER_CKPT}" ] && [ -f "${TEACHER_CKPT}/config.json" ]; then
    echo "${TEACHER_CKPT}"
    return 0
  fi
  return 1
}

prepare_train_val_split() {
  local src="$1"
  local train_out="$2"
  local val_out="$3"
  local ratio="$4"
  local seed="$5"
  local manifest_out="$6"
  "${PYTHON_BIN}" - "${src}" "${train_out}" "${val_out}" "${ratio}" "${seed}" "${manifest_out}" <<'PY'
import json
import os
import random
import sys

src, train_out, val_out, ratio_s, seed_s, manifest_out = sys.argv[1:7]
ratio = float(ratio_s)
seed = int(seed_s)
with open(src, "r", encoding="utf-8") as f:
    payload = json.load(f)
records = payload if isinstance(payload, list) else payload.get("data", [])
if not isinstance(records, list) or len(records) < 2:
    raise ValueError(f"unsupported or too-small dataset: {src}")
idx = list(range(len(records)))
rng = random.Random(seed)
rng.shuffle(idx)
n_train = max(1, min(len(records) - 1, int(round(len(records) * ratio))))
train_records = [records[i] for i in idx[:n_train]]
val_records = [records[i] for i in idx[n_train:]]
os.makedirs(os.path.dirname(os.path.abspath(train_out)) or ".", exist_ok=True)
with open(train_out, "w", encoding="utf-8") as f:
    json.dump(train_records, f, ensure_ascii=False)
with open(val_out, "w", encoding="utf-8") as f:
    json.dump(val_records, f, ensure_ascii=False)
with open(manifest_out, "w", encoding="utf-8") as f:
    json.dump({
        "source_path": os.path.abspath(src),
        "train_path": os.path.abspath(train_out),
        "val_path": os.path.abspath(val_out),
        "train_ratio": ratio,
        "seed": seed,
        "total_records": len(records),
        "train_records": len(train_records),
        "val_records": len(val_records),
    }, f, ensure_ascii=False, indent=2)
print(f"[Split] train={len(train_records)} val={len(val_records)} source={src}")
PY
}

ensure_train_val_split() {
  if [ "$(normalize_bool "${DATA_SPLIT_ENABLE}")" != "true" ]; then
    return 0
  fi
  if [ -f "${DATA_SPLIT_TRAIN_PATH}" ] && [ -f "${DATA_SPLIT_VAL_PATH}" ]; then
    return 0
  fi
  echo "[FINAL][Warn] expected split files missing; regenerating train/val split"
  prepare_train_val_split "${SOURCE_DATA_PATH}" "${DATA_SPLIT_TRAIN_PATH}" "${DATA_SPLIT_VAL_PATH}" "${DATA_SPLIT_TRAIN_RATIO}" "${SEED}" "${DATA_SPLIT_MANIFEST_PATH}"
}

resolve_local_hf_snapshot() {
  local repo_id="$1"
  local cache_name="${repo_id//\//--}"
  local root=""
  local ref=""
  local candidate=""
  for root in \
    "${HF_HOME:-}/hub/models--${cache_name}" \
    "${HF_HUB_CACHE:-}/models--${cache_name}" \
    "${HOME:-}/.cache/huggingface/hub/models--${cache_name}"
  do
    if [ -z "${root}" ] || [ ! -d "${root}" ]; then
      continue
    fi
    if [ -f "${root}/refs/main" ]; then
      ref="$(tr -d '\r\n' < "${root}/refs/main")"
      candidate="${root}/snapshots/${ref}"
      if [ -n "${ref}" ] && [ -f "${candidate}/config.json" ]; then
        echo "${candidate}"
        return 0
      fi
    fi
    candidate="$(find "${root}/snapshots" -mindepth 1 -maxdepth 1 -type d -exec test -f '{}/config.json' ';' -print 2>/dev/null | head -n 1 || true)"
    if [ -n "${candidate}" ]; then
      echo "${candidate}"
      return 0
    fi
  done
  return 1
}

resolve_flap_model() {
  if [ -n "${FLAP_MODEL}" ]; then
    echo "${FLAP_MODEL}"
    return 0
  fi
  if [ -d "${ORIGINAL_BASE_MODEL}" ] && [ -f "${ORIGINAL_BASE_MODEL}/config.json" ]; then
    echo "${ORIGINAL_BASE_MODEL}"
    return 0
  fi
  if [ "${ORIGINAL_BASE_MODEL}" = "meta-llama/Llama-3.2-3B" ]; then
    local_snapshot="$(resolve_local_hf_snapshot "${ORIGINAL_BASE_MODEL}" || true)"
    if [ -n "${local_snapshot}" ]; then
      echo "${local_snapshot}"
      return 0
    fi
  fi
  echo "${ORIGINAL_BASE_MODEL}"
}

run_flap_phase0_if_needed() {
  if [ "$(normalize_bool "${FLAP_ENABLE}")" != "true" ]; then
    echo "[FINAL] FLAP phase0 disabled; student base_model=${BASE_MODEL}"
    return 0
  fi
  FLAP_MODEL="$(resolve_flap_model)"
  if [ ! -f "${FLAP_MAIN_PY}" ]; then
    echo "[FINAL][Error] FLAP main.py not found: ${FLAP_MAIN_PY}" >&2
    exit 1
  fi
  if [ ! -f "${FLAP_CALIB_DATA_PATH}" ]; then
    echo "[FINAL][Error] FLAP calibration data not found: ${FLAP_CALIB_DATA_PATH}" >&2
    exit 1
  fi
  if [ "$(normalize_bool "${FLAP_REUSE_IF_EXISTS}")" = "true" ] && [ -f "${FLAP_OUTPUT_DIR}/config.json" ]; then
    echo "[FINAL] reusing FLAP-compressed student: ${FLAP_OUTPUT_DIR}"
  else
    mkdir -p "$(dirname "${FLAP_OUTPUT_DIR}")"
    echo "[FINAL] running FLAP phase0"
    echo "[FINAL]   flap_model=${FLAP_MODEL}"
    echo "[FINAL]   flap_output=${FLAP_OUTPUT_DIR}"
    echo "[FINAL]   ratio=${FLAP_PRUNING_RATIO} scope=${FLAP_PRUNE_SCOPE} metrics=${FLAP_METRICS} structure=${FLAP_STRUCTURE}"
    flap_help="$("${PYTHON_BIN}" "${FLAP_MAIN_PY}" --help 2>&1 || true)"
    flap_args=(
      "${PYTHON_BIN}" "${FLAP_MAIN_PY}"
      --model "${FLAP_MODEL}"
      --seed "${SEED}"
      --prune_method flap
      --pruning_ratio "${FLAP_PRUNING_RATIO}"
      --remove_heads "${FLAP_REMOVE_HEADS}"
      --metrics "${FLAP_METRICS}"
      --structure "${FLAP_STRUCTURE}"
      --nsamples "${FLAP_NSAMPLES}"
      --cache_dir "${FLAP_CACHE_DIR}"
      --save_model "${FLAP_OUTPUT_DIR}"
    )
    if grep -q -- "--model_loader" <<<"${flap_help}"; then
      flap_args+=(--model_loader "${FLAP_MODEL_LOADER}")
    else
      echo "[FINAL][Warn] FLAP CLI has no --model_loader; using legacy argument set."
    fi
    if grep -q -- "--prune_scope" <<<"${flap_help}"; then
      flap_args+=(--prune_scope "${FLAP_PRUNE_SCOPE}")
    fi
    if grep -q -- "--calib_dataset" <<<"${flap_help}"; then
      flap_args+=(--calib_dataset "${FLAP_CALIB_DATASET}")
    fi
    if grep -q -- "--calib_data_path" <<<"${flap_help}"; then
      flap_args+=(--calib_data_path "${FLAP_CALIB_DATA_PATH}")
    fi
    if grep -q -- "--seqlen" <<<"${flap_help}"; then
      flap_args+=(--seqlen "${FLAP_SEQLEN}")
    fi
    if grep -q -- "--trust_remote_code" <<<"${flap_help}"; then
      flap_args+=(--trust_remote_code "${TRUST_REMOTE_CODE}")
    fi
    if [ "$(normalize_bool "${FLAP_UNSTR}")" = "true" ]; then
      flap_args+=(--unstr)
    fi
    run_with_time "${flap_args[@]}"
  fi
  if [ ! -f "${FLAP_OUTPUT_DIR}/config.json" ]; then
    echo "[FINAL][Error] FLAP output is not a HuggingFace model dir: ${FLAP_OUTPUT_DIR}" >&2
    exit 1
  fi
  BASE_MODEL="${FLAP_OUTPUT_DIR}"
  echo "[FINAL] student initialized from FLAP-compressed base: ${BASE_MODEL}"
}

run_single() {
  run_with_time "${PYTHON_BIN}" "${PIPELINE_PY}" "$@"
}

run_distributed() {
  if [ "${NUM_GPUS}" -le 1 ]; then
    run_single "$@"
    return 0
  fi
  if [ -n "${TORCHRUN_BIN}" ]; then
    run_with_time "${TORCHRUN_BIN}" --nnodes 1 --nproc_per_node "${NUM_GPUS}" --master_addr "${MASTER_ADDR}" --master_port "${MASTER_PORT}" "${PIPELINE_PY}" "$@"
  else
    run_with_time "${PYTHON_BIN}" -m torch.distributed.run --nnodes 1 --nproc_per_node "${NUM_GPUS}" --master_addr "${MASTER_ADDR}" --master_port "${MASTER_PORT}" "${PIPELINE_PY}" "$@"
  fi
}

if [ ! -f "${PIPELINE_PY}" ]; then
  echo "[FINAL][Error] pipeline script not found: ${PIPELINE_PY}" >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}" "${RESULTS_DIR}" "${SHARED_SPLIT_DIR}"

if [ ! -f "${SOURCE_DATA_PATH}" ] && [ -f "${SOURCE_DATA_FALLBACK}" ]; then
  SOURCE_DATA_PATH="${SOURCE_DATA_FALLBACK}"
fi
if [ ! -f "${SOURCE_DATA_PATH}" ]; then
  echo "[FINAL][Error] source data not found: ${SOURCE_DATA_PATH}" >&2
  exit 1
fi

teacher_use_merged_lower="$(normalize_bool "${TEACHER_USE_MERGED}")"
if [ "${teacher_use_merged_lower}" = "auto" ] || [ "${teacher_use_merged_lower}" = "true" ]; then
  resolved_merged="$(find_latest_merged_teacher || true)"
  if [ -n "${resolved_merged}" ]; then
    TEACHER_CKPT="${resolved_merged}"
    TEACHER_LOADER="native"
    if [ -z "${TOKENIZER_NAME_OR_PATH}" ] || [ "${TOKENIZER_NAME_OR_PATH}" = "${BASE_MODEL}" ]; then
      TOKENIZER_NAME_OR_PATH="${resolved_merged}"
    fi
    echo "[FINAL] using merged teacher: ${resolved_merged}"
  elif [ "${teacher_use_merged_lower}" = "true" ]; then
    echo "[FINAL][Warn] TEACHER_USE_MERGED=True but no merged teacher was found; falling back to TEACHER_CKPT=${TEACHER_CKPT}" >&2
  fi
fi

run_flap_phase0_if_needed

if [ "$(normalize_bool "${DATA_SPLIT_ENABLE}")" = "true" ] && [ -z "${DATA_PATH}" ]; then
  echo "[FINAL] preparing train/val split"
  ensure_train_val_split
  DATA_PATH="${DATA_SPLIT_TRAIN_PATH}"
fi
if [ -z "${DATA_PATH}" ]; then
  DATA_PATH="${SOURCE_DATA_PATH}"
fi
if [ -z "${PASS1_VAL_DATA_PATH}" ]; then
  PASS1_VAL_DATA_PATH="${DATA_SPLIT_VAL_PATH}"
fi

ATLAS_DIR="${OUT_ROOT}/phase1_atlas"
if [ -n "${EXTERNAL_ATLAS_DIR}" ]; then
  ATLAS_DIR="${EXTERNAL_ATLAS_DIR}"
fi
PASS1_DIR="${OUT_ROOT}/phase1_5_pass1"
EXPORT_DIR="${OUT_ROOT}/phase4_export"

ATLAS_PATH="${ATLAS_DIR}/atlas_state.pt"
PASS1_CKPT="${PASS1_DIR}/shared_student.pt"
DEPLOY_BUNDLE="${EXPORT_DIR}/deploy_bundle.pt"

mkdir -p "${OUT_ROOT}" "${RESULTS_DIR}" "${SHARED_SPLIT_DIR}" "${EVAL_TRAJECTORY_OUTPUT_DIR}"

if [ "$(normalize_bool "${DATA_SPLIT_ENABLE}")" = "true" ]; then
  if [ "${DATA_PATH}" = "${DATA_SPLIT_TRAIN_PATH}" ] || [ "${PASS1_VAL_DATA_PATH}" = "${DATA_SPLIT_VAL_PATH}" ]; then
    ensure_train_val_split
  fi
fi

echo "[FINAL] run_tag=${RUN_TAG}"
echo "[FINAL] out_root=${OUT_ROOT}"
echo "[FINAL] data_path=${DATA_PATH}"
echo "[FINAL] base_model=${BASE_MODEL}"
echo "[FINAL] original_base_model=${ORIGINAL_BASE_MODEL}"
echo "[FINAL] flap_enable=${FLAP_ENABLE} flap_output=${FLAP_OUTPUT_DIR}"
echo "[FINAL] teacher_ckpt=${TEACHER_CKPT}"
echo "[FINAL] teacher_deploy_bundle=${TEACHER_DEPLOY_BUNDLE:-none}"
echo "[FINAL] num_gpus=${NUM_GPUS} batch_size_per_rank=${BATCH_SIZE} grad_accum_steps=${PASS1_GRAD_ACCUM_STEPS} effective_global_batch=$((NUM_GPUS * BATCH_SIZE * PASS1_GRAD_ACCUM_STEPS))"
echo "[FINAL] direct_train_enable=${DIRECT_TRAIN_ENABLE}"
if [ "$(normalize_bool "${DIRECT_TRAIN_ENABLE}")" = "true" ]; then
  if numeric_is_zero "${DIRECT_LAMBDA_KD}"; then
    DIRECT_OBJECTIVE_LABEL="CE"
    DIRECT_RESULT_PREFIX="direct_ce"
  else
    DIRECT_OBJECTIVE_LABEL="CE+KD"
    DIRECT_RESULT_PREFIX="direct_ce_kd"
  fi
  echo "[FINAL] objective=${DIRECT_OBJECTIVE_LABEL} lambda_ce=${DIRECT_LAMBDA_CE} lambda_kd=${DIRECT_LAMBDA_KD} kd_temperature=${DIRECT_KD_TEMPERATURE}"
  echo "[FINAL] atlas_enable=False core_loss=False"
else
  echo "[FINAL] pass1 distill_mode=${PASS1_DISTILL_MODE} lambda_ce=${LAMBDA_CE} lambda_kd=${LAMBDA_KD} lambda_hidden_mse=${LAMBDA_HIDDEN_MSE} kd_temperature=${KD_TEMPERATURE}"
  echo "[FINAL] pass1 lambda_core=${LAMBDA_CORE}"
  echo "[FINAL] core_use_metric_whitening=${CORE_USE_METRIC_WHITENING} core_metric_trace_normalize=${CORE_METRIC_TRACE_NORMALIZE} core_metric_diag_mode=${CORE_METRIC_DIAG_MODE} core_metric_diag_path=${CORE_METRIC_DIAG_PATH:-atlas_default} core_use_reliability_weighting=${CORE_USE_RELIABILITY_WEIGHTING}"
  echo "[FINAL] core_use_information_weighting=${CORE_USE_INFORMATION_WEIGHTING:-False} core_information_power=${CORE_INFORMATION_POWER:-1.0} lambda_geodesic_core=${LAMBDA_GEODESIC_CORE:-0.0} lambda_relational_core=${LAMBDA_RELATIONAL_CORE} lambda_variance_core=${LAMBDA_VARIANCE_CORE} lambda_manifold_core=${LAMBDA_MANIFOLD_CORE:-0.0} manifold_temperature=${MANIFOLD_CORE_TEMPERATURE:-1.0}"
  echo "[FINAL] generic_replay_path=${GENERIC_REPLAY_DATA_PATH:-disabled} interval=${GENERIC_REPLAY_INTERVAL} batch_size=${GENERIC_REPLAY_BATCH_SIZE} lambda=${LAMBDA_GENERIC_REPLAY}"
  echo "[FINAL] core_token_selection=${CORE_TOKEN_SELECTION} candidate_tokens=${CORE_CANDIDATE_TOKENS} iets_temperature=${CORE_IETS_TEMPERATURE} anchor_boost=${CORE_IETS_ANCHOR_BOOST}"
  echo "[FINAL] sharing_policy_mode=${SHARING_POLICY_MODE} upstream_similarity_threshold=${UPSTREAM_SIMILARITY_THRESHOLD}"
  if [ "${TARGET_PROTO_COUNT}" != "0" ]; then
    echo "[FINAL] target_proto_count=${TARGET_PROTO_COUNT}"
  fi
  if [ "$(normalize_bool "${FUNCTIONAL_POLICY_ENABLE}")" = "true" ]; then
    echo "[FINAL] functional_policy_enable=True obs_dir=${FUNCTIONAL_OBS_DIR} max_bidirectional_cost=${FUNCTIONAL_POLICY_MAX_BIDIRECTIONAL_COST} max_layer_gap=${FUNCTIONAL_POLICY_MAX_LAYER_GAP}"
    echo "[FINAL] functional_policy_target_saved_mlps=${FUNCTIONAL_POLICY_TARGET_SAVED_MLPS} target_compression_ratio=${FUNCTIONAL_POLICY_TARGET_COMPRESSION_RATIO}"
  fi
fi
if [ -n "${EXTERNAL_ATLAS_DIR}" ]; then
  echo "[FINAL] external_atlas_dir=${EXTERNAL_ATLAS_DIR}"
fi

if [ "$(normalize_bool "${DIRECT_TRAIN_ENABLE}")" = "true" ]; then
  if [ ! -f "${DIRECT_TRAIN_PY}" ]; then
    echo "[FINAL][Error] direct trainer script not found: ${DIRECT_TRAIN_PY}" >&2
    exit 1
  fi
  if numeric_is_zero "${DIRECT_LAMBDA_KD}"; then
    DIRECT_RESULT_PREFIX="direct_ce"
    echo "[FINAL] skipping atlas/pass1; starting direct CE training"
  else
    DIRECT_RESULT_PREFIX="direct_ce_kd"
    echo "[FINAL] skipping atlas/pass1; starting direct CE+KD training"
  fi
  direct_train_args=(
    --student_model "${BASE_MODEL}"
    --teacher_ckpt "${TEACHER_CKPT}"
    --tokenizer_name_or_path "${TOKENIZER_NAME_OR_PATH}"
    --data_path "${DATA_PATH}"
    --output_dir "${DIRECT_TRAIN_OUTPUT_DIR}"
    --trust_remote_code "${TRUST_REMOTE_CODE}"
    --dtype "${DIRECT_DTYPE}"
    --device "${DEVICE}"
    --batch_size "${BATCH_SIZE}"
    --cutoff_len "${CUTOFF_LEN}"
    --max_records "${MAX_RECORDS}"
    --shuffle_records "${SHUFFLE_RECORDS}"
    --seed "${SEED}"
    --steps "${DIRECT_TRAIN_STEPS}"
    --lr "${DIRECT_TRAIN_LR}"
    --weight_decay "${DIRECT_TRAIN_WEIGHT_DECAY}"
    --warmup_steps "${DIRECT_TRAIN_WARMUP_STEPS}"
    --warmup_ratio "${DIRECT_TRAIN_WARMUP_RATIO}"
    --lambda_ce "${DIRECT_LAMBDA_CE}"
    --lambda_kd "${DIRECT_LAMBDA_KD}"
    --kd_temperature "${DIRECT_KD_TEMPERATURE}"
    --grad_clip "${DIRECT_GRAD_CLIP}"
    --log_every "${DIRECT_TRAIN_LOG_EVERY}"
    --save_every "${DIRECT_TRAIN_SAVE_EVERY}"
    --gradient_checkpointing "${DIRECT_GRADIENT_CHECKPOINTING}"
    --preserve_zero_mask "${DIRECT_PRESERVE_ZERO_MASK}"
  )
  run_with_time "${PYTHON_BIN}" "${DIRECT_TRAIN_PY}" "${direct_train_args[@]}"

  if [ "$(normalize_bool "${EVAL_ENABLE}")" = "true" ]; then
    read -r -a DATASET_ARR <<< "${EVAL_DATASETS}"
    for ds in "${DATASET_ARR[@]}"; do
      eval_args=()
      if [ "$(normalize_bool "${EVAL_PRINT_SCORES}")" = "true" ]; then
        eval_args+=(--print_scores)
      fi
      direct_eval_args=(
        eval
        --model_variant baseline
        --base_model "${DIRECT_TRAIN_OUTPUT_DIR}"
        --tokenizer_name_or_path "${DIRECT_TRAIN_OUTPUT_DIR}"
        --trust_remote_code "${TRUST_REMOTE_CODE}"
        --dataset "${ds}"
        --test_data_root "${TEST_DATA_ROOT}"
        --batch_size "${EVAL_BATCH_SIZE}"
        --eval_mode "${EVAL_MODE}"
        --length_norm "${EVAL_LENGTH_NORM}"
        --max_new_tokens "${EVAL_MAX_NEW_TOKENS}"
        --device "${DEVICE}"
        --seed "${SEED}"
        --output_json "${RESULTS_DIR}/${DIRECT_RESULT_PREFIX}_${ds}.json"
        "${eval_args[@]}"
      )
      run_distributed "${direct_eval_args[@]}"
    done
  fi

  echo "[FINAL] done"
  echo "  - ${DIRECT_TRAIN_OUTPUT_DIR}"
  echo "  - ${RESULTS_DIR}"
  exit 0
fi

if [ -z "${EXTERNAL_ATLAS_DIR}" ]; then
  atlas_args=(
    atlas
    --output_dir "${ATLAS_DIR}"
    --base_model "${BASE_MODEL}"
    --teacher_ckpt "${TEACHER_CKPT}"
    --teacher_loader "${TEACHER_LOADER}"
    --tokenizer_name_or_path "${TOKENIZER_NAME_OR_PATH}"
    --trust_remote_code "${TRUST_REMOTE_CODE}"
    --data_path "${DATA_PATH}"
    --batch_size "${BATCH_SIZE}"
    --cutoff_len "${CUTOFF_LEN}"
    --max_records "${MAX_RECORDS}"
    --max_batches "${MAX_BATCHES}"
    --shuffle_records "${SHUFFLE_RECORDS}"
    --seed "${SEED}"
    --device "${DEVICE}"
    --analysis_rank "${ANALYSIS_RANK}"
    --projection_basis_source "${ATLAS_PROJECTION_BASIS_SOURCE}"
    --random_basis_seed "${RANDOM_BASIS_SEED}"
    --pca_mode "${PCA_MODE}"
    --pca_device "${PCA_DEVICE}"
    --pca_stream_chunk_size "${PCA_STREAM_CHUNK_SIZE}"
    --reservoir_size "${RESERVOIR_SIZE}"
    --num_codes "${NUM_CODES}"
    --kmeans_mode "${KMEANS_MODE}"
    --kmeans_iters "${KMEANS_ITERS}"
    --kmeans_batch_size "${KMEANS_BATCH_SIZE}"
    --kmeans_warmup_size "${KMEANS_WARMUP_SIZE}"
    --kmeans_warmup_iters "${KMEANS_WARMUP_ITERS}"
    --kmeans_refine_iters "${KMEANS_REFINE_ITERS}"
    --kmeans_assign_chunk_size "${KMEANS_ASSIGN_CHUNK_SIZE}"
    --token_rule "${TOKEN_RULE}"
    --tau_eps "${TAU_EPS}"
    --tau_nmin "${TAU_NMIN}"
    --tau_shrink_lambda "${TAU_SHRINK_LAMBDA}"
    --sharing_policy_mode "${SHARING_POLICY_MODE}"
    --upstream_similarity_threshold "${UPSTREAM_SIMILARITY_THRESHOLD}"
    --ckpt_every_batches "${ATLAS_CKPT_EVERY_BATCHES}"
  )
  run_distributed "${atlas_args[@]}"
else
  if [ ! -f "${ATLAS_PATH}" ]; then
    echo "[FINAL][Error] external atlas_state missing: ${ATLAS_PATH}" >&2
    exit 1
  fi
  if [ ! -f "${SHARING_POLICY_PATH}" ]; then
    echo "[FINAL][Warn] sharing policy not found at ${SHARING_POLICY_PATH}; final will fall back to per-layer groups." >&2
  fi
  if [ ! -f "${ATLAS_DIR}/final_structure_prior.pt" ]; then
    echo "[FINAL][Error] final_structure_prior.pt not found under ${ATLAS_DIR}; pass1 now requires prior layer_metric_diag." >&2
    exit 1
  fi
  echo "[FINAL] skipping atlas stage and reusing precompiled atlas artifacts"
fi

if [ "${TARGET_PROTO_COUNT}" != "0" ]; then
  RETARGET_SHARING_PY="${RETARGET_SHARING_PY:-${PROJECT_ROOT}/core/retarget_sharing_policy_final_llama.py}"
  if [ ! -f "${RETARGET_SHARING_PY}" ]; then
    echo "[FINAL][Error] retarget sharing script not found: ${RETARGET_SHARING_PY}" >&2
    exit 1
  fi
  if [ ! -f "${SHARING_POLICY_PATH}" ]; then
    echo "[FINAL][Error] sharing policy not found for retarget: ${SHARING_POLICY_PATH}" >&2
    exit 1
  fi
  echo "[FINAL] retargeting sharing policy to ${TARGET_PROTO_COUNT} prototypes"
  run_with_time "${PYTHON_BIN}" "${RETARGET_SHARING_PY}" \
    --policy "${SHARING_POLICY_PATH}" \
    --target-proto-count "${TARGET_PROTO_COUNT}"
fi

if [ "$(normalize_bool "${FUNCTIONAL_POLICY_ENABLE}")" = "true" ]; then
  if [ ! -f "${FUNCTIONAL_POLICY_PY}" ]; then
    echo "[FINAL][Error] functional policy script not found: ${FUNCTIONAL_POLICY_PY}" >&2
    exit 1
  fi
  if [ ! -d "${FUNCTIONAL_OBS_DIR}" ]; then
    echo "[FINAL][Error] functional observation dir not found: ${FUNCTIONAL_OBS_DIR}" >&2
    exit 1
  fi
  if [ ! -f "${SHARING_POLICY_PATH}" ]; then
    echo "[FINAL][Error] sharing policy not found for functional retarget: ${SHARING_POLICY_PATH}" >&2
    exit 1
  fi
  functional_policy_args=(
    --policy "${SHARING_POLICY_PATH}"
    --obs-dir "${FUNCTIONAL_OBS_DIR}"
    --max-bidirectional-cost "${FUNCTIONAL_POLICY_MAX_BIDIRECTIONAL_COST}"
    --max-layer-gap "${FUNCTIONAL_POLICY_MAX_LAYER_GAP}"
    --target-saved-mlps "${FUNCTIONAL_POLICY_TARGET_SAVED_MLPS}"
    --target-compression-ratio "${FUNCTIONAL_POLICY_TARGET_COMPRESSION_RATIO}"
    --saved-mlp-whole-model-ratio "${FUNCTIONAL_POLICY_SAVED_MLP_WHOLE_MODEL_RATIO}"
    --max-group-size "${FUNCTIONAL_POLICY_MAX_GROUP_SIZE}"
  )
  if [ -n "${FUNCTIONAL_POLICY_ELIGIBLE_LAYERS}" ]; then
    functional_policy_args+=(--eligible-layers "${FUNCTIONAL_POLICY_ELIGIBLE_LAYERS}")
  fi
  if [ "$(normalize_bool "${FUNCTIONAL_POLICY_REQUIRE_SAME_REGIME}")" = "true" ]; then
    functional_policy_args+=(--require-same-regime)
  else
    functional_policy_args+=(--no-require-same-regime)
  fi
  echo "[FINAL] retargeting sharing policy from measured functional intervention costs"
  run_with_time "${PYTHON_BIN}" "${FUNCTIONAL_POLICY_PY}" "${functional_policy_args[@]}"
fi

if [ "$(normalize_bool "${DATA_SPLIT_ENABLE}")" = "true" ]; then
  if [ "${DATA_PATH}" = "${DATA_SPLIT_TRAIN_PATH}" ] || [ "${PASS1_VAL_DATA_PATH}" = "${DATA_SPLIT_VAL_PATH}" ]; then
    ensure_train_val_split
  fi
fi

pass1_args=(
  pass1
  --atlas_path "${ATLAS_PATH}"
  --output_dir "${PASS1_DIR}"
  --base_model "${BASE_MODEL}"
  --teacher_ckpt "${TEACHER_CKPT}"
  --teacher_loader "${TEACHER_LOADER}"
  --tokenizer_name_or_path "${TOKENIZER_NAME_OR_PATH}"
  --trust_remote_code "${TRUST_REMOTE_CODE}"
  --student_gradient_checkpointing "${STUDENT_GRADIENT_CHECKPOINTING}"
  --teacher_gradient_checkpointing "${TEACHER_GRADIENT_CHECKPOINTING}"
  --data_path "${DATA_PATH}"
  --training_prompt_mode "${TRAINING_PROMPT_MODE}"
  --batch_size "${BATCH_SIZE}"
  --cutoff_len "${CUTOFF_LEN}"
  --max_records "${MAX_RECORDS}"
  --max_batches "${MAX_BATCHES}"
  --shuffle_records "${SHUFFLE_RECORDS}"
  --seed "${SEED}"
  --device "${DEVICE}"
  --num_gpus "${NUM_GPUS}"
  --master_addr "${MASTER_ADDR}"
  --master_port "${MASTER_PORT}"
  --sharing_policy_path "${SHARING_POLICY_PATH}"
  --private_down_rank "${PRIVATE_DOWN_RANK}"
  --private_down_alpha "${PRIVATE_DOWN_ALPHA}"
  --proto_seed_strategy "${PROTO_SEED_STRATEGY}"
  --steps "${PASS1_STEPS}"
  --grad_accum_steps "${PASS1_GRAD_ACCUM_STEPS}"
  --lr "${PASS1_LR}"
  --lr_bank "${PASS1_LR_BANK}"
  --lr_adapter "${PASS1_LR_ADAPTER}"
  --lr_schedule "${PASS1_LR_SCHEDULE}"
  --lr_warmup_steps "${PASS1_LR_WARMUP_STEPS}"
  --lr_warmup_ratio "${PASS1_LR_WARMUP_RATIO}"
  --lr_min_ratio "${PASS1_LR_MIN_RATIO}"
  --weight_decay "${PASS1_WEIGHT_DECAY}"
  --distill_mode "${PASS1_DISTILL_MODE}"
  --lambda_ce "${LAMBDA_CE}"
  --lambda_kd "${LAMBDA_KD}"
  --kd_temperature "${KD_TEMPERATURE}"
  --lambda_hidden_mse "${LAMBDA_HIDDEN_MSE}"
  --loss_scope "${LOSS_SCOPE}"
  --loss_exclude_eos "${LOSS_EXCLUDE_EOS}"
  --lambda_core "${LAMBDA_CORE}"
  --core_layers "all_shared_layers"
  --core_metric_eps "${TAU_EPS}"
  --core_use_metric_whitening "${CORE_USE_METRIC_WHITENING}"
  --core_metric_trace_normalize "${CORE_METRIC_TRACE_NORMALIZE}"
  --core_metric_diag_path "${CORE_METRIC_DIAG_PATH}"
  --core_metric_diag_mode "${CORE_METRIC_DIAG_MODE}"
  --core_use_reliability_weighting "${CORE_USE_RELIABILITY_WEIGHTING}"
  --core_token_selection "${CORE_TOKEN_SELECTION}"
  --core_candidate_tokens "${CORE_CANDIDATE_TOKENS}"
  --core_iets_temperature "${CORE_IETS_TEMPERATURE}"
  --core_iets_anchor_boost "${CORE_IETS_ANCHOR_BOOST}"
  --core_iets_energy_alpha "${CORE_IETS_ENERGY_ALPHA}"
  --core_iets_entropy_beta "${CORE_IETS_ENTROPY_BETA}"
  --core_iets_topk "${CORE_IETS_TOPK}"
  --core_use_information_weighting "${CORE_USE_INFORMATION_WEIGHTING:-False}"
  --core_information_power "${CORE_INFORMATION_POWER:-1.0}"
  --lambda_layer_mixture "${LAMBDA_LAYER_MIXTURE}"
  --layer_mixture_entropy_tau "${LAYER_MIXTURE_ENTROPY_TAU}"
  --layer_mixture_assignment_temperature "${LAYER_MIXTURE_ASSIGNMENT_TEMPERATURE}"
  --layer_mixture_gate_hidden "${LAYER_MIXTURE_GATE_HIDDEN}"
  --layer_mixture_lr "${LAYER_MIXTURE_LR}"
  --layer_mixture_covariance_trace_normalize "${LAYER_MIXTURE_COVARIANCE_TRACE_NORMALIZE}"
  --layer_mixture_delta_l2 "${LAYER_MIXTURE_DELTA_L2}"
  --lambda_geodesic_core "${LAMBDA_GEODESIC_CORE:-0.0}"
  --geodesic_core_max_layer_gap "${GEODESIC_CORE_MAX_LAYER_GAP:-1}"
  --lambda_relational_core "${LAMBDA_RELATIONAL_CORE}"
  --lambda_variance_core "${LAMBDA_VARIANCE_CORE}"
  --variance_core_floor_ratio "${VARIANCE_CORE_FLOOR_RATIO}"
  --generic_replay_data_path "${GENERIC_REPLAY_DATA_PATH}"
  --generic_replay_interval "${GENERIC_REPLAY_INTERVAL}"
  --generic_replay_batch_size "${GENERIC_REPLAY_BATCH_SIZE}"
  --generic_replay_max_records "${GENERIC_REPLAY_MAX_RECORDS}"
  --lambda_generic_replay "${LAMBDA_GENERIC_REPLAY}"
  --lambda_manifold_core "${LAMBDA_MANIFOLD_CORE:-0.0}"
  --manifold_core_temperature "${MANIFOLD_CORE_TEMPERATURE:-1.0}"
  --lambda_delta_manifold_core "${LAMBDA_DELTA_MANIFOLD_CORE:-0.0}"
  --delta_manifold_core_temperature "${DELTA_MANIFOLD_CORE_TEMPERATURE:-1.0}"
  --delta_manifold_risk_weight "${DELTA_MANIFOLD_RISK_WEIGHT:-0.25}"
  --grad_clip "1.0"
  --log_every "${PASS1_LOG_EVERY}"
  --val_data_path "${PASS1_VAL_DATA_PATH}"
  --val_batch_size "${PASS1_VAL_BATCH_SIZE}"
  --val_every "${PASS1_VAL_EVERY}"
  --val_max_records "${PASS1_VAL_MAX_RECORDS}"
  --val_max_batches "${PASS1_VAL_MAX_BATCHES}"
  --val_seed "${PASS1_VAL_SEED}"
  --val_selection_metric "${PASS1_VAL_SELECTION_METRIC}"
  --val_include_step0_candidate "${PASS1_VAL_INCLUDE_STEP0_CANDIDATE}"
  --val_min_improvement "${PASS1_VAL_MIN_IMPROVEMENT}"
  --val_subspace_viz_enable "${PASS1_VAL_SUBSPACE_VIZ_ENABLE}"
  --val_subspace_viz_max_points_per_regime "${PASS1_VAL_SUBSPACE_VIZ_MAX_POINTS_PER_REGIME}"
  --resume_data_step "${RESUME_DATA_STEP}"
  --latest_checkpoint_every "${PASS1_LATEST_CHECKPOINT_EVERY}"
  --latest_checkpoint_include_optimizer "${PASS1_LATEST_CHECKPOINT_INCLUDE_OPTIMIZER}"
  --residual_svd_init_mode "${RESIDUAL_SVD_INIT_MODE}"
  --residual_svd_init_records "${RESIDUAL_SVD_INIT_RECORDS}"
  --residual_svd_init_ridge "${RESIDUAL_SVD_INIT_RIDGE}"
  --residual_svd_init_oversample "${RESIDUAL_SVD_INIT_OVERSAMPLE}"
  --residual_svd_metric_complement_floor "${RESIDUAL_SVD_METRIC_COMPLEMENT_FLOOR}"
)
if [ -n "${TEACHER_DEPLOY_BUNDLE}" ]; then
  if [ ! -f "${TEACHER_DEPLOY_BUNDLE}" ]; then
    echo "[FINAL][Error] teacher deploy bundle missing: ${TEACHER_DEPLOY_BUNDLE}" >&2
    exit 1
  fi
  pass1_args+=(--teacher_deploy_bundle "${TEACHER_DEPLOY_BUNDLE}")
fi
if [ -n "${RESUME_TRAINING_CHECKPOINT}" ] && [ -n "${INIT_SHARED_STUDENT_CKPT}" ]; then
  echo "[FINAL][Error] set only one of RESUME_TRAINING_CHECKPOINT or INIT_SHARED_STUDENT_CKPT" >&2
  exit 1
fi
if [ -n "${RESUME_TRAINING_CHECKPOINT}" ]; then
  if [ ! -f "${RESUME_TRAINING_CHECKPOINT}" ]; then
    echo "[FINAL][Error] resume training checkpoint missing: ${RESUME_TRAINING_CHECKPOINT}" >&2
    exit 1
  fi
  pass1_args+=(--resume_training_checkpoint "${RESUME_TRAINING_CHECKPOINT}")
  echo "[FINAL] periodic resume checkpoint=${RESUME_TRAINING_CHECKPOINT}"
elif [ -n "${INIT_SHARED_STUDENT_CKPT}" ]; then
  if [ ! -f "${INIT_SHARED_STUDENT_CKPT}" ]; then
    echo "[FINAL][Error] init shared student checkpoint missing: ${INIT_SHARED_STUDENT_CKPT}" >&2
    exit 1
  fi
  pass1_args+=(--init_shared_student_ckpt "${INIT_SHARED_STUDENT_CKPT}")
  echo "[FINAL] balanced-polish init checkpoint=${INIT_SHARED_STUDENT_CKPT}"
fi
run_distributed "${pass1_args[@]}"

ENERGY_TRAJECTORY_SCRIPT="${PROJECT_ROOT}/core/plot_subspace_energy_trajectory_final_llama.py"
PASS1_VAL_SUBSPACE_DIR="${PASS1_DIR}/compress_val_subspace"
if [ "$(normalize_bool "${PASS1_VAL_SUBSPACE_VIZ_ENABLE}")" = "true" ] && [ -f "${ENERGY_TRAJECTORY_SCRIPT}" ]; then
  if compgen -G "${PASS1_VAL_SUBSPACE_DIR}/step_*_subspace_plot_data.pt" >/dev/null; then
    echo "[FINAL] rendering deviation-energy alignment plots"
    run_with_time "${PYTHON_BIN}" "${ENERGY_TRAJECTORY_SCRIPT}" --input_dir "${PASS1_VAL_SUBSPACE_DIR}"
  else
    echo "[FINAL][Warn] no subspace plot data found under ${PASS1_VAL_SUBSPACE_DIR}; skipping energy alignment plots" >&2
  fi
fi

echo "[FINAL] exporting pass1 checkpoint directly"

export_args=(
  export
  --atlas_path "${ATLAS_PATH}"
  --shared_student_ckpt "${PASS1_CKPT}"
  --output_dir "${EXPORT_DIR}"
  --quant_bits "${QUANT_BITS}"
)
run_distributed "${export_args[@]}"

if [ "$(normalize_bool "${EVAL_ENABLE}")" = "true" ]; then
  read -r -a DATASET_ARR <<< "${EVAL_DATASETS}"
  for ds in "${DATASET_ARR[@]}"; do
    eval_args=()
    if [ "$(normalize_bool "${EVAL_PRINT_SCORES}")" = "true" ]; then
      eval_args+=(--print_scores)
    fi
    phase_eval_args=(
      eval
      --model_variant phase1_5
      --deploy_bundle "${DEPLOY_BUNDLE}"
      --dataset "${ds}"
      --test_data_root "${TEST_DATA_ROOT}"
      --batch_size "${EVAL_BATCH_SIZE}"
      --eval_mode "${EVAL_MODE}"
      --length_norm "${EVAL_LENGTH_NORM}"
      --max_new_tokens "${EVAL_MAX_NEW_TOKENS}"
      --use_quant_bank_int4 "${EVAL_USE_QUANT_BANK_INT4}"
      --trust_remote_code "${TRUST_REMOTE_CODE}"
      --device "${DEVICE}"
      --seed "${SEED}"
      --output_json "${RESULTS_DIR}/phase1_5_${ds}.json"
      "${eval_args[@]}"
    )
    run_distributed "${phase_eval_args[@]}"
    if [ "$(normalize_bool "${EVAL_BASELINE_ENABLE}")" = "true" ]; then
      baseline_eval_args=(
        eval
        --model_variant baseline
        --base_model "${BASE_MODEL}"
        --tokenizer_name_or_path "${TOKENIZER_NAME_OR_PATH}"
        --trust_remote_code "${TRUST_REMOTE_CODE}"
        --dataset "${ds}"
        --test_data_root "${TEST_DATA_ROOT}"
        --batch_size "${EVAL_BATCH_SIZE}"
        --eval_mode "${EVAL_MODE}"
        --length_norm "${EVAL_LENGTH_NORM}"
        --max_new_tokens "${EVAL_MAX_NEW_TOKENS}"
        --device "${DEVICE}"
        --seed "${SEED}"
        --output_json "${RESULTS_DIR}/baseline_${ds}.json"
        "${eval_args[@]}"
      )
      run_distributed "${baseline_eval_args[@]}"
    else
      echo "[FINAL] baseline eval disabled; skipping baseline dataset=${ds}"
    fi
  done
fi

echo "[FINAL] done"
echo "  - ${OUT_ROOT}"
echo "  - ${RESULTS_DIR}"
