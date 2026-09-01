#!/usr/bin/env bash
set -eo pipefail

source /opt/dtk/env.sh
set -u

PROJECT_ROOT="${PROJECT_ROOT:-/public/home/dongshou/projects/OiaFed}"
RUN_BASE="${RUN_BASE:-/public/home/dongshou/OiaFed_oneshot_baseline_runs/effect_alignment_v3}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/dongshou/anaconda/envs/ct/bin/python}"
DATASET="${DATASET:-cifar10}"
ALPHA="${ALPHA:-0.05}"
NUM_CLIENTS="${NUM_CLIENTS:-5}"
GPU0_SEED="${GPU0_SEED:-42}"
GPU1_SEED="${GPU1_SEED:-0}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}"

run_seed() {
  local physical_gpu="$1"
  local seed="$2"
  local group_root="${RUN_BASE}/${DATASET}_k${NUM_CLIENTS}_a${ALPHA}/seed${seed}"
  local log_file="${group_root}/logs/gpu${physical_gpu}_fafi.log"
  mkdir -p "$(dirname "${log_file}")"
  printf '[%s] start FAFI seed=%s gpu=%s commit=%s\n' \
    "$(date --iso-8601=seconds)" "${seed}" "${physical_gpu}" "$(git rev-parse HEAD)"
  HIP_VISIBLE_DEVICES="${physical_gpu}" CUDA_VISIBLE_DEVICES="${physical_gpu}" \
    "${PYTHON_BIN}" -u examples/cluster/oneshot_baselines/run_effect_cell.py \
      --method fafi \
      --dataset "${DATASET}" \
      --alpha "${ALPHA}" \
      --num-clients "${NUM_CLIENTS}" \
      --seed "${seed}" \
      --run-base "${RUN_BASE}" \
      > "${log_file}" 2>&1
  printf '[%s] completed FAFI seed=%s gpu=%s\n' \
    "$(date --iso-8601=seconds)" "${seed}" "${physical_gpu}"
}

run_seed 0 "${GPU0_SEED}" &
PID0=$!
run_seed 1 "${GPU1_SEED}" &
PID1=$!

wait "${PID0}"
wait "${PID1}"

echo "Two-DCU FAFI alignment validation completed."
