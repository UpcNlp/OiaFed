#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:?usage: run_dataset_queue.sh GPU_ID DATASET [EPOCHS] [SEEDS...]}"
DATASET="${2:?usage: run_dataset_queue.sh GPU_ID DATASET [EPOCHS] [SEEDS...]}"
EPOCHS="${3:-300}"
shift 3 || true
SEEDS=("${@:-0 42 123}")

PROJECT_ROOT="${PROJECT_ROOT:-/public/home/dongshou/projects/OiaFed}"
RUN_BASE="${RUN_BASE:-/public/home/dongshou/OiaFed_fedsra_runs}"
CELL_SCRIPT="${PROJECT_ROOT}/examples/cluster/fedsra_validation/run_dataset_validation_cell.sh"
RUN_ROOT="${RUN_BASE}/${DATASET}_k5_a0.05"
mkdir -p "${RUN_ROOT}/logs"

for seed in ${SEEDS[*]}; do
  log_file="${RUN_ROOT}/logs/gpu${GPU_ID}_seed${seed}.log"
  printf '[%s] start dataset=%s seed=%s gpu=%s epochs=%s\n' \
    "$(date --iso-8601=seconds)" "${DATASET}" "${seed}" "${GPU_ID}" "${EPOCHS}"
  if PROJECT_ROOT="${PROJECT_ROOT}" RUN_BASE="${RUN_BASE}" \
      bash "${CELL_SCRIPT}" "${GPU_ID}" "${DATASET}" "${seed}" "${EPOCHS}" \
      > "${log_file}" 2>&1; then
    printf '[%s] completed dataset=%s seed=%s\n' \
      "$(date --iso-8601=seconds)" "${DATASET}" "${seed}"
  else
    printf '[%s] failed dataset=%s seed=%s log=%s\n' \
      "$(date --iso-8601=seconds)" "${DATASET}" "${seed}" "${log_file}" >&2
    exit 1
  fi
done
