#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/public/home/dongshou/projects/OiaFed}"
RUN_ROOT="${RUN_ROOT:-/public/home/dongshou/OiaFed_fedsra_runs/cifar10_k5_a0.05}"
EPOCHS="${EPOCHS:-600}"
SCRIPT="${PROJECT_ROOT}/examples/cluster/fedsra_validation/run_validation_cell.sh"

mkdir -p "${RUN_ROOT}/logs"

nohup bash "${SCRIPT}" 0 42 "${EPOCHS}" \
  > "${RUN_ROOT}/logs/gpu0_seed42.log" 2>&1 &
PID0=$!

nohup bash "${SCRIPT}" 1 43 "${EPOCHS}" \
  > "${RUN_ROOT}/logs/gpu1_seed43.log" 2>&1 &
PID1=$!

printf 'gpu0_seed42_pid=%s\ngpu1_seed43_pid=%s\n' "${PID0}" "${PID1}" \
  | tee "${RUN_ROOT}/pids.txt"
