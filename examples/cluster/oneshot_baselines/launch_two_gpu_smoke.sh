#!/usr/bin/env bash
set -eo pipefail

source /opt/dtk/env.sh
set -u

PROJECT_ROOT="${PROJECT_ROOT:-/public/home/dongshou/projects/OiaFed}"
RUN_ROOT="${RUN_ROOT:-/public/home/dongshou/OiaFed_oneshot_baseline_runs/native_smoke}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/dongshou/anaconda/envs/ct/bin/python}"

mkdir -p "${RUN_ROOT}/logs"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}"

run_chain() {
  local physical_gpu="$1"
  shift
  HIP_VISIBLE_DEVICES="${physical_gpu}" CUDA_VISIBLE_DEVICES="${physical_gpu}" \
    "${PYTHON_BIN}" -u examples/cluster/oneshot_baselines/run_smoke.py \
      --methods "$@" \
      --run-root "${RUN_ROOT}" \
      > "${RUN_ROOT}/logs/gpu${physical_gpu}.log" 2>&1
}

run_chain 0 ofedavg ensemble fafi &
PID0=$!
run_chain 1 fusefl fedcgs coboosting &
PID1=$!

wait "${PID0}"
wait "${PID1}"

echo "All native one-shot baseline smoke checks completed."
