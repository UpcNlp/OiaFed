#!/usr/bin/env bash
set -eo pipefail

source /opt/dtk/env.sh
set -u

PROJECT_ROOT="${PROJECT_ROOT:-/public/home/dongshou/projects/OiaFed}"
RUN_BASE="${RUN_BASE:-/public/home/dongshou/OiaFed_oneshot_baseline_runs/effect_v1}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/dongshou/anaconda/envs/ct/bin/python}"
DATASET="${DATASET:-cifar10}"
ALPHA="${ALPHA:-0.05}"
NUM_CLIENTS="${NUM_CLIENTS:-5}"
SEED="${SEED:-42}"
GROUP_ROOT="${RUN_BASE}/${DATASET}_k${NUM_CLIENTS}_a${ALPHA}/seed${SEED}"

mkdir -p "${GROUP_ROOT}/logs"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}"

run_cell() {
  local physical_gpu="$1"
  local method="$2"
  local log_file="${GROUP_ROOT}/logs/gpu${physical_gpu}_${method}.log"
  printf '[%s] start method=%s dataset=%s alpha=%s K=%s seed=%s gpu=%s\n' \
    "$(date --iso-8601=seconds)" "${method}" "${DATASET}" "${ALPHA}" \
    "${NUM_CLIENTS}" "${SEED}" "${physical_gpu}"
  HIP_VISIBLE_DEVICES="${physical_gpu}" CUDA_VISIBLE_DEVICES="${physical_gpu}" \
    "${PYTHON_BIN}" -u examples/cluster/oneshot_baselines/run_effect_cell.py \
      --method "${method}" \
      --dataset "${DATASET}" \
      --alpha "${ALPHA}" \
      --num-clients "${NUM_CLIENTS}" \
      --seed "${SEED}" \
      --run-base "${RUN_BASE}" \
      > "${log_file}" 2>&1
  printf '[%s] completed method=%s\n' "$(date --iso-8601=seconds)" "${method}"
}

run_gpu0() {
  # O-FedAvg materializes the shared CE teacher checkpoints. Ensemble then
  # reuses those exact clients before FAFI runs its own objective.
  run_cell 0 ofedavg
  run_cell 0 ensemble
  run_cell 0 fafi
}

run_gpu1() {
  run_cell 1 fusefl
  run_cell 1 fedcgs
  # Co-Boosting must consume the same CE teachers as O-FedAvg/Ensemble.
  while [[ ! -f "${GROUP_ROOT}/ofedavg/result.json" ]] || \
        ! grep -q '"status": "completed"' "${GROUP_ROOT}/ofedavg/result.json"; do
    sleep 30
  done
  run_cell 1 coboosting
}

run_gpu0 &
PID0=$!
run_gpu1 &
PID1=$!
wait "${PID0}"
wait "${PID1}"

echo "All paper-scale effect-gate cells completed."
