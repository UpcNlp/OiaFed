#!/usr/bin/env bash
set -eo pipefail

source /opt/dtk/env.sh
set -u

GPU_ID="${1:?usage: run_validation_cell.sh GPU_ID SEED [EPOCHS]}"
SEED="${2:?usage: run_validation_cell.sh GPU_ID SEED [EPOCHS]}"
EPOCHS="${3:-600}"

PROJECT_ROOT="${PROJECT_ROOT:-/public/home/dongshou/projects/OiaFed}"
RUN_ROOT="${RUN_ROOT:-/public/home/dongshou/OiaFed_fedsra_runs/cifar10_k5_a0.05}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/dongshou/anaconda/envs/ct/bin/python}"
CELL_ROOT="${RUN_ROOT}/seed${SEED}"
CONFIG_DIR="${CELL_ROOT}/configs"
RESULT_FILE="${CELL_ROOT}/result.json"

export PYTHONPATH="${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export HIP_VISIBLE_DEVICES="${GPU_ID}"
export FEDSRA_EXP_NAME="fedsra_cifar10_k5_a0.05_s${SEED}"
export FEDSRA_CHECKPOINT_DIR="${CELL_ROOT}/checkpoints"

mkdir -p "${CONFIG_DIR}" "${CELL_ROOT}/logs"
cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" -m oiafed run \
  --paper fedsra \
  --config examples/configs/fedsra_cifar10_validation/base.yaml \
  --num-clients 5 \
  --mode serial \
  --local-epochs "${EPOCHS}" \
  --data-dir /public/home/dongshou/data \
  --seed "${SEED}" \
  --dry-run \
  --save-config "${CONFIG_DIR}"

"${PYTHON_BIN}" examples/cluster/fedsra_validation/run_validation.py \
  --config-dir "${CONFIG_DIR}" \
  --result-file "${RESULT_FILE}" \
  --seed "${SEED}"
