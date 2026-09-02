#!/usr/bin/env bash
set -eo pipefail

source /opt/dtk/env.sh
set -u

GPU_ID="${1:?usage: run_dataset_validation_cell.sh GPU_ID DATASET SEED [EPOCHS]}"
DATASET="${2:?usage: run_dataset_validation_cell.sh GPU_ID DATASET SEED [EPOCHS]}"
SEED="${3:?usage: run_dataset_validation_cell.sh GPU_ID DATASET SEED [EPOCHS]}"
EPOCHS="${4:-300}"

case "${DATASET}" in
  cifar100)
    CONFIG_FILE="examples/configs/fedsra_cifar100_validation/base.yaml"
    ;;
  tiny_imagenet)
    CONFIG_FILE="examples/configs/fedsra_tiny_imagenet_validation/base.yaml"
    ;;
  *)
    echo "unsupported dataset: ${DATASET}" >&2
    exit 2
    ;;
esac

PROJECT_ROOT="${PROJECT_ROOT:-/public/home/dongshou/projects/OiaFed}"
RUN_BASE="${RUN_BASE:-/public/home/dongshou/OiaFed_fedsra_runs}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/dongshou/anaconda/envs/ct/bin/python}"
RUN_ROOT="${RUN_BASE}/${DATASET}_k5_a0.05"
CELL_ROOT="${RUN_ROOT}/seed${SEED}"
CONFIG_DIR="${CELL_ROOT}/configs"
RESULT_FILE="${CELL_ROOT}/result.json"

export PYTHONPATH="${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export HIP_VISIBLE_DEVICES="${GPU_ID}"
export FEDSRA_EXP_NAME="fedsra_${DATASET}_k5_a0.05_s${SEED}"
export FEDSRA_CHECKPOINT_DIR="${CELL_ROOT}/checkpoints"

mkdir -p "${CONFIG_DIR}" "${CELL_ROOT}/logs"
cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" -m oiafed run \
  --paper fedsra \
  --config "${CONFIG_FILE}" \
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
