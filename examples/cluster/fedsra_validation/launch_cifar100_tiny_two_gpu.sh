#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/public/home/dongshou/projects/OiaFed}"
RUN_BASE="${RUN_BASE:-/public/home/dongshou/OiaFed_fedsra_runs}"
EPOCHS="${EPOCHS:-300}"
QUEUE_SCRIPT="${PROJECT_ROOT}/examples/cluster/fedsra_validation/run_dataset_queue.sh"
CONTROL_ROOT="${RUN_BASE}/cifar100_tiny_k5_a0.05"
mkdir -p "${CONTROL_ROOT}/logs"

nohup env PROJECT_ROOT="${PROJECT_ROOT}" RUN_BASE="${RUN_BASE}" \
  bash "${QUEUE_SCRIPT}" 0 cifar100 "${EPOCHS}" 0 42 123 \
  > "${CONTROL_ROOT}/logs/gpu0_cifar100_queue.log" 2>&1 &
PID0=$!

nohup env PROJECT_ROOT="${PROJECT_ROOT}" RUN_BASE="${RUN_BASE}" \
  bash "${QUEUE_SCRIPT}" 1 tiny_imagenet "${EPOCHS}" 0 42 123 \
  > "${CONTROL_ROOT}/logs/gpu1_tiny_imagenet_queue.log" 2>&1 &
PID1=$!

printf 'gpu0_cifar100_pid=%s\ngpu1_tiny_imagenet_pid=%s\n' "${PID0}" "${PID1}" \
  | tee "${CONTROL_ROOT}/pids.txt"
