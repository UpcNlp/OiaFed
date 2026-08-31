#!/usr/bin/env bash
set -eo pipefail

source /opt/dtk/env.sh
set -u

PROJECT_ROOT="${PROJECT_ROOT:-/public/home/dongshou/projects/OiaFed}"
RUN_BASE="${RUN_BASE:-/public/home/dongshou/OiaFed_oneshot_baseline_runs/effect_alignment_v2}"
REUSE_BASE="${REUSE_BASE:-/public/home/dongshou/OiaFed_oneshot_baseline_runs/effect_v1}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/dongshou/anaconda/envs/ct/bin/python}"
DATASET="${DATASET:-cifar10}"
ALPHA="${ALPHA:-0.05}"
NUM_CLIENTS="${NUM_CLIENTS:-5}"
SEED="${SEED:-42}"
GROUP="${DATASET}_k${NUM_CLIENTS}_a${ALPHA}/seed${SEED}"
GROUP_ROOT="${RUN_BASE}/${GROUP}"

mkdir -p "${GROUP_ROOT}/logs"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}"

prepare_ce_reuse() {
  OLD_CHECKPOINT_ROOT="${REUSE_BASE}/${GROUP}/shared_ce/checkpoints" \
  NEW_CHECKPOINT_ROOT="${RUN_BASE}/${GROUP}/shared_ce/checkpoints" \
  DATASET="${DATASET}" ALPHA="${ALPHA}" NUM_CLIENTS="${NUM_CLIENTS}" SEED="${SEED}" \
    "${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

import torch

from examples.cluster.oneshot_baselines.run_effect_cell import (
    _checkpoint_signature,
    _git_commit,
    _protocol,
)

project_root = Path.cwd()
old_root = Path(os.environ["OLD_CHECKPOINT_ROOT"])
new_root = Path(os.environ["NEW_CHECKPOINT_ROOT"])
dataset = os.environ["DATASET"]
alpha = float(os.environ["ALPHA"])
num_clients = int(os.environ["NUM_CLIENTS"])
seed = int(os.environ["SEED"])
protocol = _protocol("coboosting", dataset, alpha, num_clients, seed, _git_commit(project_root))
signature = _checkpoint_signature(protocol)

for client_index in range(num_clients):
    source = old_root / f"learner_{client_index}" / "model.pt"
    target = new_root / f"learner_{client_index}" / "model.pt"
    if target.exists():
        continue
    if not source.exists():
        raise FileNotFoundError(f"missing reusable CE checkpoint: {source}")
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if int(payload.get("epochs", -1)) != 300:
        raise ValueError(f"unexpected CE epoch count in {source}")
    payload["signature"] = signature
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, target)
    print(f"reused {source} -> {target}")
PY
}

run_cell() {
  local physical_gpu="$1"
  local method="$2"
  local log_file="${GROUP_ROOT}/logs/gpu${physical_gpu}_${method}.log"
  printf '[%s] start method=%s gpu=%s\n' "$(date --iso-8601=seconds)" "${method}" "${physical_gpu}"
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

prepare_ce_reuse

run_cell 0 fafi &
PID0=$!

(
  run_cell 1 fusefl
  run_cell 1 coboosting
) &
PID1=$!

wait "${PID0}"
wait "${PID1}"

echo "FAFI, FuseFL, and Co-Boosting alignment effects completed."
