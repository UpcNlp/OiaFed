"""Run one paper-scale one-shot baseline cell with durable metadata.

The effect protocol intentionally differs from ``run_smoke.py``: it uses the
complete dataset, K=5 by default, the paper method's full training budget, and
an exact non-overlapping Dirichlet split.  O-FedAvg, direct Ensemble, and
Co-Boosting reuse one shared set of CE client checkpoints so their server-side
comparison is not confounded by independently retrained teachers.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time
from typing import Any

import numpy as np
import torch

from oiafed.papers.loader import reload_registry
from oiafed.runner import FederationRunner


METHODS = ("ofedavg", "ensemble", "fafi", "fusefl", "fedcgs", "coboosting")
CE_TEACHER_METHODS = {"ofedavg", "ensemble", "coboosting"}
DATASETS = {
    "cifar10": {"type": "cifar10", "num_classes": 10, "image_size": 32},
    "cifar100": {"type": "cifar100", "num_classes": 100, "image_size": 32},
    "tiny_imagenet": {"type": "tiny_imagenet", "num_classes": 200, "image_size": 64},
}


def _git_commit(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _alpha_tag(alpha: float) -> str:
    return format(float(alpha), "g")


def _protocol(
    method: str,
    dataset: str,
    alpha: float,
    num_clients: int,
    seed: int,
    commit: str,
) -> dict[str, Any]:
    partition = "fafi_dirichlet" if method == "fafi" else "fedsra_dirichlet"
    return {
        "version": 2,
        "method": method,
        "dataset": dataset,
        "alpha": float(alpha),
        "num_clients": int(num_clients),
        "seed": int(seed),
        "complete_dataset": True,
        "partition": partition,
        "transport": "memory",
        "git_commit": commit,
    }


def _checkpoint_signature(protocol: dict[str, Any]) -> str:
    teacher_protocol = dict(protocol)
    if teacher_protocol["method"] in CE_TEACHER_METHODS:
        teacher_protocol["method"] = "shared_ce_teacher"
    canonical = json.dumps(teacher_protocol, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _override(
    method: str,
    spec: dict[str, Any],
    data_dir: str,
    alpha: float,
    num_clients: int,
    seed: int,
    checkpoint_dir: Path,
    checkpoint_signature: str,
) -> dict[str, Any]:
    num_classes = int(spec["num_classes"])
    partition = "fafi_dirichlet" if method == "fafi" else "fedsra_dirichlet"
    return {
        "global_config": {
            "exp_name": f"{method}_{spec['type']}_k{num_clients}_a{_alpha_tag(alpha)}_s{seed}",
        },
        "trainer": {
            "device": "cuda",
            "fit_timeout": 86400,
            "eval_batch_size": 256,
            "eval_num_workers": 4,
            "num_classes": num_classes,
            "image_size": int(spec["image_size"]),
            "shutdown_wait_time": 5,
            "calibration_seed": int(seed),
        },
        "learner": {
            "device": "cuda",
            "num_classes": num_classes,
            "checkpoint_dir": str(checkpoint_dir),
            "checkpoint_signature": checkpoint_signature,
            "resume": True,
        },
        "model": {
            "num_classes": num_classes,
            **(
                {"initialization_seed": int(seed)}
                if method in {"fafi", "fusefl"}
                else {}
            ),
        },
        "dataset": {
            "type": spec["type"],
            "data_dir": data_dir,
            "download": True,
            "server_test": True,
            "partition": {
                "strategy": partition,
                "alpha": float(alpha),
                "seed": int(seed),
            },
        },
        "tracker": {"enabled": False, "backends": []},
        "transport": {"mode": "memory"},
        "logging": {"console": True, "level": "INFO"},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--dataset", choices=tuple(DATASETS), default="cifar10")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--num-clients", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", default="/public/home/dongshou/data")
    parser.add_argument(
        "--run-base",
        type=Path,
        default=Path("/public/home/dongshou/OiaFed_oneshot_baseline_runs/effect_v1"),
    )
    args = parser.parse_args()

    if args.num_clients < 1:
        raise ValueError("num_clients must be positive")
    if args.alpha <= 0:
        raise ValueError("alpha must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()

    project_root = Path(__file__).resolve().parents[3]
    commit = _git_commit(project_root)
    protocol = _protocol(
        args.method,
        args.dataset,
        args.alpha,
        args.num_clients,
        args.seed,
        commit,
    )
    cell_group = (
        args.run_base
        / f"{args.dataset}_k{args.num_clients}_a{_alpha_tag(args.alpha)}"
        / f"seed{args.seed}"
    )
    cell = cell_group / args.method
    result_file = cell / "result.json"
    if result_file.exists():
        existing = json.loads(result_file.read_text(encoding="utf-8"))
        if existing.get("status") == "completed" and existing.get("protocol") == protocol:
            print(f"[resume] completed result exists: {result_file}")
            return 0

    if args.method in CE_TEACHER_METHODS:
        checkpoint_dir = cell_group / "shared_ce" / "checkpoints"
    else:
        checkpoint_dir = cell / "checkpoints"
    signature = _checkpoint_signature(protocol)
    config_dir = cell / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    reload_registry().generate_node_configs(
        args.method,
        override=_override(
            args.method,
            DATASETS[args.dataset],
            args.data_dir,
            args.alpha,
            args.num_clients,
            args.seed,
            checkpoint_dir,
            signature,
        ),
        num_clients=args.num_clients,
        output_dir=str(config_dir),
    )

    started = time.time()
    payload: dict[str, Any] = {
        "status": "running",
        "protocol": protocol,
        "checkpoint_signature": signature,
        "checkpoint_dir": str(checkpoint_dir),
        "visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "torch_version": torch.__version__,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    _write_json(result_file, payload)
    try:
        result = FederationRunner(config_dir).run_sync()
        metrics = result.get("trainer", {}).get("final_round_metrics", {})
        payload.update(
            {
                "status": "completed",
                "result": result,
                "accuracy_percent": (
                    100.0 * float(metrics["eval_accuracy"])
                    if "eval_accuracy" in metrics
                    else None
                ),
                "elapsed_seconds": round(time.time() - started, 3),
                "gpu_peak_mb": (
                    round(torch.cuda.max_memory_allocated() / 1024**2, 3)
                    if torch.cuda.is_available()
                    else 0.0
                ),
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        _write_json(result_file, payload)
        print(json.dumps(payload, ensure_ascii=False))
        return 0
    except Exception as error:
        payload.update(
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "elapsed_seconds": round(time.time() - started, 3),
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        _write_json(result_file, payload)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
