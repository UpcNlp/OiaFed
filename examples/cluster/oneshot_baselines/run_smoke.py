"""Run reduced end-to-end checks for the six native one-shot baselines."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
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


def _git_commit(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _override(method: str, data_dir: str, seed: int) -> dict[str, Any]:
    trainer = {
        "device": "cuda",
        "fit_timeout": 900,
        "eval_batch_size": 64,
        "eval_num_workers": 0,
        "local_epochs": 1,
        "local_epochs_per_stage": 1,
        "server_epochs": 1,
        "generator_steps": 1,
        "kd_steps": 1,
        "generator_width": 4,
        "synthesis_batch_size": 4,
        "batch_size": 4,
        "shutdown_wait_time": 2,
    }
    learner = {
        "device": "cuda",
        "batch_size": 16,
        "num_workers": 0,
        "persistent_workers": False,
        "drop_last": False,
        "two_view_augmentation": method == "fafi",
    }
    model: dict[str, Any] = {}
    if method == "fedcgs":
        model["pretrained"] = False
    elif method == "fusefl":
        model["base_width"] = 4
    dataset = {
        "data_dir": data_dir,
        "download": True,
        "server_test": True,
        "max_samples": 256,
        "subset_seed": seed,
        "partition": {
            "strategy": "fedsra_dirichlet",
            "alpha": 0.5,
            "seed": seed,
        },
    }
    return {
        "global_config": {"exp_name": f"{method}_native_smoke_s{seed}"},
        "trainer": trainer,
        "learner": learner,
        "model": model,
        "dataset": dataset,
        # ConfigGenerator otherwise serializes an omitted backend collection as
        # YAML null, while legacy OiaFed releases expect an iterable here.
        "tracker": {"enabled": False, "backends": []},
        # Each physical accelerator runs an independent FederationRunner.  Keep
        # their transports process-local so concurrent smoke chains cannot
        # connect to the same default gRPC ports.
        "transport": {"mode": "memory"},
        "logging": {"console": True, "level": "INFO"},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", choices=METHODS, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--data-dir", default="/public/home/dongshou/data")
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    project_root = Path(__file__).resolve().parents[3]
    papers = reload_registry()
    for method in args.methods:
        cell = args.run_root / method
        config_dir = cell / "configs"
        result_file = cell / "result.json"
        if result_file.exists():
            current = json.loads(result_file.read_text(encoding="utf-8"))
            if current.get("status") == "completed" and current.get("git_commit") == _git_commit(project_root):
                print(f"[skip] {method}: completed result for current commit")
                continue

        config_dir.mkdir(parents=True, exist_ok=True)
        papers.generate_node_configs(
            method,
            override=_override(method, args.data_dir, args.seed),
            num_clients=2,
            output_dir=str(config_dir),
        )
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        started = time.time()
        payload: dict[str, Any] = {
            "status": "running",
            "method": method,
            "seed": args.seed,
            "git_commit": _git_commit(project_root),
            "visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_json(result_file, payload)
        try:
            result = FederationRunner(config_dir).run_sync()
            payload.update(
                {
                    "status": "completed",
                    "result": result,
                    "elapsed_seconds": round(time.time() - started, 3),
                    "gpu_peak_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 3)
                    if torch.cuda.is_available()
                    else 0.0,
                    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
            _write_json(result_file, payload)
            print(json.dumps(payload, ensure_ascii=False))
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
