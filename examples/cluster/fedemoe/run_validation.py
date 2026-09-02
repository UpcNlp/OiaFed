"""Run the native OiaFed FedEMoE integration with durable validation metadata.

The ``paper`` preset is the CIFAR-10/CNN/Dirichlet(0.5) configuration from
the supplied FedEMoE paper artifact.  The smaller presets exercise the same
native components and unchanged numerical core before the long 500-round run.
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


PRESETS: dict[str, dict[str, Any]] = {
    "smoke": {
        "num_clients": 10,
        "num_rounds": 1,
        "local_epochs": 1,
        "clients_per_round": 10,
        "max_samples": 1000,
        "num_workers": 0,
    },
    "short": {
        "num_clients": 100,
        "num_rounds": 2,
        "local_epochs": 1,
        "clients_per_round": 10,
        "max_samples": None,
        "num_workers": 0,
    },
    "paper": {
        "num_clients": 100,
        "num_rounds": 500,
        "local_epochs": 5,
        "clients_per_round": 10,
        "max_samples": None,
        "num_workers": 4,
    },
}


def _git_commit(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def _override(
    preset: dict[str, Any], data_dir: str, run_dir: Path
) -> dict[str, Any]:
    dataset: dict[str, Any] = {
        "data_dir": data_dir,
        "download": True,
        "server_test": True,
        "learner_test": False,
        "shared_cache": True,
        "augmentation": False,
        "transform_profile": "fedemoe",
        "partition": {
            "strategy": "fedemoe_dirichlet",
            "alpha": 0.5,
            "seed": 42,
        },
    }
    if preset["max_samples"] is not None:
        dataset["max_samples"] = int(preset["max_samples"])
        dataset["subset_seed"] = 42

    workers = int(preset["num_workers"])
    return {
        "global_config": {
            "exp_name": "fedemoe_cifar10_cnn_dir05_seed42",
            "run_name": run_dir.name,
            "log_dir": str(run_dir / "logs"),
        },
        "trainer": {
            "num_rounds": int(preset["num_rounds"]),
            "local_epochs": int(preset["local_epochs"]),
            "clients_per_round": int(preset["clients_per_round"]),
            "client_fraction": 0.1,
            "eval_interval": 10,
            "seed": 42,
            "device": "cuda",
            "batch_size": 64,
            "num_workers": workers,
            "eval_batch_size": 64,
            "eval_num_workers": workers,
            "fit_timeout": 86400,
        },
        "learner": {
            "learning_rate": 0.01,
            "batch_size": 64,
            "momentum": 0.9,
            "weight_decay": 0.0001,
            "optimizer": "sgd",
            "num_classes": 10,
            "edl_lambda1": 1.0,
            "edl_lambda2": 0.1,
            "annealing_epochs": 10,
            "num_workers": workers,
            "device": "cuda",
        },
        "aggregator": {
            "pool_size": 10,
            "num_experts": 8,
            "num_classes": 10,
            "num_parents": 4,
            "symbiosis_mode": "adaptive",
            "adaptive_mode": "diversity",
            "endo_ratio": 0.5,
            "switch_round": 100,
            "diversity_threshold": 0.5,
            "smoothing_alpha": 0.0,
            "ema_momentum": 1.0,
        },
        "model": {
            "num_classes": 10,
            "num_experts": 8,
            "backbone": "cnn",
            "input_channels": 3,
            "input_size": 32,
            "expert_hidden_dim": 256,
            "k_min": 1,
            "k_max": 7,
            "seed": 42,
        },
        "dataset": dataset,
        "tracker": {"enabled": False, "backends": []},
        "transport": {"mode": "memory"},
        "logging": {"console": True, "level": "INFO"},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=tuple(PRESETS), required=True)
    parser.add_argument("--data-dir", default="/public/home/dongshou/data")
    parser.add_argument(
        "--run-base",
        type=Path,
        default=Path("/public/home/dongshou/OiaFed_FedEMoE_validation/runs"),
    )
    args = parser.parse_args()

    preset = PRESETS[args.preset]
    project_root = Path(__file__).resolve().parents[3]
    commit = _git_commit(project_root)
    run_dir = args.run_base / commit[:7] / args.preset
    config_dir = run_dir / "configs"
    result_file = run_dir / "result.json"
    done_file = run_dir / "DONE"
    failed_file = run_dir / "FAILED"
    manifest = (
        project_root
        / "oiafed"
        / "methods"
        / "fedemoe_reference"
        / "UPSTREAM_MANIFEST.sha256"
    )

    protocol = {
        "version": 1,
        "preset": args.preset,
        "paper_exact": args.preset == "paper",
        "git_commit": commit,
        "upstream_manifest_sha256": _sha256(manifest),
        "dataset": "cifar10",
        "model": "cnn",
        "partition": "fedemoe_dirichlet",
        "alpha": 0.5,
        "seed": 42,
        **preset,
    }
    if result_file.exists():
        existing = json.loads(result_file.read_text(encoding="utf-8"))
        if existing.get("status") == "completed" and existing.get("protocol") == protocol:
            print(f"[resume] completed result exists: {result_file}")
            return 0

    config_dir.mkdir(parents=True, exist_ok=True)
    reload_registry().generate_node_configs(
        "fedemoe",
        override=_override(preset, args.data_dir, run_dir),
        num_clients=int(preset["num_clients"]),
        output_dir=str(config_dir),
    )

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.cuda.reset_peak_memory_stats()

    started = time.time()
    payload: dict[str, Any] = {
        "status": "running",
        "protocol": protocol,
        "config_dir": str(config_dir),
        "visible_devices": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES"),
        },
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "torch_version": torch.__version__,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    _write_json(result_file, payload)
    try:
        result = FederationRunner(config_dir).run_sync()
        trainer_result = result.get("trainer", {})
        final_accuracy = trainer_result.get("final_accuracy")
        if final_accuracy is None:
            final_accuracy = trainer_result.get("final_round_metrics", {}).get(
                "eval_accuracy"
            )
        payload.update(
            {
                "status": "completed",
                "result": result,
                "final_accuracy_percent": (
                    100.0 * float(final_accuracy) if final_accuracy is not None else None
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
        done_file.touch()
        failed_file.unlink(missing_ok=True)
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
        failed_file.touch()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
