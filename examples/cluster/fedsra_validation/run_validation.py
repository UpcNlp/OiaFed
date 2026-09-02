"""Run one generated FedSRA/OiaFed cell and write durable result metadata."""

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

from oiafed.runner import FederationRunner


def _git_commit(project_root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        text=True,
    ).strip()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--result-file", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    if args.result_file.exists():
        existing = json.loads(args.result_file.read_text(encoding="utf-8"))
        if existing.get("status") == "completed":
            print(f"[resume] completed result exists: {args.result_file}")
            return 0

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()

    project_root = Path(__file__).resolve().parents[3]
    started_at = datetime.now(timezone.utc)
    start = time.time()
    metadata: dict[str, Any] = {
        "status": "running",
        "seed": args.seed,
        "config_dir": str(args.config_dir.resolve()),
        "git_commit": _git_commit(project_root),
        "started_at_utc": started_at.isoformat(),
        "torch_version": torch.__version__,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    _write_json(args.result_file, metadata)

    try:
        result = FederationRunner(args.config_dir).run_sync()
        metadata.update(
            {
                "status": "completed",
                "elapsed_seconds": round(time.time() - start, 3),
                "gpu_peak_mb": round(
                    torch.cuda.max_memory_allocated() / 1024**2,
                    3,
                ) if torch.cuda.is_available() else 0.0,
                "result": result,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        _write_json(args.result_file, metadata)
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:
        metadata.update(
            {
                "status": "failed",
                "elapsed_seconds": round(time.time() - start, 3),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        _write_json(args.result_file, metadata)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
