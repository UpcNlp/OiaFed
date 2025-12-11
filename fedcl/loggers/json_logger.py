"""
JSON Logger 实现
fedcl/loggers/json_logger.py

特点：
- 轻量级，零依赖
- 本地 JSON 文件存储
- 适合快速开发和离线环境
"""

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Union
from datetime import datetime


from .base_logger import Logger


class JSONLogger(Logger):
    """
    JSON Logger 实现

    特点：
    - 零依赖，直接保存到 JSON 文件
    - 轻量级，适合快速测试
    - 每个 run 保存为独立的 JSON 文件

    使用示例：
        logger = JSONLogger(
            save_dir="experiments/results",
            experiment_name="my_exp",
            run_name="run_001"
        )
        logger.log_params({"lr": 0.01})
        logger.log_metrics({"accuracy": 0.95}, step=1)
        logger.finalize()

    文件结构：
        experiments/results/
        └── my_exp/
            └── run_001_20231201_120000.json
    """

    def __init__(
        self,
        save_dir: str = "experiments/results",
        experiment_name: str = "default",
        run_name: Optional[str] = None,
        **kwargs
    ):
        """
        Args:
            save_dir: 结果保存根目录
            experiment_name: 实验名称
            run_name: Run 名称（可选，默认自动生成）
        """
        super().__init__()

        self._experiment_name = experiment_name
        self._run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._run_id = f"{experiment_name}_{self._run_name}"

        # 创建保存目录
        self.save_dir = Path(save_dir) / experiment_name
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 数据存储
        self.data = {
            "experiment_name": experiment_name,
            "run_name": self._run_name,
            "run_id": self._run_id,
            "start_time": datetime.now().isoformat(),
            "params": {},
            "metrics": {},
            "artifacts": []
        }

        print(f"[JSONLogger] Run started: {self._run_name}")

    def log_metrics(
        self,
        metrics: Dict[str, Union[int, float]],
        step: Optional[int] = None
    ) -> None:
        """记录指标"""
        for name, value in metrics.items():
            if name not in self.data["metrics"]:
                self.data["metrics"][name] = []

            self.data["metrics"][name].append({
                "step": step if step is not None else len(self.data["metrics"][name]),
                "value": float(value),
                "timestamp": datetime.now().isoformat()
            })

    def log_params(self, params: Dict[str, Any]) -> None:
        """记录参数"""
        self.data["params"].update(params)

    def log_artifact(
        self,
        local_path: str,
        artifact_path: Optional[str] = None
    ) -> None:
        """复制文件到结果目录"""
        try:
            src = Path(local_path)
            if not src.exists():
                print(f"[JSONLogger] Warning: File not found: {local_path}")
                return

            # 创建 artifacts 子目录
            artifacts_dir = self.save_dir / "artifacts"
            artifacts_dir.mkdir(exist_ok=True)

            # 目标文件名
            dst_name = artifact_path or src.name
            dst = artifacts_dir / dst_name

            # 复制文件
            shutil.copy2(src, dst)

            # 记录 artifact 信息
            self.data["artifacts"].append({
                "original_path": str(local_path),
                "saved_path": str(dst.relative_to(self.save_dir)),
                "timestamp": datetime.now().isoformat()
            })

        except Exception as e:
            print(f"[JSONLogger] Failed to log artifact: {e}")

    def finalize(self, status: str = "success") -> None:
        """保存 JSON 文件"""
        self.data["end_time"] = datetime.now().isoformat()
        self.data["status"] = status

        # 计算持续时间
        if "start_time" in self.data:
            start = datetime.fromisoformat(self.data["start_time"])
            end = datetime.fromisoformat(self.data["end_time"])
            self.data["duration_seconds"] = (end - start).total_seconds()

        # 保存到文件
        output_file = self.save_dir / f"{self._run_name}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

        print(f"[JSONLogger] Results saved to {output_file}")
