"""
Sacred 记录器封装（简化版本）
experiments/sacred_wrapper.py

功能：
- 简单可靠的实验记录
- 直接使用JSON文件保存结果
- 支持三种模式下的统一记录
"""

import json
import threading
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


class SacredRecorder:
    """
    简化版实验记录器
    直接使用JSON文件保存实验结果,避免Sacred复杂性
    """
    _instance = None
    _lock = threading.Lock()

    def __init__(self, experiment_name: str, role: str, node_id: str,
                 base_dir: str = "experiments/results"):
        """
        Args:
            experiment_name: 实验名称（可读ID）
            role: "server" 或 "client"
            node_id: 节点ID（如 "server_1", "client_0"）
            base_dir: 结果保存目录
        """
        self.experiment_name = experiment_name
        self.role = role
        self.node_id = node_id

        # 结果保存路径：experiments/results/{exp_name}/{role}_{node_id}/
        self.save_dir = Path(base_dir) / experiment_name / f"{role}_{node_id}"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 数据存储
        self.config: Dict[str, Any] = {}
        self.metrics: Dict[str, list] = {}  # {metric_name: [(step, value), ...]}
        self.info: Dict[str, Any] = {}
        self.start_time = None

    @classmethod
    def get_instance(cls) -> Optional['SacredRecorder']:
        """获取当前实例（如果已初始化）"""
        return cls._instance

    @classmethod
    def initialize(cls, experiment_name: str, role: str, node_id: str,
                   base_dir: str = "experiments/results") -> 'SacredRecorder':
        """初始化全局实例"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(experiment_name, role, node_id, base_dir)
        return cls._instance

    @classmethod
    def reset(cls):
        """重置全局实例（主要用于测试）"""
        with cls._lock:
            cls._instance = None

    def start_run(self, config: dict):
        """开始一次运行"""
        self.config = config.copy()
        self.start_time = datetime.now()
        self.info['start_time'] = self.start_time.isoformat()
        print(f"[Recorder] {self.role}_{self.node_id}: Run started")

    def log_scalar(self, name: str, value: float, step: int = None):
        """记录标量指标

        Args:
            name: 指标名称
            value: 指标值
            step: 步骤/轮次编号
        """
        try:
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append({
                'step': step if step is not None else len(self.metrics[name]),
                'value': float(value),
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            print(f"[Recorder] Failed to log scalar {name}: {e}")

    def log_info(self, key: str, value):
        """记录实验信息

        Args:
            key: 信息键
            value: 信息值
        """
        try:
            self.info[key] = value
        except Exception as e:
            print(f"[Recorder] Failed to log info {key}: {e}")

    def add_artifact(self, filepath: str, name: str = None):
        """添加附件文件

        Args:
            filepath: 文件路径
            name: 附件名称（可选）
        """
        try:
            # 简单记录文件路径
            if 'artifacts' not in self.info:
                self.info['artifacts'] = []
            self.info['artifacts'].append({
                'path': filepath,
                'name': name or Path(filepath).name
            })
        except Exception as e:
            print(f"[Recorder] Failed to add artifact {filepath}: {e}")

    def finish(self, status: str = "COMPLETED"):
        """结束实验并保存结果

        Args:
            status: 实验状态（COMPLETED, FAILED等）
        """
        try:
            # 记录结束时间
            end_time = datetime.now()
            self.info['end_time'] = end_time.isoformat()
            self.info['final_status'] = status
            if self.start_time:
                duration = (end_time - self.start_time).total_seconds()
                self.info['duration_seconds'] = duration

            # 保存所有数据到JSON文件
            result = {
                'experiment_name': self.experiment_name,
                'role': self.role,
                'node_id': self.node_id,
                'config': self.config,
                'metrics': self.metrics,
                'info': self.info
            }

            # 保存到run.json
            output_file = self.save_dir / "run.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            print(f"[Recorder] {self.role}_{self.node_id}: Results saved to {output_file}")

        except Exception as e:
            print(f"[Recorder] Failed to finish run: {e}")

        # 重置实例以便下次使用
        self.config = {}
        self.metrics = {}
        self.info = {}
        self.start_time = None
