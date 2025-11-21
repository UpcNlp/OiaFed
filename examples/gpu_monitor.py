"""
GPU显存监控器 - 用于智能调度批量实验
examples/gpu_monitor.py

个人需求工具，不属于开源项目核心代码
"""

import subprocess
import re
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class GPUInfo:
    """GPU信息"""
    gpu_id: int
    name: str
    total_memory_mb: int
    used_memory_mb: int
    free_memory_mb: int
    utilization: int  # GPU利用率 (%)

    @property
    def usage_percent(self) -> float:
        """显存使用率（百分比）"""
        if self.total_memory_mb == 0:
            return 0.0
        return (self.used_memory_mb / self.total_memory_mb) * 100


class GPUMonitor:
    """
    GPU显存监控器

    功能：
    - 查询当前GPU显存使用情况
    - 估算实验所需显存
    - 判断是否有足够显存启动新实验
    """

    def __init__(self):
        """初始化GPU监控器"""
        self.experiment_memory_history = []  # 存储历史实验的显存使用记录

    def get_gpu_info(self) -> List[GPUInfo]:
        """
        获取所有GPU的当前状态信息

        Returns:
            List[GPUInfo]: GPU信息列表
        """
        try:
            # 使用nvidia-smi查询GPU信息
            result = subprocess.run(
                [
                    'nvidia-smi',
                    '--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu',
                    '--format=csv,noheader,nounits'
                ],
                capture_output=True,
                text=True,
                check=True
            )

            gpu_info_list = []
            for line in result.stdout.strip().split('\n'):
                if not line.strip():
                    continue

                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 6:
                    gpu_info = GPUInfo(
                        gpu_id=int(parts[0]),
                        name=parts[1],
                        total_memory_mb=int(parts[2]),
                        used_memory_mb=int(parts[3]),
                        free_memory_mb=int(parts[4]),
                        utilization=int(parts[5])
                    )
                    gpu_info_list.append(gpu_info)

            return gpu_info_list

        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"警告: 无法获取GPU信息: {e}")
            return []

    def get_max_free_memory(self) -> int:
        """
        获取所有GPU中最大的可用显存（MB）

        Returns:
            int: 最大可用显存（MB），如果没有GPU则返回0
        """
        gpu_info_list = self.get_gpu_info()
        if not gpu_info_list:
            return 0

        return max(gpu.free_memory_mb for gpu in gpu_info_list)

    def get_gpu_with_most_free_memory(self) -> Optional[GPUInfo]:
        """
        获取可用显存最多的GPU

        Returns:
            Optional[GPUInfo]: GPU信息，如果没有GPU则返回None
        """
        gpu_info_list = self.get_gpu_info()
        if not gpu_info_list:
            return None

        return max(gpu_info_list, key=lambda g: g.free_memory_mb)

    def record_experiment_memory(self, memory_used_mb: int):
        """
        记录实验使用的显存

        Args:
            memory_used_mb: 实验使用的显存（MB）
        """
        self.experiment_memory_history.append(memory_used_mb)

    def estimate_experiment_memory(self, percentile: int = 90) -> int:
        """
        估算实验所需显存（基于历史记录）

        Args:
            percentile: 使用第N百分位数作为估算值（默认90，表示保守估计）

        Returns:
            int: 估算的显存需求（MB）
        """
        if not self.experiment_memory_history:
            # 没有历史记录时，使用保守的默认值（5GB）
            return 5000

        import numpy as np
        history = np.array(self.experiment_memory_history)
        estimated = int(np.percentile(history, percentile))

        # 添加20%的安全边际
        return int(estimated * 1.2)

    def can_launch_experiment(self, estimated_memory_mb: Optional[int] = None,
                             safety_margin_mb: int = 1000) -> bool:
        """
        判断是否有足够显存启动新实验

        Args:
            estimated_memory_mb: 估算的显存需求（MB），如果为None则自动估算
            safety_margin_mb: 安全边际（MB），默认1000MB

        Returns:
            bool: True表示可以启动新实验，False表示显存不足
        """
        if estimated_memory_mb is None:
            estimated_memory_mb = self.estimate_experiment_memory()

        max_free = self.get_max_free_memory()
        required = estimated_memory_mb + safety_margin_mb

        return max_free >= required

    def wait_for_available_memory(self, estimated_memory_mb: Optional[int] = None,
                                  safety_margin_mb: int = 1000,
                                  check_interval_sec: int = 30) -> bool:
        """
        等待直到有足够显存可用（阻塞式）

        Args:
            estimated_memory_mb: 估算的显存需求（MB），如果为None则自动估算
            safety_margin_mb: 安全边际（MB）
            check_interval_sec: 检查间隔（秒）

        Returns:
            bool: True表示显存已可用，False表示中断等待
        """
        import time

        if estimated_memory_mb is None:
            estimated_memory_mb = self.estimate_experiment_memory()

        required = estimated_memory_mb + safety_margin_mb

        print(f"等待显存可用... (需要: {required}MB)")

        while not self.can_launch_experiment(estimated_memory_mb, safety_margin_mb):
            max_free = self.get_max_free_memory()
            print(f"  当前最大可用显存: {max_free}MB / {required}MB")
            time.sleep(check_interval_sec)

        print(f"✓ 显存已可用: {self.get_max_free_memory()}MB >= {required}MB")
        return True

    def print_gpu_status(self):
        """打印GPU状态信息"""
        gpu_info_list = self.get_gpu_info()

        if not gpu_info_list:
            print("未检测到GPU")
            return

        print("\n" + "="*80)
        print("GPU显存状态")
        print("="*80)
        print(f"{'ID':<5s} {'名称':<25s} {'已用':<10s} {'可用':<10s} {'总计':<10s} {'使用率':<10s}")
        print("-"*80)

        for gpu in gpu_info_list:
            print(
                f"{gpu.gpu_id:<5d} "
                f"{gpu.name:<25s} "
                f"{gpu.used_memory_mb:<10d} "
                f"{gpu.free_memory_mb:<10d} "
                f"{gpu.total_memory_mb:<10d} "
                f"{gpu.usage_percent:<10.1f}%"
            )

        print("="*80)

        # 打印历史记录统计
        if self.experiment_memory_history:
            print(f"\n实验显存历史记录: {len(self.experiment_memory_history)} 次")
            print(f"  平均显存使用: {sum(self.experiment_memory_history) / len(self.experiment_memory_history):.0f}MB")
            print(f"  最大显存使用: {max(self.experiment_memory_history)}MB")
            print(f"  估算显存需求 (90th percentile + 20%): {self.estimate_experiment_memory()}MB")
            print()


# 使用示例
if __name__ == "__main__":
    monitor = GPUMonitor()

    # 打印GPU状态
    monitor.print_gpu_status()

    # 模拟记录一些实验显存使用
    monitor.record_experiment_memory(4500)
    monitor.record_experiment_memory(4800)
    monitor.record_experiment_memory(4200)

    # 检查是否可以启动新实验
    print("\n检查是否可以启动新实验:")
    if monitor.can_launch_experiment():
        print("✓ 可以启动新实验")
        print(f"  最大可用显存: {monitor.get_max_free_memory()}MB")
        print(f"  估算显存需求: {monitor.estimate_experiment_memory()}MB")
    else:
        print("✗ 显存不足，无法启动新实验")
        print(f"  最大可用显存: {monitor.get_max_free_memory()}MB")
        print(f"  估算显存需求: {monitor.estimate_experiment_memory()}MB")
