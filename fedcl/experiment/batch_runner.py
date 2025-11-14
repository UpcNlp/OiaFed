"""
批量实验运行器
fedcl/experiment/batch_runner.py

功能：
- 支持批量运行多组实验
- 自动管理实验配置
- 并行或串行执行
"""

import asyncio
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime

from ..federated_learning import FederatedLearning


class BatchExperimentRunner:
    """
    批量实验运行器

    支持串行或并行运行多组实验
    """

    def __init__(self, base_config: str, experiment_variants: List[Dict[str, Any]]):
        """
        Args:
            base_config: 基础配置文件路径
            experiment_variants: 实验变体列表，每个变体是配置覆盖
                [
                    {
                        'name': 'exp1_fedavg',
                        'overrides': {'trainer.name': 'FedAvg', ...}
                    },
                    {
                        'name': 'exp2_fedprox',
                        'overrides': {'trainer.name': 'FedProx', ...}
                    }
                ]
        """
        self.base_config = base_config
        self.experiment_variants = experiment_variants
        self.results = []

    async def run_all(self,
                      parallel: bool = False,
                      max_parallel: int = 3,
                      callback: Optional[Callable] = None) -> List[Dict[str, Any]]:
        """
        运行所有实验

        Args:
            parallel: 是否并行运行
            max_parallel: 最大并行数量
            callback: 每个实验完成后的回调函数

        Returns:
            实验结果列表
        """
        if parallel:
            return await self._run_parallel(max_parallel, callback)
        else:
            return await self._run_serial(callback)

    async def _run_serial(self, callback: Optional[Callable] = None) -> List[Dict[str, Any]]:
        """串行运行所有实验"""
        results = []

        for i, variant in enumerate(self.experiment_variants):
            print(f"\n{'='*80}")
            print(f"运行实验 {i+1}/{len(self.experiment_variants)}: {variant['name']}")
            print(f"{'='*80}\n")

            start_time = time.time()

            try:
                result = await self._run_single_experiment(variant)
                result['duration'] = time.time() - start_time
                result['status'] = 'success'
                results.append(result)

                if callback:
                    callback(variant, result)

                print(f"✓ 实验完成: {variant['name']}")
                print(f"  准确率: {result.get('accuracy', 'N/A')}")
                print(f"  耗时: {result['duration']:.2f}s")

            except Exception as e:
                print(f"✗ 实验失败: {variant['name']}")
                print(f"  错误: {e}")
                results.append({
                    'name': variant['name'],
                    'status': 'failed',
                    'error': str(e),
                    'duration': time.time() - start_time
                })

        return results

    async def _run_parallel(self, max_parallel: int, callback: Optional[Callable] = None) -> List[Dict[str, Any]]:
        """并行运行实验"""
        semaphore = asyncio.Semaphore(max_parallel)

        async def run_with_semaphore(variant):
            async with semaphore:
                start_time = time.time()
                try:
                    result = await self._run_single_experiment(variant)
                    result['duration'] = time.time() - start_time
                    result['status'] = 'success'

                    if callback:
                        callback(variant, result)

                    return result
                except Exception as e:
                    return {
                        'name': variant['name'],
                        'status': 'failed',
                        'error': str(e),
                        'duration': time.time() - start_time
                    }

        tasks = [run_with_semaphore(v) for v in self.experiment_variants]
        results = await asyncio.gather(*tasks)
        return list(results)

    async def _run_single_experiment(self, variant: Dict[str, Any]) -> Dict[str, Any]:
        """运行单个实验"""
        exp_name = variant['name']

        # 创建 FederatedLearning 实例
        fl = FederatedLearning(self.base_config)

        try:
            # 初始化
            await fl.initialize()

            # 应用配置覆盖（如果有）
            # TODO: 实现配置覆盖逻辑

            # 准备实验记录配置
            exp_config = {
                'enabled': True,
                'name': exp_name,
                'base_dir': 'experiments/results'
            }

            # 运行训练（fl.run会自动处理实验记录的设置和清理）
            result = await fl.run(exp_config=exp_config)

            # 返回结果摘要
            return {
                'name': exp_name,
                'accuracy': result.final_accuracy if result else None,
                'loss': result.final_loss if result else None,
                'rounds': result.completed_rounds if result else None,
                'total_time': result.total_time if result else None
            }

        finally:
            # 清理资源（包括全局状态）
            await fl.cleanup(force_clear_global_state=True)


def create_grid_search_experiments(
    base_name: str,
    param_grid: Dict[str, List[Any]]
) -> List[Dict[str, Any]]:
    """
    创建网格搜索实验列表

    Args:
        base_name: 实验基础名称
        param_grid: 参数网格
            {
                'learning_rate': [0.01, 0.001],
                'batch_size': [32, 64],
                'aggregator': ['fedavg', 'fedprox']
            }

    Returns:
        实验变体列表
    """
    import itertools

    # 生成所有参数组合
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))

    experiments = []
    for i, combo in enumerate(combinations):
        # 生成实验名称
        param_str = '_'.join([f"{k}{v}" for k, v in zip(keys, combo)])
        exp_name = f"{base_name}_{i+1}_{param_str}"

        # 生成配置覆盖
        overrides = {k: v for k, v in zip(keys, combo)}

        experiments.append({
            'name': exp_name,
            'overrides': overrides
        })

    return experiments


def create_algorithm_comparison_experiments(
    base_name: str,
    algorithms: List[str],
    common_config: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    创建算法对比实验

    Args:
        base_name: 实验基础名称
        algorithms: 算法列表 ['fedavg', 'fedprox', 'scaffold']
        common_config: 公共配置

    Returns:
        实验变体列表
    """
    experiments = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for algo in algorithms:
        exp_name = f"{base_name}_{algo}_{timestamp}"
        overrides = {'trainer.name': algo}

        if common_config:
            overrides.update(common_config)

        experiments.append({
            'name': exp_name,
            'overrides': overrides
        })

    return experiments
