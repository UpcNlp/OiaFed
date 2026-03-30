"""
FOT 持续学习训练器 — 在任务边界执行 GPSE

扩展 ContinualTrainer, 添加:
- 任务结束时的 GPSE 通信轮次 (收集客户端激活 → 服务端 SVD)
- 将 orth_set 传递给 FedProject 聚合器
- 广播 orth_set 给客户端 (GPSE 去重所需)

与官方代码流程一致:
1. 正常 FL 轮次: 训练 → FedProject 聚合 → 广播
2. 任务结束轮: 上述 + GPSE (收集激活 → expand_orth_set)
"""

import asyncio
from typing import Dict, Any, List, Optional

from ...registry import trainer
from .continual import ContinualTrainer


@trainer(
    name='FOTContinual',
    description='FOT 持续学习训练器 — 带 GPSE 通信轮次',
    version='1.0',
    author='FOT (Bakman et al.)',
    algorithms=['fot', 'federated_orthogonal_training']
)
class FOTContinualTrainer(ContinualTrainer):
    """
    FOT 持续学习训练器

    继承 ContinualTrainer 的所有功能 (任务调度, CL 指标), 添加:
    1. 任务结束时的 GPSE 激活收集
    2. 与 FedProject 聚合器的协同

    要求:
    - aggregator 必须是 FedProjectAggregator (有 expand_orth_set 和 orth_set)
    - learner 必须是 FOTLearner (有 collect_activations 方法)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger.info("FOTContinualTrainer 初始化完成")

    async def train_round(self, round_num: int):
        """
        执行一轮联邦训练

        在 ContinualTrainer 的基础上, 任务结束时增加 GPSE 轮次
        """
        # --- 调用父类的 train_round ---
        result = await super().train_round(round_num)

        # --- 检查是否是任务结束轮 ---
        config = self.config
        max_rounds = config.get("num_rounds", config.get("max_rounds", 100))
        is_task_end = (round_num % self.rounds_per_task == 0) or (round_num == max_rounds)

        if is_task_end and self.current_task_id < self.num_tasks - 1:
            # 检查聚合器是否支持 GPSE
            if hasattr(self.aggregator, 'expand_orth_set'):
                await self._run_gpse_round(round_num)
            else:
                self.logger.warning(
                    "聚合器不支持 GPSE (无 expand_orth_set 方法). "
                    "请使用 FedProjectAggregator."
                )

        return result

    async def _run_gpse_round(self, round_num: int):
        """
        执行 GPSE 通信轮次

        流程:
        1. 从所有客户端收集当前任务的层激活
        2. 传给 FedProject 聚合器做 SVD, 扩展 orth_set
        """
        self.logger.info(f"\n{'='*60}")
        self.logger.info(
            f"GPSE 轮次: Task {self.current_task_id} 结束, "
            f"开始全局主子空间提取"
        )
        self.logger.info(f"{'='*60}")

        # 获取当前 orth_set (用于激活去重)
        orth_set = {}
        if hasattr(self.aggregator, 'orth_set'):
            orth_set = self.aggregator.orth_set

        # 准备配置
        gpse_config = {
            'task_id': self.current_task_id,
            'orth_set': orth_set,
        }

        # 从所有客户端收集激活
        connected = self.get_connected_learners()
        self.logger.info(f"  从 {len(connected)} 个客户端收集激活...")

        activation_dict = {}
        for i, learner_proxy in enumerate(connected):
            try:
                # 调用 learner 的 collect_activations 方法
                activations = await learner_proxy.collect_activations(gpse_config)
                if activations:
                    learner_id = getattr(learner_proxy, '_target_id', f'learner_{i}')
                    activation_dict[learner_id] = activations
                    self.logger.info(
                        f"  [{learner_id}] 激活收集成功: "
                        f"{len(activations)} 层"
                    )
            except Exception as e:
                self.logger.error(f"  客户端 {i} 激活收集失败: {e}")

        if not activation_dict:
            self.logger.warning("GPSE: 未收集到任何激活, 跳过")
            return

        # 传给聚合器做 GPSE
        self.logger.info(f"  聚合 {len(activation_dict)} 个客户端的激活...")
        self.aggregator.expand_orth_set(activation_dict)

        # 打印正交空间使用率
        if hasattr(self.aggregator, 'orth_set'):
            for key, val in self.aggregator.orth_set.items():
                if val is not None:
                    usage = val.shape[1] / val.shape[0]
                    self.logger.info(
                        f"  orth_set[{key}]: {val.shape} "
                        f"(空间使用率: {usage:.3f})"
                    )

        self.logger.info(f"GPSE 轮次完成\n")
