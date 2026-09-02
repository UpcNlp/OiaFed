"""
FedEraser 遗忘训练器 — 编排联邦遗忘完整流程

流程:
1. 正常 FL 训练 (记录历史)   — round 1 ~ unlearn_after_round
2. 触发遗忘                  — 回滚到目标客户端参与前的检查点
3. 校准重训                  — 排除目标客户端，用剩余客户端重训 calibration_rounds 轮
4. 遗忘完成                  — 验证 & 报告

配合 FedEraserAggregator 使用。
"""

from typing import Dict, Any, List, Optional

from ...registry import trainer
from ...core.types import ClientUpdate, RoundResult, RoundMetrics
from .default import DefaultTrainer


@trainer(
    name='FedEraserTrainer',
    description='FedEraser 遗忘训练器 — 支持客户端级数据删除',
    version='1.0',
    author='OiaFed',
    algorithms=['faderaser', 'federated_unlearning']
)
class FedEraserTrainer(DefaultTrainer):
    """
    FedEraser 遗忘训练器

    在 DefaultTrainer 基础上添加:
    1. 在 unlearn_after_round 轮触发遗忘
    2. 回滚模型到检查点
    3. 排除目标客户端进行校准重训
    4. 评估遗忘效果

    配置参数:
        unlearn_after_round: int    — 第几轮后触发遗忘 (默认 10)
        target_clients: list[str]   — 要遗忘的客户端 ID 列表
        calibration_rounds: int     — 校准重训轮数 (来自 aggregator 配置)
        eval_interval: int          — 评估间隔
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.unlearn_after_round = self.config.get('unlearn_after_round', 10)
        self.target_clients = self.config.get('target_clients', [])
        self.eval_interval = self.config.get('eval_interval', 5)

        # 遗忘状态
        self._unlearn_triggered = False
        self._unlearn_completed = False
        self._pre_unlearn_metrics = {}
        self._post_unlearn_metrics = {}

        self.logger.info(
            f"FedEraserTrainer 初始化: "
            f"unlearn_after_round={self.unlearn_after_round}, "
            f"target_clients={self.target_clients}"
        )

    async def train_round(self, round_num: int) -> RoundResult:
        """
        执行一轮训练，在指定轮次触发遗忘
        """
        # === 阶段判断 ===
        if (round_num == self.unlearn_after_round
                and not self._unlearn_triggered
                and self.target_clients):
            return await self._trigger_unlearning(round_num)

        # === 正常训练轮次 ===
        result = await self._normal_train_round(round_num)

        # 定期评估
        if round_num % self.eval_interval == 0:
            await self._evaluate_and_log(round_num)

        return result

    # ------------------------------------------------------------------
    #  正常训练 (带历史记录)
    # ------------------------------------------------------------------
    async def _normal_train_round(self, round_num: int) -> RoundResult:
        """标准训练轮次 — 与 DefaultTrainer 一致，但聚合器会自动记录历史"""
        if self.callbacks:
            await self.callbacks.on_round_begin(self, round_num, {})

        config = self.config
        client_fraction = config.get("client_fraction", 1.0)
        fit_config = config.get("fit_config", {})

        # 选择客户端
        connected = self.get_connected_learners()

        # 遗忘触发后，排除目标客户端
        if self._unlearn_triggered and self.target_clients:
            connected = [
                l for l in connected
                if getattr(l, '_target_id', '') not in self.target_clients
            ]

        num_selected = max(1, int(len(connected) * client_fraction))
        selected = self.select_learners(num_selected, strategy="random")

        selected_ids = [getattr(l, '_target_id', 'unknown') for l in selected]
        phase = "校准重训" if self._unlearn_triggered else "正常训练"
        self.logger.info(
            f"Round {round_num} ({phase}): "
            f"选择 {len(selected)} 个客户端 {selected_ids}"
        )

        # 训练
        fit_config_with_round = {**fit_config, "round_number": round_num}
        results = await self.collect_results(selected, "fit", fit_config_with_round)

        # 收集更新
        updates = []
        for i, (learner, result) in enumerate(zip(selected, results)):
            learner_id = getattr(learner, '_target_id', f'learner_{i}')
            if isinstance(result, Exception):
                self.logger.error(f"  [{learner_id}] 训练失败: {result}")
                continue

            from ...core.types import TrainResult
            is_train_result = isinstance(result, TrainResult) or (
                type(result).__name__ == 'TrainResult'
                and hasattr(result, 'num_samples')
                and hasattr(result, 'metrics')
            )
            if is_train_result:
                updates.append(ClientUpdate.from_result(learner_id, result))

        if not updates:
            raise RuntimeError(f"Round {round_num}: 所有客户端训练失败")

        # 聚合 (FedEraserAggregator 会自动过滤遗忘客户端 + 记录历史)
        new_weights = self.aggregator.aggregate(updates, self.model)
        if self.model:
            self.set_weights(new_weights)

        # 广播
        await self.broadcast_to_learners("set_weights", new_weights)

        # 指标
        round_metrics = self._compute_round_metrics(updates, round_num)

        result = RoundResult(
            round_num=round_num,
            updates=updates,
            aggregated_weights=new_weights,
            metrics=round_metrics,
            metadata={"phase": phase}
        )

        if self.callbacks:
            await self.callbacks.on_round_end(self, round_num, {
                "metrics": round_metrics, "phase": phase
            })

        return result

    # ------------------------------------------------------------------
    #  触发遗忘
    # ------------------------------------------------------------------
    async def _trigger_unlearning(self, round_num: int) -> RoundResult:
        """
        触发遗忘流程

        1. 记录遗忘前的模型性能
        2. 调用 aggregator.request_unlearn() 回滚到检查点
        3. 设置遗忘状态，后续轮次自动排除目标客户端
        """
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Round {round_num}: 触发联邦遗忘!")
        self.logger.info(f"目标客户端: {self.target_clients}")
        self.logger.info(f"{'='*60}")

        # 记录遗忘前性能
        self.logger.info("评估遗忘前模型性能...")
        self._pre_unlearn_metrics = await self._evaluate_model("遗忘前")

        # 调用聚合器执行遗忘
        if hasattr(self.aggregator, 'request_unlearn'):
            checkpoint = self.aggregator.request_unlearn(
                client_ids=self.target_clients
            )

            if checkpoint is not None:
                # 回滚模型
                self.logger.info("回滚模型到检查点...")
                if self.model:
                    self.set_weights(checkpoint)
                await self.broadcast_to_learners("set_weights", checkpoint)
                self.logger.info("模型已回滚")
            else:
                self.logger.warning("未找到有效检查点，继续当前模型")
        else:
            self.logger.warning(
                "聚合器不支持 request_unlearn，仅排除目标客户端继续训练"
            )

        self._unlearn_triggered = True

        # 评估回滚后性能
        rollback_metrics = await self._evaluate_model("回滚后")

        self.logger.info(
            f"遗忘已触发。后续轮次将排除 {self.target_clients} 进行校准重训"
        )
        self.logger.info(f"{'='*60}\n")

        # 返回一个虚拟的 round result
        return RoundResult(
            round_num=round_num,
            updates=[],
            aggregated_weights=self.get_weights() if self.model else {},
            metrics=RoundMetrics(
                round_num=round_num,
                num_clients=0,
                total_samples=0,
                metrics={
                    "phase": "unlearn_trigger",
                    **{f"pre_{k}": v for k, v in self._pre_unlearn_metrics.items()},
                    **{f"rollback_{k}": v for k, v in rollback_metrics.items()},
                }
            ),
            metadata={"phase": "unlearn_trigger"}
        )

    # ------------------------------------------------------------------
    #  评估
    # ------------------------------------------------------------------
    async def _evaluate_model(self, tag: str = "") -> Dict[str, float]:
        """评估当前全局模型"""
        connected = self.get_connected_learners()
        if not connected:
            return {}

        # 选第一个客户端做评估
        eval_learner = connected[0]
        try:
            eval_results = await self.collect_results(
                [eval_learner], "evaluate", {}
            )
            result = eval_results[0]
            if hasattr(result, 'metrics') and isinstance(result.metrics, dict):
                metrics = result.metrics
                self.logger.info(
                    f"  [{tag}] 准确率={metrics.get('accuracy', 'N/A'):.4f}, "
                    f"损失={metrics.get('loss', 'N/A'):.4f}"
                )
                return metrics
        except Exception as e:
            self.logger.error(f"  [{tag}] 评估失败: {e}")
        return {}

    async def _evaluate_and_log(self, round_num: int):
        """定期评估并记录"""
        phase = "校准重训" if self._unlearn_triggered else "正常训练"
        metrics = await self._evaluate_model(f"Round {round_num} ({phase})")

        if (self._unlearn_triggered
                and not self._unlearn_completed
                and hasattr(self.aggregator, 'complete_unlearning')):
            # 检查是否达到校准轮数
            calibration_rounds = getattr(
                self.aggregator, 'calibration_rounds', 5
            )
            unlearn_round = self.unlearn_after_round
            rounds_since_unlearn = round_num - unlearn_round

            if rounds_since_unlearn >= calibration_rounds:
                self.logger.info(f"\n{'='*60}")
                self.logger.info(
                    f"校准重训完成 ({rounds_since_unlearn} 轮)"
                )

                # 记录遗忘后性能
                self._post_unlearn_metrics = await self._evaluate_model("遗忘后")

                # 对比
                pre_acc = self._pre_unlearn_metrics.get('accuracy', 0)
                post_acc = self._post_unlearn_metrics.get('accuracy', 0)
                self.logger.info(
                    f"性能对比: 遗忘前={pre_acc:.4f} → 遗忘后={post_acc:.4f} "
                    f"(差异={post_acc - pre_acc:+.4f})"
                )

                self.aggregator.complete_unlearning()
                self._unlearn_completed = True
                self.logger.info(f"{'='*60}\n")