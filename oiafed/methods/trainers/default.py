"""
内置训练器实现
"""

from typing import Any, Dict, List, Optional
import asyncio

from ...core.trainer import Trainer
from ...core.types import ClientUpdate, TrainResult, RoundResult, RoundMetrics
from ...registry import trainer
from ...infra.logging import get_module_logger

logger = get_module_logger(__name__)


@trainer(
    name='default',
    description='默认FedAvg训练器 - 标准联邦平均训练流程',
    version='1.0',
    author='Federation Framework',
    algorithms=['fedavg']
)
class DefaultTrainer(Trainer):
    """
    默认 FedAvg 训练器

    标准联邦平均训练流程：
    1. 广播初始权重
    2. 轮次循环：
       a. 选择客户端
       b. 收集更新
       c. 聚合
       d. 广播新权重
       e. 评估（可选）
    """

    async def train_round(self, round_num: int) -> RoundResult:
        """
        单轮次训练 - 处理一个联邦学习轮次

        Args:
            round_num: 轮次编号

        Returns:
            RoundResult: 包含更新、权重、指标
        """
        # 触发轮次开始回调
        if self.callbacks:
            await self.callbacks.on_round_begin(self, round_num, {})

        config = self.config
        client_fraction = config.get("client_fraction", 1.0)
        fit_config = config.get("fit_config", {"epochs": 5})
        eval_interval = config.get("eval_interval", 10)

        # DEBUG: 输出配置信息
        self.logger.info(f"[Round {round_num}] DEBUG: config keys = {list(config.keys())}")
        self.logger.info(f"[Round {round_num}] DEBUG: client_fraction = {client_fraction}")

        # 1. 选择学习器
        connected = self.get_connected_learners()
        self.logger.info(f"[Round {round_num}] DEBUG: len(connected) = {len(connected)}")
        
        num_selected = max(1, int(len(connected) * client_fraction))
        self.logger.info(f"[Round {round_num}] DEBUG: num_selected = {num_selected}")
        
        selected = self.select_learners(num_selected, strategy="random")

        # DEBUG: 记录所有已连接和选中的学习器
        connected_ids = [getattr(l, '_target_id', 'unknown') for l in connected]
        selected_ids = [getattr(l, '_target_id', 'unknown') for l in selected]
        self.logger.debug(f"[Round {round_num}] 已连接学习器: {connected_ids}")
        self.logger.debug(f"[Round {round_num}] 选中学习器: {selected_ids}")

        self.logger.info(f"轮次 {round_num}: 选择了 {len(selected)} 个学习器")

        # 2. 收集训练结果（把 fit_config 作为位置参数传递）
        self.logger.debug(f"[Round {round_num}] 开始向 {selected_ids} 发送 fit 请求，配置: {fit_config}")
        results = await self.collect_results(selected, "fit", fit_config)
        self.logger.debug(f"[Round {round_num}] 收集到 {len(results)} 个响应")

        # 3. 过滤成功的结果并创建更新
        updates = []
        for i, (learner, result) in enumerate(zip(selected, results)):
            learner_id = getattr(learner, '_target_id', f'learner_{i}')

            # CRITICAL DEBUG: 强制记录每个learner的返回结果
            self.logger.debug(f"[Round {round_num}] Learner {learner_id} 返回: type={type(result).__name__}, is_Exception={isinstance(result, Exception)}")

            # 详细调试信息
            # self.logger.debug(f"[DEBUG-{learner_id}] result.__class__: {result.__class__}")
            # self.logger.debug(f"[DEBUG-{learner_id}] result.__class__.__module__: {result.__class__.__module__}")
            # self.logger.debug(f"[DEBUG-{learner_id}] TrainResult class: {TrainResult}")
            # self.logger.debug(f"[DEBUG-{learner_id}] TrainResult.__module__: {TrainResult.__module__}")
            # self.logger.debug(f"[DEBUG-{learner_id}] isinstance check: {isinstance(result, TrainResult)}")
            # self.logger.debug(f"[DEBUG-{learner_id}] type match: {type(result) == TrainResult}")
            # self.logger.debug(f"[DEBUG-{learner_id}] hasattr num_samples: {hasattr(result, 'num_samples')}")
            if hasattr(result, 'num_samples'):
                self.logger.debug(f"[DEBUG-{learner_id}] result.num_samples: {result.num_samples}")

            if isinstance(result, TrainResult):
                self.logger.info(f"samples={result.num_samples}, metrics_type={type(result.metrics)}")

            if isinstance(result, Exception):
                # DEBUG: 记录失败的学习器
                self.logger.error(f"[Round {round_num}] Learner {learner_id} 失败: {type(result).__name__}: {result}")
                # 使用 train_logger 记录完整的异常堆栈
                self.logger.exception(f"学习器失败", exc_info=result)
                raise result

            # 兼容性检查: 支持从不同模块路径导入的 TrainResult
            # (src.core.types.TrainResult 和 federation.core.types.TrainResult)
            is_train_result = isinstance(result, TrainResult) or (
                type(result).__name__ == 'TrainResult' and
                hasattr(result, 'num_samples') and
                hasattr(result, 'metrics')
            )

            if is_train_result:
                # DEBUG: 记录成功的学习器
                # result.metrics 是 TrainMetrics 对象，有 final_loss 和 metrics 字典
                loss_value = result.metrics.final_loss if hasattr(result.metrics, 'final_loss') else result.metrics.metrics.get('loss', 'N/A')
                self.logger.debug(f"[Round {round_num}] Learner {learner_id} 成功: samples={result.num_samples}, loss={loss_value}")
                updates.append(ClientUpdate.from_result(learner_id, result))

        if not updates:
            self.logger.info(f"轮次 {round_num}: 没有成功的更新", level="error")
            raise RuntimeError(f"轮次 {round_num}: 所有学习器都失败了")

        # 4. 聚合
        self.logger.debug(f"[Round {round_num}] 开始聚合，updates数量: {len(updates)}")
        self.logger.debug(f"[Round {round_num}] updates来源: {[u.client_id for u in updates]}")
        new_weights = self.aggregator.aggregate(updates, self.model)
        if self.model:
            self.model.set_weights(new_weights)
        self.logger.info(f"轮次 {round_num}: 聚合完成")

        # 5. 广播新权重到选中的学习器（而非所有学习器）
        self.logger.info(f"[Round {round_num}] 开始广播新权重到 {len(selected)} 个选中的学习器")
        self.logger.debug(f"[Round {round_num}] 广播目标: {[getattr(l, '_target_id', 'unknown') for l in selected]}")
        await self.broadcast_to_selected(selected, "set_weights", new_weights)
        self.logger.info(f"[Round {round_num}] 广播完成")

        # 6. 聚合后立即评估全局模型（如果配置）
        post_agg_metrics = {}
        if config.get("evaluate_after_aggregation", False):
            post_agg_metrics = await self._evaluate_global_model(round_num)

        # 7. 计算轮次指标（训练指标，不记录到tracker）
        round_metrics = self._compute_round_metrics(updates, round_num)
        # 合并聚合后评估指标
        round_metrics.metrics.update(post_agg_metrics)

        # 简化日志：只显示数据统计，不显示训练准确率（因为没意义）
        self.logger.info(
            f"轮次 {round_num}: "
            f"客户端数量={round_metrics.num_clients}, "
            f"训练样本数={round_metrics.total_samples}"
        )

        # 8. 定期评估（在所有客户端上评估，如果到了评估间隔）
        if eval_interval > 0 and round_num % eval_interval == 0:
            eval_metrics = await self._evaluate_round(round_num, selected)
            # 将评估指标合并到轮次指标
            round_metrics.metrics.update(eval_metrics)

            # 显著的日志输出：评估结果
            if eval_metrics:
                eval_str = ", ".join([f"{k}={v:.4f}" for k, v in eval_metrics.items()])
                self.logger.info(f"轮次 {round_num} 评估结果: {eval_str}")

        result = RoundResult(
            round_num=round_num,
            updates=updates,
            aggregated_weights=new_weights,
            metrics=round_metrics
        )

        # 触发轮次结束回调
        print(f"[DefaultTrainer-DEBUG] After round {round_num}: self.callbacks={self.callbacks}, bool={bool(self.callbacks)}")
        logger.info(f"[DefaultTrainer] After round {round_num}: self.callbacks={self.callbacks}, bool={bool(self.callbacks)}")

        if self.callbacks:
            print(f"[DefaultTrainer-DEBUG] Calling callbacks.on_round_end for round {round_num}")
            logger.info(f"[DefaultTrainer] Calling callbacks.on_round_end for round {round_num}")
            await self.callbacks.on_round_end(self, round_num, {"metrics": round_metrics})
            print(f"[DefaultTrainer-DEBUG] callbacks.on_round_end completed for round {round_num}")
            logger.info(f"[DefaultTrainer] callbacks.on_round_end completed for round {round_num}")
        else:
            print(f"[DefaultTrainer-DEBUG] NO CALLBACKS! self.callbacks is None or empty")
            logger.warning(f"[DefaultTrainer] NO CALLBACKS! self.callbacks is None or empty")

        return result

    def _compute_round_metrics(self, updates: List[ClientUpdate], round_num: int) -> RoundMetrics:
        """计算轮次指标"""
        if not updates:
            return RoundMetrics(
                round_num=round_num,
                num_clients=0,
                total_samples=0,
                metrics={}
            )

        total_samples = sum(u.num_samples for u in updates)

        # 聚合客户端指标
        all_metrics: Dict[str, List[float]] = {}
        for update in updates:
            for key, value in update.metrics.items():
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(value)

        # 计算平均值
        aggregated_metrics = {}
        for key, values in all_metrics.items():
            if len(values) > 0:  # 防止除零
                aggregated_metrics[f"avg_{key}"] = sum(values) / len(values)
            else:
                self.logger.warning(f"No values for metric '{key}', skipping average calculation")
                aggregated_metrics[f"avg_{key}"] = 0.0

        return RoundMetrics(
            round_num=round_num,
            num_clients=len(updates),
            total_samples=total_samples,
            metrics=aggregated_metrics
        )
    
    async def _evaluate_round(
        self,
        round_num: int,
        learners: List[Any],
    ) -> Dict[str, float]:
        """执行评估（在客户端测试集上）"""
        self.logger.info(f"轮次 {round_num} 开始评估...")

        eval_config = self.config.get("eval_config", {})
        results = await self.collect_results(learners, "evaluate", eval_config)

        # 聚合评估结果
        total_samples = 0
        weighted_metrics: Dict[str, float] = {}

        for result in results:
            if isinstance(result, Exception):
                continue

            samples = result.num_samples
            total_samples += samples

            for key, value in result.metrics.items():
                if key not in weighted_metrics:
                    weighted_metrics[key] = 0
                weighted_metrics[key] += value * samples

        # 计算加权平均
        eval_metrics = {}
        if total_samples > 0:
            for key, value in weighted_metrics.items():
                eval_metrics[f"eval_{key}"] = value / total_samples
            eval_metrics["eval_samples"] = total_samples

        # 注意：不在这里记录到 tracker，由上层统一记录
        # 这样可以避免重复记录

        return eval_metrics

    async def _evaluate_global_model(self, round_num: int) -> Dict[str, float]:
        """
        评估全局模型（在服务端测试集上）

        如果 Trainer 有 dataset，则在服务端数据上评估
        否则返回空字典
        """
        if not self.dataset:
            return {}

        self.logger.info(f"轮次 {round_num}: 评估全局模型")

        try:
            # 这里需要 Trainer 也有 evaluate 方法，或者使用模型直接评估
            # 简化实现：如果有 model 和 dataset，直接调用 model 的评估
            if hasattr(self.model, 'evaluate') and self.dataset:
                # 假设 model.evaluate 返回指标字典
                metrics = self.model.evaluate(self.dataset)
                global_metrics = {f"global_{k}": v for k, v in metrics.items()}
                if self.tracker:
                    self.tracker.log_metrics(global_metrics, step=round_num)
                return global_metrics
        except Exception as e:
            self.logger.exception(f"全局模型评估失败: {e}")

        return {}


@trainer(
    name='async',
    description='异步训练器 - 不等待所有客户端完成，收到更新即聚合',
    version='1.0',
    author='Federation Framework',
    algorithms=['async_fedavg']
)
class AsyncTrainer(Trainer):
    """
    异步训练器
    
    不等待所有客户端完成，收到更新即聚合
    """
    
    async def run(self) -> Dict[str, Any]:
        """运行异步训练流程"""
        config = self.config
        max_updates = config.get("max_updates", 1000)
        staleness_threshold = config.get("staleness_threshold", 10)
        fit_config = config.get("fit_config", {"epochs": 1})

        # 获取学习器
        learners = self.get_connected_learners()
        self.logger.info(f"异步训练开始，学习器数量: {len(learners)}")

        # 广播初始权重
        if self.model:
            await self.broadcast_to_learners("set_weights", self.model.get_weights())

        update_count = 0
        client_versions: Dict[str, int] = {}  # 客户端的模型版本
        current_version = 0

        while update_count < max_updates:
            connected = self.get_connected_learners()
            if not connected:
                self.logger.warning("没有连接的学习器，等待中...")
                await asyncio.sleep(1)
                continue

            # 对每个客户端发起训练请求
            for learner in connected:
                learner_id = getattr(learner, '_target_id', 'unknown')
                if learner_id not in client_versions:
                    client_versions[learner_id] = current_version

                # 检查过时程度
                staleness = current_version - client_versions[learner_id]
                if staleness > staleness_threshold and self.model:
                    # 发送最新权重
                    await self.broadcast_to_learners("set_weights", self.model.get_weights())
                    client_versions[learner_id] = current_version

            # 收集一个更新
            results = await self.collect_results(connected[:1], "fit", fit_config)

            for learner, result in zip(connected[:1], results):
                if isinstance(result, Exception):
                    continue

                learner_id = getattr(learner, '_target_id', 'unknown')
                update = ClientUpdate.from_result(learner_id, result)

                # 异步聚合（简化：直接平均）
                staleness = current_version - client_versions[learner_id]
                weight = 1.0 / (1.0 + staleness * 0.1)  # 过时惩罚

                if self.model:
                    current_weights = self.model.get_weights()
                    new_weights = self._weighted_average(
                        current_weights, update.weights, 1 - weight, weight
                    )
                    self.model.set_weights(new_weights)

                current_version += 1
                client_versions[learner_id] = current_version
                update_count += 1

                if update_count % 100 == 0:
                    self.logger.info(f"已处理 {update_count} 个更新")

        self.logger.success("异步训练完成")

        return {
            "total_updates": update_count,
            "final_version": current_version,
            "client_versions": client_versions,
        }