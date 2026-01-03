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


            if hasattr(result, 'num_samples'):
                self.logger.debug(f"[DEBUG-{learner_id}] result.num_samples: {result.num_samples}")

            if isinstance(result, TrainResult):
                self.logger.debug(f"[Round {round_num}] Learner {learner_id} 返回: num_samples={result.num_samples}, metrics={result.metrics}")
                # self.logger.info(f"samples={result.num_samples}, metrics_type={type(result.metrics)}")

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
            self.set_weights(new_weights)
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

        # 8. 定期评估（使用聚合后的全局模型在全局测试集上评估）
        if eval_interval > 0 and round_num % eval_interval == 0:
            if self.has_global_test:
                eval_metrics = await self._evaluate_on_global_test(round_num)
                # 将评估指标合并到轮次指标
                round_metrics.metrics.update(eval_metrics)

                # 显著的日志输出：评估结果
                if eval_metrics:
                    eval_str = ", ".join([f"{k}={v:.4f}" for k, v in eval_metrics.items() if isinstance(v, float)])
                    self.logger.info(f"轮次 {round_num} 评估结果: {eval_str}")
            else:
                self.logger.warning(
                    f"轮次 {round_num}: 跳过评估（Trainer 未配置全局测试集）。"
                    f"请在 trainer.yaml 中添加 datasets 配置。"
                )

        result = RoundResult(
            round_num=round_num,
            updates=updates,
            aggregated_weights=new_weights,
            metrics=round_metrics
        )

        # 触发轮次结束回调
        if self.callbacks:
            await self.callbacks.on_round_end(self, round_num, {"metrics": round_metrics})

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
    
    async def _evaluate_on_global_test(self, round_num: int) -> Dict[str, float]:
        """
        在 Trainer 端用全局测试集评估聚合后的模型
        
        这是真正的全局评估，使用完整的测试集（不划分）
        
        Args:
            round_num: 当前轮次号
            
        Returns:
            评估指标字典，包含 eval_accuracy, eval_loss, eval_samples
        """
        # 检查是否有全局测试集
        if not self.has_global_test:
            self.logger.warning("全局测试集不存在，跳过 Trainer 端评估")
            return {}
        
        # 检查是否有模型
        if not self.model:
            self.logger.warning("模型不存在，跳过 Trainer 端评估")
            return {}
        
        self.logger.info(f"轮次 {round_num}: 在全局测试集上评估...")
        
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader
            
            # 获取全局测试集
            test_dataset = self.test_dataset
            
            # 获取设备配置
            device_str = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
            device = torch.device(device_str)
            
            # 获取模型（支持 Model 包装器和原生 nn.Module）
            if hasattr(self.model, '_model'):
                # Model 包装器
                torch_model = self.model._model
            else:
                # 原生 nn.Module
                torch_model = self.model
            
            # 移动模型到设备并设置为评估模式
            torch_model = torch_model.to(device)
            torch_model.eval()
            
            # 创建 DataLoader
            batch_size = self.config.get("eval_batch_size", 64)
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,  # 避免多进程问题
            )
            
            # 评估
            criterion = nn.CrossEntropyLoss()
            correct = 0
            total = 0
            total_loss = 0.0
            num_batches = 0
            
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    
                    # 前向传播
                    outputs = torch_model(batch_x)
                    
                    # 支持字典输出（如 {"logits": ...}）
                    if isinstance(outputs, dict):
                        outputs = outputs.get("logits", outputs.get("output", outputs))
                    
                    loss = criterion(outputs, batch_y)
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
                    total_loss += loss.item()
                    num_batches += 1
            
            # 计算指标
            accuracy = correct / total if total > 0 else 0.0
            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            
            eval_metrics = {
                "eval_accuracy": accuracy,
                "eval_loss": avg_loss,
                "eval_samples": total,
            }
            
            # 记录到 tracker
            if self.tracker:
                self.tracker.log_metrics(eval_metrics, step=round_num)
            
            self.logger.info(
                f"轮次 {round_num} 全局评估: "
                f"accuracy={accuracy:.4f}, loss={avg_loss:.4f}, samples={total}"
            )
            
            return eval_metrics
            
        except ImportError as e:
            self.logger.error(f"PyTorch 未安装，无法进行全局评估: {e}")
            return {}
        except Exception as e:
            self.logger.exception(f"全局评估失败: {e}")
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
            await self.broadcast_to_learners("set_weights", self.get_weights())

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
                    await self.broadcast_to_learners("set_weights", self.get_weights())
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
                    current_weights = self.get_weights()
                    new_weights = self._weighted_average(
                        current_weights, update.weights, 1 - weight, weight
                    )
                    self.set_weights(new_weights)

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