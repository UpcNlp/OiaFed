"""
Learner（客户端学习器）抽象基类

分层训练架构：
fit() → train() → train_epoch() → train_step()
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, TYPE_CHECKING, List
import asyncio

from .types import TrainResult, EvalResult, StepMetrics, EpochMetrics, TrainMetrics, NodeState
from ..infra.logging import get_logger

if TYPE_CHECKING:
    from .model import Model
    from ..infra.tracker import Tracker
    from ..callback.manager import CallbackManager


class Learner(ABC):
    """
    客户端学习器抽象基类

    职责：
    - 本地模型训练
    - 本地模型评估
    - 管理本地模型状态

    用户需要实现 train_step() 方法来定义单批次训练逻辑

    Callback 触发点：
    - on_fit_start: fit() 开始时
    - on_fit_end: fit() 结束时
    - on_epoch_start: train_epoch() 开始时
    - on_epoch_end: train_epoch() 结束时
    """

    def __init__(
        self,
        # ========== 核心依赖 ==========
        model: "Model",                          # 本地模型
        data: Optional[Any] = None,              # 数据集/数据提供者

        # ========== 可选依赖 ==========
        tracker: Optional["Tracker"] = None,     # 指标追踪器（所有记录通过它）
        callbacks: Optional["CallbackManager"] = None,  # 回调管理器

        # ========== 配置 ==========
        config: Optional[Dict[str, Any]] = None, # 学习器配置
        node_id: Optional[str] = None,           # 节点ID（用于日志）
    ):
        """
        初始化学习器

        Args:
            model: 本地模型
            data: 数据集/数据提供者
            tracker: 指标追踪器（所有记录通过它）
            callbacks: 回调管理器
            config: 学习器配置
            node_id: 节点ID（用于日志）
        """
        self._model = model
        self._data = data
        self._tracker = tracker
        self._callbacks = callbacks
        self._config = config or {}
        self._node_id = node_id or "unknown"

        # 获取 loguru logger（从已初始化的日志系统）
        self.logger = get_logger(self._node_id, "training")

        # 节点状态
        self._state = NodeState.IDLE

        # 训练状态
        self._current_epoch = 0
        self._current_step = 0
        self._global_step = 0
        self._global_epoch_counter = 0  # 全局 epoch 计数器，用于 MLflow step（跨多次 fit 调用）
    
    @property
    def model(self) -> "Model":
        """获取模型"""
        return self._model

    @property
    def data(self) -> Optional[Any]:
        """获取数据集"""
        return self._data

    @property
    def tracker(self) -> Optional["Tracker"]:
        """获取追踪器"""
        return self._tracker

    @property
    def callbacks(self) -> Optional[Any]:
        """获取回调管理器"""
        return self._callbacks

    @property
    def config(self) -> Dict[str, Any]:
        """获取配置"""
        return self._config

    @property
    def node_id(self) -> str:
        """获取节点ID"""
        return self._node_id
    
    # ==================== 会话层 (Session Level) ====================

    async def fit(self, config: Optional[Dict[str, Any]] = None) -> TrainResult:
        """
        训练会话入口 - 联邦学习的标准接口

        职责:
        1. 合并运行配置
        2. 初始化训练环境（setup）
        3. 执行训练（调用 train）
        4. 收集结果并返回
        5. 触发 callbacks

        Args:
            config: 运行时配置（会覆盖初始化配置）

        Returns:
            TrainResult: 包含权重、指标、元数据的训练结果
        """
        # 设置状态为TRAINING
        self._state = NodeState.TRAINING

        try:
            # DEBUG: 记录接收到 fit 请求
            self.logger.debug(f"[{self._node_id}] 接收到 fit 请求, config={config}")

            # 合并配置
            run_config = {**self._config, **(config or {})}
            epochs = run_config.get("epochs", 1)

            self.logger.debug(f"[{self._node_id}] 合并后配置: epochs={epochs}, run_config={run_config}")

            # 触发 fit 开始回调
            if self._callbacks:
                await self._callbacks.on_fit_start(self, run_config)

            # 初始化训练环境
            await self.setup(run_config)

            try:
                # 执行训练
                train_metrics = await self.train(epochs=epochs)

                # 训练后自动评估（如果配置了 evaluate_after_fit）
                eval_metrics = None
                if run_config.get("evaluate_after_fit", False):
                    try:
                        eval_result = await self.evaluate(config=run_config.get("eval_config", {}))
                        # 将评估指标合并到训练指标中
                        if eval_result and eval_result.metrics:
                            eval_metrics = {f"post_train_{k}": v for k, v in eval_result.metrics.items()}
                            train_metrics.metrics.update(eval_metrics)

                            # 触发评估结束回调
                            if self._callbacks:
                                await self._callbacks.on_evaluate_end(
                                    self,
                                    eval_result,
                                    context={
                                        "type": "post_training",
                                        "epochs": epochs,
                                    }
                                )
                    except Exception as e:
                        self.logger.warning(f"Post-training evaluation failed: {e}")

                # 收集结果
                result = TrainResult(
                    weights=self.get_weights(),
                    num_samples=self.get_num_samples(),
                    metrics=train_metrics,
                    metadata=self.get_metadata()
                )

                # 触发 fit 结束回调
                if self._callbacks:
                    await self._callbacks.on_fit_end(self, result)

                return result

            finally:
                # 清理资源
                await self.teardown()

        finally:
            # 恢复状态为IDLE
            self._state = NodeState.IDLE

    # ==================== 训练层 (Training Level) ====================

    async def train(self, epochs: int) -> TrainMetrics:
        """
        训练循环 - 管理多个 epoch

        职责：
        1. 循环执行 train_epoch
        2. 收集和聚合所有 epoch 的指标
        3. 返回完整训练指标

        Args:
            epochs: 训练轮数

        Returns:
            TrainMetrics: 完整训练过程的指标

        扩展点：
        - 重写此方法实现自定义训练流程（如课程学习）
        """
        epoch_history = []

        # 检查配置中是否启用进度条（默认禁用，避免多进程干扰）
        enable_pbar = self._config.get('enable_progress_bar', False) if self._config else False
        disable_pbar = not enable_pbar

        # 创建 epoch 级别的进度条
        try:
            from tqdm import tqdm
            epoch_pbar = tqdm(
                range(1, epochs + 1),
                desc=f"[{self._node_id}] Training",
                position=None,
                leave=True,
                ncols=100,
                disable=disable_pbar,
                file=None  # 使用默认输出流
            )
            use_pbar = not disable_pbar
        except ImportError:
            self.logger.warning("tqdm not installed, progress bar disabled")
            epoch_pbar = range(1, epochs + 1)
            use_pbar = False

        for epoch in epoch_pbar:
            self._current_epoch = epoch

            # 执行单个 epoch
            epoch_metrics = await self.train_epoch(epoch_idx=epoch)
            epoch_history.append(epoch_metrics)

            # 更新进度条显示的指标
            if use_pbar:
                postfix_dict = {
                    'loss': f'{epoch_metrics.avg_loss:.4f}'
                }
                if epoch_metrics.metrics:
                    # 添加准确率等其他指标
                    for k, v in epoch_metrics.metrics.items():
                        if isinstance(v, (int, float)) and k != 'loss':
                            postfix_dict[k] = f'{v:.4f}'
                epoch_pbar.set_postfix(postfix_dict)
            else:
                # 如果没有进度条，输出日志
                metrics_str = f"loss={epoch_metrics.avg_loss:.4f}"
                if epoch_metrics.metrics:
                    for k, v in epoch_metrics.metrics.items():
                        if isinstance(v, (int, float)) and k != 'loss':
                            metrics_str += f", {k}={v:.4f}"
                self.logger.info(f"[{self._node_id}] Epoch {epoch}/{epochs} completed - {metrics_str}")

        # 汇总指标
        total_samples = sum(em.total_samples for em in epoch_history)
        final_loss = epoch_history[-1].avg_loss if epoch_history else 0.0

        # 聚合 epoch 指标
        aggregated_metrics = self._aggregate_epoch_metrics(epoch_history)
        # 将 final_loss 添加到指标字典中，方便轮次聚合时访问
        aggregated_metrics['loss'] = final_loss

        train_metrics = TrainMetrics(
            total_epochs=epochs,
            final_loss=final_loss,
            total_samples=total_samples,
            metrics=aggregated_metrics,
            epoch_history=epoch_history
        )

        # 记录训练指标到 Tracker（如果有）
        if self._tracker:
            metrics_to_log = {
                'train_loss': final_loss,
                'train_samples': float(total_samples),
            }
            # 添加其他指标
            for k, v in aggregated_metrics.items():
                if isinstance(v, (int, float)):
                    metrics_to_log[f'train_{k}'] = float(v)
            
            # 使用全局 epoch 计数器作为 step
            self._tracker.log_metrics(metrics_to_log, step=self._global_epoch_counter)
            self.logger.debug(f"[{self._node_id}] Logged train metrics to tracker: {list(metrics_to_log.keys())}")

        return train_metrics

    # ==================== 轮次层 (Epoch Level) ====================

    async def train_epoch(self, epoch_idx: int) -> EpochMetrics:
        """
        单轮训练 - 遍历所有批次

        职责：
        1. 触发 epoch 开始回调
        2. 循环调用 train_step 处理所有批次
        3. 收集和聚合 step 指标
        4. 触发 epoch 结束回调
        5. 返回 epoch 指标

        Args:
            epoch_idx: 当前 epoch 索引（从1开始）

        Returns:
            EpochMetrics: 单轮训练的指标

        扩展点：
        - 重写此方法实现梯度累积、混合精度等
        """
        # 触发 epoch 开始回调
        if self._callbacks:
            await self._callbacks.on_epoch_start(self, epoch_idx)

        # 获取数据加载器（子类需要实现 get_dataloader）
        dataloader = self.get_dataloader()

        step_metrics_list = []
        total_samples = 0

        # 检查配置中是否启用进度条（默认禁用，避免多进程干扰）
        enable_pbar = self._config.get('enable_progress_bar', False) if self._config else False
        disable_pbar = not enable_pbar

        # 创建批次级别的进度条
        try:
            from tqdm import tqdm
            batch_pbar = tqdm(
                enumerate(dataloader),
                total=len(dataloader),
                desc=f"[{self._node_id}] Epoch {epoch_idx}",
                position=None,
                leave=False,
                ncols=100,
                disable=disable_pbar,
                file=None  # 使用默认输出流
            )
            use_pbar = not disable_pbar
        except (ImportError, TypeError):
            # ImportError: tqdm 未安装
            # TypeError: dataloader 没有 __len__ 方法
            batch_pbar = enumerate(dataloader)
            use_pbar = False

        # 遍历所有批次
        for batch_idx, batch in batch_pbar:
            self._current_step = batch_idx
            self._global_step += 1

            # 执行单步训练
            step_metrics = await self.train_step(batch, batch_idx)
            step_metrics_list.append(step_metrics)
            total_samples += step_metrics.batch_size

            # 更新批次进度条
            if use_pbar:
                postfix_dict = {'loss': f'{step_metrics.loss:.4f}'}
                if step_metrics.metrics:
                    for k, v in step_metrics.metrics.items():
                        if isinstance(v, (int, float)) and k != 'loss':
                            postfix_dict[k] = f'{v:.4f}'
                batch_pbar.set_postfix(postfix_dict)

        # 如果没有使用进度条，在epoch结束时记录简要统计

        # 计算平均指标
        avg_loss = sum(sm.loss * sm.batch_size for sm in step_metrics_list) / total_samples if total_samples > 0 else 0.0

        epoch_metrics = EpochMetrics(
            epoch=epoch_idx,
            avg_loss=avg_loss,
            total_samples=total_samples,
            metrics=self._aggregate_step_metrics(step_metrics_list)
        )

        # 增加全局 epoch 计数器（用于 MLflow step）
        self._global_epoch_counter += 1

        # 触发 epoch 结束回调
        if self._callbacks:
            await self._callbacks.on_epoch_end(self, epoch_idx, epoch_metrics)

        return epoch_metrics

    # ==================== 批次层 (Step Level) ====================

    @abstractmethod
    async def train_step(self, batch: Any, batch_idx: int) -> StepMetrics:
        """
        单批次训练 - 用户必须实现

        职责：
        1. 将数据移到设备
        2. 前向传播计算损失
        3. 反向传播
        4. 优化器更新
        5. 返回指标

        Args:
            batch: 批次数据
            batch_idx: 批次索引

        Returns:
            StepMetrics: 包含 loss、batch_size、metrics

        典型实现：
        ```python
        async def train_step(self, batch, batch_idx):
            inputs, targets = batch
            inputs = inputs.to(self._device)
            targets = targets.to(self._device)

            # 前向传播
            outputs = self.model(inputs)
            loss = self._criterion(outputs, targets)

            # 反向传播
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

            # 返回指标
            return StepMetrics(
                loss=loss.item(),
                batch_size=inputs.size(0),
                metrics={"accuracy": compute_accuracy(outputs, targets)}
            )
        ```

        扩展点：
        - 标准训练：实现上述典型流程
        - 对抗训练：在此方法中生成对抗样本并训练
        - 多任务学习：计算多个损失并联合优化
        """
        pass

    # ==================== 环境管理（用户需要实现）====================

    @abstractmethod
    async def setup(self, config: Dict) -> None:
        """
        初始化训练环境

        职责：
        1. 创建 DataLoader
        2. 创建 Optimizer
        3. 创建 Criterion
        4. 设置 Device
        5. 其他初始化工作

        典型实现：
        ```python
        async def setup(self, config):
            # 创建 DataLoader
            self._train_dataloader = DataLoader(
                self.data,
                batch_size=config.get('batch_size', 32),
                shuffle=True
            )

            # 创建优化器
            self._optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=config.get('lr', 0.01)
            )

            # 创建损失函数
            self._criterion = nn.CrossEntropyLoss()

            # 设置设备
            self._device = torch.device(
                config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
            )
            self.model.to(self._device)
        ```
        """
        pass

    async def teardown(self) -> None:
        """
        清理资源（可选重写）

        默认实现：无操作
        用户可重写以添加自定义清理逻辑
        """
        pass

    @abstractmethod
    def get_dataloader(self) -> Any:
        """
        获取数据加载器

        用户必须实现此方法以返回训练数据的迭代器

        Returns:
            可迭代的数据加载器
        """
        pass

    @abstractmethod
    def get_num_samples(self) -> int:
        """
        获取训练样本数

        典型实现：
        ```python
        def get_num_samples(self):
            return len(self.data)
        ```
        """
        pass

    def get_metadata(self) -> Dict[str, Any]:
        """
        获取训练元数据（可选重写）

        返回训练过程的额外信息
        """
        return {
            "node_id": self._node_id,
            "total_steps": self._global_step,
        }

    # ==================== 辅助方法 ====================

    def _aggregate_step_metrics(self, step_metrics_list: List[StepMetrics]) -> Dict[str, float]:
        """聚合多个 step 的指标"""
        if not step_metrics_list:
            return {}

        # 收集所有指标的 key
        all_keys = set()
        for sm in step_metrics_list:
            all_keys.update(sm.metrics.keys())

        # 加权平均
        aggregated = {}
        total_samples = sum(sm.batch_size for sm in step_metrics_list)

        if total_samples == 0:
            return {}

        for key in all_keys:
            weighted_sum = sum(
                sm.metrics.get(key, 0) * sm.batch_size
                for sm in step_metrics_list
            )
            aggregated[key] = weighted_sum / total_samples

        return aggregated

    def _aggregate_epoch_metrics(self, epoch_metrics_list: List[EpochMetrics]) -> Dict[str, float]:
        """聚合多个 epoch 的指标"""
        if not epoch_metrics_list:
            return {}

        # 收集所有指标的 key
        all_keys = set()
        for em in epoch_metrics_list:
            all_keys.update(em.metrics.keys())

        # 加权平均
        aggregated = {}
        total_samples = sum(em.total_samples for em in epoch_metrics_list)

        if total_samples == 0:
            return {}

        for key in all_keys:
            weighted_sum = sum(
                em.metrics.get(key, 0) * em.total_samples
                for em in epoch_metrics_list
            )
            aggregated[key] = weighted_sum / total_samples

        return aggregated

    # ==================== 评估方法（保持不变）====================

    @abstractmethod
    async def evaluate(self, config: Optional[Dict[str, Any]] = None) -> EvalResult:
        """
        执行本地评估

        Args:
            config: 评估配置

        Returns:
            EvalResult 包含:
            - num_samples: 评估样本数
            - metrics: 评估指标（如 accuracy, f1）
            - metadata: 其他信息
        """
        pass
    
    # ==================== 框架提供的方法（无需覆盖） ====================
    
    def get_model(self) -> "Model":
        """获取模型"""
        return self._model
    
    def set_model(self, model: "Model") -> None:
        """设置模型"""
        self._model = model
    
    def get_weights(self) -> Any:
        """获取模型权重"""
        # 检查是否是 PyTorch 模型
        try:
            import torch.nn as nn
            if isinstance(self._model, nn.Module):
                return self._model.state_dict()
        except ImportError:
            pass

        # 尝试调用模型的 get_weights 方法（如果有的话）
        if hasattr(self._model, 'get_weights'):
            return self._model.get_weights()

        # 否则直接返回模型（可能是numpy数组等）
        return self._model
    
    def set_weights(self, weights: Any) -> bool:
        """设置模型权重

        Returns:
            True 表示设置成功
        """
        self.logger.info(f"[{self.node_id}] 收到 set_weights 请求")
        try:
            # 检查是否是 PyTorch 模型
            try:
                import torch.nn as nn
                if isinstance(self._model, nn.Module):
                    # 使用 strict=False 以支持部分参数加载（例如 FedBN 不包含 BN 层参数）
                    self._model.load_state_dict(weights, strict=False)
                    self.logger.info(f"[{self.node_id}] set_weights 成功 (PyTorch)")
                    return True
            except ImportError:
                pass

            # 尝试调用模型的 set_weights 方法
            if hasattr(self._model, 'set_weights'):
                self._model.set_weights(weights)
                self.logger.info(f"[{self.node_id}] set_weights 成功")
                return True

            # 直接设置模型（可能是numpy数组等）
            self._model = weights
            self.logger.info(f"[{self.node_id}] set_weights 成功 (direct)")
            return True

        except Exception as e:
            self.logger.exception(f"Failed to set weights: {e}")
            return str(e)
    
    def get_data_info(self) -> Dict[str, Any]:
        """
        获取数据信息
        
        Returns:
            数据信息字典，包含 num_samples 等
        """
        if self._data is None:
            return {"num_samples": 0}
        return self._data.get_info()
    
    def set_data(self, data: "DataProvider") -> None:
        """设置数据提供者"""
        self._data = data
    
    def set_tracker(self, tracker: "Tracker") -> None:
        """设置追踪器"""
        self._tracker = tracker
    
    # ==================== 生命周期钩子（可选覆盖） ====================
    
    def on_init(self) -> None:
        """初始化后回调"""
        pass
    
    def on_model_received(self, weights: Any) -> None:
        """收到新模型权重时回调"""
        pass
    
    def before_fit(self, config: Dict[str, Any]) -> None:
        """训练前回调"""
        pass
    
    def after_fit(self, result: TrainResult) -> None:
        """训练后回调"""
        pass
    
    def on_shutdown(self) -> None:
        """关闭前回调"""
        pass
    
    # ==================== 辅助方法 ====================

    def log(self, message: str, level: str = "info", **kwargs) -> None:
        """
        记录日志/事件

        使用 loguru
        """
        getattr(self.logger, level)(message)

    async def _apply_sync_info(self, sync_info: Dict[str, Any]):
        """
        应用同步信息（Learner端）

        遍历所有 SyncCallback，应用同步信息
        """
        if not self._callbacks:
            return

        from ..callback.sync_callback import SyncCallback

        # 应用所有 SyncCallback 的同步信息
        for callback in self._callbacks._callbacks:
            if isinstance(callback, SyncCallback):
                try:
                    await callback.apply_sync_info(sync_info)
                except Exception as e:
                    self.logger.error(f"Failed to apply sync info to {callback}: {e}")

    async def sync_training_info(self, **sync_info) -> Dict[str, Any]:
        """
        同步训练信息（从 Trainer 接收，RPC 方法）

        Args:
            **sync_info: 从 Trainer 收集的同步信息（作为关键字参数）

        Returns:
            确认字典
        """
        self.logger.info(f"Received training sync info: {list(sync_info.keys())}")
        await self._apply_sync_info(sync_info)
        return {"status": "ok"}

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """记录指标"""
        if self._tracker:
            self._tracker.log_metrics(metrics, step=step)

    # ==================== 属性暴露机制（新增）====================

    # 暴露声明（子类可覆盖）
    _exposed_properties: set = {
        "state",           # 节点状态
        "num_samples",     # 训练样本数
        "device",          # 计算设备
        "node_id",         # 节点 ID
        "metadata",        # 完整元数据
    }

    _exposed_methods: set = {
        "fit",
        "evaluate",
        "set_weights",
        "get_weights",
        "sync_training_info",  # 添加：支持 Trainer 同步训练信息
    }

    async def _fed_get_property(self, params: Dict[str, Any]) -> Any:
        """
        系统方法：获取远程属性

        Args:
            params: {"property_name": "num_samples"}

        Returns:
            属性值（必须可序列化）

        此方法由 Proxy 自动调用，用户无感
        """
        property_name = params.get("property_name")

        if property_name not in self._exposed_properties:
            raise AttributeError(
                f"Property '{property_name}' is not exposed. "
                f"Available: {self._exposed_properties}"
            )

        try:
            value = await self._get_property_value(property_name)
            return value
        except Exception as e:
            raise RuntimeError(
                f"Failed to get property '{property_name}': {e}"
            )

    async def _get_property_value(self, property_name: str) -> Any:
        """
        内部方法：获取属性值

        子类可以覆盖此方法来自定义属性获取逻辑
        """
        from .types import NodeState

        if property_name == "state":
            # 返回节点状态（目前总是 IDLE，后续可扩展）
            return getattr(self, "_state", NodeState.IDLE).value if hasattr(NodeState.IDLE, "value") else "idle"

        elif property_name == "num_samples":
            return self.get_num_samples()

        elif property_name == "device":
            return getattr(self, "_device", "cpu")

        elif property_name == "node_id":
            return self._node_id

        elif property_name == "metadata":
            return await self._get_metadata()

        else:
            # 尝试直接访问属性
            if hasattr(self, property_name):
                attr = getattr(self, property_name)
                if callable(attr):
                    if asyncio.iscoroutinefunction(attr):
                        return await attr()
                    else:
                        return attr()
                else:
                    return attr
            else:
                raise AttributeError(f"Property '{property_name}' not found")

    async def _get_metadata(self) -> Dict[str, Any]:
        """获取完整元数据（子类可覆盖）"""
        from .types import NodeState

        return {
            "node_id": self._node_id,
            "state": getattr(self, "_state", NodeState.IDLE).value if hasattr(NodeState.IDLE, "value") else "idle",
            "num_samples": self.get_num_samples(),
            "device": getattr(self, "_device", "cpu"),
            "model_type": type(self._model).__name__,
        }