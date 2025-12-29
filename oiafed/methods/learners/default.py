"""
内置学习器实现
"""

from typing import Any, Dict, Optional

from ...core.learner import Learner
from ...core.types import TrainResult, EvalResult, FitStatus, StepMetrics
from ...registry import learner


@learner(
    name='default',
    description='默认学习器 - 使用PyTorch风格的训练循环',
    version='1.0',
    author='Federation Framework'
)
class DefaultLearner(Learner):
    """
    默认学习器
    
    使用 PyTorch 风格的训练循环
    """
    
    def __init__(
        self,
        model: Any,  # PyTorch nn.Module
        datasets: Optional[Dict[str, Any]] = None,  # 数据集字典（按 split 分组）
        tracker: Optional[Any] = None,
        callbacks: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        optimizer: Optional[Any] = None,
        criterion: Optional[Any] = None,
        node_id: Optional[str] = None,
        **kwargs  # 忽略其他参数（如 data，向后兼容）
    ):
        """
        初始化

        Args:
            model: PyTorch 模型 (nn.Module)
            datasets: 数据集字典，按 split 分组，如 {"train": [...], "test": [...], "valid": [...]}
            tracker: 追踪器
            callbacks: 回调管理器
            config: 配置（可包含 'device': 'cpu' 或 'cuda' 或 'cuda:0'）
            optimizer: 优化器（可选，默认 SGD）
            criterion: 损失函数（可选，默认 CrossEntropy）
            node_id: 节点ID（用于日志）
        """
        # 注意：基类 Learner 仍然接受 data 参数，这里传 None
        super().__init__(model, None, tracker, callbacks, config, node_id)
        self._optimizer = optimizer
        self._criterion = criterion
        self._datasets = datasets or {}

        # 设备配置（CPU/GPU）
        self._device = self._setup_device()

        self.logger.info(f"初始化完毕")

    def _setup_device(self) -> str:
        """
        设置计算设备（CPU/GPU）

        优先级：
        1. 配置中指定的 device
        2. 自动检测 CUDA 是否可用

        Returns:
            device: 'cpu' 或 'cuda' 或 'cuda:0' 等
        """
        try:
            import torch

            # 从配置中获取设备
            device_config = self._config.get("device", None) if self._config else None

            if device_config:
                # 配置中指定了设备
                if device_config.startswith("cuda"):
                    # 检查 CUDA 是否可用
                    if torch.cuda.is_available():
                        self.logger.info(f"[{self._node_id}] 使用配置的设备: {device_config}")
                        return device_config
                    else:
                        self.logger.warning(f"[{self._node_id}] CUDA 不可用，回退到 CPU")
                        return "cpu"
                else:
                    self.logger.info(f"[{self._node_id}] 使用配置的设备: {device_config}")
                    return device_config
            else:
                # 未指定设备，自动检测
                if torch.cuda.is_available():
                    device = "cuda"
                    self.logger.info(f"[{self._node_id}] 自动检测到 CUDA，使用设备: {device}")
                else:
                    device = "cpu"
                    self.logger.info(f"[{self._node_id}] 使用设备: {device}")
                return device

        except ImportError as e:
            self.logger.error(f"{e},{self._node_id}] PyTorch 未安装，使用 CPU")
            return "cpu"

    # ==================== 实现 Learner 抽象方法 ====================

    async def setup(self, config: Dict) -> None:
        """
        初始化训练环境

        创建 DataLoader、Optimizer、Criterion、设置 Device
        """
        self.logger.debug(f"初始化训练环境...")
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader

            # 获取配置
            batch_size = config.get("batch_size", 32)
            learning_rate = config.get("learning_rate", 0.01)

            # 获取 PyTorch 模型
            torch_model = self._model

            # 将模型移动到设备（在线程池中执行，避免阻塞事件循环）
            self._device_obj = torch.device(self._device)
            torch_model = torch_model.to(self._device_obj)
            # self.logger.info(f"模型已成功移动到设备: {self._device}")

            # 设置训练模式
            torch_model.train()

            # 创建 DataLoader - 从 datasets 字典获取训练数据集
            train_datasets = self._datasets.get("train", [])
            if train_datasets:
                # 使用第一个训练数据集
                train_dataset = train_datasets[0]
                self._train_dataloader = DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True
                )
                self.logger.info(f"[{self._node_id}] 训练数据加载器已创建，样本数: {len(train_dataset)}, batch_size: {batch_size}")
            else:
                self._train_dataloader = []
                self.logger.warning(f"[{self._node_id}] 未提供训练数据，训练数据加载器为空")

            # 创建优化器
            if self._optimizer is None:
                self._optimizer = optim.SGD(
                    torch_model.parameters(),
                    lr=learning_rate,
                )
                self.logger.info(f"[{self._node_id}] 优化器已创建: SGD, lr={learning_rate}")

            # 创建损失函数
            if self._criterion is None:
                self._criterion = nn.CrossEntropyLoss()
                self.logger.info(f"[{self._node_id}] 损失函数已创建: CrossEntropyLoss")
            else:
                self.logger.info(f"[{self._node_id}] 使用提供的损失函数: {type(self._criterion)}")

        except ImportError:
            raise NotImplementedError("PyTorch is not installed")

    async def train_step(self, batch: Any, batch_idx: int) -> StepMetrics:
        """
        单批次训练

        Args:
            batch: (batch_x, batch_y) 元组
            batch_idx: 批次索引

        Returns:
            StepMetrics: 包含 loss、batch_size、metrics
        """
        import torch

        batch_x, batch_y = batch

        # 将数据移动到设备
        batch_x = batch_x.to(self._device_obj)
        batch_y = batch_y.to(self._device_obj)

        # 前向传播
        self._optimizer.zero_grad()
        outputs = self._model(batch_x)
        loss = self._criterion(outputs, batch_y)

        # 反向传播
        loss.backward()
        self._optimizer.step()

        # 计算准确率
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == batch_y).sum().item()
        accuracy = correct / batch_y.size(0)

        return StepMetrics(
            loss=loss.item(),
            batch_size=batch_x.size(0),
            metrics={"accuracy": accuracy}
        )

    def get_dataloader(self) -> Any:
        """获取数据加载器"""
        return self._train_dataloader

    def get_num_samples(self) -> int:
        """获取训练样本数"""
        train_datasets = self._datasets.get("train", [])
        if not train_datasets:
            return 0
        try:
            # 使用第一个训练数据集
            return len(train_datasets[0])
        except Exception:
            return 0

    async def teardown(self) -> None:
        """清理资源"""
        # PyTorch 不需要特殊清理
        pass

    async def evaluate(self, config: Optional[Dict[str, Any]] = None) -> EvalResult:
        """执行本地评估"""
        test_datasets = self._datasets.get("test", [])
        if not test_datasets:
            return EvalResult(num_samples=0, metrics={})

        try:
            import torch

            # 获取 PyTorch 模型（直接就是 nn.Module）
            torch_model = self._model

            # 将模型移动到指定设备
            device = torch.device(self._device)
            torch_model = torch_model.to(device)

            # 设置评估模式（PyTorch 标准方式）
            torch_model.eval()

            correct = 0
            total = 0
            total_loss = 0.0
            num_batches = 0

            criterion = self._criterion or torch.nn.CrossEntropyLoss()

            # 使用第一个测试数据集
            test_dataset = test_datasets[0]

            # 创建 DataLoader（PyTorch 标准方式）
            from torch.utils.data import DataLoader
            test_loader = DataLoader(
                test_dataset,
                batch_size=64,
                shuffle=False
            )

            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    # 将数据移动到设备
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)

                    outputs = torch_model(batch_x)
                    loss = criterion(outputs, batch_y)

                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
                    total_loss += loss.item()
                    num_batches += 1

            accuracy = correct / total if total > 0 else 0.0
            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

            return EvalResult(
                num_samples=total,
                metrics={
                    "accuracy": accuracy,
                    "loss": avg_loss,
                },
            )

        except ImportError:
            # 没有 PyTorch，使用简化评估
            raise NotImplementedError("PyTorch is not installed, simplified evaluation not implemented.")
        except Exception as e:
            self.logger.exception(f"Evaluation failed: {e}")
            return EvalResult(num_samples=0, metrics={"error": str(e)})