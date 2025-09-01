"""
联邦学习客户端学习器抽象基类

根据项目规范，learner负责客户端的本地训练逻辑，与执行模式解耦。
不同的执行模式(local/pseudo/distributed)会以不同方式实例化和调用learner。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger


class AbstractLearner(ABC):
    """
    抽象学习器基类 - 定义客户端学习器的标准接口
    
    设计原则：
    1. 强制子类实现核心方法，避免默认实现掩盖问题
    2. 禁止使用任何模拟数据，必须实现真实逻辑  
    3. 支持任意评估指标，解除指标硬绑定
    4. 与通信模式完全解耦，专注本地训练逻辑
    """
    
    def __init__(self, client_id: str, config: Dict[str, Any]):
        """
        初始化学习器
        
        Args:
            client_id: 客户端ID，用于标识不同的客户端
            config: 配置字典，包含训练相关的所有参数
        """
        self.client_id = client_id
        self.config = config
        self.logger = logger.bind(component=f"Learner-{client_id}")
        
        # 核心组件 - 由子类初始化
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.criterion: Optional[nn.Module] = None
        self.device = torch.device(config.get("device", "cpu"))
        
        # 训练状态
        self._current_round = 0
        self._training_history: List[Dict[str, Any]] = []
        
    @abstractmethod
    async def train_epoch(self, **kwargs) -> Dict[str, Any]:
        """
        执行一个epoch的本地训练
        
        这是learner的核心方法，子类必须实现具体的训练逻辑。
        必须返回真实的训练结果，禁止使用模拟数据。
        
        Args:
            **kwargs: 训练参数，可能包括：
                - round_num: 当前轮次
                - global_weights: 全局模型权重
                - local_epochs: 本地训练轮数
                - learning_rate: 学习率
                
        Returns:
            Dict[str, Any]: 训练结果，必须包括：
                - model_weights: 训练后的模型权重
                - num_samples: 训练样本数量
                - loss: 训练损失
                - 其他自定义指标
        
        Raises:
            NotImplementedError: 子类必须实现此方法
        """
        pass
    
    @abstractmethod
    async def evaluate(self, **kwargs) -> Dict[str, Any]:
        """
        执行本地评估
        
        支持任意评估指标，不限制于accuracy和loss。
        
        Args:
            **kwargs: 评估参数，可能包括：
                - test_data: 测试数据
                - metrics: 需要计算的指标列表
                
        Returns:
            Dict[str, Any]: 评估结果，包含用户指定的所有指标
            
        Raises:
            NotImplementedError: 子类必须实现此方法
        """
        pass
    
    @abstractmethod
    def get_model_weights(self) -> Dict[str, Any]:
        """
        获取当前模型权重
        
        Returns:
            Dict[str, Any]: 模型权重字典
        """
        pass
    
    @abstractmethod
    def set_model_weights(self, weights: Dict[str, Any]) -> None:
        """
        设置模型权重
        
        Args:
            weights: 要设置的模型权重
        """
        pass
    
    # 以下是通用方法，可以在基类中提供默认实现
    
    def save_checkpoint(self, path: str) -> None:
        """保存学习器检查点"""
        if self.model is None or self.optimizer is None:
            raise ValueError("Model and optimizer must be initialized before saving")
            
        checkpoint = {
            'client_id': self.client_id,
            'current_round': self._current_round,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self._training_history,
            'config': self.config
        }
        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str) -> None:
        """加载学习器检查点"""
        if self.model is None or self.optimizer is None:
            raise ValueError("Model and optimizer must be initialized before loading")
            
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self._current_round = checkpoint.get('current_round', 0)
        self._training_history = checkpoint.get('training_history', [])
        self.logger.info(f"Checkpoint loaded from {path}")
    
    def get_training_history(self) -> List[Dict[str, Any]]:
        """获取训练历史记录"""
        return self._training_history.copy()
    
    def update_round(self, round_num: int) -> None:
        """更新当前轮次"""
        self._current_round = round_num
        self.logger.debug(f"Updated to round {round_num}")
    
    def log_training_result(self, result: Dict[str, Any]) -> None:
        """记录训练结果到历史"""
        result_with_meta = {
            'round': self._current_round,
            'client_id': self.client_id,
            **result
        }
        self._training_history.append(result_with_meta)
        
    def reset_training_state(self) -> None:
        """重置训练状态"""
        self._current_round = 0
        self._training_history.clear()
        self.logger.info("Training state reset")


class StandardLearner(AbstractLearner):
    """
    标准联邦学习器实现
    
    提供最基本的联邦学习客户端训练逻辑，适用于大部分场景。
    """
    
    def __init__(self, client_id: str, config: Dict[str, Any]):
        super().__init__(client_id, config)
        
        # 初始化模型（这里需要从config或registry获取）
        # 注意：实际实现中应该通过工厂模式或注册表创建模型
        self._initialize_model()
        self._initialize_optimizer()
        self._initialize_criterion()
    
    def _initialize_model(self) -> None:
        """初始化模型 - 实际实现中应该从配置创建"""
        # 这里是示例，实际应该从registry获取
        model_name = self.config.get("model", "simple_mlp")
        if model_name == "simple_mlp":
            from fedcl.methods.models import SimpleMLP  # 示例
            self.model = SimpleMLP(
                input_dim=self.config.get("input_dim", 784),
                hidden_dims=self.config.get("hidden_dims", [128, 64]),
                output_dim=self.config.get("output_dim", 10)
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")
            
        self.model.to(self.device)
        self.logger.info(f"Model initialized: {model_name}")
    
    def _initialize_optimizer(self) -> None:
        """初始化优化器"""
        optimizer_name = self.config.get("optimizer", "sgd")
        lr = self.config.get("learning_rate", 0.01)
        
        if optimizer_name.lower() == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(), 
                lr=lr,
                momentum=self.config.get("momentum", 0.9),
                weight_decay=self.config.get("weight_decay", 1e-4)
            )
        elif optimizer_name.lower() == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=self.config.get("weight_decay", 1e-4)
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
            
        self.logger.info(f"Optimizer initialized: {optimizer_name}")
    
    def _initialize_criterion(self) -> None:
        """初始化损失函数"""
        criterion_name = self.config.get("criterion", "cross_entropy")
        
        if criterion_name == "cross_entropy":
            self.criterion = nn.CrossEntropyLoss()
        elif criterion_name == "mse":
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unknown criterion: {criterion_name}")
            
        self.criterion.to(self.device)
        self.logger.info(f"Criterion initialized: {criterion_name}")
    
    async def train_epoch(self, **kwargs) -> Dict[str, Any]:
        """执行标准的本地训练"""
        self.model.train()
        
        # 获取训练参数
        round_num = kwargs.get("round_num", self._current_round)
        local_epochs = kwargs.get("local_epochs", self.config.get("local_epochs", 5))
        
        # 更新全局权重（如果提供）
        global_weights = kwargs.get("global_weights")
        if global_weights:
            self.set_model_weights(global_weights)
        
        # 获取训练数据（实际实现中应该从data manager获取）
        train_loader = kwargs.get("train_loader")
        if train_loader is None:
            # 这里应该从data manager获取客户端的训练数据
            raise ValueError("Training data loader not provided")
        
        total_loss = 0.0
        total_samples = 0
        
        for epoch in range(local_epochs):
            epoch_loss = 0.0
            epoch_samples = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item() * data.size(0)
                epoch_samples += data.size(0)
            
            total_loss += epoch_loss
            total_samples += epoch_samples
            
            self.logger.debug(
                f"Round {round_num}, Epoch {epoch+1}/{local_epochs}, "
                f"Loss: {epoch_loss/epoch_samples:.4f}"
            )
        
        # 计算平均损失
        avg_loss = total_loss / (total_samples * local_epochs)
        
        # 准备返回结果
        result = {
            'model_weights': self.get_model_weights(),
            'num_samples': total_samples,
            'loss': avg_loss,
            'local_epochs': local_epochs,
            'round_num': round_num
        }
        
        # 记录训练结果
        self.update_round(round_num)
        self.log_training_result(result)
        
        self.logger.info(
            f"Training completed - Round: {round_num}, "
            f"Loss: {avg_loss:.4f}, Samples: {total_samples}"
        )
        
        return result
    
    async def evaluate(self, **kwargs) -> Dict[str, Any]:
        """执行标准评估"""
        self.model.eval()
        
        # 获取测试数据
        test_loader = kwargs.get("test_loader")
        if test_loader is None:
            raise ValueError("Test data loader not provided")
        
        # 获取需要计算的指标
        metrics = kwargs.get("metrics", ["accuracy", "loss"])
        
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                # 计算损失
                if "loss" in metrics:
                    loss = self.criterion(output, target)
                    total_loss += loss.item() * data.size(0)
                
                # 计算准确率
                if "accuracy" in metrics:
                    pred = output.argmax(dim=1, keepdim=True)
                    correct_predictions += pred.eq(target.view_as(pred)).sum().item()
                
                total_samples += data.size(0)
        
        # 构建结果字典
        results = {}
        
        if "loss" in metrics:
            results["loss"] = total_loss / total_samples
        
        if "accuracy" in metrics:
            results["accuracy"] = correct_predictions / total_samples
        
        # 添加样本数量
        results["num_samples"] = total_samples
        
        self.logger.info(f"Evaluation completed - {results}")
        
        return results
    
    def get_model_weights(self) -> Dict[str, Any]:
        """获取模型权重"""
        if self.model is None:
            raise ValueError("Model not initialized")
        return {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
    
    def set_model_weights(self, weights: Dict[str, Any]) -> None:
        """设置模型权重"""
        if self.model is None:
            raise ValueError("Model not initialized")
            
        # 将权重移动到正确的设备
        device_weights = {k: v.to(self.device) for k, v in weights.items()}
        self.model.load_state_dict(device_weights)
        self.logger.debug("Model weights updated")