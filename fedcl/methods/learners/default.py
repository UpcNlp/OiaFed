"""
默认客户端学习器

实现标准的联邦学习客户端训练逻辑，损失计算放在模型中。
模型由用户传入，学习器选择也由用户决定。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Optional
from loguru import logger

from ...execution.base_learner import AbstractLearner
from ...api.decorators import learner


@learner("default", description="默认联邦学习客户端")
class DefaultLearner(AbstractLearner):
    """默认联邦学习客户端学习器"""
    
    def __init__(self, client_id: str, config: Dict[str, Any]):
        super().__init__(client_id, config)
        
        # 初始化模型、优化器和损失函数
        self._initialize_model()
        self._initialize_optimizer()
        self._initialize_criterion()
        
        self.logger.info(f"✅ 默认学习器初始化完成 - 客户端: {client_id}")
    
    def _initialize_model(self) -> None:
        """初始化模型 - 从配置获取用户传入的模型"""
        model_config = self.config.get("model", {})
        
        # 用户应该传入模型实例或模型类
        model = model_config.get("instance")
        if model is None:
            raise ValueError("用户必须提供模型实例，请在配置中设置 'model.instance'")
        
        if not isinstance(model, nn.Module):
            raise ValueError("模型必须是 torch.nn.Module 的实例")
        
        self.model = model
        self.model.to(self.device)
        self.logger.info(f"模型初始化完成 - 类型: {type(model).__name__}")
    
    def _initialize_optimizer(self) -> None:
        """初始化优化器"""
        optimizer_config = self.config.get("optimizer", {})
        optimizer_type = optimizer_config.get("type", "sgd")
        learning_rate = optimizer_config.get("learning_rate", 0.01)
        
        if optimizer_type.lower() == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=optimizer_config.get("momentum", 0.9)
            )
        elif optimizer_type.lower() == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=learning_rate
            )
        else:
            raise ValueError(f"不支持的优化器类型: {optimizer_type}")
        
        self.logger.info(f"优化器初始化完成")
    
    def _initialize_criterion(self) -> None:
        """初始化损失函数 - 从模型获取或用户指定"""
        # 优先从模型获取损失函数
        if hasattr(self.model, 'criterion'):
            self.criterion = self.model.criterion
        else:
            # 用户指定损失函数
            criterion_config = self.config.get("criterion", {})
            criterion_type = criterion_config.get("type", "cross_entropy")
            
            if criterion_type == "cross_entropy":
                self.criterion = nn.CrossEntropyLoss()
            elif criterion_type == "mse":
                self.criterion = nn.MSELoss()
            else:
                raise ValueError(f"不支持的损失函数类型: {criterion_type}")
        
        self.criterion.to(self.device)
        self.logger.info(f"损失函数初始化完成")
    
    async def train_epoch(self, **kwargs) -> Dict[str, Any]:
        """执行一个epoch的本地训练"""
        self.model.train()
        
        round_num = kwargs.get("round_num", self._current_round)
        local_epochs = kwargs.get("local_epochs", self.config.get("local_epochs", 1))
        train_loader = kwargs.get("train_loader")
        
        if train_loader is None:
            raise ValueError("训练数据加载器未提供")
        
        global_weights = kwargs.get("global_weights")
        if global_weights:
            self.set_model_weights(global_weights)
        
        total_loss = 0.0
        total_samples = 0
        correct_predictions = 0
        
        for epoch in range(local_epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                
                # 前向传播 - 检查模型是否支持损失计算
                if hasattr(self.model, 'forward_with_loss'):
                    # 模型支持内置损失计算
                    output, loss = self.model.forward_with_loss(data, target)
                else:
                    # 标准前向传播 + 外部损失计算
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item() * data.size(0)
                total_samples += data.size(0)
                
                pred = output.argmax(dim=1, keepdim=True)
                correct_predictions += pred.eq(target.view_as(pred)).sum().item()
        
        avg_loss = total_loss / total_samples
        avg_accuracy = correct_predictions / total_samples
        
        result = {
            'model_weights': self.get_model_weights(),
            'num_samples': total_samples,
            'loss': avg_loss,
            'accuracy': avg_accuracy,
            'local_epochs': local_epochs,
            'round_num': round_num,
            'client_id': self.client_id
        }
        
        self.update_round(round_num)
        self.log_training_result(result)
        
        return result
    
    async def evaluate(self, **kwargs) -> Dict[str, Any]:
        """执行本地评估"""
        self.model.eval()
        
        test_loader = kwargs.get("test_loader")
        if test_loader is None:
            raise ValueError("测试数据加载器未提供")
        
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # 前向传播 - 检查模型是否支持损失计算
                if hasattr(self.model, 'forward_with_loss'):
                    output, loss = self.model.forward_with_loss(data, target)
                else:
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                total_loss += loss.item() * data.size(0)
                total_samples += data.size(0)
                
                pred = output.argmax(dim=1, keepdim=True)
                correct_predictions += pred.eq(target.view_as(pred)).sum().item()
        
        results = {
            "num_samples": total_samples,
            "client_id": self.client_id,
            "loss": total_loss / total_samples,
            "accuracy": correct_predictions / total_samples
        }
        
        return results
    
    def get_model_weights(self) -> Dict[str, Any]:
        """获取模型权重"""
        if self.model is None:
            raise ValueError("模型未初始化")
        return {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
    
    def set_model_weights(self, weights: Dict[str, Any]) -> None:
        """设置模型权重"""
        if self.model is None:
            raise ValueError("模型未初始化")
        
        device_weights = {k: v.to(self.device) for k, v in weights.items()}
        self.model.load_state_dict(device_weights)
    
    async def train_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """兼容性方法 - 执行训练任务"""
        round_num = task_data.get("round_num", 0)
        local_epochs = task_data.get("local_epochs", self.config.get("local_epochs", 1))
        global_weights = task_data.get("model_weights")
        train_loader = task_data.get("train_loader")
        
        return await self.train_epoch(
            round_num=round_num,
            local_epochs=local_epochs,
            global_weights=global_weights,
            train_loader=train_loader
        )
    
    async def evaluate_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """兼容性方法 - 执行评估任务"""
        test_loader = task_data.get("test_loader")
        
        return await self.evaluate(test_loader=test_loader)
