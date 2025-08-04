# fedcl/implementations/learners/l2p_learner.py
"""
Learning to Prompt for Continual Learning (L2P) 实现

基于论文: https://arxiv.org/abs/2112.08654
L2P通过学习任务特定的提示词来实现持续学习，避免灾难性遗忘。

主要特性:
- 可学习的提示词池
- 任务特定的提示词选择机制
- 冻结预训练模型参数，仅训练提示词
- 支持类增量和域增量学习场景
"""

from typing import Dict, Any, Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from loguru import logger
import math

from ...core.base_learner import BaseLearner
from ...core.execution_context import ExecutionContext
from ...data.results import TaskResults
from ...registry import registry


class PromptPool(nn.Module):
    """
    可学习的提示词池
    
    包含多个可学习的提示词向量，通过注意力机制选择任务相关的提示词。
    """
    
    def __init__(
        self,
        pool_size: int,
        prompt_length: int,
        prompt_dim: int,
        top_k: int = 5,
        temperature: float = 1.0
    ):
        """
        初始化提示词池
        
        Args:
            pool_size: 提示词池大小
            prompt_length: 每个提示词的长度
            prompt_dim: 提示词维度
            top_k: 选择的top-k提示词数量
            temperature: softmax温度参数
        """
        super().__init__()
        self.pool_size = pool_size
        self.prompt_length = prompt_length
        self.prompt_dim = prompt_dim
        self.top_k = top_k
        self.temperature = temperature
        
        # 可学习的提示词池
        self.prompts = nn.Parameter(
            torch.randn(pool_size, prompt_length, prompt_dim)
        )
        
        # 提示词键，用于相似度计算
        self.keys = nn.Parameter(torch.randn(pool_size, prompt_dim))
        
        # 初始化
        self._init_parameters()
        
    def _init_parameters(self):
        """初始化参数"""
        nn.init.xavier_uniform_(self.prompts)
        nn.init.xavier_uniform_(self.keys)
        
    def forward(self, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        根据查询向量选择提示词
        
        Args:
            query: 查询向量 [batch_size, dim]
            
        Returns:
            selected_prompts: 选中的提示词 [batch_size, top_k * prompt_length, prompt_dim]
            attention_weights: 注意力权重 [batch_size, pool_size]
        """
        batch_size = query.shape[0]
        
        # 计算查询向量与提示词键的相似度
        similarities = torch.matmul(query, self.keys.t())  # [batch_size, pool_size]
        similarities = similarities / self.temperature
        
        # 计算注意力权重
        attention_weights = F.softmax(similarities, dim=-1)
        
        # 选择top-k提示词
        _, top_indices = torch.topk(similarities, self.top_k, dim=-1)
        
        # 获取选中的提示词
        selected_prompts = []
        for i in range(batch_size):
            batch_prompts = self.prompts[top_indices[i]]  # [top_k, prompt_length, prompt_dim]
            batch_prompts = batch_prompts.view(-1, self.prompt_dim)  # [top_k * prompt_length, prompt_dim]
            selected_prompts.append(batch_prompts)
            
        selected_prompts = torch.stack(selected_prompts, dim=0)  # [batch_size, top_k * prompt_length, prompt_dim]
        
        return selected_prompts, attention_weights


@registry.learner("l2p", metadata={
    "description": "Learning to Prompt for Continual Learning",
    "paper": "https://arxiv.org/abs/2112.08654",
    "supported_tasks": ["class_incremental", "domain_incremental"],
    "requires_pretrained": True
})
class L2PLearner(BaseLearner):
    """
    L2P持续学习算法实现
    
    通过学习任务特定的提示词来实现持续学习，主要特点：
    1. 冻结预训练模型参数
    2. 学习可复用的提示词池
    3. 基于查询的提示词选择机制
    4. 防止灾难性遗忘
    """
    
    def __init__(self, context: ExecutionContext, config: DictConfig):
        """
        初始化L2P学习器
        
        Args:
            context: 执行上下文
            config: 配置参数，应包含：
                - backbone_model: 预训练模型名称
                - pool_size: 提示词池大小
                - prompt_length: 提示词长度
                - top_k: 选择的提示词数量
                - learning_rate: 学习率
                - temperature: softmax温度
        """
        super().__init__(context, config)
        
        # L2P特定配置
        self.pool_size = config.get("pool_size", 100)
        self.prompt_length = config.get("prompt_length", 5)
        self.top_k = config.get("top_k", 5)
        self.temperature = config.get("temperature", 1.0)
        self.freeze_backbone = config.get("freeze_backbone", True)
        
        # 构建模型
        self._build_model()
        self._build_optimizer()
        
        logger.info(f"L2PLearner initialized with pool_size={self.pool_size}, "
                   f"prompt_length={self.prompt_length}, top_k={self.top_k}")
        
    def _create_default_model(self) -> nn.Module:
        """
        创建默认模型（实现抽象方法）
        
        由于L2P有特殊的模型构建逻辑，这里返回一个简单的模型作为占位符
        实际的模型构建在_build_model中完成
        """
        # 返回一个简单的模型作为占位符
        return nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 10)
        )
        
    def _build_model(self) -> None:
        """构建L2P模型"""
        # 获取预训练backbone
        backbone_name = self.config.get("backbone_model", "vit_base_patch16_224")
        self.backbone = self._load_pretrained_model(backbone_name)
        
        # 冻结backbone参数
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        # 获取模型维度
        self.hidden_dim = self._get_model_hidden_dim()
        
        # 创建提示词池
        self.prompt_pool = PromptPool(
            pool_size=self.pool_size,
            prompt_length=self.prompt_length,
            prompt_dim=self.hidden_dim,
            top_k=self.top_k,
            temperature=self.temperature
        )
        
        # 分类头 (可选)
        num_classes = self.config.get("num_classes", 1000)
        self.classifier = nn.Linear(self.hidden_dim, num_classes)
        
        self.model = nn.ModuleDict({
            "backbone": self.backbone,
            "prompt_pool": self.prompt_pool,
            "classifier": self.classifier
        })
        
    def _load_pretrained_model(self, model_name: str) -> nn.Module:
        """加载预训练模型"""
        # 这里应该根据model_name加载相应的预训练模型
        # 简化示例，使用一个简单的CNN
        return nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
    def _get_model_hidden_dim(self) -> int:
        """获取模型隐藏层维度"""
        # 简化示例
        return 64
        
    def _build_optimizer(self) -> None:
        """构建优化器，只优化提示词池和分类头"""
        trainable_params = []
        
        # 添加提示词池参数
        trainable_params.extend(self.prompt_pool.parameters())
        
        # 添加分类头参数
        trainable_params.extend(self.classifier.parameters())
        
        lr = self.config.get("learning_rate", 0.001)
        self.optimizer = torch.optim.Adam(trainable_params, lr=lr)
        
    def train_task(self, task_data: DataLoader) -> TaskResults:
        """
        训练单个任务
        
        Args:
            task_data: 任务数据加载器
            
        Returns:
            TaskResults: 训练结果
        """
        self.model.train()
        
        total_loss = 0.0
        total_samples = 0
        correct_predictions = 0
        
        for batch_idx, (inputs, targets) in enumerate(task_data):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            # 前向传播
            outputs = self._forward_with_prompts(inputs)
            
            # 计算损失
            loss = F.cross_entropy(outputs, targets)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            total_samples += inputs.size(0)
            _, predicted = outputs.max(1)
            correct_predictions += predicted.eq(targets).sum().item()
            
            if batch_idx % 50 == 0:
                logger.debug(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
                
        # 计算平均指标
        avg_loss = total_loss / len(task_data)
        accuracy = correct_predictions / total_samples
        
        # 返回结果
        return TaskResults(
            task_id=self.current_task_id,
            metrics={
                "loss": avg_loss,
                "accuracy": accuracy,
                "total_samples": total_samples
            },
            metadata={
                "algorithm": "L2P",
                "prompt_pool_size": self.pool_size,
                "top_k": self.top_k,
                "model_state": self.get_model_state()
            }
        )
        
    def _forward_with_prompts(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        使用提示词的前向传播
        
        Args:
            inputs: 输入数据
            
        Returns:
            模型输出
        """
        batch_size = inputs.size(0)
        
        # 提取特征作为查询（不使用no_grad，以便梯度传播）
        features = self.backbone(inputs)  # [batch_size, hidden_dim]
            
        # 选择提示词
        selected_prompts, attention_weights = self.prompt_pool(features)
        
        # 将提示词信息融入特征
        # 简化实现：将selected_prompts的信息加到features上
        # selected_prompts: [batch_size, top_k * prompt_length, prompt_dim]
        # 我们对第二个维度（top_k * prompt_length）取平均
        prompt_features = selected_prompts.mean(dim=1)  # [batch_size, prompt_dim] = [batch_size, hidden_dim]
        enhanced_features = features + 0.1 * prompt_features  # 使用较小的权重
        
        # 分类
        outputs = self.classifier(enhanced_features)
        
        return outputs
        
    def evaluate_task(self, task_data: DataLoader) -> Dict[str, float]:
        """
        评估任务性能
        
        Args:
            task_data: 任务数据加载器
            
        Returns:
            评估指标字典
        """
        self.model.eval()
        
        total_samples = 0
        correct_predictions = 0
        total_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in task_data:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self._forward_with_prompts(inputs)
                loss = F.cross_entropy(outputs, targets)
                
                total_loss += loss.item()
                total_samples += inputs.size(0)
                _, predicted = outputs.max(1)
                correct_predictions += predicted.eq(targets).sum().item()
                
        return {
            "accuracy": correct_predictions / total_samples,
            "loss": total_loss / len(task_data),
            "total_samples": total_samples
        }
        
    def get_model_update(self) -> Dict[str, torch.Tensor]:
        """
        获取模型更新（仅包含可训练参数）
        
        Returns:
            模型更新字典
        """
        update = {}
        
        # 只包含提示词池和分类头的参数
        for name, param in self.prompt_pool.named_parameters():
            update[f"prompt_pool.{name}"] = param.data.clone()
            
        for name, param in self.classifier.named_parameters():
            update[f"classifier.{name}"] = param.data.clone()
            
        return update
        
    def apply_model_update(self, update: Dict[str, torch.Tensor]) -> None:
        """
        应用模型更新
        
        Args:
            update: 模型更新字典
        """
        for name, param_data in update.items():
            if name.startswith("prompt_pool."):
                param_name = name[len("prompt_pool."):]
                if hasattr(self.prompt_pool, param_name):
                    getattr(self.prompt_pool, param_name).data.copy_(param_data)
            elif name.startswith("classifier."):
                param_name = name[len("classifier."):]
                if hasattr(self.classifier, param_name):
                    getattr(self.classifier, param_name).data.copy_(param_data)
                    
    def get_prompt_statistics(self) -> Dict[str, Any]:
        """
        获取提示词使用统计
        
        Returns:
            提示词统计信息
        """
        with torch.no_grad():
            # 计算提示词的范数分布
            prompt_norms = torch.norm(self.prompt_pool.prompts, dim=-1).mean(dim=1)
            key_norms = torch.norm(self.prompt_pool.keys, dim=-1)
            
            return {
                "prompt_norms_mean": prompt_norms.mean().item(),
                "prompt_norms_std": prompt_norms.std().item(),
                "key_norms_mean": key_norms.mean().item(),
                "key_norms_std": key_norms.std().item(),
                "pool_size": self.pool_size,
                "top_k": self.top_k
            }
