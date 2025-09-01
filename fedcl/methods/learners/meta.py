"""
元学习器

实现联邦元学习，通过元学习算法提高模型在新任务上的适应能力。
适用于多任务联邦学习和快速适应场景。
"""

import torch
from typing import Dict, Any, Optional
from loguru import logger

from ...api.decorators import learner


@learner("meta", description="联邦元学习器")
class MetaLearner:
    """元学习器实现"""
    
    def __init__(self, config: Dict[str, Any] = None, context: Optional[Any] = None):
        self.config = config or {}
        self.context = context
        
        # 元学习参数
        self.inner_lr = self.config.get("inner_lr", 0.01)
        self.meta_lr = self.config.get("meta_lr", 0.001)
        self.inner_steps = self.config.get("inner_steps", 5)
        self.meta_batch_size = self.config.get("meta_batch_size", 4)
        
        # 元学习状态
        self.meta_weights = {}
        self.task_history = []
        
        logger.info(f"✅ 元学习器初始化完成 - 内部步数: {self.inner_steps}")
    
    def train_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行元学习训练"""
        # 获取支持集和查询集
        support_data = task_data.get("support_data", self._generate_support_data())
        query_data = task_data.get("query_data", self._generate_query_data())
        
        # 初始化元权重
        if not self.meta_weights:
            self._initialize_meta_weights(task_data)
        
        # MAML风格的元学习
        meta_loss = 0.0
        meta_accuracy = 0.0
        
        for task_idx in range(self.meta_batch_size):
            # 内循环：在支持集上快速适应
            adapted_weights = self._inner_loop_adaptation(support_data)
            
            # 外循环：在查询集上计算元梯度
            task_loss, task_acc = self._compute_query_loss(adapted_weights, query_data)
            meta_loss += task_loss
            meta_accuracy += task_acc
        
        # 平均元损失和准确率
        meta_loss /= self.meta_batch_size
        meta_accuracy /= self.meta_batch_size
        
        # 更新元权重
        self._update_meta_weights(meta_loss)
        
        # 记录任务历史
        self.task_history.append({
            "meta_loss": meta_loss,
            "meta_accuracy": meta_accuracy,
            "inner_steps": self.inner_steps
        })
        
        return {
            "loss": meta_loss,
            "accuracy": meta_accuracy,
            "num_samples": len(support_data) + len(query_data),
            "meta_learning_rate": self.meta_lr,
            "adaptation_steps": self.inner_steps
        }
    
    def evaluate_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """评估元学习效果"""
        # 快速适应到新任务
        support_data = task_data.get("support_data", self._generate_support_data())
        test_data = task_data.get("test_data", self._generate_query_data())
        
        # 在少量支持样本上适应
        adapted_weights = self._inner_loop_adaptation(support_data)
        
        # 在测试集上评估
        test_loss, test_accuracy = self._compute_query_loss(adapted_weights, test_data)
        
        # 计算适应性指标
        adaptation_speed = self._compute_adaptation_speed()
        
        return {
            "test_accuracy": test_accuracy,
            "test_loss": test_loss,
            "adaptation_speed": adaptation_speed,
            "few_shot_performance": test_accuracy
        }
    
    def _initialize_meta_weights(self, task_data: Dict[str, Any]):
        """初始化元权重（必须使用真实模型）"""
        # 检查是否提供了真实的模型权重
        if "model_weights" in task_data:
            self.meta_weights = task_data["model_weights"].copy()
            logger.info("使用提供的真实模型权重初始化元学习器")
        elif "model" in task_data:
            # 从真实模型提取权重
            model = task_data["model"]
            if hasattr(model, 'state_dict'):
                self.meta_weights = {k: v.clone().detach() for k, v in model.state_dict().items()}
                logger.info("从真实模型提取权重初始化元学习器")
            else:
                raise NotImplementedError(
                    "元学习器必须提供真实的模型权重。"
                    "请在task_data中提供'model_weights'或'model'。"
                    "不允许使用模拟权重，这是生产环境。"
                )
        else:
            raise NotImplementedError(
                "元学习器必须提供真实的模型权重。"
                "请在task_data中提供'model_weights'或'model'。"
                "不允许使用torch.randn()等模拟权重，这是生产环境。"
            )
        
        logger.debug(f"元学习权重初始化完成 - {len(self.meta_weights)} 个参数")
    
    def _inner_loop_adaptation(self, support_data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """内循环适应（必须使用真实数据和梯度）"""
        if "data" not in support_data or "labels" not in support_data:
            raise ValueError(
                "支持集必须包含真实的'data'和'labels'。"
                "元学习器不允许使用模拟数据进行适应。"
            )
        
        # 检查是否有损失函数和优化器
        if not hasattr(self, '_loss_function') or self._loss_function is None:
            raise NotImplementedError(
                "元学习器必须提供真实的损失函数。"
                "请设置self._loss_function或重写此方法。"
            )
        
        adapted_weights = {k: v.clone().detach().requires_grad_(True) 
                          for k, v in self.meta_weights.items()}
        
        # 使用真实数据进行内循环适应
        data = support_data["data"]
        labels = support_data["labels"]
        
        for step in range(self.inner_steps):
            # 计算真实的损失和梯度
            loss = self._compute_adaptation_loss(data, labels, adapted_weights)
            
            # 计算真实梯度
            grads = torch.autograd.grad(loss, adapted_weights.values(), 
                                      create_graph=True, retain_graph=True)
            
            # 更新适应权重
            for (param_name, param), grad in zip(adapted_weights.items(), grads):
                adapted_weights[param_name] = param - self.inner_lr * grad
        
        return adapted_weights
    

    
    def _compute_query_loss(self, weights: Dict[str, torch.Tensor], query_data: Dict[str, Any]) -> tuple:
        """计算查询集损失和准确率（必须使用真实数据）"""
        if "data" not in query_data or "labels" not in query_data:
            raise ValueError(
                "查询集必须包含真实的'data'和'labels'。"
                "元学习器不允许使用模拟损失和准确率。"
            )
        
        data = query_data["data"]
        labels = query_data["labels"]
        
        # 使用适应后的权重计算真实损失
        loss = self._compute_adaptation_loss(data, labels, weights)
        
        # 计算真实准确率
        with torch.no_grad():
            predictions = self._forward_with_weights(data, weights)
            if predictions.dim() > 1 and predictions.shape[1] > 1:
                predicted_labels = torch.argmax(predictions, dim=1)
            else:
                predicted_labels = (predictions > 0.5).long().squeeze()
            
            accuracy = (predicted_labels == labels).float().mean().item()
        
        return loss.item(), accuracy
    
    def _update_meta_weights(self, meta_loss: torch.Tensor):
        """更新元权重（必须使用真实梯度）"""
        if not isinstance(meta_loss, torch.Tensor):
            raise ValueError("meta_loss必须是torch.Tensor类型以计算真实梯度")
        
        # 计算真实的元梯度
        meta_grads = torch.autograd.grad(meta_loss, self.meta_weights.values(), 
                                       retain_graph=False, allow_unused=True)
        
        # 使用真实梯度更新元权重
        for (param_name, param), grad in zip(self.meta_weights.items(), meta_grads):
            if grad is not None:
                self.meta_weights[param_name] = param - self.meta_lr * grad
            else:
                logger.warning(f"参数 {param_name} 没有梯度，跳过更新")
    
    def _compute_adaptation_speed(self) -> float:
        """计算适应速度指标"""
        if len(self.task_history) < 2:
            return 1.0
        
        # 基于历史任务的适应速度
        recent_accuracy = [task["meta_accuracy"] for task in self.task_history[-5:]]
        if len(recent_accuracy) > 1:
            improvement = recent_accuracy[-1] - recent_accuracy[0]
            return max(0.0, improvement * 10)  # 归一化适应速度
        
        return 1.0
    
    def _compute_adaptation_loss(self, data: torch.Tensor, labels: torch.Tensor, 
                               weights: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算适应损失（必须使用真实损失函数）"""
        predictions = self._forward_with_weights(data, weights)
        return self._loss_function(predictions, labels)
    
    def _forward_with_weights(self, data: torch.Tensor, 
                            weights: Dict[str, torch.Tensor]) -> torch.Tensor:
        """使用指定权重进行前向传播（必须实现真实的前向传播）"""
        raise NotImplementedError(
            "元学习器必须实现真实的前向传播方法。"
            "请重写_forward_with_weights方法，使用真实的模型架构。"
            "不允许使用模拟的前向传播，这是生产环境。"
        )
    
    def get_adaptation_history(self) -> list:
        """获取适应历史"""
        return self.task_history.copy()