"""
FOT: Federated Orthogonal Training for Continual Learning
NeurIPS 2022 Workshop / arXiv 2022

Paper: Federated Continual Learning via Knowledge Fusion: A Survey (Section 3.2)
       https://arxiv.org/abs/2312.16475
       
Related: Orthogonal Gradient Descent for Continual Learning
         https://arxiv.org/abs/1910.07104

核心思想：
1. 正交投影：将新任务的梯度投影到旧任务参数的正交补空间
2. 防止遗忘：确保更新方向不干扰已学习的知识
3. 知识保持：通过正交约束保持旧任务的表征能力

与其他方法的区别：
- EWC: 使用Fisher信息矩阵加权惩罚
- SI: 使用参数重要性在线估计
- FOT: 使用正交投影，更直接地保护旧知识

适用场景：
- 联邦持续学习（FCL）
- 类别增量学习（Class-Incremental Learning）
- 任务增量学习（Task-Incremental Learning）
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from typing import Dict, Any, Optional, List
import copy

from ....core.learner import Learner
from ....core.types import EpochMetrics, EvalResult, StepMetrics
from ....registry import learner


class OrthogonalProjector:
    """
    正交投影器
    
    用于将梯度投影到旧任务参数空间的正交补空间
    """
    
    def __init__(self, device: torch.device):
        self.device = device
        self.projection_matrices: Dict[str, torch.Tensor] = {}
        self.feature_spaces: Dict[str, torch.Tensor] = {}
    
    def update_space(self, name: str, gradient: torch.Tensor, alpha: float = 0.5) -> None:
        """
        更新特征空间
        
        Args:
            name: 参数名
            gradient: 当前梯度
            alpha: 更新比例
        """
        grad_flat = gradient.view(-1).to(self.device)
        
        if name not in self.feature_spaces:
            # 初始化特征空间
            self.feature_spaces[name] = grad_flat.clone().unsqueeze(1)
        else:
            # 正交化新梯度
            space = self.feature_spaces[name]
            # Gram-Schmidt 正交化
            proj = torch.mm(space.t(), grad_flat.unsqueeze(1))
            orthogonal = grad_flat - torch.mm(space, proj).squeeze()
            
            # 归一化
            norm = torch.norm(orthogonal)
            if norm > 1e-8:
                orthogonal = orthogonal / norm
                # 添加到特征空间
                self.feature_spaces[name] = torch.cat([
                    space, orthogonal.unsqueeze(1)
                ], dim=1)
        
        # 更新投影矩阵
        self._update_projection_matrix(name)
    
    def _update_projection_matrix(self, name: str) -> None:
        """计算正交补空间的投影矩阵"""
        if name not in self.feature_spaces:
            return
        
        space = self.feature_spaces[name]
        # P = I - U @ U.T (投影到正交补空间)
        I = torch.eye(space.size(0), device=self.device)
        self.projection_matrices[name] = I - torch.mm(space, space.t())
    
    def project(self, name: str, gradient: torch.Tensor) -> torch.Tensor:
        """
        将梯度投影到正交补空间
        
        Args:
            name: 参数名
            gradient: 原始梯度
            
        Returns:
            投影后的梯度
        """
        if name not in self.projection_matrices:
            return gradient
        
        original_shape = gradient.shape
        grad_flat = gradient.view(-1).to(self.device)
        
        P = self.projection_matrices[name]
        projected = torch.mv(P, grad_flat)
        
        return projected.view(original_shape)
    
    def clear(self) -> None:
        """清空所有空间"""
        self.projection_matrices.clear()
        self.feature_spaces.clear()


@learner('cl.fot', description='FOT: Federated Orthogonal Training for Continual Learning')
class FOTLearner(Learner):
    """
    FOT (Federated Orthogonal Training) 学习器
    
    使用正交投影防止灾难性遗忘的联邦持续学习方法
    
    配置示例:
    {
        "learner": {
            "name": "cl.fot",
            "learning_rate": 0.01,
            "batch_size": 32,
            "local_epochs": 5,
            "num_tasks": 5,
            "classes_per_task": 2,
            "orthogonal_weight": 1.0,
            "projection_threshold": 0.1,
            "memory_strength": 0.5
        }
    }
    """
    
    def __init__(
        self,
        model,
        datasets=None,
        tracker=None,
        callbacks=None,
        config=None,
        node_id=None
    ):
        super().__init__(model, None, tracker, callbacks, config, node_id)
        
        self._datasets = datasets or {}
        
        # 基础训练参数
        self.learning_rate = self._config.get('learning_rate', 0.01)
        self.batch_size = self._config.get('batch_size', 32)
        self.local_epochs = self._config.get('local_epochs', 5)
        self.momentum = self._config.get('momentum', 0.9)
        self.weight_decay = self._config.get('weight_decay', 5e-4)
        
        # 持续学习参数
        self.num_tasks = self._config.get('num_tasks', 5)
        self.classes_per_task = self._config.get('classes_per_task', 2)
        
        # FOT 特定参数
        self.orthogonal_weight = self._config.get('orthogonal_weight', 1.0)
        self.projection_threshold = self._config.get('projection_threshold', 0.1)
        self.memory_strength = self._config.get('memory_strength', 0.5)
        self.use_projection = self._config.get('use_projection', True)
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 组件
        self.torch_model: Optional[nn.Module] = None
        self._optimizer: Optional[optim.Optimizer] = None
        self._criterion: Optional[nn.Module] = None
        self._train_loader: Optional[DataLoader] = None
        self._test_loader: Optional[DataLoader] = None
        
        # FOT 组件
        self.projector = OrthogonalProjector(self.device)
        self.previous_model: Optional[nn.Module] = None
        self.task_gradients: Dict[int, Dict[str, torch.Tensor]] = {}
        
        # 任务状态
        self.current_task_id = 0
        self.seen_classes: List[int] = []
        
        self.logger.info(
            f"FOTLearner {node_id} 初始化: "
            f"num_tasks={self.num_tasks}, classes_per_task={self.classes_per_task}, "
            f"orthogonal_weight={self.orthogonal_weight}"
        )
    
    async def setup(self, config: Dict) -> None:
        """初始化训练环境"""
        # 获取模型
        if hasattr(self._model, 'get_model'):
            self.torch_model = self._model.get_model()
        else:
            self.torch_model = self._model
        
        self.torch_model.to(self.device)
        
        # 创建优化器
        self._optimizer = optim.SGD(
            self.torch_model.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )
        
        # 创建损失函数
        self._criterion = nn.CrossEntropyLoss()
        
        # 创建数据加载器
        train_datasets = self._datasets.get("train", [])
        if train_datasets:
            self._train_loader = DataLoader(
                train_datasets[0],
                batch_size=self.batch_size,
                shuffle=True
            )
        
        test_datasets = self._datasets.get("test", [])
        if test_datasets:
            self._test_loader = DataLoader(
                test_datasets[0],
                batch_size=self.batch_size,
                shuffle=False
            )
        
        self.logger.info("FOT setup 完成")
    
    def _get_task_data_loader(self, task_id: int) -> DataLoader:
        """获取特定任务的数据加载器"""
        train_datasets = self._datasets.get("train", [])
        if not train_datasets:
            return self._train_loader
        
        dataset = train_datasets[0]
        
        # 计算当前任务的类别
        start_class = task_id * self.classes_per_task
        end_class = start_class + self.classes_per_task
        task_classes = list(range(start_class, end_class))
        
        # 筛选样本
        indices = []
        for idx in range(len(dataset)):
            _, label = dataset[idx]
            if isinstance(label, torch.Tensor):
                label = label.item()
            if label in task_classes:
                indices.append(idx)
        
        if not indices:
            return self._train_loader
        
        task_dataset = Subset(dataset, indices)
        return DataLoader(
            task_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
    
    def _compute_gradient_projection(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        计算正交投影后的梯度
        
        Args:
            gradients: 原始梯度字典
            
        Returns:
            投影后的梯度字典
        """
        if not self.use_projection or self.current_task_id == 0:
            return gradients
        
        projected_gradients = {}
        for name, grad in gradients.items():
            projected_gradients[name] = self.projector.project(name, grad)
        
        return projected_gradients
    
    def _update_projection_space(self) -> None:
        """使用当前任务的梯度更新投影空间"""
        for name, param in self.torch_model.named_parameters():
            if param.grad is not None:
                self.projector.update_space(name, param.grad.data)
    
    def _compute_orthogonal_loss(self) -> torch.Tensor:
        """
        计算正交约束损失
        
        确保当前更新与旧任务方向正交
        """
        if self.current_task_id == 0 or not self.task_gradients:
            return torch.tensor(0.0, device=self.device)
        
        ortho_loss = torch.tensor(0.0, device=self.device)
        count = 0
        
        for name, param in self.torch_model.named_parameters():
            if param.grad is None:
                continue
            
            current_grad = param.grad.view(-1)
            
            # 与所有旧任务梯度计算内积
            for task_id, task_grads in self.task_gradients.items():
                if task_id >= self.current_task_id:
                    continue
                
                if name in task_grads:
                    old_grad = task_grads[name].view(-1).to(self.device)
                    # 内积越小越正交
                    inner_product = torch.abs(torch.dot(current_grad, old_grad))
                    ortho_loss += inner_product
                    count += 1
        
        if count > 0:
            ortho_loss /= count
        
        return ortho_loss
    
    async def train_epoch(self, epoch: int) -> EpochMetrics:
        """
        单轮训练 - FOT 训练循环
        """
        # 获取任务ID
        task_id = self._config.get('task_id', self.current_task_id)
        
        # 检查任务切换
        if task_id != self.current_task_id:
            self.logger.info(f"切换任务: {self.current_task_id} -> {task_id}")
            
            # 保存当前任务的梯度方向
            self._save_task_gradients(self.current_task_id)
            
            # 更新投影空间
            self._update_projection_space()
            
            # 保存旧模型
            self.previous_model = copy.deepcopy(self.torch_model)
            self.previous_model.eval()
            
            # 更新任务状态
            self.current_task_id = task_id
            start_class = task_id * self.classes_per_task
            end_class = start_class + self.classes_per_task
            self.seen_classes.extend(range(start_class, end_class))
        
        # 获取任务数据
        task_loader = self._get_task_data_loader(task_id)
        
        self.torch_model.train()
        
        total_loss = 0.0
        total_ce_loss = 0.0
        total_ortho_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch_idx, (data, target) in enumerate(task_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self._optimizer.zero_grad()
            
            # 前向传播
            output = self.torch_model(data)
            ce_loss = self._criterion(output, target)
            
            # 反向传播
            ce_loss.backward()
            
            # 计算正交约束损失
            ortho_loss = self._compute_orthogonal_loss()
            
            # 如果有正交损失，添加到梯度
            if self.orthogonal_weight > 0 and ortho_loss.item() > 0:
                # 再次反向传播正交损失
                (self.orthogonal_weight * ortho_loss).backward()
            
            # 应用梯度投影
            if self.use_projection and self.current_task_id > 0:
                self._apply_gradient_projection()
            
            # 更新参数
            self._optimizer.step()
            
            # 统计
            total_ce_loss += ce_loss.item() * data.size(0)
            total_ortho_loss += ortho_loss.item() * data.size(0)
            total_loss += (ce_loss.item() + self.orthogonal_weight * ortho_loss.item()) * data.size(0)
            
            _, predicted = output.max(1)
            total_correct += predicted.eq(target).sum().item()
            total_samples += target.size(0)
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        avg_ce_loss = total_ce_loss / total_samples if total_samples > 0 else 0
        avg_ortho_loss = total_ortho_loss / total_samples if total_samples > 0 else 0
        avg_accuracy = total_correct / total_samples if total_samples > 0 else 0
        
        self.logger.info(
            f"[{self._node_id}] Task {task_id} Epoch {epoch}: "
            f"Loss={avg_loss:.4f}, CE={avg_ce_loss:.4f}, Ortho={avg_ortho_loss:.4f}, "
            f"Acc={avg_accuracy:.4f}"
        )
        
        return EpochMetrics(
            epoch=epoch,
            avg_loss=avg_loss,
            total_samples=total_samples,
            metrics={
                'accuracy': avg_accuracy,
                'ce_loss': avg_ce_loss,
                'ortho_loss': avg_ortho_loss,
                'task_id': task_id,
                'seen_classes': len(self.seen_classes)
            }
        )
    
    def _apply_gradient_projection(self) -> None:
        """应用梯度投影"""
        for name, param in self.torch_model.named_parameters():
            if param.grad is not None:
                projected_grad = self.projector.project(name, param.grad.data)
                param.grad.data = projected_grad
    
    def _save_task_gradients(self, task_id: int) -> None:
        """保存任务的平均梯度方向"""
        gradients = {}
        for name, param in self.torch_model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.data.clone()
        
        self.task_gradients[task_id] = gradients
    
    async def evaluate(self, config: Optional[Dict] = None) -> EvalResult:
        """评估模型"""
        test_loader = self._test_loader or self._train_loader
        if test_loader is None:
            return EvalResult(num_samples=0, metrics={})
        
        self.torch_model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        # 按任务统计准确率
        task_correct = {i: 0 for i in range(self.num_tasks)}
        task_total = {i: 0 for i in range(self.num_tasks)}
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.torch_model(data)
                loss = self._criterion(output, target)
                
                total_loss += loss.item() * data.size(0)
                _, predicted = output.max(1)
                total_correct += predicted.eq(target).sum().item()
                total_samples += target.size(0)
                
                # 按任务统计
                for i in range(target.size(0)):
                    label = target[i].item()
                    task_id = label // self.classes_per_task
                    if task_id < self.num_tasks:
                        task_total[task_id] += 1
                        if predicted[i] == target[i]:
                            task_correct[task_id] += 1
        
        # 计算各任务准确率
        task_accuracies = {}
        for task_id in range(self.num_tasks):
            if task_total[task_id] > 0:
                task_accuracies[f'task_{task_id}_acc'] = task_correct[task_id] / task_total[task_id]
        
        metrics = {
            'accuracy': total_correct / total_samples if total_samples > 0 else 0,
            'loss': total_loss / total_samples if total_samples > 0 else 0,
            **task_accuracies
        }
        
        return EvalResult(num_samples=total_samples, metrics=metrics)
    
    async def train_step(self, batch: Any, batch_idx: int) -> StepMetrics:
        """单步训练"""
        data, target = batch
        data, target = data.to(self.device), target.to(self.device)
        
        self._optimizer.zero_grad()
        
        output = self.torch_model(data)
        loss = self._criterion(output, target)
        loss.backward()
        
        # 应用梯度投影
        if self.use_projection and self.current_task_id > 0:
            self._apply_gradient_projection()
        
        self._optimizer.step()
        
        _, predicted = output.max(1)
        accuracy = predicted.eq(target).sum().item() / target.size(0)
        
        return StepMetrics(
            loss=loss.item(),
            batch_size=data.size(0),
            metrics={'accuracy': accuracy}
        )
    
    async def get_weights(self) -> Dict[str, Any]:
        """获取模型权重"""
        return {
            name: param.data.clone()
            for name, param in self.torch_model.state_dict().items()
        }
    
    async def set_weights(self, weights: Dict[str, Any]) -> None:
        """设置模型权重"""
        torch_weights = {}
        for k, v in weights.items():
            if torch.is_tensor(v):
                torch_weights[k] = v
            else:
                torch_weights[k] = torch.from_numpy(v)
        
        self.torch_model.load_state_dict(torch_weights)
    
    def get_dataloader(self) -> DataLoader:
        return self._train_loader
    
    def get_num_samples(self) -> int:
        train_datasets = self._datasets.get("train", [])
        return len(train_datasets[0]) if train_datasets else 0
    
    async def teardown(self) -> None:
        """清理资源"""
        self.projector.clear()
        self.task_gradients.clear()
