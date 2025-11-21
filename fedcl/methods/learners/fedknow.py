"""
FedKNOW: Federated Continual Learning with Signature Task Knowledge Integration
INFOCOM 2023

Paper: https://ieeexplore.ieee.org/document/10228927
GitHub: https://github.com/LINC-BIT/FedKNOW

核心思想：
1. 从每个任务提取"签名任务知识"(Signature Task Knowledge)
2. 使用知识集成机制避免灾难性遗忘
3. 支持多种网络架构：LeNet, AlexNet, VGG, ResNet等
4. 在本地维护任务知识库
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from typing import Dict, Any, Optional, List
from datetime import datetime
import copy
import numpy as np

from ...api.decorators import learner
from ...learner.base_learner import BaseLearner
from ...types import TrainingResponse, EvaluationResult, ModelData
from ...methods.models.base import get_weights_as_dict, set_weights_from_dict


@learner('FedKNOW',
         description='FedKNOW: Federated Continual Learning with Signature Task Knowledge Integration (INFOCOM 2023)',
         version='1.0',
         author='MOE-FedCL')
class FedKNOWLearner(BaseLearner):
    """
    FedKNOW 持续学习方法的Learner实现

    特点：
    - 提取和保存签名任务知识(Signature Task Knowledge)
    - 知识集成机制(Knowledge Integration)
    - 支持Task-Incremental和Class-Incremental场景
    """

    def __init__(self, client_id: str, config: Optional[Dict[str, Any]] = None, lazy_init: bool = True):
        # 提取配置
        learner_params = (config or {}).get('learner', {}).get('params', {})

        # 保存FedKNOW特定配置
        self._model_cfg = learner_params.get('model', {})
        self._optimizer_cfg = learner_params.get('optimizer', {
            'type': 'SGD',
            'lr': learner_params.get('learning_rate', 0.01),
            'momentum': 0.9
        })
        self._loss_cfg = learner_params.get('loss', 'CrossEntropyLoss')

        # 训练参数
        self._lr = learner_params.get('learning_rate', 0.01)
        self._bs = learner_params.get('batch_size', 32)
        self._epochs = learner_params.get('local_epochs', 5)

        # 持续学习参数
        self.num_tasks = learner_params.get('num_tasks', 5)
        self.classes_per_task = learner_params.get('classes_per_task', 2)
        self.scenario = learner_params.get('scenario', 'class_incremental')

        # FedKNOW特定参数
        self.signature_ratio = learner_params.get('signature_ratio', 0.1)  # 签名样本比例
        self.knowledge_weight = learner_params.get('knowledge_weight', 0.5)  # 知识蒸馏权重
        self.distill_temperature = learner_params.get('distill_temperature', 2.0)
        self.integration_method = learner_params.get('integration_method', 'weighted_sum')

        # 清理配置
        clean_config = config.copy() if config else {}
        if 'learner' in clean_config and 'params' in clean_config['learner']:
            clean_params = clean_config['learner']['params'].copy()
            clean_params.pop('model', None)
            clean_params.pop('optimizer', None)
            clean_params.pop('loss', None)
            clean_config['learner'] = clean_config['learner'].copy()
            clean_config['learner']['params'] = clean_params

        super().__init__(client_id, clean_config, lazy_init)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 持续学习状态
        self.current_task_id = 0
        self.seen_classes = []

        # 知识库（存储每个任务的签名知识）
        self.task_knowledge = {}  # {task_id: {'model': model_copy, 'data_indices': indices}}

        # 组件占位符
        self._model = None
        self._optimizer = None
        self._criterion = None
        self._train_loader = None

        self.logger.info(
            f"FedKNOWLearner {client_id} initialized: "
            f"num_tasks={self.num_tasks}, classes_per_task={self.classes_per_task}, "
            f"signature_ratio={self.signature_ratio}"
        )

    @property
    def model(self):
        """延迟加载模型"""
        if self._model is None:
            from ...api.registry import registry

            # 优先使用learner.params.model
            if self._model_cfg and self._model_cfg.get('name'):
                model_name = self._model_cfg['name']
                model_params = self._model_cfg.get('params', {})
            # 否则使用顶层model配置
            elif 'model' in self.config:
                model_config = self.config['model']
                model_name = model_config.get('name')
                model_params = model_config.get('params', {})
            else:
                raise ValueError("未找到模型配置")

            model_class = registry.get_model(model_name)
            self._model = model_class(**model_params).to(self.device)
            self.logger.info(f"Client {self.client_id}: 模型 {model_name} 创建完成")
        return self._model

    @property
    def optimizer(self):
        """延迟加载优化器"""
        if self._optimizer is None:
            opt_type = self._optimizer_cfg.get('type', 'SGD').upper()
            lr = self._optimizer_cfg.get('lr', self._lr)

            if opt_type == 'SGD':
                self._optimizer = optim.SGD(
                    self.model.parameters(),
                    lr=lr,
                    momentum=self._optimizer_cfg.get('momentum', 0.9),
                    weight_decay=self._optimizer_cfg.get('weight_decay', 5e-4)
                )
            elif opt_type == 'ADAM':
                self._optimizer = optim.Adam(
                    self.model.parameters(),
                    lr=lr
                )
            else:
                raise ValueError(f"不支持的优化器类型: {opt_type}")

            self.logger.debug(f"Client {self.client_id}: 优化器 {opt_type} 创建完成")
        return self._optimizer

    @property
    def criterion(self):
        """延迟加载损失函数"""
        if self._criterion is None:
            if isinstance(self._loss_cfg, str):
                if self._loss_cfg == 'CrossEntropyLoss':
                    self._criterion = nn.CrossEntropyLoss()
                elif self._loss_cfg == 'MSELoss':
                    self._criterion = nn.MSELoss()
                else:
                    raise ValueError(f"不支持的损失函数: {self._loss_cfg}")
            self.logger.debug(f"Client {self.client_id}: 损失函数创建完成")
        return self._criterion

    @property
    def train_loader(self):
        """延迟加载数据加载器"""
        if self._train_loader is None:
            dataset = self.dataset
            self._train_loader = DataLoader(
                dataset,
                batch_size=self._bs,
                shuffle=True,
                drop_last=False
            )
            self.logger.info(
                f"Client {self.client_id}: 数据加载器创建完成 "
                f"(samples={len(dataset)}, batch_size={self._bs})"
            )
        return self._train_loader

    def _get_task_data_loader(self, task_id: int) -> DataLoader:
        """
        获取特定任务的数据加载器

        根据当前任务ID筛选对应的类别数据
        """
        dataset = self.dataset

        # 计算当前任务的类别范围
        start_class = task_id * self.classes_per_task
        end_class = min(start_class + self.classes_per_task,
                       self.num_tasks * self.classes_per_task)
        task_classes = list(range(start_class, end_class))

        # 筛选当前任务的样本
        indices = []
        for idx in range(len(dataset)):
            _, label = dataset[idx]
            if isinstance(label, torch.Tensor):
                label = label.item()
            if label in task_classes:
                indices.append(idx)

        if not indices:
            self.logger.warning(
                f"Task {task_id} has no samples for classes {task_classes}"
            )
            return self.train_loader

        task_dataset = Subset(dataset, indices)

        self.logger.info(
            f"Task {task_id} data loader: {len(indices)} samples, "
            f"classes {task_classes}"
        )

        return DataLoader(
            task_dataset,
            batch_size=self._bs,
            shuffle=True,
            drop_last=False
        )

    async def train(self, params: Dict[str, Any]) -> TrainingResponse:
        """
        执行FedKNOW方法的本地训练

        FedKNOW的训练流程：
        1. 使用新任务数据训练
        2. 结合历史任务知识进行蒸馏
        3. 提取当前任务的签名知识
        """
        start_time = datetime.now()

        # 提取参数
        num_epochs = params.get("num_epochs", self._epochs)
        round_number = params.get("round_number", 1)
        task_id = params.get("task_id", self.current_task_id)

        # 更新任务状态
        if task_id != self.current_task_id:
            self.logger.info(
                f"[{self.client_id}] Switching task: {self.current_task_id} -> {task_id}"
            )
            self.current_task_id = task_id

            # 更新已见类别
            start_class = task_id * self.classes_per_task
            end_class = min(start_class + self.classes_per_task,
                           self.num_tasks * self.classes_per_task)
            new_classes = list(range(start_class, end_class))
            self.seen_classes.extend(new_classes)

        # 获取当前任务的数据
        task_loader = self._get_task_data_loader(task_id)

        self.logger.info(
            f"[{self.client_id}] Task {task_id} Round {round_number}, "
            f"Training {num_epochs} epochs..."
        )

        # 训练
        self.model.train()

        total_loss = 0.0
        total_accuracy = 0.0
        total_samples = 0

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0

            for batch_idx, (data, target) in enumerate(task_loader):
                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()

                # 前向传播
                output = self.model(data)

                # 1. 当前任务的分类损失
                ce_loss = self.criterion(output, target)

                # 2. 知识集成损失（如果有历史任务知识）
                integration_loss = 0.0
                if self.task_knowledge and task_id > 0:
                    integration_loss = self._compute_knowledge_integration_loss(data, output)

                # 总损失
                if task_id > 0 and self.task_knowledge:
                    loss = ce_loss + self.knowledge_weight * integration_loss
                else:
                    loss = ce_loss

                # 反向传播
                loss.backward()
                self.optimizer.step()

                # 统计
                epoch_loss += loss.item()
                _, predicted = output.max(1)
                epoch_total += target.size(0)
                epoch_correct += predicted.eq(target).sum().item()

            # Epoch统计
            epoch_accuracy = epoch_correct / epoch_total if epoch_total > 0 else 0
            avg_epoch_loss = epoch_loss / len(task_loader) if len(task_loader) > 0 else 0

            self.logger.info(
                f"  [{self.client_id}] Task {task_id} Epoch {epoch+1}/{num_epochs}: "
                f"Loss={avg_epoch_loss:.4f}, Acc={epoch_accuracy:.4f}"
            )

            total_loss += epoch_loss
            total_accuracy += epoch_accuracy
            total_samples = epoch_total

        # 训练完成后，提取签名任务知识
        self._extract_signature_knowledge(task_loader, task_id)

        # 计算平均值
        avg_loss = total_loss / (num_epochs * len(task_loader)) if num_epochs > 0 else 0
        avg_accuracy = total_accuracy / num_epochs if num_epochs > 0 else 0

        # 更新统计
        self.training_count += 1
        self.last_training_time = datetime.now()

        # 提取模型权重
        model_weights = get_weights_as_dict(self.model)

        # 返回结果（使用TrainingResponse）
        response = TrainingResponse(
            request_id="",
            client_id=self.client_id,
            success=True,
            result={
                'loss': avg_loss,
                'accuracy': avg_accuracy,
                'task_id': task_id,
                'seen_classes': len(self.seen_classes),
                'num_task_knowledge': len(self.task_knowledge),
                'model_weights': model_weights,
                'epochs_completed': num_epochs,
                'samples_used': total_samples
            },
            execution_time=(datetime.now() - start_time).total_seconds()
        )

        # 触发回调
        for callback in self._callbacks.get('after_train', []):
            callback(response)

        return response

    def _compute_knowledge_integration_loss(
        self,
        data: torch.Tensor,
        current_output: torch.Tensor
    ) -> torch.Tensor:
        """
        计算知识集成损失

        将历史任务的知识集成到当前模型中
        使用温度缩放的KL散度
        """
        if not self.task_knowledge:
            return torch.tensor(0.0, device=self.device)

        total_loss = 0.0
        num_tasks = len(self.task_knowledge)

        # 遍历所有历史任务
        for tid, knowledge in self.task_knowledge.items():
            teacher_model = knowledge['model']
            teacher_model.eval()

            with torch.no_grad():
                teacher_output = teacher_model(data)

            # 使用温度缩放的KL散度
            teacher_probs = F.softmax(teacher_output / self.distill_temperature, dim=1)
            student_log_probs = F.log_softmax(current_output / self.distill_temperature, dim=1)

            kl_div = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
            kl_div *= (self.distill_temperature ** 2)

            total_loss += kl_div

        # 平均损失
        return total_loss / num_tasks if num_tasks > 0 else torch.tensor(0.0, device=self.device)

    def _extract_signature_knowledge(self, train_loader: DataLoader, task_id: int):
        """
        提取签名任务知识

        从当前任务中选择代表性样本，保存为签名知识
        """
        if self._model is None:
            return

        # 保存当前任务的模型副本
        model_copy = copy.deepcopy(self._model)
        model_copy.eval()

        # 提取签名数据索引（基于不确定性采样）
        dataset = train_loader.dataset
        num_signature = max(1, int(len(dataset) * self.signature_ratio))

        # 计算样本的不确定性
        uncertainties = []
        self.model.eval()

        with torch.no_grad():
            for idx in range(len(dataset)):
                data, _ = dataset[idx]
                data = data.unsqueeze(0).to(self.device)
                output = self.model(data)
                probs = F.softmax(output, dim=1)

                # 使用熵作为不确定性度量
                entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
                uncertainties.append((idx, entropy))

        self.model.train()

        # 选择不确定性最高的样本作为签名数据
        uncertainties.sort(key=lambda x: x[1], reverse=True)
        signature_indices = [idx for idx, _ in uncertainties[:num_signature]]

        # 保存到知识库
        self.task_knowledge[task_id] = {
            'model': model_copy,
            'data_indices': signature_indices,
            'num_samples': len(signature_indices)
        }

        self.logger.info(
            f"[{self.client_id}] Extracted signature knowledge for Task {task_id}: "
            f"{len(signature_indices)} samples (based on uncertainty)"
        )

    async def evaluate(self, params: Dict[str, Any]) -> EvaluationResult:
        """评估模型性能"""
        start_time = datetime.now()

        task_id = params.get("task_id", self.current_task_id)

        # 获取评估数据
        if task_id is not None:
            eval_loader = self._get_task_data_loader(task_id)
        else:
            eval_loader = self.train_loader

        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in eval_loader:
                data, target = data.to(self.device), target.to(self.device)

                # 根据场景选择预测方式
                if self.scenario == 'class_incremental' and self.task_knowledge:
                    # CIL场景：使用知识集成预测
                    output = self._integrated_prediction(data)
                else:
                    # TIL场景：直接使用当前模型
                    output = self.model(data)

                loss = self.criterion(output, target)

                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        accuracy = correct / total if total > 0 else 0
        avg_loss = total_loss / len(eval_loader) if len(eval_loader) > 0 else 0

        self.evaluation_count += 1
        self.last_evaluation_time = datetime.now()

        result = EvaluationResult(
            success=True,
            result={
                'accuracy': accuracy,
                'loss': avg_loss,
                'task_id': task_id
            },
            metadata={
                'duration_seconds': (datetime.now() - start_time).total_seconds(),
                'samples_evaluated': total
            }
        )

        # 触发回调
        for callback in self._callbacks.get('after_evaluate', []):
            callback(result)

        return result

    def _integrated_prediction(self, data: torch.Tensor) -> torch.Tensor:
        """
        集成所有任务知识进行预测

        用于Class-Incremental场景（不知道任务ID）
        """
        if not self.task_knowledge:
            # 如果没有历史知识，使用当前模型
            return self.model(data)

        if self.integration_method == 'weighted_sum':
            # 加权求和所有任务的输出
            outputs = []

            # 包含当前模型
            outputs.append(self.model(data))

            # 包含所有历史任务模型
            for tid, knowledge in self.task_knowledge.items():
                teacher_model = knowledge['model']
                teacher_model.eval()
                with torch.no_grad():
                    output = teacher_model(data)
                outputs.append(output)

            # 简单平均
            integrated_output = torch.stack(outputs).mean(dim=0)
            return integrated_output

        elif self.integration_method == 'ensemble':
            # 集成投票
            predictions = []

            # 当前模型的预测
            predictions.append(self.model(data).argmax(dim=1))

            # 历史任务模型的预测
            for tid, knowledge in self.task_knowledge.items():
                teacher_model = knowledge['model']
                teacher_model.eval()
                with torch.no_grad():
                    pred = teacher_model(data).argmax(dim=1)
                predictions.append(pred)

            # 多数投票
            predictions = torch.stack(predictions)
            # 返回众数
            mode_result = torch.mode(predictions, dim=0)
            final_pred = mode_result.values

            # 转换为one-hot输出
            num_classes = self.num_tasks * self.classes_per_task
            output = torch.zeros(data.size(0), num_classes, device=self.device)
            output.scatter_(1, final_pred.unsqueeze(1), 1.0)
            return output

        else:
            # 默认使用当前模型
            return self.model(data)

    async def get_local_model(self) -> ModelData:
        """获取本地模型参数"""
        return ModelData(
            model_data=get_weights_as_dict(self.model),
            metadata={
                'client_id': self.client_id,
                'task_id': self.current_task_id,
                'seen_classes': self.seen_classes,
                'num_task_knowledge': len(self.task_knowledge)
            }
        )

    async def set_local_model(self, model_data: ModelData) -> bool:
        """设置本地模型参数"""
        try:
            if isinstance(model_data, ModelData):
                weights = model_data.model_data
            elif isinstance(model_data, dict):
                weights = model_data.get('model_data', model_data)
            else:
                weights = model_data

            set_weights_from_dict(self.model, weights)
            return True
        except Exception as e:
            self.logger.error(f"Failed to set model: {e}")
            return False

    def get_knowledge_summary(self) -> Dict[str, Any]:
        """
        获取知识库摘要

        Returns:
            知识库统计信息
        """
        return {
            'num_tasks': len(self.task_knowledge),
            'current_task': self.current_task_id,
            'total_signature_samples': sum(
                k['num_samples'] for k in self.task_knowledge.values()
            ),
            'tasks': list(self.task_knowledge.keys()),
            'integration_method': self.integration_method
        }
