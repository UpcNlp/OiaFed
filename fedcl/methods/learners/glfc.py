"""
GLFC (FCIL): Federated Class-Incremental Learning via Global-Local Compensation
CVPR 2022

Paper: https://openaccess.thecvf.com/content/CVPR2022/papers/Dong_Federated_Class-Incremental_Learning_CVPR_2022_paper.pdf
GitHub: https://github.com/conditionWang/FCIL

核心思想：
1. 类感知梯度补偿(Class-Aware Gradient Compensation)
2. 类语义关系蒸馏(Class-Semantic Relation Distillation)
3. 代理服务器辅助模型选择(Proxy Server for Model Selection)
4. 解决联邦场景下的类增量学习问题
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


@learner('GLFC',
         description='GLFC: Federated Class-Incremental Learning via Global-Local Compensation (CVPR 2022)',
         version='1.0',
         author='MOE-FedCL')
class GLFCLearner(BaseLearner):
    """
    GLFC/FCIL 持续学习方法的Learner实现

    特点：
    - 类感知梯度补偿机制
    - 类语义关系蒸馏
    - 全局-本地知识平衡
    - 专注于Class-Incremental Learning (CIL)
    """

    def __init__(self, client_id: str, config: Optional[Dict[str, Any]] = None, lazy_init: bool = True):
        # 提取配置
        learner_params = (config or {}).get('learner', {}).get('params', {})

        # 保存GLFC特定配置
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

        # GLFC特定参数
        self.gradient_compensation_weight = learner_params.get('gradient_compensation_weight', 0.5)
        self.use_class_aware_compensation = learner_params.get('use_class_aware_compensation', True)
        self.semantic_distill_weight = learner_params.get('semantic_distill_weight', 1.0)
        self.relation_temperature = learner_params.get('relation_temperature', 4.0)
        self.distill_temperature = learner_params.get('distill_temperature', 2.0)

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
        self.previous_model = None  # 保存上一个任务的模型用于蒸馏

        # 保存每个任务的类原型(class prototypes)
        self.class_prototypes = {}  # {class_id: prototype_vector}

        # 组件占位符
        self._model = None
        self._optimizer = None
        self._criterion = None
        self._train_loader = None

        self.logger.info(
            f"GLFCLearner {client_id} initialized: "
            f"num_tasks={self.num_tasks}, classes_per_task={self.classes_per_task}, "
            f"gradient_comp={self.gradient_compensation_weight}"
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
        执行GLFC方法的本地训练

        GLFC的训练流程：
        1. 使用新任务数据训练
        2. 类感知梯度补偿(防止遗忘旧类)
        3. 类语义关系蒸馏
        4. 提取新类的原型
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
            # 保存当前模型作为previous_model
            if self._model is not None:
                self.previous_model = copy.deepcopy(self._model)
                self.previous_model.eval()

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

                # 前向传播(获取特征和输出)
                features, output = self._forward_with_features(self.model, data)

                # 1. 分类损失
                ce_loss = self.criterion(output, target)

                # 2. 类感知梯度补偿损失(如果有旧类)
                gradient_comp_loss = 0.0
                if self.use_class_aware_compensation and len(self.class_prototypes) > 0 and task_id > 0:
                    gradient_comp_loss = self._compute_gradient_compensation_loss(features, target)

                # 3. 类语义关系蒸馏损失(如果有历史模型)
                semantic_distill_loss = 0.0
                if self.previous_model is not None and task_id > 0:
                    semantic_distill_loss = self._compute_semantic_distillation_loss(data, features)

                # 总损失
                if task_id > 0:
                    loss = (ce_loss +
                            self.gradient_compensation_weight * gradient_comp_loss +
                            self.semantic_distill_weight * semantic_distill_loss)
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

        # 训练完成后，提取新类的原型
        self._extract_class_prototypes(task_loader, task_id)

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
                'num_prototypes': len(self.class_prototypes),
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

    def _forward_with_features(self, model, data):
        """
        前向传播并返回中间特征和最终输出

        Args:
            model: 模型
            data: 输入数据

        Returns:
            (features, output): 特征向量和分类输出
        """
        # 执行前向传播
        output = model(data)

        # 对于简单模型，使用输出作为特征
        # 实际应用中应该从模型的倒数第二层提取特征
        # 这里使用输出的detach作为特征的简化实现
        features = output.detach()

        return features, output

    def _compute_gradient_compensation_loss(
        self,
        features: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        计算类感知梯度补偿损失

        目的：防止新类训练时遗忘旧类的知识
        通过让新类特征与旧类原型保持一定距离来实现

        Args:
            features: 当前batch的特征向量
            target: 当前batch的标签

        Returns:
            补偿损失
        """
        if not self.class_prototypes:
            return torch.tensor(0.0, device=self.device)

        loss = 0.0
        batch_size = features.size(0)

        # 对每个样本计算与所有旧类原型的距离
        for i in range(batch_size):
            feature = features[i]
            label = target[i].item()

            # 如果当前样本是新类，计算与所有旧类的分离损失
            if label not in self.class_prototypes:
                # 应该远离所有旧类原型
                for class_id, prototype in self.class_prototypes.items():
                    # 归一化特征和原型
                    feature_norm = F.normalize(feature.unsqueeze(0), p=2, dim=1)
                    prototype_norm = F.normalize(prototype.unsqueeze(0), p=2, dim=1)

                    # 计算余弦相似度
                    similarity = (feature_norm * prototype_norm).sum()

                    # 鼓励低相似度(高距离)
                    # 使用hinge loss: max(0, similarity - margin)
                    loss += torch.relu(similarity - 0.3)  # margin = 0.3

        return loss / batch_size if batch_size > 0 else torch.tensor(0.0, device=self.device)

    def _compute_semantic_distillation_loss(
        self,
        data: torch.Tensor,
        student_features: torch.Tensor
    ) -> torch.Tensor:
        """
        计算类语义关系蒸馏损失

        不仅蒸馏输出，还蒸馏类之间的关系

        Args:
            data: 输入数据
            student_features: 学生模型的特征

        Returns:
            关系蒸馏损失
        """
        if self.previous_model is None:
            return torch.tensor(0.0, device=self.device)

        self.previous_model.eval()

        with torch.no_grad():
            teacher_features, teacher_output = self._forward_with_features(self.previous_model, data)

        student_output = self.model(data)

        # 1. 标准知识蒸馏(输出层)
        teacher_probs = F.softmax(teacher_output / self.distill_temperature, dim=1)
        student_log_probs = F.log_softmax(student_output / self.distill_temperature, dim=1)

        output_distill_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
        output_distill_loss *= (self.distill_temperature ** 2)

        # 2. 特征层关系蒸馏
        # 计算样本间的相似度矩阵(关系)
        student_relations = self._compute_relation_matrix(student_features)
        teacher_relations = self._compute_relation_matrix(teacher_features)

        # 使用KL散度约束关系矩阵
        relation_distill_loss = self._relation_kl_div(
            student_relations,
            teacher_relations
        )

        # 组合损失
        total_loss = 0.5 * output_distill_loss + 0.5 * relation_distill_loss

        return total_loss

    def _compute_relation_matrix(self, features: torch.Tensor) -> torch.Tensor:
        """
        计算特征向量之间的关系矩阵(相似度矩阵)

        Args:
            features: [batch_size, feature_dim]

        Returns:
            [batch_size, batch_size] 关系矩阵
        """
        # 归一化特征
        features_norm = F.normalize(features, p=2, dim=1)

        # 计算余弦相似度矩阵
        relation_matrix = torch.mm(features_norm, features_norm.t())

        return relation_matrix

    def _relation_kl_div(
        self,
        student_relations: torch.Tensor,
        teacher_relations: torch.Tensor
    ) -> torch.Tensor:
        """
        计算关系矩阵的KL散度

        Args:
            student_relations: 学生关系矩阵
            teacher_relations: 教师关系矩阵

        Returns:
            KL散度损失
        """
        # 温度缩放的softmax
        student_probs = F.softmax(student_relations / self.relation_temperature, dim=1)
        teacher_probs = F.softmax(teacher_relations / self.relation_temperature, dim=1)

        # KL散度
        kl_loss = F.kl_div(
            student_probs.log(),
            teacher_probs,
            reduction='batchmean'
        )

        return kl_loss * (self.relation_temperature ** 2)

    def _extract_class_prototypes(self, train_loader: DataLoader, task_id: int):
        """
        提取当前任务中每个类别的原型向量

        原型 = 该类所有样本特征的平均值

        Args:
            train_loader: 训练数据加载器
            task_id: 任务ID
        """
        if self._model is None:
            return

        self.model.eval()

        # 收集每个类别的特征
        class_features = {}  # {class_id: [features]}

        with torch.no_grad():
            for data, target in train_loader:
                data = data.to(self.device)
                features, _ = self._forward_with_features(self.model, data)

                for i in range(len(target)):
                    class_id = target[i].item()
                    feature = features[i]

                    if class_id not in class_features:
                        class_features[class_id] = []
                    class_features[class_id].append(feature)

        self.model.train()

        # 计算每个类的原型(平均特征)
        for class_id, features in class_features.items():
            features_tensor = torch.stack(features)
            prototype = features_tensor.mean(dim=0)
            self.class_prototypes[class_id] = prototype

        self.logger.info(
            f"[{self.client_id}] Extracted {len(class_features)} class prototypes "
            f"for Task {task_id}: classes {list(class_features.keys())}"
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
                if self.scenario == 'class_incremental' and self.class_prototypes:
                    # CIL场景：使用最近原型分类
                    output = self._nearest_prototype_prediction(data)
                else:
                    # TIL场景：直接使用当前模型
                    _, output = self._forward_with_features(self.model, data)

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

    def _nearest_prototype_prediction(self, data: torch.Tensor) -> torch.Tensor:
        """
        基于最近原型的预测(用于Class-Incremental场景)

        Args:
            data: 输入数据

        Returns:
            预测输出(logits形式)
        """
        if not self.class_prototypes:
            # 没有原型，使用模型直接预测
            _, output = self._forward_with_features(self.model, data)
            return output

        features, output = self._forward_with_features(self.model, data)

        # 计算与每个原型的距离，转换为logits
        batch_size = features.size(0)
        num_classes = self.num_tasks * self.classes_per_task

        # 初始化输出logits
        pred_logits = torch.zeros(batch_size, num_classes, device=self.device)

        # 对每个样本
        for i in range(batch_size):
            feature = features[i]

            # 计算与所有原型的相似度
            for class_id, prototype in self.class_prototypes.items():
                # 归一化
                feature_norm = F.normalize(feature.unsqueeze(0), p=2, dim=1)
                prototype_norm = F.normalize(prototype.unsqueeze(0), p=2, dim=1)

                # 余弦相似度
                similarity = (feature_norm * prototype_norm).sum()

                # 转换为logit (相似度越大，logit越大)
                pred_logits[i, class_id] = similarity * 10.0  # 缩放因子

        # 对于没有原型的类，使用模型输出
        for i in range(batch_size):
            for class_id in range(num_classes):
                if class_id not in self.class_prototypes:
                    pred_logits[i, class_id] = output[i, class_id]

        return pred_logits

    async def get_local_model(self) -> ModelData:
        """获取本地模型参数"""
        return ModelData(
            model_data=get_weights_as_dict(self.model),
            metadata={
                'client_id': self.client_id,
                'task_id': self.current_task_id,
                'seen_classes': self.seen_classes,
                'num_prototypes': len(self.class_prototypes)
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

    def get_prototype_summary(self) -> Dict[str, Any]:
        """
        获取原型摘要

        Returns:
            原型统计信息
        """
        return {
            'num_classes': len(self.class_prototypes),
            'class_ids': list(self.class_prototypes.keys()),
            'current_task': self.current_task_id,
            'seen_classes': self.seen_classes
        }
