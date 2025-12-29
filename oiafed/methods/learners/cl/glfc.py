"""
GLFC (FCIL): Federated Class-Incremental Learning via Global-Local Compensation
CVPR 2022

从 methods/learners/cl/glfc.py 迁移到新架构

Paper: https://openaccess.thecvf.com/content/CVPR2022/papers/Dong_Federated_Class-Incremental_Learning_CVPR_2022_paper.pdf
GitHub: https://github.com/conditionWang/FCIL

核心思想:
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
import copy
import numpy as np

from ....core.learner import Learner
from ....core.types import EpochMetrics, EvalResult
from ....registry import learner


@learner('cl.glfc', description='GLFC: Federated Class-Incremental Learning via Global-Local Compensation (CVPR 2022)')
class GLFCLearner(Learner):
    """
    GLFC/FCIL 持续学习方法的Learner实现

    特点:
    - 类感知梯度补偿机制
    - 类语义关系蒸馏
    - 全局-本地知识平衡
    - 专注于Class-Incremental Learning (CIL)

    配置示例:
    {
        "learner": {
            "name": "cl.glfc",
            "learning_rate": 0.01,
            "batch_size": 32,
            "local_epochs": 5,
            "optimizer": "SGD",
            "momentum": 0.9,
            "num_tasks": 5,
            "classes_per_task": 2,
            "scenario": "class_incremental",
            "gradient_compensation_weight": 0.5,
            "use_class_aware_compensation": true,
            "semantic_distill_weight": 1.0,
            "relation_temperature": 4.0,
            "distill_temperature": 2.0
        }
    }
    """

    def __init__(
        self,
        model,
        datasets,
        tracker=None,
        callbacks=None,
        config=None,
        node_id=None
    ):
        super().__init__(model, None, tracker, callbacks, config, node_id)

        self._datasets = datasets or {}

        # 从配置提取参数
        self.learning_rate = self._config.get('learning_rate', 0.01)
        self.batch_size = self._config.get('batch_size', 32)
        self.local_epochs = self._config.get('local_epochs', 5)
        self.momentum = self._config.get('momentum', 0.9)
        self.optimizer_type = self._config.get('optimizer', 'SGD').upper()

        # 持续学习参数
        self.num_tasks = self._config.get('num_tasks', 5)
        self.classes_per_task = self._config.get('classes_per_task', 2)
        self.scenario = self._config.get('scenario', 'class_incremental')

        # GLFC特定参数
        self.gradient_compensation_weight = self._config.get('gradient_compensation_weight', 0.5)
        self.use_class_aware_compensation = self._config.get('use_class_aware_compensation', True)
        self.semantic_distill_weight = self._config.get('semantic_distill_weight', 1.0)
        self.relation_temperature = self._config.get('relation_temperature', 4.0)
        self.distill_temperature = self._config.get('distill_temperature', 2.0)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 持续学习状态
        self.current_task_id = 0
        self.seen_classes = []
        self.previous_model = None  # 保存上一个任务的模型用于蒸馏

        # 保存每个任务的类原型(class prototypes)
        self.class_prototypes = {}  # {class_id: prototype_vector}

        # 组件占位符
        self.torch_model = None
        self._optimizer = None
        self._criterion = None
        self._train_loader = None
        self._test_loader = None

        self.logger.info(
            f"GLFCLearner {node_id} 初始化完成: "
            f"num_tasks={self.num_tasks}, classes_per_task={self.classes_per_task}, "
            f"gradient_comp={self.gradient_compensation_weight}"
        )

    async def setup(self, config: Dict) -> None:
        """初始化训练环境"""
        # 获取PyTorch模型
        self.torch_model = self._model.get_model()
        self.torch_model.to(self.device)

        # 创建优化器
        if self.optimizer_type == 'SGD':
            self._optimizer = optim.SGD(
                self.torch_model.parameters(),
                lr=self.learning_rate,
                momentum=self.momentum
            )
        elif self.optimizer_type == 'ADAM':
            self._optimizer = optim.Adam(
                self.torch_model.parameters(),
                lr=self.learning_rate
            )
        else:
            raise ValueError(f"不支持的优化器类型: {self.optimizer_type}")

        # 创建损失函数
        loss_name = self._config.get('loss', 'CrossEntropyLoss')
        if loss_name == 'CrossEntropyLoss':
            self._criterion = nn.CrossEntropyLoss()
        elif loss_name == 'MSELoss':
            self._criterion = nn.MSELoss()
        else:
            raise ValueError(f"不支持的损失函数: {loss_name}")

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

        self.logger.info(f"Setup完成: train_samples={len(train_datasets[0]) if train_datasets else 0}")

    def _get_task_data_loader(self, task_id: int) -> DataLoader:
        """
        获取特定任务的数据加载器

        根据当前任务ID筛选对应的类别数据
        """
        train_datasets = self._datasets.get("train", [])
        if not train_datasets:
            return self._train_loader

        dataset = train_datasets[0]

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
            return self._train_loader

        task_dataset = Subset(dataset, indices)

        self.logger.info(
            f"Task {task_id} data loader: {len(indices)} samples, "
            f"classes {task_classes}"
        )

        return DataLoader(
            task_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

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

        # 对于简单模型,使用输出作为特征
        # 实际应用中应该从模型的倒数第二层提取特征
        features = output.detach()

        return features, output

    def _compute_gradient_compensation_loss(
        self,
        features: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        计算类感知梯度补偿损失

        目的:防止新类训练时遗忘旧类的知识
        通过让新类特征与旧类原型保持一定距离来实现
        """
        if not self.class_prototypes:
            return torch.tensor(0.0, device=self.device)

        loss = 0.0
        batch_size = features.size(0)

        # 对每个样本计算与所有旧类原型的距离
        for i in range(batch_size):
            feature = features[i]
            label = target[i].item()

            # 如果当前样本是新类,计算与所有旧类的分离损失
            if label not in self.class_prototypes:
                # 应该远离所有旧类原型
                for class_id, prototype in self.class_prototypes.items():
                    # 归一化特征和原型
                    feature_norm = F.normalize(feature.unsqueeze(0), p=2, dim=1)
                    prototype_norm = F.normalize(prototype.unsqueeze(0), p=2, dim=1)

                    # 计算余弦相似度
                    similarity = (feature_norm * prototype_norm).sum()

                    # 鼓励低相似度(高距离) - 使用hinge loss
                    loss += torch.relu(similarity - 0.3)  # margin = 0.3

        return loss / batch_size if batch_size > 0 else torch.tensor(0.0, device=self.device)

    def _compute_semantic_distillation_loss(
        self,
        data: torch.Tensor,
        student_features: torch.Tensor
    ) -> torch.Tensor:
        """
        计算类语义关系蒸馏损失

        不仅蒸馏输出,还蒸馏类之间的关系
        """
        if self.previous_model is None:
            return torch.tensor(0.0, device=self.device)

        self.previous_model.eval()

        with torch.no_grad():
            teacher_features, teacher_output = self._forward_with_features(self.previous_model, data)

        student_output = self.torch_model(data)

        # 1. 标准知识蒸馏(输出层)
        teacher_probs = F.softmax(teacher_output / self.distill_temperature, dim=1)
        student_log_probs = F.log_softmax(student_output / self.distill_temperature, dim=1)

        output_distill_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
        output_distill_loss *= (self.distill_temperature ** 2)

        # 2. 特征层关系蒸馏 - 计算样本间的相似度矩阵(关系)
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
        """
        if self.torch_model is None:
            return

        self.torch_model.eval()

        # 收集每个类别的特征
        class_features = {}  # {class_id: [features]}

        with torch.no_grad():
            for data, target in train_loader:
                data = data.to(self.device)
                features, _ = self._forward_with_features(self.torch_model, data)

                for i in range(len(target)):
                    class_id = target[i].item()
                    feature = features[i]

                    if class_id not in class_features:
                        class_features[class_id] = []
                    class_features[class_id].append(feature)

        self.torch_model.train()

        # 计算每个类的原型(平均特征)
        for class_id, features in class_features.items():
            features_tensor = torch.stack(features)
            prototype = features_tensor.mean(dim=0)
            self.class_prototypes[class_id] = prototype

        self.logger.info(
            f"[{self._node_id}] Extracted {len(class_features)} class prototypes "
            f"for Task {task_id}: classes {list(class_features.keys())}"
        )

    async def train_epoch(self, epoch: int) -> EpochMetrics:
        """
        单轮训练 - GLFC训练循环

        GLFC的训练流程:
        1. 使用新任务数据训练
        2. 类感知梯度补偿(防止遗忘旧类)
        3. 类语义关系蒸馏
        4. 提取新类的原型
        """
        # 获取当前任务ID
        task_id = self._config.get('current_task_id', self.current_task_id)

        # 更新任务状态
        if task_id != self.current_task_id:
            self.logger.info(
                f"[{self._node_id}] Switching task: {self.current_task_id} -> {task_id}"
            )
            # 保存当前模型作为previous_model
            if self.torch_model is not None:
                self.previous_model = copy.deepcopy(self.torch_model)
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

        self.torch_model.train()

        total_loss = 0.0
        total_ce_loss = 0.0
        total_gc_loss = 0.0
        total_sd_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_idx, (data, target) in enumerate(task_loader):
            data, target = data.to(self.device), target.to(self.device)

            self._optimizer.zero_grad()

            # 前向传播(获取特征和输出)
            features, output = self._forward_with_features(self.torch_model, data)

            # 1. 分类损失
            ce_loss = self._criterion(output, target)

            # 2. 类感知梯度补偿损失(如果有旧类)
            gradient_comp_loss = torch.tensor(0.0, device=self.device)
            if self.use_class_aware_compensation and len(self.class_prototypes) > 0 and task_id > 0:
                gradient_comp_loss = self._compute_gradient_compensation_loss(features, target)

            # 3. 类语义关系蒸馏损失(如果有历史模型)
            semantic_distill_loss = torch.tensor(0.0, device=self.device)
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
            self._optimizer.step()

            # 统计
            total_loss += loss.item() * data.size(0)
            total_ce_loss += ce_loss.item() * data.size(0)
            total_gc_loss += gradient_comp_loss.item() * data.size(0)
            total_sd_loss += semantic_distill_loss.item() * data.size(0)

            _, predicted = output.max(1)
            total_correct += predicted.eq(target).sum().item()
            total_samples += data.size(0)

        avg_loss = total_loss / total_samples
        avg_accuracy = total_correct / total_samples

        self.logger.info(
            f"[{self._node_id}] Task {task_id} Epoch {epoch}: "
            f"Loss={avg_loss:.4f} (CE={total_ce_loss/total_samples:.4f}, "
            f"GC={total_gc_loss/total_samples:.4f}, SD={total_sd_loss/total_samples:.4f}), "
            f"Acc={avg_accuracy:.4f}"
        )

        # 训练完成后,提取新类的原型
        self._extract_class_prototypes(task_loader, task_id)

        epoch_metrics = EpochMetrics(
            epoch=epoch,
            avg_loss=avg_loss,
            total_samples=total_samples,
            metrics={
                'accuracy': avg_accuracy,
                'ce_loss': total_ce_loss / total_samples,
                'gc_loss': total_gc_loss / total_samples,
                'sd_loss': total_sd_loss / total_samples,
                'task_id': task_id,
                'num_prototypes': len(self.class_prototypes)
            }
        )

        # 增加全局 epoch 计数器（用于 MLflow step）
        self._global_epoch_counter += 1

        # 触发 epoch 结束回调
        if self._callbacks:
            await self._callbacks.on_epoch_end(self, epoch, epoch_metrics)

        return epoch_metrics

    def _nearest_prototype_prediction(self, data: torch.Tensor) -> torch.Tensor:
        """
        基于最近原型的预测(用于Class-Incremental场景)
        """
        if not self.class_prototypes:
            # 没有原型,使用模型直接预测
            _, output = self._forward_with_features(self.torch_model, data)
            return output

        features, output = self._forward_with_features(self.torch_model, data)

        # 计算与每个原型的距离,转换为logits
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

                # 转换为logit (相似度越大,logit越大)
                pred_logits[i, class_id] = similarity * 10.0  # 缩放因子

        # 对于没有原型的类,使用模型输出
        for i in range(batch_size):
            for class_id in range(num_classes):
                if class_id not in self.class_prototypes:
                    pred_logits[i, class_id] = output[i, class_id]

        return pred_logits

    async def evaluate_model(self, config: Optional[Dict] = None) -> EvalResult:
        """评估模型性能"""
        task_id = (config or {}).get("task_id", self.current_task_id)

        # 获取评估数据
        if task_id is not None and task_id != self.current_task_id:
            eval_loader = self._get_task_data_loader(task_id)
        else:
            eval_loader = self._test_loader if self._test_loader else self._train_loader

        self.torch_model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in eval_loader:
                data, target = data.to(self.device), target.to(self.device)

                # 根据场景选择预测方式
                if self.scenario == 'class_incremental' and self.class_prototypes:
                    # CIL场景:使用最近原型分类
                    output = self._nearest_prototype_prediction(data)
                else:
                    # TIL场景:直接使用当前模型
                    _, output = self._forward_with_features(self.torch_model, data)

                loss = self._criterion(output, target)

                total_loss += loss.item() * data.size(0)
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        accuracy = correct / total if total > 0 else 0
        avg_loss = total_loss / total if total > 0 else 0

        return EvalResult(
            num_samples=total,
            metrics={
                'accuracy': accuracy,
                'loss': avg_loss,
                'task_id': task_id,
                'num_prototypes': len(self.class_prototypes),
                'seen_classes': len(self.seen_classes)
            }
        )

    async def get_weights(self) -> Dict[str, Any]:
        """获取模型权重"""
        return {name: param.data.clone() for name, param in self.torch_model.state_dict().items()}

    async def set_weights(self, weights: Dict[str, Any]):
        """设置模型权重"""
        # 转换为torch tensor
        torch_weights = {}
        for k, v in weights.items():
            if torch.is_tensor(v):
                torch_weights[k] = v
            else:
                torch_weights[k] = torch.from_numpy(v)

        self.torch_model.load_state_dict(torch_weights)

        self.logger.debug(f"[{self._node_id}] GLFC: Updated model")

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
