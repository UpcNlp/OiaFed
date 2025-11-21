"""
联邦持续学习基类
Base class for Federated Continual Learning methods

提供持续学习的通用功能：
- 任务管理
- 知识蒸馏
- 旧任务数据管理
- 评估指标计算
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Optional
from abc import abstractmethod
import copy

from ...api.decorators import learner
from ...learner.base_learner import BaseLearner
from ...types import TrainingResult


@learner('ContinualBase',
         description='Base class for Federated Continual Learning methods',
         version='1.0',
         author='MOE-FedCL')
class ContinualLearner(BaseLearner):
    """
    持续学习Learner基类

    所有联邦持续学习方法应继承此类
    """

    def __init__(self, client_id: str, config: Optional[Dict[str, Any]] = None, lazy_init: bool = True):
        super().__init__(client_id, config, lazy_init)

        # 持续学习通用参数
        learner_params = self.config.get('learner', {}).get('params', {})

        # 任务管理
        self.current_task_id = 0
        self.num_tasks = learner_params.get('num_tasks', 10)
        self.classes_per_task = learner_params.get('classes_per_task', 10)
        self.seen_classes = []  # 已见过的所有类别

        # 知识蒸馏参数（通用）
        self.use_distillation = learner_params.get('use_distillation', True)
        self.distill_temperature = learner_params.get('distill_temperature', 2.0)
        self.distill_alpha = learner_params.get('distill_alpha', 0.5)  # 蒸馏损失权重

        # 保存旧模型用于蒸馏
        self.previous_models = []  # 保存历史模型

        # 持续学习场景类型
        self.scenario = learner_params.get('scenario', 'class_incremental')
        # 可选: task_incremental, domain_incremental, class_incremental

        # 性能跟踪
        self.task_accuracies = {}  # {task_id: accuracy}
        self.forgetting_measure = 0.0

        self.logger.info(
            f"ContinualLearner initialized: scenario={self.scenario}, "
            f"num_tasks={self.num_tasks}, classes_per_task={self.classes_per_task}"
        )

    async def train(self, training_params: Dict[str, Any]) -> TrainingResult:
        """
        持续学习训练流程（模板方法）

        子类可以重写specific的部分，但保持整体流程一致
        """
        # 1. 更新任务状态
        self._update_task_state(training_params)

        # 2. 准备数据
        train_loader = self._prepare_task_data(training_params)

        # 3. 执行特定方法的训练
        result = await self._train_on_task(train_loader, training_params)

        # 4. 更新历史模型
        self._save_current_model()

        # 5. 评估所有已见任务（可选）
        if training_params.get('eval_all_tasks', False):
            self._evaluate_all_tasks()

        return result

    @abstractmethod
    async def _train_on_task(self, train_loader, training_params: Dict[str, Any]) -> TrainingResult:
        """
        在当前任务上训练（子类必须实现）

        这是各方法差异化的核心部分
        """
        pass

    def _update_task_state(self, training_params: Dict[str, Any]):
        """更新任务状态"""
        task_id = training_params.get('task_id', self.current_task_id)
        new_classes = training_params.get('new_classes', [])

        if task_id != self.current_task_id:
            self.logger.info(
                f"[{self.client_id}] Switching task: {self.current_task_id} -> {task_id}"
            )
            self.current_task_id = task_id
            self.seen_classes.extend(new_classes)

    def _prepare_task_data(self, training_params: Dict[str, Any]):
        """
        准备任务数据

        根据任务ID筛选对应的数据
        """
        # TODO: 实现任务数据筛选逻辑
        from torch.utils.data import DataLoader

        batch_size = training_params.get('batch_size', 64)
        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True
        )

    def _save_current_model(self):
        """保存当前模型到历史"""
        if hasattr(self, '_local_model') and self._local_model is not None:
            model_copy = copy.deepcopy(self._local_model)
            model_copy.eval()  # 设置为评估模式
            self.previous_models.append(model_copy)

    def compute_distillation_loss(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
        temperature: Optional[float] = None
    ) -> torch.Tensor:
        """
        计算知识蒸馏损失（KL散度）

        Args:
            student_output: 学生模型输出 (logits)
            teacher_output: 教师模型输出 (logits)
            temperature: 温度参数（默认使用self.distill_temperature）

        Returns:
            蒸馏损失
        """
        if temperature is None:
            temperature = self.distill_temperature

        # 温度缩放的softmax
        teacher_probs = F.softmax(teacher_output / temperature, dim=1)
        student_log_probs = F.log_softmax(student_output / temperature, dim=1)

        # KL散度
        kl_loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction='batchmean'
        )

        # 温度平方缩放
        kl_loss *= (temperature ** 2)

        return kl_loss

    def get_previous_model_outputs(self, data: torch.Tensor) -> List[torch.Tensor]:
        """
        获取所有历史模型在给定数据上的输出

        用于ensemble蒸馏等场景
        """
        outputs = []
        for model in self.previous_models:
            model.eval()
            with torch.no_grad():
                output = model(data)
                outputs.append(output)
        return outputs

    def _evaluate_all_tasks(self):
        """
        评估在所有已见任务上的性能

        用于计算Average Accuracy和Forgetting Measure
        """
        # TODO: 实现多任务评估逻辑
        pass

    def compute_forgetting_measure(self) -> float:
        """
        计算遗忘度量

        FM = (1/(T-1)) * Σ(max_acc_i - final_acc_i)
        """
        if len(self.task_accuracies) < 2:
            return 0.0

        forgetting_sum = 0.0
        for task_id in range(self.current_task_id):
            if task_id in self.task_accuracies:
                max_acc = self.task_accuracies[task_id].get('max', 0.0)
                current_acc = self.task_accuracies[task_id].get('current', 0.0)
                forgetting_sum += max(0, max_acc - current_acc)

        num_old_tasks = self.current_task_id
        return forgetting_sum / num_old_tasks if num_old_tasks > 0 else 0.0

    def get_continual_learning_metrics(self) -> Dict[str, float]:
        """
        获取持续学习评估指标

        Returns:
            包含AA, FM等指标的字典
        """
        # Average Accuracy
        if self.task_accuracies:
            avg_accuracy = sum(
                acc['current'] for acc in self.task_accuracies.values()
            ) / len(self.task_accuracies)
        else:
            avg_accuracy = 0.0

        # Forgetting Measure
        forgetting_measure = self.compute_forgetting_measure()

        return {
            'average_accuracy': avg_accuracy,
            'forgetting_measure': forgetting_measure,
            'num_seen_classes': len(self.seen_classes),
            'current_task': self.current_task_id
        }
