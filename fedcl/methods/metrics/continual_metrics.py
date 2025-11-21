"""
持续学习评估指标
fedcl/methods/metrics/continual_metrics.py

实现持续学习的标准评估指标：
- Average Accuracy (AA)
- Forgetting Measure (FM)
- Backward Transfer (BWT)
- Forward Transfer (FWT)
- Learning Accuracy (LA)
- Intransigence

参考文献：
- Three scenarios for continual learning (arXiv 2019)
- Continual lifelong learning with neural networks: A review (Neural Networks 2019)
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch


class ContinualLearningMetrics:
    """
    持续学习评估指标计算器

    管理多任务、多轮次的准确率，计算持续学习专用指标
    """

    def __init__(self, num_tasks: int):
        """
        初始化

        Args:
            num_tasks: 任务总数
        """
        self.num_tasks = num_tasks

        # 存储每个任务在每轮训练后的准确率
        # accuracy_matrix[t][k] = 训练完任务k后在任务t上的准确率
        self.accuracy_matrix = {
            task_id: {} for task_id in range(num_tasks)
        }

        # 每个任务训练完成时的准确率（用于计算遗忘）
        self.initial_accuracy = {}

    def update(self,
               current_task: int,
               test_accuracies: Dict[int, float]):
        """
        更新准确率矩阵

        Args:
            current_task: 当前训练完成的任务ID
            test_accuracies: 在所有已见任务上的测试准确率
                            Dict[task_id, accuracy]
        """
        for task_id, accuracy in test_accuracies.items():
            self.accuracy_matrix[task_id][current_task] = accuracy

            # 记录任务首次训练完成时的准确率
            if task_id == current_task and task_id not in self.initial_accuracy:
                self.initial_accuracy[task_id] = accuracy

    def get_average_accuracy(self, up_to_task: Optional[int] = None) -> float:
        """
        计算平均准确率（Average Accuracy）

        AA_T = (1/T) * Σ a_T,i

        Args:
            up_to_task: 计算到哪个任务（None表示所有任务）

        Returns:
            平均准确率
        """
        if up_to_task is None:
            up_to_task = self.num_tasks - 1

        # 获取训练完任务up_to_task后的所有任务准确率
        accuracies = []
        for task_id in range(up_to_task + 1):
            if up_to_task in self.accuracy_matrix[task_id]:
                accuracies.append(self.accuracy_matrix[task_id][up_to_task])

        if not accuracies:
            return 0.0

        return float(np.mean(accuracies))

    def get_forgetting_measure(self, up_to_task: Optional[int] = None) -> float:
        """
        计算遗忘度量（Forgetting Measure）

        FM_T = (1/(T-1)) * Σ(max_j a_j,i - a_T,i)

        Args:
            up_to_task: 计算到哪个任务（None表示所有任务）

        Returns:
            遗忘度量（0表示无遗忘，越大表示遗忘越严重）
        """
        if up_to_task is None:
            up_to_task = self.num_tasks - 1

        if up_to_task == 0:
            return 0.0  # 只有一个任务，无遗忘

        forgetting_sum = 0.0
        num_old_tasks = 0

        # 对每个旧任务
        for task_id in range(up_to_task):
            if task_id not in self.accuracy_matrix:
                continue

            task_accuracies = self.accuracy_matrix[task_id]

            # 找到该任务的最大准确率
            max_accuracy = max(
                acc for t, acc in task_accuracies.items()
                if t >= task_id and t <= up_to_task
            ) if task_accuracies else 0.0

            # 当前（训练完up_to_task后）的准确率
            current_accuracy = task_accuracies.get(up_to_task, 0.0)

            # 遗忘 = 最大准确率 - 当前准确率
            forgetting = max(0, max_accuracy - current_accuracy)
            forgetting_sum += forgetting
            num_old_tasks += 1

        return forgetting_sum / num_old_tasks if num_old_tasks > 0 else 0.0

    def get_backward_transfer(self, up_to_task: Optional[int] = None) -> float:
        """
        计算后向迁移（Backward Transfer）

        BWT_T = (1/(T-1)) * Σ(a_T,i - a_i,i)

        正值表示学习新任务提升了旧任务性能（正向迁移）
        负值表示学习新任务降低了旧任务性能（灾难性遗忘）

        Args:
            up_to_task: 计算到哪个任务（None表示所有任务）

        Returns:
            后向迁移（负值表示遗忘）
        """
        if up_to_task is None:
            up_to_task = self.num_tasks - 1

        if up_to_task == 0:
            return 0.0

        transfer_sum = 0.0
        num_old_tasks = 0

        for task_id in range(up_to_task):
            if task_id not in self.accuracy_matrix:
                continue

            task_accuracies = self.accuracy_matrix[task_id]

            # 训练完该任务时的准确率
            initial_acc = task_accuracies.get(task_id, 0.0)

            # 训练完所有任务后的准确率
            final_acc = task_accuracies.get(up_to_task, 0.0)

            # 后向迁移 = 最终准确率 - 初始准确率
            transfer = final_acc - initial_acc
            transfer_sum += transfer
            num_old_tasks += 1

        return transfer_sum / num_old_tasks if num_old_tasks > 0 else 0.0

    def get_forward_transfer(self) -> float:
        """
        计算前向迁移（Forward Transfer）

        FWT = (1/(T-1)) * Σ(a_i-1,i - b_i)

        其中 b_i 是在任务i上的随机初始化性能

        正值表示之前的训练有助于新任务（正向迁移）

        注意：需要记录随机初始化时的性能（通常设为0或很小的值）
        """
        # 简化实现：假设随机初始化性能为0
        # 完整实现需要在训练每个任务前先测试一次

        if self.num_tasks <= 1:
            return 0.0

        transfer_sum = 0.0
        num_tasks = 0

        for task_id in range(1, self.num_tasks):
            if task_id not in self.accuracy_matrix:
                continue

            task_accuracies = self.accuracy_matrix[task_id]

            # 训练该任务之前（即训练完task_id-1后）在该任务上的准确率
            if task_id - 1 in task_accuracies:
                before_training_acc = task_accuracies[task_id - 1]
            else:
                before_training_acc = 0.0

            # 随机初始化性能（简化为0）
            random_init_acc = 0.0

            # 前向迁移
            transfer = before_training_acc - random_init_acc
            transfer_sum += transfer
            num_tasks += 1

        return transfer_sum / num_tasks if num_tasks > 0 else 0.0

    def get_learning_accuracy(self, up_to_task: Optional[int] = None) -> float:
        """
        计算学习准确率（Learning Accuracy）

        LA = AA + BWT

        Args:
            up_to_task: 计算到哪个任务

        Returns:
            学习准确率
        """
        aa = self.get_average_accuracy(up_to_task)
        bwt = self.get_backward_transfer(up_to_task)
        return aa + bwt

    def get_intransigence(self) -> float:
        """
        计算顽固性（Intransigence）

        INT = (1/T) * Σ(a*_i - a_i,i)

        其中 a*_i 是单独训练任务i时的最佳准确率（上界）

        衡量多任务学习相对于单任务学习的性能损失

        注意：需要单独训练每个任务作为基线
        """
        # 需要外部提供单任务基线性能
        # 这里返回一个占位值
        return 0.0

    def get_all_metrics(self, up_to_task: Optional[int] = None) -> Dict[str, float]:
        """
        获取所有指标

        Args:
            up_to_task: 计算到哪个任务

        Returns:
            所有指标的字典
        """
        return {
            'average_accuracy': self.get_average_accuracy(up_to_task),
            'forgetting_measure': self.get_forgetting_measure(up_to_task),
            'backward_transfer': self.get_backward_transfer(up_to_task),
            'forward_transfer': self.get_forward_transfer(),
            'learning_accuracy': self.get_learning_accuracy(up_to_task),
        }

    def print_accuracy_matrix(self):
        """
        打印准确率矩阵（用于可视化）
        """
        print("\n准确率矩阵 (行=测试任务, 列=训练后的任务):")
        print("=" * 60)

        # 表头
        header = "Task |"
        for train_task in range(self.num_tasks):
            header += f" After T{train_task} |"
        print(header)
        print("-" * 60)

        # 每一行
        for test_task in range(self.num_tasks):
            row = f" T{test_task}  |"
            for train_task in range(self.num_tasks):
                if train_task in self.accuracy_matrix[test_task]:
                    acc = self.accuracy_matrix[test_task][train_task]
                    row += f"   {acc:.4f}   |"
                else:
                    row += "     -      |"
            print(row)

        print("=" * 60)

    def save_to_dict(self) -> Dict:
        """
        保存为字典（用于序列化）

        Returns:
            包含所有数据的字典
        """
        return {
            'num_tasks': self.num_tasks,
            'accuracy_matrix': self.accuracy_matrix,
            'initial_accuracy': self.initial_accuracy,
            'metrics': self.get_all_metrics()
        }

    @classmethod
    def load_from_dict(cls, data: Dict) -> 'ContinualLearningMetrics':
        """
        从字典加载（用于反序列化）

        Args:
            data: 保存的数据字典

        Returns:
            ContinualLearningMetrics实例
        """
        metrics = cls(num_tasks=data['num_tasks'])
        metrics.accuracy_matrix = data['accuracy_matrix']
        metrics.initial_accuracy = data['initial_accuracy']
        return metrics


def compute_continual_metrics(
    accuracy_history: List[Dict[int, float]],
    num_tasks: int
) -> Dict[str, float]:
    """
    从准确率历史计算持续学习指标

    Args:
        accuracy_history: 准确率历史
            accuracy_history[t] = {task_id: accuracy} (训练完任务t后在各任务上的准确率)
        num_tasks: 任务总数

    Returns:
        所有指标的字典
    """
    metrics = ContinualLearningMetrics(num_tasks)

    for current_task, test_accuracies in enumerate(accuracy_history):
        metrics.update(current_task, test_accuracies)

    return metrics.get_all_metrics()


class TaskPerformanceTracker:
    """
    任务性能追踪器

    追踪每个任务在训练过程中的性能变化
    """

    def __init__(self, num_tasks: int):
        self.num_tasks = num_tasks
        self.task_histories = {
            task_id: {
                'train_losses': [],
                'train_accuracies': [],
                'test_losses': [],
                'test_accuracies': [],
                'rounds': []
            }
            for task_id in range(num_tasks)
        }

    def record(self,
              task_id: int,
              round_num: int,
              train_loss: Optional[float] = None,
              train_acc: Optional[float] = None,
              test_loss: Optional[float] = None,
              test_acc: Optional[float] = None):
        """
        记录性能数据

        Args:
            task_id: 任务ID
            round_num: 训练轮次
            train_loss: 训练损失
            train_acc: 训练准确率
            test_loss: 测试损失
            test_acc: 测试准确率
        """
        history = self.task_histories[task_id]
        history['rounds'].append(round_num)

        if train_loss is not None:
            history['train_losses'].append(train_loss)
        if train_acc is not None:
            history['train_accuracies'].append(train_acc)
        if test_loss is not None:
            history['test_losses'].append(test_loss)
        if test_acc is not None:
            history['test_accuracies'].append(test_acc)

    def get_task_history(self, task_id: int) -> Dict:
        """获取特定任务的历史"""
        return self.task_histories[task_id]

    def get_final_performance(self) -> Dict[int, Dict[str, float]]:
        """
        获取每个任务的最终性能

        Returns:
            {task_id: {'train_acc': ..., 'test_acc': ...}}
        """
        final_perf = {}
        for task_id, history in self.task_histories.items():
            final_perf[task_id] = {
                'train_acc': history['train_accuracies'][-1] if history['train_accuracies'] else 0.0,
                'test_acc': history['test_accuracies'][-1] if history['test_accuracies'] else 0.0,
                'train_loss': history['train_losses'][-1] if history['train_losses'] else 0.0,
                'test_loss': history['test_losses'][-1] if history['test_losses'] else 0.0,
            }
        return final_perf
