# fedcl/core/base_evaluator.py
"""
BaseEvaluator抽象基类模块

提供模型评估器的基础接口定义，包括性能评估、遗忘指标计算、
迁移学习指标分析等功能。支持持续学习场景下的综合评估。
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from omegaconf import DictConfig
from loguru import logger
import datetime

from .execution_context import ExecutionContext
from ..exceptions import EvaluationError, ConfigurationError


class BaseEvaluator(ABC):
    """
    评估器抽象基类
    
    定义了持续学习场景下模型评估的基础接口，包括任务性能评估、
    遗忘指标计算、知识迁移分析等功能。支持多种评估策略和指标。
    
    Attributes:
        context: 执行上下文，提供配置和状态管理
        config: 评估器配置参数
        device: 计算设备
        evaluation_history: 评估历史记录
        task_performance_history: 任务性能历史
    """
    
    def __init__(self, context: ExecutionContext, config: DictConfig) -> None:
        """
        初始化评估器
        
        Args:
            context: 执行上下文对象
            config: 评估器配置参数
            
        Raises:
            ConfigurationError: 配置参数无效时抛出
        """
        if not isinstance(context, ExecutionContext):
            raise ConfigurationError("Invalid execution context provided")
            
        if not isinstance(config, DictConfig):
            raise ConfigurationError("Invalid configuration provided")
            
        self.context = context
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.evaluation_history: List[Dict[str, Any]] = []
        self.task_performance_history: Dict[int, List[float]] = {}
        
        logger.debug(f"Initialized {self.__class__.__name__} with device: {self.device}")
    
    @abstractmethod
    def evaluate(self, model: nn.Module, data: DataLoader) -> Dict[str, float]:
        """
        评估模型性能
        
        对指定的模型和数据进行全面的性能评估，计算各种评估指标。
        这是评估器的核心功能，不同的评估器需要实现不同的评估逻辑。
        
        Args:
            model: 要评估的神经网络模型
            data: 评估数据加载器
            
        Returns:
            Dict[str, float]: 评估指标字典，键为指标名称，值为指标值
            
        Raises:
            EvaluationError: 评估过程中出现错误时抛出
        """
        pass
    
    @abstractmethod
    def compute_task_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        计算任务级别指标
        
        基于模型预测结果和真实标签计算任务特定的性能指标。
        
        Args:
            predictions: 模型预测结果张量
            targets: 真实标签张量
            
        Returns:
            Dict[str, float]: 任务指标字典
            
        Raises:
            EvaluationError: 指标计算失败时抛出
        """
        pass
    
    def compute_forgetting_metrics(self, current_results: Dict[str, float], 
                                  historical_results: List[Dict[str, float]]) -> Dict[str, float]:
        """
        计算遗忘指标
        
        分析当前性能与历史性能的差异，计算遗忘相关的指标，
        如后向迁移、遗忘程度等。
        
        Args:
            current_results: 当前任务的评估结果
            historical_results: 历史任务的评估结果列表
            
        Returns:
            Dict[str, float]: 遗忘指标字典
        """
        if not historical_results:
            return {"backward_transfer": 0.0, "forgetting": 0.0}
        
        forgetting_metrics = {}
        
        try:
            # 计算后向迁移 (Backward Transfer)
            if len(historical_results) > 1:
                initial_performance = historical_results[0].get("accuracy", 0.0)
                current_performance = current_results.get("accuracy", 0.0)
                backward_transfer = current_performance - initial_performance
                forgetting_metrics["backward_transfer"] = backward_transfer
            
            # 计算平均遗忘程度
            if len(historical_results) >= 2:
                forgetting_values = []
                for i, hist_result in enumerate(historical_results[:-1]):
                    best_performance = hist_result.get("accuracy", 0.0)
                    current_performance = current_results.get("accuracy", 0.0)
                    forgetting = max(0.0, best_performance - current_performance)
                    forgetting_values.append(forgetting)
                
                if forgetting_values:
                    forgetting_metrics["avg_forgetting"] = np.mean(forgetting_values)
                    forgetting_metrics["max_forgetting"] = np.max(forgetting_values)
            
            logger.debug(f"Computed forgetting metrics: {forgetting_metrics}")
            return forgetting_metrics
            
        except Exception as e:
            logger.error(f"Failed to compute forgetting metrics: {str(e)}")
            return {"backward_transfer": 0.0, "forgetting": 0.0}
    
    def compute_transfer_metrics(self, source_results: Dict[str, float], 
                               target_results: Dict[str, float]) -> Dict[str, float]:
        """
        计算迁移学习指标
        
        分析从源任务到目标任务的知识迁移效果。
        
        Args:
            source_results: 源任务的评估结果
            target_results: 目标任务的评估结果
            
        Returns:
            Dict[str, float]: 迁移学习指标字典
        """
        transfer_metrics = {}
        
        try:
            # 计算前向迁移 (Forward Transfer)
            source_acc = source_results.get("accuracy", 0.0)
            target_acc = target_results.get("accuracy", 0.0)
            
            # 假设有基线性能（无预训练的性能）
            baseline_acc = self.config.get("baseline_accuracy", 0.0)
            
            if baseline_acc > 0:
                forward_transfer = target_acc - baseline_acc
                transfer_metrics["forward_transfer"] = forward_transfer
            
            # 计算迁移效率
            if source_acc > 0:
                transfer_efficiency = target_acc / source_acc
                transfer_metrics["transfer_efficiency"] = transfer_efficiency
            
            # 计算知识保持率
            if source_acc > 0:
                knowledge_retention = min(1.0, target_acc / source_acc)
                transfer_metrics["knowledge_retention"] = knowledge_retention
            
            logger.debug(f"Computed transfer metrics: {transfer_metrics}")
            return transfer_metrics
            
        except Exception as e:
            logger.error(f"Failed to compute transfer metrics: {str(e)}")
            return {"forward_transfer": 0.0, "transfer_efficiency": 1.0}
    
    def generate_evaluation_report(self, results: Dict[str, float]) -> str:
        """
        生成评估报告
        
        基于评估结果生成详细的文本报告，包含各种指标的解释和分析。
        
        Args:
            results: 评估结果字典
            
        Returns:
            str: 格式化的评估报告
        """
        report_lines = ["=" * 50, "MODEL EVALUATION REPORT", "=" * 50, ""]
        
        # 基础性能指标
        if "accuracy" in results:
            report_lines.append(f"Accuracy: {results['accuracy']:.4f}")
        if "loss" in results:
            report_lines.append(f"Loss: {results['loss']:.4f}")
        
        # 持续学习指标
        if "backward_transfer" in results:
            bt = results["backward_transfer"]
            report_lines.append(f"Backward Transfer: {bt:.4f} {'(Positive)' if bt > 0 else '(Negative)'}")
        
        if "avg_forgetting" in results:
            forgetting = results["avg_forgetting"]
            report_lines.append(f"Average Forgetting: {forgetting:.4f}")
        
        if "forward_transfer" in results:
            ft = results["forward_transfer"]
            report_lines.append(f"Forward Transfer: {ft:.4f} {'(Positive)' if ft > 0 else '(Negative)'}")
        
        # 其他指标
        other_metrics = {k: v for k, v in results.items() 
                        if k not in ["accuracy", "loss", "backward_transfer", 
                                   "avg_forgetting", "forward_transfer"]}
        
        if other_metrics:
            report_lines.append("\nAdditional Metrics:")
            for metric, value in other_metrics.items():
                report_lines.append(f"  {metric}: {value:.4f}")
        
        report_lines.extend(["", "=" * 50])
        
        return "\n".join(report_lines)
    
    def supports_online_evaluation(self) -> bool:
        """
        是否支持在线评估
        
        指示此评估器是否支持在训练过程中进行在线评估。
        
        Returns:
            bool: True表示支持在线评估，False表示仅支持离线评估
        """
        return False
    
    def evaluate_continual_learning(self, model: nn.Module, 
                                   task_datasets: List[DataLoader],
                                   task_id: int) -> Dict[str, Any]:
        """
        持续学习综合评估
        
        对模型在所有已学习任务上的性能进行综合评估，计算持续学习相关指标。
        
        Args:
            model: 要评估的模型
            task_datasets: 所有任务的数据集列表
            task_id: 当前任务ID
            
        Returns:
            Dict[str, Any]: 综合评估结果
            
        Raises:
            EvaluationError: 评估失败时抛出
        """
        try:
            evaluation_results = {
                "task_results": {},
                "continual_metrics": {},
                "task_id": task_id
            }
            
            # 评估每个任务
            current_accuracies = []
            for i, dataset in enumerate(task_datasets[:task_id + 1]):
                task_results = self.evaluate(model, dataset)
                evaluation_results["task_results"][f"task_{i}"] = task_results
                current_accuracies.append(task_results.get("accuracy", 0.0))
            
            # 更新任务性能历史
            if task_id not in self.task_performance_history:
                self.task_performance_history[task_id] = []
            self.task_performance_history[task_id] = current_accuracies.copy()
            
            # 计算持续学习指标
            if len(current_accuracies) > 1:
                # 平均准确率
                evaluation_results["continual_metrics"]["average_accuracy"] = np.mean(current_accuracies)
                
                # 计算遗忘
                if task_id > 0 and (task_id - 1) in self.task_performance_history:
                    prev_accuracies = self.task_performance_history[task_id - 1]
                    if len(prev_accuracies) == len(current_accuracies) - 1:
                        # 计算已学习任务的遗忘
                        forgetting_values = []
                        for i in range(len(prev_accuracies)):
                            forgetting = max(0.0, prev_accuracies[i] - current_accuracies[i])
                            forgetting_values.append(forgetting)
                        
                        evaluation_results["continual_metrics"]["avg_forgetting"] = np.mean(forgetting_values)
                        evaluation_results["continual_metrics"]["max_forgetting"] = np.max(forgetting_values)
            
            logger.debug(f"Completed continual learning evaluation for task {task_id}")
            return evaluation_results
            
        except Exception as e:
            raise EvaluationError(f"Continual learning evaluation failed: {str(e)}")
    
    def compute_plasticity_stability_metrics(self, new_task_performance: float,
                                           old_tasks_performance: List[float]) -> Dict[str, float]:
        """
        计算可塑性-稳定性权衡指标
        
        Args:
            new_task_performance: 新任务的性能
            old_tasks_performance: 旧任务的性能列表
            
        Returns:
            Dict[str, float]: 可塑性-稳定性指标
        """
        metrics = {}
        
        if old_tasks_performance:
            # 稳定性：旧任务性能的保持程度
            stability = np.mean(old_tasks_performance)
            metrics["stability"] = stability
            
            # 可塑性：新任务的学习能力
            plasticity = new_task_performance
            metrics["plasticity"] = plasticity
            
            # 平衡指标
            if stability > 0:
                balance = (plasticity * stability) / (plasticity + stability)
                metrics["plasticity_stability_balance"] = balance
        
        return metrics
    
    def record_evaluation(self, evaluation_info: Dict[str, Any]) -> None:
        """
        记录评估信息
        
        Args:
            evaluation_info: 评估信息字典
        """
        evaluation_info["timestamp"] = datetime.datetime.now().isoformat()
        evaluation_info["evaluation_id"] = len(self.evaluation_history)
        
        self.evaluation_history.append(evaluation_info)
        
        # 限制历史记录大小
        max_history = self.config.get("max_history_size", 1000)
        if len(self.evaluation_history) > max_history:
            self.evaluation_history = self.evaluation_history[-max_history:]
    
    def get_evaluation_stats(self) -> Dict[str, Any]:
        """
        获取评估统计信息
        
        Returns:
            Dict[str, Any]: 评估器统计信息
        """
        return {
            "total_evaluations": len(self.evaluation_history),
            "device": str(self.device),
            "evaluator_type": self.__class__.__name__,
            "tasks_evaluated": len(self.task_performance_history),
            "supports_online": self.supports_online_evaluation()
        }
    
    def validate_inputs(self, model: nn.Module, data: DataLoader) -> bool:
        """
        验证评估输入的有效性
        
        Args:
            model: 要评估的模型
            data: 评估数据
            
        Returns:
            bool: True表示输入有效
            
        Raises:
            EvaluationError: 输入无效时抛出
        """
        if not isinstance(model, nn.Module):
            raise EvaluationError("Invalid model: must be a nn.Module instance")
        
        if not isinstance(data, DataLoader):
            raise EvaluationError("Invalid data: must be a DataLoader instance")
        
        if len(data) == 0:
            raise EvaluationError("Empty data loader provided")
        
        # 检查模型是否在正确的设备上
        model_device = next(model.parameters()).device
        if model_device != self.device:
            logger.warning(f"Model on {model_device}, evaluator expects {self.device}")
        
        return True
    
    def reset_history(self) -> None:
        """
        重置评估历史
        
        清空评估历史记录，通常在开始新实验时调用。
        """
        self.evaluation_history.clear()
        self.task_performance_history.clear()
        logger.debug("Evaluation history reset")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        获取内存使用情况
        
        Returns:
            Dict[str, float]: 内存使用统计信息
        """
        if torch.cuda.is_available():
            return {
                "gpu_allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
                "gpu_cached": torch.cuda.memory_reserved() / 1024**3,      # GB
            }
        return {"cpu_memory": "N/A"}
    
    def cleanup(self) -> None:
        """
        清理资源
        
        释放评估器占用的资源，在评估器生命周期结束时调用。
        """
        self.evaluation_history.clear()
        self.task_performance_history.clear()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.debug("Evaluator resources cleaned up")
    
    def __repr__(self) -> str:
        """字符串表示"""
        return (f"{self.__class__.__name__}("
                f"device={self.device}, "
                f"evaluations={len(self.evaluation_history)}, "
                f"tasks={len(self.task_performance_history)})")