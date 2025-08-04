# fedcl/engine/evaluation_engine.py (修复版本)
"""
EvaluationEngine - 修复版评估流程控制引擎

修复原有问题：
1. 实现真实的指标计算逻辑
2. 支持多Learner场景的评估
3. 提供完整的评估报告
4. 支持多种评估策略
"""

import logging
import time
import numpy as np
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import defaultdict

from ..core.execution_context import ExecutionContext
from ..core.base_learner import BaseLearner
from .exceptions import EvaluationEngineError, EngineStateError


class EvaluationState(Enum):
    """评估状态枚举"""
    UNINITIALIZED = "uninitialized"
    INITIALIZED = "初始化完成"
    RUNNING = "running"
    COMPLETED = "完成"
    FAILED = "失败"


class EvaluationType(Enum):
    """评估类型枚举"""
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    MULTI_LEARNER = "multi_learner"
    ENSEMBLE = "ensemble"
    CUSTOM = "custom"


@dataclass
class EvaluationMetrics:
    """评估指标"""
    metrics: Dict[str, float] = field(default_factory=dict)
    confusion_matrix: Optional[np.ndarray] = None
    predictions: Optional[List[Any]] = None
    ground_truths: Optional[List[Any]] = None
    class_predictions: Optional[np.ndarray] = None  # 类别预测
    prediction_probabilities: Optional[np.ndarray] = None  # 预测概率
    evaluation_time: float = 0.0
    sample_count: int = 0
    num_classes: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_metric(self, name: str, default: Any = None) -> Any:
        """获取指定指标"""
        return self.metrics.get(name, default)
    
    def add_metric(self, name: str, value: float) -> None:
        """添加指标"""
        self.metrics[name] = value
    
    def get_accuracy(self) -> Optional[float]:
        """获取准确率"""
        return self.get_metric("accuracy")
    
    def get_f1_score(self) -> Optional[float]:
        """获取F1分数"""
        return self.get_metric("f1_score")
    
    def get_precision(self) -> Optional[float]:
        """获取精确率"""
        return self.get_metric("precision")
    
    def get_recall(self) -> Optional[float]:
        """获取召回率"""
        return self.get_metric("recall")


@dataclass
class EvaluationResults:
    """评估结果"""
    evaluation_type: EvaluationType
    task_id: int
    dataset_name: str
    model_name: str
    learner_id: Optional[str] = None
    metrics: EvaluationMetrics = field(default_factory=EvaluationMetrics)
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def summary(self) -> Dict[str, Any]:
        """获取结果摘要"""
        return {
            "evaluation_type": self.evaluation_type.value,
            "task_id": self.task_id,
            "dataset_name": self.dataset_name,
            "model_name": self.model_name,
            "learner_id": self.learner_id,
            "accuracy": self.metrics.get_accuracy(),
            "precision": self.metrics.get_precision(),
            "recall": self.metrics.get_recall(),
            "f1_score": self.metrics.get_f1_score(),
            "sample_count": self.metrics.sample_count,
            "evaluation_time": self.metrics.evaluation_time,
            "timestamp": self.timestamp
        }


class EvaluationEngine:
    """
    修复版评估流程控制引擎
    
    修复和新增功能：
    1. 真实的指标计算实现
    2. 多Learner评估支持
    3. 完整的混淆矩阵计算
    4. 支持多种任务类型（分类、回归）
    5. 集成评估支持
    """
    
    def __init__(self, context: ExecutionContext, config: Dict[str, Any]):
        """
        初始化评估引擎
        
        Args:
            context: 执行上下文
            config: 评估配置
        """
        self.context = context
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 评估状态
        self._state = EvaluationState.UNINITIALIZED
        self._current_evaluation_id = 0
        
        # 评估历史
        self._evaluation_results: List[EvaluationResults] = []
        
        # 支持的评估指标
        self._supported_metrics = [
            "accuracy", "precision", "recall", "f1_score",
            "auc", "roc_auc", "mean_squared_error", "mean_absolute_error",
            "loss", "top_k_accuracy", "balanced_accuracy"
        ]
        
        # 任务类型检测
        self.task_type = self.config.get('task_type', 'classification')  # classification, regression
        
        self.logger.debug("FixedEvaluationEngine initialized")
    
    @property
    def evaluation_state(self) -> EvaluationState:
        """获取评估状态"""
        return self._state
    
    @property
    def current_evaluation_id(self) -> int:
        """获取当前评估ID"""
        return self._current_evaluation_id
    
    @property
    def supported_metrics(self) -> List[str]:
        """获取支持的评估指标"""
        return self._supported_metrics.copy()
    
    def initialize_evaluation(self) -> None:
        """初始化评估"""
        try:
            self.logger.debug("Initializing evaluation...")
            
            if self._state != EvaluationState.UNINITIALIZED:
                raise EngineStateError(
                    f"Cannot initialize evaluation in state {self._state.value}",
                    {"current_state": self._state.value}
                )
            
            # 验证配置
            self._validate_evaluation_config()
            
            # 设置状态
            self._state = EvaluationState.INITIALIZED
            
            self.logger.debug("Evaluation initialized successfully")
            
        except Exception as e:
            self._state = EvaluationState.FAILED
            self.logger.error(f"Failed to initialize evaluation: {e}")
            raise EvaluationEngineError(f"Evaluation initialization failed: {e}")
    
    def evaluate_single_learner(self, 
                               learner: BaseLearner,
                               test_data: DataLoader,
                               task_id: int = 0,
                               dataset_name: str = "test_data") -> EvaluationResults:
        """
        评估单个learner
        
        Args:
            learner: 要评估的learner
            test_data: 测试数据
            task_id: 任务ID
            dataset_name: 数据集名称
            
        Returns:
            EvaluationResults: 评估结果
        """
        try:
            self.logger.debug(f"Evaluating single learner: {learner.learner_id}")
            
            if self._state not in [EvaluationState.INITIALIZED, EvaluationState.RUNNING]:
                raise EngineStateError(
                    f"Cannot evaluate in state {self._state.value}",
                    {"current_state": self._state.value}
                )
            
            self._state = EvaluationState.RUNNING
            self._current_evaluation_id += 1
            
            start_time = time.time()
            
            # 执行实际评估
            metrics = self._compute_learner_metrics(learner, test_data)
            
            evaluation_time = time.time() - start_time
            metrics.evaluation_time = evaluation_time
            
            # 创建评估结果
            results = EvaluationResults(
                evaluation_type=EvaluationType.TEST,
                task_id=task_id,
                dataset_name=dataset_name,
                model_name=learner.__class__.__name__,
                learner_id=learner.learner_id,
                metrics=metrics,
                metadata={
                    "evaluation_id": self._current_evaluation_id,
                    "task_type": self.task_type
                }
            )
            
            # 记录结果
            self._evaluation_results.append(results)
            
            # 记录指标到上下文
            for metric_name, metric_value in metrics.metrics.items():
                self.context.log_metric(f"eval_{metric_name}", metric_value, 
                                      self._current_evaluation_id, learner.learner_id)
            
            self.logger.debug(f"Single learner evaluation completed: {learner.learner_id}")
            return results
            
        except Exception as e:
            self._state = EvaluationState.FAILED
            self.logger.error(f"Single learner evaluation failed: {e}")
            raise EvaluationEngineError(f"Single learner evaluation failed: {e}")
        finally:
            if self._state == EvaluationState.RUNNING:
                self._state = EvaluationState.COMPLETED
    
    def evaluate_multi_learners(self, 
                               learners: Dict[str, BaseLearner],
                               test_data: DataLoader,
                               task_id: int = 0,
                               dataset_name: str = "test_data") -> Dict[str, EvaluationResults]:
        """
        评估多个learner
        
        Args:
            learners: learner字典 {learner_id: learner}
            test_data: 测试数据
            task_id: 任务ID  
            dataset_name: 数据集名称
            
        Returns:
            Dict[str, EvaluationResults]: 评估结果字典
        """
        try:
            self.logger.debug(f"Evaluating {len(learners)} learners")
            
            results = {}
            
            # 评估每个learner
            for learner_id, learner in learners.items():
                try:
                    result = self.evaluate_single_learner(
                        learner=learner,
                        test_data=test_data,
                        task_id=task_id,
                        dataset_name=dataset_name
                    )
                    results[learner_id] = result
                    
                except Exception as e:
                    self.logger.error(f"Evaluation failed for learner {learner_id}: {e}")
                    if self.config.get('stop_on_learner_failure', False):
                        break
            
            # 计算聚合指标
            if len(results) > 1:
                aggregated_result = self._compute_aggregated_metrics(results, task_id, dataset_name)
                results["aggregated"] = aggregated_result
            
            self.logger.debug(f"Multi-learner evaluation completed: {len(results)} results")
            return results
            
        except Exception as e:
            self.logger.error(f"Multi-learner evaluation failed: {e}")
            raise EvaluationEngineError(f"Multi-learner evaluation failed: {e}")
    
    def evaluate_ensemble(self,
                         learners: Dict[str, BaseLearner],
                         test_data: DataLoader,
                         ensemble_method: str = "voting",
                         task_id: int = 0,
                         dataset_name: str = "test_data") -> EvaluationResults:
        """
        评估集成模型
        
        Args:
            learners: learner字典
            test_data: 测试数据
            ensemble_method: 集成方法 ("voting", "averaging", "stacking")
            task_id: 任务ID
            dataset_name: 数据集名称
            
        Returns:
            EvaluationResults: 集成评估结果
        """
        try:
            self.logger.debug(f"Evaluating ensemble with method: {ensemble_method}")
            
            self._state = EvaluationState.RUNNING 
            self._current_evaluation_id += 1
            
            start_time = time.time()
            
            # 收集所有learner的预测
            all_predictions = {}
            all_probabilities = {}
            ground_truths = None
            
            for learner_id, learner in learners.items():
                predictions, probabilities, targets = self._get_learner_predictions(learner, test_data)
                all_predictions[learner_id] = predictions
                all_probabilities[learner_id] = probabilities
                
                if ground_truths is None:
                    ground_truths = targets
            
            # 执行集成
            if ensemble_method == "voting":
                ensemble_predictions = self._voting_ensemble(all_predictions)
            elif ensemble_method == "averaging":
                ensemble_predictions = self._averaging_ensemble(all_probabilities)
            else:  # stacking or other
                ensemble_predictions = self._simple_ensemble(all_predictions)
            
            # 计算集成指标
            metrics = self._compute_metrics_from_predictions(
                ensemble_predictions, ground_truths, 
                prediction_probabilities=None
            )
            
            evaluation_time = time.time() - start_time
            metrics.evaluation_time = evaluation_time
            
            # 创建集成结果
            ensemble_result = EvaluationResults(
                evaluation_type=EvaluationType.ENSEMBLE,
                task_id=task_id,
                dataset_name=dataset_name,
                model_name=f"ensemble_{ensemble_method}",
                metrics=metrics,
                metadata={
                    "evaluation_id": self._current_evaluation_id,
                    "ensemble_method": ensemble_method,
                    "num_learners": len(learners),
                    "learner_ids": list(learners.keys())
                }
            )
            
            self._evaluation_results.append(ensemble_result)
            
            self.logger.debug(f"Ensemble evaluation completed with method: {ensemble_method}")
            return ensemble_result
            
        except Exception as e:
            self._state = EvaluationState.FAILED
            self.logger.error(f"Ensemble evaluation failed: {e}")
            raise EvaluationEngineError(f"Ensemble evaluation failed: {e}")
        finally:
            if self._state == EvaluationState.RUNNING:
                self._state = EvaluationState.COMPLETED
    
    def _compute_learner_metrics(self, learner: BaseLearner, test_data: DataLoader) -> EvaluationMetrics:
        """计算单个learner的评估指标"""
        model = learner.get_model()
        model.eval()
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        total_loss = 0.0
        num_batches = 0
        
        # 确定损失函数
        criterion = getattr(learner, 'criterion', None)
        if criterion is None and self.task_type == 'classification':
            criterion = nn.CrossEntropyLoss()
        elif criterion is None and self.task_type == 'regression':
            criterion = nn.MSELoss()
        
        with torch.no_grad():
            for batch_data in test_data:
                # 处理批次数据
                if isinstance(batch_data, (tuple, list)):
                    if len(batch_data) >= 2:
                        inputs, targets = batch_data[0], batch_data[1]
                    else:
                        inputs, targets = batch_data[0], None
                else:
                    inputs, targets = batch_data, None
                
                # 移到设备
                if hasattr(inputs, 'to'):
                    inputs = inputs.to(model.device if hasattr(model, 'device') else learner.device)
                if targets is not None and hasattr(targets, 'to'):
                    targets = targets.to(model.device if hasattr(model, 'device') else learner.device)
                
                # 前向传播
                outputs = model(inputs)
                
                # 计算损失
                if targets is not None and criterion is not None:
                    loss = criterion(outputs, targets)
                    total_loss += loss.item()
                    num_batches += 1
                
                # 处理预测结果
                if self.task_type == 'classification':
                    # 分类任务
                    if outputs.dim() > 1 and outputs.size(1) > 1:
                        # 多分类
                        probabilities = torch.softmax(outputs, dim=1)
                        predictions = torch.argmax(outputs, dim=1)
                        all_probabilities.extend(probabilities.cpu().numpy())
                    else:
                        # 二分类
                        probabilities = torch.sigmoid(outputs)
                        predictions = (probabilities > 0.5).long().squeeze()
                        all_probabilities.extend(probabilities.cpu().numpy())
                    
                    all_predictions.extend(predictions.cpu().numpy())
                    
                else:
                    # 回归任务
                    all_predictions.extend(outputs.squeeze().cpu().numpy())
                
                if targets is not None:
                    all_targets.extend(targets.cpu().numpy())
        
        # 转换为numpy数组
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets) if all_targets else None
        all_probabilities = np.array(all_probabilities) if all_probabilities else None
        
        # 计算指标
        metrics = self._compute_metrics_from_predictions(
            all_predictions, all_targets, all_probabilities
        )
        
        # 添加损失
        if num_batches > 0:
            metrics.add_metric("loss", total_loss / num_batches)
        
        return metrics
    
    def _compute_metrics_from_predictions(self, 
                                        predictions: np.ndarray,
                                        targets: Optional[np.ndarray],
                                        prediction_probabilities: Optional[np.ndarray] = None) -> EvaluationMetrics:
        """从预测结果计算指标"""
        metrics = EvaluationMetrics()
        
        metrics.predictions = predictions.tolist() if isinstance(predictions, np.ndarray) else predictions
        metrics.ground_truths = targets.tolist() if targets is not None and isinstance(targets, np.ndarray) else targets
        metrics.prediction_probabilities = prediction_probabilities
        metrics.sample_count = len(predictions)
        
        if targets is None:
            self.logger.warning("No ground truth labels available, skipping metric computation")
            return metrics
        
        if self.task_type == 'classification':
            return self._compute_classification_metrics(predictions, targets, prediction_probabilities, metrics)
        else:
            return self._compute_regression_metrics(predictions, targets, metrics)
    
    def _compute_classification_metrics(self, 
                                      predictions: np.ndarray,
                                      targets: np.ndarray,
                                      probabilities: Optional[np.ndarray],
                                      metrics: EvaluationMetrics) -> EvaluationMetrics:
        """计算分类指标"""
        try:
            from sklearn.metrics import (
                accuracy_score, precision_recall_fscore_support, 
                confusion_matrix, roc_auc_score, balanced_accuracy_score
            )
            
            # 基本准确率
            accuracy = accuracy_score(targets, predictions)
            metrics.add_metric("accuracy", accuracy)
            
            # 平衡准确率
            balanced_acc = balanced_accuracy_score(targets, predictions)
            metrics.add_metric("balanced_accuracy", balanced_acc)
            
            # 精确率、召回率、F1分数
            precision, recall, f1, support = precision_recall_fscore_support(
                targets, predictions, average='weighted', zero_division=0
            )
            metrics.add_metric("precision", precision)
            metrics.add_metric("recall", recall)
            metrics.add_metric("f1_score", f1)
            
            # 每个类别的指标
            precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
                targets, predictions, average=None, zero_division=0
            )
            
            # 混淆矩阵
            conf_matrix = confusion_matrix(targets, predictions)
            metrics.confusion_matrix = conf_matrix
            metrics.num_classes = len(np.unique(targets))
            
            # 添加每个类别的指标到元数据
            metrics.metadata.update({
                "precision_per_class": precision_per_class.tolist(),
                "recall_per_class": recall_per_class.tolist(),
                "f1_per_class": f1_per_class.tolist(),
                "support_per_class": support.tolist(),
                "confusion_matrix": conf_matrix.tolist()
            })
            
            # AUC指标（如果有概率预测）
            if probabilities is not None:
                try:
                    if metrics.num_classes == 2:
                        # 二分类AUC
                        auc = roc_auc_score(targets, probabilities[:, 1] if probabilities.ndim > 1 else probabilities)
                        metrics.add_metric("auc", auc)
                        metrics.add_metric("roc_auc", auc)
                    elif metrics.num_classes > 2:
                        # 多分类AUC
                        auc = roc_auc_score(targets, probabilities, multi_class='ovr', average='weighted')
                        metrics.add_metric("auc", auc)
                        metrics.add_metric("roc_auc", auc)
                except Exception as e:
                    self.logger.warning(f"Failed to compute AUC: {e}")
            
            # Top-k准确率（如果是多分类且有概率）
            if probabilities is not None and metrics.num_classes > 2:
                for k in [3, 5]:
                    if k < metrics.num_classes:
                        top_k_acc = self._compute_top_k_accuracy(targets, probabilities, k)
                        metrics.add_metric(f"top_{k}_accuracy", top_k_acc)
            
        except ImportError:
            self.logger.error("sklearn not available, using simplified metrics")
            # 简化指标计算
            accuracy = np.mean(predictions == targets)
            metrics.add_metric("accuracy", accuracy)
        
        return metrics
    
    def _compute_regression_metrics(self, 
                                  predictions: np.ndarray,
                                  targets: np.ndarray,
                                  metrics: EvaluationMetrics) -> EvaluationMetrics:
        """计算回归指标"""
        try:
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            # 均方误差
            mse = mean_squared_error(targets, predictions)
            metrics.add_metric("mean_squared_error", mse)
            metrics.add_metric("mse", mse)
            
            # 均方根误差
            rmse = np.sqrt(mse)
            metrics.add_metric("rmse", rmse)
            
            # 平均绝对误差
            mae = mean_absolute_error(targets, predictions)
            metrics.add_metric("mean_absolute_error", mae)
            metrics.add_metric("mae", mae)
            
            # R²分数
            r2 = r2_score(targets, predictions)
            metrics.add_metric("r2_score", r2)
            
            # 平均绝对百分比误差
            mape = np.mean(np.abs((targets - predictions) / np.maximum(np.abs(targets), 1e-8))) * 100
            metrics.add_metric("mape", mape)
            
        except ImportError:
            self.logger.error("sklearn not available, using simplified regression metrics")
            # 简化回归指标
            mse = np.mean((predictions - targets) ** 2)
            mae = np.mean(np.abs(predictions - targets))
            metrics.add_metric("mse", mse)
            metrics.add_metric("mae", mae)
        
        return metrics
    
    def _compute_top_k_accuracy(self, targets: np.ndarray, probabilities: np.ndarray, k: int) -> float:
        """计算Top-k准确率"""
        top_k_predictions = np.argsort(probabilities, axis=1)[:, -k:]
        correct = 0
        for i, target in enumerate(targets):
            if target in top_k_predictions[i]:
                correct += 1
        return correct / len(targets)
    
    def _get_learner_predictions(self, learner: BaseLearner, test_data: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """获取learner的预测结果"""
        model = learner.get_model()
        model.eval()
        
        all_predictions = []
        all_probabilities = []
        all_targets = []
        
        with torch.no_grad():
            for batch_data in test_data:
                if isinstance(batch_data, (tuple, list)):
                    inputs, targets = batch_data[0], batch_data[1]
                else:
                    inputs, targets = batch_data, None
                
                # 移到设备
                if hasattr(inputs, 'to'):
                    inputs = inputs.to(learner.device)
                if targets is not None and hasattr(targets, 'to'):
                    targets = targets.to(learner.device)
                
                outputs = model(inputs)
                
                if self.task_type == 'classification':
                    if outputs.dim() > 1 and outputs.size(1) > 1:
                        probabilities = torch.softmax(outputs, dim=1)
                        predictions = torch.argmax(outputs, dim=1)
                    else:
                        probabilities = torch.sigmoid(outputs)
                        predictions = (probabilities > 0.5).long().squeeze()
                    
                    all_probabilities.extend(probabilities.cpu().numpy())
                    all_predictions.extend(predictions.cpu().numpy())
                else:
                    all_predictions.extend(outputs.squeeze().cpu().numpy())
                    all_probabilities.extend(outputs.squeeze().cpu().numpy())  # 回归时概率就是预测值
                
                if targets is not None:
                    all_targets.extend(targets.cpu().numpy())
        
        return (np.array(all_predictions), 
                np.array(all_probabilities), 
                np.array(all_targets) if all_targets else None)
    
    def _voting_ensemble(self, all_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """投票集成"""
        prediction_arrays = list(all_predictions.values())
        stacked_predictions = np.stack(prediction_arrays, axis=1)
        
        # 对每个样本进行投票
        ensemble_predictions = []
        for sample_predictions in stacked_predictions:
            # 简单多数投票
            unique, counts = np.unique(sample_predictions, return_counts=True)
            ensemble_predictions.append(unique[np.argmax(counts)])
        
        return np.array(ensemble_predictions)
    
    def _averaging_ensemble(self, all_probabilities: Dict[str, np.ndarray]) -> np.ndarray:
        """平均概率集成"""
        probability_arrays = list(all_probabilities.values())
        stacked_probabilities = np.stack(probability_arrays, axis=0)
        
        # 计算平均概率
        avg_probabilities = np.mean(stacked_probabilities, axis=0)
        
        # 基于平均概率预测
        if avg_probabilities.ndim > 1:
            ensemble_predictions = np.argmax(avg_probabilities, axis=1)
        else:
            ensemble_predictions = (avg_probabilities > 0.5).astype(int)
        
        return ensemble_predictions
    
    def _simple_ensemble(self, all_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """简单集成（选择第一个预测）"""
        return list(all_predictions.values())[0]
    
    def _compute_aggregated_metrics(self, 
                                  results: Dict[str, EvaluationResults],
                                  task_id: int,
                                  dataset_name: str) -> EvaluationResults:
        """计算聚合指标"""
        # 计算平均指标
        all_metrics = defaultdict(list)
        
        for learner_id, result in results.items():
            if learner_id != "aggregated":  # 避免递归
                for metric_name, metric_value in result.metrics.metrics.items():
                    all_metrics[metric_name].append(metric_value)
        
        # 计算平均值
        avg_metrics = EvaluationMetrics()
        for metric_name, values in all_metrics.items():
            if values:
                avg_metrics.add_metric(metric_name, np.mean(values))
        
        # 添加聚合统计
        avg_metrics.metadata.update({
            "num_learners": len(results),
            "learner_ids": list(results.keys()),
            "metric_std": {
                metric_name: np.std(values) 
                for metric_name, values in all_metrics.items() if values
            }
        })
        
        return EvaluationResults(
            evaluation_type=EvaluationType.MULTI_LEARNER,
            task_id=task_id,
            dataset_name=dataset_name,
            model_name="aggregated_metrics",
            metrics=avg_metrics,
            metadata={"aggregation_type": "average"}
        )
    
    def _validate_evaluation_config(self) -> None:
        """验证评估配置"""
        # 检查任务类型
        if self.task_type not in ['classification', 'regression']:
            raise EvaluationEngineError(f"Unsupported task type: {self.task_type}")
        
        # 检查指标配置
        metrics_to_compute = self.config.get("metrics_to_compute", [])
        if metrics_to_compute:
            unsupported_metrics = set(metrics_to_compute) - set(self._supported_metrics)
            if unsupported_metrics:
                raise EvaluationEngineError(
                    f"Unsupported metrics: {unsupported_metrics}. "
                    f"Supported metrics: {self._supported_metrics}"
                )
    
    def get_evaluation_statistics(self) -> Dict[str, Any]:
        """获取评估统计信息"""
        if not self._evaluation_results:
            return {
                "total_evaluations": 0,
                "evaluation_state": self._state.value
            }
        
        # 计算统计信息
        evaluation_types = [r.evaluation_type.value for r in self._evaluation_results]
        model_names = [r.model_name for r in self._evaluation_results]
        learner_ids = [r.learner_id for r in self._evaluation_results if r.learner_id]
        
        # 计算平均指标
        all_accuracies = [r.metrics.get_accuracy() for r in self._evaluation_results 
                         if r.metrics.get_accuracy() is not None]
        all_f1_scores = [r.metrics.get_f1_score() for r in self._evaluation_results 
                        if r.metrics.get_f1_score() is not None]
        
        return {
            "total_evaluations": len(self._evaluation_results),
            "evaluation_types": list(set(evaluation_types)),
            "unique_models": list(set(model_names)),
            "unique_learners": list(set(learner_ids)),
            "average_evaluation_times": [r.metrics.evaluation_time for r in self._evaluation_results],
            "evaluation_state": self._state.value,
            "performance_summary": {
                "avg_accuracy": np.mean(all_accuracies) if all_accuracies else None,
                "std_accuracy": np.std(all_accuracies) if all_accuracies else None,
                "avg_f1_score": np.mean(all_f1_scores) if all_f1_scores else None,
                "std_f1_score": np.std(all_f1_scores) if all_f1_scores else None,
            }
        }
    
    def generate_evaluation_report(self, 
                                 learner_ids: Optional[List[str]] = None,
                                 output_format: str = "dict") -> Any:
        """生成评估报告"""
        if learner_ids is None:
            selected_results = self._evaluation_results
        else:
            selected_results = [
                r for r in self._evaluation_results
                if r.learner_id in learner_ids
            ]
        
        if not selected_results:
            return {"message": "No evaluation results found"}
        
        # 生成报告数据
        report_data = {
            "summary": {
                "total_evaluations": len(selected_results),
                "evaluation_types": list(set(r.evaluation_type.value for r in selected_results)),
                "evaluated_learners": list(set(r.learner_id for r in selected_results if r.learner_id)),
                "time_range": {
                    "start": min(r.timestamp for r in selected_results),
                    "end": max(r.timestamp for r in selected_results)
                }
            },
            "results": [r.summary() for r in selected_results],
            "performance_analysis": self._analyze_performance(selected_results),
            "timestamp": time.time()
        }
        
        if output_format == "dict":
            return report_data
        elif output_format == "json":
            import json
            return json.dumps(report_data, indent=2, default=str)
        else:
            raise EvaluationEngineError(f"Unsupported output format: {output_format}")
    
    def _analyze_performance(self, results: List[EvaluationResults]) -> Dict[str, Any]:
        """分析性能数据"""
        if not results:
            return {}
        
        # 按learner分组分析
        learner_performance = defaultdict(list)
        for result in results:
            if result.learner_id:
                learner_performance[result.learner_id].append(result)
        
        analysis = {}
        for learner_id, learner_results in learner_performance.items():
            accuracies = [r.metrics.get_accuracy() for r in learner_results if r.metrics.get_accuracy() is not None]
            f1_scores = [r.metrics.get_f1_score() for r in learner_results if r.metrics.get_f1_score() is not None]
            
            analysis[learner_id] = {
                "num_evaluations": len(learner_results),
                "avg_accuracy": np.mean(accuracies) if accuracies else None,
                "std_accuracy": np.std(accuracies) if accuracies else None,
                "avg_f1_score": np.mean(f1_scores) if f1_scores else None,
                "std_f1_score": np.std(f1_scores) if f1_scores else None,
                "best_accuracy": max(accuracies) if accuracies else None,
                "worst_accuracy": min(accuracies) if accuracies else None
            }
        
        return analysis
    
    def clear_evaluation_history(self) -> None:
        """清空评估历史"""
        self._evaluation_results.clear()
        self._current_evaluation_id = 0
        self.logger.debug("Evaluation history cleared")