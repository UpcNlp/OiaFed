# fedcl/training/evaluation_engine.py
"""
EvaluationEngine - 模型评估引擎

协调不同评估器进行模型评估，与Hook系统集成，
提供持续学习性能评估、结果可视化和模型比较等功能。
"""

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from ..core.base_evaluator import BaseEvaluator
from ..core.execution_context import ExecutionContext
from ..exceptions import EvaluationError
from ..core.hook_executor import HookExecutor
from ..core.hook import HookPhase


class EvaluationEngine:
    """
    模型评估引擎
    
    协调评估器执行评估任务，与Hook系统集成提供：
    - 单模型评估协调
    - 持续学习性能评估
    - 结果可视化
    - 模型性能比较
    - 自动度量记录（通过MetricsHook）
    - 自动检查点保存（通过CheckpointHook）
    
    注意：此类不重复实现度量记录和检查点管理功能，
    而是与现有的Hook系统协作完成这些任务。
    
    Attributes:
        context: 执行上下文对象
        hook_executor: Hook执行器
        config: 评估引擎配置
        device: 计算设备
        evaluation_history: 评估历史记录
    """
    
    def __init__(self, context: ExecutionContext, hook_executor: Optional[HookExecutor] = None) -> None:
        """
        初始化评估引擎
        
        Args:
            context: 执行上下文对象
            hook_executor: Hook执行器，用于自动度量记录和检查点保存
            
        Raises:
            EvaluationError: 初始化失败时抛出
        """
        if not isinstance(context, ExecutionContext):
            raise EvaluationError("Invalid execution context provided")
            
        self.context = context
        self.hook_executor = hook_executor
        self.config = context.get_config("evaluation", {})
        self.device = torch.device(
            self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        
        # 评估历史记录
        self.evaluation_history: List[Dict[str, Any]] = []
        
        # 可视化配置
        plt.style.use(self.config.get("plot_style", "default"))
        
        logger.debug(f"EvaluationEngine initialized on device: {self.device}")
        if hook_executor:
            logger.debug("Hook integration enabled for automatic metrics and checkpoints")
    
    def evaluate_model(
        self, 
        model: nn.Module, 
        test_data: DataLoader, 
        evaluator: BaseEvaluator,
        task_id: Optional[int] = None
    ) -> Dict[str, float]:
        """
        评估单个模型
        
        使用BaseEvaluator进行实际评估，自动触发Hook系统
        进行度量记录和检查点保存。
        
        Args:
            model: 要评估的模型
            test_data: 测试数据加载器
            evaluator: 评估器实例
            task_id: 可选的任务ID
            
        Returns:
            评估结果字典
            
        Raises:
            EvaluationError: 评估过程出错时抛出
        """
        try:
            logger.info("Starting model evaluation")
            
            # 确保模型在正确设备上并处于评估模式
            model = model.to(self.device)
            model.eval()
            
            # 使用BaseEvaluator进行评估
            with torch.no_grad():
                results = evaluator.evaluate(model, test_data)
            
            # 添加额外信息
            results["task_id"] = task_id
            results["device"] = str(self.device)
            results["num_samples"] = len(test_data.dataset) if hasattr(test_data.dataset, '__len__') else 0
            
            # 记录评估历史
            evaluation_record = {
                "timestamp": torch.cuda.Event().record() if torch.cuda.is_available() else None,
                "task_id": task_id,
                "results": results
            }
            self.evaluation_history.append(evaluation_record)
            
            # 触发评估Hook（自动度量记录和检查点保存）
            if self.hook_executor:
                self.hook_executor.execute_hooks(
                    HookPhase.ON_EVALUATION,
                    self.context,
                    model=model,
                    results=results,
                    task_id=task_id
                )
            
            logger.debug(f"Model evaluation completed. Accuracy: {results.get('accuracy', 0.0):.4f}")
            return results
            
        except Exception as e:
            logger.error(f"Error during model evaluation: {str(e)}")
            raise EvaluationError(f"Model evaluation failed: {str(e)}") from e
    
    def evaluate_continual_learning(
        self, 
        model: nn.Module, 
        all_task_data: List[DataLoader],
        evaluator: BaseEvaluator
    ) -> Dict[str, Any]:
        """
        评估持续学习性能
        
        使用BaseEvaluator的持续学习评估功能，
        自动触发Hook系统进行记录。
        
        Args:
            model: 要评估的模型
            all_task_data: 所有任务的测试数据
            evaluator: 评估器实例
            
        Returns:
            持续学习评估结果
            
        Raises:
            EvaluationError: 评估过程出错时抛出
        """
        try:
            logger.info("Starting continual learning evaluation")
            
            # 使用BaseEvaluator的持续学习评估方法
            results = evaluator.evaluate_continual_learning(
                model, 
                all_task_data, 
                len(all_task_data) - 1  # 当前任务ID
            )
            
            # 记录历史
            cl_evaluation_record = {
                "timestamp": torch.cuda.Event().record() if torch.cuda.is_available() else None,
                "type": "continual_learning",
                "num_tasks": len(all_task_data),
                "results": results
            }
            self.evaluation_history.append(cl_evaluation_record)
            
            # 触发持续学习评估Hook
            if self.hook_executor:
                self.hook_executor.execute_hooks(
                    HookPhase.ON_EVALUATION,
                    self.context,
                    model=model,
                    results=results,
                    evaluation_type="continual_learning",
                    num_tasks=len(all_task_data)
                )
            
            logger.info(f"Continual learning evaluation completed. Average accuracy: {results.get('avg_accuracy', 0.0):.4f}")
            return results
            
        except Exception as e:
            logger.error(f"Error in continual learning evaluation: {str(e)}")
            raise EvaluationError(f"Continual learning evaluation failed: {str(e)}") from e
    
    def compare_models(
        self, 
        models: List[nn.Module], 
        test_data: DataLoader, 
        evaluator: BaseEvaluator,
        model_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        比较多个模型的性能
        
        Args:
            models: 模型列表
            test_data: 测试数据
            evaluator: 评估器实例
            model_names: 可选的模型名称列表
            
        Returns:
            比较结果字典
        """
        try:
            if not models:
                return {"error": "No models provided"}
            
            model_names = model_names or [f"Model_{i}" for i in range(len(models))]
            model_results = []
            
            # 评估每个模型
            for i, model in enumerate(models):
                logger.info(f"Evaluating {model_names[i]}")
                results = self.evaluate_model(model, test_data, evaluator)
                results["model_name"] = model_names[i]
                model_results.append(results)
            
            # 计算比较统计
            comparison = self._compute_comparison_statistics(model_results)
            
            # 触发模型比较Hook
            if self.hook_executor:
                self.hook_executor.execute_hooks(
                    HookPhase.ON_EVALUATION,
                    self.context,
                    results=comparison,
                    evaluation_type="model_comparison",
                    num_models=len(models)
                )
            
            logger.info(f"Model comparison completed for {len(models)} models")
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing models: {str(e)}")
            return {"error": f"Model comparison failed: {str(e)}"}
    
    def visualize_results(self, results: Dict[str, Any], save_path: Path) -> None:
        """
        可视化评估结果
        
        Args:
            results: 评估结果
            save_path: 保存路径
        """
        try:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # 设置图形风格
            plt.rcParams.update({
                'font.size': self.config.get("font_size", 12),
                'figure.figsize': self.config.get("figure_size", [10, 8])
            })
            
            # 根据结果类型选择可视化方法
            if "task_results" in results:
                self._plot_continual_learning_results(results, save_path)
            elif "model_results" in results:
                self._plot_model_comparison(results, save_path)
            else:
                self._plot_single_evaluation(results, save_path)
            
            logger.debug(f"Visualization results saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Error visualizing results: {str(e)}")
            # 不抛出异常，可视化失败不应该影响主要流程
    
    def generate_report(self, results: Dict[str, Any], task_id: Optional[int] = None) -> str:
        """
        生成评估报告
        
        Args:
            results: 评估结果
            task_id: 可选的任务ID
            
        Returns:
            报告内容字符串
        """
        try:
            report_lines = []
            
            if task_id is not None:
                report_lines.extend([
                    f"=== Evaluation Report for Task {task_id} ===",
                    f"Timestamp: {torch.cuda.Event().record() if torch.cuda.is_available() else 'N/A'}",
                    ""
                ])
            else:
                report_lines.extend([
                    "=== Evaluation Report ===",
                    f"Timestamp: {torch.cuda.Event().record() if torch.cuda.is_available() else 'N/A'}",
                    ""
                ])
            
            # 根据结果类型生成不同的报告
            if "task_results" in results:
                report_lines.extend(self._generate_continual_learning_report(results))
            elif "model_results" in results:
                report_lines.extend(self._generate_comparison_report(results))
            else:
                report_lines.extend(self._generate_single_evaluation_report(results))
            
            report_lines.append("="*50)
            
            report_content = "\n".join(report_lines)
            logger.debug(f"Generated evaluation report")
            
            return report_content
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            return f"Error generating report: {str(e)}"
    
    def _compute_comparison_statistics(self, model_results: List[Dict[str, float]]) -> Dict[str, Any]:
        """计算模型比较统计信息"""
        comparison = {
            "num_models": len(model_results),
            "model_results": model_results,
            "metric_statistics": {},
            "best_model_indices": {},
            "performance_ranking": []
        }
        
        # 收集所有度量名称
        all_metrics = set()
        for result in model_results:
            all_metrics.update(k for k, v in result.items() if isinstance(v, (int, float)))
        
        # 计算每个度量的统计信息
        for metric in all_metrics:
            values = []
            for result in model_results:
                if metric in result and isinstance(result[metric], (int, float)):
                    values.append(result[metric])
            
            if values:
                comparison["metric_statistics"][metric] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "median": np.median(values)
                }
                
                # 找到最佳模型
                best_idx = np.argmax(values)
                comparison["best_model_indices"][metric] = best_idx
        
        # 基于准确率进行整体排名
        if "accuracy" in comparison["metric_statistics"]:
            accuracies = []
            for i, result in enumerate(model_results):
                acc = result.get("accuracy", 0.0)
                accuracies.append((i, result.get("model_name", f"Model_{i}"), acc))
            
            # 按准确率降序排序
            comparison["performance_ranking"] = sorted(
                accuracies, key=lambda x: x[2], reverse=True
            )
        
        return comparison
    
    def _plot_continual_learning_results(self, results: Dict[str, Any], save_path: Path) -> None:
        """绘制持续学习结果"""
        if "task_results" not in results:
            return
        
        task_results = results["task_results"]
        task_accuracies = [task.get("accuracy", 0.0) for task in task_results]
        
        plt.figure(figsize=(10, 6))
        task_ids = range(len(task_accuracies))
        plt.plot(task_ids, task_accuracies, 'b-o', linewidth=2, markersize=8)
        plt.xlabel("Task ID")
        plt.ylabel("Accuracy")
        plt.title("Continual Learning Performance")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path / "continual_learning_performance.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_model_comparison(self, results: Dict[str, Any], save_path: Path) -> None:
        """绘制模型比较结果"""
        if "model_results" not in results:
            return
        
        model_results = results["model_results"]
        model_names = [r.get("model_name", f"Model_{i}") for i, r in enumerate(model_results)]
        accuracies = [r.get("accuracy", 0.0) for r in model_results]
        
        plt.figure(figsize=(10, 6))
        plt.bar(model_names, accuracies, alpha=0.7)
        plt.xlabel("Models")
        plt.ylabel("Accuracy")
        plt.title("Model Performance Comparison")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path / "model_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_single_evaluation(self, results: Dict[str, Any], save_path: Path) -> None:
        """绘制单次评估结果"""
        numeric_metrics = {k: v for k, v in results.items() if isinstance(v, (int, float))}
        
        if not numeric_metrics:
            return
        
        plt.figure(figsize=(8, 6))
        metrics_names = list(numeric_metrics.keys())
        metrics_values = list(numeric_metrics.values())
        
        plt.bar(metrics_names, metrics_values, alpha=0.7)
        plt.xlabel("Metrics")
        plt.ylabel("Value")
        plt.title("Evaluation Results")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path / "evaluation_results.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_continual_learning_report(self, results: Dict[str, Any]) -> List[str]:
        """生成持续学习报告"""
        report_lines = ["Continual Learning Performance:"]
        
        if "avg_accuracy" in results:
            report_lines.append(f"  Average Accuracy: {results['avg_accuracy']:.4f}")
        
        if "avg_forgetting" in results:
            report_lines.append(f"  Average Forgetting: {results['avg_forgetting']:.4f}")
        
        if "backward_transfer" in results:
            report_lines.append(f"  Backward Transfer: {results['backward_transfer']:.4f}")
        
        if "task_results" in results:
            report_lines.append("\nPer-task Performance:")
            for i, task_result in enumerate(results["task_results"]):
                acc = task_result.get("accuracy", 0.0)
                report_lines.append(f"  Task {i}: {acc:.4f}")
        
        return report_lines
    
    def _generate_comparison_report(self, results: Dict[str, Any]) -> List[str]:
        """生成模型比较报告"""
        report_lines = ["Model Comparison Results:"]
        
        if "performance_ranking" in results:
            report_lines.append("\nPerformance Ranking:")
            for rank, (idx, name, acc) in enumerate(results["performance_ranking"], 1):
                report_lines.append(f"  {rank}. {name}: {acc:.4f}")
        
        return report_lines
    
    def _generate_single_evaluation_report(self, results: Dict[str, Any]) -> List[str]:
        """生成单次评估报告"""
        report_lines = ["Evaluation Results:"]
        
        for key, value in results.items():
            if isinstance(value, (int, float)):
                report_lines.append(f"  {key}: {value:.4f}")
            elif isinstance(value, str):
                report_lines.append(f"  {key}: {value}")
        
        return report_lines
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """
        获取评估引擎的统计摘要
        
        Returns:
            评估摘要字典
        """
        try:
            summary = {
                "total_evaluations": len(self.evaluation_history),
                "device": str(self.device),
                "config": dict(self.config),
                "hook_executor_enabled": self.hook_executor is not None
            }
            
            if self.evaluation_history:
                # 统计最近的度量
                recent_evaluation = self.evaluation_history[-1]
                summary["latest_evaluation"] = recent_evaluation
                
                # 计算平均性能
                all_accuracies = []
                for record in self.evaluation_history:
                    if "results" in record and "accuracy" in record["results"]:
                        all_accuracies.append(record["results"]["accuracy"])
                
                if all_accuracies:
                    summary["average_accuracy"] = np.mean(all_accuracies)
                    summary["accuracy_std"] = np.std(all_accuracies)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating evaluation summary: {str(e)}")
            return {"error": str(e)}
