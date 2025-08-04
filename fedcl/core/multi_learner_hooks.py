# fedcl/core/multi_learner_hooks.py
"""
多Learner Hook系统扩展

提供多Learner场景专用的Hook阶段和基础Hook实现，支持：
- 多Learner协调
- 特征交换
- 执行计划优化
- 资源管理
- 聚合策略
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Any, Optional
import time
import torch
from loguru import logger

from .hook import Hook, HookPhase, HookPriority
from .execution_context import ExecutionContext
from .base_learner import BaseLearner


class MultiLearnerHookPhase(Enum):
    """多Learner专用Hook阶段"""
    # 初始化阶段
    MULTI_LEARNER_INIT = "multi_learner_init"
    LEARNERS_REGISTRATION = "learners_registration"
    LEARNERS_READY = "learners_ready"
    
    # 执行计划阶段
    EXECUTION_PLANNING = "execution_planning"
    PLAN_OPTIMIZATION = "plan_optimization"
    RESOURCE_ALLOCATION = "resource_allocation"
    
    # 执行协调阶段
    BEFORE_EXECUTION_GROUP = "before_execution_group"
    AFTER_EXECUTION_GROUP = "after_execution_group"
    BEFORE_LEARNER_EXECUTION = "before_learner_execution"
    AFTER_LEARNER_EXECUTION = "after_learner_execution"
    
    # 特征交换阶段
    FEATURE_EXTRACTION = "feature_extraction"
    FEATURE_EXCHANGE = "feature_exchange"
    FEATURE_AGGREGATION = "feature_aggregation"
    FEATURE_DISTRIBUTION = "feature_distribution"
    
    # 完成阶段
    ALL_LEARNERS_COMPLETE = "all_learners_complete"
    MULTI_LEARNER_AGGREGATION = "multi_learner_aggregation"
    EXECUTION_SUMMARY = "execution_summary"
    
    # 错误处理
    LEARNER_ERROR = "learner_error"
    COORDINATION_ERROR = "coordination_error"


class MultiLearnerHook(Hook, ABC):
    """
    多Learner Hook抽象基类
    
    扩展基础Hook，提供多Learner场景的通用功能。
    """
    
    def __init__(self, phase: str, priority: int = HookPriority.NORMAL.value,
                 name: Optional[str] = None, enabled: bool = True):
        super().__init__(phase, priority, name, enabled)
        
        # 多Learner相关属性
        self.target_learners: Optional[List[str]] = None
        self.feature_dependencies: Dict[str, List[str]] = {}
        
    def set_target_learners(self, learner_ids: List[str]) -> None:
        """设置目标learner"""
        self.target_learners = learner_ids
    
    def add_feature_dependency(self, source_learner: str, target_learners: List[str]) -> None:
        """添加特征依赖关系"""
        self.feature_dependencies[source_learner] = target_learners
    
    def validate_multi_learner_context(self, context: ExecutionContext, **kwargs) -> bool:
        """验证多Learner上下文"""
        # 检查是否有learner信息
        learners = kwargs.get('learners', {})
        if not learners:
            logger.warning(f"No learners found in context for hook {self.name}")
            return False
        
        # 检查目标learner是否存在
        if self.target_learners:
            available_learners = set(learners.keys()) if isinstance(learners, dict) else set(l.learner_id for l in learners)
            missing_learners = set(self.target_learners) - available_learners
            if missing_learners:
                logger.warning(f"Target learners not found: {missing_learners}")
                return False
        
        return True
    
    def should_execute(self, context: ExecutionContext, **kwargs) -> bool:
        """重写执行判断逻辑"""
        if not super().should_execute(context, **kwargs):
            return False
        
        return self.validate_multi_learner_context(context, **kwargs)


class LearnerCoordinationHook(MultiLearnerHook):
    """Learner协调Hook - 处理多learner协调逻辑"""
    
    def __init__(self, coordination_strategy: str = "adaptive", **kwargs):
        super().__init__(MultiLearnerHookPhase.EXECUTION_PLANNING.value, **kwargs)
        self.coordination_strategy = coordination_strategy
    
    def execute(self, context: ExecutionContext, **kwargs) -> Any:
        """执行协调逻辑"""
        learners = kwargs.get('learners', {})
        execution_plan = kwargs.get('execution_plan')
        
        logger.debug(f"Executing learner coordination with strategy: {self.coordination_strategy}")
        
        # 根据策略优化执行计划
        if self.coordination_strategy == "priority_based":
            return self._optimize_by_priority(learners, execution_plan)
        elif self.coordination_strategy == "resource_based":
            return self._optimize_by_resources(learners, execution_plan)
        elif self.coordination_strategy == "dependency_based":
            return self._optimize_by_dependencies(learners, execution_plan)
        else:  # adaptive
            return self._adaptive_optimization(learners, execution_plan)
    
    def _optimize_by_priority(self, learners, execution_plan):
        """基于优先级优化"""
        logger.debug("Optimizing execution plan by priority")
        return {"optimization": "priority_based", "changes": []}
    
    def _optimize_by_resources(self, learners, execution_plan):
        """基于资源优化"""
        logger.debug("Optimizing execution plan by resources")
        return {"optimization": "resource_based", "changes": []}
    
    def _optimize_by_dependencies(self, learners, execution_plan):
        """基于依赖关系优化"""
        logger.debug("Optimizing execution plan by dependencies")
        return {"optimization": "dependency_based", "changes": []}
    
    def _adaptive_optimization(self, learners, execution_plan):
        """自适应优化"""
        logger.debug("Applying adaptive optimization")
        return {"optimization": "adaptive", "changes": []}


class FeatureExchangeHook(MultiLearnerHook):
    """特征交换Hook - 处理learner间特征共享"""
    
    def __init__(self, exchange_strategy: str = "direct", **kwargs):
        super().__init__(MultiLearnerHookPhase.FEATURE_EXCHANGE.value, **kwargs)
        self.exchange_strategy = exchange_strategy
        self.feature_cache = {}
    
    def execute(self, context: ExecutionContext, **kwargs) -> Any:
        """执行特征交换"""
        learners = kwargs.get('learners', {})
        current_learner = kwargs.get('current_learner')
        execution_results = kwargs.get('execution_results', {})
        
        logger.debug(f"Executing feature exchange with strategy: {self.exchange_strategy}")
        
        # 收集特征
        features = self._collect_features(execution_results)
        
        # 执行特征交换
        if self.exchange_strategy == "direct":
            return self._direct_exchange(features, learners, context)
        elif self.exchange_strategy == "aggregated":
            return self._aggregated_exchange(features, learners, context)
        elif self.exchange_strategy == "selective":
            return self._selective_exchange(features, learners, context)
        else:
            return self._custom_exchange(features, learners, context)
    
    def _collect_features(self, execution_results: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """收集learner输出特征"""
        features = {}
        for learner_id, result in execution_results.items():
            if hasattr(result, 'output_features') and result.output_features:
                features[learner_id] = result.output_features
        return features
    
    def _direct_exchange(self, features: Dict[str, torch.Tensor], learners, context) -> Dict[str, Any]:
        """直接特征交换"""
        logger.debug("Performing direct feature exchange")
        
        exchanges = []
        for source_learner, source_features in features.items():
            for target_learner in self.feature_dependencies.get(source_learner, []):
                if target_learner in learners:
                    context.share_features(source_learner, source_features)
                    exchanges.append({
                        'source': source_learner,
                        'target': target_learner,
                        'features': list(source_features.keys())
                    })
        
        return {"exchange_type": "direct", "exchanges": exchanges}
    
    def _aggregated_exchange(self, features: Dict[str, torch.Tensor], learners, context) -> Dict[str, Any]:
        """聚合特征交换"""
        logger.debug("Performing aggregated feature exchange")
        
        # 聚合同类型特征
        aggregated_features = {}
        for learner_id, learner_features in features.items():
            for feature_name, feature_tensor in learner_features.items():
                if feature_name not in aggregated_features:
                    aggregated_features[feature_name] = []
                aggregated_features[feature_name].append(feature_tensor)
        
        # 计算平均特征
        avg_features = {}
        for feature_name, feature_list in aggregated_features.items():
            if feature_list:
                stacked = torch.stack(feature_list)
                avg_features[feature_name] = torch.mean(stacked, dim=0)
        
        # 分发聚合特征
        context.share_features("aggregated", avg_features)
        
        return {"exchange_type": "aggregated", "aggregated_features": list(avg_features.keys())}
    
    def _selective_exchange(self, features: Dict[str, torch.Tensor], learners, context) -> Dict[str, Any]:
        """选择性特征交换"""
        logger.debug("Performing selective feature exchange")
        
        # 根据特征质量选择最佳特征
        selected_features = {}
        for feature_name in set().union(*(f.keys() for f in features.values())):
            best_learner = None
            best_quality = -1
            
            for learner_id, learner_features in features.items():
                if feature_name in learner_features:
                    # 简单的质量评估（可以根据需要改进）
                    quality = self._evaluate_feature_quality(learner_features[feature_name])
                    if quality > best_quality:
                        best_quality = quality
                        best_learner = learner_id
            
            if best_learner:
                selected_features[feature_name] = features[best_learner][feature_name]
        
        context.share_features("selected", selected_features)
        
        return {"exchange_type": "selective", "selected_features": list(selected_features.keys())}
    
    def _custom_exchange(self, features: Dict[str, torch.Tensor], learners, context) -> Dict[str, Any]:
        """自定义特征交换"""
        logger.debug("Performing custom feature exchange")
        # 用户可以重写此方法实现自定义逻辑
        return {"exchange_type": "custom", "message": "Override this method for custom logic"}
    
    def _evaluate_feature_quality(self, feature_tensor: torch.Tensor) -> float:
        """评估特征质量"""
        # 简单实现：使用方差作为质量指标
        return torch.var(feature_tensor).item()


class MultiLearnerAggregationHook(MultiLearnerHook):
    """多Learner聚合Hook - 处理多learner结果聚合"""
    
    def __init__(self, aggregation_method: str = "weighted_average", **kwargs):
        super().__init__(MultiLearnerHookPhase.MULTI_LEARNER_AGGREGATION.value, **kwargs)
        self.aggregation_method = aggregation_method
    
    def execute(self, context: ExecutionContext, **kwargs) -> Any:
        """执行多learner聚合"""
        execution_results = kwargs.get('execution_results', {})
        learners = kwargs.get('learners', {})
        
        logger.debug(f"Executing multi-learner aggregation with method: {self.aggregation_method}")
        
        if self.aggregation_method == "weighted_average":
            return self._weighted_average_aggregation(execution_results, learners)
        elif self.aggregation_method == "priority_based":
            return self._priority_based_aggregation(execution_results, learners)
        elif self.aggregation_method == "performance_based":
            return self._performance_based_aggregation(execution_results, learners)
        else:
            return self._simple_average_aggregation(execution_results, learners)
    
    def _weighted_average_aggregation(self, results, learners) -> Dict[str, Any]:
        """加权平均聚合"""
        logger.debug("Performing weighted average aggregation")
        
        aggregated_params = {}
        total_weight = 0
        
        for learner_id, result in results.items():
            if result.success and hasattr(result, 'parameters'):
                weight = self._calculate_learner_weight(learner_id, result, learners)
                
                if not aggregated_params:
                    # 初始化
                    aggregated_params = {
                        name: param.clone() * weight 
                        for name, param in result.parameters.items()
                    }
                else:
                    # 累加
                    for name, param in result.parameters.items():
                        if name in aggregated_params:
                            aggregated_params[name] += param * weight
                
                total_weight += weight
        
        # 归一化
        if total_weight > 0:
            for name in aggregated_params:
                aggregated_params[name] /= total_weight
        
        return {
            "aggregation_method": "weighted_average",
            "aggregated_parameters": aggregated_params,
            "total_weight": total_weight,
            "participating_learners": len(results)
        }
    
    def _priority_based_aggregation(self, results, learners) -> Dict[str, Any]:
        """基于优先级聚合"""
        logger.debug("Performing priority-based aggregation")
        
        # 按优先级排序
        sorted_results = sorted(
            results.items(),
            key=lambda x: learners.get(x[0], {}).get('priority', 50)
        )
        
        # 使用最高优先级的成功结果
        for learner_id, result in sorted_results:
            if result.success:
                return {
                    "aggregation_method": "priority_based",
                    "selected_learner": learner_id,
                    "result": result
                }
        
        return {"aggregation_method": "priority_based", "selected_learner": None}
    
    def _performance_based_aggregation(self, results, learners) -> Dict[str, Any]:
        """基于性能聚合"""
        logger.debug("Performing performance-based aggregation")
        
        best_learner = None
        best_performance = -float('inf')
        
        for learner_id, result in results.items():
            if result.success and result.metrics:
                # 使用准确率或其他性能指标
                performance = result.metrics.get('accuracy', result.metrics.get('loss', 0))
                if performance > best_performance:
                    best_performance = performance
                    best_learner = learner_id
        
        return {
            "aggregation_method": "performance_based",
            "best_learner": best_learner,
            "best_performance": best_performance
        }
    
    def _simple_average_aggregation(self, results, learners) -> Dict[str, Any]:
        """简单平均聚合"""
        logger.debug("Performing simple average aggregation")
        
        aggregated_params = {}
        count = 0
        
        for learner_id, result in results.items():
            if result.success and hasattr(result, 'parameters'):
                if not aggregated_params:
                    aggregated_params = {
                        name: param.clone() 
                        for name, param in result.parameters.items()
                    }
                else:
                    for name, param in result.parameters.items():
                        if name in aggregated_params:
                            aggregated_params[name] += param
                
                count += 1
        
        # 计算平均值
        if count > 0:
            for name in aggregated_params:
                aggregated_params[name] /= count
        
        return {
            "aggregation_method": "simple_average",
            "aggregated_parameters": aggregated_params,
            "participating_learners": count
        }
    
    def _calculate_learner_weight(self, learner_id: str, result, learners) -> float:
        """计算learner权重"""
        # 简单实现：基于训练样本数量
        if hasattr(result, 'num_samples'):
            return float(result.num_samples)
        
        # 基于性能
        if result.metrics and 'accuracy' in result.metrics:
            return result.metrics['accuracy']
        
        # 默认权重
        return 1.0


class ResourceMonitoringHook(MultiLearnerHook):
    """资源监控Hook - 监控多learner执行的资源使用"""
    
    def __init__(self, monitoring_interval: float = 1.0, **kwargs):
        super().__init__(MultiLearnerHookPhase.BEFORE_LEARNER_EXECUTION.value, **kwargs)
        self.monitoring_interval = monitoring_interval
        self.resource_history = []
    
    def execute(self, context: ExecutionContext, **kwargs) -> Any:
        """执行资源监控"""
        current_learner = kwargs.get('current_learner')
        all_learners = kwargs.get('learners', {})
        
        # 获取资源使用情况
        resource_usage = self._get_resource_usage()
        
        # 记录资源使用历史
        self.resource_history.append({
            'timestamp': time.time(),
            'learner_id': current_learner,
            'resource_usage': resource_usage
        })
        
        # 检查资源警告
        warnings = self._check_resource_warnings(resource_usage)
        if warnings:
            logger.warning(f"Resource warnings for learner {current_learner}: {warnings}")
        
        # 更新上下文
        context.log_metric("memory_usage", resource_usage.get('memory_mb', 0))
        context.log_metric("gpu_usage", resource_usage.get('gpu_mb', 0))
        
        return {
            "resource_usage": resource_usage,
            "warnings": warnings,
            "monitoring_active": True
        }
    
    def _get_resource_usage(self) -> Dict[str, float]:
        """获取当前资源使用情况"""
        try:
            import psutil
            
            # CPU和内存使用
            memory_usage = psutil.virtual_memory().used / 1024 / 1024  # MB
            cpu_usage = psutil.cpu_percent()
            
            resource_usage = {
                'memory_mb': memory_usage,
                'cpu_percent': cpu_usage
            }
            
            # GPU使用（如果可用）
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                resource_usage['gpu_mb'] = gpu_memory
            
            return resource_usage
            
        except ImportError:
            logger.warning("psutil not available, using simplified resource monitoring")
            return {'memory_mb': 0, 'cpu_percent': 0}
    
    def _check_resource_warnings(self, resource_usage: Dict[str, float]) -> List[str]:
        """检查资源警告"""
        warnings = []
        
        # 内存警告
        if resource_usage.get('memory_mb', 0) > 3072:  # 3GB
            warnings.append("High memory usage")
        
        # CPU警告
        if resource_usage.get('cpu_percent', 0) > 90:
            warnings.append("High CPU usage")
        
        # GPU警告
        if resource_usage.get('gpu_mb', 0) > 1536:  # 1.5GB
            warnings.append("High GPU memory usage")
        
        return warnings
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """获取资源使用摘要"""
        if not self.resource_history:
            return {"message": "No resource data available"}
        
        # 计算平均值
        avg_memory = sum(r['resource_usage'].get('memory_mb', 0) for r in self.resource_history) / len(self.resource_history)
        avg_cpu = sum(r['resource_usage'].get('cpu_percent', 0) for r in self.resource_history) / len(self.resource_history)
        
        return {
            "total_samples": len(self.resource_history),
            "avg_memory_mb": avg_memory,
            "avg_cpu_percent": avg_cpu,
            "peak_memory_mb": max(r['resource_usage'].get('memory_mb', 0) for r in self.resource_history),
            "peak_cpu_percent": max(r['resource_usage'].get('cpu_percent', 0) for r in self.resource_history)
        }


# 便利函数：创建常用的多Learner Hook

def create_knowledge_distillation_hook(teacher_learner: str, student_learners: List[str], 
                                     temperature: float = 3.0) -> MultiLearnerHook:
    """创建知识蒸馏Hook"""
    
    class KnowledgeDistillationHook(FeatureExchangeHook):
        def __init__(self):
            super().__init__(exchange_strategy="selective", 
                           name="knowledge_distillation")
            self.teacher_learner = teacher_learner
            self.student_learners = student_learners
            self.temperature = temperature
        
        def execute(self, context: ExecutionContext, **kwargs) -> Any:
            logger.debug(f"Performing knowledge distillation from {self.teacher_learner}")
            
            # 获取教师模型的输出
            teacher_features = context.get_shared_features("student", self.teacher_learner)
            
            if teacher_features:
                # 对学生模型应用知识蒸馏
                for student_id in self.student_learners:
                    context.share_features(f"teacher_for_{student_id}", teacher_features)
                
                return {
                    "distillation_type": "knowledge_distillation",
                    "teacher": self.teacher_learner,
                    "students": self.student_learners,
                    "temperature": self.temperature
                }
            
            return {"message": "No teacher features available"}
    
    return KnowledgeDistillationHook()


def create_ensemble_hook(learner_ids: List[str], ensemble_method: str = "voting") -> MultiLearnerHook:
    """创建集成Hook"""
    
    class EnsembleHook(MultiLearnerAggregationHook):
        def __init__(self):
            super().__init__(aggregation_method=ensemble_method, 
                           name="ensemble_aggregation")
            self.learner_ids = learner_ids
        
        def execute(self, context: ExecutionContext, **kwargs) -> Any:
            logger.debug(f"Performing ensemble aggregation with {len(self.learner_ids)} learners")
            
            execution_results = kwargs.get('execution_results', {})
            
            # 过滤指定的learner结果
            filtered_results = {
                lid: result for lid, result in execution_results.items() 
                if lid in self.learner_ids
            }
            
            if ensemble_method == "voting":
                return self._voting_ensemble(filtered_results)
            elif ensemble_method == "stacking":
                return self._stacking_ensemble(filtered_results)
            else:
                return super().execute(context, execution_results=filtered_results, **kwargs)
        
        def _voting_ensemble(self, results):
            """投票集成"""
            logger.debug("Performing voting ensemble")
            return {"ensemble_method": "voting", "participating_learners": len(results)}
        
        def _stacking_ensemble(self, results):
            """堆叠集成"""
            logger.debug("Performing stacking ensemble")
            return {"ensemble_method": "stacking", "participating_learners": len(results)}
    
    return EnsembleHook()


def create_adaptive_execution_hook(resource_threshold: float = 0.8) -> MultiLearnerHook:
    """创建自适应执行Hook"""
    
    class AdaptiveExecutionHook(LearnerCoordinationHook):
        def __init__(self):
            super().__init__(coordination_strategy="adaptive", 
                           name="adaptive_execution")
            self.resource_threshold = resource_threshold
        
        def execute(self, context: ExecutionContext, **kwargs) -> Any:
            logger.debug("Performing adaptive execution optimization")
            
            # 获取当前资源使用情况
            resource_usage = self._get_current_resource_usage()
            
            # 根据资源使用情况调整执行策略
            if resource_usage > self.resource_threshold:
                return self._switch_to_sequential_execution(kwargs)
            else:
                return self._optimize_parallel_execution(kwargs)
        
        def _get_current_resource_usage(self) -> float:
            """获取当前资源使用率"""
            # 简化实现
            return 0.7  # 示例值
        
        def _switch_to_sequential_execution(self, kwargs):
            """切换到串行执行"""
            logger.debug("Switching to sequential execution due to high resource usage")
            return {"execution_mode": "sequential", "reason": "resource_constraint"}
        
        def _optimize_parallel_execution(self, kwargs):
            """优化并行执行"""
            logger.debug("Optimizing parallel execution")
            return {"execution_mode": "parallel", "reason": "resource_available"}
    
    return AdaptiveExecutionHook()