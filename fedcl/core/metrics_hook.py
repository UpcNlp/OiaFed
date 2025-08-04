# fedcl/core/metrics_hook.py
"""
MetricsHook - 度量记录钩子

提供训练、评估、系统和通信度量的自动记录功能。
支持可配置的记录频率、过滤条件和输出格式。
"""

import time
import psutil
from typing import Dict, Any, Optional, List, Set
from pathlib import Path
from omegaconf import DictConfig
from loguru import logger

from .hook import Hook, HookPhase
from .execution_context import ExecutionContext
from ..exceptions import HookExecutionError, ConfigurationError


class MetricsHook(Hook):
    """
    度量记录钩子
    
    在训练和评估过程中自动记录各种度量信息，包括：
    - 训练度量（损失、准确率等）
    - 评估度量（性能指标）
    - 系统度量（内存、CPU使用率等）
    - 通信度量（数据传输量、延迟等）
    
    支持可配置的记录频率、过滤条件和输出格式。
    """
    
    def __init__(
        self, 
        phase: str, 
        metrics_config: DictConfig,
        priority: int = 0, 
        name: Optional[str] = None,
        enabled: bool = True
    ) -> None:
        """
        初始化度量记录钩子
        
        Args:
            phase: 钩子执行阶段
            priority: 执行优先级
            metrics_config: 度量配置
            name: 钩子名称
            enabled: 是否启用
            
        Raises:
            ConfigurationError: 配置无效时抛出
        """
        super().__init__(phase, priority, name, enabled)
        
        if not isinstance(metrics_config, DictConfig):
            raise ConfigurationError("metrics_config must be a DictConfig object")
        
        self.metrics_config = metrics_config
        
        # 记录配置
        self.log_training_metrics_enabled = metrics_config.get('log_training', True)
        self.log_evaluation_metrics_enabled = metrics_config.get('log_evaluation', True)
        self.log_system_metrics_enabled = metrics_config.get('log_system', True)
        self.log_communication_metrics_enabled = metrics_config.get('log_communication', True)
        
        # 记录频率配置
        self.training_log_frequency = metrics_config.get('training_frequency', 1)
        self.evaluation_log_frequency = metrics_config.get('evaluation_frequency', 1)
        self.system_log_frequency = metrics_config.get('system_frequency', 10)
        self.communication_log_frequency = metrics_config.get('communication_frequency', 1)
        
        # 过滤配置
        self.metric_filters = set(metrics_config.get('metric_filters', []))
        self.excluded_metrics = set(metrics_config.get('excluded_metrics', []))
        
        # 输出配置
        self.output_format = metrics_config.get('output_format', 'context')  # context, file, both
        self.output_path = metrics_config.get('output_path', './metrics.log')
        
        # 统计信息
        self.metrics_logged_count = 0
        self.last_system_log_time = 0.0
        self.metric_history: Dict[str, List[float]] = {}
        
        # 系统监控
        self.process = psutil.Process()
        
        logger.debug(f"MetricsHook初始化完成 - phase: {phase}, config: {dict(metrics_config)}")
    
    def execute(self, context: ExecutionContext, **kwargs) -> None:
        """
        执行度量记录
        
        根据执行阶段和配置自动记录相应的度量信息。
        
        Args:
            context: 执行上下文
            **kwargs: 阶段特定参数
                - metrics: 训练度量字典
                - results: 评估结果字典
                - comm_stats: 通信统计字典
                
        Raises:
            HookExecutionError: 执行失败时抛出
        """
        try:
            if not self.should_execute(context, **kwargs):
                return
            
            # 根据阶段执行不同的度量记录
            if self.phase in [HookPhase.AFTER_BATCH.value, HookPhase.AFTER_EPOCH.value]:
                # 训练阶段 - 记录训练度量
                if 'metrics' in kwargs and self.log_training_metrics_enabled:
                    self.log_training_metrics(context, kwargs['metrics'])
                    
            elif self.phase == HookPhase.ON_EVALUATION.value:
                # 评估阶段 - 记录评估度量
                if 'results' in kwargs and self.log_evaluation_metrics_enabled:
                    self.log_evaluation_metrics(context, kwargs['results'])
                    
            elif self.phase == HookPhase.ON_AGGREGATION.value:
                # 聚合阶段 - 记录通信度量
                if 'comm_stats' in kwargs and self.log_communication_metrics_enabled:
                    self.log_communication_metrics(context, kwargs['comm_stats'])
            
            # 定期记录系统度量
            current_time = time.time()
            if (self.log_system_metrics_enabled and 
                current_time - self.last_system_log_time >= self.system_log_frequency):
                self.log_system_metrics(context)
                self.last_system_log_time = current_time
                
            self.metrics_logged_count += 1
            
        except Exception as e:
            logger.error(f"MetricsHook执行失败: {e}")
            raise HookExecutionError(f"度量记录失败: {e}")
    
    def log_training_metrics(self, context: ExecutionContext, metrics: Dict[str, float]) -> None:
        """
        记录训练度量
        
        Args:
            context: 执行上下文
            metrics: 训练度量字典，包含损失、准确率等指标
        """
        try:
            # 过滤度量
            filtered_metrics = self._filter_metrics(metrics)
            
            if not filtered_metrics:
                return
            
            # 获取当前轮次和步数信息
            current_round = context.get_state('current_round', 'global') or 0
            current_epoch = context.get_state('current_epoch', 'global') or 0
            current_batch = context.get_state('current_batch', 'global') or 0
            
            # 添加上下文信息
            enriched_metrics = {
                'round': current_round,
                'epoch': current_epoch,
                'batch': current_batch,
                'timestamp': time.time(),
                **filtered_metrics
            }
            
            # 记录到执行上下文
            for metric_name, value in filtered_metrics.items():
                context.log_metric(f'training_{metric_name}', value, step=current_batch)
                
                # 更新历史记录
                if metric_name not in self.metric_history:
                    self.metric_history[metric_name] = []
                self.metric_history[metric_name].append(value)
                
                # 限制历史记录长度
                if len(self.metric_history[metric_name]) > 1000:
                    self.metric_history[metric_name] = self.metric_history[metric_name][-1000:]
            
            # 输出日志
            self._output_metrics("training", enriched_metrics)
            
            logger.debug(f"训练度量已记录: {list(filtered_metrics.keys())}")
            
        except Exception as e:
            logger.error(f"记录训练度量失败: {e}")
            raise HookExecutionError(f"记录训练度量失败: {e}")
    
    def log_evaluation_metrics(self, context: ExecutionContext, results: Dict[str, float]) -> None:
        """
        记录评估度量
        
        Args:
            context: 执行上下文
            results: 评估结果字典，包含准确率、F1分数等指标
        """
        try:
            # 过滤度量
            filtered_results = self._filter_metrics(results)
            
            if not filtered_results:
                return
            
            # 获取当前轮次信息
            current_round = context.get_state('current_round', 'global') or 0
            current_task = context.get_state('current_task_id', 'global') or 0
            
            # 添加上下文信息
            enriched_results = {
                'round': current_round,
                'task_id': current_task,
                'timestamp': time.time(),
                **filtered_results
            }
            
            # 记录到执行上下文
            for metric_name, value in filtered_results.items():
                context.log_metric(f'evaluation_{metric_name}', value, step=current_round)
            
            # 输出日志
            self._output_metrics("evaluation", enriched_results)
            
            logger.debug(f"评估度量已记录: {list(filtered_results.keys())}")
            
        except Exception as e:
            logger.error(f"记录评估度量失败: {e}")
            raise HookExecutionError(f"记录评估度量失败: {e}")
    
    def log_system_metrics(self, context: ExecutionContext) -> None:
        """
        记录系统度量
        
        Args:
            context: 执行上下文
        """
        try:
            # 收集系统度量
            memory_info = self.process.memory_info()
            cpu_percent = self.process.cpu_percent()
            
            system_metrics = {
                'memory_rss_mb': memory_info.rss / 1024 / 1024,  # 物理内存使用(MB)
                'memory_vms_mb': memory_info.vms / 1024 / 1024,  # 虚拟内存使用(MB)
                'cpu_percent': cpu_percent,  # CPU使用率
                'num_threads': self.process.num_threads(),  # 线程数
                'timestamp': time.time()
            }
            
            # 如果有GPU，记录GPU度量
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory_allocated = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                    gpu_memory_reserved = torch.cuda.memory_reserved() / 1024 / 1024  # MB
                    
                    system_metrics.update({
                        'gpu_memory_allocated_mb': gpu_memory_allocated,
                        'gpu_memory_reserved_mb': gpu_memory_reserved,
                        'gpu_memory_cached_mb': gpu_memory_reserved - gpu_memory_allocated
                    })
            except ImportError:
                pass  # PyTorch未安装，跳过GPU度量
            
            # 记录到执行上下文
            for metric_name, value in system_metrics.items():
                if metric_name != 'timestamp':
                    context.log_metric(f'system_{metric_name}', value)
            
            # 输出日志
            self._output_metrics("system", system_metrics)
            
            logger.debug("系统度量已记录")
            
        except Exception as e:
            logger.error(f"记录系统度量失败: {e}")
            raise HookExecutionError(f"记录系统度量失败: {e}")
    
    def log_communication_metrics(self, context: ExecutionContext, comm_stats: Dict) -> None:
        """
        记录通信度量
        
        Args:
            context: 执行上下文
            comm_stats: 通信统计字典，包含传输量、延迟等信息
        """
        try:
            # 过滤和处理通信度量
            filtered_stats = self._filter_metrics(comm_stats)
            
            if not filtered_stats:
                return
            
            # 获取当前轮次信息
            current_round = context.get_state('current_round', 'global') or 0
            
            # 添加上下文信息
            enriched_stats = {
                'round': current_round,
                'timestamp': time.time(),
                **filtered_stats
            }
            
            # 记录到执行上下文
            for metric_name, value in filtered_stats.items():
                if isinstance(value, (int, float)):
                    context.log_metric(f'communication_{metric_name}', value, step=current_round)
            
            # 输出日志
            self._output_metrics("communication", enriched_stats)
            
            logger.debug(f"通信度量已记录: {list(filtered_stats.keys())}")
            
        except Exception as e:
            logger.error(f"记录通信度量失败: {e}")
            raise HookExecutionError(f"记录通信度量失败: {e}")
    
    def should_execute(self, context: ExecutionContext, **kwargs) -> bool:
        """
        判断是否应该记录度量
        
        Args:
            context: 执行上下文
            **kwargs: 额外参数
            
        Returns:
            bool: 是否应该执行
        """
        if not super().should_execute(context, **kwargs):
            return False
        
        # 检查记录频率
        if self.phase in [HookPhase.AFTER_BATCH.value, HookPhase.AFTER_EPOCH.value]:
            current_batch = context.get_state('current_batch', 'global') or 0
            if current_batch % self.training_log_frequency != 0:
                return False
                
        elif self.phase == HookPhase.ON_EVALUATION.value:
            current_round = context.get_state('current_round', 'global') or 0
            if current_round % self.evaluation_log_frequency != 0:
                return False
                
        elif self.phase == HookPhase.ON_AGGREGATION.value:
            current_round = context.get_state('current_round', 'global') or 0
            if current_round % self.communication_log_frequency != 0:
                return False
        
        return True
    
    def _filter_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        过滤度量数据
        
        Args:
            metrics: 原始度量字典
            
        Returns:
            过滤后的度量字典
        """
        filtered = {}
        
        for name, value in metrics.items():
            # 检查排除列表
            if name in self.excluded_metrics:
                continue
                
            # 检查过滤器
            if self.metric_filters and name not in self.metric_filters:
                continue
                
            # 只保留数值类型的度量
            if isinstance(value, (int, float, bool)):
                filtered[name] = float(value) if isinstance(value, bool) else value
        
        return filtered
    
    def _output_metrics(self, category: str, metrics: Dict[str, Any]) -> None:
        """
        输出度量数据
        
        Args:
            category: 度量类别
            metrics: 度量数据
        """
        try:
            if self.output_format in ['context', 'both']:
                # 已经在上面记录到context中
                pass
            
            if self.output_format in ['file', 'both']:
                # 写入文件
                output_path = Path(self.output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, 'a', encoding='utf-8') as f:
                    import json
                    log_entry = {
                        'category': category,
                        'metrics': metrics,
                        'logged_at': time.time()
                    }
                    f.write(json.dumps(log_entry) + '\n')
                    
        except Exception as e:
            logger.warning(f"输出度量数据失败: {e}")
    
    def get_metric_summary(self) -> Dict[str, Any]:
        """
        获取度量摘要信息
        
        Returns:
            Dict[str, Any]: 度量摘要
        """
        summary = {
            'total_metrics_logged': self.metrics_logged_count,
            'metric_types': list(self.metric_history.keys()),
            'metric_count_by_type': {k: len(v) for k, v in self.metric_history.items()}
        }
        
        # 计算每个度量的统计信息
        for metric_name, values in self.metric_history.items():
            if values:
                summary[f'{metric_name}_stats'] = {
                    'count': len(values),
                    'latest': values[-1],
                    'average': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values)
                }
        
        return summary
    
    def reset_metrics(self) -> None:
        """重置度量历史"""
        self.metric_history.clear()
        self.metrics_logged_count = 0
        self.last_system_log_time = 0.0
        logger.debug("度量历史已重置")
    
    def cleanup(self) -> None:
        """清理资源"""
        try:
            # 输出最终摘要
            summary = self.get_metric_summary()
            logger.debug(f"MetricsHook度量摘要: {summary}")
            
            # 清理历史数据
            self.reset_metrics()
            
            super().cleanup()
            
        except Exception as e:
            logger.error(f"MetricsHook清理失败: {e}")
