# fedcl/core/checkpoint_hook.py
"""
CheckpointHook - 检查点保存钩子

提供模型和实验状态的自动保存功能。
支持可配置的保存频率、检查点管理和状态恢复。
"""

import json
import time
import shutil
from typing import Dict, Any, Optional, List
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from loguru import logger

from .hook import Hook, HookPhase
from .execution_context import ExecutionContext
from ..exceptions import HookExecutionError, ConfigurationError


class CheckpointHook(Hook):
    """
    检查点保存钩子
    
    在训练过程中自动保存模型检查点和实验状态，包括：
    - 模型权重和架构
    - 优化器状态
    - 学习率调度器状态
    - 实验配置和元数据
    - 训练进度和统计信息
    
    支持可配置的保存频率、检查点清理和压缩。
    """
    
    def __init__(
        self, 
        phase: str, 
        checkpoint_config: DictConfig,
        priority: int = 0,
        name: Optional[str] = None,
        enabled: bool = True
    ) -> None:
        """
        初始化检查点保存钩子
        
        Args:
            phase: 钩子执行阶段
            checkpoint_config: 检查点配置
            priority: 执行优先级
            name: 钩子名称
            enabled: 是否启用
            
        Raises:
            ConfigurationError: 配置无效时抛出
        """
        super().__init__(phase, priority, name, enabled)
        
        if not isinstance(checkpoint_config, DictConfig):
            raise ConfigurationError("checkpoint_config must be a DictConfig object")
        
        self.checkpoint_config = checkpoint_config
        
        # 保存配置
        self.save_frequency = checkpoint_config.get('save_frequency', 1)
        self.save_model = checkpoint_config.get('save_model', True)
        self.save_optimizer = checkpoint_config.get('save_optimizer', True)
        self.save_scheduler = checkpoint_config.get('save_scheduler', True)
        self.save_experiment_state_enabled = checkpoint_config.get('save_experiment_state', True)
        
        # 路径配置
        self.checkpoint_dir = Path(checkpoint_config.get('checkpoint_dir', './checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 文件命名配置
        self.naming_pattern = checkpoint_config.get('naming_pattern', 'checkpoint_round_{round}_epoch_{epoch}')
        self.include_timestamp = checkpoint_config.get('include_timestamp', True)
        
        # 管理配置
        self.max_checkpoints = checkpoint_config.get('max_checkpoints', 5)
        self.compress_checkpoints = checkpoint_config.get('compress', False)
        self.keep_best_only = checkpoint_config.get('keep_best_only', False)
        self.best_metric = checkpoint_config.get('best_metric', 'accuracy')
        self.best_mode = checkpoint_config.get('best_mode', 'max')  # max 或 min
        
        # 状态跟踪
        self.saved_checkpoints: List[Dict[str, Any]] = []
        self.best_metric_value: Optional[float] = None
        self.best_checkpoint_path: Optional[Path] = None
        self.last_save_time = 0.0
        
        logger.debug(f"CheckpointHook初始化完成 - phase: {phase}, dir: {self.checkpoint_dir}")
    
    def execute(self, context: ExecutionContext, **kwargs) -> None:
        """
        执行检查点保存
        
        根据配置和当前状态决定是否保存检查点。
        
        Args:
            context: 执行上下文
            **kwargs: 阶段特定参数
                - model: 需要保存的模型
                - optimizer: 优化器对象
                - scheduler: 学习率调度器
                - metrics: 当前度量值
                
        Raises:
            HookExecutionError: 执行失败时抛出
        """
        try:
            if not self.should_save_checkpoint(context, **kwargs):
                return
            
            # 生成检查点路径
            checkpoint_path = self._generate_checkpoint_path(context)
            
            # 创建检查点目录
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            
            # 保存模型检查点
            if 'model' in kwargs and self.save_model:
                model_path = checkpoint_path / 'model.pth'
                self.save_model_checkpoint(kwargs['model'], model_path)
            
            # 保存优化器状态
            if 'optimizer' in kwargs and self.save_optimizer:
                optimizer_path = checkpoint_path / 'optimizer.pth'
                self._save_optimizer_state(kwargs['optimizer'], optimizer_path)
            
            # 保存调度器状态
            if 'scheduler' in kwargs and self.save_scheduler:
                scheduler_path = checkpoint_path / 'scheduler.pth'
                self._save_scheduler_state(kwargs['scheduler'], scheduler_path)
            
            # 保存实验状态
            if self.save_experiment_state_enabled:
                state_path = checkpoint_path / 'experiment_state.json'
                self.save_experiment_state(context, state_path)
            
            # 记录检查点信息
            checkpoint_info = self._create_checkpoint_info(context, checkpoint_path, **kwargs)
            self.saved_checkpoints.append(checkpoint_info)
            
            # 更新最佳检查点
            if 'metrics' in kwargs:
                self._update_best_checkpoint(checkpoint_info, kwargs['metrics'])
            
            # 清理旧检查点
            if self.max_checkpoints > 0:
                self.cleanup_old_checkpoints(self.max_checkpoints)
            
            self.last_save_time = time.time()
            
            logger.debug(f"检查点已保存: {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"CheckpointHook执行失败: {e}")
            raise HookExecutionError(f"检查点保存失败: {e}")
    
    def save_model_checkpoint(self, model, path: Path) -> None:
        """
        保存模型检查点
        
        Args:
            model: 需要保存的模型对象
            path: 保存路径
        """
        try:
            import torch
            
            if hasattr(model, 'state_dict'):
                # PyTorch模型
                checkpoint_data = {
                    'model_state_dict': model.state_dict(),
                    'model_class': model.__class__.__name__,
                    'timestamp': time.time()
                }
                
                # 如果模型有配置信息，也保存
                if hasattr(model, 'config'):
                    checkpoint_data['model_config'] = model.config
                
                torch.save(checkpoint_data, path)
                
            else:
                # 其他类型的模型
                logger.warning(f"未知模型类型: {type(model)}, 尝试使用pickle保存")
                import pickle
                with open(path, 'wb') as f:
                    pickle.dump(model, f)
            
            logger.debug(f"模型检查点已保存: {path}")
            
        except Exception as e:
            logger.error(f"保存模型检查点失败: {e}")
            raise HookExecutionError(f"保存模型检查点失败: {e}")
    
    def save_experiment_state(self, context: ExecutionContext, path: Path) -> None:
        """
        保存实验状态
        
        Args:
            context: 执行上下文
            path: 保存路径
        """
        try:
            # 收集实验状态信息
            experiment_state = {
                'experiment_id': context.experiment_id,
                'timestamp': time.time(),
                'config': OmegaConf.to_container(context.config, resolve=True),
                'global_state': self._serialize_state(context._global_state),
                'metrics_summary': self._get_metrics_summary(context),
                'training_progress': {
                    'current_round': context.get_state('current_round', 'global'),
                    'current_epoch': context.get_state('current_epoch', 'global'),
                    'current_batch': context.get_state('current_batch', 'global'),
                    'current_task_id': context.get_state('current_task_id', 'global'),
                },
                'hook_stats': self.get_execution_stats()
            }
            
            # 保存到JSON文件
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(experiment_state, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"实验状态已保存: {path}")
            
        except Exception as e:
            logger.error(f"保存实验状态失败: {e}")
            raise HookExecutionError(f"保存实验状态失败: {e}")
    
    def should_save_checkpoint(self, context: ExecutionContext, **kwargs) -> bool:
        """
        判断是否应该保存检查点
        
        Args:
            context: 执行上下文
            **kwargs: 额外参数
            
        Returns:
            bool: 是否应该保存
        """
        if not super().should_execute(context, **kwargs):
            return False
        
        # 检查保存频率
        if self.phase in [HookPhase.AFTER_ROUND.value, HookPhase.AFTER_EPOCH.value]:
            if self.phase == HookPhase.AFTER_ROUND.value:
                current_step = context.get_state('current_round', 'global') or 0
            else:
                current_step = context.get_state('current_epoch', 'global') or 0
                
            if current_step % self.save_frequency != 0:
                return False
        
        # 检查是否有需要保存的内容
        has_content = False
        if 'model' in kwargs and self.save_model:
            has_content = True
        if self.save_experiment_state_enabled:
            has_content = True
            
        return has_content
    
    def cleanup_old_checkpoints(self, max_checkpoints: int) -> None:
        """
        清理旧检查点
        
        Args:
            max_checkpoints: 最大保留检查点数量
        """
        try:
            if self.keep_best_only and self.best_checkpoint_path:
                # 只保留最佳检查点
                to_remove = [cp for cp in self.saved_checkpoints 
                           if cp['path'] != str(self.best_checkpoint_path)]
            else:
                # 按时间排序，保留最新的检查点
                sorted_checkpoints = sorted(self.saved_checkpoints, 
                                          key=lambda x: x['timestamp'], 
                                          reverse=True)
                to_remove = sorted_checkpoints[max_checkpoints:]
            
            # 删除旧检查点
            for checkpoint_info in to_remove:
                checkpoint_path = Path(checkpoint_info['path'])
                if checkpoint_path.exists():
                    if checkpoint_path.is_dir():
                        shutil.rmtree(checkpoint_path)
                    else:
                        checkpoint_path.unlink()
                    
                    logger.debug(f"已删除旧检查点: {checkpoint_path}")
                
                # 从记录中移除
                if checkpoint_info in self.saved_checkpoints:
                    self.saved_checkpoints.remove(checkpoint_info)
            
            if to_remove:
                logger.debug(f"已清理 {len(to_remove)} 个旧检查点")
                
        except Exception as e:
            logger.error(f"清理旧检查点失败: {e}")
    
    def get_checkpoint_info(self, checkpoint_path: Path) -> Dict[str, Any]:
        """
        获取检查点信息
        
        Args:
            checkpoint_path: 检查点路径
            
        Returns:
            Dict[str, Any]: 检查点信息
        """
        try:
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"检查点不存在: {checkpoint_path}")
            
            info = {
                'path': str(checkpoint_path),
                'size': self._get_directory_size(checkpoint_path) if checkpoint_path.is_dir() else checkpoint_path.stat().st_size,
                'created_time': checkpoint_path.stat().st_ctime,
                'modified_time': checkpoint_path.stat().st_mtime,
                'exists': True
            }
            
            # 尝试读取实验状态文件获取更多信息
            state_file = checkpoint_path / 'experiment_state.json' if checkpoint_path.is_dir() else None
            if state_file and state_file.exists():
                with open(state_file, 'r', encoding='utf-8') as f:
                    state_data = json.load(f)
                    info.update({
                        'experiment_id': state_data.get('experiment_id'),
                        'training_progress': state_data.get('training_progress'),
                        'metrics_summary': state_data.get('metrics_summary')
                    })
            
            return info
            
        except Exception as e:
            logger.error(f"获取检查点信息失败: {e}")
            return {
                'path': str(checkpoint_path),
                'exists': False,
                'error': str(e)
            }
    
    def load_checkpoint(self, checkpoint_path: Path, context: ExecutionContext) -> Dict[str, Any]:
        """
        加载检查点
        
        Args:
            checkpoint_path: 检查点路径
            context: 执行上下文
            
        Returns:
            Dict[str, Any]: 加载的检查点数据
        """
        try:
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"检查点不存在: {checkpoint_path}")
            
            loaded_data = {}
            
            # 加载模型
            model_path = checkpoint_path / 'model.pth'
            if model_path.exists():
                import torch
                model_checkpoint = torch.load(model_path, map_location='cpu')
                loaded_data['model'] = model_checkpoint
            
            # 加载优化器状态
            optimizer_path = checkpoint_path / 'optimizer.pth'
            if optimizer_path.exists():
                import torch
                optimizer_state = torch.load(optimizer_path, map_location='cpu')
                loaded_data['optimizer'] = optimizer_state
            
            # 加载调度器状态
            scheduler_path = checkpoint_path / 'scheduler.pth'
            if scheduler_path.exists():
                import torch
                scheduler_state = torch.load(scheduler_path, map_location='cpu')
                loaded_data['scheduler'] = scheduler_state
            
            # 加载实验状态
            state_path = checkpoint_path / 'experiment_state.json'
            if state_path.exists():
                with open(state_path, 'r', encoding='utf-8') as f:
                    experiment_state = json.load(f)
                    loaded_data['experiment_state'] = experiment_state
                    
                    # 恢复训练进度到执行上下文
                    progress = experiment_state.get('training_progress', {})
                    for key, value in progress.items():
                        if value is not None:
                            context.set_state(key, value, 'global')
            
            logger.debug(f"检查点加载完成: {checkpoint_path}")
            return loaded_data
            
        except Exception as e:
            logger.error(f"加载检查点失败: {e}")
            raise HookExecutionError(f"加载检查点失败: {e}")
    
    def _generate_checkpoint_path(self, context: ExecutionContext) -> Path:
        """生成检查点保存路径"""
        # 获取当前状态
        current_round = context.get_state('current_round', 'global') or 0
        current_epoch = context.get_state('current_epoch', 'global') or 0
        current_task = context.get_state('current_task_id', 'global') or 0
        
        # 格式化文件名
        checkpoint_name = self.naming_pattern.format(
            round=current_round,
            epoch=current_epoch,
            task=current_task
        )
        
        # 添加时间戳
        if self.include_timestamp:
            timestamp = int(time.time())
            checkpoint_name += f"_{timestamp}"
        
        return self.checkpoint_dir / checkpoint_name
    
    def _save_optimizer_state(self, optimizer, path: Path) -> None:
        """保存优化器状态"""
        try:
            import torch
            if hasattr(optimizer, 'state_dict'):
                torch.save(optimizer.state_dict(), path)
                logger.debug(f"优化器状态已保存: {path}")
            else:
                logger.warning("优化器不支持state_dict方法")
        except Exception as e:
            logger.error(f"保存优化器状态失败: {e}")
    
    def _save_scheduler_state(self, scheduler, path: Path) -> None:
        """保存调度器状态"""
        try:
            import torch
            if hasattr(scheduler, 'state_dict'):
                torch.save(scheduler.state_dict(), path)
                logger.debug(f"调度器状态已保存: {path}")
            else:
                logger.warning("调度器不支持state_dict方法")
        except Exception as e:
            logger.error(f"保存调度器状态失败: {e}")
    
    def _create_checkpoint_info(self, context: ExecutionContext, checkpoint_path: Path, **kwargs) -> Dict[str, Any]:
        """创建检查点信息记录"""
        info = {
            'path': str(checkpoint_path),
            'timestamp': time.time(),
            'round': context.get_state('current_round', 'global'),
            'epoch': context.get_state('current_epoch', 'global'),
            'task_id': context.get_state('current_task_id', 'global'),
            'size': self._get_directory_size(checkpoint_path)
        }
        
        # 添加度量信息
        if 'metrics' in kwargs:
            info['metrics'] = dict(kwargs['metrics'])
        
        return info
    
    def _update_best_checkpoint(self, checkpoint_info: Dict[str, Any], metrics: Dict[str, float]) -> None:
        """更新最佳检查点"""
        if self.best_metric not in metrics:
            return
        
        current_value = metrics[self.best_metric]
        
        if self.best_metric_value is None:
            # 第一个检查点
            self.best_metric_value = current_value
            self.best_checkpoint_path = Path(checkpoint_info['path'])
        else:
            # 比较度量值
            is_better = (
                (self.best_mode == 'max' and current_value > self.best_metric_value) or
                (self.best_mode == 'min' and current_value < self.best_metric_value)
            )
            
            if is_better:
                self.best_metric_value = current_value
                self.best_checkpoint_path = Path(checkpoint_info['path'])
                logger.debug(f"发现更好的检查点: {self.best_metric}={current_value}")
    
    def _serialize_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """序列化状态数据"""
        serialized = {}
        for key, value in state.items():
            try:
                # 尝试JSON序列化以检查是否可序列化
                json.dumps(value)
                serialized[key] = value
            except (TypeError, ValueError):
                # 不可序列化的对象转换为字符串表示
                serialized[key] = str(value)
        return serialized
    
    def _get_metrics_summary(self, context: ExecutionContext) -> Dict[str, Any]:
        """获取度量摘要"""
        try:
            # 从执行上下文获取度量信息
            # 这里简化实现，实际可以根据context的度量系统来获取
            return {
                'total_metrics': len(getattr(context, '_metrics', {})),
                'last_updated': time.time()
            }
        except Exception:
            return {}
    
    def _get_directory_size(self, directory: Path) -> int:
        """计算目录大小"""
        try:
            total_size = 0
            if directory.is_file():
                return directory.stat().st_size
            
            for file_path in directory.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            return total_size
        except Exception:
            return 0
    
    def get_checkpoint_summary(self) -> Dict[str, Any]:
        """获取检查点摘要"""
        return {
            'total_checkpoints': len(self.saved_checkpoints),
            'checkpoint_dir': str(self.checkpoint_dir),
            'best_checkpoint': str(self.best_checkpoint_path) if self.best_checkpoint_path else None,
            'best_metric_value': self.best_metric_value,
            'last_save_time': self.last_save_time,
            'total_size': sum(cp.get('size', 0) for cp in self.saved_checkpoints)
        }
    
    def cleanup(self) -> None:
        """清理资源"""
        try:
            # 输出检查点摘要
            summary = self.get_checkpoint_summary()
            logger.debug(f"CheckpointHook摘要: {summary}")
            
            super().cleanup()
            
        except Exception as e:
            logger.error(f"CheckpointHook清理失败: {e}")
