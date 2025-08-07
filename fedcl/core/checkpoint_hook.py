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

from .hook import Hook, HookPhase, HookPriority
from .execution_context import ExecutionContext
from ..exceptions import HookExecutionError, ConfigurationError
from ..registry import registry


@registry.hook("checkpoint_hook", metadata={
    "description": "自动保存模型检查点的hook",
    "phases": ["after_epoch", "after_task", "after_round", "after_experiment"],
    "priority": HookPriority.HIGH.value,
    "version": "1.0.0"
})
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
        self.save_experiment_state_once = checkpoint_config.get('save_experiment_state_once', True)  # 只保存一次experiment_state
        
        logger.debug(f"CheckpointHook配置: save_frequency={self.save_frequency}, save_model={self.save_model}, save_experiment_state={self.save_experiment_state_enabled}")
        
        # 路径配置 - 支持实验级别的目录隔离
        self.base_checkpoint_dir = checkpoint_config.get('checkpoint_dir', './checkpoints')
        self.experiment_dir_created = False  # 标记是否已创建实验目录
        
        # 初始设置为基础目录，实际目录将在第一次执行时确定
        self.checkpoint_dir = Path(self.base_checkpoint_dir)
        
        # 文件命名配置
        self.naming_pattern = checkpoint_config.get('naming_pattern', 'checkpoint_{phase}_round_{round}_epoch_{epoch}')
        self.include_timestamp = checkpoint_config.get('include_timestamp', False)  # 默认不包含时间戳
        
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
        self.experiment_state_saved = False  # 跟踪experiment_state是否已保存
        
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
            logger.debug(f"CheckpointHook.execute - phase: {self.phase}, kwargs keys: {list(kwargs.keys())}")
            
            # 确保实验目录已设置
            self._ensure_experiment_directory(context)
            logger.debug(f"CheckpointHook.execute - checkpoint_dir: {self.checkpoint_dir}")
            
            # 生成检查点路径
            checkpoint_path = self._generate_checkpoint_path(context, **kwargs)
            logger.debug(f"CheckpointHook.execute - checkpoint_path: {checkpoint_path}")
            
            # 创建检查点目录
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"CheckpointHook.execute - created directory: {checkpoint_path}")
            
            # 保存模型检查点
            if 'model' in kwargs and self.save_model:
                model_path = checkpoint_path / 'model.pth'
                logger.debug(f"CheckpointHook.execute - saving model to: {model_path}")
                self.save_model_checkpoint(kwargs['model'], model_path)
                logger.debug(f"CheckpointHook.execute - model saved successfully")
            else:
                logger.debug(f"CheckpointHook.execute - not saving model: model in kwargs={('model' in kwargs)}, save_model={self.save_model}")
            
            # 保存优化器状态
            if 'optimizer' in kwargs and self.save_optimizer:
                optimizer_path = checkpoint_path / 'optimizer.pth'
                self._save_optimizer_state(kwargs['optimizer'], optimizer_path)
            
            # 保存调度器状态
            if 'scheduler' in kwargs and self.save_scheduler:
                scheduler_path = checkpoint_path / 'scheduler.pth'
                self._save_scheduler_state(kwargs['scheduler'], scheduler_path)
            
            # 保存实验状态（每个客户端只保存一次，通过检查文件是否存在）
            if self.save_experiment_state_enabled:
                state_path = checkpoint_path / 'experiment_state.json'
                logger.debug(f"CheckpointHook.execute - saving experiment state to: {state_path}")
                # 检查是否已经保存过experiment_state（在同一个checkpoint_dir中查找）
                existing_state_files = list(self.checkpoint_dir.glob('*/experiment_state.json'))
                if not existing_state_files:
                    # 第一次保存
                    self.save_experiment_state(context, state_path)
                    self.experiment_state_saved = True  # 标记已保存
                elif not self.experiment_state_saved:
                    # 如果已经有其他checkpoint保存了experiment_state，就不再保存
                    logger.debug(f"Experiment state already exists in checkpoint directory, skipping save for {checkpoint_path}")
                    self.experiment_state_saved = True
            
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
            # 检查模型是否为None
            if model is None:
                logger.warning(f"模型为None，跳过保存: {path}")
                return
            
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
                logger.debug(f"PyTorch模型检查点已保存: {path}")
                
            else:
                # 其他类型的模型，先检查是否可序列化
                try:
                    import pickle
                    with open(path, 'wb') as f:
                        pickle.dump(model, f)
                    logger.debug(f"模型检查点已保存（pickle格式）: {path}")
                except Exception as pickle_error:
                    logger.error(f"模型无法序列化，模型类型: {type(model)}, 错误: {pickle_error}")
                    # 保存模型的基本信息而不是模型本身
                    model_info = {
                        'model_type': str(type(model)),
                        'model_class': model.__class__.__name__ if hasattr(model, '__class__') else 'Unknown',
                        'timestamp': time.time(),
                        'error': f"Model serialization failed: {str(pickle_error)}"
                    }
                    with open(path.with_suffix('.json'), 'w') as f:
                        json.dump(model_info, f, indent=2)
                    logger.warning(f"保存了模型信息而非模型本身: {path.with_suffix('.json')}")
            
        except Exception as e:
            logger.error(f"保存模型检查点失败: {e}")
            # 不抛出异常，允许其他部分继续执行
            logger.debug(f"模型类型: {type(model)}, 路径: {path}")
    
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
    
    def should_execute(self, context: ExecutionContext, **kwargs) -> bool:
        """
        判断是否应该执行检查点保存
        
        Args:
            context: 执行上下文
            **kwargs: 额外参数
            
        Returns:
            bool: 是否应该执行
        """
        logger.debug(f"CheckpointHook.should_execute - phase: {self.phase}, kwargs keys: {list(kwargs.keys())}")
        
        if not super().should_execute(context, **kwargs):
            logger.debug(f"CheckpointHook.should_execute - super().should_execute returned False")
            return False
        
        # 检查保存频率
        if self.phase in [HookPhase.AFTER_EPOCH.value, HookPhase.AFTER_TASK.value]:
            if self.phase == HookPhase.AFTER_TASK.value:
                current_step = context.get_state('current_round', 'global') or 0
            else:
                current_step = context.get_state('current_epoch', 'global') or 0
                
            if current_step % self.save_frequency != 0:
                logger.debug(f"CheckpointHook.should_execute - frequency check failed: step {current_step} % {self.save_frequency} != 0")
                return False
        
        # 对于after_round阶段，也检查轮次频率
        if self.phase == HookPhase.AFTER_ROUND.value:
            current_round = context.get_state('current_round', 'global') or 0
            logger.debug(f"CheckpointHook.should_execute - after_round phase, current_round: {current_round}, save_frequency: {self.save_frequency}")
            if current_round % self.save_frequency != 0:
                logger.debug(f"CheckpointHook.should_execute - round frequency check failed: round {current_round} % {self.save_frequency} != 0")
                return False
        
        # 检查是否有需要保存的内容
        has_content = False
        if 'model' in kwargs and self.save_model:
            has_content = True
            logger.debug(f"CheckpointHook.should_execute - has model content, model type: {type(kwargs['model'])}")
        if self.save_experiment_state_enabled:
            has_content = True
            logger.debug(f"CheckpointHook.should_execute - has experiment state content")
            
        logger.debug(f"CheckpointHook.should_execute - has_content: {has_content}")
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
    
    def _generate_checkpoint_path(self, context: ExecutionContext, **kwargs) -> Path:
        """生成检查点保存路径"""
        # 获取当前状态，优先使用kwargs中的参数
        current_round = kwargs.get('round', context.get_state('current_round', 'global') or 0)
        current_epoch = kwargs.get('epoch', context.get_state('current_epoch', 'global') or 0)
        current_task = kwargs.get('task', context.get_state('current_task_id', 'global') or 0)
        server_id = kwargs.get('server_id', 'server')
        client_id = kwargs.get('client_id', context.get_state('client_id', 'global') or 'client')
        
        # 准备格式化参数
        format_params = {
            'phase': self.phase.replace('_', ''),  # 去掉下划线使文件名更简洁
            'round': current_round,
            'epoch': current_epoch,
            'task': current_task,
            'server_id': server_id,
            'client_id': client_id
        }
        
        # 格式化文件名，包含阶段信息
        checkpoint_name = self.naming_pattern.format(**format_params)
        
        # 添加时间戳
        if self.include_timestamp:
            timestamp = int(time.time())
            checkpoint_name += f"_{timestamp}"
        
        return self.checkpoint_dir / checkpoint_name
    
    def _ensure_experiment_directory(self, context: ExecutionContext) -> None:
        """确保实验目录已正确设置"""
        if self.experiment_dir_created:
            return
            
        # 检查是否需要创建实验级别的目录
        if not self._is_experiment_specific_dir(self.base_checkpoint_dir):
            # 优先从context中获取共享的实验目录信息
            experiment_dir = getattr(context, '_shared_experiment_dir', None)
            
            if experiment_dir:
                # 使用共享的实验目录
                self.checkpoint_dir = Path(experiment_dir) / 'checkpoints'
                logger.debug(f"Using shared experiment checkpoint directory: {self.checkpoint_dir}")
            else:
                # 创建新的实验级别目录
                experiment_name = getattr(context, 'experiment_name', None) or context.experiment_id or 'unknown_experiment'
                experiment_timestamp = getattr(context, 'experiment_timestamp', None) or int(time.time())
                
                # 使用统一的实验目录格式: experiments/experiment_timestamp/checkpoints
                if hasattr(context, '_base_experiment_dir'):
                    # 如果context有基础实验目录，使用它
                    base_exp_dir = getattr(context, '_base_experiment_dir')
                    self.checkpoint_dir = Path(base_exp_dir) / 'checkpoints'
                else:
                    # 创建标准的实验目录
                    experiment_dir_name = f"experiment_{experiment_timestamp}"
                    self.checkpoint_dir = Path('experiments') / experiment_dir_name / 'checkpoints'
                
                # 在context中设置共享目录，让其他实例使用相同目录
                context._shared_experiment_dir = str(self.checkpoint_dir.parent)
                
                logger.debug(f"Creating unified experiment checkpoint directory: {self.checkpoint_dir}")
        else:
            # 用户已指定具体目录，直接使用
            self.checkpoint_dir = Path(self.base_checkpoint_dir)
        
        # 创建目录
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_dir_created = True
    
    def _is_experiment_specific_dir(self, dir_path: str) -> bool:
        """检查目录路径是否已经是实验特定的（包含实验名称或ID）"""
        path_str = str(dir_path).lower()
        # 如果路径包含实验相关的关键字，认为是用户指定的实验目录
        experiment_keywords = ['experiment', 'exp_', 'run_', 'trial_', 'session_']
        return any(keyword in path_str for keyword in experiment_keywords)
    
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
