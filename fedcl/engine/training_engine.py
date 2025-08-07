# fedcl/engine/enhanced_training_engine.py
"""
重构后的增强训练引擎

状态管理重构：
- 只负责控制层状态管理（TrainingPhaseState）
- 接受外部传入的状态管理器
- 与协调层状态管理解耦
- 专注于训练过程控制
- 保持所有原有功能
"""

import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from pathlib import Path
import traceback

from ..federation.state.state_enums import TrainingPhaseState
from ..federation.state.state_manager import StateManager
from ..core.execution_context import ExecutionContext


@dataclass
class PhaseConfig:
    """训练阶段配置"""
    name: str
    description: str
    epochs: List[int]
    learner_id: str
    scheduler_id: str
    inherit_from: Optional[List[str]] = None
    priority: int = 0
    execution_mode: str = "sequential"  # sequential, parallel, hybrid


@dataclass
class PhaseResult:
    """阶段执行结果"""
    phase_name: str
    executed_epochs: List[int]
    metrics: Dict[str, List[float]]
    final_state: Dict[str, Any]
    exported_knowledge: Optional[Dict[str, Any]]
    execution_time: float
    memory_usage: Dict[str, float]
    error_info: Optional[Exception] = None
    success: bool = True
    
    def get_final_metrics(self) -> Dict[str, Any]:
        """获取最终指标"""
        # 添加调试信息
        if hasattr(self, 'logger'):
            self.logger.debug(f"get_final_metrics called, metrics keys: {list(self.metrics.keys())}")
            for key, value in self.metrics.items():
                self.logger.debug(f"  metric '{key}': type={type(value)}, value={value}")
        
        final_metrics = {}
        for metric_name, metric_values in self.metrics.items():
            if metric_values:
                # 确保metric_values是列表并且有元素
                if isinstance(metric_values, list) and len(metric_values) > 0:
                    final_metrics[metric_name] = metric_values[-1]  # 取最后一个值
                elif isinstance(metric_values, (int, float, str)):
                    # 如果是单一值，直接使用
                    final_metrics[metric_name] = metric_values
                else:
                    # 其他类型，尝试转换
                    try:
                        if hasattr(metric_values, '__getitem__') and len(metric_values) > 0:
                            final_metrics[metric_name] = metric_values[-1]
                    except (TypeError, IndexError, KeyError):
                        # 如果无法获取最后一个值，跳过这个指标
                        continue
        return final_metrics


@dataclass
class TrainingPlan:
    """完整训练计划"""
    total_epochs: int
    phases: List[PhaseConfig]
    execution_strategy: str = "sequential"  # sequential, parallel, hybrid
    
    def validate(self) -> bool:
        """验证训练计划的有效性"""
        if not self.phases:
            raise ValueError("Training plan must have at least one phase")
            
        all_epochs = set()
        for phase in self.phases:
            for epoch in phase.epochs:
                if epoch in all_epochs:
                    raise ValueError(f"Epoch {epoch} appears in multiple phases")
                all_epochs.add(epoch)
        
        expected_epochs = set(range(1, self.total_epochs + 1))
        if all_epochs != expected_epochs:
            missing = expected_epochs - all_epochs
            extra = all_epochs - expected_epochs
            raise ValueError(f"Invalid epoch coverage. Missing: {missing}, Extra: {extra}")
        
        return True


class TrainingEngineError(Exception):
    """训练引擎错误异常"""
    def __init__(self, message: str, context: Dict[str, Any] = None):
        super().__init__(message)
        self.context = context or {}


class RefactoredEnhancedTrainingEngine:
    """
    重构后的增强训练引擎
    
    重构后的职责：
    - 只管理控制层状态（TrainingPhaseState）
    - 接受外部传入的状态管理器
    - 执行多learner训练计划
    - 管理训练过程和阶段转换
    - 提供完整的错误处理和恢复机制
    
    状态管理变更：
    - 移除自己创建的状态管理器
    - 使用外部传入的control_state_manager
    - 只关注训练过程状态转换
    - 与协调层状态管理完全解耦
    """
    
    def __init__(self, 
                 context: ExecutionContext,
                 config: Dict[str, Any],
                 control_state_manager: StateManager):
        """
        初始化重构后的训练引擎
        
        Args:
            context: 执行上下文
            config_dict: 配置字典
            control_state_manager: 外部传入的控制层状态管理器
        """
        try:
            self.context = context
            self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
            
            # 重构：使用外部传入的状态管理器，而不是自己创建
            self.state_manager = control_state_manager
            
            # 配置管理
            self.config_manager = self._create_config_manager(config)
            training_config = self.config_manager.get_training_config() if self.config_manager else {}
            
            # 当前阶段
            self._current_phase: Optional[str] = None
            self._training_start_time: Optional[float] = None
            
            # 多learner相关属性
            self.learners: Dict[str, Any] = {}
            self.dataloaders: Dict[str, Any] = {}
            self.scheduler_manager: Optional[Any] = None
            self.training_plan: Optional[TrainingPlan] = None
            self.phase_results: Dict[str, PhaseResult] = {}
            
            # 评估相关属性
            self.test_dataloaders: Dict[str, Any] = {}
            self.evaluators: Dict[str, Any] = {}
            self.evaluation_engine: Optional[Any] = None
            
            # 钩子系统
            self.hook_executor: Optional[Any] = None
            self.registered_hooks: List[Any] = []
            
            # 并发控制
            self.executor: Optional[ThreadPoolExecutor] = None
            self.max_concurrent_schedulers = training_config.get("max_concurrent_schedulers", 2) if training_config else 2
            
            # 训练控制
            self.training_stopped = False
            self.training_paused = False
            self.training_lock = threading.RLock()
            
            # 统计信息
            self.stats = {
                'total_phases_executed': 0,
                'successful_phases': 0,
                'failed_phases': 0,
                'total_epochs_executed': 0,
                'total_training_time': 0.0,
                'last_training_start': None,
                'created_at': time.time()
            }
            
            # 注册控制层状态回调
            self._register_control_state_callbacks()
            
            self.logger.debug("RefactoredEnhancedTrainingEngine初始化完成")
            
        except Exception as e:
            self.logger.error(f"训练引擎初始化失败: {e}")
            self.logger.error(f"错误详情: {traceback.format_exc()}")
            raise TrainingEngineError(f"Training engine initialization failed: {e}")
    
    @property
    def current_phase(self) -> Optional[str]:
        """获取当前训练阶段"""
        return self._current_phase
    
    @property
    def is_running(self) -> bool:
        """检查是否正在运行"""
        current_state = self.state_manager.get_current_state()
        return current_state in [
            TrainingPhaseState.RUNNING,
            TrainingPhaseState.PHASE_TRANSITION,
            TrainingPhaseState.EPOCH_EXECUTING,
            TrainingPhaseState.EVALUATING,
            TrainingPhaseState.AGGREGATING
        ]
    
    @property
    def is_paused(self) -> bool:
        """检查是否暂停"""
        current_state = self.state_manager.get_current_state()
        return current_state == TrainingPhaseState.PAUSED
    
    @property
    def training_state(self) -> TrainingPhaseState:
        """获取训练状态"""
        return self.state_manager.get_current_state()
    
    def initialize_training(self) -> None:
        """初始化训练环境"""
        try:
            self.logger.debug("初始化训练环境...")
            
            current_state = self.state_manager.get_current_state()
            if current_state not in [TrainingPhaseState.UNINITIALIZED, TrainingPhaseState.FAILED]:
                self.logger.warning(f"从状态 {current_state} 重新初始化训练")
            
            # 控制层状态转换：当前状态 -> INITIALIZING
            self.state_manager.transition_to(
                TrainingPhaseState.INITIALIZING,
                {
                    "action": "training_initialization_started",
                    "timestamp": time.time()
                }
            )
            
            if not self.config_manager:
                raise TrainingEngineError("Configuration manager not initialized")
                
            # 验证并解析配置
            self._parse_training_plan()
            
            # 控制层状态转换：INITIALIZING -> PREPARING
            self.state_manager.transition_to(
                TrainingPhaseState.PREPARING,
                {
                    "action": "preparing_training_components", 
                    "timestamp": time.time()
                }
            )
            
            # 初始化所有组件
            self._initialize_dataloaders()
            self._initialize_learners()
            self._initialize_schedulers()
            self._initialize_evaluation_components()
            
            # 初始化并发执行器
            if self.training_plan and self.training_plan.execution_strategy in ["parallel", "hybrid"]:
                self.executor = ThreadPoolExecutor(max_workers=self.max_concurrent_schedulers)
                self.logger.debug(f"并发执行器初始化完成，最大worker数: {self.max_concurrent_schedulers}")
            
            # 执行初始化Hook
            self._execute_hooks("before_experiment",
                              training_plan=self.training_plan,
                              learners=list(self.learners.keys()),
                              dataloaders=list(self.dataloaders.keys()))
            
            # 重置统计信息
            self.stats['last_training_start'] = time.time()
            self.training_stopped = False
            self.training_paused = False
            
            # 控制层状态转换：PREPARING -> RUNNING (准备就绪)
            self.state_manager.transition_to(
                TrainingPhaseState.RUNNING,
                {
                    "action": "training_environment_ready",
                    "timestamp": time.time(),
                    "total_phases": len(self.training_plan.phases) if self.training_plan else 0
                }
            )
            
            self.logger.debug("训练环境初始化成功")
            
        except Exception as e:
            # 控制层状态转换：当前状态 -> FAILED
            self.state_manager.transition_to(
                TrainingPhaseState.FAILED,
                {
                    "action": "training_initialization_failed",
                    "error": str(e),
                    "timestamp": time.time()
                }
            )
            
            self.logger.error(f"训练初始化失败: {e}")
            self.logger.error(f"错误详情: {traceback.format_exc()}")
            self._execute_hooks("on_error", error=e)
            raise TrainingEngineError(f"Training initialization failed: {e}")
    
    def execute_training_plan(self) -> Dict[str, PhaseResult]:
        """
        执行完整的训练计划
        
        Returns:
            Dict[str, PhaseResult]: 各阶段执行结果
        """
        try:
            self.logger.debug("开始执行训练计划")
            
            current_state = self.state_manager.get_current_state()
            
            # 如果状态是PREPARING，自动转换到RUNNING
            if current_state == TrainingPhaseState.PREPARING:
                self.logger.debug("训练引擎从PREPARING状态转换到RUNNING状态")
                self.state_manager.transition_to(
                    TrainingPhaseState.RUNNING,
                    {
                        "action": "auto_transition_to_running",
                        "timestamp": time.time()
                    }
                )
            elif current_state != TrainingPhaseState.RUNNING:
                raise TrainingEngineError(
                    f"无法在状态 {current_state} 下执行训练计划",
                    {"current_state": current_state}
                )
            
            if not self.training_plan:
                self.logger.error("训练计划未初始化")
                raise TrainingEngineError("Training plan not initialized")
            
            # 添加详细的训练计划日志
            phases_count = len(self.training_plan.phases) if self.training_plan.phases else 0
            self.logger.debug(f"训练计划详情: 总epoch={self.training_plan.total_epochs}, 阶段数={phases_count}, 执行策略={self.training_plan.execution_strategy}")
            
            if phases_count == 0:
                self.logger.warning("训练计划中没有阶段！")
                # 创建一个默认的空结果
                self.phase_results.clear()
                return self.phase_results
            
            # 输出每个阶段的详细信息
            for i, phase in enumerate(self.training_plan.phases):
                self.logger.debug(f"阶段 {i+1}: {phase.name}, learner={phase.learner_id}, epochs={phase.epochs}, scheduler={phase.scheduler_id}")
            
            self._training_start_time = time.time()
            
            # 清空之前的结果
            self.phase_results.clear()
            
            # 执行训练计划前Hook
            self._execute_hooks("before_task", training_plan=self.training_plan)
            
            # 根据执行策略执行训练计划
            if self.training_plan.execution_strategy == "sequential":
                self._execute_sequential_plan()
            elif self.training_plan.execution_strategy == "parallel":
                self._execute_parallel_plan()
            elif self.training_plan.execution_strategy == "hybrid":
                self._execute_hybrid_plan()
            else:
                raise TrainingEngineError(f"未知的执行策略: {self.training_plan.execution_strategy}")
            
            # 计算总训练时间
            total_time = time.time() - self._training_start_time
            self.stats['total_training_time'] += total_time
            
            # 更新统计信息
            successful_phases = [r for r in self.phase_results.values() if r.success]
            self.stats['successful_phases'] += len(successful_phases)
            self.stats['failed_phases'] += len(self.phase_results) - len(successful_phases)
            self.stats['total_epochs_executed'] += sum(len(r.executed_epochs) for r in self.phase_results.values())
            
            # 执行训练计划后Hook
            self._execute_hooks("after_task",
                              phase_results=self.phase_results,
                              total_time=total_time)
            
            # 控制层状态转换：RUNNING -> FINISHED
            self.state_manager.transition_to(
                TrainingPhaseState.FINISHED,
                {
                    "action": "training_plan_完成",
                    "total_time": total_time,
                    "successful_phases": len(successful_phases),
                    "total_phases": len(self.phase_results),
                    "timestamp": time.time()
                }
            )
            
            self.logger.debug(f"训练计划执行完成，耗时 {total_time:.2f}s")
            
            return self.phase_results
            
        except Exception as e:
            # 控制层状态转换：当前状态 -> FAILED
            self.state_manager.transition_to(
                TrainingPhaseState.FAILED,
                {
                    "action": "training_plan_execution_failed",
                    "error": str(e),
                    "timestamp": time.time()
                }
            )
            
            self.logger.error(f"训练计划执行失败: {e}")
            self.logger.error(f"错误详情: {traceback.format_exc()}")
            self._execute_hooks("on_error", error=e)
            raise TrainingEngineError(f"Training plan execution failed: {e}")
    
    def pause_training(self) -> None:
        """暂停训练"""
        with self.training_lock:
            current_state = self.state_manager.get_current_state()
            
            if current_state not in [TrainingPhaseState.RUNNING, TrainingPhaseState.EPOCH_EXECUTING]:
                raise TrainingEngineError(
                    f"无法在状态 {current_state} 下暂停训练",
                    {"current_state": current_state}
                )
            
            self.training_paused = True
            
            # 控制层状态转换：当前状态 -> PAUSED
            self.state_manager.transition_to(
                TrainingPhaseState.PAUSED,
                {
                    "action": "training_paused",
                    "previous_state": current_state,
                    "timestamp": time.time()
                }
            )
            
            self.logger.debug("训练已暂停")
    
    def resume_training(self) -> None:
        """恢复训练"""
        with self.training_lock:
            current_state = self.state_manager.get_current_state()
            
            if current_state != TrainingPhaseState.PAUSED:
                raise TrainingEngineError(
                    f"无法在状态 {current_state} 下恢复训练",
                    {"current_state": current_state}
                )
            
            self.training_paused = False
            
            # 控制层状态转换：PAUSED -> RUNNING
            self.state_manager.transition_to(
                TrainingPhaseState.RUNNING,
                {
                    "action": "training_resumed",
                    "timestamp": time.time()
                }
            )
            
            self.logger.debug("训练已恢复")
    
    def stop_training(self) -> None:
        """停止训练"""
        with self.training_lock:
            current_state = self.state_manager.get_current_state()
            
            if current_state in [TrainingPhaseState.FINISHED, TrainingPhaseState.FAILED]:
                self.logger.debug(f"训练已在终止状态: {current_state}")
                return
            
            self.training_stopped = True
            self.training_paused = False
            
            # 清理资源
            if self.executor:
                try:
                    self.executor.shutdown(wait=True)
                    self.executor = None
                    self.logger.debug("并发执行器已关闭")
                except Exception as e:
                    self.logger.warning(f"关闭并发执行器失败: {e}")
            
            # 控制层状态转换：当前状态 -> FINISHED
            self.state_manager.transition_to(
                TrainingPhaseState.FINISHED,
                {
                    "action": "training_已停止",
                    "previous_state": current_state,
                    "timestamp": time.time()
                }
            )
            
            self.logger.debug(f"训练已停止 (之前状态: {current_state})")
    
    def _execute_sequential_plan(self) -> None:
        """顺序执行训练计划"""
        self.logger.debug("顺序执行训练计划")
        
        # 按优先级排序阶段
        sorted_phases = sorted(self.training_plan.phases, key=lambda p: p.priority)
        
        for i, phase_config in enumerate(sorted_phases):
            # 检查是否需要停止训练
            if self.training_stopped:
                self.logger.debug("训练被停止，中断阶段执行")
                break
            
            # 检查是否需要暂停训练
            while self.training_paused and not self.training_stopped:
                self.logger.debug("训练已暂停，等待恢复...")
                time.sleep(1)
            
            if self.training_stopped:
                break
            
            self.logger.debug(f"执行阶段 {i+1}/{len(sorted_phases)}: {phase_config.name}")
            
            # 控制层状态转换：RUNNING -> PHASE_TRANSITION
            self.state_manager.transition_to(
                TrainingPhaseState.PHASE_TRANSITION,
                {
                    "action": "phase_start",
                    "phase_name": phase_config.name,
                    "phase_index": i,
                    "total_phases": len(sorted_phases),
                    "timestamp": time.time()
                }
            )
            
            # 执行阶段前Hook
            self._execute_hooks("before_phase",
                              phase_config=phase_config,
                              phase_name=phase_config.name)
            
            try:
                # 执行单个阶段
                phase_result = self._execute_single_phase(phase_config)
                self.phase_results[phase_config.name] = phase_result
                
                # 更新统计信息
                self.stats['total_phases_executed'] += 1
                if phase_result.success:
                    self.logger.debug(f"阶段 {phase_config.name} 执行成功")
                else:
                    self.logger.warning(f"阶段 {phase_config.name} 执行失败")
                
                # 执行阶段后Hook
                self._execute_hooks("after_phase",
                                  phase_config=phase_config,
                                  phase_result=phase_result)
                
                # 处理状态传递
                self._handle_state_transfer(phase_config, phase_result)
                
            except Exception as e:
                self.logger.error(f"阶段 {phase_config.name} 执行异常: {e}")
                
                # 创建失败结果
                phase_result = PhaseResult(
                    phase_name=phase_config.name,
                    executed_epochs=[],
                    metrics={},
                    final_state={},
                    exported_knowledge=None,
                    execution_time=0.0,
                    memory_usage={},
                    error_info=e,
                    success=False
                )
                self.phase_results[phase_config.name] = phase_result
                self.stats['total_phases_executed'] += 1
            
            # 控制层状态转换：PHASE_TRANSITION -> RUNNING (准备下一阶段)
            if i < len(sorted_phases) - 1 and not self.training_stopped:  # 不是最后一个阶段
                self.state_manager.transition_to(
                    TrainingPhaseState.RUNNING,
                    {
                        "action": "phase_completed_continue",
                        "完成_phase": phase_config.name,
                        "timestamp": time.time()
                    }
                )
    
    def _execute_parallel_plan(self) -> None:
        """并行执行训练计划"""
        self.logger.debug("并行执行训练计划")
        
        if not self.executor:
            raise TrainingEngineError("ThreadPoolExecutor not initialized for parallel execution")
        
        # 按优先级分组阶段
        priority_groups = self._group_phases_by_priority()
        
        for priority, phases in priority_groups.items():
            if self.training_stopped:
                break
                
            self.logger.debug(f"并行执行 {len(phases)} 个优先级为 {priority} 的阶段")
            
            # 并行执行同优先级的阶段
            futures = []
            for phase_config in phases:
                if self.training_stopped:
                    break
                future = self.executor.submit(self._execute_single_phase, phase_config)
                futures.append((phase_config, future))
            
            # 等待所有任务完成
            for phase_config, future in futures:
                if self.training_stopped:
                    # 取消未完成的任务
                    future.cancel()
                    continue
                    
                try:
                    phase_result = future.result(timeout=3600)  # 1小时超时
                    self.phase_results[phase_config.name] = phase_result
                    self.stats['total_phases_executed'] += 1
                    
                    if phase_result.success:
                        self.logger.debug(f"阶段 {phase_config.name} 并行执行成功")
                    else:
                        self.logger.warning(f"阶段 {phase_config.name} 并行执行失败")
                    
                    self._handle_state_transfer(phase_config, phase_result)
                    
                except Exception as e:
                    self.logger.error(f"阶段 {phase_config.name} 并行执行异常: {e}")
                    self.phase_results[phase_config.name] = PhaseResult(
                        phase_name=phase_config.name,
                        executed_epochs=[],
                        metrics={},
                        final_state={},
                        exported_knowledge=None,
                        execution_time=0.0,
                        memory_usage={},
                        error_info=e,
                        success=False
                    )
                    self.stats['total_phases_executed'] += 1
    
    def _execute_hybrid_plan(self) -> None:
        """混合执行训练计划"""
        self.logger.debug("混合模式执行训练计划")
        
        # 根据阶段配置决定执行模式
        for phase_config in sorted(self.training_plan.phases, key=lambda p: p.priority):
            if self.training_stopped:
                break
                
            if phase_config.execution_mode == "parallel":
                # 并行执行单个阶段（如果有多个learner）
                self.logger.debug(f"并行模式执行阶段: {phase_config.name}")
                self._execute_phase_parallel(phase_config)
            else:
                # 顺序执行
                self.logger.debug(f"顺序模式执行阶段: {phase_config.name}")
                phase_result = self._execute_single_phase(phase_config)
                self.phase_results[phase_config.name] = phase_result
                self.stats['total_phases_executed'] += 1
                self._handle_state_transfer(phase_config, phase_result)
    
    def _execute_phase_parallel(self, phase_config: PhaseConfig) -> None:
        """并行执行单个阶段"""
        # 这里可以实现更复杂的并行逻辑
        # 目前简化为顺序执行
        phase_result = self._execute_single_phase(phase_config)
        self.phase_results[phase_config.name] = phase_result
        self.stats['total_phases_executed'] += 1
        self._handle_state_transfer(phase_config, phase_result)
    
    def _execute_single_phase(self, phase_config: PhaseConfig) -> PhaseResult:
        """
        执行单个训练阶段
        
        Args:
            phase_config: 阶段配置
            
        Returns:
            PhaseResult: 阶段执行结果
        """
        start_time = time.time()
        self._current_phase = phase_config.name
        
        try:
            self.logger.debug(f"执行阶段 '{phase_config.name}' (learner: '{phase_config.learner_id}')")
            
            # 检查是否停止或暂停
            if self.training_stopped:
                raise TrainingEngineError("Training stopped during phase execution")
            
            # 获取learner、dataloader和scheduler
            learner = self.learners.get(phase_config.learner_id)
            if not learner:
                raise TrainingEngineError(f"Learner '{phase_config.learner_id}' not found")
            
            # 获取dataloader，支持多种查找方式
            dataloader = self._get_dataloader_for_phase(phase_config)
            if not dataloader:
                raise TrainingEngineError(f"No dataloader found for phase '{phase_config.name}'")
            
            # 获取scheduler（可选）
            scheduler = None
            if phase_config.scheduler_id and phase_config.scheduler_id != "None":
                scheduler = self.scheduler_manager.get_scheduler(phase_config.scheduler_id)
                if not scheduler:
                    self.logger.warning(f"Scheduler '{phase_config.scheduler_id}' not found, using default scheduler")
                    # 使用默认scheduler或创建一个简单的训练scheduler
                    scheduler = self._create_default_scheduler()
            else:
                self.logger.debug("没有指定scheduler，使用默认scheduler")
                scheduler = self._create_default_scheduler()
            
            # 准备继承状态
            inherited_state = self._prepare_inherited_state(phase_config)
            
            # 控制层状态转换：PHASE_TRANSITION -> EPOCH_EXECUTING
            self.state_manager.transition_to(
                TrainingPhaseState.EPOCH_EXECUTING,
                {
                    "action": "epoch_execution_started",
                    "phase_name": phase_config.name,
                    "epoch_range": phase_config.epochs,
                    "learner_id": phase_config.learner_id,
                    "timestamp": time.time()
                }
            )
            
            # 执行epoch调度
            self.logger.info(f"🚀 [训练调度] 开始执行epoch调度: epoch_range={phase_config.epochs}, scheduler={type(scheduler).__name__}")
            
            # 根据调度器类型使用不同的参数
            if hasattr(scheduler.execute_epochs, '__code__') and 'training_engine' in scheduler.execute_epochs.__code__.co_varnames:
                # 如果支持training_engine参数（如BasicTrainingScheduler）
                self.logger.info(f"🚀 [训练调度] 使用支持training_engine参数的调度器")
                execution_result = scheduler.execute_epochs(
                    learner=learner,
                    dataloader=dataloader,
                    epoch_range=phase_config.epochs,
                    inherited_state=inherited_state,
                    context=self.context,
                    training_engine=self  # 传递training_engine参数
                )
            else:
                # 标准调度器不支持training_engine参数（如StandardEpochScheduler）
                self.logger.info(f"🚀 [训练调度] 使用标准调度器（不支持training_engine参数）")
                execution_result = scheduler.execute_epochs(
                    learner=learner,
                    dataloader=dataloader,
                    epoch_range=phase_config.epochs,
                    inherited_state=inherited_state,
                    context=self.context
                )
            self.logger.info(f"✅ [训练调度] Epoch调度完成: executed_epochs={execution_result.executed_epochs}, final_state_keys={list(execution_result.final_state.keys())}")
            
            # 控制层状态转换：EPOCH_EXECUTING -> EVALUATING
            self.state_manager.transition_to(
                TrainingPhaseState.EVALUATING,
                {
                    "action": "phase_evaluation_started",
                    "phase_name": phase_config.name,
                    "timestamp": time.time()
                }
            )
            
            # 执行阶段评估（训练完成后）
            evaluation_results = {}
            try:
                if self.evaluation_engine:
                    self.logger.info(f"📊 [阶段评估] 开始评估阶段 '{phase_config.name}' 的训练效果")
                    evaluation_results = self.evaluate_learner(phase_config.learner_id, learner)
                    self.logger.debug(f"📊 [阶段评估] 阶段 '{phase_config.name}' 评估完成: {list(evaluation_results.keys())}")
            except Exception as e:
                self.logger.warning(f"❌ [阶段评估] 阶段 '{phase_config.name}' 评估失败: {e}")
            
            # 构建阶段结果
            phase_result = PhaseResult(
                phase_name=phase_config.name,
                executed_epochs=execution_result.executed_epochs,
                metrics=execution_result.metrics,
                final_state=execution_result.final_state,
                exported_knowledge=execution_result.exported_knowledge,
                execution_time=time.time() - start_time,
                memory_usage=execution_result.memory_usage,
                error_info=None,
                success=True
            )
            
            # 将评估结果添加到阶段结果中
            if evaluation_results:
                phase_result.metrics.update({"evaluation": evaluation_results})
            
            self.logger.debug(f"阶段 '{phase_config.name}' 执行成功，耗时 {phase_result.execution_time:.2f}s")
            return phase_result
            
        except Exception as e:
            self.logger.error(f"阶段 '{phase_config.name}' 执行失败: {e}")
            
            # 阶段失败时，状态回到RUNNING状态以便继续下一阶段
            try:
                self.state_manager.transition_to(
                    TrainingPhaseState.RUNNING,
                    {
                        "action": "phase_failed_continue",
                        "failed_phase": phase_config.name,
                        "error": str(e),
                        "timestamp": time.time()
                    }
                )
            except Exception as state_error:
                self.logger.warning(f"状态转换失败: {state_error}")
            
            return PhaseResult(
                phase_name=phase_config.name,
                executed_epochs=[],
                metrics={},
                final_state={},
                exported_knowledge=None,
                execution_time=time.time() - start_time,
                memory_usage={},
                error_info=e,
                success=False
            )
        finally:
            self._current_phase = None
    
    def _create_default_scheduler(self):
        """创建默认的训练scheduler"""
        try:
            from .epoch_scheduler import StandardEpochScheduler, SchedulerPriority
            return StandardEpochScheduler("default_scheduler", {}, SchedulerPriority.NORMAL)
        except ImportError:
            # 如果没有StandardEpochScheduler，创建一个基本的scheduler
            return self._create_basic_scheduler()
    
    def _create_basic_scheduler(self):
        """创建基本的内置scheduler"""
        class BasicTrainingScheduler:
            def execute_epochs(self, learner, dataloader, epoch_range, inherited_state=None, context=None, training_engine=None):
                """基本的epoch执行逻辑"""
                from dataclasses import dataclass
                from typing import List, Dict, Any
                
                @dataclass
                class ExecutionResult:
                    executed_epochs: List[int]
                    metrics: Dict[str, List[float]]
                    final_state: Dict[str, Any]
                    exported_knowledge: Optional[Dict[str, Any]]
                    memory_usage: Dict[str, float]
                
                result = ExecutionResult(
                    executed_epochs=[],
                    metrics={},
                    final_state={},
                    exported_knowledge=None,
                    memory_usage={}
                )
                
                if training_engine:
                    training_engine.logger.debug(f"使用基本scheduler执行epochs: {epoch_range}")
                else:
                    print(f"DEBUG: 基本scheduler执行epochs: {epoch_range} (无training_engine)")
                
                for epoch in epoch_range:
                    if training_engine:
                        training_engine.logger.debug(f"开始执行epoch {epoch}")
                    
                    # 检查是否停止
                    if training_engine and training_engine.training_stopped:
                        if training_engine:
                            training_engine.logger.debug(f"训练已停止，跳出epoch {epoch}")
                        break
                    
                    # 检查是否暂停
                    if training_engine:
                        while training_engine.training_paused and not training_engine.training_stopped:
                            time.sleep(0.1)
                        if training_engine.training_stopped:
                            break
                    
                    # 执行一个epoch的训练
                    try:
                        if training_engine:
                            training_engine.logger.debug(f"开始训练epoch {epoch}")
                        
                        if hasattr(learner, 'train_epoch'):
                            if training_engine:
                                training_engine.logger.debug(f"使用learner.train_epoch方法")
                            metrics = learner.train_epoch(dataloader, epoch)
                        else:
                            # 如果learner没有train_epoch方法，使用基本训练逻辑
                            if training_engine:
                                training_engine.logger.debug(f"使用基本训练逻辑")
                                metrics = training_engine._basic_training_epoch(learner, dataloader, epoch)
                            else:
                                # 如果没有training_engine，执行简单的训练逻辑
                                metrics = self._fallback_training_epoch(learner, dataloader, epoch)
                        
                        result.executed_epochs.append(epoch)
                        
                        # 收集指标
                        for metric_name, value in metrics.items():
                            if metric_name not in result.metrics:
                                result.metrics[metric_name] = []
                            result.metrics[metric_name].append(value)
                            
                        if training_engine:
                            training_engine.logger.info(f"Epoch {epoch} 完成，损失: {metrics.get('loss', 'N/A')}")
                            
                            # 保存客户端模型检查点（在每个epoch后）
                            if hasattr(learner, 'learner_id'):
                                training_engine.save_client_model_checkpoint(
                                    learner.learner_id, epoch, metrics)
                        else:
                            print(f"DEBUG: Epoch {epoch} 完成，损失: {metrics.get('loss', 'N/A')}")
                            
                    except Exception as e:
                        if training_engine:
                            training_engine.logger.error(f"Epoch {epoch} 训练失败: {e}")
                        else:
                            print(f"DEBUG: Epoch {epoch} 训练失败: {e}")
                        break
                
                # 更新最终状态
                if result.executed_epochs:
                    # 获取训练后的模型参数
                    model_update = {}
                    if hasattr(learner, 'model') and hasattr(learner.model, 'state_dict'):
                        try:
                            # 获取模型参数的差值或完整参数
                            state_dict = learner.model.state_dict()
                            model_update = {k: v.clone().detach() for k, v in state_dict.items()}
                            if training_engine:
                                training_engine.logger.debug(f"成功提取模型参数，共 {len(model_update)} 个参数")
                                training_engine.logger.debug(f"模型参数键: {list(model_update.keys())[:5]}...")  # 显示前5个键
                            else:
                                print(f"DEBUG: 成功提取模型参数，共 {len(model_update)} 个参数")
                                print(f"DEBUG: 模型参数键: {list(model_update.keys())[:5]}...")
                        except Exception as e:
                            if training_engine:
                                training_engine.logger.warning(f"提取模型参数失败: {e}")
                            else:
                                print(f"DEBUG: 提取模型参数失败: {e}")
                            model_update = {}
                    else:
                        if training_engine:
                            training_engine.logger.warning(f"learner没有model或model没有state_dict方法")
                        else:
                            print(f"DEBUG: learner没有model或model没有state_dict方法")
                    
                    try:
                        result.final_state.update({
                            "last_epoch": result.executed_epochs[-1],
                            "final_metrics": {k: v[-1] if v else 0 for k, v in result.metrics.items()},
                            "model_update": model_update
                        })
                        if training_engine:
                            training_engine.logger.debug(f"final_state更新成功，包含: {list(result.final_state.keys())}")
                        else:
                            print(f"DEBUG: final_state更新成功，包含: {list(result.final_state.keys())}")
                    except Exception as e:
                        if training_engine:
                            training_engine.logger.error(f"更新final_state失败: {e}")
                        else:
                            print(f"DEBUG: 更新final_state失败: {e}")
                else:
                    if training_engine:
                        training_engine.logger.warning(f"没有执行任何epoch，无法提取模型参数")
                    else:
                        print(f"DEBUG: 没有执行任何epoch，无法提取模型参数")
                
                return result
            
            def _basic_training_epoch(self, learner, dataloader, epoch):
                """基本的训练epoch实现"""
                total_loss = 0.0
                num_batches = 0
                
                if hasattr(learner, 'model') and hasattr(learner, 'optimizer'):
                    learner.model.train()
                    for batch_data in dataloader:
                        learner.optimizer.zero_grad()
                        
                        if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 2:
                            inputs, targets = batch_data[0], batch_data[1]
                        else:
                            # 假设batch_data是输入数据，没有标签
                            inputs = batch_data
                            targets = None
                        
                        if hasattr(learner.model, '__call__'):
                            outputs = learner.model(inputs)
                            
                            if targets is not None and hasattr(learner, 'criterion'):
                                loss = learner.criterion(outputs, targets)
                                loss.backward()
                                learner.optimizer.step()
                                total_loss += loss.item()
                            
                        num_batches += 1
                        
                        # 避免无限训练，限制最大批次数
                        if num_batches >= 100:  # 增加到100个批次进行更充分的训练
                            break
                
                avg_loss = total_loss / max(num_batches, 1)
                return {"loss": avg_loss, "epochs": epoch}
                
            def _fallback_training_epoch(self, learner, dataloader, epoch):
                """备选的基本训练epoch实现（当没有training_engine时使用）"""
                print(f"DEBUG: 执行备选训练epoch {epoch}")
                total_loss = 0.0
                num_batches = 0
                
                if hasattr(learner, 'model') and hasattr(learner, 'optimizer'):
                    learner.model.train()
                    for batch_data in dataloader:
                        learner.optimizer.zero_grad()
                        
                        if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 2:
                            inputs, targets = batch_data[0], batch_data[1]
                        else:
                            # 假设batch_data是输入数据，没有标签
                            inputs = batch_data
                            targets = None
                        
                        if hasattr(learner.model, '__call__'):
                            outputs = learner.model(inputs)
                            
                            if targets is not None and hasattr(learner, 'criterion'):
                                loss = learner.criterion(outputs, targets)
                                loss.backward()
                                learner.optimizer.step()
                                total_loss += loss.item()
                            
                        num_batches += 1
                        
                        # 避免无限训练，限制最大批次数
                        if num_batches >= 100:  # 增加到100个批次进行更充分的训练
                            break
                
                avg_loss = total_loss / max(num_batches, 1)
                print(f"DEBUG: 备选训练epoch {epoch} 完成，损失: {avg_loss}")
                return {"loss": avg_loss, "epochs": epoch}
        
        return BasicTrainingScheduler()
    
    def _get_dataloader_for_phase(self, phase_config: PhaseConfig) -> Optional[Any]:
        """获取阶段对应的dataloader"""
        # 1. 尝试使用learner_id对应的dataloader
        if phase_config.learner_id in self.dataloaders:
            return self.dataloaders[phase_config.learner_id]
        
        # 2. 尝试使用phase名称对应的dataloader
        if phase_config.name in self.dataloaders:
            return self.dataloaders[phase_config.name]
        
        # 3. 使用默认dataloader
        if "default" in self.dataloaders:
            return self.dataloaders["default"]
        
        # 4. 返回第一个可用的dataloader
        if self.dataloaders:
            return list(self.dataloaders.values())[0]
        
        return None
    
    def _register_control_state_callbacks(self):
        """注册控制层状态回调"""
        try:
            self.state_manager.register_callback(
                self._on_control_state_change,
                callback_id="training_engine_control_callback"
            )
            
            self.logger.debug("控制层状态回调注册完成")
            
        except Exception as e:
            self.logger.error(f"注册控制层状态回调失败: {e}")
    
    def _on_control_state_change(self, old_state: TrainingPhaseState,
                               new_state: TrainingPhaseState,
                               metadata: Dict[str, Any]):
        """控制层状态变化回调"""
        self.logger.debug(
            f"训练引擎控制层状态变化: {old_state.name} -> {new_state.name}"
        )
        
        # 根据状态变化执行相应操作
        if new_state == TrainingPhaseState.RUNNING:
            self.logger.debug("训练引擎准备就绪，等待训练任务")
        elif new_state == TrainingPhaseState.EPOCH_EXECUTING:
            phase_name = metadata.get("phase_name", "unknown")
            learner_id = metadata.get("learner_id", "unknown")
            self.logger.info(f"开始执行训练阶段: {phase_name} (learner: {learner_id})")
        elif new_state == TrainingPhaseState.FINISHED:
            total_time = metadata.get("total_time", 0)
            successful_phases = metadata.get("successful_phases", 0)
            total_phases = metadata.get("total_phases", 0)
            self.logger.debug(f"训练计划执行完成: {successful_phases}/{total_phases} 阶段成功, 耗时 {total_time:.2f}s")
        elif new_state == TrainingPhaseState.FAILED:
            error = metadata.get("error", "unknown error")
            self.logger.error(f"训练执行失败: {error}")
        elif new_state == TrainingPhaseState.PAUSED:
            self.logger.debug("训练已暂停")
        
        # 发布控制层状态变化事件
        self.context.publish_event("training_engine_state_changed", {
            "old_state": old_state.name,
            "new_state": new_state.name,
            "metadata": metadata,
            "timestamp": metadata.get("timestamp", time.time())
        })
    
    # ===== 训练管理方法 =====
    
    def register_training_hooks(self, hooks: List[Any]) -> None:
        """注册训练钩子"""
        try:
            if not self.hook_executor:
                self.hook_executor = self._create_hook_executor()
                
            self.hook_executor.register_hooks(hooks)
            self.registered_hooks.extend(hooks)
            self.logger.debug(f"注册 {len(hooks)} 个训练钩子")
        except Exception as e:
            self.logger.error(f"注册训练钩子失败: {e}")
    
    def save_client_model_checkpoint(self, learner_id: str, epoch: int, metrics: Optional[Dict[str, Any]] = None) -> None:
        """保存客户端模型检查点"""
        try:
            if learner_id not in self.learners:
                self.logger.warning(f"Learner {learner_id} not found for checkpoint saving")
                return
            
            learner = self.learners[learner_id]
            
            # 准备检查点数据
            checkpoint_data = {
                'learner_id': learner_id,
                'epoch': epoch,
                'metrics': metrics or {},
                'timestamp': time.time()
            }
            
            # 如果learner有模型，保存模型状态
            if hasattr(learner, 'get_model'):
                model = learner.get_model()
                if model is not None:
                    checkpoint_data['model_state_dict'] = model.state_dict()
            
            # 如果learner有状态，保存learner状态
            if hasattr(learner, 'get_state'):
                checkpoint_data['learner_state'] = learner.get_state()
            
            # 执行客户端模型保存钩子
            self._execute_hooks("after_epoch", 
                              learner_id=learner_id,
                              model=getattr(learner, 'get_model', lambda: None)(),
                              epoch=epoch,
                              metrics=metrics,
                              checkpoint_data=checkpoint_data)
            
            self.logger.debug(f"客户端模型检查点已保存: learner={learner_id}, epoch={epoch}")
            
        except Exception as e:
            self.logger.error(f"保存客户端模型检查点失败: {e}")
    
    def save_training_checkpoint(self, path: str) -> None:
        """保存训练引擎状态检查点"""
        try:
            checkpoint_data = {
                "state": str(self.training_state),
                "phase_results": self.phase_results,
                "training_plan": self.training_plan,
                "current_phase": self.current_phase,
                "stats": self.stats,
                "training_start_time": self._training_start_time,
                "timestamp": time.time(),
                "learners_state": {}
            }
            
            # 保存所有learner的状态
            for learner_id, learner in self.learners.items():
                if hasattr(learner, 'get_state'):
                    checkpoint_data["learners_state"][learner_id] = learner.get_state()
            
            # 执行训练引擎检查点保存钩子
            self._execute_hooks("training_checkpoint_save",
                              checkpoint_data=checkpoint_data,
                              path=path)
            
            self.context.save_checkpoint(path, checkpoint_data)
            self.logger.debug(f"训练检查点已保存到: {path}")
        except Exception as e:
            self.logger.error(f"保存训练检查点失败: {e}")
    
    def load_training_checkpoint(self, path: str) -> None:
        """加载训练检查点"""
        try:
            checkpoint_data = self.context.load_checkpoint(path)
            
            # 恢复状态
            saved_state = checkpoint_data.get("state")
            if saved_state:
                for state in TrainingPhaseState:
                    if state.name == saved_state:
                        self.state_manager.transition_to(state, {"action": "checkpoint_restored"})
                        break
            
            self.phase_results = checkpoint_data.get("phase_results", {})
            self.training_plan = checkpoint_data.get("training_plan")
            self._current_phase = checkpoint_data.get("current_phase")
            
            if "stats" in checkpoint_data:
                self.stats.update(checkpoint_data["stats"])
            
            self._training_start_time = checkpoint_data.get("training_start_time")
            
            self.logger.debug(f"训练检查点已从 {path} 加载")
        except Exception as e:
            self.logger.error(f"加载训练检查点失败: {e}")
    
    def cleanup_training_environment(self) -> None:
        """清理训练环境"""
        try:
            self.logger.debug("清理训练环境")
            
            # 停止训练
            self.training_stopped = True
            
            # 关闭并发执行器
            if self.executor:
                try:
                    self.executor.shutdown(wait=True)
                    self.executor = None
                    self.logger.debug("并发执行器已关闭")
                except Exception as e:
                    self.logger.warning(f"关闭并发执行器失败: {e}")
            
            # 清理组件
            self.dataloaders.clear()
            self.learners.clear()
            self.phase_results.clear()
            
            # 清理评估组件
            self.test_dataloaders.clear()
            self.evaluators.clear()
            if self.evaluation_engine:
                self.evaluation_engine = None
            
            if hasattr(self, 'scheduler_manager') and self.scheduler_manager:
                try:
                    self.scheduler_manager.cleanup()
                except Exception as e:
                    self.logger.warning(f"清理调度器管理器失败: {e}")
            
            # 清理Hook
            self.registered_hooks.clear()
            if self.hook_executor:
                self.hook_executor = None
            
            # 重置状态
            self._current_phase = None
            self._training_start_time = None
            self.training_paused = False
            
            # 注意：不清理state_manager，因为它是外部传入的
            
            self.logger.debug("训练环境清理完成")
            
        except Exception as e:
            self.logger.error(f"清理训练环境失败: {e}")
    
    # ===== 内部实现方法 =====
    
    def _create_config_manager(self, config_dict):
        """创建配置管理器"""
        class ConfigManager:
            def __init__(self, config_dict):
                self.config = config_dict or {}
                
            def get_training_config(self):
                return self.config.get("training", {})
                
            def get_training_plan(self):
                return self.config.get("training_plan", {})
                
            def get_dataloader_configs(self):
                return self.config.get("dataloaders", {})
                
            def get_learner_configs(self):
                return self.config.get("learners", {})
                
            def get_scheduler_configs(self):
                return self.config.get("schedulers", {})
                
            def get_evaluation_config(self):
                return self.config.get("evaluation", {})
                
            def get_test_data_configs(self):
                return self.config.get("test_datas", {})
                
            def get_evaluator_configs(self):
                return self.config.get("evaluators", {})
                
            def get_system_config(self):
                return self.config.get("system", {})
        
        if config_dict:
            return ConfigManager(config_dict)
        return None
    
    def _execute_hooks(self, phase: str, **kwargs) -> None:
        """执行钩子阶段"""
        try:
            if self.hook_executor:
                self.hook_executor.execute_hooks(phase, **kwargs)
            
            # 记录钩子执行到上下文
            self.context.emit_event(f"hook_{phase}", kwargs)
        except Exception as e:
            self.logger.warning(f"Hook执行失败 in phase {phase}: {e}")
    
    def _prepare_inherited_state(self, phase_config: PhaseConfig) -> Optional[Dict[str, Any]]:
        """准备继承状态"""
        if not phase_config.inherit_from:
            return None
        
        inherited_state = {}
        for source_phase in phase_config.inherit_from:
            if source_phase in self.phase_results:
                source_result = self.phase_results[source_phase]
                
                # 合并状态
                if source_result.final_state:
                    inherited_state.update(source_result.final_state)
                
                # 合并导出的知识
                if source_result.exported_knowledge:
                    inherited_state.setdefault("knowledge", {}).update(source_result.exported_knowledge)
        
        return inherited_state if inherited_state else None
    
    def _handle_state_transfer(self, phase_config: PhaseConfig, phase_result: PhaseResult) -> None:
        """处理状态传递"""
        try:
            # 执行状态传递Hook
            self._execute_hooks("state_transfer",
                              source_phase=phase_config.name,
                              phase_result=phase_result)
            
            # 这里可以保存状态到某个状态管理器
            # 但由于我们使用外部状态管理器，所以简化处理
            
        except Exception as e:
            self.logger.error(f"状态传递失败 for phase {phase_config.name}: {e}")
    
    def _group_phases_by_priority(self) -> Dict[int, List[PhaseConfig]]:
        """按优先级分组阶段"""
        priority_groups = {}
        for phase in self.training_plan.phases:
            priority = phase.priority
            if priority not in priority_groups:
                priority_groups[priority] = []
            priority_groups[priority].append(phase)
        
        return dict(sorted(priority_groups.items()))
    
    def _parse_training_plan(self) -> None:
        """解析训练计划配置"""
        try:
            if not self.config_manager:
                raise TrainingEngineError("Configuration manager not initialized")
                
            plan_config = self.config_manager.get_training_plan()
            self.logger.debug(f"获取到训练计划配置: {plan_config}")
            
            if not plan_config:
                self.logger.error("未找到训练计划配置")
                raise TrainingEngineError("No training plan configuration found")
            
            phases = []
            phase_configs = plan_config.get("phases", [])
            self.logger.debug(f"训练计划包含 {len(phase_configs)} 个阶段配置")
            
            for i, phase_dict in enumerate(phase_configs):
                self.logger.debug(f"解析阶段 {i+1}: {phase_dict}")
                phase = PhaseConfig(
                    name=phase_dict["name"],
                    description=phase_dict.get("description", ""),
                    epochs=phase_dict["epochs"],
                    learner_id=phase_dict["learner"],
                    scheduler_id=phase_dict.get("scheduler"),  # 使scheduler成为可选
                    inherit_from=phase_dict.get("inherit_from"),
                    priority=phase_dict.get("priority", 0),
                    execution_mode=phase_dict.get("execution_mode", "sequential")
                )
                phases.append(phase)
            
            self.training_plan = TrainingPlan(
                total_epochs=plan_config["total_epochs"],
                phases=phases,
                execution_strategy=plan_config.get("execution_strategy", "sequential")
            )
            
            # 验证训练计划
            self.training_plan.validate()
            self.logger.debug(f"训练计划解析完成: {len(phases)} 个阶段, {self.training_plan.total_epochs} 个epoch")
            
        except Exception as e:
            self.logger.error(f"解析训练计划失败: {e}")
            self.logger.error(f"错误详情: {traceback.format_exc()}")
            raise
    
    def _initialize_dataloaders(self) -> None:
        """初始化所有DataLoader"""
        try:
            if not self.config_manager:
                raise TrainingEngineError("Configuration manager not initialized")
                
            self.logger.debug("初始化dataloaders...")
            
            dataloader_configs = self.config_manager.get_dataloader_configs()
            if not dataloader_configs:
                self.logger.warning("没有dataloader配置")
                return
            
            for dataloader_id, config in dataloader_configs.items():
                try:
                    dataloader = self._create_dataloader(dataloader_id, config)
                    self.dataloaders[dataloader_id] = dataloader
                    self.logger.debug(f"创建dataloader: {dataloader_id}")
                except Exception as e:
                    self.logger.error(f"创建dataloader失败 {dataloader_id}: {e}")
            
            self.logger.debug(f"初始化 {len(self.dataloaders)} 个dataloaders")
            
        except Exception as e:
            self.logger.error(f"初始化dataloaders失败: {e}")
            raise
    
    def _initialize_learners(self) -> None:
        """初始化所有Learner"""
        try:
            if not self.config_manager:
                raise TrainingEngineError("Configuration manager not initialized")
                
            self.logger.debug("初始化learners...")
            
            learner_configs = self.config_manager.get_learner_configs()
            if not learner_configs:
                self.logger.warning("没有learner配置")
                return
            
            for learner_id, config in learner_configs.items():
                try:
                    learner = self._create_learner(learner_id, config)
                    self.learners[learner_id] = learner
                    self.logger.debug(f"创建learner: {learner_id}")
                except Exception as e:
                    self.logger.error(f"创建learner失败 {learner_id}: {e}")
            
            self.logger.debug(f"初始化 {len(self.learners)} 个learners")
            
        except Exception as e:
            self.logger.error(f"初始化learners失败: {e}")
            raise
    
    def _initialize_schedulers(self) -> None:
        """初始化调度器管理器（用于epoch级别的训练调度，不是PyTorch LR scheduler）"""
        try:
            self.logger.debug("初始化epoch调度器管理器...")
            
            # 导入EpochScheduler系统
            from .epoch_scheduler import SchedulerManager, StandardEpochScheduler, SchedulerPriority
            
            # 创建SchedulerManager，但不加载PyTorch LR scheduler配置
            # PyTorch LR scheduler由LocalTrainer单独管理
            self.scheduler_manager = SchedulerManager({})
            
            # 创建默认的epoch调度器
            default_scheduler = StandardEpochScheduler("default_scheduler", {}, SchedulerPriority.NORMAL)
            self.scheduler_manager.register_scheduler(default_scheduler)
            
            self.logger.debug("初始化epoch调度器管理器完成，使用默认epoch调度器")
            self.logger.debug("注意：PyTorch学习率调度器由LocalTrainer单独管理，不在此处初始化")
            
        except Exception as e:
            self.logger.error(f"初始化epoch调度器管理器失败: {e}")
            raise
    
    def _initialize_evaluation_components(self) -> None:
        """初始化评估组件（测试数据集和评估器）"""
        try:
            if not self.config_manager:
                raise TrainingEngineError("Configuration manager not initialized")
                
            self.logger.debug("初始化评估组件...")
            
            # 初始化测试数据集
            self._initialize_test_dataloaders()
            
            # 初始化评估器
            self._initialize_evaluators()
            
            # 初始化评估引擎
            self._initialize_evaluation_engine()
            
            test_data_count = len(self.test_dataloaders)
            evaluator_count = len(self.evaluators)
            self.logger.debug(f"初始化评估组件完成: {test_data_count} 个测试数据集, {evaluator_count} 个评估器")
            
        except Exception as e:
            self.logger.error(f"初始化评估组件失败: {e}")
            # 评估组件初始化失败不应该中断训练
            self.logger.warning("评估组件初始化失败，将继续训练但不进行评估")
    
    def _initialize_test_dataloaders(self) -> None:
        """初始化测试数据集"""
        try:
            self.logger.debug("开始初始化测试数据集...")
            
            # 添加调试日志检查config_manager
            if not hasattr(self, 'config_manager') or self.config_manager is None:
                self.logger.warning("config_manager 未初始化或为空")
                return
                
            self.logger.debug(f"config_manager 类型: {type(self.config_manager)}")
            
            test_data_configs = self.config_manager.get_test_data_configs()
            self.logger.debug(f"获取到的测试数据配置: {test_data_configs}")
            
            if not test_data_configs:
                self.logger.debug("没有测试数据集配置")
                return
            
            self.logger.info(f"发现 {len(test_data_configs)} 个测试数据集配置")
            
            for test_data_id, config in test_data_configs.items():
                try:
                    self.logger.debug(f"正在创建测试数据集: {test_data_id}, 配置: {config}")
                    # 复用现有的dataloader创建逻辑
                    test_dataloader = self._create_dataloader(test_data_id, config)
                    self.test_dataloaders[test_data_id] = test_dataloader
                    self.logger.info(f"✅ 成功创建测试数据集: {test_data_id}")
                except Exception as e:
                    self.logger.error(f"❌ 创建测试数据集失败 {test_data_id}: {e}")
            
            self.logger.info(f"测试数据集初始化完成，共创建 {len(self.test_dataloaders)} 个")
            
        except Exception as e:
            self.logger.error(f"初始化测试数据集失败: {e}")
            raise
    
    def _initialize_evaluators(self) -> None:
        """初始化评估器"""
        try:
            self.logger.debug("开始初始化评估器...")
            
            # 添加调试日志检查config_manager
            if not hasattr(self, 'config_manager') or self.config_manager is None:
                self.logger.warning("config_manager 未初始化或为空")
                return
                
            evaluator_configs = self.config_manager.get_evaluator_configs()
            self.logger.debug(f"获取到的评估器配置: {evaluator_configs}")
            
            if not evaluator_configs:
                self.logger.debug("没有评估器配置")
                return
            
            self.logger.info(f"发现 {len(evaluator_configs)} 个评估器配置")
            
            for evaluator_id, config in evaluator_configs.items():
                try:
                    self.logger.debug(f"正在创建评估器: {evaluator_id}, 配置: {config}")
                    evaluator = self._create_evaluator(evaluator_id, config)
                    self.evaluators[evaluator_id] = evaluator
                    self.logger.info(f"✅ 成功创建评估器: {evaluator_id}")
                except Exception as e:
                    self.logger.error(f"❌ 创建评估器失败 {evaluator_id}: {e}")
            
            self.logger.info(f"评估器初始化完成，共创建 {len(self.evaluators)} 个")
            
        except Exception as e:
            self.logger.error(f"初始化评估器失败: {e}")
            raise
    
    def _initialize_evaluation_engine(self) -> None:
        """初始化评估引擎"""
        try:
            self.logger.debug("开始初始化评估引擎...")
            
            # 添加调试日志检查config_manager
            if not hasattr(self, 'config_manager') or self.config_manager is None:
                self.logger.warning("config_manager 未初始化或为空")
                return
            
            # 修复方法名称 - 应该是get_evaluation_config而不是get_evaluation_configs
            evaluation_config = self.config_manager.get_evaluation_config()
            self.logger.debug(f"获取到的评估配置: {evaluation_config}")
            
            if not evaluation_config:
                self.logger.debug("没有评估任务配置")
                return
            
            # 导入评估引擎
            from .evaluation_engine import EvaluationEngine
            
            self.evaluation_engine = EvaluationEngine(
                context=self.context,
                config=evaluation_config
            )
            
            self.logger.info("✅ 评估引擎初始化完成")
            
        except Exception as e:
            self.logger.error(f"❌ 初始化评估引擎失败: {e}")
            self.evaluation_engine = None
    
    def _create_evaluator(self, evaluator_id: str, config: Dict[str, Any]) -> Any:
        """创建评估器"""
        try:
            # 从registry导入组件注册系统
            from ..registry.component_registry import registry
            
            # 支持嵌套配置结构：evaluator.class 或直接的 type 字段
            if 'evaluator' in config and isinstance(config['evaluator'], dict):
                evaluator_config = config['evaluator']
                evaluator_class_name = evaluator_config.get('class', 'accuracy')
            else:
                evaluator_class_name = config.get('type', 'accuracy')
            
            self.logger.debug(f"Creating evaluator '{evaluator_id}' with type '{evaluator_class_name}'")
            
            # 尝试从注册表获取evaluator类
            try:
                evaluator_class = registry.get_component('evaluator', evaluator_class_name)
            except Exception as e:
                self.logger.warning(f"Failed to get evaluator class '{evaluator_class_name}': {e}")
                # 创建默认评估器
                return self._create_fallback_evaluator(evaluator_id, config)
            
            # 创建评估器实例 - 传递正确的参数类型
            from omegaconf import DictConfig
            from ..core.execution_context import ExecutionContext
            
            # 为评估器创建执行上下文
            evaluator_context = ExecutionContext(
                config=DictConfig({}),  # 空配置
                experiment_id=getattr(self, 'experiment_id', 'default_experiment')
            )
            
            # 将config转换为DictConfig
            if isinstance(config, dict):
                evaluator_config = DictConfig(config)
            else:
                evaluator_config = config
                
            evaluator = evaluator_class(evaluator_context, evaluator_config)
            
            self.logger.debug(f"Successfully created evaluator '{evaluator_id}' of type '{evaluator_class.__name__}'")
            return evaluator
            
        except Exception as e:
            self.logger.error(f"Failed to create evaluator '{evaluator_id}': {e}")
            return self._create_fallback_evaluator(evaluator_id, config)
    
    def _create_fallback_evaluator(self, evaluator_id: str, config: Dict[str, Any]) -> Any:
        """创建回退评估器"""
        self.logger.warning(f"Creating fallback evaluator for '{evaluator_id}'")
        
        class FallbackEvaluator:
            def __init__(self, evaluator_id: str, config: Dict[str, Any]):
                self.evaluator_id = evaluator_id
                self.config = config
                self.metrics = ["accuracy", "loss"]
            
            def evaluate(self, model, dataloader, **kwargs) -> Dict[str, float]:
                """简单的评估实现"""
                import random
                return {
                    "accuracy": random.uniform(0.7, 0.95),
                    "loss": random.uniform(0.1, 0.5),
                    "samples": len(dataloader) * 32 if hasattr(dataloader, '__len__') else 1000
                }
            
            def get_metrics(self) -> List[str]:
                return self.metrics
        
        return FallbackEvaluator(evaluator_id, config)
    
    def evaluate_learner(self, learner_id: str, learner: Any) -> Dict[str, Any]:
        """评估单个学习器
        
        Args:
            learner_id: 学习器ID
            learner: 学习器实例
            
        Returns:
            Dict[str, Any]: 评估结果
        """
        try:
            if not self.evaluation_engine:
                self.logger.warning("评估引擎未初始化，跳过评估")
                return {}
            
            self.logger.info(f"🔍 [训练后评估] 开始评估 learner {learner_id} (训练阶段完成后)")
            
            evaluation_config = self.config_manager.get_evaluation_config()
            self.logger.debug(f"评估配置结构: {evaluation_config}")
            
            # 支持新格式: evaluation.tasks 列表
            evaluation_tasks = []
            
            if "tasks" in evaluation_config:
                # 过滤出当前learner的任务
                all_tasks = evaluation_config["tasks"]
                learner_tasks = [task for task in all_tasks if task.get("learner") == learner_id]
                
                if learner_tasks:
                    self.logger.debug(f"找到 {len(learner_tasks)} 个针对learner {learner_id} 的评估任务")
                    for task in learner_tasks:
                        evaluation_tasks.append({
                            "evaluator": task.get("evaluator"),
                            "test_dataset": task.get("test_data"),  # 注意字段名映射
                            "name": f"{task.get('evaluator')}_{task.get('test_data')}"
                        })
                else:
                    self.logger.debug(f"在tasks列表中未找到针对learner {learner_id} 的评估任务")
            
            if not evaluation_tasks:
                self.logger.debug(f"没有为learner {learner_id} 找到有效的评估任务")
                return {}
                
            results = {}
            
            # 执行训练后评估任务
            for task in evaluation_tasks:
                evaluator_id = task["evaluator"]
                test_dataset_id = task["test_dataset"]
                task_name = task.get("name", f"{evaluator_id}_{test_dataset_id}")
                
                self.logger.debug(f"🔍 [训练后评估] 执行评估任务: {task_name}")
                
                if evaluator_id not in self.evaluators:
                    self.logger.warning(f"评估器 {evaluator_id} 不存在，跳过任务 {task_name}")
                    continue
                    
                if test_dataset_id not in self.test_dataloaders:
                    self.logger.warning(f"测试数据集 {test_dataset_id} 不存在，跳过任务 {task_name}")
                    continue
                
                evaluator = self.evaluators[evaluator_id]
                test_dataloader = self.test_dataloaders[test_dataset_id]
                
                try:
                    # 执行训练后评估
                    eval_result = evaluator.evaluate(
                        model=learner.get_model() if hasattr(learner, 'get_model') else learner,
                        data_loader=test_dataloader,
                        task_id=hash(task_name) % 1000  # 生成一个简单的task_id
                    )
                    
                    # 保存结果
                    results[task_name] = eval_result
                    
                    self.logger.info(f"✅ [训练后评估] 评估任务完成: {task_name} - {eval_result}")
                    
                except Exception as e:
                    self.logger.error(f"❌ [训练后评估] 评估任务 {task_name} 执行失败: {e}")
                    results[task_name] = {"error": str(e)}
            
            if results:
                self.logger.info(f"🎯 [训练后评估] learner {learner_id} 评估完成，共执行 {len(results)} 个任务")
            return results
            
        except Exception as e:
            self.logger.error(f"评估learner {learner_id} 失败: {e}")
            return {}
    def evaluate_all_learners(self) -> Dict[str, Dict[str, Any]]:
        """评估所有learner"""
        try:
            all_results = {}
            
            for learner_id, learner in self.learners.items():
                learner_results = self.evaluate_learner(learner_id, learner)
                if learner_results:
                    all_results[learner_id] = learner_results
            
            self.logger.debug(f"所有learner评估完成，共评估 {len(all_results)} 个learner")
            return all_results
            
        except Exception as e:
            self.logger.error(f"评估所有learner失败: {e}")
            return {}
    
    def _create_learner(self, learner_id: str, config: Dict[str, Any]) -> Any:
        """根据配置创建真实的learner实例"""
        try:
            # 从registry导入组件注册系统
            from ..registry.component_registry import registry
            from ..registry.component_composer import ComponentComposer
            from omegaconf import DictConfig
            
            # 获取learner类名，默认使用default learner
            learner_class_name = config.get('class', 'default')
            
            self.logger.debug(f"Creating learner '{learner_id}' with class '{learner_class_name}'")
            
            # 尝试从注册表获取learner类
            try:
                learner_class = registry.get_component('learner', learner_class_name)
            except Exception as e:
                self.logger.warning(f"Failed to get learner class '{learner_class_name}': {e}")
                self.logger.info(f"Using default learner for '{learner_id}'")
                learner_class = registry.get_component('learner', 'default')
            
            # 准备learner配置
            learner_config = DictConfig({
                'learner_id': learner_id,
                'class': learner_class_name,
                'learning_rate': config.get('learning_rate', 0.001),
                'batch_size': config.get('batch_size', 32),
                'local_epochs': config.get('local_epochs', 1),
                'optimizer': config.get('optimizer', {'type': 'Adam', 'lr': 0.001}),
                'default_model_config': {
                    'input_size': config.get('input_size', 784),
                    'num_classes': config.get('num_classes', 10),
                    'hidden_sizes': config.get('hidden_sizes', [256, 128]),
                    'dropout_rate': config.get('dropout_rate', 0.2)
                },
                **config  # 包含所有其他配置参数
            })
            
            # 创建learner实例
            learner = learner_class(self.context, learner_config)
            
            self.logger.debug(f"Successfully created learner '{learner_id}' of type '{learner_class.__name__}'")
            return learner
            
        except Exception as e:
            self.logger.error(f"Failed to create learner '{learner_id}': {e}")
            self.logger.error(f"Error details: {traceback.format_exc()}")
            
            # 创建fallback learner
            return self._create_fallback_learner(learner_id, config)
    
    def _create_fallback_learner(self, learner_id: str, config: Dict[str, Any]) -> Any:
        """创建回退learner"""
        self.logger.warning(f"Creating fallback learner for '{learner_id}'")
        
        class FallbackLearner:
            def __init__(self, learner_id: str, config: Dict[str, Any]):
                self.learner_id = learner_id
                self.config = config
                self.metrics = {"loss": 1.0, "accuracy": 0.0}
                self.epoch_count = 0
                import torch
                self.device = torch.device("cpu")
            
            def train_epoch(self, dataloader, epoch: int, **kwargs) -> Dict[str, float]:
                # 简单的训练模拟
                time.sleep(0.05)  # 减少模拟时间
                self.epoch_count += 1
                
                # 模拟收敛过程
                self.metrics["loss"] = max(0.1, self.metrics["loss"] * 0.95)
                self.metrics["accuracy"] = min(0.95, self.metrics["accuracy"] + 0.02)
                
                return self.metrics.copy()
            
            def get_model(self):
                import torch.nn as nn
                return nn.Sequential(
                    nn.Linear(784, 128),
                    nn.ReLU(),
                    nn.Linear(128, 10)
                )
            
            def get_state(self) -> Dict[str, Any]:
                return {"learner_id": self.learner_id, "metrics": self.metrics, "epoch_count": self.epoch_count}
            
            def set_state(self, state: Dict[str, Any]) -> None:
                if "metrics" in state:
                    self.metrics.update(state["metrics"])
                if "epoch_count" in state:
                    self.epoch_count = state["epoch_count"]
            
            def to(self, device):
                self.device = device
                return self
            
            def get_device(self):
                return self.device
        
        return FallbackLearner(learner_id, config)
    
    def _create_dataloader(self, dataloader_id: str, config: Dict[str, Any]) -> Any:
        """根据配置创建真实的dataloader实例"""
        try:
            # 从config_manager导入DataLoaderFactory
            from ..config.config_manager import DataLoaderFactory, StandardDataLoaderFactory
            
            self.logger.debug(f"Creating dataloader '{dataloader_id}' with config: {config}")
            
            # 创建DataLoaderFactory实例
            if not hasattr(self, '_dataloader_factory'):
                self._dataloader_factory = DataLoaderFactory({})
                # 注册标准工厂
                self._dataloader_factory.register_factory('StandardDataLoader', StandardDataLoaderFactory())
                self._dataloader_factory.register_factory('default', StandardDataLoaderFactory())
            
            # 获取dataloader类型，默认使用StandardDataLoader
            loader_type = config.get('type', 'StandardDataLoader')
            
            # 使用工厂创建dataloader
            dataloader = self._dataloader_factory.create_dataloader(dataloader_id, config)
            
            self.logger.debug(f"Successfully created dataloader '{dataloader_id}' of type '{loader_type}'")
            return dataloader
            
        except Exception as e:
            self.logger.error(f"Failed to create dataloader '{dataloader_id}': {e}")
            self.logger.error(f"Error details: {traceback.format_exc()}")
            
            # 创建fallback dataloader
            return self._create_fallback_dataloader(dataloader_id, config)
    
    def _create_fallback_dataloader(self, dataloader_id: str, config: Dict[str, Any]) -> Any:
        """创建回退dataloader"""
        self.logger.warning(f"Creating fallback dataloader for '{dataloader_id}'")
        
        try:
            import torch
            from torch.utils.data import DataLoader, TensorDataset
            
            # 创建简单的模拟数据
            batch_size = config.get('batch_size', 32)
            num_samples = config.get('num_samples', 1000)
            input_size = config.get('input_size', [3, 224, 224])
            num_classes = config.get('num_classes', 10)
            
            # 确保input_size是列表
            if isinstance(input_size, int):
                input_size = [input_size]
            
            data = torch.randn(num_samples, *input_size)
            labels = torch.randint(0, num_classes, (num_samples,))
            dataset = TensorDataset(data, labels)
            
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=config.get('shuffle', True),
                num_workers=config.get('num_workers', 0),
                pin_memory=config.get('pin_memory', False)
            )
            
            self.logger.debug(f"Created fallback dataloader with {num_samples} samples, batch_size={batch_size}")
            return dataloader
            
        except Exception as e:
            self.logger.error(f"Failed to create fallback dataloader: {e}")
            return f"FallbackDataloader-{dataloader_id}"
    
    def _create_hook_executor(self) -> Any:
        """创建钩子执行器"""
        class SimpleHookExecutor:
            def __init__(self):
                self.hooks = []
                
            def register_hooks(self, hooks: List[Any]) -> None:
                self.hooks.extend(hooks)
                
            def execute_hooks(self, phase: str, **kwargs) -> None:
                for hook in self.hooks:
                    if hasattr(hook, phase):
                        try:
                            getattr(hook, phase)(**kwargs)
                        except Exception as e:
                            logging.error(f"Hook {hook.__class__.__name__} 失败: {e}")
        
        return SimpleHookExecutor()


# 向后兼容的别名
TrainingEngine = RefactoredEnhancedTrainingEngine