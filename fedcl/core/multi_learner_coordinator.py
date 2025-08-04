# fedcl/core/multi_learner_coordinator.py
"""
MultiLearnerCoordinator - 多Learner协调器

负责协调多个learner的训练执行，包括：
- 依赖关系解析
- 执行计划生成
- 特征流管理
- 资源调度
- 并行/串行执行控制
"""

import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict, deque
import torch
from loguru import logger

from .base_learner import BaseLearner, LearnerExecutionMode, ResourceRequirements
from .execution_context import ExecutionContext
from ..exceptions import LearnerError


class ExecutionStrategy(Enum):
    """执行策略"""
    SEQUENTIAL = "sequential"    # 严格串行执行
    PARALLEL = "parallel"       # 尽可能并行执行
    ADAPTIVE = "adaptive"       # 自适应执行
    DEPENDENCY_BASED = "dependency_based"  # 基于依赖关系执行


@dataclass
class ExecutionGroup:
    """执行组 - 可以并行执行的learner组"""
    group_id: int
    learner_ids: List[str]
    execution_mode: str = "parallel"  # parallel, sequential
    dependencies: List[int] = field(default_factory=list)  # 依赖的组ID
    estimated_time: float = 0.0
    resource_requirement: Optional[ResourceRequirements] = None


@dataclass
class ExecutionPlan:
    """执行计划"""
    strategy: ExecutionStrategy
    execution_groups: List[ExecutionGroup]
    dependency_graph: Dict[str, List[str]]  # learner_id -> [dependent_learner_ids]
    feature_flow_graph: Dict[str, List[str]]  # source_learner -> [target_learners]
    estimated_total_time: float
    max_parallel_groups: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """执行结果"""
    learner_id: str
    success: bool
    execution_time: float
    output_features: Dict[str, torch.Tensor]
    metrics: Dict[str, float]
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionSummary:
    """执行摘要"""
    total_execution_time: float
    successful_learners: List[str]
    failed_learners: List[str]
    execution_results: Dict[str, ExecutionResult]
    feature_exchanges: int
    parallel_efficiency: float
    resource_utilization: Dict[str, float]


class MultiLearnerCoordinator:
    """
    多Learner协调器
    
    核心职责：
    1. 解析learner间的依赖关系
    2. 生成优化的执行计划
    3. 协调learner的执行顺序
    4. 管理特征在learner间的流动
    5. 处理并行执行和资源调度
    """
    
    def __init__(self, 
                 learners: List[BaseLearner], 
                 context: ExecutionContext,
                 config: Dict[str, Any] = None):
        """
        初始化多Learner协调器
        
        Args:
            learners: learner列表
            context: 执行上下文
            config: 协调器配置
        """
        self.context = context
        self.config = config or {}
        
        # Learner管理
        self.learners = {learner.learner_id: learner for learner in learners}
        self.learner_order = [learner.learner_id for learner in learners]
        
        # 执行计划
        self.execution_plan: Optional[ExecutionPlan] = None
        self.current_execution: Dict[str, Any] = {}
        
        # 资源管理
        self.available_memory = self.config.get('available_memory_mb', 4096)
        self.available_gpu = self.config.get('available_gpu_mb', 2048)
        self.max_parallel_learners = self.config.get('max_parallel_learners', 2)
        
        # 执行策略
        strategy_name = self.config.get('execution_strategy', 'adaptive')
        self.execution_strategy = ExecutionStrategy(strategy_name)
        
        # 线程池（用于并行执行）
        max_workers = min(self.max_parallel_learners, len(learners))
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        
        # 统计信息
        self.execution_history: List[ExecutionSummary] = []
        
        logger.debug(f"MultiLearnerCoordinator initialized with {len(learners)} learners")
        logger.debug(f"Execution strategy: {self.execution_strategy.value}")
        
        # 初始化时创建执行计划
        self._create_execution_plan()
    
    def execute_training_round(self, task_data: Any) -> ExecutionSummary:
        """
        执行一轮多learner训练
        
        Args:
            task_data: 训练数据
            
        Returns:
            ExecutionSummary: 执行摘要
        """
        logger.debug("Starting multi-learner training round")
        start_time = time.time()
        
        # 重置learner状态
        self._reset_learner_states()
        
        # 清除之前的特征缓存
        self.context.clear_learner_features()
        
        # 执行前检查
        if not self._validate_execution_plan():
            raise LearnerError("Execution plan validation failed")
        
        # 执行训练
        execution_results = {}
        
        try:
            if self.execution_strategy == ExecutionStrategy.SEQUENTIAL:
                execution_results = self._execute_sequential(task_data)
            elif self.execution_strategy == ExecutionStrategy.PARALLEL:
                execution_results = self._execute_parallel(task_data)
            elif self.execution_strategy == ExecutionStrategy.DEPENDENCY_BASED:
                execution_results = self._execute_dependency_based(task_data)
            else:  # ADAPTIVE
                execution_results = self._execute_adaptive(task_data)
                
        except Exception as e:
            logger.error(f"Multi-learner execution failed: {e}")
            raise LearnerError(f"Execution failed: {e}")
        
        # 计算执行摘要
        total_time = time.time() - start_time
        summary = self._create_execution_summary(execution_results, total_time)
        
        # 记录执行历史
        self.execution_history.append(summary)
        
        logger.info(f"Multi-learner training round completed in {total_time:.2f}s")
        logger.info(f"Successful: {len(summary.successful_learners)}, Failed: {len(summary.failed_learners)}")
        
        return summary
    
    def _create_execution_plan(self) -> None:
        """创建执行计划"""
        logger.debug("Creating execution plan")
        
        # 构建依赖图
        dependency_graph = self._build_dependency_graph()
        feature_flow_graph = self._build_feature_flow_graph()
        
        # 创建执行组
        execution_groups = []
        
        if self.execution_strategy == ExecutionStrategy.DEPENDENCY_BASED:
            execution_groups = self._create_dependency_based_groups(dependency_graph)
        elif self.execution_strategy == ExecutionStrategy.PARALLEL:
            execution_groups = self._create_parallel_groups()
        elif self.execution_strategy == ExecutionStrategy.SEQUENTIAL:
            execution_groups = self._create_sequential_groups()
        else:  # ADAPTIVE
            execution_groups = self._create_adaptive_groups(dependency_graph)
        
        # 估算执行时间
        estimated_time = self._estimate_execution_time(execution_groups)
        
        # 创建执行计划
        self.execution_plan = ExecutionPlan(
            strategy=self.execution_strategy,
            execution_groups=execution_groups,
            dependency_graph=dependency_graph,
            feature_flow_graph=feature_flow_graph,
            estimated_total_time=estimated_time,
            max_parallel_groups=self.max_parallel_learners,
            metadata={
                'created_at': time.time(),
                'total_learners': len(self.learners),
                'total_groups': len(execution_groups)
            }
        )
        
        logger.debug(f"Execution plan created - {len(execution_groups)} groups, estimated time: {estimated_time:.2f}s")
    
    def _build_dependency_graph(self) -> Dict[str, List[str]]:
        """构建依赖图"""
        dependency_graph = defaultdict(list)
        
        for learner_id, learner in self.learners.items():
            dependencies = learner.get_dependencies()
            for dep in dependencies:
                if dep in self.learners:
                    dependency_graph[dep].append(learner_id)
                else:
                    logger.warning(f"Dependency {dep} not found for learner {learner_id}")
        
        return dict(dependency_graph)
    
    def _build_feature_flow_graph(self) -> Dict[str, List[str]]:
        """构建特征流图"""
        feature_flow_graph = defaultdict(list)
        
        for learner_id, learner in self.learners.items():
            feature_deps = learner.get_feature_dependencies()
            for feature_dep in feature_deps:
                source = feature_dep.get('source')
                if source and source in self.learners:
                    feature_flow_graph[source].append(learner_id)
        
        return dict(feature_flow_graph)
    
    def _create_dependency_based_groups(self, dependency_graph: Dict[str, List[str]]) -> List[ExecutionGroup]:
        """基于依赖关系创建执行组"""
        groups = []
        visited = set()
        group_id = 0
        
        # 拓扑排序创建执行层级
        in_degree = defaultdict(int)
        for learner_id, deps in dependency_graph.items():
            for dep in deps:
                in_degree[dep] += 1
        
        # 初始化队列（没有依赖的learner）
        queue = deque([lid for lid in self.learners.keys() if in_degree[lid] == 0])
        
        while queue:
            # 当前层的learner（可以并行执行）
            current_level = []
            next_queue = deque()
            
            while queue:
                learner_id = queue.popleft()
                if learner_id not in visited:
                    visited.add(learner_id)
                    current_level.append(learner_id)
                    
                    # 更新依赖计数
                    for dependent in dependency_graph.get(learner_id, []):
                        in_degree[dependent] -= 1
                        if in_degree[dependent] == 0:
                            next_queue.append(dependent)
            
            # 创建执行组
            if current_level:
                # 检查是否可以并行执行
                can_parallel = self._can_execute_parallel(current_level)
                
                group = ExecutionGroup(
                    group_id=group_id,
                    learner_ids=current_level,
                    execution_mode="parallel" if can_parallel else "sequential"
                )
                groups.append(group)
                group_id += 1
            
            queue = next_queue
        
        return groups
    
    def _create_parallel_groups(self) -> List[ExecutionGroup]:
        """创建并行执行组"""
        # 尝试将所有learner放入一个并行组
        if self._can_execute_parallel(list(self.learners.keys())):
            return [ExecutionGroup(
                group_id=0,
                learner_ids=list(self.learners.keys()),
                execution_mode="parallel"
            )]
        
        # 否则按资源需求分组
        groups = []
        remaining_learners = list(self.learners.keys())
        group_id = 0
        
        while remaining_learners:
            current_group = []
            current_memory = 0
            current_gpu = 0
            
            for learner_id in remaining_learners[:]:
                learner = self.learners[learner_id]
                req = learner.get_resource_requirements()
                
                if (current_memory + req.memory_mb <= self.available_memory and
                    current_gpu + req.gpu_memory_mb <= self.available_gpu and
                    len(current_group) < self.max_parallel_learners):
                    
                    current_group.append(learner_id)
                    current_memory += req.memory_mb
                    current_gpu += req.gpu_memory_mb
                    remaining_learners.remove(learner_id)
            
            if current_group:
                groups.append(ExecutionGroup(
                    group_id=group_id,
                    learner_ids=current_group,
                    execution_mode="parallel" if len(current_group) > 1 else "sequential"
                ))
                group_id += 1
            else:
                # 如果无法分组，强制添加一个
                if remaining_learners:
                    groups.append(ExecutionGroup(
                        group_id=group_id,
                        learner_ids=[remaining_learners.pop(0)],
                        execution_mode="sequential"
                    ))
                    group_id += 1
        
        return groups
    
    def _create_sequential_groups(self) -> List[ExecutionGroup]:
        """创建串行执行组"""
        groups = []
        
        # 按优先级排序
        sorted_learners = sorted(
            self.learners.keys(),
            key=lambda x: self.learners[x].priority
        )
        
        for i, learner_id in enumerate(sorted_learners):
            groups.append(ExecutionGroup(
                group_id=i,
                learner_ids=[learner_id],
                execution_mode="sequential"
            ))
        
        return groups
    
    def _create_adaptive_groups(self, dependency_graph: Dict[str, List[str]]) -> List[ExecutionGroup]:
        """创建自适应执行组（结合依赖和资源）"""
        # 首先创建基于依赖的组
        dependency_groups = self._create_dependency_based_groups(dependency_graph)
        
        # 然后在每个组内考虑资源优化
        optimized_groups = []
        
        for group in dependency_groups:
            if len(group.learner_ids) > 1 and group.execution_mode == "parallel":
                # 检查资源是否允许并行
                if not self._can_execute_parallel(group.learner_ids):
                    # 分解成更小的并行组
                    sub_groups = self._split_group_by_resources(group)
                    optimized_groups.extend(sub_groups)
                else:
                    optimized_groups.append(group)
            else:
                optimized_groups.append(group)
        
        # 重新分配组ID
        for i, group in enumerate(optimized_groups):
            group.group_id = i
        
        return optimized_groups
    
    def _can_execute_parallel(self, learner_ids: List[str]) -> bool:
        """检查learner是否可以并行执行"""
        if len(learner_ids) <= 1:
            return True
        
        # 检查资源需求
        total_memory = sum(self.learners[lid].get_resource_requirements().memory_mb for lid in learner_ids)
        total_gpu = sum(self.learners[lid].get_resource_requirements().gpu_memory_mb for lid in learner_ids)
        
        if total_memory > self.available_memory or total_gpu > self.available_gpu:
            return False
        
        # 检查并行数量限制
        if len(learner_ids) > self.max_parallel_learners:
            return False
        
        # 检查learner间的兼容性
        for i, lid1 in enumerate(learner_ids):
            for lid2 in learner_ids[i+1:]:
                learner1 = self.learners[lid1]
                learner2 = self.learners[lid2]
                
                if not learner1.can_execute_parallel_with(
                    learner2, self.available_memory, self.available_gpu):
                    return False
        
        return True
    
    def _split_group_by_resources(self, group: ExecutionGroup) -> List[ExecutionGroup]:
        """按资源需求分解执行组"""
        sub_groups = []
        remaining_learners = group.learner_ids[:]
        group_id = group.group_id
        
        while remaining_learners:
            current_group_learners = []
            current_memory = 0
            current_gpu = 0
            
            for learner_id in remaining_learners[:]:
                req = self.learners[learner_id].get_resource_requirements()
                
                if (current_memory + req.memory_mb <= self.available_memory and
                    current_gpu + req.gpu_memory_mb <= self.available_gpu and
                    len(current_group_learners) < self.max_parallel_learners):
                    
                    current_group_learners.append(learner_id)
                    current_memory += req.memory_mb
                    current_gpu += req.gpu_memory_mb
                    remaining_learners.remove(learner_id)
            
            if current_group_learners:
                sub_groups.append(ExecutionGroup(
                    group_id=f"{group_id}_{len(sub_groups)}",
                    learner_ids=current_group_learners,
                    execution_mode="parallel" if len(current_group_learners) > 1 else "sequential",
                    dependencies=group.dependencies
                ))
            else:
                break
        
        return sub_groups
    
    def _estimate_execution_time(self, groups: List[ExecutionGroup]) -> float:
        """估算执行时间"""
        total_time = 0.0
        
        for group in groups:
            # 简化的时间估算
            if group.execution_mode == "parallel":
                # 并行执行时间为最长的learner时间
                max_time = max(
                    self._estimate_learner_time(learner_id) 
                    for learner_id in group.learner_ids
                )
                total_time += max_time
            else:
                # 串行执行时间为所有learner时间之和
                group_time = sum(
                    self._estimate_learner_time(learner_id) 
                    for learner_id in group.learner_ids
                )
                total_time += group_time
        
        return total_time
    
    def _estimate_learner_time(self, learner_id: str) -> float:
        """估算单个learner的执行时间"""
        # 简化实现，实际应该基于模型复杂度、数据量等
        learner = self.learners[learner_id]
        base_time = self.config.get('base_execution_time', 1.0)
        
        # 根据资源需求调整时间估算
        req = learner.get_resource_requirements()
        complexity_factor = (req.memory_mb + req.gpu_memory_mb) / 1000.0
        
        return base_time * (1.0 + complexity_factor * 0.1)
    
    def _execute_sequential(self, task_data: Any) -> Dict[str, ExecutionResult]:
        """串行执行所有learner"""
        results = {}
        
        # 按优先级排序
        sorted_learners = sorted(
            self.learners.keys(),
            key=lambda x: self.learners[x].priority
        )
        
        for learner_id in sorted_learners:
            logger.info(f"Executing learner {learner_id} (sequential)")
            result = self._execute_single_learner(learner_id, task_data)
            results[learner_id] = result
            
            # 如果learner失败，根据配置决定是否继续
            if not result.success and self.config.get('stop_on_error', False):
                logger.error(f"Stopping execution due to learner {learner_id} failure")
                break
            
            # 处理特征共享
            if result.success and result.output_features:
                self.context.share_features(learner_id, result.output_features)
        
        return results
    
    def _execute_parallel(self, task_data: Any) -> Dict[str, ExecutionResult]:
        """并行执行所有learner"""
        results = {}
        
        # 检查是否可以全部并行
        all_learner_ids = list(self.learners.keys())
        if self._can_execute_parallel(all_learner_ids):
            # 全部并行执行
            futures = {}
            for learner_id in all_learner_ids:
                future = self.thread_pool.submit(self._execute_single_learner, learner_id, task_data)
                futures[future] = learner_id
            
            # 收集结果
            for future in as_completed(futures):
                learner_id = futures[future]
                try:
                    result = future.result()
                    results[learner_id] = result
                    
                    # 处理特征共享
                    if result.success and result.output_features:
                        self.context.share_features(learner_id, result.output_features)
                        
                except Exception as e:
                    logger.error(f"Parallel execution failed for learner {learner_id}: {e}")
                    results[learner_id] = ExecutionResult(
                        learner_id=learner_id,
                        success=False,
                        execution_time=0.0,
                        output_features={},
                        metrics={},
                        error_message=str(e)
                    )
        else:
            # 分组并行执行
            return self._execute_parallel_groups(task_data)
        
        return results
    
    def _execute_dependency_based(self, task_data: Any) -> Dict[str, ExecutionResult]:
        """基于依赖关系执行"""
        return self._execute_by_plan(task_data)
    
    def _execute_adaptive(self, task_data: Any) -> Dict[str, ExecutionResult]:
        """自适应执行"""
        return self._execute_by_plan(task_data)
    
    def _execute_by_plan(self, task_data: Any) -> Dict[str, ExecutionResult]:
        """按执行计划执行"""
        results = {}
        
        for group in self.execution_plan.execution_groups:
            logger.info(f"Executing group {group.group_id} with {len(group.learner_ids)} learners")
            
            # 等待依赖组完成
            self._wait_for_dependencies(group, results)
            
            # 执行当前组
            if group.execution_mode == "parallel" and len(group.learner_ids) > 1:
                group_results = self._execute_group_parallel(group, task_data)
            else:
                group_results = self._execute_group_sequential(group, task_data)
            
            results.update(group_results)
            
            # 处理特征共享
            for learner_id, result in group_results.items():
                if result.success and result.output_features:
                    self.context.share_features(learner_id, result.output_features)
        
        return results
    
    def _execute_parallel_groups(self, task_data: Any) -> Dict[str, ExecutionResult]:
        """分组并行执行"""
        if not self.execution_plan:
            logger.error("No execution plan available")
            return {}
        
        return self._execute_by_plan(task_data)
    
    def _execute_group_parallel(self, group: ExecutionGroup, task_data: Any) -> Dict[str, ExecutionResult]:
        """并行执行组内learner"""
        results = {}
        futures = {}
        
        # 提交并行任务
        for learner_id in group.learner_ids:
            future = self.thread_pool.submit(self._execute_single_learner, learner_id, task_data)
            futures[future] = learner_id
        
        # 收集结果
        for future in as_completed(futures):
            learner_id = futures[future]
            try:
                result = future.result()
                results[learner_id] = result
            except Exception as e:
                logger.error(f"Parallel execution failed for learner {learner_id}: {e}")
                results[learner_id] = ExecutionResult(
                    learner_id=learner_id,
                    success=False,
                    execution_time=0.0,
                    output_features={},
                    metrics={},
                    error_message=str(e)
                )
        
        return results
    
    def _execute_group_sequential(self, group: ExecutionGroup, task_data: Any) -> Dict[str, ExecutionResult]:
        """串行执行组内learner"""
        results = {}
        
        for learner_id in group.learner_ids:
            result = self._execute_single_learner(learner_id, task_data)
            results[learner_id] = result
            
            # 如果失败且配置为停止，则跳出
            if not result.success and self.config.get('stop_on_error', False):
                break
        
        return results
    
    def _execute_single_learner(self, learner_id: str, task_data: Any) -> ExecutionResult:
        """执行单个learner"""
        learner = self.learners[learner_id]
        start_time = time.time()
        
        try:
            # 准备输入特征
            shared_features = self.context.get_all_shared_features(learner_id)
            if not learner.prepare_for_execution(shared_features):
                raise LearnerError(f"Learner {learner_id} preparation failed")
            
            # 执行训练步骤
            if hasattr(task_data, '__iter__'):
                # 如果是批次数据，执行多个步骤
                total_metrics = defaultdict(list)
                output_features = {}
                
                for batch_data in task_data:
                    step_result = learner.execute_training_step(batch_data)
                    
                    # 累积指标
                    if 'loss' in step_result and step_result['loss'] is not None:
                        total_metrics['loss'].append(step_result['loss'])
                    
                    # 更新输出特征
                    if 'features' in step_result:
                        output_features.update(step_result['features'])
                
                # 计算平均指标
                metrics = {}
                for metric_name, values in total_metrics.items():
                    if values:
                        metrics[metric_name] = sum(values) / len(values)
                
            else:
                # 单个批次
                step_result = learner.execute_training_step(task_data)
                metrics = {'loss': step_result.get('loss', 0.0)}
                output_features = step_result.get('features', {})
            
            # 完成执行
            learner.finalize_execution()
            
            execution_time = time.time() - start_time
            
            return ExecutionResult(
                learner_id=learner_id,
                success=True,
                execution_time=execution_time,
                output_features=output_features,
                metrics=metrics
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Learner {learner_id} execution failed: {e}")
            
            return ExecutionResult(
                learner_id=learner_id,
                success=False,
                execution_time=execution_time,
                output_features={},
                metrics={},
                error_message=str(e)
            )
    
    def _wait_for_dependencies(self, group: ExecutionGroup, completed_results: Dict[str, ExecutionResult]) -> None:
        """等待依赖组完成"""
        # 简化实现：检查依赖的learner是否都已完成
        for learner_id in group.learner_ids:
            learner = self.learners[learner_id]
            dependencies = learner.get_dependencies()
            
            for dep in dependencies:
                if dep not in completed_results:
                    logger.warning(f"Dependency {dep} not completed for learner {learner_id}")
                elif not completed_results[dep].success:
                    logger.warning(f"Dependency {dep} failed for learner {learner_id}")
    
    def _reset_learner_states(self) -> None:
        """重置所有learner状态"""
        for learner in self.learners.values():
            learner.reset_state()
    
    def _validate_execution_plan(self) -> bool:
        """验证执行计划"""
        if not self.execution_plan:
            logger.error("No execution plan available")
            return False
        
        # 检查是否所有learner都包含在计划中
        planned_learners = set()
        for group in self.execution_plan.execution_groups:
            planned_learners.update(group.learner_ids)
        
        missing_learners = set(self.learners.keys()) - planned_learners
        if missing_learners:
            logger.error(f"Learners not in execution plan: {missing_learners}")
            return False
        
        return True
    
    def _create_execution_summary(self, results: Dict[str, ExecutionResult], total_time: float) -> ExecutionSummary:
        """创建执行摘要"""
        successful = [lid for lid, result in results.items() if result.success]
        failed = [lid for lid, result in results.items() if not result.success]
        
        # 计算特征交换次数
        feature_exchanges = len([r for r in results.values() if r.output_features])
        
        # 计算并行效率
        sequential_time = sum(r.execution_time for r in results.values())
        parallel_efficiency = sequential_time / total_time if total_time > 0 else 0.0
        
        # 计算资源利用率
        resource_utilization = self._calculate_resource_utilization(results)
        
        return ExecutionSummary(
            total_execution_time=total_time,
            successful_learners=successful,
            failed_learners=failed,
            execution_results=results,
            feature_exchanges=feature_exchanges,
            parallel_efficiency=parallel_efficiency,
            resource_utilization=resource_utilization
        )
    
    def _calculate_resource_utilization(self, results: Dict[str, ExecutionResult]) -> Dict[str, float]:
        """计算资源利用率"""
        # 简化实现
        return {
            'memory_utilization': 0.75,  # 示例值
            'gpu_utilization': 0.80,     # 示例值
            'cpu_utilization': 0.60      # 示例值
        }
    
    def get_execution_plan(self) -> Optional[ExecutionPlan]:
        """获取执行计划"""
        return self.execution_plan
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """获取执行统计信息"""
        if not self.execution_history:
            return {"total_executions": 0}
        
        latest = self.execution_history[-1]
        return {
            "total_executions": len(self.execution_history),
            "latest_execution": {
                "total_time": latest.total_execution_time,
                "successful_learners": len(latest.successful_learners),
                "failed_learners": len(latest.failed_learners),
                "parallel_efficiency": latest.parallel_efficiency,
                "feature_exchanges": latest.feature_exchanges
            },
            "average_execution_time": sum(h.total_execution_time for h in self.execution_history) / len(self.execution_history),
            "success_rate": sum(len(h.successful_learners) for h in self.execution_history) / sum(len(h.successful_learners) + len(h.failed_learners) for h in self.execution_history)
        }
    
    def cleanup(self) -> None:
        """清理资源"""
        logger.debug("Cleaning up MultiLearnerCoordinator")
        
        # 关闭线程池
        self.thread_pool.shutdown(wait=True)
        
        # 重置learner状态
        self._reset_learner_states()
        
        # 清理执行历史
        self.execution_history.clear()
        
        logger.debug("MultiLearnerCoordinator cleanup 完成")
    
    def __repr__(self) -> str:
        return (f"MultiLearnerCoordinator("
                f"learners={len(self.learners)}, "
                f"strategy={self.execution_strategy.value}, "
                f"max_parallel={self.max_parallel_learners})")