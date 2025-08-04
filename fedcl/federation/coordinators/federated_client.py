# fedcl/federation/coordinators/multi_learner_federated_client.py
"""
多learner联邦客户端

重构后的实现，使用层级状态管理系统：
- 协调层状态管理（ClientLifecycleState）
- 与控制层状态自动同步
- 保持原有所有功能
- 提供更好的状态一致性保证
"""

import time
import threading
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass, field
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from loguru import logger
from omegaconf import OmegaConf
import traceback

# 基础通信类导入
from .base import (
    FederatedCommunicator, 
    CommunicationConfig, 
    CommunicatorRole, 
    MessageType
)

# 核心组件导入
from ...core.execution_context import ExecutionContext
from ...core.base_learner import BaseLearner

# 状态管理导入
from ...federation.state.state_manager import (
    ClientLifecycleState, 
    TrainingPhaseState,
    
)
from ...federation.state.hierarchical_state_manager import create_hierarchical_state_manager

# 其他组件导入
from ..exceptions import FederationError
from ...config.config_manager import DictConfig


@dataclass
class MultiLearnerTrainingResult:
    """多learner训练结果"""
    client_id: str
    round_id: int
    phase_results: Dict[str, Any]  # 各阶段的训练结果
    aggregated_model_update: Dict[str, torch.Tensor]  # 聚合后的模型更新
    total_samples: int
    training_metrics: Dict[str, Any]  # 聚合后的训练指标
    total_training_time: float
    learner_contributions: Dict[str, float]  # 各learner的贡献权重
    evaluation_results: Dict[str, Any] = field(default_factory=dict)  # 评估结果
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearnerInfo:
    """Learner信息"""
    learner_id: str
    learner_type: str
    learner_instance: BaseLearner
    dataloader_id: str
    scheduler_id: str
    priority: int = 0
    is_active: bool = True


@dataclass
class PhaseResult:
    """训练阶段结果"""
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
        final_metrics = {}
        for metric_name, metric_values in self.metrics.items():
            if metric_values:
                final_metrics[metric_name] = metric_values[-1]
        return final_metrics


class MultiLearnerFederatedClient(FederatedCommunicator):
    """
    多learner联邦客户端协调器
    
    重构后的实现特点：
    1. 使用HierarchicalStateManager进行层级状态管理
    2. 协调层专注于客户端生命周期管理
    3. 与训练引擎的控制层状态自动同步
    4. 保持所有原有功能不变
    5. 提供更好的状态监控和调试能力
    
    主要职责：
    - 客户端生命周期管理（协调层状态）
    - 与服务端的通信协调
    - 多learner组件的创建和管理
    - 训练任务的调度和结果聚合
    """
    
    def __init__(self, client_id: str, config: DictConfig):
        """
        初始化多learner联邦客户端
        
        Args:
            client_id: 客户端唯一标识
            config: 客户端完整配置
        """
        try:
            # 构建通信配置
            comm_config = self._build_communication_config(client_id, config)
            
            # 初始化通信基类
            super().__init__(comm_config)
            
            # 基本属性
            self.client_id = client_id
            self.client_config = config
            # 注意：不要重新绑定 self.logger，基类已经正确设置了组件日志器
            
            # 创建执行上下文
            self.context = self._create_execution_context(config)
            
            # 创建层级状态管理器
            self.hierarchical_state_manager = create_hierarchical_state_manager(
                self.context, 
                client_id,
                max_history=config.get('state_management', {}).get('max_history', 1000),
                enable_validation=config.get('state_management', {}).get('enable_validation', True)
            )
            
            # 多learner相关属性
            self.learners_info: Dict[str, LearnerInfo] = {}
            self.dataloaders: Dict[str, DataLoader] = {}
            
            # 训练引擎（延迟创建，避免循环依赖）
            self.enhanced_training_engine = None
            
            # 客户端状态
            self.current_round = 0
            self.is_training = False
            self.received_global_models: Dict[str, torch.nn.Module] = {}
            
            # 训练历史
            self.training_history: List[MultiLearnerTrainingResult] = []
            
            # 训练线程管理
            self.training_thread: Optional[threading.Thread] = None
            self.training_lock = threading.RLock()
            
            # 初始化多learner组件
            self._initialize_multi_learner_components()
            
            # 创建训练引擎
            self._initialize_training_engine()
            
            # 注册状态回调
            self._register_state_callbacks()
            
            # 注册Hook（如果配置了）
            self._register_hooks(config.get('hooks', {}))
            
            self.logger.debug(f"多学习器联邦客户端初始化完成: {client_id}")
            self.logger.debug(f"Learners: {list(self.learners_info.keys())}")
            self.logger.debug(f"状态管理: 层级化状态管理器")
            
        except Exception as e:
            self.logger.error(f"MultiLearnerFederatedClient初始化失败: {e}")
            self.logger.error(f"错误详情: {traceback.format_exc()}")
            raise FederationError(f"MultiLearnerFederatedClient initialization failed: {e}")
    
    def _build_communication_config(self, client_id: str, config: DictConfig) -> CommunicationConfig:
        """构建通信配置"""
        comm_settings = config.get('communication', {})
        
        return CommunicationConfig(
            role=CommunicatorRole.CLIENT,
            component_id=client_id,
            host=comm_settings.get('host', 'localhost'),
            port=comm_settings.get('port', 8080),
            max_workers=comm_settings.get('max_workers', 5),
            heartbeat_interval=comm_settings.get('heartbeat_interval', 30.0),
            message_timeout=comm_settings.get('timeout', 60.0)
        )
    
    def _initialize_training_engine(self):
        """初始化训练引擎"""
        try:
            # 延迟导入避免循环依赖
            from ...engine.training_engine import TrainingEngine
            
            # 构建训练引擎配置
            enhanced_config = self._build_enhanced_training_config(self.client_config)
            
            # 创建训练引擎
            self.enhanced_training_engine = TrainingEngine(
                context=self.context,
                config=enhanced_config,
                control_state_manager=self.hierarchical_state_manager.control_state_manager
            )
            
            # 将客户端的logger赋给训练引擎
            self.enhanced_training_engine.logger = self.logger
            
            self.logger.debug("训练引擎初始化完成")
            
        except Exception as e:
            self.logger.error(f"训练引擎初始化失败: {e}")
            raise
    
    # ===== FederatedCommunicator 抽象方法实现 =====
    
    def on_start(self) -> None:
        """客户端启动时的初始化"""
        try:
            self.logger.debug(f"启动多learner客户端: {self.client_id}")
            
            # 协调层状态转换：INITIALIZING -> LOADING_CONFIG
            self.hierarchical_state_manager.transition_coordination_state(
                ClientLifecycleState.LOADING_CONFIG, 
                {
                    "action": "loading_configuration",
                    "timestamp": time.time(),
                    "learner_count": len(self.learners_info)
                }
            )
            
            # 1. 初始化增强训练引擎
            if self.enhanced_training_engine:
                self.enhanced_training_engine.initialize_training()
            
            # 协调层状态转换：LOADING_CONFIG -> PREPARING_DATA
            self.hierarchical_state_manager.transition_coordination_state(
                ClientLifecycleState.PREPARING_DATA,
                {
                    "action": "preparing_multi_learner_data",
                    "timestamp": time.time()
                }
            )
            
            # 2. 加载客户端数据
            self._load_multi_learner_data()
            
            # 协调层状态转换：PREPARING_DATA -> REGISTERING
            self.hierarchical_state_manager.transition_coordination_state(
                ClientLifecycleState.REGISTERING,
                {
                    "action": "registering_to_server",
                    "timestamp": time.time()
                }
            )
            
            # 3. 向服务端注册
            registration_success = self._register_to_server()
            
            # 协调层状态转换：REGISTERING -> REGISTERED 或 ERROR
            if registration_success:
                self.hierarchical_state_manager.transition_coordination_state(
                    ClientLifecycleState.REGISTERED,
                    {
                        "action": "registration_完成",
                        "success": True,
                        "timestamp": time.time()
                    }
                )
                
                # 注册成功后转为READY状态
                self.hierarchical_state_manager.transition_coordination_state(
                    ClientLifecycleState.READY,
                    {
                        "action": "client_ready",
                        "timestamp": time.time()
                    }
                )
            else:
                self.hierarchical_state_manager.transition_coordination_state(
                    ClientLifecycleState.ERROR,
                    {
                        "action": "registration_failed",
                        "error": "Server registration failed",
                        "timestamp": time.time()
                    }
                )
                return
            
            # 4. 发布客户端启动事件
            self.context.publish_event("multi_learner_client_started", {
                "client_id": self.client_id,
                "learners": list(self.learners_info.keys()),
                "timestamp": time.time()
            })
            
            self.logger.debug(f"多learner客户端启动成功: {self.client_id}")
            
        except Exception as e:
            self.logger.error(f"客户端启动失败: {self.client_id}: {e}")
            self.logger.error(f"错误详情: {traceback.format_exc()}")
            
            # 协调层状态转换到错误状态
            self.hierarchical_state_manager.transition_coordination_state(
                ClientLifecycleState.ERROR,
                {
                    "action": "startup_failed",
                    "error": str(e),
                    "timestamp": time.time()
                }
            )
            
            raise FederationError(f"Multi-learner client startup failed: {e}")
    
    def on_stop(self) -> None:
        """客户端停止时的清理"""
        try:
            self.logger.debug(f"停止多learner客户端: {self.client_id}")
            
            # 1. 停止正在进行的训练
            with self.training_lock:
                if self.is_training and self.training_thread and self.training_thread.is_alive():
                    self.logger.debug("等待训练线程结束...")
                    if self.enhanced_training_engine:
                        self.enhanced_training_engine.stop_training()
                    self.is_training = False
                    
                    # 等待训练线程结束
                    self.training_thread.join(timeout=30)
                    if self.training_thread.is_alive():
                        self.logger.warning("训练线程未能正常结束")
            
            # 2. 清理所有learner资源
            for learner_info in self.learners_info.values():
                if hasattr(learner_info.learner_instance, 'cleanup'):
                    try:
                        learner_info.learner_instance.cleanup()
                    except Exception as e:
                        self.logger.warning(f"清理learner失败 {learner_info.learner_id}: {e}")
            
            # 3. 清理增强训练引擎
            if self.enhanced_training_engine and hasattr(self.enhanced_training_engine, 'cleanup_training_environment'):
                try:
                    self.enhanced_training_engine.cleanup_training_environment()
                except Exception as e:
                    self.logger.warning(f"清理训练引擎失败: {e}")
            
            # 4. 清理状态管理器
            try:
                self.hierarchical_state_manager.cleanup()
            except Exception as e:
                self.logger.warning(f"清理状态管理器失败: {e}")
            
            # 5. 发布客户端停止事件
            self.context.publish_event("multi_learner_client_已停止", {
                "client_id": self.client_id,
                "timestamp": time.time()
            })
            
            self.logger.debug(f"多learner客户端停止完成: {self.client_id}")
            
        except Exception as e:
            self.logger.error(f"客户端停止失败: {self.client_id}: {e}")
    
    def handle_model_distribution(self, message_data: Dict[str, Any]) -> Any:
        """处理全局模型分发（支持多模型）"""
        try:
            round_id = message_data.get('metadata', {}).get('round_id', -1)
            self.logger.info(f"📥 [模型下发] Round {round_id} - 接收全局模型，准备开始训练与评估")
            
            # 提取多个模型数据
            models_data = message_data.get('data', {}).get('models', {})
            if not models_data:
                # 兼容单模型格式
                model_state = message_data.get('data', {}).get('model_state')
                if model_state:
                    models_data = {"primary_model": model_state}
            
            if not models_data:
                self.logger.warning("消息中没有模型状态数据")
                return {"status": "error", "message": "No model states"}
            
            updated_models = []
            
            # 更新各个learner的模型
            for model_key, model_state in models_data.items():
                learner_info = self._find_learner_for_model(model_key)
                
                if learner_info:
                    try:
                        if hasattr(learner_info.learner_instance, 'update_model'):
                            learner_info.learner_instance.update_model(model_state)
                        else:
                            # 兼容性处理
                            model = learner_info.learner_instance.get_model()
                            if hasattr(model, 'load_state_dict'):
                                model.load_state_dict(model_state)
                        
                        self.received_global_models[learner_info.learner_id] = learner_info.learner_instance.get_model()
                        updated_models.append(learner_info.learner_id)
                        
                    except Exception as e:
                        self.logger.error(f"更新learner模型失败 {learner_info.learner_id}: {e}")
                else:
                    self.logger.warning(f"未找到模型key对应的learner: {model_key}")
            
            # 发布模型接收事件
            self.context.publish_event("global_models_received", {
                "client_id": self.client_id,
                "round_id": round_id,
                "updated_models": updated_models,
                "timestamp": time.time()
            })
            
            return {"status": "success", "round_id": round_id, "updated_models": updated_models}
            
        except Exception as e:
            self.logger.error(f"处理模型分发失败: {e}")
            return {"status": "error", "message": str(e)}
    
    def handle_model_update(self, message_data: Dict[str, Any]) -> Any:
        """处理模型更新请求（客户端一般不处理此消息）"""
        self.logger.warning("多learner客户端收到model_update消息 - 非预期")
        return {"status": "ignored"}
    
    def handle_training_trigger(self, message_data: Dict[str, Any]) -> Any:
        """处理训练触发（多learner训练）"""
        try:
            training_params = message_data.get('data', {})
            round_id = message_data.get('metadata', {}).get('round_id', -1)
            
            self.logger.info(f"触发多learner训练 round {round_id}")
            
            # 检查当前状态是否允许开始训练
            current_coordination_state = self.hierarchical_state_manager.get_coordination_state()
            if current_coordination_state not in [ClientLifecycleState.READY, ClientLifecycleState.REGISTERED]:
                self.logger.warning(f"当前状态不允许开始训练: {current_coordination_state}")
                return {"status": "error", "message": f"Invalid state for training: {current_coordination_state}"}
            
            # 检查是否已有训练在进行
            with self.training_lock:
                if self.is_training:
                    self.logger.warning("训练已在进行中，忽略新的训练触发")
                    return {"status": "error", "message": "Training already in progress"}
            
            # 协调层状态转换：当前状态 -> TRAINING
            self.hierarchical_state_manager.transition_coordination_state(
                ClientLifecycleState.TRAINING,
                {
                    "action": "multi_learner_training_triggered",
                    "round_id": round_id,
                    "timestamp": time.time()
                }
            )
            
            # 异步启动多learner训练
            self._start_multi_learner_training_async(training_params, round_id)
            
            return {"status": "multi_learner_training_started", "round_id": round_id}
            
        except Exception as e:
            self.logger.error(f"处理多learner训练触发失败: {e}")
            
            # 协调层状态转换到错误状态
            self.hierarchical_state_manager.transition_coordination_state(
                ClientLifecycleState.ERROR,
                {
                    "action": "training_trigger_failed",
                    "error": str(e),
                    "timestamp": time.time()
                }
            )
            
            return {"status": "error", "message": str(e)}
    
    def handle_task_notification(self, message_data: Dict[str, Any]) -> Any:
        """处理任务通知（多learner持续学习场景）"""
        try:
            task_info = message_data.get('data', {})
            self.logger.debug(f"接收多learner任务通知: {task_info}")
            
            # 处理新任务（通知所有相关learner）
            self._handle_multi_learner_new_task(task_info)
            
            return {"status": "multi_learner_task_received"}
            
        except Exception as e:
            self.logger.error(f"处理多learner任务通知失败: {e}")
            return {"status": "error", "message": str(e)}
    
    # ===== 多learner特有方法 =====
    
    def _start_multi_learner_training_async(self, training_params: Dict[str, Any], round_id: int) -> None:
        """异步启动多learner训练"""
        def multi_learner_training_worker():
            try:
                with self.training_lock:
                    self.current_round = round_id
                    self.is_training = True
                
                self.logger.info(f"开始执行多learner训练 round {round_id}")
                
                # 执行多learner训练
                result = self._execute_multi_learner_training(round_id, training_params)
                
                # 发送结果
                self._send_multi_learner_training_result(result)
                
                self.logger.debug(f"多learner训练完成 round {round_id}")
                
            except Exception as e:
                self.logger.error(f"多learner训练执行失败: {e}")
                self.logger.error(f"错误详情: {traceback.format_exc()}")
                
                # 协调层状态转换到错误状态
                self.hierarchical_state_manager.transition_coordination_state(
                    ClientLifecycleState.ERROR,
                    {
                        "action": "training_execution_failed",
                        "error": str(e),
                        "round_id": round_id,
                        "timestamp": time.time()
                    }
                )
            finally:
                with self.training_lock:
                    self.is_training = False
        
        # 启动训练线程
        self.training_thread = threading.Thread(
            target=multi_learner_training_worker, 
            name=f"TrainingWorker-{self.client_id}-R{round_id}",
            daemon=True
        )
        self.training_thread.start()
    
    def _execute_multi_learner_training(self, round_id: int, training_params: Dict[str, Any] = None) -> MultiLearnerTrainingResult:
        """执行多learner训练"""
        try:
            start_time = time.time()
            
            if not self.dataloaders:
                raise FederationError("No dataloaders available for training")
            
            if not self.enhanced_training_engine:
                raise FederationError("Training engine not initialized")
            
            self.logger.debug(f"开始多learner训练 round {round_id}")
            
            # 确保训练引擎处于可执行状态
            current_state = self.enhanced_training_engine.training_state
            if current_state == TrainingPhaseState.PREPARING:
                self.logger.debug(f"训练引擎状态为 {current_state}，直接执行训练")
            elif current_state == TrainingPhaseState.RUNNING:
                self.logger.debug(f"训练引擎状态为 {current_state}，准备执行训练")
            else:
                self.logger.debug(f"训练引擎状态为 {current_state}，需要初始化...")
                
                # 如果状态需要重新初始化（UNINITIALIZED或FAILED）
                if current_state in [TrainingPhaseState.UNINITIALIZED, TrainingPhaseState.FAILED]:
                    self.enhanced_training_engine.initialize_training()
                # 如果状态是FINISHED，先转换为PREPARING状态
                elif current_state == TrainingPhaseState.FINISHED:
                    # 从FINISHED状态可以转换到PREPARING，然后由execute_training_plan自动处理
                    self.enhanced_training_engine.state_manager.transition_to(
                        TrainingPhaseState.PREPARING,
                        {
                            "action": "reset_for_next_round",
                            "timestamp": time.time()
                        }
                    )
                    self.logger.debug("状态已转换为PREPARING，由execute_training_plan处理后续状态转换")
                else:
                    # 其他状态，尝试直接初始化
                    self.enhanced_training_engine.initialize_training()
            
            # 委托给增强训练引擎执行（训练引擎管理控制层状态）
            phase_results = self.enhanced_training_engine.execute_training_plan()
            
            total_training_time = time.time() - start_time
            
            # 聚合多learner结果
            aggregated_result = self._aggregate_multi_learner_results(
                round_id, phase_results, total_training_time
            )
            
            # 记录训练历史
            self.training_history.append(aggregated_result)
            
            self.logger.debug(f"多learner训练完成 round {round_id}, 耗时 {total_training_time:.2f}s")
            return aggregated_result
            
        except Exception as e:
            self.logger.error(f"多learner训练执行失败: {e}")
            raise FederationError(f"Multi-learner training failed: {e}")
    
    def _aggregate_multi_learner_results(self, 
                                       round_id: int, 
                                       phase_results: Dict[str, PhaseResult], 
                                       total_training_time: float) -> MultiLearnerTrainingResult:
        """聚合多learner训练结果"""
        try:
            # 1. 提取和聚合模型更新
            aggregated_model_update = self._aggregate_model_updates(phase_results)
            
            # 2. 聚合训练指标
            aggregated_metrics = self._aggregate_training_metrics(phase_results)
            
            # 3. 聚合评估结果
            aggregated_evaluation_results = self._aggregate_evaluation_results(phase_results)
            
            # 4. 计算learner贡献权重
            learner_contributions = self._calculate_learner_contributions(phase_results)
            
            # 5. 计算总样本数
            total_samples = self._calculate_total_samples()
            
            # 6. 构建聚合结果
            aggregated_result = MultiLearnerTrainingResult(
                client_id=self.client_id,
                round_id=round_id,
                phase_results=phase_results,
                aggregated_model_update=aggregated_model_update,
                total_samples=total_samples,
                training_metrics=aggregated_metrics,
                evaluation_results=aggregated_evaluation_results,
                total_training_time=total_training_time,
                learner_contributions=learner_contributions,
                metadata={
                    "phase_count": len(phase_results),
                    "learner_count": len(self.learners_info),
                    "aggregation_method": "weighted_average",
                    "timestamp": time.time()
                }
            )
            
            return aggregated_result
            
        except Exception as e:
            self.logger.error(f"聚合多learner结果失败: {e}")
            raise
    
    def _aggregate_model_updates(self, phase_results: Dict[str, PhaseResult]) -> Dict[str, torch.Tensor]:
        """聚合模型更新"""
        try:
            aggregated_update = {}
            total_weight = 0.0
            
            for phase_name, phase_result in phase_results.items():
                if not phase_result.success or not phase_result.final_state:
                    continue
                
                # 获取阶段的模型更新
                phase_model_update = phase_result.final_state.get('model_update', {})
                if not phase_model_update:
                    # 尝试从learner获取模型参数
                    learner_info = self._get_learner_info_by_phase(phase_name)
                    if learner_info:
                        try:
                            model = learner_info.learner_instance.get_model()
                            if hasattr(model, 'state_dict'):
                                phase_model_update = {k: v.clone() for k, v in model.state_dict().items()}
                        except Exception as e:
                            self.logger.warning(f"获取阶段{phase_name}模型参数失败: {e}")
                            continue
                    else:
                        continue
                
                # 计算阶段权重（基于训练时间和成功的epoch数）
                phase_weight = len(phase_result.executed_epochs) * max(phase_result.execution_time, 1.0)
                total_weight += phase_weight
                
                # 加权聚合模型参数
                for param_name, param_tensor in phase_model_update.items():
                    if not isinstance(param_tensor, torch.Tensor):
                        continue
                        
                    if param_name not in aggregated_update:
                        aggregated_update[param_name] = param_tensor.clone() * phase_weight
                    else:
                        aggregated_update[param_name] += param_tensor * phase_weight
            
            # 归一化
            if total_weight > 0:
                for param_name in aggregated_update:
                    aggregated_update[param_name] /= total_weight
            
            self.logger.info(f"聚合模型更新完成: {len(phase_results)}个阶段, {len(aggregated_update)}个参数")
            return aggregated_update
            
        except Exception as e:
            self.logger.error(f"聚合模型更新失败: {e}")
            return {}
    
    def _aggregate_training_metrics(self, phase_results: Dict[str, PhaseResult]) -> Dict[str, Any]:
        """聚合训练指标"""
        try:
            aggregated_metrics = {
                "total_phases": len(phase_results),
                "successful_phases": 0,
                "total_epochs": 0,
                "average_loss": 0.0,
                "average_accuracy": 0.0,
                "phase_metrics": {}
            }
            
            total_loss = 0.0
            total_accuracy = 0.0
            loss_count = 0
            accuracy_count = 0
            
            for phase_name, phase_result in phase_results.items():
                if phase_result.success:
                    aggregated_metrics["successful_phases"] += 1
                
                aggregated_metrics["total_epochs"] += len(phase_result.executed_epochs)
                
                # 聚合每个阶段的最终指标
                final_metrics = phase_result.get_final_metrics()
                aggregated_metrics["phase_metrics"][phase_name] = final_metrics
                
                # 累积损失和准确率
                if "loss" in final_metrics and isinstance(final_metrics["loss"], (int, float)):
                    total_loss += final_metrics["loss"]
                    loss_count += 1
                
                if "accuracy" in final_metrics and isinstance(final_metrics["accuracy"], (int, float)):
                    total_accuracy += final_metrics["accuracy"]
                    accuracy_count += 1
            
            # 计算平均值
            if loss_count > 0:
                aggregated_metrics["average_loss"] = total_loss / loss_count
            
            if accuracy_count > 0:
                aggregated_metrics["average_accuracy"] = total_accuracy / accuracy_count
            
            return aggregated_metrics
            
        except Exception as e:
            self.logger.error(f"聚合训练指标失败: {e}")
            return {"error": str(e)}
    
    def _aggregate_evaluation_results(self, phase_results: Dict[str, PhaseResult]) -> Dict[str, Any]:
        """聚合评估结果"""
        try:
            aggregated_evaluation = {
                "total_evaluation_tasks": 0,
                "successful_evaluations": 0,
                "phase_evaluations": {}
            }
            
            for phase_name, phase_result in phase_results.items():
                if not phase_result.success:
                    continue
                
                # 从阶段结果中提取评估数据
                evaluation_data = phase_result.metrics.get("evaluation", {})
                if evaluation_data:
                    aggregated_evaluation["phase_evaluations"][phase_name] = evaluation_data
                    aggregated_evaluation["total_evaluation_tasks"] += len(evaluation_data)
                    aggregated_evaluation["successful_evaluations"] += len(evaluation_data)
                    
                    self.logger.debug(f"📊 提取阶段 {phase_name} 评估结果: {list(evaluation_data.keys())}")
            
            if aggregated_evaluation["total_evaluation_tasks"] > 0:
                self.logger.info(f"📊 [评估聚合] 聚合评估结果完成: {aggregated_evaluation['total_evaluation_tasks']} 个评估任务")
            else:
                self.logger.debug("📊 [评估聚合] 未发现评估结果")
            
            return aggregated_evaluation
            
        except Exception as e:
            self.logger.error(f"❌ [评估聚合] 聚合评估结果失败: {e}")
            return {"error": str(e)}
    
    def _calculate_learner_contributions(self, phase_results: Dict[str, PhaseResult]) -> Dict[str, float]:
        """计算learner贡献权重"""
        try:
            contributions = {}
            total_contribution = 0.0
            
            for phase_name, phase_result in phase_results.items():
                if not phase_result.success:
                    contributions[phase_name] = 0.0
                    continue
                
                # 根据执行的epoch数和执行时间计算贡献
                epoch_contribution = len(phase_result.executed_epochs)
                time_contribution = max(phase_result.execution_time, 1.0)
                
                # 简单的贡献计算：epoch数 × 执行时间
                phase_contribution = epoch_contribution * time_contribution
                contributions[phase_name] = phase_contribution
                total_contribution += phase_contribution
            
            # 归一化为百分比
            if total_contribution > 0:
                for phase_name in contributions:
                    contributions[phase_name] = contributions[phase_name] / total_contribution
            
            return contributions
            
        except Exception as e:
            self.logger.error(f"计算learner贡献失败: {e}")
            return {}
    
    def _calculate_total_samples(self) -> int:
        """计算总样本数"""
        try:
            # 优先从上下文获取
            data_info = self.context.get_state(f"client_{self.client_id}_multi_data_info", scope="client")
            if data_info and 'total_samples' in data_info:
                return data_info['total_samples']
            
            # 从DataLoader推断
            total_samples = 0
            for dataloader in self.dataloaders.values():
                if hasattr(dataloader, 'dataset') and hasattr(dataloader.dataset, '__len__'):
                    total_samples += len(dataloader.dataset)
            
            return total_samples
            
        except Exception as e:
            self.logger.warning(f"计算总样本数失败: {e}")
            return 0
    
    def _create_evaluation_summary(self, evaluation_results: Dict[str, Any]) -> str:
        """创建评估结果摘要字符串"""
        try:
            if not evaluation_results or not evaluation_results.get("phase_evaluations"):
                return ""
            
            summary_parts = []
            total_tasks = evaluation_results.get("total_evaluation_tasks", 0)
            
            for phase_name, phase_eval in evaluation_results.get("phase_evaluations", {}).items():
                if isinstance(phase_eval, dict):
                    accuracy_results = []
                    loss_results = []
                    
                    for task_name, task_result in phase_eval.items():
                        if isinstance(task_result, dict):
                            accuracy = task_result.get("accuracy")
                            loss = task_result.get("loss")
                            
                            if accuracy is not None:
                                accuracy_results.append(f"{accuracy:.3f}")
                            if loss is not None:
                                loss_results.append(f"{loss:.3f}")
                    
                    if accuracy_results or loss_results:
                        phase_summary = f"{phase_name}("
                        if accuracy_results:
                            phase_summary += f"acc:{','.join(accuracy_results)}"
                        if loss_results:
                            if accuracy_results:
                                phase_summary += f", "
                            phase_summary += f"loss:{','.join(loss_results)}"
                        phase_summary += ")"
                        summary_parts.append(phase_summary)
            
            if summary_parts:
                return f"{total_tasks}个评估任务 - {'; '.join(summary_parts)}"
            else:
                return f"{total_tasks}个评估任务"
                
        except Exception as e:
            self.logger.warning(f"创建评估摘要失败: {e}")
            return "评估结果摘要创建失败"
    
    def _send_multi_learner_training_result(self, result: MultiLearnerTrainingResult) -> None:
        """发送多learner训练结果到服务端"""
        try:
            result_data = {
                "client_id": result.client_id,
                "round_id": result.round_id,
                "client_type": "multi_learner",
                "aggregated_model_update": result.aggregated_model_update,
                "total_samples": result.total_samples,
                "training_metrics": result.training_metrics,
                "evaluation_results": result.evaluation_results,  # 添加评估结果
                "total_training_time": result.total_training_time,
                "learner_contributions": result.learner_contributions,
                "metadata": result.metadata
            }
            
            # 记录即将上传的评估结果摘要
            evaluation_summary = self._create_evaluation_summary(result.evaluation_results)
            if evaluation_summary:
                self.logger.info(f"📤 [模型上传] Round {result.round_id} - 上传模型与评估结果: {evaluation_summary}")
            else:
                self.logger.info(f"📤 [模型上传] Round {result.round_id} - 上传模型 (无评估结果)")
            
            # 发送聚合的模型更新消息
            self.send_message(
                target="server",
                message_type=MessageType.MODEL_UPDATE,
                data=result_data,
                metadata={
                    "round_id": result.round_id,
                    "client_id": result.client_id,
                    "client_type": "multi_learner"
                }
            )
            
            # 协调层状态转换：TRAINING -> READY（准备下一轮）
            self.hierarchical_state_manager.transition_coordination_state(
                ClientLifecycleState.READY,
                {
                    "action": "multi_learner_training_result_sent",
                    "round_id": result.round_id,
                    "timestamp": time.time()
                }
            )
            
            self.logger.info(f"多learner训练结果已发送 round {result.round_id}")
            
        except Exception as e:
            self.logger.error(f"发送多learner训练结果失败: {e}")
            
            # 协调层状态转换到错误状态
            self.hierarchical_state_manager.transition_coordination_state(
                ClientLifecycleState.ERROR,
                {
                    "action": "send_result_failed",
                    "error": str(e),
                    "timestamp": time.time()
                }
            )
    
    # ===== 组件初始化方法 =====
    
    def _initialize_multi_learner_components(self) -> None:
        """初始化多learner组件"""
        try:
            self.logger.debug("初始化多learner组件...")
            
            # 1. 创建所有learner
            learners_config = self.client_config.get('learners', {})
            if not learners_config:
                self.logger.warning("没有找到learners配置，将创建默认learner")
                learners_config = self._create_default_learners_config()
            
            for learner_id, learner_config in learners_config.items():
                try:
                    self.logger.debug(f"开始创建learner: {learner_id}, 配置: {learner_config}")
                    learner_info = self._create_learner_info(learner_id, learner_config)
                    self.learners_info[learner_id] = learner_info
                    self.logger.debug(f"创建learner成功: {learner_id} ({learner_info.learner_type})")
                except Exception as e:
                    self.logger.error(f"创建learner失败 {learner_id}: {e}")
                    self.logger.error(f"错误详情: {traceback.format_exc()}")
            
            # 2. 创建所有dataloader
            dataloaders_config = self.client_config.get('dataloaders', {})
            if not dataloaders_config:
                self.logger.warning("没有找到dataloaders配置，将创建默认dataloader")
                dataloaders_config = self._create_default_dataloaders_config()
            
            for dataloader_id, dataloader_config in dataloaders_config.items():
                try:
                    dataloader = self._create_dataloader(dataloader_id, dataloader_config)
                    self.dataloaders[dataloader_id] = dataloader
                    self.logger.info(f"创建dataloader: {dataloader_id}")
                except Exception as e:
                    self.logger.error(f"创建dataloader失败 {dataloader_id}: {e}")
            
            if not self.learners_info:
                raise FederationError("No learners created successfully")
            
            if not self.dataloaders:
                self.logger.warning("No dataloaders created, using mock data")
                self.dataloaders["default"] = self._create_mock_data()
            
            self.logger.debug(f"多learner组件初始化完成: {len(self.learners_info)} learners, {len(self.dataloaders)} dataloaders")
            
        except Exception as e:
            self.logger.error(f"初始化多learner组件失败: {e}")
            raise
    
    def _create_default_learners_config(self) -> Dict[str, Any]:
        """创建默认learner配置"""
        return {
            "default_learner": {
                "class": "l2p",
                "model": {
                    "type": "SimpleMLP",
                    "input_dim": 784,
                    "hidden_dims": [128, 64],
                    "output_dim": 10
                },
                "optimizer": {
                    "type": "Adam",
                    "lr": 0.001
                },
                "dataloader": "default",
                "scheduler": "default_scheduler",
                "priority": 0,
                "enabled": True
            }
        }
    
    def _create_default_dataloaders_config(self) -> Dict[str, Any]:
        """创建默认dataloader配置"""
        return {
            "default": {
                "batch_size": 32,
                "shuffle": True,
                "num_workers": 0,
                "drop_last": False
            }
        }
    
    def _create_learner_info(self, learner_id: str, learner_config: Dict[str, Any]) -> LearnerInfo:
        """创建learner信息"""
        try:
            # 创建learner实例
            learner_instance = self._create_single_learner(learner_config, self.context)
            
            # 构建learner信息
            learner_info = LearnerInfo(
                learner_id=learner_id,
                learner_type=learner_config.get('type', 'UnknownLearner'),
                learner_instance=learner_instance,
                dataloader_id=learner_config.get('dataloader', f"{learner_id}_dataloader"),
                scheduler_id=learner_config.get('scheduler', f"{learner_id}_scheduler"),
                priority=learner_config.get('priority', 0),
                is_active=learner_config.get('enabled', True)
            )
            
            return learner_info
            
        except Exception as e:
            self.logger.error(f"创建learner信息失败 {learner_id}: {e}")
            raise
    
    def _create_single_learner(self, learner_config: Dict[str, Any], context: ExecutionContext) -> BaseLearner:
        """创建单个learner"""
        try:
            # 尝试使用组件注册表创建learner
            try:
                from ...registry.component_composer import ComponentComposer
                from ...registry import registry
                
                composer = ComponentComposer(registry)
                config = OmegaConf.create({'learner': learner_config})
                learner = composer.create_learner(config, context)
                return learner
            except Exception as e:
                self.logger.warning(f"使用组件注册表创建learner失败: {e}, 尝试创建mock learner")
                return self._create_mock_learner(learner_config)
            
        except Exception as e:
            self.logger.error(f"创建learner失败: {e}")
            raise
    
    def _create_mock_learner(self, learner_config: Dict[str, Any]) -> BaseLearner:
        """创建模拟learner"""
        class MockLearner:
            def __init__(self, config):
                self.config = config
                self.metrics = {"loss": 1.0, "accuracy": 0.0}
                self.model = self._create_mock_model()
            
            def _create_mock_model(self):
                """创建模拟模型"""
                import torch.nn as nn
                return nn.Sequential(
                    nn.Linear(784, 128),
                    nn.ReLU(),
                    nn.Linear(128, 10)
                )
            
            def get_model(self):
                return self.model
            
            def train_epoch(self, dataloader, epoch: int) -> Dict[str, float]:
                # 模拟训练
                import time
                time.sleep(0.1)
                self.metrics["loss"] = max(0.1, self.metrics["loss"] * 0.95)
                self.metrics["accuracy"] = min(0.95, self.metrics["accuracy"] + 0.02)
                return self.metrics.copy()
            
            def get_state(self) -> Dict[str, Any]:
                return {"metrics": self.metrics}
            
            def set_state(self, state: Dict[str, Any]) -> None:
                if "metrics" in state:
                    self.metrics.update(state["metrics"])
            
            def update_model(self, model_state):
                if hasattr(self.model, 'load_state_dict'):
                    self.model.load_state_dict(model_state)
        
        return MockLearner(learner_config)
    
    def _create_dataloader(self, dataloader_id: str, dataloader_config: Dict[str, Any]) -> DataLoader:
        """创建dataloader"""
        try:
            # 尝试使用DataLoaderFactory
            try:
                from ...config.config_manager import DataLoaderFactory
                
                factory = DataLoaderFactory({dataloader_id: dataloader_config})
                dataloader = factory.create_dataloader(dataloader_id, dataloader_config)
                return dataloader
            except Exception as e:
                self.logger.warning(f"使用DataLoaderFactory创建dataloader失败 {dataloader_id}: {e}, 使用mock数据")
                return self._create_mock_data()
            
        except Exception as e:
            self.logger.warning(f"创建dataloader失败 {dataloader_id}: {e}, 使用mock数据")
            return self._create_mock_data()
    
    def _create_mock_data(self) -> DataLoader:
        """创建模拟训练数据"""
        try:
            # 简单的模拟数据
            num_samples = 100
            input_dim = 784
            num_classes = 10
            
            X = torch.randn(num_samples, input_dim)
            y = torch.randint(0, num_classes, (num_samples,))
            
            dataset = TensorDataset(X, y)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
            
            self.logger.debug(f"创建模拟数据集: {num_samples} samples")
            return dataloader
        except Exception as e:
            self.logger.error(f"创建模拟数据失败: {e}")
            raise
    
    def _build_enhanced_training_config(self, client_config: DictConfig) -> Dict[str, Any]:
        """构建增强训练引擎配置"""
        enhanced_config = {
            "project": {
                "name": f"client_{self.client_id}_multi_learner",
                "output_dir": f"./outputs/client_{self.client_id}"
            },
            
            # 从客户端配置中提取相关配置
            "dataloaders": client_config.get('dataloaders', {}),
            "learners": client_config.get('learners', {}),
            "schedulers": client_config.get('schedulers', {}),
            "training_plan": client_config.get('training_plan', {}),
            "state_transfer": client_config.get('state_transfer', {}),
            "hooks": client_config.get('hooks', {}),
            "system": client_config.get('system', {}),
            
            # 添加评估相关配置
            "test_datas": client_config.get('test_datas', {}),
            "evaluators": client_config.get('evaluators', {}),
            "evaluation": client_config.get('evaluation', {})
        }
        
        # 如果没有训练计划，创建默认的
        if not enhanced_config.get('training_plan'):
            enhanced_config['training_plan'] = self._create_default_training_plan()
        
        return enhanced_config
    
    def _create_default_training_plan(self) -> Dict[str, Any]:
        """创建默认训练计划"""
        self.logger.debug(f"开始创建默认训练计划，当前learners_info: {list(self.learners_info.keys())}")
        
        learner_ids = list(self.learners_info.keys())
        
        if not learner_ids:
            self.logger.error("没有可用的learners用于创建训练计划！")
            raise FederationError("No learners available for training plan")
        
        # 创建简单的顺序训练计划
        phases = []
        epoch_count = 1
        epochs_per_phase = 5
        
        for i, learner_id in enumerate(learner_ids):
            learner_info = self.learners_info[learner_id]
            self.logger.debug(f"为learner {learner_id} 创建训练阶段，scheduler={learner_info.scheduler_id}")
            
            phase_epochs = list(range(epoch_count, epoch_count + epochs_per_phase))
            
            phase = {
                "name": f"phase_{learner_id}",
                "description": f"Training phase for {learner_id}",
                "epochs": phase_epochs,
                "learner": learner_id,
                "scheduler": learner_info.scheduler_id,
                "priority": learner_info.priority,
                "execution_mode": "sequential"
            }
            
            # 添加继承关系（除了第一个阶段）
            if i > 0:
                phase["inherit_from"] = [phases[i-1]["name"]]
            
            phases.append(phase)
            epoch_count += epochs_per_phase
        
        training_plan = {
            "total_epochs": epoch_count - 1,
            "execution_strategy": "sequential",
            "phases": phases
        }
        
        self.logger.debug(f"创建的训练计划: {training_plan}")
        return training_plan
    
    # ===== 辅助方法 =====
    
    def _find_learner_for_model(self, model_key: str) -> Optional[LearnerInfo]:
        """根据模型key找到对应的learner"""
        # 尝试直接匹配learner_id
        if model_key in self.learners_info:
            return self.learners_info[model_key]
        
        # 尝试根据模型key的映射规则查找
        model_mappings = self.client_config.get('model_mappings', {})
        if model_key in model_mappings:
            learner_id = model_mappings[model_key]
            return self.learners_info.get(learner_id)
        
        # 默认策略：如果只有一个learner，使用它
        if len(self.learners_info) == 1:
            return list(self.learners_info.values())[0]
        
        # 默认策略：查找主learner
        for learner_info in self.learners_info.values():
            if "primary" in learner_info.learner_id.lower() or "default" in learner_info.learner_id.lower():
                return learner_info
        
        # 如果都没找到，返回第一个
        if self.learners_info:
            return list(self.learners_info.values())[0]
        
        return None
    
    def _get_learner_info_by_phase(self, phase_name: str) -> Optional[LearnerInfo]:
        """根据阶段名称获取learner信息"""
        # 尝试从阶段名称提取learner_id
        for learner_id in self.learners_info:
            if learner_id in phase_name:
                return self.learners_info[learner_id]
        
        # 如果提取不到，返回默认learner
        return self._find_learner_for_model("default")
    
    def _load_multi_learner_data(self) -> None:
        """加载多learner数据"""
        try:
            dataset_config = self.client_config.get('dataset', {})
            
            if not dataset_config:
                self.logger.warning("没有数据集配置，数据将在dataloader中加载")
                return
            
            # 为每个dataloader加载数据（如果还没有加载的话）
            total_samples = 0
            for dataloader_id, dataloader in self.dataloaders.items():
                if hasattr(dataloader, 'dataset') and hasattr(dataloader.dataset, '__len__'):
                    num_samples = len(dataloader.dataset)
                    total_samples += num_samples
                    self.logger.debug(f"Dataloader {dataloader_id} 已有 {num_samples} 样本")
                else:
                    self.logger.debug(f"Dataloader {dataloader_id} 需要数据加载")
            
            # 存储数据集信息到上下文
            self.context.set_state(f"client_{self.client_id}_multi_data_info", {
                "total_samples": total_samples,
                "dataloader_count": len(self.dataloaders),
                "learner_count": len(self.learners_info),
                "dataset_name": dataset_config.get('name', 'unknown')
            }, scope="client")
            
        except Exception as e:
            self.logger.error(f"加载多learner数据失败: {e}")
    
    def _register_to_server(self) -> bool:
        """向服务端注册（包含多learner信息）"""
        try:
            registration_data = {
                "client_id": self.client_id,
                "client_type": "multi_learner",
                "learners": self._get_learners_capabilities(),
                "data_info": self.context.get_state(f"client_{self.client_id}_multi_data_info", scope="client")
            }
            
            # 发送注册消息
            response = self.send_message(
                target="server",
                message_type=MessageType.REGISTRATION,
                data=registration_data,
                expect_response=True,
                timeout=30.0
            )
            
            if response and response.get('status') == 'registered':
                self.logger.debug(f"多learner客户端注册成功: {self.client_id}")
                return True
            else:
                self.logger.warning(f"注册响应: {response}")
                return False
                
        except Exception as e:
            self.logger.warning(f"注册失败: {e}")
            return False
    
    def _get_learners_capabilities(self) -> Dict[str, Dict[str, Any]]:
        """获取所有learner的能力"""
        capabilities = {}
        
        for learner_id, learner_info in self.learners_info.items():
            learner_instance = learner_info.learner_instance
            
            try:
                model = learner_instance.get_model()
                model_name = type(model).__name__ if model else "Unknown"
            except:
                model_name = "Unknown"
            
            capabilities[learner_id] = {
                "learner_type": learner_info.learner_type,
                "model_architecture": model_name,
                "device": str(getattr(learner_instance, 'device', 'cpu')),
                "supported_tasks": getattr(learner_instance, 'supported_tasks', []),
                "priority": learner_info.priority,
                "is_active": learner_info.is_active,
                "dataloader_id": learner_info.dataloader_id,
                "scheduler_id": learner_info.scheduler_id
            }
        
        return capabilities
    
    def _handle_multi_learner_new_task(self, task_info: Dict[str, Any]) -> None:
        """处理多learner新任务"""
        try:
            task_id = task_info.get('task_id')
            task_type = task_info.get('task_type', 'classification')
            
            self.logger.debug(f"处理多learner新任务: {task_id} (type: {task_type})")
            
            # 通知所有相关learner准备新任务
            affected_learners = []
            for learner_id, learner_info in self.learners_info.items():
                if hasattr(learner_info.learner_instance, 'prepare_for_task'):
                    try:
                        learner_info.learner_instance.prepare_for_task(task_info)
                        affected_learners.append(learner_id)
                    except Exception as e:
                        self.logger.warning(f"learner {learner_id} 准备新任务失败: {e}")
            
            # 发布任务接收事件
            self.context.publish_event("multi_learner_new_task_received", {
                "client_id": self.client_id,
                "task_id": task_id,
                "task_type": task_type,
                "affected_learners": affected_learners,
                "timestamp": time.time()
            })
            
        except Exception as e:
            self.logger.error(f"处理多learner新任务失败: {e}")
    
    def _register_state_callbacks(self):
        """注册状态变化回调"""
        try:
            # 注册协调层状态回调
            self.hierarchical_state_manager.register_coordination_callback(
                self._on_coordination_state_change,
                callback_id="client_coordination_callback"
            )
            
            # 注册控制层状态回调
            self.hierarchical_state_manager.register_control_callback(
                self._on_control_state_change,
                callback_id="client_control_callback"
            )
            
            self.logger.debug("状态回调注册完成")
            
        except Exception as e:
            self.logger.error(f"注册状态回调失败: {e}")
    
    def _on_coordination_state_change(self, old_state: ClientLifecycleState, 
                                    new_state: ClientLifecycleState, 
                                    metadata: Dict[str, Any]):
        """协调层状态变化回调"""
        self.logger.debug(
            f"客户端协调层状态变化: {self.client_id} {old_state.name} -> {new_state.name}"
        )
        
        # 发布事件到执行上下文
        self.context.publish_event("client_coordination_state_changed", {
            "client_id": self.client_id,
            "old_state": old_state.name,
            "new_state": new_state.name,
            "metadata": metadata,
            "timestamp": metadata.get("timestamp", time.time())
        })
    
    def _on_control_state_change(self, old_state: TrainingPhaseState,
                               new_state: TrainingPhaseState,
                               metadata: Dict[str, Any]):
        """控制层状态变化回调"""
        self.logger.debug(
            f"客户端控制层状态变化: {self.client_id} {old_state.name} -> {new_state.name}"
        )
        
        # 根据控制层状态执行相应操作
        if new_state == TrainingPhaseState.FINISHED:
            self.logger.debug("训练完成，准备发送结果")
        elif new_state == TrainingPhaseState.FAILED:
            self.logger.error("训练失败")
    
    def _register_hooks(self, hooks_config: Dict[str, Any]) -> None:
        """注册Hook到增强训练引擎"""
        try:
            if not hooks_config:
                self.logger.debug("没有配置hooks")
                return
            
            # Hook注册逻辑可以委托给enhanced_training_engine处理
            if self.enhanced_training_engine and hasattr(self.enhanced_training_engine, 'register_training_hooks'):
                # 这里可以根据hooks_config创建具体的hook实例
                self.logger.debug("Hook注册委托给增强训练引擎")
            else:
                self.logger.debug("训练引擎不支持hook注册")
            
        except Exception as e:
            self.logger.error(f"注册hooks失败: {e}")
    
    def _create_execution_context(self, config: DictConfig) -> ExecutionContext:
        """创建执行上下文"""
        try:
            context_config = config.get('context', {})
            experiment_id = f"multi_learner_client_experiment_{self.client_id}"
            
            context = ExecutionContext(
                config=OmegaConf.create(context_config),
                experiment_id=experiment_id
            )
            
            # 存储完整配置
            context._client_config = config
            
            return context
        except Exception as e:
            self.logger.error(f"创建执行上下文失败: {e}")
            raise
    
    # ===== 公共接口方法 =====
    
    def get_client_status(self) -> Dict[str, Any]:
        """获取多learner客户端状态"""
        try:
            overall_status = self.hierarchical_state_manager.get_overall_status()
            
            status = {
                "client_id": self.client_id,
                "client_type": "multi_learner",
                "current_round": self.current_round,
                "is_training": self.is_training,
                "is_running": self.is_running(),
                "is_connected": self.is_connected(),
                "learner_count": len(self.learners_info),
                "dataloader_count": len(self.dataloaders),
                "active_learners": [
                    learner_id for learner_id, info in self.learners_info.items() 
                    if info.is_active
                ],
                "training_history_length": len(self.training_history),
                "state_management": overall_status,
                "training_thread_alive": self.training_thread.is_alive() if self.training_thread else False
            }
            
            return status
        except Exception as e:
            self.logger.error(f"获取客户端状态失败: {e}")
            return {
                "client_id": self.client_id,
                "error": str(e),
                "timestamp": time.time()
            }
    
    def cleanup_client(self) -> None:
        """清理多learner客户端资源"""
        try:
            self.logger.info(f"清理多learner客户端: {self.client_id}")
            
            # 协调层状态转换到完成状态
            try:
                current_state = self.hierarchical_state_manager.get_coordination_state()
                if current_state != ClientLifecycleState.COMPLETED:
                    self.hierarchical_state_manager.transition_coordination_state(
                        ClientLifecycleState.COMPLETED,
                        {
                            "action": "multi_learner_client_cleanup",
                            "timestamp": time.time()
                        }
                    )
            except Exception as e:
                self.logger.warning(f"转换到完成状态失败: {e}")
            
            # 停止通信
            self.stop()
            
            # 停止训练
            with self.training_lock:
                if self.is_training and self.training_thread and self.training_thread.is_alive():
                    if self.enhanced_training_engine:
                        self.enhanced_training_engine.stop_training()
                    self.is_training = False
                    self.training_thread.join(timeout=10)
            
            # 重置客户端状态
            self.current_round = 0
            self.is_training = False
            self.received_global_models.clear()
            self.training_history.clear()
            
            # 清理所有learner
            for learner_info in self.learners_info.values():
                if hasattr(learner_info.learner_instance, 'cleanup'):
                    try:
                        learner_info.learner_instance.cleanup()
                    except Exception as e:
                        self.logger.warning(f"清理learner失败 {learner_info.learner_id}: {e}")
            
            # 清理增强训练引擎
            if self.enhanced_training_engine and hasattr(self.enhanced_training_engine, 'cleanup_training_environment'):
                try:
                    self.enhanced_training_engine.cleanup_training_environment()
                except Exception as e:
                    self.logger.warning(f"清理训练引擎失败: {e}")
            
            # 清理层级状态管理器
            try:
                self.hierarchical_state_manager.cleanup()
            except Exception as e:
                self.logger.warning(f"清理状态管理器失败: {e}")
            
            # 清理数据结构
            self.learners_info.clear()
            self.dataloaders.clear()
            
            # 清理上下文状态
            try:
                self.context.clear_scope("client")
            except Exception as e:
                self.logger.warning(f"清理context失败: {e}")
            
            self.logger.info(f"多learner客户端清理完成: {self.client_id}")
            
        except Exception as e:
            self.logger.error(f"清理多learner客户端失败: {e}")
    
    @classmethod
    def create_from_config(cls, config: DictConfig) -> 'MultiLearnerFederatedClient':
        """从配置创建多learner客户端实例"""
        try:
            client_id = config.get('client', {}).get('id', 'multi_learner_client_0')
            logger.debug(f"从配置创建多learner客户端: {client_id}")
            
            client = cls(client_id, config)
            
            logger.debug(f"多learner客户端创建成功: {client_id}")
            return client
            
        except Exception as e:
            logger.error(f"从配置创建多learner客户端失败: {e}")
            raise FederationError(f"Multi-learner client creation failed: {e}")


# 向后兼容的别名
MultiLearnerClient = MultiLearnerFederatedClient