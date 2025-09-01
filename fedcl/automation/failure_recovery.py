# fedcl/automation/failure_recovery.py
"""
故障恢复机制

处理真联邦（多机）和伪联邦环境下的各种故障情况：
- 网络连接故障
- 客户端节点崩溃  
- 服务器故障
- 内存不足
- 训练发散
"""

import json
import pickle
import time
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Union
import shutil
import psutil

import torch
from loguru import logger

from .communication import TransparentCommunication


class FailureType(Enum):
    """故障类型"""
    NETWORK_TIMEOUT = "network_timeout"
    CLIENT_DISCONNECT = "client_disconnect"
    SERVER_UNAVAILABLE = "server_unavailable"
    MEMORY_OVERFLOW = "memory_overflow"
    TRAINING_DIVERGENCE = "training_divergence"
    MODEL_CORRUPTION = "model_corruption"
    COMMUNICATION_ERROR = "communication_error"
    RESOURCE_EXHAUSTION = "resource_exhaustion"


@dataclass
class FailureEvent:
    """故障事件"""
    failure_type: FailureType
    component: str  # 故障组件 (client_id, server, communication等)
    timestamp: float
    details: Dict[str, Any]
    severity: str = "medium"  # low, medium, high, critical
    recovery_attempts: int = 0
    resolved: bool = False


@dataclass
class Checkpoint:
    """检查点"""
    round_number: int
    timestamp: float
    global_model_state: Optional[Dict[str, Any]]
    client_states: Dict[str, Dict[str, Any]]
    training_metrics: Dict[str, Any]
    system_state: Dict[str, Any]
    checkpoint_path: str


class BaseRecoveryStrategy(ABC):
    """恢复策略基类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logger.bind(component="RecoveryStrategy")
    
    @abstractmethod
    def can_handle(self, failure: FailureEvent) -> bool:
        """判断是否能处理该故障"""
        pass
    
    @abstractmethod
    def recover(self, failure: FailureEvent, context: Dict[str, Any]) -> bool:
        """执行恢复操作"""
        pass
    
    def get_priority(self) -> int:
        """获取策略优先级（数字越小优先级越高）"""
        return 50


class NetworkTimeoutRecoveryStrategy(BaseRecoveryStrategy):
    """网络超时恢复策略"""
    
    def can_handle(self, failure: FailureEvent) -> bool:
        return failure.failure_type == FailureType.NETWORK_TIMEOUT
    
    def recover(self, failure: FailureEvent, context: Dict[str, Any]) -> bool:
        """恢复网络超时"""
        self.logger.info(f"🔄 尝试恢复网络超时故障: {failure.component}")
        
        max_retries = self.config.get("max_network_retries", 3)
        retry_interval = self.config.get("retry_interval", 5.0)
        
        for attempt in range(max_retries):
            self.logger.info(f"📡 重试网络连接 {attempt + 1}/{max_retries}")
            
            # 等待一段时间再重试
            time.sleep(retry_interval * (attempt + 1))
            
            # 尝试重新建立连接
            communication = context.get("communication")
            if communication and hasattr(communication, "backend"):
                try:
                    # 重启通信后端
                    communication.backend.stop()
                    time.sleep(1)
                    if communication.backend.start():
                        self.logger.info("✅ 网络连接恢复成功")
                        return True
                except Exception as e:
                    self.logger.warning(f"重连尝试失败: {e}")
        
        self.logger.error("❌ 网络连接恢复失败")
        return False
    
    def get_priority(self) -> int:
        return 10


class ClientDisconnectRecoveryStrategy(BaseRecoveryStrategy):
    """客户端断连恢复策略"""
    
    def can_handle(self, failure: FailureEvent) -> bool:
        return failure.failure_type == FailureType.CLIENT_DISCONNECT
    
    def recover(self, failure: FailureEvent, context: Dict[str, Any]) -> bool:
        """处理客户端断连"""
        client_id = failure.component
        self.logger.info(f"👤 处理客户端断连: {client_id}")
        
        # 从活跃客户端列表中移除
        active_clients = context.get("active_clients", set())
        if client_id in active_clients:
            active_clients.remove(client_id)
            self.logger.info(f"📝 已从活跃列表移除客户端: {client_id}")
        
        # 检查是否还有足够的客户端继续训练
        min_clients = self.config.get("min_clients_for_training", 2)
        remaining_clients = len(active_clients)
        
        if remaining_clients >= min_clients:
            self.logger.info(f"✅ 剩余 {remaining_clients} 个客户端，继续训练")
            return True
        else:
            self.logger.warning(f"⚠️ 客户端数量不足 ({remaining_clients}/{min_clients})，暂停训练")
            # 标记需要等待更多客户端
            context["waiting_for_clients"] = True
            return False
    
    def get_priority(self) -> int:
        return 20


class MemoryOverflowRecoveryStrategy(BaseRecoveryStrategy):
    """内存溢出恢复策略"""
    
    def can_handle(self, failure: FailureEvent) -> bool:
        return failure.failure_type == FailureType.MEMORY_OVERFLOW
    
    def recover(self, failure: FailureEvent, context: Dict[str, Any]) -> bool:
        """恢复内存溢出"""
        self.logger.info("💾 处理内存溢出故障")
        
        try:
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.logger.info("🧹 已清理GPU缓存")
            
            # 减小批次大小
            trainer = context.get("trainer")
            if trainer and hasattr(trainer, "config"):
                current_batch_size = trainer.config.get("batch_size", 32)
                new_batch_size = max(1, current_batch_size // 2)
                trainer.config["batch_size"] = new_batch_size
                self.logger.info(f"📉 减小批次大小: {current_batch_size} → {new_batch_size}")
            
            # 建议减少模型大小或使用梯度累积
            recommendations = [
                "考虑使用梯度累积",
                "减少模型参数数量",
                "使用混合精度训练",
                "启用数据并行"
            ]
            
            for rec in recommendations:
                self.logger.info(f"💡 建议: {rec}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"内存溢出恢复失败: {e}")
            return False
    
    def get_priority(self) -> int:
        return 30


class TrainingDivergenceRecoveryStrategy(BaseRecoveryStrategy):
    """训练发散恢复策略"""
    
    def can_handle(self, failure: FailureEvent) -> bool:
        return failure.failure_type == FailureType.TRAINING_DIVERGENCE
    
    def recover(self, failure: FailureEvent, context: Dict[str, Any]) -> bool:
        """恢复训练发散"""
        self.logger.info("📈 处理训练发散故障")
        
        try:
            # 降低学习率
            trainer = context.get("trainer")
            if trainer and hasattr(trainer, "config"):
                current_lr = trainer.config.get("learning_rate", 0.01)
                new_lr = current_lr * 0.5
                trainer.config["learning_rate"] = new_lr
                self.logger.info(f"📉 降低学习率: {current_lr} → {new_lr}")
            
            # 回滚到最近的稳定检查点
            checkpoint_manager = context.get("checkpoint_manager")
            if checkpoint_manager:
                latest_checkpoint = checkpoint_manager.get_latest_checkpoint()
                if latest_checkpoint:
                    checkpoint_manager.restore_checkpoint(latest_checkpoint.checkpoint_path)
                    self.logger.info(f"🔄 回滚到检查点: round {latest_checkpoint.round_number}")
                    return True
            
            # 重新初始化模型（最后手段）
            self.logger.warning("⚠️ 未找到可用检查点，建议重新初始化模型")
            return False
            
        except Exception as e:
            self.logger.error(f"训练发散恢复失败: {e}")
            return False
    
    def get_priority(self) -> int:
        return 40


class CheckpointManager:
    """检查点管理器"""
    
    def __init__(self, checkpoint_dir: Union[str, Path], max_checkpoints: int = 5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.logger = logger.bind(component="CheckpointManager")
        
        self.checkpoints: List[Checkpoint] = []
        self._load_existing_checkpoints()
    
    def create_checkpoint(
        self,
        round_number: int,
        global_model_state: Optional[Dict[str, Any]] = None,
        client_states: Optional[Dict[str, Dict[str, Any]]] = None,
        training_metrics: Optional[Dict[str, Any]] = None,
        system_state: Optional[Dict[str, Any]] = None
    ) -> Checkpoint:
        """创建检查点"""
        timestamp = time.time()
        checkpoint_filename = f"checkpoint_round_{round_number}_{int(timestamp)}.pkl"
        checkpoint_path = self.checkpoint_dir / checkpoint_filename
        
        checkpoint = Checkpoint(
            round_number=round_number,
            timestamp=timestamp,
            global_model_state=global_model_state,
            client_states=client_states or {},
            training_metrics=training_metrics or {},
            system_state=system_state or {},
            checkpoint_path=str(checkpoint_path)
        )
        
        # 保存检查点到文件
        try:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint, f)
            
            self.checkpoints.append(checkpoint)
            self.checkpoints.sort(key=lambda x: x.round_number)
            
            # 清理旧检查点
            self._cleanup_old_checkpoints()
            
            self.logger.info(f"💾 检查点已创建: round {round_number}")
            return checkpoint
            
        except Exception as e:
            self.logger.error(f"创建检查点失败: {e}")
            raise
    
    def restore_checkpoint(self, checkpoint_path: str) -> Optional[Checkpoint]:
        """恢复检查点"""
        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
            
            self.logger.info(f"🔄 检查点已恢复: round {checkpoint.round_number}")
            return checkpoint
            
        except Exception as e:
            self.logger.error(f"恢复检查点失败: {e}")
            return None
    
    def get_latest_checkpoint(self) -> Optional[Checkpoint]:
        """获取最新检查点"""
        if self.checkpoints:
            return max(self.checkpoints, key=lambda x: x.round_number)
        return None
    
    def get_checkpoint_by_round(self, round_number: int) -> Optional[Checkpoint]:
        """根据轮次获取检查点"""
        for checkpoint in self.checkpoints:
            if checkpoint.round_number == round_number:
                return checkpoint
        return None
    
    def _load_existing_checkpoints(self):
        """加载现有检查点"""
        for checkpoint_file in self.checkpoint_dir.glob("checkpoint_*.pkl"):
            try:
                with open(checkpoint_file, 'rb') as f:
                    checkpoint = pickle.load(f)
                self.checkpoints.append(checkpoint)
            except Exception as e:
                self.logger.warning(f"加载检查点失败 {checkpoint_file}: {e}")
        
        self.checkpoints.sort(key=lambda x: x.round_number)
        self.logger.info(f"📂 加载了 {len(self.checkpoints)} 个现有检查点")
    
    def _cleanup_old_checkpoints(self):
        """清理旧检查点"""
        while len(self.checkpoints) > self.max_checkpoints:
            old_checkpoint = self.checkpoints.pop(0)
            try:
                Path(old_checkpoint.checkpoint_path).unlink()
                self.logger.debug(f"🗑️ 已删除旧检查点: round {old_checkpoint.round_number}")
            except Exception as e:
                self.logger.warning(f"删除旧检查点失败: {e}")
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """列出所有检查点"""
        return [
            {
                "round_number": cp.round_number,
                "timestamp": cp.timestamp,
                "checkpoint_path": cp.checkpoint_path,
                "has_global_model": cp.global_model_state is not None,
                "num_clients": len(cp.client_states)
            }
            for cp in self.checkpoints
        ]


class FailureDetector:
    """故障检测器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logger.bind(component="FailureDetector")
        self.active_monitors: Dict[str, threading.Thread] = {}
        self.running = False
        
    def start_monitoring(self):
        """启动监控"""
        self.running = True
        
        # 启动各种监控线程
        self._start_network_monitor()
        self._start_memory_monitor()
        self._start_training_monitor()
        
        self.logger.info("🔍 故障检测器启动")
    
    def stop_monitoring(self):
        """停止监控"""
        self.running = False
        
        # 等待监控线程结束
        for monitor_name, thread in self.active_monitors.items():
            thread.join(timeout=5)
            
        self.active_monitors.clear()
        self.logger.info("⏹️ 故障检测器停止")
    
    def _start_network_monitor(self):
        """启动网络监控"""
        def network_monitor():
            while self.running:
                try:
                    # 检查网络连接状态
                    # 这里可以实现具体的网络检测逻辑
                    time.sleep(self.config.get("network_check_interval", 10))
                except Exception as e:
                    self.logger.error(f"网络监控错误: {e}")
        
        thread = threading.Thread(target=network_monitor, daemon=True)
        thread.start()
        self.active_monitors["network"] = thread
    
    def _start_memory_monitor(self):
        """启动内存监控"""
        def memory_monitor():
            memory_threshold = self.config.get("memory_threshold", 0.9)
            
            while self.running:
                try:
                    # 检查系统内存使用率
                    memory_percent = psutil.virtual_memory().percent / 100.0
                    
                    if memory_percent > memory_threshold:
                        self.logger.warning(f"⚠️ 内存使用率过高: {memory_percent:.1%}")
                        # 触发内存溢出故障事件
                        # 这里可以触发回调函数
                    
                    # 检查GPU内存（如果可用）
                    if torch.cuda.is_available():
                        for device_id in range(torch.cuda.device_count()):
                            gpu_memory = torch.cuda.memory_usage(device_id)
                            if gpu_memory > memory_threshold:
                                self.logger.warning(f"⚠️ GPU {device_id} 内存使用率过高: {gpu_memory:.1%}")
                    
                    time.sleep(self.config.get("memory_check_interval", 5))
                    
                except Exception as e:
                    self.logger.error(f"内存监控错误: {e}")
        
        thread = threading.Thread(target=memory_monitor, daemon=True)
        thread.start()
        self.active_monitors["memory"] = thread
    
    def _start_training_monitor(self):
        """启动训练监控"""
        def training_monitor():
            while self.running:
                try:
                    # 检查训练是否发散
                    # 这里可以实现训练状态检测逻辑
                    time.sleep(self.config.get("training_check_interval", 30))
                except Exception as e:
                    self.logger.error(f"训练监控错误: {e}")
        
        thread = threading.Thread(target=training_monitor, daemon=True)
        thread.start()
        self.active_monitors["training"] = thread


class FailureRecoveryManager:
    """
    故障恢复管理器
    
    统一管理故障检测、恢复策略和检查点系统
    """
    
    def __init__(
        self,
        checkpoint_dir: Union[str, Path] = "./checkpoints",
        config: Optional[Dict[str, Any]] = None
    ):
        self.config = config or {}
        self.logger = logger.bind(component="FailureRecoveryManager")
        
        # 初始化组件
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        self.failure_detector = FailureDetector(self.config)
        
        # 注册恢复策略
        self.recovery_strategies: List[BaseRecoveryStrategy] = []
        self._register_default_strategies()
        
        # 故障事件历史
        self.failure_history: List[FailureEvent] = []
        self.recovery_callbacks: List[Callable] = []
        
    def _register_default_strategies(self):
        """注册默认恢复策略"""
        strategies = [
            NetworkTimeoutRecoveryStrategy(self.config),
            ClientDisconnectRecoveryStrategy(self.config),
            MemoryOverflowRecoveryStrategy(self.config),
            TrainingDivergenceRecoveryStrategy(self.config)
        ]
        
        # 按优先级排序
        strategies.sort(key=lambda x: x.get_priority())
        self.recovery_strategies.extend(strategies)
        
        self.logger.info(f"📋 已注册 {len(strategies)} 个恢复策略")
    
    def register_strategy(self, strategy: BaseRecoveryStrategy):
        """注册自定义恢复策略"""
        self.recovery_strategies.append(strategy)
        self.recovery_strategies.sort(key=lambda x: x.get_priority())
        self.logger.info(f"✅ 注册新恢复策略: {strategy.__class__.__name__}")
    
    def register_recovery_callback(self, callback: Callable):
        """注册恢复回调函数"""
        self.recovery_callbacks.append(callback)
    
    def detect_failure(self, failure_type: FailureType, component: str, details: Dict[str, Any]) -> FailureEvent:
        """检测并记录故障"""
        failure = FailureEvent(
            failure_type=failure_type,
            component=component,
            timestamp=time.time(),
            details=details,
            severity=details.get("severity", "medium")
        )
        
        self.failure_history.append(failure)
        self.logger.warning(f"⚠️ 检测到故障: {failure_type.value} in {component}")
        
        return failure
    
    def recover_from_failure(self, failure: FailureEvent, context: Dict[str, Any]) -> bool:
        """从故障中恢复"""
        self.logger.info(f"🔧 尝试恢复故障: {failure.failure_type.value}")
        
        # 查找合适的恢复策略
        for strategy in self.recovery_strategies:
            if strategy.can_handle(failure):
                self.logger.info(f"🎯 使用策略: {strategy.__class__.__name__}")
                
                try:
                    success = strategy.recover(failure, context)
                    failure.recovery_attempts += 1
                    
                    if success:
                        failure.resolved = True
                        self.logger.info(f"✅ 故障恢复成功")
                        
                        # 调用恢复回调
                        for callback in self.recovery_callbacks:
                            try:
                                callback(failure, True)
                            except Exception as e:
                                self.logger.error(f"恢复回调失败: {e}")
                        
                        return True
                    else:
                        self.logger.warning(f"❌ 策略执行失败")
                        
                except Exception as e:
                    self.logger.error(f"策略执行异常: {e}")
        
        self.logger.error(f"❌ 所有恢复策略都失败了")
        
        # 调用失败回调
        for callback in self.recovery_callbacks:
            try:
                callback(failure, False)
            except Exception as e:
                self.logger.error(f"失败回调异常: {e}")
        
        return False
    
    def create_checkpoint(self, **kwargs) -> Checkpoint:
        """创建检查点的便捷方法"""
        return self.checkpoint_manager.create_checkpoint(**kwargs)
    
    def start_monitoring(self):
        """启动故障监控"""
        self.failure_detector.start_monitoring()
    
    def stop_monitoring(self):
        """停止故障监控"""
        self.failure_detector.stop_monitoring()
    
    def get_failure_stats(self) -> Dict[str, Any]:
        """获取故障统计信息"""
        if not self.failure_history:
            return {"total_failures": 0}
        
        failure_counts = {}
        resolved_count = 0
        
        for failure in self.failure_history:
            failure_type = failure.failure_type.value
            failure_counts[failure_type] = failure_counts.get(failure_type, 0) + 1
            if failure.resolved:
                resolved_count += 1
        
        return {
            "total_failures": len(self.failure_history),
            "resolved_failures": resolved_count,
            "failure_types": failure_counts,
            "success_rate": resolved_count / len(self.failure_history) if self.failure_history else 0,
            "checkpoints_available": len(self.checkpoint_manager.checkpoints)
        }