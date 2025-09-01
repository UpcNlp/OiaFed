# fedcl/automation/model_sync.py
"""
自动模型同步管理器

处理真联邦（多机）和伪联邦（本地）环境下的模型参数同步。
支持同步、异步、自适应等多种同步策略。
"""

import time
import threading
from concurrent.futures import ThreadPoolExecutor, Future, TimeoutError
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Union
import copy

import torch
import torch.nn as nn
from loguru import logger

from .communication import TransparentCommunication, Message, CommunicationMode


class FeatureTransferMode(Enum):
    """特征传递模式"""
    FORWARD_FEATURES = "forward_features"    # 前向特征传递
    BACKWARD_GRADIENTS = "backward_gradients"  # 反向梯度传递
    INTERMEDIATE_RESULTS = "intermediate_results"  # 中间结果传递


@dataclass
class FeaturePacket:
    """特征数据包"""
    packet_id: str
    source_model: str
    target_model: str
    features: torch.Tensor
    metadata: Dict[str, Any]
    timestamp: float
    requires_grad: bool = True


@dataclass 
class GradientPacket:
    """梯度数据包"""
    packet_id: str
    source_model: str
    target_model: str
    gradients: torch.Tensor
    loss_value: float
    metadata: Dict[str, Any]
    timestamp: float


class SyncMode(Enum):
    """同步模式"""
    SYNCHRONOUS = "synchronous"      # 同步模式：等待所有客户端
    ASYNCHRONOUS = "asynchronous"    # 异步模式：部分客户端完成即聚合
    ADAPTIVE = "adaptive"            # 自适应模式：根据网络状况动态调整


@dataclass
class ClientUpdate:
    """客户端更新"""
    client_id: str
    model_weights: Dict[str, torch.Tensor]
    num_samples: int
    accuracy: float
    loss: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SyncConfig:
    """同步配置"""
    mode: SyncMode = SyncMode.SYNCHRONOUS
    timeout_seconds: float = 300.0
    min_clients_ratio: float = 0.8  # 异步模式下最少等待客户端比例
    max_staleness_rounds: int = 5   # 最大过期轮次
    compression_enabled: bool = False
    quantization_bits: int = 8
    gradient_clipping: bool = True
    max_gradient_norm: float = 1.0


class ModelSynchronizer:
    """
    模型同步器
    
    负责协调服务器和客户端之间的模型参数同步，
    支持真联邦的网络同步和伪联邦的本地同步。
    """
    
    def __init__(
        self, 
        communication: TransparentCommunication,
        sync_config: Optional[SyncConfig] = None,
        is_server: bool = True
    ):
        self.communication = communication
        self.sync_config = sync_config or SyncConfig()
        self.is_server = is_server
        self.logger = logger.bind(component="ModelSynchronizer", role="server" if is_server else "client")
        
        # 服务器端状态
        if is_server:
            self.client_updates: Dict[str, ClientUpdate] = {}
            self.global_model: Optional[Dict[str, torch.Tensor]] = None
            self.current_round = 0
            self.sync_lock = threading.Lock()
            self.executor = ThreadPoolExecutor(max_workers=10)
            
        # 客户端状态
        else:
            self.local_model: Optional[Dict[str, torch.Tensor]] = None
            self.last_sync_round = 0
            
        # 注册消息处理器
        self._register_handlers()
        
    def _register_handlers(self):
        """注册消息处理器"""
        if self.is_server:
            self.communication.register_handler("model_update", self._handle_client_update)
            self.communication.register_handler("sync_request", self._handle_sync_request)
        else:
            self.communication.register_handler("global_model", self._handle_global_model)
            self.communication.register_handler("sync_command", self._handle_sync_command)
    
    def _handle_client_update(self, message: Message):
        """处理客户端模型更新"""
        try:
            payload = message.payload
            client_update = ClientUpdate(
                client_id=message.sender,
                model_weights=payload["model_weights"],
                num_samples=payload["num_samples"],
                accuracy=payload["accuracy"],
                loss=payload["loss"],
                timestamp=message.timestamp,
                metadata=payload.get("metadata", {})
            )
            
            with self.sync_lock:
                self.client_updates[message.sender] = client_update
                self.logger.info(f"📥 收到客户端 {message.sender} 的模型更新")
                
        except Exception as e:
            self.logger.error(f"处理客户端更新失败: {e}")
    
    def _handle_global_model(self, message: Message):
        """处理全局模型更新"""
        try:
            payload = message.payload
            self.global_model = payload["model_weights"]
            self.last_sync_round = payload["round"]
            
            self.logger.info(f"📥 收到全局模型更新 - 轮次: {self.last_sync_round}")
            
        except Exception as e:
            self.logger.error(f"处理全局模型失败: {e}")
    
    def _handle_sync_request(self, message: Message):
        """处理同步请求"""
        self.logger.info(f"收到来自 {message.sender} 的同步请求")
    
    def _handle_sync_command(self, message: Message):
        """处理同步命令"""
        command = message.payload.get("command")
        if command == "start_training":
            self.logger.info("📡 收到开始训练命令")
        elif command == "pause_training":
            self.logger.info("⏸️ 收到暂停训练命令")
    
    def aggregate_models(self, client_list: List[str]) -> Optional[Dict[str, torch.Tensor]]:
        """
        聚合客户端模型 - 服务器端
        
        Args:
            client_list: 期望的客户端列表
            
        Returns:
            聚合后的全局模型权重
        """
        if not self.is_server:
            raise ValueError("只有服务器端可以执行模型聚合")
            
        self.logger.info(f"🔄 开始模型聚合 - 轮次: {self.current_round + 1}")
        
        # 等待客户端更新
        collected_updates = self._collect_client_updates(client_list)
        
        if not collected_updates:
            self.logger.warning("未收到任何客户端更新")
            return None
        
        # 执行聚合
        if self.sync_config.mode == SyncMode.SYNCHRONOUS:
            aggregated_model = self._federated_averaging(collected_updates)
        elif self.sync_config.mode == SyncMode.ASYNCHRONOUS:
            aggregated_model = self._async_aggregation(collected_updates)
        else:  # ADAPTIVE
            aggregated_model = self._adaptive_aggregation(collected_updates)
        
        # 更新全局模型
        self.global_model = aggregated_model
        self.current_round += 1
        
        self.logger.info(f"✅ 模型聚合完成 - 参与客户端: {len(collected_updates)}")
        return aggregated_model
    
    def _collect_client_updates(self, client_list: List[str]) -> List[ClientUpdate]:
        """收集客户端更新"""
        collected_updates = []
        timeout = self.sync_config.timeout_seconds
        start_time = time.time()
        
        while True:
            current_time = time.time()
            elapsed = current_time - start_time
            
            # 检查超时
            if elapsed > timeout:
                self.logger.warning(f"⏰ 收集客户端更新超时: {elapsed:.2f}s")
                break
            
            # 检查已收集的更新
            with self.sync_lock:
                for client_id in client_list:
                    if client_id in self.client_updates and client_id not in [u.client_id for u in collected_updates]:
                        collected_updates.append(self.client_updates[client_id])
            
            # 检查是否满足同步条件
            if self._should_proceed_aggregation(collected_updates, client_list):
                break
                
            time.sleep(0.1)  # 短暂等待
        
        return collected_updates
    
    def _should_proceed_aggregation(self, collected_updates: List[ClientUpdate], client_list: List[str]) -> bool:
        """判断是否应该开始聚合"""
        collected_ratio = len(collected_updates) / len(client_list)
        
        if self.sync_config.mode == SyncMode.SYNCHRONOUS:
            return collected_ratio >= 1.0
        elif self.sync_config.mode == SyncMode.ASYNCHRONOUS:
            return collected_ratio >= self.sync_config.min_clients_ratio
        else:  # ADAPTIVE
            # 自适应逻辑：根据历史性能动态调整
            return collected_ratio >= max(0.5, self.sync_config.min_clients_ratio)
    
    def _federated_averaging(self, updates: List[ClientUpdate]) -> Dict[str, torch.Tensor]:
        """联邦平均聚合"""
        if not updates:
            return {}
        
        total_samples = sum(update.num_samples for update in updates)
        aggregated_weights = {}
        
        # 获取第一个更新的权重结构
        first_weights = updates[0].model_weights
        
        for param_name in first_weights.keys():
            weighted_sum = torch.zeros_like(first_weights[param_name])
            
            for update in updates:
                weight = update.num_samples / total_samples
                weighted_sum += weight * update.model_weights[param_name]
            
            aggregated_weights[param_name] = weighted_sum
        
        self.logger.debug(f"📊 联邦平均完成 - 总样本数: {total_samples}")
        return aggregated_weights
    
    def _async_aggregation(self, updates: List[ClientUpdate]) -> Dict[str, torch.Tensor]:
        """异步聚合"""
        # 过滤过期更新
        current_time = time.time()
        fresh_updates = [
            update for update in updates 
            if current_time - update.timestamp < self.sync_config.timeout_seconds
        ]
        
        if not fresh_updates:
            self.logger.warning("所有更新都已过期，使用所有可用更新")
            fresh_updates = updates
        
        return self._federated_averaging(fresh_updates)
    
    def _adaptive_aggregation(self, updates: List[ClientUpdate]) -> Dict[str, torch.Tensor]:
        """自适应聚合"""
        # 根据客户端性能加权
        performance_weights = []
        for update in updates:
            # 综合考虑准确率和时效性
            time_weight = 1.0 / (1.0 + time.time() - update.timestamp)
            accuracy_weight = update.accuracy
            performance_weight = time_weight * accuracy_weight
            performance_weights.append(performance_weight)
        
        # 归一化权重
        total_performance = sum(performance_weights)
        if total_performance > 0:
            performance_weights = [w / total_performance for w in performance_weights]
        else:
            performance_weights = [1.0 / len(updates)] * len(updates)
        
        # 加权聚合
        aggregated_weights = {}
        first_weights = updates[0].model_weights
        
        for param_name in first_weights.keys():
            weighted_sum = torch.zeros_like(first_weights[param_name])
            
            for update, weight in zip(updates, performance_weights):
                weighted_sum += weight * update.model_weights[param_name]
            
            aggregated_weights[param_name] = weighted_sum
        
        self.logger.debug("📊 自适应聚合完成")
        return aggregated_weights
    
    def distribute_global_model(self, client_list: List[str]) -> Dict[str, bool]:
        """
        分发全局模型 - 服务器端
        
        Args:
            client_list: 客户端列表
            
        Returns:
            分发结果
        """
        if not self.is_server:
            raise ValueError("只有服务器端可以分发全局模型")
        
        if self.global_model is None:
            self.logger.error("全局模型为空，无法分发")
            return {client: False for client in client_list}
        
        self.logger.info(f"📤 开始分发全局模型到 {len(client_list)} 个客户端")
        
        # 准备模型数据
        model_data = {
            "model_weights": self.global_model,
            "round": self.current_round,
            "timestamp": time.time(),
            "metadata": {
                "aggregation_mode": self.sync_config.mode.value,
                "num_participants": len(self.client_updates)
            }
        }
        
        # 广播模型
        results = self.communication.broadcast_global_model(client_list, model_data)
        
        success_count = sum(1 for success in results.values() if success)
        self.logger.info(f"📤 全局模型分发完成: {success_count}/{len(client_list)} 成功")
        
        return results
    
    def upload_model_update(
        self, 
        model_weights: Dict[str, torch.Tensor],
        num_samples: int,
        accuracy: float,
        loss: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        上传模型更新 - 客户端
        
        Args:
            model_weights: 模型权重
            num_samples: 训练样本数
            accuracy: 训练准确率
            loss: 训练损失
            metadata: 额外元数据
            
        Returns:
            上传是否成功
        """
        if self.is_server:
            raise ValueError("服务器端不能上传模型更新")
        
        self.logger.info(f"📤 上传模型更新 - 样本数: {num_samples}, 准确率: {accuracy:.4f}")
        
        # 压缩模型权重（如果启用）
        if self.sync_config.compression_enabled:
            model_weights = self._compress_model_weights(model_weights)
        
        # 准备更新数据
        update_data = {
            "model_weights": model_weights,
            "num_samples": num_samples,
            "accuracy": accuracy,
            "loss": loss,
            "metadata": metadata or {}
        }
        
        # 发送到服务器
        success = self.communication.send_model_update("server", update_data)
        
        if success:
            self.logger.info("✅ 模型更新上传成功")
        else:
            self.logger.error("❌ 模型更新上传失败")
            
        return success
    
    def _compress_model_weights(self, weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """压缩模型权重"""
        if self.sync_config.quantization_bits < 32:
            # 简单的量化压缩
            compressed_weights = {}
            for name, tensor in weights.items():
                # 量化到指定位数
                scale = tensor.abs().max()
                quantized = (tensor / scale * (2 ** (self.sync_config.quantization_bits - 1) - 1)).round()
                compressed_weights[name] = quantized * scale / (2 ** (self.sync_config.quantization_bits - 1) - 1)
            return compressed_weights
        return weights
    
    def get_sync_stats(self) -> Dict[str, Any]:
        """获取同步统计信息"""
        if self.is_server:
            return {
                "current_round": self.current_round,
                "connected_clients": len(self.client_updates),
                "sync_mode": self.sync_config.mode.value,
                "has_global_model": self.global_model is not None
            }
        else:
            return {
                "last_sync_round": self.last_sync_round,
                "has_local_model": self.local_model is not None,
                "sync_mode": self.sync_config.mode.value
            }
    
    def reset_round(self):
        """重置轮次状态"""
        if self.is_server:
            with self.sync_lock:
                self.client_updates.clear()
            self.logger.info(f"🔄 重置轮次状态 - 当前轮次: {self.current_round}")


class TransparentFeatureSync:
    """
    透明特征同步器
    
    支持服务器-客户端中间特征的透明传递：
    1. 服务器向客户端传递中间特征
    2. 客户端接收特征、计算损失、梯度回传
    3. 整个过程对用户完全透明
    """
    
    def __init__(self, communication: TransparentCommunication, node_id: str):
        self.communication = communication
        self.node_id = node_id
        self.logger = logger.bind(component="TransparentFeatureSync", node=node_id)
        
        # 特征传递的回调函数
        self.feature_handlers: Dict[str, Callable] = {}
        self.gradient_handlers: Dict[str, Callable] = {}
        
        # 注册消息处理器
        self._register_feature_handlers()
        
        # 特征传递的待处理队列
        self.pending_features: Dict[str, FeaturePacket] = {}
        self.pending_gradients: Dict[str, GradientPacket] = {}
    
    def _register_feature_handlers(self):
        """注册特征传递的消息处理器"""
        self.communication.register_handler("forward_features", self._handle_forward_features)
        self.communication.register_handler("backward_gradients", self._handle_backward_gradients)
        self.communication.register_handler("intermediate_results", self._handle_intermediate_results)
    
    def send_features_to_client(
        self, 
        client_id: str, 
        model_name: str,
        features: torch.Tensor, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        服务器向客户端发送中间特征（透明）
        
        Args:
            client_id: 目标客户端
            model_name: 模型名称
            features: 中间特征张量
            metadata: 附加元数据
            
        Returns:
            特征包ID，用于追踪回传的梯度
        """
        packet_id = f"feat_{int(time.time() * 1000000)}"
        
        feature_packet = FeaturePacket(
            packet_id=packet_id,
            source_model=f"{self.node_id}_{model_name}",
            target_model=f"{client_id}_{model_name}",
            features=features,
            metadata=metadata or {},
            timestamp=time.time(),
            requires_grad=True
        )
        
        # 发送特征数据
        message_payload = {
            "packet_id": packet_id,
            "source_model": feature_packet.source_model,
            "target_model": feature_packet.target_model,
            "features": features.detach().cpu(),  # 传输时移动到CPU
            "metadata": feature_packet.metadata,
            "timestamp": feature_packet.timestamp,
            "requires_grad": feature_packet.requires_grad
        }
        
        success = self.communication.send_model_update(client_id, {
            "message_type": "forward_features",
            "payload": message_payload
        })
        
        if success:
            # 记录待处理的特征，等待梯度回传
            self.pending_features[packet_id] = feature_packet
            self.logger.info(f"📤 特征已发送至 {client_id}: {features.shape}")
        else:
            self.logger.error(f"❌ 特征发送失败至 {client_id}")
            
        return packet_id
    
    def _handle_forward_features(self, message: Message):
        """处理接收到的前向特征（客户端）"""
        try:
            payload = message.payload["payload"]
            
            feature_packet = FeaturePacket(
                packet_id=payload["packet_id"],
                source_model=payload["source_model"],
                target_model=payload["target_model"],
                features=payload["features"].requires_grad_(payload["requires_grad"]),
                metadata=payload["metadata"],
                timestamp=payload["timestamp"],
                requires_grad=payload["requires_grad"]
            )
            
            self.logger.info(f"📥 接收到特征: {feature_packet.features.shape}")
            
            # 调用用户注册的特征处理器
            handler = self.feature_handlers.get(feature_packet.target_model)
            if handler:
                # 用户处理特征，返回损失和梯度
                loss, gradients = handler(feature_packet.features, feature_packet.metadata)
                
                # 自动回传梯度
                self._send_gradients_back(feature_packet.packet_id, 
                                        feature_packet.source_model,
                                        gradients, loss, feature_packet.metadata)
            else:
                self.logger.warning(f"未找到特征处理器: {feature_packet.target_model}")
                
        except Exception as e:
            self.logger.error(f"处理前向特征失败: {e}")
    
    def _send_gradients_back(
        self, 
        packet_id: str, 
        target_model: str,
        gradients: torch.Tensor, 
        loss_value: float,
        metadata: Dict[str, Any]
    ):
        """自动回传梯度到服务器"""
        
        gradient_packet = GradientPacket(
            packet_id=packet_id,
            source_model=self.node_id,
            target_model=target_model,
            gradients=gradients,
            loss_value=loss_value,
            metadata=metadata,
            timestamp=time.time()
        )
        
        # 提取服务器ID
        server_id = target_model.split("_")[0]
        
        message_payload = {
            "packet_id": packet_id,
            "source_model": gradient_packet.source_model,
            "target_model": gradient_packet.target_model,
            "gradients": gradients.detach().cpu(),
            "loss_value": loss_value,
            "metadata": gradient_packet.metadata,
            "timestamp": gradient_packet.timestamp
        }
        
        success = self.communication.send_model_update(server_id, {
            "message_type": "backward_gradients",
            "payload": message_payload
        })
        
        if success:
            self.logger.info(f"🔙 梯度已回传: loss={loss_value:.6f}")
        else:
            self.logger.error(f"❌ 梯度回传失败")
    
    def _handle_backward_gradients(self, message: Message):
        """处理接收到的反向梯度（服务器）"""
        try:
            payload = message.payload["payload"]
            packet_id = payload["packet_id"]
            
            # 查找对应的特征包
            if packet_id in self.pending_features:
                feature_packet = self.pending_features[packet_id]
                
                gradient_packet = GradientPacket(
                    packet_id=packet_id,
                    source_model=payload["source_model"],
                    target_model=payload["target_model"],
                    gradients=payload["gradients"],
                    loss_value=payload["loss_value"],
                    metadata=payload["metadata"],
                    timestamp=payload["timestamp"]
                )
                
                self.logger.info(f"📥 接收到梯度: loss={gradient_packet.loss_value:.6f}")
                
                # 调用用户注册的梯度处理器
                handler = self.gradient_handlers.get(feature_packet.source_model)
                if handler:
                    handler(feature_packet.features, gradient_packet.gradients, gradient_packet.loss_value)
                
                # 清理已处理的特征包
                del self.pending_features[packet_id]
            else:
                self.logger.warning(f"未找到对应的特征包: {packet_id}")
                
        except Exception as e:
            self.logger.error(f"处理反向梯度失败: {e}")
    
    def register_feature_handler(self, model_name: str, handler: Callable):
        """
        注册特征处理器（客户端使用）
        
        Args:
            model_name: 模型名称
            handler: 处理函数，签名为 (features, metadata) -> (loss, gradients)
        """
        full_model_name = f"{self.node_id}_{model_name}"
        self.feature_handlers[full_model_name] = handler
        self.logger.info(f"✅ 注册特征处理器: {full_model_name}")
    
    def register_gradient_handler(self, model_name: str, handler: Callable):
        """
        注册梯度处理器（服务器使用）
        
        Args:
            model_name: 模型名称  
            handler: 处理函数，签名为 (features, gradients, loss) -> None
        """
        full_model_name = f"{self.node_id}_{model_name}"
        self.gradient_handlers[full_model_name] = handler
        self.logger.info(f"✅ 注册梯度处理器: {full_model_name}")
    
    def _handle_intermediate_results(self, message: Message):
        """处理中间结果传递"""
        # 用于支持更复杂的中间计算结果传递
        pass


class MultiModelManager:
    """
    多模型管理器
    
    支持客户端多个模型的透明管理：
    1. 多个模型实例的自动管理
    2. 模型间通信的自动协调
    3. 业务逻辑与通信的完全解耦
    """
    
    def __init__(self, communication: TransparentCommunication, node_id: str):
        self.communication = communication
        self.node_id = node_id
        self.logger = logger.bind(component="MultiModelManager", node=node_id)
        
        # 模型实例管理
        self.model_instances: Dict[str, Any] = {}
        self.model_synchronizers: Dict[str, ModelSynchronizer] = {}
        
        # 特征同步器
        self.feature_sync = TransparentFeatureSync(communication, node_id)
        
        # 模型间通信路由
        self.model_routes: Dict[str, List[str]] = {}
        
    def register_model(self, model_name: str, model_instance: Any, 
                      sync_config: Optional[SyncConfig] = None) -> str:
        """
        注册模型实例（对用户透明）
        
        Args:
            model_name: 模型名称
            model_instance: 模型实例
            sync_config: 同步配置
            
        Returns:
            模型的全局ID
        """
        full_model_id = f"{self.node_id}_{model_name}"
        
        # 注册模型实例
        self.model_instances[full_model_id] = model_instance
        
        # 创建专用的模型同步器
        model_sync = ModelSynchronizer(
            communication=self.communication,
            sync_config=sync_config,
            is_server=False  # 默认作为客户端
        )
        self.model_synchronizers[full_model_id] = model_sync
        
        self.logger.info(f"✅ 模型注册成功: {full_model_id}")
        return full_model_id
    
    def setup_model_communication(self, model_name: str, 
                                feature_handler: Optional[Callable] = None,
                                gradient_handler: Optional[Callable] = None):
        """
        设置模型的透明通信（对用户透明）
        
        Args:
            model_name: 模型名称
            feature_handler: 特征处理函数
            gradient_handler: 梯度处理函数
        """
        if feature_handler:
            self.feature_sync.register_feature_handler(model_name, feature_handler)
            
        if gradient_handler:
            self.feature_sync.register_gradient_handler(model_name, gradient_handler)
            
        self.logger.info(f"✅ 模型通信设置完成: {model_name}")
    
    def auto_sync_model(self, model_name: str, target_nodes: List[str]) -> bool:
        """
        自动同步模型（对用户透明）
        
        Args:
            model_name: 模型名称
            target_nodes: 目标节点列表
            
        Returns:
            同步是否成功
        """
        full_model_id = f"{self.node_id}_{model_name}"
        
        if full_model_id not in self.model_synchronizers:
            self.logger.error(f"模型未注册: {model_name}")
            return False
            
        model_sync = self.model_synchronizers[full_model_id]
        model_instance = self.model_instances[full_model_id]
        
        # 获取模型权重
        if hasattr(model_instance, 'state_dict'):
            model_weights = model_instance.state_dict()
        else:
            self.logger.warning(f"模型不支持 state_dict: {model_name}")
            return False
            
        # 自动上传模型更新
        success = model_sync.upload_model_update(
            model_weights=model_weights,
            num_samples=1000,  # 可以从模型实例获取
            accuracy=0.95,     # 可以从模型实例获取
            loss=0.05,         # 可以从模型实例获取
            metadata={"model_name": model_name, "target_nodes": target_nodes}
        )
        
        return success
    
    def get_model_stats(self) -> Dict[str, Any]:
        """获取模型统计信息"""
        return {
            "registered_models": len(self.model_instances),
            "active_synchronizers": len(self.model_synchronizers),
            "model_list": list(self.model_instances.keys()),
            "communication_stats": self.communication.get_stats()
        }
    """
    自动模型同步管理器
    
    提供更高级的自动化同步接口，集成通信和同步逻辑
    """
    
    def __init__(
        self,
        node_id: str,
        communication_mode: CommunicationMode,
        is_server: bool = False,
        sync_config: Optional[SyncConfig] = None,
        network_config = None
    ):
        self.node_id = node_id
        self.is_server = is_server
        self.logger = logger.bind(component="AutoModelSync", node=node_id)
        
        # 初始化透明通信
        self.communication = TransparentCommunication(
            node_id=node_id,
            mode=communication_mode,
            config=network_config,
            is_server=is_server
        )
        
        # 初始化模型同步器
        self.synchronizer = ModelSynchronizer(
            communication=self.communication,
            sync_config=sync_config,
            is_server=is_server
        )
        
        self.is_running = False
    
    def start(self) -> bool:
        """启动自动同步"""
        if self.communication.start():
            self.is_running = True
            self.logger.info("🚀 自动模型同步已启动")
            return True
        return False
    
    def stop(self) -> bool:
        """停止自动同步"""
        self.is_running = False
        return self.communication.stop()
    
    def sync_global_model(self, learner_instances: List[Any]):
        """自动同步全局模型到所有学习器 - 简化版"""
        if not self.is_server:
            return
            
        client_ids = [f"client_{i}" for i in range(len(learner_instances))]
        
        # 收集更新并聚合
        aggregated_model = self.synchronizer.aggregate_models(client_ids)
        
        if aggregated_model:
            # 分发模型
            self.synchronizer.distribute_global_model(client_ids)
    
    def collect_model_updates(self, learner_instances: List[Any]):
        """自动收集模型更新 - 简化版"""
        pass  # 在实际实现中会调用具体的学习器接口
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        comm_stats = self.communication.get_stats()
        sync_stats = self.synchronizer.get_sync_stats()
        
        return {
            "communication": comm_stats,
            "synchronization": sync_stats,
            "is_running": self.is_running
        }