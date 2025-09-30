"""
MOE-FedCL Process模式通信管理器
moe_fedcl/communication/process_manager.py
"""

import asyncio
import json
import os
import tempfile
import uuid
from typing import Any, Dict, List, Optional
from datetime import datetime
from pathlib import Path

from .base import CommunicationManagerBase
from ..transport.process import ProcessTransport
from ..types import (
    ClientInfo, RegistrationRequest, RegistrationResponse,
    HeartbeatMessage, CommunicationConfig, RegistrationStatus
)
from ..exceptions import RegistrationError


class ProcessCommunicationManager(CommunicationManagerBase):
    """Process模式通信管理器 - 基于Unix Domain Socket的进程间通信"""
    
    def __init__(self, node_id: str, transport: ProcessTransport, config: CommunicationConfig):
        super().__init__(node_id, transport, config)
        
        print(f"[ProcessCommunicationManager] 检查节点类型: {node_id}")
        
        # 如果是服务器节点，添加"system"目标ID到transport
        if "server" in node_id.lower():
            print(f"[ProcessCommunicationManager] 服务器节点，添加system目标ID到transport")
            transport.add_target_id("system")
            print(f"[ProcessCommunicationManager] 服务器节点添加system目标ID")
        else:
            print(f"[ProcessCommunicationManager] 客户端节点，不添加system目标ID")
        
        # 初始化任务管理
        self._request_listener_task = None
        self._shared_heartbeats = {}
        
        # 设置传输层的请求处理器
        
        # 使用文件系统作为跨进程状态存储
        self.state_dir = Path(tempfile.gettempdir()) / "moe_fedcl_state"
        self.state_dir.mkdir(exist_ok=True)
        self.client_registry_file = self.state_dir / "client_registry.json"
        
        # 事件文件目录（与传输层保持一致）
        self.events_dir = Path(tempfile.gettempdir()) / "moe_fedcl_events"
        self.events_dir.mkdir(exist_ok=True)
        
        # 初始化客户端注册表文件
        if not self.client_registry_file.exists():
            self._save_client_registry({})
        
        print(f"[ProcessCommunicationManager] 初始化完成: {node_id}")
        print(f"[ProcessCommunicationManager] 状态目录: {self.state_dir}")
    
    async def _emit_event(self, event_type: str, data: dict) -> None:
        """发射事件到传输层"""
        try:
            # 在进程模式中，使用文件系统来传递事件，而不是TCP连接
            client_id = data.get("client_id")
            
            # 将事件写入文件系统，供服务器端读取
            event_data = {
                "event_type": event_type,
                "source": client_id,
                "target": "system", 
                "data": data,
                "timestamp": datetime.now().isoformat()
            }
            
            event_file = self.events_dir / f"event_{client_id}_{uuid.uuid4().hex[:8]}.json"
            with open(event_file, 'w') as f:
                json.dump(event_data, f, default=str)
            
            print(f"[ProcessCommunicationManager] 事件已发射到文件: {event_type} -> {client_id}")
        except Exception as e:
            print(f"[ProcessCommunicationManager] 事件发射失败: {e}")
    
    def _load_client_registry(self) -> Dict[str, Dict]:
        """加载客户端注册表"""
        try:
            if self.client_registry_file.exists():
                with open(self.client_registry_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"[ProcessCommunicationManager] 加载注册表失败: {e}")
        return {}
    
    def _save_client_registry(self, registry: Dict[str, Dict]):
        """保存客户端注册表"""
        try:
            with open(self.client_registry_file, 'w') as f:
                json.dump(registry, f, default=str, indent=2)
        except Exception as e:
            print(f"[ProcessCommunicationManager] 保存注册表失败: {e}")
    
    async def _handle_transport_request(self, method: str, params: Dict[str, Any]) -> Any:
        """处理传输层请求"""
        print(f"[ProcessCommunicationManager] 处理请求: {method}")
        
        if method == "register_client":
            return await self._handle_register_client_request(params)
        elif method == "get_client_info":
            return await self._handle_get_client_info_request(params)
        elif method == "heartbeat":
            return await self._handle_heartbeat_request(params)
        elif method == "train":
            return await self._handle_train_request(params)
        elif method == "evaluate":
            return await self._handle_evaluate_request(params)
        elif method == "get_model":
            return await self._handle_get_model_request(params)
        elif method == "set_model":
            return await self._handle_set_model_request(params)
        else:
            return {"error": f"Unknown method: {method}"}
    
    async def _handle_register_client_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """处理客户端注册请求"""
        try:
            # 从参数中构造RegistrationRequest
            registration = RegistrationRequest(
                client_id=params["client_id"],
                client_type=params["client_type"],
                capabilities=params["capabilities"],
                metadata=params["metadata"]
            )
            
            # 调用注册方法
            response = await self.register_client(registration)
            
            return {
                "success": response.success,
                "message": response.message,
                "client_info": response.client_info.__dict__ if response.client_info else None
            }
            
        except Exception as e:
            return {"success": False, "message": str(e)}
    
    async def _handle_get_client_info_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """处理获取客户端信息请求"""
        client_id = params.get("client_id")
        client_info = await self.get_client_info(client_id)
        
        if client_info:
            return {"success": True, "client_info": client_info.__dict__}
        else:
            return {"success": False, "message": "Client not found"}
    
    async def _handle_heartbeat_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """处理心跳请求"""
        try:
            heartbeat = HeartbeatMessage(
                client_id=params["client_id"],
                timestamp=datetime.fromisoformat(params["timestamp"]),
                status=params["status"],
                metrics=params.get("metrics", {})
            )
            
            success = await self.update_heartbeat(heartbeat)
            return {"success": success}
            
        except Exception as e:
            return {"success": False, "message": str(e)}
    
    async def _handle_train_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """处理训练请求"""
        # 这里应该转发给学习器处理
        if hasattr(self, 'learner_handler') and self.learner_handler:
            return await self.learner_handler.train(params)
        else:
            return {"error": "No learner handler available"}
    
    async def _handle_evaluate_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """处理评估请求"""
        if hasattr(self, 'learner_handler') and self.learner_handler:
            return await self.learner_handler.evaluate(params)
        else:
            return {"error": "No learner handler available"}
    
    async def _handle_get_model_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """处理获取模型请求"""
        if hasattr(self, 'learner_handler') and self.learner_handler:
            return await self.learner_handler.get_model(params)
        else:
            return {"error": "No learner handler available"}
    
    async def _handle_set_model_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """处理设置模型请求"""
        if hasattr(self, 'learner_handler') and self.learner_handler:
            return await self.learner_handler.set_model(params)
        else:
            return {"error": "No learner handler available"}
    
    def set_learner_handler(self, handler):
        """设置学习器处理器"""
        self.learner_handler = handler
    
    async def register_client(self, registration: RegistrationRequest) -> RegistrationResponse:
        """注册客户端"""
        try:
            client_id = registration.client_id
            
            # 加载当前注册表
            registry = self._load_client_registry()
            
            # 检查是否已注册
            if client_id in registry:
                return RegistrationResponse(
                    success=False,
                    client_id=client_id,
                    error_message=f"Client {client_id} already registered"
                )
            
            # 创建客户端信息
            client_info = ClientInfo(
                client_id=client_id,
                client_type=registration.client_type,
                capabilities=registration.capabilities,
                metadata=registration.metadata,
                registration_time=datetime.now(),
                last_seen=datetime.now(),
                status=RegistrationStatus.REGISTERED
            )
            
            # 添加到注册表
            registry[client_id] = {
                "client_id": client_id,
                "client_type": registration.client_type,
                "capabilities": registration.capabilities,
                "metadata": registration.metadata,
                "registration_time": datetime.now().isoformat(),
                "last_seen": datetime.now().isoformat(),
                "status": RegistrationStatus.REGISTERED.value
            }
            
            # 保存注册表
            self._save_client_registry(registry)
            
            # 触发客户端注册事件
            await self._emit_event(
                "CLIENT_REGISTERED", 
                {
                    "client_id": client_id, 
                    "client_info": client_info
                }
            )
            
            print(f"[ProcessCommunicationManager] 客户端注册成功: {client_id}")
            
            return RegistrationResponse(
                success=True,
                client_id=client_id,
                server_info={
                    "server_id": self.node_id,
                    "capabilities": ["train", "evaluate", "aggregate"],
                    "heartbeat_interval": self.config.heartbeat_interval,
                    "process_mode": True,
                    "client_info": client_info.__dict__ if hasattr(client_info, '__dict__') else str(client_info)
                }
            )
            
        except Exception as e:
            print(f"[ProcessCommunicationManager] 客户端注册失败: {e}")
            return RegistrationResponse(
                success=False,
                client_id=registration.client_id,
                error_message=str(e)
            )
    
    async def unregister_client(self, client_id: str) -> bool:
        """注销客户端"""
        try:
            # 加载当前注册表
            registry = self._load_client_registry()
            
            # 检查客户端是否存在
            if client_id not in registry:
                return False
            
            # 从注册表中移除
            del registry[client_id]
            
            # 保存注册表
            self._save_client_registry(registry)
            
            print(f"[ProcessCommunicationManager] 客户端注销成功: {client_id}")
            return True
            
        except Exception as e:
            print(f"[ProcessCommunicationManager] 客户端注销失败: {e}")
            return False
    
    async def get_client_info(self, client_id: str) -> Optional[ClientInfo]:
        """获取客户端信息"""
        try:
            registry = self._load_client_registry()
            
            if client_id not in registry:
                return None
            
            client_data = registry[client_id]
            
            # 构造ClientInfo对象
            return ClientInfo(
                client_id=client_data["client_id"],
                client_type=client_data["client_type"],
                capabilities=client_data["capabilities"],
                metadata=client_data["metadata"],
                registration_time=datetime.fromisoformat(client_data["registration_time"]),
                last_seen=datetime.fromisoformat(client_data["last_seen"]),
                status=RegistrationStatus(client_data["status"])
            )
            
        except Exception as e:
            print(f"[ProcessCommunicationManager] 获取客户端信息失败: {e}")
            return None
    
    async def list_clients(self) -> List[ClientInfo]:
        """列出所有客户端"""
        try:
            registry = self._load_client_registry()
            clients = []
            
            for client_data in registry.values():
                client_info = ClientInfo(
                    client_id=client_data["client_id"],
                    client_type=client_data["client_type"],
                    capabilities=client_data["capabilities"],
                    metadata=client_data["metadata"],
                    registration_time=datetime.fromisoformat(client_data["registration_time"]),
                    last_seen=datetime.fromisoformat(client_data["last_seen"]),
                    status=RegistrationStatus(client_data["status"])
                )
                clients.append(client_info)
            
            return clients
            
        except Exception as e:
            print(f"[ProcessCommunicationManager] 列出客户端失败: {e}")
            return []
    
    async def update_heartbeat(self, heartbeat: HeartbeatMessage) -> bool:
        """更新心跳状态"""
        try:
            registry = self._load_client_registry()
            
            if heartbeat.client_id not in registry:
                return False
            
            # 更新最后活跃时间
            registry[heartbeat.client_id]["last_seen"] = heartbeat.timestamp.isoformat()
            
            # 保存注册表
            self._save_client_registry(registry)
            
            return True
            
        except Exception as e:
            print(f"[ProcessCommunicationManager] 更新心跳失败: {e}")
            return False
    
    async def start(self):
        """启动通信管理器"""
        # 启动传输层
        await self.transport.start()
        # 启动请求监听器任务
        self._request_listener_task = asyncio.create_task(self._listen_for_requests())
        await super().start()
        print(f"[ProcessCommunicationManager] 启动完成: {self.node_id}")
    
    async def _listen_for_requests(self):
        """监听请求的任务"""
        # 这里可以添加额外的请求监听逻辑
        # 目前主要依赖transport层的TCP服务器
        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            print(f"[ProcessCommunicationManager] 请求监听器已停止: {self.node_id}")
    
    async def stop(self):
        """停止通信管理器"""
        if self._request_listener_task:
            self._request_listener_task.cancel()
            try:
                await self._request_listener_task
            except asyncio.CancelledError:
                pass
        
        await super().stop()
        await self.transport.stop()
        print(f"[ProcessCommunicationManager] 停止完成: {self.node_id}")
        return None
    
    async def list_clients(self, filters: Dict[str, Any] = None) -> List[ClientInfo]:
        """列出客户端 - 从共享状态读取"""
        clients = []
        
        try:
            for client_data in self._shared_clients.values():
                client_info = ClientInfo(
                    client_id=client_data["client_id"],
                    client_type=client_data["client_type"],
                    capabilities=client_data["capabilities"],
                    metadata=client_data["metadata"],
                    registration_time=datetime.fromisoformat(client_data["registration_time"]),
                    last_seen=datetime.fromisoformat(client_data["last_seen"]),
                    status=RegistrationStatus(client_data["status"])
                )
                
                # 应用过滤器
                if filters:
                    match = True
                    for key, value in filters.items():
                        if hasattr(client_info, key):
                            if getattr(client_info, key) != value:
                                match = False
                                break
                        elif key in client_info.metadata:
                            if client_info.metadata[key] != value:
                                match = False
                                break
                        else:
                            match = False
                            break
                    
                    if not match:
                        continue
                
                clients.append(client_info)
        
        except Exception as e:
            print(f"Error listing clients: {e}")
        
        return clients
    
    def get_active_clients(self) -> List[str]:
        """获取活跃客户端列表 - Process模式"""
        active_clients = []
        current_time = datetime.now()
        timeout_seconds = self.config.heartbeat_timeout
        
        try:
            for client_id, last_heartbeat_str in self._shared_heartbeats.items():
                last_heartbeat = datetime.fromisoformat(last_heartbeat_str)
                if (current_time - last_heartbeat).total_seconds() <= timeout_seconds:
                    active_clients.append(client_id)
        except Exception as e:
            print(f"Error getting active clients: {e}")
        
        return active_clients
    
    async def setup_request_listener(self):
        """设置请求监听器"""
        if self._request_listener_task is None:
            self._request_listener_task = asyncio.create_task(self._request_listener_loop())
    
    async def _request_listener_loop(self):
        """请求监听循环"""
        while self._running:
            try:
                # 从传输层接收请求
                request_data = await self.transport.receive(
                    self.node_id,
                    timeout=1.0  # 短超时以便响应停止信号
                )
                
                # 处理请求
                source = request_data.get("source", "unknown")
                response = await self.handle_rpc_request(source, request_data)
                
                # 发送响应
                if "request_id" in request_data:
                    await self.transport.send_response(
                        source,
                        request_data["request_id"],
                        response
                    )
                
            except asyncio.TimeoutError:
                # 正常超时，继续循环
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Request listener error: {e}")
                await asyncio.sleep(0.1)
    
    def _serialize_response(self, response: Any) -> Dict[str, Any]:
        """序列化响应对象"""
        if hasattr(response, '__dict__'):
            result = {}
            for key, value in response.__dict__.items():
                if isinstance(value, datetime):
                    result[key] = value.isoformat()
                elif hasattr(value, 'value'):  # Enum
                    result[key] = value.value
                else:
                    result[key] = value
            return result
        return response
    
    def get_process_stats(self) -> Dict[str, Any]:
        """获取进程统计信息"""
        try:
            return {
                "shared_clients_count": len(self._shared_clients),
                "shared_heartbeats_count": len(self._shared_heartbeats),
                "local_cache_count": len(self._local_client_cache),
                "transport_queue_sizes": getattr(self.transport, 'get_queue_sizes', lambda: {})(),
                "active_clients": self.get_active_clients()
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def start(self) -> None:
        """启动Process通信管理器"""
        await super().start()
        
        # 设置请求监听器
        await self.setup_request_listener()
    
    async def stop(self) -> None:
        """停止Process通信管理器"""
        await super().stop()
        
        # 停止请求监听器
        if self._request_listener_task:
            self._request_listener_task.cancel()
            try:
                await self._request_listener_task
            except asyncio.CancelledError:
                pass
    
    async def cleanup(self) -> None:
        """清理Process通信管理器资源"""
        # 清理本地缓存
        self._local_client_cache.clear()
        
        await super().cleanup()