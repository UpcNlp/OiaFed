"""
gRPC 传输层实现（支持双线程优化）

双线程模式（默认启用）：
- 主线程：处理业务调用（call/broadcast）
- 控制线程：心跳、健康检查（独立事件循环）
"""

import asyncio
import json
import threading
import time
from typing import Dict, Optional, List, Any, TYPE_CHECKING

import grpc
from grpc import aio as grpc_aio

from .base import Transport
from ..config import GrpcTransportConfig
from ..message import Message, MessageType, ErrorInfo
from ..exceptions import NodeNotConnectedError

# Proto 生成的代码
from ..proto import node_service_pb2 as pb2
from ..proto import node_service_pb2_grpc as pb2_grpc

if TYPE_CHECKING:
    from ..node import Node

# 使用 loguru 统一日志
from ...infra.logging import get_logger


class NodeServiceServicer(pb2_grpc.NodeServiceServicer):
    """gRPC 服务实现"""
    
    def __init__(self, node: "Node"):
        self.node = node
    
    async def Call(
        self,
        request: pb2.NodeMessage,
        context: grpc_aio.ServicerContext,
    ) -> pb2.NodeMessage:
        """处理请求-响应调用"""
        # 转换为内部 Message
        message = _from_proto(request)
        
        # 调用 Node 处理
        response = await self.node.on_message(message)
        
        # 转换为 proto Message
        if response:
            return _to_proto(response)
        else:
            # 不应该发生，REQUEST 应该总是有响应
            return pb2.NodeMessage(
                id="",
                type=pb2.MESSAGE_TYPE_RESPONSE,
                error=pb2.ErrorInfo(
                    code="INTERNAL_ERROR",
                    message="No response generated",
                ),
            )
    
    async def Notify(
        self,
        request: pb2.NodeMessage,
        context: grpc_aio.ServicerContext,
    ) -> pb2.Empty:
        """处理单向通知"""
        message = _from_proto(request)
        await self.node.on_message(message)
        return pb2.Empty()


class GrpcTransport(Transport):
    """
    gRPC 传输层（支持双线程优化）

    双线程模式（默认启用）：
    - 主线程：处理业务调用（call/broadcast）
    - 控制线程：心跳、健康检查（独立事件循环）

    单线程模式（兼容模式）：
    - 所有逻辑在主线程（与 Memory 模式类似）
    """

    def __init__(self, node: "Node", config: Optional[GrpcTransportConfig] = None):
        self.node = node
        self.config = config or GrpcTransportConfig()
        self.node_id = node.node_id

        # 创建绑定了 node_id 和 log_type 的 logger
        self.logger = get_logger(node_id=self.node_id, log_type="system")

        # gRPC 服务端和客户端
        self._server: Optional[grpc_aio.Server] = None
        self._channels: Dict[str, grpc_aio.Channel] = {}
        self._stubs: Dict[str, pb2_grpc.NodeServiceStub] = {}
        self._running = False

        # 双线程相关
        self._dual_thread_enabled = self.config.dual_thread_enabled
        self._control_thread: Optional[threading.Thread] = None
        self._control_loop: Optional[asyncio.AbstractEventLoop] = None
        self._control_running = False
        self._control_stop_event: Optional[asyncio.Event] = None  # 控制线程的停止信号

        # 共享状态（线程安全）
        self._state_lock = threading.RLock()
        self._shared_state = {
            "last_heartbeat": {},    # {node_id: timestamp}
            "peer_status": {},       # {node_id: {"health": "healthy"/"unhealthy"}}
        }

        # 连接失败跟踪（用于自动shutdown）
        self._first_failure_time: Dict[str, float] = {}  # {node_id: 首次失败时间戳}
        self._critical_peers = set(self.config.critical_peers)  # 关键节点集合
        self._max_wait_time = self.config.max_connection_wait_time  # 最大等待时间
        self._auto_shutdown = self.config.auto_shutdown_on_failure  # 是否自动shutdown
        self._shutdown_triggered = False  # 是否已触发shutdown

        # 记录自动shutdown配置
        # 注意：只有设置了 critical_peers 时才记录，否则保持静默
        # （Trainer 不需要 critical_peers，Learner 会自动设置）
        if self._auto_shutdown and self._critical_peers:
            self.logger.info(
                f"[{self.node_id}] 自动shutdown已启用: "
                f"关键节点={list(self._critical_peers)}, "
                f"最大等待时间={self._max_wait_time}秒"
            )
    
    async def start(self) -> None:
        """启动 gRPC 服务端"""
        self.logger.debug(f"[{self.node_id}] 正在启动 gRPC 服务端")

        # 设置 gRPC server 选项（消息大小限制）
        server_options = [
            ('grpc.max_send_message_length', self.config.max_message_length),
            ('grpc.max_receive_message_length', self.config.max_message_length),
        ]
        self._server = grpc_aio.server(options=server_options)

        # 注册服务
        servicer = NodeServiceServicer(self.node)
        pb2_grpc.add_NodeServiceServicer_to_server(servicer, self._server)

        # 从 node.config.listen 获取绑定地址
        if hasattr(self.node.config, 'listen') and self.node.config.listen:
            host = self.node.config.listen.get("host", "0.0.0.0")
            port = self.node.config.listen.get("port", 50051)
        else:
            # 降级到 transport 配置（向后兼容）
            host = self.config.get("host", "0.0.0.0") if isinstance(self.config, dict) else getattr(self.config, "host", "0.0.0.0")
            port = self.config.get("port", 50051) if isinstance(self.config, dict) else getattr(self.config, "port", 50051)

        address = f"{host}:{port}"

        if isinstance(self.config, dict):
            tls_enabled = self.config.get("tls", {}).get("enabled", False)
        else:
            tls_enabled = self.config.tls.enabled if hasattr(self.config, 'tls') else False

        if tls_enabled:
            # TODO: 实现 TLS
            raise NotImplementedError("TLS not yet implemented")
        else:
            self._server.add_insecure_port(address)

        await self._server.start()
        self._running = True
        self.logger.debug(f"[{self.node_id}] gRPC 服务端已启动: {address}")

        # 启动控制线程（如果启用）
        if self._dual_thread_enabled and self.config.heartbeat_enabled:
            self._start_control_thread()
            self.logger.debug(f"[{self.node_id}] 双线程模式已启用（心跳独立运行）")
        else:
            self.logger.debug(f"[{self.node_id}] 单线程模式（兼容模式）")

    async def stop(self) -> None:
        """停止 gRPC 服务端并关闭所有连接"""
        self._running = False

        # *** 关键修复：先清空连接表，防止心跳继续发送 ***
        # 保存 channels 引用用于后续关闭
        channels_to_close = list(self._channels.values())
        self._channels.clear()
        self._stubs.clear()

        # 停止控制线程（如果启用）
        # 此时连接表已清空，心跳循环不会再尝试发送
        if self._control_running:
            self._stop_control_thread()

        # 关闭所有 channel
        for channel in channels_to_close:
            await channel.close()

        # 停止 server
        if self._server:
            await self._server.stop(grace=5.0)
            self._server = None

        self.logger.debug(f"gRPC server stopped for node '{self.node_id}'")
    
    async def send(self, message: Message) -> None:
        """发送消息到目标节点"""
        stub = self._stubs.get(message.target)
        if not stub:
            raise NodeNotConnectedError(message.target)
        
        # 转换为 protobuf 消息
        proto_msg = _to_proto(message)
        
        if message.type == MessageType.NOTIFY:
            await stub.Notify(proto_msg)
        else:
            # REQUEST: 等待响应
            proto_response = await stub.Call(proto_msg)
            response = _from_proto(proto_response)
            await self.node.on_message(response)
    
    async def connect(self, node_id: str, address: Optional[str] = None, retry_config: Optional[Dict[str, Any]] = None) -> None:
        """建立到目标节点的连接"""
        if not address:
            raise ValueError("Address is required for gRPC transport")

        if node_id in self._stubs:
            self.logger.debug(f"节点 '{node_id}' 已连接，跳过")
            return

        self.logger.debug(f"[{self.node_id}] 正在连接到节点 '{node_id}' (地址: {address})")

        # 创建 channel
        if self.config.tls.enabled:
            # TODO: 实现 TLS
            raise NotImplementedError("TLS not yet implemented")
        else:
            # 设置 gRPC channel 选项
            options = [
                ('grpc.max_send_message_length', self.config.max_message_length),
                ('grpc.max_receive_message_length', self.config.max_message_length),
            ]
            channel = grpc_aio.insecure_channel(address, options=options)

        # 使用 retry_config 或默认值
        max_retries = 3
        retry_delay = 1.0
        if retry_config:
            max_retries = retry_config.get("max_retries", 3)
            retry_delay = retry_config.get("retry_interval", 1.0)

        for attempt in range(max_retries):
            try:
                self.logger.debug(f"[{self.node_id}] 尝试连接到 '{node_id}'，第 {attempt + 1}/{max_retries} 次")
                await asyncio.wait_for(
                    channel.channel_ready(),
                    timeout=5.0,
                )
                self.logger.debug(f"[{self.node_id}] 成功连接到 '{node_id}'")

                # 重置失败跟踪
                self._track_connection_success(node_id)

                break  # 成功连接
            except asyncio.TimeoutError:
                if attempt < max_retries - 1:
                    self.logger.debug(f"连接 '{node_id}' 超时，{retry_delay}秒后重试...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # 指数退避
                else:
                    self.logger.error(f"连接 '{node_id}' 失败，已达到最大重试次数")

                    # 跟踪连接失败
                    self._track_connection_failure(node_id)

                    await channel.close()
                    raise NodeNotConnectedError(node_id)

        # 创建 stub
        stub = pb2_grpc.NodeServiceStub(channel)

        self._channels[node_id] = channel
        self._stubs[node_id] = stub

        await self.node._on_connect(node_id, address)
        self.logger.debug(f"[{self.node_id}] 节点 '{self.node_id}' 已连接到 '{node_id}' (地址: {address})")
    
    async def disconnect(self, node_id: str) -> None:
        """断开与目标节点的连接"""
        channel = self._channels.pop(node_id, None)
        self._stubs.pop(node_id, None)
        
        if channel:
            await channel.close()
            await self.node._on_disconnect(node_id, "manual disconnect")
            self.logger.debug(f"Node '{self.node_id}' disconnected from '{node_id}'")
    
    def is_connected(self, node_id: str) -> bool:
        """检查是否已连接到指定节点"""
        return node_id in self._stubs

    async def broadcast(
        self,
        targets: List[str],
        method: str,
        payload: Any,
        timeout: Optional[float] = None,
        wait_for_all: bool = True,
        min_responses: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        广播调用多个目标节点（并行调用）

        gRPC 模式下使用真正的网络并行广播

        改进机制:
        - 支持超时控制，不必等待所有节点返回
        - 支持最小响应数，达到数量即可返回
        - 超时或未响应的节点返回 TimeoutError

        Args:
            targets: 目标节点 ID 列表
            method: 方法名
            payload: 参数（字典格式）
            timeout: 超时时间（秒），None 表示无限等待
            wait_for_all: 是否等待所有节点返回，False 时结合 min_responses 使用
            min_responses: 最少需要的响应数，达到后即可返回（仅在 wait_for_all=False 时生效）

        Returns:
            结果字典 {node_id: result}
            - 成功的节点: 返回结果
            - 失败的节点: Exception 对象
            - 超时的节点: TimeoutError 对象
            - 未连接的节点: NodeNotConnectedError 对象
        """
        # 创建所有调用任务
        tasks = {}
        for target_id in targets:
            if target_id in self._stubs:
                task = self._call_single(target_id, method, payload, timeout)
                tasks[target_id] = task
            else:
                # 节点未连接，记录为错误
                tasks[target_id] = None

        # 如果没有可调用的任务，直接返回
        if not tasks:
            return {}

        # 准备结果字典
        results = {}

        # 先处理未连接的节点
        for target_id, task in tasks.items():
            if task is None:
                results[target_id] = NodeNotConnectedError(target_id)

        # 获取所有有效任务
        valid_tasks = {tid: t for tid, t in tasks.items() if t is not None}

        if not valid_tasks:
            return results

        # 情况 1: 等待所有节点返回 (原有逻辑)
        if wait_for_all:
            results_list = await asyncio.gather(
                *valid_tasks.values(),
                return_exceptions=True
            )

            for idx, target_id in enumerate(valid_tasks.keys()):
                results[target_id] = results_list[idx]

            return results

        # 情况 2: 不等待所有节点，支持最小响应数和超时
        if min_responses is None:
            min_responses = len(valid_tasks)  # 默认仍需要全部响应

        # 创建任务到ID的映射
        task_to_id = {task: tid for tid, task in valid_tasks.items()}
        pending_tasks = set(valid_tasks.values())
        completed_count = 0

        # 使用 asyncio.wait 实现灵活的等待策略
        try:
            while pending_tasks and completed_count < min_responses:
                done, pending_tasks = await asyncio.wait(
                    pending_tasks,
                    timeout=timeout,
                    return_when=asyncio.FIRST_COMPLETED
                )

                # 处理已完成的任务
                for task in done:
                    target_id = task_to_id[task]
                    try:
                        result = await task  # 获取结果或异常
                        results[target_id] = result
                    except Exception as e:
                        results[target_id] = e
                    completed_count += 1

                # 如果已达到最小响应数，可以提前返回
                if completed_count >= min_responses:
                    break

                # 如果设置了超时且还有未完成的任务，标记为超时
                if timeout and pending_tasks:
                    # 这里的pending已经是超时后剩余的任务
                    break

        except asyncio.TimeoutError:
            pass  # 超时是预期行为

        # 处理超时或未完成的任务
        for task in pending_tasks:
            target_id = task_to_id[task]
            task.cancel()  # 取消未完成的任务
            results[target_id] = TimeoutError(f"Node {target_id} did not respond within {timeout}s")

        return results

    async def _call_single(
        self,
        target_id: str,
        method: str,
        payload: Any,
        timeout: Optional[float] = None,
    ) -> Any:
        """
        单个节点调用（内部方法）

        Args:
            target_id: 目标节点 ID
            method: 方法名
            payload: 参数
            timeout: 超时时间

        Returns:
            调用结果
        """
        stub = self._stubs.get(target_id)
        if not stub:
            raise NodeNotConnectedError(target_id)

        # 构造 REQUEST 消息
        import uuid
        import time

        # 获取序列化器（使用 Node 的序列化器）
        serializer = self.node._get_serializer(method)

        # 序列化 payload
        payload_bytes = serializer.serialize(payload)

        message = Message(
            id=str(uuid.uuid4()),
            type=MessageType.REQUEST,
            source=self.node_id,
            target=target_id,
            method=method,
            payload=payload_bytes,
            correlation_id=None,
            timestamp=time.time_ns(),
            metadata={},
            error=None,
        )

        # 转换为 protobuf 消息
        proto_msg = _to_proto(message)

        try:
            # 调用远程节点（带超时）
            if timeout:
                proto_response = await asyncio.wait_for(
                    stub.Call(proto_msg),
                    timeout=timeout
                )
            else:
                proto_response = await stub.Call(proto_msg)

            # 转换响应
            response = _from_proto(proto_response)

            # 检查是否有错误
            if response.type == MessageType.RESPONSE:
                # 反序列化 payload
                if isinstance(response.payload, bytes):
                    return serializer.deserialize(response.payload)
                return response.payload
            elif response.error:
                # 抛出远程异常
                from ..exceptions import RemoteExecutionError
                raise RemoteExecutionError(
                    response.error.message,
                    response.error.code,
                )

            return None

        except asyncio.TimeoutError:
            from ..exceptions import CallTimeoutError
            raise CallTimeoutError(target_id, method, timeout)
        except grpc.RpcError as e:
            # *** 关键修复：关闭时不记录错误日志，避免大量无用日志 ***
            if self._running:
                self.logger.error(f"gRPC error calling {target_id}.{method}: {e}")
                # 跟踪连接失败并检查是否需要shutdown
                self._track_connection_failure(target_id)
            else:
                # 节点正在关闭，降低日志级别
                self.logger.debug(f"gRPC error during shutdown calling {target_id}.{method}: {e}")

            raise

    # ========== 控制线程实现 ==========

    def _start_control_thread(self):
        """启动控制线程"""
        if self._control_running:
            self.logger.warning(f"[{self.node_id}] Control thread already running")
            return

        self._control_running = True
        self._control_thread = threading.Thread(
            target=self._run_control_loop,
            name=f"GRPCControl-{self.node_id}",
            daemon=True,
        )
        self._control_thread.start()
        self.logger.info(f"[{self.node_id}] 控制线程已启动")

    def _stop_control_thread(self):
        """停止控制线程"""
        if not self._control_running:
            return

        self._control_running = False

        # 通过事件循环设置停止信号
        if self._control_loop and self._control_stop_event:
            try:
                # 跨线程调用：在控制线程的事件循环中设置Event
                self._control_loop.call_soon_threadsafe(
                    self._control_stop_event.set
                )
            except Exception as e:
                self.logger.debug(f"[{self.node_id}] 设置停止信号失败: {e}")

        # 等待线程结束（减少超时时间到3秒）
        if self._control_thread and self._control_thread.is_alive():
            self._control_thread.join(timeout=3.0)
            if self._control_thread.is_alive():
                self.logger.warning(f"[{self.node_id}] 控制线程未在3秒内结束")

        self.logger.debug(f"[{self.node_id}] 控制线程已停止")

    def _run_control_loop(self):
        """控制线程入口（在独立线程中运行）"""
        # 创建新的事件循环
        self._control_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._control_loop)

        try:
            self._control_loop.run_until_complete(self._control_main())
        except Exception as e:
            self.logger.error(f"[{self.node_id}] Control loop error: {e}")
        finally:
            self._control_loop.close()

    async def _control_main(self):
        """控制线程主逻辑"""
        # 在控制线程的事件循环中创建停止事件
        self._control_stop_event = asyncio.Event()

        tasks = []

        # 心跳任务
        if self.config.heartbeat_enabled:
            task = asyncio.create_task(self._heartbeat_loop())
            tasks.append(task)
            self.logger.debug(f"[{self.node_id}] 心跳任务已启动")

        # 健康检查任务
        task = asyncio.create_task(self._health_check_loop())
        tasks.append(task)
        self.logger.debug(f"[{self.node_id}] 健康检查任务已启动")

        # 等待所有任务
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            self.logger.debug(f"[{self.node_id}] Control tasks cancelled")
        except Exception as e:
            self.logger.exception(f"[{self.node_id}] Control tasks error: {e}")

    async def _heartbeat_loop(self):
        """心跳循环（控制线程）"""
        interval = self.config.heartbeat_interval
        timeout = self.config.heartbeat_timeout

        self.logger.debug(f"[{self.node_id}] 心跳循环启动: interval={interval}s, timeout={timeout}s")

        while True:
            try:
                # 使用wait_for实现可中断等待
                await asyncio.wait_for(
                    self._control_stop_event.wait(),
                    timeout=interval
                )
                # 收到停止信号，退出循环
                self.logger.debug(f"[{self.node_id}] 心跳循环收到停止信号")
                break
            except asyncio.TimeoutError:
                # 正常超时，继续发送心跳
                pass
            except asyncio.CancelledError:
                # 任务被取消
                self.logger.info(f"[{self.node_id}] 心跳任务被取消")
                break
            except Exception as e:
                self.logger.error(f"[{self.node_id}] 心跳循环异常: {e}")
                break

            # 发送心跳
            await self._send_heartbeats()

            # 检查超时
            await self._check_heartbeat_timeout(timeout)

        self.logger.debug(f"[{self.node_id}] 心跳循环已停止")

    async def _send_heartbeats(self):
        """发送心跳到所有连接的节点"""
        # *** 关键修复：检查运行状态，避免在关闭时继续发送 ***
        if not self._running:
            self.logger.debug(f"[{self.node_id}] 节点已停止，跳过心跳发送")
            return

        # 获取所有连接的节点
        connected_nodes = list(self._stubs.keys())
        if not connected_nodes:
            return

        # 发送心跳（跨线程调用主线程的 broadcast）
        try:
            results = await self._async_call_main_thread(
                self.broadcast,
                targets=connected_nodes,
                method="_heartbeat",
                payload={"timestamp": time.time()},
                timeout=self.config.heartbeat_interval,
            )

            # *** 关键修复：检查每个节点的结果，失败的节点跟踪连接失败 ***
            if results:
                for node_id, result in results.items():
                    if isinstance(result, Exception):
                        # 心跳失败，跟踪连接失败
                        self._track_connection_failure(node_id)
                    else:
                        # 心跳成功，记录成功
                        self._track_connection_success(node_id)

        except Exception as e:
            # 只在运行时记录错误，关闭时忽略
            if self._running:
                self.logger.debug(f"[{self.node_id}] 心跳发送失败: {e}")
                # 如果整个调用失败，对所有节点调用 _track_connection_failure
                for node_id in connected_nodes:
                    self._track_connection_failure(node_id)

    async def _check_heartbeat_timeout(self, timeout: float):
        """检查心跳超时"""
        now = time.time()
        with self._state_lock:
            connected_nodes = list(self._stubs.keys())
            for node_id in connected_nodes:
                last_time = self._shared_state["last_heartbeat"].get(node_id, now)
                if now - last_time > timeout:
                    self.logger.warning(
                        f"[{self.node_id}] 心跳超时: {node_id} "
                        f"(沉默 {now - last_time:.1f}s)"
                    )
                    # 标记为不健康
                    self._shared_state["peer_status"][node_id] = {"health": "unhealthy"}

    async def _health_check_loop(self):
        """健康检查循环（控制线程）"""
        interval = self.config.heartbeat_check_interval

        self.logger.info(f"[{self.node_id}] 健康检查循环启动: interval={interval}s")

        while True:
            try:
                # 使用wait_for实现可中断等待
                await asyncio.wait_for(
                    self._control_stop_event.wait(),
                    timeout=interval
                )
                # 收到停止信号，退出循环
                self.logger.debug(f"[{self.node_id}] 健康检查循环收到停止信号")
                break
            except asyncio.TimeoutError:
                # 正常超时，继续健康检查
                pass
            except asyncio.CancelledError:
                # 任务被取消
                self.logger.debug(f"[{self.node_id}] 健康检查任务被取消")
                break
            except Exception as e:
                self.logger.error(f"[{self.node_id}] 健康检查循环异常: {e}")
                break

            # 检查节点健康状态
            with self._state_lock:
                for node_id, status in list(self._shared_state["peer_status"].items()):
                    if status.get("health") == "unhealthy":
                        self.logger.warning(f"[{self.node_id}] 检测到不健康的节点: {node_id}")
                        # 可以触发告警、自动隔离等策略

        self.logger.debug(f"[{self.node_id}] 健康检查循环已停止")

    async def _async_call_main_thread(self, func, *args, **kwargs):
        """
        从控制线程异步调用主线程方法

        使用 run_coroutine_threadsafe 跨事件循环调用
        """
        main_loop = self.node._main_loop  # Node 保存的主循环引用
        if main_loop is None:
            self.logger.error(f"[{self.node_id}] Main loop not available")
            return None

        # 在主线程的事件循环中执行协程
        future = asyncio.run_coroutine_threadsafe(
            func(*args, **kwargs),
            main_loop
        )

        # 等待结果（在控制线程的事件循环中等待）
        try:
            # 使用 asyncio.wrap_future 将 concurrent.futures.Future 转为 asyncio.Future
            result = await asyncio.wrap_future(future)
            return result
        except Exception as e:
            self.logger.error(f"[{self.node_id}] Call main thread error: {e}")
            return None

    # ========== 连接失败跟踪和自动shutdown ==========

    def _track_connection_failure(self, node_id: str):
        """
        跟踪连接失败并检查是否需要触发shutdown

        Args:
            node_id: 失败的目标节点ID
        """
        self.logger.debug(
            f"[{self.node_id}] _track_connection_failure called for {node_id}, "
            f"critical_peers={list(self._critical_peers)}, "
            f"auto_shutdown={self._auto_shutdown}, "
            f"shutdown_triggered={self._shutdown_triggered}"
        )

        # 如果已经触发了shutdown，不再处理
        if self._shutdown_triggered:
            return

        # 如果不是关键节点，只记录日志
        if node_id not in self._critical_peers:
            self.logger.debug(
                f"[{self.node_id}] 节点 {node_id} 不在关键节点列表中，跳过失败跟踪"
            )
            return

        # 如果自动shutdown未启用，只记录日志
        if not self._auto_shutdown:
            self.logger.warning(
                f"[{self.node_id}] 自动shutdown未启用，跳过失败跟踪"
            )
            return

        current_time = time.time()

        # 记录首次失败时间
        with self._state_lock:
            if node_id not in self._first_failure_time:
                self._first_failure_time[node_id] = current_time
                self.logger.warning(
                    f"[{self.node_id}] 开始跟踪关键节点 {node_id} 的连接失败，"
                    f"将在 {self._max_wait_time}秒 后触发自动shutdown"
                )

            # 检查是否超过最大等待时间
            first_failure = self._first_failure_time[node_id]
            elapsed_time = current_time - first_failure

            if elapsed_time >= self._max_wait_time:
                self.logger.error(
                    f"[{self.node_id}] 无法连接到关键节点 {node_id}，"
                    f"已超过最大等待时间 {self._max_wait_time}秒（实际: {elapsed_time:.1f}秒），触发自动shutdown"
                )
                self._trigger_shutdown(node_id, elapsed_time)
            else:
                # 每30秒提醒一次
                if int(elapsed_time) % 30 == 0 and int(elapsed_time) > 0:
                    remaining = self._max_wait_time - elapsed_time
                    self.logger.warning(
                        f"[{self.node_id}] 仍无法连接到关键节点 {node_id}，"
                        f"已等待 {elapsed_time:.1f}秒，剩余 {remaining:.1f}秒"
                    )

    def _track_connection_success(self, node_id: str):
        """
        记录连接成功，重置失败跟踪

        Args:
            node_id: 成功连接的节点ID
        """
        with self._state_lock:
            if node_id in self._first_failure_time:
                elapsed = time.time() - self._first_failure_time[node_id]
                self.logger.info(
                    f"[{self.node_id}] 成功连接到关键节点 {node_id}，"
                    f"重置失败跟踪（之前失败 {elapsed:.1f}秒）"
                )
                del self._first_failure_time[node_id]

    def _trigger_shutdown(self, failed_node_id: str, elapsed_time: float):
        """
        触发系统shutdown

        Args:
            failed_node_id: 失败的节点ID
            elapsed_time: 经过的时间（秒）
        """
        if self._shutdown_triggered:
            return

        self._shutdown_triggered = True

        self.logger.critical(
            f"[{self.node_id}] ==================== 自动SHUTDOWN触发 ====================\n"
            f"  原因: 无法连接到关键节点 {failed_node_id}\n"
            f"  等待时间: {elapsed_time:.1f}秒 (最大: {self._max_wait_time}秒)\n"
            f"  关键节点列表: {list(self._critical_peers)}\n"
            f"  将在3秒后退出..."
            f"\n============================================================"
        )

        # 延迟3秒让日志有时间输出
        import sys
        import os
        time.sleep(3)

        # 触发进程退出
        # 使用os._exit而不是sys.exit，确保立即退出，不被异常处理捕获
        os._exit(1)

    # ========== 共享状态访问（线程安全）==========

    def update_heartbeat(self, node_id: str, timestamp: float):
        """更新心跳时间（任何线程都可调用）"""
        with self._state_lock:
            self._shared_state["last_heartbeat"][node_id] = timestamp
            # 恢复健康状态
            if node_id in self._shared_state["peer_status"]:
                self._shared_state["peer_status"][node_id] = {"health": "healthy"}

    def get_peer_status(self, node_id: str) -> Optional[Dict[str, Any]]:
        """获取节点状态（任何线程都可调用）"""
        with self._state_lock:
            return self._shared_state["peer_status"].get(node_id)


# ========== Proto 转换工具函数 ==========

def _to_proto(message: Message) -> pb2.NodeMessage:
    """将内部 Message 转换为 protobuf Message"""
    proto_error = None
    if message.error:
        proto_error = pb2.ErrorInfo(
            code=message.error.code,
            message=message.error.message,
            details_json=json.dumps(message.error.details) if message.error.details else "",
            stack_trace=message.error.stack_trace or "",
        )
    
    return pb2.NodeMessage(
        id=message.id,
        type=_message_type_to_proto(message.type),
        method=message.method,
        source=message.source,
        target=message.target,
        payload=message.payload,
        correlation_id=message.correlation_id or "",
        timestamp=message.timestamp,
        metadata=message.metadata,
        error=proto_error,
    )


def _from_proto(proto_msg: pb2.NodeMessage) -> Message:
    """将 protobuf Message 转换为内部 Message"""
    error = None
    if proto_msg.HasField("error") and proto_msg.error.code:
        error = ErrorInfo(
            code=proto_msg.error.code,
            message=proto_msg.error.message,
            details=json.loads(proto_msg.error.details_json) if proto_msg.error.details_json else None,
            stack_trace=proto_msg.error.stack_trace or None,
        )
    
    return Message(
        id=proto_msg.id,
        type=_message_type_from_proto(proto_msg.type),
        method=proto_msg.method,
        source=proto_msg.source,
        target=proto_msg.target,
        payload=proto_msg.payload,
        correlation_id=proto_msg.correlation_id or None,
        timestamp=proto_msg.timestamp,
        metadata=dict(proto_msg.metadata),
        error=error,
    )


def _message_type_to_proto(msg_type: MessageType) -> int:
    """转换消息类型到 proto 枚举"""
    mapping = {
        MessageType.REQUEST: pb2.MESSAGE_TYPE_REQUEST,
        MessageType.RESPONSE: pb2.MESSAGE_TYPE_RESPONSE,
        MessageType.NOTIFY: pb2.MESSAGE_TYPE_NOTIFY,
    }
    return mapping.get(msg_type, pb2.MESSAGE_TYPE_UNSPECIFIED)


def _message_type_from_proto(proto_type: int) -> MessageType:
    """从 proto 枚举转换消息类型"""
    mapping = {
        pb2.MESSAGE_TYPE_REQUEST: MessageType.REQUEST,
        pb2.MESSAGE_TYPE_RESPONSE: MessageType.RESPONSE,
        pb2.MESSAGE_TYPE_NOTIFY: MessageType.NOTIFY,
    }
    return mapping.get(proto_type, MessageType.REQUEST)
