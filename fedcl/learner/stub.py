"""
MOE-FedCL 客户端存根实现
moe_fedcl/learner/stub.py
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, Optional, Callable

from .base_learner import BaseLearner
from ..communication.base import CommunicationManagerBase
from ..connection.manager import ConnectionManager
from ..exceptions import RegistrationError, ValidationError
from ..types import (
    RegistrationRequest, RegistrationResponse, RegistrationStatus,
    TrainingRequest, TrainingResponse
)
from ..utils.auto_logger import get_sys_logger, get_train_logger


class StubConfig:
    """存根配置"""
    
    def __init__(self,
                 auto_register: bool = True,
                 registration_retry_attempts: int = 3,
                 registration_retry_delay: float = 5.0,
                 request_timeout: float = 120.0,
                 max_concurrent_requests: int = 5):
        self.auto_register = auto_register
        self.registration_retry_attempts = registration_retry_attempts
        self.registration_retry_delay = registration_retry_delay
        self.request_timeout = request_timeout
        self.max_concurrent_requests = max_concurrent_requests


class LearnerStub:
    """客户端存根统一类 - 接收服务端请求并调用本地BaseLearner"""
    
    def __init__(self,
                 learner: BaseLearner,
                 communication_manager: CommunicationManagerBase,
                 connection_manager: ConnectionManager,
                 config: Optional[StubConfig] = None):
        """
        初始化客户端存根
        
        Args:
            learner: 本地学习器实例
            communication_manager: 通信管理器
            connection_manager: 连接管理器
            config: 存根配置
        """
        self.learner = learner
        self.logger = get_sys_logger()
        self.train_logger = get_train_logger(learner.client_id)
        self.communication_manager = communication_manager
        self.connection_manager = connection_manager
        self.config = config or StubConfig()
        
        # 注册状态
        self._registration_status = RegistrationStatus.UNREGISTERED
        self._server_info: Optional[Dict[str, Any]] = None
        
        # 请求处理
        self._request_handlers: Dict[str, Callable] = {}
        self._active_requests: Dict[str, asyncio.Task] = {}
        self._request_semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        
        # 生命周期
        self._listening = False
        self._listener_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        
        # 注册请求处理器
        self._setup_request_handlers()

    
    def _setup_request_handlers(self):
        """设置请求处理器"""
        self._request_handlers = {
            "train": self.handle_train_request,
            "evaluate": self.handle_evaluate_request,
            "get_model": self.handle_get_model_request,
            "set_model": self.handle_set_model_request,
            "get_info": self.handle_get_info_request,
            "ping": self.handle_ping_request
        }

        
        # 注意：动态调用在process_business_request中特殊处理
        # 不在这里注册，避免与标准方法冲突
    
    # ==================== 客户端注册方法 ====================
    
    async def register_to_server(self) -> RegistrationResponse:
        """向服务端注册客户端
        
        Returns:
            RegistrationResponse: 注册响应结果
            
        Raises:
            RegistrationError: 注册失败
        """
        async with self._lock:
            if self._registration_status in [RegistrationStatus.REGISTERED, RegistrationStatus.ACTIVE]:
                return RegistrationResponse(
                    success=True,
                    client_id=self.learner.client_id,
                    server_info=self._server_info or {}
                )
            
            self._registration_status = RegistrationStatus.REGISTERING
        
        # 获取客户端地址信息（从transport获取）
        client_address = self._get_client_address()

        # 创建注册请求
        registration_request = RegistrationRequest(
            client_id=self.learner.client_id,
            client_type="learner",
            capabilities=["train", "evaluate"],
            metadata={
                "data_statistics": self.learner.get_data_statistics(),
                "model_config": self.learner.model_config,
                "training_config": self.learner.training_config,
                "registration_time": datetime.now().isoformat(),
                # 添加客户端地址信息，供服务端向客户端发送请求使用
                "client_address": client_address
            }
        )
        
        # 尝试注册
        for attempt in range(self.config.registration_retry_attempts):
            try:
                response = await self.communication_manager.register_client(registration_request)
                
                if response.success:
                    async with self._lock:
                        self._registration_status = RegistrationStatus.REGISTERED
                        self._server_info = response.server_info
                    
                    self.logger.debug(f"Client {self.learner.client_id} registered successfully")
                    return response
                else:
                    self.logger.error(f"Registration attempt {attempt + 1} failed: {response.error_message}")
                    
            except Exception as e:
                self.logger.exception(f"Registration attempt {attempt + 1} error: {e}")
            
            # 等待重试
            if attempt < self.config.registration_retry_attempts - 1:
                await asyncio.sleep(self.config.registration_retry_delay)
        
        # 所有尝试都失败
        async with self._lock:
            self._registration_status = RegistrationStatus.ERROR
        
        raise RegistrationError(f"Failed to register after {self.config.registration_retry_attempts} attempts")
    
    async def unregister_from_server(self) -> bool:
        """从服务端注销
        
        Returns:
            bool: 注销是否成功
        """
        try:
            async with self._lock:
                if self._registration_status == RegistrationStatus.UNREGISTERED:
                    return True
                
                self._registration_status = RegistrationStatus.UNREGISTERED
            
            # 发送注销请求
            success = await self.communication_manager.unregister_client(self.learner.client_id)
            
            if success:
                self._server_info = None
                self.logger.info(f"Client {self.learner.client_id} unregistered successfully")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Unregistration error: {e}")
            return False
    
    async def update_registration_info(self, updates: Dict[str, Any]) -> bool:
        """更新注册信息
        
        Args:
            updates: 更新的信息
            
        Returns:
            bool: 更新是否成功
        """
        try:
            return await self.communication_manager.update_client_info(
                self.learner.client_id, updates
            )
        except Exception as e:
            self.logger.error(f"Update registration info error: {e}")
            return False
    
    def get_registration_status(self) -> RegistrationStatus:
        """获取注册状态"""
        return self._registration_status
    
    async def handle_registration_failure(self):
        """处理注册失败"""
        async with self._lock:
            self._registration_status = RegistrationStatus.ERROR

        self.logger.error(f"Registration failed for client {self.learner.client_id}")

        # 可以添加自动重试逻辑
        if self.config.auto_register:
            self.logger.info("Attempting automatic re-registration...")
            asyncio.create_task(self._auto_retry_registration())
    
    async def _auto_retry_registration(self):
        """自动重试注册"""
        await asyncio.sleep(30)  # 等待30秒后重试
        try:
            await self.register_to_server()
        except Exception as e:
            self.logger.error(f"Auto retry registration failed: {e}")
    
    # ==================== RPC处理方法 ====================
    
    async def handle_train_request(self, request: TrainingRequest) -> TrainingResponse:
        """处理训练请求

        Args:
            request: 训练请求

        Returns:
            TrainingResponse: 训练响应
        """
        start_time = datetime.now()

        try:
            self.logger.debug(f"[STUB] 收到训练请求: {request.client_id}, 方法: {request.method_name}")
            self.logger.debug(f"[STUB] 训练参数: {request.parameters}")

            # 参数验证
            if not self.validate_request_parameters(request):
                return TrainingResponse(
                    request_id=request.request_id,
                    client_id=request.client_id,
                    success=False,
                    error_message="Invalid training parameters"
                )

            # 确保学习器已初始化
            if not self.learner._is_initialized:
                await self.learner.initialize()

            # 执行训练
            result = await self.learner.train(request.parameters)
            self.train_logger.debug(f"[STUB] 接收到learner {result.client_id} 的训练结果")

            # 记录训练历史
            await self.learner._record_training(request.parameters, result)

            # learner.train()已经返回TrainingResponse对象，直接使用它
            # 只需要填充request_id字段（如果learner没有填充的话）
            if isinstance(result, TrainingResponse):
                if not result.request_id:
                    result.request_id = request.request_id
                self.logger.debug(f"[STUB] 返回响应: success={result.success}, result={result.result.keys()}")
                return result
            else:
                # 如果learner返回的不是TrainingResponse（兼容旧接口），则包装它
                execution_time = (datetime.now() - start_time).total_seconds()
                response = TrainingResponse(
                    request_id=request.request_id,
                    client_id=request.client_id,
                    success=True,
                    result=result,
                    execution_time=execution_time
                )
                self.logger.debug(f"[STUB] 返回响应: success={response.success}, result={response.result.keys()}")
                return response

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Training failed: {str(e)}"
            self.logger.exception(f"[STUB] 训练失败: {error_msg}")

            return TrainingResponse(
                request_id=request.request_id,
                client_id=request.client_id,
                success=False,
                error_message=error_msg,
                execution_time=execution_time
            )
    
    async def handle_evaluate_request(self, request: TrainingRequest) -> TrainingResponse:
        """处理评估请求"""
        start_time = datetime.now()
        
        try:
            # 参数验证
            if not self.validate_request_parameters(request):
                return TrainingResponse(
                    request_id=request.request_id,
                    client_id=request.client_id,
                    success=False,
                    error_message="Invalid evaluation parameters"
                )
            
            # 确保学习器已初始化
            if not self.learner._is_initialized:
                await self.learner.initialize()
            
            # 执行评估
            result = await self.learner.evaluate(request.parameters)
            
            # 记录评估历史
            await self.learner._record_evaluation(request.parameters, result)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return TrainingResponse(
                request_id=request.request_id,
                client_id=request.client_id,
                success=True,
                result=result,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Evaluation failed: {str(e)}"
            self.logger.exception(error_msg)
            
            return TrainingResponse(
                request_id=request.request_id,
                client_id=request.client_id,
                success=False,
                error_message=error_msg,
                execution_time=execution_time
            )
    
    async def handle_get_model_request(self, request: TrainingRequest) -> TrainingResponse:
        """处理获取模型请求"""
        try:
            model_data = await self.learner.get_local_model()
            
            return TrainingResponse(
                request_id=request.request_id,
                client_id=request.client_id,
                success=True,
                result=model_data
            )
            
        except Exception as e:
            error_msg = f"Get model failed: {str(e)}"
            self.logger.exception(error_msg)
            
            return TrainingResponse(
                request_id=request.request_id,
                client_id=request.client_id,
                success=False,
                error_message=error_msg
            )
    
    async def handle_set_model_request(self, request: TrainingRequest) -> TrainingResponse:
        """处理设置模型请求"""
        try:
            # 获取模型数据
            model_data = request.parameters.get("model_data")
            if model_data is None:
                return TrainingResponse(
                    request_id=request.request_id,
                    client_id=request.client_id,
                    success=False,
                    error_message="No model data provided"
                )
            
            # 设置模型
            success = await self.learner.set_local_model(model_data)
            
            return TrainingResponse(
                request_id=request.request_id,
                client_id=request.client_id,
                success=success,
                result={"model_updated": success}
            )
            
        except Exception as e:
            error_msg = f"Set model failed: {str(e)}"
            self.logger.exception(error_msg)
            
            return TrainingResponse(
                request_id=request.request_id,
                client_id=request.client_id,
                success=False,
                error_message=error_msg
            )
    
    async def handle_get_info_request(self, request: TrainingRequest) -> TrainingResponse:
        """处理获取信息请求"""
        try:
            info = self.learner.get_learner_info()
            info.update({
                "registration_status": self._registration_status.value,
                "server_info": self._server_info,
                "active_requests": len(self._active_requests)
            })
            
            return TrainingResponse(
                request_id=request.request_id,
                client_id=request.client_id,
                success=True,
                result=info
            )
            
        except Exception as e:
            error_msg = f"Get info failed: {str(e)}"
            self.logger.exception(error_msg)
            
            return TrainingResponse(
                request_id=request.request_id,
                client_id=request.client_id,
                success=False,
                error_message=error_msg
            )
    
    async def handle_ping_request(self, request: TrainingRequest) -> TrainingResponse:
        """处理ping请求"""
        return TrainingResponse(
            request_id=request.request_id,
            client_id=request.client_id,
            success=True,
            result={
                "status": "alive",
                "timestamp": datetime.now().isoformat(),
                "registration_status": self._registration_status.value
            }
        )
    
    # ==================== 动态方法调用处理 ====================
    
    async def handle_dynamic_call_request(self, request: TrainingRequest, method_name: str) -> TrainingResponse:
        """处理动态方法调用请求
        
        Args:
            request: 训练请求对象
            method_name: 要调用的实际方法名
            
        Returns:
            TrainingResponse: 响应对象
        """
        try:
            # 检查learner是否有这个方法
            if not hasattr(self.learner, method_name):
                return TrainingResponse(
                    request_id=request.request_id,
                    client_id=request.client_id,
                    success=False,
                    error_message=f"Method '{method_name}' does not exist on learner"
                )
            
            # 获取方法对象
            method = getattr(self.learner, method_name)
            
            # 检查是否为可调用对象
            if not callable(method):
                return TrainingResponse(
                    request_id=request.request_id,
                    client_id=request.client_id,
                    success=False,
                    error_message=f"'{method_name}' is not a callable method"
                )
            
            # 安全检查：防止调用危险方法
            if self._is_method_safe(method_name):
                return TrainingResponse(
                    request_id=request.request_id,
                    client_id=request.client_id,
                    success=False,
                    error_message=f"Method '{method_name}' is not allowed for security reasons"
                )
            
            # 从请求参数中提取args和kwargs
            parameters = request.parameters
            args = parameters.get('args', [])
            kwargs = parameters.get('kwargs', {})
            
            # 执行方法调用
            if asyncio.iscoroutinefunction(method):
                # 异步方法
                result = await method(*args, **kwargs)
            else:
                # 同步方法
                result = method(*args, **kwargs)
            
            # 构造成功响应
            return TrainingResponse(
                request_id=request.request_id,
                client_id=request.client_id,
                success=True,
                result=result
            )
            
        except TypeError as e:
            # 参数错误
            return TrainingResponse(
                request_id=request.request_id,
                client_id=request.client_id,
                success=False,
                error_message=f"Invalid arguments for method '{method_name}': {str(e)}"
            )
        except Exception as e:
            # 其他执行错误
            return TrainingResponse(
                request_id=request.request_id,
                client_id=request.client_id,
                success=False,
                error_message=f"Error executing method '{method_name}': {str(e)}"
            )
    
    def _is_method_safe(self, method_name: str) -> bool:
        """检查方法是否安全（返回True表示不安全，应该被阻止）
        
        Args:
            method_name: 方法名
            
        Returns:
            bool: True表示不安全，应该阻止调用
        """
        # 危险方法黑名单
        dangerous_methods = [
            '__', 'eval', 'exec', 'compile', 'open', 'file', 'input', 
            'raw_input', 'reload', '__import__', 'delattr', 'setattr',
            'globals', 'locals', 'vars', 'dir', 'help', 'copyright', 
            'credits', 'license', 'quit', 'exit', 'breakpoint',
            'execfile', 'reload', 'apply', 'buffer', 'coerce'
        ]
        
        # 检查方法名是否包含危险关键字
        method_lower = method_name.lower()
        for dangerous in dangerous_methods:
            if dangerous in method_lower:
                return True
        
        # 检查是否为私有方法（以_开头但不是标准的魔术方法）
        if method_name.startswith('_') and not (method_name.startswith('__') and method_name.endswith('__')):
            return True
        
        return False
    
    # ==================== 请求分发方法 ====================
    
    async def process_business_request(self, source: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理业务请求
        
        Args:
            source: 请求来源
            request_data: 请求数据
            
        Returns:
            Dict[str, Any]: 响应数据
        """
        try:
            # 解析请求
            if isinstance(request_data, dict) and "data" in request_data:
                # 处理封装的请求
                inner_data = request_data["data"]
                if isinstance(inner_data, dict):
                    method_name = inner_data.get("method_name", "unknown")
                    parameters = inner_data.get("parameters", {})
                    request_id = inner_data.get("request_id", "unknown")
                    client_id = inner_data.get("client_id", source)
                else:
                    method_name = "unknown"
                    parameters = inner_data
                    request_id = request_data.get("request_id", "unknown")
                    client_id = source
            else:
                # 直接请求格式
                method_name = request_data.get("method_name", "unknown")
                parameters = request_data.get("parameters", {})
                request_id = request_data.get("request_id", "unknown")
                client_id = request_data.get("client_id", source)
            
            # 创建训练请求对象
            request = TrainingRequest(
                request_id=request_id,
                client_id=client_id,
                method_name=method_name,
                parameters=parameters
            )
            
            # 路由到对应方法
            response = await self.route_request_to_method(method_name, request)
            
            # 转换为字典格式
            return self._serialize_response(response)
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Request processing failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    async def route_request_to_method(self, method_name: str, request: TrainingRequest) -> TrainingResponse:
        """将请求路由到对应方法
        
        Args:
            method_name: 方法名
            request: 请求对象
            
        Returns:
            TrainingResponse: 响应对象
        """
        # 限制并发请求数
        async with self._request_semaphore:
            # 首先检查是否为动态调用
            if method_name.startswith("__dynamic_call__"):
                # 提取真实方法名
                real_method_name = method_name[len("__dynamic_call__"):]
                return await self.handle_dynamic_call_request(request, real_method_name)
            
            # 查找标准处理器
            if method_name in self._request_handlers:
                handler = self._request_handlers[method_name]
                
                # 记录活跃请求
                task_id = f"{method_name}_{request.request_id}"
                
                try:
                    # 创建处理任务
                    task = asyncio.create_task(handler(request))
                    self._active_requests[task_id] = task
                    
                    # 等待结果
                    response = await asyncio.wait_for(task, timeout=self.config.request_timeout)
                    return response
                    
                except asyncio.TimeoutError:
                    return TrainingResponse(
                        request_id=request.request_id,
                        client_id=request.client_id,
                        success=False,
                        error_message=f"Request timeout after {self.config.request_timeout}s"
                    )
                finally:
                    # 清理活跃请求记录
                    if task_id in self._active_requests:
                        del self._active_requests[task_id]
            else:
                return self.handle_method_not_found(method_name, request)
    
    def validate_request_parameters(self, request: TrainingRequest) -> bool:
        """验证请求参数
        
        Args:
            request: 请求对象
            
        Returns:
            bool: 参数是否有效
        """
        try:
            if not request.request_id or not request.client_id:
                return False
            
            if not request.method_name:
                return False
            
            if not isinstance(request.parameters, dict):
                return False
            
            return True
            
        except Exception:
            return False
    
    def handle_method_not_found(self, method_name: str, request: TrainingRequest) -> TrainingResponse:
        """处理方法未找到的情况"""
        available_methods = list(self._request_handlers.keys())
        error_msg = f"Method '{method_name}' not found. Available methods: {available_methods}"
        
        return TrainingResponse(
            request_id=request.request_id,
            client_id=request.client_id,
            success=False,
            error_message=error_msg
        )
    
    def handle_validation_error(self, error: ValidationError, request: TrainingRequest) -> TrainingResponse:
        """处理验证错误"""
        return TrainingResponse(
            request_id=request.request_id,
            client_id=request.client_id,
            success=False,
            error_message=f"Validation error: {str(error)}"
        )
    
    def handle_execution_error(self, error: Exception, request: TrainingRequest) -> TrainingResponse:
        """处理执行错误"""
        return TrainingResponse(
            request_id=request.request_id,
            client_id=request.client_id,
            success=False,
            error_message=f"Execution error: {str(error)}"
        )
    
    def log_request_error(self, request: Any, error: Exception):
        """记录请求错误"""
        self.logger.error(f"Request details: {request}")
        self.logger.exception(f"Request error for client {self.learner.client_id}: {error}")
        
    
    def _serialize_response(self, response: TrainingResponse) -> Dict[str, Any]:
        """序列化响应对象"""
        return {
            "request_id": response.request_id,
            "client_id": response.client_id,
            "success": response.success,
            "result": response.result,
            "error_message": response.error_message,
            "execution_time": response.execution_time,
            "timestamp": response.timestamp.isoformat()
        }
    
    # ==================== 生命周期方法 ====================
    
    async def start_listening(self) -> None:
        """启动监听服务端请求"""
        # 设置通信管理器的消息处理器
        await self._setup_communication_handlers()

        if self._listening:
            self.logger.info(f"Client {self.learner.client_id} is already listening")
            return
        
        self._listening = True
        
        try:
            # 初始化学习器
            if not self.learner._is_initialized:
                await self.learner.initialize()
            
            # 如果配置了自动注册，则注册到服务端
            if self.config.auto_register:
                await self.register_to_server()

            
            # 启动监听任务
            # self._listener_task = asyncio.create_task(self._request_listener())

            self.logger.info(f"Client {self.learner.client_id} started listening for requests")

        except Exception as e:
            self._listening = False
            raise e
    
    async def stop_listening(self) -> None:
        """停止监听"""
        if not self._listening:
            return
        
        self._listening = False
        
        # 取消监听任务
        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
        
        # 取消所有活跃请求
        for task in list(self._active_requests.values()):
            if not task.done():
                task.cancel()
        
        self._active_requests.clear()
        
        # 从服务端注销
        await self.unregister_from_server()
        
        self.logger.info(f"Client {self.learner.client_id} stopped listening")
    
    async def cleanup(self) -> None:
        """清理存根资源"""
        await self.stop_listening()
        
        # 清理学习器
        await self.learner.cleanup()
        
        # 重置状态
        async with self._lock:
            self._registration_status = RegistrationStatus.UNREGISTERED
            self._server_info = None
            self._request_handlers.clear()
    
    async def _setup_communication_handlers(self):
        """设置通信处理器"""
        # 为不同的通信管理器类型设置处理器
        if hasattr(self.communication_manager, 'register_message_handler'):
            # 注册业务消息处理器
            self.communication_manager.register_message_handler(
                "business_request", self.process_business_request
            )
            self.logger.debug("Registered business request handler")

        self.logger.debug(f"message_handlers:{self.communication_manager}, stub:{self}")
        # 设置传输层处理器
        if hasattr(self.communication_manager.transport, 'register_request_handler'):
            self.communication_manager.transport.register_request_handler(
                self.learner.client_id, self.process_business_request
            )
    
    async def _request_listener(self):
        """请求监听循环"""
        while self._listening:
            try:
                # 这里的具体实现依赖于通信管理器的类型
                # Memory模式: 直接处理函数调用
                # Process模式: 从队列接收消息
                # Network模式: HTTP服务器自动处理

                await asyncio.sleep(0.1)  # 防止过度占用CPU

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.exception(f"Request listener error: {e}")
                await asyncio.sleep(1)

    def _get_client_address(self) -> Dict[str, Any]:
        """获取客户端地址信息（供服务端回调使用）

        Returns:
            Dict包含:
            - host: 客户端监听的主机地址（可连接的地址）
            - port: 客户端监听的端口
            - url: 完整的URL地址（http://host:port）
        """
        try:
            transport = self.communication_manager.transport

            # 从transport获取实际监听的地址和端口
            host = getattr(transport, 'host', '127.0.0.1')
            port = getattr(transport, 'port', None)

            # ✅ 智能转换 0.0.0.0 为可连接的地址
            if host == '0.0.0.0' or host == '' or host is None:
                # 检测通信模式
                mode = getattr(transport, 'mode', 'process')

                if mode == 'network':
                    # Network 模式：尝试获取实际的网卡 IP
                    try:
                        import socket
                        # 获取本机的实际 IP 地址（不是 127.0.0.1）
                        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                        try:
                            # 连接到外部地址（不会真正发送数据）
                            s.connect(('8.8.8.8', 80))
                            actual_ip = s.getsockname()[0]
                        finally:
                            s.close()

                        if actual_ip and actual_ip != '0.0.0.0':
                            host = actual_ip
                            self.logger.info(f"Network模式：将监听地址 0.0.0.0 转换为网卡地址: {host}")
                        else:
                            host = '127.0.0.1'
                            self.logger.warning(f"Network模式：无法获取网卡IP，使用本地地址: {host}")
                    except Exception as e:
                        host = '127.0.0.1'
                        self.logger.warning(f"Network模式：获取网卡IP失败({e})，使用本地地址: {host}")
                else:
                    # Process/Memory 模式：使用本地地址
                    host = '127.0.0.1'
                    self.logger.debug(f"{mode}模式：将监听地址 0.0.0.0 转换为本地地址: {host}")

            # Process模式和Network模式都使用NetworkTransport
            # 需要获取HTTP服务器的实际端口
            if port is None or port == 0:
                # 尝试从transport的配置中获取
                if hasattr(transport, 'config') and hasattr(transport.config, 'specific_config'):
                    port = transport.config.specific_config.get('port', 0)

            # 如果仍然是0，说明是自动分配的端口，从_server_port获取
            if port == 0 and hasattr(transport, '_server_port'):
                port = transport._server_port

            # 构造完整URL
            if port and port != 0:
                url = f"http://{host}:{port}"
            else:
                url = None

            self.logger.debug(f"客户端注册地址: host={host}, port={port}, url={url}")

            return {
                "host": host,
                "port": port,
                "url": url
            }
        except Exception as e:
            self.logger.warning(f"Failed to get client address: {e}")
            return {
                "host": "127.0.0.1",
                "port": 0,
                "url": None
            }
