"""
Learner代理系统

提供在trainer中透明访问客户端learner功能的代理接口，支持真联邦和伪联邦模式。
用户可以像操作本地learner一样操作远程客户端learner，大大降低开发复杂度。

设计理念：
- 基于现有的learner概念，符合架构一致性
- 支持任意learner类型和方法调用  
- 自动适配真伪联邦通信模式
- 提供简洁直观的API接口
"""

import time
import uuid
from typing import Any, Dict, List, Optional, Union, Callable
from contextlib import contextmanager
from loguru import logger

from ..comm.transparent_communication import BaseCommunicationBackend, Message


class ModelProxy:
    """
    模型代理类
    
    提供对客户端learner内部模型的透明访问，支持：
    1. 前向传播
    2. 模型状态管理（train/eval模式）
    3. 参数获取和设置
    """
    
    def __init__(self, client_id: str, learner_name: str, communication_backend):
        self.client_id = client_id
        self.learner_name = learner_name
        self.comm = communication_backend
        self.logger = logger.bind(component="ModelProxy", client=client_id)
        
        # 模型状态缓存
        self._is_training = True
        
    def forward(self, data: Any) -> Any:
        """
        模型前向传播代理
        
        Args:
            data: 输入数据
            
        Returns:
            Any: 模型输出结果
        """
        return self._call_model_method("forward", data)
    
    def __call__(self, data: Any) -> Any:
        """支持直接调用语法: model(data)"""
        return self.forward(data)
    
    @contextmanager
    def eval(self):
        """
        模型评估模式上下文管理器
        
        使用方式:
            with model_proxy.eval():
                predictions = model_proxy(test_data)
        """
        original_mode = self._is_training
        try:
            self.set_eval_mode()
            yield self
        finally:
            if original_mode:
                self.set_train_mode()
    
    @contextmanager  
    def train(self):
        """
        模型训练模式上下文管理器
        
        使用方式:
            with model_proxy.train():
                model_proxy.train_step(batch_data)
        """
        original_mode = self._is_training
        try:
            self.set_train_mode()
            yield self
        finally:
            if not original_mode:
                self.set_eval_mode()
    
    def set_train_mode(self):
        """设置为训练模式"""
        self._call_model_method("train")
        self._is_training = True
    
    def set_eval_mode(self):
        """设置为评估模式"""
        self._call_model_method("eval")
        self._is_training = False
    
    def get_parameters(self) -> Dict[str, Any]:
        """获取模型参数"""
        return self._call_model_method("state_dict")
    
    def set_parameters(self, state_dict: Dict[str, Any]):
        """设置模型参数"""
        return self._call_model_method("load_state_dict", state_dict)
    
    def _call_model_method(self, method_name: str, *args, **kwargs) -> Any:
        """调用客户端模型的方法"""
        # 构造模型方法调用请求
        request = {
            "target": "model",
            "method": method_name, 
            "args": args,
            "kwargs": kwargs
        }
        
        return self.comm.call_learner_method(
            self.client_id, self.learner_name, "_proxy_call_model", request
        )


class LearnerProxy:
    """
    Learner代理类
    
    提供对客户端learner的透明访问，支持：
    1. 调用learner的任意方法
    2. 访问learner的模型代理
    3. 获取和设置learner状态
    4. 批量操作支持
    5. 🆕 用户自定义方法的自动映射（train_task、evaluate_task等）
    """
    
    def __init__(self, client_id: str, learner_name: str, communication_backend):
        self.client_id = client_id
        self.learner_name = learner_name  
        self.comm = communication_backend
        self.logger = logger.bind(component="LearnerProxy", client=client_id, learner=learner_name)
        
        # 创建模型代理
        self.model = ModelProxy(client_id, learner_name, communication_backend)
        
        # 🆕 方法映射缓存 - 用于自动适配用户自定义方法
        self._method_mapping_cache: Optional[Dict[str, str]] = None
        
    def __getattr__(self, name: str) -> Callable:
        """
        动态代理learner的所有方法
        
        🆕 增强功能：自动适配用户自定义方法名：
        - train_epoch -> train_task（如果用户定义了train_task）
        - evaluate -> evaluate_task（如果用户定义了evaluate_task）
        
        使用方式:
            proxy.train_task(data)  # 自动代理到远程learner.train_task()
            proxy.train_epoch(data)  # 自动映射到train_task（如果存在）
            proxy.get_current_personal_weights()  # 代理任意方法
        """
        def method_proxy(*args, **kwargs):
            # 🆕 使用增强的call_method，支持自动方法映射
            return self.call_method_with_mapping(name, *args, **kwargs)
        
        method_proxy.__name__ = name
        method_proxy.__doc__ = f"代理方法: {self.learner_name}.{name}()"
        return method_proxy
    
    def call_method(self, method_name: str, *args, **kwargs) -> Any:
        """
        调用客户端learner的指定方法
        
        Args:
            method_name: 方法名
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            Any: 方法执行结果
        """
        try:
            start_time = time.time()
            
            result = self.comm.call_learner_method(
                self.client_id, self.learner_name, method_name, *args, **kwargs
            )
            
            execution_time = time.time() - start_time
            self.logger.debug(f"方法调用成功: {method_name} (耗时: {execution_time:.3f}s)")
            
            return result
            
        except Exception as e:
            self.logger.error(f"方法调用失败: {method_name} - {e}")
            raise
    
    def call_method_with_mapping(self, method_name: str, *args, **kwargs) -> Any:
        """
        🆕 增强的方法调用，支持用户自定义方法的自动映射
        
        自动适配逻辑：
        1. 先尝试调用原始方法名
        2. 如果失败，尝试映射到用户自定义方法名
        
        Args:
            method_name: 方法名
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            Any: 方法执行结果
        """
        # 先尝试直接调用原始方法名
        try:
            return self.call_method(method_name, *args, **kwargs)
        except Exception as original_error:
            # 如果失败，尝试方法映射
            mapped_method = self._get_mapped_method_name(method_name)
            
            if mapped_method and mapped_method != method_name:
                try:
                    self.logger.debug(f"方法映射: {method_name} -> {mapped_method}")
                    return self.call_method(mapped_method, *args, **kwargs)
                except Exception as mapped_error:
                    self.logger.debug(f"映射方法也失败: {mapped_method} - {mapped_error}")
            
            # 两种方法都失败，抛出原始错误
            raise original_error
    
    def _get_mapped_method_name(self, method_name: str) -> Optional[str]:
        """
        获取方法名的映射
        
        映射规则：
        - train_epoch -> train_task
        - evaluate -> evaluate_task  
        - train_on_client -> train_task
        
        Args:
            method_name: 原始方法名
            
        Returns:
            Optional[str]: 映射后的方法名，如果没有映射则返回None
        """
        # 常用方法映射表
        mapping_rules = {
            "train_epoch": "train_task",
            "evaluate": "evaluate_task",
            "train_on_client": "train_task"
        }
        
        return mapping_rules.get(method_name)
    
    async def _get_learner_methods(self) -> List[str]:
        """
        🆕 获取远程 learner 的所有可用方法名列表
        
        用于动态检测用户自定义learner的实际方法名
        
        Returns:
            List[str]: 方法名列表
        """
        try:
            # 调用特殊方法获取learner的所有方法
            methods = await self.call_method("__dir__")
            # 过滤出公开方法（不以_开头）
            return [method for method in methods if not method.startswith("_")]
        except Exception as e:
            self.logger.debug(f"无法获取远程learner方法列表: {e}")
            return []
    
    def forward(self, data: Any) -> Any:
        """快捷方式：模型前向传播"""
        return self.model.forward(data)
    
    def evaluate_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """快捷方式：执行评估任务（🆕 支持自动映射）"""
        return self.call_method_with_mapping("evaluate_task", task_data)
    
    def train_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """快捷方式：执行训练任务（🆕 支持自动映射）"""
        return self.call_method_with_mapping("train_task", task_data)
    
    def train_epoch(self, **kwargs) -> Dict[str, Any]:
        """🆕 新增：兼容性快捷方式，自动映射到train_task"""
        return self.call_method_with_mapping("train_epoch", **kwargs)
    
    def evaluate(self, **kwargs) -> Dict[str, Any]:
        """🆕 新增：兼容性快捷方式，自动映射到evaluate_task"""
        return self.call_method_with_mapping("evaluate", **kwargs)
    
    def get_state(self) -> Dict[str, Any]:
        """获取learner状态"""
        return self.call_method("get_state") if hasattr(self, "get_state") else {}
    
    def set_state(self, state: Dict[str, Any]):
        """设置learner状态"""
        if hasattr(self, "set_state"):
            return self.call_method("set_state", state)


class BatchLearnerProxy:
    """
    批量Learner代理类
    
    支持对多个客户端learner进行批量操作
    """
    
    def __init__(self, learner_proxies: List[LearnerProxy]):
        self.proxies = learner_proxies
        self.logger = logger.bind(component="BatchLearnerProxy")
    
    def batch_call(self, method_name: str, *args, **kwargs) -> Dict[str, Any]:
        """
        批量调用多个learner的相同方法（🆕 支持自动映射）
        
        Args:
            method_name: 方法名
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            Dict[str, Any]: {client_id: result} 的结果字典
        """
        results = {}
        
        for proxy in self.proxies:
            try:
                # 🆕 使用增强的call_method_with_mapping
                result = proxy.call_method_with_mapping(method_name, *args, **kwargs)
                results[proxy.client_id] = result
            except Exception as e:
                self.logger.error(f"批量调用失败 - 客户端 {proxy.client_id}: {e}")
                results[proxy.client_id] = {"error": str(e)}
        
        return results
    
    def batch_forward(self, data: Any) -> Dict[str, Any]:
        """批量前向传播"""
        return self.batch_call("forward", data)
    
    def batch_evaluate(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """批量评估（🆕 支持自动映射evaluate->evaluate_task）"""
        return self.batch_call("evaluate_task", task_data)
    
    def batch_train(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """🆕 新增：批量训练（支持自动映射train_epoch->train_task）"""
        return self.batch_call("train_task", task_data)


# ================ 通信层扩展 ================

class LearnerProxyMixin:
    """
    为通信后端添加learner代理支持的混入类
    
    扩展现有通信系统，添加learner方法调用能力
    """
    
    def call_learner_method(self, client_id: str, learner_name: str, 
                          method_name: str, *args, **kwargs) -> Any:
        """
        调用客户端learner方法的统一接口
        
        Args:
            client_id: 客户端ID
            learner_name: learner名称
            method_name: 方法名
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            Any: 方法执行结果
        """
        if self.is_local_mode():
            return self._call_local_learner_method(client_id, learner_name, method_name, *args, **kwargs)
        else:
            return self._call_remote_learner_method(client_id, learner_name, method_name, *args, **kwargs)
    
    def _call_local_learner_method(self, client_id: str, learner_name: str, 
                                 method_name: str, *args, **kwargs) -> Any:
        """本地模式：直接调用learner方法"""
        # 这里需要访问本地的learner实例
        # 具体实现取决于本地模拟器的设计
        raise NotImplementedError("本地learner方法调用需要在具体通信后端中实现")
    
    def _call_remote_learner_method(self, client_id: str, learner_name: str, 
                                  method_name: str, *args, **kwargs) -> Any:
        """远程模式：发送消息调用learner方法"""
        message = Message(
            id=str(uuid.uuid4()),
            sender=self.node_id,
            receiver=client_id,
            message_type="learner_method_call",
            payload={
                "learner_name": learner_name,
                "method_name": method_name,
                "args": args,
                "kwargs": kwargs
            },
            timestamp=time.time(),
            metadata={"timeout": 30.0}
        )
        
        # 发送消息并等待结果
        success = self.send_message(message)
        if not success:
            raise RuntimeError(f"发送learner方法调用消息失败: {client_id}.{learner_name}.{method_name}")
        
        # 等待结果消息
        result_message = self.receive_message(timeout=30.0)
        if not result_message or result_message.message_type != "learner_method_result":
            raise RuntimeError(f"未收到learner方法调用结果: {client_id}.{learner_name}.{method_name}")
        
        result_payload = result_message.payload
        if "error" in result_payload:
            raise RuntimeError(f"远程learner方法执行错误: {result_payload['error']}")
        
        return result_payload.get("result")
    
    def is_local_mode(self) -> bool:
        """判断是否为本地模式"""
        # 这个方法需要在具体的通信后端中实现
        return False