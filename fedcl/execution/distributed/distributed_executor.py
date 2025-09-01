"""
分布式执行器 - 真实网络通信

用于生产环境，客户端运行在不同的机器上，通过网络进行通信。
支持真正的分布式联邦学习部署。
"""

import asyncio
from typing import Any, Dict, List, Optional
from loguru import logger

from ...comm.network_transport import NetworkTransport


class DistributedExecutor:
    """
    真联邦分布式执行器
    
    特点：
    1. 客户端跨机器部署
    2. 真实的网络通信
    3. 支持TLS加密
    4. 生产环境就绪
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化分布式执行器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.logger = logger.bind(component="DistributedExecutor")
        
        # 网络配置
        host = config.get("server_host", "localhost")
        port = config.get("server_port", 8080)
        use_ssl = config.get("use_ssl", False)
        timeout = config.get("network_timeout", 30.0)
        
        # 初始化网络传输器
        self._transport = NetworkTransport(host, port, use_ssl, timeout)
        
        # 客户端端点映射
        self._client_endpoints: Dict[str, str] = config.get("client_endpoints", {})
        
        # 注册客户端端点
        for client_id, endpoint in self._client_endpoints.items():
            self._transport.register_client_endpoint(client_id, endpoint)
        
        # 执行状态
        self._is_running = False
        
        self.logger.info(f"分布式执行器已初始化，客户端数: {len(self._client_endpoints)}")
    
    async def execute_client_training(self, client_id: str, **kwargs) -> Dict[str, Any]:
        """
        执行客户端训练 - 分布式版本
        
        Args:
            client_id: 客户端ID
            **kwargs: 训练参数
            
        Returns:
            Dict[str, Any]: 训练结果
        """
        try:
            if client_id not in self._client_endpoints:
                raise ValueError(f"客户端 {client_id} 端点未注册")
            
            # 通过网络调用客户端训练
            result = await self._call_remote_method(client_id, "train_epoch", **kwargs)
            
            self.logger.debug(f"客户端 {client_id} 训练完成（分布式模式）")
            return result
            
        except Exception as e:
            self.logger.error(f"客户端 {client_id} 训练失败: {e}")
            raise
    
    async def execute_client_evaluation(self, client_id: str, **kwargs) -> Dict[str, Any]:
        """
        执行客户端评估 - 分布式版本
        
        Args:
            client_id: 客户端ID
            **kwargs: 评估参数
            
        Returns:
            Dict[str, Any]: 评估结果
        """
        try:
            if client_id not in self._client_endpoints:
                raise ValueError(f"客户端 {client_id} 端点未注册")
            
            # 通过网络调用客户端评估
            result = await self._call_remote_method(client_id, "evaluate", **kwargs)
            
            self.logger.debug(f"客户端 {client_id} 评估完成（分布式模式）")
            return result
            
        except Exception as e:
            self.logger.error(f"客户端 {client_id} 评估失败: {e}")
            raise
    
    async def get_client_weights(self, client_id: str) -> Dict[str, Any]:
        """
        获取客户端模型权重 - 分布式版本
        
        Args:
            client_id: 客户端ID
            
        Returns:
            Dict[str, Any]: 模型权重
        """
        try:
            if client_id not in self._client_endpoints:
                raise ValueError(f"客户端 {client_id} 端点未注册")
            
            # 通过网络获取权重
            weights = await self._call_remote_method(client_id, "get_model_weights")
            
            self.logger.debug(f"获取客户端 {client_id} 权重完成（分布式模式）")
            return weights
            
        except Exception as e:
            self.logger.error(f"获取客户端 {client_id} 权重失败: {e}")
            raise
    
    async def set_client_weights(self, client_id: str, weights: Dict[str, Any]) -> None:
        """
        设置客户端模型权重 - 分布式版本
        
        Args:
            client_id: 客户端ID
            weights: 模型权重
        """
        try:
            if client_id not in self._client_endpoints:
                raise ValueError(f"客户端 {client_id} 端点未注册")
            
            # 通过网络设置权重
            await self._call_remote_method(client_id, "set_model_weights", weights=weights)
            
            self.logger.debug(f"设置客户端 {client_id} 权重完成（分布式模式）")
            
        except Exception as e:
            self.logger.error(f"设置客户端 {client_id} 权重失败: {e}")
            raise
    
    async def _call_remote_method(self, client_id: str, method: str, **kwargs) -> Any:
        """
        调用远程客户端方法
        
        Args:
            client_id: 客户端ID
            method: 方法名
            **kwargs: 方法参数
            
        Returns:
            Any: 方法执行结果
        """
        try:
            endpoint = self._client_endpoints[client_id]
            
            # 构建请求数据（包含配置信息，远程客户端可从中获取learner_name等参数）
            request_data = {
                'method': method,
                'client_id': client_id,
                'config': self.config,  # 传递配置信息给远程客户端
                'kwargs': kwargs
            }
            
            # 发送HTTP请求
            session = await self._transport._get_session()
            url = f"{endpoint}/execute_method"
            
            async with session.post(url, json=request_data) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    if result.get('success'):
                        return result.get('data')
                    else:
                        error_msg = result.get('error', 'Unknown error')
                        raise RuntimeError(f"Remote method failed: {error_msg}")
                else:
                    error_text = await response.text()
                    raise RuntimeError(f"HTTP error {response.status}: {error_text}")
                    
        except Exception as e:
            self.logger.error(f"远程调用失败 {client_id}.{method}: {e}")
            raise
    
    async def broadcast_weights(self, client_ids: List[str], weights: Dict[str, Any]) -> Dict[str, bool]:
        """
        广播模型权重到多个客户端 - 分布式版本
        
        Args:
            client_ids: 客户端ID列表
            weights: 模型权重
            
        Returns:
            Dict[str, bool]: 每个客户端的设置结果
        """
        results = {}
        
        # 并发设置所有客户端权重
        tasks = [
            self.set_client_weights(client_id, weights)
            for client_id in client_ids
        ]
        
        set_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for client_id, result in zip(client_ids, set_results):
            if isinstance(result, Exception):
                results[client_id] = False
                self.logger.error(f"广播权重失败 {client_id}: {result}")
            else:
                results[client_id] = True
        
        self.logger.info(f"权重广播完成（分布式模式）: {sum(results.values())}/{len(client_ids)} 成功")
        return results
    
    async def gather_weights(self, client_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        收集多个客户端的模型权重 - 分布式版本
        
        Args:
            client_ids: 客户端ID列表
            
        Returns:
            Dict[str, Dict[str, Any]]: 每个客户端的权重
        """
        results = {}
        
        # 并发获取所有客户端权重
        tasks = [
            self.get_client_weights(client_id)
            for client_id in client_ids
        ]
        
        weight_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for client_id, result in zip(client_ids, weight_results):
            if isinstance(result, Exception):
                self.logger.error(f"收集权重失败 {client_id}: {result}")
                results[client_id] = None
            else:
                results[client_id] = result
        
        # 过滤掉失败的结果
        results = {k: v for k, v in results.items() if v is not None}
        
        self.logger.info(f"权重收集完成（分布式模式）: {len(results)}/{len(client_ids)} 成功")
        return results
    
    async def health_check_all_clients(self) -> Dict[str, bool]:
        """
        检查所有客户端健康状态
        
        Returns:
            Dict[str, bool]: 每个客户端的健康状态
        """
        results = {}
        
        # 并发检查所有客户端
        tasks = [
            self._transport.health_check(client_id)
            for client_id in self._client_endpoints.keys()
        ]
        
        health_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for client_id, result in zip(self._client_endpoints.keys(), health_results):
            if isinstance(result, Exception):
                results[client_id] = False
                self.logger.error(f"健康检查失败 {client_id}: {result}")
            else:
                results[client_id] = result
        
        healthy_count = sum(results.values())
        total_count = len(results)
        
        self.logger.info(f"健康检查完成: {healthy_count}/{total_count} 客户端健康")
        return results
    
    async def start(self) -> bool:
        """启动分布式执行器"""
        if self._is_running:
            self.logger.warning("分布式执行器已经在运行")
            return True
        
        try:
            # 检查客户端连接
            health_status = await self.health_check_all_clients()
            healthy_clients = sum(health_status.values())
            
            if healthy_clients == 0:
                self.logger.error("没有健康的客户端，无法启动")
                return False
            elif healthy_clients < len(self._client_endpoints):
                self.logger.warning(f"只有 {healthy_clients}/{len(self._client_endpoints)} 客户端健康")
            
            self._is_running = True
            self.logger.info(f"分布式执行器已启动，健康客户端: {healthy_clients}")
            return True
            
        except Exception as e:
            self.logger.error(f"启动分布式执行器失败: {e}")
            self._is_running = False
            return False
    
    async def stop(self) -> bool:
        """停止分布式执行器"""
        if not self._is_running:
            self.logger.warning("分布式执行器未运行")
            return True
        
        try:
            # 清理网络传输器
            await self._transport.cleanup()
            
            self._is_running = False
            self.logger.info("分布式执行器已停止")
            return True
            
        except Exception as e:
            self.logger.error(f"停止分布式执行器失败: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """获取执行器状态"""
        return {
            "executor_type": "distributed",
            "is_running": self._is_running,
            "registered_clients": len(self._client_endpoints),
            "client_endpoints": self._client_endpoints.copy(),
            "transport_status": self._transport.get_status()
        }