"""
网络传输器 - 真实分布式通信

用于真联邦模式，客户端运行在不同的机器上，通过网络进行通信。
支持HTTP/HTTPS协议，适合生产环境部署。
"""

import asyncio
import json
import pickle
import aiohttp
from typing import Any, Dict, List, Optional, Union
from loguru import logger


class NetworkTransport:
    """
    网络传输器 - 真实分布式通信
    
    特点：
    1. 真实的网络通信
    2. 支持跨机器部署
    3. 支持TLS加密
    4. 适合生产环境
    """
    
    def __init__(self, 
                 host: str = "localhost", 
                 port: int = 8080,
                 use_ssl: bool = False,
                 timeout: float = 30.0):
        """
        初始化网络传输器
        
        Args:
            host: 服务器主机地址
            port: 服务器端口
            use_ssl: 是否使用SSL/TLS加密
            timeout: 默认超时时间
        """
        self.host = host
        self.port = port
        self.use_ssl = use_ssl
        self.timeout = timeout
        
        # 构建基础URL
        protocol = "https" if use_ssl else "http"
        self.base_url = f"{protocol}://{host}:{port}"
        
        # HTTP会话
        self._session: Optional[aiohttp.ClientSession] = None
        
        # 客户端端点映射
        self._client_endpoints: Dict[str, str] = {}
        
        self.logger = logger.bind(component="NetworkTransport")
        self.logger.info(f"网络传输器已初始化: {self.base_url}")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """获取或创建HTTP会话"""
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(
                ssl=False if not self.use_ssl else None,
                keepalive_timeout=60,
                limit=100,
                limit_per_host=30
            )
            
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={"Content-Type": "application/json"}
            )
            
        return self._session
    
    def register_client_endpoint(self, client_id: str, endpoint: str) -> None:
        """
        注册客户端端点
        
        Args:
            client_id: 客户端标识
            endpoint: 客户端端点URL
        """
        self._client_endpoints[client_id] = endpoint
        self.logger.info(f"客户端端点已注册: {client_id} -> {endpoint}")
    
    def _serialize_data(self, data: Any) -> bytes:
        """序列化数据"""
        try:
            # 尝试JSON序列化（更快，更通用）
            return json.dumps(data, ensure_ascii=False).encode('utf-8')
        except (TypeError, ValueError):
            # 回退到pickle（支持更多类型）
            return pickle.dumps(data)
    
    def _deserialize_data(self, data: bytes, use_pickle: bool = False) -> Any:
        """反序列化数据"""
        try:
            if use_pickle:
                return pickle.loads(data)
            else:
                return json.loads(data.decode('utf-8'))
        except:
            # 如果JSON失败，尝试pickle
            return pickle.loads(data)
    
    async def send(self, source: str, target: str, data: Any) -> bool:
        """
        发送数据到目标客户端
        
        Args:
            source: 发送方标识
            target: 接收方标识  
            data: 要发送的数据
            
        Returns:
            bool: 发送是否成功
        """
        try:
            # 获取目标客户端端点
            if target not in self._client_endpoints:
                self.logger.error(f"目标客户端端点未注册: {target}")
                return False
                
            target_endpoint = self._client_endpoints[target]
            
            # 构建请求数据
            request_data = {
                'source': source,
                'target': target, 
                'data': data,
                'timestamp': asyncio.get_event_loop().time()
            }
            
            # 发送HTTP请求
            session = await self._get_session()
            url = f"{target_endpoint}/receive_data"
            
            async with session.post(url, json=request_data) as response:
                if response.status == 200:
                    self.logger.debug(f"数据已发送 {source} -> {target}")
                    return True
                else:
                    error_text = await response.text()
                    self.logger.error(f"发送失败 {source}->{target}: {response.status} {error_text}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"发送数据失败 {source}->{target}: {e}")
            return False
    
    async def receive(self, target: str, source: str, timeout: float = 10.0) -> Optional[Any]:
        """
        从源客户端接收数据
        
        注意：在网络模式下，这个方法通常由HTTP服务器的端点处理
        这里提供一个轮询实现作为备选方案
        
        Args:
            target: 接收方标识
            source: 发送方标识
            timeout: 超时时间（秒）
            
        Returns:
            接收到的数据，超时或失败返回None
        """
        try:
            # 构建轮询URL
            if source not in self._client_endpoints:
                self.logger.error(f"源客户端端点未注册: {source}")
                return None
                
            source_endpoint = self._client_endpoints[source]
            url = f"{source_endpoint}/poll_data"
            
            # 轮询数据
            session = await self._get_session()
            start_time = asyncio.get_event_loop().time()
            
            while asyncio.get_event_loop().time() - start_time < timeout:
                params = {'target': target, 'source': source}
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        result = await response.json()
                        if result.get('data') is not None:
                            self.logger.debug(f"数据已接收 {source} -> {target}")
                            return result['data']
                    elif response.status != 404:  # 404表示没有数据
                        error_text = await response.text()
                        self.logger.warning(f"轮询错误 {response.status}: {error_text}")
                
                # 短暂等待后重试
                await asyncio.sleep(0.1)
            
            self.logger.warning(f"接收超时 {source} -> {target}")
            return None
            
        except Exception as e:
            self.logger.error(f"接收数据失败 {source}->{target}: {e}")
            return None
    
    async def broadcast(self, source: str, targets: List[str], data: Any) -> Dict[str, bool]:
        """
        广播数据到多个目标客户端
        
        Args:
            source: 发送方标识
            targets: 接收方标识列表
            data: 要广播的数据
            
        Returns:
            Dict[str, bool]: 每个目标的发送结果
        """
        results = {}
        
        # 并发发送到所有目标
        tasks = [
            self.send(source, target, data) 
            for target in targets
        ]
        
        send_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for target, result in zip(targets, send_results):
            if isinstance(result, Exception):
                results[target] = False
                self.logger.error(f"广播失败 {source}->{target}: {result}")
            else:
                results[target] = result
                
        self.logger.info(f"广播完成 {source} -> {len(targets)}个客户端")
        return results
    
    async def gather(self, target: str, sources: List[str], timeout: float = 30.0) -> Dict[str, Any]:
        """
        从多个源客户端收集数据
        
        Args:
            target: 接收方标识
            sources: 发送方标识列表
            timeout: 超时时间（秒）
            
        Returns:
            Dict[str, Any]: 从每个源收集到的数据
        """
        results = {}
        
        # 并发从所有源接收
        tasks = [
            self.receive(target, source, timeout)
            for source in sources
        ]
        
        receive_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for source, result in zip(sources, receive_results):
            if isinstance(result, Exception):
                self.logger.error(f"收集失败 {source}->{target}: {result}")
                results[source] = None
            else:
                results[source] = result
                
        # 过滤掉None结果
        results = {k: v for k, v in results.items() if v is not None}
        
        self.logger.info(f"收集完成 {len(results)}/{len(sources)} 成功")
        return results
    
    async def health_check(self, client_id: str) -> bool:
        """
        检查客户端健康状态
        
        Args:
            client_id: 客户端标识
            
        Returns:
            bool: 客户端是否健康
        """
        try:
            if client_id not in self._client_endpoints:
                return False
                
            endpoint = self._client_endpoints[client_id]
            session = await self._get_session()
            
            async with session.get(f"{endpoint}/health") as response:
                return response.status == 200
                
        except Exception as e:
            self.logger.warning(f"健康检查失败 {client_id}: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """获取传输器状态"""
        return {
            "transport_type": "network",
            "base_url": self.base_url,
            "use_ssl": self.use_ssl,
            "registered_clients": len(self._client_endpoints),
            "client_endpoints": self._client_endpoints.copy(),
            "session_active": self._session is not None and not self._session.closed
        }
    
    async def cleanup(self) -> None:
        """清理资源"""
        if self._session and not self._session.closed:
            await self._session.close()
            
        self._client_endpoints.clear()
        self.logger.info("网络传输器已清理")