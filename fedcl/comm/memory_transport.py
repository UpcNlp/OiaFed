"""
内存传输器 - 单机内存通信

用于本地模拟模式，所有客户端在同一进程中，通过内存共享数据。
这是最简单的通信方式，主要用于算法开发和调试。
"""

import asyncio
import copy
from typing import Any, Dict, List, Optional
from loguru import logger


class MemoryTransport:
    """
    内存传输器 - 单机多线程/协程通信
    
    特点：
    1. 零延迟通信
    2. 无序列化开销
    3. 共享内存存储
    4. 适合算法开发和调试
    """
    
    def __init__(self):
        """初始化内存传输器"""
        self._memory_store: Dict[str, Dict[str, Any]] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self.logger = logger.bind(component="MemoryTransport")
        
        self.logger.info("内存传输器已初始化")
        
    async def send(self, source: str, target: str, data: Any) -> bool:
        """
        发送数据到目标
        
        Args:
            source: 发送方标识
            target: 接收方标识  
            data: 要发送的数据
            
        Returns:
            bool: 发送是否成功
        """
        try:
            # 确保目标存在锁
            if target not in self._locks:
                self._locks[target] = asyncio.Lock()
                
            # 确保目标存在存储空间
            if target not in self._memory_store:
                self._memory_store[target] = {}
                
            # 使用锁保证线程安全
            async with self._locks[target]:
                # 深拷贝数据避免引用问题
                self._memory_store[target][source] = copy.deepcopy(data)
                
            self.logger.debug(f"数据从 {source} 发送到 {target}")
            return True
            
        except Exception as e:
            self.logger.error(f"发送数据失败 {source}->{target}: {e}")
            return False
    
    async def receive(self, target: str, source: str, timeout: float = 10.0) -> Optional[Any]:
        """
        从源接收数据
        
        Args:
            target: 接收方标识
            source: 发送方标识
            timeout: 超时时间（秒）
            
        Returns:
            接收到的数据，超时或失败返回None
        """
        try:
            # 确保目标存在
            if target not in self._locks:
                self._locks[target] = asyncio.Lock()
            if target not in self._memory_store:
                self._memory_store[target] = {}
                
            # 等待数据到达
            start_time = asyncio.get_event_loop().time()
            
            while True:
                async with self._locks[target]:
                    if source in self._memory_store[target]:
                        # 获取数据并删除
                        data = self._memory_store[target].pop(source)
                        self.logger.debug(f"数据从 {source} 接收到 {target}")
                        return data
                        
                # 检查超时
                if asyncio.get_event_loop().time() - start_time > timeout:
                    self.logger.warning(f"接收超时 {source}->{target}")
                    return None
                    
                # 短暂等待后重试
                await asyncio.sleep(0.001)
                
        except Exception as e:
            self.logger.error(f"接收数据失败 {source}->{target}: {e}")
            return None
    
    async def broadcast(self, source: str, targets: List[str], data: Any) -> Dict[str, bool]:
        """
        广播数据到多个目标
        
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
                
        self.logger.info(f"广播完成 {source} -> {len(targets)}个目标")
        return results
    
    async def gather(self, target: str, sources: List[str], timeout: float = 30.0) -> Dict[str, Any]:
        """
        从多个源收集数据
        
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
    
    def get_status(self) -> Dict[str, Any]:
        """获取传输器状态"""
        return {
            "transport_type": "memory",
            "active_connections": len(self._memory_store),
            "pending_messages": sum(
                len(messages) for messages in self._memory_store.values()
            )
        }
    
    async def cleanup(self) -> None:
        """清理资源"""
        self._memory_store.clear()
        self._locks.clear()
        self.logger.info("内存传输器已清理")