"""
进程传输器 - 多进程通信

用于伪联邦模式，每个客户端运行在独立进程中，通过进程间通信传递数据。
主要用于本地测试真实的并发场景。
"""

import asyncio
import pickle
import multiprocessing as mp
from typing import Any, Dict, List, Optional, Union
from concurrent.futures import ProcessPoolExecutor
from loguru import logger


class ProcessTransport:
    """
    进程传输器 - 多进程间通信
    
    特点：
    1. 真实的进程隔离
    2. 数据需要序列化
    3. 可测试并发问题
    4. 适合本地多进程测试
    """
    
    def __init__(self, max_processes: int = 4):
        """
        初始化进程传输器
        
        Args:
            max_processes: 最大进程数量
        """
        self.max_processes = max_processes
        self._process_pool: Optional[ProcessPoolExecutor] = None
        self._queues: Dict[str, mp.Queue] = {}
        self._manager = mp.Manager()
        self.logger = logger.bind(component="ProcessTransport")
        
        # 创建共享字典用于进程间状态管理
        self._shared_state = self._manager.dict()
        
        self.logger.info(f"进程传输器已初始化，最大进程数: {max_processes}")
    
    def _get_queue(self, target: str) -> mp.Queue:
        """获取或创建目标进程的队列"""
        if target not in self._queues:
            self._queues[target] = mp.Queue()
        return self._queues[target]
    
    async def send(self, source: str, target: str, data: Any) -> bool:
        """
        发送数据到目标进程
        
        Args:
            source: 发送方标识
            target: 接收方标识  
            data: 要发送的数据
            
        Returns:
            bool: 发送是否成功
        """
        try:
            # 序列化数据
            serialized_data = pickle.dumps({
                'source': source,
                'target': target,
                'data': data,
                'timestamp': asyncio.get_event_loop().time()
            })
            
            # 获取目标队列
            target_queue = self._get_queue(target)
            
            # 在线程池中执行阻塞的队列操作
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, 
                target_queue.put, 
                serialized_data
            )
            
            self.logger.debug(f"数据已发送 {source} -> {target}")
            return True
            
        except Exception as e:
            self.logger.error(f"发送数据失败 {source}->{target}: {e}")
            return False
    
    async def receive(self, target: str, source: str, timeout: float = 10.0) -> Optional[Any]:
        """
        从源进程接收数据
        
        Args:
            target: 接收方标识
            source: 发送方标识
            timeout: 超时时间（秒）
            
        Returns:
            接收到的数据，超时或失败返回None
        """
        try:
            target_queue = self._get_queue(target)
            
            # 在线程池中执行阻塞的队列操作
            loop = asyncio.get_event_loop()
            
            async def _receive_with_timeout():
                while True:
                    try:
                        # 非阻塞获取
                        serialized_data = await loop.run_in_executor(
                            None, 
                            target_queue.get, 
                            True,  # block
                            0.1    # timeout
                        )
                        
                        # 反序列化数据
                        message = pickle.loads(serialized_data)
                        
                        # 检查是否来自期望的源
                        if message['source'] == source:
                            self.logger.debug(f"数据已接收 {source} -> {target}")
                            return message['data']
                        else:
                            # 如果不是期望的源，重新放回队列
                            await loop.run_in_executor(
                                None,
                                target_queue.put,
                                serialized_data
                            )
                            
                    except:
                        # 队列为空，继续等待
                        await asyncio.sleep(0.01)
            
            # 使用asyncio.wait_for添加超时
            return await asyncio.wait_for(_receive_with_timeout(), timeout=timeout)
            
        except asyncio.TimeoutError:
            self.logger.warning(f"接收超时 {source} -> {target}")
            return None
        except Exception as e:
            self.logger.error(f"接收数据失败 {source}->{target}: {e}")
            return None
    
    async def broadcast(self, source: str, targets: List[str], data: Any) -> Dict[str, bool]:
        """
        广播数据到多个目标进程
        
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
                
        self.logger.info(f"广播完成 {source} -> {len(targets)}个进程")
        return results
    
    async def gather(self, target: str, sources: List[str], timeout: float = 30.0) -> Dict[str, Any]:
        """
        从多个源进程收集数据
        
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
    
    def start_process_pool(self) -> None:
        """启动进程池"""
        if self._process_pool is None:
            self._process_pool = ProcessPoolExecutor(max_workers=self.max_processes)
            self.logger.info(f"进程池已启动，工作进程数: {self.max_processes}")
    
    def stop_process_pool(self) -> None:
        """停止进程池"""
        if self._process_pool is not None:
            self._process_pool.shutdown(wait=True)
            self._process_pool = None
            self.logger.info("进程池已停止")
    
    def get_status(self) -> Dict[str, Any]:
        """获取传输器状态"""
        queue_sizes = {}
        for name, queue in self._queues.items():
            try:
                queue_sizes[name] = queue.qsize()
            except:
                queue_sizes[name] = -1  # 无法获取大小
                
        return {
            "transport_type": "process",
            "max_processes": self.max_processes,
            "active_queues": len(self._queues),
            "queue_sizes": queue_sizes,
            "process_pool_active": self._process_pool is not None
        }
    
    async def cleanup(self) -> None:
        """清理资源"""
        self.stop_process_pool()
        
        # 清空所有队列
        for queue in self._queues.values():
            try:
                while not queue.empty():
                    queue.get_nowait()
            except:
                pass
                
        self._queues.clear()
        self.logger.info("进程传输器已清理")