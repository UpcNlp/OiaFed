"""
伪联邦执行器 - 多进程执行

用于本地测试真实的并发场景，每个客户端运行在独立的进程中。
可以验证并发问题和进程间通信，适合本地测试。
"""

import asyncio
import multiprocessing as mp
from typing import Any, Dict, List, Optional
from concurrent.futures import ProcessPoolExecutor
from loguru import logger

from ...comm.process_transport import ProcessTransport


def _run_learner_in_process(client_id: str, config: Dict[str, Any], method: str, **kwargs) -> Dict[str, Any]:
    """
    在独立进程中运行learner方法
    
    Args:
        client_id: 客户端ID
        config: 配置
        method: 要调用的方法名
        **kwargs: 方法参数
        
    Returns:
        Dict[str, Any]: 方法执行结果
    """
    try:
        # 在进程中导入并创建learner
        from ...execution.base_learner import StandardLearner
        from ...registry import registry
        
        # 尝试从注册表获取用户自定义learner
        learner_name = config.get("learner_name")
        learner_type = config.get("learner_type", "standard")
        
        if learner_name and learner_name in registry.learners:
            # 使用用户自定义learner
            learner_cls = registry.get_learner(learner_name)
            learner = learner_cls(client_id, config)
        elif learner_type == "standard":
            # 使用内置StandardLearner
            learner = StandardLearner(client_id, config)
        else:
            raise ValueError(f"Unknown learner type: {learner_type}")
        
        # 执行指定方法
        if method == "train_epoch":
            result = asyncio.run(learner.train_epoch(**kwargs))
        elif method == "evaluate":
            result = asyncio.run(learner.evaluate(**kwargs))
        elif method == "get_model_weights":
            result = learner.get_model_weights()
        elif method == "set_model_weights":
            learner.set_model_weights(kwargs["weights"])
            result = {"success": True}
        else:
            raise ValueError(f"Unknown method: {method}")
            
        return {
            "success": True,
            "result": result,
            "client_id": client_id
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "client_id": client_id
        }


class PseudoExecutor:
    """
    伪联邦多进程执行器
    
    特点：
    1. 每个客户端独立进程
    2. 真实的进程隔离
    3. 可测试并发问题
    4. 使用进程传输器通信
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化伪联邦执行器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.logger = logger.bind(component="PseudoExecutor")
        
        # 进程相关配置
        self.max_processes = config.get("max_processes", mp.cpu_count())
        
        # 初始化进程传输器
        self._transport = ProcessTransport(self.max_processes)
        
        # 进程池
        self._process_pool: Optional[ProcessPoolExecutor] = None
        
        # 客户端进程状态
        self._client_processes: Dict[str, int] = {}  # client_id -> process_id
        
        # 执行状态
        self._is_running = False
        
        self.logger.info(f"伪联邦执行器已初始化，最大进程数: {self.max_processes}")
    
    async def execute_client_training(self, client_id: str, **kwargs) -> Dict[str, Any]:
        """
        执行客户端训练 - 多进程版本
        
        Args:
            client_id: 客户端ID
            **kwargs: 训练参数
            
        Returns:
            Dict[str, Any]: 训练结果
        """
        try:
            # 在进程池中执行训练
            loop = asyncio.get_event_loop()
            
            result = await loop.run_in_executor(
                self._process_pool,
                _run_learner_in_process,
                client_id,
                self.config,
                "train_epoch",
                **kwargs
            )
            
            if result["success"]:
                self.logger.debug(f"客户端 {client_id} 训练完成（进程模式）")
                return result["result"]
            else:
                error_msg = result["error"]
                self.logger.error(f"客户端 {client_id} 训练失败: {error_msg}")
                raise RuntimeError(error_msg)
                
        except Exception as e:
            self.logger.error(f"客户端 {client_id} 训练异常: {e}")
            raise
    
    async def execute_client_evaluation(self, client_id: str, **kwargs) -> Dict[str, Any]:
        """
        执行客户端评估 - 多进程版本
        
        Args:
            client_id: 客户端ID
            **kwargs: 评估参数
            
        Returns:
            Dict[str, Any]: 评估结果
        """
        try:
            # 在进程池中执行评估
            loop = asyncio.get_event_loop()
            
            result = await loop.run_in_executor(
                self._process_pool,
                _run_learner_in_process,
                client_id,
                self.config,
                "evaluate",
                **kwargs
            )
            
            if result["success"]:
                self.logger.debug(f"客户端 {client_id} 评估完成（进程模式）")
                return result["result"]
            else:
                error_msg = result["error"]
                self.logger.error(f"客户端 {client_id} 评估失败: {error_msg}")
                raise RuntimeError(error_msg)
                
        except Exception as e:
            self.logger.error(f"客户端 {client_id} 评估异常: {e}")
            raise
    
    async def get_client_weights(self, client_id: str) -> Dict[str, Any]:
        """
        获取客户端模型权重 - 多进程版本
        
        Args:
            client_id: 客户端ID
            
        Returns:
            Dict[str, Any]: 模型权重
        """
        try:
            # 在进程池中获取权重
            loop = asyncio.get_event_loop()
            
            result = await loop.run_in_executor(
                self._process_pool,
                _run_learner_in_process,
                client_id,
                self.config,
                "get_model_weights"
            )
            
            if result["success"]:
                self.logger.debug(f"获取客户端 {client_id} 权重完成（进程模式）")
                return result["result"]
            else:
                error_msg = result["error"]
                self.logger.error(f"获取客户端 {client_id} 权重失败: {error_msg}")
                raise RuntimeError(error_msg)
                
        except Exception as e:
            self.logger.error(f"获取客户端 {client_id} 权重异常: {e}")
            raise
    
    async def set_client_weights(self, client_id: str, weights: Dict[str, Any]) -> None:
        """
        设置客户端模型权重 - 多进程版本
        
        Args:
            client_id: 客户端ID
            weights: 模型权重
        """
        try:
            # 在进程池中设置权重
            loop = asyncio.get_event_loop()
            
            result = await loop.run_in_executor(
                self._process_pool,
                _run_learner_in_process,
                client_id,
                self.config,
                "set_model_weights",
                weights=weights
            )
            
            if result["success"]:
                self.logger.debug(f"设置客户端 {client_id} 权重完成（进程模式）")
            else:
                error_msg = result["error"]
                self.logger.error(f"设置客户端 {client_id} 权重失败: {error_msg}")
                raise RuntimeError(error_msg)
                
        except Exception as e:
            self.logger.error(f"设置客户端 {client_id} 权重异常: {e}")
            raise
    
    async def broadcast_weights(self, client_ids: List[str], weights: Dict[str, Any]) -> Dict[str, bool]:
        """
        广播模型权重到多个客户端 - 多进程版本
        
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
        
        self.logger.info(f"权重广播完成（进程模式）: {sum(results.values())}/{len(client_ids)} 成功")
        return results
    
    async def gather_weights(self, client_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        收集多个客户端的模型权重 - 多进程版本
        
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
        
        self.logger.info(f"权重收集完成（进程模式）: {len(results)}/{len(client_ids)} 成功")
        return results
    
    async def start(self) -> bool:
        """启动伪联邦执行器"""
        if self._is_running:
            self.logger.warning("伪联邦执行器已经在运行")
            return True
        
        try:
            # 启动进程池
            self._process_pool = ProcessPoolExecutor(max_workers=self.max_processes)
            
            # 启动进程传输器
            self._transport.start_process_pool()
            
            self._is_running = True
            self.logger.info(f"伪联邦执行器已启动，进程数: {self.max_processes}")
            return True
            
        except Exception as e:
            self.logger.error(f"启动伪联邦执行器失败: {e}")
            self._is_running = False
            return False
    
    async def stop(self) -> bool:
        """停止伪联邦执行器"""
        if not self._is_running:
            self.logger.warning("伪联邦执行器未运行")
            return True
        
        try:
            # 停止进程池
            if self._process_pool:
                self._process_pool.shutdown(wait=True)
                self._process_pool = None
            
            # 清理传输器
            await self._transport.cleanup()
            
            # 清理状态
            self._client_processes.clear()
            
            self._is_running = False
            self.logger.info("伪联邦执行器已停止")
            return True
            
        except Exception as e:
            self.logger.error(f"停止伪联邦执行器失败: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """获取执行器状态"""
        return {
            "executor_type": "pseudo",
            "is_running": self._is_running,
            "max_processes": self.max_processes,
            "active_clients": len(self._client_processes),
            "client_processes": self._client_processes.copy(),
            "transport_status": self._transport.get_status()
        }