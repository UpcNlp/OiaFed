"""
本地执行器 - 单机串行执行

用于本地开发和调试，所有learner在当前进程中串行或并发执行。
这是最简单的执行模式，启动快，便于调试。
"""

import asyncio
from typing import Any, Dict, List, Optional
from loguru import logger

from ...comm.memory_transport import MemoryTransport
from ..base_learner import AbstractLearner, StandardLearner
from ...registry import registry


class LocalExecutor:
    """
    本地执行器 - 最简单的实现
    
    特点：
    1. 所有learner在同一进程
    2. 使用内存传输器
    3. 启动快速，便于调试
    4. 支持异步并发执行
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化本地执行器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.logger = logger.bind(component="LocalExecutor")
        
        # 初始化内存传输器
        self._transport = MemoryTransport()
        
        # 客户端learner实例
        self._learners: Dict[str, AbstractLearner] = {}
        
        # 执行状态
        self._is_running = False
        
        self.logger.info("本地执行器已初始化")
    
    def _create_learner(self, client_id: str) -> AbstractLearner:
        """
        从注册表或内置实现创建learner实例
        
        优先级：
        1. 注册表中的用户自定义learner
        2. 内置StandardLearner
        
        Args:
            client_id: 客户端ID
            
        Returns:
            AbstractLearner: learner实例
        """
        from ...registry import registry
        
        learner_name = self.config.get("learner_name")
        learner_type = self.config.get("learner_type", "standard")
        
        # 优先从注册表获取用户自定义learner
        if learner_name and learner_name in registry.learners:
            learner_cls = registry.get_learner(learner_name)
            learner = learner_cls(client_id, self.config)
            self.logger.debug(f"为客户端 {client_id} 创建用户自定义learner: {learner_name}")
            return learner
        
        # 回退到内置learner
        if learner_type == "standard":
            learner = StandardLearner(client_id, self.config)
            self.logger.debug(f"为客户端 {client_id} 创建内置learner: {learner_type}")
            return learner
        else:
            raise ValueError(f"Unknown learner type: {learner_type}")
    
    async def execute_client_training(self, client_id: str, **kwargs) -> Dict[str, Any]:
        """
        执行客户端训练 - 本地版本
        
        Args:
            client_id: 客户端ID
            **kwargs: 训练参数
            
        Returns:
            Dict[str, Any]: 训练结果
        """
        try:
            # 获取或创建learner
            if client_id not in self._learners:
                self._learners[client_id] = self._create_learner(client_id)
            
            learner = self._learners[client_id]
            
            # 执行训练
            result = await learner.train_epoch(**kwargs)
            
            self.logger.debug(f"客户端 {client_id} 训练完成")
            return result
            
        except Exception as e:
            self.logger.error(f"客户端 {client_id} 训练失败: {e}")
            raise
    
    async def execute_client_evaluation(self, client_id: str, **kwargs) -> Dict[str, Any]:
        """
        执行客户端评估
        
        Args:
            client_id: 客户端ID
            **kwargs: 评估参数
            
        Returns:
            Dict[str, Any]: 评估结果
        """
        try:
            if client_id not in self._learners:
                self._learners[client_id] = self._create_learner(client_id)
            
            learner = self._learners[client_id]
            
            # 执行评估
            result = await learner.evaluate(**kwargs)
            
            self.logger.debug(f"客户端 {client_id} 评估完成")
            return result
            
        except Exception as e:
            self.logger.error(f"客户端 {client_id} 评估失败: {e}")
            raise
    
    async def get_client_weights(self, client_id: str) -> Dict[str, Any]:
        """
        获取客户端模型权重
        
        Args:
            client_id: 客户端ID
            
        Returns:
            Dict[str, Any]: 模型权重
        """
        try:
            if client_id not in self._learners:
                raise ValueError(f"客户端 {client_id} 不存在")
            
            learner = self._learners[client_id]
            weights = learner.get_model_weights()
            
            self.logger.debug(f"获取客户端 {client_id} 模型权重")
            return weights
            
        except Exception as e:
            self.logger.error(f"获取客户端 {client_id} 权重失败: {e}")
            raise
    
    async def set_client_weights(self, client_id: str, weights: Dict[str, Any]) -> None:
        """
        设置客户端模型权重
        
        Args:
            client_id: 客户端ID
            weights: 模型权重
        """
        try:
            if client_id not in self._learners:
                self._learners[client_id] = self._create_learner(client_id)
            
            learner = self._learners[client_id]
            learner.set_model_weights(weights)
            
            self.logger.debug(f"设置客户端 {client_id} 模型权重")
            
        except Exception as e:
            self.logger.error(f"设置客户端 {client_id} 权重失败: {e}")
            raise
    
    async def broadcast_weights(self, client_ids: List[str], weights: Dict[str, Any]) -> Dict[str, bool]:
        """
        广播模型权重到多个客户端
        
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
        
        self.logger.info(f"权重广播完成: {sum(results.values())}/{len(client_ids)} 成功")
        return results
    
    async def gather_weights(self, client_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        收集多个客户端的模型权重
        
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
        
        self.logger.info(f"权重收集完成: {len(results)}/{len(client_ids)} 成功")
        return results
    
    async def start(self) -> bool:
        """启动本地执行器"""
        if self._is_running:
            self.logger.warning("本地执行器已经在运行")
            return True
        
        try:
            self._is_running = True
            self.logger.info("本地执行器已启动")
            return True
            
        except Exception as e:
            self.logger.error(f"启动本地执行器失败: {e}")
            self._is_running = False
            return False
    
    async def stop(self) -> bool:
        """停止本地执行器"""
        if not self._is_running:
            self.logger.warning("本地执行器未运行")
            return True
        
        try:
            # 清理资源
            await self._transport.cleanup()
            self._learners.clear()
            
            self._is_running = False
            self.logger.info("本地执行器已停止")
            return True
            
        except Exception as e:
            self.logger.error(f"停止本地执行器失败: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """获取执行器状态"""
        return {
            "executor_type": "local",
            "is_running": self._is_running,
            "active_learners": len(self._learners),
            "learner_ids": list(self._learners.keys()),
            "transport_status": self._transport.get_status()
        }