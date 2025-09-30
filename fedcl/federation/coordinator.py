"""
MOE-FedCL 联邦学习协调器
moe_fedcl/federation/coordinator.py
"""

import asyncio
from typing import Any, Dict, List, Optional, Callable, TYPE_CHECKING
from datetime import datetime, timedelta
import uuid

from ..trainer.base_trainer import BaseTrainer, TrainingConfig
from ..types import FederationConfig, FederationStatus, RoundResult, ModelData
from ..exceptions import FederationError, TrainingError

if TYPE_CHECKING:
    from ..federation.server import FederationServer


class FederationResult:
    """联邦学习结果"""
    
    def __init__(self):
        self.success = False
        self.total_rounds = 0
        self.completed_rounds = 0
        self.final_accuracy = 0.0
        self.final_loss = float('inf')
        self.total_time = 0.0
        self.convergence_round = None
        self.best_model: Optional[ModelData] = None
        self.training_history: List[RoundResult] = []
        self.error_message: Optional[str] = None
        self.termination_reason = "unknown"


class FederationCoordinator:
    """联邦学习协调器 - 专注联邦学习流程协调，不管理底层组件"""
    
    def __init__(self,
                 federation_server: "FederationServer",
                 federation_config: Optional[FederationConfig] = None):
        """
        初始化联邦学习协调器
        
        Args:
            federation_server: 联邦服务端管理器
            federation_config: 联邦学习配置
        """
        self.federation_server = federation_server
        self.trainer = federation_server.get_trainer()
        if not self.trainer:
            raise FederationError("FederationServer must be initialized with trainer before creating coordinator")
        
        self.federation_config = federation_config or FederationConfig()
        
        # 协调器状态
        self.status = FederationStatus.INITIALIZING
        self.federation_id = str(uuid.uuid4())
        
        # 训练控制
        self._training_task: Optional[asyncio.Task] = None
        self._is_paused = False
        self._should_stop = False
        self._lock = asyncio.Lock()
        
        # 事件回调
        self.federation_callbacks: List[Callable] = []
        self.round_callbacks: List[Callable] = []
        
        # 统计和结果
        self.federation_result = FederationResult()
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        
        # 客户端管理（通过trainer查询，不直接管理）
        self.active_clients: List[str] = []
        self.failed_clients: List[str] = []
        
        print(f"FederationCoordinator initialized for federation_id: {self.federation_id}")
    
    # ==================== 核心协调方法 ====================
    
    async def start_federation(self) -> FederationResult:
        """启动联邦学习
        
        Returns:
            FederationResult: 联邦学习最终结果
            
        Raises:
            FederationError: 联邦学习启动或执行失败
        """
        async with self._lock:
            if self.status in [FederationStatus.TRAINING, FederationStatus.PAUSED]:
                raise FederationError("Federation is already running or paused")
            
            self.status = FederationStatus.INITIALIZING
            self._should_stop = False
            self._is_paused = False
        
        try:
            # 初始化阶段
            await self._trigger_federation_callbacks("FEDERATION_STARTING", {
                "federation_id": self.federation_id,
                "server_status": self.federation_server.get_server_status()
            })
            
            # 1. 确保服务端已启动
            print("Starting federation server...")
            server_started = await self.federation_server.start_server()
            if not server_started:
                raise FederationError("Failed to start federation server")
            
            # 2. 等待客户端就绪（通过trainer查询）
            print("Waiting for clients to be ready...")
            clients_ready = await self.wait_for_clients(timeout=120.0)
            if not clients_ready:
                raise FederationError("Not enough clients ready within timeout")
            
            # 3. 检查客户端就绪状态
            ready_clients = await self.check_client_readiness()
            active_count = sum(ready_clients.values())
            if active_count < self.federation_config.min_clients:
                raise FederationError(f"Insufficient ready clients: {active_count} < {self.federation_config.min_clients}")
            
            print(f"Starting federation with {active_count} active clients")
            
            # 4. 开始训练循环
            self.status = FederationStatus.TRAINING
            self.start_time = datetime.now()
            
            await self._trigger_federation_callbacks("FEDERATION_STARTED", {
                "start_time": self.start_time.isoformat(),
                "active_clients": active_count,
                "max_rounds": self.federation_config.max_rounds
            })
            
            # 启动训练任务
            self._training_task = asyncio.create_task(self._training_loop())
            
            # 等待训练完成
            await self._training_task
            
            # 完成联邦学习
            await self._finalize_federation()
            
            return self.federation_result
            
        except Exception as e:
            await self._handle_federation_error(e)
            raise FederationError(f"Federation failed: {str(e)}")
    
    async def stop_federation(self) -> bool:
        """停止联邦学习
        
        Returns:
            bool: 停止是否成功
        """
        async with self._lock:
            if self.status not in [FederationStatus.TRAINING, FederationStatus.PAUSED]:
                return False
            
            self._should_stop = True
            print("Federation stop requested")
        
        try:
            # 等待训练任务结束
            if self._training_task and not self._training_task.done():
                await asyncio.wait_for(self._training_task, timeout=30.0)
            
            await self._trigger_federation_callbacks("FEDERATION_STOPPED", {
                "reason": "user_requested",
                "completed_rounds": self.federation_result.completed_rounds
            })
            
            return True
            
        except Exception as e:
            print(f"Error stopping federation: {e}")
            return False
    
    async def pause_federation(self) -> bool:
        """暂停联邦学习
        
        Returns:
            bool: 暂停是否成功
        """
        async with self._lock:
            if self.status != FederationStatus.TRAINING:
                return False
            
            self._is_paused = True
            self.status = FederationStatus.PAUSED
            
            print("Federation paused")
        
        await self._trigger_federation_callbacks("FEDERATION_PAUSED", {
            "pause_time": datetime.now().isoformat(),
            "current_round": self.federation_result.completed_rounds
        })
        
        return True
    
    async def resume_federation(self) -> bool:
        """恢复联邦学习
        
        Returns:
            bool: 恢复是否成功
        """
        async with self._lock:
            if self.status != FederationStatus.PAUSED:
                return False
            
            self._is_paused = False
            self.status = FederationStatus.TRAINING
            
            print("Federation resumed")
        
        await self._trigger_federation_callbacks("FEDERATION_RESUMED", {
            "resume_time": datetime.now().isoformat(),
            "current_round": self.federation_result.completed_rounds
        })
        
        return True
    
    async def reset_federation(self) -> bool:
        """重置联邦学习状态
        
        Returns:
            bool: 重置是否成功
        """
        try:
            # 停止当前训练
            await self.stop_federation()
            
            async with self._lock:
                # 重置状态
                self.status = FederationStatus.INITIALIZING
                self._is_paused = False
                self._should_stop = False
                
                # 重置结果
                self.federation_result = FederationResult()
                self.start_time = None
                self.end_time = None
                
                # 重置客户端状态
                self.active_clients.clear()
                self.failed_clients.clear()
                
                # 清理训练器状态
                await self.trainer.cleanup()
            
            await self._trigger_federation_callbacks("FEDERATION_RESET", {
                "federation_id": self.federation_id,
                "reset_time": datetime.now().isoformat()
            })
            
            print("Federation reset successfully")
            return True
            
        except Exception as e:
            print(f"Error resetting federation: {e}")
            return False
    
    # ==================== 训练控制方法 ====================
    
    async def run_training_round(self, round_num: int) -> RoundResult:
        """执行单轮训练
        
        Args:
            round_num: 轮次编号
            
        Returns:
            RoundResult: 轮次训练结果
        """
        print(f"Starting training round {round_num}")
        round_start_time = datetime.now()
        
        try:
            # 选择参与客户端
            selected_clients = self.trainer.select_clients_for_round(round_num)
            print(f"Round {round_num}: Selected {len(selected_clients)} clients: {selected_clients}")
            
            # 检查客户端就绪状态
            client_readiness = await self.trainer.check_client_readiness(selected_clients)
            ready_clients = [cid for cid, ready in client_readiness.items() if ready]
            
            if len(ready_clients) < self.federation_config.min_clients:
                raise TrainingError(f"Insufficient ready clients for round {round_num}: {len(ready_clients)} < {self.federation_config.min_clients}")
            
            # 执行训练轮次
            round_result = await self.trainer.train_round(round_num, ready_clients)
            
            # 更新全局模型
            if "aggregated_model" in round_result:
                self.trainer.global_model = round_result["aggregated_model"]
            
            # 计算轮次时间
            round_time = (datetime.now() - round_start_time).total_seconds()
            round_result["round_time"] = round_time
            round_result["round_number"] = round_num
            
            # 触发轮次回调
            await self._trigger_round_callbacks(round_num, round_result)
            
            print(f"Round {round_num} completed in {round_time:.2f}s")
            return round_result
            
        except Exception as e:
            error_msg = f"Round {round_num} failed: {str(e)}"
            print(error_msg)
            
            round_result = {
                "round_number": round_num,
                "success": False,
                "error": error_msg,
                "participants": [],
                "successful_clients": [],
                "failed_clients": selected_clients if 'selected_clients' in locals() else [],
                "round_time": (datetime.now() - round_start_time).total_seconds()
            }
            
            await self._trigger_round_callbacks(round_num, round_result)
            raise TrainingError(error_msg)
    
    async def get_global_model(self) -> ModelData:
        """获取当前全局模型
        
        Returns:
            ModelData: 全局模型数据
        """
        return self.trainer.global_model
    
    async def set_global_model(self, model_data: ModelData) -> bool:
        """设置全局模型
        
        Args:
            model_data: 模型数据
            
        Returns:
            bool: 设置是否成功
        """
        try:
            self.trainer.global_model = model_data
            
            # 可选：将新模型推送到所有客户端
            # await self._broadcast_model_to_clients(model_data)
            
            return True
        except Exception as e:
            print(f"Failed to set global model: {e}")
            return False
    
    def get_federation_status(self) -> Dict[str, Any]:
        """获取联邦学习状态
        
        Returns:
            Dict[str, Any]: 联邦学习状态信息
        """
        return {
            "federation_id": self.federation_id,
            "status": self.status.value,
            "is_paused": self._is_paused,
            "current_round": self.federation_result.completed_rounds,
            "max_rounds": self.federation_config.max_rounds,
            "progress": self.federation_result.completed_rounds / self.federation_config.max_rounds if self.federation_config.max_rounds > 0 else 0,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "active_clients": len(self.active_clients),
            "failed_clients": len(self.failed_clients),
            "server_status": self.federation_server.get_server_status(),
            "best_accuracy": self.federation_result.final_accuracy,
            "trainer_status": self.trainer.get_training_status()
        }
    
    # ==================== 客户端状态查询方法 ====================
    
    async def wait_for_clients(self, timeout: float = 120.0) -> bool:
        """等待客户端就绪 - 通过trainer查询状态
        
        Args:
            timeout: 等待超时时间
            
        Returns:
            bool: 是否有足够的客户端就绪
        """
        import time
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # 通过trainer查询可用客户端
            available_clients = self.trainer.get_available_clients()
            
            if len(available_clients) >= self.federation_config.min_clients:
                self.active_clients = available_clients
                print(f"Sufficient clients ready: {available_clients}")
                return True
            
            print(f"Waiting for clients... Current: {len(available_clients)}, Required: {self.federation_config.min_clients}")
            await asyncio.sleep(2.0)  # 轮询间隔
        
        print(f"Timeout waiting for clients. Available: {len(self.trainer.get_available_clients())}")
        return False
    
    async def check_client_readiness(self) -> Dict[str, bool]:
        """检查客户端就绪状态 - 通过trainer查询
        
        Returns:
            Dict[str, bool]: 客户端就绪状态映射
        """
        readiness = {}
        available_clients = self.trainer.get_available_clients()
        
        for client_id in available_clients:
            readiness[client_id] = self.trainer.is_client_ready(client_id)
        
        return readiness
    
    # ==================== 内部训练循环 ====================
    
    async def _training_loop(self):
        """训练循环主逻辑"""
        try:
            self.federation_result.total_rounds = self.federation_config.max_rounds
            
            for round_num in range(1, self.federation_config.max_rounds + 1):
                # 检查停止条件
                if self._should_stop:
                    self.federation_result.termination_reason = "user_stopped"
                    break
                
                # 检查暂停条件
                while self._is_paused:
                    await asyncio.sleep(1.0)
                    if self._should_stop:
                        self.federation_result.termination_reason = "user_stopped"
                        return
                
                # 执行训练轮次
                try:
                    round_result = await self.run_training_round(round_num)
                    
                    # 更新结果
                    self.federation_result.completed_rounds = round_num
                    self.federation_result.training_history.append(round_result)
                    
                    # 更新最佳指标
                    round_metrics = round_result.get("round_metrics", {})
                    if "avg_accuracy" in round_metrics:
                        accuracy = round_metrics["avg_accuracy"]
                        if accuracy > self.federation_result.final_accuracy:
                            self.federation_result.final_accuracy = accuracy
                            self.federation_result.best_model = self.trainer.global_model.copy() if isinstance(self.trainer.global_model, dict) else self.trainer.global_model
                    
                    if "avg_loss" in round_metrics:
                        loss = round_metrics["avg_loss"]
                        if loss < self.federation_result.final_loss:
                            self.federation_result.final_loss = loss
                    
                    # 检查收敛条件
                    if self.trainer.should_stop_training(round_num, round_result):
                        self.federation_result.termination_reason = "converged"
                        self.federation_result.convergence_round = round_num
                        print(f"Training converged at round {round_num}")
                        break
                    
                except Exception as e:
                    print(f"Round {round_num} failed: {e}")
                    # 可以选择继续下一轮或终止训练
                    continue
            
            # 正常完成所有轮次
            if self.federation_result.termination_reason == "unknown":
                self.federation_result.termination_reason = "max_rounds_reached"
            
            self.federation_result.success = True
            
        except Exception as e:
            self.federation_result.error_message = str(e)
            print(f"Training loop error: {e}")
            raise
    
    async def _finalize_federation(self):
        """完成联邦学习"""
        self.end_time = datetime.now()
        self.status = FederationStatus.COMPLETED
        
        if self.start_time:
            self.federation_result.total_time = (self.end_time - self.start_time).total_seconds()
        
        # 最终评估
        try:
            final_evaluation = await self.trainer.evaluate_global_model()
            if final_evaluation.get("accuracy") is not None:
                self.federation_result.final_accuracy = final_evaluation["accuracy"]
            if final_evaluation.get("loss") is not None:
                self.federation_result.final_loss = final_evaluation["loss"]
        except Exception as e:
            print(f"Final evaluation failed: {e}")
        
        # 保存最终模型
        if self.federation_result.best_model is None:
            self.federation_result.best_model = self.trainer.global_model
        
        await self._trigger_federation_callbacks("FEDERATION_COMPLETED", {
            "end_time": self.end_time.isoformat(),
            "total_rounds": self.federation_result.completed_rounds,
            "final_accuracy": self.federation_result.final_accuracy,
            "final_loss": self.federation_result.final_loss,
            "total_time": self.federation_result.total_time,
            "termination_reason": self.federation_result.termination_reason
        })
        
        print(f"Federation completed: {self.federation_result.completed_rounds} rounds in {self.federation_result.total_time:.2f}s")
        print(f"Final accuracy: {self.federation_result.final_accuracy:.4f}, Final loss: {self.federation_result.final_loss:.4f}")
    
    async def _handle_federation_error(self, error: Exception):
        """处理联邦学习错误"""
        self.end_time = datetime.now()
        self.status = FederationStatus.ERROR
        
        self.federation_result.error_message = str(error)
        
        if self.start_time:
            self.federation_result.total_time = (self.end_time - self.start_time).total_seconds()
        
        await self._trigger_federation_callbacks("FEDERATION_ERROR", {
            "error": str(error),
            "error_time": self.end_time.isoformat(),
            "completed_rounds": self.federation_result.completed_rounds
        })
        
        print(f"Federation error: {error}")
    
    # ==================== 事件管理方法 ====================
    
    def register_round_callback(self, callback: Callable) -> str:
        """注册轮次回调
        
        Args:
            callback: 回调函数，签名为 callback(round_num: int, round_result: RoundResult)
            
        Returns:
            str: 回调ID
        """
        callback_id = f"round_{len(self.round_callbacks)}"
        self.round_callbacks.append((callback_id, callback))
        return callback_id
    
    def register_federation_callback(self, callback: Callable) -> str:
        """注册联邦学习回调
        
        Args:
            callback: 回调函数，签名为 callback(event: str, data: Any)
            
        Returns:
            str: 回调ID
        """
        callback_id = f"federation_{len(self.federation_callbacks)}"
        self.federation_callbacks.append((callback_id, callback))
        return callback_id
    
    def unregister_callback(self, callback_id: str) -> bool:
        """取消注册回调
        
        Args:
            callback_id: 回调ID
            
        Returns:
            bool: 是否成功取消
        """
        # 检查轮次回调
        for i, (cid, callback) in enumerate(self.round_callbacks):
            if cid == callback_id:
                del self.round_callbacks[i]
                return True
        
        # 检查联邦回调
        for i, (cid, callback) in enumerate(self.federation_callbacks):
            if cid == callback_id:
                del self.federation_callbacks[i]
                return True
        
        return False
    
    async def _trigger_round_callbacks(self, round_num: int, round_result: RoundResult):
        """触发轮次回调"""
        for callback_id, callback in self.round_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(round_num, round_result)
                else:
                    callback(round_num, round_result)
            except Exception as e:
                print(f"Round callback {callback_id} error: {e}")
    
    async def _trigger_federation_callbacks(self, event: str, data: Any):
        """触发联邦学习回调"""
        for callback_id, callback in self.federation_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event, data)
                else:
                    callback(event, data)
            except Exception as e:
                print(f"Federation callback {callback_id} error: {e}")