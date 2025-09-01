"""
标准联邦训练器 - 生产环境实现

实现标准FedAvg算法，使用真实的底层组件：
- 从注册表获取聚合器
- 使用真实的执行引擎
- 调用真实的learner
- 异常直接抛出，不使用模拟结果
"""

import time
from typing import Dict, Any, List, Optional
from loguru import logger

from ...fl.abstract_trainer import AbstractFederationTrainer
from ...transparent.base_federation_engine import TrainingResult, EvaluationResult
from ...registry import registry
from ...api.decorators import trainer


@trainer("standard_federation", description="标准联邦训练器 - FedAvg实现")
class StandardFederationTrainer(AbstractFederationTrainer):
    """
    标准联邦训练器 - FedAvg算法实现
    
    核心特点：
    1. 🎯 实现标准FedAvg算法
    2. 🔧 使用真实的底层组件
    3. 🌐 支持多种数据分布（IID/Non-IID）
    4. 📦 与底层执行模式解耦
    5. ⚡ 生产环境就绪，异常直接抛出
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        
        # FedAvg特有配置
        self.num_clients = self.config.get("num_clients", 3)
        self.local_epochs = self.config.get("local_epochs", 1)
        self.learning_rate = self.config.get("learning_rate", 0.01)
        self.batch_size = self.config.get("batch_size", 32)
        
        # 客户端选择配置
        self.client_selection_ratio = self.config.get("client_selection_ratio", 1.0)
        self.min_clients = self.config.get("min_clients", 1)
        
        # 获取聚合器 - 根据用户配置，默认使用fedavg
        self.aggregator_name = self.config.get("aggregator", "fedavg")
        self._aggregator = self._create_aggregator()
        
        # 获取learner名称
        self.learner_name = self.config.get("learner", "default")
        
        logger.info("✅ 标准联邦训练器初始化完成")
        logger.info(f"   客户端数量: {self.num_clients}")
        logger.info(f"   本地训练轮数: {self.local_epochs}")
        logger.info(f"   学习率: {self.learning_rate}")
        logger.info(f"   聚合器: {self.aggregator_name}")
        logger.info(f"   学习器: {self.learner_name}")
    
    def _create_aggregator(self):
        """从注册表创建聚合器实例"""
        try:
            aggregator_cls = registry.get_aggregator(self.aggregator_name)
            aggregator_config = self.config.get("aggregator_config", {})
            return aggregator_cls(aggregator_config)
        except Exception as e:
            logger.error(f"❌ 创建聚合器失败: {e}")
            raise
    
    def train(self, num_rounds: int, **kwargs) -> TrainingResult:
        """
        执行标准FedAvg联邦训练
        
        Args:
            num_rounds: 训练轮次
            **kwargs: 训练参数
            
        Returns:
            TrainingResult: 训练结果
        """
        start_time = time.time()
        
        logger.info(f"🚀 开始FedAvg联邦训练 - {num_rounds} 轮次")
        
        try:
            # 初始化全局模型
            self._initialize_global_model()
            
            # 执行联邦训练轮次
            for round_num in range(1, num_rounds + 1):
                logger.info(f"🔄 执行FedAvg轮次 {round_num}")
                
                # 1. 客户端选择
                selected_clients = self._select_clients_for_round(round_num)
                logger.info(f"   选中 {len(selected_clients)} 个客户端: {selected_clients}")
                
                # 2. 获取当前全局模型权重
                global_weights = self.get_global_state("global_model_weights", {})
                
                # 3. 执行客户端训练
                client_results = self._execute_client_training(
                    selected_clients, round_num, global_weights, **kwargs
                )
                
                # 4. 执行聚合
                aggregated_weights = self._execute_aggregation(client_results)
                
                # 5. 更新全局模型
                self.update_global_state("global_model_weights", aggregated_weights)
                
                # 6. 记录轮次结果
                round_result = {
                    "num_participants": len(client_results),
                    "total_samples": sum(r.get("num_samples", 0) for r in client_results),
                    "avg_loss": sum(r.get("loss", 0) for r in client_results) / len(client_results),
                    "aggregation_method": self.aggregator_name
                }
                
                self.add_round_result(round_num, round_result)
                
                # 7. 更新训练状态
                self.update_global_state("current_round", round_num)
                self.update_global_state("best_loss", min(
                    self.get_global_state("best_loss", float('inf')),
                    round_result["avg_loss"]
                ))
                
                logger.info(f"   平均损失: {round_result['avg_loss']:.4f}")
            
            # 构建训练结果
            training_time = time.time() - start_time
            result = self.build_training_result(num_rounds, training_time, "fedavg")
            
            logger.info(f"✅ FedAvg训练完成，耗时: {training_time:.2f}秒")
            return result
            
        except Exception as e:
            logger.error(f"❌ FedAvg训练失败: {e}")
            raise
    
    def evaluate(self, test_data: Optional[Any] = None, **kwargs) -> EvaluationResult:
        """
        执行模型评估
        
        Args:
            test_data: 测试数据
            **kwargs: 评估参数
            
        Returns:
            EvaluationResult: 评估结果
        """
        logger.info("🔍 执行FedAvg模型评估")
        
        # 获取全局模型权重
        global_weights = self.get_global_state("global_model_weights", {})
        
        if not global_weights:
            raise ValueError("全局模型权重未初始化，无法进行评估")
        
        # 获取评估器
        evaluator_name = self.config.get("evaluator", "accuracy")
        try:
            evaluator_cls = registry.get_evaluator(evaluator_name)
            evaluator_config = self.config.get("evaluator_config", {})
            evaluator = evaluator_cls(evaluator_config)
        except Exception as e:
            logger.error(f"❌ 创建评估器失败: {e}")
            raise
        
        # 执行评估
        try:
            evaluation_start = time.time()
            metrics = evaluator.evaluate(global_weights, test_data, **kwargs)
            evaluation_time = time.time() - evaluation_start
            
            return EvaluationResult(
                metrics=metrics,
                task_metrics={},  # 可根据需要扩展
                metadata={
                    "evaluation_mode": "fedavg",
                    "evaluator": evaluator_name,
                    "global_weights_available": bool(global_weights)
                },
                evaluation_time=evaluation_time,
                primary_metric="accuracy"
            )
            
        except Exception as e:
            logger.error(f"❌ 模型评估失败: {e}")
            raise
    
    def _initialize_global_model(self) -> None:
        """初始化全局模型"""
        logger.info("🔧 初始化全局模型")
        
        # 获取learner类来初始化模型
        try:
            learner_cls = registry.get_learner(self.learner_name)
            # 创建临时learner实例来获取初始模型权重
            temp_learner = learner_cls("temp_client", self.config)
            initial_weights = temp_learner.get_model_weights()
            
            self.update_global_state("global_model_weights", initial_weights)
            self.update_global_state("best_loss", float('inf'))
            
            logger.info("✅ 全局模型初始化完成")
            
        except Exception as e:
            logger.error(f"❌ 全局模型初始化失败: {e}")
            raise
    
    def _select_clients_for_round(self, round_num: int) -> List[str]:
        """
        为当前轮次选择客户端
        
        Args:
            round_num: 轮次编号
            
        Returns:
            List[str]: 选中的客户端ID列表
        """
        # 计算选择数量
        num_to_select = max(
            self.min_clients,
            int(self.num_clients * self.client_selection_ratio)
        )
        num_to_select = min(num_to_select, self.num_clients)
        
        # 简单的轮转选择策略
        start_idx = (round_num - 1) % self.num_clients
        selected = []
        
        for i in range(num_to_select):
            client_idx = (start_idx + i) % self.num_clients
            selected.append(f"client_{client_idx}")
        
        return selected
    
    def _execute_client_training(
        self, 
        client_ids: List[str], 
        round_num: int, 
        global_weights: Dict[str, Any], 
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        执行客户端训练
        
        Args:
            client_ids: 客户端ID列表
            round_num: 轮次编号
            global_weights: 全局模型权重
            **kwargs: 额外参数
            
        Returns:
            List[Dict[str, Any]]: 客户端训练结果列表
        """
        logger.debug(f"🏃 执行 {len(client_ids)} 个客户端训练")
        
        client_results = []
        
        for client_id in client_ids:
            try:
                # 创建learner实例
                learner_cls = registry.get_learner(self.learner_name)
                learner = learner_cls(client_id, self.config)
                
                # 设置全局模型权重
                learner.set_model_weights(global_weights)
                
                # 构建训练任务数据
                task_data = {
                    "client_id": client_id,
                    "round_num": round_num,
                    "local_epochs": self.local_epochs,
                    "learning_rate": self.learning_rate,
                    "batch_size": self.batch_size,
                    **kwargs
                }
                
                # 执行训练
                training_result = learner.train_task(task_data)
                
                # 获取更新后的模型权重
                updated_weights = learner.get_model_weights()
                
                # 构建客户端结果
                client_result = {
                    "client_id": client_id,
                    "model_weights": updated_weights,
                    "num_samples": training_result.get("num_samples", 0),
                    "loss": training_result.get("loss", 0.0),
                    "accuracy": training_result.get("accuracy", 0.0),
                    "local_epochs": self.local_epochs,
                    "round_num": round_num
                }
                
                client_results.append(client_result)
                logger.debug(f"✅ 客户端 {client_id} 训练完成")
                
            except Exception as e:
                logger.error(f"❌ 客户端 {client_id} 训练失败: {e}")
                raise
        
        if not client_results:
            raise ValueError(f"轮次 {round_num}: 所有客户端训练都失败了")
        
        return client_results
    
    def _execute_aggregation(self, client_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        执行聚合
        
        Args:
            client_results: 客户端训练结果列表
            
        Returns:
            Dict[str, Any]: 聚合后的模型权重
        """
        logger.debug(f"🔄 执行聚合 - {len(client_results)} 个客户端")
        
        if not client_results:
            raise ValueError("没有客户端结果可供聚合")
        
        try:
            # 使用注册的聚合器执行聚合
            aggregated_result = self._aggregator.aggregate(client_results)
            
            # 提取聚合后的权重
            if "aggregated_weights" in aggregated_result:
                aggregated_weights = aggregated_result["aggregated_weights"]
            else:
                # 兼容不同的聚合器返回格式
                aggregated_weights = aggregated_result
            
            logger.debug("✅ 聚合完成")
            return aggregated_weights
            
        except Exception as e:
            logger.error(f"❌ 聚合失败: {e}")
            raise