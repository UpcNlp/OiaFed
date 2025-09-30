
import asyncio
from typing import Dict, Any, Optional, List

from ...comm.rpc_layer import LearnerProxy
from ...api.decorators import trainer
from ...fl.server import FLTrainerBase


# ==================== 示例实现 ====================
@trainer("SimpleTrainer")
class SimpleTrainer(FLTrainerBase):
    """
    简单训练器实现示例
    
    演示如何继承FLTrainerBase来实现具体的业务逻辑
    """
    
    def __init__(self, trainer_id: str, learners: Dict[str, LearnerProxy]):
        super().__init__(trainer_id, learners)
        
        # 业务状态
        self.global_model = {'param1': 1.0, 'param2': 2.0}
        
        # 向现有学习器发送初始模型
        asyncio.create_task(self._send_initial_models())
    
    async def _send_initial_models(self):
        """向所有学习器发送初始模型"""
        for learner_id, proxy in self.learners.items():
            try:
                await proxy.set_model(self.global_model)
                self.logger.info(f"已向 {learner_id} 发送初始模型")
            except Exception as e:
                self.logger.warning(f"向 {learner_id} 发送初始模型失败: {e}")
    
    def add_learner(self, learner_id: str, proxy: LearnerProxy):
        """添加学习器时发送初始模型"""
        super().add_learner(learner_id, proxy)
        
        # 异步发送初始模型
        async def send_model():
            try:
                await proxy.set_model(self.global_model)
                self.logger.info(f"已向新学习器 {learner_id} 发送初始模型")
            except Exception as e:
                self.logger.warning(f"向新学习器 {learner_id} 发送初始模型失败: {e}")
        
        asyncio.create_task(send_model())
    
    async def train_round(self, round_num: int, config: Optional[Dict] = None) -> Dict[str, Any]:
        """执行一轮训练"""
        # 1. 并行调用所有学习器进行训练
        train_results = await self.call_all_learners('train', config)
        
        # 2. 过滤有效结果
        valid_results = {k: v for k, v in train_results.items() if v is not None}
        if not valid_results:
            raise ValueError("没有有效的训练结果")
        
        # 3. 聚合模型
        models = [result['model'] for result in valid_results.values()]
        self.global_model = self.aggregate_models(models)
        
        # 4. 广播全局模型（尝试使用transport的广播功能）
        transport = None
        if self.learners:
            first_proxy = next(iter(self.learners.values()))
            transport = getattr(first_proxy, 'transport', None)
        
        await self.broadcast_to_learners({'global_model': self.global_model}, transport)
        
        # 5. 返回结果
        return {
            'participating_learners': list(valid_results.keys()),
            'global_model': self.global_model.copy(),
            'metrics': self._calculate_avg_metrics(valid_results)
        }
    
    def aggregate_models(self, models: List[Dict]) -> Dict:
        """简单模型聚合（平均）"""
        if not models:
            return self.global_model
        
        aggregated = {}
        for key in models[0].keys():
            values = [model[key] for model in models if key in model]
            if values and all(isinstance(v, (int, float)) for v in values):
                aggregated[key] = sum(values) / len(values)
            else:
                aggregated[key] = models[0][key]  # fallback
        
        return aggregated
    
    def _calculate_avg_metrics(self, results: Dict) -> Dict:
        """计算平均指标"""
        metrics = {}
        for result in results.values():
            if 'metrics' in result:
                for key, value in result['metrics'].items():
                    if key not in metrics:
                        metrics[key] = []
                    if isinstance(value, (int, float)):
                        metrics[key].append(value)
        
        return {key: sum(values) / len(values) for key, values in metrics.items()}
    
    async def on_round_end(self, round_num: int, result: Dict):
        """轮次结束后记录日志"""
        participants = len(result['participating_learners'])
        metrics = result.get('metrics', {})
        self.logger.info(f"第 {round_num} 轮完成 - 参与者: {participants}, 指标: {metrics}")

    # ===== 与装饰器兼容的必需方法 =====
    def setup_training(self, **kwargs) -> None:
        """设置训练环境（与装饰器约定保持一致）"""
        self.logger.info("🔧 简单训练器训练环境已设置")
    
    def execute_client_round(self, round_num: int, client_ids: list, global_model_weights: dict = None, **kwargs) -> list:
        """执行一次客户端轮次（与装饰器约定保持一致）"""
        self.logger.info(f"🏃 执行第 {round_num} 轮客户端训练，客户端: {client_ids}")
        
        # 模拟客户端训练结果
        client_results = []
        for client_id in client_ids:
            result = {
                'client_id': client_id,
                'model': {'param1': 1.0 + round_num * 0.1, 'param2': 2.0 + round_num * 0.1},
                'metrics': {'loss': 0.5 - round_num * 0.01, 'accuracy': 0.8 + round_num * 0.01}
            }
            client_results.append(result)
        
        return client_results
    
    def execute_server_aggregation(self, client_results: list, round_num: int = None, **kwargs) -> dict:
        """执行服务端聚合（与装饰器约定保持一致）"""
        self.logger.info(f"🔄 执行服务端聚合，轮次: {round_num}")
        
        if not client_results:
            return {}
        
        # 简单聚合：平均所有客户端的模型参数
        models = [result.get('model', {}) for result in client_results]
        aggregated_model = self.aggregate_models(models)
        
        return {
            'aggregated_weights': aggregated_model,
            'num_participants': len(client_results),
            'aggregation_method': 'simple_average'
        }
