"""
新联邦学习架构演示

演示基于长连接的MVP联邦学习架构：
1. 轻量化的基础设施
2. 动态RPC代理
3. 推送机制
4. 用户自定义业务逻辑
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from typing import Dict, Any
from loguru import logger

# 导入新架构的组件
from fedcl.fl import (
    BaseFLNode,
    TrainerBase
)
from fedcl.comm import MemoryTransport
from fedcl.utils.auto_logger import setup_auto_logging, get_train_logger, get_sys_logger
from fedcl import learner, trainer

# ==================== 用户自定义学习器 ====================

class CustomLearner(BaseFLNode):
    """
    用户自定义学习器
    演示如何实现自己的业务逻辑
    """
    
    def __init__(self, learner_id: str, dataset_size: int = 100):
        super().__init__(learner_id, auto_connect=False)  # 手动控制连接
        
        # 业务逻辑相关
        self.dataset_size = dataset_size
        self.model = {'weights': [1.0, 2.0, 3.0], 'bias': 0.1}
        self.local_epochs = 5
        self.learning_rate = 0.01
        
        # RPC方法注册
        self._rpc_handlers = {
            'train': self.train,
            'evaluate': self.evaluate,
            'get_model': self.get_model,
            'set_model': self.set_model,
            'get_dataset_info': self.get_dataset_info,
            'ping': self.ping,
            '__get_methods__': self.get_methods
        }
        
    async def train(self, config: Dict = None) -> Dict[str, Any]:
        """自定义训练逻辑"""
        # 使用训练日志记录器
        try:
            train_logger = get_train_logger(self.node_id)
            train_logger.info(f"开始训练 - 数据集大小: {self.dataset_size}")
        except:
            self.logger.info(f"开始训练 - 数据集大小: {self.dataset_size}")
        
        # 解析训练配置
        epochs = config.get('epochs', self.local_epochs) if config else self.local_epochs
        lr = config.get('learning_rate', self.learning_rate) if config else self.learning_rate
        
        # 模拟训练过程
        initial_loss = 1.0
        for epoch in range(epochs):
            # 模拟训练迭代
            await asyncio.sleep(0.01)  # 模拟计算时间
            
            # 更新模型参数
            for i in range(len(self.model['weights'])):
                self.model['weights'][i] += lr * 0.1 * (0.5 - epoch/epochs)
            self.model['bias'] += lr * 0.05
            
        final_loss = initial_loss * (1 - epochs * 0.1)
        accuracy = min(0.95, 0.6 + epochs * 0.05)
        
        result = {
            'model': self.model.copy(),
            'metrics': {
                'loss': final_loss,
                'accuracy': accuracy,
                'epochs': epochs,
                'samples': self.dataset_size
            },
            'learner_id': self.node_id
        }
        
        # 记录训练完成
        try:
            train_logger = get_train_logger(self.node_id)
            train_logger.info(f"训练完成 - Loss: {final_loss:.3f}, Acc: {accuracy:.3f}")
        except:
            self.logger.info(f"训练完成 - Loss: {final_loss:.3f}, Acc: {accuracy:.3f}")
        return result
        
    async def evaluate(self, test_data: Any = None) -> Dict[str, Any]:
        """模型评估"""
        try:
            train_logger = get_train_logger(self.node_id)
            train_logger.info("开始评估")
        except:
            self.logger.info("开始评估")
        
        # 模拟评估过程
        await asyncio.sleep(0.05)
        
        test_loss = 0.15
        test_accuracy = 0.92
        
        return {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'test_samples': 50
        }
        
    def get_model(self) -> Dict:
        """获取模型"""
        return self.model.copy()
        
    def set_model(self, model: Dict):
        """设置模型"""
        self.model = model.copy()
        self.logger.info("模型已更新")
        
    def get_dataset_info(self) -> Dict:
        """获取数据集信息"""
        return {
            'dataset_size': self.dataset_size,
            'features': len(self.model['weights']),
            'data_type': 'simulated'
        }
        
    async def ping(self) -> str:
        return f"pong from {self.node_id}"
        
    def get_methods(self) -> Dict[str, Dict]:
        """返回可调用方法信息"""
        methods = {}
        for name, method in self._rpc_handlers.items():
            if not name.startswith('_'):
                methods[name] = {
                    'is_async': asyncio.iscoroutinefunction(method),
                    'description': method.__doc__ or 'No description'
                }
        return methods
        
    async def on_push_received(self, data: Any):
        """接收全局模型推送"""
        if isinstance(data, dict) and 'global_model' in data:
            self.set_model(data['global_model'])
            self.logger.info("接收到全局模型推送")
            
    async def handle_rpc(self, request: Dict) -> Dict:
        """处理RPC请求"""
        method_name = request.get('method')
        args = request.get('args', ())
        kwargs = request.get('kwargs', {})
        
        try:
            handler = self._rpc_handlers.get(method_name)
            if not handler:
                raise ValueError(f"未知方法: {method_name}")
                
            if asyncio.iscoroutinefunction(handler):
                result = await handler(*args, **kwargs)
            else:
                result = handler(*args, **kwargs)
                
            return {'result': result}
            
        except Exception as e:
            return {'error': str(e)}


# ==================== 用户自定义训练器 ====================

class CustomTrainer(TrainerBase):
    """
    用户自定义训练器
    演示如何实现自己的联邦学习算法
    """
    
    def __init__(self, trainer_id: str):
        super().__init__(trainer_id)
        
        # 算法相关参数
        self.global_model = {'weights': [1.0, 2.0, 3.0], 'bias': 0.1}
        self.min_participants = 2
        self.aggregation_strategy = 'weighted_avg'
        self.round_timeout = 30.0  # 30秒超时
        
        # 训练历史
        self.training_history = []
        
    async def federated_round(self, round_num: int, config: Dict = None) -> Dict:
        """执行一轮联邦训练"""
        self.logger.info(f"=== 开始第 {round_num} 轮联邦训练 ===")
        start_time = asyncio.get_event_loop().time()
        
        try:
            # 1. 选择参与客户端（这里简单选择所有）
            participants = list(self.learners.keys())
            if len(participants) < self.min_participants:
                raise ValueError(f"参与客户端数量不足: {len(participants)} < {self.min_participants}")
                
            self.logger.info(f"选择参与客户端: {participants}")
            
            # 2. 准备训练配置
            training_config = {
                'round': round_num,
                'global_model': self.global_model,
                'epochs': config.get('epochs', 3) if config else 3,
                'learning_rate': config.get('learning_rate', 0.01) if config else 0.01
            }
            
            # 3. 并行训练所有参与客户端
            self.logger.info("开始并行训练...")
            train_results = await self.call_all_learners('train', training_config)
            
            # 4. 过滤有效结果
            valid_results = {k: v for k, v in train_results.items() if v is not None}
            if len(valid_results) < self.min_participants:
                raise ValueError(f"有效训练结果不足: {len(valid_results)} < {self.min_participants}")
                
            # 5. 聚合模型
            self.logger.info(f"聚合 {len(valid_results)} 个模型...")
            self.global_model = self._custom_aggregate(valid_results)
            
            # 6. 广播全局模型
            await self.broadcast_global_model(self.global_model)
            
            # 7. 可选：评估全局模型
            eval_results = await self.call_all_learners('evaluate')
            avg_eval = self._calculate_avg_evaluation(eval_results)
            
            # 8. 记录轮次结果
            round_time = asyncio.get_event_loop().time() - start_time
            round_result = {
                'round': round_num,
                'participants': list(valid_results.keys()),
                'global_model': self.global_model.copy(),
                'training_metrics': self._aggregate_metrics(valid_results),
                'evaluation_metrics': avg_eval,
                'round_time': round_time
            }
            
            self.training_history.append(round_result)
            
            self.logger.info(f"=== 第 {round_num} 轮训练完成，用时 {round_time:.2f}s ===")
            return round_result
            
        except Exception as e:
            self.logger.exception(f"第 {round_num} 轮训练失败: {e}")
            raise
            
    def _custom_aggregate(self, results: Dict[str, Dict]) -> Dict:
        """自定义聚合策略"""
        if self.aggregation_strategy == 'weighted_avg':
            return self._weighted_average_aggregate(results)
        elif self.aggregation_strategy == 'simple_avg':
            return self._simple_average_aggregate(results)
        else:
            raise ValueError(f"未知聚合策略: {self.aggregation_strategy}")
            
    def _weighted_average_aggregate(self, results: Dict[str, Dict]) -> Dict:
        """加权平均聚合"""
        total_samples = sum(r['metrics']['samples'] for r in results.values())
        
        # 初始化聚合模型
        aggregated = {}
        first_model = list(results.values())[0]['model']
        
        for key in first_model:
            if isinstance(first_model[key], list):
                aggregated[key] = [0.0] * len(first_model[key])
            else:
                aggregated[key] = 0.0
                
        # 加权聚合
        for result in results.values():
            model = result['model']
            weight = result['metrics']['samples'] / total_samples
            
            for key in model:
                if isinstance(model[key], list):
                    for i in range(len(model[key])):
                        aggregated[key][i] += model[key][i] * weight
                else:
                    aggregated[key] += model[key] * weight
                    
        return aggregated
        
    def _simple_average_aggregate(self, results: Dict[str, Dict]) -> Dict:
        """简单平均聚合"""
        num_models = len(results)
        
        # 初始化聚合模型
        aggregated = {}
        first_model = list(results.values())[0]['model']
        
        for key in first_model:
            if isinstance(first_model[key], list):
                aggregated[key] = [0.0] * len(first_model[key])
            else:
                aggregated[key] = 0.0
                
        # 简单平均
        for result in results.values():
            model = result['model']
            
            for key in model:
                if isinstance(model[key], list):
                    for i in range(len(model[key])):
                        aggregated[key][i] += model[key][i] / num_models
                else:
                    aggregated[key] += model[key] / num_models
                    
        return aggregated
        
    def _aggregate_metrics(self, results: Dict[str, Dict]) -> Dict:
        """聚合训练指标"""
        metrics = {}
        for result in results.values():
            for key, value in result['metrics'].items():
                if key not in metrics:
                    metrics[key] = []
                if isinstance(value, (int, float)):
                    metrics[key].append(value)
                    
        # 计算平均值
        return {key: sum(values) / len(values) for key, values in metrics.items()}
        
    def _calculate_avg_evaluation(self, eval_results: Dict) -> Dict:
        """计算平均评估指标"""
        if not eval_results:
            return {}
            
        valid_evals = [v for v in eval_results.values() if v is not None]
        if not valid_evals:
            return {}
            
        avg_metrics = {}
        for eval_result in valid_evals:
            for key, value in eval_result.items():
                if key not in avg_metrics:
                    avg_metrics[key] = []
                if isinstance(value, (int, float)):
                    avg_metrics[key].append(value)
                    
        return {key: sum(values) / len(values) for key, values in avg_metrics.items()}
        
    async def broadcast_global_model(self, model: Dict):
        """广播全局模型"""
        await self.broadcast_to_learners({'global_model': model})
        self.logger.info("全局模型已广播")
        
    def get_training_summary(self) -> Dict:
        """获取训练摘要"""
        if not self.training_history:
            return {'status': 'no_training'}
            
        latest = self.training_history[-1]
        return {
            'total_rounds': len(self.training_history),
            'latest_round': latest['round'],
            'latest_metrics': latest.get('training_metrics', {}),
            'latest_evaluation': latest.get('evaluation_metrics', {}),
            'global_model': self.global_model.copy()
        }


# ==================== 演示主函数 ====================

async def main():
    """演示新架构的使用"""
    logger.info("🚀 开始演示新联邦学习架构")
    
    # 0. 初始化自动分流日志系统
    from datetime import datetime
    experiment_date = datetime.now().strftime("%Y%m%d-%H-%M-%S")
    auto_logger = setup_auto_logging(experiment_date)
    sys_logger = get_sys_logger()
    
    logger.info(f"📋 日志保存到: logs/exp_{experiment_date}/")
    logger.info("  ├── comm/    # 通信日志")
    logger.info("  ├── train/   # 训练日志")
    logger.info("  └── sys/     # 系统日志")
    
    # 1. 创建内存传输（模拟分布式环境）
    server_transport = MemoryTransport("server")
    
    # 2. 创建训练器
    trainer = CustomTrainer("server")
    trainer.transport = server_transport
    
    # 3. 创建学习器（模拟3个客户端）
    learners = []
    client_transports = {}  # 存储每个客户端的传输实例
    
    for i in range(3):
        learner_id = f"client_{i+1}"
        dataset_size = 50 + i * 25  # 不同的数据集大小
        
        # 为每个客户端创建独立的传输实例
        client_transport = MemoryTransport(learner_id)
        client_transports[learner_id] = client_transport
        
        # 创建学习器
        learner = CustomLearner(learner_id, dataset_size)
        learner.transport = client_transport
        learners.append(learner)
        
        # 在训练器中添加学习器代理（使用客户端的传输实例）
        proxy = trainer.add_learner(learner_id, client_transport)
        
        # 设置学习器的RPC处理（模拟真实的RPC调用）
        def create_mock_rpc_call(learner_obj):
            async def mock_rpc_call(method_name, *args, **kwargs):
                request = {
                    'method': method_name,
                    'args': args,
                    'kwargs': kwargs
                }
                result = await learner_obj.handle_rpc(request)
                return result.get('result') if 'result' in result else None
            return mock_rpc_call
            
        # 重写代理的RPC调用方法（在实际环境中这由传输层处理）
        proxy._rpc_call = create_mock_rpc_call(learner)
        
    logger.info(f"创建了 {len(learners)} 个学习器")
    
    # 4. 演示方法发现
    logger.info("\n📡 演示动态方法发现:")
    for learner_id, proxy in trainer.learners.items():
        try:
            methods = await proxy.__get_methods__()
            logger.info(f"{learner_id} 可用方法: {list(methods.keys())}")
        except Exception as e:
            logger.warning(f"方法发现失败: {e}")
            
    # 5. 演示数据集信息获取
    logger.info("\n📊 获取数据集信息:")
    dataset_info = await trainer.call_all_learners('get_dataset_info')
    for learner_id, info in dataset_info.items():
        if info:
            logger.info(f"{learner_id}: {info}")
            
    # 6. 执行联邦训练
    logger.info("\n🎯 开始联邦训练演示:")
    
    training_config = {
        'epochs': 3,
        'learning_rate': 0.01
    }
    
    for round_num in range(1, 4):  # 执行3轮训练
        try:
            result = await trainer.federated_round(round_num, training_config)
            
            # 显示轮次结果
            metrics = result['training_metrics']
            eval_metrics = result.get('evaluation_metrics', {})
            
            logger.info(f"第 {round_num} 轮结果:")
            logger.info(f"  参与者: {len(result['participants'])}")
            logger.info(f"  平均Loss: {metrics.get('loss', 0):.4f}")
            logger.info(f"  平均Acc: {metrics.get('accuracy', 0):.4f}")
            if eval_metrics:
                logger.info(f"  测试Acc: {eval_metrics.get('test_accuracy', 0):.4f}")
            logger.info(f"  用时: {result['round_time']:.2f}s")
            
            # 稍微等待
            await asyncio.sleep(0.5)
            
        except Exception as e:
            logger.error(f"第 {round_num} 轮训练失败: {e}")
            break
            
    # 7. 显示最终结果
    logger.info("\n📈 训练总结:")
    summary = trainer.get_training_summary()
    if 'total_rounds' in summary:
        logger.info(f"总轮数: {summary['total_rounds']}")
        logger.info(f"最终指标: {summary.get('latest_metrics', {})}")
        logger.info(f"最终模型: {summary['global_model']}")
    else:
        logger.info(f"训练状态: {summary.get('status', 'unknown')}")
    
    logger.info("✅ 演示完成！")


if __name__ == "__main__":
    asyncio.run(main())
