"""
装饰器注册系统演示
demo_decorators.py

展示如何使用MOE-FedCL的装饰器系统注册和发现组件。
"""

import asyncio
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fedcl.api import learner, trainer, aggregator, evaluator
from fedcl.api.discovery import auto_discover_components, list_registered_components
from fedcl.registry import registry
from fedcl.learner.base_learner import BaseLearner
from fedcl.trainer.base_trainer import BaseTrainer


# ==================== 示例1：使用装饰器注册学习器 ====================

@learner('CustomMNIST', 
         description='自定义MNIST学习器',
         version='1.0',
         author='用户示例',
         dataset='MNIST')
class CustomMNISTLearner(BaseLearner):
    """自定义MNIST学习器示例"""
    
    def __init__(self, client_id: str, config: dict, logger=None):
        super().__init__(client_id, config, logger)
        self.model_params = {'weights': [1, 2, 3], 'bias': [0.1, 0.2]}
        print(f"CustomMNISTLearner {client_id} 初始化完成")
    
    async def train(self, request):
        """训练方法"""
        print(f"[{self.client_id}] 开始MNIST训练...")
        await asyncio.sleep(0.1)  # 模拟训练
        
        return {
            'client_id': self.client_id,
            'success': True,
            'loss': 0.5,
            'accuracy': 0.85,
            'epochs_completed': request.get('num_epochs', 1)
        }
    
    async def evaluate(self, model_data=None):
        """评估方法"""
        print(f"[{self.client_id}] 开始MNIST评估...")
        await asyncio.sleep(0.05)
        
        return {
            'accuracy': 0.88,
            'loss': 0.3,
            'samples': 1000
        }
    
    async def get_local_model(self):
        """获取本地模型"""
        return self.model_params
    
    async def set_local_model(self, model_data):
        """设置本地模型"""
        self.model_params = model_data
        return True


@learner('CustomCIFAR', 
         description='自定义CIFAR学习器',
         version='2.0',
         author='用户示例',
         dataset='CIFAR-10')
class CustomCIFARLearner(BaseLearner):
    """自定义CIFAR学习器示例"""
    
    def __init__(self, client_id: str, config: dict, logger=None):
        super().__init__(client_id, config, logger)
        self.model_params = {'conv_layers': 3, 'fc_layers': 2}
        print(f"CustomCIFARLearner {client_id} 初始化完成")
    
    async def train(self, request):
        """训练方法"""
        print(f"[{self.client_id}] 开始CIFAR训练...")
        await asyncio.sleep(0.2)
        
        return {
            'client_id': self.client_id,
            'success': True,
            'loss': 0.7,
            'accuracy': 0.75,
            'epochs_completed': request.get('num_epochs', 1)
        }
    
    async def evaluate(self, model_data=None):
        """评估方法"""
        print(f"[{self.client_id}] 开始CIFAR评估...")
        await asyncio.sleep(0.05)
        
        return {
            'accuracy': 0.78,
            'loss': 0.6,
            'samples': 5000
        }
    
    async def get_local_model(self):
        """获取本地模型"""
        return self.model_params
    
    async def set_local_model(self, model_data):
        """设置本地模型"""
        self.model_params = model_data
        return True


# ==================== 示例2：使用装饰器注册训练器 ====================

@trainer('CustomFedAvg', 
         description='自定义联邦平均训练器',
         version='1.0',
         author='用户示例',
         algorithms=['fedavg', 'weighted_avg'])
class CustomFedAvgTrainer(BaseTrainer):
    """自定义联邦平均训练器"""
    
    def __init__(self, config=None):
        super().__init__(config or {})
        self.global_model = None
        print("CustomFedAvgTrainer 初始化完成")
    
    async def train_round(self, round_num: int, client_ids: list):
        """训练轮次"""
        print(f"开始第 {round_num} 轮训练，客户端: {client_ids}")
        
        results = {}
        for client_id in client_ids:
            if self.is_client_ready(client_id):
                proxy = self.proxy_manager.get_proxy(client_id)
                if proxy:
                    result = await proxy.train({'num_epochs': 1})
                    results[client_id] = result
        
        return {
            'round': round_num,
            'participants': client_ids,
            'results': results,
            'success_count': len(results)
        }
    
    async def aggregate_models(self, client_results):
        """聚合模型"""
        print("开始模型聚合...")
        # 简化的聚合逻辑
        return {'aggregated': True, 'participants': len(client_results)}
    
    async def evaluate_global_model(self):
        """评估全局模型"""
        print("评估全局模型...")
        return {'accuracy': 0.85, 'loss': 0.4}
    
    def should_stop_training(self, round_num: int, round_result):
        """判断是否停止训练"""
        return round_num >= 3  # 简单停止条件


# ==================== 示例3：使用装饰器注册聚合器 ====================

@aggregator('WeightedAvg', 
           description='加权平均聚合器',
           version='1.0',
           author='用户示例',
           algorithm='weighted_average')
class WeightedAverageAggregator:
    """加权平均聚合器"""
    
    def __init__(self, config=None):
        self.config = config or {}
        print("WeightedAverageAggregator 初始化完成")
    
    def aggregate(self, client_models, weights=None):
        """聚合客户端模型"""
        print(f"聚合 {len(client_models)} 个客户端模型")
        
        if weights is None:
            weights = [1.0 / len(client_models)] * len(client_models)
        
        # 简化的聚合逻辑
        aggregated_model = {}
        for i, model in enumerate(client_models):
            weight = weights[i]
            print(f"  - 客户端 {i}: 权重 {weight}")
        
        return {'aggregated_model': aggregated_model, 'total_weight': sum(weights)}


# ==================== 示例4：使用装饰器注册评估器 ====================

@evaluator('AccuracyMetrics', 
          description='准确率指标评估器',
          version='1.0',
          author='用户示例',
          metrics=['accuracy', 'precision', 'recall'])
class AccuracyMetricsEvaluator:
    """准确率指标评估器"""
    
    def __init__(self, config=None):
        self.config = config or {}
        print("AccuracyMetricsEvaluator 初始化完成")
    
    def evaluate(self, predictions, ground_truth):
        """评估预测结果"""
        print("计算准确率指标...")
        
        # 模拟计算
        accuracy = 0.85
        precision = 0.83
        recall = 0.87
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': 2 * (precision * recall) / (precision + recall)
        }


# ==================== 演示函数 ====================

def demo_component_registration():
    """演示组件注册功能"""
    print("🎯 MOE-FedCL 装饰器注册系统演示")
    print("=" * 60)
    
    # 1. 显示已注册的组件
    print("\n1. 已注册的组件:")
    registered = list_registered_components()
    
    for comp_type, components in registered.items():
        print(f"\n{comp_type.upper()}:")
        for name, info in components.items():
            print(f"  - {name}: {info.get('description', 'N/A')} (v{info.get('version', '?')})")
            if info.get('author'):
                print(f"    作者: {info['author']}")
            if info.get('algorithms'):
                print(f"    算法: {info['algorithms']}")
            if info.get('metrics'):
                print(f"    指标: {info['metrics']}")
    
    # 2. 显示注册表统计
    print("\n2. 注册表统计:")
    stats = registry.get_component_count()
    for comp_type, count in stats.items():
        print(f"  {comp_type}: {count} 个")
    
    # 3. 测试组件获取
    print("\n3. 测试组件获取:")
    
    try:
        # 获取学习器
        mnist_learner_cls = registry.get_learner('CustomMNIST')
        print(f"✅ 成功获取学习器: {mnist_learner_cls.__name__}")
        
        # 获取训练器
        trainer_cls = registry.get_trainer('CustomFedAvg')
        print(f"✅ 成功获取训练器: {trainer_cls.__name__}")
        
        # 获取聚合器
        aggregator_cls = registry.get_aggregator('WeightedAvg')
        print(f"✅ 成功获取聚合器: {aggregator_cls.__name__}")
        
        # 获取评估器
        evaluator_cls = registry.get_evaluator('AccuracyMetrics')
        print(f"✅ 成功获取评估器: {evaluator_cls.__name__}")
        
    except ValueError as e:
        print(f"❌ 获取组件失败: {e}")
    
    # 4. 测试组件实例化
    print("\n4. 测试组件实例化:")
    
    try:
        # 实例化学习器
        learner = mnist_learner_cls('demo_client', {})
        print(f"✅ 学习器实例化成功: {type(learner).__name__}")
        
        # 实例化聚合器
        aggregator = aggregator_cls()
        print(f"✅ 聚合器实例化成功: {type(aggregator).__name__}")
        
        # 实例化评估器
        evaluator = evaluator_cls()
        print(f"✅ 评估器实例化成功: {type(evaluator).__name__}")
        
    except Exception as e:
        print(f"❌ 实例化失败: {e}")


async def demo_component_usage():
    """演示组件使用"""
    print("\n5. 演示组件使用:")
    
    try:
        # 创建学习器实例
        learner_cls = registry.get_learner('CustomMNIST')
        learner = learner_cls('demo_client', {})
        
        # 测试训练
        train_result = await learner.train({'num_epochs': 1})
        print(f"✅ 训练结果: {train_result}")
        
        # 测试评估
        eval_result = await learner.evaluate()
        print(f"✅ 评估结果: {eval_result}")
        
        # 测试聚合器
        aggregator_cls = registry.get_aggregator('WeightedAvg')
        aggregator = aggregator_cls()
        
        agg_result = aggregator.aggregate([{'model': 1}, {'model': 2}], [0.6, 0.4])
        print(f"✅ 聚合结果: {agg_result}")
        
        # 测试评估器
        evaluator_cls = registry.get_evaluator('AccuracyMetrics')
        evaluator = evaluator_cls()
        
        metrics = evaluator.evaluate([1, 1, 0], [1, 0, 0])
        print(f"✅ 评估指标: {metrics}")
        
    except Exception as e:
        print(f"❌ 组件使用失败: {e}")
        import traceback
        traceback.print_exc()


def demo_auto_discovery():
    """演示自动发现功能"""
    print("\n6. 演示自动发现功能:")
    
    # 从当前文件发现组件
    discovered = auto_discover_components([__file__])
    print(f"从当前文件发现的组件: {discovered}")


if __name__ == "__main__":
    try:
        # 运行演示
        demo_component_registration()
        
        # 运行异步演示
        asyncio.run(demo_component_usage())
        
        # 演示自动发现
        demo_auto_discovery()
        
        print("\n🎉 装饰器注册系统演示完成！")
        
    except Exception as e:
        print(f"❌ 演示失败: {e}")
        import traceback
        traceback.print_exc()
