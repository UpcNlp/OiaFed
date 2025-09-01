"""
经典联邦学习方法

本模块提供了预设的经典联邦学习算法实现，用户可以直接使用而无需自己实现。
主要包含：

1. **聚合器 (Aggregators)**：实现不同的参数聚合策略
   - FedAvg: 经典的加权平均聚合
   - FedProx: 带正则化的聚合
   - SCAFFOLD: 带控制变量的聚合
   - FedNova: 归一化聚合
   - FedAdam/FedYogi: 自适应优化聚合
   - FedDyn: 动态正则化聚合

2. **学习器 (Learners)**：实现特殊的客户端学习策略
   - ContrastiveLearner: 对比学习
   - PersonalizedLearner: 个性化学习
   - MetaLearner: 元学习

3. **评估器 (Evaluators)**：实现特殊的评估方法
   - PrototypeEvaluator: 基于原型的评估
   - FairnessEvaluator: 公平性评估

使用方式：
```python
# 配置文件方式
config = {
    "aggregator": "fedavg",  # 使用FedAvg聚合器
    "learner": "simple_learner",  # 使用标准学习器
    "trainer": "standard"  # 使用标准训练器
}

# 或者直接导入使用
from fedcl.methods.aggregators import FedAvgAggregator
from fedcl.methods.learners import ContrastiveLearner
```
"""

# 导入聚合器
from .aggregators import (
    FedAvgAggregator,
    FedProxAggregator,
    SCAFFOLDAggregator,
    FedNovaAggregator,
    FedAdamAggregator,
    FedYogiAggregator,
    FedDynAggregator
)

# 导入学习器
from .learners import (
    ContrastiveLearner,
    PersonalizedClientLearner,
    MetaLearner
)

# 导入评估器
from .evaluators import (
    PrototypeEvaluator,
    FairnessEvaluator
)

__all__ = [
    # 聚合器
    "FedAvgAggregator",
    "FedProxAggregator", 
    "SCAFFOLDAggregator",
    "FedNovaAggregator",
    "FedAdamAggregator",
    "FedYogiAggregator",
    "FedDynAggregator",
    
    # 学习器
    "ContrastiveLearner",
    "PersonalizedClientLearner",
    "MetaLearner",
    
    # 评估器
    "PrototypeEvaluator",
    "FairnessEvaluator"
]