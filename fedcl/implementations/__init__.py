# fedcl/implementations/__init__.py
"""
FedCL 具体算法实现层

提供各种持续学习算法、联邦聚合算法、评估器和钩子的具体实现。
所有实现都基于相应的抽象基类，并通过组件注册系统进行管理。

主要模块:
- learners: 持续学习算法实现 (L2P, EWC, Replay等)
- aggregators: 联邦聚合算法实现 (FedAvg, FedProx, SCAFFOLD等)
- evaluators: 评估器实现 (Accuracy, Forgetting, Transfer等)
- hooks: 钩子实现 (WandB, TensorBoard, Custom等)

使用示例:
    from fedcl.implementations.learners import L2PLearner, EWCLearner
    from fedcl.implementations.aggregators import FedAvgAggregator
    from fedcl.implementations.evaluators import AccuracyEvaluator
    
    # 通过注册系统使用
    from fedcl.registry import registry
    learner_cls = registry.get_learner("l2p")
"""

# 导入所有实现以触发注册
from . import learners
from . import aggregators  
from . import evaluators
from . import hooks

# 版本信息
__version__ = "1.0.0"
__author__ = "FedCL Team"

# API 导出
__all__ = [
    "learners",
    "aggregators", 
    "evaluators",
    "hooks"
]
