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
   - GenericLearner: 通用学习器（推荐）
   - MNISTLearner: MNIST专用学习器
   - ContrastiveLearner: 对比学习
   - PersonalizedLearner: 个性化学习
   - MetaLearner: 元学习

3. **训练器 (Trainers)**：实现联邦训练协调
   - GenericTrainer: 通用训练器（推荐）
   - FedAvgMNISTTrainer: FedAvg MNIST训练器

4. **模型 (Models)**：实现神经网络模型
   - MNISTCNNModel: MNIST CNN模型

5. **数据集 (Datasets)**：实现数据加载和划分
   - MNISTFederatedDataset: MNIST联邦数据集

6. **评估器 (Evaluators)**：实现特殊的评估方法
   - PrototypeEvaluator: 基于原型的评估
   - FairnessEvaluator: 公平性评估

使用方式：
```python
# 只需导入 fedcl.methods，所有组件自动注册
import fedcl.methods

# 然后在配置文件中使用
config = {
    "trainer": {"name": "Generic"},
    "learner": {"name": "Generic"},
    "dataset": {"name": "MNIST"}
}
```
"""

# 导入所有子模块，触发自动注册
from . import aggregators  # noqa: F401
from . import learners  # noqa: F401
from . import evaluators  # noqa: F401
from . import trainers  # noqa: F401
from . import models  # noqa: F401
from . import datasets  # noqa: F401

# 同时保留旧的导入方式（向后兼容）
from .aggregators import *  # noqa: F401, F403
from .learners import *  # noqa: F401, F403
from .evaluators import *  # noqa: F401, F403
from .trainers import *  # noqa: F401, F403
