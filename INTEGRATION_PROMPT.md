# 联邦学习算法集成完整指南 - LLM Prompt

## 任务概述

你是一位联邦学习/持续学习/遗忘学习算法专家。你的任务是将经典算法**完整地**集成到本开源框架中，并精确复现论文中的实验结果。

**重要**：本框架支持三大类算法：
- **fl.xxx**: 联邦学习算法 (Federated Learning)
- **cl.xxx**: 持续学习算法 (Continual Learning)
- **ul.xxx**: 遗忘学习算法 (Unlearning)

所有算法共享统一的实验管理数据库和评估体系。

---

## 框架核心设计理念

### 1. 装饰器驱动的组件注册系统 ⭐

本框架的**核心特性**是基于装饰器的自动注册系统，所有组件（Learner、Model、Dataset、Trainer、Aggregator、Metric）都通过装饰器注册到全局注册表中。

**为什么使用装饰器？**
- ✅ **自动发现**: 框架启动时自动扫描并注册所有组件，无需手动维护列表
- ✅ **解耦**: 算法实现与框架核心完全解耦，易于扩展
- ✅ **类型安全**: 装饰器提供元数据验证和类型检查
- ✅ **配置驱动**: 通过YAML配置文件即可切换算法，无需修改代码
- ✅ **统一管理**: 所有算法类型（fl/cl/ul）使用相同的注册机制

**关键装饰器**:
```python
from fedcl.methods.learners._decorators import learner
from fedcl.api.decorators import model, dataset, trainer, aggregator

@learner(namespace='fl', name='FedAvg', description='...')  # 注册学习器 (必须)
@model(name='ResNet18', ...)         # 注册模型 (可选)
@dataset(name='CIFAR10', ...)        # 注册数据集 (可选)
@trainer(name='FLTrainer', ...)      # 注册训练器 (可选)
@aggregator(name='FedAvg', ...)      # 注册聚合器 (可选)
```

### 2. 业务层通信协议 ⭐

联邦学习的核心是分布式通信。框架提供了完整的业务层通信抽象：

**通信流程**:
```
Server                           Client
  │                                 │
  ├─► broadcast(global_model) ─────►│  # 服务器广播全局模型
  │                                 │
  │◄──── upload(local_update) ──────┤  # 客户端上传本地更新
  │                                 │
  ├─► aggregate(updates) ───────────┤  # 服务器聚合
  │                                 │
  └─► broadcast(new_global_model) ─►│  # 新一轮通信
```

**需要实现的通信接口**:
- `get_local_model()` / `get_model()`: 定义上传哪些参数（全部/部分/特征）
- `set_local_model()` / `set_model()`: 定义如何接收并更新参数
- `aggregate()`: 自定义聚合逻辑（在Aggregator中实现）

注：BaseLearner定义了抽象方法 `get_local_model()`/`set_local_model()`，子类通常实现 `get_model()`/`set_model()` 并委托调用。

### 3. 完整的组件生态系统

每个算法不仅仅是一个Learner，而是一个完整的系统：

```
算法系统 = Learner + Trainer + Model + Aggregator + Dataset + Metrics
         ↓        ↓        ↓         ↓          ↓         ↓
      核心逻辑  训练流程  网络结构  聚合策略   数据分区  评估指标
```

**为什么需要这么多组件？**
- 不同算法的训练流程可能完全不同（同步/异步/半监督）
- 模型架构可能有特殊设计（个性化层/共享层/原型网络）
- 聚合策略各异（加权平均/中位数/自适应权重）
- 评估指标需要定制（准确率/遗忘率/公平性）

---

## 框架目录结构

```
MOE-FedCL/
├── fedcl/
│   ├── api/
│   │   └── decorators.py              # 🔑 核心装饰器定义
│   │
│   ├── methods/
│   │   ├── learners/                  # 学习算法实现
│   │   │   ├── _registry.py           # 全局注册表
│   │   │   ├── _decorators.py         # @learner装饰器
│   │   │   ├── fl/                    # 联邦学习: fl.xxx
│   │   │   │   ├── fedavg.py          # 示例: fl.FedAvg
│   │   │   │   ├── fedprox.py         # 示例: fl.FedProx
│   │   │   │   └── [your_algorithm].py
│   │   │   ├── cl/                    # 持续学习: cl.xxx
│   │   │   │   ├── ewc.py             # 示例: cl.EWC
│   │   │   │   └── [your_algorithm].py
│   │   │   └── ul/                    # 遗忘学习: ul.xxx
│   │   │       ├── retrain.py         # 示例: ul.Retrain
│   │   │       └── [your_algorithm].py
│   │   │
│   │   ├── models/                    # 模型定义
│   │   │   ├── __init__.py
│   │   │   ├── lenet.py               # LeNet-5
│   │   │   ├── resnet.py              # ResNet系列
│   │   │   ├── vgg.py                 # VGG系列
│   │   │   └── [algorithm_name]_net.py # 算法特定模型
│   │   │
│   │   ├── datasets/                  # 数据集实现
│   │   │   ├── __init__.py
│   │   │   ├── mnist.py               # @dataset装饰
│   │   │   ├── cifar10.py
│   │   │   └── [dataset_name].py
│   │   │
│   │   ├── trainers/                  # 训练器
│   │   │   ├── __init__.py
│   │   │   ├── generic.py             # 通用FL训练器
│   │   │   ├── continual.py           # 持续学习训练器
│   │   │   └── [custom_trainer].py
│   │   │
│   │   ├── aggregators/               # 聚合器 ⭐ 新增
│   │   │   ├── __init__.py
│   │   │   ├── fedavg.py              # 加权平均聚合
│   │   │   ├── fedopt.py              # 服务器优化器聚合
│   │   │   └── [custom_aggregator].py
│   │   │
│   │   └── metrics/                   # 评估指标 ⭐ 新增
│   │       ├── __init__.py
│   │       ├── accuracy.py            # 准确率
│   │       ├── forgetting.py          # 遗忘率（CL专用）
│   │       └── fairness.py            # 公平性（FL专用）
│   │
│   ├── federation/                    # 联邦基础设施
│   │   ├── server.py                  # 服务器逻辑
│   │   ├── client.py                  # 客户端逻辑
│   │   └── communication.py           # 通信协议
│   │
│   └── utils/
│
├── configs/
│   └── distributed/
│       ├── base/
│       │   ├── server_base.yaml       # 服务器基础配置
│       │   └── client_base.yaml       # 客户端基础配置
│       └── experiments/
│
├── papers/                            # 🔑 所有论文复现脚本
│   ├── fedavg_mcmahan2017/
│   │   ├── reproduce.py               # 复现脚本
│   │   ├── configs/                   # 实验配置
│   │   └── README.md                  # 论文信息和结果
│   ├── fedprox_li2020/
│   ├── moon_li2021/
│   └── [paper_name_author_year]/     # 统一命名规范
│
└── examples/
    └── smart_batch_runner.py          # 批量实验调度器
```

---

## 完整集成流程（8大步骤）

### 步骤1: 深入理解论文算法

**必须提取的信息**：

#### 1.1 算法分类
```
算法类型: [ ] fl.xxx (联邦学习)
         [ ] cl.xxx (持续学习)
         [ ] ul.xxx (遗忘学习)
```

#### 1.2 核心创新点
- **一句话总结**: [用1句话概括算法核心思想]
- **与基线的差异**:
  - 模型结构: [是否有变化？如何变化？]
  - 训练过程: [是否有特殊损失/正则化/约束？]
  - 通信内容: [上传什么？下载什么？与FedAvg有何不同？]
  - 聚合策略: [如何聚合？是否有自适应权重？]

#### 1.3 关键组件需求分析
```python
需要自定义的组件:
[ ] Learner      - 核心算法逻辑
[ ] Trainer      - 是否需要特殊训练流程？
[ ] Model        - 是否需要特殊网络结构？
[ ] Aggregator   - 是否需要自定义聚合？
[ ] Dataset      - 是否需要特殊数据处理？
[ ] Metric       - 是否需要新的评估指标？
```

#### 1.4 超参数提取
```yaml
# 从论文Table/Appendix提取所有超参数
算法特有参数:
  - param_name_1: [默认值] # [说明]
  - param_name_2: [默认值] # [说明]

标准参数:
  - learning_rate: [值]
  - batch_size: [值]
  - local_epochs: [值]
  - communication_rounds: [值]
```

#### 1.5 实验设置
```yaml
数据集: [MNIST, CIFAR10, ...]
Non-IID设置:
  - [Dirichlet(α=0.5)]
  - [Pathological(#C=2)]
  - [...]
客户端数量: [10, 100, ...]
参与率: [1.0, 0.1, ...]
```

---

### 步骤2: 分析开源代码（如果有）

**关键代码位置识别**：

```python
# 1. 客户端本地训练逻辑
def local_train(self, ...):
    # 找到这个函数，分析:
    # - 损失函数的组成
    # - 是否有特殊的正则化项
    # - 是否使用了辅助模型/数据结构
    pass

# 2. 参数上传逻辑
def get_params_to_upload(self):
    # 上传全部参数？部分参数？还是其他信息？
    pass

# 3. 参数更新逻辑
def update_from_server(self, params):
    # 如何处理服务器下发的参数？
    # 是覆盖？融合？还是只更新部分层？
    pass

# 4. 服务器聚合逻辑
def aggregate(self, client_updates):
    # 加权平均？中位数？还是更复杂的策略？
    pass
```

---

### 步骤3: 实现Learner（核心组件）

#### 3.1 创建文件
```bash
# 根据算法类型选择目录
fedcl/methods/learners/fl/[algorithm_name].py   # 联邦学习
fedcl/methods/learners/cl/[algorithm_name].py   # 持续学习
fedcl/methods/learners/ul/[algorithm_name].py   # 遗忘学习
```

#### 3.2 完整Learner模板

```python
"""
[算法全名] ([算法简称])

论文: [标题]
作者: [作者列表]
会议/期刊: [venue, year]
链接: [arXiv/DOI]

核心思想:
    [2-3句话描述算法的核心创新点]

关键特性:
    1. [特性1]
    2. [特性2]
    3. [特性3]

与FedAvg的主要区别:
    - [区别1]
    - [区别2]
"""
from typing import Dict, Any, Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import copy

from fedcl.methods.learners._decorators import learner
from fedcl.learner.base_learner import BaseLearner  # 基类


@learner(
    namespace='[fl|cl|ul]',  # 命名空间: fl(联邦学习), cl(持续学习), ul(遗忘学习)
    name='[算法名]',          # 例如: 'FedAvg', 'MOON', 'TARGET'
    description='[一句话描述（可选）]'  # 例如: 'FedAvg: Federated Averaging'
)
class [AlgorithmName]Learner(BaseLearner):  # 继承BaseLearner基类
    """
    [算法名称] Learner实现

    参数说明:
        model: 神经网络模型
        device: 训练设备
        learning_rate: 学习率
        local_epochs: 本地训练轮数
        batch_size: 批次大小

        # 算法特有参数（以MOON为例）
        mu: 对比损失权重（默认: 1.0）
        temperature: 对比学习温度（默认: 0.5）

    通信协议:
        上传: [描述上传什么，如: 模型参数, 特征原型, 统计信息等]
        下载: [描述下载什么，如: 全局模型, 聚合特征等]

    示例:
        >>> learner = [AlgorithmName]Learner(
        ...     model=model,
        ...     device='cuda',
        ...     learning_rate=0.01,
        ...     mu=1.0
        ... )
        >>> results = learner.local_train(train_loader)
    """

    def __init__(self, client_id: str, config: Dict[str, Any] = None, lazy_init: bool = True):
        """
        初始化学习器

        Args:
            client_id: 客户端唯一标识
            config: 组件配置字典（由框架传入）
            lazy_init: 是否延迟初始化组件（默认True）
        """
        # 提取learner配置
        learner_params = (config or {}).get('learner', {}).get('params', {})

        # 提取模型、优化器、损失函数配置
        self._model_cfg = learner_params.get('model', {})
        self._optimizer_cfg = learner_params.get('optimizer', {
            'type': 'SGD',
            'lr': learner_params.get('learning_rate', 0.01),
            'momentum': 0.9
        })
        self._loss_cfg = learner_params.get('loss', 'CrossEntropyLoss')

        # 标准训练参数
        self._lr = learner_params.get('learning_rate', 0.01)
        self._bs = learner_params.get('batch_size', 32)
        self._epochs = learner_params.get('local_epochs', 5)

        # 算法特有参数（从learner_params中提取）
        self.special_param_1 = learner_params.get('special_param_1', 1.0)
        self.special_param_2 = learner_params.get('special_param_2', 10)

        # 调用父类初始化
        super().__init__(client_id, config, lazy_init)

        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 组件占位符（延迟加载）
        self._model = None
        self._optimizer = None
        self._criterion = None
        self._train_loader = None

        # 初始化算法特有的数据结构
        self.global_model = None      # 用于保存全局模型副本
        self.prev_model = None         # 用于保存上一轮模型
        self.prototypes = None         # 类原型（如果需要）

        self.logger.info(
            f"{self.__class__.__name__} {client_id} 初始化完成: "
            f"model={self._model_cfg.get('name')}, "
            f"special_param_1={self.special_param_1}"
        )

    # 使用@property实现延迟加载
    @property
    def model(self):
        """延迟加载模型"""
        if self._model is None:
            from fedcl.api.registry import registry
            model_name = self._model_cfg['name']
            model_params = self._model_cfg.get('params', {})
            model_class = registry.get_model(model_name)
            self._model = model_class(**model_params).to(self.device)
            self.logger.debug(f"模型 {model_name} 创建完成")
        return self._model

    @property
    def optimizer(self):
        """延迟加载优化器"""
        if self._optimizer is None:
            opt_type = self._optimizer_cfg.get('type', 'SGD').upper()
            lr = self._optimizer_cfg.get('lr', self._lr)

            if opt_type == 'SGD':
                self._optimizer = optim.SGD(
                    self.model.parameters(),
                    lr=lr,
                    momentum=self._optimizer_cfg.get('momentum', 0.9)
                )
            elif opt_type == 'ADAM':
                self._optimizer = optim.Adam(
                    self.model.parameters(),
                    lr=lr
                )
            self.logger.debug(f"优化器 {opt_type} 创建完成")
        return self._optimizer

    @property
    def criterion(self):
        """延迟加载损失函数"""
        if self._criterion is None:
            if isinstance(self._loss_cfg, str):
                if self._loss_cfg == 'CrossEntropyLoss':
                    self._criterion = nn.CrossEntropyLoss()
                elif self._loss_cfg == 'MSELoss':
                    self._criterion = nn.MSELoss()
            self.logger.debug("损失函数创建完成")
        return self._criterion

    @property
    def train_loader(self):
        """延迟加载数据加载器"""
        if self._train_loader is None:
            dataset = self.dataset  # 从BaseLearner继承
            self._train_loader = DataLoader(
                dataset,
                batch_size=self._bs,
                shuffle=True
            )
            self.logger.debug(f"DataLoader创建完成: batch_size={self._bs}")
        return self._train_loader

    def local_train(
        self,
        train_loader: DataLoader,
        current_round: int = 0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        本地训练函数 - 算法核心实现

        注意：实际框架中使用 async def train()，这里为了简化示例使用同步版本

        Args:
            train_loader: 训练数据加载器
            current_round: 当前通信轮数
            **kwargs: 其他参数

        Returns:
            训练结果字典:
                - loss: 平均损失
                - accuracy: 训练准确率
                - num_samples: 训练样本数
                - [其他自定义指标]
        """
        self.model.train()

        # 统计信息
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        # 算法特有的统计（例如：对比损失、正则化损失等）
        total_ce_loss = 0.0
        total_special_loss = 0.0

        for epoch in range(self.local_epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                data = data.to(self.device)
                target = target.to(self.device)

                self.optimizer.zero_grad()

                # ====== 核心：算法特有的前向传播 ======
                output = self.model(data)

                # 基础分类损失
                ce_loss = self.criterion(output, target)

                # 算法特有的损失项（根据算法添加）
                special_loss = self._compute_special_loss(
                    output=output,
                    data=data,
                    target=target,
                    # 传入需要的上下文信息
                )

                # 总损失
                loss = ce_loss + self.special_param_1 * special_loss

                # 反向传播
                loss.backward()
                self.optimizer.step()

                # 统计
                total_loss += loss.item() * data.size(0)
                total_ce_loss += ce_loss.item() * data.size(0)
                total_special_loss += special_loss.item() * data.size(0)

                pred = output.argmax(dim=1, keepdim=True)
                total_correct += pred.eq(target.view_as(pred)).sum().item()
                total_samples += data.size(0)

        # 返回详细的训练结果
        return {
            'loss': total_loss / total_samples,
            'ce_loss': total_ce_loss / total_samples,
            'special_loss': total_special_loss / total_samples,
            'accuracy': total_correct / total_samples,
            'num_samples': total_samples,
            'current_round': current_round,
        }

    def _compute_special_loss(
        self,
        output: torch.Tensor,
        data: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算算法特有的损失项

        示例（根据算法类型选择）:
            - FedProx: proximal term
            - MOON: contrastive loss
            - EWC: Fisher regularization
            - ...

        Args:
            output: 模型输出
            data: 输入数据
            target: 标签

        Returns:
            损失张量
        """
        # 示例：对比损失（MOON类算法）
        if self.global_model is not None and self.prev_model is not None:
            # 提取特征
            with torch.no_grad():
                global_features = self.global_model.get_features(data)
                prev_features = self.prev_model.get_features(data)

            current_features = self.model.get_features(data)

            # 计算对比损失
            # ...（具体实现）

            return contrastive_loss

        return torch.tensor(0.0, device=self.device)

    # ====== 通信协议实现 ======

    async def get_model(self) -> Dict[str, Any]:
        """
        获取需要上传到服务器的模型数据

        不同算法上传不同内容:
            - FedAvg: 所有模型参数
            - FedPer: 只上传base层参数
            - FedProto: 上传类原型
            - ...

        Returns:
            包含模型参数和元数据的字典:
            {
                'model_type': str,
                'parameters': {'weights': Dict[str, torch.Tensor]},
                'metadata': {...}
            }
        """
        # 示例：上传所有参数
        weights = {
            name: param.detach().cpu().clone()
            for name, param in self.model.named_parameters()
        }

        # 如果只上传部分参数（例如FedPer）
        # weights = {
        #     name: param.detach().cpu().clone()
        #     for name, param in self.model.named_parameters()
        #     if 'base' in name  # 只上传base层
        # }

        return {
            'model_type': self._model_cfg['name'],
            'parameters': {'weights': weights},
            'metadata': {
                'client_id': self.client_id,
                'samples': len(self.dataset),
            }
        }

    async def set_model(self, model_data: Dict[str, Any]) -> bool:
        """
        接收并设置服务器下发的模型参数

        Args:
            model_data: 服务器聚合后的模型数据

        Returns:
            bool: 设置是否成功
        """
        try:
            if 'parameters' in model_data and 'weights' in model_data['parameters']:
                weights = model_data['parameters']['weights']

                # 更新模型参数
                state_dict = self.model.state_dict()
                for name, value in weights.items():
                    if name in state_dict:
                        if not isinstance(value, torch.Tensor):
                            value = torch.from_numpy(value)
                        state_dict[name] = value.to(self.device)

                self.model.load_state_dict(state_dict, strict=True)

                # 如果只更新部分层（例如FedPer）
                # for name, value in weights.items():
                #     if name in state_dict and 'base' in name:
                #         state_dict[name] = value.to(self.device)

                return True
        except Exception as e:
            self.logger.exception(f"Failed to set model: {e}")
        return False

    async def get_local_model(self) -> Dict[str, Any]:
        """BaseLearner抽象方法 - 委托给get_model()"""
        return await self.get_model()

    async def set_local_model(self, model_data: Dict[str, Any]) -> bool:
        """BaseLearner抽象方法 - 委托给set_model()"""
        return await self.set_model(model_data)

    # ====== 评估相关 ======

    async def evaluate(
        self,
        evaluation_params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        本地测试/评估（BaseLearner抽象方法）

        注意：实际框架中使用 async def evaluate()，
        通常使用 self.train_loader 或从 evaluation_params 获取测试数据

        Args:
            evaluation_params: 评估参数，可能包含测试模型等

        Returns:
            评估结果字典
        """
        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        # 用于计算更多指标
        all_preds = []
        all_targets = []

        with torch.no_grad():
            # 通常使用 self.train_loader 进行评估（或专门的test_loader）
            for data, target in self.train_loader:
                data = data.to(self.device)
                target = target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                total_loss += loss.item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                total_correct += pred.eq(target.view_as(pred)).sum().item()
                total_samples += data.size(0)

                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        return {
            'loss': total_loss / total_samples,
            'accuracy': total_correct / total_samples,
            'samples': total_samples,
        }

    # ====== 辅助方法 ======

    def save_checkpoint(self, path: str) -> None:
        """保存检查点"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'special_param_1': self.special_param_1,
            'special_param_2': self.special_param_2,
            # 保存算法特有的状态
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> None:
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.special_param_1 = checkpoint['special_param_1']
        self.special_param_2 = checkpoint['special_param_2']
```

---

### 步骤4: 实现Trainer（训练流程控制器）

**何时需要自定义Trainer？**
- 训练流程与标准FL流程不同（如半监督、元学习、课程学习）
- 需要特殊的通信模式（如异步、分层）
- 需要在训练过程中动态调整策略

#### 4.1 创建Trainer

```python
# fedcl/methods/trainers/[algorithm_name]_trainer.py

"""
[算法名] 专用训练器

用于处理特殊的训练流程，例如:
    - 异步更新
    - 半监督学习
    - 元学习训练循环
"""
from typing import Dict, List, Any
from fedcl.api.decorators import trainer
from fedcl.methods.trainers.generic import GenericFLTrainer


@trainer(
    name='[AlgorithmName]Trainer',
    trainer_type='[federated|continual|unlearning]',
    description='[描述特殊的训练流程]'
)
class [AlgorithmName]Trainer(GenericFLTrainer):
    """
    [算法名] 训练器

    特殊功能:
        - [功能1]
        - [功能2]
    """

    def train_round(
        self,
        round_idx: int,
        selected_clients: List[int],
        **kwargs
    ) -> Dict[str, Any]:
        """
        单轮训练流程

        可以覆盖这个方法来实现特殊的训练逻辑
        """
        # 自定义训练流程
        # ...

        return results
```

---

### 步骤5: 实现Model（网络结构）

**何时需要自定义Model？**
- 模型有特殊结构（如分离的全局层/个性化层）
- 需要返回中间特征（用于对比学习、蒸馏等）
- 需要特殊的初始化方式

#### 5.1 创建Model

```python
# fedcl/methods/models/[model_name].py

"""
[模型名称]

论文: [如果是论文特定模型，注明论文]
用途: [描述模型的用途和特点]
"""
import torch
import torch.nn as nn
from fedcl.api.decorators import model


@model(
    name='[ModelName]',
    model_type='[cnn|mlp|transformer|...]',
    description='[模型描述]',
    input_shape=(3, 32, 32),  # 示例
    num_classes=10,            # 示例
)
class [ModelName](nn.Module):
    """
    [模型名称]

    参数:
        num_classes: 分类数量
        feature_dim: 特征维度（如果需要）

    Forward:
        支持返回中间特征用于特殊算法
    """

    def __init__(
        self,
        num_classes: int = 10,
        feature_dim: int = 128,
    ):
        super().__init__()

        # 如果是个性化算法，明确区分全局层和个性化层
        self.base = nn.Sequential(
            # 全局共享层
            # ...
        )

        self.head = nn.Sequential(
            # 个性化层
            # ...
        )

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ):
        """
        前向传播

        Args:
            x: 输入张量
            return_features: 是否返回中间特征

        Returns:
            如果return_features=False: 分类logits
            如果return_features=True: (logits, features)
        """
        features = self.base(x)
        logits = self.head(features)

        if return_features:
            return logits, features
        return logits

    def get_base_params(self):
        """获取全局层参数（用于部分参数聚合）"""
        return self.base.parameters()

    def get_head_params(self):
        """获取个性化层参数"""
        return self.head.parameters()
```

---

### 步骤6: 实现Aggregator（聚合策略）

**何时需要自定义Aggregator？**
- 聚合策略不是简单的加权平均（如中位数、修剪均值、自适应权重）
- 需要使用服务器端优化器（如FedOpt、FedAdam）
- 需要过滤或调整客户端上传的内容

#### 6.1 创建Aggregator

```python
# fedcl/methods/aggregators/[aggregator_name].py

"""
[聚合器名称]

用于: [描述聚合策略]
论文: [如果来自特定论文]
"""
from typing import Dict, List
import torch
from fedcl.api.decorators import aggregator


@aggregator(
    name='[AggregatorName]',
    aggregator_type='[weighted_avg|median|adaptive|...]',
    description='[描述]'
)
class [AggregatorName]:
    """
    [聚合器名称]

    聚合策略:
        [描述具体的聚合逻辑]

    参数:
        特定聚合器的参数
    """

    def __init__(
        self,
        # 聚合器特有参数
        **kwargs
    ):
        self.kwargs = kwargs

    def aggregate(
        self,
        client_params_list: List[Dict[str, torch.Tensor]],
        client_weights: List[float],
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        执行聚合

        Args:
            client_params_list: 客户端参数列表
            client_weights: 客户端权重（通常基于数据量）

        Returns:
            聚合后的全局参数
        """
        # 实现聚合逻辑
        # ...

        return aggregated_params
```

---

### 步骤7: 实现Dataset（数据集）

**何时需要自定义Dataset？**
- 使用新的数据集（框架中不存在）
- 需要特殊的数据预处理
- 需要特殊的数据分区策略

#### 7.1 创建Dataset

```python
# fedcl/methods/datasets/[dataset_name].py

"""
[数据集名称]

数据来源: [URL或描述]
任务类型: [分类/回归/...]
"""
from typing import Dict, Any
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from fedcl.api.decorators import dataset
from fedcl.methods.datasets.base import FederatedDataset


@dataset(
    name='[DatasetName]',
    dataset_type='[image_classification|text_classification|...]',
    num_classes=10,
    input_shape=(3, 32, 32),
    download_url='[如果可以自动下载]'
)
class [DatasetName]FederatedDataset(FederatedDataset):
    """
    [数据集名称]

    统计信息:
        - 训练集大小: [数量]
        - 测试集大小: [数量]
        - 类别数: [数量]
        - 输入形状: [形状]

    数据分区:
        支持的Non-IID设置:
            - iid: 独立同分布
            - dirichlet: Dirichlet分布
            - pathological: 病理性Non-IID
    """

    def __init__(
        self,
        root: str = './data',
        train: bool = True,
        download: bool = True,
        **kwargs
    ):
        super().__init__(root, train, download)

        # 数据转换
        if train:
            transform = transforms.Compose([
                # 训练集数据增强
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((...), (...)),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((...), (...)),
            ])

        # 加载数据
        self.dataset = self._load_dataset(root, train, transform, download)

        # 设置属性
        self.num_classes = 10
        self.input_shape = (3, 32, 32)

    def _load_dataset(self, root, train, transform, download):
        """加载原始数据集"""
        # 实现数据加载逻辑
        # ...
        pass

    def get_statistics(self) -> Dict[str, Any]:
        """返回数据集统计信息"""
        return {
            'dataset_name': '[DatasetName]',
            'num_samples': len(self.dataset),
            'num_classes': self.num_classes,
            'input_shape': self.input_shape,
            'train': self.train,
        }
```

---

### 步骤8: 实现Metrics（评估指标）

**何时需要自定义Metric？**
- 需要算法特定的评估指标（如遗忘率、公平性）
- 需要多任务评估
- 需要复杂的统计分析

#### 8.1 创建Metric

```python
# fedcl/methods/metrics/[metric_name].py

"""
[指标名称]

用于: [描述指标的用途]
计算方式: [描述计算方法]
"""
from typing import List, Dict, Any
import numpy as np
from fedcl.api.decorators import metric


@metric(
    name='[MetricName]',
    metric_type='[classification|fairness|forgetting|...]',
    description='[描述]'
)
class [MetricName]:
    """
    [指标名称]

    计算方法:
        [详细描述计算公式]

    用例:
        >>> metric = [MetricName]()
        >>> score = metric.compute(predictions, targets)
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def compute(
        self,
        predictions: List,
        targets: List,
        **kwargs
    ) -> Dict[str, float]:
        """
        计算指标

        Args:
            predictions: 预测结果
            targets: 真实标签

        Returns:
            指标字典
        """
        # 实现指标计算
        # ...

        return {
            '[metric_name]': score
        }
```

---

### 步骤9: 注册所有组件

#### 9.1 更新 `__init__.py`

```python
# fedcl/methods/learners/__init__.py
from fedcl.methods.learners.fl.[algorithm_name] import [AlgorithmName]Learner

__all__ = [
    # ... 现有的
    '[AlgorithmName]Learner',
]

# fedcl/methods/models/__init__.py
from fedcl.methods.models.[model_name] import [ModelName]

__all__ = [
    # ... 现有的
    '[ModelName]',
]

# 类似地更新其他组件的 __init__.py
```

---

### 步骤10: 创建实验配置

#### 10.1 目录结构
```
configs/distributed/experiments/[algorithm_name]/
├── server.yaml          # 服务器配置
├── client_0.yaml        # 客户端0配置
├── client_1.yaml        # 客户端1配置
├── ...
└── client_9.yaml        # 客户端9配置
```

#### 10.2 服务器配置模板

```yaml
# configs/distributed/experiments/[algorithm_name]/server.yaml

extends: "../../base/server_base.yaml"

# 联邦配置
federation:
  aggregation:
    method: "[AggregatorName]"  # 使用注册的聚合器名称
    params:
      # 聚合器特有参数
      adaptive_weight: true

# 训练配置
training:
  rounds: 100              # 通信轮数（从论文获取）
  sample_ratio: 1.0        # 每轮参与率

  # 如果需要自定义训练器
  trainer:
    name: "[AlgorithmName]Trainer"
    params:
      # 训练器特有参数

# 日志和检查点
logging:
  log_level: "INFO"
  save_checkpoints: true
  checkpoint_freq: 10
```

#### 10.3 客户端配置模板

```yaml
# configs/distributed/experiments/[algorithm_name]/client_0.yaml

extends: "../../base/client_base.yaml"

node_id: "client_0"

# 训练配置
training:
  learner:
    name: "[namespace].[算法名]"  # 例如: "fl.FedAvg", "cl.TARGET", "ul.SISA"
    params:
      client_index: 0

      # 标准参数
      batch_size: 32
      local_epochs: 5
      learning_rate: 0.01
      momentum: 0.9
      weight_decay: 0.0001

      # 算法特有参数（从论文获取）
      special_param_1: 1.0
      special_param_2: 10

  # 模型配置
  model:
    name: "[ModelName]"
    params:
      num_classes: 10
      # 模型特有参数

  # 数据集配置
  dataset:
    name: "CIFAR10"
    partition:
      method: "dirichlet"   # iid | dirichlet | pathological
      num_clients: 10
      alpha: 0.5            # Dirichlet参数
      seed: 42

# 评估配置
evaluation:
  metrics:
    - name: "Accuracy"
    - name: "[CustomMetricName]"  # 如果有自定义指标
      params:
        # 指标参数
```

---

### 步骤11: 编写论文复现脚本 ⭐

**重要**: 所有复现脚本统一放在 `papers/` 目录下！

#### 11.1 创建论文目录

```bash
papers/
└── [method name]/
    ├── reproduce.py              # 复现脚本
    ├── configs/                  # 实验配置（可选，如果不放在configs/distributed/）
    │   ├── server.yaml
    │   └── client_*.yaml
    ├── README.md                 # 论文信息和复现说明
    └── results/                  # 结果存储
        ├── results.csv
        └── figures/

# 示例:
papers/fedavg_mcmahan2017/
papers/fedprox_li2020/
papers/moon_li2021/
papers/fedper_arivazhagan2019/
```

#### 11.2 复现脚本模板

```python
# papers/[method name]/reproduce.py

"""
论文复现: [论文标题]

论文信息:
    标题: [完整标题]
    作者: [作者列表]
    会议/期刊: [venue, year]
    链接: [arXiv/DOI链接]

实验设置:
    数据集: [列表]
    Non-IID: [设置列表]
    算法: [算法名]
    客户端数: [数量]
    通信轮数: [数量]
    重复次数: [次数]

预期结果:
    [从论文Table/Figure中提取的结果]

运行命令:
    python papers/[paper_name]_[author]_[year]/reproduce.py \\
        --dataset CIFAR10 \\
        --noniid dirichlet \\
        --alpha 0.5

数据库:
    所有实验结果统一写入: experiments/unified_results.db
    表结构:
        - experiment_name: 实验名称
        - algorithm_type: fl | cl | ul
        - algorithm_name: 具体算法名
        - dataset: 数据集名
        - accuracy: 准确率
        - ...
"""
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import pandas as pd
from typing import List, Dict

from examples.smart_batch_runner import SmartBatchRunner, ExperimentConfig


# ==================== 论文配置 ====================

PAPER_INFO = {
    'title': '[论文标题]',
    'authors': '[作者]',
    'venue': '[会议/期刊, 年份]',
    'arxiv': '[链接]',
    'algorithm_type': 'fl',  # 命名空间: fl | cl | ul
    'algorithm_name': '[AlgorithmName]',
}

# 实验超参数（从论文获取）
HYPERPARAMETERS = {
    'learning_rate': 0.01,
    'batch_size': 32,
    'local_epochs': 5,
    'communication_rounds': 100,
    'num_clients': 10,
    'sample_ratio': 1.0,

    # 算法特有参数
    'special_param_1': 1.0,
    'special_param_2': 10,
}

# 实验设置（复现论文的Table/Figure）
EXPERIMENTS = {
    'table1': {
        'description': '复现论文Table 1: 不同数据集上的性能对比',
        'datasets': ['MNIST', 'FMNIST', 'CIFAR10', 'CIFAR100'],
        'noniid_settings': [
            {'method': 'iid', 'name': 'IID'},
        ],
    },
    'table2': {
        'description': '复现论文Table 2: 不同Non-IID程度的影响',
        'datasets': ['CIFAR10'],
        'noniid_settings': [
            {'method': 'dirichlet', 'alpha': 0.1, 'name': 'Dir(0.1)'},
            {'method': 'dirichlet', 'alpha': 0.5, 'name': 'Dir(0.5)'},
            {'method': 'dirichlet', 'alpha': 1.0, 'name': 'Dir(1.0)'},
        ],
    },
}


# ==================== 实验配置生成 ====================

def create_experiment_configs(
    experiment_name: str = 'table1',
    datasets: List[str] = None,
    noniid_settings: List[Dict] = None,
) -> List[ExperimentConfig]:
    """
    创建实验配置列表

    Args:
        experiment_name: 实验名称（对应论文的Table/Figure）
        datasets: 数据集列表（如果为None，使用EXPERIMENTS中的配置）
        noniid_settings: Non-IID设置列表

    Returns:
        实验配置列表
    """
    if experiment_name in EXPERIMENTS:
        exp_config = EXPERIMENTS[experiment_name]
        datasets = datasets or exp_config['datasets']
        noniid_settings = noniid_settings or exp_config['noniid_settings']
        print(f"\n复现实验: {exp_config['description']}")

    configs = []
    algo_name = PAPER_INFO['algorithm_name']

    for dataset in datasets:
        for noniid in noniid_settings:
            exp_name = f"{dataset}_{noniid['name']}_{algo_name}"

            # 构建配置覆盖
            config_overrides = {
                # Learner配置
                'training.learner.name': f"{PAPER_INFO['algorithm_type']}.{algo_name}",  # 例如: "fl.FedAvg"
                'training.learner.params.learning_rate': HYPERPARAMETERS['learning_rate'],
                'training.learner.params.batch_size': HYPERPARAMETERS['batch_size'],
                'training.learner.params.local_epochs': HYPERPARAMETERS['local_epochs'],
                'training.learner.params.special_param_1': HYPERPARAMETERS['special_param_1'],
                'training.learner.params.special_param_2': HYPERPARAMETERS['special_param_2'],

                # Dataset配置
                'training.dataset.name': dataset,
                'training.dataset.partition.method': noniid['method'],
                'training.dataset.partition.num_clients': HYPERPARAMETERS['num_clients'],

                # Server配置
                'training.rounds': HYPERPARAMETERS['communication_rounds'],
                'training.sample_ratio': HYPERPARAMETERS['sample_ratio'],
            }

            # 添加Non-IID特定参数
            if 'alpha' in noniid:
                config_overrides['training.dataset.partition.alpha'] = noniid['alpha']
            if 'num_classes' in noniid:
                config_overrides['training.dataset.partition.num_classes'] = noniid['num_classes']

            configs.append(ExperimentConfig(
                name=exp_name,
                dataset=dataset,
                algorithm=algo_name,
                algorithm_type=PAPER_INFO['algorithm_type'],  # fl | cl | ul
                noniid_type=noniid['name'],
                config_overrides=config_overrides,
                metadata={
                    'paper': PAPER_INFO['title'],
                    'experiment': experiment_name,
                }
            ))

    return configs


# ==================== 结果分析 ====================

def analyze_results(results_csv: str, paper_results: Dict = None):
    """
    分析实验结果并与论文对比

    Args:
        results_csv: 结果CSV文件路径
        paper_results: 论文中报告的结果（可选）
    """
    df = pd.read_csv(results_csv)

    print("\n" + "="*80)
    print("实验结果汇总")
    print("="*80)

    # 按数据集和Non-IID分组统计
    summary = df.groupby(['dataset', 'noniid_type']).agg({
        'accuracy': ['mean', 'std', 'min', 'max'],
        'loss': ['mean', 'std'],
    }).round(4)

    print(summary)

    # 如果提供了论文结果，进行对比
    if paper_results:
        print("\n" + "="*80)
        print("与论文结果对比")
        print("="*80)

        comparison = []
        for (dataset, noniid), paper_acc in paper_results.items():
            our_acc = df[
                (df['dataset'] == dataset) &
                (df['noniid_type'] == noniid)
            ]['accuracy'].mean()

            gap = our_acc - paper_acc

            comparison.append({
                'Dataset': dataset,
                'Non-IID': noniid,
                'Paper': f"{paper_acc:.2%}",
                'Ours': f"{our_acc:.2%}",
                'Gap': f"{gap:+.2%}"
            })

        comparison_df = pd.DataFrame(comparison)
        print(comparison_df.to_string(index=False))

        # 分析差距
        avg_gap = abs(comparison_df['Gap'].str.rstrip('%').astype(float).mean())
        print(f"\n平均差距: {avg_gap:.2f}%")

        if avg_gap < 2.0:
            print("✅ 复现成功！结果与论文基本一致（差距<2%）")
        elif avg_gap < 5.0:
            print("⚠️  结果可接受，但存在一定差距（2-5%）")
        else:
            print("❌ 结果差距较大（>5%），需要排查原因")


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(
        description=f"复现论文: {PAPER_INFO['title']}"
    )

    # 实验选择
    parser.add_argument(
        '--experiment',
        type=str,
        default='table1',
        choices=list(EXPERIMENTS.keys()),
        help='选择要复现的实验（对应论文的Table/Figure）'
    )

    # 运行选项
    parser.add_argument(
        '--mode',
        type=str,
        default='full',
        choices=['test', 'full', 'resume'],
        help='运行模式: test(测试单个), full(完整运行), resume(断点续跑)'
    )

    parser.add_argument(
        '--repetitions',
        type=int,
        default=3,
        help='每个实验重复次数（默认3次）'
    )

    parser.add_argument(
        '--max-concurrent',
        type=int,
        default=5,
        help='最大并发实验数'
    )

    # 数据库配置
    parser.add_argument(
        '--db-path',
        type=str,
        default='experiments/unified_results.db',
        help='统一的实验结果数据库路径'
    )

    # 结果分析
    parser.add_argument(
        '--analyze-only',
        action='store_true',
        help='只分析已有结果，不运行实验'
    )

    args = parser.parse_args()

    # 设置路径
    paper_dir = Path(__file__).parent
    project_root = paper_dir.parent.parent
    db_path = project_root / args.db_path
    results_csv = paper_dir / 'results' / f'{args.experiment}_results.csv'

    # 只分析结果
    if args.analyze_only:
        if results_csv.exists():
            analyze_results(str(results_csv))
        else:
            print(f"结果文件不存在: {results_csv}")
        return

    # 创建实验配置
    exp_configs = create_experiment_configs(experiment_name=args.experiment)

    print("\n" + "="*80)
    print(f"论文: {PAPER_INFO['title']}")
    print(f"算法: {PAPER_INFO['algorithm_type']}.{PAPER_INFO['algorithm_name']}")
    print(f"实验: {args.experiment}")
    print("="*80)
    print(f"总实验配置数: {len(exp_configs)}")
    print(f"每个配置重复: {args.repetitions} 次")
    print(f"总运行数: {len(exp_configs) * args.repetitions}")
    print(f"数据库: {db_path}")
    print("="*80)

    # 测试模式：只运行第一个配置
    if args.mode == 'test':
        print("\n🧪 测试模式：只运行第一个配置")
        exp_configs = exp_configs[:1]
        args.repetitions = 1

    # 配置基础目录
    config_base_dir = str(project_root / 'configs' / 'distributed' / 'experiments' /
                         f"{PAPER_INFO['algorithm_name'].lower()}")

    # 创建智能调度器
    runner = SmartBatchRunner(
        config_base_dir=config_base_dir,
        experiments=exp_configs,
        max_repetitions=args.repetitions,
        db_path=str(db_path),
        log_dir=str(paper_dir / 'logs'),
        enable_gpu_scheduling=True,
        max_concurrent_experiments=args.max_concurrent,

        # 数据集特定并发限制
        dataset_concurrent_limits={
            'MNIST': 10,
            'FMNIST': 10,
            'CIFAR10': 5,
            'CIFAR100': 3,
        }
    )

    # 运行实验
    print("\n🚀 开始运行实验...")
    runner.run_multiprocess()

    # 导出结果
    print("\n📊 导出结果...")
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    runner.export_results_to_csv(str(results_csv))

    # 分析结果
    print("\n📈 分析结果...")

    # 如果有论文报告的结果，进行对比
    # paper_results = {
    #     ('CIFAR10', 'IID'): 0.85,
    #     ('CIFAR10', 'Dir(0.5)'): 0.78,
    #     # ...
    # }
    # analyze_results(str(results_csv), paper_results)

    analyze_results(str(results_csv))

    print(f"\n✅ 完成！结果已保存到: {results_csv}")


if __name__ == '__main__':
    main()
```

#### 11.3 README.md模板

```markdown
# [论文标题] - 复现

## 论文信息

- **标题**: [完整标题]
- **作者**: [作者列表]
- **会议/期刊**: [venue, year]
- **论文链接**: [arXiv/DOI]
- **开源代码**: [GitHub链接（如有）]

## 算法简介

[2-3段描述算法的核心思想和创新点]

## 实验设置

### Table 1: [描述]

| 数据集 | Non-IID | 论文结果 | 复现结果 | 差距 |
|--------|---------|----------|----------|------|
| MNIST  | IID     | 98.5%    | 98.3%    | -0.2%|
| CIFAR10| Dir(0.5)| 78.2%    | 78.0%    | -0.2%|

### Table 2: [描述]

...

## 运行方法

### 复现所有实验

```bash
# 复现Table 1
python papers/[paper_name]_[author]_[year]/reproduce.py \
    --experiment table1 \
    --repetitions 3

# 复现Table 2
python papers/[paper_name]_[author]_[year]/reproduce.py \
    --experiment table2 \
    --repetitions 3
```

### 测试单个实验

```bash
python papers/[paper_name]_[author]_[year]/reproduce.py \
    --mode test \
    --experiment table1
```

### 只分析已有结果

```bash
python papers/[paper_name]_[author]_[year]/reproduce.py \
    --analyze-only \
    --experiment table1
```

## 结果分析

[分析复现结果与论文的对比]

### 成功复现 ✅

- [列出成功复现的实验]

### 存在差距 ⚠️

- [列出有差距的实验]
- 可能原因: [分析]

## 依赖和环境

```bash
# Python版本
python >= 3.8

# 依赖包
torch >= 1.10
torchvision
numpy
pandas
pyyaml
```

## 文件结构

```
papers/[paper_name]_[author]_[year]/
├── reproduce.py          # 复现脚本
├── README.md             # 本文件
├── configs/              # 实验配置（可选）
├── logs/                 # 运行日志
└── results/              # 结果文件
    ├── table1_results.csv
    ├── table2_results.csv
    └── figures/
```

## 引用

如果使用本复现代码，请引用原论文:

```bibtex
@inproceedings{...}
```
```

---

## 统一数据库管理 ⭐

### 数据库设计

所有算法类型（fl.xxx, cl.xxx, ul.xxx）的实验结果统一存储在一个数据库中：

```sql
-- 数据库: experiments/unified_results.db

-- 实验配置表
CREATE TABLE experiments (
    config_hash TEXT PRIMARY KEY,
    exp_name TEXT NOT NULL,
    algorithm_type TEXT NOT NULL,  -- 'fl' | 'cl' | 'ul'
    algorithm_name TEXT NOT NULL,   -- 具体算法名
    dataset TEXT NOT NULL,
    noniid_type TEXT,
    config_json TEXT,
    paper_name TEXT,                -- 来自哪篇论文
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 实验运行表
CREATE TABLE experiment_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    config_hash TEXT NOT NULL,
    run_number INTEGER NOT NULL,
    status TEXT NOT NULL,  -- 'pending' | 'running' | 'success' | 'failed'

    -- 结果指标
    accuracy REAL,
    loss REAL,
    rounds INTEGER,
    duration_sec REAL,

    -- 算法特定指标
    forgetting_rate REAL,      -- 遗忘学习专用
    fairness_score REAL,       -- 公平性指标
    custom_metric_1 REAL,
    custom_metric_2 REAL,

    -- 元信息
    error_msg TEXT,
    log_file TEXT,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,

    FOREIGN KEY (config_hash) REFERENCES experiments(config_hash),
    UNIQUE(config_hash, run_number)
);

-- 索引
CREATE INDEX idx_algorithm_type ON experiments(algorithm_type);
CREATE INDEX idx_algorithm_name ON experiments(algorithm_name);
CREATE INDEX idx_dataset ON experiments(dataset);
CREATE INDEX idx_paper ON experiments(paper_name);
CREATE INDEX idx_status ON experiment_runs(status);
```

### 查询示例

```python
import sqlite3
import pandas as pd

# 连接数据库
conn = sqlite3.connect('experiments/unified_results.db')

# 查询所有联邦学习算法的结果
df = pd.read_sql_query("""
    SELECT
        e.algorithm_name,
        e.dataset,
        e.noniid_type,
        AVG(r.accuracy) as avg_accuracy,
        AVG(r.loss) as avg_loss
    FROM experiments e
    JOIN experiment_runs r ON e.config_hash = r.config_hash
    WHERE e.algorithm_type = 'fl' AND r.status = 'success'
    GROUP BY e.algorithm_name, e.dataset, e.noniid_type
""", conn)

print(df)

# 查询特定论文的所有实验
df = pd.read_sql_query("""
    SELECT * FROM experiments e
    JOIN experiment_runs r ON e.config_hash = r.config_hash
    WHERE e.paper_name = 'fedavg_mcmahan2017'
    AND r.status = 'success'
""", conn)

# 对比不同算法类型
df = pd.read_sql_query("""
    SELECT
        algorithm_type,
        algorithm_name,
        AVG(accuracy) as avg_accuracy
    FROM experiments e
    JOIN experiment_runs r ON e.config_hash = r.config_hash
    WHERE dataset = 'CIFAR10' AND status = 'success'
    GROUP BY algorithm_type, algorithm_name
    ORDER BY avg_accuracy DESC
""", conn)
```

---

## 检查清单 ✅

在提交集成前，确认以下所有项：

### 代码实现
- [ ] Learner实现完整，核心算法逻辑正确
- [ ] 所有需要的组件都已实现（Trainer、Model、Aggregator、Dataset、Metric）
- [ ] 装饰器正确使用，所有组件已注册
- [ ] 通信协议正确实现（get_local_model, set_local_model, get_model, set_model）
- [ ] 代码有充分的文档字符串和注释
- [ ] 变量命名清晰，符合Python规范

### 配置文件
- [ ] 实验配置与论文完全一致
- [ ] 所有超参数可以通过YAML配置
- [ ] 配置文件结构清晰，有注释说明

### 复现脚本
- [ ] 复现脚本放在 `papers/[paper_name]_[author]_[year]/`
- [ ] README.md详细记录论文信息和运行方法
- [ ] 脚本支持不同运行模式（test、full、resume）
- [ ] 结果自动分析和对比

### 测试验证
- [ ] 至少在1个数据集上测试通过
- [ ] 结果与论文差距在合理范围（±2%）
- [ ] 代码在不同环境下可运行
- [ ] 无内存泄漏或GPU显存溢出

### 数据库
- [ ] 实验结果正确写入统一数据库
- [ ] algorithm_type字段正确（fl/cl/ul）
- [ ] 可以通过SQL查询和分析结果

### 文档
- [ ] README.md完整
- [ ] 所有代码有文档字符串
- [ ] 特殊设计有说明注释

---

## 常见问题 FAQ

### Q1: 如何选择正确的基类？

**A**: 所有Learner都继承自 `BaseLearner`：

```python
from fedcl.learner.base_learner import BaseLearner
from fedcl.methods.learners._decorators import learner

# 联邦学习算法
@learner(namespace='fl', name='MyFLAlgo')
class MyFLLearner(BaseLearner):
    pass

# 持续学习算法
@learner(namespace='cl', name='MyCLAlgo')
class MyCLLearner(BaseLearner):
    pass

# 遗忘学习算法
@learner(namespace='ul', name='MyULAlgo')
class MyULLearner(BaseLearner):
    pass
```

**核心方法需要实现**:
- `async def train()`: 本地训练逻辑（异步方法）
- `async def evaluate()`: 评估逻辑（异步方法）
- `async def get_local_model()`: 返回需要上传的模型参数（异步方法）
- `async def set_local_model()`: 接收服务器下发的模型参数（异步方法）

**注意**：框架使用异步编程（async/await），所有核心方法都是异步的。

### Q2: 装饰器参数有什么作用？

**A**: Learner装饰器采用简洁的三参数设计：

```python
from fedcl.methods.learners._decorators import learner

@learner(
    namespace='fl',              # 命名空间: fl(联邦学习), cl(持续学习), ul(遗忘学习)
    name='FedAvg',              # 方法名，用于配置文件引用
    description='FedAvg: ...'   # 可选描述，用于文档
)
class FedAvgLearner(BaseLearner):
    pass
```

**注册结果**:
- `namespace='fl', name='FedAvg'` → 注册为 `'fl.FedAvg'`
- `namespace='cl', name='TARGET'` → 注册为 `'cl.TARGET'`
- `namespace='ul', name='SISA'` → 注册为 `'ul.SISA'`

**在配置文件中使用**:
```yaml
training:
  learner:
    name: "fl.FedAvg"  # 或 "cl.TARGET", "ul.SISA"
```

### Q3: 如何处理算法需要的特殊模型结构？

**A**: 创建自定义Model并用装饰器注册：

```python
@model(name='MySpecialNet')
class MySpecialNet(nn.Module):
    def __init__(self, ...):
        # 定义特殊结构
        pass
```

然后在配置文件中指定：

```yaml
training:
  model:
    name: "MySpecialNet"
```

### Q4: 通信协议中上传/下载什么？

**A**: BaseLearner定义了两个抽象方法，子类需要实现：

```python
# 抽象方法（BaseLearner中定义）
async def get_local_model(self) -> Dict[str, Any]:
    """返回要上传到服务器的模型数据"""
    pass

async def set_local_model(self, model_data: Dict[str, Any]) -> bool:
    """接收服务器下发的模型数据"""
    pass
```

**实际实现模式**（推荐）：
```python
class MyLearner(BaseLearner):
    # 实现具体逻辑
    async def get_model(self) -> Dict[str, Any]:
        weights = {...}  # 获取需要上传的参数
        return {
            'model_type': 'MyModel',
            'parameters': {'weights': weights},
            'metadata': {'client_id': self.client_id, 'samples': 1000}
        }

    async def set_model(self, model_data: Dict[str, Any]) -> bool:
        # 更新本地模型
        weights = model_data['parameters']['weights']
        # ...
        return True

    # 委托给具体实现
    async def get_local_model(self):
        return await self.get_model()

    async def set_local_model(self, model_data):
        return await self.set_model(model_data)
```

**不同算法的上传策略**：
```python
# 标准FL: 上传全部参数
async def get_model(self):
    weights = {name: param for name, param in self.model.named_parameters()}
    return {'parameters': {'weights': weights}, ...}

# 个性化FL (FedPer): 只上传共享层
async def get_model(self):
    weights = {name: param for name, param in self.model.named_parameters()
               if 'base' in name}
    return {'parameters': {'weights': weights}, ...}

# 原型FL (FedProto): 在metadata中上传类原型
async def get_model(self):
    return {
        'parameters': {'weights': {}},
        'metadata': {'prototypes': self.compute_prototypes()}
    }
```

### Q5: 如何调试实验？

**A**: 使用测试模式：

```bash
# 只运行一个配置，快速测试
python papers/xxx/reproduce.py --mode test --experiment table1

# 查看详细日志
tail -f papers/xxx/logs/experiment.log

# 查看数据库
sqlite3 experiments/unified_results.db "SELECT * FROM experiment_runs ORDER BY run_id DESC LIMIT 10"
```

### Q6: 结果与论文差距大怎么办？

**A**: 系统化排查：

1. **检查超参数**: 确认所有参数与论文一致
2. **检查数据处理**: 归一化、数据增强
3. **检查随机种子**: 固定种子增加可复现性
4. **检查优化器**: SGD vs Adam，学习率调度
5. **检查评估方式**: 测试集、评估指标
6. **查看日志**: 分析训练曲线，查找异常
7. **参考开源代码**: 如果有官方实现，对比细节

### Q7: 如何添加自定义评估指标？

**A**: 创建Metric并注册：

```python
@metric(name='MyMetric')
class MyMetric:
    def compute(self, predictions, targets, **kwargs):
        # 计算指标
        return {'my_metric': score}
```

在Learner中使用：

```python
def local_test(self, test_loader, **kwargs):
    # ... 测试逻辑

    # 使用自定义指标
    from fedcl.methods.metrics.my_metric import MyMetric
    metric = MyMetric()
    custom_scores = metric.compute(all_preds, all_targets)

    return {
        'accuracy': accuracy,
        **custom_scores
    }
```

---

## 进阶技巧

### 1. 支持多GPU训练

```python
class MyLearner(FedAvgLearner):
    def __init__(self, model, device, **kwargs):
        super().__init__(model, device, **kwargs)

        # 自动使用DataParallel
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
```

### 2. 动态学习率调度

```python
class MyLearner(FedAvgLearner):
    def __init__(self, model, device, **kwargs):
        super().__init__(model, device, **kwargs)

        # 添加学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100
        )

    def local_train(self, train_loader, current_round, **kwargs):
        results = super().local_train(train_loader, **kwargs)

        # 每轮调整学习率
        self.scheduler.step()

        return results
```

### 3. 早停机制

```python
class MyLearner(FedAvgLearner):
    def __init__(self, model, device, patience=10, **kwargs):
        super().__init__(model, device, **kwargs)
        self.patience = patience
        self.best_loss = float('inf')
        self.patience_counter = 0

    def local_train(self, train_loader, **kwargs):
        results = super().local_train(train_loader, **kwargs)

        # 早停检查
        if results['loss'] < self.best_loss:
            self.best_loss = results['loss']
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        if self.patience_counter >= self.patience:
            results['early_stopped'] = True

        return results
```

---

## 参考资源

### 框架文档
- `README.md` - 框架总览
- `ARCHITECTURE.md` - 架构设计
- `fedcl/api/decorators.py` - 装饰器源码和文档

### 示例代码
- `fedcl/methods/learners/fl/fedper.py` - 个性化联邦学习示例
  ```python
  @learner('fl', 'FedPer', description='FedPer: Federated Learning with Personalization Layers')
  class FedPerLearner(BaseLearner):
      pass
  ```
- `fedcl/methods/learners/fl/moon.py` - 对比学习示例
  ```python
  @learner('fl', 'MOON', description='MOON: Model-Contrastive Federated Learning')
  class MOONLearner(BaseLearner):
      pass
  ```
- `fedcl/methods/learners/cl/target.py` - 持续学习示例
  ```python
  @learner('cl', 'TARGET', description='TARGET: Federated Class-Continual Learning via Exemplar-Free Distillation (ICCV 2023)')
  class TARGETLearner(BaseLearner):
      pass
  ```

### 外部资源
- [FedML](https://github.com/FedML-AI/FedML) - 参考实现
- [Flower](https://github.com/adap/flower) - 另一个FL框架
- [PFL-Non-IID](https://github.com/TsingZ0/PFL-Non-IID) - 个性化FL算法集合

---

## 总结

集成新算法到本框架需要：

1. ✅ **理解装饰器系统** - 所有组件通过装饰器注册
2. ✅ **实现完整组件** - Learner + Trainer + Model + Aggregator + Dataset + Metric
3. ✅ **设计通信协议** - 明确上传/下载内容
4. ✅ **编写复现脚本** - 放在 `papers/` 目录
5. ✅ **统一数据库管理** - 所有结果写入 `experiments/unified_results.db`
6. ✅ **充分测试验证** - 确保结果正确

**记住**: 框架的核心是**装饰器驱动的组件注册**和**统一的实验管理**。所有算法（fl/cl/ul）共享相同的基础设施，但可以有各自的特殊实现。

祝你成功复现论文结果！🚀
