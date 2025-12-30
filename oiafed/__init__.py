"""
Federation Framework - 联邦学习框架

一个完整、灵活、可扩展的联邦学习框架

特点：
1. 完整的系统容器（FederatedSystem）
2. Callback 和 Tracker 支持
3. 装饰器自动继承机制
4. 4种场景抽象类（标准FL/持续学习/遗忘学习/纵向FL）
5. 直接复用 PyTorch 和 Transformers
6. 配置驱动执行

使用方式：
    # 方式1：FederationRunner 类（推荐）
    from federation import FederationRunner

    # 单配置文件
    runner = FederationRunner("configs/trainer.yaml")
    result = runner.run_sync()

    # 配置文件夹（自动加载所有 yaml 文件）
    runner = FederationRunner("configs/experiment/")
    result = runner.run_sync()

    # 方式2：配置文件驱动函数
    from federation import run_from_config_sync
    result = run_from_config_sync("config.yaml")

    # 方式3：编程方式
    from federation import FederatedSystem
    system = FederatedSystem(config)
    await system.initialize()
    await system.run()

    # 方式4：CLI
    $ federation run config.yaml
    $ federation run configs/experiment/  # 文件夹
    $ federation list
    $ federation init -n my_experiment

设计原则：
    - 直接使用 PyTorch nn.Module，不重复定义模型基类
    - 直接使用 torch.utils.data.Dataset，不重复定义数据集基类
    - 支持 Hugging Face Transformers 和 Datasets
    - 配置驱动，统一初始化
"""
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python" # 避免 grpcio 与 protobuf 版本冲突问题

__version__ = "0.2.2"

# 核心组件
from .core import (
    # 类型
    TrainResult,
    EvalResult,
    ClientUpdate,
    ClientInfo,
    FitConfig,
    EvalConfig,
    FitStatus,

    # 抽象基类
    Learner,
    Trainer,
    Aggregator,

    # 系统容器
    FederatedSystem,
)

# 模型工具函数
from .core.model import (
    get_model_weights,
    set_model_weights,
    clone_model_weights,
    get_num_parameters,
    get_transformers_model_weights,
    set_transformers_model_weights,
)

# Proxy 代理
from .proxy import (
    PeerProxy,
    ProxyCollection,
)

# 注册系统
from .registry import (
    Registry,
    registry,
    register,
    get_component,
    create_component,
    list_components,
)

# Callback 系统
from .callback import (
    Callback,
    CallbackManager,
    ModelCheckpoint,
    EarlyStopping,
    SyncCallback,
    TrackerSyncCallback,
)

# Tracker 系统
from .tracker import (
    Tracker,
    MLflowTracker,
    CompositeTracker,
)
# 导入以触发注册
from .callback import builtin as callbacks
from .callback import tracker_sync  # 新增
from .tracker import mlflow_tracker
# 数据
from .data import (
    DataProvider,
    DataManager,
    Partitioner,
    IIDPartitioner,
    DirichletPartitioner,
)

# 数据集工具
from .data.datasets import (
    DataProvider as TorchDataProvider,
    HuggingFaceDataProvider,
    create_dataloader,
    create_hf_dataloader,
    get_dataset_info,
)

# 配置
from .config import (
    FederationConfig,
    NodeConfig,
    TransportConfig,
    TrackerConfig,
    TrackerBackendConfig,
    MLflowConfig,
    WandbConfig,
    TensorBoardConfig,
    LogConfig,
    load_config,
    load_node_config,
    load_config_from_dict,
)

# Runner
from .runner import (
    FederationRunner
)

# 导入内置组件（触发注册）
from . import methods


__all__ = [
    # 版本
    "__version__",

    # 核心类型
    "TrainResult",
    "EvalResult",
    "ClientUpdate",
    "ClientInfo",
    "FitConfig",
    "EvalConfig",
    "FitStatus",

    # 抽象基类
    "Learner",
    "Trainer",
    "Aggregator",

    # 系统容器
    "FederatedSystem",

    # 模型工具函数
    "get_model_weights",
    "set_model_weights",
    "clone_model_weights",
    "get_num_parameters",
    "get_transformers_model_weights",
    "set_transformers_model_weights",

    # Proxy 代理
    "PeerProxy",
    "ProxyCollection",

    # 注册系统
    "Registry",
    "registry",
    "register",
    "get_component",
    "create_component",
    "list_components",

    # Callback 系统
    "Callback",
    "CallbackManager",
    "ModelCheckpoint",
    "EarlyStopping",
    "SyncCallback",
    "TrackerSyncCallback",

    # Tracker 系统
    "Tracker",
    "LoguruTracker",
    "MLflowTracker",
    "CompositeTracker",

    # 数据
    "DataProvider",
    "TorchDataProvider",
    "HuggingFaceDataProvider",
    "DataManager",
    "Partitioner",
    "IIDPartitioner",
    "DirichletPartitioner",
    "create_dataloader",
    "create_hf_dataloader",
    "get_dataset_info",

    # Runner
    "FederationRunner",
    
    # 模块
    "builtin",
]
