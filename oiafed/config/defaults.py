"""
配置默认值定义

所有默认值集中管理，便于维护和文档化。
修改默认值只需修改此文件。
"""

# ==================== 网络配置 ====================
DEFAULT_TRAINER_PORT = 50051
DEFAULT_LEARNER_BASE_PORT = 50052
DEFAULT_HOST = "0.0.0.0"
DEFAULT_LOCALHOST = "localhost"
DEFAULT_MAX_MESSAGE_SIZE = 1000 * 1024 * 1024  # 100MB

# ==================== 超时配置 ====================
DEFAULT_TIMEOUT = 300.0  # 5 分钟（RPC 调用超时）
DEFAULT_CONNECTION_TIMEOUT = 60.0  # 连接超时
DEFAULT_RETRY_INTERVAL = 2.0  # 重试间隔
DEFAULT_MAX_RETRIES = 10  # 最大重试次数
DEFAULT_BACKOFF_FACTOR = 1.5  # 退避因子

# ==================== 组件类型 ====================
DEFAULT_TRAINER_TYPE = "default"
DEFAULT_LEARNER_TYPE = "default"
DEFAULT_AGGREGATOR_TYPE = "fedavg"
DEFAULT_MODEL_TYPE = "cnn"
DEFAULT_DATASET_TYPE = "cifar10"

# ==================== 训练配置 ====================
DEFAULT_NUM_ROUNDS = 100
DEFAULT_LOCAL_EPOCHS = 5
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 0.01

# ==================== 数据划分 ====================
DEFAULT_PARTITION_STRATEGY = "dirichlet"
DEFAULT_PARTITION_ALPHA = 0.5
DEFAULT_PARTITION_SEED = 42

# ==================== 日志配置 ====================
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_DIR = "./logs"
DEFAULT_LOG_CONSOLE = True

# ==================== 序列化 ====================
DEFAULT_SERIALIZATION = "pickle"

# ==================== 传输模式 ====================
DEFAULT_TRANSPORT_MODE = "grpc"

# ==================== 心跳配置 ====================
DEFAULT_HEARTBEAT_INTERVAL = 30.0
DEFAULT_HEARTBEAT_TIMEOUT = 10.0
DEFAULT_HEARTBEAT_MAX_FAILURES = 3

# ==================== 实验配置 ====================
DEFAULT_EXP_NAME = "default"


__all__ = [
    # 网络
    "DEFAULT_TRAINER_PORT",
    "DEFAULT_LEARNER_BASE_PORT",
    "DEFAULT_HOST",
    "DEFAULT_LOCALHOST",
    "DEFAULT_MAX_MESSAGE_SIZE",
    # 超时
    "DEFAULT_TIMEOUT",
    "DEFAULT_CONNECTION_TIMEOUT",
    "DEFAULT_RETRY_INTERVAL",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_BACKOFF_FACTOR",
    # 组件
    "DEFAULT_TRAINER_TYPE",
    "DEFAULT_LEARNER_TYPE",
    "DEFAULT_AGGREGATOR_TYPE",
    "DEFAULT_MODEL_TYPE",
    "DEFAULT_DATASET_TYPE",
    # 训练
    "DEFAULT_NUM_ROUNDS",
    "DEFAULT_LOCAL_EPOCHS",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_LEARNING_RATE",
    # 数据划分
    "DEFAULT_PARTITION_STRATEGY",
    "DEFAULT_PARTITION_ALPHA",
    "DEFAULT_PARTITION_SEED",
    # 日志
    "DEFAULT_LOG_LEVEL",
    "DEFAULT_LOG_DIR",
    "DEFAULT_LOG_CONSOLE",
    # 序列化
    "DEFAULT_SERIALIZATION",
    # 传输
    "DEFAULT_TRANSPORT_MODE",
    # 心跳
    "DEFAULT_HEARTBEAT_INTERVAL",
    "DEFAULT_HEARTBEAT_TIMEOUT",
    "DEFAULT_HEARTBEAT_MAX_FAILURES",
    # 实验
    "DEFAULT_EXP_NAME",
]