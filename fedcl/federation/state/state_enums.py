# fedcl/federation/state/state_enums.py
"""
状态枚举定义模块

定义联邦学习过程中各类组件的状态枚举。
"""

from enum import Enum, auto


class ServerState(Enum):
    """
    服务端状态枚举
    
    定义服务端在联邦学习过程中的各种状态。
    """
    # 初始化阶段
    INITIALIZING = auto()        # 正在初始化服务端组件
    LOADING_CONFIG = auto()      # 正在加载配置
    REGISTERING_COMPONENTS = auto()  # 正在注册组件
    
    # 等待阶段
    WAITING_FOR_CLIENTS = auto() # 等待客户端注册
    READY = auto()               # 准备开始训练
    
    # 训练阶段
    TRAINING = auto()            # 正在进行联邦训练
    COORDINATING = auto()        # 正在协调客户端
    WAITING_FOR_UPDATES = auto() # 等待客户端更新
    
    # 聚合阶段
    AGGREGATING = auto()         # 正在聚合客户端更新
    VALIDATING_AGGREGATION = auto()  # 正在验证聚合结果
    
    # 下发阶段
    DISTRIBUTING = auto()        # 正在下发聚合结果
    BROADCASTING = auto()        # 正在广播模型
    
    # 评估阶段
    EVALUATING = auto()          # 正在评估模型
    
    # 结束阶段
    ROUND_COMPLETED = auto()     # 轮次完成
    COMPLETED = auto()           # 训练完成
    STOPPING = auto()            # 正在停止
    
    # 异常状态
    ERROR = auto()               # 发生错误
    TIMEOUT = auto()             # 超时
    DISCONNECTED = auto()        # 连接断开


class ClientState(Enum):
    """
    客户端状态枚举（保持向后兼容）
    
    定义客户端在联邦学习过程中的各种状态。
    """
    # 初始化阶段
    INITIALIZING = auto()        # 客户端正在初始化
    LOADING_CONFIG = auto()      # 正在加载配置
    PREPARING_DATA = auto()      # 正在准备数据
    
    # 注册阶段
    REGISTERING = auto()         # 正在向服务端注册
    REGISTERED = auto()          # 已成功注册到服务端
    
    # 等待阶段
    WAITING = auto()             # 等待训练指令
    IDLE = auto()                # 空闲状态
    
    # 数据处理阶段
    LOADING_DATA = auto()        # 正在加载数据集
    PREPROCESSING_DATA = auto()  # 正在预处理数据
    
    # 训练阶段
    TRAINING = auto()            # 正在进行本地训练
    TRAINING_EPOCH = auto()      # 正在训练轮次
    VALIDATING = auto()          # 正在验证模型
    
    # 通信阶段
    UPLOADING = auto()           # 正在上传训练结果
    DOWNLOADING = auto()         # 正在下载全局模型
    SYNCHRONIZING = auto()       # 正在同步状态
    
    # 评估阶段
    EVALUATING = auto()          # 正在评估模型
    
    # 结束阶段
    ROUND_COMPLETED = auto()     # 轮次完成
    COMPLETED = auto()           # 训练完成
    
    # 异常状态
    DISCONNECTED = auto()        # 与服务端断开连接
    ERROR = auto()               # 发生错误
    TIMEOUT = auto()             # 操作超时
    PAUSED = auto()              # 暂停状态


class ClientLifecycleState(Enum):
    """
    客户端生命周期状态枚举（协调层管理）
    
    重构后新增，用于管理客户端整体生命周期状态。
    """
    INITIALIZING = auto()        # 客户端初始化中
    LOADING_CONFIG = auto()      # 加载配置中  
    PREPARING_DATA = auto()      # 准备数据中
    REGISTERING = auto()         # 向服务端注册中
    REGISTERED = auto()          # 已注册到服务端
    READY = auto()              # 准备就绪，等待任务
    CONNECTED = auto()          # 与服务端连接中
    TRAINING = auto()           # 正在训练
    COMPLETED = auto()          # 任务完成
    ERROR = auto()              # 错误状态
    DISCONNECTED = auto()       # 与服务端断开连接


class TrainingPhaseState(Enum):
    """
    训练阶段状态枚举（控制层管理）
    
    重构后新增，用于管理训练过程的具体状态。
    """
    UNINITIALIZED = auto()       # 未初始化
    INITIALIZING = auto()        # 初始化中
    PREPARING = auto()           # 准备训练环境
    RUNNING = auto()             # 运行中
    PHASE_TRANSITION = auto()    # 阶段转换中
    EPOCH_EXECUTING = auto()     # 执行epoch中
    EVALUATING = auto()          # 评估中
    AGGREGATING = auto()         # 聚合中
    FINISHED = auto()            # 训练完成
    PAUSED = auto()             # 暂停
    FAILED = auto()             # 失败


class AuxiliaryState(Enum):
    """
    辅助模型状态枚举
    
    定义辅助模型（如教师模型）在联邦学习过程中的各种状态。
    """
    # 初始化阶段
    INITIALIZING = auto()        # 正在初始化
    LOADING_MODEL = auto()       # 正在加载模型
    
    # 注册阶段
    REGISTERING = auto()         # 正在向服务端注册
    REGISTERED = auto()          # 已注册
    
    # 工作阶段
    READY = auto()               # 准备就绪
    PROVIDING_GUIDANCE = auto()  # 正在提供指导
    UPDATING = auto()            # 正在更新模型
    
    # 结束阶段
    COMPLETED = auto()           # 任务完成
    
    # 异常状态
    ERROR = auto()               # 发生错误
    DISCONNECTED = auto()        # 连接断开


class StateTransition:
    """
    状态转换定义类
    
    定义各状态之间的合法转换关系。
    """
    
    # 服务端状态转换规则
    SERVER_TRANSITIONS = {
        ServerState.INITIALIZING: [
            ServerState.LOADING_CONFIG,
            ServerState.ERROR
        ],
        ServerState.LOADING_CONFIG: [
            ServerState.REGISTERING_COMPONENTS,
            ServerState.ERROR
        ],
        ServerState.REGISTERING_COMPONENTS: [
            ServerState.WAITING_FOR_CLIENTS,
            ServerState.ERROR
        ],
        ServerState.WAITING_FOR_CLIENTS: [
            ServerState.READY,
            ServerState.ERROR,
            ServerState.TIMEOUT
        ],
        ServerState.READY: [
            ServerState.TRAINING,
            ServerState.STOPPING,
            ServerState.ERROR
        ],
        ServerState.TRAINING: [
            ServerState.COORDINATING,
            ServerState.WAITING_FOR_UPDATES,
            ServerState.ERROR,
            ServerState.TIMEOUT
        ],
        ServerState.COORDINATING: [
            ServerState.WAITING_FOR_UPDATES,
            ServerState.ERROR
        ],
        ServerState.WAITING_FOR_UPDATES: [
            ServerState.AGGREGATING,
            ServerState.TIMEOUT,
            ServerState.ERROR
        ],
        ServerState.AGGREGATING: [
            ServerState.VALIDATING_AGGREGATION,
            ServerState.ERROR
        ],
        ServerState.VALIDATING_AGGREGATION: [
            ServerState.DISTRIBUTING,
            ServerState.EVALUATING,
            ServerState.ERROR
        ],
        ServerState.DISTRIBUTING: [
            ServerState.BROADCASTING,
            ServerState.ERROR
        ],
        ServerState.BROADCASTING: [
            ServerState.ROUND_COMPLETED,
            ServerState.ERROR
        ],
        ServerState.EVALUATING: [
            ServerState.ROUND_COMPLETED,
            ServerState.ERROR
        ],
        ServerState.ROUND_COMPLETED: [
            ServerState.TRAINING,
            ServerState.COMPLETED,
            ServerState.ERROR
        ],
        ServerState.COMPLETED: [
            ServerState.STOPPING
        ],
        ServerState.STOPPING: [],
        ServerState.ERROR: [
            ServerState.STOPPING,
            ServerState.INITIALIZING  # 允许重新初始化
        ],
        ServerState.TIMEOUT: [
            ServerState.ERROR,
            ServerState.STOPPING
        ]
    }
    
    # 客户端状态转换规则（保持向后兼容）
    CLIENT_TRANSITIONS = {
        ClientState.INITIALIZING: [
            ClientState.LOADING_CONFIG,
            ClientState.ERROR
        ],
        ClientState.LOADING_CONFIG: [
            ClientState.PREPARING_DATA,
            ClientState.ERROR
        ],
        ClientState.PREPARING_DATA: [
            ClientState.REGISTERING,
            ClientState.ERROR
        ],
        ClientState.REGISTERING: [
            ClientState.REGISTERED,
            ClientState.ERROR,
            ClientState.TIMEOUT
        ],
        ClientState.REGISTERED: [
            ClientState.WAITING,
            ClientState.DISCONNECTED,
            ClientState.ERROR
        ],
        ClientState.WAITING: [
            ClientState.LOADING_DATA,
            ClientState.DOWNLOADING,
            ClientState.DISCONNECTED,
            ClientState.PAUSED
        ],
        ClientState.IDLE: [
            ClientState.WAITING,
            ClientState.TRAINING,
            ClientState.DISCONNECTED
        ],
        ClientState.LOADING_DATA: [
            ClientState.PREPROCESSING_DATA,
            ClientState.ERROR
        ],
        ClientState.PREPROCESSING_DATA: [
            ClientState.TRAINING,
            ClientState.ERROR
        ],
        ClientState.DOWNLOADING: [
            ClientState.TRAINING,
            ClientState.ERROR,
            ClientState.TIMEOUT
        ],
        ClientState.TRAINING: [
            ClientState.TRAINING_EPOCH,
            ClientState.VALIDATING,
            ClientState.UPLOADING,
            ClientState.ERROR
        ],
        ClientState.TRAINING_EPOCH: [
            ClientState.TRAINING,
            ClientState.VALIDATING,
            ClientState.ERROR
        ],
        ClientState.VALIDATING: [
            ClientState.UPLOADING,
            ClientState.EVALUATING,
            ClientState.ERROR
        ],
        ClientState.UPLOADING: [
            ClientState.SYNCHRONIZING,
            ClientState.ROUND_COMPLETED,
            ClientState.ERROR,
            ClientState.TIMEOUT
        ],
        ClientState.SYNCHRONIZING: [
            ClientState.WAITING,
            ClientState.ROUND_COMPLETED,
            ClientState.ERROR
        ],
        ClientState.EVALUATING: [
            ClientState.ROUND_COMPLETED,
            ClientState.ERROR
        ],
        ClientState.ROUND_COMPLETED: [
            ClientState.WAITING,
            ClientState.COMPLETED,
            ClientState.DISCONNECTED
        ],
        ClientState.COMPLETED: [],
        ClientState.DISCONNECTED: [
            ClientState.REGISTERING,  # 允许重新注册
            ClientState.ERROR
        ],
        ClientState.ERROR: [
            ClientState.INITIALIZING,  # 允许重新初始化
            ClientState.DISCONNECTED
        ],
        ClientState.TIMEOUT: [
            ClientState.ERROR,
            ClientState.WAITING
        ],
        ClientState.PAUSED: [
            ClientState.WAITING,
            ClientState.DISCONNECTED
        ]
    }
    
    # 客户端生命周期状态转换规则（新增）
    CLIENT_LIFECYCLE_TRANSITIONS = {
        ClientLifecycleState.INITIALIZING: [
            ClientLifecycleState.LOADING_CONFIG,
            ClientLifecycleState.ERROR
        ],
        ClientLifecycleState.LOADING_CONFIG: [
            ClientLifecycleState.PREPARING_DATA,
            ClientLifecycleState.ERROR
        ],
        ClientLifecycleState.PREPARING_DATA: [
            ClientLifecycleState.REGISTERING,
            ClientLifecycleState.ERROR
        ],
        ClientLifecycleState.REGISTERING: [
            ClientLifecycleState.REGISTERED,
            ClientLifecycleState.ERROR,
            ClientLifecycleState.DISCONNECTED
        ],
        ClientLifecycleState.REGISTERED: [
            ClientLifecycleState.READY,
            ClientLifecycleState.CONNECTED,
            ClientLifecycleState.ERROR,
            ClientLifecycleState.DISCONNECTED
        ],
        ClientLifecycleState.READY: [
            ClientLifecycleState.TRAINING,
            ClientLifecycleState.CONNECTED,
            ClientLifecycleState.COMPLETED,
            ClientLifecycleState.ERROR,
            ClientLifecycleState.DISCONNECTED
        ],
        ClientLifecycleState.CONNECTED: [
            ClientLifecycleState.TRAINING,
            ClientLifecycleState.READY,
            ClientLifecycleState.ERROR,
            ClientLifecycleState.DISCONNECTED
        ],
        ClientLifecycleState.TRAINING: [
            ClientLifecycleState.READY,
            ClientLifecycleState.COMPLETED,
            ClientLifecycleState.ERROR,
            ClientLifecycleState.DISCONNECTED
        ],
        ClientLifecycleState.COMPLETED: [
            ClientLifecycleState.READY,
            ClientLifecycleState.DISCONNECTED
        ],
        ClientLifecycleState.ERROR: [
            ClientLifecycleState.INITIALIZING,
            ClientLifecycleState.DISCONNECTED
        ],
        ClientLifecycleState.DISCONNECTED: [
            ClientLifecycleState.REGISTERING,
            ClientLifecycleState.ERROR
        ]
    }
    
    # 训练阶段状态转换规则（新增）
    TRAINING_PHASE_TRANSITIONS = {
        TrainingPhaseState.UNINITIALIZED: [
            TrainingPhaseState.INITIALIZING,
            TrainingPhaseState.FAILED
        ],
        TrainingPhaseState.INITIALIZING: [
            TrainingPhaseState.PREPARING,
            TrainingPhaseState.FAILED
        ],
        TrainingPhaseState.PREPARING: [
            TrainingPhaseState.RUNNING,
            TrainingPhaseState.FAILED
        ],
        TrainingPhaseState.RUNNING: [
            TrainingPhaseState.PHASE_TRANSITION,
            TrainingPhaseState.EPOCH_EXECUTING,
            TrainingPhaseState.EVALUATING,
            TrainingPhaseState.FINISHED,
            TrainingPhaseState.PAUSED,
            TrainingPhaseState.FAILED
        ],
        TrainingPhaseState.PHASE_TRANSITION: [
            TrainingPhaseState.EPOCH_EXECUTING,
            TrainingPhaseState.RUNNING,
            TrainingPhaseState.FAILED
        ],
        TrainingPhaseState.EPOCH_EXECUTING: [
            TrainingPhaseState.EVALUATING,
            TrainingPhaseState.RUNNING,
            TrainingPhaseState.PHASE_TRANSITION,
            TrainingPhaseState.FAILED
        ],
        TrainingPhaseState.EVALUATING: [
            TrainingPhaseState.AGGREGATING,
            TrainingPhaseState.RUNNING,
            TrainingPhaseState.FINISHED,
            TrainingPhaseState.FAILED
        ],
        TrainingPhaseState.AGGREGATING: [
            TrainingPhaseState.RUNNING,
            TrainingPhaseState.FINISHED,
            TrainingPhaseState.FAILED
        ],
        TrainingPhaseState.FINISHED: [
            TrainingPhaseState.UNINITIALIZED,  # 允许重新开始
            TrainingPhaseState.PREPARING       # 准备下一轮
        ],
        TrainingPhaseState.PAUSED: [
            TrainingPhaseState.RUNNING,
            TrainingPhaseState.FAILED,
            TrainingPhaseState.FINISHED
        ],
        TrainingPhaseState.FAILED: [
            TrainingPhaseState.UNINITIALIZED,  # 允许重新开始
            TrainingPhaseState.PREPARING       # 尝试恢复
        ]
    }
    
    @classmethod
    def is_valid_transition(cls, from_state, to_state):
        """
        检查状态转换是否合法
        
        Args:
            from_state: 源状态
            to_state: 目标状态
            
        Returns:
            bool: 转换是否合法
        """
        if isinstance(from_state, ServerState):
            transitions = cls.SERVER_TRANSITIONS
        elif isinstance(from_state, ClientState):
            transitions = cls.CLIENT_TRANSITIONS
        elif isinstance(from_state, ClientLifecycleState):
            transitions = cls.CLIENT_LIFECYCLE_TRANSITIONS
        elif isinstance(from_state, TrainingPhaseState):
            transitions = cls.TRAINING_PHASE_TRANSITIONS
        else:
            return False
            
        return to_state in transitions.get(from_state, [])
    
    @classmethod
    def get_valid_transitions(cls, current_state):
        """
        获取当前状态的所有合法转换
        
        Args:
            current_state: 当前状态
            
        Returns:
            List: 合法的目标状态列表
        """
        if isinstance(current_state, ServerState):
            transitions = cls.SERVER_TRANSITIONS
        elif isinstance(current_state, ClientState):
            transitions = cls.CLIENT_TRANSITIONS
        elif isinstance(current_state, ClientLifecycleState):
            transitions = cls.CLIENT_LIFECYCLE_TRANSITIONS
        elif isinstance(current_state, TrainingPhaseState):
            transitions = cls.TRAINING_PHASE_TRANSITIONS
        else:
            return []
            
        return transitions.get(current_state, [])


# 向后兼容的状态映射
STATE_MAPPING = {
    # ClientState -> ClientLifecycleState 映射
    ClientState.INITIALIZING: ClientLifecycleState.INITIALIZING,
    ClientState.LOADING_CONFIG: ClientLifecycleState.LOADING_CONFIG,
    ClientState.PREPARING_DATA: ClientLifecycleState.PREPARING_DATA,
    ClientState.REGISTERING: ClientLifecycleState.REGISTERING,
    ClientState.REGISTERED: ClientLifecycleState.REGISTERED,
    ClientState.WAITING: ClientLifecycleState.READY,
    ClientState.TRAINING: ClientLifecycleState.TRAINING,
    ClientState.COMPLETED: ClientLifecycleState.COMPLETED,
    ClientState.ERROR: ClientLifecycleState.ERROR,
    ClientState.DISCONNECTED: ClientLifecycleState.DISCONNECTED,
}

def map_old_state_to_new(old_state):
    """
    将旧的状态映射到新的状态
    
    Args:
        old_state: 旧的状态枚举
        
    Returns:
        新的状态枚举
    """
    return STATE_MAPPING.get(old_state, old_state)