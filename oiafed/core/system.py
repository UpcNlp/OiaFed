"""
联邦学习系统容器

提供统一的系统初始化和管理

设计原则：
- 延迟连接：initialize() 不阻塞，连接在 run() 入口建立
- 支持多种运行模式：本地串行、本地并行、分布式
- 只使用配置类，不使用字典操作
- 运行时信息同步：Trainer 生成统一的 run_name，Learner 连接后同步
"""

import asyncio
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..config import (
    NodeConfig,
    LogConfig,
    TrackerConfig,
    ConnectionRetryConfig,
    ComponentConfig,
)
from ..proxy import ProxyCollection
from .node import Node
from ..registry import registry
from ..callback import CallbackManager
from ..tracker import CompositeTracker
from ..infra.logging import setup_logging, get_logger, cleanup_logging


class FederatedSystem:
    """
    联邦学习系统容器

    职责：
    1. 统一节点管理（不区分角色）
    2. 配置驱动组件创建（根据配置中存在的组件来初始化）
    3. 依赖注入
    4. 提供统一入口

    设计原则：
    - 只接受 NodeConfig，不支持字典配置
    - 延迟连接：initialize() 只做本地准备，不等待连接
    - 连接在 run() 入口建立，支持重试

    Example:
        from federation import load_config, FederatedSystem

        config = load_config("configs/trainer.yaml")
        system = FederatedSystem(config)
        await system.initialize()
        result = await system.run()
        await system.stop()
    """

    def __init__(self, config: NodeConfig):
        """
        初始化系统

        Args:
            config: NodeConfig 实例
        """
        self._config = config

        # Node 将在 initialize 中创建
        self.node: Optional[Node] = None

        # 组件容器
        self.trainer = None
        self.learner = None
        self.learners: Optional[ProxyCollection] = None

        # 基础设施
        self.tracker: Optional[CompositeTracker] = None
        self.callbacks: Optional[CallbackManager] = None

        # 系统日志
        self.sys_logger = None

        # ComponentBuilder
        from ..builder import ComponentBuilder
        self.builder = ComponentBuilder()

        # 延迟连接状态
        self._connected: bool = False
        self._initialized: bool = False

        # Shutdown 状态
        self._shutdown_event: Optional[asyncio.Event] = None
        
        # ===== 运行时信息（用于分布式同步）=====
        # Trainer 会生成这些信息，Learner 会在连接后同步获取
        self._runtime_run_name: Optional[str] = None
        self._runtime_mlflow_run_id: Optional[str] = None
        self._runtime_synced: bool = False  # Learner 是否已同步

    # ========== 属性访问器 ==========

    @property
    def config(self) -> NodeConfig:
        """获取配置对象"""
        return self._config

    @property
    def node_id(self) -> str:
        """获取节点 ID"""
        return self._config.node_id

    @property
    def exp_name(self) -> str:
        """获取实验名称"""
        return self._config.exp_name

    @property
    def run_name(self) -> Optional[str]:
        """获取运行名称"""
        return self._config.run_name

    @property
    def log_dir(self) -> str:
        """获取日志目录"""
        return self._config.log_dir

    # ========== 工厂方法 ==========

    @classmethod
    def from_node_config(cls, config: NodeConfig) -> "FederatedSystem":
        """
        从 NodeConfig 创建系统

        Args:
            config: NodeConfig 实例

        Returns:
            FederatedSystem 实例
        """
        return cls(config)

    # ========== 核心生命周期方法 ==========

    async def initialize(self):
        """
        初始化系统（不阻塞）

        此方法只做本地准备工作，不等待任何远程连接：
        1. 初始化日志系统
        2. 创建并启动 Node
        3. 初始化基础设施（Tracker/Callbacks）
        4. 初始化 Learner 组件并 bind
        5. [Trainer] 注册运行时信息同步方法
        """
        try:
            if self._initialized:
                if self.sys_logger:
                    self.sys_logger.warning(f"系统已初始化: {self.node_id}")
                return

            # 0. [Trainer] 生成统一的 run_name
            if self._config.is_trainer():
                self._generate_runtime_info()

            # 1. 初始化日志系统
            self._setup_logging()
            self.sys_logger = get_logger(self.node_id, "system")
            self.sys_logger.info(f"开始初始化联邦学习节点: node_id={self.node_id}")
            self.sys_logger.info(f"实验: {self.exp_name}")
            if self.run_name:
                self.sys_logger.info(f"运行: {self.run_name}")
            self.sys_logger.info(f"日志目录: {self.log_dir}")

            # 2. 创建 Node
            self.sys_logger.info(f"正在创建通信节点: {self.node_id}")
            await self._create_node()
            self.sys_logger.debug(f"通信节点创建完成: {self.node_id}")

            # 3. 注册事件处理器
            self.sys_logger.debug("注册连接事件处理器")
            self.node.on("connect", self._on_peer_connect)
            self.node.on("disconnect", self._on_peer_disconnect)

            # 4. 启动 Node
            self.sys_logger.debug("启动通信节点")
            await self.node.start()
            self.sys_logger.debug("通信节点已启动")

            # 5. 初始化基础设施
            self.sys_logger.debug("初始化基础设施")
            await self._initialize_infrastructure()

            # 6. 初始化 Learner
            if self._config.learner:
                self.sys_logger.debug("初始化 Learner 组件")
                await self._initialize_learner()

            # 7. [Trainer] 注册运行时信息同步方法
            if self._config.is_trainer():
                self._register_runtime_sync_handler()

            self._initialized = True
            self.sys_logger.info(f"联邦学习系统初始化完成: node_id={self.node_id}")

        except Exception as e:
            if self.sys_logger:
                self.sys_logger.exception(f"系统初始化失败: {e}")
            else:
                import traceback
                print(f"ERROR: 系统初始化失败 (node_id={self.node_id}): {e}", file=sys.stderr)
                traceback.print_exc()
            raise

    async def run(self) -> Dict[str, Any]:
        """
        运行系统

        Returns:
            运行结果字典
        """
        if not self._initialized:
            raise RuntimeError("System not initialized. Call initialize() first.")

        try:
            # 1. 确保连接已建立
            await self._ensure_connected()

            # 2. [Learner] 同步运行时信息（从 Trainer 获取 run_name 和 mlflow_run_id）
            if self._config.is_learner() and not self._config.is_trainer():
                await self._sync_runtime_info_from_trainer()

            # 3. 初始化 Trainer（需要连接信息）
            if self._config.trainer and self.trainer is None:
                self.sys_logger.info(f"[{self.node_id}] 开始初始化 Trainer")
                await self._initialize_trainer()
                self.sys_logger.info(f"[{self.node_id}] Trainer 初始化完成")

            # 4. 触发系统启动钩子
            if self.callbacks:
                self.sys_logger.debug(f"[{self.node_id}] 触发系统启动钩子")
                await self.callbacks.on_system_start(self)

            # 5. 运行主逻辑
            if self.trainer:
                return await self._run_as_trainer()
            else:
                return await self._run_as_learner()

        except Exception as e:
            self.sys_logger.exception(f"系统运行失败: {e}")
            raise

    async def stop(self):
        """停止系统"""
        self.sys_logger.info(f"Stopping FederatedSystem: {self.node_id}")

        if self.callbacks:
            await self.callbacks.on_system_stop(self)

        if self.tracker:
            self.tracker.close()
            self.sys_logger.info("Tracker closed")

        if self.node:
            await self.node.stop()

        self._connected = False
        self.sys_logger.debug(f"FederatedSystem stopped: {self.node_id}")

    # ========== 日志初始化 ==========

    def _setup_logging(self):
        """初始化日志系统"""
        log_config = self._config.logging or LogConfig()
        setup_logging(node_id=self.node_id, log_config=log_config)

    # ========== Node 创建 ==========

    async def _create_node(self):
        """创建通信节点"""
        # *** 关键修复：在调用 get_comm_config() 之前设置 critical_peers ***
        # 自动设置 critical_peers：如果配置为空且使用 gRPC，将 connect_to 中的节点设为关键节点
        if (self._config.transport.mode == "grpc" and self._config.connect_to):

            # heartbeat 配置中可能包含 critical_peers
            heartbeat_config = self._config.heartbeat or {}
            existing_critical_peers = heartbeat_config.get("critical_peers", [])

            # 如果 critical_peers 为空，则自动填充
            if not existing_critical_peers:
                # 从 connect_to 提取节点 ID（格式: "trainer@localhost:50051" -> "trainer"）
                critical_peers = []
                for addr in self._config.connect_to:
                    if '@' in addr:
                        node_id = addr.split('@')[0]
                        critical_peers.append(node_id)

                # 更新 heartbeat 配置（必须在 get_comm_config() 之前）
                if not self._config.heartbeat:
                    self._config.heartbeat = {}
                self._config.heartbeat["critical_peers"] = critical_peers

                if self.sys_logger and critical_peers:
                    self.sys_logger.info(
                        f"[{self.node_id}] 自动设置关键节点: {critical_peers}"
                    )

        # 从 NodeConfig 获取 NodeCommConfig（此时 heartbeat 配置已更新）
        comm_config = self._config.get_comm_config()

        self.node = Node(comm_config)
        await self.node.initialize()

    # ========== 连接管理 ==========

    async def _ensure_connected(self):
        """确保连接已建立"""
        if self._connected:
            return

        self.sys_logger.debug(f"[{self.node_id}] 开始建立网络连接")

        # 1. 连接到远程节点（Learner 模式）
        if self._config.connect_to:
            retry_config = self._get_retry_config()
            self.sys_logger.debug(f"[{self.node_id}] 连接到远程节点: {self._config.connect_to}")

            for addr in self._config.connect_to:
                target_id, address = self._parse_address(addr)
                self.sys_logger.debug(f"[{self.node_id}] 正在连接到 {target_id}")
                await self.node.connect(target_id, address, retry_config=retry_config)
                self.sys_logger.debug(f"[{self.node_id}] 已成功连接到 {target_id}")

        # 2. 等待足够的 peer 连接（Trainer 模式）
        if self._config.listen and self._config.min_peers > 0:
            timeout = 120  # 默认超时
            self.sys_logger.info(
                f"[{self.node_id}] 等待 {self._config.min_peers} 个节点连接"
            )
            await self.node.wait_for_connections(
                min_peers=self._config.min_peers,
                timeout=timeout
            )
            connected_peers = self.node.get_connected_nodes()
            self.sys_logger.info(f"[{self.node_id}] 所有节点已连接: {connected_peers}")

        self._connected = True
        self.sys_logger.info(f"[{self.node_id}] 网络连接建立完成")

    def _get_retry_config(self) -> Dict[str, Any]:
        """获取连接重试配置"""
        cfg = self._config.connection_retry
        return {
            "enabled": cfg.enabled,
            "max_retries": cfg.max_retries,
            "retry_interval": cfg.retry_interval,
            "timeout": cfg.timeout,
            "backoff": cfg.backoff,
            "backoff_factor": cfg.backoff_factor,
        }

    def _parse_address(self, addr: str) -> tuple:
        """解析地址字符串"""
        if "@" in addr:
            target_id, address = addr.split("@", 1)
        else:
            target_id = addr
            address = addr if ":" in addr else None
        return target_id, address

    # ========== 运行逻辑 ==========

    async def _run_as_trainer(self) -> Dict[str, Any]:
        """以 Trainer 角色运行"""
        self.sys_logger.info(f"[{self.node_id}] 开始运行 Trainer")
        result = await self.trainer.run()
        self.sys_logger.info(f"[{self.node_id}] Trainer 运行完成")
        return result

    async def _run_as_learner(self) -> Dict[str, Any]:
        """以 Learner 角色运行"""
        self.sys_logger.debug(f"[{self.node_id}] Learner 等待被调用")

        if self._shutdown_event:
            try:
                await self._shutdown_event.wait()
                self.sys_logger.debug(f"[{self.node_id}] 收到 shutdown 信号")

                if self.node:
                    self.sys_logger.info(f"[{self.node_id}] 正在停止 Node...")
                    await self.node.stop()

                if self.learner and hasattr(self.learner, 'on_shutdown'):
                    self.learner.on_shutdown()

                self.sys_logger.info(f"[{self.node_id}] Learner 已关闭")
            except asyncio.CancelledError:
                self.sys_logger.info(f"[{self.node_id}] Learner cancelled")
        else:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.sys_logger.info(f"[{self.node_id}] Learner stopped")

        return {
            "node_id": self.node_id,
            "role": "learner",
            "status": "shutdown",
        }

    # ========== 事件处理 ==========

    async def _on_peer_connect(self, peer_id: str, client_proxy):
        """连接事件处理器"""
        self.sys_logger.debug(f"Peer connected: {peer_id}")

        if self.learners:
            self.sys_logger.debug(f"Current learners count: {len(self.learners)}")

        if self.tracker:
            self.tracker.log_params({f"peer_{peer_id}": "connected"})

    async def _on_peer_disconnect(self, peer_id: str, reason: str = "unknown"):
        """断开连接事件处理器"""
        self.sys_logger.warning(f"Peer disconnected: {peer_id}, reason: {reason}")

        if self.learners:
            self.sys_logger.debug(f"Remaining learners count: {len(self.learners)}")

        if self.tracker:
            self.tracker.log_params({f"peer_{peer_id}": f"disconnected_{reason}"})

    # ========== 基础设施初始化 ==========

    async def _initialize_infrastructure(self):
        """初始化 Tracker 和 Callbacks"""
        self.sys_logger.debug("Initializing infrastructure...")

        # Tracker - 直接传递 TrackerConfig
        self.sys_logger.info(f"[DEBUG] self._config.tracker = {self._config.tracker}")
        if self._config.tracker:
            self.sys_logger.info(f"[DEBUG] tracker.enabled = {self._config.tracker.enabled}")
            self.sys_logger.info(f"[DEBUG] tracker.backends = {self._config.tracker.backends}")
            self.sys_logger.info(f"[DEBUG] exp_name = {self.exp_name}")
            
            try:
                self.tracker = CompositeTracker.from_config(
                    tracker_config=self._config.tracker,
                    node_id=self.node_id,
                    is_trainer=self._config.is_trainer(),
                    exp_name=self.exp_name,  # 传入实验名称
                )
                self.sys_logger.info(f"[DEBUG] Created tracker = {self.tracker}")
            except Exception as e:
                self.sys_logger.error(f"[DEBUG] Failed to create tracker: {e}")
                import traceback
                self.sys_logger.error(traceback.format_exc())
                self.tracker = None

            if self.tracker:
                params = {
                    "node_id": self.node_id,
                    "exp_name": self.exp_name,
                }
                if self.run_name:
                    params["run_name"] = self.run_name

                # 添加 trainer 参数
                trainer_config = self._config.get_trainer_config()
                if trainer_config:
                    params.update(trainer_config.get_args())

                self.tracker.log_params(params)
                self.tracker.set_tags({
                    "node_id": self.node_id,
                    "exp_name": self.exp_name,
                    "framework": "federation",
                })

        # Callbacks
        self.callbacks = CallbackManager()
        for cb_config in self._config.get_callbacks():
            cb_args = cb_config.get_args()

            if cb_config.type == 'tracker_sync' and self.tracker:
                cb_args['tracker'] = self.tracker

            cb = registry.create(
                namespace=f"federated.callback.{cb_config.type}",
                **cb_args
            )
            self.callbacks.add(cb)
            self.sys_logger.debug(f"Added Callback: {cb_config.type}")

        self.sys_logger.debug("Infrastructure initialized")

    # ========== 业务组件初始化 ==========

    async def _initialize_trainer(self):
        """初始化 Trainer"""
        self.sys_logger.info("Initializing Trainer...")

        # 1. 获取已连接的节点
        connected_peer_ids = self.node.get_connected_nodes()
        self.sys_logger.info(f"已连接的节点: {connected_peer_ids}")

        # 2. 创建 ProxyCollection
        self.learners = ProxyCollection(self.node, connected_peer_ids)
        self.sys_logger.info(f"Created ProxyCollection with {len(self.learners)} learners")

        # 3. 使用 Builder 构建依赖
        components = await self.builder.build(self._config)

        aggregator = components.get("aggregator")
        datasets_dict = components.get("datasets", {})

        # 向后兼容：取第一个训练数据集
        train_datasets = datasets_dict.get("train", [])
        dataset = train_datasets[0] if train_datasets else None

        # 4. 创建 Trainer
        trainer_config = self._config.get_trainer_config()
        self.sys_logger.info(f"正在创建 Trainer: {trainer_config.type}")
        self.sys_logger.info(f"[DEBUG] 创建 Trainer 时 self.tracker = {self.tracker}")
        self.sys_logger.info(f"[DEBUG] trainer_config.get_args() = {trainer_config.get_args()}")

        # 构建 namespace
        trainer_type = trainer_config.type
        if "." not in trainer_type:
            namespace = f"trainer.{trainer_type}"
        else:
            namespace = trainer_type

        self.trainer = registry.create(
            namespace=namespace,
            config=trainer_config.get_args(),
            learners=self.learners,
            aggregator=aggregator,
            dataset=dataset,         # 向后兼容：第一个训练数据集
            datasets=datasets_dict,  # 新方式：完整的数据集字典
            tracker=self.tracker,
            callbacks=self.callbacks,
        )
        
        self.sys_logger.info(f"[DEBUG] 创建后 self.trainer._tracker = {getattr(self.trainer, '_tracker', 'NO ATTR')}")

        self.sys_logger.info(f"Trainer initialized: {trainer_config.type}")

    async def _initialize_learner(self):
        """初始化 Learner"""
        self.sys_logger.info("Initializing Learner...")

        # 使用 Builder 构建组件
        components = await self.builder.build(self._config)

        self.learner = components.get("learner")

        if self.learner:
            self.sys_logger.info(f"Created Learner: {type(self.learner).__name__}")

            if self.tracker and hasattr(self.learner, 'set_tracker'):
                self.learner.set_tracker(self.tracker)
            if self.callbacks and hasattr(self.learner, '_callbacks'):
                self.learner._callbacks = self.callbacks

            self.node.bind(self.learner, name="learner")

            self._shutdown_event = asyncio.Event()
            self.node.register("_fed_shutdown", self._handle_shutdown)
            self.sys_logger.info("Learner initialized and bound")
        else:
            self.sys_logger.warning("Learner not created")

    # ========== Shutdown 处理 ==========

    async def _handle_shutdown(self, payload: Dict[str, Any], ctx: Any) -> Dict[str, Any]:
        """处理 shutdown 信号"""
        reason = payload.get("reason", "unknown")
        self.sys_logger.debug(f"[{self.node_id}] 收到 shutdown 信号: {reason}")

        await self._stop_heartbeat_safely()

        if self._shutdown_event:
            self._shutdown_event.set()

        return {"status": "shutting_down", "reason": reason}

    async def _stop_heartbeat_safely(self):
        """安全停止心跳任务"""
        try:
            if not self.node or not hasattr(self.node, '_comm'):
                return

            comm = self.node._comm
            if not hasattr(comm, '_transport'):
                return

            transport = comm._transport

            if hasattr(transport, '_control_running') and transport._control_running:
                self.sys_logger.debug(f"[{self.node_id}] 停止心跳任务")
                transport._stop_control_thread()
                await asyncio.sleep(0.1)

        except Exception as e:
            self.sys_logger.warning(f"停止心跳任务时出错: {e}")

    # ========== 上下文管理器 ==========

    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.stop()
        return False

    # ========== 辅助方法 ==========

    def get_experiment_info(self) -> Dict[str, Any]:
        """获取实验信息"""
        return {
            "node_id": self.node_id,
            "exp_name": self.exp_name,
            "run_name": self.run_name,
            "log_dir": self.log_dir,
            "is_trainer": self._config.is_trainer(),
            "is_learner": self._config.is_learner(),
            "initialized": self._initialized,
            "connected": self._connected,
        }

    # ========== 运行时信息同步 ==========

    def _generate_runtime_info(self) -> None:
        """
        [Trainer] 生成运行时信息
        
        在 Trainer 初始化时调用，生成统一的 run_name。
        这个 run_name 会通过 RPC 同步给所有 Learner。
        """
        # 使用配置中的 run_name，如果没有则生成
        if self._config.logging and self._config.logging.run_name:
            self._runtime_run_name = self._config.logging.run_name
        else:
            self._runtime_run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 更新配置中的 logging.run_name（确保日志系统使用统一的值）
        if self._config.logging:
            # LogConfig 是 dataclass，需要通过 object.__setattr__ 修改
            object.__setattr__(self._config.logging, 'run_name', self._runtime_run_name)

    def _register_runtime_sync_handler(self) -> None:
        """
        [Trainer] 注册运行时信息同步 RPC 方法
        
        Learner 连接后会调用此方法获取运行时信息。
        """
        async def handle_sync_runtime_info(
            payload: Dict[str, Any],
            ctx: Any,
        ) -> Dict[str, Any]:
            """处理运行时信息同步请求"""
            self.sys_logger.debug(f"收到运行时信息同步请求: from={ctx.source}")
            
            # 调用可覆盖的钩子方法
            result = self.get_sync_info()
            
            self.sys_logger.info(f"同步运行时信息到 {ctx.source}: {result}")
            return result
        
        self.node.register("_sync_runtime_info", handle_sync_runtime_info)
        self.sys_logger.debug("已注册运行时信息同步方法: _sync_runtime_info")

    def get_sync_info(self) -> Dict[str, Any]:
        """
        [Trainer] 获取要同步给 Learner 的运行时信息
        
        子类可覆盖此方法添加自定义同步信息。
        
        Returns:
            包含运行时信息的字典
            
        Example:
            class MySystem(FederatedSystem):
                def get_sync_info(self) -> Dict[str, Any]:
                    info = super().get_sync_info()
                    info["global_seed"] = 42
                    info["custom_config"] = {"key": "value"}
                    return info
        """
        info = {
            "exp_name": self.exp_name,
            "run_name": self._runtime_run_name,
        }
        
        # 如果有 MLflow Tracker，添加 run_id
        if self.tracker:
            for backend in getattr(self.tracker, 'trackers', []):
                if hasattr(backend, 'get_sync_info'):
                    info.update(backend.get_sync_info())
                    break
        
        return info

    async def _sync_runtime_info_from_trainer(self) -> None:
        """
        [Learner] 从 Trainer 同步运行时信息
        
        获取 Trainer 的 run_name 和 mlflow_run_id 等信息。
        """
        if self._runtime_synced:
            return
        
        # 获取 Trainer 节点 ID
        trainer_id = None
        if self._config.connect_to:
            for addr in self._config.connect_to:
                if '@' in addr:
                    trainer_id = addr.split('@')[0]
                    break
                else:
                    trainer_id = addr
                    break
        
        if not trainer_id:
            self.sys_logger.warning("无法确定 Trainer 节点 ID，跳过运行时信息同步")
            return
        
        try:
            self.sys_logger.info(f"从 Trainer ({trainer_id}) 同步运行时信息...")
            
            # 调用 Trainer 的同步方法
            result = await self.node.call(
                trainer_id,
                "_sync_runtime_info",
                {},
                timeout=10.0,
            )
            
            if result:
                self._runtime_run_name = result.get("run_name")
                self._runtime_mlflow_run_id = result.get("mlflow_run_id")
                
                self.sys_logger.info(
                    f"运行时信息同步成功: "
                    f"run_name={self._runtime_run_name}, "
                    f"mlflow_run_id={self._runtime_mlflow_run_id}"
                )
                
                # 调用可覆盖的钩子方法处理同步信息
                self.on_sync_info(result)
                
                self._runtime_synced = True
            else:
                self.sys_logger.warning("运行时信息同步返回空结果")
                
        except Exception as e:
            self.sys_logger.warning(f"运行时信息同步失败: {e}")
            # 不抛出异常，允许继续运行（使用本地配置）

    def on_sync_info(self, info: Dict[str, Any]) -> None:
        """
        [Learner] 处理从 Trainer 同步的运行时信息
        
        子类可覆盖此方法处理自定义同步信息。
        
        Args:
            info: 从 Trainer 同步的信息字典
            
        Example:
            class MySystem(FederatedSystem):
                def on_sync_info(self, info: Dict[str, Any]) -> None:
                    super().on_sync_info(info)  # 先处理默认逻辑
                    if "global_seed" in info:
                        torch.manual_seed(info["global_seed"])
        """
        # 默认处理：让 MLflow Tracker 加入 Trainer 的 run
        mlflow_run_id = info.get("mlflow_run_id")
        
        self.sys_logger.info(f"on_sync_info: mlflow_run_id={mlflow_run_id}, tracker={self.tracker}")
        
        if mlflow_run_id and self.tracker:
            backends = getattr(self.tracker, 'trackers', [])
            self.sys_logger.info(f"Found {len(backends)} tracker backends")
            
            for i, backend in enumerate(backends):
                backend_name = type(backend).__name__
                has_join_run = hasattr(backend, 'join_run')
                self.sys_logger.info(f"Backend[{i}]: {backend_name}, has_join_run={has_join_run}")
                
                if has_join_run:
                    backend.join_run(mlflow_run_id)
                    self.sys_logger.info(f"MLflow Tracker 已加入 run: {mlflow_run_id}")
                    break
        elif not mlflow_run_id:
            self.sys_logger.warning("on_sync_info: mlflow_run_id is None")
        elif not self.tracker:
            self.sys_logger.warning("on_sync_info: self.tracker is None")

    def __repr__(self) -> str:
        """字符串表示"""
        role = "trainer" if self._config.is_trainer() else "learner"
        return f"FederatedSystem(node_id={self.node_id}, role={role}, exp={self.exp_name})"