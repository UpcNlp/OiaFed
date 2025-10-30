"""
MOE-FedCL 统一入口类
fedcl/federated_learning.py

提供一个统一的入口类，整合 FederationCoordinator, FederationServer, FederationClient
用户只需提供配置文件和 Trainer/Learner 类即可启动完整系统
"""

import asyncio
from typing import Type, Dict, Any, List, Optional, Tuple, Union

from .config import (
    ServerConfig, ClientConfig,
    load_server_config, load_client_config,
    create_default_server_config, create_default_client_config
)
from .federation.client import FederationClient
from .federation.coordinator import FederationCoordinator, FederationResult
from .federation.server import FederationServer
from .learner.base_learner import BaseLearner
from .trainer.trainer import BaseTrainer
from .types import ModelData, FederationConfig
from .utils.auto_logger import get_sys_logger, setup_auto_logging


class FederatedLearning:
    """
    联邦学习统一入口类

    整合了 FederationServer, FederationClient, FederationCoordinator
    提供最简单的方式从配置文件启动完整的联邦学习系统

    Example:
        >>> # 方式1: 使用配置文件
        >>> fl = FederatedLearning(
        ...     trainer_class=MyTrainer,
        ...     learner_class=MyLearner,
        ...     global_model={"weight": 1.0},
        ...     server_config_path="configs/server.yaml",
        ...     client_config_path="configs/client.yaml",
        ...     num_clients=5
        ... )
        >>> result = await fl.run(max_rounds=10)

        >>> # 方式2: 使用配置对象
        >>> fl = FederatedLearning(
        ...     trainer_class=MyTrainer,
        ...     learner_class=MyLearner,
        ...     global_model={"weight": 1.0},
        ...     server_config=ServerConfig(...),
        ...     client_config=ClientConfig(...),
        ...     num_clients=5
        ... )
        >>> result = await fl.run(max_rounds=10)

        >>> # 方式3: 使用自定义节点ID
        >>> fl = FederatedLearning(
        ...     trainer_class=MyTrainer,
        ...     learner_class=MyLearner,
        ...     global_model={"weight": 1.0},
        ...     server_id="my_custom_server",
        ...     client_ids=["alice", "bob", "charlie"],
        ...     num_clients=3
        ... )
        >>> result = await fl.run(max_rounds=10)
    """

    def __init__(
        self,
        trainer_class: Type[BaseTrainer],
        learner_class: Type[BaseLearner],
        global_model: ModelData,
        # 服务端配置
        server_config_path: Optional[str] = None,
        server_config: Optional[ServerConfig] = None,
        # 客户端配置
        client_config_path: Optional[str] = None,
        client_configs: Optional[Union[List[ClientConfig], ClientConfig]] = None,
        num_clients: int = 2,
        # 额外配置
        trainer_config: Optional[Dict[str, Any]] = None,
        learner_config: Optional[Dict[str, Any]] = None,
        federation_config: Optional[FederationConfig] = None,
        # 其他选项
        auto_setup_logging: bool = True
    ):
        """
        初始化联邦学习系统

        Args:
            trainer_class: 训练器类（继承自BaseTrainer）
            learner_class: 学习器类（继承自BaseLearner）
            global_model: 初始全局模型
            server_config_path: 服务端配置文件路径（与server_config二选一）
            server_config: 服务端配置对象（与server_config_path二选一）
            client_config_path: 客户端配置路径（与client_configs互斥）
                - 如果是文件路径：该文件作为所有客户端的共享配置
                - 如果是文件夹路径：文件夹下的每个YAML文件作为一个客户端的独立配置
            client_configs: 客户端配置（与client_config_path互斥）
                - 如果是单个 ClientConfig 对象：所有客户端共享该配置
                - 如果是 List[ClientConfig]：每个客户端使用独立配置
            num_clients: 客户端数量
            trainer_config: 训练器额外配置
            learner_config: 学习器额外配置
            federation_config: 联邦学习配置
            auto_setup_logging: 是否自动设置日志

        Note:
            - server_id 从 ServerConfig.server_id 中读取（如果未设置则自动生成）
            - client_id 从 ClientConfig.client_id 中读取（如果未设置则自动生成）

        Example:
            >>> # 方式1: 所有客户端使用相同配置文件
            >>> fl = FederatedLearning(
            ...     trainer_class=MyTrainer,
            ...     learner_class=MyLearner,
            ...     global_model={"weight": 1.0},
            ...     server_config_path="configs/server.yaml",
            ...     client_config_path="configs/client.yaml",  # 单个文件
            ...     num_clients=3
            ... )

            >>> # 方式2: 使用文件夹，为每个客户端提供独立配置
            >>> fl = FederatedLearning(
            ...     trainer_class=MyTrainer,
            ...     learner_class=MyLearner,
            ...     global_model={"weight": 1.0},
            ...     server_config_path="configs/server.yaml",
            ...     client_config_path="configs/clients/",  # 文件夹（需要3个配置文件）
            ...     num_clients=3
            ... )

            >>> # 方式3: 使用单个配置对象（所有客户端共享）
            >>> from fedcl.config import ClientConfig, TransportLayerConfig
            >>> client_config = ClientConfig(
            ...     mode="process",
            ...     transport=TransportLayerConfig(port=0)  # 自动分配端口
            ... )
            >>> fl = FederatedLearning(
            ...     trainer_class=MyTrainer,
            ...     learner_class=MyLearner,
            ...     global_model={"weight": 1.0},
            ...     server_config_path="configs/server.yaml",
            ...     client_configs=client_config,  # 单个对象
            ...     num_clients=3
            ... )

            >>> # 方式4: 使用配置对象列表（每个客户端独立配置）
            >>> client_configs = [
            ...     ClientConfig(
            ...         mode="process",
            ...         client_id="alice",
            ...         transport=TransportLayerConfig(port=8001)
            ...     ),
            ...     ClientConfig(
            ...         mode="process",
            ...         client_id="bob",
            ...         transport=TransportLayerConfig(port=8002)
            ...     ),
            ...     ClientConfig(
            ...         mode="process",
            ...         client_id="charlie",
            ...         transport=TransportLayerConfig(port=8003)
            ...     )
            ... ]
            >>> fl = FederatedLearning(
            ...     trainer_class=MyTrainer,
            ...     learner_class=MyLearner,
            ...     global_model={"weight": 1.0},
            ...     server_config_path="configs/server.yaml",
            ...     client_configs=client_configs,  # 列表
            ...     num_clients=3
            ... )
        """
        # 设置日志
        if auto_setup_logging:
            setup_auto_logging()

        self.logger = get_sys_logger()

        # 保存类和模型
        self.trainer_class = trainer_class
        self.learner_class = learner_class
        self.global_model = global_model
        self.num_clients = num_clients

        # 验证客户端配置参数互斥性
        if client_config_path is not None and client_configs is not None:
            raise ValueError(
                "client_config_path 和 client_configs 参数互斥，只能指定其中一个"
            )

        # 保存额外配置
        self.trainer_config = trainer_config or {}
        self.learner_config = learner_config or {}
        self.federation_config = federation_config or FederationConfig()

        # 加载服务端配置
        self.server_config = self._load_server_config(server_config_path, server_config)

        # 处理客户端配置
        self.client_configs_list: Optional[List[ClientConfig]] = None
        self.client_config_single: Optional[ClientConfig] = None

        if client_configs is not None:
            # 方式1: 直接提供配置对象（单个或列表）
            if isinstance(client_configs, list):
                # 列表：每个客户端独立配置
                if len(client_configs) != num_clients:
                    raise ValueError(
                        f"client_configs 列表的长度 ({len(client_configs)}) 必须等于 num_clients ({num_clients})"
                    )
                self.client_configs_list = client_configs
                self.logger.info(f"使用 {len(client_configs)} 个独立客户端配置")
            else:
                # 单个对象：所有客户端共享
                self.client_config_single = client_configs
                self.logger.info(f"使用单个配置对象（{num_clients}个客户端共享）")

        elif client_config_path is not None:
            # 方式2: 从路径加载（可能是文件或文件夹）
            self.client_configs_list, self.client_config_single = self._load_client_configs_from_path(
                client_config_path, num_clients
            )

        else:
            # 方式3: 使用默认配置
            self.client_config_single = create_default_client_config()
            self.logger.info(f"使用默认客户端配置（{num_clients}个客户端共享）")

        # 组件实例（延迟初始化）
        self.server: Optional[FederationServer] = None
        self.clients: List[FederationClient] = []
        self.coordinator: Optional[FederationCoordinator] = None

        # 状态
        self._is_initialized = False
        self._is_running = False

        self.logger.info(f"✅ FederatedLearning 已创建（{num_clients}个客户端）")

    def _load_server_config(
        self,
        config_path: Optional[str],
        config: Optional[ServerConfig]
    ) -> ServerConfig:
        """加载服务端配置"""
        if config_path:
            return load_server_config(config_path)
        elif config:
            return config
        else:
            # 使用默认配置
            return create_default_server_config()

    def _load_client_configs_from_path(
        self,
        config_path: str,
        num_clients: int
    ) -> Tuple[Optional[List[ClientConfig]], Optional[ClientConfig]]:
        """从路径加载客户端配置（支持文件或文件夹）

        Args:
            config_path: 配置文件或文件夹路径
            num_clients: 期望的客户端数量

        Returns:
            tuple: (configs_list, single_config)
                - 如果是文件夹：返回 (配置列表, None)
                - 如果是文件：返回 (None, 单个配置)

        Raises:
            FileNotFoundError: 路径不存在
            ValueError: 文件夹下的配置文件数量与 num_clients 不匹配
        """
        import os

        # 检查路径是否存在
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置路径不存在: {config_path}")

        # 情况1: 路径是文件
        if os.path.isfile(config_path):
            self.logger.info(f"从文件加载客户端配置（所有客户端共享）: {config_path}")
            single_config = load_client_config(config_path)
            return None, single_config

        # 情况2: 路径是文件夹
        elif os.path.isdir(config_path):
            self.logger.info(f"从文件夹加载客户端配置（每个客户端独立配置）: {config_path}")

            # 查找文件夹下所有 YAML 配置文件
            config_files = []
            for filename in os.listdir(config_path):
                if filename.endswith(('.yaml', '.yml')):
                    config_files.append(os.path.join(config_path, filename))

            # 按文件名排序，确保顺序一致
            config_files.sort()

            # 验证配置文件数量
            if len(config_files) == 0:
                raise ValueError(f"文件夹 {config_path} 中没有找到 YAML 配置文件")

            if len(config_files) != num_clients:
                raise ValueError(
                    f"文件夹 {config_path} 中的配置文件数量 ({len(config_files)}) "
                    f"与 num_clients ({num_clients}) 不匹配。\n"
                    f"找到的配置文件: {[os.path.basename(f) for f in config_files]}"
                )

            # 加载所有配置文件
            configs = []
            for config_file in config_files:
                try:
                    config = load_client_config(config_file)
                    configs.append(config)
                    self.logger.info(f"  ✓ 加载配置: {os.path.basename(config_file)}")
                except Exception as e:
                    self.logger.error(f"  ✗ 加载配置失败: {os.path.basename(config_file)} - {e}")
                    raise ValueError(f"加载配置文件失败: {config_file}") from e

            self.logger.info(f"✅ 成功从文件夹加载 {len(configs)} 个客户端配置")
            return configs, None

        else:
            raise ValueError(f"配置路径既不是文件也不是文件夹: {config_path}")

    async def initialize(self):
        """初始化所有组件"""
        if self._is_initialized:
            self.logger.warning("系统已初始化，跳过")
            return

        self.logger.info("="*60)
        self.logger.info("开始初始化联邦学习系统")
        self.logger.info("="*60)

        # 1. 初始化服务端
        await self._initialize_server()

        # 2. 等待服务端完全启动
        await asyncio.sleep(1)

        # 3. 初始化客户端
        await self._initialize_clients()

        # 4. 等待客户端注册
        await asyncio.sleep(1)

        # 5. 创建协调器
        self._create_coordinator()

        self._is_initialized = True
        self.logger.info("✅ 联邦学习系统初始化完成")

    async def _initialize_server(self):
        """初始化服务端"""
        self.logger.info("🚀 初始化服务端...")

        # 创建服务端（使用配置中的 server_id，如果未设置则由 FederationServer 自动生成）
        config_dict = self.server_config.to_dict()

        # 如果配置中指定了 server_id，则传递给 FederationServer
        if self.server_config.server_id:
            self.server = FederationServer(config_dict, server_id=self.server_config.server_id)
        else:
            self.server = FederationServer(config_dict)

        # 初始化训练器
        await self.server.initialize_with_trainer(
            trainer_class=self.trainer_class,
            global_model=self.global_model,
            trainer_config=self.trainer_config
        )

        # 启动服务端
        await self.server.start_server()

        self.logger.info(f"✅ 服务端已启动: {self.server.server_id}")

    async def _initialize_clients(self):
        """初始化所有客户端"""
        self.logger.info(f"🚀 初始化 {self.num_clients} 个客户端...")

        tasks = []
        for i in range(self.num_clients):
            task = self._create_and_start_client(i)
            tasks.append(task)

        # 并发启动所有客户端
        self.clients = await asyncio.gather(*tasks)

        self.logger.info(f"✅ {len(self.clients)} 个客户端已启动")

    async def _create_and_start_client(self, index: int) -> FederationClient:
        """创建并启动单个客户端

        Args:
            index: 客户端索引

        Returns:
            FederationClient: 已启动的客户端实例
        """
        # 确定使用哪个配置
        if self.client_configs_list is not None:
            # 使用独立配置列表
            client_config = self.client_configs_list[index]
        else:
            # 使用共享配置，为每个客户端创建独立副本
            client_config = ClientConfig.from_dict(self.client_config_single.to_dict())

        # 如果配置中没有设置 client_id，则自动生成
        if client_config.client_id is None:
            client_config.client_id = f"client_{index+1}"

        # 创建客户端
        client = FederationClient(
            client_config.to_dict(),
            client_id=client_config.client_id
        )

        # 初始化学习器
        await client.initialize_with_learner(
            learner_class=self.learner_class,
            learner_config=self.learner_config
        )

        # 启动客户端
        await client.start_client()

        self.logger.info(f"  ✓ 客户端 {client.client_id} 已启动")

        return client

    def _create_coordinator(self):
        """创建协调器"""
        self.logger.info("🚀 创建联邦学习协调器...")

        self.coordinator = FederationCoordinator(
            federation_server=self.server,
            federation_config=self.federation_config
        )

        self.logger.info("✅ 协调器已创建")

    async def run(self, max_rounds: Optional[int] = None) -> FederationResult:
        """
        运行联邦学习训练

        Args:
            max_rounds: 最大训练轮数（覆盖配置中的值）

        Returns:
            FederationResult: 训练结果
        """
        # 初始化（如果还没初始化）
        if not self._is_initialized:
            await self.initialize()

        # 更新最大轮数
        if max_rounds is not None:
            self.federation_config.max_rounds = max_rounds
            self.coordinator.federation_config.max_rounds = max_rounds

        self.logger.info("="*60)
        self.logger.info(f"开始联邦学习训练（{self.federation_config.max_rounds}轮）")
        self.logger.info("="*60)

        self._is_running = True

        try:
            # 运行训练
            result = await self.coordinator.start_federation()

            self.logger.info("="*60)
            self.logger.info("联邦学习训练完成")
            self.logger.info(f"  完成轮数: {result.completed_rounds}")
            self.logger.info(f"  最终准确率: {result.final_accuracy:.4f}")
            self.logger.info(f"  最终损失: {result.final_loss:.4f}")
            self.logger.info(f"  总时间: {result.total_time:.2f}秒")
            self.logger.info("="*60)

            return result

        except Exception as e:
            self.logger.error(f"训练失败: {e}")
            raise
        finally:
            self._is_running = False

    async def cleanup(self):
        """清理所有资源"""
        self.logger.info("开始清理资源...")

        # 停止所有客户端
        if self.clients:
            self.logger.info(f"停止 {len(self.clients)} 个客户端...")
            tasks = [client.stop_client() for client in self.clients]
            await asyncio.gather(*tasks, return_exceptions=True)
            self.logger.info("✅ 所有客户端已停止")

        # 停止服务端
        if self.server:
            self.logger.info("停止服务端...")
            await self.server.stop_server()
            self.logger.info("✅ 服务端已停止")

        self._is_initialized = False
        self._is_running = False

        self.logger.info("✅ 资源清理完成")

    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.cleanup()


# ============================================
# 便捷函数
# ============================================

async def run_federated_learning(
    trainer_class: Type[BaseTrainer],
    learner_class: Type[BaseLearner],
    global_model: ModelData,
    server_config_path: str,
    client_config_path: str = None,
    client_configs = None,
    num_clients: int = 2,
    max_rounds: int = 10,
    trainer_config: Optional[Dict[str, Any]] = None,
    learner_config: Optional[Dict[str, Any]] = None,
    federation_config: Optional[FederationConfig] = None
) -> FederationResult:
    """
    便捷函数：一行代码运行完整的联邦学习系统

    Args:
        trainer_class: 训练器类
        learner_class: 学习器类
        global_model: 初始全局模型
        server_config_path: 服务端配置文件
        client_config_path: 客户端配置路径（与client_configs二选一）
            - 如果是文件路径：该文件作为所有客户端的共享配置
            - 如果是文件夹路径：文件夹下的每个YAML文件作为一个客户端的独立配置
        client_configs: 客户端配置（与client_config_path二选一）
            - 如果是单个 ClientConfig 对象：所有客户端共享该配置
            - 如果是 List[ClientConfig]：每个客户端使用独立配置
        num_clients: 客户端数量
        max_rounds: 训练轮数
        trainer_config: 训练器配置
        learner_config: 学习器配置
        federation_config: 联邦学习配置

    Returns:
        FederationResult: 训练结果

    Note:
        - server_id 从 ServerConfig.server_id 中读取
        - client_id 从 ClientConfig.client_id 中读取

    Example:
        >>> # 方式1: 使用配置文件（所有客户端共享配置）
        >>> result = await run_federated_learning(
        ...     MyTrainer, MyLearner,
        ...     {"weight": 1.0},
        ...     "configs/server.yaml",
        ...     "configs/client.yaml",  # 单个文件
        ...     num_clients=5,
        ...     max_rounds=10
        ... )

        >>> # 方式2: 使用配置文件夹（每个客户端独立配置）
        >>> result = await run_federated_learning(
        ...     MyTrainer, MyLearner,
        ...     {"weight": 1.0},
        ...     "configs/server.yaml",
        ...     "configs/clients/",  # 文件夹路径，包含5个配置文件
        ...     num_clients=5,
        ...     max_rounds=10
        ... )

        >>> # 方式3: 使用单个配置对象（所有客户端共享）
        >>> from fedcl.config import ClientConfig, TransportLayerConfig
        >>> client_config = ClientConfig(
        ...     mode="process",
        ...     transport=TransportLayerConfig(port=0)
        ... )
        >>> result = await run_federated_learning(
        ...     MyTrainer, MyLearner,
        ...     {"weight": 1.0},
        ...     "configs/server.yaml",
        ...     client_configs=client_config,  # 单个对象
        ...     num_clients=3,
        ...     max_rounds=10
        ... )

        >>> # 方式4: 使用独立客户端配置对象列表
        >>> client_configs = [
        ...     ClientConfig(mode="process", client_id="alice", transport=TransportLayerConfig(port=8001)),
        ...     ClientConfig(mode="process", client_id="bob", transport=TransportLayerConfig(port=8002)),
        ...     ClientConfig(mode="process", client_id="charlie", transport=TransportLayerConfig(port=8003))
        ... ]
        >>> result = await run_federated_learning(
        ...     MyTrainer, MyLearner,
        ...     {"weight": 1.0},
        ...     "configs/server.yaml",
        ...     client_configs=client_configs,  # 列表
        ...     num_clients=3,
        ...     max_rounds=10
        ... )
    """
    async with FederatedLearning(
        trainer_class=trainer_class,
        learner_class=learner_class,
        global_model=global_model,
        server_config_path=server_config_path,
        client_config_path=client_config_path,
        client_configs=client_configs,
        num_clients=num_clients,
        trainer_config=trainer_config,
        learner_config=learner_config,
        federation_config=federation_config
    ) as fl:
        return await fl.run(max_rounds=max_rounds)
