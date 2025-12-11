"""
联邦学习统一入口类（重构版）
fedcl/federated_learning.py

职责：
    - 从配置文件或文件夹加载配置
    - 根据配置创建对应的Server/Client实例
    - 直接调用 Server.Trainer 进行训练
"""

import asyncio
import os
from typing import List, Optional, Dict, Any, Union, Tuple

from .config import ConfigLoader, CommunicationConfig, TrainingConfig
from .federation.client import FederationClient
from .federation.server import FederationServer
from .trainer.trainer import FederationResult
from .utils.auto_logger import get_sys_logger, setup_auto_logging


class FederatedLearning:
    """
    联邦学习统一入口类

    职责：
        - 从配置文件/文件夹加载配置（每个配置文件指定role和mode）
        - 根据配置创建对应的FederationServer/Client实例
        - 如果有1个Server + 多个Client，调用Trainer进行训练

    使用方式：
        # 方式1：从配置文件夹（推荐）
        fl = FederatedLearning("configs/")  # 加载文件夹中所有YAML
        await fl.run(max_rounds=10)

        # 方式2：从单个配置文件
        fl = FederatedLearning("config.yaml")
        await fl.run()

        # 方式3：从多个配置文件路径
        fl = FederatedLearning(["configs/server.yaml", "configs/client1.yaml", "configs/client2.yaml"])
        await fl.run(max_rounds=10)

        # 方式4：从配置对象
        configs = [
            (server_comm_config, server_train_config),
            (client1_comm_config, client1_train_config),
            (client2_comm_config, client2_train_config),
        ]
        fl = FederatedLearning(configs)
        await fl.run(max_rounds=10)
    """

    def __init__(
        self,
        config: Union[
            str,  # 配置文件路径或文件夹路径
            List[str],  # 多个配置文件路径
            Tuple[CommunicationConfig, TrainingConfig],  # 单个配置对象元组
            List[Tuple[CommunicationConfig, TrainingConfig]]  # 多个配置对象元组
        ],
        auto_setup_logging: bool = True,
        logging_config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化联邦学习系统

        Args:
            config: 配置，可以是：
                - str: 配置文件路径或文件夹路径
                    - 如果是文件：加载单个节点配置
                    - 如果是文件夹：加载所有YAML文件，每个文件对应一个节点
                - List[str]: 多个配置文件路径列表
                - Tuple[CommunicationConfig, TrainingConfig]: 单个配置对象元组
                - List[Tuple[...]]: 多个配置对象元组列表
            auto_setup_logging: 是否自动设置日志
            logging_config: 日志配置字典，支持以下键：
                - console_enabled: 是否启用控制台输出（默认True）
                - level: 日志级别（默认DEBUG）
                - format: 日志格式
                - rotation: 日志轮转大小
                - retention: 日志保留时间
                - compression: 压缩格式
        """
        # 存储从YAML提取的日志配置（用于自动提取）
        self._extracted_logging_config: Optional[Dict[str, Any]] = None

        # 解析配置，得到配置列表（在解析过程中会提取日志配置）
        # 返回格式: List[Tuple[CommunicationConfig, TrainingConfig, LoggingConfig]]
        self.config_list: List[Tuple[CommunicationConfig, TrainingConfig, 'LoggingConfig']] = self._parse_config(config)

        if not self.config_list:
            raise ValueError("No valid configuration found")

        # 分类配置（按role分类，包含logging_config）
        self.server_configs: List[Tuple[CommunicationConfig, TrainingConfig, 'LoggingConfig']] = []
        self.client_configs: List[Tuple[CommunicationConfig, TrainingConfig, 'LoggingConfig']] = []

        for comm_cfg, train_cfg, log_cfg in self.config_list:
            role = comm_cfg.role
            if not role:
                raise ValueError(
                    f"Role must be specified in configuration (node_id: {comm_cfg.node_id})"
                )

            if role == "server":
                self.server_configs.append((comm_cfg, train_cfg, log_cfg))
            elif role == "client":
                self.client_configs.append((comm_cfg, train_cfg, log_cfg))
            else:
                raise ValueError(f"Invalid role: {role}. Must be 'server' or 'client'")

        # 如果用户未提供logging_config，但YAML文件中有logging配置，则使用提取的配置
        if logging_config is None and self._extracted_logging_config is not None:
            logging_config = self._extracted_logging_config

        # 提取实验配置（从server配置中提取）
        experiment_config = None
        if self.server_configs:
            comm_cfg, train_cfg, log_cfg = self.server_configs[0]

            # 从 TrainingConfig 中提取扁平化配置
            experiment_config = {}

            # 数据集配置（从客户端配置中获取）
            if self.client_configs:
                _, client_train_cfg, _ = self.client_configs[0]

                # 安全地提取数据集配置
                if client_train_cfg.dataset and isinstance(client_train_cfg.dataset, dict):
                    experiment_config['dataset'] = client_train_cfg.dataset.get('name', 'UNKNOWN')
                    experiment_config['data_dir'] = client_train_cfg.dataset.get('data_dir', '')

                    # 数据划分配置
                    partition_cfg = client_train_cfg.dataset.get('partition', {})
                    if isinstance(partition_cfg, dict):
                        experiment_config['noniid_type'] = partition_cfg.get('type', 'iid')
                        experiment_config['alpha'] = partition_cfg.get('alpha')
                        experiment_config['num_clients'] = partition_cfg.get('num_clients', len(self.client_configs))
                        experiment_config['samples_per_client'] = partition_cfg.get('samples_per_client')

                # 模型配置
                if client_train_cfg.local_model and isinstance(client_train_cfg.local_model, dict):
                    experiment_config['model_name'] = client_train_cfg.local_model.get('name')
                    experiment_config['model_config'] = client_train_cfg.local_model

                # 学习器配置（训练参数）
                if client_train_cfg.learner and isinstance(client_train_cfg.learner, dict):
                    learner_cfg = client_train_cfg.learner
                    experiment_config['learning_rate'] = learner_cfg.get('learning_rate')
                    experiment_config['batch_size'] = learner_cfg.get('batch_size')
                    experiment_config['local_epochs'] = learner_cfg.get('epochs')
                    experiment_config['optimizer'] = learner_cfg.get('optimizer', 'SGD')
                    experiment_config['loss_fn'] = learner_cfg.get('loss_fn', 'CrossEntropyLoss')

            # 服务端配置
            experiment_config['num_rounds'] = train_cfg.max_rounds
            experiment_config['clients_per_round'] = train_cfg.min_clients

            # 聚合器配置
            if train_cfg.aggregator and isinstance(train_cfg.aggregator, dict):
                experiment_config['aggregator'] = train_cfg.aggregator.get('name', 'FedAvgAggregator')
                experiment_config['algorithm'] = train_cfg.aggregator.get('name', 'FedAvg').replace('Aggregator', '')

            # 通信配置
            experiment_config['comm_mode'] = comm_cfg.mode if hasattr(comm_cfg, 'mode') else 'ProcessAndNetwork'
            experiment_config['backend'] = comm_cfg.backend if hasattr(comm_cfg, 'backend') else None

            # 其他配置
            experiment_config['device'] = 'cuda'  # 默认
            experiment_config['seed'] = None

        # 自动设置日志（传递实验配置）
        if auto_setup_logging:
            setup_auto_logging(config=logging_config, experiment_config=experiment_config)

        self.logger = get_sys_logger()

        # 组件实例（延迟创建）
        self.servers: List[FederationServer] = []
        self.clients: List[FederationClient] = []

        # 状态
        self._is_initialized = False
        self._is_running = False

        self.logger.info(
            f"FederatedLearning created: {len(self.server_configs)} server(s), "
            f"{len(self.client_configs)} client(s)"
        )

    def _parse_config(
        self,
        config: Union[
            str,
            List[str],
            Tuple[CommunicationConfig, TrainingConfig],
            List[Tuple[CommunicationConfig, TrainingConfig]]
        ]
    ) -> List[Tuple[CommunicationConfig, TrainingConfig, 'LoggingConfig']]:
        """
        解析配置，返回配置列表（包含 LoggingConfig）

        Args:
            config: 配置输入

        Returns:
            List[Tuple[CommunicationConfig, TrainingConfig, LoggingConfig]]
        """
        if isinstance(config, str):
            # 单个路径（文件或文件夹）
            return self._load_from_path(config)

        elif isinstance(config, list):
            if not config:
                raise ValueError("Config list is empty")

            if isinstance(config[0], str):
                # List[str] - 多个文件路径
                result = []
                for path in config:
                    result.extend(self._load_from_path(path))
                return result

            elif isinstance(config[0], tuple):
                # List[Tuple[...]] - 配置对象列表
                return config

            else:
                raise TypeError(f"Unsupported config list element type: {type(config[0])}")

        elif isinstance(config, tuple) and len(config) == 2:
            # Single Tuple[CommunicationConfig, TrainingConfig] (backwards compat)
            comm_config, train_config = config
            if not isinstance(comm_config, CommunicationConfig):
                raise TypeError("First element must be CommunicationConfig")
            if not isinstance(train_config, TrainingConfig):
                raise TypeError("Second element must be TrainingConfig")
            # Create empty LoggingConfig for backwards compatibility
            from .config.logging_config import LoggingConfig
            log_config = LoggingConfig()
            return [(comm_config, train_config, log_config)]

        else:
            raise TypeError(
                "config must be one of: str (file/folder path), List[str], "
                "Tuple[CommunicationConfig, TrainingConfig], or List[Tuple[...]]"
            )

    def _extract_logging_config_from_file(self, file_path: str):
        """
        从YAML文件中提取日志配置

        如果已经提取过日志配置，则跳过（使用第一个找到的）

        Args:
            file_path: 配置文件路径
        """
        # 如果已经提取过日志配置，则跳过（使用第一个找到的）
        if self._extracted_logging_config is not None:
            return

        try:
            # 直接加载YAML文件以提取logging配置
            config_dict = ConfigLoader.load_with_inheritance(file_path)

            if 'logging' in config_dict:
                self._extracted_logging_config = config_dict['logging']
                # 此时logger还未创建，不能使用logger.info
                # print(f"  Extracted logging config from {os.path.basename(file_path)}")
        except Exception:
            # 提取失败不影响主流程
            pass

    def _load_from_path(self, path: str) -> List[Tuple[CommunicationConfig, TrainingConfig, 'LoggingConfig']]:
        """
        从路径加载配置（支持文件或文件夹）

        Args:
            path: 配置文件或文件夹路径

        Returns:
            配置列表（包含 LoggingConfig）
        """
        from .config.logging_config import LoggingConfig

        if not os.path.exists(path):
            raise FileNotFoundError(f"Config path not found: {path}")

        # 情况1: 路径是文件
        if os.path.isfile(path):
            # 注意：此时logger还未初始化
            comm_cfg, train_cfg = ConfigLoader.load(path)

            # 提取日志配置（如果存在）
            self._extract_logging_config_from_file(path)

            # 从提取的logging字典创建LoggingConfig对象
            if self._extracted_logging_config:
                log_cfg = LoggingConfig(**self._extracted_logging_config)
            else:
                log_cfg = LoggingConfig()

            return [(comm_cfg, train_cfg, log_cfg)]

        # 情况2: 路径是文件夹
        elif os.path.isdir(path):
            # 注意：此时logger还未初始化

            # 查找文件夹下所有 YAML 配置文件
            config_files = []
            for filename in os.listdir(path):
                if filename.endswith(('.yaml', '.yml')):
                    config_files.append(os.path.join(path, filename))

            # 按文件名排序，确保顺序一致
            config_files.sort()

            if not config_files:
                raise ValueError(f"No YAML config files found in folder: {path}")

            # 加载所有配置文件
            result = []
            for config_file in config_files:
                try:
                    comm_cfg, train_cfg = ConfigLoader.load(config_file)

                    # 提取日志配置（如果存在）
                    # 优先使用第一个包含logging配置的文件，或server配置文件
                    self._extract_logging_config_from_file(config_file)

                    # 为每个配置创建LoggingConfig（共享同一个配置）
                    if self._extracted_logging_config:
                        log_cfg = LoggingConfig(**self._extracted_logging_config)
                    else:
                        log_cfg = LoggingConfig()

                    result.append((comm_cfg, train_cfg, log_cfg))

                except Exception as e:
                    # Note: logger not initialized yet
                    raise ValueError(f"Failed to load config file: {config_file}") from e

            # Note: logger not initialized yet
            return result

        else:
            raise ValueError(f"Path is neither file nor folder: {path}")

    # ========== 初始化流程 ==========

    async def initialize(self):
        """初始化所有节点"""
        if self._is_initialized:
            self.logger.warning("System already initialized")
            return

        self.logger.info("=" * 60)
        self.logger.info("Initializing FederatedLearning...")
        self.logger.info("=" * 60)

        # 1. 初始化所有Server
        if self.server_configs:
            self.logger.info(f"Initializing {len(self.server_configs)} server(s)...")
            for i, (comm_cfg, train_cfg, log_cfg) in enumerate(self.server_configs):
                server = FederationServer(comm_cfg, train_cfg, logging_config=log_cfg)
                await server.initialize()
                await server.start_server()
                self.servers.append(server)
                self.logger.info(f"✓ Server {i+1}/{len(self.server_configs)} started: {server.server_id}")

        # 2. 等待Server就绪
        if self.servers:
            await self._wait_for_servers_ready(timeout=30)

        # 3. 初始化所有Client
        if self.client_configs:
            self.logger.info(f"Initializing {len(self.client_configs)} client(s)...")
            for i, (comm_cfg, train_cfg, log_cfg) in enumerate(self.client_configs):
                client = FederationClient(comm_cfg, train_cfg, logging_config=log_cfg)
                await client.initialize()
                await client.start_client()
                self.clients.append(client)
                self.logger.info(f"✓ Client {i+1}/{len(self.client_configs)} started: {client.client_id}")

        self._is_initialized = True
        self.logger.info("FederatedLearning initialized successfully")

    async def _wait_for_servers_ready(self, timeout: float = 30):
        """等待所有Server就绪"""
        import time
        start_time = time.time()

        while True:
            all_ready = all(server.is_running for server in self.servers)
            if all_ready:
                self.logger.info(f"✓ All {len(self.servers)} server(s) ready")
                return

            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise TimeoutError(f"Servers failed to start within {timeout}s")

            await asyncio.sleep(0.1)

    async def _wait_for_clients_ready(self, min_clients: int, timeout: float = 30):
        """等待足够数量的客户端注册并就绪

        Args:
            min_clients: 最少需要的客户端数量
            timeout: 超时时间（秒）

        Raises:
            TimeoutError: 如果超时仍未达到最少客户端数量
        """
        import time
        start_time = time.time()

        self.logger.info(f"Waiting for at least {min_clients} clients to be ready...")

        while True:
            # 检查训练器的可用客户端数量
            if self.servers and self.servers[0].trainer:
                available_clients = self.servers[0].trainer.get_available_clients()
                num_available = len(available_clients)

                if num_available >= min_clients:
                    self.logger.info(f"✓ {num_available} clients ready (minimum: {min_clients})")
                    self.logger.debug(f"  Available clients: {available_clients}")
                    return

                # 显示进度
                self.logger.debug(f"  Currently {num_available}/{min_clients} clients ready...")

            elapsed = time.time() - start_time
            if elapsed > timeout:
                num_available = len(self.servers[0].trainer.get_available_clients()) if self.servers and self.servers[0].trainer else 0
                raise TimeoutError(
                    f"Only {num_available} clients ready after {timeout}s (minimum: {min_clients})"
                )

            await asyncio.sleep(0.5)

    # ========== 实验记录 ==========

    def setup_experiment_recording(self, exp_config: Optional[Dict[str, Any]] = None):
        """
        设置实验记录

        Args:
            exp_config: 实验配置字典，可选。如果为None，则从配置文件中读取
                {
                    'enabled': True,
                    'name': 'my_experiment',  # 可选，默认自动生成
                    'base_dir': 'experiments/results',  # 可选
                    'record_server': True,  # 可选，默认True
                    'record_clients': True  # 可选，默认True
                }

        Returns:
            Tuple[Optional[Recorder], List[Recorder]]: (server_recorder, client_recorders)
        """
        try:
            from .experiment import Recorder, create_callbacks
        except ImportError:
            self.logger.warning("Experiment module not available")
            return None, []

        # 如果未提供配置，尝试从配置文件读取
        if exp_config is None:
            # 尝试从第一个配置中获取experiment配置
            if self.config_list:
                _, train_cfg, _ = self.config_list[0]
                exp_config = getattr(train_cfg, 'experiment', None)
                if isinstance(exp_config, dict):
                    pass
                else:
                    exp_config = None

        if not exp_config or not exp_config.get('enabled', False):
            self.logger.debug("Experiment recording not enabled")
            return None, []

        # 提取配置
        exp_name = exp_config.get('name')
        if not exp_name:
            # 自动生成实验名称
            import time
            exp_name = f"exp_{int(time.time())}"

        base_dir = exp_config.get('base_dir', 'experiments/results')
        record_server = exp_config.get('record_server', True)
        record_clients = exp_config.get('record_clients', True)

        self.logger.info(f"[实验记录] 设置实验记录: {exp_name}")

        server_recorder = None
        client_recorders = []

        # 为Server设置
        if record_server and self.servers:
            server = self.servers[0]
            server_recorder = Recorder.initialize(exp_name, "server", server.server_id, base_dir)
            server_recorder.start_run({"mode": str(server.comm_config.mode), "role": "server"})

            callbacks = create_callbacks(server_recorder)
            server.trainer.add_callback('after_round', callbacks['round_callback'])
            server.trainer.add_callback('after_evaluation', callbacks['eval_callback'])
            self.logger.info(f"  ✓ Server实验记录已启用")

        # 为Clients设置
        if record_clients and self.clients:
            for client in self.clients:
                Recorder.reset()
                client_recorder = Recorder.initialize(exp_name, "client", client.client_id, base_dir)
                client_recorder.start_run({"mode": str(client.comm_config.mode), "role": "client"})

                client_callbacks = create_callbacks(client_recorder)
                if hasattr(client, 'learner') and client.learner:
                    client.learner.add_callback('after_train', client_callbacks['client_train_callback'])

                client_recorders.append(client_recorder)
            self.logger.info(f"  ✓ {len(client_recorders)} Client实验记录已启用")

        return server_recorder, client_recorders

    # ========== 运行和清理 ==========

    async def run(self, max_rounds: Optional[int] = None, exp_config: Optional[Dict[str, Any]] = None) -> Optional[FederationResult]:
        """
        运行联邦学习（自动处理实验记录）

        Args:
            max_rounds: 最大训练轮数
            exp_config: 实验记录配置（可选）。如果为None，会从配置文件中读取

        Returns:
            FederationResult（如果有Server + Clients）或 None
        """
        if not self._is_initialized:
            await self.initialize()

        self._is_running = True

        # 自动设置实验记录
        server_recorder = None
        client_recorders = []

        try:
            # 尝试从配置文件读取实验配置（如果用户未提供）
            if exp_config is None and self.config_list:
                _, train_cfg, _ = self.config_list[0]
                exp_config = getattr(train_cfg, 'experiment', None)
                if isinstance(exp_config, dict) and exp_config.get('enabled', False):
                    self.logger.info("[实验记录] 从配置文件检测到实验记录配置，自动启用")

            # 如果有有效的实验配置，自动设置
            if exp_config and exp_config.get('enabled', False):
                server_recorder, client_recorders = self.setup_experiment_recording(exp_config)
                self.logger.info(f"[实验记录] 实验记录已自动启用: {exp_config.get('name', 'auto')}")

            # 如果有1个Server + 多个Client：运行训练
            if len(self.servers) == 1 and len(self.clients) > 0:
                server = self.servers[0]
                if not server.trainer:
                    raise ValueError("Server trainer not initialized")

                # 如果未提供max_rounds，尝试从服务器配置中获取
                if not max_rounds:
                    max_rounds = getattr(server.trainer.training_config, 'max_rounds', None)
                    if not max_rounds:
                        raise ValueError("max_rounds is required when running training with server and clients")

                # 获取训练器要求的最少客户端数量
                min_clients = getattr(server.trainer.training_config, 'min_clients', 2)

                # 等待足够数量的客户端就绪
                self.logger.info(f"Waiting for at least {min_clients} clients to register and be ready...")
                await self._wait_for_clients_ready(min_clients=min_clients, timeout=30)

                self.logger.info(f"Starting federation training for {max_rounds} rounds...")
                result = await server.trainer.run_training(max_rounds=max_rounds)

                # 自动记录最终结果
                if server_recorder and result:
                    server_recorder.log_info("final_accuracy", result.final_accuracy)
                    server_recorder.log_info("final_loss", result.final_loss)
                    server_recorder.log_info("completed_rounds", result.completed_rounds)
                    server_recorder.log_info("total_time", result.total_time)
                    server_recorder.finish(status="COMPLETED")
                    self.logger.info("[实验记录] Server实验记录已自动保存")

                for client_rec in client_recorders:
                    client_rec.finish(status="COMPLETED")
                if client_recorders:
                    self.logger.info(f"[实验记录] {len(client_recorders)} Client实验记录已自动保存")

                return result

            else:
                # 其他情况：保持运行，等待外部控制
                self.logger.info("Nodes are running. Press Ctrl+C to stop.")
                # 保持运行（由外部控制停止）
                return None

        except Exception as e:
            # 异常时也要保存记录
            if server_recorder:
                server_recorder.log_info("error", str(e))
                server_recorder.finish(status="FAILED")
            for client_rec in client_recorders:
                client_rec.finish(status="FAILED")
            raise
        finally:
            self._is_running = False

    async def cleanup(self, force_clear_global_state: bool = False):
        """清理资源

        Args:
            force_clear_global_state: 是否强制清理全局状态（适用于批量实验场景）
        """
        self.logger.debug("Cleaning up resources...")

        # 停止所有Client
        for client in self.clients:
            try:
                await client.stop_client()
            except Exception as e:
                self.logger.error(f"Error stopping client {client.client_id}: {e}")

        # 停止所有Server
        for server in self.servers:
            try:
                await server.stop_server()
            except Exception as e:
                self.logger.error(f"Error stopping server {server.server_id}: {e}")

        # 如果使用 Memory 模式，清理全局状态（适用于批量实验）
        if force_clear_global_state or self._should_clear_global_state():
            try:
                from .transport.memory import MemoryTransport
                MemoryTransport.clear_global_state()
                self.logger.debug("✓ Global state cleared (Memory mode)")
            except Exception as e:
                self.logger.debug(f"Skip global state cleanup: {e}")

        # 清理实验记录器的单例状态（适用于批量实验）
        if force_clear_global_state:
            try:
                from .experiment import Recorder
                Recorder.reset()
                self.logger.debug("✓ Recorder singleton reset")
            except Exception as e:
                self.logger.debug(f"Skip recorder cleanup: {e}")

        self.logger.debug("Cleanup completed")

    def _should_clear_global_state(self) -> bool:
        """判断是否应该清理全局状态（检测是否为 Memory 模式）"""
        # 检查是否所有配置都是 Memory 模式
        for comm_cfg, _, _ in self.config_list:
            if comm_cfg.mode == "memory":
                return True
        return False

    # ========== 状态查询 ==========

    def get_status(self) -> Dict[str, Any]:
        """
        获取系统状态

        Returns:
            状态字典
        """
        return {
            "num_servers": len(self.servers),
            "num_clients": len(self.clients),
            "is_initialized": self._is_initialized,
            "is_running": self._is_running,
            "servers_status": [server.get_server_status() for server in self.servers],
            "clients_status": [client.get_client_status() for client in self.clients]
        }

    # ========== 便捷访问属性 ==========

    @property
    def server(self) -> Optional[FederationServer]:
        """获取第一个Server实例（便捷访问）"""
        return self.servers[0] if self.servers else None

    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.cleanup()
