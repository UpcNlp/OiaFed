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
        auto_setup_logging: bool = True
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
        """
        # 自动设置日志
        if auto_setup_logging:
            setup_auto_logging()

        self.logger = get_sys_logger()

        # 解析配置，得到配置列表
        self.config_list: List[Tuple[CommunicationConfig, TrainingConfig]] = self._parse_config(config)

        if not self.config_list:
            raise ValueError("No valid configuration found")

        # 分类配置（按role分类）
        self.server_configs: List[Tuple[CommunicationConfig, TrainingConfig]] = []
        self.client_configs: List[Tuple[CommunicationConfig, TrainingConfig]] = []

        for comm_cfg, train_cfg in self.config_list:
            role = comm_cfg.role
            if not role:
                raise ValueError(
                    f"Role must be specified in configuration (node_id: {comm_cfg.node_id})"
                )

            if role == "server":
                self.server_configs.append((comm_cfg, train_cfg))
            elif role == "client":
                self.client_configs.append((comm_cfg, train_cfg))
            else:
                raise ValueError(f"Invalid role: {role}. Must be 'server' or 'client'")

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
    ) -> List[Tuple[CommunicationConfig, TrainingConfig]]:
        """
        解析配置，返回配置列表

        Args:
            config: 配置输入

        Returns:
            List[Tuple[CommunicationConfig, TrainingConfig]]
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
            # Single Tuple[CommunicationConfig, TrainingConfig]
            comm_config, train_config = config
            if not isinstance(comm_config, CommunicationConfig):
                raise TypeError("First element must be CommunicationConfig")
            if not isinstance(train_config, TrainingConfig):
                raise TypeError("Second element must be TrainingConfig")
            return [config]

        else:
            raise TypeError(
                "config must be one of: str (file/folder path), List[str], "
                "Tuple[CommunicationConfig, TrainingConfig], or List[Tuple[...]]"
            )

    def _load_from_path(self, path: str) -> List[Tuple[CommunicationConfig, TrainingConfig]]:
        """
        从路径加载配置（支持文件或文件夹）

        Args:
            path: 配置文件或文件夹路径

        Returns:
            配置列表
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config path not found: {path}")

        # 情况1: 路径是文件
        if os.path.isfile(path):
            self.logger.info(f"Loading config from file: {path}")
            return [ConfigLoader.load(path)]

        # 情况2: 路径是文件夹
        elif os.path.isdir(path):
            self.logger.info(f"Loading configs from folder: {path}")

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
                    config_tuple = ConfigLoader.load(config_file)
                    result.append(config_tuple)
                    comm_cfg, _ = config_tuple
                    self.logger.info(
                        f"  ✓ Loaded: {os.path.basename(config_file)} "
                        f"(role={comm_cfg.role}, mode={comm_cfg.mode})"
                    )
                except Exception as e:
                    self.logger.error(f"  ✗ Failed to load {os.path.basename(config_file)}: {e}")
                    raise ValueError(f"Failed to load config file: {config_file}") from e

            self.logger.info(f"✅ Loaded {len(result)} config(s) from folder")
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
            for i, (comm_cfg, train_cfg) in enumerate(self.server_configs):
                server = FederationServer(comm_cfg, train_cfg)
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
            for i, (comm_cfg, train_cfg) in enumerate(self.client_configs):
                client = FederationClient(comm_cfg, train_cfg)
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
                _, train_cfg = self.config_list[0]
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
                _, train_cfg = self.config_list[0]
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
        for comm_cfg, _ in self.config_list:
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
