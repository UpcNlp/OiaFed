"""
配置驱动的实验运行器

支持从 YAML 配置文件运行联邦学习实验
每个节点独立配置和运行

设计原则：
- 入口是一个类（FederationRunner）
- 支持单个配置文件或文件夹输入
- 统一的节点启动流程
- 只使用配置类，不使用字典操作
"""

# 必须在所有 import 之前设置，解决 protobuf 兼容性问题
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import asyncio
import atexit
import signal
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict, List, Union

from .config import (
    ConfigManager,
    NodeConfig,
    LogConfig,
)
from .core.system import FederatedSystem
from .infra import get_module_logger
from .infra.logging import setup_logging

# 导入 methods 模块以触发组件注册
from . import methods
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

logger = get_module_logger(__name__)


def _run_single_node(config_path: str, adjusted_min_peers: int = None) -> None:
    """
    子进程入口：运行单个节点

    Args:
        config_path: 配置文件路径
        adjusted_min_peers: 调整后的 min_peers 值（多进程模式下传递）
    """
    import sys
    from pathlib import Path

    src_dir = Path(__file__).parent
    project_root = src_dir.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # 导入 methods 模块以触发组件注册
    from . import methods  # noqa: F401

    runner = FederationRunner(config_path)

    # 应用主进程调整的 min_peers（如果有）
    if adjusted_min_peers is not None and runner.configs:
        for config in runner.configs:
            if config.is_trainer():
                old_min_peers = config.min_peers
                config.min_peers = adjusted_min_peers
                logger.info(
                    f"[子进程应用] {config.node_id}: "
                    f"min_peers 从 {old_min_peers} 设置为 {adjusted_min_peers}"
                )

    runner.run_sync()


class FederationRunner:
    """
    联邦学习运行器

    职责：
    - 统一的节点启动入口
    - 支持单配置文件或配置文件夹
    - 管理节点生命周期
    - 根据配置自动选择执行模式

    使用方式：
        # 方式1：单个配置文件
        runner = FederationRunner("configs/trainer.yaml")
        await runner.run()

        # 方式2：配置文件夹
        runner = FederationRunner("configs/experiment/")
        await runner.run()

        # 方式3：同步运行
        runner = FederationRunner("configs/trainer.yaml")
        runner.run_sync()
    """

    def __init__(
        self,
        config_path: Union[str, Path],
        auto_generate_run_name: bool = True,
        port_offset: int = 0,
    ):
        """
        初始化运行器

        Args:
            config_path: 配置文件路径或文件夹路径
            auto_generate_run_name: 是否自动生成 run_name
            port_offset: 端口偏移量，用于并发运行多个实验（默认为0）
        """
        self.config_path = Path(config_path)
        self.configs: List[NodeConfig] = []
        self.config_paths: List[Path] = []
        self.systems: List[FederatedSystem] = []
        self._running = False
        self.port_offset = port_offset

        # 使用配置管理器
        self._config_manager = ConfigManager(
            auto_generate_run_name=auto_generate_run_name
        )

        # 加载配置
        self._load_configs()

        # *** 先初始化日志，确保后续操作的日志能被输出 ***
        if self.configs:
            self._setup_global_logging()

        # *** 自动调整 min_peers（日志已初始化，可以正常输出）***
        self._auto_adjust_min_peers()

        # 应用端口偏移
        if self.port_offset > 0:
            self._apply_port_offset()

    def _load_configs(self) -> None:
        """加载配置文件"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config path not found: {self.config_path}")

        if self.config_path.is_file():
            logger.debug(f"Loading single config file: {self.config_path}")
            config = self._config_manager.load(str(self.config_path))
            self.configs.append(config)
            self.config_paths.append(self.config_path)
            logger.debug(f"Loaded config for node: {config.node_id}")

        elif self.config_path.is_dir():
            logger.debug(f"Loading configs from directory: {self.config_path}")

            yaml_files = sorted(
                list(self.config_path.glob("*.yaml"))
                + list(self.config_path.glob("*.yml"))
            )

            if not yaml_files:
                raise ValueError(f"No YAML config files found in: {self.config_path}")

            for yaml_file in yaml_files:
                if yaml_file.name.lower() in ["base.yaml", "base.yml"]:
                    logger.debug(f"Skipping base config file: {yaml_file.name}")
                    continue

                try:
                    config = self._config_manager.load(str(yaml_file))
                    self.configs.append(config)
                    self.config_paths.append(yaml_file)
                    logger.debug(f"Loaded config for node: {config.node_id}")
                except Exception as e:
                    logger.error(f"Failed to load {yaml_file.name}: {e}")
                    raise

            if not self.configs:
                raise ValueError(f"No valid node configs found in: {self.config_path}")

            logger.debug(f"Loaded {len(self.configs)} configs from directory")

        else:
            raise ValueError(f"Config path must be a file or directory: {self.config_path}")

        # 验证配置
        for config in self.configs:
            self._config_manager.validate(config)
            logger.debug(f"Config validated: {config.node_id}")

    def _auto_adjust_min_peers(self) -> None:
        """
        自动设置 Trainer 的 min_peers 配置

        规则：
        - 如果 min_peers = 0（或未设置），自动设置为 learner 数量
        - 如果 min_peers > 0，遵循配置文件，不自动调整

        设计原则：
        - 尊重用户配置：如果用户明确设置了 min_peers，就使用用户的值
        - 提供便利：如果用户没有设置，自动检测并设置
        """
        # 统计 learner 数量
        learner_count = sum(1 for config in self.configs if config.is_learner() and not config.is_trainer())

        # 找到 trainer 配置
        trainer_configs = [config for config in self.configs if config.is_trainer()]

        if not trainer_configs:
            return  # 没有 trainer，不需要调整

        if learner_count == 0:
            return  # 没有 learner，不需要调整

        # 处理每个 trainer 的 min_peers
        for trainer_config in trainer_configs:
            old_min_peers = trainer_config.min_peers

            # 只有未设置（0）时才自动设置
            if old_min_peers == 0:
                trainer_config.min_peers = learner_count
                logger.info(
                    f"[自动检测] {trainer_config.node_id}: "
                    f"min_peers 未设置，自动设置为 {learner_count} "
                    f"(检测到 {learner_count} 个 learner 配置)"
                )
            else:
                # 已设置，遵循配置文件
                logger.info(
                    f"[配置检查] {trainer_config.node_id}: "
                    f"使用配置文件中的 min_peers={old_min_peers} "
                    f"(检测到 {learner_count} 个 learner 配置)"
                )

                # 提示潜在问题（但不强制修改）
                if old_min_peers > learner_count:
                    logger.warning(
                        f"[配置提示] min_peers={old_min_peers} 大于实际 learner 数量 {learner_count}，"
                        f"可能导致 trainer 永久等待"
                    )
                elif old_min_peers < learner_count:
                    logger.warning(
                        f"[配置提示] min_peers={old_min_peers} 小于实际 learner 数量 {learner_count}，"
                        f"部分 learner 可能不会参与训练"
                    )

    def _apply_port_offset(self) -> None:
        """应用端口偏移量以避免并发实验的端口冲突"""
        logger.info(f"Applying port offset: {self.port_offset}")

        for config in self.configs:
            # 更新 listen 端口
            if config.listen and isinstance(config.listen, dict) and 'port' in config.listen:
                old_port = config.listen['port']
                config.listen['port'] += self.port_offset
                logger.debug(f"[{config.node_id}] Listen port: {old_port} -> {config.listen['port']}")

            # 更新 connect_to 中的端口
            if config.connect_to:
                updated_connect_to = []
                for connection in config.connect_to:
                    if '@' in connection:
                        # 格式: "trainer@localhost:50051"
                        node_id, address = connection.split('@', 1)
                        if ':' in address:
                            host, port_str = address.rsplit(':', 1)
                            try:
                                old_port = int(port_str)
                                new_port = old_port + self.port_offset
                                new_connection = f"{node_id}@{host}:{new_port}"
                                updated_connect_to.append(new_connection)
                                logger.debug(f"[{config.node_id}] Connect to {node_id}: {old_port} -> {new_port}")
                            except ValueError:
                                # 端口不是数字，保持原样
                                updated_connect_to.append(connection)
                        else:
                            # 没有端口，保持原样
                            updated_connect_to.append(connection)
                    else:
                        # 只有节点ID，保持原样
                        updated_connect_to.append(connection)

                config.connect_to = updated_connect_to

    def _setup_global_logging(self) -> None:
        """初始化全局日志系统"""
        first_config = self.configs[0]
        log_config = first_config.logging or LogConfig()
        setup_logging(node_id=first_config.node_id, log_config=log_config)

        logger.info(f"Experiment: {first_config.exp_name}")
        if first_config.run_name:
            logger.info(f"Run: {first_config.run_name}")
        logger.info(f"Log directory: {first_config.log_dir}")

    def _get_execution_mode(self) -> str:
        """检测执行模式"""
        if len(self.configs) == 1:
            return "single_process"

        if all(c.transport.mode == "memory" for c in self.configs):
            return "single_process"

        if any(c.transport.mode == "grpc" for c in self.configs):
            return "multi_process"

        return "single_process"

    async def run(self) -> Dict[str, Any]:
        """运行联邦学习实验"""
        mode = self._get_execution_mode()
        logger.debug(f"Execution mode: {mode}")

        if mode == "multi_process":
            return self._run_multi_process()
        else:
            return await self._run_single_process()

    def _run_multi_process(self) -> Dict[str, Any]:
        """多进程模式（改进版：分批启动 + 启动确认）"""
        logger.debug(f"Starting {len(self.configs)} nodes in separate processes...")

        ctx = get_context("spawn")
        processes = []

        # *** 收集调整后的 min_peers（用于传递给子进程）***
        adjusted_min_peers_map = {}
        for config in self.configs:
            if config.is_trainer():
                adjusted_min_peers_map[config.node_id] = config.min_peers

        def cleanup_processes():
            logger.debug("Cleaning up child processes...")
            for p in processes:
                if p.is_alive():
                    logger.warning(f"Terminating process: {p.name}")
                    p.terminate()

            import time
            deadline = time.time() + 5
            for p in processes:
                if p.is_alive():
                    timeout = max(0, deadline - time.time())
                    p.join(timeout=timeout)
                    if p.is_alive():
                        logger.error(f"Force killing process: {p.name}")
                        p.kill()
                        p.join()

            logger.debug("All child processes cleaned up")

        atexit.register(cleanup_processes)

        def signal_handler(signum, frame):
            logger.debug(f"Received signal {signum}, cleaning up...")
            cleanup_processes()
            import sys
            sys.exit(1)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        try:
            # *** 改进1：创建所有进程 ***
            for config_path, config in zip(self.config_paths, self.configs):
                logger.debug(f"Spawning process for node: {config.node_id}")

                # 获取调整后的 min_peers（仅对 trainer）
                adjusted_min_peers = adjusted_min_peers_map.get(config.node_id, None)

                p = ctx.Process(
                    target=_run_single_node,
                    args=(str(config_path), adjusted_min_peers),
                    name=f"node-{config.node_id}",
                )
                processes.append(p)

            # *** 改进2：分批启动（每批启动后等待短暂时间）***
            import time
            batch_size = 5  # 每批启动5个进程
            stagger_delay = 0.5  # 每批之间延迟0.5秒

            logger.info(f"分批启动进程（每批 {batch_size} 个，间隔 {stagger_delay}秒）...")

            for i in range(0, len(processes), batch_size):
                batch = processes[i:i+batch_size]
                batch_names = [p.name for p in batch]
                logger.debug(f"启动批次 {i//batch_size + 1}: {batch_names}")

                for p in batch:
                    p.start()
                    logger.debug(f"Process started: {p.name} (pid={p.pid})")

                # 批次之间短暂延迟
                if i + batch_size < len(processes):
                    time.sleep(stagger_delay)

            # *** 改进3：启动后检查进程状态 ***
            logger.info("等待所有进程启动...")
            time.sleep(2)  # 给进程一些启动时间

            alive_processes = [p for p in processes if p.is_alive()]
            dead_processes = [p for p in processes if not p.is_alive()]

            logger.info(f"启动完成: {len(alive_processes)}/{len(processes)} 个进程运行中")

            # *** 关键改进：如果有进程启动失败，报错并退出 ***
            if dead_processes:
                failed_names = [p.name for p in dead_processes]
                failed_info = [(p.name, p.exitcode) for p in dead_processes]

                logger.error(f"❌ 启动失败！{len(dead_processes)}/{len(processes)} 个进程未能启动")
                logger.error(f"失败的进程: {failed_names}")
                logger.error(f"退出码: {failed_info}")

                # 清理已启动的进程
                logger.info("正在清理已启动的进程...")
                cleanup_processes()

                # 抛出异常，停止实验
                raise RuntimeError(
                    f"进程启动失败：{len(dead_processes)} 个进程未能启动。"
                    f"失败的进程: {failed_names}"
                )

            # 所有进程都启动成功
            logger.info("✓ 所有进程启动成功")

            # *** 改进4：等待所有进程完成 ***
            for p in processes:
                p.join()
                logger.debug(f"Process finished: {p.name} (exitcode={p.exitcode})")

            failed = [p for p in processes if p.exitcode != 0]
            if failed:
                logger.warning(f"进程异常退出: {[p.name for p in failed]}")

            logger.debug("All processes completed")
            return {}

        except KeyboardInterrupt:
            logger.debug("Interrupted by user")
            cleanup_processes()
            raise
        except Exception as e:
            logger.error(f"Error in multi-process mode: {e}")
            cleanup_processes()
            raise
        finally:
            atexit.unregister(cleanup_processes)

    async def _run_single_process(self) -> Dict[str, Any]:
        """单进程模式"""
        if self._running:
            raise RuntimeError("Runner is already running")

        self._running = True
        results = {}

        try:
            # 1. 创建 FederatedSystem 实例
            logger.debug("Creating FederatedSystem instances...")
            for config in self.configs:
                system = FederatedSystem(config)
                self.systems.append(system)
                logger.debug(f"Created system: {config.node_id}")

            logger.debug(f"Created {len(self.systems)} systems")

            # 2. 并行初始化
            logger.debug("Initializing all nodes...")
            init_tasks = [system.initialize() for system in self.systems]
            await asyncio.gather(*init_tasks)
            logger.debug("All nodes initialized")

            # 3. 并发运行
            logger.debug("Starting all nodes...")
            run_tasks = [asyncio.create_task(system.run()) for system in self.systems]

            trainer_indices = [
                i for i, config in enumerate(self.configs)
                if config.is_trainer()
            ]
            learner_indices = [
                i for i, config in enumerate(self.configs)
                if not config.is_trainer()
            ]

            if trainer_indices:
                logger.debug(
                    f"Waiting for Trainer(s): "
                    f"{[self.configs[i].node_id for i in trainer_indices]}"
                )
                trainer_tasks = [run_tasks[i] for i in trainer_indices]
                learner_tasks = [run_tasks[i] for i in learner_indices]

                trainer_results = await asyncio.gather(
                    *trainer_tasks, return_exceptions=True
                )
                logger.debug("All Trainers completed")

                logger.debug("Waiting for Learners to shutdown...")
                learner_results = await asyncio.gather(
                    *learner_tasks, return_exceptions=True
                )
                logger.debug("All Learners shutdown")

                all_results = [None] * len(self.systems)
                for i, result in zip(trainer_indices, trainer_results):
                    all_results[i] = result
                for i, result in zip(learner_indices, learner_results):
                    all_results[i] = (
                        result if not isinstance(result, asyncio.CancelledError) else {}
                    )
            else:
                all_results = await asyncio.gather(*run_tasks, return_exceptions=True)

            # 4. 收集结果
            for system, result in zip(self.systems, all_results):
                node_id = system.node_id
                if isinstance(result, Exception):
                    logger.error(f"Node {node_id} failed: {result}")
                    results[node_id] = {"error": str(result)}
                else:
                    results[node_id] = result

            logger.debug("All nodes completed")
            return results

        except KeyboardInterrupt:
            logger.debug("Interrupted by user")
            return {}
        except Exception as e:
            logger.error(f"Error during execution: {e}")
            raise
        finally:
            logger.debug("Stopping all nodes...")
            await self._stop_all_systems()
            self._running = False
            logger.debug("Runner stopped")

    async def _stop_all_systems(self):
        """停止所有系统"""
        for system in self.systems:
            try:
                await system.stop()
                logger.debug(f"Stopped: {system.node_id}")
            except Exception as e:
                logger.error(f"Error stopping {system.node_id}: {e}")

    def run_sync(self) -> Dict[str, Any]:
        """同步运行"""
        return asyncio.run(self.run())

    def get_node_ids(self) -> List[str]:
        """获取所有节点 ID"""
        return [config.node_id for config in self.configs]

    def get_experiment_info(self) -> Dict[str, Any]:
        """获取实验信息"""
        if not self.configs:
            return {}

        first_config = self.configs[0]
        return {
            "exp_name": first_config.exp_name,
            "run_name": first_config.run_name,
            "log_dir": first_config.log_dir,
            "num_nodes": len(self.configs),
            "node_ids": self.get_node_ids(),
            "execution_mode": self._get_execution_mode(),
        }

    def __repr__(self) -> str:
        """字符串表示"""
        mode = self._get_execution_mode()
        if self.configs:
            exp_name = self.configs[0].exp_name
            return (
                f"FederationRunner("
                f"exp={exp_name}, "
                f"nodes={len(self.configs)}, "
                f"mode={mode})"
            )
        return f"FederationRunner(config_path={self.config_path}, nodes=0)"


# ========== 便捷函数 ==========


def run_experiment(
    config_path: Union[str, Path],
    auto_generate_run_name: bool = True,
) -> Dict[str, Any]:
    """
    运行联邦学习实验

    Args:
        config_path: 配置文件或文件夹路径
        auto_generate_run_name: 是否自动生成 run_name

    Returns:
        实验结果字典
    """
    runner = FederationRunner(
        config_path,
        auto_generate_run_name=auto_generate_run_name,
    )
    return runner.run_sync()


def create_runner(
    config_path: Union[str, Path],
    **kwargs,
) -> FederationRunner:
    """
    创建运行器实例

    Args:
        config_path: 配置文件或文件夹路径
        **kwargs: 其他参数

    Returns:
        FederationRunner 实例
    """
    return FederationRunner(config_path, **kwargs)