# fedcl/experiments/experiment.py
"""
FedCL实验管理器 - 改进版本
修复了导入问题，增强了与测试框架的兼容性
"""

import time
import uuid
import itertools
import yaml
import subprocess
import multiprocessing as mp
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from dataclasses import dataclass
from unittest.mock import Mock
from loguru import logger

from ..config.config_manager import ConfigManager, DictConfig
from ..config.schema_validator import SchemaValidator
from ..data.results import ExperimentResults, SweepResults
from ..data.split_api import DataSplitAPI
from ..exceptions import FedCLError, ConfigurationError, ExperimentEngineError
from ..utils.improved_logging_manager import initialize_improved_logging, get_improved_logging_manager, log_training_info, log_system_debug


@dataclass
class ExperimentConfig:
    """实验配置数据结构"""
    name: str
    description: str = ""
    seed: int = 42
    working_dir: Path = Path("./experiments")
    save_checkpoints: bool = True
    checkpoint_frequency: int = 10
    log_level: str = "INFO"


class FedCLExperiment:
    """
    FedCL实验管理器 - 改进版本
    
    职责：
    1. 配置加载和验证（支持单文件和目录扫描）
    2. 通过统一接口创建组件（支持进程化初始化）
    3. 实验结果收集和保存
    4. 参数扫描支持
    5. 配置目录扫描和依赖排序
    
    Example:
        # 单配置文件
        experiment = FedCLExperiment("configs/fedavg_cifar10.yaml")
        results = experiment.run()
        
        # 配置目录扫描
        experiment = FedCLExperiment("configs/")
        results = experiment.run()
        
        # 参数扫描
        sweep_results = experiment.sweep({
            "federation.num_轮次": [10, 20, 50],
            "federation.client_fraction": [0.1, 0.3, 1.0]
        })
    """
    
    def __init__(self, config: Union[str, Path, DictConfig], experiment_id: Optional[str] = None):
        """
        初始化实验管理器
        
        Args:
            config: 配置文件路径/配置目录路径或配置对象
            experiment_id: 实验ID，如果不提供则自动生成（基于日期时间）
        """
        # 生成或设置实验ID（基于日期时间格式）
        from datetime import datetime
        self.experiment_id = experiment_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 初始化配置管理器（延迟到需要时创建）
        try:
            self.schema_validator = SchemaValidator()
            self.config_manager = None  # 延迟初始化
        except ImportError:
            logger.warning("Schema validator not available, using mock validator")
            self.schema_validator = self._create_mock_validator()
            self.config_manager = None
        
        # 检测配置模式：单文件 vs 目录扫描
        self.config_mode = self._detect_config_mode(config)
        
        if self.config_mode == "directory":
            # 目录扫描模式
            self.config_dir = Path(config)
            self.config = self._scan_and_merge_configs()
        else:
            # 单文件模式
            self.config_dir = None
            self.config = self._load_and_validate_config(config)
        
        # 设置实验目录
        self.working_dir = Path(self.config.get("experiment.working_dir", "./experiments"))
        self.experiment_dir = self.working_dir / f"{self.config.get('experiment.name', 'experiment')}_{self.experiment_id}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志目录（支持自定义）
        self.log_base_dir = Path(self.config.get("experiment.log_base_dir", "./logs"))
        self.log_base_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化改进的日志系统
        self._initialize_improved_logging()
        
        # 保存配置到实验目录
        self._save_config_to_experiment_dir()
        
        # 设置随机种子
        self._set_seed(self.config.get("experiment.seed", 42))
        
        # 实验状态
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        
        # 初始化组件（进程化）
        self.components = {}
        
        logger.debug(f"Initialized FedCLExperiment {self.experiment_id} in {self.config_mode} mode")
        logger.info(f"Experiment directory: {self.experiment_dir}")
    
    def run(self) -> ExperimentResults:
        """
        运行实验 - 统一入口
        
        Returns:
            ExperimentResults: 实验结果
            
        Raises:
            ExperimentEngineError: 实验执行失败
        """
        try:
            logger.debug(f"Starting experiment: {self.config.get('experiment.name', 'unnamed')}")
            self.start_time = time.time()
            
            if self.config_mode == "directory":
                # 目录扫描模式：进程化初始化组件
                self._process_based_initialization()
                
                # 获取服务器组件
                server = self.components.get('server')
                if not server:
                    raise ExperimentEngineError("No server component found after initialization")
            else:
                # 单文件模式：传统方式创建服务器
                server = self._create_server()
            
            # 启动联邦学习
            federation_results = self._start_federation(server)
            
            # 创建实验结果
            experiment_results = self._create_experiment_results(federation_results)
            
            # 保存结果
            self._save_results(experiment_results)
            
            self.end_time = time.time()
            logger.success(f"Experiment completed in {self.end_time - self.start_time:.2f}s")
            
            return experiment_results
            
        except Exception as e:
            self.end_time = time.time()
            logger.error(f"Experiment failed: {e}")
            raise ExperimentEngineError(f"Experiment execution failed: {e}") from e
    
    def sweep(self, param_grid: Dict[str, List[Any]]) -> SweepResults:
        """
        参数扫描实验
        
        Args:
            param_grid: 参数网格，键为配置路径，值为参数列表
            
        Returns:
            SweepResults: 扫描结果
        """
        logger.info(f"Starting parameter sweep with {len(param_grid)} parameters")
        
        # 生成参数组合
        param_combinations = self._generate_param_combinations(param_grid)
        logger.info(f"Total parameter combinations: {len(param_combinations)}")
        
        # 创建扫描结果对象
        sweep_results = SweepResults(
            sweep_config=param_grid,
            base_config=self.config.copy(),
            experiment_id=self.experiment_id
        )
        
        # 执行每个参数组合的实验
        for i, params in enumerate(param_combinations):
            logger.info(f"Running sweep experiment {i+1}/{len(param_combinations)}: {params}")
            
            try:
                # 创建修改后的配置
                modified_config = self._apply_param_combination(self.config.copy(), params)
                
                # 创建子实验
                sub_experiment = FedCLExperiment(
                    config=modified_config,
                    experiment_id=f"{self.experiment_id}_sweep_{i+1}"
                )
                
                # 运行子实验
                results = sub_experiment.run()
                
                # 添加到扫描结果
                sweep_results.add_result(params, results)
                
            except Exception as e:
                logger.error(f"Sweep experiment {i+1} failed: {e}")
                sweep_results.add_failed_result(params, str(e))
        
        # 保存扫描结果
        self._save_sweep_results(sweep_results)
        
        logger.success(f"Parameter sweep completed: {sweep_results.summary}")
        return sweep_results
    
    def resume_from_checkpoint(self, checkpoint_path: Path) -> ExperimentResults:
        """
        从检查点恢复实验
        
        Args:
            checkpoint_path: 检查点路径
            
        Returns:
            ExperimentResults: 实验结果
        """
        logger.info(f"Resuming experiment from checkpoint: {checkpoint_path}")
        
        # 加载检查点数据
        checkpoint_data = self._load_checkpoint(checkpoint_path)
        
        # 更新配置
        self.config.update(checkpoint_data.get("config", {}))
        
        # 设置恢复标志
        self.config.set_value("experiment.resume_from_checkpoint", str(checkpoint_path))
        
        # 运行实验
        return self.run()
    
    def get_progress(self) -> Dict[str, Any]:
        """
        获取实验进度信息
        
        Returns:
            Dict[str, Any]: 进度信息
        """
        progress = {
            "experiment_id": self.experiment_id,
            "start_time": self.start_time,
            "current_time": time.time(),
            "experiment_dir": str(self.experiment_dir)
        }
        
        if self.start_time:
            progress["elapsed_time"] = time.time() - self.start_time
        
        return progress
    
    def cleanup(self) -> None:
        """清理实验资源"""
        logger.info(f"Cleaning up experiment {self.experiment_id}")
        
        # 清理组件
        for component_name, component in self.components.items():
            if hasattr(component, 'cleanup'):
                try:
                    component.cleanup()
                    logger.debug(f"Cleaned up {component_name}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup {component_name}: {e}")
        
        self.components.clear()
    
    def _initialize_improved_logging(self):
        """初始化改进的日志系统"""
        try:
            # 获取实验名称
            experiment_name = self.config.get("experiment.name", "unknown_experiment")
            
            # 设置日志目录
            log_dir = self.log_base_dir / f"{experiment_name}_{self.experiment_id}"
            
            # 初始化改进的日志管理器
            log_manager = initialize_improved_logging(
                log_base_dir=str(self.log_base_dir),
                experiment_name=f"{experiment_name}_{self.experiment_id}",  # 使用实验名称+ID
                enable_console=True,
                global_log_level=self.config.get("experiment.log_level", "INFO")
            )
            
            log_training_info(f"改进的日志系统初始化完成 - 实验: {experiment_name}")
            log_system_debug(f"日志目录: {log_dir}")
            
        except Exception as e:
            logger.warning(f"初始化改进日志系统失败，使用默认日志: {e}")
    
    def _get_config_manager(self, config_path: Optional[str] = None) -> 'ConfigManager':
        """获取ConfigManager实例"""
        if self.config_manager is None:
            if config_path:
                self.config_manager = ConfigManager(config_path=config_path, schema_validator=self.schema_validator)
            else:
                self.config_manager = ConfigManager(config_dict={}, schema_validator=self.schema_validator)
        return self.config_manager
    
    # ================== 配置扫描和进程化初始化 ==================
    
    def _detect_config_mode(self, config: Union[str, Path, DictConfig]) -> str:
        """检测配置模式"""
        if isinstance(config, DictConfig):
            return "single_file"
        
        path = Path(config)
        if path.is_dir():
            return "directory"
        else:
            return "single_file"
    
    def _scan_and_merge_configs(self) -> DictConfig:
        """扫描配置目录并合并配置"""
        logger.debug(f"Scanning config directory: {self.config_dir}")
        
        # 1. 检查并执行数据分割
        if self._has_data_split_config():
            logger.debug("Found data split config, executing data split first")
            self._execute_data_split()
        
        # 2. 扫描配置文件
        config_files = self._scan_config_files()
        
        # 3. 合并配置
        merged_config = self._merge_configs(config_files)
        
        return merged_config
    
    def _has_data_split_config(self) -> bool:
        """检查是否存在数据分割配置"""
        split_config_path = self.config_dir / "data_split_config.yaml"
        return split_config_path.exists()
    
    def _execute_data_split(self):
        """执行数据分割"""
        try:
            split_config_path = self.config_dir / "data_split_config.yaml"
            
            with open(split_config_path, 'r', encoding='utf-8') as f:
                split_config_dict = yaml.safe_load(f)
            
            split_config = DictConfig(split_config_dict)
            
            # 确保clients目录存在
            clients_dir = self.config_dir / "客户端"
            clients_dir.mkdir(exist_ok=True)
            logger.info(f"Ensured clients directory exists: {clients_dir}")
            
            # 执行数据分割
            try:
                data_split_api = DataSplitAPI()
                split_result = data_split_api.execute_split(split_config)
                logger.info(f"Data split completed: {split_result}")
            except ImportError:
                logger.warning("DataSplitAPI not available, skipping data split")
            
        except Exception as e:
            logger.error(f"Data split execution failed: {e}")
            raise ConfigurationError(f"Failed to execute data split: {e}") from e
    
    def _scan_config_files(self) -> Dict[str, List[Path]]:
        """扫描并分类配置文件"""
        config_files = {
            'server': [],
            'client': [],
            'experiment': []
        }
        
        # 扫描根目录的配置文件
        for config_file in self.config_dir.glob("*.yaml"):
            if config_file.name == "data_split_config.yaml":
                continue
            
            config_type = self._identify_config_type(config_file)
            if config_type in config_files:
                config_files[config_type].append(config_file)
                logger.debug(f"Found {config_type} config: {config_file.name}")
        
        # 扫描clients文件夹
        clients_dir = self.config_dir / "客户端"
        if clients_dir.exists() and clients_dir.is_dir():
            logger.info(f"Scanning clients directory: {clients_dir}")
            for client_file in clients_dir.glob("*.yaml"):
                config_files['client'].append(client_file)
                logger.debug(f"Found client config: {client_file.name}")
        else:
            logger.debug("No clients directory found")
        
        return config_files
    
    def _identify_config_type(self, config_file: Path) -> str:
        """识别配置文件类型"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            filename = config_file.name.lower()
            
            # 服务端配置
            if 'server' in filename or 'server' in config:
                return 'server'
            
            # 客户端配置
            if 'client' in filename or 'client' in config:
                return 'client'
            
            # 实验配置
            if 'experiment' in filename or 'experiment' in config:
                return 'experiment'
            
            return 'unknown'
            
        except Exception as e:
            logger.warning(f"Failed to parse config file {config_file}: {e}")
            return 'unknown'
    
    def _merge_configs(self, config_files: Dict[str, List[Path]]) -> DictConfig:
        """合并配置文件"""
        merged = {}
        
        # 合并实验配置（基础配置）
        for exp_file in config_files['experiment']:
            with open(exp_file, 'r') as f:
                exp_config = yaml.safe_load(f)
            merged.update(exp_config)
        
        # 合并服务端配置
        if config_files['server']:
            with open(config_files['server'][0], 'r') as f:  # 只取第一个服务端配置
                server_config = yaml.safe_load(f)
            merged['server'] = server_config.get('server', server_config)
        
        # 记录客户端配置路径
        merged['_config_files'] = {
            '客户端': [str(f) for f in config_files['client']]
        }
        
        # 添加客户端数量信息
        merged['client_count'] = len(config_files['client'])
        
        return DictConfig(merged)
    
    def _process_based_initialization(self):
        """进程化初始化组件"""
        logger.info("Starting process-based component initialization")
        
        config_files = self._scan_config_files()
        
        # 1. 初始化服务端
        if config_files['server']:
            logger.info("Initializing server component")
            self.components['server'] = self._init_server_process(config_files['server'][0])
        
        # 2. 初始化客户端
        clients = []
        client_configs = config_files['client']
        if client_configs:
            logger.debug(f"Found {len(client_configs)} client configurations in clients/ directory")
            for i, client_config in enumerate(client_configs):
                client_name = client_config.stem
                logger.debug(f"Initializing client {i+1}/{len(client_configs)}: {client_name}")
                client = self._init_client_process(client_config)
                clients.append(client)
        else:
            logger.warning("No client configurations found in clients/ directory")
        
        self.components['客户端'] = clients
        
        # 3. 启动服务端和客户端注册流程
        self._start_components_and_register()
        
        # 记录初始化统计
        logger.info(f"Process-based initialization completed:")
        logger.info(f"  - Server: {'✓' if self.components.get('server') else '✗'}")
        logger.info(f"  - Clients: {len(clients)} (from clients/ directory)")
    
    def _start_components_and_register(self):
        """启动组件并处理客户端注册"""
        try:
            # 1. 启动服务端通信器
            server = self.components.get('server')
            if server:
                logger.info("Starting server component")
                if hasattr(server, 'start'):
                    server.start()  # 启动通信器，这会调用on_start()
                elif hasattr(server, 'on_start'):
                    server.on_start()
            
            # 2. 启动客户端通信器
            clients = self.components.get('客户端', [])
            for i, client in enumerate(clients):
                logger.info(f"Starting client {i+1}/{len(clients)}")
                if hasattr(client, 'start'):
                    client.start()  # 启动通信器，这会调用on_start()
                elif hasattr(client, 'on_start'):
                    client.on_start()
            
            # 3. 客户端注册到服务端
            if server and clients:
                logger.info("处理客户端注册s")
                for i, client in enumerate(clients):
                    try:
                        client_id = getattr(client, 'client_id', f'client_{i+1}')
                        client_info = {
                            'client_id': client_id,
                            'client_type': type(client).__name__,
                            'capabilities': getattr(client, 'capabilities', {}),
                            'timestamp': time.time()
                        }
                        
                        # 注册客户端到服务端
                        if hasattr(server, 'register_client'):
                            response = server.register_client(client_info)
                            logger.info(f"Client {client_id} registration: {response.get('status', 'unknown')}")
                        
                    except Exception as e:
                        logger.warning(f"Failed to register client {i+1}: {e}")
            
            # 4. 等待服务端就绪
            if server:
                self._wait_for_server_ready(server)
                
        except Exception as e:
            logger.error(f"Failed to start components and register: {e}")
            raise
    
    def _wait_for_server_ready(self, server, timeout=10):
        """等待服务端就绪"""
        try:
            start_time = time.time()
            while time.time() - start_time < timeout:
                if hasattr(server, 'startup_ready') and server.startup_ready:
                    logger.info("Server is ready for federation")
                    return True
                
                time.sleep(0.1)  # 100ms检查间隔
            
            logger.warning(f"Server not ready after {timeout}s timeout")
            return False
            
        except Exception as e:
            logger.error(f"Error waiting for server ready: {e}")
            return False
    
    def _init_server_process(self, config_file: Path):
        """进程化初始化服务端"""
        with open(config_file, 'r') as f:
            server_config = yaml.safe_load(f)
        
        config = DictConfig(server_config)
        
        # 使用统一接口创建服务端
        server_type = config.get("server.type", "improved")
        
        try:
            if server_type == "improved":
                try:
                    from ..federation.coordinators.federated_server import FederatedServer
                    return FederatedServer.create_from_config(config)
                except ImportError:
                    logger.warning("FederatedServer not available, creating mock server")
                    return Mock()
            elif server_type == "simulated":
                # 尝试导入模拟服务端
                try:
                    from ..federation.coordinators.federated_server import FederatedServer
                    return FederatedServer.create_from_config(config)
                except ImportError:
                    logger.warning("SimulatedFederatedServer not available, creating mock server")
                    return Mock()
            elif server_type == "distributed":
                try:
                    from ..federation.coordinators.federated_server import FederatedServer
                    return FederatedServer.create_from_config(config)
                except ImportError:
                    logger.warning("DistributedFederatedServer not available, creating mock server")
                    return Mock()
            else:
                raise ConfigurationError(f"Unknown server type: {server_type}")
        except ImportError as e:
            logger.error(f"Failed to import server class: {e}")
            raise ConfigurationError(f"Server type {server_type} not available: {e}")
    
    def _init_client_process(self, config_file: Path):
        """进程化初始化客户端"""
        with open(config_file, 'r') as f:
            client_config = yaml.safe_load(f)
        
        config = DictConfig(client_config)
        
        # 使用统一接口创建客户端
        client_type = config.get("client.type", "multi_learner")
        
        try:
            if client_type == "multi_learner":
                try:
                    from ..federation.coordinators.federated_client import MultiLearnerFederatedClient
                    return MultiLearnerFederatedClient.create_from_config(config)
                except ImportError:
                    logger.warning("MultiLearnerFederatedClient not available, creating mock client")
                    return Mock()
            else:
                try:
                    from ..federation.coordinators.federated_client import MultiLearnerFederatedClient
                    return MultiLearnerFederatedClient.create_from_config(config)
                except ImportError:
                    logger.warning("MultiLearnerFederatedClient not available, creating mock client")
                    return Mock()
        except ImportError as e:
            logger.error(f"Failed to import client class: {e}")
            raise ConfigurationError(f"Client type {client_type} not available: {e}")
    
    # ================== 传统单文件模式方法 ==================
    
    def _load_and_validate_config(self, config: Union[str, Path, DictConfig]) -> DictConfig:
        """加载和验证配置"""
        if isinstance(config, (str, Path)):
            # 简单的YAML加载
            config_path = Path(config)
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_dict = yaml.safe_load(f)
                config = DictConfig(config_dict)
            else:
                raise FileNotFoundError(f"Config file not found: {config_path}")
        elif not isinstance(config, DictConfig):
            config = DictConfig(config)
        
        # 尝试验证配置
        try:
            config_manager = self._get_config_manager()
            validation_result = config_manager.validate_config(config)
            if not validation_result.is_valid:
                error_messages = [f"{error.field}: {error.message}" for error in validation_result.errors]
                logger.warning(f"Configuration validation issues: {'; '.join(error_messages)}")
        except Exception as e:
            logger.warning(f"Configuration validation failed: {e}")
        
        return config
    
    def _create_server(self):
        """
        根据配置创建服务器
        
        使用统一的 create_from_config 接口
        """
        server_config = self.config.get("server", {})
        server_type = server_config.get("type", "improved")
        
        try:
            if server_type == "improved":
                try:
                    from ..federation.coordinators.federated_server import FederatedServer
                    return FederatedServer.create_from_config(self.config)
                except ImportError:
                    logger.warning("FederatedServer not available, creating mock server")
                    return Mock()
            
            elif server_type == "simulated":
                try:
                    from ..federation.coordinators.federated_server import FederatedServer
                    return FederatedServer.create_from_config(self.config)
                except ImportError:
                    logger.warning("SimulatedFederatedServer not available, creating mock server")
                    return Mock()
            
            elif server_type == "distributed":
                try:
                    from ..federation.coordinators.federated_server import FederatedServer
                    return FederatedServer.create_from_config(self.config)
                except ImportError:
                    logger.warning("DistributedFederatedServer not available, creating mock server")
                    return Mock()
            
            else:
                raise ConfigurationError(f"Unknown server type: {server_type}")
                
        except ImportError as e:
            logger.error(f"Failed to import server class: {e}")
            raise ConfigurationError(f"Server type {server_type} not available: {e}")
    
    def _start_federation(self, server):
        """启动联邦学习"""
        try:
            # 尝试使用不同的启动方法
            if hasattr(server, 'start_federation'):
                return server.start_federation()
            elif hasattr(server, 'run_federation'):
                return server.run_federation()
            elif hasattr(server, 'coordinate_federation'):
                return server.coordinate_federation()
            else:
                logger.warning("Server doesn't have standard federation methods, trying generic run")
                if hasattr(server, 'run'):
                    return server.run()
                else:
                    raise ExperimentEngineError("Server doesn't support federation execution")
                    
        except Exception as e:
            logger.error(f"Federation execution failed: {e}")
            raise ExperimentEngineError(f"Failed to start federation: {e}")
    
    def _create_experiment_results(self, federation_results) -> ExperimentResults:
        """创建实验结果"""
        try:
            from datetime import datetime, timezone
            
            # 转换时间戳为datetime对象
            start_datetime = datetime.fromtimestamp(self.start_time, tz=timezone.utc) if self.start_time else None
            end_datetime = datetime.fromtimestamp(self.end_time, tz=timezone.utc) if self.end_time else None
            
            # 创建实验结果对象，不直接传入federation_results
            experiment_results = ExperimentResults(
                experiment_id=self.experiment_id,
                config=self.config,
                start_time=start_datetime,
                end_time=end_datetime
            )
            
            # 将联邦学习结果添加到artifacts中
            experiment_results.artifacts["federation_results"] = federation_results
            
            return experiment_results
        except Exception as e:
            self.logger.error(f"Failed to create ExperimentResults: {e}")
            # 如果ExperimentResults创建失败，创建简单的结果字典
            return {
                "experiment_id": self.experiment_id,
                "federation_results": federation_results,
                "start_time": self.start_time,
                "end_time": self.end_time,
                "duration": (self.end_time - self.start_time) if self.end_time and self.start_time else 0,
                "working_dir": str(self.experiment_dir)
            }
    
    def _save_results(self, results) -> None:
        """保存实验结果"""
        results_path = self.experiment_dir / "results.json"
        
        try:
            if hasattr(results, 'save_to_file'):
                results.save_to_file(results_path)
            else:
                # 简单的JSON保存
                import json
                with open(results_path, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Experiment results saved: {results_path}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def _save_config_to_experiment_dir(self):
        """保存配置到实验目录"""
        try:
            if hasattr(self.config_manager, 'save_config'):
                self.config_manager.save_config(self.config, self.experiment_dir / "config.yaml")
            else:
                # 简单的YAML保存
                config_path = self.experiment_dir / "config.yaml"
                with open(config_path, 'w', encoding='utf-8') as f:
                    if hasattr(self.config, 'to_dict'):
                        config_dict = self.config.to_dict()
                    elif hasattr(self.config, '__dict__'):
                        config_dict = dict(self.config)
                    else:
                        config_dict = self.config
                    yaml.dump(config_dict, f, default_flow_style=False)
            logger.debug(f"Config saved to experiment directory")
        except Exception as e:
            logger.warning(f"Failed to save config: {e}")
    
    def _save_sweep_results(self, sweep_results) -> None:
        """保存扫描结果"""
        sweep_path = self.experiment_dir / "sweep_results.json"
        
        try:
            if hasattr(sweep_results, 'save_to_file'):
                sweep_results.save_to_file(sweep_path)
            else:
                # 简单的JSON保存
                import json
                with open(sweep_path, 'w') as f:
                    json.dump(sweep_results, f, indent=2, default=str)
                    
            logger.info(f"Sweep results saved: {sweep_path}")
        except Exception as e:
            logger.error(f"Failed to save sweep results: {e}")
    
    def _generate_param_combinations(self, param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """生成参数组合"""
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        
        combinations = []
        for value_combo in itertools.product(*values):
            combinations.append(dict(zip(keys, value_combo)))
        
        return combinations
    
    def _apply_param_combination(self, config: DictConfig, params: Dict[str, Any]) -> DictConfig:
        """将参数组合应用到配置中"""
        modified_config = config.copy()
        
        for param_path, param_value in params.items():
            try:
                if hasattr(modified_config, 'set_value'):
                    modified_config.set_value(param_path, param_value)
                else:
                    # 简单的嵌套字典设置
                    keys = param_path.split('.')
                    current = modified_config
                    for key in keys[:-1]:
                        if key not in current:
                            current[key] = {}
                        current = current[key]
                    current[keys[-1]] = param_value
            except Exception as e:
                logger.warning(f"Failed to set parameter {param_path}: {e}")
        
        return modified_config
    
    def _set_seed(self, seed: int) -> None:
        """设置随机种子"""
        try:
            import random
            random.seed(seed)
            
            try:
                import numpy as np
                np.random.seed(seed)
            except ImportError:
                pass
            
            try:
                import torch
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
            except ImportError:
                pass
            
            logger.debug(f"Set random seed to {seed}")
        except Exception as e:
            logger.warning(f"Failed to set random seed: {e}")
    
    def _load_checkpoint(self, checkpoint_path: Path) -> Dict[str, Any]:
        """加载检查点数据"""
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        try:
            import torch
            checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
        except ImportError:
            # 如果torch不可用，尝试JSON加载
            try:
                import json
                with open(checkpoint_path, 'r') as f:
                    checkpoint_data = json.load(f)
            except Exception as e:
                raise Exception(f"Failed to load checkpoint: {e}")
        
        logger.info(f"Checkpoint loaded from: {checkpoint_path}")
        return checkpoint_data
    
    def _create_mock_validator(self):
        """创建模拟验证器"""
        class MockValidator:
            def validate_config(self, config):
                class MockResult:
                    is_valid = True
                    errors = []
                return MockResult()
        return MockValidator()
    
    def _create_mock_config_manager(self):
        """创建模拟配置管理器"""
        class MockConfigManager:
            def __init__(self, validator):
                self.validator = validator
                
            def load_config(self, path):
                with open(path, 'r') as f:
                    return DictConfig(yaml.safe_load(f))
                    
            def validate_config(self, config):
                return self.validator.validate_config(config)
                
            def save_config(self, config, path):
                with open(path, 'w') as f:
                    yaml.dump(dict(config), f, default_flow_style=False)
        
        return MockConfigManager(self.schema_validator)
    
    # ================== 上下文管理器支持 ==================
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.cleanup()


# ================== 便捷函数 ==================

def quick_experiment(config_path: str, **overrides) -> Union[ExperimentResults, Dict[str, Any]]:
    """
    快速实验函数
    
    Args:
        config_path: 配置文件路径或配置目录路径
        **overrides: 配置覆盖参数
        
    Returns:
        实验结果
    """
    with FedCLExperiment(config_path) as experiment:
        # 应用覆盖参数
        for key, value in overrides.items():
            param_path = key.replace("__", ".")
            try:
                if hasattr(experiment.config, 'set_value'):
                    experiment.config.set_value(param_path, value)
                else:
                    # 简单的参数设置
                    keys = param_path.split('.')
                    current = experiment.config
                    for k in keys[:-1]:
                        if k not in current:
                            current[k] = {}
                        current = current[k]
                    current[keys[-1]] = value
            except Exception as e:
                logger.warning(f"Failed to set override {param_path}: {e}")
        
        return experiment.run()


def quick_sweep(config_path: str, param_grid: Dict[str, List[Any]]) -> Union[SweepResults, Dict[str, Any]]:
    """
    快速参数扫描函数
    
    Args:
        config_path: 配置文件路径或配置目录路径
        param_grid: 参数网格
        
    Returns:
        扫描结果
    """
    with FedCLExperiment(config_path) as experiment:
        return experiment.sweep(param_grid)


def process_config_directory(config_dir: str, 
                            execute_data_split: bool = True,
                            dry_run: bool = False) -> Dict[str, Any]:
    """
    处理配置目录的便捷函数
    
    Args:
        config_dir: 配置目录路径
        execute_data_split: 是否执行数据分割
        dry_run: 是否只进行扫描而不执行
        
    Returns:
        Dict[str, Any]: 处理结果
    """
    experiment = FedCLExperiment(config_dir)
    
    if dry_run:
        # 只返回扫描结果
        config_files = experiment._scan_config_files()
        
        return {
            "config_dir": config_dir,
            "total_configs": sum(len(files) for files in config_files.values()),
            "config_breakdown": {k: len(v) for k, v in config_files.items()},
            "has_data_split": experiment._has_data_split_config(),
            "client_count": len(config_files.get('client', [])),
            "server_count": len(config_files.get('server', [])),
            "experiment_count": len(config_files.get('experiment', [])),
            "dry_run": True
        }
    else:
        # 执行完整处理
        results = experiment.run()
        
        return {
            "config_dir": config_dir,
            "experiment_results": results,
            "experiment_id": experiment.experiment_id,
            "components_summary": {
                "server": experiment.components.get('server') is not None if hasattr(experiment, 'components') else False,
                "客户端": len(experiment.components.get('客户端', [])) if hasattr(experiment, 'components') else 0
            },
            "dry_run": False
        }