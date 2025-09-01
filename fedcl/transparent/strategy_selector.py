# fedcl/transparent/strategy_selector.py
"""
执行策略选择器

根据检测到的运行环境，自动选择最优的执行策略。
支持真联邦、伪联邦和本地模拟三种模式的智能切换。
"""

from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from loguru import logger

from .mode_detector import ExecutionMode, SystemResources, NetworkEnvironment


class StrategyType(Enum):
    """执行策略类型"""
    TRUE_FEDERATION = "true_federation"
    PSEUDO_FEDERATION = "pseudo_federation"
    LOCAL_SIMULATION = "local_simulation"
    HYBRID = "hybrid"


@dataclass
class ExecutionStrategy:
    """执行策略配置"""
    strategy_type: StrategyType
    num_processes: int
    communication_backend: str
    data_distribution: str
    resource_allocation: Dict[str, Any]
    optimization_config: Dict[str, Any]
    metadata: Dict[str, Any]


class StrategySelector:
    """执行策略选择器"""
    
    def __init__(self):
        self.logger = logger.bind(component="StrategySelector")
        self._strategy_cache: Dict[str, ExecutionStrategy] = {}
        self._mode_detector = None  # 将在需要时初始化
    
    def analyze_system_resources(self) -> Optional[SystemResources]:
        """分析系统资源"""
        if not self._mode_detector:
            from .mode_detector import ModeDetector
            self._mode_detector = ModeDetector()
        
        return self._mode_detector.analyze_system_resources()
    
    def select_strategy(
        self,
        mode: ExecutionMode,
        config: Dict[str, Any],
        system_resources: Optional[SystemResources] = None,
        network_env: Optional[NetworkEnvironment] = None,
        **kwargs
    ) -> ExecutionStrategy:
        """
        选择执行策略
        
        Args:
            mode: 检测到的执行模式
            config: 用户配置
            system_resources: 系统资源信息
            network_env: 网络环境信息
            **kwargs: 额外参数
            
        Returns:
            ExecutionStrategy: 选定的执行策略
        """
        self.logger.info(f"🎯 开始选择执行策略，模式: {mode.value}")
        
        # 如果没有提供系统资源信息，尝试自动检测
        if system_resources is None:
            system_resources = self.analyze_system_resources()
        
        # 创建缓存键
        cache_key = self._create_cache_key(mode, config, system_resources, network_env)
        
        # 检查缓存
        if cache_key in self._strategy_cache:
            self.logger.info("📋 使用缓存的策略配置")
            return self._strategy_cache[cache_key]
        
        # 根据模式选择策略
        if mode == ExecutionMode.TRUE_FEDERATION:
            strategy = self._create_true_federation_strategy(config, system_resources, network_env)
        elif mode == ExecutionMode.PSEUDO_FEDERATION:
            strategy = self._create_pseudo_federation_strategy(config, system_resources)
        elif mode == ExecutionMode.LOCAL_SIMULATION:
            strategy = self._create_local_simulation_strategy(config, system_resources)
        else:
            # 默认策略
            self.logger.warning(f"未知模式 {mode}，使用默认本地模拟策略")
            strategy = self._create_local_simulation_strategy(config, system_resources)
        
        # 优化策略配置
        strategy = self._optimize_strategy(strategy, system_resources, network_env)
        
        # 缓存策略
        self._strategy_cache[cache_key] = strategy
        
        self.logger.info(f"✅ 策略选择完成: {strategy.strategy_type.value}")
        self.logger.info(f"📊 进程数: {strategy.num_processes}, 通信后端: {strategy.communication_backend}")
        
        return strategy
    
    def _create_true_federation_strategy(
        self,
        config: Dict[str, Any],
        system_resources: Optional[SystemResources],
        network_env: Optional[NetworkEnvironment]
    ) -> ExecutionStrategy:
        """创建真联邦策略"""
        self.logger.info("🌐 创建真联邦策略")
        
        num_clients = config.get("num_clients", 3)
        
        # 选择通信后端
        communication_backend = self._select_communication_backend(network_env, "distributed")
        
        # 资源分配策略
        resource_allocation = {
            "client_cpu_cores": system_resources.cpu_cores // max(num_clients, 1) if system_resources else 1,
            "client_memory_gb": system_resources.memory_gb // max(num_clients, 1) if system_resources else 2,
            "use_gpu": system_resources.has_gpu if system_resources else False,
            "network_bandwidth": "auto",
        }
        
        # 优化配置
        optimization_config = {
            "compression": True,
            "quantization": network_env.bandwidth_mbps < 100 if network_env else False,
            "async_updates": True,
            "timeout_seconds": 300,
            "retry_attempts": 3,
        }
        
        return ExecutionStrategy(
            strategy_type=StrategyType.TRUE_FEDERATION,
            num_processes=num_clients + 1,  # 客户端 + 服务器
            communication_backend=communication_backend,
            data_distribution="federated",
            resource_allocation=resource_allocation,
            optimization_config=optimization_config,
            metadata={
                "requires_network": True,
                "scalability": "high",
                "fault_tolerance": "medium"
            }
        )
    
    def _create_pseudo_federation_strategy(
        self,
        config: Dict[str, Any],
        system_resources: Optional[SystemResources]
    ) -> ExecutionStrategy:
        """创建伪联邦策略"""
        self.logger.info("🖥️ 创建伪联邦策略")
        
        num_clients = config.get("num_clients", 3)
        
        # 根据系统资源调整进程数
        max_processes = system_resources.cpu_cores if system_resources else 4
        actual_processes = min(num_clients + 1, max_processes)
        
        # 资源分配策略
        resource_allocation = {
            "process_cpu_cores": max(1, system_resources.cpu_cores // actual_processes) if system_resources else 1,
            "process_memory_gb": max(1, system_resources.memory_gb // actual_processes) if system_resources else 1,
            "shared_memory_mb": 512,
            "use_multiprocessing": True,
        }
        
        # 优化配置
        optimization_config = {
            "ipc_method": "shared_memory",
            "data_sharing": True,
            "process_pool": True,
            "memory_efficient": True,
        }
        
        return ExecutionStrategy(
            strategy_type=StrategyType.PSEUDO_FEDERATION,
            num_processes=actual_processes,
            communication_backend="local_multiprocessing",
            data_distribution="simulated_federated",
            resource_allocation=resource_allocation,
            optimization_config=optimization_config,
            metadata={
                "requires_network": False,
                "scalability": "medium",
                "fault_tolerance": "high"
            }
        )
    
    def _create_local_simulation_strategy(
        self,
        config: Dict[str, Any],
        system_resources: Optional[SystemResources]
    ) -> ExecutionStrategy:
        """创建本地模拟策略"""
        self.logger.info("🏠 创建本地模拟策略")
        
        # 资源分配策略（单进程）
        resource_allocation = {
            "cpu_cores": system_resources.cpu_cores if system_resources else 1,
            "memory_gb": system_resources.memory_gb if system_resources else 4,
            "use_gpu": system_resources.has_gpu if system_resources else False,
            "simulation_mode": "sequential",
        }
        
        # 优化配置
        optimization_config = {
            "batch_simulation": True,
            "memory_optimization": True,
            "fast_mode": config.get("fast_simulation", False),
            "debug_mode": config.get("debug", False),
        }
        
        return ExecutionStrategy(
            strategy_type=StrategyType.LOCAL_SIMULATION,
            num_processes=1,
            communication_backend="local_memory",
            data_distribution="centralized_split",
            resource_allocation=resource_allocation,
            optimization_config=optimization_config,
            metadata={
                "requires_network": False,
                "scalability": "low",
                "fault_tolerance": "high"
            }
        )
    
    def _select_communication_backend(
        self,
        network_env: Optional[NetworkEnvironment],
        preferred_type: str = "auto"
    ) -> str:
        """选择通信后端"""
        if not network_env:
            return "local_memory"
        
        if preferred_type == "distributed":
            # 根据网络条件选择
            if network_env.bandwidth_mbps > 100:
                return "grpc"
            elif network_env.bandwidth_mbps > 10:
                return "tcp_socket"
            else:
                return "http_rest"
        
        return "local_memory"
    
    def _optimize_strategy(
        self,
        strategy: ExecutionStrategy,
        system_resources: Optional[SystemResources],
        network_env: Optional[NetworkEnvironment]
    ) -> ExecutionStrategy:
        """优化策略配置"""
        
        # 内存优化
        if system_resources and system_resources.memory_gb < 8:
            strategy.optimization_config["memory_efficient"] = True
            strategy.optimization_config["batch_size_reduction"] = 0.5
        
        # 网络优化
        if network_env and network_env.bandwidth_mbps < 50:
            strategy.optimization_config["compression"] = True
            strategy.optimization_config["gradient_compression_ratio"] = 0.1
        
        # GPU优化
        if system_resources and system_resources.has_gpu:
            strategy.resource_allocation["gpu_memory_fraction"] = 0.8
            strategy.optimization_config["mixed_precision"] = True
        
        return strategy
    
    def _create_cache_key(
        self,
        mode: ExecutionMode,
        config: Dict[str, Any],
        system_resources: Optional[SystemResources],
        network_env: Optional[NetworkEnvironment]
    ) -> str:
        """创建缓存键"""
        key_parts = [
            mode.value,
            str(config.get("num_clients", 3)),
            str(system_resources.cpu_cores if system_resources else "unknown"),
            str(system_resources.memory_gb if system_resources else "unknown"),
            str(network_env.bandwidth_mbps if network_env else "unknown")
        ]
        return "_".join(key_parts)
    
    def clear_cache(self):
        """清空策略缓存"""
        self._strategy_cache.clear()
        self.logger.info("🔄 策略缓存已清空")
    
    def get_available_strategies(self) -> List[StrategyType]:
        """获取可用的策略类型"""
        return list(StrategyType)
    
    def get_strategy_info(self, strategy_type: StrategyType) -> Dict[str, Any]:
        """获取策略信息"""
        strategy_info = {
            StrategyType.TRUE_FEDERATION: {
                "name": "真联邦",
                "description": "分布式多机器联邦学习",
                "requirements": ["网络连接", "多机器环境"],
                "scalability": "高",
                "complexity": "高"
            },
            StrategyType.PSEUDO_FEDERATION: {
                "name": "伪联邦",
                "description": "单机多进程模拟联邦学习",
                "requirements": ["多核CPU", "充足内存"],
                "scalability": "中",
                "complexity": "中"
            },
            StrategyType.LOCAL_SIMULATION: {
                "name": "本地模拟",
                "description": "单进程顺序模拟联邦学习",
                "requirements": ["基础计算资源"],
                "scalability": "低",
                "complexity": "低"
            }
        }
        
        return strategy_info.get(strategy_type, {})