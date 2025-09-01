# fedcl/transparent/mode_detector.py
"""
智能模式检测器

自动检测运行环境并选择最优的联邦学习执行模式：
- 真联邦：多机器分布式环境
- 伪联邦：单机多进程模拟
- 本地模拟：单进程快速测试

让用户完全无感知地在不同环境中运行相同的代码。
"""

import os
import socket
import psutil
import platform
from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from loguru import logger


class ExecutionMode(Enum):
    """执行模式枚举"""
    TRUE_FEDERATION = "true_federation"    # 真联邦：分布式多机器
    PSEUDO_FEDERATION = "pseudo_federation"  # 伪联邦：单机多进程
    LOCAL_SIMULATION = "local_simulation"   # 本地模拟：单进程


@dataclass
class SystemResources:
    """系统资源信息"""
    cpu_cores: int
    memory_gb: float
    available_memory_gb: float
    gpu_count: int
    has_gpu: bool
    network_interfaces: List[str]
    is_distributed_capable: bool
    confidence_score: float


@dataclass
class NetworkEnvironment:
    """网络环境信息"""
    hostname: str
    ip_addresses: List[str]
    open_ports: List[int]
    can_bind_server_port: bool
    can_connect_to_peers: bool
    network_quality: str  # "good", "fair", "poor"
    bandwidth_mbps: float  # 估算的网络带宽


class ModeDetector:
    """
    智能模式检测器
    
    自动分析运行环境的各种特征，智能选择最适合的执行模式。
    检测逻辑：
    1. 网络环境检测：是否有多机器可用
    2. 系统资源检测：CPU、内存、GPU资源
    3. 配置文件分析：用户是否提供了分布式配置
    4. 环境变量检测：是否在容器/集群环境中运行
    """
    
    def __init__(self):
        self.logger = logger.bind(component="ModeDetector")
        self._detection_cache: Optional[ExecutionMode] = None
        self._system_resources: Optional[SystemResources] = None
        self._network_env: Optional[NetworkEnvironment] = None
    
    def detect_execution_mode(self, config: Optional[Dict[str, Any]] = None) -> ExecutionMode:
        """
        智能检测最优执行模式
        
        Args:
            config: 可选的配置信息，可能包含用户偏好
            
        Returns:
            ExecutionMode: 推荐的执行模式
        """
        self.logger.info("🔍 开始智能模式检测...")
        
        # 检查用户是否强制指定模式
        if config and config.get("force_mode"):
            forced_mode = config.get("execution_mode")
            if forced_mode in [mode.value for mode in ExecutionMode]:
                mode = ExecutionMode(forced_mode)
                self.logger.info(f"👤 用户强制指定模式: {mode.value}")
                return mode
        
        # 执行自动检测
        system_resources = self._analyze_system_resources()
        network_env = self._analyze_network_environment()
        container_env = self._detect_container_environment()
        config_hints = self._analyze_config_hints(config)
        
        # 综合决策
        mode = self._make_intelligent_decision(
            system_resources, network_env, container_env, config_hints
        )
        
        self.logger.info(f"🎯 检测完成，推荐模式: {mode.value}")
        self._log_detection_summary(mode, system_resources, network_env)
        
        return mode
    
    def _analyze_system_resources(self) -> SystemResources:
        """分析系统资源"""
        if self._system_resources:
            return self._system_resources
        
        self.logger.debug("📊 分析系统资源...")
        
        # CPU信息
        cpu_cores = psutil.cpu_count(logical=False) or 1
        
        # 内存信息
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        available_memory_gb = memory.available / (1024**3)
        
        # GPU信息（简化检测）
        gpu_count = 0
        has_gpu = False
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                has_gpu = gpu_count > 0
        except ImportError:
            pass
        
        # 网络接口
        network_interfaces = list(psutil.net_if_addrs().keys())
        
        # 分布式能力评估
        is_distributed_capable = (
            cpu_cores >= 4 and 
            memory_gb >= 8 and 
            len(network_interfaces) >= 2
        )
        
        # 置信度评分
        confidence_score = min(1.0, (cpu_cores / 8 + memory_gb / 16) / 2)
        
        self._system_resources = SystemResources(
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            available_memory_gb=available_memory_gb,
            gpu_count=gpu_count,
            has_gpu=has_gpu,
            network_interfaces=network_interfaces,
            is_distributed_capable=is_distributed_capable,
            confidence_score=confidence_score
        )
        
        self.logger.debug(f"💻 系统资源: CPU={cpu_cores}核, 内存={memory_gb:.1f}GB, GPU={gpu_count}个")
        return self._system_resources
    
    def _analyze_network_environment(self) -> NetworkEnvironment:
        """分析网络环境"""
        if self._network_env:
            return self._network_env
        
        self.logger.debug("🌐 分析网络环境...")
        
        # 主机名和IP
        hostname = socket.gethostname()
        ip_addresses = []
        
        try:
            # 获取所有网络接口的IP地址
            for interface, addrs in psutil.net_if_addrs().items():
                for addr in addrs:
                    if addr.family == socket.AF_INET and not addr.address.startswith('127.'):
                        ip_addresses.append(addr.address)
        except Exception as e:
            self.logger.debug(f"获取IP地址失败: {e}")
        
        # 检测可用端口
        open_ports = self._scan_available_ports([8080, 8081, 8082, 9000, 9001])
        
        # 测试服务器端口绑定能力
        can_bind_server_port = self._test_port_binding(8080)
        
        # 测试对等连接能力（简化版）
        can_connect_to_peers = len(ip_addresses) > 0
        
        # 网络质量评估
        network_quality = "good" if can_bind_server_port and can_connect_to_peers else "fair"
        
        # 估算网络带宽（简化版）
        bandwidth_mbps = self._estimate_network_bandwidth()
        
        self._network_env = NetworkEnvironment(
            hostname=hostname,
            ip_addresses=ip_addresses,
            open_ports=open_ports,
            can_bind_server_port=can_bind_server_port,
            can_connect_to_peers=can_connect_to_peers,
            network_quality=network_quality,
            bandwidth_mbps=bandwidth_mbps
        )
        
        self.logger.debug(f"🌐 网络环境: {hostname}, IPs={len(ip_addresses)}, 可用端口={len(open_ports)}")
        return self._network_env
    
    def _detect_container_environment(self) -> Dict[str, Any]:
        """检测容器/集群环境"""
        env_info = {
            "is_docker": False,
            "is_kubernetes": False,
            "is_slurm": False,
            "is_cloud": False
        }
        
        # Docker检测
        if os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER'):
            env_info["is_docker"] = True
        
        # Kubernetes检测
        if os.environ.get('KUBERNETES_SERVICE_HOST'):
            env_info["is_kubernetes"] = True
        
        # SLURM检测
        if os.environ.get('SLURM_JOB_ID'):
            env_info["is_slurm"] = True
        
        # 云环境检测
        cloud_indicators = ['AWS_', 'AZURE_', 'GCP_', 'CLOUD_']
        if any(env_var for env_var in os.environ if any(indicator in env_var for indicator in cloud_indicators)):
            env_info["is_cloud"] = True
        
        return env_info
    
    def _analyze_config_hints(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """分析配置文件中的提示信息"""
        hints = {
            "suggests_distributed": False,
            "num_clients": 1,
            "has_server_config": False,
            "has_client_configs": False
        }
        
        if not config:
            return hints
        
        # 客户端数量提示
        num_clients = config.get("num_clients", 1)
        hints["num_clients"] = num_clients
        
        # 分布式配置提示
        if num_clients > 1:
            hints["suggests_distributed"] = True
        
        # 服务器配置检测
        if config.get("server_config") or config.get("server"):
            hints["has_server_config"] = True
            hints["suggests_distributed"] = True
        
        # 客户端配置检测
        if config.get("client_configs") or config.get("clients"):
            hints["has_client_configs"] = True
            hints["suggests_distributed"] = True
        
        return hints
    
    def _make_intelligent_decision(
        self, 
        system_resources: SystemResources,
        network_env: NetworkEnvironment,
        container_env: Dict[str, Any],
        config_hints: Dict[str, Any]
    ) -> ExecutionMode:
        """综合决策选择最优模式"""
        
        score_true_fed = 0
        score_pseudo_fed = 0
        score_local_sim = 0
        
        # 系统资源评分
        if system_resources.is_distributed_capable:
            score_true_fed += 3
            score_pseudo_fed += 2
        else:
            score_local_sim += 3
        
        if system_resources.cpu_cores >= 8:
            score_pseudo_fed += 2
            score_true_fed += 1
        
        # 网络环境评分
        if network_env.can_bind_server_port and len(network_env.ip_addresses) > 0:
            score_true_fed += 3
        
        if len(network_env.open_ports) >= 3:
            score_true_fed += 1
            score_pseudo_fed += 1
        
        # 容器环境评分
        if container_env["is_kubernetes"] or container_env["is_slurm"]:
            score_true_fed += 4
        elif container_env["is_docker"]:
            score_pseudo_fed += 2
        
        # 配置提示评分
        if config_hints["suggests_distributed"]:
            score_true_fed += 2
            score_pseudo_fed += 1
        
        if config_hints["num_clients"] > 3:
            score_true_fed += 1
        elif config_hints["num_clients"] <= 1:
            score_local_sim += 2
        
        # 决策逻辑
        max_score = max(score_true_fed, score_pseudo_fed, score_local_sim)
        
        if max_score == score_true_fed and score_true_fed >= 5:
            return ExecutionMode.TRUE_FEDERATION
        elif max_score == score_pseudo_fed and score_pseudo_fed >= 3:
            return ExecutionMode.PSEUDO_FEDERATION
        else:
            return ExecutionMode.LOCAL_SIMULATION
    
    def _scan_available_ports(self, ports: List[int]) -> List[int]:
        """扫描可用端口"""
        available = []
        for port in ports:
            if self._test_port_binding(port):
                available.append(port)
        return available
    
    def _test_port_binding(self, port: int) -> bool:
        """测试端口绑定能力"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return True
        except OSError:
            return False
    
    def _log_detection_summary(
        self, 
        mode: ExecutionMode, 
        system_resources: SystemResources,
        network_env: NetworkEnvironment
    ):
        """记录检测总结"""
        self.logger.info("📋 模式检测总结:")
        self.logger.info(f"  🎯 推荐模式: {mode.value}")
        self.logger.info(f"  💻 系统资源: CPU={system_resources.cpu_cores}核, 内存={system_resources.memory_gb:.1f}GB")
        self.logger.info(f"  🌐 网络环境: IP数量={len(network_env.ip_addresses)}, 可绑定端口={'是' if network_env.can_bind_server_port else '否'}")
        self.logger.info(f"  📊 置信度: {system_resources.confidence_score:.2f}")
    
    def get_mode_explanation(self, mode: ExecutionMode) -> str:
        """获取模式选择的详细解释"""
        explanations = {
            ExecutionMode.TRUE_FEDERATION: "检测到分布式环境，使用真联邦模式进行多机器训练",
            ExecutionMode.PSEUDO_FEDERATION: "检测到单机多核环境，使用伪联邦模式进行多进程模拟",
            ExecutionMode.LOCAL_SIMULATION: "检测到受限环境，使用本地模拟模式进行快速测试"
        }
        return explanations.get(mode, "未知模式")
    
    def get_detection_summary(self) -> Dict[str, Any]:
        """获取检测结果摘要"""
        return {
            "system_resources": self._system_resources,
            "network_environment": self._network_env,
            "cached_mode": self._detection_cache,
            "detection_confidence": self._system_resources.confidence_score if self._system_resources else 0.0
        }
    
    def _estimate_network_bandwidth(self) -> float:
        """估算网络带宽（简化版）"""
        try:
            # 检查网络接口统计
            net_io = psutil.net_io_counters()
            if net_io:
                # 简化的带宽估算，基于网络接口速度
                # 这里使用一个保守的估算
                return 100.0  # 默认假设100Mbps
            else:
                return 10.0   # 保守估算
        except Exception:
            return 10.0  # 默认值
    
    def detect_mode(self, config: Optional[Dict[str, Any]] = None) -> ExecutionMode:
        """检测执行模式的简化方法（用于向后兼容）"""
        return self.detect_execution_mode(config)
    
    def analyze_system_resources(self) -> SystemResources:
        """分析系统资源的公开方法"""
        return self._analyze_system_resources()
    
    def force_mode(self, mode: ExecutionMode) -> None:
        """强制设置模式（用于测试和调试）"""
        self._detection_cache = mode
        self.logger.warning(f"🔧 强制设置模式: {mode.value}")