"""
配置生成器

职责：生成完整、有效的 NodeConfig
- 这是唯一的配置生成入口
- CLI 和 PaperRegistry 都通过它生成配置
- 自动填充默认值、处理 partition 等

Usage:
    from oiafed.config import ConfigGenerator
    
    generator = ConfigGenerator()
    
    # 生成完整联邦配置（1 Trainer + N Learners）
    configs = generator.generate_federation(
        num_clients=10,
        trainer_args={"num_rounds": 100},
        learner_args={"learning_rate": 0.01},
    )
    
    # 或单独生成
    trainer = generator.generate_trainer(...)
    learner = generator.generate_learner(index=0, num_clients=10, ...)
    
    # 保存到文件
    generator.save_configs(configs, "./configs")
"""
from typing import Any, Dict, List, Optional, Union
from dataclasses import asdict
from pathlib import Path
import copy

from .defaults import (
    DEFAULT_TRAINER_PORT,
    DEFAULT_LEARNER_BASE_PORT,
    DEFAULT_HOST,
    DEFAULT_LOCALHOST,
    DEFAULT_MAX_MESSAGE_SIZE,
    DEFAULT_TIMEOUT,
    DEFAULT_TRAINER_TYPE,
    DEFAULT_LEARNER_TYPE,
    DEFAULT_AGGREGATOR_TYPE,
    DEFAULT_MODEL_TYPE,
    DEFAULT_DATASET_TYPE,
    DEFAULT_PARTITION_STRATEGY,
    DEFAULT_PARTITION_ALPHA,
    DEFAULT_LOG_LEVEL,
    DEFAULT_LOG_DIR,
    DEFAULT_SERIALIZATION,
    DEFAULT_TRANSPORT_MODE,
    DEFAULT_EXP_NAME,
)


class ConfigGenerator:
    """
    配置生成器
    
    统一的配置生成入口，确保生成的配置：
    1. 符合 NodeConfig Schema
    2. 包含所有必要的默认值
    3. 自动处理 partition_id, num_partitions 等
    
    设计原则：
    - 所有生成方法返回 NodeConfig 实例（不是字典）
    - 与 ConfigManager 协作：生成后可用 manager.save() 保存
    - 支持完全自定义，所有参数都可覆盖
    """
    
    def __init__(self):
        # 延迟导入避免循环依赖
        from .manager import ConfigManager
        self._manager = ConfigManager()
    
    # ==================== 内部方法 ====================
    
    def _generate_exp_name(
        self,
        model_type: str,
        aggregator_type: str,
        dataset_type: str,
        partition_strategy: str,
        partition_alpha: float,
        num_clients: int,
    ) -> str:
        """
        自动生成实验名称
        
        格式: {model}_{method}_{dataset}_{partition}{alpha}_n{num_clients}
        例如: cnn_fedavg_cifar10_dirichlet0.1_n10
        
        Args:
            model_type: 模型类型
            aggregator_type: 聚合器类型（作为方法名）
            dataset_type: 数据集类型
            partition_strategy: 分区策略
            partition_alpha: Dirichlet alpha
            num_clients: 客户端数量
            
        Returns:
            生成的实验名称
        """
        # 提取简短名称（去掉前缀如 "methods.models."）
        model = model_type.split(".")[-1]
        method = aggregator_type.split(".")[-1]
        dataset = dataset_type.split(".")[-1]
        
        # 构建分区部分
        if partition_strategy == "iid":
            partition_part = "iid"
        else:
            partition_part = f"{partition_strategy}{partition_alpha}"
        
        # 组合: model_method_dataset_partition_nX
        return f"{model}_{method}_{dataset}_{partition_part}_n{num_clients}"
    
    # ==================== 主要 API ====================
    
    def generate_federation(
        self,
        num_clients: int,
        *,
        # 全局配置
        exp_name: Optional[str] = None,  # None 表示自动生成
        run_name: Optional[str] = None,
        log_dir: str = DEFAULT_LOG_DIR,
        
        # 组件类型（可从 PaperDef 获取）
        trainer_type: str = DEFAULT_TRAINER_TYPE,
        learner_type: str = DEFAULT_LEARNER_TYPE,
        aggregator_type: str = DEFAULT_AGGREGATOR_TYPE,
        model_type: str = DEFAULT_MODEL_TYPE,
        dataset_type: str = DEFAULT_DATASET_TYPE,
        
        # 组件参数
        trainer_args: Optional[Dict[str, Any]] = None,
        learner_args: Optional[Dict[str, Any]] = None,
        aggregator_args: Optional[Dict[str, Any]] = None,
        model_args: Optional[Dict[str, Any]] = None,
        dataset_args: Optional[Dict[str, Any]] = None,
        
        # 数据划分
        partition_strategy: str = DEFAULT_PARTITION_STRATEGY,
        partition_alpha: float = DEFAULT_PARTITION_ALPHA,
        partition_seed: Optional[int] = None,
        
        # 可选配置
        tracker: Optional[Dict[str, Any]] = None,
        callbacks: Optional[List[Dict[str, Any]]] = None,
        logging: Optional[Dict[str, Any]] = None,
        
        # 网络配置
        trainer_host: str = DEFAULT_LOCALHOST,
        trainer_port: int = DEFAULT_TRAINER_PORT,
        learner_host: str = DEFAULT_LOCALHOST,
        learner_base_port: int = DEFAULT_LEARNER_BASE_PORT,
        
        # 其他
        default_timeout: float = DEFAULT_TIMEOUT,
        transport_mode: str = DEFAULT_TRANSPORT_MODE,
        
    ) -> List:
        """
        生成完整的联邦学习配置
        
        Args:
            num_clients: 客户端数量
            exp_name: 实验名称（None 则自动生成 model_method_dataset_partition_nX 格式）
            run_name: 运行名称（可选，自动生成）
            log_dir: 日志目录
            trainer_type: Trainer 类型
            learner_type: Learner 类型
            aggregator_type: Aggregator 类型
            model_type: 模型类型
            dataset_type: 数据集类型
            trainer_args: Trainer 参数（num_rounds, local_epochs 等）
            learner_args: Learner 参数（learning_rate, batch_size 等）
            aggregator_args: Aggregator 参数
            model_args: 模型参数（num_classes 等）
            dataset_args: 数据集参数（data_dir, download 等）
            partition_strategy: 数据划分策略（dirichlet, iid, label 等）
            partition_alpha: Dirichlet alpha（越小越 non-IID）
            partition_seed: 划分随机种子
            tracker: 追踪配置（MLflow, WandB 等）
            callbacks: 回调配置列表
            logging: 日志配置
            trainer_host: Trainer 主机地址
            trainer_port: Trainer 端口
            learner_host: Learner 主机地址
            learner_base_port: Learner 基础端口（第 i 个 Learner 使用 base_port + i）
            default_timeout: 默认 RPC 超时时间
            transport_mode: 传输模式（grpc, memory）
            
        Returns:
            配置列表 [trainer_config, learner_0, learner_1, ..., learner_{n-1}]
        """
        # 自动生成 exp_name（如果未指定）
        if exp_name is None:
            exp_name = self._generate_exp_name(
                model_type=model_type,
                aggregator_type=aggregator_type,
                dataset_type=dataset_type,
                partition_strategy=partition_strategy,
                partition_alpha=partition_alpha,
                num_clients=num_clients,
            )
        
        configs = []
        
        # 生成 Trainer 配置
        trainer_config = self.generate_trainer(
            exp_name=exp_name,
            run_name=run_name,
            log_dir=log_dir,
            trainer_type=trainer_type,
            aggregator_type=aggregator_type,
            model_type=model_type,
            trainer_args=trainer_args,
            aggregator_args=aggregator_args,
            model_args=model_args,
            tracker=tracker,
            callbacks=callbacks,
            logging=logging,
            host=trainer_host,
            port=trainer_port,
            min_peers=num_clients,
            default_timeout=default_timeout,
            transport_mode=transport_mode,
        )
        configs.append(trainer_config)
        
        # 生成 Learner 配置
        trainer_address = f"trainer@{trainer_host}:{trainer_port}"
        
        for i in range(num_clients):
            learner_config = self.generate_learner(
                index=i,
                num_clients=num_clients,
                exp_name=exp_name,
                run_name=run_name,
                log_dir=log_dir,
                learner_type=learner_type,
                model_type=model_type,
                dataset_type=dataset_type,
                learner_args=learner_args,
                model_args=model_args,
                dataset_args=dataset_args,
                partition_strategy=partition_strategy,
                partition_alpha=partition_alpha,
                partition_seed=partition_seed,
                tracker=tracker,
                callbacks=callbacks,
                logging=logging,
                connect_to=[trainer_address],
                host=learner_host,
                port=learner_base_port + i,
                default_timeout=default_timeout,
                transport_mode=transport_mode,
            )
            configs.append(learner_config)
        
        return configs
    
    def generate_trainer(
        self,
        *,
        node_id: str = "trainer",
        exp_name: str = DEFAULT_EXP_NAME,
        run_name: Optional[str] = None,
        log_dir: str = DEFAULT_LOG_DIR,
        trainer_type: str = DEFAULT_TRAINER_TYPE,
        aggregator_type: str = DEFAULT_AGGREGATOR_TYPE,
        model_type: str = DEFAULT_MODEL_TYPE,
        trainer_args: Optional[Dict[str, Any]] = None,
        aggregator_args: Optional[Dict[str, Any]] = None,
        model_args: Optional[Dict[str, Any]] = None,
        tracker: Optional[Dict[str, Any]] = None,
        callbacks: Optional[List[Dict[str, Any]]] = None,
        logging: Optional[Dict[str, Any]] = None,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_TRAINER_PORT,
        min_peers: int = 1,
        default_timeout: float = DEFAULT_TIMEOUT,
        transport_mode: str = DEFAULT_TRANSPORT_MODE,
    ):
        """
        生成 Trainer 配置
        
        Args:
            node_id: 节点 ID
            exp_name: 实验名称
            run_name: 运行名称
            log_dir: 日志目录
            trainer_type: Trainer 类型
            aggregator_type: Aggregator 类型
            model_type: 模型类型
            trainer_args: Trainer 参数
            aggregator_args: Aggregator 参数
            model_args: 模型参数
            tracker: 追踪配置
            callbacks: 回调配置
            logging: 日志配置
            host: 监听地址
            port: 监听端口
            min_peers: 最少等待的客户端数
            default_timeout: 默认超时
            transport_mode: 传输模式
            
        Returns:
            NodeConfig 实例
        """
        config_dict = {
            "node_id": node_id,
            "role": "trainer",
            "global_config": {
                "exp_name": exp_name,
                "run_name": run_name,
                "log_dir": log_dir,
            },
            "listen": {
                "host": host,
                "port": port,
            },
            "min_peers": min_peers,
            "default_timeout": default_timeout,
            "transport": {
                "mode": transport_mode,
                "grpc": {"max_message_size": DEFAULT_MAX_MESSAGE_SIZE},
            },
            "serialization": {"default": DEFAULT_SERIALIZATION},
            "trainer": {
                "type": trainer_type,
                "args": trainer_args or {},
            },
            "aggregator": {
                "type": aggregator_type,
                "args": aggregator_args or {},
            },
            "model": {
                "type": model_type,
                "args": model_args or {},
            },
        }
        
        # 添加可选配置
        self._add_optional_config(config_dict, tracker, callbacks, logging)
        
        return self._manager.from_dict(config_dict)
    
    def generate_learner(
        self,
        *,
        index: int,
        num_clients: int,
        node_id: Optional[str] = None,
        exp_name: str = DEFAULT_EXP_NAME,
        run_name: Optional[str] = None,
        log_dir: str = DEFAULT_LOG_DIR,
        learner_type: str = DEFAULT_LEARNER_TYPE,
        model_type: str = DEFAULT_MODEL_TYPE,
        dataset_type: str = DEFAULT_DATASET_TYPE,
        learner_args: Optional[Dict[str, Any]] = None,
        model_args: Optional[Dict[str, Any]] = None,
        dataset_args: Optional[Dict[str, Any]] = None,
        partition_strategy: str = DEFAULT_PARTITION_STRATEGY,
        partition_alpha: float = DEFAULT_PARTITION_ALPHA,
        partition_seed: Optional[int] = None,
        tracker: Optional[Dict[str, Any]] = None,
        callbacks: Optional[List[Dict[str, Any]]] = None,
        logging: Optional[Dict[str, Any]] = None,
        connect_to: Optional[List[str]] = None,
        host: str = DEFAULT_HOST,
        port: Optional[int] = None,
        default_timeout: float = DEFAULT_TIMEOUT,
        transport_mode: str = DEFAULT_TRANSPORT_MODE,
    ):
        """
        生成 Learner 配置
        
        Args:
            index: Learner 索引（0, 1, 2, ...）
            num_clients: 总客户端数（用于 partition）
            node_id: 节点 ID（默认 learner_{index}）
            exp_name: 实验名称
            run_name: 运行名称
            log_dir: 日志目录
            learner_type: Learner 类型
            model_type: 模型类型
            dataset_type: 数据集类型
            learner_args: Learner 参数
            model_args: 模型参数
            dataset_args: 数据集参数
            partition_strategy: 划分策略
            partition_alpha: Dirichlet alpha
            partition_seed: 随机种子
            tracker: 追踪配置
            callbacks: 回调配置
            logging: 日志配置
            connect_to: 连接目标
            host: 监听地址
            port: 监听端口（默认 base_port + index）
            default_timeout: 默认超时
            transport_mode: 传输模式
            
        Returns:
            NodeConfig 实例
        """
        # 默认值处理
        if node_id is None:
            node_id = f"learner_{index}"
        
        if connect_to is None:
            connect_to = [f"trainer@{DEFAULT_LOCALHOST}:{DEFAULT_TRAINER_PORT}"]
        
        if port is None:
            port = DEFAULT_LEARNER_BASE_PORT + index
        
        # 构建数据集配置
        datasets = self._build_datasets(
            dataset_type=dataset_type,
            dataset_args=dataset_args or {},
            partition_strategy=partition_strategy,
            partition_alpha=partition_alpha,
            partition_seed=partition_seed,
            num_partitions=num_clients,
            partition_id=index,
        )
        
        config_dict = {
            "node_id": node_id,
            "role": "learner",
            "global_config": {
                "exp_name": exp_name,
                "run_name": run_name,
                "log_dir": log_dir,
            },
            "listen": {
                "host": host,
                "port": port,
            },
            "connect_to": connect_to,
            "default_timeout": default_timeout,
            "transport": {
                "mode": transport_mode,
                "grpc": {"max_message_size": DEFAULT_MAX_MESSAGE_SIZE},
            },
            "serialization": {"default": DEFAULT_SERIALIZATION},
            "learner": {
                "type": learner_type,
                "args": learner_args or {},
            },
            "model": {
                "type": model_type,
                "args": model_args or {},
            },
            "datasets": datasets,
        }
        
        # 添加可选配置
        self._add_optional_config(config_dict, tracker, callbacks, logging)
        
        return self._manager.from_dict(config_dict)
    
    # ==================== 文件操作 ====================
    
    def save_configs(
        self,
        configs: List,
        output_dir: str,
    ) -> List[str]:
        """
        保存配置到文件
        
        Args:
            configs: NodeConfig 列表
            output_dir: 输出目录
            
        Returns:
            保存的文件路径列表
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        for config in configs:
            file_path = output_path / f"{config.node_id}.yaml"
            self._manager.save(config, str(file_path))
            saved_files.append(str(file_path))
        
        return saved_files
    
    # ==================== 辅助方法 ====================
    
    def _build_datasets(
        self,
        dataset_type: str,
        dataset_args: Dict[str, Any],
        partition_strategy: str,
        partition_alpha: float,
        partition_seed: Optional[int],
        num_partitions: int,
        partition_id: int,
    ) -> List[Dict[str, Any]]:
        """
        构建数据集配置列表（train + test）
        
        train 数据集带 partition 配置
        test 数据集不划分（共享）
        """
        # 构建 partition 配置
        partition_config = {
            "strategy": partition_strategy,
            "num_partitions": num_partitions,
            "partition_id": partition_id,
        }
        
        # 根据策略添加特定参数
        if partition_strategy == "dirichlet":
            partition_config["alpha"] = partition_alpha
        
        if partition_seed is not None:
            partition_config["seed"] = partition_seed
        
        # 训练集参数
        train_args = copy.deepcopy(dataset_args)
        
        # 测试集参数（排除 download）
        test_args = {k: v for k, v in dataset_args.items() if k != "download"}
        
        return [
            # 训练集（带划分）
            {
                "type": dataset_type,
                "split": "train",
                "args": train_args,
                "partition": partition_config,
            },
            # 测试集（不划分）
            {
                "type": dataset_type,
                "split": "test",
                "args": test_args,
            },
        ]
    
    def _add_optional_config(
        self,
        config_dict: Dict[str, Any],
        tracker: Optional[Dict[str, Any]],
        callbacks: Optional[List[Dict[str, Any]]],
        logging: Optional[Dict[str, Any]],
    ) -> None:
        """添加可选配置（仅当提供时）"""
        
        if tracker is not None:
            config_dict["tracker"] = tracker
        
        if callbacks is not None:
            config_dict["callbacks"] = callbacks
        
        if logging is not None:
            config_dict["logging"] = logging
        else:
            # 默认日志配置
            config_dict["logging"] = {
                "level": DEFAULT_LOG_LEVEL,
                "console": True,
            }
    
    # ==================== PaperRegistry 接口 ====================
    
    def from_paper_def(
        self,
        paper_defaults: Dict[str, Any],
        paper_components: Dict[str, str],
        num_clients: int,
        override: Optional[Dict[str, Any]] = None,
    ) -> List:
        """
        从论文定义生成配置
        
        这是 PaperRegistry 调用的入口点，将论文定义转换为配置。
        
        Args:
            paper_defaults: 论文默认配置
            paper_components: 论文组件类型映射 {"trainer": "...", "learner": "...", ...}
            num_clients: 客户端数量
            override: 用户覆盖配置
            
        Returns:
            NodeConfig 列表 [trainer, learner_0, learner_1, ...]
        """
        # 合并配置: paper_defaults + override
        merged = self._deep_merge(paper_defaults, override or {})
        
        # 提取参数
        params = self._extract_params_from_merged(merged, paper_components, num_clients)
        
        # 生成配置
        return self.generate_federation(**params)
    
    def _extract_params_from_merged(
        self,
        merged: Dict[str, Any],
        paper_components: Dict[str, str],
        num_clients: int,
    ) -> Dict[str, Any]:
        """从合并后的配置中提取 generate_federation 需要的参数"""
        
        params: Dict[str, Any] = {
            "num_clients": num_clients,
            
            # 全局配置
            "exp_name": merged.get("global_config", {}).get("exp_name", DEFAULT_EXP_NAME),
            "run_name": merged.get("global_config", {}).get("run_name"),
            "log_dir": merged.get("global_config", {}).get("log_dir", DEFAULT_LOG_DIR),
            
            # 组件类型：优先用 paper_components，其次用 merged 中的 override
            "trainer_type": paper_components.get("trainer") or merged.get("trainer", {}).get("type") or DEFAULT_TRAINER_TYPE,
            "learner_type": paper_components.get("learner") or merged.get("learner", {}).get("type") or DEFAULT_LEARNER_TYPE,
            "aggregator_type": paper_components.get("aggregator") or merged.get("aggregator", {}).get("type") or DEFAULT_AGGREGATOR_TYPE,
            "model_type": paper_components.get("model") or merged.get("model", {}).get("type") or DEFAULT_MODEL_TYPE,
            "dataset_type": paper_components.get("dataset") or DEFAULT_DATASET_TYPE,
            
            # 组件参数
            "trainer_args": self._extract_component_args(merged.get("trainer", {})),
            "learner_args": self._extract_component_args(merged.get("learner", {})),
            "aggregator_args": self._extract_component_args(merged.get("aggregator", {})),
            "model_args": self._extract_component_args(merged.get("model", {})),
        }
        
        # 处理数据集配置
        self._extract_dataset_params(merged, params)
        
        # 可选配置
        if "tracker" in merged:
            params["tracker"] = merged["tracker"]
        if "callbacks" in merged:
            params["callbacks"] = merged["callbacks"]
        if "logging" in merged:
            params["logging"] = merged["logging"]
        if "default_timeout" in merged:
            params["default_timeout"] = merged["default_timeout"]
        if "transport" in merged:
            params["transport_mode"] = merged["transport"].get("mode", DEFAULT_TRANSPORT_MODE)
        
        return params
    
    def _extract_component_args(self, component_config: Dict[str, Any]) -> Dict[str, Any]:
        """提取组件参数（排除 type 字段）"""
        if isinstance(component_config, dict):
            # 如果有 args 字段，使用它
            if "args" in component_config:
                return copy.deepcopy(component_config["args"])
            # 否则排除 type 字段
            return {k: v for k, v in component_config.items() if k != "type"}
        return {}
    
    def _extract_dataset_params(
        self,
        merged: Dict[str, Any],
        params: Dict[str, Any],
    ) -> None:
        """提取数据集相关参数"""
        
        # 标准格式: datasets (复数)
        if "datasets" in merged:
            datasets_list = merged["datasets"]
            # 查找训练集配置
            train_ds = next(
                (ds for ds in datasets_list if ds.get("split") == "train"),
                datasets_list[0] if datasets_list else None
            )
            if train_ds:
                params["dataset_args"] = train_ds.get("args", {})
                params["dataset_type"] = train_ds.get("type", params.get("dataset_type", DEFAULT_DATASET_TYPE))
                
                partition = train_ds.get("partition", {})
                if partition:
                    params["partition_strategy"] = partition.get("strategy", DEFAULT_PARTITION_STRATEGY)
                    params["partition_alpha"] = partition.get("alpha", DEFAULT_PARTITION_ALPHA)
                    if "seed" in partition:
                        params["partition_seed"] = partition["seed"]
        
        # 简化格式: dataset (单数，向后兼容)
        elif "dataset" in merged:
            dataset_config = copy.deepcopy(merged["dataset"])
            partition = dataset_config.pop("partition", {})
            
            # dataset_type 可能在顶层
            if "type" in dataset_config:
                params["dataset_type"] = dataset_config.pop("type")
            
            params["dataset_args"] = dataset_config
            
            if partition:
                params["partition_strategy"] = partition.get("strategy", DEFAULT_PARTITION_STRATEGY)
                params["partition_alpha"] = partition.get("alpha", DEFAULT_PARTITION_ALPHA)
                if "seed" in partition:
                    params["partition_seed"] = partition["seed"]
    
    def _deep_merge(
        self,
        base: Dict[str, Any],
        override: Dict[str, Any],
    ) -> Dict[str, Any]:
        """深度合并字典"""
        result = copy.deepcopy(base)
        
        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        
        return result


# ==================== 便捷函数 ====================

def generate_federation(num_clients: int, **kwargs) -> List:
    """
    生成联邦配置的便捷函数
    
    等同于 ConfigGenerator().generate_federation(num_clients, **kwargs)
    """
    return ConfigGenerator().generate_federation(num_clients, **kwargs)


__all__ = [
    "ConfigGenerator",
    "generate_federation",
]