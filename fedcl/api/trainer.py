# fedcl/api/trainer.py
"""
统一的联邦训练器

提供简洁统一的联邦学习接口，支持多种初始化方式和自动模式检测。
实现真联邦和伪联邦的完全透明切换。
"""

import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from ..transparent.base_federation_engine import BaseFederationEngine as TransparentExecutionEngine


@dataclass
class TrainingResult:
    """联邦训练结果"""

    experiment_name: str
    total_rounds: int
    final_metrics: Dict[str, float]
    round_history: List[Dict[str, Any]]
    training_time: float
    client_results: Dict[str, Any] = field(default_factory=dict)
    global_model_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def average_accuracy(self) -> float:
        """获取平均准确率"""
        if "average_accuracy" in self.final_metrics:
            return self.final_metrics["average_accuracy"]
        return 0.0

    @property
    def forgetting(self) -> float:
        """获取遗忘度"""
        if "forgetting" in self.final_metrics:
            return self.final_metrics["forgetting"]
        return 0.0


@dataclass
class EvaluationResult:
    """评估结果"""

    metrics: Dict[str, float]
    task_accuracies: Dict[str, float] = field(default_factory=dict)
    evaluation_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class FederatedTrainer:
    """
    统一的联邦训练器

    提供简洁统一的联邦学习接口，支持：
    1. 多种初始化方式（配置文件、字典、参数）
    2. 自动模式检测（真联邦/伪联邦/本地模拟）
    3. 透明的执行引擎
    4. 统一的训练和评估接口

    使用示例：
        # 方式1：最简单的使用
        trainer = FederatedTrainer(
            learner="ewc_mnist",
            dataset="mnist",
            num_clients=3
        )
        result = trainer.train(num_rounds=10)

        # 方式2：使用配置文件
        trainer = FederatedTrainer.from_config("config.yaml")
        result = trainer.train()

        # 方式3：使用配置字典
        config = {"learner": "ewc_mnist", "dataset": "mnist"}
        trainer = FederatedTrainer(config)
        result = trainer.train()
    """

    def __init__(
        self,
        config: Optional[Union[str, Path, Dict[str, Any], DictConfig]] = None,
        **kwargs,
    ):
        """
        初始化联邦训练器

        Args:
            config: 配置文件路径、配置字典或DictConfig对象
            **kwargs: 额外的配置参数，会覆盖config中的设置

        支持的kwargs参数：
            - learner: 学习器名称
            - aggregator: 聚合器名称
            - evaluator: 评估器名称
            - dataset: 数据集名称
            - num_clients: 客户端数量
            - num_rounds: 训练轮次
            - execution_mode: 执行模式（auto/true_federation/pseudo_federation）
            - experiment_name: 实验名称
        """
        self.start_time = time.time()

        # 解析和合并配置
        self.config = self._parse_config(config, **kwargs)

        # 设置实验名称
        self.experiment_name = self.config.get(
            "experiment_name", "federated_experiment"
        )

        # 创建组件日志器
        self.logger = logger.bind(component="FederatedTrainer", experiment=self.experiment_name)

        # 延迟初始化的组件（在需要时创建）
        self._execution_engine = None
        self._is_initialized = False

        self.logger.info(f"✅ FederatedTrainer 初始化完成: {self.experiment_name}")
        self.logger.debug(f"📝 配置: {dict(self.config)}")

    @classmethod
    def from_config(cls, config_path: Union[str, Path], **kwargs) -> "FederatedTrainer":
        """
        从配置文件创建训练器

        Args:
            config_path: 配置文件路径
            **kwargs: 额外参数，会覆盖配置文件中的设置

        Returns:
            FederatedTrainer实例
        """
        return cls(config=config_path, **kwargs)

    def _parse_config(
        self, config: Optional[Union[str, Path, Dict[str, Any], DictConfig]], **kwargs
    ) -> DictConfig:
        """
        解析和合并配置

        优先级（从高到低）：
        1. kwargs参数
        2. config参数
        3. 默认配置
        """
        # 默认配置
        default_config = {
            "experiment_name": "federated_experiment",
            "execution_mode": "auto",  # auto, true_federation, pseudo_federation
            "learner": "simple_learner",
            "learner_name": None,  # 🆕 从注册表获取用户自定义learner
            "learner_type": "standard",  # 🆕 内置learner类型回退
            "aggregator": "fedavg",
            "aggregator_name": None,  # 🆕 从注册表获取用户自定义聚合器
            "aggregator_type": "fedavg",  # 🆕 内置聚合器类型回退
            "evaluator": "accuracy",
            "evaluator_name": None,  # 🆕 从注册表获取用户自定义评估器
            "evaluator_type": "accuracy",  # 🆕 内置评估器类型回退
            "trainer_name": None,  # 🆕 从注册表获取用户自定义trainer
            "trainer_type": "standard",  # 🆕 内置trainer类型回退
            "dataset": "mnist",
            "num_clients": 3,
            "num_rounds": 10,
            "federation": {"client_selection": "random", "participation_rate": 1.0},
            "training": {"local_epochs": 1, "batch_size": 32, "learning_rate": 0.01},
            "logging": {"level": "INFO", "enable_debug": False},
        }

        # 创建基础配置
        final_config = OmegaConf.create(default_config)

        # 合并输入配置
        if config is not None:
            if isinstance(config, (str, Path)):
                # 配置文件路径
                try:
                    file_config = OmegaConf.load(config)
                    final_config = OmegaConf.merge(final_config, file_config)
                except Exception as e:
                    warnings.warn(f"无法加载配置文件 {config}: {e}，使用默认配置")
            elif isinstance(config, dict):
                # 配置字典
                dict_config = OmegaConf.create(config)
                final_config = OmegaConf.merge(final_config, dict_config)
            elif isinstance(config, DictConfig):
                # DictConfig对象
                final_config = OmegaConf.merge(final_config, config)

        # 合并kwargs参数（最高优先级）
        if kwargs:
            # 处理嵌套配置参数
            processed_kwargs = {}
            for key, value in kwargs.items():
                if key in ["local_epochs", "batch_size", "learning_rate"]:
                    # 这些参数属于training配置
                    if "training" not in processed_kwargs:
                        processed_kwargs["training"] = {}
                    processed_kwargs["training"][key] = value
                elif key in ["client_selection", "participation_rate"]:
                    # 这些参数属于federation配置
                    if "federation" not in processed_kwargs:
                        processed_kwargs["federation"] = {}
                    processed_kwargs["federation"][key] = value
                elif key in ["level", "enable_debug"]:
                    # 这些参数属于logging配置
                    if "logging" not in processed_kwargs:
                        processed_kwargs["logging"] = {}
                    processed_kwargs["logging"][key] = value
                else:
                    # 顶级配置
                    processed_kwargs[key] = value
            
            kwargs_config = OmegaConf.create(processed_kwargs)
            final_config = OmegaConf.merge(final_config, kwargs_config)

        # 🆕 验证和规范化装饰器组件配置
        final_config = self._normalize_component_config(final_config)

        return final_config
    
    def _normalize_component_config(self, config: DictConfig) -> DictConfig:
        """
        🆕 验证和规范化装饰器组件配置
        
        处理优先级：
        1. xxx_name (从注册表获取用户自定义组件) - 最高优先级
        2. xxx (兼容性字段，映射到xxx_name)
        3. xxx_type (内置组件类型) - 回退选项
        
        Args:
            config: 原始配置
            
        Returns:
            DictConfig: 规范化后的配置
        """
        from ..registry import registry
        
        # 对于learner配置的处理
        self._normalize_single_component_config(
            config, "learner", registry.learners, 
            "learner_name", "learner_type"
        )
        
        # 对于aggregator配置的处理
        self._normalize_single_component_config(
            config, "aggregator", registry.aggregators,
            "aggregator_name", "aggregator_type"
        )
        
        # 对于evaluator配置的处理
        self._normalize_single_component_config(
            config, "evaluator", registry.evaluators,
            "evaluator_name", "evaluator_type"
        )
        
        # 对于trainer配置的处理
        self._normalize_single_component_config(
            config, "trainer", registry.trainers,
            "trainer_name", "trainer_type"
        )
        
        return config
    
    def _normalize_single_component_config(self, config: DictConfig, 
                                         legacy_key: str, registry_dict: dict,
                                         name_key: str, type_key: str) -> None:
        """
        规范化单个组件的配置
        
        Args:
            config: 配置对象
            legacy_key: 遗留字段名 (learner/aggregator/evaluator/trainer)
            registry_dict: 注册表字典
            name_key: 用户自定义组件名字段 (learner_name/aggregator_name...)
            type_key: 内置类型字段 (learner_type/aggregator_type...)
        """
        # 获取各种配置值
        name_value = config.get(name_key)
        legacy_value = config.get(legacy_key) 
        type_value = config.get(type_key)
        
        # 优先级处理：
        # 1. 如果指定了xxx_name，使用用户自定义组件
        if name_value and name_value in registry_dict:
            self.logger.debug(f"🆕 使用用户自定义{legacy_key}: {name_value}")
            return
        
        # 2. 如果指定了legacy字段，尝试映射到注册表
        if legacy_value and legacy_value in registry_dict:
            config[name_key] = legacy_value
            self.logger.debug(f"🆕 将{legacy_key}='{legacy_value}'映射到{name_key}")
            return
        
        # 3. 都没找到，使用内置类型作为回退
        if not type_value:
            # 设置默认的内置类型
            default_types = {
                "learner": "standard",
                "aggregator": "fedavg", 
                "evaluator": "accuracy",
                "trainer": "standard"
            }
            config[type_key] = default_types.get(legacy_key, "standard")
            self.logger.debug(f"🆕 使用默认{type_key}: {config[type_key]}")

    def _get_execution_engine(self) -> TransparentExecutionEngine:
        """获取执行引擎（懒加载）"""
        if self._execution_engine is None:
            self.logger.info("🔧 初始化透明执行引擎")
            self._execution_engine = TransparentExecutionEngine(self.config)
            self._is_initialized = True
        return self._execution_engine

    def train(self, num_rounds: Optional[int] = None, **kwargs) -> TrainingResult:
        """
        执行联邦训练

        Args:
            num_rounds: 训练轮次，默认使用配置中的值
            **kwargs: 额外参数

        Returns:
            TrainingResult: 训练结果
        """
        if num_rounds is None:
            num_rounds = self.config.get("num_rounds", 10)

        self.logger.info(f"🚀 开始联邦训练 - 轮次: {num_rounds}")
        self.logger.info(f"📊 实验: {self.experiment_name}")

        # 获取执行引擎
        execution_engine = self._get_execution_engine()

        # 执行训练
        result = execution_engine.execute_training(num_rounds, **kwargs)

        # 创建训练结果
        training_result = TrainingResult(
            experiment_name=self.experiment_name,
            total_rounds=result.total_rounds,
            final_metrics=result.final_metrics,
            round_history=result.round_history,
            training_time=result.training_time,
            client_results=result.client_results,
            global_model_path=result.global_model_path,
            metadata={
                "execution_mode": result.execution_mode,
                "config": dict(self.config),
                **getattr(result, 'metadata', {})
            }
        )

        self.logger.info(f"✅ 训练完成 - 耗时: {training_result.training_time:.2f}秒")
        self.logger.info(f"📊 最终准确率: {training_result.average_accuracy:.4f}")

        return training_result

    def continue_training(
        self, additional_rounds: int = 5, **kwargs
    ) -> TrainingResult:
        """
        继续训练

        Args:
            additional_rounds: 额外的训练轮次
            **kwargs: 额外参数

        Returns:
            TrainingResult: 训练结果
        """
        self.logger.info(f"🔄 继续训练 - 额外轮次: {additional_rounds}")
        
        # 目前简化实现，直接调用train
        return self.train(num_rounds=additional_rounds, **kwargs)

    def evaluate(self, test_data: Optional[Any] = None, **kwargs) -> EvaluationResult:
        """
        执行模型评估

        Args:
            test_data: 测试数据
            **kwargs: 额外参数

        Returns:
            EvaluationResult: 评估结果
        """
        self.logger.info("🔍 开始模型评估")

        # 获取执行引擎
        execution_engine = self._get_execution_engine()

        # 执行评估
        result = execution_engine.execute_evaluation(test_data, **kwargs)

        # 创建评估结果
        evaluation_result = EvaluationResult(
            metrics=result.metrics,
            task_accuracies=result.task_accuracies,
            evaluation_time=result.evaluation_time,
            metadata={
                "experiment_name": self.experiment_name,
                "config": dict(self.config),
                **result.metadata
            }
        )

        self.logger.info(f"✅ 评估完成 - 耗时: {evaluation_result.evaluation_time:.2f}秒")
        self.logger.info(f"📊 准确率: {evaluation_result.metrics.get('accuracy', 0):.4f}")

        return evaluation_result

    def get_config(self) -> DictConfig:
        """获取当前配置"""
        return self.config

    def get_execution_mode(self) -> Optional[str]:
        """获取当前执行模式"""
        if self._execution_engine:
            mode = self._execution_engine.get_current_mode()
            return mode.value if mode else None
        return None

    def reset(self):
        """重置训练器状态"""
        self.logger.info("🔄 重置训练器状态")
        if self._execution_engine:
            self._execution_engine.reset_state()
        self._is_initialized = False

    def __repr__(self) -> str:
        return (
            f"FederatedTrainer("
            f"experiment='{self.experiment_name}', "
            f"learner='{self.config.get('learner', 'unknown')}', "
            f"clients={self.config.get('num_clients', 0)}, "
            f"mode='{self.get_execution_mode() or 'auto'}'"
            f")"
        )