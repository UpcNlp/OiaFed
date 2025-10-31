"""
业务层初始化器 - 负责从配置创建业务组件（trainer/learner）
fedcl/federation/business_initializer.py
"""

from typing import Optional, Type, Any

from ..config import TrainingConfig
from ..learner.base_learner import BaseLearner
from ..trainer.trainer import BaseTrainer
from ..types import ModelData
from ..utils.auto_logger import get_sys_logger
from .components import ServerBusinessComponents, ClientBusinessComponents


class BusinessInitializer:
    """
    业务层初始化器

    职责：
        - 接收 TrainingConfig 配置对象
        - 从装饰器注册表创建业务组件（trainer/learner/aggregator）
        - 返回 ServerBusinessComponents 或 ClientBusinessComponents

    使用方式：
        >>> train_config = TrainingConfig(trainer={"name": "FedAvgTrainer"})
        >>> initializer = BusinessInitializer(train_config, "server")
        >>> server_components = await initializer.initialize_server_components("server_main")
        >>> # server_components 包含 trainer, aggregator, global_model
    """

    def __init__(self, train_config: TrainingConfig, node_role: str):
        """
        初始化业务层初始化器

        Args:
            train_config: 训练配置对象（TrainingConfig）
            node_role: 节点角色（"server" 或 "client"）
        """
        self.train_config = train_config
        self.node_role = node_role
        self.logger = get_sys_logger()

        self.logger.info(
            f"BusinessInitializer created: role={node_role}"
        )

    async def initialize_server_components(self, server_id: str) -> ServerBusinessComponents:
        """
        初始化服务端业务组件

        流程：
            1. 导入组件模块（触发装饰器注册）
            2. 从 TrainingConfig 创建 Trainer
            3. 创建 Aggregator（可选）
            4. 创建全局模型

        Args:
            server_id: 服务端ID

        Returns:
            ServerBusinessComponents: 服务端业务组件集合

        Raises:
            ValueError: 如果配置缺失或组件未找到
        """
        self.logger.info("Initializing server business components...")

        try:
            # 1. 导入组件模块（触发装饰器注册）
            self._import_component_modules()

            # 2. 创建全局模型（需要在 trainer 之前创建）
            global_model = self._create_global_model_from_config()

            # 3. 从 TrainingConfig 创建 Trainer
            trainer = self._create_trainer_from_config(global_model)

            # 4. 创建 Aggregator（可选）
            aggregator = self._create_aggregator_from_config()

            self.logger.info("Server business components initialized successfully")

            return ServerBusinessComponents(
                trainer=trainer,
                aggregator=aggregator,
                global_model=global_model
            )

        except Exception as e:
            self.logger.exception(f"Server business components initialization failed: {e}")
            raise

    async def initialize_client_components(self, client_id: str) -> ClientBusinessComponents:
        """
        初始化客户端业务组件

        流程：
            1. 导入组件模块（触发装饰器注册）
            2. 从 TrainingConfig 创建 Learner
            3. 创建数据集（可选）

        Args:
            client_id: 客户端ID

        Returns:
            ClientBusinessComponents: 客户端业务组件集合

        Raises:
            ValueError: 如果配置缺失或组件未找到
        """
        self.logger.debug("Initializing client business components...")

        try:
            # 1. 导入组件模块（触发装饰器注册）
            self._import_component_modules()

            # 2. 从 TrainingConfig 创建 Learner
            learner = self._create_learner_from_config(client_id)

            # 3. 创建数据集（可选）
            dataset = self._create_dataset_from_config()

            self.logger.debug("Client business components initialized successfully")

            return ClientBusinessComponents(
                learner=learner,
                dataset=dataset
            )

        except Exception as e:
            self.logger.exception(f"Client business components initialization failed: {e}")
            raise

    def _import_component_modules(self):
        """
        导入用户定义的组件模块（触发装饰器注册）

        从 TrainingConfig.components 字段读取模块列表并导入
        """
        components = self.train_config.components or []

        if not components:
            self.logger.warning("No component modules specified in training config")
            return

        for module_name in components:
            try:
                __import__(module_name)
                self.logger.debug(f"✓ Imported component module: {module_name}")
            except ImportError as e:
                self.logger.warning(f"✗ Failed to import module: {module_name} - {e}")
            except Exception as e:
                self.logger.exception(f"✗ Error importing module: {module_name} - {e}")

    def _create_trainer_from_config(self, global_model: ModelData) -> BaseTrainer:
        """
        从 TrainingConfig 创建 Trainer

        Args:
            global_model: 全局模型

        Returns:
            BaseTrainer: Trainer 实例

        Raises:
            ValueError: 如果配置缺失或 Trainer 类未找到
        """
        if not self.train_config.trainer:
            raise ValueError("Missing trainer configuration in training config")

        trainer_name = self.train_config.trainer.get("name")
        if not trainer_name:
            raise ValueError("Missing 'name' in trainer configuration")

        trainer_params = self.train_config.trainer.get("params", {})

        # 从注册表获取 Trainer 类
        trainer_class = self._get_component_from_registry(trainer_name, "trainer")

        # 检查是否有 parsed_config（新的统一初始化策略）
        if hasattr(self.train_config, 'parsed_config') and self.train_config.parsed_config:
            # 使用统一初始化策略
            trainer = trainer_class(
                config=self.train_config.parsed_config,
                lazy_init=True
            )
        else:
            # 使用旧的初始化方式（向后兼容）
            trainer = trainer_class(
                global_model=global_model,
                training_config=trainer_params,
            )

        self.logger.debug(f"Trainer created: {trainer_class.__name__}")

        return trainer

    def _create_learner_from_config(self, client_id: str) -> BaseLearner:
        """
        从 TrainingConfig 创建 Learner

        Args:
            client_id: 客户端ID

        Returns:
            BaseLearner: Learner 实例

        Raises:
            ValueError: 如果配置缺失或 Learner 类未找到
        """
        if not self.train_config.learner:
            raise ValueError("Missing learner configuration in training config")

        learner_name = self.train_config.learner.get("name")
        if not learner_name:
            raise ValueError("Missing 'name' in learner configuration")

        learner_params = self.train_config.learner.get("params", {})

        # 从注册表获取 Learner 类
        learner_class = self._get_component_from_registry(learner_name, "learner")

        # 检查是否有 parsed_config（新的统一初始化策略）
        if hasattr(self.train_config, 'parsed_config') and self.train_config.parsed_config:
            # 使用统一初始化策略
            learner = learner_class(
                client_id=client_id,
                config=self.train_config.parsed_config,
                lazy_init=True
            )
        else:
            # 使用旧的初始化方式（向后兼容）
            learner = learner_class(
                client_id=client_id,
                config=learner_params,
            )

        self.logger.info(f"✓ Learner created: {learner_class.__name__}")

        return learner

    def _create_aggregator_from_config(self) -> Optional[Any]:
        """
        从 TrainingConfig 创建 Aggregator（可选）

        Returns:
            Aggregator 实例，如果配置中未指定则返回 None
        """
        if not self.train_config.aggregator:
            self.logger.debug("No aggregator configuration, skipping")
            return None

        aggregator_name = self.train_config.aggregator.get("name")
        if not aggregator_name:
            self.logger.warning("Aggregator config exists but missing 'name' field")
            return None

        aggregator_params = self.train_config.aggregator.get("params", {})

        # 从注册表获取 Aggregator 类
        aggregator_class = self._get_component_from_registry(aggregator_name, "aggregator")

        # 创建 Aggregator 实例
        aggregator = aggregator_class(**aggregator_params)

        self.logger.info(f"✓ Aggregator created: {aggregator_class.__name__}")

        return aggregator

    def _create_global_model_from_config(self) -> ModelData:
        """
        从 TrainingConfig 创建全局模型

        Returns:
            ModelData: 全局模型数据
        """
        if not self.train_config.model:
            # 使用默认模型
            self.logger.info("No model config, using default initial weights")
            return {"weights": 1.0, "round": 0}

        model_config = self.train_config.model

        # 方式1：从检查点加载
        checkpoint_path = model_config.get("checkpoint")
        if checkpoint_path:
            self.logger.info(f"Loading model from checkpoint: {checkpoint_path}")
            return self._load_model_from_checkpoint(checkpoint_path)

        # 方式2：使用初始权重
        initial_weights = model_config.get("initial_weights", 1.0)
        self.logger.info(f"Creating model with initial weights: {initial_weights}")

        return {
            "weights": initial_weights,
            "round": 0
        }

    def _load_model_from_checkpoint(self, checkpoint_path: str) -> ModelData:
        """
        从检查点加载模型

        Args:
            checkpoint_path: 检查点文件路径

        Returns:
            ModelData: 加载的模型数据

        Note:
            这是一个简化实现，实际应该根据模型类型加载
        """
        # TODO: 实现真正的检查点加载逻辑
        self.logger.warning(f"Checkpoint loading not fully implemented: {checkpoint_path}")

        return {
            "weights": 1.0,
            "round": 0,
            "checkpoint": checkpoint_path
        }

    def _create_dataset_from_config(self) -> Optional[Any]:
        """
        从 TrainingConfig 创建数据集（可选）

        Returns:
            数据集对象，如果配置中未指定则返回 None
        """
        if not self.train_config.dataset:
            self.logger.debug("No dataset configuration, skipping")
            return None

        dataset_name = self.train_config.dataset.get("name")
        if not dataset_name:
            self.logger.warning("Dataset config exists but missing 'name' field")
            return None

        dataset_params = self.train_config.dataset.get("params", {})

        # TODO: 从注册表或工厂创建数据集
        self.logger.info(f"Dataset config found: {dataset_name}")

        return {
            "name": dataset_name,
            **dataset_params
        }

    def _get_component_from_registry(
        self,
        component_name: str,
        component_type: str
    ) -> Type:
        """
        从装饰器注册表获取组件类

        Args:
            component_name: 组件名称（例如 "FedAvgTrainer"）
            component_type: 组件类型（例如 "trainer", "learner", "aggregator"）

        Returns:
            组件类

        Raises:
            ValueError: 如果组件未在注册表中找到
        """
        from ..api.registry import registry

        component_class = registry.get(component_name, component_type)

        if component_class is None:
            available = registry.list_all_components()
            raise ValueError(
                f"{component_type.capitalize()} '{component_name}' not found in registry. "
                f"Available components: {available}"
            )

        return component_class
