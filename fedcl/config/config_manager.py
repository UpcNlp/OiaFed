"""
Unified Configuration Manager for FedCL Framework

This module provides:
- Configuration loading (YAML/JSON/TOML)
- Validation, merging, inheritance, env expansion
- DataLoaderFactory & StateManager integration
- Backward-compatible ConfigManager interface
"""

import os
import re
import copy
import json
import yaml
import torch
from pathlib import Path
import toml
import pickle
import torch
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Callable, Union, Optional
from dataclasses import dataclass

from loguru import logger
from omegaconf import DictConfig as OmegaDictConfig, OmegaConf

from .exceptions import (
    ConfigValidationError, ConfigLoadError, ConfigMergeError,
    ConfigReferenceError, ConfigFormatError
)
from .schema_validator import SchemaValidator, ValidationResult

# ===================== 异常类（统一） =====================
class ConfigError(ConfigLoadError): pass
class DataLoaderError(Exception): pass
class StateManagerError(Exception): pass
class SchedulerError(Exception): pass
class ExecutionError(Exception): pass

# ===================== DictConfig（增强配置类） =====================
class DictConfig(OmegaDictConfig):
    def __init__(self, data=None):
        if data is None:
            data = {}
        elif isinstance(data, (DictConfig, OmegaDictConfig)):
            data = OmegaConf.to_container(data, resolve=True)
        super().__init__(data)

    def get(self, key: str, default=None):
        return OmegaConf.select(self, key) or default

    def to_dict(self):
        return OmegaConf.to_container(self, resolve=True)

    def copy(self):
        return DictConfig(self.to_dict())

    def update(self, other):
        other = OmegaConf.to_container(other, resolve=True) if isinstance(other, (DictConfig, OmegaDictConfig)) else other
        merged = OmegaConf.merge(self, other)
        self.clear()
        self.update_config(merged)

    def update_config(self, other):
        if isinstance(other, dict):
            other = OmegaConf.create(other)
        for k, v in other.items():
            self[k] = v

    def has_key(self, key: str) -> bool:
        return OmegaConf.select(self, key) is not None

    def set_value(self, key: str, value: Any):
        OmegaConf.set(self, key, value)

    def __repr__(self):
        return f"DictConfig({OmegaConf.to_yaml(self)})"

    def pretty_print(self):
        print(OmegaConf.to_yaml(self))

# ===================== 配置管理器（兼容版） =====================
class ConfigManager:
    SUPPORTED_FORMATS = {'.yaml', '.yml', '.json', '.toml'}
    ENV_VAR_PATTERN = re.compile(r'\$\{([^}]+)\}')
    REF_PATTERN = re.compile(r'\$\{ref:([^}]+)\}')

    def __init__(self,
                 config_path: Optional[str] = None,
                 config_dict: Optional[Dict[str, Any]] = None,
                 schema_validator: Optional[SchemaValidator] = None):
        self.schema_validator = schema_validator or SchemaValidator()
        self._config_cache: Dict[str, DictConfig] = {}
        self._config_history: List[Tuple[datetime, DictConfig]] = []
        self._config_hooks: List[Callable[[DictConfig], DictConfig]] = []
        self._reference_stack: List[str] = []

        if config_dict:
            self.config = DictConfig(config_dict)
            logger.debug("Configuration loaded from dictionary")
        elif config_path:
            self.config = self.load_config(config_path)
            logger.debug(f"Configuration loaded from file: {config_path}")
        else:
            raise ConfigError("Either config_path or config_dict must be provided")

        # 只对非客户端配置进行验证
        if not self._is_client_config(self.config.to_dict()):
            self._validate_config()

    # ========== 配置加载与处理 ==========
    def load_config(self, config_path: Union[str, Path]) -> DictConfig:
        config_path = Path(config_path)
        cache_key = str(config_path.absolute())

        if cache_key in self._config_cache:
            return self._config_cache[cache_key].copy()

        if not config_path.exists():
            raise ConfigLoadError("Configuration file not found", config_path=str(config_path))

        try:
            raw_config = self._load_raw_config(config_path)
            config = DictConfig(raw_config)

            if 'base_config' in config:
                config = self._resolve_base_config(config, config_path.parent)

            config = self.resolve_references(config)
            config = self.expand_environment_variables(config)

            for hook in self._config_hooks:
                config = hook(config)

            # 判断是否为客户端配置，如果是则跳过验证
            if self._is_client_config(config.to_dict()):
                logger.debug("Detected client configuration, skipping experiment validation")
            else:
                result = self.schema_validator.validate_experiment_config(config.to_dict())
                if not result.is_valid:
                    raise ConfigValidationError("Configuration validation failed", details=result.errors)

            self._config_cache[cache_key] = config.copy()
            self._add_to_history(config)
            logger.success(f"Loaded configuration from: {config_path}")
            return config.copy()

        except Exception as e:
            raise ConfigLoadError("Failed to load configuration", config_path=str(config_path), cause=e)

    def _is_client_config(self, config: Dict) -> bool:
        """判断是否为客户端配置
        
        Args:
            config: 配置字典
            
        Returns:
            bool: 是否为客户端配置
        """
        # 客户端配置通常包含这些字段，而实验配置不包含
        client_indicators = ['client', 'dataloaders', 'learners', 'schedulers']
        # 实验配置必须包含的字段
        experiment_required = ['name', 'method', 'federation']
        
        # 检查是否包含客户端特有字段
        has_client_fields = any(field in config for field in client_indicators)
        # 检查是否缺少实验配置必需字段
        missing_experiment_fields = any(field not in config for field in experiment_required)
        
        # 如果有客户端字段且缺少实验必需字段，判断为客户端配置
        return has_client_fields and missing_experiment_fields

    # ========== 配置验证 ==========
    def _validate_config(self):
        required_sections = ['dataloaders', 'learners', 'schedulers', 'training_plan']
        for section in required_sections:
            if not self.config.has_key(section):
                raise ConfigError(f"Missing required configuration section: {section}")

        training_plan = self.config.get('training_plan')
        if not training_plan.get('total_epochs'):
            raise ConfigError("Missing 'total_epochs' in training_plan")
        if not training_plan.get('phases'):
            raise ConfigError("Missing 'phases' in training_plan")

        for phase in training_plan['phases']:
            for field in ['name', 'epochs', 'learner', 'scheduler']:
                if field not in phase:
                    raise ConfigError(f"Missing field '{field}' in phase '{phase.get('name')}'")

    # ========== 配置获取方法（兼容旧接口） ==========
    def get_training_config(self) -> Dict[str, Any]:
        return self.config.get('training', {})

    def get_dataloader_configs(self) -> Dict[str, Dict[str, Any]]:
        return self.config.get('dataloaders', {})

    def get_learner_configs(self) -> Dict[str, Dict[str, Any]]:
        return self.config.get('learners', {})

    def get_scheduler_configs(self) -> Dict[str, Dict[str, Any]]:
        return self.config.get('schedulers', {})

    def get_training_plan(self) -> Dict[str, Any]:
        return self.config.get('training_plan', {})

    def get_hook_configs(self) -> Dict[str, Any]:
        return self.config.get('hooks', {})

    def get_system_config(self) -> Dict[str, Any]:
        return self.config.get('system', {})
    
    def get_test_data_configs(self) -> Dict[str, Dict[str, Any]]:
        """获取测试数据集配置"""
        logger.debug(f"尝试获取test_datas配置，当前config键: {list(self.config.keys())}")
        test_data_configs = self.config.get('test_datas', {})
        logger.debug(f"获取到test_datas配置: {test_data_configs}")
        return test_data_configs
    
    def get_evaluator_configs(self) -> Dict[str, Dict[str, Any]]:
        """获取评估器配置"""
        return self.config.get('evaluators', {})
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """获取评估配置"""
        return self.config.get('evaluation', {})

    def get_config_value(self, key_path: str, default=None):
        return self.config.get(key_path, default)

    # ========== 内部工具方法 ==========
    def _load_raw_config(self, config_path: Path) -> Dict[str, Any]:
        suffix = config_path.suffix.lower()
        if suffix not in self.SUPPORTED_FORMATS:
            raise ConfigFormatError("Unsupported config format", format_type=suffix)

        with open(config_path, 'r', encoding='utf-8') as f:
            if suffix in {'.yaml', '.yml'}:
                return yaml.safe_load(f) or {}
            elif suffix == '.json':
                return json.load(f)
            elif suffix == '.toml':
                return toml.load(f)
        return {}

    def resolve_references(self, config: DictConfig) -> DictConfig:
        resolved = self._resolve_references_recursive(config.to_dict(), config.to_dict())
        self._reference_stack.clear()
        return DictConfig(resolved)

    def expand_environment_variables(self, config: DictConfig) -> DictConfig:
        expanded = self._expand_env_vars_recursive(config.to_dict())
        return DictConfig(expanded)

    def register_config_hook(self, hook: Callable[[DictConfig], DictConfig]):
        self._config_hooks.append(hook)

    def _resolve_base_config(self, config: DictConfig, base_dir: Path) -> DictConfig:
        base_path = config.get('base_config')
        if not base_path:
            return config
        if not Path(base_path).is_absolute():
            base_path = base_dir / base_path
        base_config = self.load_config(base_path)
        current = DictConfig({k: v for k, v in config.to_dict().items() if k != 'base_config'})
        merged = OmegaConf.merge(base_config, current)
        return DictConfig(merged)

    def _resolve_references_recursive(self, data, root):
        if isinstance(data, dict):
            return {k: self._resolve_references_recursive(v, root) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._resolve_references_recursive(item, root) for item in data]
        elif isinstance(data, str):
            return self._resolve_string_references(data, root)
        return data

    def _resolve_string_references(self, text: str, root: Dict[str, Any]) -> Any:
        matches = list(self.REF_PATTERN.finditer(text))
        if not matches:
            return text
        if len(matches) == 1 and matches[0].group(0) == text:
            return self._get_reference_value(matches[0].group(1), root)
        result = text
        for match in reversed(matches):
            ref_value = self._get_reference_value(match.group(1), root)
            result = result[:match.start()] + str(ref_value) + result[match.end():]
        return result

    def _get_reference_value(self, ref_path: str, root: Dict[str, Any]) -> Any:
        if ref_path in self._reference_stack:
            raise ConfigReferenceError("Circular reference detected", reference_path=ref_path)
        self._reference_stack.append(ref_path)
        try:
            current = root
            for key in ref_path.split('.'):
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    raise ConfigReferenceError("Reference not found", reference_path=ref_path)
            return self._resolve_references_recursive(current, root)
        finally:
            self._reference_stack.pop()

    def _expand_env_vars_recursive(self, data):
        if isinstance(data, dict):
            return {k: self._expand_env_vars_recursive(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._expand_env_vars_recursive(item) for item in data]
        elif isinstance(data, str):
            return self.ENV_VAR_PATTERN.sub(lambda m: os.getenv(m.group(1).split(':')[0], m.group(1).split(':')[1] if ':' in m.group(1) else m.group(0)), data)
        return data

    def _add_to_history(self, config: DictConfig):
        self._config_history.append((datetime.now(), config.copy()))
        if len(self._config_history) > 100:
            self._config_history = self._config_history[-100:]

# ===================== DataLoaderFactory =====================
class BaseDataLoaderFactory(ABC):
    @abstractmethod
    def create_dataloader(self, config: Dict[str, Any]) -> Any: ...

class StandardDataLoaderFactory(BaseDataLoaderFactory):
    def create_dataloader(self, config: Dict[str, Any]) -> Any:
        try:
            # 检查是否有dataset_config，如果有则创建真实数据集
            if 'dataset_config' in config:
                dataset_config = config['dataset_config']
                dataset_name = dataset_config.get('name', '').upper()
                
                if dataset_name == 'MNIST':
                    return self._create_mnist_dataloader(config, dataset_config)
                elif dataset_name == 'CIFAR10':
                    return self._create_cifar10_dataloader(config, dataset_config)
            
            # 如果没有dataset_config，创建默认的mock数据
            from torch.utils.data import DataLoader, TensorDataset
            batch_size = config.get('loader_params', {}).get('batch_size', 32)
            num_samples = config.get('num_samples', 1000)
            input_size = config.get('input_size', [3, 224, 224])
            num_classes = config.get('num_classes', 10)

            data = torch.randn(num_samples, *input_size)
            labels = torch.randint(0, num_classes, (num_samples,))
            dataset = TensorDataset(data, labels)
            loader_params = config.get('loader_params', {})
            return DataLoader(
                dataset,
                batch_size=loader_params.get('batch_size', 32),
                shuffle=loader_params.get('shuffle', True),
                num_workers=loader_params.get('num_workers', 0),
                pin_memory=loader_params.get('pin_memory', False)
            )
        except Exception as e:
            raise DataLoaderError(f"Failed to create standard dataloader: {e}")
    
    def _create_mnist_dataloader(self, config: Dict[str, Any], dataset_config: Dict[str, Any]) -> Any:
        """创建MNIST数据加载器"""
        try:
            import torchvision
            import torchvision.transforms as transforms
            from torch.utils.data import DataLoader, Subset
            from pathlib import Path
            import torch
            
            # 数据变换
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            
            # 创建MNIST数据集
            data_path = dataset_config.get('path', 'data/MNIST')
            split = dataset_config.get('split', 'train')
            download = dataset_config.get('download', True)
            
            full_dataset = torchvision.datasets.MNIST(
                root=data_path,
                train=(split == 'train'),
                download=download,
                transform=transform
            )
            
            # 检查是否需要联邦数据分割
            federated_config = config.get('federated_config', {})
            if federated_config:
                client_id = federated_config.get('client_id', 1)
                num_clients = federated_config.get('num_clients', 1) 
                samples_per_client = federated_config.get('samples_per_client')
                distribution = federated_config.get('distribution', 'iid')
                
                logger.debug(f"Creating federated MNIST dataset for client {client_id}/{num_clients}")
                logger.debug(f"Full dataset size: {len(full_dataset)}")
                
                if samples_per_client:
                    # 使用指定的每客户端样本数
                    if distribution == 'iid':
                        # IID分布：随机选择样本
                        total_samples = len(full_dataset)
                        start_idx = (client_id - 1) * samples_per_client
                        end_idx = min(start_idx + samples_per_client, total_samples)
                        
                        # 生成随机索引
                        torch.manual_seed(42)  # 确保可重现
                        indices = torch.randperm(total_samples).tolist()
                        client_indices = indices[start_idx:end_idx]
                        
                        dataset = Subset(full_dataset, client_indices)
                        logger.debug(f"Created IID subset for client {client_id}: {len(dataset)} samples (indices {start_idx}-{end_idx-1})")
                    else:
                        # 非IID分布：使用专门的分割API
                        try:
                            from fedcl.data.split_api import DataSplitAPI
                            from omegaconf import OmegaConf
                            
                            # 创建联邦数据分割API
                            split_api = DataSplitAPI()
                            
                            # 构建分割配置
                            split_config = OmegaConf.create({
                                'name': f'mnist_split_client_{client_id}',
                                'dataset': {
                                    'name': 'MNIST',
                                    'path': dataset_config.get('path', 'data/MNIST'),
                                    'split': dataset_config.get('split', 'train')
                                },
                                'split': {
                                    'num_客户端': num_clients,
                                    'strategy': {
                                        'method': distribution,
                                        'params': {
                                            'samples_per_client': samples_per_client
                                        }
                                    }
                                },
                                'output': {
                                    'path': f'./temp_split_client_{client_id}'
                                }
                            })
                            
                            # 执行分割并获取结果
                            split_results = split_api.execute_split(split_config)
                            
                            # 从分割结果中获取当前客户端的数据集
                            client_datasets = split_results.get('split_datasets', {})
                            if str(client_id) in client_datasets:
                                dataset = client_datasets[str(client_id)]
                                logger.debug(f"Created non-IID dataset for client {client_id}: {len(dataset)} samples")
                            else:
                                logger.error(f"Client ID {client_id} not found in split results")
                                # 回退到简单的索引分割
                                total_samples = len(full_dataset)
                                start_idx = (client_id - 1) * samples_per_client
                                end_idx = min(start_idx + samples_per_client, total_samples)
                                indices = list(range(start_idx, end_idx))
                                dataset = Subset(full_dataset, indices)
                                logger.debug(f"Used fallback split for client {client_id}: {len(dataset)} samples")
                                
                        except ImportError as e:
                            logger.warning(f"DataSplitAPI not available: {e}, using IID fallback")
                            # 回退到IID分布
                            total_samples = len(full_dataset)
                            start_idx = (client_id - 1) * samples_per_client
                            end_idx = min(start_idx + samples_per_client, total_samples)
                            
                            torch.manual_seed(42)
                            indices = torch.randperm(total_samples).tolist()
                            client_indices = indices[start_idx:end_idx]
                            
                            dataset = Subset(full_dataset, client_indices)
                            logger.debug(f"Created IID fallback dataset for client {client_id}: {len(dataset)} samples")
                            
                        except Exception as e:
                            logger.error(f"Failed to create non-IID split: {e}, using simple fallback")
                            # 最简单的回退：按顺序分割
                            total_samples = len(full_dataset)
                            start_idx = (client_id - 1) * samples_per_client
                            end_idx = min(start_idx + samples_per_client, total_samples)
                            indices = list(range(start_idx, end_idx))
                            dataset = Subset(full_dataset, indices)
                            logger.debug(f"Used simple fallback for client {client_id}: {len(dataset)} samples")
                else:
                    # 均匀分割数据集
                    total_samples = len(full_dataset)
                    samples_per_client = total_samples // num_clients
                    start_idx = (client_id - 1) * samples_per_client
                    end_idx = start_idx + samples_per_client if client_id < num_clients else total_samples
                    
                    indices = list(range(start_idx, end_idx))
                    dataset = Subset(full_dataset, indices)
                    logger.debug(f"Created evenly split subset for client {client_id}: {len(dataset)} samples")
            else:
                # 没有联邦配置，使用完整数据集
                dataset = full_dataset
                logger.debug(f"Using full MNIST dataset: {len(dataset)} samples")
            
            # 获取loader参数
            loader_params = config.get('loader_params', {})
            
            dataloader = DataLoader(
                dataset,
                batch_size=loader_params.get('batch_size', 32),
                shuffle=loader_params.get('shuffle', True),
                num_workers=loader_params.get('num_workers', 0),
                pin_memory=loader_params.get('pin_memory', False),
                drop_last=loader_params.get('drop_last', False)
            )
            
            logger.info(f"Created MNIST DataLoader: {len(dataset)} samples, {len(dataloader)} batches, batch_size={dataloader.batch_size}")
            return dataloader
            
        except Exception as e:
            raise DataLoaderError(f"Failed to create MNIST dataloader: {e}")
    
    def _create_cifar10_dataloader(self, config: Dict[str, Any], dataset_config: Dict[str, Any]) -> Any:
        """创建CIFAR10数据加载器"""
        try:
            import torchvision
            import torchvision.transforms as transforms
            from torch.utils.data import DataLoader
            
            # 数据变换
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
            
            # 创建CIFAR10数据集
            data_path = dataset_config.get('path', 'data/CIFAR10')
            split = dataset_config.get('split', 'train')
            download = dataset_config.get('download', True)
            
            dataset = torchvision.datasets.CIFAR10(
                root=data_path,
                train=(split == 'train'),
                download=download,
                transform=transform
            )
            
            # 获取loader参数
            loader_params = config.get('loader_params', {})
            
            return DataLoader(
                dataset,
                batch_size=loader_params.get('batch_size', 32),
                shuffle=loader_params.get('shuffle', True),
                num_workers=loader_params.get('num_workers', 0),
                pin_memory=loader_params.get('pin_memory', False),
                drop_last=loader_params.get('drop_last', False)
            )
        except Exception as e:
            raise DataLoaderError(f"Failed to create CIFAR10 dataloader: {e}")

class DataLoaderFactory:
    def __init__(self, dataloader_configs: Dict[str, Dict[str, Any]]):
        self.dataloader_configs = dataloader_configs
        self.factories = {
            'StandardDataLoader': StandardDataLoaderFactory(),
            'DiffusionDataLoader': StandardDataLoaderFactory(),
            'FeatureDataLoader': StandardDataLoaderFactory(),
        }
        self.created_dataloaders = {}
        logger.debug("DataLoaderFactory initialized")

    def register_factory(self, loader_type: str, factory: BaseDataLoaderFactory):
        self.factories[loader_type] = factory

    def create_dataloader(self, dataloader_id: str, config: Dict[str, Any]):
        if dataloader_id in self.created_dataloaders:
            return self.created_dataloaders[dataloader_id]

        loader_type = config.get('type', 'StandardDataLoader')
        if loader_type not in self.factories:
            raise DataLoaderError(f"Unknown dataloader type: {loader_type}")

        dataloader = self.factories[loader_type].create_dataloader(config)
        self.created_dataloaders[dataloader_id] = dataloader
        logger.debug(f"Created dataloader '{dataloader_id}' of type '{loader_type}'")
        return dataloader

    def get_dataloader(self, dataloader_id: str):
        if dataloader_id not in self.created_dataloaders:
            raise DataLoaderError(f"Dataloader '{dataloader_id}' not found")
        return self.created_dataloaders[dataloader_id]

    def cleanup(self):
        self.created_dataloaders.clear()
        logger.debug("DataLoaderFactory cleaned up")

# ===================== StateManager =====================
@dataclass
class StateInfo:
    phase_name: str
    timestamp: float
    data: Dict[str, Any]
    size_bytes: int

class StateManager:
    def __init__(self,
                 checkpoint_dir: Optional[str] = None,
                 max_states_in_memory: int = 10):
        self.checkpoint_dir = Path(checkpoint_dir or "./checkpoints")
        self.max_states_in_memory = max_states_in_memory
        self.phase_states: Dict[str, StateInfo] = {}
        self.knowledge_store: Dict[str, Dict[str, Any]] = {}
        self.transfer_rules: Dict[str, List[str]] = {}
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"StateManager initialized with checkpoint_dir: {self.checkpoint_dir}")

    def save_phase_state(self, phase_name: str, state: Dict[str, Any]):
        import time
        try:
            state_bytes = len(pickle.dumps(state))
            info = StateInfo(phase_name, time.time(), state, state_bytes)
            self.phase_states[phase_name] = info
            if len(self.phase_states) > self.max_states_in_memory:
                self._persist_oldest_states()
            self._save_state_to_disk(phase_name, state)
            logger.debug(f"Saved state for phase '{phase_name}' ({state_bytes} bytes)")
        except Exception as e:
            raise StateManagerError(f"Failed to save phase state '{phase_name}': {e}")

    def load_phase_state(self, phase_name: str) -> Optional[Dict[str, Any]]:
        try:
            if phase_name in self.phase_states:
                return self.phase_states[phase_name].data
            state = self._load_state_from_disk(phase_name)
            if state:
                logger.debug(f"Loaded state for phase '{phase_name}' from disk")
                return state
            logger.warning(f"No state found for phase '{phase_name}'")
            return None
        except Exception as e:
            raise StateManagerError(f"Failed to load phase state '{phase_name}': {e}")

    def save_knowledge(self, phase_name: str, knowledge: Dict[str, Any]):
        try:
            self.knowledge_store[phase_name] = knowledge
            with open(self.checkpoint_dir / f"{phase_name}_knowledge.pkl", 'wb') as f:
                pickle.dump(knowledge, f)
            logger.debug(f"Saved knowledge for phase '{phase_name}'")
        except Exception as e:
            raise StateManagerError(f"Failed to save knowledge: {e}")

    def load_knowledge(self, phase_name: str) -> Optional[Dict[str, Any]]:
        try:
            if phase_name in self.knowledge_store:
                return self.knowledge_store[phase_name]
            path = self.checkpoint_dir / f"{phase_name}_knowledge.pkl"
            if path.exists():
                with open(path, 'rb') as f:
                    knowledge = pickle.load(f)
                self.knowledge_store[phase_name] = knowledge
                return knowledge
            return None
        except Exception as e:
            raise StateManagerError(f"Failed to load knowledge: {e}")

    def prepare_inherited_state(self, target_phase: str, source_phases: List[str]) -> Dict[str, Any]:
        inherited = {}
        for phase in source_phases:
            state = self.load_phase_state(phase)
            if state:
                inherited.update(state)
            knowledge = self.load_knowledge(phase)
            if knowledge:
                inherited.setdefault('knowledge', {}).update(knowledge)
        logger.debug(f"Prepared inherited state for '{target_phase}' from {source_phases}")
        return inherited

    def _save_state_to_disk(self, phase_name: str, state: Dict[str, Any]):
        with open(self.checkpoint_dir / f"{phase_name}_state.pkl", 'wb') as f:
            pickle.dump(state, f)

    def _load_state_from_disk(self, phase_name: str) -> Optional[Dict[str, Any]]:
        path = self.checkpoint_dir / f"{phase_name}_state.pkl"
        if path.exists():
            with open(path, 'rb') as f:
                return pickle.load(f)
        return None

    def _persist_oldest_states(self):
        sorted_states = sorted(self.phase_states.items(), key=lambda x: x[1].timestamp)
        num = len(self.phase_states) - self.max_states_in_memory + 1
        for i in range(num):
            del self.phase_states[sorted_states[i][0]]

    def get_state_summary(self) -> Dict[str, Any]:
        return {
            "states_in_memory": len(self.phase_states),
            "knowledge_items": len(self.knowledge_store),
            "total_memory_size": sum(info.size_bytes for info in self.phase_states.values()),
            "checkpoint_dir": str(self.checkpoint_dir),
            "phases": list(self.phase_states.keys())
        }

    def cleanup(self):
        self.phase_states.clear()
        self.knowledge_store.clear()
        self.transfer_rules.clear()
        logger.debug("StateManager cleaned up")

# ===================== 默认配置模板（兼容） =====================
def get_default_config_template() -> Dict[str, Any]:
    return {
        "project": {
            "name": "federated_continual_learning",
            "version": "1.0",
            "output_dir": "./outputs",
            "log_level": "INFO"
        },
        "dataloaders": {
            "primary_dataloader": {
                "type": "StandardDataLoader",
                "num_samples": 1000,
                "input_size": [3, 224, 224],
                "num_classes": 10,
                "loader_params": {
                    "batch_size": 64,
                    "shuffle": True,
                    "num_workers": 4,
                    "pin_memory": True
                }
            }
        },
        "learners": {
            "primary_learner": {
                "type": "ContinualLearner",
                "model": {
                    "name": "ResNet18",
                    "num_classes": 10
                },
                "optimizer": {
                    "name": "Adam",
                    "lr": 0.001
                }
            }
        },
        "schedulers": {
            "primary_scheduler": {
                "type": "StandardEpochScheduler",
                "priority": "NORMAL",
                "config": {
                    "error_tolerance": "continue",
                    "max_retries": 3
                }
            }
        },
        "training_plan": {
            "total_epochs": 10,
            "execution_strategy": "sequential",
            "phases": [
                {
                    "name": "main_training",
                    "description": "Main training phase",
                    "epochs": list(range(1, 11)),
                    "learner": "primary_learner",
                    "scheduler": "primary_scheduler",
                    "priority": 0,
                    "execution_mode": "sequential"
                }
            ]
        },
        "system": {
            "device": "cuda",
            "max_concurrent_schedulers": 2,
            "memory_management": {
                "max_memory_usage": "8GB",
                "garbage_collection_frequency": 10
            },
            "logging": {
                "level": "INFO",
                "file": "./outputs/training.log",
                "console": True
            }
        }
    }