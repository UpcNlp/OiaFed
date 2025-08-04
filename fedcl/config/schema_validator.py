# fedcl/config/schema_validator.py
"""
配置模式验证器

提供配置验证功能，支持：
- 多种配置类型验证（实验、模型、数据、通信）
- 嵌套配置验证
- 自定义验证器
- 字段间依赖验证
- 高性能验证（1000字段<100ms）
- 动态组件注册支持（与装饰器系统集成）
"""

from typing import Dict, List, Any, Optional, Callable, Union
from pathlib import Path
from dataclasses import dataclass, field
import json
import yaml
import re
from collections import defaultdict
import time
from loguru import logger

from .exceptions import (
    ConfigValidationError, SchemaNotFoundError, SchemaLoadError,
    RequiredFieldError, TypeValidationError, RangeValidationError,
    EnumValidationError, PatternValidationError, CustomValidationError,
    handle_validation_errors
)
from .utils import load_config_file, get_nested_value


@dataclass
class ValidationError:
    """验证错误信息"""
    field: str
    message: str
    error_code: str = ""
    value: Any = None
    
    def __str__(self) -> str:
        return f"Field '{self.field}': {self.message}"


@dataclass
class ValidationWarning:
    """验证警告信息"""
    field: str
    message: str
    warning_code: str = ""
    value: Any = None
    
    def __str__(self) -> str:
        return f"Field '{self.field}': {self.message}"


@dataclass
class ValidationResult:
    """验证结果"""
    is_valid: bool = True
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationWarning] = field(default_factory=list)
    
    def add_error(self, field: str, message: str, error_code: str = "", value: Any = None) -> None:
        """添加验证错误"""
        self.errors.append(ValidationError(field, message, error_code, value))
        self.is_valid = False
    
    def add_warning(self, field: str, message: str, warning_code: str = "", value: Any = None) -> None:
        """添加验证警告"""
        self.warnings.append(ValidationWarning(field, message, warning_code, value))
    
    def merge(self, other: 'ValidationResult') -> 'ValidationResult':
        """合并验证结果"""
        merged = ValidationResult(
            is_valid=self.is_valid and other.is_valid,
            errors=self.errors + other.errors,
            warnings=self.warnings + other.warnings
        )
        return merged


class SchemaValidator:
    """配置模式验证器
    
    支持多种配置类型的验证，包括类型检查、范围检查、必需字段检查等。
    现已支持动态组件注册与装饰器系统集成。
    """
    
    @handle_validation_errors
    def __init__(self, schema_path: Optional[Path] = None):
        """初始化验证器"""
        self.schemas: Dict[str, Dict] = {}
        self.custom_validators: Dict[str, Callable] = {}
        self._validation_errors: List[ValidationError] = []
        self._component_registry = None
        
        # 加载默认模式
        self._load_default_schemas()
        
        # 如果提供了模式文件路径，尝试加载外部模式
        if schema_path and schema_path.exists():
            try:
                self._load_schemas_from_file(schema_path)
            except Exception as e:
                # 记录错误但不抛出异常，使验证器仍能正常工作
                logger.error(f"Failed to load schemas from {schema_path}: {e}")
                logger.warning("Using default schemas only")
                # 可选：将加载错误记录到验证错误列表中
                self._validation_errors.append(
                    ValidationError("schema_loading", f"Failed to load schema file: {e}", "SCHEMA_LOAD_ERROR")
                )
        
        logger.debug(f"SchemaValidator initialized with {len(self.schemas)} schemas")
    
    def _get_component_registry(self):
        """获取组件注册表实例"""
        if self._component_registry is None:
            try:
                from fedcl.registry.component_registry import registry
                self._component_registry = registry
                logger.debug("Successfully connected to global ComponentRegistry")
            except ImportError:
                logger.warning("ComponentRegistry not available, using static validation")
                self._component_registry = None
            except Exception as e:
                logger.warning(f"Failed to get ComponentRegistry: {e}, using static validation")
                self._component_registry = None
        return self._component_registry
    
    def _get_registered_components(self, component_type: str) -> List[str]:
        """获取已注册的组件列表"""
        registry = self._get_component_registry()
        if registry is None:
            return []
        
        try:
            # 使用注册表的 list_components 方法
            if hasattr(registry, 'list_components'):
                components = registry.list_components(component_type)
                logger.debug(f"Found {component_type}s via list_components: {components}")
                return components if isinstance(components, list) else []
            
            # 如果没有 list_components 方法，尝试其他方式
            logger.debug(f"No list_components method, trying direct attribute access")
            return []
            
        except Exception as e:
            logger.debug(f"Error getting {component_type}: {e}")
            return []
    
    def _validate_dynamic_enum(self, value: str, component_type: str, field_name: str) -> tuple[bool, str]:
        """动态验证枚举值（基于已注册的组件）
        
        Args:
            value: 要验证的值
            component_type: 组件类型
            field_name: 字段名称
            
        Returns:
            tuple[bool, str]: (是否有效, 错误消息)
        """
        if not isinstance(value, str):
            return False, f"Expected string value for {field_name}, got {type(value).__name__}"
        
        # 获取已注册的组件
        registered_components = self._get_registered_components(component_type)
        
        # 定义静态默认值作为回退
        static_defaults = {
            "learner": ["L2P", "EWC", "Replay", "DDDR"],
            "method": ["L2P", "EWC", "Replay", "DDDR"],
            "dataset": ["CIFAR10", "CIFAR100", "ImageNet-R", "MNIST", "mnist"],
            "aggregator": ["FedAvg", "FedProx", "SCAFFOLD"],
            "optimizer": ["SGD", "Adam", "AdamW", "RMSprop"],
            "loss": ["cross_entropy", "focal_loss", "label_smoothing"]
        }
        
        # 如果有注册的组件，优先检查注册列表
        if registered_components:
            if value in registered_components:
                logger.debug(f"✅ {field_name}='{value}' found in registered {component_type}s")
                return True, ""
            else:
                # 还要检查静态默认值，以防某些内置组件没有通过注册表注册
                default_list = static_defaults.get(component_type, [])
                if value in default_list:
                    logger.debug(f"✅ {field_name}='{value}' found in static defaults for {component_type}")
                    return True, ""
                else:
                    available_components = sorted(set(registered_components + default_list))
                    error_msg = f"'{value}' not found in registered {component_type}s: {available_components}"
                    return False, error_msg
        
        # 如果没有注册的组件，使用静态默认列表
        default_list = static_defaults.get(component_type, [])
        if default_list:
            if value in default_list:
                logger.debug(f"✅ {field_name}='{value}' found in static defaults")
                return True, ""
            else:
                error_msg = f"'{value}' not in available {component_type}s: {default_list}"
                return False, error_msg
        
        # 如果既没有注册组件也没有静态默认值，使用宽松验证
        logger.debug(f"⚠️ No validation constraints for {component_type}, accepting value: '{value}'")
        return True, ""
    
    def validate_experiment_config(self, config: Dict) -> ValidationResult:
        """验证实验配置
        
        Args:
            config: 实验配置字典
            
        Returns:
            ValidationResult: 验证结果
        """
        logger.debug("Validating experiment config")
        start_time = time.time()
        
        result = self._validate_config(config, "experiment")
        
        # 特定的实验配置校验逻辑
        result = self._validate_experiment_specific(config, result)
        
        validation_time = (time.time() - start_time) * 1000
        logger.debug(f"Experiment config validation completed in {validation_time:.2f}ms")
        
        return result
    
    def validate_model_config(self, config: Dict) -> ValidationResult:
        """验证模型配置
        
        Args:
            config: 模型配置字典
            
        Returns:
            ValidationResult: 验证结果
        """
        logger.debug("Validating model config")
        start_time = time.time()
        
        result = self._validate_config(config, "model")
        
        # 特定的模型配置校验逻辑
        result = self._validate_model_specific(config, result)
        
        validation_time = (time.time() - start_time) * 1000
        logger.debug(f"Model config validation completed in {validation_time:.2f}ms")
        
        return result
    
    def validate_data_config(self, config: Dict) -> ValidationResult:
        """验证数据配置
        
        Args:
            config: 数据配置字典
            
        Returns:
            ValidationResult: 验证结果
        """
        logger.debug("Validating data config")
        start_time = time.time()
        
        result = self._validate_config(config, "data")
        
        # 特定的数据配置校验逻辑
        result = self._validate_data_specific(config, result)
        
        validation_time = (time.time() - start_time) * 1000
        logger.debug(f"Data config validation completed in {validation_time:.2f}ms")
        
        return result
    
    def validate_communication_config(self, config: Dict) -> ValidationResult:
        """验证通信配置
        
        Args:
            config: 通信配置字典
            
        Returns:
            ValidationResult: 验证结果
        """
        logger.debug("Validating communication config")
        start_time = time.time()
        
        result = self._validate_config(config, "communication")
        
        # 特定的通信配置校验逻辑
        result = self._validate_communication_specific(config, result)
        
        validation_time = (time.time() - start_time) * 1000
        logger.debug(f"Communication config validation completed in {validation_time:.2f}ms")
        
        return result
    
    def validate_client_config(self, config: Dict) -> ValidationResult:
        """验证客户端配置
        
        Args:
            config: 客户端配置字典
            
        Returns:
            ValidationResult: 验证结果
        """
        logger.debug("Validating client config")
        start_time = time.time()
        
        result = ValidationResult()
        
        # 客户端配置的基本验证
        # 检查必需的顶级字段
        required_sections = ['client', 'dataloaders', 'learners', 'schedulers']
        for section in required_sections:
            if section not in config:
                result.add_error(section, f"Required section '{section}' is missing", "MISSING_REQUIRED")
        
        # 验证评估配置（如果存在）
        if 'test_datas' in config:
            result = self._validate_test_datas_config(config['test_datas'], result)
        
        if 'evaluators' in config:
            result = self._validate_evaluators_config(config['evaluators'], result)
        
        if 'evaluation' in config:
            result = self._validate_evaluation_config(config['evaluation'], result)
        
        validation_time = (time.time() - start_time) * 1000
        logger.debug(f"Client config validation completed in {validation_time:.2f}ms")
        
        return result
    
    def _validate_test_datas_config(self, test_datas: Dict, result: ValidationResult) -> ValidationResult:
        """验证测试数据配置"""
        for test_data_name, test_data_config in test_datas.items():
            if not isinstance(test_data_config, dict):
                result.add_error(f"test_datas.{test_data_name}", "Test data config must be a dictionary", "INVALID_TYPE")
                continue
            
            # 检查必需字段
            if 'dataset_config' not in test_data_config:
                result.add_error(f"test_datas.{test_data_name}", "Missing dataset_config", "MISSING_REQUIRED")
        
        return result
    
    def _validate_evaluators_config(self, evaluators: Dict, result: ValidationResult) -> ValidationResult:
        """验证评估器配置"""
        for evaluator_name, evaluator_config in evaluators.items():
            if not isinstance(evaluator_config, dict):
                result.add_error(f"evaluators.{evaluator_name}", "Evaluator config must be a dictionary", "INVALID_TYPE")
                continue
            
            # 检查必需字段
            required_fields = ['class', 'test_data']
            for field in required_fields:
                if field not in evaluator_config:
                    result.add_error(f"evaluators.{evaluator_name}", f"Missing required field '{field}'", "MISSING_REQUIRED")
        
        return result
    
    def _validate_evaluation_config(self, evaluation: Dict, result: ValidationResult) -> ValidationResult:
        """验证评估任务配置"""
        if not isinstance(evaluation, dict):
            result.add_error("evaluation", "Evaluation config must be a dictionary", "INVALID_TYPE")
            return result
        
        # 检查频率
        frequency = evaluation.get('frequency')
        if frequency is not None:
            if not isinstance(frequency, int) or frequency < 1:
                result.add_error("evaluation.frequency", "Frequency must be a positive integer", "INVALID_VALUE")
        
        # 检查任务列表
        tasks = evaluation.get('tasks')
        if tasks is not None:
            if not isinstance(tasks, list):
                result.add_error("evaluation.tasks", "Tasks must be a list", "INVALID_TYPE")
            else:
                for i, task in enumerate(tasks):
                    if not isinstance(task, dict):
                        result.add_error(f"evaluation.tasks[{i}]", "Task must be a dictionary", "INVALID_TYPE")
                        continue
                    
                    # 检查任务的必需字段
                    required_task_fields = ['learner', 'evaluator', 'test_data']
                    for field in required_task_fields:
                        if field not in task:
                            result.add_error(f"evaluation.tasks[{i}]", f"Missing required field '{field}'", "MISSING_REQUIRED")
        
        return result
    
    def check_required_fields(self, config: Dict, schema: Dict) -> List[str]:
        """检查必需字段
        
        Args:
            config: 配置字典
            schema: 验证模式
            
        Returns:
            List[str]: 缺失的必需字段列表
        """
        missing_fields = []
        required_fields = schema.get("required", [])
        
        for field in required_fields:
            if not self._check_field_exists(config, field):
                missing_fields.append(field)
        
        return missing_fields
    
    def check_field_types(self, config: Dict, schema: Dict) -> List[str]:
        """检查字段类型
        
        Args:
            config: 配置字典
            schema: 验证模式
            
        Returns:
            List[str]: 类型错误的字段列表
        """
        type_errors = []
        properties = schema.get("properties", {})
        
        for field, field_schema in properties.items():
            value = self._get_nested_value(config, field)
            if value is not None:
                expected_type = field_schema.get("type")
                if expected_type and not self._check_field_type(value, expected_type):
                    type_errors.append(field)
        
        return type_errors
    
    def check_field_ranges(self, config: Dict, schema: Dict) -> List[str]:
        """检查字段范围
        
        Args:
            config: 配置字典
            schema: 验证模式
            
        Returns:
            List[str]: 范围错误的字段列表
        """
        range_errors = []
        properties = schema.get("properties", {})
        
        for field, field_schema in properties.items():
            value = self._get_nested_value(config, field)
            if value is not None:
                is_valid, _ = self._check_field_range_with_message(value, field_schema, field)
                if not is_valid:
                    range_errors.append(field)
        
        return range_errors
    
    def add_custom_validator(self, field: str, validator: Callable) -> None:
        """添加自定义验证器
        
        Args:
            field: 字段路径（支持嵌套，如 "model.learning_rate"）
            validator: 验证函数，接受value参数，返回bool或(bool, str)
        """
        self.custom_validators[field] = validator
        logger.debug(f"Added custom validator for field: {field}")
    
    def get_validation_errors(self) -> List[ValidationError]:
        """获取验证错误列表
        
        Returns:
            List[ValidationError]: 验证错误列表
        """
        return self._validation_errors.copy()
    
    def register_schema(self, name: str, schema: Dict) -> None:
        """注册验证模式
        
        Args:
            name: 模式名称
            schema: 验证模式字典
        """
        self.schemas[name] = schema
        logger.debug(f"Registered schema: {name}")
    
    def _load_default_schemas(self) -> None:
        """加载默认验证模式（移除硬编码枚举，支持动态组件）"""
        # 实验配置模式 - 移除硬编码的 method 和 dataset 枚举
        self.schemas["experiment"] = {
            "type": "object",
            "required": ["name", "method", "dataset", "federation"],
            "properties": {
                "name": {"type": "string", "minLength": 1},
                "method": {"type": "string", "_dynamic_enum": "learner"},  # 标记为动态枚举
                "dataset": {"type": "string", "_dynamic_enum": "dataset"},  # 标记为动态枚举
                "federation": {"type": "object"},
                "seed": {"type": "integer", "minimum": 0, "maximum": 2147483647},
                "num_轮次": {"type": "integer", "minimum": 1, "maximum": 10000},
                "tasks_per_round": {"type": "integer", "minimum": 1, "maximum": 100}
            }
        }
        
        # 模型配置模式
        self.schemas["model"] = {
            "type": "object",
            "required": ["architecture", "learning_rate"],
            "properties": {
                "architecture": {"type": "string"},
                "learning_rate": {"type": "number", "minimum": 1e-6, "maximum": 1.0},
                "batch_size": {"type": "integer", "minimum": 1, "maximum": 10000},
                "num_epochs": {"type": "integer", "minimum": 1, "maximum": 1000},
                "optimizer": {"type": "string", "enum": ["SGD", "Adam", "AdamW", "RMSprop"]},  # 保留静态枚举
                "weight_decay": {"type": "number", "minimum": 0.0, "maximum": 1.0}
            }
        }
        
        # 数据配置模式
        self.schemas["data"] = {
            "type": "object",
            "required": ["name", "num_tasks"],
            "properties": {
                "name": {"type": "string"},
                "num_tasks": {"type": "integer", "minimum": 1, "maximum": 100},
                "classes_per_task": {"type": "integer", "minimum": 1, "maximum": 1000},
                "split_strategy": {"type": "string", "enum": ["iid", "non_iid", "dirichlet"]},
                "data_dir": {"type": "string"},
                "download": {"type": "boolean"}
            }
        }
        
        # 通信配置模式
        self.schemas["communication"] = {
            "type": "object",
            "required": ["protocol", "num_客户端"],
            "properties": {
                "protocol": {"type": "string", "enum": ["tcp", "udp", "grpc", "http"]},
                "num_客户端": {"type": "integer", "minimum": 1, "maximum": 10000},
                "timeout": {"type": "number", "minimum": 0.1, "maximum": 3600.0},
                "max_retries": {"type": "integer", "minimum": 0, "maximum": 100},
                "compression": {"type": "boolean"},
                "encryption": {"type": "boolean"}
            }
        }
    
    def _load_schemas_from_file(self, schema_path: Path) -> None:
        """从文件加载验证模式
        
        Args:
            schema_path: 模式文件路径
        """
        try:
            external_schemas = load_config_file(schema_path)
            
            for name, schema in external_schemas.items():
                self.schemas[name] = schema
            
            logger.debug(f"Loaded {len(external_schemas)} schemas from {schema_path}")
        
        except Exception as e:
            logger.error(f"Failed to load schemas from {schema_path}: {e}")
            raise SchemaLoadError(str(schema_path), e)
    
    def _validate_config(self, config: Dict, schema_name: str) -> ValidationResult:
        """验证配置
        
        Args:
            config: 配置字典
            schema_name: 模式名称
            
        Returns:
            ValidationResult: 验证结果
            
        Raises:
            SchemaNotFoundError: 模式未找到时抛出
        """
        result = ValidationResult()
        
        if schema_name not in self.schemas:
            # 返回验证失败结果而不是抛出异常
            result.add_error(
                "schema", 
                f"Validation schema '{schema_name}' not found", 
                "SCHEMA_NOT_FOUND"
            )
            return result
        
        schema = self.schemas[schema_name]
        
        # 检查必需字段
        missing_fields = self.check_required_fields(config, schema)
        for field in missing_fields:
            result.add_error(field, f"Required field '{field}' is missing", "MISSING_REQUIRED")
        
        # 检查字段类型
        type_errors = self.check_field_types(config, schema)
        for field in type_errors:
            expected_type = schema["properties"][field]["type"]
            actual_value = self._get_nested_value(config, field)
            result.add_error(
                field, 
                f"Field '{field}' has incorrect type. Expected {expected_type}, got {type(actual_value).__name__}",
                "INVALID_TYPE",
                actual_value
            )
        
        # 检查字段范围（改进版，支持动态枚举）
        range_errors = self._check_field_ranges_with_dynamic_enum(config, schema)
        for field, error_msg in range_errors:
            actual_value = self._get_nested_value(config, field)
            result.add_error(
                field,
                error_msg,
                "OUT_OF_RANGE",
                actual_value
            )
        
        # 运行自定义验证器
        result = self._run_custom_validators(config, result)
        
        return result
    
    def _check_field_ranges_with_dynamic_enum(self, config: Dict, schema: Dict) -> List[tuple[str, str]]:
        """检查字段范围，支持动态枚举验证
        
        Args:
            config: 配置字典
            schema: 验证模式
            
        Returns:
            List[tuple[str, str]]: 错误字段和错误消息的列表
        """
        range_errors = []
        properties = schema.get("properties", {})
        
        for field, field_schema in properties.items():
            value = self._get_nested_value(config, field)
            if value is None:
                continue
            
            # 检查是否是动态枚举字段
            dynamic_enum_type = field_schema.get("_dynamic_enum")
            if dynamic_enum_type:
                is_valid, error_msg = self._validate_dynamic_enum(value, dynamic_enum_type, field)
                if not is_valid:
                    range_errors.append((field, error_msg))
                continue
            
            # 传统的范围检查
            is_valid, error_msg = self._check_field_range_with_message(value, field_schema, field)
            if not is_valid:
                range_errors.append((field, error_msg))
        
        return range_errors
    
    def _check_field_range_with_message(self, value: Any, field_schema: Dict, field_name: str) -> tuple[bool, str]:
        """检查字段范围并返回详细错误消息
        
        Args:
            value: 字段值
            field_schema: 字段模式
            field_name: 字段名称
            
        Returns:
            tuple[bool, str]: (是否有效, 错误消息)
        """
        # 检查数值范围
        if isinstance(value, (int, float)):
            minimum = field_schema.get("minimum")
            maximum = field_schema.get("maximum")
            
            if minimum is not None and value < minimum:
                return False, f"Value {value} is below minimum {minimum}"
            if maximum is not None and value > maximum:
                return False, f"Value {value} is above maximum {maximum}"
        
        # 检查字符串长度
        if isinstance(value, str):
            min_length = field_schema.get("minLength")
            max_length = field_schema.get("maxLength")
            
            if min_length is not None and len(value) < min_length:
                return False, f"String length {len(value)} is below minimum {min_length}"
            if max_length is not None and len(value) > max_length:
                return False, f"String length {len(value)} is above maximum {max_length}"
        
        # 检查静态枚举值
        enum_values = field_schema.get("enum")
        if enum_values is not None and value not in enum_values:
            return False, f"Value '{value}' is not in allowed values: {enum_values}"
        
        # 检查正则表达式模式
        pattern = field_schema.get("pattern")
        if pattern is not None and isinstance(value, str):
            if not re.match(pattern, value):
                return False, f"Value '{value}' does not match required pattern: {pattern}"
        
        return True, ""
    
    def _validate_experiment_specific(self, config: Dict, result: ValidationResult) -> ValidationResult:
        """实验配置特定验证
        
        Args:
            config: 配置字典
            result: 当前验证结果
            
        Returns:
            ValidationResult: 更新后的验证结果
        """
        # 检查方法和数据集的兼容性
        method = config.get("method")
        dataset = config.get("dataset")
        
        if method == "Replay" and dataset in ["ImageNet-R"]:
            result.add_warning(
                "method",
                f"Replay method with {dataset} may require large memory",
                "MEMORY_WARNING"
            )
        
        # 检查联邦配置与任务数的一致性
        federation = config.get("federation", {})
        num_rounds = config.get("num_轮次", 1)
        tasks_per_round = config.get("tasks_per_round", 1)
        
        if num_rounds * tasks_per_round > 100:
            result.add_warning(
                "experiment",
                "Large number of total tasks may lead to long experiment time",
                "PERFORMANCE_WARNING"
            )
        
        return result
    
    def _validate_model_specific(self, config: Dict, result: ValidationResult) -> ValidationResult:
        """模型配置特定验证
        
        Args:
            config: 配置字典
            result: 当前验证结果
            
        Returns:
            ValidationResult: 更新后的验证结果
        """
        # 检查学习率和优化器的兼容性
        learning_rate = config.get("learning_rate", 0.001)
        optimizer = config.get("optimizer", "Adam")
        
        if optimizer == "SGD" and learning_rate > 0.1:
            result.add_warning(
                "learning_rate",
                "High learning rate with SGD optimizer may cause instability",
                "OPTIMIZER_WARNING"
            )
        
        return result
    
    def _validate_data_specific(self, config: Dict, result: ValidationResult) -> ValidationResult:
        """数据配置特定验证
        
        Args:
            config: 配置字典
            result: 当前验证结果
            
        Returns:
            ValidationResult: 更新后的验证结果
        """
        # 检查任务数和每任务类别数的合理性
        num_tasks = config.get("num_tasks", 1)
        classes_per_task = config.get("classes_per_task")
        
        if classes_per_task and num_tasks * classes_per_task > 1000:
            result.add_warning(
                "data",
                "Large number of total classes may impact performance",
                "SCALE_WARNING"
            )
        
        return result
    
    def _validate_communication_specific(self, config: Dict, result: ValidationResult) -> ValidationResult:
        """通信配置特定验证
        
        Args:
            config: 配置字典
            result: 当前验证结果
            
        Returns:
            ValidationResult: 更新后的验证结果
        """
        # 检查客户端数量和协议的兼容性
        num_clients = config.get("num_客户端", 1)
        protocol = config.get("protocol", "tcp")
        
        if num_clients > 1000 and protocol in ["tcp", "http"]:
            result.add_warning(
                "communication",
                f"Large number of clients ({num_clients}) with {protocol} protocol may cause performance issues",
                "SCALABILITY_WARNING"
            )
        
        return result
    
    def _run_custom_validators(self, config: Dict, result: ValidationResult) -> ValidationResult:
        """运行自定义验证器
        
        Args:
            config: 配置字典
            result: 当前验证结果
            
        Returns:
            ValidationResult: 更新后的验证结果
        """
        for field, validator in self.custom_validators.items():
            try:
                value = self._get_nested_value(config, field)
                if value is not None:
                    validation_result = validator(value)
                    
                    if isinstance(validation_result, tuple):
                        is_valid, message = validation_result
                        if not is_valid:
                            result.add_error(field, message, "CUSTOM_VALIDATION")
                    elif not validation_result:
                        result.add_error(field, f"Custom validation failed for field '{field}'", "CUSTOM_VALIDATION")
                        
            except Exception as e:
                logger.error(f"Custom validator for field '{field}' failed: {e}")
                result.add_error(field, f"Custom validator error: {str(e)}", "VALIDATOR_ERROR")
        
        return result
    
    def _check_field_exists(self, config: Dict, field: str) -> bool:
        """检查字段是否存在
        
        Args:
            config: 配置字典
            field: 字段路径
            
        Returns:
            bool: 字段是否存在
        """
        return self._get_nested_value(config, field) is not None
    
    def _get_nested_value(self, config: Dict, field: str) -> Any:
        """获取嵌套字段值
        
        Args:
            config: 配置字典
            field: 字段路径（如 "model.learning_rate"）
            
        Returns:
            Any: 字段值，不存在时返回None
        """
        return get_nested_value(config, field)
    
    def _check_field_type(self, value: Any, expected_type: str) -> bool:
        """检查字段类型
        
        Args:
            value: 字段值
            expected_type: 期望类型
            
        Returns:
            bool: 类型是否匹配
        """
        type_mapping = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict
        }
        
        expected_python_type = type_mapping.get(expected_type)
        if expected_python_type is None:
            return True  # 未知类型，跳过检查
        
        return isinstance(value, expected_python_type)
    
    def _check_field_range(self, value: Any, field_schema: Dict) -> bool:
        """检查字段范围（保持向后兼容性）
        
        Args:
            value: 字段值
            field_schema: 字段模式
            
        Returns:
            bool: 值是否在范围内
        """
        is_valid, _ = self._check_field_range_with_message(value, field_schema, "")
        return is_valid