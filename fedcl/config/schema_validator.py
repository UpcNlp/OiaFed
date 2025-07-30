# fedcl/config/schema_validator.py
"""
配置模式验证器

提供配置验证功能，支持：
- 多种配置类型验证（实验、模型、数据、通信）
- 嵌套配置验证
- 自定义验证器
- 字段间依赖验证
- 高性能验证（1000字段<100ms）
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
    """
    
    @handle_validation_errors
    def __init__(self, schema_path: Optional[Path] = None):
        """初始化验证器"""
        self.schemas: Dict[str, Dict] = {}
        self.custom_validators: Dict[str, Callable] = {}
        self._validation_errors: List[ValidationError] = []
        
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
        
        logger.info(f"SchemaValidator initialized with {len(self.schemas)} schemas")
    
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
                if not self._check_field_range(value, field_schema):
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
        """加载默认验证模式"""
        # 实验配置模式
        self.schemas["experiment"] = {
            "type": "object",
            "required": ["name", "method", "dataset", "federation"],
            "properties": {
                "name": {"type": "string", "minLength": 1},
                "method": {"type": "string", "enum": ["L2P", "EWC", "Replay", "DDDR"]},
                "dataset": {"type": "string", "enum": ["CIFAR10", "CIFAR100", "ImageNet-R", "MNIST"]},
                "federation": {"type": "object"},
                "seed": {"type": "integer", "minimum": 0, "maximum": 2147483647},
                "num_rounds": {"type": "integer", "minimum": 1, "maximum": 10000},
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
                "optimizer": {"type": "string", "enum": ["SGD", "Adam", "AdamW", "RMSprop"]},
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
            "required": ["protocol", "num_clients"],
            "properties": {
                "protocol": {"type": "string", "enum": ["tcp", "udp", "grpc", "http"]},
                "num_clients": {"type": "integer", "minimum": 1, "maximum": 10000},
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
            
            logger.info(f"Loaded {len(external_schemas)} schemas from {schema_path}")
        
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
        
        # 检查字段范围
        range_errors = self.check_field_ranges(config, schema)
        for field in range_errors:
            field_schema = schema["properties"][field]
            actual_value = self._get_nested_value(config, field)
            result.add_error(
                field,
                f"Field '{field}' value {actual_value} is out of range",
                "OUT_OF_RANGE",
                actual_value
            )
        
        # 运行自定义验证器
        result = self._run_custom_validators(config, result)
        
        return result
    
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
        num_rounds = config.get("num_rounds", 1)
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
        num_clients = config.get("num_clients", 1)
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
        """检查字段范围
        
        Args:
            value: 字段值
            field_schema: 字段模式
            
        Returns:
            bool: 值是否在范围内
        """
        # 检查数值范围
        if isinstance(value, (int, float)):
            minimum = field_schema.get("minimum")
            maximum = field_schema.get("maximum")
            
            if minimum is not None and value < minimum:
                return False
            if maximum is not None and value > maximum:
                return False
        
        # 检查字符串长度
        if isinstance(value, str):
            min_length = field_schema.get("minLength")
            max_length = field_schema.get("maxLength")
            
            if min_length is not None and len(value) < min_length:
                return False
            if max_length is not None and len(value) > max_length:
                return False
        
        # 检查枚举值
        enum_values = field_schema.get("enum")
        if enum_values is not None and value not in enum_values:
            return False
        
        # 检查正则表达式模式
        pattern = field_schema.get("pattern")
        if pattern is not None and isinstance(value, str):
            if not re.match(pattern, value):
                return False
        
        return True