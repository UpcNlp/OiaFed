# fedcl/config/exceptions.py
"""
配置验证相关异常类

定义配置加载、验证过程中可能出现的各种异常情况。
包含ConfigManager和SchemaValidator使用的所有异常类型。
"""

import json
import yaml
from typing import List, Optional, Any


class ConfigValidationError(Exception):
    """配置验证异常基类"""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Any = None, details: Optional[Any] = None):
        """初始化验证异常
        
        Args:
            message: 错误信息
            field: 出错的字段路径
            value: 字段值
            details: 额外的错误详情
        """
        super().__init__(message)
        self.field = field
        self.value = value
        self.message = message
        self.details = details
    
    def __str__(self) -> str:
        result = self.message
        if self.field:
            result = f"Field '{self.field}': {result}"
        if self.details:
            result = f"{result}\nDetails: {self.details}"
        return result


class SchemaNotFoundError(ConfigValidationError):
    """验证模式未找到异常"""
    
    def __init__(self, schema_name: str):
        message = f"Validation schema '{schema_name}' not found"
        super().__init__(message)
        self.schema_name = schema_name


class SchemaLoadError(ConfigValidationError):
    """模式文件加载异常"""
    
    def __init__(self, schema_path: str, cause: Exception):
        message = f"Failed to load schema from {schema_path}: {cause}"
        super().__init__(message)
        self.schema_path = schema_path
        self.cause = cause


class FieldValidationError(ConfigValidationError):
    """字段验证异常"""
    
    def __init__(self, field: str, message: str, value: Any = None, error_code: str = ""):
        super().__init__(message, field, value)
        self.error_code = error_code


class RequiredFieldError(FieldValidationError):
    """必需字段缺失异常"""
    
    def __init__(self, field: str):
        message = f"Required field '{field}' is missing"
        super().__init__(field, message, error_code="MISSING_REQUIRED")


class TypeValidationError(FieldValidationError):
    """类型验证异常"""
    
    def __init__(self, field: str, expected_type: str, actual_type: str, value: Any):
        message = f"Expected type {expected_type}, got {actual_type}"
        super().__init__(field, message, value, "INVALID_TYPE")
        self.expected_type = expected_type
        self.actual_type = actual_type


class RangeValidationError(FieldValidationError):
    """范围验证异常"""
    
    def __init__(self, field: str, value: Any, constraint: str):
        message = f"Value {value} violates constraint: {constraint}"
        super().__init__(field, message, value, "OUT_OF_RANGE")
        self.constraint = constraint


class EnumValidationError(FieldValidationError):
    """枚举值验证异常"""
    
    def __init__(self, field: str, value: Any, allowed_values: List[Any]):
        allowed_str = ", ".join(str(v) for v in allowed_values)
        message = f"Value '{value}' not in allowed values: [{allowed_str}]"
        super().__init__(field, message, value, "INVALID_ENUM")
        self.allowed_values = allowed_values


class PatternValidationError(FieldValidationError):
    """正则模式验证异常"""
    
    def __init__(self, field: str, value: str, pattern: str):
        message = f"Value '{value}' does not match pattern: {pattern}"
        super().__init__(field, message, value, "PATTERN_MISMATCH")
        self.pattern = pattern


class CustomValidationError(FieldValidationError):
    """自定义验证器异常"""
    
    def __init__(self, field: str, message: str, validator_name: str = ""):
        super().__init__(field, message, error_code="CUSTOM_VALIDATION")
        self.validator_name = validator_name


class ConfigLoadError(ConfigValidationError):
    """配置文件加载异常"""
    
    def __init__(self, message: str, config_path: Optional[str] = None, cause: Optional[Exception] = None):
        """初始化配置加载异常
        
        Args:
            message: 错误信息
            config_path: 配置文件路径
            cause: 原始异常
        """
        if config_path and not message.startswith("Failed to load config"):
            full_message = f"Failed to load config from {config_path}: {message}"
        else:
            full_message = message
            
        super().__init__(full_message)
        self.config_path = config_path
        self.cause = cause
    
    def __str__(self) -> str:
        result = self.message
        if self.config_path and "from" not in result:
            result = f"{result}\nFile: {self.config_path}"
        if self.cause:
            result = f"{result}\nCause: {self.cause}"
        return result


class ConfigSaveError(ConfigValidationError):
    """配置文件保存异常"""
    
    def __init__(self, config_path: str, cause: Exception):
        message = f"Failed to save config to {config_path}: {cause}"
        super().__init__(message)
        self.config_path = config_path
        self.cause = cause


class ConfigMergeError(ConfigValidationError):
    """配置合并异常"""
    
    def __init__(self, message: str, conflicts: Optional[List[str]] = None, conflict_path: Optional[str] = None):
        """初始化配置合并异常
        
        Args:
            message: 错误信息
            conflicts: 冲突列表
            conflict_path: 冲突路径
        """
        super().__init__(message)
        self.conflicts = conflicts or []
        self.conflict_path = conflict_path
    
    def __str__(self) -> str:
        result = self.message
        if self.conflict_path:
            result = f"{result}\nConflict at path: {self.conflict_path}"
        if self.conflicts:
            conflicts_str = '\n  - '.join(self.conflicts)
            result = f"{result}\nConflicts:\n  - {conflicts_str}"
        return result


class PerformanceError(ConfigValidationError):
    """性能要求不满足异常"""
    
    def __init__(self, operation: str, actual_time: float, max_time: float):
        message = f"{operation} took {actual_time:.2f}ms, exceeding limit of {max_time:.2f}ms"
        super().__init__(message)
        self.operation = operation
        self.actual_time = actual_time
        self.max_time = max_time


# ConfigManager专用异常类
class ConfigSchemaError(ConfigValidationError):
    """
    配置模式异常
    
    当配置模式本身无效或存在错误时抛出此异常。
    """
    pass


class ConfigReferenceError(ConfigValidationError):
    """
    配置引用解析异常
    
    当配置引用无法解析时抛出此异常，如循环引用、引用不存在等。
    """
    
    def __init__(self, message: str, reference_path: Optional[str] = None, reference_chain: Optional[List[str]] = None):
        super().__init__(message)
        self.reference_path = reference_path
        self.reference_chain = reference_chain or []
    
    def __str__(self) -> str:
        result = self.message
        if self.reference_path:
            result = f"{result}\nReference path: {self.reference_path}"
        if self.reference_chain:
            chain = ' -> '.join(self.reference_chain)
            result = f"{result}\nReference chain: {chain}"
        return result


class ConfigEnvironmentError(ConfigValidationError):
    """
    环境变量展开异常
    
    当环境变量无法正确展开或不存在时抛出此异常。
    """
    
    def __init__(self, message: str, env_var: Optional[str] = None):
        super().__init__(message)
        self.env_var = env_var
    
    def __str__(self) -> str:
        if self.env_var:
            return f"{self.message}\nEnvironment variable: {self.env_var}"
        return self.message


class ConfigPermissionError(ConfigValidationError):
    """
    配置文件权限异常
    
    当无权限访问配置文件时抛出此异常。
    """
    pass


class ConfigFormatError(ConfigValidationError):
    """
    配置文件格式异常
    
    当配置文件格式不支持或格式错误时抛出此异常。
    """
    
    def __init__(self, message: str, format_type: Optional[str] = None, line_number: Optional[int] = None):
        super().__init__(message)
        self.format_type = format_type
        self.line_number = line_number
    
    def __str__(self) -> str:
        result = self.message
        if self.format_type:
            result = f"{result}\nFormat: {self.format_type}"
        if self.line_number:
            result = f"{result}\nLine: {self.line_number}"
        return result


class ConfigVersionError(ConfigValidationError):
    """
    配置版本兼容性异常
    
    当配置文件版本不兼容时抛出此异常。
    """
    
    def __init__(self, message: str, config_version: Optional[str] = None, supported_versions: Optional[List[str]] = None):
        super().__init__(message)
        self.config_version = config_version
        self.supported_versions = supported_versions or []
    
    def __str__(self) -> str:
        result = self.message
        if self.config_version:
            result = f"{result}\nConfig version: {self.config_version}"
        if self.supported_versions:
            versions = ', '.join(self.supported_versions)
            result = f"{result}\nSupported versions: {versions}"
        return result


# 向后兼容性
class ValidationError(ConfigValidationError):
    """通用验证异常（向后兼容）"""
    pass


# 异常工厂函数
def create_validation_error(error_type: str, **kwargs) -> ConfigValidationError:
    """创建特定类型的验证异常
    
    Args:
        error_type: 异常类型
        **kwargs: 异常参数
        
    Returns:
        ConfigValidationError: 对应的异常实例
    """
    error_classes = {
        # 原有异常类型
        "schema_not_found": SchemaNotFoundError,
        "schema_load": SchemaLoadError,
        "required_field": RequiredFieldError,
        "type_validation": TypeValidationError,
        "range_validation": RangeValidationError,
        "enum_validation": EnumValidationError,
        "pattern_validation": PatternValidationError,
        "custom_validation": CustomValidationError,
        "config_load": ConfigLoadError,
        "config_save": ConfigSaveError,
        "config_merge": ConfigMergeError,
        "performance": PerformanceError,
        # ConfigManager新增异常类型
        "config_schema": ConfigSchemaError,
        "config_reference": ConfigReferenceError,
        "config_environment": ConfigEnvironmentError,
        "config_permission": ConfigPermissionError,
        "config_format": ConfigFormatError,
        "config_version": ConfigVersionError,
    }
    
    error_class = error_classes.get(error_type, ConfigValidationError)
    return error_class(**kwargs)


# 异常处理装饰器
def handle_validation_errors(func):
    """验证错误处理装饰器
    
    将内部异常转换为适当的ConfigValidationError子类
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ConfigValidationError:
            # 重新抛出已知的验证异常
            raise
        except FileNotFoundError as e:
            raise ConfigLoadError(str(e), cause=e)
        except PermissionError as e:
            raise ConfigPermissionError(f"Permission denied: {e}")
        except (json.JSONDecodeError, yaml.YAMLError) as e:
            raise ConfigFormatError("Invalid file format", cause=e)
        except KeyError as e:
            raise RequiredFieldError(str(e).strip("'\""))
        except TypeError as e:
            raise TypeValidationError("unknown", "unknown", "unknown", str(e))
        except ValueError as e:
            raise RangeValidationError("unknown", "unknown", str(e))
        except Exception as e:
            raise ConfigValidationError(f"Unexpected error: {e}")
    
    return wrapper


# 异常收集器，用于批量验证
class ValidationErrorCollector:
    """
    验证错误收集器
    
    用于收集多个验证错误，然后一次性抛出。
    """
    
    def __init__(self):
        self.errors: List[ConfigValidationError] = []
    
    def add_error(self, error: ConfigValidationError) -> None:
        """添加验证错误"""
        self.errors.append(error)
    
    def add_field_error(self, field: str, message: str, error_type: str = "field_validation", **kwargs) -> None:
        """添加字段验证错误"""
        if error_type in ["required_field", "type_validation", "range_validation", 
                         "enum_validation", "pattern_validation", "custom_validation"]:
            error = create_validation_error(error_type, field=field, message=message, **kwargs)
        else:
            error = FieldValidationError(field, message, **kwargs)
        self.add_error(error)
    
    def has_errors(self) -> bool:
        """是否有错误"""
        return len(self.errors) > 0
    
    def get_error_count(self) -> int:
        """获取错误数量"""
        return len(self.errors)
    
    def get_errors(self) -> List[ConfigValidationError]:
        """获取所有错误"""
        return self.errors.copy()
    
    def get_error_messages(self) -> List[str]:
        """获取所有错误消息"""
        return [str(error) for error in self.errors]
    
    def raise_if_errors(self, message: str = "Multiple validation errors occurred") -> None:
        """如果有错误则抛出异常"""
        if self.has_errors():
            error_messages = self.get_error_messages()
            full_message = f"{message}:\n" + "\n".join(f"  - {msg}" for msg in error_messages)
            raise ConfigValidationError(full_message, details=self.errors)
    
    def clear(self) -> None:
        """清空所有错误"""
        self.errors.clear()