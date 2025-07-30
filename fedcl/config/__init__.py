# fedcl/config/__init__.py
"""
FedCL配置管理模块

提供配置加载、验证、管理等功能，包括：
- SchemaValidator: 配置模式验证器
- ConfigManager: 配置管理器 
- ValidationResult: 验证结果数据结构
- ValidationError: 验证错误信息
- ValidationWarning: 验证警告信息

使用示例:
    from fedcl.config import SchemaValidator, ValidationResult
    
    validator = SchemaValidator()
    result = validator.validate_experiment_config(config)
    
    if result.is_valid:
        print("配置验证通过")
    else:
        for error in result.errors:
            print(f"错误: {error}")
"""

from .schema_validator import (
    SchemaValidator,
    ValidationResult,
    ValidationError,
    ValidationWarning
)

# 版本信息
__version__ = "1.0.0"
__author__ = "FedCL Team"

# 导出的公共接口
__all__ = [
    "SchemaValidator",
    "ValidationResult", 
    "ValidationError",
    "ValidationWarning"
]

# 默认配置路径
DEFAULT_SCHEMA_PATH = None
DEFAULT_CONFIG_DIR = "configs"

# 支持的配置文件格式
SUPPORTED_CONFIG_FORMATS = [".yaml", ".yml", ".json"]

# 内置验证模式类型
BUILTIN_SCHEMA_TYPES = [
    "experiment",
    "model", 
    "data",
    "communication"
]