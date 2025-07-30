# fedcl/config/utils.py
"""
配置验证工具函数

提供配置处理相关的实用工具函数。
"""

from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
import json
import yaml
import re
from datetime import datetime
import hashlib
from loguru import logger

from .exceptions import ConfigValidationError, ConfigLoadError


def detect_config_format(file_path: Path) -> str:
    """检测配置文件格式
    
    Args:
        file_path: 配置文件路径
        
    Returns:
        str: 文件格式 ('json', 'yaml', 'unknown')
    """
    suffix = file_path.suffix.lower()
    
    if suffix == '.json':
        return 'json'
    elif suffix in ['.yaml', '.yml']:
        return 'yaml'
    else:
        # 尝试通过内容检测
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                
            # 尝试解析为JSON
            try:
                json.loads(content)
                return 'json'
            except json.JSONDecodeError:
                pass
            
            # 尝试解析为YAML
            try:
                yaml.safe_load(content)
                return 'yaml'
            except yaml.YAMLError:
                pass
                
        except Exception:
            pass
    
    return 'unknown'


def load_config_file(file_path: Path) -> Dict[str, Any]:
    """加载配置文件
    
    Args:
        file_path: 配置文件路径
        
    Returns:
        Dict[str, Any]: 配置字典
        
    Raises:
        ConfigLoadError: 配置加载失败
    """
    if not file_path.exists():
        raise ConfigLoadError(str(file_path), FileNotFoundError(f"File not found: {file_path}"))
    
    file_format = detect_config_format(file_path)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_format == 'json':
                return json.load(f)
            elif file_format == 'yaml':
                return yaml.safe_load(f) or {}
            else:
                raise ConfigLoadError(str(file_path), ValueError(f"Unsupported file format: {file_format}"))
    
    except Exception as e:
        if isinstance(e, ConfigLoadError):
            raise
        raise ConfigLoadError(str(file_path), e)


def save_config_file(config: Dict[str, Any], file_path: Path, format: str = None) -> None:
    """保存配置文件
    
    Args:
        config: 配置字典
        file_path: 保存路径
        format: 文件格式，如果为None则根据文件扩展名自动检测
        
    Raises:
        ConfigValidationError: 保存失败
    """
    if format is None:
        format = detect_config_format(file_path)
        if format == 'unknown':
            format = 'yaml'  # 默认使用YAML格式
    
    # 确保目录存在
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            if format == 'json':
                json.dump(config, f, indent=2, ensure_ascii=False)
            elif format == 'yaml':
                yaml.dump(config, f, default_flow_style=False, indent=2, 
                         allow_unicode=True, sort_keys=False)
            else:
                raise ConfigValidationError(f"Unsupported save format: {format}")
    
    except Exception as e:
        raise ConfigValidationError(f"Failed to save config to {file_path}: {e}")


def get_nested_value(data: Dict[str, Any], path: str, default: Any = None) -> Any:
    """获取嵌套字典值
    
    Args:
        data: 数据字典
        path: 键路径，用点分隔（如 "model.learning_rate"）
        default: 默认值
        
    Returns:
        Any: 获取的值或默认值
    """
    keys = path.split('.')
    current = data
    
    try:
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current
    except (TypeError, KeyError):
        return default


def set_nested_value(data: Dict[str, Any], path: str, value: Any) -> None:
    """设置嵌套字典值
    
    Args:
        data: 数据字典
        path: 键路径，用点分隔
        value: 要设置的值
    """
    keys = path.split('.')
    current = data
    
    # 创建中间层级
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        elif not isinstance(current[key], dict):
            raise ValueError(f"Cannot set nested value: '{key}' is not a dict")
        current = current[key]
    
    # 设置最终值
    current[keys[-1]] = value


def delete_nested_value(data: Dict[str, Any], path: str) -> bool:
    """删除嵌套字典值
    
    Args:
        data: 数据字典
        path: 键路径，用点分隔
        
    Returns:
        bool: 是否成功删除
    """
    keys = path.split('.')
    current = data
    
    try:
        # 导航到父级
        for key in keys[:-1]:
            current = current[key]
        
        # 删除最终键
        if keys[-1] in current:
            del current[keys[-1]]
            return True
        return False
    
    except (TypeError, KeyError):
        return False


def flatten_config(config: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """扁平化配置字典
    
    Args:
        config: 配置字典
        parent_key: 父键名
        sep: 分隔符
        
    Returns:
        Dict[str, Any]: 扁平化后的字典
    """
    items = []
    
    for key, value in config.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        
        if isinstance(value, dict):
            items.extend(flatten_config(value, new_key, sep).items())
        else:
            items.append((new_key, value))
    
    return dict(items)


def unflatten_config(flat_config: Dict[str, Any], sep: str = '.') -> Dict[str, Any]:
    """反扁平化配置字典
    
    Args:
        flat_config: 扁平化的配置字典
        sep: 分隔符
        
    Returns:
        Dict[str, Any]: 嵌套配置字典
    """
    result = {}
    
    for flat_key, value in flat_config.items():
        set_nested_value(result, flat_key.replace(sep, '.'), value)
    
    return result


def compare_configs(config1: Dict[str, Any], config2: Dict[str, Any]) -> Dict[str, Any]:
    """比较两个配置字典
    
    Args:
        config1: 配置字典1
        config2: 配置字典2
        
    Returns:
        Dict[str, Any]: 比较结果，包含added, removed, modified, unchanged字段
    """
    flat1 = flatten_config(config1)
    flat2 = flatten_config(config2)
    
    keys1 = set(flat1.keys())
    keys2 = set(flat2.keys())
    
    added = {key: flat2[key] for key in keys2 - keys1}
    removed = {key: flat1[key] for key in keys1 - keys2}
    
    common_keys = keys1 & keys2
    modified = {}
    unchanged = {}
    
    for key in common_keys:
        if flat1[key] != flat2[key]:
            modified[key] = {'old': flat1[key], 'new': flat2[key]}
        else:
            unchanged[key] = flat1[key]
    
    return {
        'added': added,
        'removed': removed,
        'modified': modified,
        'unchanged': unchanged
    }


def generate_config_hash(config: Dict[str, Any]) -> str:
    """生成配置哈希值
    
    Args:
        config: 配置字典
        
    Returns:
        str: MD5哈希值
    """
    # 规范化配置并序列化
    normalized = json.dumps(config, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(normalized.encode('utf-8')).hexdigest()


def validate_config_paths(config: Dict[str, Any], required_paths: List[str]) -> List[str]:
    """验证配置中必需的路径是否存在
    
    Args:
        config: 配置字典
        required_paths: 必需的路径列表
        
    Returns:
        List[str]: 缺失的路径列表
    """
    missing_paths = []
    
    for path in required_paths:
        if get_nested_value(config, path) is None:
            missing_paths.append(path)
    
    return missing_paths


def merge_configs_deep(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """深度合并配置字典
    
    Args:
        base: 基础配置
        override: 覆盖配置
        
    Returns:
        Dict[str, Any]: 合并后的配置
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs_deep(result[key], value)
        else:
            result[key] = value
    
    return result


def extract_config_section(config: Dict[str, Any], section: str) -> Dict[str, Any]:
    """提取配置的指定部分
    
    Args:
        config: 完整配置
        section: 部分名称
        
    Returns:
        Dict[str, Any]: 提取的配置部分
    """
    return config.get(section, {})


def validate_config_types(config: Dict[str, Any], type_mapping: Dict[str, type]) -> List[str]:
    """验证配置字段类型
    
    Args:
        config: 配置字典
        type_mapping: 字段类型映射 {path: expected_type}
        
    Returns:
        List[str]: 类型错误的字段列表
    """
    type_errors = []
    
    for path, expected_type in type_mapping.items():
        value = get_nested_value(config, path)
        if value is not None and not isinstance(value, expected_type):
            type_errors.append(f"{path}: expected {expected_type.__name__}, got {type(value).__name__}")
    
    return type_errors


def create_config_template(schema: Dict[str, Any]) -> Dict[str, Any]:
    """根据验证模式创建配置模板
    
    Args:
        schema: 验证模式
        
    Returns:
        Dict[str, Any]: 配置模板
    """
    template = {}
    
    if 'properties' in schema:
        for field, field_schema in schema['properties'].items():
            if 'default' in field_schema:
                template[field] = field_schema['default']
            elif field_schema.get('type') == 'object':
                template[field] = create_config_template(field_schema)
            elif field_schema.get('type') == 'array':
                template[field] = []
            elif field_schema.get('type') == 'string':
                template[field] = ""
            elif field_schema.get('type') == 'integer':
                template[field] = 0
            elif field_schema.get('type') == 'number':
                template[field] = 0.0
            elif field_schema.get('type') == 'boolean':
                template[field] = False
            else:
                template[field] = None
    
    return template


def sanitize_config_value(value: Any, field_type: str) -> Any:
    """清理配置值
    
    Args:
        value: 原始值
        field_type: 字段类型
        
    Returns:
        Any: 清理后的值
    """
    if value is None:
        return None
    
    try:
        if field_type == 'string':
            return str(value).strip()
        elif field_type == 'integer':
            return int(float(value))
        elif field_type == 'number':
            return float(value)
        elif field_type == 'boolean':
            if isinstance(value, str):
                return value.lower() in ('true', '1', 'yes', 'on')
            return bool(value)
        else:
            return value
    except (ValueError, TypeError):
        return value


def get_config_summary(config: Dict[str, Any]) -> Dict[str, Any]:
    """获取配置摘要信息
    
    Args:
        config: 配置字典
        
    Returns:
        Dict[str, Any]: 摘要信息
    """
    flat_config = flatten_config(config)
    
    type_counts = {}
    for value in flat_config.values():
        value_type = type(value).__name__
        type_counts[value_type] = type_counts.get(value_type, 0) + 1
    
    return {
        'total_fields': len(flat_config),
        'nested_levels': max(key.count('.') for key in flat_config.keys()) + 1 if flat_config else 0,
        'type_distribution': type_counts,
        'config_hash': generate_config_hash(config),
        'size_bytes': len(json.dumps(config).encode('utf-8'))
    }


def format_validation_report(errors: List[Any], warnings: List[Any]) -> str:
    """格式化验证报告
    
    Args:
        errors: 错误列表
        warnings: 警告列表
        
    Returns:
        str: 格式化的报告
    """
    report_lines = []
    
    if errors:
        report_lines.append("🔴 ERRORS:")
        for i, error in enumerate(errors, 1):
            report_lines.append(f"  {i}. {error}")
        report_lines.append("")
    
    if warnings:
        report_lines.append("🟡 WARNINGS:")
        for i, warning in enumerate(warnings, 1):
            report_lines.append(f"  {i}. {warning}")
        report_lines.append("")
    
    if not errors and not warnings:
        report_lines.append("✅ All validations passed!")
    
    return "\n".join(report_lines)


# 装饰器：自动配置验证
def require_valid_config(validator_func):
    """配置验证装饰器
    
    自动验证函数的config参数
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # 假设第一个参数是config
            if args and isinstance(args[0], dict):
                config = args[0]
                result = validator_func(config)
                if not result.is_valid:
                    error_messages = [str(error) for error in result.errors]
                    raise ConfigValidationError("Invalid configuration:\n" + "\n".join(error_messages))
            
            return func(*args, **kwargs)
        return wrapper
    return decorator