# fedcl/config/config_manager.py
"""
Configuration Manager for FedCL Framework

This module provides comprehensive configuration management capabilities including:
- Multi-format config loading (YAML, JSON, TOML)
- Configuration validation and merging
- Environment variable expansion
- Configuration history and rollback
- Reference resolution and inheritance
- Hot reload support
"""

import os
import re
import copy
import json
import yaml
import toml
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Callable, Union, Optional
from loguru import logger
from omegaconf import DictConfig as OmegaDictConfig, OmegaConf


from .schema_validator import SchemaValidator, ValidationResult
from .exceptions import (
    ConfigValidationError, ConfigLoadError, ConfigMergeError, 
    ConfigReferenceError, ConfigFormatError
)
from ..utils.file_utils import FileUtils
class DictConfig(OmegaDictConfig):
    """
    继承自 OmegaConf DictConfig 的配置类，
    添加了一些便利方法，完全兼容 OmegaConf.select()
    """
    
    def __init__(self, data: Union[Dict[str, Any], 'DictConfig', OmegaDictConfig] = None):
        """
        初始化配置对象
        
        Args:
            data: 字典数据、DictConfig对象或OmegaDictConfig对象
        """
        if data is None:
            data = {}
        elif isinstance(data, (DictConfig, OmegaDictConfig)):
            # 如果是配置对象，转换为字典
            data = OmegaConf.to_container(data, resolve=True)
        elif not isinstance(data, dict):
            raise TypeError(f"Expected dict or DictConfig, got {type(data)}")
        
        # 调用父类构造函数
        super().__init__(data)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值，支持嵌套键路径
        
        Args:
            key: 键名，支持点分隔的嵌套路径如 'model.name'
            default: 默认值
            
        Returns:
            配置值或默认值
        """
        result = OmegaConf.select(self, key)
        return result if result is not None else default
    
    def select(self, key: str, default: Any = None) -> Any:
        """
        选择配置值，等同于 OmegaConf.select()
        
        Args:
            key: 键名，支持点分隔的嵌套路径
            default: 默认值
            
        Returns:
            配置值或默认值
        """
        result = OmegaConf.select(self, key)
        return result if result is not None else default
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为普通字典
        
        Returns:
            Dict: 配置的字典表示
        """
        return OmegaConf.to_container(self, resolve=True)
    
    def copy(self) -> 'DictConfig':
        """
        创建深拷贝
        
        Returns:
            DictConfig: 配置对象的深拷贝
        """
        return DictConfig(self.to_dict())
    
    def update(self, other: Union[Dict, 'DictConfig', OmegaDictConfig]) -> None:
        """
        更新配置
        
        Args:
            other: 要合并的配置数据
        """
        if isinstance(other, (DictConfig, OmegaDictConfig)):
            other = OmegaConf.to_container(other, resolve=True)
        
        # 使用 OmegaConf.merge 进行深度合并
        merged = OmegaConf.merge(self, other)
        
        # 更新当前对象
        self.clear()
        self.update_config(merged)
    
    def update_config(self, other: Union[Dict, OmegaDictConfig]) -> None:
        """内部方法：更新配置内容"""
        if isinstance(other, dict):
            other = OmegaConf.create(other)
        
        for key, value in other.items():
            self[key] = value
    
    def has_key(self, key: str) -> bool:
        """
        检查是否存在指定键（支持嵌套路径）
        
        Args:
            key: 键名，支持点分隔的嵌套路径
            
        Returns:
            bool: 是否存在该键
        """
        return OmegaConf.select(self, key) is not None
    
    def set_value(self, key: str, value: Any) -> None:
        """
        设置嵌套键的值
        
        Args:
            key: 键名，支持点分隔的嵌套路径如 'model.name'
            value: 要设置的值
        """
        OmegaConf.set(self, key, value)
    
    def __repr__(self) -> str:
        """字符串表示"""
        return f"DictConfig({OmegaConf.to_yaml(self)})"
    
    def pretty_print(self) -> None:
        """美化打印配置"""
        print(OmegaConf.to_yaml(self))

class ConfigManager:
    """
    Comprehensive configuration manager for FedCL framework.
    
    Provides functionality for loading, validating, merging, and managing
    configuration files with support for multiple formats, environment variables,
    and configuration inheritance.
    """
    
    # Supported configuration file formats
    SUPPORTED_FORMATS = {'.yaml', '.yml', '.json', '.toml'}
    
    # Environment variable pattern for substitution
    ENV_VAR_PATTERN = re.compile(r'\$\{([^}]+)\}')
    
    # Reference pattern for config inheritance
    REF_PATTERN = re.compile(r'\$\{ref:([^}]+)\}')
    
    def __init__(self, schema_validator: SchemaValidator):
        """
        Initialize ConfigManager with a schema validator.
        
        Args:
            schema_validator: Validator for configuration schemas
        """
        self.schema_validator = schema_validator
        self._config_cache: Dict[str, DictConfig] = {}
        self._config_history: List[Tuple[datetime, DictConfig]] = []
        self._config_hooks: List[Callable[[DictConfig], DictConfig]] = []
        self._reference_stack: List[str] = []  # For circular reference detection
        
        logger.info("ConfigManager initialized")
    
    def load_config(self, config_path: Union[str, Path]) -> DictConfig:
        """
        Load configuration from file with validation and processing.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            DictConfig: Validated and processed configuration
            
        Raises:
            ConfigLoadError: If file cannot be loaded or parsed
            ConfigValidationError: If configuration is invalid
        """
        config_path = Path(config_path)
        cache_key = str(config_path.absolute())
        
        # Check cache first
        if cache_key in self._config_cache:
            logger.debug(f"Loading config from cache: {config_path}")
            return self._config_cache[cache_key].copy()
        
        try:
            logger.info(f"Loading configuration from: {config_path}")
            
            # Check if file exists
            if not config_path.exists():
                raise ConfigLoadError(
                    "Configuration file not found", 
                    config_path=str(config_path)
                )
            
            # Load raw configuration based on file extension
            raw_config = self._load_raw_config(config_path)
            
            # Convert to DictConfig
            config = DictConfig(raw_config)
            
            # Process configuration inheritance if base config specified
            if 'base_config' in config:
                config = self._resolve_base_config(config, config_path.parent)
            
            # Resolve references
            config = self.resolve_references(config)
            
            # Expand environment variables
            config = self.expand_environment_variables(config)
            
            # Apply configuration hooks
            for hook in self._config_hooks:
                config = hook(config)
            
            # Validate configuration
            validation_result = self.validate_config(config)
            if not validation_result.is_valid:
                raise ConfigValidationError(
                    "Configuration validation failed",
                    details=validation_result.errors
                )
            
            # Cache the processed configuration
            self._config_cache[cache_key] = config.copy()
            
            # Add to history
            self._add_to_history(config)
            
            logger.success(f"Successfully loaded configuration from: {config_path}")
            return config.copy()
            
        except (yaml.YAMLError, json.JSONDecodeError, toml.TomlDecodeError) as e:
            raise ConfigLoadError("Failed to parse configuration file", 
                                config_path=str(config_path), cause=e)
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise ConfigLoadError("Failed to load configuration", 
                                config_path=str(config_path), cause=e)
    
    def validate_config(self, config: DictConfig) -> ValidationResult:
        """
        Validate configuration against schema.
        
        Args:
            config: Configuration to validate
            
        Returns:
            ValidationResult: Validation result
        """
        try:
            logger.debug("Validating configuration")
            result = self.schema_validator.validate_experiment_config(config.to_dict())
            
            if result.is_valid:
                logger.debug("Configuration validation passed")
            else:
                logger.warning(f"Configuration validation failed: {result.errors}")
            
            return result
            
        except Exception as e:
            logger.error(f"Configuration validation error: {e}")
            return ValidationResult(is_valid=False, errors=[str(e)])
    
    def merge_configs(self, base: DictConfig, override: DictConfig) -> DictConfig:
        """
        Merge two configurations, with override taking precedence.
        
        Args:
            base: Base configuration
            override: Override configuration
            
        Returns:
            DictConfig: Merged configuration
            
        Raises:
            ConfigMergeError: If merge operation fails
        """
        try:
            logger.debug("Merging configurations")
            
            merged_data = self._deep_merge(base.to_dict(), override.to_dict())
            merged_config = DictConfig(merged_data)
            
            # Add to history
            self._add_to_history(merged_config)
            
            logger.debug("Configuration merge completed")
            return merged_config
            
        except Exception as e:
            raise ConfigMergeError("Failed to merge configurations", 
                                 conflicts=[str(e)])
    
    def resolve_references(self, config: DictConfig) -> DictConfig:
        """
        Resolve configuration references (${ref:path.to.value}).
        
        Args:
            config: Configuration with possible references
            
        Returns:
            DictConfig: Configuration with resolved references
        """
        logger.debug("Resolving configuration references")
        
        resolved_data = self._resolve_references_recursive(
            config.to_dict(), config.to_dict()
        )
        
        # Clear reference stack after resolution
        self._reference_stack.clear()
        
        return DictConfig(resolved_data)
    
    def save_config(self, config: DictConfig, path: Path) -> None:
        """
        Save configuration to file.
        
        Args:
            config: Configuration to save
            path: Target file path
            
        Raises:
            ConfigLoadError: If file cannot be saved
        """
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Determine format from extension
            suffix = path.suffix.lower()
            config_data = config.to_dict()
            
            if suffix in {'.yaml', '.yml'}:
                with open(path, 'w', encoding='utf-8') as f:
                    yaml.dump(config_data, f, default_flow_style=False, indent=2)
            elif suffix == '.json':
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(config_data, f, indent=2, ensure_ascii=False)
            elif suffix == '.toml':
                with open(path, 'w', encoding='utf-8') as f:
                    toml.dump(config_data, f)
            else:
                raise ConfigLoadError(
                    f"Unsupported config format: {suffix}", 
                    config_path=str(path)
                )
            
            logger.info(f"Configuration saved to: {path}")
            
        except Exception as e:
            raise ConfigLoadError("Failed to save configuration", 
                                config_path=str(path), cause=e)
    
    def get_nested_value(self, config: DictConfig, path: str, default: Any = None) -> Any:
        """
        Get nested value using dot notation path.
        
        Args:
            config: Configuration object
            path: Dot-separated path (e.g., 'model.optimizer.lr')
            default: Default value if path not found
            
        Returns:
            Any: Value at the specified path or default
        """
        try:
            current = config.to_dict()
            
            for key in path.split('.'):
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    return default
            
            return current
            
        except Exception:
            return default
    
    def set_nested_value(self, config: DictConfig, path: str, value: Any) -> None:
        """
        Set nested value using dot notation path.
        
        Args:
            config: Configuration object to modify
            path: Dot-separated path (e.g., 'model.optimizer.lr')
            value: Value to set
        """
        keys = path.split('.')
        current = config._data
        
        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            elif not isinstance(current[key], dict):
                current[key] = {}
            current = current[key]
        
        # Set the final value
        current[keys[-1]] = value
    
    def expand_environment_variables(self, config: DictConfig) -> DictConfig:
        """
        Expand environment variables in configuration values.
        
        Args:
            config: Configuration with possible environment variables
            
        Returns:
            DictConfig: Configuration with expanded environment variables
        """
        logger.debug("Expanding environment variables")
        
        expanded_data = self._expand_env_vars_recursive(config.to_dict())
        return DictConfig(expanded_data)
    
    def register_config_hook(self, hook: Callable[[DictConfig], DictConfig]) -> None:
        """
        Register a configuration processing hook.
        
        Args:
            hook: Function that takes and returns a DictConfig
        """
        self._config_hooks.append(hook)
        logger.debug(f"Registered configuration hook: {hook.__name__}")
    
    def get_config_history(self) -> List[Tuple[datetime, DictConfig]]:
        """
        Get configuration history.
        
        Returns:
            List[Tuple[datetime, DictConfig]]: List of (timestamp, config) tuples
        """
        return [(timestamp, config.copy()) for timestamp, config in self._config_history]
    
    def rollback_config(self, steps: int = 1) -> DictConfig:
        """
        Rollback configuration to a previous version.
        
        Args:
            steps: Number of steps to rollback
            
        Returns:
            DictConfig: Previous configuration
            
        Raises:
            ValueError: If rollback steps exceed history
        """
        if steps <= 0:
            raise ValueError("Rollback steps must be positive")
        
        if len(self._config_history) < steps:
            raise ValueError(
                f"Cannot rollback {steps} steps, only {len(self._config_history)} "
                "configurations in history"
            )
        
        target_config = self._config_history[-(steps + 1)][1]
        logger.info(f"Rolling back configuration {steps} steps")
        
        return target_config.copy()
    
    def _load_raw_config(self, config_path: Path) -> Dict[str, Any]:
        """Load raw configuration data from file."""
        suffix = config_path.suffix.lower()
        
        if suffix not in self.SUPPORTED_FORMATS:
            raise ConfigFormatError(
                "Unsupported config format", 
                format_type=suffix
            )
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if suffix in {'.yaml', '.yml'}:
                    return yaml.safe_load(f) or {}
                elif suffix == '.json':
                    return json.load(f)
                elif suffix == '.toml':
                    return toml.load(f)
        except Exception as e:
            raise ConfigLoadError(
                f"Failed to load {suffix} file",
                config_path=str(config_path), 
                cause=e
            )
        
        return {}
    
    def _resolve_base_config(self, config: DictConfig, base_dir: Path) -> DictConfig:
        """Resolve base configuration inheritance."""
        base_config_path = config.get('base_config')
        if not base_config_path:
            return config
        
        # Remove base_config key from current config
        current_data = config.to_dict()
        del current_data['base_config']
        current_config = DictConfig(current_data)
        
        # Load base configuration
        if not Path(base_config_path).is_absolute():
            base_config_path = base_dir / base_config_path
        
        base_config = self.load_config(base_config_path)
        
        # Merge base with current (current overrides base)
        return self.merge_configs(base_config, current_config)
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = copy.deepcopy(base)
        
        for key, value in override.items():
            if (key in result and 
                isinstance(result[key], dict) and 
                isinstance(value, dict)):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        
        return result
    
    def _resolve_references_recursive(self, data: Any, root_config: Dict[str, Any]) -> Any:
        """Recursively resolve configuration references."""
        if isinstance(data, dict):
            resolved = {}
            for key, value in data.items():
                resolved[key] = self._resolve_references_recursive(value, root_config)
            return resolved
        elif isinstance(data, list):
            return [self._resolve_references_recursive(item, root_config) for item in data]
        elif isinstance(data, str):
            return self._resolve_string_references(data, root_config)
        else:
            return data
    
    def _resolve_string_references(self, text: str, root_config: Dict[str, Any]) -> Any:
        """Resolve references in a string value."""
        matches = list(self.REF_PATTERN.finditer(text))
        
        if not matches:
            return text
        
        # If the entire string is a single reference, return the referenced value
        if len(matches) == 1 and matches[0].group(0) == text:
            ref_path = matches[0].group(1)
            return self._get_reference_value(ref_path, root_config)
        
        # Otherwise, substitute references in the string
        result = text
        for match in reversed(matches):  # Reverse to maintain positions
            ref_path = match.group(1)
            ref_value = self._get_reference_value(ref_path, root_config)
            result = result[:match.start()] + str(ref_value) + result[match.end():]
        
        return result
    
    def _get_reference_value(self, ref_path: str, root_config: Dict[str, Any]) -> Any:
        """Get value from reference path, checking for circular references."""
        if ref_path in self._reference_stack:
            raise ConfigReferenceError(
                "Circular reference detected", 
                reference_path=ref_path,
                reference_chain=self._reference_stack + [ref_path]
            )
        
        self._reference_stack.append(ref_path)
        
        try:
            # Navigate to the referenced value
            current = root_config
            for key in ref_path.split('.'):
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    raise ConfigReferenceError(
                        "Reference not found", 
                        reference_path=ref_path
                    )
            
            # Recursively resolve the referenced value
            resolved_value = self._resolve_references_recursive(current, root_config)
            
            return resolved_value
            
        finally:
            self._reference_stack.pop()
    
    def _expand_env_vars_recursive(self, data: Any) -> Any:
        """Recursively expand environment variables."""
        if isinstance(data, dict):
            return {key: self._expand_env_vars_recursive(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._expand_env_vars_recursive(item) for item in data]
        elif isinstance(data, str):
            return self._expand_env_vars_in_string(data)
        else:
            return data
    
    def _expand_env_vars_in_string(self, text: str) -> str:
        """Expand environment variables in a string."""
        def replace_env_var(match):
            env_var = match.group(1)
            # Support default values: ${VAR:default}
            if ':' in env_var:
                var_name, default_value = env_var.split(':', 1)
                return os.environ.get(var_name.strip(), default_value.strip())
            else:
                return os.environ.get(env_var, match.group(0))
        
        return self.ENV_VAR_PATTERN.sub(replace_env_var, text)
    
    def _add_to_history(self, config: DictConfig) -> None:
        """Add configuration to history."""
        timestamp = datetime.now()
        self._config_history.append((timestamp, config.copy()))
        
        # Keep only last 100 configurations in history
        if len(self._config_history) > 100:
            self._config_history = self._config_history[-100:]