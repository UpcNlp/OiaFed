"""
MOE-FedCL 配置管理器
moe_fedcl/config/manager.py
"""

import os
import json
import yaml
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from dataclasses import dataclass

from ..exceptions import ConfigurationError
from .validator import ConfigValidator


@dataclass
class ConfigSource:
    """配置源信息"""
    source_type: str  # file/dict/env
    source_path: Optional[str] = None
    priority: int = 0  # 优先级，数值越高越优先
    loaded_time: Optional[str] = None
    checksum: Optional[str] = None


class ConfigManager:
    """配置管理器 - 统一管理各种配置源"""
    
    def __init__(self, base_config_dir: str = "configs"):
        """
        初始化配置管理器
        
        Args:
            base_config_dir: 基础配置目录
        """
        self.base_config_dir = Path(base_config_dir)
        self.validator = ConfigValidator()
        
        # 配置存储
        self._config_data: Dict[str, Any] = {}
        self._config_sources: List[ConfigSource] = []
        
        # 配置监听器
        self._change_listeners: List[callable] = []
        
        # 环境变量前缀
        self.env_prefix = "MOE_FEDCL_"
    
    def load_configs(self, config_files: Optional[List[str]] = None) -> Dict[str, Any]:
        """加载配置文件
        
        Args:
            config_files: 配置文件路径列表，如果为None则自动发现
            
        Returns:
            Dict[str, Any]: 合并后的配置字典
        """
        # 清空现有配置
        self._config_data.clear()
        self._config_sources.clear()
        
        # 自动发现配置文件
        if config_files is None:
            config_files = self._discover_config_files()
        
        # 按优先级排序并加载配置文件
        for config_file in config_files:
            try:
                config_data = self._load_single_config(config_file)
                if config_data:
                    # 记录配置源
                    source = ConfigSource(
                        source_type="file",
                        source_path=config_file,
                        priority=self._get_file_priority(config_file),
                        loaded_time=self._get_current_time(),
                        checksum=self._calculate_checksum(config_data)
                    )
                    self._config_sources.append(source)
                    
                    # 合并配置
                    self._merge_config(config_data)
                    
            except Exception as e:
                print(f"Warning: Failed to load config file {config_file}: {e}")
        
        # 加载环境变量配置
        self._load_env_config()
        
        # 验证最终配置
        try:
            self.validator.validate_config(self._config_data)
        except Exception as e:
            raise ConfigurationError(f"Configuration validation failed: {str(e)}")
        
        return self._config_data.copy()
    
    def get_config(self, key: str = None, default: Any = None) -> Any:
        """获取配置值
        
        Args:
            key: 配置键，支持点号分隔的嵌套键 (如 'transport.timeout')
            default: 默认值
            
        Returns:
            Any: 配置值
        """
        if key is None:
            return self._config_data.copy()
        
        return self._get_nested_value(self._config_data, key, default)
    
    def set_config(self, key: str, value: Any) -> bool:
        """设置配置值
        
        Args:
            key: 配置键
            value: 配置值
            
        Returns:
            bool: 设置是否成功
        """
        try:
            self._set_nested_value(self._config_data, key, value)
            
            # 触发变更监听器
            self._notify_config_change(key, value)
            
            return True
        except Exception as e:
            print(f"Failed to set config {key}={value}: {e}")
            return False
    
    def merge_config(self, new_config: Dict[str, Any], priority: int = 100) -> None:
        """合并新配置
        
        Args:
            new_config: 新配置字典
            priority: 优先级
        """
        # 记录配置源
        source = ConfigSource(
            source_type="dict",
            priority=priority,
            loaded_time=self._get_current_time(),
            checksum=self._calculate_checksum(new_config)
        )
        self._config_sources.append(source)
        
        # 合并配置
        self._merge_config(new_config)
        
        # 通知变更
        self._notify_config_change("*", new_config)
    
    def get_config_by_mode(self, mode: str) -> Dict[str, Any]:
        """根据模式获取配置
        
        Args:
            mode: 通信模式 ('memory', 'process', 'network')
            
        Returns:
            Dict[str, Any]: 该模式的完整配置
        """
        base_config = self._config_data.copy()
        base_config["mode"] = mode
        
        # 加载模式特定配置
        mode_config_file = self.base_config_dir / f"{mode}.yaml"
        if mode_config_file.exists():
            try:
                mode_config = self._load_single_config(str(mode_config_file))
                self._deep_merge_dict(base_config, mode_config)
            except Exception as e:
                print(f"Warning: Failed to load mode config {mode_config_file}: {e}")
        
        return base_config
    
    def get_client_config(self, client_id: str) -> Dict[str, Any]:
        """获取特定客户端的配置
        
        Args:
            client_id: 客户端ID
            
        Returns:
            Dict[str, Any]: 客户端配置
        """
        # 基础配置
        client_config = self._config_data.copy()
        
        # 客户端特定配置
        clients_config = self.get_config("clients", {})
        if client_id in clients_config:
            self._deep_merge_dict(client_config, clients_config[client_id])
        
        return client_config
    
    def save_config(self, file_path: str, config_data: Dict[str, Any] = None) -> bool:
        """保存配置到文件
        
        Args:
            file_path: 保存路径
            config_data: 配置数据，如果为None则保存当前配置
            
        Returns:
            bool: 保存是否成功
        """
        try:
            if config_data is None:
                config_data = self._config_data
            
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 根据文件扩展名选择格式
            if file_path.suffix.lower() in ['.yaml', '.yml']:
                with open(file_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
            elif file_path.suffix.lower() == '.json':
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(config_data, f, indent=2, ensure_ascii=False)
            else:
                raise ConfigurationError(f"Unsupported config file format: {file_path.suffix}")
            
            return True
            
        except Exception as e:
            print(f"Failed to save config to {file_path}: {e}")
            return False
    
    def add_config_change_listener(self, listener: callable) -> str:
        """添加配置变更监听器
        
        Args:
            listener: 监听器函数，签名为 listener(key: str, value: Any)
            
        Returns:
            str: 监听器ID
        """
        listener_id = f"listener_{len(self._change_listeners)}"
        self._change_listeners.append((listener_id, listener))
        return listener_id
    
    def remove_config_change_listener(self, listener_id: str) -> bool:
        """移除配置变更监听器
        
        Args:
            listener_id: 监听器ID
            
        Returns:
            bool: 移除是否成功
        """
        for i, (lid, listener) in enumerate(self._change_listeners):
            if lid == listener_id:
                del self._change_listeners[i]
                return True
        return False
    
    def reload_configs(self) -> Dict[str, Any]:
        """重新加载所有配置"""
        config_files = [source.source_path for source in self._config_sources if source.source_path]
        return self.load_configs(config_files)
    
    def get_config_info(self) -> Dict[str, Any]:
        """获取配置信息
        
        Returns:
            Dict[str, Any]: 配置源信息和统计
        """
        return {
            "sources": [
                {
                    "type": source.source_type,
                    "path": source.source_path,
                    "priority": source.priority,
                    "loaded_time": source.loaded_time
                }
                for source in self._config_sources
            ],
            "total_sources": len(self._config_sources),
            "config_keys": list(self._config_data.keys()),
            "listeners_count": len(self._change_listeners)
        }
    
    # ==================== 私有方法 ====================
    
    def _discover_config_files(self) -> List[str]:
        """自动发现配置文件"""
        config_files = []
        
        # 标准配置文件顺序
        standard_files = [
            "global.yaml",
            "global.yml", 
            "global.json",
            "memory.yaml",
            "process.yaml", 
            "network.yaml",
            "federation.yaml"
        ]
        
        for filename in standard_files:
            file_path = self.base_config_dir / filename
            if file_path.exists():
                config_files.append(str(file_path))
        
        # 查找其他配置文件
        if self.base_config_dir.exists():
            for file_path in self.base_config_dir.glob("*.yaml"):
                if str(file_path) not in config_files:
                    config_files.append(str(file_path))
            
            for file_path in self.base_config_dir.glob("*.yml"):
                if str(file_path) not in config_files:
                    config_files.append(str(file_path))
            
            for file_path in self.base_config_dir.glob("*.json"):
                if str(file_path) not in config_files:
                    config_files.append(str(file_path))
        
        return config_files
    
    def _load_single_config(self, config_file: str) -> Dict[str, Any]:
        """加载单个配置文件"""
        file_path = Path(config_file)
        
        if not file_path.exists():
            return {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix.lower() in ['.yaml', '.yml']:
                    return yaml.safe_load(f) or {}
                elif file_path.suffix.lower() == '.json':
                    return json.load(f) or {}
                else:
                    print(f"Warning: Unsupported config file format: {file_path.suffix}")
                    return {}
        except Exception as e:
            raise ConfigurationError(f"Failed to load config file {config_file}: {str(e)}")
    
    def _load_env_config(self) -> None:
        """加载环境变量配置"""
        env_config = {}
        
        for key, value in os.environ.items():
            if key.startswith(self.env_prefix):
                # 移除前缀并转换为小写
                config_key = key[len(self.env_prefix):].lower()
                
                # 处理嵌套键 (如 MOE_FEDCL_TRANSPORT__TIMEOUT -> transport.timeout)
                config_key = config_key.replace('__', '.')
                
                # 尝试解析值的类型
                parsed_value = self._parse_env_value(value)
                
                # 设置嵌套值
                self._set_nested_value(env_config, config_key, parsed_value)
        
        if env_config:
            # 记录环境变量配置源
            source = ConfigSource(
                source_type="env",
                priority=1000,  # 环境变量优先级最高
                loaded_time=self._get_current_time(),
                checksum=self._calculate_checksum(env_config)
            )
            self._config_sources.append(source)
            
            # 合并环境变量配置
            self._merge_config(env_config)
    
    def _parse_env_value(self, value: str) -> Any:
        """解析环境变量值的类型"""
        # 尝试解析为JSON
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
        
        # 布尔值
        if value.lower() in ('true', '1', 'yes', 'on'):
            return True
        elif value.lower() in ('false', '0', 'no', 'off'):
            return False
        
        # 数字
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # 字符串
        return value
    
    def _merge_config(self, new_config: Dict[str, Any]) -> None:
        """合并配置到主配置"""
        self._deep_merge_dict(self._config_data, new_config)
    
    def _deep_merge_dict(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """深度合并字典"""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge_dict(target[key], value)
            else:
                target[key] = value
    
    def _get_nested_value(self, data: Dict[str, Any], key: str, default: Any = None) -> Any:
        """获取嵌套字典的值"""
        try:
            keys = key.split('.')
            current = data
            
            for k in keys:
                current = current[k]
            
            return current
        except (KeyError, TypeError):
            return default
    
    def _set_nested_value(self, data: Dict[str, Any], key: str, value: Any) -> None:
        """设置嵌套字典的值"""
        keys = key.split('.')
        current = data
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            elif not isinstance(current[k], dict):
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    def _get_file_priority(self, config_file: str) -> int:
        """获取配置文件的优先级"""
        file_name = Path(config_file).stem.lower()
        
        # 优先级映射
        priority_map = {
            'global': 10,
            'memory': 20,
            'process': 20,
            'network': 20,
            'federation': 30,
            'local': 40,
            'override': 50
        }
        
        return priority_map.get(file_name, 0)
    
    def _get_current_time(self) -> str:
        """获取当前时间字符串"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def _calculate_checksum(self, data: Any) -> str:
        """计算数据的校验和"""
        import hashlib
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def _notify_config_change(self, key: str, value: Any) -> None:
        """通知配置变更监听器"""
        for listener_id, listener in self._change_listeners:
            try:
                listener(key, value)
            except Exception as e:
                print(f"Config change listener {listener_id} error: {e}")


# ==================== 便捷函数 ====================

# 全局配置管理器实例
_global_config_manager: Optional[ConfigManager] = None


def get_global_config_manager() -> ConfigManager:
    """获取全局配置管理器实例"""
    global _global_config_manager
    
    if _global_config_manager is None:
        _global_config_manager = ConfigManager()
    
    return _global_config_manager


def load_global_configs(config_files: Optional[List[str]] = None) -> Dict[str, Any]:
    """加载全局配置"""
    manager = get_global_config_manager()
    return manager.load_configs(config_files)


def get_global_config(key: str = None, default: Any = None) -> Any:
    """获取全局配置值"""
    manager = get_global_config_manager()
    return manager.get_config(key, default)


def set_global_config(key: str, value: Any) -> bool:
    """设置全局配置值"""
    manager = get_global_config_manager()
    return manager.set_config(key, value)


def create_mode_config(mode: str, **overrides) -> Dict[str, Any]:
    """创建指定模式的配置
    
    Args:
        mode: 通信模式
        **overrides: 配置覆盖
        
    Returns:
        Dict[str, Any]: 模式配置
    """
    manager = get_global_config_manager()
    config = manager.get_config_by_mode(mode)
    
    # 应用覆盖
    for key, value in overrides.items():
        manager._set_nested_value(config, key, value)
    
    return config


def create_client_config(client_id: str, **overrides) -> Dict[str, Any]:
    """创建客户端配置
    
    Args:
        client_id: 客户端ID
        **overrides: 配置覆盖
        
    Returns:
        Dict[str, Any]: 客户端配置
    """
    manager = get_global_config_manager()
    config = manager.get_client_config(client_id)
    
    # 应用覆盖
    for key, value in overrides.items():
        manager._set_nested_value(config, key, value)
    
    return config