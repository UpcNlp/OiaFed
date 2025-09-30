"""
MOE-FedCL 配置验证器
moe_fedcl/config/validator.py
"""

from typing import Any, Dict, List, Optional, Union, Callable, Set
from dataclasses import dataclass
import re
import ipaddress

from ..types import CommunicationMode
from ..exceptions import ConfigurationError, ValidationError


@dataclass
class ValidationRule:
    """验证规则"""
    field_path: str                           # 字段路径，支持点号分隔
    rule_type: str                           # 规则类型
    required: bool = True                    # 是否必需
    default_value: Any = None                # 默认值
    validator: Optional[Callable] = None     # 自定义验证函数
    error_message: Optional[str] = None      # 错误信息
    depends_on: Optional[List[str]] = None   # 依赖的其他字段


class ConfigValidator:
    """配置验证器"""
    
    def __init__(self):
        self.validation_rules: List[ValidationRule] = []
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """初始化默认验证规则"""
        # 基础配置规则
        self.validation_rules.extend([
            # 通信模式
            ValidationRule(
                field_path="mode",
                rule_type="enum",
                required=True,
                validator=lambda x: x in ["memory", "process", "network"],
                error_message="mode must be one of: memory, process, network"
            ),
            
            # 传输配置
            ValidationRule(
                field_path="transport.type",
                rule_type="string",
                required=True,
                validator=lambda x: x in ["memory", "process", "network"]
            ),
            
            ValidationRule(
                field_path="transport.timeout",
                rule_type="number",
                required=False,
                default_value=30.0,
                validator=lambda x: isinstance(x, (int, float)) and x > 0,
                error_message="transport.timeout must be a positive number"
            ),
            
            ValidationRule(
                field_path="transport.retry_attempts",
                rule_type="integer",
                required=False,
                default_value=3,
                validator=lambda x: isinstance(x, int) and 0 <= x <= 10,
                error_message="transport.retry_attempts must be an integer between 0 and 10"
            ),
            
            # 通信配置
            ValidationRule(
                field_path="communication.heartbeat_interval",
                rule_type="number",
                required=False,
                default_value=30.0,
                validator=lambda x: isinstance(x, (int, float)) and x > 0
            ),
            
            ValidationRule(
                field_path="communication.heartbeat_timeout",
                rule_type="number", 
                required=False,
                default_value=90.0,
                validator=lambda x: isinstance(x, (int, float)) and x > 0
            ),
            
            ValidationRule(
                field_path="communication.max_clients",
                rule_type="integer",
                required=False,
                default_value=100,
                validator=lambda x: isinstance(x, int) and x > 0
            ),
            
            # 联邦配置
            ValidationRule(
                field_path="federation.max_rounds",
                rule_type="integer",
                required=False,
                default_value=100,
                validator=lambda x: isinstance(x, int) and x > 0
            ),
            
            ValidationRule(
                field_path="federation.min_clients",
                rule_type="integer",
                required=False,
                default_value=2,
                validator=lambda x: isinstance(x, int) and x >= 1
            ),
        ])
        
        # 网络模式特定规则
        self._add_network_validation_rules()
        
        # 进程模式特定规则
        self._add_process_validation_rules()
    
    def _add_network_validation_rules(self):
        """添加网络模式验证规则"""
        network_rules = [
            ValidationRule(
                field_path="transport.specific_config.host",
                rule_type="string",
                required=False,
                default_value="0.0.0.0",
                validator=self._validate_ip_address,
                error_message="Invalid IP address format",
                depends_on=["mode"]
            ),
            
            ValidationRule(
                field_path="transport.specific_config.port",
                rule_type="integer",
                required=False,
                default_value=8000,
                validator=lambda x: isinstance(x, int) and 1024 <= x <= 65535,
                error_message="Port must be between 1024 and 65535",
                depends_on=["mode"]
            ),
            
            ValidationRule(
                field_path="transport.specific_config.websocket_port",
                rule_type="integer",
                required=False,
                default_value=8001,
                validator=lambda x: isinstance(x, int) and 1024 <= x <= 65535,
                error_message="WebSocket port must be between 1024 and 65535",
                depends_on=["mode"]
            ),
        ]
        
        self.validation_rules.extend(network_rules)
    
    def _add_process_validation_rules(self):
        """添加进程模式验证规则"""
        process_rules = [
            ValidationRule(
                field_path="transport.specific_config.max_queue_size",
                rule_type="integer",
                required=False,
                default_value=10000,
                validator=lambda x: isinstance(x, int) and x > 0,
                depends_on=["mode"]
            ),
            
            ValidationRule(
                field_path="transport.specific_config.serialization",
                rule_type="string",
                required=False,
                default_value="pickle",
                validator=lambda x: x in ["pickle", "json"],
                error_message="serialization must be 'pickle' or 'json'",
                depends_on=["mode"]
            ),
        ]
        
        self.validation_rules.extend(process_rules)
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """验证配置并返回处理后的配置
        
        Args:
            config: 原始配置字典
            
        Returns:
            Dict[str, Any]: 验证和处理后的配置字典
            
        Raises:
            ValidationError: 验证失败
        """
        validated_config = config.copy()
        validation_errors = []
        
        # 获取模式信息（用于条件验证）
        mode = config.get("mode", "memory")
        
        # 遍历所有验证规则
        for rule in self.validation_rules:
            try:
                # 检查依赖条件
                if rule.depends_on:
                    if not self._check_dependencies(config, rule.depends_on, mode):
                        continue
                
                # 获取字段值
                field_value = self._get_nested_value(validated_config, rule.field_path)
                
                # 检查必需字段
                if rule.required and field_value is None:
                    if rule.default_value is not None:
                        # 使用默认值
                        self._set_nested_value(validated_config, rule.field_path, rule.default_value)
                        field_value = rule.default_value
                    else:
                        error_msg = rule.error_message or f"Required field missing: {rule.field_path}"
                        validation_errors.append(error_msg)
                        continue
                
                # 如果字段不存在但有默认值，设置默认值
                if field_value is None and rule.default_value is not None:
                    self._set_nested_value(validated_config, rule.field_path, rule.default_value)
                    field_value = rule.default_value
                
                # 执行验证
                if field_value is not None and rule.validator:
                    if not rule.validator(field_value):
                        error_msg = rule.error_message or f"Validation failed for {rule.field_path}: {field_value}"
                        validation_errors.append(error_msg)
                
            except Exception as e:
                validation_errors.append(f"Validation error for {rule.field_path}: {str(e)}")
        
        # 执行跨字段验证
        cross_field_errors = self._validate_cross_field_rules(validated_config)
        validation_errors.extend(cross_field_errors)
        
        # 如果有验证错误，抛出异常
        if validation_errors:
            raise ValidationError(f"Configuration validation failed:\n" + "\n".join(f"- {error}" for error in validation_errors))
        
        # 执行最后的配置清理和优化
        self._finalize_config(validated_config)
        
        return validated_config
    
    def add_validation_rule(self, rule: ValidationRule):
        """添加自定义验证规则"""
        self.validation_rules.append(rule)
    
    def remove_validation_rule(self, field_path: str) -> bool:
        """移除验证规则"""
        for i, rule in enumerate(self.validation_rules):
            if rule.field_path == field_path:
                del self.validation_rules[i]
                return True
        return False
    
    def validate_field(self, field_path: str, value: Any, config_context: Dict[str, Any] = None) -> bool:
        """验证单个字段
        
        Args:
            field_path: 字段路径
            value: 字段值
            config_context: 配置上下文
            
        Returns:
            bool: 验证是否通过
        """
        for rule in self.validation_rules:
            if rule.field_path == field_path:
                try:
                    # 检查依赖
                    if rule.depends_on and config_context:
                        mode = config_context.get("mode", "memory")
                        if not self._check_dependencies(config_context, rule.depends_on, mode):
                            continue
                    
                    # 执行验证
                    if rule.validator:
                        return rule.validator(value)
                    
                except Exception:
                    return False
        
        return True  # 没有找到规则则认为验证通过
    
    # ==================== 私有方法 ====================
    
    def _check_dependencies(self, config: Dict[str, Any], dependencies: List[str], mode: str) -> bool:
        """检查依赖条件"""
        for dep in dependencies:
            if dep == "mode":
                # 模式相关的依赖检查
                current_mode = config.get("mode", "memory")
                if current_mode != mode:
                    continue
            else:
                # 其他字段依赖
                dep_value = self._get_nested_value(config, dep)
                if dep_value is None:
                    return False
        
        return True
    
    def _validate_cross_field_rules(self, config: Dict[str, Any]) -> List[str]:
        """执行跨字段验证"""
        errors = []
        
        # 验证心跳超时必须大于心跳间隔
        heartbeat_interval = self._get_nested_value(config, "communication.heartbeat_interval", 30.0)
        heartbeat_timeout = self._get_nested_value(config, "communication.heartbeat_timeout", 90.0)
        
        if heartbeat_timeout <= heartbeat_interval:
            errors.append("communication.heartbeat_timeout must be greater than heartbeat_interval")
        
        # 验证网络模式的端口不能冲突
        mode = config.get("mode")
        if mode == "network":
            port = self._get_nested_value(config, "transport.specific_config.port", 8000)
            ws_port = self._get_nested_value(config, "transport.specific_config.websocket_port", 8001)
            
            if port == ws_port:
                errors.append("HTTP port and WebSocket port cannot be the same")
        
        # 验证最小客户端数不能超过最大客户端数
        min_clients = self._get_nested_value(config, "federation.min_clients", 2)
        max_clients = self._get_nested_value(config, "communication.max_clients", 100)
        
        if min_clients > max_clients:
            errors.append("federation.min_clients cannot exceed communication.max_clients")
        
        return errors
    
    def _finalize_config(self, config: Dict[str, Any]):
        """最终配置处理"""
        # 确保传输类型与模式一致
        mode = config.get("mode", "memory")
        transport_config = config.setdefault("transport", {})
        transport_config["type"] = mode
        
        # 确保特定配置存在
        if "specific_config" not in transport_config:
            transport_config["specific_config"] = {}
        
        # 根据模式设置默认的特定配置
        specific_config = transport_config["specific_config"]
        
        if mode == "memory":
            specific_config.setdefault("shared_memory_size", "1GB")
            specific_config.setdefault("event_queue_size", 1000)
            
        elif mode == "process":
            specific_config.setdefault("queue_backend", "multiprocessing")
            specific_config.setdefault("serialization", "pickle")
            
        elif mode == "network":
            specific_config.setdefault("protocol", "http")
            specific_config.setdefault("ssl_enabled", False)
    
    def _get_nested_value(self, data: Dict[str, Any], path: str, default: Any = None) -> Any:
        """获取嵌套字典值"""
        try:
            keys = path.split('.')
            current = data
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default
    
    def _set_nested_value(self, data: Dict[str, Any], path: str, value: Any):
        """设置嵌套字典值"""
        keys = path.split('.')
        current = data
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            elif not isinstance(current[key], dict):
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    # ==================== 验证函数 ====================
    
    def _validate_ip_address(self, ip: str) -> bool:
        """验证IP地址格式"""
        if not isinstance(ip, str):
            return False
        
        # 允许特殊值
        if ip in ["0.0.0.0", "localhost", "127.0.0.1"]:
            return True
        
        try:
            ipaddress.ip_address(ip)
            return True
        except ValueError:
            # 检查是否是主机名
            return self._validate_hostname(ip)
    
    def _validate_hostname(self, hostname: str) -> bool:
        """验证主机名格式"""
        if not isinstance(hostname, str) or len(hostname) == 0 or len(hostname) > 255:
            return False
        
        # 允许的主机名模式
        hostname_pattern = re.compile(
            r'^(?![0-9]+$)(?!.*-$)(?!-)[a-zA-Z0-9-]{1,63}(?:\.[a-zA-Z0-9-]{1,63})*$'
        )
        
        return bool(hostname_pattern.match(hostname))
    
    def _validate_url(self, url: str) -> bool:
        """验证URL格式"""
        if not isinstance(url, str):
            return False
        
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain
            r'localhost|'  # localhost
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # IP
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE
        )
        
        return bool(url_pattern.match(url))


# ==================== 预定义验证配置 ====================

def create_development_validator() -> ConfigValidator:
    """创建开发环境验证器（宽松验证）"""
    validator = ConfigValidator()
    
    # 移除一些严格的验证规则
    validator.remove_validation_rule("transport.specific_config.host")
    
    # 添加开发环境特定规则
    dev_rules = [
        ValidationRule(
            field_path="debug_mode",
            rule_type="boolean",
            required=False,
            default_value=True
        ),
        
        ValidationRule(
            field_path="log_level",
            rule_type="string",
            required=False,
            default_value="DEBUG",
            validator=lambda x: x in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        )
    ]
    
    for rule in dev_rules:
        validator.add_validation_rule(rule)
    
    return validator


def create_production_validator() -> ConfigValidator:
    """创建生产环境验证器（严格验证）"""
    validator = ConfigValidator()
    
    # 添加生产环境特定的严格规则
    prod_rules = [
        ValidationRule(
            field_path="debug_mode",
            rule_type="boolean",
            required=False,
            default_value=False,
            validator=lambda x: x is False,
            error_message="debug_mode must be False in production"
        ),
        
        ValidationRule(
            field_path="log_level",
            rule_type="string",
            required=False,
            default_value="WARNING",
            validator=lambda x: x in ["WARNING", "ERROR", "CRITICAL"],
            error_message="log_level should be WARNING or higher in production"
        ),
        
        ValidationRule(
            field_path="communication.max_clients",
            rule_type="integer",
            required=True,
            validator=lambda x: isinstance(x, int) and 1 <= x <= 1000,
            error_message="max_clients must be between 1 and 1000 in production"
        ),
        
        # 网络模式在生产环境中需要SSL
        ValidationRule(
            field_path="transport.specific_config.ssl_enabled",
            rule_type="boolean",
            required=False,
            default_value=True,
            depends_on=["mode"],
            validator=lambda x: x is True,
            error_message="SSL must be enabled in production network mode"
        )
    ]
    
    for rule in prod_rules:
        validator.add_validation_rule(rule)
    
    return validator


def create_testing_validator() -> ConfigValidator:
    """创建测试环境验证器"""
    validator = ConfigValidator()
    
    # 测试环境特定规则
    test_rules = [
        ValidationRule(
            field_path="testing_mode",
            rule_type="boolean",
            required=False,
            default_value=True
        ),
        
        ValidationRule(
            field_path="federation.max_rounds",
            rule_type="integer",
            required=False,
            default_value=5,  # 测试环境使用较少的轮次
            validator=lambda x: isinstance(x, int) and 1 <= x <= 20
        ),
        
        ValidationRule(
            field_path="communication.heartbeat_interval",
            rule_type="number",
            required=False,
            default_value=5.0,  # 测试环境使用较短的心跳间隔
            validator=lambda x: isinstance(x, (int, float)) and 1.0 <= x <= 30.0
        )
    ]
    
    for rule in test_rules:
        validator.add_validation_rule(rule)
    
    return validator


# ==================== 便捷函数 ====================

def validate_config(config: Dict[str, Any], env: str = "development") -> Dict[str, Any]:
    """便捷的配置验证函数
    
    Args:
        config: 配置字典
        env: 环境类型 ("development", "production", "testing")
        
    Returns:
        Dict[str, Any]: 验证后的配置
    """
    if env == "production":
        validator = create_production_validator()
    elif env == "testing":
        validator = create_testing_validator()
    else:
        validator = create_development_validator()
    
    return validator.validate_config(config)


def quick_validate(config: Dict[str, Any]) -> bool:
    """快速验证配置是否基本有效
    
    Args:
        config: 配置字典
        
    Returns:
        bool: 是否有效
    """
    try:
        validator = ConfigValidator()
        validator.validate_config(config)
        return True
    except (ValidationError, ConfigurationError):
        return False