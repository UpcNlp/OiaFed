"""
配置验证器
验证配置的结构、值和逻辑正确性
"""

from typing import Dict, Any, List, Callable, Optional
from dataclasses import dataclass


class ConfigValidationError(Exception):
    """配置验证错误"""

    def __init__(self, errors: List[str]):
        self.errors = errors
        message = "Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors)
        super().__init__(message)


@dataclass
class ValidationRule:
    """验证规则"""
    field_path: str                             # 字段路径，如 "transport.port"
    validator: Callable[[Any], bool]            # 验证函数
    error_message: str                          # 错误消息
    required: bool = True                       # 是否必需字段


@dataclass
class ConditionalValidationRule:
    """条件验证规则：当满足某个条件时，才检查某些字段"""
    condition: Callable[[Dict], bool]           # 条件函数
    rules: List[ValidationRule]                 # 如果条件满足，应用这些规则
    error_message: str = ""                     # 可选的上下文错误消息


class ConfigValidator:
    """
    配置验证器

    验证层次：
    1. 结构验证：必需字段是否存在、字段类型是否正确
    2. 值验证：数值范围、枚举值、格式验证
    3. 基本逻辑验证：字段之间的依赖关系
    """

    def __init__(
        self,
        rules: Optional[List[ValidationRule]] = None,
        conditional_rules: Optional[List[ConditionalValidationRule]] = None
    ):
        """
        初始化验证器

        Args:
            rules: 基本验证规则列表
            conditional_rules: 条件验证规则列表
        """
        self.rules = rules or []
        self.conditional_rules = conditional_rules or []

    def validate(self, config: Dict[str, Any]) -> List[str]:
        """
        验证配置，返回错误消息列表

        Args:
            config: 配置字典

        Returns:
            错误消息列表，空列表表示验证通过
        """
        errors = []

        # 1. 验证基本规则
        for rule in self.rules:
            error = self._validate_rule(config, rule)
            if error:
                errors.append(error)

        # 2. 验证条件规则
        for cond_rule in self.conditional_rules:
            if cond_rule.condition(config):
                for rule in cond_rule.rules:
                    error = self._validate_rule(config, rule)
                    if error:
                        if cond_rule.error_message:
                            errors.append(f"{cond_rule.error_message}: {error}")
                        else:
                            errors.append(error)

        return errors

    def validate_or_raise(self, config: Dict[str, Any]) -> None:
        """
        验证配置，如果有错误则抛出异常

        Args:
            config: 配置字典

        Raises:
            ConfigValidationError: 验证失败
        """
        errors = self.validate(config)
        if errors:
            raise ConfigValidationError(errors)

    def _validate_rule(self, config: Dict[str, Any], rule: ValidationRule) -> Optional[str]:
        """
        验证单个规则

        Args:
            config: 配置字典
            rule: 验证规则

        Returns:
            错误消息，None 表示验证通过
        """
        # 获取字段值
        value = self._get_nested_value(config, rule.field_path)

        # 检查必需字段
        if value is None:
            if rule.required:
                return f"Missing required field: {rule.field_path}"
            else:
                return None  # 可选字段，跳过验证

        # 执行验证函数
        try:
            if not rule.validator(value):
                return rule.error_message
        except Exception as e:
            return f"Validation error for {rule.field_path}: {str(e)}"

        return None

    def _get_nested_value(self, config: Dict[str, Any], path: str) -> Any:
        """
        获取嵌套字段的值

        Args:
            config: 配置字典
            path: 字段路径，如 "transport.port"

        Returns:
            字段值，如果路径不存在返回 None
        """
        keys = path.split('.')
        value = config

        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return None
            else:
                return None

        return value


# ========== 预定义的验证规则 ==========

# 通信配置验证规则
COMMUNICATION_VALIDATION_RULES = [
    # 部署模式验证
    ValidationRule(
        field_path="deployment.mode",
        validator=lambda x: x in ["memory", "process", "network"],
        error_message="deployment.mode must be one of: memory, process, network"
    ),

    # 角色验证
    ValidationRule(
        field_path="deployment.role",
        validator=lambda x: x in ["server", "client", "local"],
        error_message="deployment.role must be one of: server, client, local"
    ),

    # 传输类型验证
    ValidationRule(
        field_path="transport.type",
        validator=lambda x: x in ["tcp", "websocket", "grpc", "in_memory", "multiprocessing_queue"],
        error_message="transport.type must be one of: tcp, websocket, grpc, in_memory, multiprocessing_queue",
        required=False
    ),

    # 端口范围验证
    ValidationRule(
        field_path="transport.port",
        validator=lambda x: 1 <= x <= 65535,
        error_message="transport.port must be between 1 and 65535",
        required=False
    ),

    # 超时时间验证
    ValidationRule(
        field_path="transport.timeout",
        validator=lambda x: x > 0,
        error_message="transport.timeout must be positive",
        required=False
    ),

    # 心跳间隔验证
    ValidationRule(
        field_path="communication.heartbeat.interval",
        validator=lambda x: x > 0,
        error_message="communication.heartbeat.interval must be positive",
        required=False
    ),

    # 心跳超时验证
    ValidationRule(
        field_path="communication.heartbeat.timeout",
        validator=lambda x: x > 0,
        error_message="communication.heartbeat.timeout must be positive",
        required=False
    ),

    # 序列化格式验证
    ValidationRule(
        field_path="communication.serialization.format",
        validator=lambda x: x in ["pickle", "json", "protobuf"],
        error_message="communication.serialization.format must be one of: pickle, json, protobuf",
        required=False
    ),

    # 压缩方式验证
    ValidationRule(
        field_path="communication.serialization.compression",
        validator=lambda x: x in ["none", "gzip", "lz4"],
        error_message="communication.serialization.compression must be one of: none, gzip, lz4",
        required=False
    ),
]

# 条件验证规则
COMMUNICATION_CONDITIONAL_RULES = [
    # 如果是客户端，必须有 server 配置
    ConditionalValidationRule(
        condition=lambda cfg: cfg.get("deployment", {}).get("role") == "client",
        rules=[
            ValidationRule(
                field_path="transport.server.host",
                validator=lambda x: x is not None and len(str(x).strip()) > 0,
                error_message="Client must specify transport.server.host"
            ),
            ValidationRule(
                field_path="transport.server.port",
                validator=lambda x: x is not None and 1 <= x <= 65535,
                error_message="Client must specify valid transport.server.port"
            ),
        ],
        error_message="Client configuration incomplete"
    ),

    # 如果是服务端，必须有监听地址
    ConditionalValidationRule(
        condition=lambda cfg: cfg.get("deployment", {}).get("role") == "server",
        rules=[
            ValidationRule(
                field_path="transport.host",
                validator=lambda x: x is not None and len(str(x).strip()) > 0,
                error_message="Server must specify transport.host"
            ),
            ValidationRule(
                field_path="transport.port",
                validator=lambda x: x is not None and 1 <= x <= 65535,
                error_message="Server must specify valid transport.port"
            ),
        ],
        error_message="Server configuration incomplete"
    ),

    # 心跳超时应该大于心跳间隔
    ConditionalValidationRule(
        condition=lambda cfg: (
            cfg.get("communication", {}).get("heartbeat", {}).get("interval") is not None and
            cfg.get("communication", {}).get("heartbeat", {}).get("timeout") is not None
        ),
        rules=[
            ValidationRule(
                field_path="communication.heartbeat",
                validator=lambda hb: hb.get("timeout", 0) > hb.get("interval", 1),
                error_message="heartbeat.timeout must be greater than heartbeat.interval"
            ),
        ],
        error_message="Heartbeat configuration invalid"
    ),
]

# 训练配置验证规则
TRAINING_VALIDATION_RULES = [
    # 组件名称不能为空
    ValidationRule(
        field_path="trainer.name",
        validator=lambda x: isinstance(x, str) and len(x.strip()) > 0,
        error_message="trainer.name must be a non-empty string",
        required=False
    ),

    ValidationRule(
        field_path="learner.name",
        validator=lambda x: isinstance(x, str) and len(x.strip()) > 0,
        error_message="learner.name must be a non-empty string",
        required=False
    ),

    # 训练轮数验证
    ValidationRule(
        field_path="trainer.params.max_rounds",
        validator=lambda x: isinstance(x, int) and x > 0,
        error_message="trainer.params.max_rounds must be a positive integer",
        required=False
    ),

    # 最小客户端数验证
    ValidationRule(
        field_path="trainer.params.min_clients",
        validator=lambda x: isinstance(x, int) and x > 0,
        error_message="trainer.params.min_clients must be a positive integer",
        required=False
    ),

    # 本地训练轮次验证
    ValidationRule(
        field_path="learner.params.local_epochs",
        validator=lambda x: isinstance(x, int) and x > 0,
        error_message="learner.params.local_epochs must be a positive integer",
        required=False
    ),

    # 批次大小验证
    ValidationRule(
        field_path="learner.params.batch_size",
        validator=lambda x: isinstance(x, int) and x > 0,
        error_message="learner.params.batch_size must be a positive integer",
        required=False
    ),

    # 学习率验证
    ValidationRule(
        field_path="learner.params.learning_rate",
        validator=lambda x: isinstance(x, (int, float)) and x > 0,
        error_message="learner.params.learning_rate must be a positive number",
        required=False
    ),
]


# ========== 便捷验证器工厂 ==========

def create_communication_validator() -> ConfigValidator:
    """创建通信配置验证器"""
    return ConfigValidator(
        rules=COMMUNICATION_VALIDATION_RULES,
        conditional_rules=COMMUNICATION_CONDITIONAL_RULES
    )


def create_training_validator() -> ConfigValidator:
    """创建训练配置验证器"""
    return ConfigValidator(
        rules=TRAINING_VALIDATION_RULES,
        conditional_rules=[]
    )
