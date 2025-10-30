"""
基础配置类定义
支持自定义字段的灵活配置系统
"""

from typing import Any, Dict, Optional
from dataclasses import dataclass, field, fields


@dataclass
class BaseConfig:
    """
    基础配置类，所有配置类的父类

    特性：
    - 支持自定义字段：用户可以添加任意字段，存储在 extra_fields 中
    - 类型安全：已定义的字段有类型检查
    - 灵活访问：通过 get/set 方法访问标准字段和自定义字段
    """

    # 存储所有额外的自定义字段
    extra_fields: Dict[str, Any] = field(default_factory=dict, repr=False)

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取字段值，支持标准字段和自定义字段

        Args:
            key: 字段名称
            default: 如果字段不存在，返回的默认值

        Returns:
            字段值或默认值
        """
        # 首先检查是否是标准字段
        if hasattr(self, key) and key != 'extra_fields':
            return getattr(self, key)
        # 然后检查自定义字段
        return self.extra_fields.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """
        设置字段值
        - 如果是标准字段，直接设置属性
        - 如果是自定义字段，存入 extra_fields

        Args:
            key: 字段名称
            value: 字段值
        """
        if hasattr(self, key) and key != 'extra_fields':
            setattr(self, key, value)
        else:
            self.extra_fields[key] = value

    def has(self, key: str) -> bool:
        """
        检查字段是否存在

        Args:
            key: 字段名称

        Returns:
            True 如果字段存在
        """
        if hasattr(self, key) and key != 'extra_fields':
            return True
        return key in self.extra_fields

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式，包含所有字段（标准字段 + 自定义字段）

        Returns:
            包含所有配置的字典
        """
        result = {}

        # 添加所有标准字段
        for f in fields(self):
            if f.name == 'extra_fields':
                continue
            value = getattr(self, f.name)

            # 如果字段值也是 BaseConfig，递归转换
            if isinstance(value, BaseConfig):
                result[f.name] = value.to_dict()
            # 如果是列表，检查列表中的元素
            elif isinstance(value, list):
                result[f.name] = [
                    item.to_dict() if isinstance(item, BaseConfig) else item
                    for item in value
                ]
            # 如果是字典，检查字典中的值
            elif isinstance(value, dict):
                result[f.name] = {
                    k: v.to_dict() if isinstance(v, BaseConfig) else v
                    for k, v in value.items()
                }
            else:
                result[f.name] = value

        # 添加所有自定义字段
        result.update(self.extra_fields)

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseConfig':
        """
        从字典创建配置对象

        Args:
            data: 配置字典

        Returns:
            配置对象实例
        """
        # 获取类的所有标准字段
        known_fields = {f.name for f in fields(cls) if f.name != 'extra_fields'}

        # 分离标准字段和自定义字段
        known_data = {}
        extra_data = {}

        for key, value in data.items():
            if key in known_fields:
                known_data[key] = value
            else:
                extra_data[key] = value

        # 创建实例
        instance = cls(**known_data)

        # 添加自定义字段
        instance.extra_fields = extra_data

        return instance

    def __getitem__(self, key: str) -> Any:
        """支持字典式访问：config['key']"""
        return self.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """支持字典式设置：config['key'] = value"""
        self.set(key, value)

    def __contains__(self, key: str) -> bool:
        """支持 in 操作符：'key' in config"""
        return self.has(key)
