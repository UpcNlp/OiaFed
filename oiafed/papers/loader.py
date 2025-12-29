"""
论文注册表加载器

功能：
1. 从 YAML 文件加载论文定义
2. 提供查询接口（列表、搜索、获取）
3. 配置生成与合并
4. 参数验证
"""

import os
import copy
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import yaml

from ..infra import get_module_logger

logger = get_module_logger(__name__)


# ==================== 数据类 ====================

@dataclass
class ParamDef:
    """参数定义"""
    name: str
    type: str = "str"
    desc: str = ""
    range: Optional[List] = None
    choices: Optional[List] = None
    default: Any = None
    
    @classmethod
    def from_dict(cls, name: str, data: Dict) -> "ParamDef":
        return cls(
            name=name,
            type=data.get("type", "str"),
            desc=data.get("desc", ""),
            range=data.get("range"),
            choices=data.get("choices"),
            default=data.get("default"),
        )
    
    def validate(self, value: Any) -> Optional[str]:
        """验证参数值，返回错误信息或None"""
        # 类型检查
        type_map = {
            "int": int,
            "float": (int, float),
            "str": str,
            "bool": bool,
            "list": list,
        }
        
        expected_type = type_map.get(self.type)
        if expected_type and not isinstance(value, expected_type):
            return f"{self.name}: 期望 {self.type}, 实际 {type(value).__name__}"
        
        # 范围检查
        if self.range and len(self.range) == 2:
            if value < self.range[0] or value > self.range[1]:
                return f"{self.name}: {value} 不在范围 {self.range} 内"
        
        # 选项检查
        if self.choices and value not in self.choices:
            return f"{self.name}: {value} 不在可选项 {self.choices} 中"
        
        return None


@dataclass
class PaperDef:
    """论文定义"""
    id: str
    name: str
    category: str
    venue: str = ""
    year: int = 0
    url: str = ""
    description: str = ""
    
    components: Dict[str, str] = field(default_factory=dict)
    defaults: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, ParamDef] = field(default_factory=dict)
    
    citation: str = ""
    notes: str = ""
    
    # 源文件路径
    _source_file: str = ""
    
    @classmethod
    def from_dict(cls, data: Dict, source_file: str = "") -> "PaperDef":
        """从字典创建"""
        # 解析参数定义
        params = {}
        for param_name, param_data in data.get("params", {}).items():
            params[param_name] = ParamDef.from_dict(param_name, param_data)
        
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            category=data.get("category", ""),
            venue=data.get("venue", ""),
            year=data.get("year", 0),
            url=data.get("url", ""),
            description=data.get("description", "").strip(),
            components=data.get("components", {}),
            defaults=data.get("defaults", {}),
            params=params,
            citation=data.get("citation", "").strip(),
            notes=data.get("notes", "").strip(),
            _source_file=source_file,
        )
    
    def get_component(self, component_type: str) -> Optional[str]:
        """获取组件类型"""
        return self.components.get(component_type)
    
    def get_default(self, path: str, default: Any = None) -> Any:
        """获取默认值，支持点号路径如 'learner.learning_rate'"""
        parts = path.split(".")
        value = self.defaults
        
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        
        return value
    
    def get_param_def(self, path: str) -> Optional[ParamDef]:
        """获取参数定义"""
        return self.params.get(path)


# ==================== 注册表 ====================

class PaperRegistry:
    """
    论文注册表
    
    从 YAML 文件加载论文定义，提供查询和配置生成功能。
    
    使用方式:
        registry = PaperRegistry()
        
        # 列出所有论文
        papers = registry.list_all()
        
        # 获取论文
        paper = registry.get("fot")
        
        # 生成配置
        configs = registry.generate_configs("fot", override={"learner": {"lr": 0.001}})
    """
    
    # 类别名称映射
    CATEGORY_NAMES = {
        "HFL": "横向联邦学习 (Horizontal FL)",
        "VFL": "纵向联邦学习 (Vertical FL)",
        "FCL": "联邦持续学习 (Federated Continual Learning)",
        "FU": "联邦遗忘 (Federated Unlearning)",
    }
    
    def __init__(self, defs_dir: Optional[str] = None):
        """
        初始化注册表
        
        Args:
            defs_dir: 论文定义目录，默认为 src/papers/defs/
        """
        self.papers: Dict[str, PaperDef] = {}
        
        if defs_dir is None:
            defs_dir = Path(__file__).parent / "defs"
        
        self.defs_dir = Path(defs_dir)
        self._load_all()
    
    def _load_all(self) -> None:
        """加载所有论文定义（支持子目录）"""
        if not self.defs_dir.exists():
            logger.warning(f"论文定义目录不存在: {self.defs_dir}")
            return
        
        # 递归扫描所有 yaml 文件
        for yaml_file in self.defs_dir.rglob("*.yaml"):
            # 跳过模板文件
            if yaml_file.name.startswith("_"):
                continue
            
            try:
                self._load_file(yaml_file)
            except Exception as e:
                logger.error(f"加载论文定义失败 {yaml_file}: {e}")
        
        logger.info(f"已加载 {len(self.papers)} 篇论文定义")
    
    def _load_file(self, yaml_file: Path) -> None:
        """加载单个 YAML 文件"""
        with open(yaml_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        if not data:
            return
        
        paper = PaperDef.from_dict(data, str(yaml_file))
        
        # 使用文件名作为 ID（如果未指定）
        if not paper.id:
            paper.id = yaml_file.stem
        
        self.papers[paper.id] = paper
        # logger.debug(f"已加载论文: {paper.id} ({paper.name})")
    
    # ==================== 查询接口 ====================
    
    def list_all(self) -> List[str]:
        """列出所有论文 ID"""
        return list(self.papers.keys())
    
    def list_by_category(self, category: str) -> List[str]:
        """按类别列出论文"""
        category = category.upper()
        return [
            paper_id for paper_id, paper in self.papers.items()
            if paper.category.upper() == category
        ]
    
    def get(self, paper_id: str) -> Optional[PaperDef]:
        """获取论文定义"""
        return self.papers.get(paper_id)
    
    def search(self, keyword: str) -> List[PaperDef]:
        """搜索论文（按名称、描述）"""
        keyword = keyword.lower()
        results = []
        
        for paper in self.papers.values():
            if (keyword in paper.id.lower() or
                keyword in paper.name.lower() or
                keyword in paper.description.lower()):
                results.append(paper)
        
        return results
    
    def get_categories(self) -> Dict[str, str]:
        """获取所有类别"""
        return self.CATEGORY_NAMES.copy()
    
    def get_papers_grouped(self) -> Dict[str, List[PaperDef]]:
        """按类别分组获取论文"""
        grouped = {}
        for paper in self.papers.values():
            category = paper.category.upper()
            if category not in grouped:
                grouped[category] = []
            grouped[category].append(paper)
        return grouped
    
    # ==================== 配置生成 ====================
    
    def get_defaults(self, paper_id: str) -> Dict[str, Any]:
        """获取论文默认配置"""
        paper = self.get(paper_id)
        if not paper:
            raise ValueError(f"未找到论文: {paper_id}")
        
        return copy.deepcopy(paper.defaults)
    
    def merge_override(
        self,
        paper_id: str,
        override: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        合并用户覆盖配置
        
        Args:
            paper_id: 论文 ID
            override: 用户覆盖配置
            
        Returns:
            合并后的配置
        """
        defaults = self.get_defaults(paper_id)
        
        if not override:
            return defaults
        
        return self._deep_merge(defaults, override)
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """深度合并字典"""
        result = copy.deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        
        return result
    
    def apply_set_params(
        self,
        config: Dict[str, Any],
        set_params: List[str],
    ) -> Dict[str, Any]:
        """
        应用 --set 参数
        
        Args:
            config: 配置字典
            set_params: 参数列表，格式如 ["learner.lr=0.01", "trainer.rounds=50"]
            
        Returns:
            更新后的配置
        """
        result = copy.deepcopy(config)
        
        for param in set_params:
            if "=" not in param:
                logger.warning(f"无效的参数格式: {param}")
                continue
            
            key, value = param.split("=", 1)
            parts = key.strip().split(".")
            
            # 解析值类型
            value = self._parse_value(value.strip())
            
            # 设置值
            self._set_nested(result, parts, value)
        
        return result
    
    def _parse_value(self, value: str) -> Any:
        """解析参数值的类型"""
        # bool
        if value.lower() in ("true", "false"):
            return value.lower() == "true"
        
        # int
        try:
            return int(value)
        except ValueError:
            pass
        
        # float
        try:
            return float(value)
        except ValueError:
            pass
        
        # list (简单格式: "a,b,c")
        if "," in value:
            return [self._parse_value(v.strip()) for v in value.split(",")]
        
        # string
        return value
    
    def _set_nested(self, config: Dict, parts: List[str], value: Any) -> None:
        """设置嵌套字典的值"""
        current = config
        
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        current[parts[-1]] = value
    
    # ==================== 配置文件生成 ====================
    
    def generate_node_configs(
        self,
        paper_id: str,
        override: Optional[Dict[str, Any]] = None,
        num_clients: Optional[int] = None,
        output_dir: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        生成节点配置文件
        
        Args:
            paper_id: 论文 ID
            override: 用户覆盖配置
            num_clients: 客户端数量（覆盖默认值）
            output_dir: 输出目录（如果指定，则写入文件）
            
        Returns:
            配置字典列表 [trainer_config, learner_0_config, learner_1_config, ...]
        """
        paper = self.get(paper_id)
        if not paper:
            raise ValueError(f"未找到论文: {paper_id}")
        
        # 合并配置
        merged = self.merge_override(paper_id, override)
        
        # 确定客户端数量
        if num_clients is None:
            num_clients = merged.get("experiment", {}).get("num_clients", 2)
        
        configs = []
        
        # 生成 Trainer 配置
        trainer_config = self._generate_trainer_config(paper, merged, num_clients)
        configs.append(trainer_config)
        
        # 生成 Learner 配置
        for i in range(num_clients):
            learner_config = self._generate_learner_config(paper, merged, i, num_clients)
            configs.append(learner_config)
        
        # 写入文件（如果指定）
        if output_dir:
            self._write_configs(configs, output_dir)
        
        return configs
    
    def _generate_trainer_config(
        self,
        paper: PaperDef,
        merged: Dict,
        num_clients: int,
    ) -> Dict[str, Any]:
        """生成 Trainer 配置"""
        config = {
            "node_id": "trainer",
            "role": "trainer",
            "listen": {
                "host": "localhost",
                "port": 50051,
            },
            "min_peers": num_clients,
            "transport": {
                "mode": "grpc",
                "grpc": {"max_message_size": 104857600},
            },
            "logging": {
                "level": "INFO",
                "console": True,
            },
            "serialization": {"default": "pickle"},
        }
        
        # Trainer 组件
        trainer_type = paper.get_component("trainer") or "default"
        config["trainer"] = {
            "type": trainer_type,
            "args": merged.get("trainer", {}),
        }
        
        # Aggregator 组件
        aggregator_type = paper.get_component("aggregator") or "fedavg"
        config["aggregator"] = {
            "type": aggregator_type,
            "args": merged.get("aggregator", {}),
        }
        
        # Model
        model_type = paper.get_component("model") or "mnist_cnn"
        config["model"] = {
            "type": model_type,
            "args": merged.get("model", {}),
        }
        
        return config
    
    def _generate_learner_config(
        self,
        paper: PaperDef,
        merged: Dict,
        index: int,
        num_clients: int,
    ) -> Dict[str, Any]:
        """生成 Learner 配置"""
        config = {
            "node_id": f"learner_{index}",
            "role": "learner",
            "listen": {
                "host": "localhost",
                "port": 50052 + index,
            },
            "connect_to": ["trainer@localhost:50051"],
            "transport": {
                "mode": "grpc",
                "grpc": {"max_message_size": 104857600},
            },
            "logging": {
                "level": "INFO",
                "console": True,
            },
            "serialization": {"default": "pickle"},
        }
        
        # Learner 组件
        learner_type = paper.get_component("learner") or "default"
        config["learner"] = {
            "type": learner_type,
            "args": merged.get("learner", {}),
        }
        
        # Model
        model_type = paper.get_component("model") or "mnist_cnn"
        config["model"] = {
            "type": model_type,
            "args": merged.get("model", {}),
        }
        
        # Dataset
        dataset_type = paper.get_component("dataset") or "mnist"
        dataset_config = merged.get("dataset", {})
        partition_config = dataset_config.pop("partition", {})
        
        config["datasets"] = [
            {
                "type": dataset_type,
                "split": "train",
                "args": dataset_config.copy(),
                "partition": {
                    **partition_config,
                    "num_partitions": num_clients,
                    "partition_id": index,
                },
            },
            {
                "type": dataset_type,
                "split": "test",
                "args": {k: v for k, v in dataset_config.items() if k != "download"},
            },
        ]
        
        return config
    
    def _write_configs(self, configs: List[Dict], output_dir: str) -> None:
        """写入配置文件"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for config in configs:
            node_id = config["node_id"]
            file_path = output_path / f"{node_id}.yaml"
            
            with open(file_path, "w", encoding="utf-8") as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            
            logger.info(f"已生成: {file_path}")
    
    # ==================== 验证 ====================
    
    def validate_override(
        self,
        paper_id: str,
        override: Dict[str, Any],
    ) -> List[str]:
        """
        验证用户覆盖配置
        
        Returns:
            警告/错误信息列表
        """
        paper = self.get(paper_id)
        if not paper:
            return [f"未找到论文: {paper_id}"]
        
        warnings = []
        
        # 递归验证
        self._validate_dict(paper, override, "", warnings)
        
        return warnings
    
    def _validate_dict(
        self,
        paper: PaperDef,
        data: Dict,
        prefix: str,
        warnings: List[str],
    ) -> None:
        """递归验证字典"""
        for key, value in data.items():
            path = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                self._validate_dict(paper, value, path, warnings)
            else:
                param_def = paper.get_param_def(path)
                if param_def:
                    error = param_def.validate(value)
                    if error:
                        warnings.append(error)
    
    # ==================== 格式化输出 ====================
    
    def format_paper_info(self, paper_id: str, show_params: bool = False) -> str:
        """格式化论文信息"""
        paper = self.get(paper_id)
        if not paper:
            return f"未找到论文: {paper_id}"
        
        lines = [
            f"{'=' * 60}",
            f" {paper.name}",
            f"{'=' * 60}",
            f"",
            f"ID:       {paper.id}",
            f"类别:     {paper.category} ({self.CATEGORY_NAMES.get(paper.category, '')})",
            f"发表:     {paper.venue} {paper.year}",
            f"链接:     {paper.url}",
            f"",
            f"描述:",
            f"  {paper.description}",
            f"",
            f"组件:",
        ]
        
        for comp_type, comp_name in paper.components.items():
            lines.append(f"  - {comp_type}: {comp_name}")
        
        if show_params and paper.params:
            lines.extend([
                f"",
                f"可调参数:",
            ])
            
            # 按组件分组
            grouped = {}
            for param_path, param_def in paper.params.items():
                group = param_path.split(".")[0]
                if group not in grouped:
                    grouped[group] = []
                grouped[group].append((param_path, param_def))
            
            for group, params in grouped.items():
                lines.append(f"  [{group}]")
                for param_path, param_def in params:
                    short_name = param_path.split(".", 1)[1] if "." in param_path else param_path
                    default = paper.get_default(param_path, "N/A")
                    
                    constraint = ""
                    if param_def.range:
                        constraint = f" [{param_def.range[0]}, {param_def.range[1]}]"
                    elif param_def.choices:
                        constraint = f" {param_def.choices}"
                    
                    lines.append(
                        f"    {short_name:<20} {param_def.type:<6}{constraint:<20} "
                        f"{param_def.desc} (默认: {default})"
                    )
        
        if paper.notes:
            lines.extend([
                f"",
                f"备注:",
                f"  {paper.notes}",
            ])
        
        lines.append(f"{'=' * 60}")
        
        return "\n".join(lines)
    
    def format_paper_list(self) -> str:
        """格式化论文列表"""
        grouped = self.get_papers_grouped()
        
        lines = [
            f"{'=' * 60}",
            f" OiaFed 论文库",
            f"{'=' * 60}",
            f"",
        ]
        
        for category in ["HFL", "VFL", "FCL", "FU"]:
            if category not in grouped:
                continue
            
            category_name = self.CATEGORY_NAMES.get(category, category)
            lines.append(f"[{category}] {category_name}")
            lines.append(f"-" * 40)
            
            for paper in grouped[category]:
                lines.append(f"  {paper.id:<15} {paper.name}")
            
            lines.append(f"")
        
        lines.extend([
            f"共 {len(self.papers)} 篇论文",
            f"",
            f"使用方法:",
            f"  oiafed papers show <paper_id>          # 查看详情",
            f"  oiafed papers show <paper_id> --params # 查看可调参数",
            f"  oiafed papers run <paper_id>           # 运行实验",
            f"{'=' * 60}",
        ])
        
        return "\n".join(lines)


# ==================== 全局单例 ====================

_registry: Optional[PaperRegistry] = None


def get_registry() -> PaperRegistry:
    """获取全局论文注册表"""
    global _registry
    if _registry is None:
        _registry = PaperRegistry()
    return _registry


def reload_registry() -> PaperRegistry:
    """重新加载论文注册表"""
    global _registry
    _registry = PaperRegistry()
    return _registry
