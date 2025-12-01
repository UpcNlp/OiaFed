"""
Learner Registry with Namespace Support
联邦学习方法注册中心 (命名空间版本)

命名空间:
- fl: 标准联邦学习 (federated_learning)
- cl: 联邦持续学习 (continual_learning)
- ul: 联邦遗忘学习 (unlearning)

使用格式: "fl.FedAvg", "cl.TARGET", "ul.FedEraser"
"""

from typing import Dict, Type, List, Optional, Any
from pathlib import Path


class LearnerRegistry:
    """Learner注册中心 (带命名空间)"""

    def __init__(self):
        """初始化注册中心"""
        # 主注册表: {'fl.FedAvg': FedAvgLearner, 'cl.TARGET': TARGETLearner, ...}
        self._registry: Dict[str, Type] = {}

        # 命名空间元数据
        self._namespace_meta = {
            'fl': {
                'full_name': 'federated_learning',
                'description': '标准联邦学习算法',
                'color': 'blue'
            },
            'cl': {
                'full_name': 'continual_learning',
                'description': '联邦持续学习算法',
                'color': 'green'
            },
            'ul': {
                'full_name': 'unlearning',
                'description': '联邦遗忘学习算法',
                'color': 'red'
            }
        }

    def register(self, namespace: str, name: str, cls: Type) -> None:
        """
        注册一个learner

        Args:
            namespace: 命名空间 (fl/cl/ul)
            name: 方法名 (如 FedAvg, TARGET)
            cls: Learner类

        Raises:
            ValueError: 命名空间无效或已存在重名
        """
        # 验证命名空间
        if namespace not in self._namespace_meta:
            raise ValueError(
                f"Invalid namespace '{namespace}'. "
                f"Available: {', '.join(self._namespace_meta.keys())}"
            )

        # 构造完整名称
        full_name = f"{namespace}.{name}"

        # 检查重复
        if full_name in self._registry:
            raise ValueError(
                f"Learner '{full_name}' already registered. "
                f"Existing class: {self._registry[full_name].__name__}"
            )

        # 注册
        self._registry[full_name] = cls

    def get(self, full_name: str) -> Type:
        """
        获取learner类

        Args:
            full_name: 完整名称 (如 "fl.FedAvg", "cl.TARGET")

        Returns:
            Learner类

        Raises:
            ValueError: 名称格式错误或不存在
        """
        # 验证格式
        if '.' not in full_name:
            raise ValueError(
                f"Invalid learner name format '{full_name}'. "
                f"Expected format: 'namespace.Method' (e.g., 'fl.FedAvg', 'cl.TARGET')\n"
                f"Available namespaces: {', '.join(self._namespace_meta.keys())}"
            )

        # 解析命名空间
        namespace, method = full_name.split('.', 1)

        # 验证命名空间
        if namespace not in self._namespace_meta:
            raise ValueError(
                f"Unknown namespace '{namespace}' in '{full_name}'. "
                f"Available namespaces: {', '.join(self._namespace_meta.keys())}"
            )

        # 获取learner
        if full_name not in self._registry:
            # 提供有用的错误信息
            available = self.list_learners(namespace)
            error_msg = f"Learner '{full_name}' not found in namespace '{namespace}'."

            if available:
                error_msg += f"\nAvailable learners in '{namespace}': {', '.join(available)}"

                # 尝试提供拼写建议
                suggestion = self._find_closest_match(method, available)
                if suggestion:
                    error_msg += f"\nDid you mean: '{namespace}.{suggestion}'?"
            else:
                error_msg += f"\nNo learners registered in namespace '{namespace}'."

            raise ValueError(error_msg)

        return self._registry[full_name]

    def exists(self, full_name: str) -> bool:
        """检查learner是否存在"""
        return full_name in self._registry

    def list_namespaces(self) -> List[str]:
        """列出所有命名空间"""
        return list(self._namespace_meta.keys())

    def list_learners(self, namespace: str) -> List[str]:
        """
        列出命名空间下的所有learner

        Args:
            namespace: 命名空间 (fl/cl/ul)

        Returns:
            方法名列表 (不含命名空间前缀)
        """
        prefix = f"{namespace}."
        return [
            name[len(prefix):]
            for name in self._registry.keys()
            if name.startswith(prefix)
        ]

    def get_all(self) -> Dict[str, Type]:
        """获取所有注册的learner"""
        return self._registry.copy()

    def get_namespace_info(self, namespace: str) -> Dict[str, Any]:
        """获取命名空间元数据"""
        if namespace not in self._namespace_meta:
            raise ValueError(f"Unknown namespace: {namespace}")
        return self._namespace_meta[namespace].copy()

    def count(self, namespace: Optional[str] = None) -> int:
        """
        统计learner数量

        Args:
            namespace: 命名空间 (None表示所有)

        Returns:
            数量
        """
        if namespace is None:
            return len(self._registry)
        else:
            return len(self.list_learners(namespace))

    def search(self, query: str, case_sensitive: bool = False, partial: bool = True) -> List[str]:
        """
        搜索learner

        Args:
            query: 搜索关键词
            case_sensitive: 是否大小写敏感
            partial: 是否部分匹配

        Returns:
            匹配的完整名称列表
        """
        if not case_sensitive:
            query = query.lower()

        results = []
        for full_name in self._registry.keys():
            # 提取方法名用于匹配
            method_name = full_name.split('.', 1)[1]
            target = method_name if case_sensitive else method_name.lower()

            if partial:
                if query in target:
                    results.append(full_name)
            else:
                if query == target:
                    results.append(full_name)

        return sorted(results)

    def validate(self, full_name: str) -> bool:
        """
        验证learner名称

        Args:
            full_name: 完整名称

        Returns:
            是否有效 (会抛出异常说明原因)
        """
        try:
            self.get(full_name)
            return True
        except ValueError:
            raise

    def _find_closest_match(self, query: str, candidates: List[str]) -> Optional[str]:
        """
        找到最接近的匹配 (简单的编辑距离)

        Args:
            query: 查询字符串
            candidates: 候选列表

        Returns:
            最接近的候选 (如果足够相似)
        """
        def levenshtein_distance(s1: str, s2: str) -> int:
            """计算编辑距离"""
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)
            if len(s2) == 0:
                return len(s1)

            previous_row = range(len(s2) + 1)
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row

            return previous_row[-1]

        # 不区分大小写比较
        query_lower = query.lower()
        best_match = None
        best_distance = float('inf')

        for candidate in candidates:
            distance = levenshtein_distance(query_lower, candidate.lower())
            if distance < best_distance:
                best_distance = distance
                best_match = candidate

        # 只在距离足够小时返回建议
        if best_match and best_distance <= max(2, len(query) // 3):
            return best_match

        return None

    def summary(self) -> str:
        """生成注册中心摘要"""
        lines = ["Learner Registry Summary", "=" * 50]

        for namespace in self.list_namespaces():
            info = self._namespace_meta[namespace]
            learners = self.list_learners(namespace)
            lines.append(f"\n{namespace} ({info['full_name']})")
            lines.append(f"  Description: {info['description']}")
            lines.append(f"  Learners ({len(learners)}):")
            for learner in sorted(learners):
                lines.append(f"    - {namespace}.{learner}")

        return "\n".join(lines)


# 全局注册中心实例
_global_registry = LearnerRegistry()


# 导出便捷函数
def register(namespace: str, name: str, cls: Type) -> None:
    """注册learner到全局registry"""
    _global_registry.register(namespace, name, cls)


def get(full_name: str) -> Type:
    """从全局registry获取learner"""
    return _global_registry.get(full_name)


def get_registry() -> LearnerRegistry:
    """获取全局registry实例"""
    return _global_registry
