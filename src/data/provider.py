"""
DataProvider - 数据提供者接口

定义数据访问的统一接口
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Tuple


class DataProvider(ABC):
    """
    数据提供者抽象基类
    
    职责：
    - 提供训练数据迭代器
    - 提供评估数据迭代器
    - 提供数据信息
    """
    
    @abstractmethod
    def get_train_loader(self) -> Iterator[Tuple[Any, Any]]:
        """
        获取训练数据加载器
        
        Returns:
            迭代器，每次返回 (batch_x, batch_y)
        """
        pass
    
    @abstractmethod
    def get_eval_loader(self) -> Iterator[Tuple[Any, Any]]:
        """
        获取评估数据加载器
        
        Returns:
            迭代器，每次返回 (batch_x, batch_y)
        """
        pass
    
    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """
        获取数据信息
        
        Returns:
            数据信息字典，至少包含 num_samples
        """
        pass
    
    def get_num_samples(self) -> int:
        """获取样本数量"""
        return self.get_info().get("num_samples", 0)


class InMemoryDataProvider(DataProvider):
    """
    内存数据提供者
    
    简单的内存数据提供者，用于测试或小数据集
    """
    
    def __init__(
        self,
        train_x: Any,
        train_y: Any,
        eval_x: Optional[Any] = None,
        eval_y: Optional[Any] = None,
        batch_size: int = 32,
    ):
        """
        初始化
        
        Args:
            train_x: 训练特征
            train_y: 训练标签
            eval_x: 评估特征（可选）
            eval_y: 评估标签（可选）
            batch_size: 批大小
        """
        self._train_x = train_x
        self._train_y = train_y
        self._eval_x = eval_x if eval_x is not None else train_x
        self._eval_y = eval_y if eval_y is not None else train_y
        self._batch_size = batch_size
    
    def get_train_loader(self) -> Iterator[Tuple[Any, Any]]:
        """获取训练数据加载器"""
        return self._batch_iterator(self._train_x, self._train_y)
    
    def get_eval_loader(self) -> Iterator[Tuple[Any, Any]]:
        """获取评估数据加载器"""
        return self._batch_iterator(self._eval_x, self._eval_y)
    
    def get_info(self) -> Dict[str, Any]:
        """获取数据信息"""
        return {
            "num_samples": len(self._train_x),
            "num_eval_samples": len(self._eval_x),
            "batch_size": self._batch_size,
        }
    
    def _batch_iterator(self, x: Any, y: Any) -> Iterator[Tuple[Any, Any]]:
        """批次迭代器"""
        n = len(x)
        for i in range(0, n, self._batch_size):
            end = min(i + self._batch_size, n)
            yield x[i:end], y[i:end]


class SubsetDataProvider(DataProvider):
    """
    子集数据提供者
    
    从完整数据集中按 indices 选择子集
    """
    
    def __init__(
        self,
        full_provider: DataProvider,
        indices: List[int],
        batch_size: int = 32,
    ):
        """
        初始化
        
        Args:
            full_provider: 完整数据提供者
            indices: 选择的索引列表
            batch_size: 批大小
        """
        self._full_provider = full_provider
        self._indices = indices
        self._batch_size = batch_size
        
        # 提取子集（这里假设 full_provider 支持索引访问）
        self._subset_loaded = False
        self._train_x = None
        self._train_y = None
    
    def _ensure_loaded(self) -> None:
        """确保数据已加载"""
        if self._subset_loaded:
            return
        
        # 尝试从 full_provider 获取数据
        info = self._full_provider.get_info()
        if "data" in info:
            full_x, full_y = info["data"]
            self._train_x = [full_x[i] for i in self._indices]
            self._train_y = [full_y[i] for i in self._indices]
            self._subset_loaded = True
    
    def get_train_loader(self) -> Iterator[Tuple[Any, Any]]:
        """获取训练数据加载器"""
        self._ensure_loaded()
        if self._train_x is None:
            return iter([])
        
        n = len(self._train_x)
        for i in range(0, n, self._batch_size):
            end = min(i + self._batch_size, n)
            yield self._train_x[i:end], self._train_y[i:end]
    
    def get_eval_loader(self) -> Iterator[Tuple[Any, Any]]:
        """获取评估数据加载器"""
        return self.get_train_loader()
    
    def get_info(self) -> Dict[str, Any]:
        """获取数据信息"""
        return {
            "num_samples": len(self._indices),
            "indices": self._indices,
            "batch_size": self._batch_size,
        }
