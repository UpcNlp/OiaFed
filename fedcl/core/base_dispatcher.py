# fedcl/core/base_dispatcher.py
"""
下发器基类模块

定义下发钩子的基础接口，与现有基类保持一致的设计模式。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from loguru import logger

from ..config.config_manager import DictConfig
from ..exceptions import FedCLError


class DispatchError(FedCLError):
    """下发器错误"""
    pass


class BaseDispatcher(ABC):
    """
    下发器基类
    
    定义联邦学习中模型下发的标准接口。下发器负责：
    1. 准备下发数据
    2. 选择下发目标
    3. 为特定客户端定制数据
    """
    
    def __init__(self, config: DictConfig):
        """
        初始化下发器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.name = getattr(self.__class__, '__name__', 'UnknownDispatcher')
        self.version = getattr(self.__class__, '__version__', '1.0.0')
        
        # 下发统计
        self.dispatch_count = 0
        self.last_dispatch_time = None
        
        logger.debug(f"下发器初始化: {self.name}")
    
    @abstractmethod
    def prepare_dispatch_data(self, aggregated_result: Dict[str, Any]) -> Any:
        """
        准备下发数据
        
        Args:
            aggregated_result: 聚合结果
            
        Returns:
            Any: 准备好的下发数据
            
        Raises:
            DispatchError: 数据准备失败
        """
        raise NotImplementedError("子类必须实现 prepare_dispatch_data 方法")
    
    def select_targets(self, available_clients: List[Any]) -> List[Any]:
        """
        选择下发目标
        
        Args:
            available_clients: 可用客户端列表
            
        Returns:
            List[Any]: 选中的客户端列表
        """
        # 默认下发给所有客户端
        return available_clients
    
    def customize_for_client(self, dispatch_data: Any, client: Any) -> Any:
        """
        为特定客户端定制下发内容
        
        Args:
            dispatch_data: 原始下发数据
            client: 目标客户端
            
        Returns:
            Any: 定制后的数据
        """
        # 默认不进行定制
        return dispatch_data
    
    def validate_dispatch_data(self, dispatch_data: Any) -> bool:
        """
        验证下发数据
        
        Args:
            dispatch_data: 下发数据
            
        Returns:
            bool: 数据是否有效
        """
        return dispatch_data is not None
    
    def pre_dispatch_hook(self, dispatch_data: Any, targets: List[Any]) -> None:
        """
        下发前钩子
        
        Args:
            dispatch_data: 下发数据
            targets: 目标客户端列表
        """
        pass
    
    def post_dispatch_hook(self, dispatch_data: Any, targets: List[Any], 
                          results: List[bool]) -> None:
        """
        下发后钩子
        
        Args:
            dispatch_data: 下发数据
            targets: 目标客户端列表
            results: 下发结果列表
        """
        pass
    
    def get_dispatch_statistics(self) -> Dict[str, Any]:
        """
        获取下发统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        return {
            'name': self.name,
            'version': self.version,
            'dispatch_count': self.dispatch_count,
            'last_dispatch_time': self.last_dispatch_time
        }


class ModelDispatcher(BaseDispatcher):
    """
    模型下发器
    
    专门用于下发训练好的模型参数。
    """
    
    def prepare_dispatch_data(self, aggregated_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        准备模型下发数据
        
        Args:
            aggregated_result: 包含聚合模型的结果
            
        Returns:
            Dict[str, Any]: 模型参数字典
        """
        try:
            # 提取模型参数
            if 'model_state_dict' in aggregated_result:
                return aggregated_result['model_state_dict']
            elif 'model_parameters' in aggregated_result:
                return aggregated_result['model_parameters']
            elif 'aggregated_model' in aggregated_result:
                return aggregated_result['aggregated_model']
            else:
                raise DispatchError("聚合结果中未找到模型参数")
                
        except Exception as e:
            raise DispatchError(f"模型数据准备失败: {e}") from e


class GradientDispatcher(BaseDispatcher):
    """
    梯度下发器
    
    专门用于下发梯度信息而非完整模型。
    """
    
    def prepare_dispatch_data(self, aggregated_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        准备梯度下发数据
        
        Args:
            aggregated_result: 包含聚合梯度的结果
            
        Returns:
            Dict[str, Any]: 梯度信息
        """
        try:
            if 'aggregated_gradients' in aggregated_result:
                return aggregated_result['aggregated_gradients']
            else:
                raise DispatchError("聚合结果中未找到梯度信息")
                
        except Exception as e:
            raise DispatchError(f"梯度数据准备失败: {e}") from e


class SelectiveDispatcher(BaseDispatcher):
    """
    选择性下发器
    
    根据客户端特征选择性地下发不同的内容。
    """
    
    def __init__(self, config: DictConfig):
        super().__init__(config)
        
        # 客户端选择策略
        self.selection_strategy = config.get('selection_strategy', 'all')
        self.selection_ratio = config.get('selection_ratio', 1.0)
        self.selection_criteria = config.get('selection_criteria', {})
    
    def select_targets(self, available_clients: List[Any]) -> List[Any]:
        """
        根据策略选择客户端
        
        Args:
            available_clients: 可用客户端列表
            
        Returns:
            List[Any]: 选中的客户端列表
        """
        if self.selection_strategy == 'all':
            return available_clients
        elif self.selection_strategy == 'random':
            import random
            num_selected = int(len(available_clients) * self.selection_ratio)
            return random.sample(available_clients, min(num_selected, len(available_clients)))
        elif self.selection_strategy == 'top_performers':
            # 根据性能选择客户端（需要客户端有性能指标）
            return self._select_top_performers(available_clients)
        else:
            logger.warning(f"未知选择策略: {self.selection_strategy}，使用默认策略")
            return available_clients
    
    def _select_top_performers(self, clients: List[Any]) -> List[Any]:
        """选择表现最好的客户端"""
        # 这里需要根据实际的客户端对象结构来实现
        # 暂时返回所有客户端
        return clients
    
    def prepare_dispatch_data(self, aggregated_result: Dict[str, Any]) -> Dict[str, Any]:
        """准备选择性下发数据"""
        return aggregated_result


class CustomizableDispatcher(BaseDispatcher):
    """
    可定制下发器
    
    支持为不同客户端定制不同的下发内容。
    """
    
    def __init__(self, config: DictConfig):
        super().__init__(config)
        
        # 定制规则
        self.customization_rules = config.get('customization_rules', {})
    
    def customize_for_client(self, dispatch_data: Any, client: Any) -> Any:
        """
        为客户端定制数据
        
        Args:
            dispatch_data: 原始数据
            client: 客户端对象
            
        Returns:
            Any: 定制后的数据
        """
        try:
            # 获取客户端ID
            client_id = getattr(client, 'client_id', str(client))
            
            # 应用定制规则
            if client_id in self.customization_rules:
                rules = self.customization_rules[client_id]
                customized_data = self._apply_customization_rules(dispatch_data, rules)
                logger.debug(f"为客户端 {client_id} 定制数据")
                return customized_data
            
            return dispatch_data
            
        except Exception as e:
            logger.warning(f"客户端数据定制失败: {e}")
            return dispatch_data
    
    def _apply_customization_rules(self, data: Any, rules: Dict[str, Any]) -> Any:
        """应用定制规则"""
        # 实现具体的定制逻辑
        # 例如：部分模型参数、压缩、量化等
        return data
    
    def prepare_dispatch_data(self, aggregated_result: Dict[str, Any]) -> Any:
        """准备可定制的下发数据"""
        return aggregated_result
