# fedcl/data/federated_data_manager.py
"""
联邦数据管理器 - 支持真实和伪联邦场景的数据管理
"""

from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
from loguru import logger

from .dataset_manager import DatasetManager
from ..core.execution_context import ExecutionContext


class FederatedDataManager:
    """
    联邦数据管理器
    
    支持两种模式：
    1. 真实联邦模式：客户端本地加载自己的数据
    2. 伪联邦模式：服务端统一管理和分发数据
    """
    
    def __init__(self, 
                 mode: str = "pseudo_federated",  # "real_federated" or "pseudo_federated"
                 config: Dict[str, Any] = None):
        """
        初始化联邦数据管理器
        
        Args:
            mode: 联邦模式 ("real_federated" 或 "pseudo_federated")
            config: 数据配置
        """
        self.mode = mode
        self.config = config or {}
        self.dataset_manager = DatasetManager()
        
        logger.debug(f"Initialized FederatedDataManager in {mode} mode")
    
    def setup_client_data(self, 
                         client_id: str, 
                         data_path: Optional[Path] = None,
                         task_id: Optional[int] = None) -> DataLoader:
        """
        为客户端设置数据
        
        Args:
            client_id: 客户端ID
            data_path: 数据路径（真实联邦模式）
            task_id: 任务ID（伪联邦模式）
            
        Returns:
            数据加载器
        """
        if self.mode == "real_federated":
            return self._setup_real_federated_data(client_id, data_path)
        else:
            return self._setup_pseudo_federated_data(client_id, task_id)
    
    def _setup_real_federated_data(self, 
                                  client_id: str, 
                                  data_path: Path) -> DataLoader:
        """真实联邦模式：客户端本地加载数据"""
        if not data_path or not data_path.exists():
            raise ValueError(f"Data path not found for client {client_id}: {data_path}")
        
        logger.debug(f"Loading local data for client {client_id} from {data_path}")
        
        # 客户端本地加载数据
        dataset = self.dataset_manager.load_local_dataset(data_path, client_id)
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.get("batch_size", 32),
            shuffle=True,
            num_workers=self.config.get("num_workers", 0)
        )
        
        return dataloader
    
    def _setup_pseudo_federated_data(self, 
                                   client_id: str, 
                                   task_id: int) -> DataLoader:
        """伪联邦模式：从服务端获取分配的数据"""
        logger.debug(f"Setting up pseudo federated data for client {client_id}, task {task_id}")
        
        # 从服务端获取该客户端的数据分片
        client_data = self.dataset_manager.get_client_data_split(client_id, task_id)
        
        if client_data is None:
            raise ValueError(f"No data available for client {client_id}, task {task_id}")
        
        dataloader = DataLoader(
            client_data,
            batch_size=self.config.get("batch_size", 32),
            shuffle=True,
            num_workers=self.config.get("num_workers", 0)
        )
        
        return dataloader