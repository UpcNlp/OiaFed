# fedcl/engine/dataloader_manager.py
"""
DataLoader管理器

为TrainingEngine提供DataLoader管理功能，集成到多Learner架构中。
"""

from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import yaml
from loguru import logger

from ..data.dataloader_factory import (
    DataLoaderFactory, dataloader_factory, 
    DataLoaderCreationError, validate_dataloader_config
)
from ..data.dataloader import DataLoader
from ..config.config_manager import DictConfig


class DataLoaderManagerError(Exception):
    """DataLoader管理器错误"""
    pass


class DataLoaderManager:
    """
    DataLoader管理器
    
    为TrainingEngine提供统一的DataLoader管理接口，支持：
    - 从配置文件批量创建DataLoader
    - DataLoader生命周期管理
    - 内存和性能监控
    - 错误处理和恢复
    """
    
    def __init__(self, factory: Optional[DataLoaderFactory] = None):
        """
        初始化DataLoader管理器
        
        Args:
            factory: DataLoader工厂实例（可选）
        """
        self.factory = factory or dataloader_factory
        self.dataloaders: Dict[str, DataLoader] = {}
        self.configs: Dict[str, Dict[str, Any]] = {}
        self.stats = {
            'total_created': 0,
            'active_dataloaders': 0,
            'total_memory_mb': 0.0,
            'creation_errors': 0
        }
        
    def load_dataloaders_from_config(self, config: Union[str, Dict[str, Any]]) -> Dict[str, DataLoader]:
        """
        从配置加载多个DataLoader
        
        Args:
            config: 配置文件路径或配置字典
            
        Returns:
            DataLoader字典
            
        Raises:
            DataLoaderManagerError: 加载失败时抛出
        """
        try:
            # 解析配置
            if isinstance(config, str):
                config_path = Path(config)
                if not config_path.exists():
                    raise DataLoaderManagerError(f"Config file not found: {config_path}")
                
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_dict = yaml.safe_load(f)
            else:
                config_dict = config
            
            # 获取dataloaders配置
            dataloader_configs = config_dict.get('dataloaders', {})
            if not dataloader_configs:
                logger.warning("No dataloaders configuration found")
                return {}
            
            # 批量创建DataLoader
            created_dataloaders = {}
            for name, loader_config in dataloader_configs.items():
                try:
                    # 验证配置
                    if not validate_dataloader_config(loader_config):
                        logger.error(f"Invalid configuration for DataLoader '{name}'")
                        self.stats['creation_errors'] += 1
                        continue
                    
                    # 创建DataLoader
                    dataloader = self.factory.create_dataloader(name, loader_config)
                    created_dataloaders[name] = dataloader
                    
                    # 保存配置和实例
                    self.configs[name] = loader_config
                    self.dataloaders[name] = dataloader
                    
                    # 更新统计
                    self.stats['total_created'] += 1
                    self.stats['active_dataloaders'] += 1
                    
                    logger.debug(f"Successfully created DataLoader: {name}")
                    
                except Exception as e:
                    logger.error(f"Failed to create DataLoader '{name}': {e}")
                    self.stats['creation_errors'] += 1
                    continue
            
            # 更新内存统计
            self._update_memory_stats()
            
            logger.debug(f"Created {len(created_dataloaders)}/{len(dataloader_configs)} DataLoaders")
            return created_dataloaders
            
        except Exception as e:
            logger.error(f"Failed to load DataLoaders from config: {e}")
            raise DataLoaderManagerError(f"Failed to load DataLoaders: {e}")
    
    def get_dataloader(self, name: str) -> Optional[DataLoader]:
        """
        获取指定名称的DataLoader
        
        Args:
            name: DataLoader名称
            
        Returns:
            DataLoader实例或None
        """
        return self.dataloaders.get(name)
    
    def get_dataloader_for_learner(self, learner_id: str, 
                                  learner_config: Dict[str, Any]) -> Optional[DataLoader]:
        """
        为指定learner获取对应的DataLoader
        
        Args:
            learner_id: Learner ID
            learner_config: Learner配置
            
        Returns:
            DataLoader实例或None
        """
        dataloader_name = learner_config.get('dataloader')
        if not dataloader_name:
            logger.warning(f"No dataloader specified for learner {learner_id}")
            return None
        
        dataloader = self.get_dataloader(dataloader_name)
        if not dataloader:
            logger.error(f"DataLoader '{dataloader_name}' not found for learner {learner_id}")
            return None
        
        return dataloader
    
    def create_dataloader(self, name: str, config: Dict[str, Any]) -> DataLoader:
        """
        创建单个DataLoader
        
        Args:
            name: DataLoader名称
            config: DataLoader配置
            
        Returns:
            DataLoader实例
        """
        try:
            # 验证配置
            if not validate_dataloader_config(config):
                raise DataLoaderManagerError(f"Invalid configuration for DataLoader '{name}'")
            
            # 创建DataLoader
            dataloader = self.factory.create_dataloader(name, config)
            
            # 保存配置和实例
            self.configs[name] = config
            self.dataloaders[name] = dataloader
            
            # 更新统计
            self.stats['total_created'] += 1
            self.stats['active_dataloaders'] += 1
            self._update_memory_stats()
            
            logger.debug(f"Created DataLoader: {name}")
            return dataloader
            
        except Exception as e:
            logger.error(f"Failed to create DataLoader '{name}': {e}")
            self.stats['creation_errors'] += 1
            raise DataLoaderManagerError(f"Failed to create DataLoader '{name}': {e}")
    
    def remove_dataloader(self, name: str) -> bool:
        """
        移除DataLoader
        
        Args:
            name: DataLoader名称
            
        Returns:
            是否成功移除
        """
        if name in self.dataloaders:
            del self.dataloaders[name]
            if name in self.configs:
                del self.configs[name]
            
            self.stats['active_dataloaders'] -= 1
            self._update_memory_stats()
            
            logger.debug(f"Removed DataLoader: {name}")
            return True
        
        return False
    
    def list_dataloaders(self) -> List[str]:
        """列出所有DataLoader名称"""
        return list(self.dataloaders.keys())
    
    def get_dataloader_config(self, name: str) -> Optional[Dict[str, Any]]:
        """获取DataLoader配置"""
        return self.configs.get(name)
    
    def validate_learner_dataloader_mapping(self, learners_config: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        验证learner与dataloader的映射关系
        
        Args:
            learners_config: Learner配置字典
            
        Returns:
            验证结果字典
        """
        validation_result = {
            'valid_mappings': [],
            'invalid_mappings': [],
            'missing_dataloaders': []
        }
        
        for learner_id, learner_config in learners_config.items():
            dataloader_name = learner_config.get('dataloader')
            
            if not dataloader_name:
                validation_result['invalid_mappings'].append(
                    f"Learner '{learner_id}' has no dataloader specified"
                )
                continue
            
            if dataloader_name not in self.dataloaders:
                validation_result['missing_dataloaders'].append(
                    f"DataLoader '{dataloader_name}' for learner '{learner_id}' not found"
                )
                continue
            
            validation_result['valid_mappings'].append(
                f"Learner '{learner_id}' -> DataLoader '{dataloader_name}'"
            )
        
        return validation_result
    
    def get_memory_usage_summary(self) -> Dict[str, Any]:
        """获取内存使用摘要"""
        total_memory = 0.0
        dataloader_memory = {}
        
        for name, dataloader in self.dataloaders.items():
            try:
                memory_info = dataloader.get_memory_info()
                memory_mb = memory_info.get('estimated_total_memory_mb', 0.0)
                total_memory += memory_mb
                dataloader_memory[name] = memory_mb
            except Exception as e:
                logger.warning(f"Failed to get memory info for DataLoader '{name}': {e}")
                dataloader_memory[name] = 0.0
        
        return {
            'total_memory_mb': total_memory,
            'dataloader_memory': dataloader_memory,
            'average_memory_mb': total_memory / len(self.dataloaders) if self.dataloaders else 0.0
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        performance_info = {}
        
        for name, dataloader in self.dataloaders.items():
            try:
                config = dataloader.get_config()
                performance_info[name] = {
                    'dataset_size': config.get('dataset_size', 0),
                    'batch_size': config.get('batch_size', 0),
                    'num_batches': config.get('num_batches', 0),
                    'num_workers': config.get('num_workers', 0),
                    'memory_mb': dataloader.get_memory_info().get('estimated_total_memory_mb', 0.0)
                }
            except Exception as e:
                logger.warning(f"Failed to get performance info for DataLoader '{name}': {e}")
        
        return performance_info
    
    def get_stats(self) -> Dict[str, Any]:
        """获取管理器统计信息"""
        stats = self.stats.copy()
        stats.update({
            'factory_stats': self.factory.get_stats(),
            'memory_summary': self.get_memory_usage_summary(),
            'available_creators': self.factory.list_creators()
        })
        return stats
    
    def cleanup(self) -> None:
        """清理所有DataLoader"""
        self.dataloaders.clear()
        self.configs.clear()
        self.factory.clear_cache()
        
        self.stats['active_dataloaders'] = 0
        self.stats['total_memory_mb'] = 0.0
        
        logger.debug("DataLoader manager cleaned up")
    
    def _update_memory_stats(self) -> None:
        """更新内存统计"""
        try:
            memory_summary = self.get_memory_usage_summary()
            self.stats['total_memory_mb'] = memory_summary['total_memory_mb']
        except Exception as e:
            logger.warning(f"Failed to update memory stats: {e}")
    
    def export_config(self, output_path: str) -> None:
        """
        导出当前DataLoader配置
        
        Args:
            output_path: 输出文件路径
        """
        try:
            config_dict = {
                'dataloaders': self.configs.copy(),
                'stats': self.get_stats(),
                'metadata': {
                    'total_dataloaders': len(self.dataloaders),
                    'created_by': 'DataLoaderManager'
                }
            }
            
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
            
            logger.debug(f"Exported DataLoader configuration to: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to export configuration: {e}")
            raise DataLoaderManagerError(f"Failed to export configuration: {e}")


def create_dataloader_manager_from_config(config_path: str) -> DataLoaderManager:
    """
    从配置文件创建DataLoader管理器
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        DataLoader管理器实例
    """
    manager = DataLoaderManager()
    manager.load_dataloaders_from_config(config_path)
    return manager
