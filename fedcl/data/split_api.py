# fedcl/data/split_api.py
"""
数据分割API模块

基于现有DatasetManager和SplitStrategy实现用户友好的数据分割接口。
提供简化的API用于数据分割和客户端配置生成。
"""

import os
import pickle
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from loguru import logger

from .split_strategy import SplitStrategy
from ..config.config_manager import DictConfig
from ..exceptions import FedCLError


class DataSplitError(FedCLError):
    """数据分割错误"""
    pass


class DataSplitAPI:
    """
    用户友好的数据分割API
    
    基于现有DatasetManager和SplitStrategy实现，提供：
    1. 简化的数据分割接口
    2. 自动配置文件生成
    3. 分割结果可视化和验证
    """
    
    def __init__(self):
        """初始化数据分割API"""
        # 不再依赖 DatasetManager，直接使用分割策略
        pass
        
    def execute_split(self, split_config: DictConfig) -> Dict[str, Any]:
        """
        执行数据分割操作
        
        Args:
            split_config: 数据分割配置
            
        Returns:
            Dict[str, Any]: 分割结果信息
            
        Raises:
            DataSplitError: 数据分割失败
        """
        try:
            logger.debug(f"开始执行数据分割: {split_config.get('name', 'unnamed')}")
            
            # 解析分割配置
            split_params = self._parse_split_config(split_config)
            
            # 执行数据分割
            split_results = self._perform_split(split_params)
            
            # 保存分割结果
            output_info = self._save_split_results(split_results, split_params)
            
            # 生成客户端配置文件（如果需要）
            if split_params.get('generate_configs', True):
                config_files = self._generate_client_configs(split_results, split_params)
                output_info['config_files'] = config_files
            
            # 生成可视化（如果需要）
            if split_params.get('visualization', {}).get('enable', False):
                visualization_info = self._generate_visualization(split_results, split_params)
                output_info['visualization'] = visualization_info
            
            logger.debug(f"数据分割完成: {output_info}")
            return output_info
            
        except Exception as e:
            logger.error(f"数据分割失败: {e}")
            raise DataSplitError(f"Failed to execute data split: {e}") from e
    
    def split_and_generate_configs(self, 
                                 dataset_path: str,
                                 num_clients: int,
                                 strategy: str = "iid",
                                 output_dir: str = "./client_data") -> List[str]:
        """
        分割数据并生成客户端配置文件
        
        Args:
            dataset_path: 数据集路径
            num_clients: 客户端数量
            strategy: 分割策略 ('iid', 'noniid', 'dirichlet')
            output_dir: 输出目录
            
        Returns:
            List[str]: 生成的客户端配置文件路径列表
        """
        # 构建分割配置
        split_config = DictConfig({
            'data_split': {
                'strategy': {
                    'type': 'federated_split',
                    'method': strategy
                },
                'dataset': {
                    'path': dataset_path
                },
                '客户端': {
                    'num_客户端': num_clients
                },
                'output': {
                    'save_path': output_dir,
                    'format': 'pickle'
                },
                'generate_configs': True
            }
        })
        
        # 执行分割
        result = self.execute_split(split_config)
        return result.get('config_files', [])
    
    def _parse_split_config(self, split_config: DictConfig) -> Dict[str, Any]:
        """
        解析分割配置
        
        Args:
            split_config: 原始配置
            
        Returns:
            Dict[str, Any]: 解析后的参数
        """
        # 获取数据分割部分
        split_section = split_config.select('data_split') or split_config
        
        # 从数据集路径推断数据集名称
        dataset_path = split_section.get('dataset_path', '')
        dataset_name = split_section.get('dataset', {}).get('name', '')
        
        # 如果没有明确指定数据集名称，从路径推断
        if not dataset_name:
            if 'MNIST' in dataset_path.upper() or 'mnist' in dataset_path.lower():
                dataset_name = 'mnist'
            elif 'CIFAR' in dataset_path.upper() or 'cifar' in dataset_path.lower():
                dataset_name = 'cifar10'
            else:
                dataset_name = 'custom'
        
        return {
            # 分割策略
            'strategy_type': split_section.get('strategy', {}).get('type', 'federated_split'),
            'strategy_method': split_section.get('split_method', 'iid'),  # 更新这里
            
            # 数据集配置
            'dataset_name': dataset_name,
            'dataset_path': dataset_path,
            'dataset_root': split_section.get('dataset', {}).get('root', './data'),
            'download': split_section.get('dataset', {}).get('download', True),
            
            # 客户端配置
            'num_客户端': split_section.get('num_客户端', 10),
            'samples_per_client': split_section.get('客户端', {}).get('samples_per_client', None),
            
            # Non-IID配置
            'alpha': split_section.get('non_iid_config', {}).get('alpha', 0.5),  # 更新这里
            'min_samples': split_section.get('non_iid_config', {}).get('min_samples_per_client', 50),  # 更新这里
            'max_samples': split_section.get('non_iid', {}).get('max_samples', 1000),
            
            # 输出配置
            'output_path': split_section.get('output_dir', './data/splits'),  # 更新这里
            'output_format': split_section.get('output', {}).get('format', 'pickle'),
            'save_metadata': split_section.get('output', {}).get('metadata', True),
            
            # 验证配置
            'validation_enable': split_section.get('validation', {}).get('enable', True),
            'validation_ratio': split_section.get('validation', {}).get('split_ratio', 0.1),
            
            # 可视化配置
            'visualization': split_section.get('visualization', {}),
            
            # 配置生成
            'generate_configs': split_section.get('generation', {}).get('auto_generate_configs', True)  # 更新这里
        }
    
    def _perform_split(self, split_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行实际的数据分割
        
        Args:
            split_params: 分割参数
            
        Returns:
            Dict[str, Any]: 分割结果
        """
        try:
            # 1. 加载数据集
            dataset = self._load_dataset(split_params)
            
            # 2. 获取分割策略
            strategy = self._get_split_strategy(split_params)
            
            # 3. 执行数据分割
            split_datasets = strategy.split_data(dataset, split_params['num_客户端'])
            
            # 4. 构建分割结果
            split_result = {
                'client_data': [],
                'split_datasets': split_datasets,
                'metadata': {
                    'num_客户端': len(split_datasets),
                    'strategy': split_params['strategy_method'],
                    'total_samples': len(dataset),
                    'dataset_name': split_params['dataset_name']
                }
            }
            
            # 5. 保存客户端数据索引
            for client_id, client_dataset in split_datasets.items():
                # 保存客户端数据索引
                indices_file = self._save_client_indices(
                    client_id, 
                    client_dataset, 
                    split_params['output_path']
                )
                
                client_data_info = {
                    'client_id': client_id,
                    'data_path': split_params['output_path'],
                    'indices_file': indices_file,
                    'num_samples': len(client_dataset),
                    'classes': list(client_dataset.get_classes()) if hasattr(client_dataset, 'get_classes') else []
                }
                split_result['client_data'].append(client_data_info)
            
            logger.debug(f"数据分割完成，生成 {len(split_result['client_data'])} 个客户端数据")
            return split_result
            
        except Exception as e:
            raise DataSplitError(f"Failed to perform data split: {e}") from e
    
    def _get_split_strategy(self, split_params: Dict[str, Any]) -> SplitStrategy:
        """
        获取分割策略实例
        
        Args:
            split_params: 分割参数
            
        Returns:
            SplitStrategy: 分割策略实例
        """
        strategy_method = split_params['strategy_method'].lower()
        
        try:
            if strategy_method == 'iid':
                from .split_strategy import IIDSplitStrategy
                strategy_config = DictConfig({
                    'random_seed': 42,
                    'stratified': True,
                    'shuffle': True,
                    'min_samples_per_client': split_params.get('min_samples', 1)
                })
                return IIDSplitStrategy(strategy_config)
                
            elif strategy_method in ['noniid', 'non_iid']:
                from .split_strategy import NonIIDSplitStrategy
                strategy_config = DictConfig({
                    'alpha': split_params.get('alpha', 0.5),
                    'method': 'dirichlet',
                    'min_samples_per_client': split_params.get('min_samples', 50),
                    'random_seed': 42
                })
                return NonIIDSplitStrategy(strategy_config)
                
            else:
                logger.warning(f"未知分割策略: {strategy_method}，使用默认IID策略")
                from .split_strategy import IIDSplitStrategy
                strategy_config = DictConfig({
                    'random_seed': 42,
                    'stratified': True,
                    'shuffle': True,
                    'min_samples_per_client': 1
                })
                return IIDSplitStrategy(strategy_config)
                
        except ImportError as e:
            logger.error(f"分割策略导入失败: {e}")
            raise DataSplitError(f"Failed to import split strategy: {e}") from e
    
    def _load_dataset(self, split_params: Dict[str, Any]):
        """
        加载数据集
        
        Args:
            split_params: 分割参数
            
        Returns:
            Dataset: 加载的数据集
        """
        try:
            dataset_name = split_params['dataset_name'].lower()
            dataset_path = split_params['dataset_path'] or split_params['dataset_root']
            
            if dataset_name == 'mnist':
                # 直接加载 MNIST 数据集，不依赖 DatasetManager
                logger.debug(f"加载MNIST数据集: {dataset_path}")
                
                from torchvision.datasets import MNIST
                from .dataset import Dataset
                
                # 加载训练数据集 - 不应用transform，让Dataset类处理
                torch_dataset = MNIST(
                    root=dataset_path,
                    train=True,
                    download=split_params.get('download', True),
                    transform=None  # 不在这里应用transform
                )
                
                # 转换为我们的Dataset格式
                dataset = Dataset(
                    name='mnist_train',
                    data=torch_dataset.data,
                    targets=torch_dataset.targets,
                    transform=None  # 暂时不使用transform，避免分割时的问题
                )
                
            elif dataset_name == 'cifar10':
                # 加载 CIFAR-10 数据集
                logger.debug(f"加载CIFAR-10数据集: {dataset_path}")
                
                from torchvision.datasets import CIFAR10
                from .dataset import Dataset
                
                torch_dataset = CIFAR10(
                    root=dataset_path,
                    train=True,
                    download=split_params.get('download', True),
                    transform=None  # 不在这里应用transform
                )
                
                dataset = Dataset(
                    name='cifar10_train',
                    data=torch_dataset.data,
                    targets=torch_dataset.targets,
                    transform=None  # 暂时不使用transform
                )
                
            else:
                # 自定义数据集路径
                logger.warning("自定义数据集加载需要用户自己实现")
                raise DataSplitError("Custom dataset loading not implemented")
            
            logger.debug(f"成功加载数据集: {dataset_name}, 样本数: {len(dataset)}")
            return dataset
            
        except Exception as e:
            logger.error(f"数据集加载失败: {e}")
            raise DataSplitError(f"Failed to load dataset: {e}") from e
    
    def _save_client_indices(self, client_id: str, client_dataset, output_path: str) -> str:
        """
        保存客户端数据索引
        
        Args:
            client_id: 客户端ID
            client_dataset: 客户端数据集
            output_path: 输出路径
            
        Returns:
            str: 索引文件路径
        """
        try:
            output_dir = Path(output_path)
            indices_dir = output_dir / "indices"
            indices_dir.mkdir(parents=True, exist_ok=True)
            
            # 获取数据集索引
            if hasattr(client_dataset, 'indices'):
                indices = client_dataset.indices
            else:
                # 如果没有索引属性，创建顺序索引
                indices = list(range(len(client_dataset)))
            
            # 保存索引文件
            indices_file = indices_dir / f"{client_id}_indices.json"
            with open(indices_file, 'w', encoding='utf-8') as f:
                json.dump(indices, f, indent=2)
            
            logger.debug(f"保存客户端索引: {indices_file}")
            return str(indices_file)
            
        except Exception as e:
            logger.error(f"保存客户端索引失败: {e}")
            raise DataSplitError(f"Failed to save client indices: {e}") from e
    
    def _save_split_results(self, split_results: Dict[str, Any], split_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        保存分割结果
        
        Args:
            split_results: 分割结果
            split_params: 分割参数
            
        Returns:
            Dict[str, Any]: 保存信息
        """
        output_path = Path(split_params['output_path'])
        output_path.mkdir(parents=True, exist_ok=True)
        
        output_info = {
            'output_path': str(output_path),
            'saved_files': []
        }
        
        # 保存客户端数据信息
        for client_data in split_results['client_data']:
            client_dir = Path(client_data['data_path'])
            client_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存数据信息文件
            info_file = client_dir / 'data_info.json'
            with open(info_file, 'w', encoding='utf-8') as f:
                json.dump(client_data, f, indent=2, ensure_ascii=False)
            output_info['saved_files'].append(str(info_file))
        
        # 保存元数据
        if split_params.get('save_metadata', True):
            metadata_file = output_path / 'split_metadata.json'
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(split_results['metadata'], f, indent=2, ensure_ascii=False)
            output_info['saved_files'].append(str(metadata_file))
        
        return output_info
    
    def _generate_client_configs(self, split_results: Dict[str, Any], split_params: Dict[str, Any]) -> List[str]:
        """
        生成客户端配置文件
        
        Args:
            split_results: 分割结果
            split_params: 分割参数
            
        Returns:
            List[str]: 生成的配置文件路径列表
        """
        config_files = []
        output_path = Path(split_params['output_path'])
        
        for client_data in split_results['client_data']:
            client_id = client_data['client_id']
            
            # 生成客户端配置
            client_config = {
                'client': {
                    'id': client_id,
                    'description': f"自动生成的客户端配置 - {client_id}"
                },
                'server': {
                    'host': 'localhost',
                    'port': 8080,
                    'registration_timeout': 30,
                    'heartbeat_interval': 10
                },
                'model': {
                    'architecture': 'CNN',
                    'config': {
                        'conv_layers': [32, 64],
                        'fc_layers': [128, 10],
                        'dropout': 0.5,
                        'input_channels': 1,
                        'input_size': 28
                    }
                },
                'learner': {
                    'type': 'federated_learner',
                    'config': {
                        'optimizer': 'SGD',
                        'learning_rate': 0.01,
                        'momentum': 0.9
                    }
                },
                'dataset': {
                    'source': split_params.get('dataset_name', 'mnist'),
                    'path': split_params.get('dataset_path', 'data/MNIST'),
                    'split': 'train',
                    'indices_file': os.path.basename(client_data['indices_file']),
                    'preprocessing': {
                        'normalize': True,
                        'augmentation': False
                    }
                },
                'training': {
                    'batch_size': 32,
                    'epochs_per_round': 3,
                    'local_learning_rate': 0.01
                },
                'aggregation': {
                    'upload_strategy': 'full_model',
                    'upload_params': []
                },
                'state_management': {
                    'enabled': True
                }
            }
            
            # 保存配置文件
            config_file = output_path / f'{client_id}_config.yaml'
            import yaml
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(client_config, f, default_flow_style=False, allow_unicode=True, indent=2)
            
            config_files.append(str(config_file))
            logger.debug(f"生成客户端配置: {config_file}")
        
        return config_files
    
    def _generate_visualization(self, split_results: Dict[str, Any], split_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成数据分割可视化
        
        Args:
            split_results: 分割结果
            split_params: 分割参数
            
        Returns:
            Dict[str, Any]: 可视化信息
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            output_path = Path(split_params['output_path'])
            viz_info = {'plots': []}
            
            # 生成客户端数据分布图
            if split_params['visualization'].get('show_distribution', True):
                plt.figure(figsize=(12, 6))
                
                client_ids = [data['client_id'] for data in split_results['client_data']]
                sample_counts = [data['num_samples'] for data in split_results['client_data']]
                
                plt.bar(client_ids, sample_counts)
                plt.xlabel('客户端ID')
                plt.ylabel('样本数量')
                plt.title('客户端数据分布')
                plt.xticks(client_ids)
                
                plot_file = output_path / 'client_distribution.png'
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                plt.close()
                
                viz_info['plots'].append(str(plot_file))
                logger.debug(f"生成数据分布图: {plot_file}")
            
            return viz_info
            
        except ImportError:
            logger.warning("matplotlib未安装，跳过可视化生成")
            return {'plots': []}
        except Exception as e:
            logger.warning(f"可视化生成失败: {e}")
            return {'plots': []}
