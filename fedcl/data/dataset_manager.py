# fedcl/data/dataset_manager.py
"""
Dataset Manager implementation for FedCL framework.

This module provides comprehensive dataset management capabilities including:
- Multi-format dataset loading (torchvision, custom datasets)
- Intelligent caching with memory management
- Task sequence creation for continual learning
- Client data distribution for federated learning
- Dataset validation and integrity checks
- Concurrent access safety
"""

from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from pathlib import Path
import threading
import hashlib
import pickle
import time
import weakref
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import torch
import torchvision
from torch.utils.data import DataLoader as TorchDataLoader
from loguru import logger

from ..config.config_manager import DictConfig
from ..config.schema_validator import ValidationResult, ValidationError, ValidationWarning
from ..config.exceptions import ConfigValidationError
from .dataset import Dataset, DatasetError, DatasetValidationError
from .dataloader import DataLoader
from .task import Task, TaskType
from .task_generator import TaskGenerator, TaskGeneratorError
from ..exceptions import FedCLError


class DatasetManagerError(FedCLError):
    """Dataset manager related errors"""
    pass


class DatasetNotFoundError(DatasetManagerError):
    """Dataset not found error"""
    pass


class DatasetCacheError(DatasetManagerError):
    """Dataset cache related errors"""
    pass


class DatasetValidationManager:
    """Dataset validation manager for integrity checks"""
    
    def __init__(self):
        self.validators: Dict[str, Callable] = {
            'integrity': self._validate_integrity,
            'format': self._validate_format,
            'size': self._validate_size,
            'labels': self._validate_labels,
            'distribution': self._validate_distribution
        }
    
    def validate_dataset(self, dataset: Dataset) -> ValidationResult:
        """
        Comprehensive dataset validation
        
        Args:
            dataset: Dataset to validate
            
        Returns:
            ValidationResult: Validation results with errors and warnings
        """
        result = ValidationResult()
        
        for validator_name, validator_func in self.validators.items():
            try:
                validator_result = validator_func(dataset)
                result = result.merge(validator_result)
            except Exception as e:
                result.add_error(
                    f"validation.{validator_name}",
                    f"Validation failed: {str(e)}",
                    "VALIDATION_ERROR"
                )
        
        return result
    
    def _validate_integrity(self, dataset: Dataset) -> ValidationResult:
        """Validate dataset integrity"""
        result = ValidationResult()
        
        if len(dataset.data) != len(dataset.targets):
            result.add_error(
                "integrity.size_mismatch",
                f"Data length ({len(dataset.data)}) doesn't match targets length ({len(dataset.targets)})",
                "SIZE_MISMATCH"
            )
        
        if len(dataset) == 0:
            result.add_error(
                "integrity.empty_dataset",
                "Dataset is empty",
                "EMPTY_DATASET"
            )
        
        return result
    
    def _validate_format(self, dataset: Dataset) -> ValidationResult:
        """Validate dataset format"""
        result = ValidationResult()
        
        if not isinstance(dataset.data, torch.Tensor):
            result.add_error(
                "format.data_type",
                f"Data should be torch.Tensor, got {type(dataset.data)}",
                "INVALID_DATA_TYPE"
            )
        
        if not isinstance(dataset.targets, torch.Tensor):
            result.add_error(
                "format.targets_type",
                f"Targets should be torch.Tensor, got {type(dataset.targets)}",
                "INVALID_TARGETS_TYPE"
            )
        
        return result
    
    def _validate_size(self, dataset: Dataset) -> ValidationResult:
        """Validate dataset size constraints"""
        result = ValidationResult()
        
        if len(dataset) < 10:
            result.add_warning(
                "size.too_small",
                f"Dataset has only {len(dataset)} samples, which might be too small",
                "SMALL_DATASET"
            )
        
        return result
    
    def _validate_labels(self, dataset: Dataset) -> ValidationResult:
        """Validate dataset labels"""
        result = ValidationResult()
        
        unique_labels = torch.unique(dataset.targets)
        if len(unique_labels) < 2:
            result.add_warning(
                "labels.single_class",
                f"Dataset has only {len(unique_labels)} unique class(es)",
                "SINGLE_CLASS"
            )
        
        # Check for negative labels
        if torch.min(dataset.targets) < 0:
            result.add_error(
                "labels.negative_labels",
                "Dataset contains negative labels",
                "NEGATIVE_LABELS"
            )
        
        return result
    
    def _validate_distribution(self, dataset: Dataset) -> ValidationResult:
        """Validate dataset class distribution"""
        result = ValidationResult()
        
        class_counts = torch.bincount(dataset.targets)
        if len(class_counts) > 0:
            min_count = torch.min(class_counts[class_counts > 0])
            max_count = torch.max(class_counts)
            
            if max_count > min_count * 10:  # Imbalance ratio > 10
                result.add_warning(
                    "distribution.imbalanced",
                    f"Severe class imbalance detected (ratio: {max_count/min_count:.2f})",
                    "CLASS_IMBALANCE"
                )
        
        return result


class DatasetCache:
    """Intelligent LRU cache for datasets with memory management"""
    
    def __init__(self, max_size: str = "2GB", strategy: str = "LRU"):
        self.max_size_bytes = self._parse_size(max_size)
        self.strategy = strategy
        self.cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.current_size = 0
        self._lock = threading.RLock()
        
        logger.debug(f"Dataset cache initialized with max size: {max_size}")
    
    def _parse_size(self, size_str: str) -> int:
        """Parse size string to bytes"""
        units = {'B': 1, 'KB': 1024, 'MB': 1024**2, 'GB': 1024**3}
        size_str = size_str.upper().strip()
        
        # Sort units by length (longest first) to avoid matching issues
        for unit in sorted(units.keys(), key=len, reverse=True):
            if size_str.endswith(unit):
                number_part = size_str[:-len(unit)].strip()
                return int(float(number_part) * units[unit])
        
        # Default to bytes if no unit specified
        return int(size_str)
    
    def get(self, key: str) -> Optional[Dataset]:
        """Get dataset from cache"""
        with self._lock:
            if key not in self.cache:
                return None
            
            # Move to end (most recently used)
            entry = self.cache.pop(key)
            self.cache[key] = entry
            
            # Update access time
            entry['last_accessed'] = time.time()
            entry['access_count'] += 1
            
            logger.debug(f"Cache hit for dataset: {key}")
            return entry['dataset']
    
    def put(self, key: str, dataset: Dataset, metadata: Dict[str, Any] = None) -> None:
        """Put dataset in cache"""
        with self._lock:
            # Calculate dataset size
            dataset_size = self._calculate_dataset_size(dataset)
            
            # Check if dataset is too large for cache
            if dataset_size > self.max_size_bytes:
                logger.warning(f"Dataset {key} too large for cache ({dataset_size} bytes)")
                return
            
            # Make room if necessary
            self._make_room(dataset_size)
            
            # Add to cache
            entry = {
                'dataset': dataset,
                'size': dataset_size,
                'created': time.time(),
                'last_accessed': time.time(),
                'access_count': 1,
                'metadata': metadata or {}
            }
            
            self.cache[key] = entry
            self.current_size += dataset_size
            
            logger.debug(f"Dataset {key} cached ({dataset_size} bytes)")
    
    def _calculate_dataset_size(self, dataset: Dataset) -> int:
        """Calculate approximate memory size of dataset"""
        try:
            data_size = dataset.data.element_size() * dataset.data.nelement()
            targets_size = dataset.targets.element_size() * dataset.targets.nelement()
            return data_size + targets_size
        except Exception as e:
            logger.warning(f"Could not calculate dataset size: {e}")
            return 0
    
    def _make_room(self, needed_size: int) -> None:
        """Make room in cache using LRU strategy"""
        while self.current_size + needed_size > self.max_size_bytes and self.cache:
            if self.strategy == "LRU":
                # Remove least recently used
                key, entry = self.cache.popitem(last=False)
            else:
                # Default to LRU
                key, entry = self.cache.popitem(last=False)
            
            self.current_size -= entry['size']
            logger.debug(f"Evicted dataset {key} from cache")
    
    def clear(self, pattern: Optional[str] = None) -> None:
        """Clear cache completely or by pattern"""
        with self._lock:
            if pattern is None:
                self.cache.clear()
                self.current_size = 0
                logger.debug("Cache cleared completely")
            else:
                keys_to_remove = [key for key in self.cache.keys() if pattern in key]
                for key in keys_to_remove:
                    entry = self.cache.pop(key)
                    self.current_size -= entry['size']
                logger.debug(f"Cleared {len(keys_to_remove)} cache entries matching pattern: {pattern}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_accesses = sum(entry['access_count'] for entry in self.cache.values())
            return {
                'size': len(self.cache),
                'memory_usage': self.current_size,
                'memory_usage_mb': self.current_size / (1024**2),
                'max_size_mb': self.max_size_bytes / (1024**2),
                'utilization': self.current_size / self.max_size_bytes,
                'total_accesses': total_accesses
            }


class DatasetManager:
    """
    Comprehensive dataset manager for federated continual learning.
    
    Provides dataset loading, registration, caching, validation, and task sequence
    creation with support for multiple dataset formats and intelligent memory management.
    """
    
    def __init__(self, config: DictConfig, task_generator: TaskGenerator):
        """
        Initialize DatasetManager.
        
        Args:
            config: Configuration containing dataset management parameters
            task_generator: Task generator for creating task sequences
            
        Raises:
            ConfigValidationError: If configuration is invalid
            DatasetManagerError: If initialization fails
        """
        self.config = config
        self.task_generator = task_generator
        
        # Extract dataset configuration
        self.dataset_config = config.get('datasets', {})
        self.cache_config = config.get('cache', {})
        
        # Initialize components
        self.validator = DatasetValidationManager()
        self._init_cache()
        
        # Dataset registry
        self.datasets: Dict[str, Dataset] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
        
        # Dataset loaders
        self.loaders: Dict[str, Callable] = {
            'torchvision': self._load_torchvision_dataset,
            'custom': self._load_custom_dataset,
            'file': self._load_file_dataset
        }
        
        # Thread safety
        self._registry_lock = threading.RLock()
        
        # Statistics
        self.stats = {
            'datasets_loaded': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'validation_errors': 0
        }
        
        logger.debug("DatasetManager initialized successfully")
    
    def _init_cache(self) -> None:
        """Initialize dataset cache"""
        if self.cache_config.get('enable', True):
            max_size = self.cache_config.get('max_size', '2GB')
            strategy = self.cache_config.get('strategy', 'LRU')
            self.cache = DatasetCache(max_size, strategy)
        else:
            self.cache = None
            logger.debug("Dataset caching disabled")
    
    def load_dataset(self, name: str, config: DictConfig) -> Dataset:
        """
        Load dataset by name and configuration.
        
        Args:
            name: Dataset name
            config: Dataset configuration
            
        Returns:
            Dataset: Loaded dataset
            
        Raises:
            DatasetNotFoundError: If dataset cannot be found or loaded
            DatasetValidationError: If dataset validation fails
        """
        logger.debug(f"Loading dataset: {name}")
        
        # Check cache first
        if self.cache:
            cached_dataset = self.cache.get(name)
            if cached_dataset is not None:
                self.stats['cache_hits'] += 1
                logger.debug(f"Dataset {name} loaded from cache")
                return cached_dataset
            else:
                self.stats['cache_misses'] += 1
        
        try:
            # Determine dataset type and load
            dataset_type = config.get('type', 'torchvision')
            if dataset_type not in self.loaders:
                raise DatasetNotFoundError(f"Unknown dataset type: {dataset_type}")
            
            # Load dataset
            dataset = self.loaders[dataset_type](name, config)
            
            # Validate dataset
            validation_result = self.validator.validate_dataset(dataset)
            if not validation_result.is_valid:
                error_msg = "; ".join([str(error) for error in validation_result.errors])
                raise DatasetValidationError(f"Dataset validation failed: {error_msg}")
            
            # Log warnings
            for warning in validation_result.warnings:
                logger.warning(f"Dataset {name}: {warning}")
            
            # Cache dataset
            if self.cache:
                metadata = {
                    'type': dataset_type,
                    'config': config.to_dict(),
                    'validation_result': validation_result
                }
                self.cache.put(name, dataset, metadata)
            
            self.stats['datasets_loaded'] += 1
            logger.debug(f"Dataset {name} loaded successfully")
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to load dataset {name}: {e}")
            raise DatasetNotFoundError(f"Could not load dataset {name}: {str(e)}")
    
    def _load_torchvision_dataset(self, name: str, config: DictConfig) -> Dataset:
        """Load torchvision dataset"""
        dataset_name = config.get('name', name.upper())
        root = Path(config.get('root', './data'))
        download = config.get('download', True)
        
        # Get torchvision dataset class
        if not hasattr(torchvision.datasets, dataset_name):
            raise DatasetNotFoundError(f"Torchvision dataset {dataset_name} not found")
        
        dataset_class = getattr(torchvision.datasets, dataset_name)
        
        # Load training data
        try:
            torch_dataset = dataset_class(
                root=str(root),
                train=True,
                download=download,
                transform=None  # We'll handle transforms separately
            )
            
            # Convert to our Dataset format
            if hasattr(torch_dataset, 'data') and hasattr(torch_dataset, 'targets'):
                data = torch.tensor(torch_dataset.data, dtype=torch.float32)
                targets = torch.tensor(torch_dataset.targets, dtype=torch.long)
            else:
                # Fallback: iterate through dataset
                data_list, targets_list = [], []
                for item, target in torch_dataset:
                    data_list.append(torch.tensor(item))
                    targets_list.append(target)
                data = torch.stack(data_list)
                targets = torch.tensor(targets_list, dtype=torch.long)
            
            # Handle transforms if specified
            transform = self._build_transform(config.get('transforms', {}))
            
            dataset = Dataset(name, data, targets, transform)
            return dataset
            
        except Exception as e:
            raise DatasetNotFoundError(f"Failed to load torchvision dataset {dataset_name}: {e}")
    
    def _load_custom_dataset(self, name: str, config: DictConfig) -> Dataset:
        """Load custom dataset"""
        data_path = Path(config.get('data_path', './data'))
        loader_class = config.get('loader_class')
        
        if not data_path.exists():
            raise DatasetNotFoundError(f"Custom dataset path does not exist: {data_path}")
        
        # For now, implement a simple file-based loader
        # In practice, this would use the specified loader_class
        try:
            # This is a placeholder implementation
            # Real implementation would use the loader_class
            data = torch.randn(1000, 32, 32, 3)  # Placeholder data
            targets = torch.randint(0, 10, (1000,))  # Placeholder targets
            
            transform = self._build_transform(config.get('preprocessing', {}))
            dataset = Dataset(name, data, targets, transform)
            return dataset
            
        except Exception as e:
            raise DatasetNotFoundError(f"Failed to load custom dataset: {e}")
    
    def _load_file_dataset(self, name: str, config: DictConfig) -> Dataset:
        """Load dataset from file"""
        file_path = Path(config.get('file_path'))
        
        if not file_path.exists():
            raise DatasetNotFoundError(f"Dataset file does not exist: {file_path}")
        
        try:
            # Load from pickle file
            with open(file_path, 'rb') as f:
                data_dict = pickle.load(f)
            
            data = torch.tensor(data_dict['data'], dtype=torch.float32)
            targets = torch.tensor(data_dict['targets'], dtype=torch.long)
            
            dataset = Dataset(name, data, targets)
            return dataset
            
        except Exception as e:
            raise DatasetNotFoundError(f"Failed to load dataset from file: {e}")
    
    def _build_transform(self, transform_config: Dict[str, Any]) -> Optional[Callable]:
        """Build transform pipeline from configuration"""
        if not transform_config:
            return None
        
        # This is a simplified implementation
        # Real implementation would build a complex transform pipeline
        def simple_transform(x):
            return x
        
        return simple_transform
    
    def register_dataset(self, name: str, dataset: Dataset, metadata: Dict = None) -> None:
        """
        Register a dataset in the manager.
        
        Args:
            name: Dataset name
            dataset: Dataset object
            metadata: Optional metadata dictionary
            
        Raises:
            DatasetValidationError: If dataset validation fails
        """
        with self._registry_lock:
            # Validate dataset
            validation_result = self.validator.validate_dataset(dataset)
            if not validation_result.is_valid:
                self.stats['validation_errors'] += 1
                error_msg = "; ".join([str(error) for error in validation_result.errors])
                raise DatasetValidationError(f"Dataset validation failed: {error_msg}")
            
            # Register dataset
            self.datasets[name] = dataset
            self.metadata[name] = metadata or {}
            
            # Cache if enabled
            if self.cache:
                self.cache.put(name, dataset, self.metadata[name])
            
            logger.debug(f"Dataset {name} registered successfully")
    
    def create_task_sequence(self, dataset_name: str, num_tasks: int) -> List[Task]:
        """
        Create task sequence from dataset.
        
        Args:
            dataset_name: Name of the dataset
            num_tasks: Number of tasks to create
            
        Returns:
            List[Task]: Generated task sequence
            
        Raises:
            DatasetNotFoundError: If dataset is not found
            TaskGeneratorError: If task generation fails
        """
        # Get dataset
        if dataset_name in self.datasets:
            dataset = self.datasets[dataset_name]
        else:
            # Try to load from configuration
            if dataset_name in self.dataset_config:
                dataset = self.load_dataset(dataset_name, DictConfig(self.dataset_config[dataset_name]))
            else:
                raise DatasetNotFoundError(f"Dataset {dataset_name} not found")
        
        try:
            # Generate tasks using task generator
            task_type = self.task_generator.task_type
            if task_type == 'class_incremental':
                tasks = self.task_generator.generate_class_incremental_tasks(
                    dataset, self.task_generator.classes_per_task
                )
            elif task_type == 'domain_incremental':
                tasks = self.task_generator.generate_domain_incremental_tasks([dataset])
            elif task_type == 'task_incremental':
                tasks = self.task_generator.generate_task_incremental_tasks(
                    dataset, num_tasks
                )
            else:
                # Default to class incremental
                tasks = self.task_generator.generate_class_incremental_tasks(
                    dataset, self.task_generator.classes_per_task
                )
            
            logger.debug(f"Created {len(tasks)} tasks from dataset {dataset_name}")
            return tasks
            
        except Exception as e:
            logger.error(f"Failed to create task sequence: {e}")
            raise TaskGeneratorError(f"Task generation failed: {str(e)}")
    
    def get_client_data(self, client_id: str, task_id: int) -> DataLoader:
        """
        Get client-specific data for a task.
        
        Args:
            client_id: Client identifier
            task_id: Task identifier
            
        Returns:
            DataLoader: Client's data loader for the task
            
        Raises:
            DatasetNotFoundError: If data is not available
        """
        # This would typically interact with the federation engine
        # For now, return a placeholder implementation
        try:
            # Get task data (simplified implementation)
            # In practice, this would use the split strategy to get client-specific data
            data = torch.randn(100, 32, 32, 3)  # Placeholder
            targets = torch.randint(0, 10, (100,))  # Placeholder
            
            dataset = Dataset(f"client_{client_id}_task_{task_id}", data, targets)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
            
            logger.debug(f"Retrieved data for client {client_id}, task {task_id}")
            return dataloader
            
        except Exception as e:
            raise DatasetNotFoundError(f"Could not get client data: {str(e)}")
    
    def cache_dataset(self, name: str, dataset: Dataset) -> None:
        """
        Cache a dataset.
        
        Args:
            name: Dataset name
            dataset: Dataset to cache
        """
        if self.cache:
            self.cache.put(name, dataset)
            logger.debug(f"Dataset {name} cached")
        else:
            logger.warning("Caching is disabled")
    
    def clear_cache(self, dataset_name: Optional[str] = None) -> None:
        """
        Clear dataset cache.
        
        Args:
            dataset_name: Optional specific dataset to clear, None for all
        """
        if self.cache:
            if dataset_name:
                self.cache.clear(dataset_name)
                logger.debug(f"Cache cleared for dataset: {dataset_name}")
            else:
                self.cache.clear()
                logger.debug("All dataset cache cleared")
        else:
            logger.warning("Caching is disabled")
    
    def get_dataset_metadata(self, name: str) -> Dict[str, Any]:
        """
        Get dataset metadata.
        
        Args:
            name: Dataset name
            
        Returns:
            Dict[str, Any]: Dataset metadata
            
        Raises:
            DatasetNotFoundError: If dataset is not found
        """
        if name in self.metadata:
            return self.metadata[name].copy()
        elif name in self.dataset_config:
            return self.dataset_config[name].copy()
        else:
            raise DatasetNotFoundError(f"Metadata for dataset {name} not found")
    
    def validate_dataset(self, dataset: Dataset) -> ValidationResult:
        """
        Validate a dataset.
        
        Args:
            dataset: Dataset to validate
            
        Returns:
            ValidationResult: Validation results
        """
        return self.validator.validate_dataset(dataset)
    
    def download_dataset(self, name: str, download_path: Path) -> Path:
        """
        Download dataset to specified path.
        
        Args:
            name: Dataset name
            download_path: Path to download to
            
        Returns:
            Path: Actual download path
            
        Raises:
            DatasetNotFoundError: If dataset cannot be downloaded
        """
        if name not in self.dataset_config:
            raise DatasetNotFoundError(f"Dataset {name} not found in configuration")
        
        config = self.dataset_config[name]
        dataset_type = config.get('type', 'torchvision')
        
        if dataset_type == 'torchvision':
            # Download torchvision dataset
            dataset_name = config.get('name', name.upper())
            if hasattr(torchvision.datasets, dataset_name):
                dataset_class = getattr(torchvision.datasets, dataset_name)
                try:
                    download_path.mkdir(parents=True, exist_ok=True)
                    dataset_class(root=str(download_path), download=True)
                    logger.debug(f"Dataset {name} downloaded to {download_path}")
                    return download_path
                except Exception as e:
                    raise DatasetNotFoundError(f"Failed to download dataset: {e}")
        
        raise DatasetNotFoundError(f"Download not supported for dataset type: {dataset_type}")
    
    def list_available_datasets(self) -> List[str]:
        """
        List all available datasets.
        
        Returns:
            List[str]: List of available dataset names
        """
        available = list(self.datasets.keys())
        available.extend(self.dataset_config.keys())
        return list(set(available))  # Remove duplicates
    
    def get_dataset_statistics(self, name: str) -> Dict[str, Any]:
        """
        Get dataset statistics.
        
        Args:
            name: Dataset name
            
        Returns:
            Dict[str, Any]: Dataset statistics
            
        Raises:
            DatasetNotFoundError: If dataset is not found
        """
        # Get dataset
        if name in self.datasets:
            dataset = self.datasets[name]
        else:
            try:
                if name in self.dataset_config:
                    dataset = self.load_dataset(name, DictConfig(self.dataset_config[name]))
                else:
                    raise DatasetNotFoundError(f"Dataset {name} not found")
            except Exception as e:
                raise DatasetNotFoundError(f"Could not load dataset for statistics: {e}")
        
        try:
            stats = {
                'name': name,
                'size': len(dataset),
                'num_classes': len(torch.unique(dataset.targets)),
                'data_shape': list(dataset.data.shape),
                'targets_shape': list(dataset.targets.shape),
                'memory_usage_mb': self._calculate_memory_usage(dataset),
                'class_distribution': self._get_class_distribution(dataset),
                'data_type': str(dataset.data.dtype),
                'targets_type': str(dataset.targets.dtype)
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to compute statistics for dataset {name}: {e}")
            raise DatasetManagerError(f"Statistics computation failed: {str(e)}")
    
    def _calculate_memory_usage(self, dataset: Dataset) -> float:
        """Calculate memory usage of dataset in MB"""
        try:
            data_size = dataset.data.element_size() * dataset.data.nelement()
            targets_size = dataset.targets.element_size() * dataset.targets.nelement()
            total_size = data_size + targets_size
            return total_size / (1024 ** 2)  # Convert to MB
        except:
            return 0.0
    
    def _get_class_distribution(self, dataset: Dataset) -> Dict[int, int]:
        """Get class distribution of dataset"""
        try:
            class_counts = torch.bincount(dataset.targets)
            return {i: int(count) for i, count in enumerate(class_counts) if count > 0}
        except:
            return {}
    
    def get_manager_statistics(self) -> Dict[str, Any]:
        """
        Get dataset manager statistics.
        
        Returns:
            Dict[str, Any]: Manager statistics
        """
        stats = self.stats.copy()
        stats.update({
            'registered_datasets': len(self.datasets),
            'configured_datasets': len(self.dataset_config),
            'memory_usage': psutil.Process().memory_info().rss / (1024**2),  # MB
        })
        
        if self.cache:
            stats['cache_stats'] = self.cache.get_stats()
        
        return stats
