# fedcl/data/dataset.py
"""
Dataset wrapper implementation for FedCL framework.

This module provides a Dataset wrapper class that extends PyTorch's Dataset
with additional functionality for federated continual learning.
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
import copy
import pickle
import hashlib
from pathlib import Path
import torch
from torch.utils.data import Dataset as PyTorchDataset
import numpy as np
import logging

logger = logging.getLogger(__name__)


class DatasetError(Exception):
    """Base exception for dataset-related errors."""
    pass


class DatasetValidationError(DatasetError):
    """Exception raised when dataset validation fails."""
    pass


class DatasetTransformError(DatasetError):
    """Exception raised when dataset transformation fails."""
    pass


class Dataset(PyTorchDataset):
    """
    Dataset wrapper class that extends PyTorch Dataset with additional functionality.
    
    This class provides enhanced functionality for federated continual learning,
    including data validation, transformation management, and memory optimization.
    
    Attributes:
        name: Human-readable name for the dataset
        data: Tensor containing the dataset features
        targets: Tensor containing the dataset labels
        transform: Optional transformation function to apply to data
    """
    
    def __init__(
        self,
        name: str,
        data: torch.Tensor,
        targets: torch.Tensor,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    ):
        """
        Initialize the Dataset wrapper.
        
        Args:
            name: Human-readable name for the dataset
            data: Tensor containing the dataset features
            targets: Tensor containing the dataset labels
            transform: Optional transformation function to apply to data
            
        Raises:
            DatasetValidationError: If dataset validation fails
        """
        self.name = name
        self.data = data
        self.targets = targets
        self.transform = transform
        
        # Validate dataset integrity
        if not self.validate_integrity():
            raise DatasetValidationError(f"Dataset '{name}' failed integrity validation")
        
        # Cache for lazy loading and optimization
        self._cache: Dict[int, Tuple[torch.Tensor, int]] = {}
        self._stats_cache: Optional[Dict[str, Any]] = None
        self._classes_cache: Optional[List[int]] = None
        self._class_distribution_cache: Optional[Dict[int, int]] = None
        
        logger.debug(f"Dataset '{name}' initialized with {len(self)} samples")
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single item from the dataset.
        
        Args:
            index: Index of the item to retrieve
            
        Returns:
            Tuple of (data, target) for the given index
            
        Raises:
            IndexError: If index is out of bounds
            DatasetError: If data retrieval fails
        """
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of bounds for dataset of size {len(self)}")
        
        try:
            # Check cache first for performance optimization
            if index in self._cache:
                return self._cache[index]
            
            # Get data and target
            data_item = self.data[index]
            target_item = int(self.targets[index])
            
            # Apply transform if specified
            if self.transform is not None:
                try:
                    data_item = self.transform(data_item)
                except Exception as e:
                    logger.error(f"Transform failed for index {index}: {e}")
                    raise DatasetTransformError(f"Transform failed: {e}")
            
            # Cache the result for future access
            result = (data_item, target_item)
            self._cache[index] = result
            
            return result
            
        except Exception as e:
            if not isinstance(e, (IndexError, DatasetTransformError)):
                logger.error(f"Failed to get item at index {index}: {e}")
                raise DatasetError(f"Failed to retrieve item: {e}")
            raise
    
    def __len__(self) -> int:
        """
        Get the length of the dataset.
        
        Returns:
            Number of samples in the dataset
        """
        return len(self.data)
    
    def get_classes(self) -> List[int]:
        """
        Get the unique class labels in the dataset.
        
        Returns:
            Sorted list of unique class labels
        """
        if self._classes_cache is None:
            unique_classes = torch.unique(self.targets).tolist()
            self._classes_cache = sorted(unique_classes)
            logger.debug(f"Dataset '{self.name}': cached {len(self._classes_cache)} unique classes")
        
        return copy.deepcopy(self._classes_cache)
    
    def get_class_distribution(self) -> Dict[int, int]:
        """
        Get the distribution of classes in the dataset.
        
        Returns:
            Dictionary mapping class labels to their counts
        """
        if self._class_distribution_cache is None:
            class_counts = {}
            for target in self.targets:
                target_int = int(target)
                class_counts[target_int] = class_counts.get(target_int, 0) + 1
            
            self._class_distribution_cache = class_counts
            logger.debug(f"Dataset '{self.name}': cached class distribution with {len(class_counts)} classes")
        
        return copy.deepcopy(self._class_distribution_cache)
    
    def apply_transform(self, transform: Callable[[torch.Tensor], torch.Tensor]) -> Dataset:
        """
        Create a new dataset with the specified transform applied.
        
        Args:
            transform: Transformation function to apply
            
        Returns:
            New Dataset instance with the transform applied
            
        Raises:
            DatasetTransformError: If transform application fails
        """
        try:
            new_dataset = Dataset(
                name=f"{self.name}_transformed",
                data=self.data.clone(),
                targets=self.targets.clone(),
                transform=transform
            )
            
            logger.debug(f"Applied transform to dataset '{self.name}'")
            return new_dataset
            
        except Exception as e:
            logger.error(f"Failed to apply transform to dataset '{self.name}': {e}")
            raise DatasetTransformError(f"Failed to apply transform: {e}")
    
    def subset(self, indices: List[int]) -> Dataset:
        """
        Create a subset of the dataset with the specified indices.
        
        Args:
            indices: List of indices to include in the subset
            
        Returns:
            New Dataset instance containing only the specified indices
            
        Raises:
            DatasetError: If subset creation fails
        """
        try:
            # Validate indices
            if not indices:
                raise ValueError("Indices list cannot be empty")
            
            if min(indices) < 0 or max(indices) >= len(self):
                raise ValueError(f"Indices out of bounds: {min(indices)}-{max(indices)} for dataset of size {len(self)}")
            
            # Create subset
            subset_data = self.data[indices]
            subset_targets = self.targets[indices]
            
            subset_dataset = Dataset(
                name=f"{self.name}_subset_{len(indices)}",
                data=subset_data,
                targets=subset_targets,
                transform=self.transform
            )
            
            logger.debug(f"Created subset of dataset '{self.name}' with {len(indices)} samples")
            return subset_dataset
            
        except Exception as e:
            logger.error(f"Failed to create subset of dataset '{self.name}': {e}")
            raise DatasetError(f"Failed to create subset: {e}")
    
    def split(self, ratios: List[float]) -> List[Dataset]:
        """
        Split the dataset into multiple datasets based on the given ratios.
        
        Args:
            ratios: List of ratios for each split (must sum to 1.0)
            
        Returns:
            List of Dataset instances corresponding to each split
            
        Raises:
            DatasetError: If split creation fails
        """
        try:
            # Validate ratios
            if not ratios or len(ratios) < 2:
                raise ValueError("At least 2 ratios must be provided")
            
            if abs(sum(ratios) - 1.0) > 1e-6:
                raise ValueError(f"Ratios must sum to 1.0, got {sum(ratios)}")
            
            if any(r <= 0 for r in ratios):
                raise ValueError("All ratios must be positive")
            
            # Calculate split sizes
            total_size = len(self)
            split_sizes = [int(ratio * total_size) for ratio in ratios]
            
            # Adjust for rounding errors
            remaining = total_size - sum(split_sizes)
            for i in range(remaining):
                split_sizes[i % len(split_sizes)] += 1
            
            # Create random permutation for fair splitting
            indices = torch.randperm(total_size).tolist()
            
            # Create splits
            datasets = []
            start_idx = 0
            
            for i, size in enumerate(split_sizes):
                end_idx = start_idx + size
                split_indices = indices[start_idx:end_idx]
                
                split_dataset = self.subset(split_indices)
                split_dataset.name = f"{self.name}_split_{i}"
                datasets.append(split_dataset)
                
                start_idx = end_idx
            
            logger.debug(f"Split dataset '{self.name}' into {len(datasets)} parts with sizes {split_sizes}")
            return datasets
            
        except Exception as e:
            logger.error(f"Failed to split dataset '{self.name}': {e}")
            raise DatasetError(f"Failed to split dataset: {e}")
    
    def validate_integrity(self) -> bool:
        """
        Validate the integrity of the dataset.
        
        Returns:
            True if dataset is valid, False otherwise
        """
        try:
            # Check basic types
            if not isinstance(self.name, str) or not self.name.strip():
                logger.error(f"Invalid dataset name: {self.name}")
                return False
            
            if not isinstance(self.data, torch.Tensor):
                logger.error(f"Data must be a torch.Tensor, got {type(self.data)}")
                return False
            
            if not isinstance(self.targets, torch.Tensor):
                logger.error(f"Targets must be a torch.Tensor, got {type(self.targets)}")
                return False
            
            # Check dimensions
            if len(self.data) != len(self.targets):
                logger.error(f"Data and targets length mismatch: {len(self.data)} vs {len(self.targets)}")
                return False
            
            if len(self.data) == 0:
                logger.error("Dataset cannot be empty")
                return False
            
            # Check data types
            if not self.data.dtype.is_floating_point and not self.data.dtype == torch.uint8:
                logger.warning(f"Data dtype {self.data.dtype} may not be suitable for training")
            
            if not self.targets.dtype in [torch.int32, torch.int64, torch.long]:
                logger.error(f"Targets dtype {self.targets.dtype} not suitable for classification")
                return False
            
            # Check for NaN or infinite values
            if torch.isnan(self.data).any():
                logger.error("Data contains NaN values")
                return False
            
            if torch.isinf(self.data).any():
                logger.error("Data contains infinite values")
                return False
            
            # Check transform if present
            if self.transform is not None and not callable(self.transform):
                logger.error(f"Transform must be callable, got {type(self.transform)}")
                return False
            
            logger.debug(f"Dataset '{self.name}' passed integrity validation")
            return True
            
        except Exception as e:
            logger.error(f"Dataset integrity validation failed: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the dataset.
        
        Returns:
            Dictionary containing dataset statistics
        """
        if self._stats_cache is None:
            try:
                # Basic statistics
                stats = {
                    'name': self.name,
                    'size': len(self),
                    'data_shape': list(self.data.shape),
                    'data_dtype': str(self.data.dtype),
                    'targets_dtype': str(self.targets.dtype),
                    'num_classes': len(self.get_classes()),
                    'classes': self.get_classes(),
                    'class_distribution': self.get_class_distribution(),
                }
                
                # Data statistics
                if self.data.dtype.is_floating_point:
                    stats.update({
                        'data_mean': float(torch.mean(self.data)),
                        'data_std': float(torch.std(self.data)),
                        'data_min': float(torch.min(self.data)),
                        'data_max': float(torch.max(self.data)),
                    })
                
                # Memory usage
                data_memory = self.data.element_size() * self.data.nelement()
                targets_memory = self.targets.element_size() * self.targets.nelement()
                total_memory_mb = (data_memory + targets_memory) / (1024 * 1024)
                
                stats.update({
                    'memory_usage_mb': float(total_memory_mb),
                    'data_memory_mb': float(data_memory / (1024 * 1024)),
                    'targets_memory_mb': float(targets_memory / (1024 * 1024)),
                })
                
                # Class balance metrics
                class_counts = list(self.get_class_distribution().values())
                if class_counts:
                    stats.update({
                        'class_balance_ratio': float(min(class_counts) / max(class_counts)),
                        'most_frequent_class_count': max(class_counts),
                        'least_frequent_class_count': min(class_counts),
                    })
                
                self._stats_cache = stats
                logger.debug(f"Computed statistics for dataset '{self.name}'")
                
            except Exception as e:
                logger.error(f"Failed to compute statistics for dataset '{self.name}': {e}")
                # Return basic stats if computation fails
                self._stats_cache = {
                    'name': self.name,
                    'size': len(self),
                    'error': str(e)
                }
        
        return copy.deepcopy(self._stats_cache)
    
    def clear_cache(self) -> None:
        """Clear all cached data to free memory."""
        self._cache.clear()
        self._stats_cache = None
        self._classes_cache = None
        self._class_distribution_cache = None
        logger.debug(f"Cleared cache for dataset '{self.name}'")
    
    def __repr__(self) -> str:
        """String representation of the dataset."""
        return (f"Dataset(name='{self.name}', size={len(self)}, "
                f"classes={len(self.get_classes())}, shape={list(self.data.shape)})")
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        stats = self.get_stats()
        return (f"Dataset '{self.name}' with {stats['size']} samples, "
                f"{stats['num_classes']} classes, "
                f"memory usage: {stats.get('memory_usage_mb', 0):.2f} MB")
    
    def create_subset(self, indices: List[int]) -> 'Dataset':
        """
        Create a subset of the dataset using specified indices.
        
        Args:
            indices: List of indices to include in the subset
            
        Returns:
            New Dataset object containing only the specified indices
            
        Raises:
            DatasetError: If indices are invalid
        """
        try:
            # 检查空索引
            if isinstance(indices, torch.Tensor):
                if indices.numel() == 0:
                    raise DatasetError("Cannot create subset with empty indices list")
            elif not indices:
                raise DatasetError("Cannot create subset with empty indices list")
            
            # 转换为张量索引
            if isinstance(indices, list):
                indices_tensor = torch.tensor(indices, dtype=torch.long)
            else:
                indices_tensor = indices.long()
            
            # 验证索引
            max_idx = indices_tensor.max().item()
            min_idx = indices_tensor.min().item()
            
            if min_idx < 0 or max_idx >= len(self):
                raise DatasetError(f"Indices out of range: got [{min_idx}, {max_idx}], "
                                 f"valid range is [0, {len(self)-1}]")
            
            # 创建子集数据
            subset_data = self.data[indices_tensor]
            subset_targets = self.targets[indices_tensor]
            
            # 创建新的数据集
            subset_name = f"{self.name}_subset_{indices_tensor.numel()}"
            subset = Dataset(
                name=subset_name,
                data=subset_data,
                targets=subset_targets,
                transform=self.transform
            )
            
            logger.debug(f"Created subset '{subset_name}' with {indices_tensor.numel()} samples")
            return subset
            
        except Exception as e:
            logger.error(f"Failed to create subset: {e}")
            raise DatasetError(f"Failed to create subset: {e}")