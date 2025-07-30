# fedcl/data/dataloader.py
"""
DataLoader wrapper implementation for FedCL framework.

This module provides a DataLoader wrapper class that extends PyTorch's DataLoader
with additional functionality for federated continual learning.
"""

from __future__ import annotations
from typing import Iterator, Tuple, Any, Optional, Dict
import torch
from torch.utils.data import DataLoader as PyTorchDataLoader, Dataset as PyTorchDataset
from torch.utils.data.distributed import DistributedSampler
import logging

logger = logging.getLogger(__name__)


class DataLoaderError(Exception):
    """Base exception for DataLoader-related errors."""
    pass


class DataLoaderConfigurationError(DataLoaderError):
    """Exception raised when DataLoader configuration is invalid."""
    pass


class DataLoader:
    """
    DataLoader wrapper class that extends PyTorch DataLoader with additional functionality.
    
    This class provides enhanced functionality for federated continual learning,
    including dynamic batch size adjustment, distributed training support,
    and improved error handling.
    
    Attributes:
        dataset: The dataset to load data from
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle the data
        num_workers: Number of subprocesses to use for data loading
    """
    
    def __init__(
        self,
        dataset: PyTorchDataset,
        batch_size: int = 32,
        shuffle: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn: Optional[callable] = None,
        multiprocessing_context: Optional[str] = None,
        generator: Optional[torch.Generator] = None,
        prefetch_factor: int = 2,
        persistent_workers: bool = False,
        **kwargs
    ):
        """
        Initialize the DataLoader wrapper.
        
        Args:
            dataset: Dataset to load data from
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle the data at every epoch
            num_workers: Number of subprocesses to use for data loading
            pin_memory: If True, tensors are copied into CUDA pinned memory
            drop_last: If True, drop the last incomplete batch
            timeout: Timeout for collecting a batch from workers
            worker_init_fn: Function to call on each worker subprocess
            multiprocessing_context: Multiprocessing context
            generator: Random number generator for sampling
            prefetch_factor: Number of samples loaded in advance by each worker
            persistent_workers: If True, workers are not shutdown after each epoch
            **kwargs: Additional arguments passed to PyTorch DataLoader
            
        Raises:
            DataLoaderConfigurationError: If configuration is invalid
        """
        # Validate inputs
        if not isinstance(dataset, PyTorchDataset):
            raise DataLoaderConfigurationError(f"Dataset must be a PyTorch Dataset, got {type(dataset)}")
        
        if batch_size <= 0:
            raise DataLoaderConfigurationError(f"Batch size must be positive, got {batch_size}")
        
        if num_workers < 0:
            raise DataLoaderConfigurationError(f"Number of workers must be non-negative, got {num_workers}")
        
        self.dataset = dataset
        self._batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.timeout = timeout
        self.worker_init_fn = worker_init_fn
        self.multiprocessing_context = multiprocessing_context
        self.generator = generator
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        
        # Store additional kwargs
        self._kwargs = kwargs
        
        # Current epoch for distributed training
        self._current_epoch = 0
        
        # Create the underlying PyTorch DataLoader
        self._create_dataloader()
        
        logger.debug(f"DataLoader initialized with batch_size={batch_size}, "
                    f"num_workers={num_workers}, dataset_size={len(dataset)}")
    
    def _create_dataloader(self) -> None:
        """Create or recreate the underlying PyTorch DataLoader."""
        try:
            # Handle distributed sampling - 只在创建时设置一次
            sampler = self._kwargs.get('sampler', None)
            # 注释掉这行，避免重复设置
            # if isinstance(sampler, DistributedSampler):
            #     sampler.set_epoch(self._current_epoch)
            
            # Fix prefetch_factor when num_workers=0
            effective_prefetch_factor = None if self.num_workers == 0 else self.prefetch_factor
            
            self._dataloader = PyTorchDataLoader(
                dataset=self.dataset,
                batch_size=self._batch_size,
                shuffle=self.shuffle if sampler is None else False,
                sampler=sampler,
                num_workers=self.num_workers,
                collate_fn=self._kwargs.get('collate_fn', None),
                pin_memory=self.pin_memory,
                drop_last=self.drop_last,
                timeout=self.timeout,
                worker_init_fn=self.worker_init_fn,
                multiprocessing_context=self.multiprocessing_context,
                generator=self.generator,
                prefetch_factor=effective_prefetch_factor,
                persistent_workers=self.persistent_workers
            )
            
        except Exception as e:
            logger.error(f"Failed to create DataLoader: {e}")
            raise DataLoaderError(f"Failed to create DataLoader: {e}")
    
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Return an iterator over the dataset.
        
        Returns:
            Iterator yielding batches of (data, targets)
            
        Raises:
            DataLoaderError: If iteration fails
        """
        try:
            return iter(self._dataloader)
        except Exception as e:
            logger.error(f"DataLoader iteration failed: {e}")
            raise DataLoaderError(f"DataLoader iteration failed: {e}")
    
    def __len__(self) -> int:
        """
        Get the number of batches in the DataLoader.
        
        Returns:
            Number of batches
        """
        return len(self._dataloader)
    
    def set_epoch(self, epoch: int) -> None:
        """
        Set the epoch for distributed training.
        
        This method is important for proper shuffling in distributed training
        scenarios where DistributedSampler is used.
        
        Args:
            epoch: Current epoch number
            
        Raises:
            DataLoaderError: If epoch setting fails
        """
        try:
            self._current_epoch = epoch
            
            # Update distributed sampler if present
            sampler = self._kwargs.get('sampler', None)
            if isinstance(sampler, DistributedSampler):
                sampler.set_epoch(epoch)
                logger.debug(f"Set epoch {epoch} for DistributedSampler")
            
            # 只有当 epoch 改变时才重新创建 dataloader
            # 或者移除这行，让用户手动调用 reset() 如果需要的话
            # if hasattr(self, '_dataloader'):
            #     self._create_dataloader()
                
        except Exception as e:
            logger.error(f"Failed to set epoch {epoch}: {e}")
            raise DataLoaderError(f"Failed to set epoch: {e}")
    
    def get_batch_size(self) -> int:
        """
        Get the current batch size.
        
        Returns:
            Current batch size
        """
        return self._batch_size
    
    def set_batch_size(self, batch_size: int) -> None:
        """
        Dynamically adjust the batch size.
        
        Args:
            batch_size: New batch size
            
        Raises:
            DataLoaderConfigurationError: If batch size is invalid
            DataLoaderError: If batch size adjustment fails
        """
        if batch_size <= 0:
            raise DataLoaderConfigurationError(f"Batch size must be positive, got {batch_size}")
        
        try:
            old_batch_size = self._batch_size
            self._batch_size = batch_size
            
            # Recreate the DataLoader with new batch size
            self._create_dataloader()
            
            logger.debug(f"Batch size changed from {old_batch_size} to {batch_size}")
            
        except Exception as e:
            # Revert batch size on error
            self._batch_size = getattr(self, '_batch_size', batch_size)
            logger.error(f"Failed to set batch size to {batch_size}: {e}")
            raise DataLoaderError(f"Failed to set batch size: {e}")
    
    def get_dataset_size(self) -> int:
        """
        Get the size of the underlying dataset.
        
        Returns:
            Number of samples in the dataset
        """
        return len(self.dataset)
    
    def get_effective_batch_size(self) -> int:
        """
        Get the effective batch size considering drop_last setting.
        
        Returns:
            Effective batch size for the last batch
        """
        dataset_size = self.get_dataset_size()
        last_batch_size = dataset_size % self._batch_size
        
        if self.drop_last or last_batch_size == 0:
            return self._batch_size
        else:
            return last_batch_size if len(self) > 0 else self._batch_size
    
    def reset(self) -> None:
        """
        Reset the DataLoader state.
        
        This method recreates the underlying DataLoader, which can be useful
        after changing configuration or recovering from errors.
        """
        try:
            self._create_dataloader()
            logger.debug("DataLoader reset successfully")
        except Exception as e:
            logger.error(f"Failed to reset DataLoader: {e}")
            raise DataLoaderError(f"Failed to reset DataLoader: {e}")
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the current configuration of the DataLoader.
        
        Returns:
            Dictionary containing DataLoader configuration
        """
        config = {
            'batch_size': self._batch_size,
            'shuffle': self.shuffle,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            'drop_last': self.drop_last,
            'timeout': self.timeout,
            'prefetch_factor': self.prefetch_factor,
            'persistent_workers': self.persistent_workers,
            'dataset_size': self.get_dataset_size(),
            'num_batches': len(self),
            'current_epoch': self._current_epoch
        }
        
        # Add sampler information if present
        sampler = self._kwargs.get('sampler', None)
        if sampler is not None:
            config['sampler_type'] = type(sampler).__name__
            if isinstance(sampler, DistributedSampler):
                config['distributed'] = True
                config['world_size'] = getattr(sampler, 'num_replicas', None)
                config['rank'] = getattr(sampler, 'rank', None)
        
        return config
    
    def get_memory_info(self) -> Dict[str, Any]:
        """
        Get memory usage information for the DataLoader.
        
        Returns:
            Dictionary containing memory usage information
        """
        try:
            # Estimate memory usage based on batch size and dataset
            sample_batch = next(iter(self))
            if isinstance(sample_batch, (tuple, list)) and len(sample_batch) >= 2:
                data_tensor, target_tensor = sample_batch[0], sample_batch[1]
                
                # Calculate memory per batch
                data_memory = data_tensor.element_size() * data_tensor.nelement()
                target_memory = target_tensor.element_size() * target_tensor.nelement()
                batch_memory_mb = (data_memory + target_memory) / (1024 * 1024)
                
                # Estimate total memory usage with workers
                total_memory_mb = batch_memory_mb * (self.num_workers + 1) * self.prefetch_factor
                
                return {
                    'batch_memory_mb': float(batch_memory_mb),
                    'estimated_total_memory_mb': float(total_memory_mb),
                    'data_memory_per_batch_mb': float(data_memory / (1024 * 1024)),
                    'target_memory_per_batch_mb': float(target_memory / (1024 * 1024)),
                    'num_workers': self.num_workers,
                    'prefetch_factor': self.prefetch_factor
                }
                
        except Exception as e:
            logger.warning(f"Failed to compute memory info: {e}")
            
        return {
            'batch_memory_mb': 0.0,
            'estimated_total_memory_mb': 0.0,
            'error': 'Unable to compute memory information'
        }
    
    def __repr__(self) -> str:
        """String representation of the DataLoader."""
        return (f"DataLoader(dataset_size={self.get_dataset_size()}, "
                f"batch_size={self._batch_size}, num_batches={len(self)}, "
                f"num_workers={self.num_workers})")
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        config = self.get_config()
        return (f"DataLoader with {config['dataset_size']} samples, "
                f"{config['num_batches']} batches of size {config['batch_size']}, "
                f"{config['num_workers']} workers")
    
    # Delegate attribute access to underlying DataLoader for compatibility
    def __getattr__(self, name: str) -> Any:
        """
        Delegate attribute access to the underlying PyTorch DataLoader.
        
        Args:
            name: Attribute name
            
        Returns:
            Attribute value from underlying DataLoader
            
        Raises:
            AttributeError: If attribute doesn't exist
        """
        if hasattr(self, '_dataloader') and hasattr(self._dataloader, name):
            return getattr(self._dataloader, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")