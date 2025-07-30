# fedcl/data/task.py
"""
Task data structure implementation for FedCL framework.

This module defines the Task dataclass which represents a learning task
in the federated continual learning context.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
from enum import Enum
import copy
import pickle
import hashlib
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import logging

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Task type enumeration for different continual learning scenarios."""
    
    CLASS_INCREMENTAL = "class_incremental"
    DOMAIN_INCREMENTAL = "domain_incremental"
    TASK_INCREMENTAL = "task_incremental"
    MIXED = "mixed"


class TaskError(Exception):
    """Base exception for Task-related errors."""
    pass


class TaskValidationError(TaskError):
    """Exception raised when task validation fails."""
    pass


class TaskSerializationError(TaskError):
    """Exception raised when task serialization/deserialization fails."""
    pass


@dataclass
class Task:
    """
    Represents a learning task in federated continual learning.
    
    A Task encapsulates all the necessary information for training on a specific
    learning task, including data, class information, and metadata.
    
    Attributes:
        task_id: Unique identifier for the task
        data: DataLoader containing the task's training data
        classes: List of class labels present in this task
        metadata: Additional metadata about the task
        task_type: Type of the task (class/domain/task incremental)
    """
    
    task_id: int
    data: DataLoader
    classes: List[int]
    metadata: Dict[str, Any] = field(default_factory=dict)
    task_type: TaskType = TaskType.CLASS_INCREMENTAL
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        if not self.validate():
            raise TaskValidationError(f"Task {self.task_id} validation failed")
        
        # Freeze non-metadata attributes by creating a frozen copy
        self._frozen_attrs = {
            'task_id': self.task_id,
            'data': self.data,
            'classes': copy.deepcopy(self.classes),
            'task_type': self.task_type
        }
    
    def get_class_distribution(self) -> Dict[int, int]:
        """
        Get the distribution of classes in the task data.
        
        Returns:
            Dictionary mapping class labels to their counts
            
        Raises:
            RuntimeError: If unable to compute class distribution
        """
        try:
            class_counts = {}
            total_samples = 0
            
            for batch_data, batch_targets in self.data:
                if isinstance(batch_targets, torch.Tensor):
                    targets = batch_targets.tolist()
                else:
                    targets = batch_targets
                
                for target in targets:
                    class_counts[target] = class_counts.get(target, 0) + 1
                    total_samples += 1
            
            logger.debug(f"Task {self.task_id}: computed class distribution for {total_samples} samples")
            return class_counts
            
        except Exception as e:
            logger.error(f"Failed to compute class distribution for task {self.task_id}: {e}")
            raise RuntimeError(f"Unable to compute class distribution: {e}")
    
    def get_sample_count(self) -> int:
        """
        Get the total number of samples in the task.
        
        Returns:
            Total number of samples
            
        Raises:
            RuntimeError: If unable to count samples
        """
        try:
            if hasattr(self.data.dataset, '__len__'):
                sample_count = len(self.data.dataset)
            else:
                # Fallback: iterate through data to count
                sample_count = sum(len(batch_targets) for _, batch_targets in self.data)
            
            logger.debug(f"Task {self.task_id}: counted {sample_count} samples")
            return sample_count
            
        except Exception as e:
            logger.error(f"Failed to count samples for task {self.task_id}: {e}")
            raise RuntimeError(f"Unable to count samples: {e}")
    
    def get_memory_usage(self) -> float:
        """
        Estimate memory usage of the task in MB.
        
        Returns:
            Estimated memory usage in megabytes
            
        Raises:
            RuntimeError: If unable to estimate memory usage
        """
        try:
            memory_mb = 0.0
            
            # Estimate data memory usage
            if hasattr(self.data.dataset, 'data'):
                if isinstance(self.data.dataset.data, torch.Tensor):
                    memory_mb += self.data.dataset.data.element_size() * self.data.dataset.data.nelement() / (1024 * 1024)
            
            # Estimate targets memory usage
            if hasattr(self.data.dataset, 'targets'):
                if isinstance(self.data.dataset.targets, torch.Tensor):
                    memory_mb += self.data.dataset.targets.element_size() * self.data.dataset.targets.nelement() / (1024 * 1024)
            
            # Add metadata memory estimation
            try:
                metadata_bytes = len(pickle.dumps(self.metadata))
                memory_mb += metadata_bytes / (1024 * 1024)
            except:
                # Fallback estimation for metadata
                memory_mb += 0.1  # Assume 0.1 MB for metadata
            
            logger.debug(f"Task {self.task_id}: estimated memory usage {memory_mb:.2f} MB")
            return memory_mb
            
        except Exception as e:
            logger.error(f"Failed to estimate memory usage for task {self.task_id}: {e}")
            raise RuntimeError(f"Unable to estimate memory usage: {e}")
    
    def validate(self) -> bool:
        """
        Validate the task structure and data integrity.
        
        Returns:
            True if task is valid, False otherwise
        """
        try:
            # Validate task_id
            if not isinstance(self.task_id, int) or self.task_id < 0:
                logger.error(f"Invalid task_id: {self.task_id}")
                return False
            
            # Validate data
            if not isinstance(self.data, DataLoader):
                logger.error(f"Invalid data type: {type(self.data)}")
                return False
            
            # Validate classes
            if not isinstance(self.classes, list) or not all(isinstance(c, int) for c in self.classes):
                logger.error(f"Invalid classes: {self.classes}")
                return False
            
            # Validate task_type
            if not isinstance(self.task_type, TaskType):
                logger.error(f"Invalid task_type: {self.task_type}")
                return False
            
            # Validate metadata
            if not isinstance(self.metadata, dict):
                logger.error(f"Invalid metadata type: {type(self.metadata)}")
                return False
            
            # Check if data is accessible
            try:
                iter(self.data)
            except Exception as e:
                logger.error(f"Data is not iterable: {e}")
                return False
            
            logger.debug(f"Task {self.task_id} validation successful")
            return True
            
        except Exception as e:
            logger.error(f"Task validation error: {e}")
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert task to dictionary representation.
        
        Returns:
            Dictionary representation of the task
            
        Raises:
            TaskSerializationError: If serialization fails
        """
        try:
            task_dict = {
                'task_id': self.task_id,
                'classes': copy.deepcopy(self.classes),
                'metadata': copy.deepcopy(self.metadata),
                'task_type': self.task_type.value,
                'sample_count': self.get_sample_count(),
                'class_distribution': self.get_class_distribution(),
                'memory_usage_mb': self.get_memory_usage()
            }
            
            # Note: DataLoader is not directly serializable, so we store metadata about it
            task_dict['data_info'] = {
                'batch_size': self.data.batch_size,
                'dataset_length': len(self.data.dataset) if hasattr(self.data.dataset, '__len__') else None,
                'num_workers': self.data.num_workers,
                'shuffle': getattr(self.data, 'shuffle', False)
            }
            
            logger.debug(f"Task {self.task_id} serialized to dictionary")
            return task_dict
            
        except Exception as e:
            logger.error(f"Failed to serialize task {self.task_id}: {e}")
            raise TaskSerializationError(f"Failed to serialize task: {e}")
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], data_loader: DataLoader) -> Task:
        """
        Create task from dictionary representation.
        
        Args:
            data: Dictionary containing task data
            data_loader: DataLoader instance (must be provided separately)
            
        Returns:
            Task instance
            
        Raises:
            TaskSerializationError: If deserialization fails
        """
        try:
            task_type = TaskType(data['task_type']) if isinstance(data['task_type'], str) else data['task_type']
            
            task = cls(
                task_id=data['task_id'],
                data=data_loader,
                classes=copy.deepcopy(data['classes']),
                metadata=copy.deepcopy(data.get('metadata', {})),
                task_type=task_type
            )
            
            logger.debug(f"Task {task.task_id} deserialized from dictionary")
            return task
            
        except Exception as e:
            logger.error(f"Failed to deserialize task: {e}")
            raise TaskSerializationError(f"Failed to deserialize task: {e}")
    
    def __eq__(self, other: object) -> bool:
        """
        Check equality with another task.
        
        Args:
            other: Another object to compare with
            
        Returns:
            True if tasks are equal, False otherwise
        """
        if not isinstance(other, Task):
            return False
        
        return (
            self.task_id == other.task_id and
            self.classes == other.classes and
            self.task_type == other.task_type and
            self.metadata == other.metadata
        )
    
    def __hash__(self) -> int:
        """
        Compute hash of the task.
        
        Returns:
            Hash value of the task
        """
        # Create a deterministic hash based on immutable attributes
        hash_data = (
            self.task_id,
            tuple(sorted(self.classes)),
            self.task_type.value,
            tuple(sorted(self.metadata.items()))
        )
        return hash(hash_data)
    
    def __deepcopy__(self, memo: Dict[int, Any]) -> Task:
        """
        Create a deep copy of the task.
        
        Args:
            memo: Memoization dictionary
            
        Returns:
            Deep copy of the task
        """
        # DataLoader cannot be deep copied directly, so we create a new task
        # with the same DataLoader reference but deep copied other attributes
        copied_task = Task(
            task_id=self.task_id,
            data=self.data,  # DataLoader reference is shared
            classes=copy.deepcopy(self.classes, memo),
            metadata=copy.deepcopy(self.metadata, memo),
            task_type=self.task_type
        )
        
        logger.debug(f"Task {self.task_id} deep copied")
        return copied_task
    
    def __repr__(self) -> str:
        """String representation of the task."""
        return (f"Task(task_id={self.task_id}, task_type={self.task_type.value}, "
                f"classes={self.classes}, num_samples={self.get_sample_count()})")