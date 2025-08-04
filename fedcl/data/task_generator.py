# fedcl/data/task_generator.py
"""
Task Generator for FedCL Framework

This module provides task generation functionality for different continual learning scenarios,
including class incremental, domain incremental, and task incremental learning.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
from loguru import logger

from ..config.config_manager import DictConfig
from ..config.exceptions import ConfigValidationError
from .dataset import Dataset
from .task import Task, TaskType
from .split_strategy import SplitStrategy
from ..exceptions import FedCLError


class TaskGeneratorError(FedCLError):
    """Task generation related errors"""
    pass


class TaskValidationError(TaskGeneratorError):
    """Task validation errors"""
    pass


class TaskGenerator:
    """
    Task Generator for creating different types of continual learning tasks.
    
    This class provides comprehensive task generation capabilities for federated
    continual learning scenarios, supporting class incremental, domain incremental,
    and task incremental learning paradigms.
    """
    
    def __init__(self, config: DictConfig, split_strategy: SplitStrategy):
        """
        Initialize the TaskGenerator.
        
        Args:
            config: Configuration containing task generation parameters
            split_strategy: Strategy for splitting data among clients
            
        Raises:
            ConfigValidationError: If configuration is invalid
        """
        self.config = config
        self.split_strategy = split_strategy
        
        # Extract task generation configuration
        self.task_config = config.get('task_generation', {})
        self.num_tasks = self.task_config.get('num_tasks', 10)
        self.classes_per_task = self.task_config.get('classes_per_task', 10)
        
        # Random seed for reproducibility
        self.random_seed = self.task_config.get('random_seed', 42)
        self._set_random_seeds(self.random_seed)
        
        # Task type configuration
        self.task_type = self.task_config.get('type', 'class_incremental')
        
        # Replay buffer configuration
        self.replay_config = self.task_config.get('replay', {})
        self.replay_enabled = self.replay_config.get('enable', False)
        self.buffer_size_per_class = self.replay_config.get('buffer_size_per_class', 50)
        self.selection_strategy = self.replay_config.get('selection_strategy', 'random')
        
        # Internal state
        self._class_order: Optional[List[int]] = None
        self._task_statistics: Dict[str, Any] = {}
        
        logger.debug(f"TaskGenerator initialized with type: {self.task_type}, "
                   f"tasks: {self.num_tasks}, classes_per_task: {self.classes_per_task}")
    
    def _set_random_seeds(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    def generate_class_incremental_tasks(
        self, 
        dataset: Dataset, 
        classes_per_task: int
    ) -> List[Task]:
        """
        Generate class incremental learning tasks.
        
        In class incremental learning, each task contains a disjoint set of classes.
        The model needs to learn new classes while retaining knowledge of previous ones.
        
        Args:
            dataset: Source dataset to split into tasks
            classes_per_task: Number of classes per task
            
        Returns:
            List of tasks with disjoint class sets
            
        Raises:
            TaskGeneratorError: If task generation fails
        """
        try:
            logger.debug(f"Generating {self.num_tasks} class incremental tasks")
            
            # Get unique classes from dataset
            unique_classes = self._get_unique_classes(dataset)
            total_classes = len(unique_classes)
            
            # Validate configuration
            if classes_per_task * self.num_tasks > total_classes:
                raise TaskGeneratorError(
                    f"Cannot create {self.num_tasks} tasks with {classes_per_task} "
                    f"classes each from {total_classes} total classes"
                )
            
            # Shuffle classes if configured
            class_order = self.shuffle_classes(self.random_seed)
            if not class_order:
                class_order = sorted(unique_classes)
            
            # Generate tasks
            tasks = []
            for task_id in range(self.num_tasks):
                start_idx = task_id * classes_per_task
                end_idx = start_idx + classes_per_task
                task_classes = class_order[start_idx:end_idx]
                
                # Create task dataset
                task_dataset = self._create_class_subset(dataset, task_classes)
                task_dataloader = DataLoader(
                    task_dataset,
                    batch_size=self.config.get('batch_size', 32),
                    shuffle=True,
                    num_workers=self.config.get('num_workers', 0)
                )
                
                # Create task with metadata
                task = Task(
                    task_id=task_id,
                    data=task_dataloader,
                    classes=task_classes,
                    task_type=TaskType.CLASS_INCREMENTAL,
                    metadata={
                        'dataset_name': dataset.name,
                        'generation_seed': self.random_seed,
                        'total_tasks': self.num_tasks,
                        'classes_per_task': classes_per_task,
                        'sample_count': len(task_dataset)
                    }
                )
                
                tasks.append(task)
                logger.debug(f"Created task {task_id} with classes {task_classes}")
            
            # Validate task sequence
            if not self.validate_task_sequence(tasks):
                raise TaskGeneratorError("Generated task sequence validation failed")
            
            logger.debug(f"Successfully generated {len(tasks)} class incremental tasks")
            return tasks
            
        except Exception as e:
            logger.error(f"Failed to generate class incremental tasks: {e}")
            raise TaskGeneratorError(f"Class incremental task generation failed: {e}")
    
    def generate_domain_incremental_tasks(self, datasets: List[Dataset]) -> List[Task]:
        """
        Generate domain incremental learning tasks.
        
        In domain incremental learning, each task represents a different domain
        (e.g., different datasets) but with the same classes.
        
        Args:
            datasets: List of datasets representing different domains
            
        Returns:
            List of tasks with different domains
            
        Raises:
            TaskGeneratorError: If task generation fails
        """
        try:
            logger.debug(f"Generating {len(datasets)} domain incremental tasks")
            
            if len(datasets) != self.num_tasks:
                logger.warning(
                    f"Number of datasets ({len(datasets)}) doesn't match "
                    f"configured tasks ({self.num_tasks}), adjusting"
                )
                self.num_tasks = len(datasets)
            
            tasks = []
            for task_id, dataset in enumerate(datasets):
                # Create dataloader for the domain
                task_dataloader = DataLoader(
                    dataset,
                    batch_size=self.config.get('batch_size', 32),
                    shuffle=True,
                    num_workers=self.config.get('num_workers', 0)
                )
                
                # Get classes from dataset
                unique_classes = self._get_unique_classes(dataset)
                
                # Create task with metadata
                task = Task(
                    task_id=task_id,
                    data=task_dataloader,
                    classes=unique_classes,
                    task_type=TaskType.DOMAIN_INCREMENTAL,
                    metadata={
                        'dataset_name': dataset.name,
                        'domain_id': task_id,
                        'generation_seed': self.random_seed,
                        'total_tasks': self.num_tasks,
                        'sample_count': len(dataset)
                    }
                )
                
                tasks.append(task)
                logger.debug(f"Created domain task {task_id} with dataset {dataset.name}")
            
            # Validate task sequence
            if not self.validate_task_sequence(tasks):
                raise TaskGeneratorError("Generated task sequence validation failed")
            
            logger.debug(f"Successfully generated {len(tasks)} domain incremental tasks")
            return tasks
            
        except Exception as e:
            logger.error(f"Failed to generate domain incremental tasks: {e}")
            raise TaskGeneratorError(f"Domain incremental task generation failed: {e}")
    
    def generate_task_incremental_tasks(
        self, 
        dataset: Dataset, 
        task_configs: List[DictConfig]
    ) -> List[Task]:
        """
        Generate task incremental learning tasks.
        
        In task incremental learning, each task has a specific configuration
        and potentially different objectives or transformations.
        
        Args:
            dataset: Source dataset
            task_configs: List of configurations for each task
            
        Returns:
            List of configured tasks
            
        Raises:
            TaskGeneratorError: If task generation fails
        """
        try:
            logger.debug(f"Generating {len(task_configs)} task incremental tasks")
            
            if len(task_configs) != self.num_tasks:
                logger.warning(
                    f"Number of task configs ({len(task_configs)}) doesn't match "
                    f"configured tasks ({self.num_tasks}), adjusting"
                )
                self.num_tasks = len(task_configs)
            
            tasks = []
            for task_id, task_config in enumerate(task_configs):
                # Extract task-specific configuration
                task_name = task_config.get('name', f'task_{task_id}')
                task_classes = list(task_config.get('classes', [])) if task_config.get('classes') else []
                
                # Create task dataset based on classes
                if task_classes:
                    task_dataset = self._create_class_subset(dataset, task_classes)
                else:
                    task_dataset = dataset
                
                # Create dataloader with task-specific settings
                batch_size = task_config.get('batch_size', self.config.get('batch_size', 32))
                task_dataloader = DataLoader(
                    task_dataset,
                    batch_size=batch_size,
                    shuffle=task_config.get('shuffle', True),
                    num_workers=self.config.get('num_workers', 0)
                )
                
                # Create task with comprehensive metadata
                task = Task(
                    task_id=task_id,
                    data=task_dataloader,
                    classes=task_classes if task_classes else self._get_unique_classes(task_dataset),
                    task_type=TaskType.TASK_INCREMENTAL,
                    metadata={
                        'dataset_name': dataset.name,
                        'task_name': task_name,
                        'task_config': dict(task_config),
                        'generation_seed': self.random_seed,
                        'total_tasks': self.num_tasks,
                        'sample_count': len(task_dataset)
                    }
                )
                
                tasks.append(task)
                logger.debug(f"Created task incremental task {task_id}: {task_name}")
            
            # Validate task sequence
            if not self.validate_task_sequence(tasks):
                raise TaskGeneratorError("Generated task sequence validation failed")
            
            logger.debug(f"Successfully generated {len(tasks)} task incremental tasks")
            return tasks
            
        except Exception as e:
            logger.error(f"Failed to generate task incremental tasks: {e}")
            raise TaskGeneratorError(f"Task incremental task generation failed: {e}")
    
    def shuffle_classes(self, seed: Optional[int] = None) -> List[int]:
        """
        Shuffle class order for task generation.
        
        Args:
            seed: Random seed for reproducible shuffling
            
        Returns:
            Shuffled list of class indices
        """
        if seed is not None:
            self._set_random_seeds(seed)
        
        # Generate class order from configuration
        class_config = self.task_config.get('class_incremental', {})
        
        if not class_config.get('shuffle_classes', True):
            logger.debug("Class shuffling disabled in configuration")
            return []
        
        # Generate class indices for the configured number of tasks and classes
        total_classes = self.num_tasks * self.classes_per_task
        class_order = list(range(total_classes))
        
        # Shuffle the classes
        random.shuffle(class_order)
        
        # Cache the order only if no specific seed is provided for this call
        if seed is None:
            self._class_order = class_order
        
        logger.debug(f"Generated shuffled class order with {len(class_order)} classes")
        return class_order.copy()
    
    def validate_task_sequence(self, tasks: List[Task]) -> bool:
        """
        Validate the generated task sequence.
        
        Args:
            tasks: List of tasks to validate
            
        Returns:
            True if task sequence is valid, False otherwise
        """
        try:
            if not tasks:
                logger.error("Empty task sequence")
                return False
            
            # Check task IDs are sequential
            for i, task in enumerate(tasks):
                if task.task_id != i:
                    logger.error(f"Non-sequential task ID: expected {i}, got {task.task_id}")
                    return False
            
            # Validate based on task type
            task_type = tasks[0].task_type
            
            if task_type == TaskType.CLASS_INCREMENTAL:
                return self._validate_class_incremental_sequence(tasks)
            elif task_type == TaskType.DOMAIN_INCREMENTAL:
                return self._validate_domain_incremental_sequence(tasks)
            elif task_type == TaskType.TASK_INCREMENTAL:
                return self._validate_task_incremental_sequence(tasks)
            else:
                logger.error(f"Unknown task type: {task_type}")
                return False
                
        except Exception as e:
            logger.error(f"Task sequence validation failed: {e}")
            return False
    
    def _validate_class_incremental_sequence(self, tasks: List[Task]) -> bool:
        """Validate class incremental task sequence."""
        all_classes = set()
        
        for task in tasks:
            # Check for class overlap
            task_classes = set(task.classes)
            if all_classes & task_classes:
                logger.error(f"Class overlap detected in task {task.task_id}")
                return False
            
            all_classes.update(task_classes)
            
            # Validate task has data
            if task.get_sample_count() == 0:
                logger.error(f"Task {task.task_id} has no samples")
                return False
        
        logger.debug("Class incremental task sequence validation passed")
        return True
    
    def _validate_domain_incremental_sequence(self, tasks: List[Task]) -> bool:
        """Validate domain incremental task sequence."""
        # In domain incremental, all tasks should have the same classes
        if not tasks:
            return False
        
        reference_classes = set(tasks[0].classes)
        
        for task in tasks[1:]:
            task_classes = set(task.classes)
            if task_classes != reference_classes:
                logger.warning(f"Task {task.task_id} has different classes than reference")
                # Don't fail for domain incremental, just warn
        
        logger.debug("Domain incremental task sequence validation passed")
        return True
    
    def _validate_task_incremental_sequence(self, tasks: List[Task]) -> bool:
        """Validate task incremental task sequence."""
        # Task incremental is most flexible, just check basic properties
        for task in tasks:
            if task.get_sample_count() == 0:
                logger.error(f"Task {task.task_id} has no samples")
                return False
        
        logger.debug("Task incremental task sequence validation passed")
        return True
    
    def get_task_statistics(self, tasks: List[Task]) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the task sequence.
        
        Args:
            tasks: List of tasks to analyze
            
        Returns:
            Dictionary containing task statistics
        """
        try:
            if not tasks:
                return {}
            
            stats = {
                'num_tasks': len(tasks),
                'task_type': tasks[0].task_type.value,
                'total_samples': sum(task.get_sample_count() for task in tasks),
                'total_classes': len(set().union(*[task.classes for task in tasks])),
                'tasks_info': []
            }
            
            # Per-task statistics
            for task in tasks:
                task_info = {
                    'task_id': task.task_id,
                    'classes': task.classes,
                    'num_classes': len(task.classes),
                    'sample_count': task.get_sample_count(),
                    'class_distribution': task.get_class_distribution(),
                    'memory_usage_mb': task.get_memory_usage()
                }
                stats['tasks_info'].append(task_info)
            
            # Overall statistics
            samples_per_task = [task.get_sample_count() for task in tasks]
            stats['sample_statistics'] = {
                'mean_samples_per_task': np.mean(samples_per_task),
                'std_samples_per_task': np.std(samples_per_task),
                'min_samples_per_task': np.min(samples_per_task),
                'max_samples_per_task': np.max(samples_per_task)
            }
            
            # Class balance analysis
            if tasks[0].task_type == TaskType.CLASS_INCREMENTAL:
                classes_per_task = [len(task.classes) for task in tasks]
                stats['class_balance'] = {
                    'mean_classes_per_task': np.mean(classes_per_task),
                    'std_classes_per_task': np.std(classes_per_task),
                    'classes_per_task_uniform': len(set(classes_per_task)) == 1
                }
            
            # Cache statistics
            self._task_statistics = stats
            
            logger.debug(f"Generated statistics for {len(tasks)} tasks")
            return stats
            
        except Exception as e:
            logger.error(f"Failed to generate task statistics: {e}")
            return {}
    
    def visualize_task_sequence(
        self, 
        tasks: List[Task], 
        save_path: Optional[Path] = None
    ) -> None:
        """
        Visualize the task sequence and class distribution.
        
        Args:
            tasks: List of tasks to visualize
            save_path: Optional path to save the visualization
        """
        try:
            if not tasks:
                logger.warning("No tasks to visualize")
                return
            
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Task Sequence Analysis', fontsize=16)
            
            # 1. Sample count per task
            task_ids = [task.task_id for task in tasks]
            sample_counts = [task.get_sample_count() for task in tasks]
            
            axes[0, 0].bar(task_ids, sample_counts, color='skyblue', alpha=0.7)
            axes[0, 0].set_xlabel('Task ID')
            axes[0, 0].set_ylabel('Sample Count')
            axes[0, 0].set_title('Samples per Task')
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Classes per task
            classes_per_task = [len(task.classes) for task in tasks]
            
            axes[0, 1].bar(task_ids, classes_per_task, color='lightcoral', alpha=0.7)
            axes[0, 1].set_xlabel('Task ID')
            axes[0, 1].set_ylabel('Number of Classes')
            axes[0, 1].set_title('Classes per Task')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Class distribution heatmap (for class incremental)
            if tasks[0].task_type == TaskType.CLASS_INCREMENTAL:
                all_classes = sorted(set().union(*[task.classes for task in tasks]))
                class_matrix = np.zeros((len(tasks), len(all_classes)))
                
                for i, task in enumerate(tasks):
                    for cls in task.classes:
                        if cls in all_classes:
                            j = all_classes.index(cls)
                            class_matrix[i, j] = 1
                
                if HAS_SEABORN:
                    sns.heatmap(
                        class_matrix,
                        xticklabels=all_classes,
                        yticklabels=task_ids,
                        cmap='Blues',
                        ax=axes[1, 0],
                        cbar_kws={'label': 'Class Present'}
                    )
                else:
                    # Fallback to matplotlib imshow
                    im = axes[1, 0].imshow(class_matrix, cmap='Blues', aspect='auto')
                    axes[1, 0].set_xticks(range(len(all_classes)))
                    axes[1, 0].set_xticklabels(all_classes)
                    axes[1, 0].set_yticks(range(len(task_ids)))
                    axes[1, 0].set_yticklabels(task_ids)
                    plt.colorbar(im, ax=axes[1, 0], label='Class Present')
                
                axes[1, 0].set_xlabel('Classes')
                axes[1, 0].set_ylabel('Task ID')
                axes[1, 0].set_title('Class Distribution Matrix')
            else:
                axes[1, 0].text(0.5, 0.5, 'Class Distribution\nNot Applicable', 
                              ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Class Distribution')
            
            # 4. Memory usage per task
            memory_usage = [task.get_memory_usage() for task in tasks]
            
            axes[1, 1].bar(task_ids, memory_usage, color='lightgreen', alpha=0.7)
            axes[1, 1].set_xlabel('Task ID')
            axes[1, 1].set_ylabel('Memory Usage (MB)')
            axes[1, 1].set_title('Memory Usage per Task')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save or show
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.debug(f"Task sequence visualization saved to {save_path}")
            else:
                plt.show()
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Failed to visualize task sequence: {e}")
            plt.close()
    
    def create_replay_buffer(self, task: Task, buffer_size: int) -> Dataset:
        """
        Create a replay buffer from a task's data.
        
        Args:
            task: Task to create replay buffer from
            buffer_size: Size of the replay buffer
            
        Returns:
            Dataset containing replay buffer samples
            
        Raises:
            TaskGeneratorError: If replay buffer creation fails
        """
        try:
            if not self.replay_enabled:
                logger.warning("Replay buffer creation requested but replay is disabled")
                # Create a minimal valid dataset for empty replay buffer
                empty_data = torch.empty(1, 1)  # Minimal non-empty tensor
                empty_targets = torch.empty(1, dtype=torch.long)  # Minimal non-empty tensor
                
                class EmptyDataset(Dataset):
                    def __len__(self):
                        return 0
                    
                    def __getitem__(self, index):
                        raise IndexError("Empty replay buffer")
                
                empty_dataset = EmptyDataset("empty_replay_buffer", empty_data, empty_targets)
                return empty_dataset
            
            logger.debug(f"Creating replay buffer for task {task.task_id} with size {buffer_size}")
            
            # Collect all data from the task
            all_data = []
            all_targets = []
            
            for batch_data, batch_targets in task.data:
                if isinstance(batch_data, torch.Tensor):
                    all_data.append(batch_data)
                if isinstance(batch_targets, torch.Tensor):
                    all_targets.append(batch_targets)
            
            if not all_data:
                logger.warning(f"No data found in task {task.task_id} for replay buffer")
                return Dataset("empty_replay_buffer", torch.empty(0), torch.empty(0))
            
            # Concatenate all data
            full_data = torch.cat(all_data, dim=0)
            full_targets = torch.cat(all_targets, dim=0)
            
            # Select samples based on strategy
            selected_indices = self._select_replay_samples(
                full_data, full_targets, buffer_size, task.classes
            )
            
            # Create replay buffer dataset
            replay_data = full_data[selected_indices]
            replay_targets = full_targets[selected_indices]
            
            replay_buffer = Dataset(
                name=f"replay_buffer_task_{task.task_id}",
                data=replay_data,
                targets=replay_targets
            )
            
            logger.debug(f"Created replay buffer with {len(replay_buffer)} samples")
            return replay_buffer
            
        except Exception as e:
            logger.error(f"Failed to create replay buffer for task {task.task_id}: {e}")
            raise TaskGeneratorError(f"Replay buffer creation failed: {e}")
    
    def _select_replay_samples(
        self, 
        data: torch.Tensor, 
        targets: torch.Tensor, 
        buffer_size: int,
        task_classes: List[int]
    ) -> torch.Tensor:
        """Select samples for replay buffer based on selection strategy."""
        if self.selection_strategy == 'random':
            return self._random_selection(data, targets, buffer_size)
        elif self.selection_strategy == 'herding':
            return self._herding_selection(data, targets, buffer_size, task_classes)
        elif self.selection_strategy == 'gradient_based':
            return self._gradient_based_selection(data, targets, buffer_size, task_classes)
        else:
            logger.warning(f"Unknown selection strategy: {self.selection_strategy}, using random")
            return self._random_selection(data, targets, buffer_size)
    
    def _random_selection(
        self, 
        data: torch.Tensor, 
        targets: torch.Tensor, 
        buffer_size: int
    ) -> torch.Tensor:
        """Random sample selection for replay buffer."""
        total_samples = data.shape[0]
        if buffer_size >= total_samples:
            return torch.arange(total_samples)
        
        indices = torch.randperm(total_samples)[:buffer_size]
        return indices
    
    def _herding_selection(
        self, 
        data: torch.Tensor, 
        targets: torch.Tensor, 
        buffer_size: int,
        task_classes: List[int]
    ) -> torch.Tensor:
        """Herding-based sample selection for replay buffer."""
        # Simplified herding: select samples closest to class centroids
        selected_indices = []
        samples_per_class = buffer_size // len(task_classes)
        
        for cls in task_classes:
            class_mask = targets == cls
            class_data = data[class_mask]
            class_indices = torch.where(class_mask)[0]
            
            if len(class_data) == 0:
                continue
            
            # Compute class centroid
            centroid = class_data.mean(dim=0)
            
            # Find samples closest to centroid
            distances = torch.norm(class_data - centroid, dim=1)
            _, sorted_indices = torch.sort(distances)
            
            # Select closest samples
            selected_count = min(samples_per_class, len(class_indices))
            selected_indices.extend(class_indices[sorted_indices[:selected_count]].tolist())
        
        return torch.tensor(selected_indices)
    
    def _gradient_based_selection(
        self, 
        data: torch.Tensor, 
        targets: torch.Tensor, 
        buffer_size: int,
        task_classes: List[int]
    ) -> torch.Tensor:
        """Gradient-based sample selection for replay buffer."""
        # Simplified gradient-based selection: random for now
        # In practice, this would require model gradients
        logger.warning("Gradient-based selection not fully implemented, using random")
        return self._random_selection(data, targets, buffer_size)
    
    def _get_unique_classes(self, dataset: Dataset) -> List[int]:
        """Get unique classes from a dataset."""
        try:
            if hasattr(dataset, 'targets'):
                if isinstance(dataset.targets, torch.Tensor):
                    return sorted(torch.unique(dataset.targets).tolist())
                else:
                    return sorted(set(dataset.targets))
            else:
                # Iterate through dataset to find unique classes
                all_targets = []
                try:
                    # Try to iterate through the dataset
                    iterator = iter(dataset)
                    for _, target in iterator:
                        if isinstance(target, torch.Tensor):
                            all_targets.extend(target.tolist() if target.dim() > 0 else [target.item()])
                        else:
                            all_targets.append(target)
                    return sorted(set(all_targets))
                except (TypeError, AttributeError) as e:
                    # Handle mock objects or other non-iterable datasets
                    logger.warning(f"Dataset is not properly iterable: {e}, returning empty class list")
                    return []
        except Exception as e:
            logger.error(f"Failed to get unique classes: {e}")
            return []
    
    def _create_class_subset(self, dataset: Dataset, classes: List[int]) -> Dataset:
        """Create a subset of dataset containing only specified classes."""
        try:
            # Find indices of samples belonging to specified classes
            indices = []
            
            if hasattr(dataset, 'targets'):
                targets = dataset.targets
                if isinstance(targets, torch.Tensor):
                    for cls in classes:
                        class_indices = torch.where(targets == cls)[0].tolist()
                        indices.extend(class_indices)
                else:
                    for i, target in enumerate(targets):
                        if target in classes:
                            indices.append(i)
            else:
                # Iterate through dataset
                for i, (_, target) in enumerate(dataset):
                    if isinstance(target, torch.Tensor):
                        target = target.item()
                    if target in classes:
                        indices.append(i)
            
            # Create subset
            subset = Subset(dataset, indices)
            
            # Wrap in Dataset class
            subset_data = []
            subset_targets = []
            
            for idx in indices:
                data_item, target_item = dataset[idx]
                subset_data.append(data_item.unsqueeze(0) if data_item.dim() == len(dataset.data.shape[1:]) else data_item)
                subset_targets.append(target_item)
            
            if subset_data:
                subset_data_tensor = torch.cat(subset_data, dim=0)
                subset_targets_tensor = torch.tensor(subset_targets)
                subset_dataset = Dataset(
                    name=f"{dataset.name}_classes_{classes}",
                    data=subset_data_tensor,
                    targets=subset_targets_tensor
                )
                return subset_dataset
            else:
                # Create minimal valid tensors for empty subset
                empty_data = torch.empty(1, *dataset.data.shape[1:])
                empty_targets = torch.empty(1, dtype=torch.long)
                
                class EmptySubset(Dataset):
                    def __len__(self):
                        return 0
                    
                    def __getitem__(self, index):
                        raise IndexError("Empty class subset")
                
                return EmptySubset(
                    name=f"{dataset.name}_classes_{classes}",
                    data=empty_data,
                    targets=empty_targets
                )
            
        except Exception as e:
            logger.error(f"Failed to create class subset: {e}")
            # Return minimal valid dataset on failure
            empty_data = torch.empty(1, *dataset.data.shape[1:]) if len(dataset.data.shape) > 1 else torch.empty(1, 1)
            empty_targets = torch.empty(1, dtype=torch.long)
            
            class EmptySubset(Dataset):
                def __len__(self):
                    return 0
                
                def __getitem__(self, index):
                    raise IndexError("Empty class subset")
            
            return EmptySubset(
                name=f"{dataset.name}_empty_subset",
                data=empty_data,
                targets=empty_targets
            )
