# fedcl/data/results.py
"""
Results management implementation for FedCL framework.

This module provides TaskResults and ExperimentResults classes for managing
and analyzing experimental results in federated continual learning.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timezone
from pathlib import Path
import copy
import pickle
import json
import logging
from omegaconf import DictConfig
import torch
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


class ResultsError(Exception):
    """Base exception for results-related errors."""
    pass


class ResultsSerializationError(ResultsError):
    """Exception raised when results serialization/deserialization fails."""
    pass


class ResultsVisualizationError(ResultsError):
    """Exception raised when results visualization fails."""
    pass


@dataclass
class TaskResults:
    """
    Container for results from a single task in continual learning.
    
    This class stores all relevant metrics and metadata for a single task's
    training and evaluation results.
    
    Attributes:
        task_id: Unique identifier for the task
        metrics: Dictionary of metric names to their values
        training_time: Time taken for training in seconds
        memory_usage: Peak memory usage during training in MB
        model_size: Size of the model in bytes
        convergence_step: Training step at which convergence was achieved
        metadata: Additional metadata about the task results
    """
    
    task_id: int
    metrics: Dict[str, float] = field(default_factory=dict)
    training_time: float = 0.0
    memory_usage: float = 0.0
    model_size: int = 0
    convergence_step: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization validation."""
        if self.task_id < 0:
            raise ValueError(f"Task ID must be non-negative, got {self.task_id}")
        
        if self.training_time < 0:
            raise ValueError(f"Training time must be non-negative, got {self.training_time}")
        
        if self.memory_usage < 0:
            raise ValueError(f"Memory usage must be non-negative, got {self.memory_usage}")
        
        if self.model_size < 0:
            raise ValueError(f"Model size must be non-negative, got {self.model_size}")
        
        logger.debug(f"TaskResults initialized for task {self.task_id} with {len(self.metrics)} metrics")
    
    def get_metric(self, name: str, default: Any = None) -> Any:
        """
        Get a specific metric value.
        
        Args:
            name: Name of the metric
            default: Default value if metric doesn't exist
            
        Returns:
            Metric value or default
        """
        return self.metrics.get(name, default)
    
    def add_metric(self, name: str, value: float) -> None:
        """
        Add or update a metric.
        
        Args:
            name: Name of the metric
            value: Value of the metric
        """
        self.metrics[name] = float(value)
        logger.debug(f"Added metric '{name}={value}' to task {self.task_id}")
    
    def update_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Update multiple metrics at once.
        
        Args:
            metrics: Dictionary of metrics to update
        """
        for name, value in metrics.items():
            self.add_metric(name, value)
        logger.debug(f"Updated {len(metrics)} metrics for task {self.task_id}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert TaskResults to dictionary representation.
        
        Returns:
            Dictionary representation of the results
        """
        try:
            result_dict = asdict(self)
            result_dict['timestamp'] = datetime.now(timezone.utc).isoformat()
            logger.debug(f"Converted TaskResults for task {self.task_id} to dictionary")
            return result_dict
        except Exception as e:
            logger.error(f"Failed to convert TaskResults to dict: {e}")
            raise ResultsSerializationError(f"Failed to convert to dictionary: {e}")
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TaskResults:
        """
        Create TaskResults from dictionary representation.
        
        Args:
            data: Dictionary containing results data
            
        Returns:
            TaskResults instance
            
        Raises:
            ResultsSerializationError: If deserialization fails
        """
        try:
            # Remove timestamp if present (it's not part of the dataclass)
            data = copy.deepcopy(data)
            data.pop('timestamp', None)
            
            results = cls(**data)
            logger.debug(f"Created TaskResults for task {results.task_id} from dictionary")
            return results
            
        except Exception as e:
            logger.error(f"Failed to create TaskResults from dict: {e}")
            raise ResultsSerializationError(f"Failed to create from dictionary: {e}")
    
    def merge_with(self, other: TaskResults) -> TaskResults:
        """
        Merge with another TaskResults instance.
        
        Args:
            other: Another TaskResults instance to merge with
            
        Returns:
            New TaskResults instance with merged data
            
        Raises:
            ResultsError: If merge fails or task IDs don't match
        """
        if not isinstance(other, TaskResults):
            raise ResultsError(f"Can only merge with TaskResults, got {type(other)}")
        
        if self.task_id != other.task_id:
            raise ResultsError(f"Cannot merge results from different tasks: {self.task_id} vs {other.task_id}")
        
        try:
            # Merge metrics (other takes precedence for conflicts)
            merged_metrics = copy.deepcopy(self.metrics)
            merged_metrics.update(other.metrics)
            
            # Merge metadata (other takes precedence for conflicts)
            merged_metadata = copy.deepcopy(self.metadata)
            merged_metadata.update(other.metadata)
            
            # Use values from other for scalar fields (assumes other is more recent)
            merged_results = TaskResults(
                task_id=self.task_id,
                metrics=merged_metrics,
                training_time=max(self.training_time, other.training_time),
                memory_usage=max(self.memory_usage, other.memory_usage),
                model_size=max(self.model_size, other.model_size),
                convergence_step=max(self.convergence_step, other.convergence_step),
                metadata=merged_metadata
            )
            
            logger.debug(f"Merged TaskResults for task {self.task_id}")
            return merged_results
            
        except Exception as e:
            logger.error(f"Failed to merge TaskResults: {e}")
            raise ResultsError(f"Failed to merge results: {e}")
    
    def __repr__(self) -> str:
        """String representation of TaskResults."""
        return (f"TaskResults(task_id={self.task_id}, metrics={len(self.metrics)}, "
                f"training_time={self.training_time:.2f}s)")
        
    def get_summary(self) -> str:
        """
        获取任务结果摘要
        
        Returns:
            str: 任务结果的简要摘要信息
        """
        metrics_summary = ", ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" 
                                    for k, v in self.metrics.items()])
        
        training_time_str = f"{self.training_time:.2f}s" if hasattr(self, 'training_time') and self.training_time else "N/A"
        
        return f"Task {self.task_id} - Metrics: {metrics_summary}, Training time: {training_time_str}"


@dataclass
class ExperimentResults:
    """
    Container for results from a complete experiment.
    
    This class manages results from multiple tasks and provides functionality
    for analysis, visualization, and persistence.
    
    Attributes:
        experiment_id: Unique identifier for the experiment
        config: Configuration used for the experiment
        metrics: Global metrics tracked across tasks
        task_results: Results from individual tasks
        checkpoints: List of checkpoint file paths
        artifacts: Additional experiment artifacts
        start_time: When the experiment started
        end_time: When the experiment ended (None if ongoing)
    """
    
    experiment_id: str
    config: DictConfig
    metrics: Dict[str, List[float]] = field(default_factory=dict)
    task_results: List[TaskResults] = field(default_factory=list)
    checkpoints: List[Path] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        if not self.experiment_id or not isinstance(self.experiment_id, str):
            raise ValueError(f"Experiment ID must be a non-empty string, got {self.experiment_id}")
        
        # Ensure all paths are Path objects
        self.checkpoints = [Path(p) if not isinstance(p, Path) else p for p in self.checkpoints]
        
        logger.debug(f"ExperimentResults initialized for experiment '{self.experiment_id}'")
    
    def add_task_result(self, result: TaskResults) -> None:
        """
        Add results from a single task.
        
        Args:
            result: TaskResults instance to add
            
        Raises:
            ResultsError: If adding task result fails
        """
        if not isinstance(result, TaskResults):
            raise ResultsError(f"Expected TaskResults, got {type(result)}")
        
        try:
            # Check if results for this task already exist
            existing_idx = None
            for i, existing_result in enumerate(self.task_results):
                if existing_result.task_id == result.task_id:
                    existing_idx = i
                    break
            
            if existing_idx is not None:
                # Merge with existing results
                self.task_results[existing_idx] = self.task_results[existing_idx].merge_with(result)
                logger.debug(f"Merged results for task {result.task_id}")
            else:
                # Add new results
                self.task_results.append(result)
                logger.debug(f"Added new results for task {result.task_id}")
            
            # Update global metrics
            for metric_name, metric_value in result.metrics.items():
                if metric_name not in self.metrics:
                    self.metrics[metric_name] = []
                
                # Ensure the list is long enough for this task
                while len(self.metrics[metric_name]) <= result.task_id:
                    self.metrics[metric_name].append(0.0)
                
                self.metrics[metric_name][result.task_id] = metric_value
            
        except Exception as e:
            logger.error(f"Failed to add task result: {e}")
            raise ResultsError(f"Failed to add task result: {e}")
    
    def get_best_result(self, metric: str, maximize: bool = True) -> Optional[TaskResults]:
        """
        Get the task result with the best value for a specific metric.
        
        Args:
            metric: Name of the metric to optimize
            maximize: If True, find maximum value; if False, find minimum
            
        Returns:
            TaskResults with the best metric value, or None if not found
        """
        if not self.task_results:
            return None
        
        try:
            valid_results = [r for r in self.task_results if metric in r.metrics]
            if not valid_results:
                logger.warning(f"No results found with metric '{metric}'")
                return None
            
            best_result = max(valid_results, key=lambda r: r.metrics[metric]) if maximize else \
                         min(valid_results, key=lambda r: r.metrics[metric])
            
            logger.debug(f"Found best result for metric '{metric}': task {best_result.task_id}")
            return best_result
            
        except Exception as e:
            logger.error(f"Failed to find best result for metric '{metric}': {e}")
            return None
    
    def get_task_result(self, task_id: int) -> Optional[TaskResults]:
        """
        Get results for a specific task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            TaskResults for the specified task, or None if not found
        """
        for result in self.task_results:
            if result.task_id == task_id:
                return result
        return None
    
    def generate_summary(self) -> Dict[str, Any]:
        """
        Generate a comprehensive summary of the experiment results.
        
        Returns:
            Dictionary containing experiment summary
        """
        try:
            duration = None
            if self.end_time:
                duration = (self.end_time - self.start_time).total_seconds()
            
            summary = {
                'experiment_id': self.experiment_id,
                'start_time': self.start_time.isoformat(),
                'end_time': self.end_time.isoformat() if self.end_time else None,
                'duration_seconds': duration,
                'num_tasks': len(self.task_results),
                'num_checkpoints': len(self.checkpoints),
                'num_artifacts': len(self.artifacts),
                'status': 'completed' if self.end_time else 'ongoing'
            }
            
            # Task-level statistics
            if self.task_results:
                training_times = [r.training_time for r in self.task_results]
                memory_usages = [r.memory_usage for r in self.task_results]
                model_sizes = [r.model_size for r in self.task_results]
                
                summary.update({
                    'total_training_time': sum(training_times),
                    'avg_task_training_time': np.mean(training_times),
                    'max_memory_usage': max(memory_usages) if memory_usages else 0,
                    'avg_memory_usage': np.mean(memory_usages) if memory_usages else 0,
                    'final_model_size': model_sizes[-1] if model_sizes else 0,
                })
            
            # Metric summaries
            metric_summaries = {}
            for metric_name, values in self.metrics.items():
                if values:
                    metric_summaries[metric_name] = {
                        'final_value': values[-1],
                        'best_value': max(values),
                        'worst_value': min(values),
                        'mean_value': np.mean(values),
                        'std_value': np.std(values)
                    }
            
            summary['metric_summaries'] = metric_summaries
            
            logger.debug(f"Generated summary for experiment '{self.experiment_id}'")
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            return {'error': str(e), 'experiment_id': self.experiment_id}
    
    def plot_learning_curves(self, save_path: Optional[Path] = None, metrics: Optional[List[str]] = None) -> None:
        """
        Plot learning curves for the experiment metrics.
        
        Args:
            save_path: Path to save the plot (if None, display only)
            metrics: List of metrics to plot (if None, plot all)
            
        Raises:
            ResultsVisualizationError: If plotting fails
        """
        try:
            if not self.metrics:
                logger.warning("No metrics available for plotting")
                return
            
            # Determine which metrics to plot
            plot_metrics = metrics if metrics else list(self.metrics.keys())
            plot_metrics = [m for m in plot_metrics if m in self.metrics and self.metrics[m]]
            
            if not plot_metrics:
                logger.warning("No valid metrics found for plotting")
                return
            
            # Create subplots
            n_metrics = len(plot_metrics)
            n_cols = min(3, n_metrics)
            n_rows = (n_metrics + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
            if n_metrics == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes if n_metrics > 1 else [axes]
            else:
                axes = axes.flatten()
            
            # Plot each metric
            for i, metric_name in enumerate(plot_metrics):
                ax = axes[i]
                values = self.metrics[metric_name]
                tasks = list(range(len(values)))
                
                ax.plot(tasks, values, marker='o', linewidth=2, markersize=4)
                ax.set_title(f'{metric_name.replace("_", " ").title()}')
                ax.set_xlabel('Task')
                ax.set_ylabel(metric_name)
                ax.grid(True, alpha=0.3)
                
                # Add best value annotation
                best_idx = np.argmax(values)
                ax.annotate(f'Best: {values[best_idx]:.3f}', 
                           xy=(best_idx, values[best_idx]),
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            
            # Hide unused subplots
            for i in range(n_metrics, len(axes)):
                axes[i].set_visible(False)
            
            plt.suptitle(f'Learning Curves - Experiment {self.experiment_id}', fontsize=16)
            plt.tight_layout()
            
            # Save or display
            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.debug(f"Saved learning curves to {save_path}")
            else:
                plt.show()
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Failed to plot learning curves: {e}")
            raise ResultsVisualizationError(f"Failed to plot learning curves: {e}")
    
    def _serialize_for_json(self, obj):
        """
        Properly serialize objects for JSON compatibility.
        
        Args:
            obj: Object to serialize
            
        Returns:
            JSON-serializable object
        """
        try:
            from omegaconf import DictConfig, OmegaConf
            
            if isinstance(obj, DictConfig):
                # Convert DictConfig to regular dict while preserving structure
                return OmegaConf.to_container(obj, resolve=True)
            elif hasattr(obj, 'to_dict') and callable(getattr(obj, 'to_dict')):
                return obj.to_dict()
            elif isinstance(obj, (list, tuple)):
                return [self._serialize_for_json(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: self._serialize_for_json(value) for key, value in obj.items()}
            elif hasattr(obj, '__dict__'):
                # For custom objects, try to serialize their attributes
                return {key: self._serialize_for_json(value) 
                    for key, value in obj.__dict__.items() 
                    if not key.startswith('_')}
            else:
                # For basic types and numpy types
                if hasattr(obj, 'item'):  # numpy scalar
                    return obj.item()
                return obj
                
        except Exception as e:
            logger.warning(f"Failed to serialize {type(obj)}: {e}, falling back to string")
            return str(obj)
    
    def save_to_file(self, path: Path) -> None:
        """
        Save experiment results to file.
        
        Args:
            path: Path to save the results
            
        Raises:
            ResultsSerializationError: If saving fails
        """
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare data for serialization with proper handling
            save_data = {
                'experiment_id': self.experiment_id,
                'config': self._serialize_for_json(self.config),  # 使用新的序列化方法
                'metrics': copy.deepcopy(self.metrics),
                'task_results': [result.to_dict() for result in self.task_results],
                'checkpoints': [str(p) for p in self.checkpoints],
                'artifacts': self._serialize_for_json(self.artifacts),  # 也处理artifacts
                'start_time': self.start_time.isoformat(),
                'end_time': self.end_time.isoformat() if self.end_time else None,
                'summary': self.generate_summary()
            }
            
            # Save based on file extension
            if path.suffix.lower() == '.json':
                with open(path, 'w') as f:
                    # 移除 default=str，因为我们已经预处理了数据
                    json.dump(save_data, f, indent=2, ensure_ascii=False)
            else:
                # Default to pickle
                with open(path, 'wb') as f:
                    pickle.dump(save_data, f)
            
            logger.debug(f"Saved experiment results to {path}")
            
        except Exception as e:
            logger.error(f"Failed to save results to {path}: {e}")
            raise ResultsSerializationError(f"Failed to save results: {e}")
    
    @classmethod
    def load_from_file(cls, path: Path) -> ExperimentResults:
        """
        Load experiment results from file.
        
        Args:
            path: Path to load the results from
            
        Returns:
            ExperimentResults instance
            
        Raises:
            ResultsSerializationError: If loading fails
        """
        try:
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"Results file not found: {path}")
            
            # Load based on file extension
            if path.suffix.lower() == '.json':
                with open(path, 'r') as f:
                    data = json.load(f)
            else:
                # Default to pickle
                with open(path, 'rb') as f:
                    data = pickle.load(f)
            
            # Reconstruct ExperimentResults with proper config handling
            config_data = data.get('config', {})
            if config_data:
                # 确保config重新创建为DictConfig
                from omegaconf import DictConfig
                config = DictConfig(config_data)
            else:
                config = DictConfig({})
                
            experiment = cls(
                experiment_id=data['experiment_id'],
                config=config,
                metrics=data.get('metrics', {}),
                checkpoints=[Path(p) for p in data.get('checkpoints', [])],
                artifacts=data.get('artifacts', {}),
                start_time=datetime.fromisoformat(data['start_time']),
                end_time=datetime.fromisoformat(data['end_time']) if data.get('end_time') else None
            )
            
            # Reconstruct task results
            for task_data in data.get('task_results', []):
                task_result = TaskResults.from_dict(task_data)
                experiment.task_results.append(task_result)
            
            logger.debug(f"Loaded experiment results from {path}")
            return experiment
            
        except Exception as e:
            logger.error(f"Failed to load results from {path}: {e}")
            raise ResultsSerializationError(f"Failed to load results: {e}")
    
    def finalize(self) -> None:
        """Mark the experiment as completed."""
        if self.end_time is None:
            self.end_time = datetime.now(timezone.utc)
            logger.debug(f"Experiment '{self.experiment_id}' finalized")
    
    def add_checkpoint(self, checkpoint_path: Path) -> None:
        """
        Add a checkpoint path to the experiment.
        
        Args:
            checkpoint_path: Path to the checkpoint file
        """
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path not in self.checkpoints:
            self.checkpoints.append(checkpoint_path)
            logger.debug(f"Added checkpoint: {checkpoint_path}")
    
    def add_artifact(self, name: str, artifact: Any) -> None:
        """
        Add an artifact to the experiment.
        
        Args:
            name: Name of the artifact
            artifact: Artifact data
        """
        self.artifacts[name] = artifact
        logger.debug(f"Added artifact: {name}")
    
    def __repr__(self) -> str:
        """String representation of ExperimentResults."""
        status = "completed" if self.end_time else "ongoing"
        return (f"ExperimentResults(id='{self.experiment_id}', tasks={len(self.task_results)}, "
                f"status={status})")