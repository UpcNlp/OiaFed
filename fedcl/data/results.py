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
                'status': '完成' if self.end_time else 'ongoing'
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
        status = "完成" if self.end_time else "ongoing"
        return (f"ExperimentResults(id='{self.experiment_id}', tasks={len(self.task_results)}, "
                f"status={status})")


@dataclass
class RoundResults:
    """
    Container for results from a single federation round.
    
    This class stores all relevant metrics and metadata for a single federation round's
    training and aggregation results.
    """
    round_number: int
    aggregated_metrics: Dict[str, float] = field(default_factory=dict)
    participating_clients: List[str] = field(default_factory=list)
    convergence_metrics: Dict[str, float] = field(default_factory=dict)
    round_duration: float = 0.0
    client_updates_received: int = 0
    client_updates_expected: int = 0
    client_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    is_successful: bool = True
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def get_metric(self, metric_name: str, default: float = 0.0) -> float:
        """
        Get a specific aggregated metric.
        
        Args:
            metric_name: Name of the metric
            default: Default value if metric not found
            
        Returns:
            Metric value
        """
        return self.aggregated_metrics.get(metric_name, default)
    
    def add_client_result(self, client_id: str, result: Dict[str, Any]) -> None:
        """
        Add client result to this round.
        
        Args:
            client_id: ID of the client
            result: Client training result
        """
        self.client_results[client_id] = result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)


@dataclass  
class FederationResults:
    """
    Container for overall federation learning results.
    
    This class stores all relevant metrics and metadata for the entire federation
    learning process across all rounds.
    """
    federation_state: Any  # Should be FederationState enum
    total_rounds: int
    best_round: int = 0
    best_metrics: Dict[str, float] = field(default_factory=dict)
    final_metrics: Dict[str, float] = field(default_factory=dict)
    round_results: List[RoundResults] = field(default_factory=list)
    convergence_achieved: bool = False
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    def get_best_metric(self, metric_name: str) -> Optional[float]:
        """
        Get the best value for a specific metric across all rounds.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            Best metric value or None if not found
        """
        if not self.round_results:
            return None
        
        values = [
            round_result.aggregated_metrics.get(metric_name)
            for round_result in self.round_results
            if metric_name in round_result.aggregated_metrics
        ]
        
        return max(values) if values else None
    
    def get_final_metric(self, metric_name: str) -> Optional[float]:
        """
        Get the final value for a specific metric.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            Final metric value or None if not found
        """
        return self.final_metrics.get(metric_name)
    
    def add_round_result(self, round_result: RoundResults) -> None:
        """
        Add a round result to the federation results.
        
        Args:
            round_result: Round result to add
        """
        self.round_results.append(round_result)
        
        # Update best metrics if this round is better
        for metric_name, value in round_result.aggregated_metrics.items():
            if (metric_name not in self.best_metrics or 
                value > self.best_metrics[metric_name]):
                self.best_metrics[metric_name] = value
                if metric_name == "accuracy":  # Primary metric for best round
                    self.best_round = round_result.round_number
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)
    

# 在您的 fedcl/data/results.py 文件末尾添加以下内容：

@dataclass
class SweepResults:
    """
    Container for parameter sweep experiment results.
    
    This class manages results from parameter sweep experiments, providing functionality
    for parameter analysis, best result discovery, and performance comparison.
    
    Attributes:
        sweep_id: Unique identifier for the sweep
        base_config: Base configuration used for all experiments
        sweep_config: Parameter sweep configuration
        experiment_results: Dictionary mapping parameter combinations to results
        failed_experiments: Dictionary mapping parameter combinations to error messages
        best_result: Best experiment result found so far
        best_parameters: Parameters that produced the best result
        start_time: When the sweep started
        end_time: When the sweep ended (None if ongoing)
    """
    
    sweep_id: str
    base_config: DictConfig
    sweep_config: Dict[str, List[Any]]
    experiment_results: Dict[str, ExperimentResults] = field(default_factory=dict)
    failed_experiments: Dict[str, str] = field(default_factory=dict)
    best_result: Optional[ExperimentResults] = None
    best_parameters: Optional[Dict[str, Any]] = None
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        if not self.sweep_id or not isinstance(self.sweep_id, str):
            raise ValueError(f"Sweep ID must be a non-empty string, got {self.sweep_id}")
        
        if not self.sweep_config:
            raise ValueError("Sweep config cannot be empty")
        
        logger.debug(f"SweepResults initialized for sweep '{self.sweep_id}'")
    
    def add_experiment_result(self, parameters: Dict[str, Any], result: ExperimentResults) -> None:
        """
        Add experiment result for a specific parameter combination.
        
        Args:
            parameters: Parameter combination used for this experiment
            result: Experiment results
            
        Raises:
            ResultsError: If adding experiment result fails
        """
        if not isinstance(result, ExperimentResults):
            raise ResultsError(f"Expected ExperimentResults, got {type(result)}")
        
        try:
            param_key = self._parameters_to_key(parameters)
            self.experiment_results[param_key] = result
            
            # Update best result if this is better
            self._update_best_result(parameters, result)
            
            logger.debug(f"Added experiment result for parameters: {parameters}")
            
        except Exception as e:
            logger.error(f"Failed to add experiment result: {e}")
            raise ResultsError(f"Failed to add experiment result: {e}")
    
    def add_failed_experiment(self, parameters: Dict[str, Any], error_message: str) -> None:
        """
        Add failed experiment for a specific parameter combination.
        
        Args:
            parameters: Parameter combination that failed
            error_message: Error message describing the failure
        """
        try:
            param_key = self._parameters_to_key(parameters)
            self.failed_experiments[param_key] = error_message
            
            logger.debug(f"Added failed experiment for parameters: {parameters}")
            
        except Exception as e:
            logger.error(f"Failed to add failed experiment: {e}")
    
    def get_best_result(self, metric: str = 'accuracy', maximize: bool = True) -> Tuple[Optional[Dict[str, Any]], Optional[ExperimentResults]]:
        """
        Get the best experiment result based on a specific metric.
        
        Args:
            metric: Metric name to optimize
            maximize: If True, find maximum value; if False, find minimum
            
        Returns:
            Tuple of (best_parameters, best_experiment_result)
        """
        if not self.experiment_results:
            return None, None
        
        try:
            best_params = None
            best_result = None
            best_value = float('-inf') if maximize else float('inf')
            
            for param_key, result in self.experiment_results.items():
                # Get metric value from the experiment result
                metric_value = None
                
                # Try to get from final metrics first
                if hasattr(result, 'task_results') and result.task_results:
                    final_task = result.task_results[-1]
                    metric_value = final_task.get_metric(metric)
                
                # If not found, try summary metrics
                if metric_value is None:
                    summary = result.generate_summary()
                    metric_summaries = summary.get('metric_summaries', {})
                    if metric in metric_summaries:
                        metric_value = metric_summaries[metric].get('final_value')
                
                if metric_value is not None:
                    if (maximize and metric_value > best_value) or (not maximize and metric_value < best_value):
                        best_value = metric_value
                        best_params = self._key_to_parameters(param_key)
                        best_result = result
            
            logger.debug(f"Found best result for metric '{metric}': {best_value}")
            return best_params, best_result
            
        except Exception as e:
            logger.error(f"Failed to find best result: {e}")
            return None, None
    
    def get_sorted_results(self, metric: str = 'accuracy', maximize: bool = True) -> List[Tuple[Dict[str, Any], ExperimentResults, float]]:
        """
        Get experiment results sorted by a specific metric.
        
        Args:
            metric: Metric name to sort by
            maximize: If True, sort in descending order; if False, ascending
            
        Returns:
            List of (parameters, result, metric_value) tuples sorted by metric
        """
        results_with_metrics = []
        
        try:
            for param_key, result in self.experiment_results.items():
                # Get metric value
                metric_value = None
                
                if hasattr(result, 'task_results') and result.task_results:
                    final_task = result.task_results[-1]
                    metric_value = final_task.get_metric(metric)
                
                if metric_value is None:
                    summary = result.generate_summary()
                    metric_summaries = summary.get('metric_summaries', {})
                    if metric in metric_summaries:
                        metric_value = metric_summaries[metric].get('final_value')
                
                if metric_value is not None:
                    params = self._key_to_parameters(param_key)
                    results_with_metrics.append((params, result, metric_value))
            
            # Sort by metric value
            results_with_metrics.sort(key=lambda x: x[2], reverse=maximize)
            
            logger.debug(f"Sorted {len(results_with_metrics)} results by metric '{metric}'")
            return results_with_metrics
            
        except Exception as e:
            logger.error(f"Failed to sort results: {e}")
            return []
    
    def get_parameter_analysis(self, metric: str = 'accuracy') -> Dict[str, Dict[str, Any]]:
        """
        Analyze the impact of different parameters on the specified metric.
        
        Args:
            metric: Metric name to analyze
            
        Returns:
            Dictionary with parameter analysis results
        """
        analysis = {}
        
        try:
            # Group results by parameter values
            for param_name in self.sweep_config.keys():
                param_analysis = {}
                param_groups = {}
                
                # Group experiments by this parameter value
                for param_key, result in self.experiment_results.items():
                    params = self._key_to_parameters(param_key)
                    param_value = params.get(param_name)
                    
                    if param_value is not None:
                        if param_value not in param_groups:
                            param_groups[param_value] = []
                        
                        # Get metric value
                        metric_value = None
                        if hasattr(result, 'task_results') and result.task_results:
                            final_task = result.task_results[-1]
                            metric_value = final_task.get_metric(metric)
                        
                        if metric_value is not None:
                            param_groups[param_value].append(metric_value)
                
                # Calculate statistics for each parameter value
                for param_value, metric_values in param_groups.items():
                    if metric_values:
                        param_analysis[str(param_value)] = {
                            'mean': np.mean(metric_values),
                            'std': np.std(metric_values),
                            'min': np.min(metric_values),
                            'max': np.max(metric_values),
                            'count': len(metric_values)
                        }
                
                analysis[param_name] = param_analysis
            
            logger.debug(f"Generated parameter analysis for metric '{metric}'")
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to generate parameter analysis: {e}")
            return {}
    
    def generate_summary(self) -> Dict[str, Any]:
        """
        Generate a comprehensive summary of the sweep results.
        
        Returns:
            Dictionary containing sweep summary
        """
        try:
            duration = None
            if self.end_time:
                duration = (self.end_time - self.start_time).total_seconds()
            
            total_experiments = len(self.experiment_results) + len(self.failed_experiments)
            success_rate = len(self.experiment_results) / total_experiments if total_experiments > 0 else 0
            
            summary = {
                'sweep_id': self.sweep_id,
                'start_time': self.start_time.isoformat(),
                'end_time': self.end_time.isoformat() if self.end_time else None,
                'duration_seconds': duration,
                'total_experiments': total_experiments,
                'successful_experiments': len(self.experiment_results),
                'failed_experiments': len(self.failed_experiments),
                'success_rate': success_rate,
                'sweep_parameters': list(self.sweep_config.keys()),
                'parameter_space_size': self._calculate_parameter_space_size(),
                'status': '完成' if self.end_time else 'ongoing'
            }
            
            # Add best result summary if available
            if self.best_result and self.best_parameters:
                summary['best_parameters'] = self.best_parameters
                summary['best_result_summary'] = self.best_result.generate_summary()
            
            # Add parameter statistics
            param_stats = {}
            for param_name, param_values in self.sweep_config.items():
                param_stats[param_name] = {
                    'num_values': len(param_values),
                    'values': param_values
                }
            summary['parameter_statistics'] = param_stats
            
            logger.debug(f"Generated summary for sweep '{self.sweep_id}'")
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            return {'error': str(e), 'sweep_id': self.sweep_id}
    
    def plot_parameter_impact(self, metric: str = 'accuracy', save_path: Optional[Path] = None) -> None:
        """
        Plot the impact of different parameters on the specified metric.
        
        Args:
            metric: Metric name to analyze
            save_path: Path to save the plot (if None, display only)
            
        Raises:
            ResultsVisualizationError: If plotting fails
        """
        try:
            analysis = self.get_parameter_analysis(metric)
            
            if not analysis:
                logger.warning("No parameter analysis data available for plotting")
                return
            
            # Create subplots for each parameter
            n_params = len(analysis)
            n_cols = min(3, n_params)
            n_rows = (n_params + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
            if n_params == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes if n_params > 1 else [axes]
            else:
                axes = axes.flatten()
            
            # Plot each parameter's impact
            for i, (param_name, param_data) in enumerate(analysis.items()):
                ax = axes[i]
                
                param_values = []
                means = []
                stds = []
                
                for param_value, stats in param_data.items():
                    param_values.append(param_value)
                    means.append(stats['mean'])
                    stds.append(stats['std'])
                
                # Create bar plot with error bars
                x_pos = range(len(param_values))
                bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)
                
                ax.set_title(f'Impact of {param_name} on {metric}')
                ax.set_xlabel(param_name)
                ax.set_ylabel(f'{metric} (mean ± std)')
                ax.set_xticks(x_pos)
                ax.set_xticklabels(param_values, rotation=45)
                ax.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar, mean in zip(bars, means):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + max(stds) * 0.01,
                           f'{mean:.3f}', ha='center', va='bottom', fontsize=8)
            
            # Hide unused subplots
            for i in range(n_params, len(axes)):
                axes[i].set_visible(False)
            
            plt.suptitle(f'Parameter Impact Analysis - Sweep {self.sweep_id}', fontsize=16)
            plt.tight_layout()
            
            # Save or display
            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.debug(f"Saved parameter impact plot to {save_path}")
            else:
                plt.show()
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Failed to plot parameter impact: {e}")
            raise ResultsVisualizationError(f"Failed to plot parameter impact: {e}")
    
    def save_to_file(self, path: Path) -> None:
        """
        Save sweep results to file.
        
        Args:
            path: Path to save the results
            
        Raises:
            ResultsSerializationError: If saving fails
        """
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare data for serialization
            save_data = {
                'sweep_id': self.sweep_id,
                'base_config': self._serialize_for_json(self.base_config),
                'sweep_config': self.sweep_config,
                'start_time': self.start_time.isoformat(),
                'end_time': self.end_time.isoformat() if self.end_time else None,
                'failed_experiments': self.failed_experiments,
                'experiment_results': {},
                'best_parameters': self.best_parameters,
                'summary': self.generate_summary()
            }
            
            # Serialize experiment results (save only summaries to reduce file size)
            for param_key, result in self.experiment_results.items():
                save_data['experiment_results'][param_key] = {
                    'experiment_id': result.experiment_id,
                    'summary': result.generate_summary(),
                    'parameters': self._key_to_parameters(param_key)
                }
            
            # Save based on file extension
            if path.suffix.lower() == '.json':
                with open(path, 'w') as f:
                    json.dump(save_data, f, indent=2, ensure_ascii=False)
            else:
                # Default to pickle for full data preservation
                with open(path, 'wb') as f:
                    pickle.dump(save_data, f)
            
            logger.debug(f"Saved sweep results to {path}")
            
        except Exception as e:
            logger.error(f"Failed to save sweep results to {path}: {e}")
            raise ResultsSerializationError(f"Failed to save sweep results: {e}")
    
    @classmethod
    def load_from_file(cls, path: Path) -> 'SweepResults':
        """
        Load sweep results from file.
        
        Args:
            path: Path to load the results from
            
        Returns:
            SweepResults instance
            
        Raises:
            ResultsSerializationError: If loading fails
        """
        try:
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"Sweep results file not found: {path}")
            
            # Load based on file extension
            if path.suffix.lower() == '.json':
                with open(path, 'r') as f:
                    data = json.load(f)
            else:
                with open(path, 'rb') as f:
                    data = pickle.load(f)
            
            # Reconstruct SweepResults
            sweep = cls(
                sweep_id=data['sweep_id'],
                base_config=DictConfig(data.get('base_config', {})),
                sweep_config=data.get('sweep_config', {}),
                failed_experiments=data.get('failed_experiments', {}),
                best_parameters=data.get('best_parameters'),
                start_time=datetime.fromisoformat(data['start_time']),
                end_time=datetime.fromisoformat(data['end_time']) if data.get('end_time') else None
            )
            
            # Note: For JSON files, we only have summaries, not full ExperimentResults
            # For pickle files, we could have full data
            logger.debug(f"Loaded sweep results from {path}")
            return sweep
            
        except Exception as e:
            logger.error(f"Failed to load sweep results from {path}: {e}")
            raise ResultsSerializationError(f"Failed to load sweep results: {e}")
    
    def finalize(self) -> None:
        """Mark the sweep as completed."""
        if self.end_time is None:
            self.end_time = datetime.now(timezone.utc)
            logger.debug(f"Sweep '{self.sweep_id}' finalized")
    
    def _parameters_to_key(self, parameters: Dict[str, Any]) -> str:
        """Convert parameter dictionary to a unique string key."""
        # Sort parameters by key for consistent ordering
        sorted_params = sorted(parameters.items())
        return json.dumps(sorted_params, sort_keys=True, default=str)
    
    def _key_to_parameters(self, key: str) -> Dict[str, Any]:
        """Convert string key back to parameter dictionary."""
        try:
            sorted_params = json.loads(key)
            return dict(sorted_params)
        except Exception as e:
            logger.error(f"Failed to convert key to parameters: {e}")
            return {}
    
    def _update_best_result(self, parameters: Dict[str, Any], result: ExperimentResults) -> None:
        """Update best result if this result is better."""
        try:
            # Use accuracy as the primary metric for determining best result
            current_accuracy = None
            if hasattr(result, 'task_results') and result.task_results:
                final_task = result.task_results[-1]
                current_accuracy = final_task.get_metric('accuracy')
            
            if current_accuracy is not None:
                if self.best_result is None:
                    self.best_result = result
                    self.best_parameters = parameters
                else:
                    # Compare with current best
                    best_accuracy = None
                    if hasattr(self.best_result, 'task_results') and self.best_result.task_results:
                        best_final_task = self.best_result.task_results[-1]
                        best_accuracy = best_final_task.get_metric('accuracy')
                    
                    if best_accuracy is None or current_accuracy > best_accuracy:
                        self.best_result = result
                        self.best_parameters = parameters
                        logger.debug(f"Updated best result with accuracy: {current_accuracy}")
            
        except Exception as e:
            logger.error(f"Failed to update best result: {e}")
    
    def _calculate_parameter_space_size(self) -> int:
        """Calculate the total size of the parameter space."""
        size = 1
        for param_values in self.sweep_config.values():
            size *= len(param_values)
        return size
    
    def _serialize_for_json(self, obj):
        """Serialize object for JSON compatibility (same as ExperimentResults)."""
        try:
            from omegaconf import DictConfig, OmegaConf
            
            if isinstance(obj, DictConfig):
                return OmegaConf.to_container(obj, resolve=True)
            elif hasattr(obj, 'to_dict') and callable(getattr(obj, 'to_dict')):
                return obj.to_dict()
            elif isinstance(obj, (list, tuple)):
                return [self._serialize_for_json(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: self._serialize_for_json(value) for key, value in obj.items()}
            elif hasattr(obj, '__dict__'):
                return {key: self._serialize_for_json(value) 
                    for key, value in obj.__dict__.items() 
                    if not key.startswith('_')}
            else:
                if hasattr(obj, 'item'):  # numpy scalar
                    return obj.item()
                return obj
                
        except Exception as e:
            logger.warning(f"Failed to serialize {type(obj)}: {e}, falling back to string")
            return str(obj)
    
    def __repr__(self) -> str:
        """String representation of SweepResults."""
        status = "完成" if self.end_time else "ongoing"
        success_rate = len(self.experiment_results) / (len(self.experiment_results) + len(self.failed_experiments)) if (len(self.experiment_results) + len(self.failed_experiments)) > 0 else 0
        return (f"SweepResults(id='{self.sweep_id}', experiments={len(self.experiment_results)}, "
                f"success_rate={success_rate:.2%}, status={status})")