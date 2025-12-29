"""
Tracker - 分布式指标追踪器

支持 MLflow, WandB, TensorBoard 等后端
"""

from typing import Any, Dict, List, Optional, Union
from abc import ABC, abstractmethod
import time
import uuid
import json
import os
from .logging import get_module_logger

logger = get_module_logger(__name__)


class TrackerBackend(ABC):
    """Tracker 后端抽象基类"""
    
    @abstractmethod
    def start_run(self, run_name: Optional[str] = None) -> str:
        """开始一个 run，返回 run_id"""
        pass
    
    @abstractmethod
    def join_run(self, run_id: str) -> None:
        """加入已有的 run"""
        pass
    
    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """记录指标"""
        pass
    
    @abstractmethod
    def log_params(self, params: Dict[str, Any]) -> None:
        """记录参数"""
        pass
    
    @abstractmethod
    def log_artifact(self, path: str, name: Optional[str] = None) -> None:
        """记录文件"""
        pass
    
    @abstractmethod
    def end_run(self) -> None:
        """结束 run"""
        pass


class ConsoleBackend(TrackerBackend):
    """控制台后端（调试用）"""
    
    def __init__(self, config: Dict[str, Any]):
        self._run_id = None
        self._verbose = config.get("verbose", True)
    
    def start_run(self, run_name: Optional[str] = None) -> str:
        self._run_id = str(uuid.uuid4())[:8]
        if self._verbose:
            logger.info(f"Started run: {self._run_id} ({run_name or 'unnamed'})")
        return self._run_id
    
    def join_run(self, run_id: str) -> None:
        self._run_id = run_id
        if self._verbose:
            logger.info(f"Joined run: {run_id}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if self._verbose:
            step_str = f"[step={step}] " if step is not None else ""
            metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            logger.info(f"{step_str}{metrics_str}")
    
    def log_params(self, params: Dict[str, Any]) -> None:
        if self._verbose:
            params_str = ", ".join(f"{k}={v}" for k, v in params.items())
            logger.info(f"Params: {params_str}")
    
    def log_artifact(self, path: str, name: Optional[str] = None) -> None:
        if self._verbose:
            logger.info(f"Artifact: {name or path}")
    
    def end_run(self) -> None:
        if self._verbose:
            logger.info(f"Ended run: {self._run_id}")
        self._run_id = None


class FileBackend(TrackerBackend):
    """文件后端"""
    
    def __init__(self, config: Dict[str, Any]):
        self._log_dir = config.get("log_dir", "./logs")
        self._run_id = None
        self._log_file = None
    
    def start_run(self, run_name: Optional[str] = None) -> str:
        self._run_id = str(uuid.uuid4())[:8]
        run_dir = os.path.join(self._log_dir, self._run_id)
        os.makedirs(run_dir, exist_ok=True)
        
        self._log_file = open(os.path.join(run_dir, "metrics.jsonl"), "a")
        
        # 记录元信息
        meta = {"run_id": self._run_id, "run_name": run_name, "start_time": time.time()}
        with open(os.path.join(run_dir, "meta.json"), "w") as f:
            json.dump(meta, f)
        
        return self._run_id
    
    def join_run(self, run_id: str) -> None:
        self._run_id = run_id
        run_dir = os.path.join(self._log_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)
        self._log_file = open(os.path.join(run_dir, "metrics.jsonl"), "a")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if self._log_file:
            record = {"metrics": metrics, "step": step, "timestamp": time.time()}
            self._log_file.write(json.dumps(record) + "\n")
            self._log_file.flush()
    
    def log_params(self, params: Dict[str, Any]) -> None:
        run_dir = os.path.join(self._log_dir, self._run_id)
        with open(os.path.join(run_dir, "params.json"), "w") as f:
            json.dump(params, f, indent=2)
    
    def log_artifact(self, path: str, name: Optional[str] = None) -> None:
        import shutil
        run_dir = os.path.join(self._log_dir, self._run_id, "artifacts")
        os.makedirs(run_dir, exist_ok=True)
        
        dest_name = name or os.path.basename(path)
        shutil.copy(path, os.path.join(run_dir, dest_name))
    
    def end_run(self) -> None:
        if self._log_file:
            self._log_file.close()
            self._log_file = None


class Tracker:
    """
    分布式指标追踪器
    
    特点：
    - 支持多种后端（Console, File, MLflow, WandB, TensorBoard）
    - 支持 pending 状态（客户端在注册前缓存指标）
    - 自动添加节点前缀
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        role: str = "server",
        node_id: str = "node",
        run_id: Optional[str] = None,
    ):
        """
        初始化追踪器
        
        Args:
            config: 追踪器配置
            role: 角色（server / client）
            node_id: 节点 ID
            run_id: run ID（客户端使用服务端分发的）
        """
        self._config = config
        self._role = role
        self._node_id = node_id
        self._run_id = run_id
        
        # 根据角色确定前缀
        if role == "server":
            self._prefix = "server"
        else:
            self._prefix = f"client_{node_id}"
        
        # 状态
        self._pending = run_id is None and role == "client"
        self._buffer: List[Dict[str, Any]] = []
        
        # 创建后端
        self._backend = self._create_backend()
        
        # 初始化 run
        if not self._pending:
            if run_id:
                self._backend.join_run(run_id)
            else:
                self._run_id = self._backend.start_run(
                    config.get("run_name", f"run_{int(time.time())}")
                )
    
    @property
    def run_id(self) -> Optional[str]:
        """获取 run_id"""
        return self._run_id
    
    @property
    def prefix(self) -> str:
        """获取指标前缀"""
        return self._prefix
    
    def activate(self, run_id: str) -> None:
        """
        激活追踪器（客户端收到 run_id 后调用）
        
        Args:
            run_id: 服务端分发的 run_id
        """
        if not self._pending:
            return
        
        self._run_id = run_id
        self._backend.join_run(run_id)
        self._pending = False
        
        # 上传缓存的指标
        for record in self._buffer:
            self._write(record)
        self._buffer.clear()
        
        logger.info(f"Tracker activated with run_id: {run_id}")
    
    def log(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        round: Optional[int] = None,
        prefix: Optional[str] = None,
    ) -> None:
        """
        记录指标
        
        Args:
            metrics: 指标字典
            step: 步数
            round: 轮次（会自动转换为 step）
            prefix: 自定义前缀（覆盖默认前缀）
        """
        # 确定 step
        actual_step = step if step is not None else round
        
        # 添加前缀
        actual_prefix = prefix or self._prefix
        prefixed_metrics = {
            f"{actual_prefix}/{k}": v for k, v in metrics.items()
        }
        
        record = {"metrics": prefixed_metrics, "step": actual_step}
        
        if self._pending:
            self._buffer.append(record)
        else:
            self._write(record)
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """
        记录超参数（通常仅服务端调用）
        
        Args:
            params: 参数字典
        """
        if not self._pending:
            self._backend.log_params(params)
    
    def log_artifact(self, path: str, name: Optional[str] = None) -> None:
        """
        记录文件
        
        Args:
            path: 文件路径
            name: 文件名（可选）
        """
        if not self._pending:
            self._backend.log_artifact(path, name)
    
    def log_model(self, model: Any, name: str) -> None:
        """
        记录模型
        
        Args:
            model: 模型对象
            name: 模型名称
        """
        import tempfile
        
        # 序列化模型到临时文件
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            temp_path = f.name
            model_bytes = model.serialize()
            f.write(model_bytes)
        
        self.log_artifact(temp_path, name)
        os.unlink(temp_path)
    
    def get_run_id(self) -> Optional[str]:
        """获取 run_id（供分发）"""
        return self._run_id
    
    def close(self) -> None:
        """关闭追踪器"""
        if self._role == "server":
            self._backend.end_run()
    
    def _write(self, record: Dict[str, Any]) -> None:
        """写入记录"""
        self._backend.log_metrics(record["metrics"], record.get("step"))
    
    def _create_backend(self) -> TrackerBackend:
        """创建后端"""
        backend_type = self._config.get("backend", "console")
        
        if backend_type == "console":
            return ConsoleBackend(self._config)
        elif backend_type == "file":
            return FileBackend(self._config)
        elif backend_type == "mlflow":
            return self._create_mlflow_backend()
        elif backend_type == "wandb":
            return self._create_wandb_backend()
        elif backend_type == "tensorboard":
            return self._create_tensorboard_backend()
        else:
            logger.warning(f"Unknown backend: {backend_type}, using console")
            return ConsoleBackend(self._config)
    
    def _create_mlflow_backend(self) -> TrackerBackend:
        """创建 MLflow 后端"""
        try:
            import mlflow

            class MLflowBackend(TrackerBackend):
                def __init__(self, config: Dict[str, Any]):
                    self._config = config

                    # 获取tracking_uri，优先级：配置 > 环境变量 > 默认值
                    tracking_uri = config.get("tracking_uri")

                    # 如果配置中没有，尝试从环境变量获取
                    if not tracking_uri:
                        tracking_uri = os.environ.get(
                            "MLFLOW_TRACKING_URI",
                            "http://172.19.138.200:5000"  # 默认使用远程服务器而不是本地
                        )

                    logger.info(f"MLflow tracking URI: {tracking_uri}")
                    mlflow.set_tracking_uri(tracking_uri)
                    mlflow.set_experiment(config.get("experiment_name", "federation"))
                
                def start_run(self, run_name: Optional[str] = None) -> str:
                    run = mlflow.start_run(run_name=run_name)
                    return run.info.run_id
                
                def join_run(self, run_id: str) -> None:
                    mlflow.start_run(run_id=run_id)
                
                def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
                    mlflow.log_metrics(metrics, step=step)
                
                def log_params(self, params: Dict[str, Any]) -> None:
                    mlflow.log_params(params)
                
                def log_artifact(self, path: str, name: Optional[str] = None) -> None:
                    mlflow.log_artifact(path)
                
                def end_run(self) -> None:
                    mlflow.end_run()
            
            return MLflowBackend(self._config.get("mlflow", {}))
        except ImportError:
            logger.warning("mlflow not installed, falling back to console")
            return ConsoleBackend(self._config)
    
    def _create_wandb_backend(self) -> TrackerBackend:
        """创建 WandB 后端"""
        try:
            import wandb
            
            class WandbBackend(TrackerBackend):
                def __init__(self, config: Dict[str, Any]):
                    self._config = config
                    self._run = None
                
                def start_run(self, run_name: Optional[str] = None) -> str:
                    self._run = wandb.init(
                        project=self._config.get("project", "federation"),
                        entity=self._config.get("entity"),
                        name=run_name,
                    )
                    return self._run.id
                
                def join_run(self, run_id: str) -> None:
                    self._run = wandb.init(
                        project=self._config.get("project", "federation"),
                        id=run_id,
                        resume="allow",
                    )
                
                def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
                    wandb.log(metrics, step=step)
                
                def log_params(self, params: Dict[str, Any]) -> None:
                    wandb.config.update(params)
                
                def log_artifact(self, path: str, name: Optional[str] = None) -> None:
                    artifact = wandb.Artifact(name or "artifact", type="file")
                    artifact.add_file(path)
                    wandb.log_artifact(artifact)
                
                def end_run(self) -> None:
                    if self._run:
                        wandb.finish()
            
            return WandbBackend(self._config.get("wandb", {}))
        except ImportError:
            logger.warning("wandb not installed, falling back to console")
            return ConsoleBackend(self._config)
    
    def _create_tensorboard_backend(self) -> TrackerBackend:
        """创建 TensorBoard 后端"""
        try:
            from torch.utils.tensorboard import SummaryWriter
            
            class TensorBoardBackend(TrackerBackend):
                def __init__(self, config: Dict[str, Any]):
                    self._log_dir = config.get("log_dir", "./runs")
                    self._writer = None
                    self._run_id = None
                
                def start_run(self, run_name: Optional[str] = None) -> str:
                    self._run_id = run_name or str(uuid.uuid4())[:8]
                    self._writer = SummaryWriter(
                        os.path.join(self._log_dir, self._run_id)
                    )
                    return self._run_id
                
                def join_run(self, run_id: str) -> None:
                    self._run_id = run_id
                    self._writer = SummaryWriter(
                        os.path.join(self._log_dir, run_id)
                    )
                
                def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
                    for k, v in metrics.items():
                        self._writer.add_scalar(k, v, step or 0)
                
                def log_params(self, params: Dict[str, Any]) -> None:
                    self._writer.add_text("params", json.dumps(params, indent=2))
                
                def log_artifact(self, path: str, name: Optional[str] = None) -> None:
                    # TensorBoard 不直接支持 artifact
                    pass
                
                def end_run(self) -> None:
                    if self._writer:
                        self._writer.close()
            
            return TensorBoardBackend(self._config.get("tensorboard", {}))
        except ImportError:
            logger.warning("tensorboard not installed, falling back to console")
            return ConsoleBackend(self._config)
