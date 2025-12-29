"""
MLflow Tracker

使用 MLflow 记录实验
"""

import sys
from typing import Dict, Any, Optional
from .base import Tracker
from ..registry import register
from ..infra import get_module_logger

logger = get_module_logger(__name__)


def _print_error(msg: str) -> None:
    """打印错误到 stderr，确保用户能看到"""
    print(f"[MLflowTracker] ❌ {msg}", file=sys.stderr)


def _print_warning(msg: str) -> None:
    """打印警告到 stderr"""
    print(f"[MLflowTracker] ⚠️  {msg}", file=sys.stderr)


def _print_info(msg: str) -> None:
    """打印信息"""
    print(f"[MLflowTracker] ✓ {msg}")


@register("federated.tracker.mlflow")
class MLflowTracker(Tracker):
    """
    MLflow Tracker

    使用 MLflow 记录实验
    """

    def __init__(
        self,
        tracking_uri: str = "./mlruns",
        experiment_name: str = "federated_learning",
        run_name: str = None,
        auto_end_run: bool = True,
        auto_start_run: bool = True,
        node_id: str = None,  # 新增：节点 ID，用于指标前缀
        username: str = None,  # MLflow 认证用户名
        password: str = None,  # MLflow 认证密码
    ):
        """
        Args:
            tracking_uri: MLflow 存储路径
            experiment_name: 实验名称
            run_name: 运行名称（可选）
            auto_end_run: 是否自动结束运行
            auto_start_run: 是否自动启动运行（Trainer=True，Learner=False）
            node_id: 节点 ID，用于给指标添加前缀（格式：{node_id}/metric_name）
            username: MLflow 认证用户名（可选）
            password: MLflow 认证密码（可选）
        """
        self.node_id = node_id
        self.mlflow = None
        self.run_id = None
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.auto_end_run = auto_end_run
        self.auto_start_run = auto_start_run
        self._init_error: Optional[str] = None  # 记录初始化错误
        
        try:
            import mlflow
            import os

            # 设置 MLflow 认证环境变量（如果提供）
            if username:
                os.environ['MLFLOW_TRACKING_USERNAME'] = username
            if password:
                os.environ['MLFLOW_TRACKING_PASSWORD'] = password

            self.mlflow = mlflow

            # 设置 MLflow
            mlflow.set_tracking_uri(tracking_uri)

            # 创建或获取实验（处理各种边界情况）
            from mlflow.tracking import MlflowClient
            client = MlflowClient(tracking_uri=tracking_uri)
            
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                # 实验不存在，创建新实验
                mlflow.create_experiment(experiment_name)
            elif experiment.lifecycle_stage == "deleted":
                # 实验已被软删除，尝试恢复
                try:
                    _print_info(f"Restoring deleted experiment: {experiment_name}")
                    client.restore_experiment(experiment.experiment_id)
                except Exception as e:
                    # 恢复失败（可能已被永久删除），创建新实验
                    _print_warning(f"Failed to restore experiment: {e}")
                    _print_info(f"Creating new experiment: {experiment_name}")
                    try:
                        mlflow.create_experiment(experiment_name)
                    except Exception as create_err:
                        # 如果创建也失败，使用带时间戳的新名称
                        import time
                        new_name = f"{experiment_name}_{int(time.time())}"
                        _print_warning(f"Creating experiment with new name: {new_name}")
                        mlflow.create_experiment(new_name)
                        experiment_name = new_name
            
            mlflow.set_experiment(experiment_name)

            # Trainer 端：立即启动 run
            if auto_start_run:
                # 确保没有活动的 run（清理残留）
                if mlflow.active_run() is not None:
                    _print_warning("Ending previous active run before starting new one")
                    mlflow.end_run()

                mlflow.start_run(run_name=run_name)
                self.run_id = mlflow.active_run().info.run_id
                _print_info(
                    f"Initialized: run_id={self.run_id}, "
                    f"uri={tracking_uri}, experiment={experiment_name}"
                )
                logger.info(
                    f"MLflowTracker initialized with run_id: {self.run_id}, "
                    f"tracking_uri={tracking_uri}, experiment={experiment_name}"
                )
            else:
                _print_info(
                    f"Initialized (waiting for run_id): "
                    f"uri={tracking_uri}, experiment={experiment_name}"
                )
                logger.info(
                    f"MLflowTracker initialized (no active run): "
                    f"tracking_uri={tracking_uri}, experiment={experiment_name}"
                )

        except ImportError as e:
            error_msg = str(e)
            self._init_error = error_msg
            
            # 提供具体的解决建议
            if "databricks" in error_msg:
                _print_error(f"MLflow 依赖缺失: {error_msg}")
                _print_error("解决方法: pip install databricks-sdk")
            elif "mlflow" in error_msg.lower():
                _print_error(f"MLflow 未安装: {error_msg}")
                _print_error("解决方法: pip install mlflow")
            else:
                _print_error(f"导入错误: {error_msg}")
            
            _print_warning("MLflowTracker 已禁用，指标将不会记录到 MLflow")
            logger.warning(f"MLflowTracker disabled due to ImportError: {error_msg}")
            
        except Exception as e:
            error_msg = str(e)
            self._init_error = error_msg
            
            # 连接错误等
            _print_error(f"MLflow 初始化失败: {error_msg}")
            
            if "connection" in error_msg.lower() or "refused" in error_msg.lower():
                _print_error(f"无法连接到 MLflow server: {tracking_uri}")
                _print_error("请检查 MLflow server 是否运行，或使用本地路径如 './mlruns'")
            
            _print_warning("MLflowTracker 已禁用，指标将不会记录到 MLflow")
            logger.warning(f"MLflowTracker disabled due to error: {error_msg}")

    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """
        记录指标

        如果设置了 node_id，指标名称会自动添加前缀：{node_id}/metric_name
        例如：node_id="learner_0" 时，"accuracy" -> "learner_0/accuracy"

        注意：MLflow 只支持标量值（int/float），复杂类型（list/dict/tensor）会被跳过
        """
        if not self.mlflow:
            logger.debug("MLflow not available, skipping metric logging")
            return
            
        # 必须有 run_id（来自 Trainer 同步或自己创建）
        if not self.run_id:
            logger.warning("No run_id available. Skipping metric logging.")
            return

        # 展平和过滤指标
        flat_metrics = self._flatten_metrics(metrics)
        logger.debug(f"Logging {len(flat_metrics)} metrics to MLflow")

        for key, value in flat_metrics.items():
            try:
                # 如果有 node_id，添加前缀
                metric_name = f"{self.node_id}/{key}" if self.node_id else key

                # 使用 client API 直接指定 run_id 记录
                from mlflow.tracking import MlflowClient
                client = MlflowClient(tracking_uri=self.tracking_uri)
                client.log_metric(self.run_id, metric_name, value, step=step or 0)
                
            except Exception as e:
                logger.error(f"Failed to log metric {key}: {e}")

    def _flatten_metrics(self, metrics: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
        """
        展平指标字典，并过滤出标量值

        Args:
            metrics: 原始指标字典（可能包含嵌套结构）
            prefix: 键前缀（用于递归）

        Returns:
            展平后的指标字典，只包含 int/float 值
        """
        flat = {}

        for key, value in metrics.items():
            # 构造完整的键名
            full_key = f"{prefix}/{key}" if prefix else key

            # 检查值的类型
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                # 标量值：直接记录
                flat[full_key] = float(value)
            elif isinstance(value, dict):
                # 嵌套字典：递归展平
                nested = self._flatten_metrics(value, prefix=full_key)
                flat.update(nested)
            elif isinstance(value, (list, tuple)):
                # 列表/元组：跳过（MLflow 不支持）
                logger.debug(f"Skipping metric '{full_key}' (type: {type(value).__name__})")
            elif hasattr(value, 'item'):
                # NumPy/Torch 标量：提取值
                try:
                    flat[full_key] = float(value.item())
                except (AttributeError, ValueError):
                    logger.debug(f"Skipping metric '{full_key}' (cannot convert to float)")
            else:
                # 其他类型（如 tensor/array）：跳过
                logger.debug(f"Skipping metric '{full_key}' (type: {type(value).__name__})")

        return flat

    def log_params(self, params: Dict[str, Any]):
        """记录参数"""
        if self.mlflow:
            # 检查是否有活动的 run
            if self.mlflow.active_run() is None:
                logger.debug("No active MLflow run. Skipping param logging.")
                return

            try:
                self.mlflow.log_params(params)
            except Exception as e:
                logger.error(f"Failed to log params: {e}")

    def log_artifact(self, local_path: str, artifact_path: str = None):
        """记录文件"""
        if self.mlflow:
            try:
                self.mlflow.log_artifact(local_path, artifact_path=artifact_path)
            except Exception as e:
                logger.error(f"Failed to log artifact {local_path}: {e}")

    def set_tags(self, tags: Dict[str, str]):
        """设置标签"""
        if self.mlflow:
            # 检查是否有活动的 run
            if self.mlflow.active_run() is None:
                logger.debug("No active MLflow run. Skipping tag setting.")
                return

            try:
                self.mlflow.set_tags(tags)
            except Exception as e:
                logger.error(f"Failed to set tags: {e}")

    def close(self):
        """结束运行"""
        if self.mlflow and self.auto_end_run:
            try:
                self.mlflow.end_run()
                _print_info("MLflow run ended")
                logger.info("MLflow run ended")
            except Exception as e:
                logger.error(f"Failed to end MLflow run: {e}")

    # ========== 状态检查 ==========

    @property
    def is_available(self) -> bool:
        """检查 MLflow 是否可用"""
        return self.mlflow is not None

    @property
    def init_error(self) -> Optional[str]:
        """获取初始化错误信息"""
        return self._init_error

    # ========== 分布式同步接口 ==========

    def get_sync_info(self) -> Dict[str, Any]:
        """
        获取同步信息（Trainer 端）

        Returns:
            包含 run_id 的字典，用于 Learner 同步
        """
        if self.run_id:
            return {"mlflow_run_id": self.run_id}
        return {}

    def join_run(self, run_id: str):
        """
        加入已有的 run（Learner 端）

        直接保存 run_id，之后使用 MlflowClient 记录指标，无需 start_run

        Args:
            run_id: Trainer 创建的 run_id
        """
        if not self.mlflow:
            _print_warning(f"Cannot join run {run_id}: MLflow not available")
            return

        # 如果已经有 run，先结束它（切换到 Trainer 的 run）
        if self.run_id is not None:
            if self.run_id == run_id:
                logger.info(f"Already in the same run {run_id}, no need to join")
                return
            
            # 结束当前 run（如果是活动的）
            try:
                if self.mlflow.active_run() is not None:
                    logger.info(f"Ending previous run {self.run_id} to join Trainer's run")
                    self.mlflow.end_run()
            except Exception as e:
                logger.warning(f"Failed to end previous run: {e}")

        # 保存新的 run_id，使用 MlflowClient 记录指标
        self.run_id = run_id
        _print_info(f"Joined MLflow run: {run_id}")
        logger.info(f"Joined MLflow run: {run_id}")