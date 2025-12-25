"""
MLflow Tracker

使用 MLflow 记录实验
"""

from typing import Dict, Any
from .base import Tracker
from ..registry import register
from ..infra import get_module_logger

logger = get_module_logger(__name__)


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
        self.node_id = node_id  # 保存 node_id
        try:
            import mlflow
            import os

            # 设置 MLflow 认证环境变量（如果提供）
            if username:
                os.environ['MLFLOW_TRACKING_USERNAME'] = username
            if password:
                os.environ['MLFLOW_TRACKING_PASSWORD'] = password

            self.mlflow = mlflow
            self.tracking_uri = tracking_uri
            self.experiment_name = experiment_name
            self.run_name = run_name
            self.auto_end_run = auto_end_run
            self.auto_start_run = auto_start_run
            self.run_id = None

            # 设置 MLflow
            mlflow.set_tracking_uri(tracking_uri)

            # 创建或获取实验
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                mlflow.create_experiment(experiment_name)

            mlflow.set_experiment(experiment_name)

            # Trainer 端：立即启动 run
            if auto_start_run:
                # 确保没有活动的 run（清理残留）
                if mlflow.active_run() is not None:
                    logger.warning("Ending previous active run before starting new one")
                    mlflow.end_run()

                mlflow.start_run(run_name=run_name)
                self.run_id = mlflow.active_run().info.run_id
                logger.info(
                    f"MLflowTracker initialized with run_id: {self.run_id}, "
                    f"tracking_uri={tracking_uri}, experiment={experiment_name}"
                )
            else:
                logger.info(
                    f"MLflowTracker initialized (no active run): "
                    f"tracking_uri={tracking_uri}, experiment={experiment_name}"
                )

        except ImportError:
            logger.warning(
                "MLflow not installed. Install with: pip install mlflow. "
                "MLflowTracker will be disabled."
            )
            self.mlflow = None
            self.run_id = None

    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """
        记录指标

        如果设置了 node_id，指标名称会自动添加前缀：{node_id}/metric_name
        例如：node_id="learner_0" 时，"accuracy" -> "learner_0/accuracy"

        注意：MLflow 只支持标量值（int/float），复杂类型（list/dict/tensor）会被跳过
        """
        print(f"[MLflowTracker-DEBUG] log_metrics called: mlflow={bool(self.mlflow)}, run_id={self.run_id}, step={step}")
        print(f"[MLflowTracker-DEBUG] tracking_uri={self.tracking_uri}")
        print(f"[MLflowTracker-DEBUG] node_id={self.node_id}")
        print(f"[MLflowTracker-DEBUG] metrics keys: {list(metrics.keys()) if isinstance(metrics, dict) else type(metrics)}")

        if self.mlflow:
            # 必须有 run_id（来自 Trainer 同步或自己创建）
            if not self.run_id:
                print(f"[MLflowTracker-DEBUG] ❌ SKIPPING: No run_id available!")
                logger.warning("No run_id available. Skipping metric logging.")
                return

            print(f"[MLflowTracker-DEBUG] ✓ run_id exists: {self.run_id}")

            # 展平和过滤指标
            flat_metrics = self._flatten_metrics(metrics)
            print(f"[MLflowTracker-DEBUG] Flattened to {len(flat_metrics)} metrics: {list(flat_metrics.keys())}")

            for key, value in flat_metrics.items():
                try:
                    # 如果有 node_id，添加前缀
                    metric_name = f"{self.node_id}/{key}" if self.node_id else key
                    print(f"[MLflowTracker-DEBUG] Logging metric: {metric_name}={value} at step={step or 0}")

                    # 使用 client API 直接指定 run_id 记录
                    from mlflow.tracking import MlflowClient
                    client = MlflowClient(tracking_uri=self.tracking_uri)
                    client.log_metric(self.run_id, metric_name, value, step=step or 0)

                    print(f"[MLflowTracker-DEBUG] ✓ Successfully logged: {metric_name}")
                except Exception as e:
                    print(f"[MLflowTracker-DEBUG] ❌ Failed to log {key}: {e}")
                    logger.error(f"Failed to log metric {key}: {e}")

            print(f"[MLflowTracker-DEBUG] ✓ Completed logging {len(flat_metrics)} metrics")
        else:
            print(f"[MLflowTracker-DEBUG] ❌ MLflow not available (self.mlflow is None)")

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
                logger.info("MLflow run ended")
            except Exception as e:
                logger.error(f"Failed to end MLflow run: {e}")

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
            return

        if self.run_id is not None:
            logger.warning(f"Already in run {self.run_id}, skipping join")
            return

        # 直接保存 run_id，不需要 start_run（避免冲突）
        self.run_id = run_id
        logger.info(f"Joined MLflow run: {run_id}")
