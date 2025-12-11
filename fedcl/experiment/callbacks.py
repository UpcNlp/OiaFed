"""
实验回调函数集合
fedcl/experiment/callbacks.py

功能：
- 提供预定义的实验回调函数
- 用于记录训练过程中的各种指标
- 支持新的 Logger 系统（参考 PyTorch Lightning 设计）
"""

from typing import Dict, Any, Callable, Union, List, Optional


def create_callbacks(loggers: Union['Logger', List['Logger']]) -> Dict[str, Callable]:
    """创建实验回调函数集合

    Args:
        loggers: Logger 实例或 Logger 列表

    Returns:
        回调函数字典

    使用示例：
        from fedcl.loggers import MLflowLogger, JSONLogger

        # 单个 logger
        logger = MLflowLogger(experiment_name="my_exp")
        callbacks = create_callbacks(logger)

        # 多个 logger
        loggers = [
            MLflowLogger(experiment_name="my_exp"),
            JSONLogger(save_dir="results/"),
        ]
        callbacks = create_callbacks(loggers)

        # 注册回调
        trainer.add_callback('after_round', callbacks['round_callback'])
        trainer.add_callback('after_evaluation', callbacks['eval_callback'])
    """
    # 统一处理单个或多个 logger
    if not isinstance(loggers, list):
        loggers = [loggers]

    def _log_metrics(metrics: Dict[str, Union[int, float]], step: Optional[int] = None):
        """Helper: 记录指标到所有 logger"""
        print(f"[Callback Debug] _log_metrics 被调用:")
        print(f"  metrics: {metrics}")
        print(f"  step: {step}")
        print(f"  loggers 数量: {len(loggers)}")

        for i, logger in enumerate(loggers):
            try:
                print(f"  [{i}] 调用 {logger}.log_metrics()")
                logger.log_metrics(metrics, step=step)
                print(f"  [{i}] 成功")
            except Exception as e:
                print(f"  [{i}] 失败: {e}")
                print(f"Warning: {logger.name if hasattr(logger, 'name') else logger} failed to log metrics: {e}")

    def round_callback(round_num: int, round_result: dict):
        """轮次训练回调

        Args:
            round_num: 轮次编号
            round_result: 轮次结果
        """
        print(f"\n[Callback Debug] round_callback 被调用:")
        print(f"  round_num: {round_num}")
        print(f"  round_result keys: {list(round_result.keys())}")
        print(f"  round_result: {round_result}")

        metrics = round_result.get("round_metrics", {})

        # 收集所有指标
        metrics_to_log = {}

        # 服务端聚合指标
        if "avg_accuracy" in metrics:
            metrics_to_log["server/avg_accuracy"] = metrics["avg_accuracy"]

        if "avg_loss" in metrics:
            metrics_to_log["server/avg_loss"] = metrics["avg_loss"]

        if "successful_count" in metrics:
            metrics_to_log["server/num_participants"] = metrics["successful_count"]

        # 其他指标
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and key not in ["avg_accuracy", "avg_loss", "successful_count"]:
                metrics_to_log[f"server/{key}"] = value

        print(f"[Callback Debug] metrics_to_log: {metrics_to_log}")

        # 记录到所有 logger
        if metrics_to_log:
            print(f"[Callback Debug] 调用 _log_metrics")
            _log_metrics(metrics_to_log, step=round_num)
            print(f"[Callback Debug] _log_metrics 调用完成")
        else:
            print(f"[Callback Debug] 没有指标需要记录")

    def eval_callback(eval_result: dict):
        """评估回调

        Args:
            eval_result: 评估结果
        """
        # 记录最终评估结果作为 metrics
        metrics_to_log = {}

        if "accuracy" in eval_result:
            metrics_to_log["final/test_accuracy"] = eval_result["accuracy"]

        if "loss" in eval_result:
            metrics_to_log["final/test_loss"] = eval_result["loss"]

        # 其他评估指标
        for key, value in eval_result.items():
            if isinstance(value, (int, float)) and key not in ["accuracy", "loss"]:
                metrics_to_log[f"final/{key}"] = value

        # 记录到所有 logger
        if metrics_to_log:
            _log_metrics(metrics_to_log)

    def client_train_callback(params: dict, response):
        """客户端训练回调

        Args:
            params: 训练参数
            response: 训练响应
        """
        # 提取轮次编号
        round_num = params.get("round_number", 0)

        # 提取客户端ID
        client_id = getattr(response, 'client_id', None)
        if not client_id:
            return

        # 提取训练结果
        result = getattr(response, 'result', None)
        if not result or not isinstance(result, dict):
            return

        # 收集客户端指标
        metrics_to_log = {}

        if "loss" in result:
            metrics_to_log[f"client/{client_id}/loss"] = result["loss"]

        if "accuracy" in result:
            metrics_to_log[f"client/{client_id}/accuracy"] = result["accuracy"]

        if "samples_count" in result:
            metrics_to_log[f"client/{client_id}/samples"] = result["samples_count"]

        # 其他客户端指标
        for key, value in result.items():
            if isinstance(value, (int, float)) and key not in ["loss", "accuracy", "samples_count", "model_update", "model_weights"]:
                metrics_to_log[f"client/{client_id}/{key}"] = value

        # 记录到所有 logger
        if metrics_to_log:
            _log_metrics(metrics_to_log, step=round_num)

    def client_eval_callback(result: dict):
        """客户端评估回调

        Args:
            result: 评估结果
        """
        # 提取客户端评估指标
        if not result or not isinstance(result, dict):
            return

        metrics_to_log = {}

        # 提取客户端ID（如果有）
        client_id = result.get('client_id', 'unknown')

        if "accuracy" in result:
            metrics_to_log[f"client/{client_id}/eval_accuracy"] = result["accuracy"]

        if "loss" in result:
            metrics_to_log[f"client/{client_id}/eval_loss"] = result["loss"]

        # 其他评估指标
        for key, value in result.items():
            if isinstance(value, (int, float)) and key not in ["accuracy", "loss", "client_id"]:
                metrics_to_log[f"client/{client_id}/eval_{key}"] = value

        # 记录到所有 logger
        if metrics_to_log:
            _log_metrics(metrics_to_log)

    return {
        'round_callback': round_callback,
        'eval_callback': eval_callback,
        'client_train_callback': client_train_callback,
        'client_eval_callback': client_eval_callback
    }
