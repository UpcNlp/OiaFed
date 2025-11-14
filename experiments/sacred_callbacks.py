"""
Sacred 回调函数集合
experiments/sacred_callbacks.py

功能：
- 提供预定义的 Sacred 回调函数
- 用于记录训练过程中的各种指标
"""

from typing import Dict, Any
from .sacred_wrapper import SacredRecorder


def create_sacred_callbacks(recorder: SacredRecorder) -> Dict[str, callable]:
    """创建 Sacred 回调函数集合

    Args:
        recorder: SacredRecorder 实例

    Returns:
        回调函数字典
    """

    def round_callback(round_num: int, round_result: dict):
        """轮次训练回调

        Args:
            round_num: 轮次编号
            round_result: 轮次结果
        """
        metrics = round_result.get("round_metrics", {})

        # 记录服务端聚合指标
        if "avg_accuracy" in metrics:
            recorder.log_scalar("server/avg_accuracy", metrics["avg_accuracy"], round_num)

        if "avg_loss" in metrics:
            recorder.log_scalar("server/avg_loss", metrics["avg_loss"], round_num)

        if "successful_count" in metrics:
            recorder.log_scalar("server/num_participants", metrics["successful_count"], round_num)

        # 记录其他指标
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and key not in ["avg_accuracy", "avg_loss", "successful_count"]:
                recorder.log_scalar(f"server/{key}", value, round_num)

    def eval_callback(eval_result: dict):
        """评估回调

        Args:
            eval_result: 评估结果
        """
        # 记录最终评估结果
        if "accuracy" in eval_result:
            recorder.log_info("final_test_accuracy", eval_result["accuracy"])

        if "loss" in eval_result:
            recorder.log_info("final_test_loss", eval_result["loss"])

        # 记录其他评估指标
        for key, value in eval_result.items():
            if isinstance(value, (int, float, str, bool)) and key not in ["accuracy", "loss"]:
                recorder.log_info(f"final_{key}", value)

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

        # 记录客户端本地训练指标
        if "loss" in result:
            recorder.log_scalar(f"client/{client_id}/loss", result["loss"], round_num)

        if "accuracy" in result:
            recorder.log_scalar(f"client/{client_id}/accuracy", result["accuracy"], round_num)

        if "samples_count" in result:
            recorder.log_scalar(f"client/{client_id}/samples", result["samples_count"], round_num)

        # 记录其他客户端指标
        for key, value in result.items():
            if isinstance(value, (int, float)) and key not in ["loss", "accuracy", "samples_count", "model_update", "model_weights"]:
                recorder.log_scalar(f"client/{client_id}/{key}", value, round_num)

    def client_eval_callback(result: dict):
        """客户端评估回调

        Args:
            result: 评估结果
        """
        # 可以记录客户端的评估结果
        pass

    return {
        'round_callback': round_callback,
        'eval_callback': eval_callback,
        'client_train_callback': client_train_callback,
        'client_eval_callback': client_eval_callback
    }
