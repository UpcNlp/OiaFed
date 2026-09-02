"""
FedProject 聚合器 — FOT (ICLR 2024) 的核心反遗忘机制

移植自官方实现:
https://github.com/duygunuryldz/Federated_Orthogonal_Training/blob/main/
    FedML/fedml_api/distributed/fedavg_seq_cont/FedAVGAggregator.py

原理:
1. 先做标准 FedAvg (加权平均)
2. 计算聚合更新方向: grad = old_global - averaged
3. FedProject: 将 grad 投影到旧任务主子空间的正交补空间
   - 对 conv 层: projected = U @ U^T @ grad.view(out_ch, -1).T
   - 对 fc 层:   projected = U @ U^T @ grad.T
   - grad = grad - projected (移除旧任务方向的分量)
4. 应用: new_global = old_global - projected_grad
5. GPSE (expand_orth_set): 任务结束时用各客户端的激活做 SVD, 扩展正交集
"""

import time
import logging
from typing import Any, List, Optional, Dict, TYPE_CHECKING

import torch
import numpy as np

from ...core.aggregator import Aggregator
from ...core.types import ClientUpdate
from ...registry import aggregator
from ...infra import get_module_logger

if TYPE_CHECKING:
    from ...core.model import Model

logger = get_module_logger(__name__)


@aggregator(
    name='fedproject',
    description='FedProject 聚合器 — FOT (ICLR 2024) 正交投影聚合',
    version='1.0',
    author='FOT (Bakman et al.)',
    weighted=True
)
class FedProjectAggregator(Aggregator):
    """
    FedProject 聚合器

    在标准 FedAvg 之上添加正交投影, 防止新任务更新干扰旧任务知识。

    配置参数:
        epsilon: float = 0.87       — SVD 能量保留比例 (越大保留越多旧知识)
        eps_inc: float = 0.0        — 每个任务结束后 epsilon 的增量
        orth_layer_names: list      — 需要做正交投影的层名列表
        weighted: bool = True       — 是否按样本数加权
    """

    def __init__(
        self,
        epsilon: float = 0.87,
        eps_inc: float = 0.0,
        orth_layer_names: Optional[List[str]] = None,
        weighted: bool = True,
        **kwargs
    ):
        self._weighted = weighted
        self.epsilon = epsilon
        self.eps_inc = eps_inc

        # 需要做投影的层名 (默认: FOTAlexNet)
        if orth_layer_names is None:
            self.orth_layer_names = [
                'conv1.weight', 'conv2.weight', 'conv3.weight',
                'fc1.weight', 'fc2.weight',
            ]
        else:
            self.orth_layer_names = orth_layer_names

        # 正交集合: {layer_name: Tensor(n, k) or None}
        self.orth_set: Dict[str, Optional[torch.Tensor]] = {}
        for name in self.orth_layer_names:
            self.orth_set[name] = None

        logger.info(
            f"FedProject 初始化: epsilon={epsilon}, eps_inc={eps_inc}, "
            f"orth_layers={self.orth_layer_names}"
        )

    # ------------------------------------------------------------------
    #  核心聚合: FedAvg + FedProject
    # ------------------------------------------------------------------
    def aggregate(
        self,
        updates: List[ClientUpdate],
        global_model: Optional["Model"] = None,
    ) -> Any:
        """
        FedProject 聚合

        流程 (与官方实现一致):
        1. FedAvg 加权平均
        2. 计算更新方向 (gradient = global - averaged)
        3. 对 orth_layer 做正交投影
        4. 应用投影后的更新
        """
        if not updates:
            raise ValueError("No updates to aggregate")

        start_time = time.time()

        # --- Step 1: FedAvg ---
        updates = self.pre_aggregate(updates)

        if self._weighted:
            total_samples = sum(u.num_samples for u in updates)
            if total_samples == 0:
                weights = [1.0 / len(updates)] * len(updates)
            else:
                weights = [u.num_samples / total_samples for u in updates]
        else:
            weights = [1.0 / len(updates)] * len(updates)

        # 加权平均 (统一在 CPU 上操作)
        averaged_params = None
        for update, weight in zip(updates, weights):
            client_weights = update.weights
            if averaged_params is None:
                averaged_params = {
                    k: v.clone().cpu() * weight if torch.is_tensor(v) else v * weight
                    for k, v in client_weights.items()
                }
            else:
                for k, v in client_weights.items():
                    if torch.is_tensor(v):
                        averaged_params[k] += v.cpu() * weight
                    else:
                        averaged_params[k] += v * weight

        # --- Step 2: 获取当前全局模型参数 ---
        if global_model is None:
            # 无全局模型时退化为 FedAvg
            logger.warning("FedProject: 无全局模型, 退化为 FedAvg")
            return self.post_aggregate(averaged_params, updates)

        if hasattr(global_model, 'get_model'):
            torch_model = global_model.get_model()
        elif hasattr(global_model, '_model'):
            torch_model = global_model._model
        else:
            torch_model = global_model

        global_params = {k: v.clone().cpu() for k, v in torch_model.state_dict().items()}

        # --- Step 3: 计算梯度 (gradient = global - averaged) ---
        global_gradients = {}
        for k in global_params.keys():
            if k in averaged_params:
                avg_k = averaged_params[k].cpu() if torch.is_tensor(averaged_params[k]) else averaged_params[k]
                global_gradients[k] = global_params[k] - avg_k
            else:
                global_gradients[k] = torch.zeros_like(global_params[k])

        # --- Step 4: FedProject — 正交投影 ---
        projected_count = 0
        for key in self.orth_layer_names:
            if key not in global_gradients:
                continue
            if self.orth_set.get(key) is None:
                continue

            U = self.orth_set[key]  # (n, k)
            grad = global_gradients[key]

            if 'conv' in key or 'shortcut' in key:
                # Conv 层: grad shape = (out_ch, in_ch, kH, kW)
                # 展平为 (out_ch, in_ch*kH*kW), 转置, 投影
                flat = grad.view(grad.size(0), -1).T              # (in_ch*kH*kW, out_ch)
                projected = U @ (U.T @ flat)                       # (n, out_ch)
                global_gradients[key] = grad - projected.T.view(grad.size())
            else:
                # FC 层: grad shape = (out_features, in_features)
                projected = U @ (U.T @ grad.T)                    # (in_features, out_features)
                global_gradients[key] = grad - projected.T

            projected_count += 1

        if projected_count > 0:
            logger.info(f"FedProject: 对 {projected_count} 层做了正交投影")

        # --- Step 5: 应用投影后的梯度 ---
        for k in global_params.keys():
            if k in global_gradients:
                averaged_params[k] = global_params[k] - global_gradients[k]

        elapsed = time.time() - start_time
        logger.debug(f"FedProject 聚合耗时: {elapsed:.3f}s")

        return self.post_aggregate(averaged_params, updates)

    # ------------------------------------------------------------------
    #  GPSE: 全局主子空间提取
    # ------------------------------------------------------------------
    def expand_orth_set(self, activation_dict: Dict[int, Dict]) -> None:
        """
        GPSE — 全局主子空间提取 (在任务结束时调用)

        移植自官方 FedAVGAggregator.expand_orth_set()

        Args:
            activation_dict: {client_id: {layer_name: (random_proj_matrix, ratio, num_samples)}}

        流程:
        1. 聚合所有客户端的随机投影激活矩阵 (求和)
        2. 根据加权 ratio 调整 epsilon
        3. SVD, 保留 new_eps 比例能量的奇异向量
        4. 追加到 orth_set, QR 分解保持正交性
        """
        logger.info("GPSE: 开始扩展正交集合")

        act_list = list(activation_dict.values())
        if not act_list:
            logger.warning("GPSE: 无激活数据, 跳过")
            return

        # 检查是否有 None
        act_list = [a for a in act_list if a is not None]
        if not act_list:
            logger.warning("GPSE: 所有激活数据为 None, 跳过")
            return

        keys = act_list[0].keys()
        activations = {}
        ratios = {}
        num_samples = {}

        # 聚合各客户端的激活
        for k in keys:
            for i, local_act in enumerate(act_list):
                if k not in local_act:
                    continue
                if i == 0:
                    activations[k] = local_act[k][0]         # random_proj_matrix
                    ratios[k] = [local_act[k][1]]            # ratio
                    num_samples[k] = [local_act[k][2]]       # num_samples
                else:
                    activations[k] = activations[k] + local_act[k][0]
                    ratios[k].append(local_act[k][1])
                    num_samples[k].append(local_act[k][2])

        # 对每一层做 SVD 并扩展 orth_set
        for key in activations.keys():
            if key not in self.orth_layer_names:
                continue

            # 加权平均 ratio
            w = np.array(num_samples[key]) / np.sum(num_samples[key])
            weighted_avg_ratio = np.sum(w * np.array(ratios[key]))

            # 根据 ratio 调整 epsilon
            org_eps = self.epsilon
            if weighted_avg_ratio > 0:
                new_eps = (weighted_avg_ratio - (1 - org_eps)) / weighted_avg_ratio
            else:
                new_eps = org_eps
            new_eps = max(0.0, min(1.0, new_eps))  # clamp

            # SVD
            mat = activations[key]
            if not torch.is_tensor(mat):
                mat = torch.tensor(mat, dtype=torch.float32)

            U, S, V = torch.svd(mat)

            # 找到保留 new_eps 比例能量的奇异向量数
            total = torch.norm(mat) ** 2
            cutoff_idx = 0
            for i in range(len(S)):
                hand = torch.norm(S[0:i + 1]) ** 2
                if hand / total > new_eps:
                    cutoff_idx = i
                    break
            else:
                cutoff_idx = len(S) - 1

            new_vectors = U[:, 0:cutoff_idx + 1]

            # 追加到 orth_set
            if self.orth_set[key] is None:
                self.orth_set[key] = new_vectors
            else:
                self.orth_set[key] = torch.cat(
                    (self.orth_set[key], new_vectors), dim=1
                )

            # QR 分解保持正交性
            self.orth_set[key], _ = torch.linalg.qr(self.orth_set[key])

            shape = self.orth_set[key].shape
            logger.info(
                f"  GPSE [{key}]: orth_set shape = {shape}, "
                f"space usage = {shape[1]}/{shape[0]} = {shape[1]/shape[0]:.3f}"
            )

        # epsilon 递增
        self.epsilon += self.eps_inc
        logger.info(
            f"GPSE 完成. epsilon 更新为 {self.epsilon:.4f}"
        )