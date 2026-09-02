"""
Multi-Parent Evidence-Guided Symbiosis (MP-EGS) 模块。

在原始 FedSym 共生池框架的基础上:
1. 将共生从"2 父模型"推广到"M 父模型"
2. 利用 EDL 证据签名引导融合 (Endo) 和重组 (Ecto)

核心洞察:
  M=2 时，Pathological 场景下两模型的类分布几乎不重叠，导致类级别
  权重极端化 (0/1 二值)，层间耦合断裂。增大 M 使参与者覆盖更多类，
  class_weights 分布更平滑，证据引导的精细化才能生效。

  M 是一个自然的控制参数:
    M=2  → 传统两父共生 (FedSym)
    M=K  → 全池 CEGA 聚合（池多样性消失）
    最优 M 在两者之间

分层聚合策略 (Endo):
  Backbone:           FedAvg (等权 1/M)
  Router 第 e 行:      feat_weights[e] (类级别权重的均值)
  Expert e hidden 层:  feat_weights[e]
  Expert e 输出层第 c 行: class_weights[e, c]

重组策略 (Ecto):
  Backbone:           按总证据量概率选一个父模型整块复制
  Expert 位置 e:       从 M×E 候选中按签名相似度贪心选择
  Router 第 e 行:      跟着被选中的 expert 走 (coherence)
"""

import copy
import math
import random
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 辅助函数
# ============================================================

def _cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> float:
    na, nb = a.norm(), b.norm()
    if na < eps or nb < eps:
        return 0.0
    return float(((a * b).sum() / (na * nb)).item())


def _compute_model_total_evidence(
    signatures: Dict[int, torch.Tensor], num_experts: int
) -> float:
    total = 0.0
    for e in range(num_experts):
        sig = signatures.get(e)
        if sig is not None:
            total += float(sig.abs().sum().item())
    return total


def _pool_diversity_score(
    pool_signatures: List[Dict[int, torch.Tensor]], num_experts: int
) -> float:
    K = len(pool_signatures)
    if K < 2:
        return 0.0
    diversity_sum = 0.0
    n_pairs = 0
    for i in range(K):
        for j in range(i + 1, K):
            vec_i = torch.cat([
                pool_signatures[i].get(e, torch.zeros(1))
                for e in range(num_experts)
            ])
            vec_j = torch.cat([
                pool_signatures[j].get(e, torch.zeros(1))
                for e in range(num_experts)
            ])
            if vec_i.shape != vec_j.shape:
                continue
            sim = _cosine_similarity(vec_i, vec_j)
            diversity_sum += (1.0 - sim)
            n_pairs += 1
    return diversity_sum / n_pairs if n_pairs > 0 else 0.0


# ============================================================
# Multi-Parent Evidence-Guided Endosymbiosis
# ============================================================

class EvidenceEndoSymbiosis:
    """
    多父代证据引导的内共生。

    本质: CEGA 聚合逻辑在 M 个池内模型上的应用。

    M=2 时退化为两模型共生 (FedSym-Endo 的证据引导版)；
    M=K 时等价于对整个池做 CEGA 聚合。
    """

    def __init__(
        self,
        num_experts: int,
        num_classes: int,
        smoothing_alpha: float = 0.0,
    ):
        self.num_experts = num_experts
        self.num_classes = num_classes
        self.smoothing_alpha = smoothing_alpha

    def symbiose(
        self,
        models: List[nn.Module],
        signatures: List[Dict[int, torch.Tensor]],
    ) -> nn.Module:
        """
        对 M 个父模型执行证据引导的内共生。

        参数:
            models: M 个父模型
            signatures: M 个对应的证据签名

        返回:
            融合后的新模型
        """
        M = len(models)
        eps = 1e-8
        alpha = self.smoothing_alpha
        device = next(models[0].parameters()).device

        # ============================================================
        # Phase 0: 预计算所有 expert 的权重
        # ============================================================
        # class_weights[e]: [M, C] — 第 i 个模型对 class c 的权重
        # feat_weights[e]:  [M]   — 第 i 个模型的综合权重
        all_class_weights: Dict[int, torch.Tensor] = {}
        all_feat_weights: Dict[int, torch.Tensor] = {}

        for e in range(self.num_experts):
            # 收集 M 个模型在 expert e 上的签名
            sig_matrix = torch.zeros(M, self.num_classes, device=device)
            for i in range(M):
                sig = signatures[i].get(e)
                if sig is not None:
                    sig_matrix[i] = sig.abs().to(device)

            # 类级别权重 (带平滑)
            # w[i, c] = (sig[i][c] + α) / (Σ_j sig[j][c] + M*α)
            col_sums = sig_matrix.sum(dim=0, keepdim=True) + M * alpha + eps
            class_weights = (sig_matrix + alpha) / col_sums  # [M, C]
            all_class_weights[e] = class_weights

            # 综合权重: 只在"有效类"上取均值
            # 有效类 = 至少一个 parent 有非零证据的类
            # 避免零证据列（退化为 1/M）淹没真实信号
            active_mask = sig_matrix.sum(dim=0) > eps  # [C]
            n_active = active_mask.sum().item()
            if n_active > 0:
                feat_weights = class_weights[:, active_mask].mean(dim=1)  # [M]
            else:
                feat_weights = torch.ones(M, device=device) / M
            feat_weights = feat_weights / (feat_weights.sum() + eps)  # 归一化
            all_feat_weights[e] = feat_weights

        # ============================================================
        # Phase 1: Backbone → FedAvg (等权 1/M)
        # ============================================================
        new_model = copy.deepcopy(models[0])

        with torch.no_grad():
            # 清零 backbone 参数
            for p in new_model.backbone.parameters():
                p.data.zero_()
            for m in models:
                for p_new, p_m in zip(
                    new_model.backbone.parameters(), m.backbone.parameters()
                ):
                    p_new.data.add_(p_m.data / M)

            # Backbone BN buffers
            new_bufs = dict(new_model.backbone.named_buffers())
            for name, buf in new_bufs.items():
                if 'running_mean' in name or 'running_var' in name:
                    buf.data.zero_()
            for m in models:
                m_bufs = dict(m.backbone.named_buffers())
                for name, buf_new in new_bufs.items():
                    if 'running_mean' in name or 'running_var' in name:
                        if name in m_bufs:
                            buf_new.data.add_(m_bufs[name].data / M)

            # ============================================================
            # Phase 2: Router 第 e 行 → feat_weights[e]
            # ============================================================
            router_W_new = new_model.router.evidence_network.weight.data
            router_b_new = new_model.router.evidence_network.bias.data
            router_W_new.zero_()
            router_b_new.zero_()

            for e in range(self.num_experts):
                fw = all_feat_weights[e]  # [M]
                for i, m in enumerate(models):
                    w = float(fw[i].item())
                    router_W_new[e].add_(
                        w * m.router.evidence_network.weight.data[e]
                    )
                    router_b_new[e].add_(
                        w * m.router.evidence_network.bias.data[e]
                    )

            # ============================================================
            # Phase 3: Expert e
            #   hidden 层: feat_weights[e]
            #   输出层第 c 行: class_weights[e, c]
            # ============================================================
            for e in range(self.num_experts):
                fw = all_feat_weights[e]   # [M]
                cw = all_class_weights[e]  # [M, C]

                expert_new = new_model.experts.experts[e]
                layers_new = list(expert_new.network.children())

                # 找输出层
                output_idx = None
                for idx in reversed(range(len(layers_new))):
                    if isinstance(layers_new[idx], nn.Linear):
                        output_idx = idx
                        break

                # 收集各父模型对应 expert 的层
                all_layers = [
                    list(models[i].experts.experts[e].network.children())
                    for i in range(M)
                ]

                for idx, layer_new in enumerate(layers_new):
                    if not isinstance(layer_new, nn.Linear):
                        continue

                    layer_new.weight.data.zero_()
                    if layer_new.bias is not None:
                        layer_new.bias.data.zero_()

                    if idx == output_idx:
                        # 输出层: 逐类加权
                        for c in range(self.num_classes):
                            for i in range(M):
                                w_ic = float(cw[i, c].item())
                                layer_new.weight.data[c].add_(
                                    w_ic * all_layers[i][idx].weight.data[c]
                                )
                                layer_new.bias.data[c].add_(
                                    w_ic * all_layers[i][idx].bias.data[c]
                                )
                    else:
                        # Hidden 层: 综合权重
                        for i in range(M):
                            w_i = float(fw[i].item())
                            layer_new.weight.data.add_(
                                w_i * all_layers[i][idx].weight.data
                            )
                            if layer_new.bias is not None:
                                layer_new.bias.data.add_(
                                    w_i * all_layers[i][idx].bias.data
                                )

        return new_model


# ============================================================
# Multi-Parent Evidence-Guided Ectosymbiosis
# ============================================================

class EvidenceEctoSymbiosis:
    """
    多父代证据引导的外共生。

    Backbone: 从 M 个父模型中按总证据量概率选一个整块复制
    Expert:   从 M×E 候选中按签名相似度贪心匹配
    Router:   第 e 行跟着被选中的 (model_idx, expert_idx) 走
    """

    def __init__(self, num_experts: int, num_classes: int):
        self.num_experts = num_experts
        self.num_classes = num_classes

    def symbiose(
        self,
        models: List[nn.Module],
        signatures: List[Dict[int, torch.Tensor]],
    ) -> nn.Module:
        """
        对 M 个父模型执行证据引导的外共生。
        """
        M = len(models)
        eps = 1e-8

        # ======== Backbone: 按总证据量概率选一个 ========
        evidence_totals = [
            _compute_model_total_evidence(signatures[i], self.num_experts)
            for i in range(M)
        ]
        total_sum = sum(evidence_totals) + eps
        probs = [e / total_sum for e in evidence_totals]

        # 按概率选择
        r = random.random()
        cumulative = 0.0
        backbone_idx = M - 1
        for i, p in enumerate(probs):
            cumulative += p
            if r < cumulative:
                backbone_idx = i
                break

        new_model = copy.deepcopy(models[0])

        with torch.no_grad():
            new_model.backbone.load_state_dict(
                models[backbone_idx].backbone.state_dict()
            )

            # ======== Expert: M×E 候选池贪心匹配 ========
            # 收集所有有效候选: (model_idx, expert_idx, signature)
            candidates = []
            for i in range(M):
                for e in range(self.num_experts):
                    sig = signatures[i].get(e)
                    if sig is not None and sig.norm() >= eps:
                        candidates.append((i, e, sig))

            # 贪心匹配
            used = set()  # (model_idx, expert_idx)
            assignment = {}  # target_e -> (model_idx, expert_idx)

            for e in range(self.num_experts):
                # 锚点: 按模型顺序找第一个有效签名
                anchor = None
                for i in range(M):
                    sig = signatures[i].get(e)
                    if sig is not None and sig.norm() >= eps:
                        anchor = sig
                        break

                if anchor is None or len(candidates) == 0:
                    # 无信号，随机选一个父模型的位置 e
                    rand_model = random.randint(0, M - 1)
                    assignment[e] = (rand_model, e)
                    used.add((rand_model, e))
                    continue

                # 从未使用的候选中找最相似的
                best_sim = -2.0
                best_cand = None
                for (mi, ei, sig_cand) in candidates:
                    if (mi, ei) in used:
                        continue
                    sim = _cosine_similarity(anchor, sig_cand)
                    if sim > best_sim:
                        best_sim = sim
                        best_cand = (mi, ei)

                if best_cand is None:
                    rand_model = random.randint(0, M - 1)
                    assignment[e] = (rand_model, e)
                else:
                    assignment[e] = best_cand
                    used.add(best_cand)

            # ======== 复制 Expert 参数 ========
            for e, (mi, ei) in assignment.items():
                new_model.experts.experts[e].load_state_dict(
                    models[mi].experts.experts[ei].state_dict()
                )

            # ======== Router 逐行跟着 expert ========
            router_W_new = new_model.router.evidence_network.weight.data
            router_b_new = new_model.router.evidence_network.bias.data

            for e, (mi, ei) in assignment.items():
                src_W = models[mi].router.evidence_network.weight.data
                src_b = models[mi].router.evidence_network.bias.data
                router_W_new[e] = src_W[ei].clone()
                router_b_new[e] = src_b[ei].clone()

        return new_model


# ============================================================
# Multi-Parent Adaptive Symbiosis
# ============================================================

class EvidenceAdaptiveSymbiosis:
    """
    多父代自适应共生。

    两种调度模式:
    - 'round':     按轮数调度 (和原 FedEMoE 一致)
    - 'diversity': 按池多样性分数自适应调度
    """

    def __init__(
        self,
        num_experts: int,
        num_classes: int,
        mode: str = "round",
        endo_ratio: float = 0.5,
        switch_round: int = 100,
        diversity_threshold: float = 0.3,
        smoothing_alpha: float = 0.0,
    ):
        self.endo = EvidenceEndoSymbiosis(
            num_experts, num_classes, smoothing_alpha=smoothing_alpha
        )
        self.ecto = EvidenceEctoSymbiosis(num_experts, num_classes)
        self.mode = mode
        self.endo_ratio = endo_ratio
        self.switch_round = switch_round
        self.diversity_threshold = diversity_threshold
        self.num_experts = num_experts

    def _endo_prob_by_round(self, current_round: int) -> float:
        if current_round < self.switch_round:
            return self.endo_ratio * (current_round / self.switch_round)
        return self.endo_ratio + (1 - self.endo_ratio) * min(
            1.0, (current_round - self.switch_round) / self.switch_round
        )

    def _endo_prob_by_diversity(self, pool_signatures) -> float:
        div = _pool_diversity_score(pool_signatures, self.num_experts)
        return min(1.0, max(0.0, div / (2 * self.diversity_threshold)))

    def symbiose(
        self,
        models: List[nn.Module],
        signatures: List[Dict[int, torch.Tensor]],
        current_round: int = 0,
        pool_signatures: Optional[List[Dict[int, torch.Tensor]]] = None,
    ) -> nn.Module:
        if self.mode == "diversity" and pool_signatures is not None:
            endo_prob = self._endo_prob_by_diversity(pool_signatures)
        else:
            endo_prob = self._endo_prob_by_round(current_round)

        if random.random() < endo_prob:
            return self.endo.symbiose(models, signatures)
        else:
            return self.ecto.symbiose(models, signatures)


# ============================================================
# Evidence-Guided Symbiosis Pool
# ============================================================

class EvidenceGuidedSymbiosisPool:
    """
    多父代证据引导的共生池。

    核心参数 num_parents (M):
    - M=2: 传统两父共生
    - M=K: 全池 CEGA 聚合（池多样性消失）
    - 推荐 M=4-6 (覆盖足够多类别，同时保持池多样性)
    """

    def __init__(
        self,
        pool_size: int,
        model_template: nn.Module,
        num_experts: int,
        num_classes: int,
        num_parents: int = 5,
        symbiosis_mode: str = "adaptive",
        adaptive_mode: str = "round",
        endo_ratio: float = 0.5,
        switch_round: int = 100,
        diversity_threshold: float = 0.3,
        smoothing_alpha: float = 0.0,
        ema_momentum: float = 1.0,
    ):
        self.pool_size = pool_size
        self.num_experts = num_experts
        self.num_classes = num_classes
        self.num_parents = min(num_parents, pool_size - 1)
        self.ema_momentum = ema_momentum  # 1.0 = 完全覆盖（无EMA），0.7 = 保留30%旧模型

        self.models: List[nn.Module] = [
            copy.deepcopy(model_template) for _ in range(pool_size)
        ]
        self.signatures: List[Dict[int, torch.Tensor]] = [
            {} for _ in range(pool_size)
        ]

        # 池快照: 用于 EMA 时保留上一轮广谱共生结果
        self._pool_snapshot: Optional[List[nn.Module]] = None

        self.symbiosis_mode = symbiosis_mode
        self.adaptive_mode = adaptive_mode

        if symbiosis_mode == "endo":
            self.symbiosis = EvidenceEndoSymbiosis(
                num_experts, num_classes, smoothing_alpha=smoothing_alpha
            )
            self._needs_round = False
        elif symbiosis_mode == "ecto":
            self.symbiosis = EvidenceEctoSymbiosis(num_experts, num_classes)
            self._needs_round = False
        else:  # adaptive
            self.symbiosis = EvidenceAdaptiveSymbiosis(
                num_experts=num_experts,
                num_classes=num_classes,
                mode=adaptive_mode,
                endo_ratio=endo_ratio,
                switch_round=switch_round,
                diversity_threshold=diversity_threshold,
                smoothing_alpha=smoothing_alpha,
            )
            self._needs_round = True

    def get_model(self, idx: int) -> nn.Module:
        return self.models[idx]

    def set_model(
        self,
        idx: int,
        model: nn.Module,
        signature: Optional[Dict[int, torch.Tensor]] = None,
    ):
        self.models[idx] = model
        if signature is not None:
            self.signatures[idx] = signature

    def get_signature(self, idx: int) -> Dict[int, torch.Tensor]:
        return self.signatures[idx]

    def save_pool_snapshot(self):
        """
        保存当前池模型的快照（深拷贝）。

        在 receive_models 覆盖池之前调用，这样快照保留的是
        上一轮共生后的广谱模型，用于后续 EMA 计算。
        """
        self._pool_snapshot = [
            copy.deepcopy(m) for m in self.models
        ]

    def perform_symbiosis(self, current_round: int = 0) -> List[nn.Module]:
        """
        对池中每个位置执行多父代共生。

        对每个位置 i，从其他 K-1 个位置中随机选 M 个作为父代，
        用它们的模型和签名生成新模型。

        不同位置选的 M 个父代不同 → 天然保持池多样性。
        """
        new_models = []

        for i in range(self.pool_size):
            candidates = [j for j in range(self.pool_size) if j != i]

            if len(candidates) < self.num_parents:
                # 候选不够，用全部
                selected = candidates
            else:
                selected = random.sample(candidates, self.num_parents)

            if len(selected) == 0:
                new_models.append(copy.deepcopy(self.models[i]))
                continue

            parent_models = [self.models[j] for j in selected]
            parent_sigs = [self.signatures[j] for j in selected]

            if isinstance(self.symbiosis, EvidenceAdaptiveSymbiosis):
                new_model = self.symbiosis.symbiose(
                    parent_models, parent_sigs,
                    current_round=current_round,
                    pool_signatures=self.signatures,
                )
            else:
                new_model = self.symbiosis.symbiose(
                    parent_models, parent_sigs
                )

            new_models.append(new_model)

        return new_models

    def update_pool(self, new_models: List[nn.Module]):
        """
        用新模型更新池。

        如果 ema_momentum < 1.0，对每个位置做 EMA:
            pool[i] = (1 - mu) × snapshot[i] + mu × new_models[i]

        其中 snapshot 是 receive_models 之前保存的上一轮广谱共生结果，
        new_models 是本轮共生产生的广谱模型。两侧都是广谱模型，
        保证 EMA 在同质模型之间进行。

        如果没有快照（首轮或 ema_momentum=1.0），直接覆盖。
        """
        assert len(new_models) == self.pool_size

        if self.ema_momentum >= 1.0:
            self.models = new_models
            self._pool_snapshot = None
            return

        mu = self.ema_momentum

        # 确定 EMA 基准: 优先使用快照（上一轮广谱），无快照时直接覆盖
        ema_base = self._pool_snapshot
        if ema_base is None:
            self.models = new_models
            self._pool_snapshot = None
            return

        with torch.no_grad():
            for i in range(self.pool_size):
                base_model = ema_base[i]       # 上一轮广谱共生结果
                new_model = new_models[i]       # 本轮广谱共生结果
                # 结果写入 base_model 就地修改, 然后赋给 pool
                for p_base, p_new in zip(
                    base_model.parameters(), new_model.parameters()
                ):
                    p_base.data.mul_(1.0 - mu).add_(mu * p_new.data)

                base_bufs = dict(base_model.named_buffers())
                new_bufs = dict(new_model.named_buffers())
                for name, buf_base in base_bufs.items():
                    if 'running_mean' in name or 'running_var' in name:
                        if name in new_bufs:
                            buf_base.data.mul_(1.0 - mu).add_(
                                mu * new_bufs[name].data
                            )

                self.models[i] = base_model

        # 快照已使用，清空
        self._pool_snapshot = None

    def aggregate_to_global(
        self, weights: Optional[List[float]] = None
    ) -> nn.Module:
        if weights is None:
            weights = [1.0 / self.pool_size] * self.pool_size
        total = sum(weights)
        weights = [w / total for w in weights]

        global_model = copy.deepcopy(self.models[0])
        with torch.no_grad():
            for param in global_model.parameters():
                param.data.zero_()
            for model, w in zip(self.models, weights):
                for p_g, p_m in zip(
                    global_model.parameters(), model.parameters()
                ):
                    p_g.data.add_(w * p_m.data)

            global_bufs = dict(global_model.named_buffers())
            for name, buf in global_bufs.items():
                if 'running_mean' in name or 'running_var' in name:
                    buf.data.zero_()
            for model, w in zip(self.models, weights):
                m_bufs = dict(model.named_buffers())
                for name, buf_g in global_bufs.items():
                    if 'running_mean' in name or 'running_var' in name:
                        if name in m_bufs:
                            buf_g.data.add_(w * m_bufs[name].data)

        return global_model

    def get_pool_diversity(self) -> float:
        return _pool_diversity_score(self.signatures, self.num_experts)