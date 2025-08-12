import math
import logging
from collections import defaultdict, Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm
import random

# ============ 日志设置 ============
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("CIL-MoE")

# ============ 随机种子 ============
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============ 数据准备：MNIST，5 个任务（每次 2 类）===========
TASKS = [[0,1],[2,3],[4,5],[6,7],[8,9]]

def get_task_indices(targets, classes):
    idx = [i for i, y in enumerate(targets) if int(y) in classes]
    return idx

def get_mnist_splits(batch_size=128):
    tr = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = datasets.MNIST(root="./data", train=True, download=True, transform=tr)
    test_set  = datasets.MNIST(root="./data", train=False, download=True, transform=tr)

    train_loaders = []
    test_loaders  = []

    for t, cls in enumerate(TASKS):
        tr_idx = get_task_indices(train_set.targets, cls)
        te_idx = get_task_indices(test_set.targets,  cls)
        train_sub = Subset(train_set, tr_idx)
        test_sub  = Subset(test_set,  te_idx)
        train_loaders.append(DataLoader(train_sub, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True))
        test_loaders.append(DataLoader(test_sub,  batch_size=batch_size, shuffle=False, num_workers=2))
    return train_loaders, test_loaders

# ============ 工具 ============
def angle_deg(u, v, eps=1e-7):
    u = F.normalize(u, dim=-1, eps=eps)
    v = F.normalize(v, dim=-1, eps=eps)
    cosv = torch.clamp((u*v).sum(dim=-1), -1.0 + eps, 1.0 - eps)
    return torch.arccos(cosv) * 180.0 / math.pi

def has_nan(t, name):
    if torch.isnan(t).any():
        tmin = float(torch.nanmin(t))
        tmax = float(torch.nanmax(t))
        logger.error(f"[NaN-Guard] {name} has NaN! range=({tmin:.4f},{tmax:.4f})")
        return True
    return False

# ============ 模型 ============
class SmallCNN(nn.Module):
    """1x28x28 -> 128-d L2 归一化嵌入"""
    def __init__(self, out_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 7x7
        )
        self.fc = nn.Linear(64*7*7, out_dim)

    def forward(self, x):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        z = self.fc(h)
        z = F.normalize(z, dim=1, eps=1e-7)   # FIX: eps
        return z

class ExpertBank(nn.Module):
    """
    维护 E 个专家：
      A_e: 慢锚(Ke+) -> 路由/记忆，不吃梯度，只 EMA 更新
      C_e: 快原型     -> 可训练，吃梯度（被正样本拉近）
      B_e: 反锚(Ke-)  -> 仅在“被侵入”时 EMA
    """
    def __init__(self, E=12, dim=128, k_top=2, beta=0.5,
                 tau_route=0.2, tau_margin=0.1, tau_intrude=0.3,
                 m_grow=0.95, m_stable=0.99, m_sentry=0.999,
                 delta_max_deg=3.0, delta_max_neg_deg=5.0, gamma_neg=0.99):
        super().__init__()
        self.E = E
        self.dim = dim
        self.k_top = k_top
        self.beta = beta

        self.tau_route  = tau_route
        self.tau_margin = tau_margin
        self.tau_intrude = tau_intrude

        self.m_states = {0: m_grow, 1: m_stable, 2: m_sentry}
        self.state = torch.zeros(E, dtype=torch.long)

        self.delta_max = math.radians(delta_max_deg)
        self.delta_max_neg = math.radians(delta_max_neg_deg)
        self.gamma_neg = gamma_neg

        # 初始化
        A = torch.randn(E, dim); A = F.normalize(A, dim=1)
        C = torch.randn(E, dim); C = F.normalize(C, dim=1)
        B = torch.randn(E, dim); B = F.normalize(B, dim=1)

        self.A = nn.Parameter(A, requires_grad=False)
        self.C = nn.Parameter(C, requires_grad=True)
        self.B = nn.Parameter(B, requires_grad=False)

        self.w_importance = nn.Parameter(torch.ones(E), requires_grad=False)
        self.active = torch.zeros(E, dtype=torch.bool)

        self.register_buffer("A_prev", self.A.detach().clone())
        self.register_buffer("B_prev", self.B.detach().clone())
        self.intrude_count = torch.zeros(E, dtype=torch.long)

    @torch.no_grad()
    def set_active(self, idxs, value=True):
        self.active[idxs] = value

    def forward_scores(self, z):
        # [B,E]
        cosA = torch.clamp(torch.matmul(z, self.A.t()), -1.0, 1.0)
        cosB = torch.clamp(torch.matmul(z, self.B.t()), -1.0, 1.0)
        s = cosA - self.beta * cosB
        topk_val, topk_idx = torch.topk(s, k=self.k_top, dim=1)
        return s, topk_idx, cosA, cosB

    def eligibility(self, z, e_star, cosA_row, cosB_row):
        s_e = cosA_row[e_star] - self.beta * cosB_row[e_star]
        gap = cosA_row[e_star] - self.beta * torch.max(cosB_row)
        ok = (s_e >= self.tau_route) and (gap >= self.tau_margin)
        return ok, float(s_e), float(gap)

    @torch.no_grad()
    def _angle_clip(self, vec_new, vec_old, max_rad, eps=1e-7):
        vec_new = F.normalize(vec_new, dim=-1, eps=eps)
        vec_old = F.normalize(vec_old, dim=-1, eps=eps)
        cosang = torch.clamp((vec_new * vec_old).sum(), -1.0 + eps, 1.0 - eps)
        ang = torch.arccos(cosang)
        if ang <= max_rad:
            return vec_new
        t = float(max_rad / (ang + 1e-12))
        # slerp
        sin_ang = torch.sin(ang) + 1e-12
        part1 = torch.sin((1-t)*ang)/sin_ang * vec_old
        part2 = torch.sin(t*ang)/sin_ang * vec_new
        vec_clip = F.normalize(part1 + part2, dim=-1, eps=eps)
        return vec_clip

    @torch.no_grad()
    def ema_update_A(self, e, m):
        A_old = self.A.data[e].clone()
        A_new = F.normalize(m*self.A.data[e] + (1-m)*self.C.data[e], dim=0, eps=1e-7)
        A_new = self._angle_clip(A_new, A_old, self.delta_max)
        self.A.data[e].copy_(A_new)         # FIX: 原地 copy

    @torch.no_grad()
    def ema_update_B(self, e, z):
        B_old = self.B.data[e].clone()
        B_new = F.normalize(self.gamma_neg*self.B.data[e] + (1-self.gamma_neg)*z, dim=0, eps=1e-7)
        B_new = self._angle_clip(B_new, B_old, self.delta_max_neg)
        self.B.data[e].copy_(B_new)         # FIX: 原地 copy

    @torch.no_grad()
    def anchor_drift_deg_and_reset(self):
        drift_A = angle_deg(self.A.data, self.A_prev)
        drift_B = angle_deg(self.B.data, self.B_prev)
        self.A_prev.copy_(self.A.data)
        self.B_prev.copy_(self.B.data)
        return drift_A.cpu().numpy(), drift_B.cpu().numpy()

# ============ 损失（数值稳定） ============
def info_nce_route_stable(z, A, B, e_star, tau=0.07, beta=0.5):
    """
    单样本 InfoNCE（logsumexp 版本，数值稳定）
    """
    simA = torch.mv(A, z)            # [E]
    simB = torch.mv(B, z)            # [E]
    pos = simA[e_star] / tau
    # logsumexp over all negatives + positive
    all_terms = torch.cat([simA/tau, beta*simB/tau])  # [E + E]
    lse = torch.logsumexp(all_terms, dim=0)
    # 但要把正项算一次，negA/negB也在 all_terms 中
    # 常见做法是：-pos + logsumexp([pos, negA..., negB...])
    loss = -pos + torch.logsumexp(torch.cat([pos.unsqueeze(0), all_terms]), dim=0)
    return loss

def hinge_gap(z, A_e, B, m_gap=0.2, beta=0.5):
    cosA = torch.dot(z, A_e)
    cosB_max = torch.max(torch.mv(B, z))
    gap = cosA - beta * cosB_max
    loss = F.relu(m_gap - gap)
    return loss, float(gap)

def prox_loss(C_e, A_e, w=1.0, lam=1e-3):
    ang = angle_deg(C_e.unsqueeze(0), A_e.unsqueeze(0))[0] * math.pi / 180.0
    return lam * w * (ang**2)

def pos_pull_loss(z, C_e, alpha=1.0):
    # 把快原型朝当前正样本方向拉近
    return -alpha * torch.dot(z.detach(), F.normalize(C_e, dim=0, eps=1e-7))

# ============ 评测：专家多数映射 ============
class ExpertLabelMap:
    def __init__(self, num_experts, num_classes=10):
        self.counts = np.zeros((num_experts, num_classes), dtype=np.int64)

    def update(self, e_star, y_true):
        self.counts[e_star, int(y_true)] += 1

    def majority_label(self, e):
        row = self.counts[e]
        if row.sum() == 0:
            return None
        return int(row.argmax())

    def predict_label(self, e):
        return self.majority_label(e)

    def __repr__(self):
        tops = []
        for e in range(self.counts.shape[0]):
            if self.counts[e].sum() == 0:
                continue
            tops.append(f"e{e}-> {self.counts[e].argmax()} (n={self.counts[e].max()})")
        return " | ".join(tops)

# ============ 训练一个任务 ============
def train_one_task(task_id, classes, encoder, bank, train_loader, label_map,
                   epochs=2, lr=1e-3, tau=0.07, beta=0.5, m_gap=0.2,
                   lam_prox=1e-3, alpha_pos=0.5):
    encoder.train()
    opt = torch.optim.Adam(list(encoder.parameters()) + [bank.C], lr=lr)

    for ep in range(1, epochs+1):
        gaps = []
        intrusions_this_epoch = 0
        nan_flag = False

        for x, y in tqdm(train_loader, desc=f"Task {task_id} Epoch {ep}", leave=False):
            x, y = x.to(device), y.to(device)
            z = encoder(x)
            if has_nan(z, "z"): nan_flag = True; break

            s, topk_idx, cosA_all, cosB_all = bank.forward_scores(z)
            batch_loss = 0.0

            for i in range(z.size(0)):
                e_star = int(torch.argmax(s[i]).item())

                # InfoNCE（稳定版）
                l_nce = info_nce_route_stable(
                    z[i], bank.A.data, bank.B.data, e_star, tau=tau, beta=beta
                )

                # gap hinge
                l_gap, gap_val = hinge_gap(z[i], bank.A.data[e_star], bank.B.data, m_gap=m_gap, beta=beta)

                # prox 近端
                l_prox = prox_loss(bank.C[e_star], bank.A.data[e_star], w=float(bank.w_importance[e_star].item()), lam=lam_prox)

                # 正样本拉近 C_e（关键修复）
                l_pos = pos_pull_loss(z[i], bank.C[e_star], alpha=alpha_pos)  # FIX

                loss_i = l_nce + l_gap + l_prox + l_pos
                batch_loss += loss_i

                gaps.append(gap_val)
                label_map.update(e_star, int(y[i].item()))

            batch_loss = batch_loss / z.size(0)

            opt.zero_grad()
            batch_loss.backward()
            # 梯度裁剪（防守）
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + [bank.C], max_norm=5.0)
            opt.step()

            # 规范化快原型
            with torch.no_grad():
                bank.C.data.copy_(F.normalize(bank.C.data, dim=1, eps=1e-7))

            # === 锚点与反锚更新（EMA + 角度裁剪） ===
            with torch.no_grad():
                for i in range(z.size(0)):
                    e_star = int(torch.argmax(s[i]).item())

                    ok, s_e, gap_val = bank.eligibility(z[i], e_star, cosA_all[i], cosB_all[i])
                    if ok:
                        m = bank.m_states[int(bank.state[e_star].item())]
                        bank.ema_update_A(e_star, m)

                    # 侵入检测
                    cosA_row = cosA_all[i]
                    for j in range(bank.E):
                        if j == e_star:
                            continue
                        if cosA_row[j] >= bank.tau_intrude:
                            bank.ema_update_B(j, z[i])
                            bank.intrude_count[j] += 1
                            intrusions_this_epoch += 1

            # NaN 哨兵
            if has_nan(bank.A.data, "A") or has_nan(bank.B.data, "B") or has_nan(bank.C.data, "C"):
                nan_flag = True
                break

        drift_A, drift_B = bank.anchor_drift_deg_and_reset()
        logger.info(f"[Task {task_id}][Epoch {ep}] "
                    f"loss={float(batch_loss):.4f} "
                    f"avg_gap={np.nanmean(gaps) if gaps else float('nan'):.4f} | "
                    f"A_drift_deg(mean)={np.nanmean(drift_A):.3f} "
                    f"B_drift_deg(mean)={np.nanmean(drift_B):.3f} "
                    f"| intrusions={intrusions_this_epoch} "
                    f"| map: {label_map}")

        if nan_flag:
            logger.error(f"[Task {task_id}][Epoch {ep}] NaN detected. Stopping this task early.")
            break

# ============ 新类初始化 ============
@torch.no_grad()
def initialize_new_classes(task_id, classes, encoder, bank, train_loader, label_map, warm_k=128):
    """
    用 warm_k 个样本均值初始化每个新类的一个专家：
      C_e <- mean_z, A_e <- C_e，B_e <- 随机正交
    """
    if len(classes) == 0:
        return
    encoder.eval()
    cls_vecs = {c: [] for c in classes}
    collected = {c: 0 for c in classes}
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        z = encoder(x)
        for i in range(z.size(0)):
            yi = int(y[i].item())
            if yi in classes and collected[yi] < warm_k:
                cls_vecs[yi].append(z[i].detach().cpu())
                collected[yi] += 1
        if all(collected[c] >= warm_k for c in classes):
            break

    for c in classes:
        if len(cls_vecs[c]) == 0:
            logger.warning(f"[Task {task_id}] class {c} not found in warm-up; skip init")
            continue
        zmean = torch.stack(cls_vecs[c], dim=0).mean(dim=0)
        zmean = F.normalize(zmean, dim=0, eps=1e-7).to(device)

        # 找未激活 expert
        free_idx = None
        for e in range(bank.E):
            if not bool(bank.active[e].item()):
                free_idx = e; break
        if free_idx is None:
            norms = angle_deg(bank.A.data, bank.C.data).cpu().numpy()
            free_idx = int(np.argmin(norms))
            logger.warning(f"[Task {task_id}] No free expert; reusing e={free_idx}")

        # FIX: 原地拷贝，不创建新的 Parameter
        bank.C.data[free_idx].copy_(zmean)
        bank.A.data[free_idx].copy_(zmean)

        rand = torch.randn_like(zmean)
        ortho = rand - (rand @ zmean) * zmean
        if ortho.norm() < 1e-6:
            ortho = torch.randn_like(zmean)
        bank.B.data[free_idx].copy_(F.normalize(ortho, dim=0, eps=1e-7))

        bank.set_active(torch.tensor([free_idx]), True)
        logger.info(f"[Task {task_id}] Init class {c} -> expert e{free_idx}")

# ============ 评测 ============
@torch.no_grad()
def eval_on_loader(encoder, bank, label_map, loader):
    encoder.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        z = encoder(x)
        s, topk_idx, _, _ = bank.forward_scores(z)
        e_star = torch.argmax(s, dim=1)
        pred = []
        for i in range(e_star.size(0)):
            e = int(e_star[i].item())
            lab = label_map.predict_label(e)
            pred.append(-1 if lab is None else lab)
        pred = torch.tensor(pred, device=device)
        correct += (pred == y).sum().item()
        total   += y.numel()
    return correct / max(total, 1)

# ============ 主流程 ============
def main():
    embed_dim = 128
    E = 12
    k_top = 2
    beta = 0.5
    tau = 0.07
    tau_route = 0.2
    tau_margin = 0.1
    tau_intrude = 0.3
    m_gap = 0.2
    lam_prox = 1e-3
    alpha_pos = 0.5   # FIX: 正样本拉近强度
    lr = 1e-3
    epochs_per_task = 2
    batch_size = 128

    train_loaders, test_loaders = get_mnist_splits(batch_size=batch_size)

    encoder = SmallCNN(out_dim=embed_dim).to(device)
    bank = ExpertBank(
        E=E, dim=embed_dim, k_top=k_top, beta=beta,
        tau_route=tau_route, tau_margin=tau_margin, tau_intrude=tau_intrude,
        m_grow=0.95, m_stable=0.99, m_sentry=0.999,
        delta_max_deg=3.0, delta_max_neg_deg=5.0, gamma_neg=0.99
    ).to(device)

    label_map = ExpertLabelMap(num_experts=E, num_classes=10)
    ACC = np.zeros((len(TASKS), len(TASKS)), dtype=np.float32)

    seen_classes = set()
    for t, classes in enumerate(TASKS):
        logger.info(f"========== Start Task {t}: classes {classes} ==========")

        new_classes = [c for c in classes if c not in seen_classes]
        initialize_new_classes(t, new_classes, encoder, bank, train_loaders[t], label_map, warm_k=128)
        for c in new_classes:
            seen_classes.add(c)

        train_one_task(
            task_id=t, classes=classes,
            encoder=encoder, bank=bank, train_loader=train_loaders[t], label_map=label_map,
            epochs=epochs_per_task, lr=lr, tau=tau, beta=beta, m_gap=m_gap,
            lam_prox=lam_prox, alpha_pos=alpha_pos
        )

        for j in range(t+1):
            acc = eval_on_loader(encoder, bank, label_map, test_loaders[j])
            ACC[t, j] = acc
        logger.info(f"[After Task {t}] Acc row (0..{t}) = {np.round(ACC[t,:t+1], 4)}")

    logger.info("========== Accuracy Matrix (rows: after task t, cols: on task j) ==========")
    for i in range(len(TASKS)):
        row = " ".join([f"{ACC[i,j]:.4f}" for j in range(len(TASKS))])
        logger.info(f"t={i}: {row}")

    print("\n=== Final Accuracy Matrix ===")
    for i in range(len(TASKS)):
        row = " ".join([f"{ACC[i,j]:.4f}" for j in range(len(TASKS))])
        print(f"After Task {i}: {row}")

if __name__ == "__main__":
    main()
