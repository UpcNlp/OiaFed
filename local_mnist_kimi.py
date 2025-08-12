"""
brain_like_fix.py
三时间尺度突触 + 无任务 ID 回放
pip install torch torchvision tqdm
python brain_like_fix.py
"""

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np
import random

DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH    = 64
LR       = 1e-3
EPOCHS   = 3
LAMBDA_EWC = 2000
RANK     = 4
REPLAY   = 200
KL_THRESH = 0.15
SEED     = 42
torch.manual_seed(SEED); np.random.seed(SEED)

# ---------- Conv 特征提取 ----------
class ConvNet(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(64*4*4, hidden)
        )
    def forward(self, x): return self.f(x)

# ---------- 三记忆模块 ----------
class TriSynapse(nn.Module):
    def __init__(self, hidden=256, n_classes=10):
        super().__init__()
        self.backbone = ConvNet(hidden)
        # 慢权重：冻结
        for p in self.backbone.parameters():
            p.requires_grad = False
        # 中权重：LoRA 低秩 + EWC
        self.mid_A = nn.Parameter(torch.randn(hidden, RANK) * 0.01)
        self.mid_B = nn.Parameter(torch.zeros(RANK, hidden))
        # 快权重：对角矩阵，每样本更新
        self.register_buffer('fast', torch.zeros(hidden))
        # 分类头
        self.head = nn.Linear(hidden, n_classes)
        # EWC
        self.fisher = {}
        self.optpar = {}

    def forward(self, x):
        z = self.backbone(x)
        z = z + torch.einsum('bi,ir,rj->bj', z, self.mid_A, self.mid_B)
        z = z * self.fast            # 逐通道乘快权重
        return self.head(z)

    # ---------- EWC ----------
    def estimate_fisher(self, loader):
        self.eval()
        fisher = {n: torch.zeros_like(p) for n, p in self.named_parameters() if p.requires_grad}
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            self.zero_grad()
            loss = F.cross_entropy(self.forward(x), y)
            loss.backward()
            for n, p in self.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.pow(2) * x.size(0)
        for n in fisher:
            fisher[n] /= len(loader.dataset)
        self.fisher = fisher
        self.optpar = {n: p.clone() for n, p in self.named_parameters() if p.requires_grad}

    def ewc_loss(self):
        if not self.fisher: return 0.
        return LAMBDA_EWC * sum((self.fisher[n] * (p - self.optpar[n]).pow(2)).sum()
                                for n, p in self.named_parameters() if p.requires_grad)

    # ---------- 无监督漂移 ----------
    def drift_detect(self, logits):
        if not hasattr(self, '_prev_logits'):
            self._prev_logits = logits.detach().mean(0)
            return True
        kl = F.kl_div(
            F.log_softmax(logits.mean(0), dim=0),
            F.softmax(self._prev_logits, dim=0),
            reduction='sum'
        ).item()
        self._prev_logits = logits.detach().mean(0)
        return kl > KL_THRESH

    # ---------- 快权重 Hebb ----------
    def update_fast(self, x, y):
        with torch.no_grad():
            z = self.backbone(x)
            z = z + torch.einsum('bi,ir,rj->bj', z, self.mid_A, self.mid_B)
            # 逐通道 Hebb: Δw_i = η * mean(z_i) * y_signal
            y_oh = F.one_hot(y, 10).float()
            y_signal = y_oh.sum(0).mean()  # 简化的学习信号
            delta = 0.01 * z.mean(0) * y_signal  # 确保 delta 形状是 (hidden,)
            self.fast = 0.9 * self.fast + delta
            self.fast = torch.clamp(self.fast, -1, 1)

# ---------- 数据 ----------
def get_tasks():
    trans = transforms.Compose([transforms.ToTensor()])  # 移除展平操作，保持 [1, 28, 28] 形状
    full = datasets.MNIST('./data', train=True,  download=True, transform=trans)
    test = datasets.MNIST('./data', train=False, download=True, transform=trans)
    tasks = []
    for i in range(0, 10, 2):
        tr_idx = (full.targets >= i) & (full.targets < i+2)
        te_idx = (test.targets >= i) & (test.targets < i+2)
        tasks.append((
            Subset(full, tr_idx.nonzero().squeeze()),
            Subset(test , te_idx.nonzero().squeeze())
        ))
    return tasks

def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

# ---------- 任务无关回放 ----------
class ReplayDataset(torch.utils.data.Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        return self.data_list[idx]

class ProtoReplay:
    def __init__(self, size):
        self.size = size
        self.pool = []

    def add(self, loader):
        for x_batch, y_batch in loader:
            # 将批次中的每个样本单独添加到池中
            for i in range(x_batch.size(0)):
                # 确保数据和标签都是正确的张量格式
                x_sample = x_batch[i].clone()
                
                # 处理标签：确保是0维标量张量
                if isinstance(y_batch, torch.Tensor):
                    y_sample = y_batch[i]
                    # 如果标签是标量张量但有多余的维度，去掉多余维度
                    if y_sample.dim() > 0:
                        y_sample = y_sample.squeeze()
                    # 确保是长整型
                    y_sample = y_sample.long()
                else:
                    # 如果不是张量，直接转换
                    y_sample = torch.tensor(y_batch[i], dtype=torch.long)
                
                self.pool.append((x_sample, y_sample))
                if len(self.pool) > self.size:
                    self.pool.pop(random.randrange(len(self.pool)))

    def get_dataset(self):
        return ReplayDataset(self.pool)

    def sample(self):
        return DataLoader(self.pool, BATCH, shuffle=True)

def main():
    tasks = get_tasks()
    model = TriSynapse().to(DEVICE)
    opt   = torch.optim.Adam([model.mid_A, model.mid_B, model.head.weight], lr=LR)
    replay = ProtoReplay(REPLAY)
    acc = np.zeros((5, 5))

    for task_id, (train_ds, test_ds) in enumerate(tasks):
        train_loader = DataLoader(train_ds, BATCH, shuffle=True)
        test_loader  = DataLoader(test_ds, BATCH)

        # 1. 先收集回放
        replay.add(train_loader)

        # 2. 混合训练（当前 + 回放）
        if replay.pool:  # 只有当回放池不为空时才混合
            replay_dataset = replay.get_dataset()
            mixed_dataset = ConcatDataset([train_ds, replay_dataset])
            # 使用自定义collate函数确保所有数据都是张量
            def collate_fn(batch):
                xs, ys = zip(*batch)
                xs = torch.stack([x if isinstance(x, torch.Tensor) else torch.tensor(x) for x in xs])
                ys = torch.stack([y if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=torch.long) for y in ys])
                return xs, ys
            mixed_loader = DataLoader(mixed_dataset, BATCH, shuffle=True, collate_fn=collate_fn)
        else:
            # 对原始数据也使用相同的collate函数
            def collate_fn(batch):
                xs, ys = zip(*batch)
                xs = torch.stack([x if isinstance(x, torch.Tensor) else torch.tensor(x) for x in xs])
                ys = torch.stack([y if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=torch.long) for y in ys])
                return xs, ys
            mixed_loader = DataLoader(train_ds, BATCH, shuffle=True, collate_fn=collate_fn)

        print(f"\nTask {task_id} classes {task_id*2}-{task_id*2+1}")
        for epoch in range(EPOCHS):
            model.train()
            for x, y in mixed_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                opt.zero_grad()
                loss = F.cross_entropy(model(x), y) + model.ewc_loss()
                loss.backward()
                opt.step()

        # 3. 记录 Fisher
        model.estimate_fisher(train_loader)

        # 4. 快权重在线微调
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            if model.drift_detect(model(x)):
                model.update_fast(x, y)

        # 5. 评估
        for test_task_id in range(task_id+1):
            test_loader = DataLoader(tasks[test_task_id][1], BATCH)
            acc[task_id, test_task_id] = evaluate(model, test_loader)
            print(f"  Test Task{test_task_id}: {acc[task_id, test_task_id]:.3f}")

    print("\nAccuracy matrix (行=训练任务，列=测试任务)")
    print(np.round(acc, 3))

if __name__ == '__main__':
    main()