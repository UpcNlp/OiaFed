"""
SplitNN 训练器 — 编排纵向联邦 (VFL) 的 split learning 通信

与 HFL 的根本区别:
- HFL: 每轮 → 各客户端独立训练 → 聚合权重 → 广播
- VFL/SplitNN: 每个 batch → client forward → server forward+backward → client backward

流程:
1. 初始化: 分割全局模型 → client_layers 发给客户端, server_layers 留在 trainer
2. 每轮: 遍历训练数据, 每个 batch 做 split forward/backward
3. 评估: 完整 forward (client → server)

多客户端: 每轮各客户端轮流做一个 epoch 的 split 训练
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Optional

from ...registry import trainer
from ...core.types import RoundResult, RoundMetrics
from .default import DefaultTrainer

# 复用 SplitModel
from ..learners.vfl.splitnn import SplitModel


@trainer(
    name='SplitNNTrainer',
    description='SplitNN 训练器 — 纵向联邦 split learning',
    version='1.0',
    author='OiaFed',
    algorithms=['splitnn', 'vfl', 'split_learning']
)
class SplitNNTrainer(DefaultTrainer):
    """
    SplitNN 训练器

    配置参数:
        split_layer: int       — 模型切分层索引 (默认自动)
        learning_rate: float   — server 端学习率
        batch_size: int        — 批大小
        local_epochs: int      — 每轮每个客户端的 epoch 数
        eval_interval: int     — 评估间隔
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.split_layer = self.config.get('split_layer', -1)
        self.server_lr = self.config.get('learning_rate', 0.01)
        self.batch_size = self.config.get('batch_size', 64)
        self.local_epochs = self.config.get('local_epochs', 1)
        self.eval_interval = self.config.get('eval_interval', 2)

        # server 端组件 (在 first round 初始化)
        self.split_model: Optional[SplitModel] = None
        self.server_optimizer: Optional[optim.Optimizer] = None
        self.criterion = nn.CrossEntropyLoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._initialized = False

        self.logger.info(
            f"SplitNNTrainer: split_layer={self.split_layer}, "
            f"server_lr={self.server_lr}"
        )

    # ------------------------------------------------------------------
    #  初始化: 分割模型
    # ------------------------------------------------------------------
    async def _init_split(self):
        """首次调用时分割模型"""
        if self._initialized:
            return

        # 获取全局模型
        if hasattr(self.model, 'get_model'):
            base_model = self.model.get_model()
        elif hasattr(self.model, '_model'):
            base_model = self.model._model
        else:
            base_model = self.model

        # 分割
        self.split_model = SplitModel(base_model, self.split_layer)
        self.split_model.to(self.device)

        # server 端优化器 (只优化 server_layers)
        self.server_optimizer = optim.SGD(
            self.split_model.server_layers.parameters(),
            lr=self.server_lr,
            momentum=0.9,
        )

        client_layers_info = list(self.split_model.client_layers.state_dict().keys())
        server_layers_info = list(self.split_model.server_layers.state_dict().keys())
        self.logger.info(
            f"模型分割完成: client={len(client_layers_info)} 层, "
            f"server={len(server_layers_info)} 层"
        )

        self._initialized = True

    # ------------------------------------------------------------------
    #  Server 端前向 + 反向
    # ------------------------------------------------------------------
    def _server_forward_backward(
        self,
        smashed_data: torch.Tensor,
        labels: torch.Tensor,
    ):
        """
        Server 端: 接收 smashed_data → forward → loss → backward → 返回梯度

        Returns:
            (grad_smashed, loss_value, accuracy)
        """
        smashed_data = smashed_data.to(self.device).detach().requires_grad_(True)
        labels = labels.to(self.device)

        self.split_model.server_layers.train()
        output = self.split_model.server_forward(smashed_data)

        loss = self.criterion(output, labels)

        _, predicted = output.max(1)
        accuracy = predicted.eq(labels).sum().item() / labels.size(0)

        self.server_optimizer.zero_grad()
        loss.backward()

        grad_smashed = smashed_data.grad.clone().cpu()
        self.server_optimizer.step()

        return grad_smashed, loss.item(), accuracy

    # ------------------------------------------------------------------
    #  训练轮次
    # ------------------------------------------------------------------
    async def train_round(self, round_num: int) -> RoundResult:
        """
        一轮 SplitNN 训练

        每个客户端轮流做 local_epochs 个 epoch 的 split 训练:
          for batch in data:
            smashed = client.client_forward(batch)
            grad = server.forward_backward(smashed, labels)
            client.client_backward(grad)
        """
        await self._init_split()

        if self.callbacks:
            await self.callbacks.on_round_begin(self, round_num, {})

        connected = self.get_connected_learners()
        self.logger.info(
            f"\nRound {round_num}: SplitNN 训练, {len(connected)} 个客户端"
        )

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        batches_count = 0

        # 每个客户端轮流做 split 训练
        for learner in connected:
            learner_id = getattr(learner, '_target_id', 'unknown')

            # 确保 learner 已 setup
            try:
                await learner.setup({
                    "round_number": round_num,
                    "split_layer": self.split_layer,
                })
            except Exception as e:
                self.logger.warning(f"[{learner_id}] setup 异常 (可能已初始化): {e}")

            for epoch in range(self.local_epochs):
                # Trainer 侧驱动 split 训练: 逐 batch client↔server 交互
                epoch_loss, epoch_correct, epoch_samples = \
                    await self._drive_split_training(learner, learner_id)

                total_loss += epoch_loss
                total_correct += epoch_correct
                total_samples += epoch_samples
                batches_count += 1

                self.logger.info(
                    f"  [{learner_id}] Epoch {epoch}: "
                    f"Loss={epoch_loss/max(epoch_samples,1):.4f}, "
                    f"Acc={epoch_correct/max(epoch_samples,1):.4f}"
                )

        avg_loss = total_loss / max(total_samples, 1)
        avg_acc = total_correct / max(total_samples, 1)

        self.logger.info(
            f"Round {round_num} 完成: "
            f"Loss={avg_loss:.4f}, Acc={avg_acc:.4f}, "
            f"Samples={total_samples}"
        )

        # 评估
        if round_num % self.eval_interval == 0:
            await self._evaluate_split_model(round_num, connected)

        round_metrics = RoundMetrics(
            round_num=round_num,
            num_clients=len(connected),
            total_samples=total_samples,
            metrics={'accuracy': avg_acc, 'loss': avg_loss}
        )

        result = RoundResult(
            round_num=round_num,
            updates=[],
            aggregated_weights={},
            metrics=round_metrics,
            metadata={"mode": "splitnn"}
        )

        if self.callbacks:
            await self.callbacks.on_round_end(self, round_num, {
                "metrics": round_metrics
            })

        return result

    # ------------------------------------------------------------------
    #  Trainer 侧驱动 split 训练 (逐 batch)
    # ------------------------------------------------------------------
    async def _drive_split_training(self, learner, learner_id: str):
        """
        用 trainer 侧的数据驱动 split 训练

        对每个 batch:
        1. await learner.client_forward(batch) → smashed, labels
        2. server forward+backward → grad
        3. await learner.client_backward(grad)
        """
        # 从 trainer 的 datasets 创建 dataloader
        train_datasets = self._datasets.get("train", [])
        if not train_datasets:
            self.logger.warning("Trainer 没有训练数据")
            return 0.0, 0, 0

        loader = DataLoader(
            train_datasets[0],
            batch_size=self.batch_size,
            shuffle=True,
        )

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_idx, (data, labels) in enumerate(loader):
            try:
                # 1. Client forward
                smashed_result = await learner.client_forward((data, labels))

                # 解析返回值
                if isinstance(smashed_result, tuple):
                    smashed_data, ret_labels = smashed_result
                else:
                    smashed_data = smashed_result
                    ret_labels = labels

                # 2. Server forward + backward
                grad_smashed, loss_val, acc = self._server_forward_backward(
                    smashed_data, ret_labels
                )

                # 3. Client backward
                await learner.client_backward(grad_smashed)

                total_loss += loss_val * data.size(0)
                total_correct += acc * data.size(0)
                total_samples += data.size(0)

            except Exception as e:
                self.logger.error(
                    f"  [{learner_id}] Batch {batch_idx} 失败: {e}"
                )
                continue

        return total_loss, total_correct, total_samples

    # ------------------------------------------------------------------
    #  评估
    # ------------------------------------------------------------------
    async def _evaluate_split_model(self, round_num: int, learners=None):
        """
        评估完整 split 模型
        
        评估前需要从 Learner 端同步 client_layers 权重，因为训练时
        client_layers 在 Learner 侧更新，Trainer 侧的副本不会自动同步。
        """
        test_datasets = self._datasets.get("test", [])
        if not test_datasets:
            return

        loader = DataLoader(test_datasets[0], batch_size=self.batch_size, shuffle=False)

        # ⭐ 从 Learner 同步 client_layers 权重
        # 多客户端时对每个 learner 的 client_weights 分别评估
        eval_learners = learners or []
        if not eval_learners:
            # 没有可用 learner，使用 trainer 自身的（可能未更新的）权重
            self.split_model.eval()
            acc = self._run_eval_loop(loader)
            self.logger.info(
                f"  [Eval] Round {round_num}: 准确率={acc:.4f} (无 learner 同步)"
            )
            return

        for learner in eval_learners:
            learner_id = getattr(learner, '_target_id', 'unknown')
            try:
                # 从 learner 获取已训练的 client_layers 权重
                client_weights = await learner.get_client_weights()
                
                # 加载到 trainer 的 split_model.client_layers
                state_dict = {}
                for key, value in client_weights.items():
                    # 移除 "client." 前缀（如果有）
                    clean_key = key.replace("client.", "")
                    if torch.is_tensor(value):
                        state_dict[clean_key] = value.to(self.device)
                    else:
                        state_dict[clean_key] = torch.tensor(value).to(self.device)
                
                self.split_model.client_layers.load_state_dict(state_dict)
                self.split_model.eval()
                
                acc = self._run_eval_loop(loader)
                self.logger.info(
                    f"  [Eval] Round {round_num} [{learner_id}]: "
                    f"准确率={acc:.4f}"
                )
            except Exception as e:
                self.logger.error(
                    f"  [Eval] Round {round_num} [{learner_id}] 同步权重失败: {e}"
                )

    def _run_eval_loop(self, loader) -> float:
        """执行评估循环，返回准确率"""
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for data, labels in loader:
                data = data.to(self.device)
                labels = labels.to(self.device)
                output = self.split_model(data)
                _, predicted = output.max(1)
                total_correct += predicted.eq(labels).sum().item()
                total_samples += labels.size(0)

        return total_correct / max(total_samples, 1)