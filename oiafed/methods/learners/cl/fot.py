"""
FOT Learner — 联邦正交训练的客户端学习器 (ICLR 2024)

移植自官方实现:
https://github.com/duygunuryldz/Federated_Orthogonal_Training/blob/main/trainer/cifar_trainer.py

核心要点:
- 客户端做标准 SGD 训练 (无额外计算开销!)
- 使用多头模型, 每个任务用自己的分类头
- 在任务结束时收集各层激活 (GPSE 所需)
- 正交投影在 SERVER 端的 FedProject 聚合器中完成
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from typing import Dict, Any, Optional, List

from ....core.learner import Learner
from ....core.types import EpochMetrics, EvalResult, StepMetrics
from ....registry import learner


@learner('cl.fot', description='FOT: Federated Orthogonal Training (ICLR 2024)')
class FOTLearner(Learner):
    """
    FOT 客户端学习器

    关键特性:
    1. 标准 SGD 训练 — 客户端无额外计算
    2. 多头模型 — output[task_id] 选择当前任务的分类头
    3. collect_activations() — 任务结束时收集激活供 GPSE 使用
    """

    def __init__(self, model, datasets=None, tracker=None,
                 callbacks=None, config=None, node_id=None):
        super().__init__(model, None, tracker, callbacks, config, node_id)

        self._datasets = datasets or {}

        # 训练参数
        self.learning_rate = self._config.get('learning_rate', 0.01)
        self.batch_size = self._config.get('batch_size', 32)
        self.local_epochs = self._config.get('local_epochs', 5)
        self.momentum = self._config.get('momentum', 0.9)
        self.weight_decay = self._config.get('weight_decay', 1e-5)

        # CL 参数
        self.num_tasks = self._config.get('num_tasks', 5)
        self.classes_per_task = self._config.get('classes_per_task', 2)

        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 组件 (setup 中初始化)
        self.torch_model: Optional[nn.Module] = None
        self._optimizer: Optional[optim.Optimizer] = None
        self._criterion: Optional[nn.Module] = None
        self._train_loader: Optional[DataLoader] = None
        self._test_loader: Optional[DataLoader] = None

        # 任务状态
        self.current_task_id = 0

        self.logger.info(
            f"FOTLearner {node_id}: "
            f"num_tasks={self.num_tasks}, classes_per_task={self.classes_per_task}"
        )

    async def setup(self, config: Dict) -> None:
        """初始化训练环境"""
        if 'task_id' in config:
            self.current_task_id = config['task_id']
            self._config['task_id'] = config['task_id']

        if hasattr(self._model, 'get_model'):
            self.torch_model = self._model.get_model()
        else:
            self.torch_model = self._model
        self.torch_model.to(self.device)

        self._optimizer = optim.SGD(
            self.torch_model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        self._criterion = nn.CrossEntropyLoss()

        train_datasets = self._datasets.get("train", [])
        if train_datasets:
            self._train_loader = DataLoader(
                train_datasets[0], batch_size=self.batch_size, shuffle=True
            )
        test_datasets = self._datasets.get("test", [])
        if test_datasets:
            self._test_loader = DataLoader(
                test_datasets[0], batch_size=self.batch_size, shuffle=False
            )

    # ------------------------------------------------------------------
    #  训练 — 标准 SGD, 与官方 normal_train() 一致
    # ------------------------------------------------------------------
    async def train_epoch(self, epoch_idx: int) -> EpochMetrics:
        task_id = self._config.get('task_id', self.current_task_id)
        self.current_task_id = task_id

        task_loader = self._get_task_data_loader(task_id)
        self.torch_model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for data, target in task_loader:
            data, target = data.to(self.device), target.to(self.device)
            self._optimizer.zero_grad()

            output = self.torch_model(data)

            # 多头: output[task_id]; 单头: output
            if isinstance(output, list):
                logits = output[task_id]
                target_local = target - task_id * self.classes_per_task
                target_local = target_local.clamp(0, self.classes_per_task - 1)
            else:
                logits = output
                target_local = target

            loss = self._criterion(logits, target_local)
            loss.backward()
            self._optimizer.step()

            total_loss += loss.item() * data.size(0)
            _, predicted = logits.max(1)
            total_correct += predicted.eq(target_local).sum().item()
            total_samples += target.size(0)

        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        avg_acc = total_correct / total_samples if total_samples > 0 else 0

        self.logger.info(
            f"[{self._node_id}] Task {task_id} Epoch {epoch_idx}: "
            f"Loss={avg_loss:.4f}, Acc={avg_acc:.4f}"
        )

        return EpochMetrics(
            epoch=epoch_idx, avg_loss=avg_loss, total_samples=total_samples,
            metrics={'accuracy': avg_acc, 'task_id': task_id}
        )

    # ------------------------------------------------------------------
    #  GPSE 激活收集 (任务结束时由 trainer 调用)
    # ------------------------------------------------------------------
    async def collect_activations(self, config: Optional[Dict] = None) -> Dict:
        """
        收集各层激活 — GPSE 所需

        移植自官方 collect_activations():
        1. 前向传播整个训练集, 捕获各层输入激活
        2. Conv 层做 Unfold; FC 层直接转置
        3. 已有 orth_set 时投影到正交补空间
        4. 乘随机高斯矩阵 (隐私保护 + 降维)

        Returns:
            {layer_name: (random_proj_matrix, ratio, num_samples)}
        """
        config = config or {}
        orth_set = config.get('orth_set', {})
        task_id = config.get('task_id', self.current_task_id)

        task_loader = self._get_task_data_loader(task_id)
        self.torch_model.eval()

        if not hasattr(self.torch_model, 'act'):
            self.logger.warning("模型没有 act 属性, 无法收集激活")
            return {}

        model_cls = type(self.torch_model)
        layer_names = getattr(model_cls, 'ORTH_LAYER_NAMES', [
            'conv1.weight', 'conv2.weight', 'conv3.weight',
            'fc1.weight', 'fc2.weight',
        ])

        activation = {key: [] for key in layer_names}

        with torch.no_grad():
            for data, _ in task_loader:
                data = data.to(self.device)
                _ = self.torch_model(data)
                for key in layer_names:
                    if key in self.torch_model.act:
                        activation[key].append(self.torch_model.act[key].detach().cpu())

        for name in activation:
            if activation[name]:
                activation[name] = torch.cat(activation[name], dim=0)
            else:
                activation[name] = None

        bsz = len(task_loader.dataset) if task_loader else 0
        result = {}
        ksize_list = getattr(self.torch_model, 'ksize', [])
        num_conv = len(ksize_list)

        for i, key in enumerate(layer_names):
            if activation[key] is None:
                continue

            if i < num_conv:
                # Conv 层: unfold → 矩阵
                ksz = ksize_list[i]
                act = activation[key]
                unfolder = nn.Unfold(ksz, dilation=1, padding=0, stride=1)
                mat = unfolder(act.to(self.device))
                mat = mat.permute(0, 2, 1).reshape(-1, mat.shape[1]).T.to(self.device)

                ratio = 1.0
                if orth_set.get(key) is not None:
                    U = orth_set[key].to(self.device)
                    projected = U @ (U.T @ mat)
                    remaining = mat - projected
                    ratio = (torch.norm(remaining) / torch.norm(mat)).cpu().item()
                    mat = remaining

                rand_mat = torch.normal(0, 1, size=(mat.shape[1], mat.shape[0])).to(self.device)
                result[key] = ((mat @ rand_mat).cpu(), ratio, bsz)
            else:
                # FC 层
                mat = activation[key].T.to(self.device)
                ratio = 1.0
                if orth_set.get(key) is not None:
                    U = orth_set[key].to(self.device)
                    projected = U @ (U.T @ mat)
                    remaining = mat - projected
                    ratio = (torch.norm(remaining) / torch.norm(mat)).cpu().item()
                    mat = remaining

                rand_mat = torch.normal(0, 1, size=(mat.shape[1], mat.shape[0] * 5)).to(self.device)
                result[key] = ((mat @ rand_mat).cpu(), ratio, bsz)

        self.logger.info(f"[{self._node_id}] GPSE 激活收集: {len(result)} 层, {bsz} 样本")
        return result

    # ------------------------------------------------------------------
    #  评估 — 支持按 task_id 评估
    # ------------------------------------------------------------------
    async def evaluate(self, config: Optional[Dict] = None) -> EvalResult:
        config = config or {}
        task_id = config.get("task_id", None)

        if task_id is not None:
            test_loader = self._get_task_data_loader(task_id, split="test")
        else:
            test_loader = self._test_loader or self._train_loader

        if test_loader is None:
            return EvalResult(num_samples=0, metrics={})

        self.torch_model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.torch_model(data)

                if task_id is not None and isinstance(output, list):
                    logits = output[task_id]
                    target_local = target - task_id * self.classes_per_task
                    target_local = target_local.clamp(0, self.classes_per_task - 1)
                else:
                    logits = output[0] if isinstance(output, list) else output
                    target_local = target

                loss = self._criterion(logits, target_local)
                total_loss += loss.item() * data.size(0)
                _, predicted = logits.max(1)
                total_correct += predicted.eq(target_local).sum().item()
                total_samples += target.size(0)

        return EvalResult(num_samples=total_samples, metrics={
            'accuracy': total_correct / total_samples if total_samples > 0 else 0,
            'loss': total_loss / total_samples if total_samples > 0 else 0,
        })

    # ------------------------------------------------------------------
    #  辅助方法
    # ------------------------------------------------------------------
    def _get_task_data_loader(self, task_id: int, split: str = "train") -> DataLoader:
        datasets = self._datasets.get(split, [])
        if not datasets:
            datasets = self._datasets.get("train", [])
        if not datasets:
            return self._train_loader

        dataset = datasets[0]
        start_class = task_id * self.classes_per_task
        end_class = start_class + self.classes_per_task
        task_classes = set(range(start_class, end_class))

        indices = []
        for idx in range(len(dataset)):
            _, label = dataset[idx]
            if isinstance(label, torch.Tensor):
                label = label.item()
            if label in task_classes:
                indices.append(idx)

        if not indices:
            return self._train_loader

        return DataLoader(Subset(dataset, indices), batch_size=self.batch_size,
                          shuffle=(split == "train"))

    async def train_step(self, batch: Any, batch_idx: int) -> StepMetrics:
        data, target = batch
        data, target = data.to(self.device), target.to(self.device)
        self._optimizer.zero_grad()
        output = self.torch_model(data)
        if isinstance(output, list):
            logits = output[self.current_task_id]
            target_local = (target - self.current_task_id * self.classes_per_task).clamp(0, self.classes_per_task - 1)
        else:
            logits, target_local = output, target
        loss = self._criterion(logits, target_local)
        loss.backward()
        self._optimizer.step()
        _, predicted = logits.max(1)
        return StepMetrics(loss=loss.item(), batch_size=data.size(0),
                           metrics={'accuracy': predicted.eq(target_local).sum().item() / target.size(0)})

    def get_weights(self) -> Dict[str, Any]:
        return {name: param.data.clone() for name, param in self.torch_model.state_dict().items()}

    def set_weights(self, weights: Dict[str, Any]) -> None:
        model = self.torch_model or self._model
        if model is None:
            return
        torch_weights = {k: (v if torch.is_tensor(v) else torch.from_numpy(v)) for k, v in weights.items()}
        model.load_state_dict(torch_weights)

    def get_dataloader(self) -> DataLoader:
        return self._train_loader

    def get_num_samples(self) -> int:
        train_datasets = self._datasets.get("train", [])
        return len(train_datasets[0]) if train_datasets else 0

    async def teardown(self) -> None:
        pass
