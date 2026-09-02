"""
基线联邦学习方法实现模块。
包含 FedAvg、FedProx、FedSym、FedProto、FedProc、FedNTD、FedSOL、FedLESAM、pFedHB 的对比实验实现。
"""

import copy
import random
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .baseline import BaselineModel, create_baseline_model
from .helpers import AverageMeter, create_optimizer, average_models
from .metrics import Metrics


# ============================================================================
# 基线客户端
# ============================================================================

class BaselineClient:
    """
    基线联邦学习客户端。
    用于 FedAvg、FedProx 和 FedSym。
    """

    def __init__(
            self,
            client_id: int,
            dataloader: DataLoader,
            device: torch.device,
            num_classes: int,
            lr: float = 0.01,
            momentum: float = 0.9,
            weight_decay: float = 1e-4,
            optimizer_name: str = "sgd",
            mu: float = 0.0  # FedProx 的近端项系数
    ):
        """
        初始化客户端。

        参数:
            client_id: 客户端 ID
            dataloader: 本地数据 DataLoader
            device: 训练设备
            num_classes: 类别数
            lr: 学习率
            momentum: SGD 动量
            weight_decay: 权重衰减
            optimizer_name: 优化器类型
            mu: FedProx 近端项系数（0 表示 FedAvg）
        """
        self.client_id = client_id
        self.dataloader = dataloader
        self.device = device
        self.num_classes = num_classes

        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer_name
        self.mu = mu

        self.model: Optional[BaselineModel] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.global_model: Optional[BaselineModel] = None  # 用于 FedProx

        self.num_samples = len(dataloader.dataset)

    def set_model(self, model: BaselineModel, keep_global: bool = False):
        """
        设置客户端模型。

        参数:
            model: 要使用的模型
            keep_global: 是否保存全局模型副本（用于 FedProx）
        """
        self.model = copy.deepcopy(model).to(self.device)

        if keep_global and self.mu > 0:
            self.global_model = copy.deepcopy(model).to(self.device)
            for p in self.global_model.parameters():
                p.requires_grad = False

        self.optimizer = create_optimizer(
            self.model,
            self.optimizer_name,
            self.lr,
            self.momentum,
            self.weight_decay
        )

    def get_model(self) -> BaselineModel:
        """获取当前模型。"""
        return self.model

    def train_one_epoch(self) -> Dict[str, float]:
        """训练一个本地轮次。"""
        self.model.train()

        loss_meter = AverageMeter("loss")
        acc_meter = AverageMeter("accuracy")

        criterion = nn.CrossEntropyLoss()

        for data, target in self.dataloader:
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(data)
            loss = criterion(logits, target)

            # FedProx 近端项
            if self.mu > 0 and self.global_model is not None:
                prox_loss = 0.0
                for p, g_p in zip(self.model.parameters(), self.global_model.parameters()):
                    prox_loss += (p - g_p).pow(2).sum()
                loss = loss + (self.mu / 2) * prox_loss

            loss.backward()
            self.optimizer.step()

            # 更新指标
            batch_size = data.size(0)
            loss_meter.update(loss.item(), batch_size)

            pred = logits.argmax(dim=1)
            correct = pred.eq(target).sum().item()
            acc_meter.update(correct / batch_size, batch_size)

        return {
            "loss": loss_meter.avg,
            "accuracy": acc_meter.avg
        }

    def train(self, num_epochs: int) -> Dict[str, float]:
        """执行本地训练。"""
        if self.model is None:
            raise ValueError("模型未设置。")

        metrics = {}
        for _ in range(num_epochs):
            metrics = self.train_one_epoch()

        return metrics


# ============================================================================
# 基线服务器
# ============================================================================

class BaselineServer(ABC):
    """基线联邦服务器抽象基类。"""

    def __init__(
            self,
            model_config: Dict,
            device: torch.device,
            test_loader: Optional[DataLoader] = None
    ):
        self.device = device
        self.test_loader = test_loader
        self.model_config = model_config

        # 创建全局模型
        self.global_model = create_baseline_model(**model_config, device=device)

    @abstractmethod
    def aggregate(self, client_models: List[BaselineModel], weights: List[float]):
        """聚合客户端模型。"""
        pass

    def sample_clients(self, num_clients: int, total_clients: int) -> List[int]:
        """采样客户端。"""
        return random.sample(range(total_clients), min(num_clients, total_clients))

    def get_global_model(self) -> BaselineModel:
        """获取全局模型。"""
        return self.global_model

    def evaluate(self, model: Optional[BaselineModel] = None) -> Dict[str, float]:
        """在测试集上评估模型。"""
        if self.test_loader is None:
            return {}

        if model is None:
            model = self.global_model

        model.eval()
        model = model.to(self.device)

        correct = 0
        total = 0
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()

        all_preds = []
        all_targets = []
        all_confidences = []

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                logits = model(data)

                loss = criterion(logits, target)
                total_loss += loss.item() * target.size(0)

                probs = F.softmax(logits, dim=1)
                confidence, pred = probs.max(dim=1)

                correct += pred.eq(target).sum().item()
                total += target.size(0)

                all_preds.append(pred.cpu())
                all_targets.append(target.cpu())
                all_confidences.append(confidence.cpu())

        # 计算指标
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        all_confidences = torch.cat(all_confidences)

        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / total if total > 0 else 0.0

        return {
            "accuracy": accuracy,
            "loss": avg_loss,
            "avg_confidence": all_confidences.mean().item()
        }

    def save_checkpoint(self, path: str, round_num: int):
        """保存检查点。"""
        checkpoint = {
            "round": round_num,
            "model_config": self.model_config,
            "global_model_state_dict": self.global_model.state_dict()
        }
        torch.save(checkpoint, path)


class FedAvgServer(BaselineServer):
    """FedAvg 服务器。"""

    def aggregate(self, client_models: List[BaselineModel], weights: List[float]):
        """使用加权平均聚合客户端模型。"""
        self.global_model = average_models(client_models, weights)


class FedProxServer(BaselineServer):
    """FedProx 服务器（聚合方式与 FedAvg 相同）。"""

    def aggregate(self, client_models: List[BaselineModel], weights: List[float]):
        """使用加权平均聚合客户端模型。"""
        self.global_model = average_models(client_models, weights)


class FedSymServer(BaselineServer):
    """
    FedSym 服务器。

    特点:
    - 维护模型池而非单一全局模型
    - 使用共生机制而非聚合
    - 仅在部署阶段聚合全局模型
    """

    def __init__(
            self,
            model_config: Dict,
            pool_size: int,
            device: torch.device,
            test_loader: Optional[DataLoader] = None,
            symbiosis_mode: str = "adaptive"
    ):
        super().__init__(model_config, device, test_loader)

        self.pool_size = pool_size
        self.symbiosis_mode = symbiosis_mode

        # 初始化模型池
        self.model_pool: List[BaselineModel] = []
        for _ in range(pool_size):
            model = create_baseline_model(**model_config, device=device)
            self.model_pool.append(model)

    def distribute_models(self, client_ids: List[int]) -> Dict[int, BaselineModel]:
        """分发模型给客户端。"""
        client_models = {}
        for i, client_id in enumerate(client_ids):
            model_idx = i % self.pool_size
            model = copy.deepcopy(self.model_pool[model_idx])
            client_models[client_id] = model
        return client_models

    def receive_models(self, client_models: Dict[int, BaselineModel]):
        """接收客户端更新后的模型。"""
        for i, (_, model) in enumerate(client_models.items()):
            pool_idx = i % self.pool_size
            self.model_pool[pool_idx] = copy.deepcopy(model)

    def perform_symbiosis(self, current_round: int):
        """
        执行模型共生。

        根据 FedSym 论文 (Algorithm 1):
        对于每个模型 wi，从模型列表中随机选择两个其他不同的模型 wp 和 wq 进行共生。

        注意：wi 本身不参与自己位置的共生，但会参与其他位置的共生。
        这种设计让模型向更平坦的损失景观移动。
        """
        new_pool = []

        for i in range(self.pool_size):
            # 从其他模型中随机选择两个进行共生（不包括位置 i 的模型）
            other_indices = [j for j in range(self.pool_size) if j != i]
            if len(other_indices) < 2:
                # 如果候选模型不足，保留原模型
                new_pool.append(copy.deepcopy(self.model_pool[i]))
                continue

            p_idx, q_idx = random.sample(other_indices, 2)

            model_p = self.model_pool[p_idx]
            model_q = self.model_pool[q_idx]

            # 根据共生模式选择共生方式
            if self.symbiosis_mode == "endo":
                new_model = self._endosymbiosis(model_p, model_q)
            elif self.symbiosis_mode == "ecto":
                new_model = self._ectosymbiosis(model_p, model_q)
            else:  # adaptive
                # 交替使用内共生和外共生
                if current_round % 2 == 0:
                    new_model = self._endosymbiosis(model_p, model_q)
                else:
                    new_model = self._ectosymbiosis(model_p, model_q)

            new_pool.append(new_model)

        self.model_pool = new_pool

    def _endosymbiosis(
            self, model_p: BaselineModel, model_q: BaselineModel
    ) -> BaselineModel:
        """内共生：参数级融合。"""
        return average_models([model_p, model_q])

    def _ectosymbiosis(
            self, model_p: BaselineModel, model_q: BaselineModel
    ) -> BaselineModel:
        """外共生：层级重组。"""
        new_model = copy.deepcopy(model_p)

        # 随机选择从哪个模型复制每个层
        for (name_p, param_p), (name_q, param_q) in zip(
                new_model.named_parameters(), model_q.named_parameters()
        ):
            if random.random() < 0.5:
                param_p.data.copy_(param_q.data)

        return new_model

    def aggregate(self, client_models: List[BaselineModel], weights: List[float]):
        """FedSym 不使用传统聚合，此方法仅用于接口兼容。"""
        pass

    def aggregate_global_model(self):
        """聚合池中模型以创建全局模型（用于评估/部署）。"""
        self.global_model = average_models(self.model_pool)


# ============================================================================
# 基线训练器
# ============================================================================

class BaselineTrainer:
    """基线方法训练器。"""

    # 支持的方法列表
    SUPPORTED_METHODS = [
        "fedavg", "fedprox", "fedsym", "fedproto",
        "fedproc", "fedntd", "fedsol", "fedlesam", "pfedhb",
        "fedmoeda", "fedevi"
    ]

    def __init__(
            self,
            method: str,
            config,
            client_loaders: List[DataLoader],
            test_loader: DataLoader,
            device: torch.device,
            logger
    ):
        """
        初始化训练器。

        参数:
            method: 方法名称
            config: 配置对象
            client_loaders: 客户端 DataLoader 列表
            test_loader: 测试 DataLoader
            device: 训练设备
            logger: 日志器
        """
        self.method = method.lower()
        if self.method not in self.SUPPORTED_METHODS:
            raise ValueError(f"未知的方法: {self.method}。支持的方法: {self.SUPPORTED_METHODS}")

        self.config = config
        self.client_loaders = client_loaders
        self.test_loader = test_loader
        self.device = device
        self.logger = logger

        # 模型配置
        self.model_config = {
            "num_classes": config.num_classes,
            "backbone": config.backbone,
            "input_channels": config.input_channels,
            "input_size": config.input_size,
            "hidden_dim": getattr(config, 'expert_hidden_dim', 256)
        }

        # 初始化服务器
        self._init_server()

        # 初始化客户端
        self._init_clients()

        # 最佳指标
        self.best_accuracy = 0.0
        self.best_round = 0

    def _init_server(self):
        """初始化服务器。"""
        if self.method == "fedavg":
            self.server = FedAvgServer(
                model_config=self.model_config,
                device=self.device,
                test_loader=self.test_loader
            )
        elif self.method == "fedprox":
            self.server = FedProxServer(
                model_config=self.model_config,
                device=self.device,
                test_loader=self.test_loader
            )
        elif self.method == "fedsym":
            self.server = FedSymServer(
                model_config=self.model_config,
                pool_size=getattr(self.config, 'symbiosis_pool_size', 10),
                device=self.device,
                test_loader=self.test_loader,
                symbiosis_mode=getattr(self.config, 'symbiosis_mode', 'adaptive')
            )
        elif self.method == "fedproto":
            self.server = FedProtoServer(
                model_config=self.model_config,
                device=self.device,
                test_loader=self.test_loader,
                num_classes=self.config.num_classes
            )
        elif self.method == "fedproc":
            self.server = FedProcServer(
                model_config=self.model_config,
                device=self.device,
                test_loader=self.test_loader,
                num_classes=self.config.num_classes
            )
        elif self.method == "fedntd":
            self.server = FedNTDServer(
                model_config=self.model_config,
                device=self.device,
                test_loader=self.test_loader
            )
        elif self.method == "fedsol":
            self.server = FedSOLServer(
                model_config=self.model_config,
                device=self.device,
                test_loader=self.test_loader
            )
        elif self.method == "fedlesam":
            self.server = FedLESAMServer(
                model_config=self.model_config,
                device=self.device,
                test_loader=self.test_loader
            )
        elif self.method == "pfedhb":
            self.server = pFedHBServer(
                model_config=self.model_config,
                device=self.device,
                test_loader=self.test_loader,
                prior_var=getattr(self.config, 'pfedhb_prior_var', 1.0)
            )
        elif self.method == "fedmoeda":
            from .moe import create_moe_model
            self.server = FedMoEDAServer(
                model_config=self.model_config,
                device=self.device,
                test_loader=self.test_loader,
                moe_config={
                    "num_classes": self.config.num_classes,
                    "num_experts": self.config.num_experts,
                    "backbone": self.config.backbone,
                    "input_channels": self.config.input_channels,
                    "input_size": self.config.input_size,
                    "expert_hidden_dim": getattr(self.config, 'expert_hidden_dim', 256),
                    "top_k": getattr(self.config, 'fedmoeda_top_k', 2),
                }
            )
        elif self.method == "fedevi":
            self.server = FedEviServer(
                model_config=self.model_config,
                device=self.device,
                test_loader=self.test_loader
            )
        else:
            raise ValueError(f"未知的方法: {self.method}")

    def _init_clients(self):
        """初始化客户端。"""
        self.clients = []

        for client_id, dataloader in enumerate(self.client_loaders):
            if self.method == "fedavg":
                client = BaselineClient(
                    client_id=client_id,
                    dataloader=dataloader,
                    device=self.device,
                    num_classes=self.config.num_classes,
                    lr=self.config.lr,
                    momentum=self.config.momentum,
                    weight_decay=self.config.weight_decay,
                    optimizer_name=self.config.optimizer,
                    mu=0.0
                )
            elif self.method == "fedprox":
                client = BaselineClient(
                    client_id=client_id,
                    dataloader=dataloader,
                    device=self.device,
                    num_classes=self.config.num_classes,
                    lr=self.config.lr,
                    momentum=self.config.momentum,
                    weight_decay=self.config.weight_decay,
                    optimizer_name=self.config.optimizer,
                    mu=getattr(self.config, 'fedprox_mu', 0.1)
                )
            elif self.method == "fedsym":
                client = BaselineClient(
                    client_id=client_id,
                    dataloader=dataloader,
                    device=self.device,
                    num_classes=self.config.num_classes,
                    lr=self.config.lr,
                    momentum=self.config.momentum,
                    weight_decay=self.config.weight_decay,
                    optimizer_name=self.config.optimizer,
                    mu=0.0
                )
            elif self.method == "fedproto":
                client = FedProtoClient(
                    client_id=client_id,
                    dataloader=dataloader,
                    device=self.device,
                    num_classes=self.config.num_classes,
                    lr=self.config.lr,
                    momentum=self.config.momentum,
                    weight_decay=self.config.weight_decay,
                    optimizer_name=self.config.optimizer,
                    proto_lambda=getattr(self.config, 'fedproto_lambda', 1.0)
                )
            elif self.method == "fedproc":
                client = FedProcClient(
                    client_id=client_id,
                    dataloader=dataloader,
                    device=self.device,
                    num_classes=self.config.num_classes,
                    lr=self.config.lr,
                    momentum=self.config.momentum,
                    weight_decay=self.config.weight_decay,
                    optimizer_name=self.config.optimizer,
                    proto_weight=getattr(self.config, 'proto_weight', 0.1),
                    temperature=getattr(self.config, 'proto_temperature', 0.5)
                )
            elif self.method == "fedntd":
                client = FedNTDClient(
                    client_id=client_id,
                    dataloader=dataloader,
                    device=self.device,
                    num_classes=self.config.num_classes,
                    lr=self.config.lr,
                    momentum=self.config.momentum,
                    weight_decay=self.config.weight_decay,
                    optimizer_name=self.config.optimizer,
                    ntd_weight=getattr(self.config, 'ntd_weight', 1.0),
                    temperature=getattr(self.config, 'ntd_temperature', 1.0)
                )
            elif self.method == "fedsol":
                client = FedSOLClient(
                    client_id=client_id,
                    dataloader=dataloader,
                    device=self.device,
                    num_classes=self.config.num_classes,
                    lr=self.config.lr,
                    momentum=self.config.momentum,
                    weight_decay=self.config.weight_decay,
                    optimizer_name=self.config.optimizer,
                    rho=getattr(self.config, 'sol_rho', 0.5)
                )
            elif self.method == "fedlesam":
                client = FedLESAMClient(
                    client_id=client_id,
                    dataloader=dataloader,
                    device=self.device,
                    num_classes=self.config.num_classes,
                    lr=self.config.lr,
                    momentum=self.config.momentum,
                    weight_decay=self.config.weight_decay,
                    optimizer_name=self.config.optimizer,
                    rho=getattr(self.config, 'lesam_rho', 0.5)
                )
            elif self.method == "pfedhb":
                client = pFedHBClient(
                    client_id=client_id,
                    dataloader=dataloader,
                    device=self.device,
                    num_classes=self.config.num_classes,
                    lr=self.config.lr,
                    momentum=self.config.momentum,
                    weight_decay=self.config.weight_decay,
                    optimizer_name=self.config.optimizer,
                    prior_var=getattr(self.config, 'pfedhb_prior_var', 1.0),
                    posterior_var=getattr(self.config, 'pfedhb_posterior_var', 0.1),
                    kl_weight=getattr(self.config, 'pfedhb_kl_weight', 0.01)
                )
            elif self.method == "fedmoeda":
                client = FedMoEDAClient(
                    client_id=client_id,
                    dataloader=dataloader,
                    device=self.device,
                    num_classes=self.config.num_classes,
                    lr=self.config.lr,
                    momentum=self.config.momentum,
                    weight_decay=self.config.weight_decay,
                    optimizer_name=self.config.optimizer,
                    moe_config={
                        "num_classes": self.config.num_classes,
                        "num_experts": self.config.num_experts,
                        "backbone": self.config.backbone,
                        "input_channels": self.config.input_channels,
                        "input_size": self.config.input_size,
                        "expert_hidden_dim": getattr(self.config, "expert_hidden_dim", 256),
                        "top_k": getattr(self.config, "fedmoeda_top_k", 2),
                    }
                )
            elif self.method == "fedevi":
                client = FedEviClient(
                    client_id=client_id,
                    dataloader=dataloader,
                    device=self.device,
                    num_classes=self.config.num_classes,
                    lr=self.config.lr,
                    momentum=self.config.momentum,
                    weight_decay=self.config.weight_decay,
                    optimizer_name=self.config.optimizer
                )

            self.clients.append(client)

    def train_round(self, round_num: int) -> Dict[str, float]:
        """执行一轮训练。"""
        # 采样客户端
        selected_ids = self.server.sample_clients(
            self.config.clients_per_round,
            len(self.clients)
        )

        # 训练客户端
        client_metrics = {}
        updated_models = []
        weights = []

        if self.method == "fedsym":
            # FedSym: 从池中分发模型
            client_models = self.server.distribute_models(selected_ids)

            for client_id in selected_ids:
                client = self.clients[client_id]
                model = client_models[client_id]

                client.set_model(model)
                metrics = client.train(self.config.local_epochs)

                client_metrics[client_id] = metrics
                updated_models.append(client.get_model())
                weights.append(client.num_samples)

            # 接收模型并执行共生
            client_model_dict = {
                cid: self.clients[cid].get_model()
                for cid in selected_ids
            }
            self.server.receive_models(client_model_dict)
            self.server.perform_symbiosis(round_num)

        elif self.method == "fedproto":
            # FedProto (Algorithm 1): 客户端保持独立本地模型，仅通过原型通信。
            # 与 FedAvg 的关键区别：不会每轮将全局模型分发给客户端。
            # 每个客户端只在首次被选中时从全局模型初始化，之后保持并持续训练自己的本地模型。
            # 注: 从聚合模型初始化（而非随机初始化）是为了让不同客户端的特征空间
            # 在初始阶段即具有一致性，使得原型聚合更有意义。
            global_prototypes = self.server.get_global_prototypes()
            prototype_mask = self.server.get_prototype_mask()

            for client_id in selected_ids:
                client = self.clients[client_id]

                # FedProto: 仅在首次被选中时初始化模型（此后保持本地模型）
                if client.model is None:
                    global_model = self.server.get_global_model()
                    client.set_model(global_model)

                # 设置全局原型用于正则化 (Eq. 7-8)
                client.set_global_prototypes(global_prototypes, prototype_mask)
                metrics = client.train(self.config.local_epochs)

                # 训练后计算本地原型，更新到服务器缓存
                proto, mask, counts = client.compute_local_prototypes()
                self.server.update_client_prototypes(client_id, proto, mask, counts)

                client_metrics[client_id] = metrics

            # 聚合所有缓存的原型（而非仅本轮选中的客户端）
            # 这使得全局原型覆盖更多类别，且跨轮更稳定
            self.server.aggregate_cached_prototypes()

            # 聚合所有已初始化的客户端模型用于全局评估
            # FedProto 客户端独立训练导致模型高度发散，
            # 仅平均 10 个客户端会使 eval 结果在不同轮次间剧烈震荡。
            # 使用所有已训练客户端的模型使全局评估模型更稳定。
            all_models = []
            all_weights = []
            for c in self.clients:
                if c.model is not None:
                    all_models.append(c.get_model())
                    all_weights.append(c.num_samples)
            if all_models:
                self.server.aggregate(all_models, all_weights)

        elif self.method == "fedproc":
            # FedProc: 分发模型和全局原型
            global_model = self.server.get_global_model()
            global_prototypes = self.server.get_global_prototypes()

            client_prototypes = []

            for client_id in selected_ids:
                client = self.clients[client_id]

                client.set_model(global_model)
                client.set_global_prototypes(global_prototypes)
                metrics = client.train(self.config.local_epochs)

                # 计算本地原型
                proto, mask = client.compute_local_prototypes()
                client_prototypes.append((proto, mask))

                client_metrics[client_id] = metrics
                updated_models.append(client.get_model())
                weights.append(client.num_samples)

            # 聚合模型和原型
            self.server.aggregate(updated_models, weights)
            self.server.aggregate_prototypes(client_prototypes, weights)

        elif self.method == "fedntd":
            # FedNTD: 分发全局模型作为 Teacher
            global_model = self.server.get_global_model()

            for client_id in selected_ids:
                client = self.clients[client_id]

                client.set_model(global_model)
                client.set_teacher_model(global_model)
                metrics = client.train(self.config.local_epochs)

                client_metrics[client_id] = metrics
                updated_models.append(client.get_model())
                weights.append(client.num_samples)

            # 聚合
            self.server.aggregate(updated_models, weights)

        elif self.method == "fedsol":
            # FedSOL: 分发全局模型
            global_model = self.server.get_global_model()

            for client_id in selected_ids:
                client = self.clients[client_id]

                client.set_model(global_model, keep_global=True)
                metrics = client.train(self.config.local_epochs)

                client_metrics[client_id] = metrics
                updated_models.append(client.get_model())
                weights.append(client.num_samples)

            # 聚合
            self.server.aggregate(updated_models, weights)

        elif self.method == "fedlesam":
            # FedLESAM: 分发当前和上一轮全局模型
            global_model = self.server.get_global_model()
            prev_global = self.server.get_prev_global_model()

            for client_id in selected_ids:
                client = self.clients[client_id]

                client.set_model(global_model)
                client.set_global_direction(global_model, prev_global)
                metrics = client.train(self.config.local_epochs)

                client_metrics[client_id] = metrics
                updated_models.append(client.get_model())
                weights.append(client.num_samples)

            # 聚合（会自动保存历史模型）
            self.server.aggregate(updated_models, weights)

        elif self.method == "pfedhb":
            # pFedHB: 分发全局模型作为先验
            global_model = self.server.get_global_model()
            client_posteriors = []

            for client_id in selected_ids:
                client = self.clients[client_id]

                client.set_model(global_model)
                client.set_global_prior(global_model)
                metrics = client.train(self.config.local_epochs)

                # 收集后验参数
                posterior = client.get_posterior_params()
                client_posteriors.append(posterior)

                client_metrics[client_id] = metrics
                updated_models.append(client.get_model())
                weights.append(client.num_samples)

            # 聚合模型和后验参数
            self.server.aggregate(updated_models, weights)
            self.server.aggregate_posteriors(client_posteriors, weights)

        elif self.method == "fedmoeda":
            # FedMoE-DA: 分发全局 MoE 模型，收集训练后的 MoE 模型 + router 权重
            global_moe = self.server.get_global_moe()

            client_moe_models = []
            router_weights_list = []

            for client_id in selected_ids:
                client = self.clients[client_id]
                client.set_moe_model(global_moe)
                metrics = client.train(self.config.local_epochs)
                client_metrics[client_id] = metrics

                moe_model = client.get_moe_model()
                client_moe_models.append(moe_model)
                router_weights_list.append(client.get_router_weights())
                weights.append(client.num_samples)

            # Domain-aware 聚合
            self.server.aggregate_domain_aware(
                client_moe_models, weights, router_weights_list
            )

        elif self.method == "fedevi":
            # FedEvi: 标准训练 + evidential uncertainty 调整聚合权重
            global_model = self.server.get_global_model()
            uncertainty_list = []

            for client_id in selected_ids:
                client = self.clients[client_id]
                client.set_model(global_model)
                metrics = client.train(self.config.local_epochs)

                # 训练后计算 uncertainty
                u_stats = client.compute_uncertainty()
                uncertainty_list.append(u_stats)

                client_metrics[client_id] = metrics
                updated_models.append(client.get_model())
                weights.append(client.num_samples)

            # 使用 uncertainty 加权聚合
            self.server.aggregate_with_uncertainty(
                updated_models, weights, uncertainty_list
            )

        else:
            # FedAvg/FedProx: 标准流程
            global_model = self.server.get_global_model()

            for client_id in selected_ids:
                client = self.clients[client_id]

                client.set_model(global_model, keep_global=(self.method == "fedprox"))
                metrics = client.train(self.config.local_epochs)

                client_metrics[client_id] = metrics
                updated_models.append(client.get_model())
                weights.append(client.num_samples)

            # 聚合
            self.server.aggregate(updated_models, weights)

        # 聚合客户端指标
        avg_metrics = {}
        if client_metrics:
            keys = list(client_metrics.values())[0].keys()
            for key in keys:
                values = [m[key] for m in client_metrics.values()]
                avg_metrics[key] = sum(values) / len(values)

        return avg_metrics

    def evaluate(self) -> Dict[str, float]:
        """评估模型。"""
        if self.method == "fedsym":
            self.server.aggregate_global_model()
        return self.server.evaluate()

    def train(self) -> Dict:
        """主训练循环。"""
        self.logger.info("=" * 60)
        self.logger.info(f"开始 {self.method.upper()} 训练")
        self.logger.info("=" * 60)

        for round_num in tqdm(range(1, self.config.num_rounds + 1), desc=f"{self.method}"):
            # 训练
            train_metrics = self.train_round(round_num)

            # 日志和评估
            if round_num % self.config.log_interval == 0:
                self.logger.log_metrics(train_metrics, round_num, prefix="train")

                eval_metrics = self.evaluate()
                self.logger.log_metrics(eval_metrics, round_num, prefix="eval")

                if eval_metrics.get("accuracy", 0) > self.best_accuracy:
                    self.best_accuracy = eval_metrics["accuracy"]
                    self.best_round = round_num
                    self.logger.info(f"新的最佳准确率: {self.best_accuracy:.4f}")

        # 最终结果
        self.logger.info("=" * 60)
        self.logger.info(f"{self.method.upper()} 训练完成!")
        self.logger.info(f"最佳准确率: {self.best_accuracy:.4f} (第 {self.best_round} 轮)")
        self.logger.info("=" * 60)

        final_metrics = self.evaluate()

        return {
            "method": self.method,
            "best_accuracy": self.best_accuracy,
            "best_round": self.best_round,
            "final_metrics": final_metrics
        }


# ============================================================================
# FedProto: 联邦原型学习
# ============================================================================

class FedProtoClient(BaselineClient):
    """
    FedProto 客户端。

    使用原型聚合进行联邦学习，通过MSE距离正则化使本地原型趋近全局原型。
    论文: Tan et al., "FedProto: Federated Prototype Learning across Heterogeneous Clients", AAAI 2022

    核心思想（与 FedAvg 的关键区别）：
    - 每个客户端保持独立的本地模型，不接收全局模型参数（Algorithm 1）
    - 知识传递仅通过类原型进行，通信效率极高
    - 训练损失: L = L_S(F(ω;x), y) + λ * Σ_j MSE(C_i^(j), C̄^(j))  (Eq. 7-8)
    - 距离度量使用 MSE (均方误差)，与官方实现一致，使 λ 与特征维度解耦
    """

    def __init__(
            self,
            client_id: int,
            dataloader: DataLoader,
            device: torch.device,
            num_classes: int,
            lr: float = 0.01,
            momentum: float = 0.9,
            weight_decay: float = 1e-4,
            optimizer_name: str = "sgd",
            proto_lambda: float = 1.0
    ):
        """
        初始化 FedProto 客户端。

        参数:
            client_id: 客户端 ID
            dataloader: 本地数据 DataLoader
            device: 训练设备
            num_classes: 类别数
            lr: 学习率
            momentum: SGD 动量
            weight_decay: 权重衰减
            optimizer_name: 优化器类型
            proto_lambda: 原型正则化损失权重 λ
        """
        super().__init__(
            client_id, dataloader, device, num_classes,
            lr, momentum, weight_decay, optimizer_name, mu=0.0
        )
        self.proto_lambda = proto_lambda
        self.global_prototypes: Optional[torch.Tensor] = None  # [num_classes, feature_dim]
        self.prototype_mask: Optional[torch.Tensor] = None  # [num_classes] 标记哪些类有全局原型

    def set_global_prototypes(self, prototypes: Optional[torch.Tensor], mask: Optional[torch.Tensor] = None):
        """
        设置全局原型。

        使用 clone().detach() 确保全局原型不参与客户端的计算图。

        参数:
            prototypes: 全局原型 [num_classes, feature_dim]
            mask: 标记哪些类有有效的全局原型 [num_classes]
        """
        if prototypes is not None:
            self.global_prototypes = prototypes.clone().detach().to(self.device)
        else:
            self.global_prototypes = None

        if mask is not None:
            self.prototype_mask = mask.clone().detach().to(self.device)
        else:
            self.prototype_mask = None

    def compute_local_prototypes(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算本地类原型 (FedProto Eq. 3)。

        C_i^(j) = (1/|D_{i,j}|) * Σ_{(x,y)∈D_{i,j}} f_i(φ_i; x)

        原型定义为同一类别下所有样本特征向量的均值。
        在 eval 模式下计算以获得稳定的特征表示（BatchNorm 使用 running stats）。
        同时返回每个类的样本数，用于服务器端加权聚合 (Eq. 6)。

        返回:
            prototypes: 本地原型 [num_classes, feature_dim]
            mask: 有效类别掩码 [num_classes]
            class_counts: 每个类的样本数 [num_classes]（用于 Eq. 6 加权）
        """
        self.model.eval()

        class_features = defaultdict(list)

        with torch.no_grad():
            for data, target in self.dataloader:
                data, target = data.to(self.device), target.to(self.device)
                features = self.model.get_features(data)

                for i, label in enumerate(target):
                    class_features[label.item()].append(features[i])

        # 推断特征维度（安全处理空数据集的边界情况）
        if class_features:
            feature_dim = next(iter(class_features.values()))[0].size(-1)
        else:
            # 边界情况：空数据集 → 通过 dummy 前向推断维度
            feature_dim = 1  # 回退值，实际不应触发

        # 计算每个类的平均特征作为原型
        prototypes = torch.zeros(self.num_classes, feature_dim, device=self.device)
        prototype_mask = torch.zeros(self.num_classes, device=self.device)
        class_counts = torch.zeros(self.num_classes, device=self.device)

        for cls, feats in class_features.items():
            if feats:
                prototypes[cls] = torch.stack(feats).mean(dim=0)
                prototype_mask[cls] = 1.0
                class_counts[cls] = float(len(feats))

        self.model.train()
        return prototypes, prototype_mask, class_counts

    def train_one_epoch(self) -> Dict[str, float]:
        """
        训练一个本地轮次 (FedProto Eq. 7)。

        L = L_S(F(ω;x), y) + λ * L_R(C̄, C)

        其中 L_R = Σ_j d(C_i^(j), C̄^(j)) (Eq. 8)

        实现说明:
        - 使用 MSE (均方误差) 作为距离度量 d(·,·)，与官方实现一致。
          MSE = (1/feature_dim) * ||C_i^(j) - C̄^(j)||²₂
          这使得损失值与特征维度无关，lambda 可直接使用论文推荐值。
        - 对 batch 内有效类别的 MSE 求和（不取平均），近似 Eq. 8。
        - batch 级原型是全数据集原型的无偏随机近似，适用于 SGD 优化。
        """
        self.model.train()

        loss_meter = AverageMeter("loss")
        acc_meter = AverageMeter("accuracy")

        criterion = nn.CrossEntropyLoss()

        for data, target in self.dataloader:
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()

            # 获取特征和 logits
            features = self.model.get_features(data)
            logits = self.model.classifier(features)

            # 分类损失 L_S
            task_loss = criterion(logits, target)

            # 原型正则化损失 L_R (FedProto Eq. 8)
            # L_R = Σ_j MSE(C_i^(j), C̄^(j))
            proto_loss = torch.tensor(0.0, device=self.device)
            if self.global_prototypes is not None and self.proto_lambda > 0:
                # 计算当前 batch 中每个类的本地原型
                unique_labels = target.unique()
                num_valid_classes = 0
                for label in unique_labels:
                    label_val = label.item()
                    # 检查索引越界 + 该类是否有有效的全局原型
                    if (self.prototype_mask is not None and
                            label_val < self.prototype_mask.size(0) and
                            self.prototype_mask[label_val] > 0):
                        # 当前 batch 中该类样本的特征均值作为本地原型
                        label_mask = (target == label)
                        local_proto = features[label_mask].mean(dim=0)
                        global_proto = self.global_prototypes[label_val]
                        # MSE: (1/d) * ||C_i^(j) - C̄^(j)||²₂
                        # 使用 MSE 而非 sum-of-squares 使损失与特征维度解耦
                        proto_loss = proto_loss + F.mse_loss(
                            local_proto, global_proto
                        )
                        num_valid_classes += 1

            loss = task_loss + self.proto_lambda * proto_loss

            loss.backward()
            self.optimizer.step()

            # 更新指标
            batch_size = data.size(0)
            loss_meter.update(loss.item(), batch_size)

            pred = logits.argmax(dim=1)
            correct = pred.eq(target).sum().item()
            acc_meter.update(correct / batch_size, batch_size)

        return {
            "loss": loss_meter.avg,
            "accuracy": acc_meter.avg
        }

    def train(self, num_epochs: int) -> Dict[str, float]:
        """
        执行本地训练。

        与官方 FedProto 实现一致：每轮重新创建优化器。
        原因：FedProto 客户端保持独立的本地模型，跨轮持续训练。
        如果不重置优化器，SGD 的 momentum buffer 会跨轮累积，
        导致后期梯度方向被过时的历史梯度主导，训练极不稳定。
        """
        if self.model is None:
            raise ValueError("模型未设置。")

        # 每轮重新创建优化器（匹配官方实现）
        self.optimizer = create_optimizer(
            self.model,
            self.optimizer_name,
            self.lr,
            self.momentum,
            self.weight_decay
        )

        metrics = {}
        for _ in range(num_epochs):
            metrics = self.train_one_epoch()

        return metrics


class FedProtoServer(BaselineServer):
    """
    FedProto 服务器。

    特点:
    - 聚合全局类原型，使用每类样本数加权平均 (Eq. 6)
    - 原始 FedProto 不聚合模型参数，仅交换原型
    - 为了统一评估，额外聚合客户端模型（不分发回客户端）

    注意: 由于 FedProto 允许异构模型，客户端各自独立训练，聚合后的全局模型
    可能不如个性化模型表现好。此处聚合仅用于与 FedAvg 等方法公平对比。

    论文: Tan et al., "FedProto: Federated Prototype Learning across Heterogeneous Clients", AAAI 2022
    """

    def __init__(
            self,
            model_config: Dict,
            device: torch.device,
            test_loader: Optional[DataLoader] = None,
            num_classes: int = 10
    ):
        super().__init__(model_config, device, test_loader)
        self.num_classes = num_classes
        self.global_prototypes: Optional[torch.Tensor] = None
        self.prototype_mask: Optional[torch.Tensor] = None
        self.feature_dim: Optional[int] = None

        # 原型缓存：保存所有曾参与训练的客户端的最新原型
        # 这样全局原型聚合不限于当前轮选中的客户端，
        # 使得全局原型更稳定、类别覆盖更完整
        self.client_proto_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def update_client_prototypes(
            self,
            client_id: int,
            proto: torch.Tensor,
            mask: torch.Tensor,
            counts: torch.Tensor
    ):
        """
        更新某个客户端的原型缓存。

        参数:
            client_id: 客户端 ID
            proto: 该客户端的本地原型 [num_classes, feature_dim]
            mask: 有效类别掩码 [num_classes]
            counts: 每个类的样本数 [num_classes]
        """
        self.client_proto_cache[client_id] = (
            proto.clone().detach().to(self.device),
            mask.clone().detach().to(self.device),
            counts.clone().detach().to(self.device)
        )

    def aggregate_cached_prototypes(self):
        """
        从所有缓存的客户端原型中聚合全局原型。

        与仅聚合当前轮选中客户端相比，此方法的优势:
        1. 全局原型覆盖更多类别（减少类别缺失）
        2. 全局原型跨轮更稳定（减少震荡）
        3. 正则化目标更可靠
        """
        if not self.client_proto_cache:
            return
        all_protos = list(self.client_proto_cache.values())
        self.aggregate_prototypes(all_protos)

    def aggregate_prototypes(
            self,
            client_prototypes: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ):
        """
        聚合客户端原型 (FedProto Eq. 6)。

        C̄^(j) = Σ_{i∈N_j} (|D_{i,j}| / Σ_k |D_{k,j}|) * C_i^(j)

        使用每个类的样本数 |D_{i,j}| 进行加权平均，而非客户端总样本数。

        参数:
            client_prototypes: [(prototypes, mask, class_counts), ...]
                prototypes: [num_classes, feature_dim]
                mask: [num_classes]
                class_counts: [num_classes] 每个类的样本数
        """
        if not client_prototypes:
            return

        proto, mask, counts = client_prototypes[0]
        self.feature_dim = proto.size(-1)

        # 使用每类样本数加权聚合 (Eq. 6)
        weighted_prototypes = torch.zeros(
            self.num_classes, self.feature_dim, device=self.device
        )
        total_class_counts = torch.zeros(self.num_classes, device=self.device)

        for (proto, mask, counts) in client_prototypes:
            proto = proto.to(self.device)
            counts = counts.to(self.device)

            # 权重为每类样本数 |D_{i,j}|
            weighted_prototypes += proto * counts.unsqueeze(-1)
            total_class_counts += counts

        # 归一化: 除以每类总样本数
        valid_classes = total_class_counts > 0
        weighted_prototypes[valid_classes] /= total_class_counts[valid_classes].unsqueeze(-1)

        self.global_prototypes = weighted_prototypes
        self.prototype_mask = valid_classes.float()

    def get_global_prototypes(self) -> Optional[torch.Tensor]:
        """获取全局原型。"""
        return self.global_prototypes

    def get_prototype_mask(self) -> Optional[torch.Tensor]:
        """获取全局原型掩码。"""
        return self.prototype_mask

    def aggregate(self, client_models: List[BaselineModel], weights: List[float]):
        """
        聚合客户端模型（仅用于全局评估）。

        注意：原始 FedProto 不聚合模型参数，知识传递仅通过原型进行。
        此处聚合仅用于生成统一的评估模型，聚合后的模型不会分发回客户端。
        """
        self.global_model = average_models(client_models, weights)


# ============================================================================
# FedProc: 原型对比学习
# ============================================================================

class FedProcClient(BaselineClient):
    """
    FedProc 客户端。

    使用原型对比损失进行训练，利用全局类原型校正本地训练。
    论文: Mu et al., "FedProc: Prototypical Contrastive FL on Non-IID Data", FGCS 2023
    """

    def __init__(
            self,
            client_id: int,
            dataloader: DataLoader,
            device: torch.device,
            num_classes: int,
            lr: float = 0.01,
            momentum: float = 0.9,
            weight_decay: float = 1e-4,
            optimizer_name: str = "sgd",
            proto_weight: float = 0.1,
            temperature: float = 0.5
    ):
        super().__init__(
            client_id, dataloader, device, num_classes,
            lr, momentum, weight_decay, optimizer_name, mu=0.0
        )
        self.proto_weight = proto_weight
        self.temperature = temperature
        self.global_prototypes: Optional[torch.Tensor] = None  # [num_classes, feature_dim]
        self.local_prototypes: Optional[torch.Tensor] = None

    def set_global_prototypes(self, prototypes: torch.Tensor):
        """设置全局原型。"""
        if prototypes is not None:
            self.global_prototypes = prototypes.to(self.device)
        else:
            self.global_prototypes = None

    def compute_local_prototypes(self) -> torch.Tensor:
        """计算本地类原型。"""
        self.model.eval()

        # 收集每个类的特征
        class_features = defaultdict(list)

        with torch.no_grad():
            for data, target in self.dataloader:
                data, target = data.to(self.device), target.to(self.device)
                features = self.model.get_features(data)  # 需要模型支持

                for i, label in enumerate(target):
                    class_features[label.item()].append(features[i])

        # 计算每个类的平均特征作为原型
        feature_dim = features.size(-1)
        prototypes = torch.zeros(self.num_classes, feature_dim, device=self.device)
        prototype_mask = torch.zeros(self.num_classes, device=self.device)

        for cls, feats in class_features.items():
            if feats:
                prototypes[cls] = torch.stack(feats).mean(dim=0)
                prototype_mask[cls] = 1.0

        self.local_prototypes = prototypes
        return prototypes, prototype_mask

    def train_one_epoch(self) -> Dict[str, float]:
        """训练一个本地轮次（带原型对比损失）。"""
        self.model.train()

        loss_meter = AverageMeter("loss")
        acc_meter = AverageMeter("accuracy")

        criterion = nn.CrossEntropyLoss()

        for data, target in self.dataloader:
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()

            # 获取特征和 logits
            features = self.model.get_features(data)
            logits = self.model.classifier(features)

            # 任务损失
            task_loss = criterion(logits, target)

            # 原型对比损失
            proto_loss = torch.tensor(0.0, device=self.device)
            if self.global_prototypes is not None and self.proto_weight > 0:
                # 计算样本与所有类原型的相似度
                # features: [B, D], global_prototypes: [C, D]
                similarities = F.cosine_similarity(
                    features.unsqueeze(1),  # [B, 1, D]
                    self.global_prototypes.unsqueeze(0),  # [1, C, D]
                    dim=-1
                ) / self.temperature  # [B, C]

                # 对比损失: -log(exp(sim(z, P^y)) / sum_c(exp(sim(z, P^c))))
                proto_loss = F.cross_entropy(similarities, target)

            loss = task_loss + self.proto_weight * proto_loss

            loss.backward()
            self.optimizer.step()

            # 更新指标
            batch_size = data.size(0)
            loss_meter.update(loss.item(), batch_size)

            pred = logits.argmax(dim=1)
            correct = pred.eq(target).sum().item()
            acc_meter.update(correct / batch_size, batch_size)

        return {
            "loss": loss_meter.avg,
            "accuracy": acc_meter.avg
        }


class FedProcServer(BaselineServer):
    """FedProc 服务器，聚合全局类原型。"""

    def __init__(
            self,
            model_config: Dict,
            device: torch.device,
            test_loader: Optional[DataLoader] = None,
            num_classes: int = 10
    ):
        super().__init__(model_config, device, test_loader)
        self.num_classes = num_classes
        self.global_prototypes: Optional[torch.Tensor] = None
        self.feature_dim: Optional[int] = None

    def aggregate_prototypes(
            self,
            client_prototypes: List[Tuple[torch.Tensor, torch.Tensor]],
            weights: List[float]
    ):
        """
        聚合客户端原型。

        参数:
            client_prototypes: [(prototypes, mask), ...] 每个客户端的原型和掩码
            weights: 客户端权重
        """
        if not client_prototypes:
            return

        # 初始化
        proto, mask = client_prototypes[0]
        self.feature_dim = proto.size(-1)

        # 加权聚合
        weighted_prototypes = torch.zeros(
            self.num_classes, self.feature_dim, device=self.device
        )
        total_weights = torch.zeros(self.num_classes, device=self.device)

        for (proto, mask), weight in zip(client_prototypes, weights):
            proto = proto.to(self.device)
            mask = mask.to(self.device)

            weighted_prototypes += proto * mask.unsqueeze(-1) * weight
            total_weights += mask * weight

        # 归一化
        valid_classes = total_weights > 0
        weighted_prototypes[valid_classes] /= total_weights[valid_classes].unsqueeze(-1)

        self.global_prototypes = weighted_prototypes

    def get_global_prototypes(self) -> Optional[torch.Tensor]:
        """获取全局原型。"""
        return self.global_prototypes

    def aggregate(self, client_models: List[BaselineModel], weights: List[float]):
        """使用加权平均聚合客户端模型。"""
        self.global_model = average_models(client_models, weights)


# ============================================================================
# FedNTD: 非真实标签蒸馏
# ============================================================================

class FedNTDClient(BaselineClient):
    """
    FedNTD 客户端。

    使用非真实标签蒸馏保留全局知识。
    论文: Lee et al., "Preservation of Global Knowledge by Not-True Distillation in FL", NeurIPS 2022
    """

    def __init__(
            self,
            client_id: int,
            dataloader: DataLoader,
            device: torch.device,
            num_classes: int,
            lr: float = 0.01,
            momentum: float = 0.9,
            weight_decay: float = 1e-4,
            optimizer_name: str = "sgd",
            ntd_weight: float = 1.0,
            temperature: float = 1.0
    ):
        super().__init__(
            client_id, dataloader, device, num_classes,
            lr, momentum, weight_decay, optimizer_name, mu=0.0
        )
        self.ntd_weight = ntd_weight
        self.temperature = temperature
        self.teacher_model: Optional[BaselineModel] = None

    def set_teacher_model(self, model: BaselineModel):
        """设置教师模型（全局模型）。"""
        self.teacher_model = copy.deepcopy(model).to(self.device)
        self.teacher_model.eval()
        for p in self.teacher_model.parameters():
            p.requires_grad = False

    def _compute_ntd_loss(
            self,
            student_logits: torch.Tensor,
            teacher_logits: torch.Tensor,
            targets: torch.Tensor
    ) -> torch.Tensor:
        """
        计算非真实标签蒸馏损失。

        只在非真实类别上进行知识蒸馏，避免干扰真实标签的学习。
        """
        batch_size, num_classes = student_logits.size()

        # 创建掩码：排除真实类别
        mask = torch.ones_like(student_logits, dtype=torch.bool)
        mask.scatter_(1, targets.unsqueeze(1), False)

        # 提取非真实类别的 logits
        student_not_true = student_logits[mask].view(batch_size, num_classes - 1)
        teacher_not_true = teacher_logits[mask].view(batch_size, num_classes - 1)

        # 计算 KL 散度
        student_probs = F.log_softmax(student_not_true / self.temperature, dim=1)
        teacher_probs = F.softmax(teacher_not_true / self.temperature, dim=1)

        ntd_loss = F.kl_div(student_probs, teacher_probs, reduction='batchmean')
        ntd_loss = ntd_loss * (self.temperature ** 2)

        return ntd_loss

    def train_one_epoch(self) -> Dict[str, float]:
        """训练一个本地轮次（带 NTD 损失）。"""
        self.model.train()

        loss_meter = AverageMeter("loss")
        acc_meter = AverageMeter("accuracy")

        criterion = nn.CrossEntropyLoss()

        for data, target in self.dataloader:
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()

            # 学生模型前向
            student_logits = self.model(data)

            # 任务损失
            task_loss = criterion(student_logits, target)

            # NTD 损失
            ntd_loss = torch.tensor(0.0, device=self.device)
            if self.teacher_model is not None and self.ntd_weight > 0:
                with torch.no_grad():
                    teacher_logits = self.teacher_model(data)
                ntd_loss = self._compute_ntd_loss(student_logits, teacher_logits, target)

            loss = task_loss + self.ntd_weight * ntd_loss

            loss.backward()
            self.optimizer.step()

            # 更新指标
            batch_size = data.size(0)
            loss_meter.update(loss.item(), batch_size)

            pred = student_logits.argmax(dim=1)
            correct = pred.eq(target).sum().item()
            acc_meter.update(correct / batch_size, batch_size)

        return {
            "loss": loss_meter.avg,
            "accuracy": acc_meter.avg
        }


class FedNTDServer(BaselineServer):
    """FedNTD 服务器。"""

    def aggregate(self, client_models: List[BaselineModel], weights: List[float]):
        """使用加权平均聚合客户端模型。"""
        self.global_model = average_models(client_models, weights)


# ============================================================================
# FedSOL: 稳定正交学习
# ============================================================================

class FedSOLClient(BaselineClient):
    """
    FedSOL 客户端。

    在近端扰动点计算梯度，使本地更新与近端目标正交。
    论文: Lee et al., "FedSOL: Stabilized Orthogonal Learning with Proximal Restrictions in FL", CVPR 2024
    """

    def __init__(
            self,
            client_id: int,
            dataloader: DataLoader,
            device: torch.device,
            num_classes: int,
            lr: float = 0.01,
            momentum: float = 0.9,
            weight_decay: float = 1e-4,
            optimizer_name: str = "sgd",
            rho: float = 0.5
    ):
        super().__init__(
            client_id, dataloader, device, num_classes,
            lr, momentum, weight_decay, optimizer_name, mu=0.0
        )
        self.rho = rho

    def set_model(self, model: BaselineModel, keep_global: bool = True):
        """设置客户端模型。"""
        self.model = copy.deepcopy(model).to(self.device)

        # 保存全局模型用于计算扰动方向
        self.global_model = copy.deepcopy(model).to(self.device)
        for p in self.global_model.parameters():
            p.requires_grad = False

        self.optimizer = create_optimizer(
            self.model,
            self.optimizer_name,
            self.lr,
            self.momentum,
            self.weight_decay
        )

    def _compute_perturbation_direction(self) -> List[torch.Tensor]:
        """计算近端扰动方向: d = (w - w_global) / ||w - w_global||"""
        direction = []
        norm_sq = 0.0

        for p, g_p in zip(self.model.parameters(), self.global_model.parameters()):
            diff = p.data - g_p.data
            direction.append(diff)
            norm_sq += diff.pow(2).sum().item()

        norm = max(norm_sq ** 0.5, 1e-8)

        # 归一化
        for d in direction:
            d.div_(norm)

        return direction

    def train_one_epoch(self) -> Dict[str, float]:
        """训练一个本地轮次（带近端扰动）。"""
        self.model.train()

        loss_meter = AverageMeter("loss")
        acc_meter = AverageMeter("accuracy")

        criterion = nn.CrossEntropyLoss()

        for data, target in self.dataloader:
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()

            # 计算扰动方向
            direction = self._compute_perturbation_direction()

            # 扰动模型: w' = w + ρ * d
            for p, d in zip(self.model.parameters(), direction):
                p.data.add_(d, alpha=self.rho)

            # 在扰动点计算梯度
            logits = self.model(data)
            loss = criterion(logits, target)
            loss.backward()

            # 恢复模型: w = w - ρ * d
            for p, d in zip(self.model.parameters(), direction):
                p.data.sub_(d, alpha=self.rho)

            # 使用扰动点的梯度更新
            self.optimizer.step()

            # 更新指标
            batch_size = data.size(0)
            loss_meter.update(loss.item(), batch_size)

            pred = logits.argmax(dim=1)
            correct = pred.eq(target).sum().item()
            acc_meter.update(correct / batch_size, batch_size)

        return {
            "loss": loss_meter.avg,
            "accuracy": acc_meter.avg
        }


class FedSOLServer(BaselineServer):
    """FedSOL 服务器。"""

    def aggregate(self, client_models: List[BaselineModel], weights: List[float]):
        """使用加权平均聚合客户端模型。"""
        self.global_model = average_models(client_models, weights)


# ============================================================================
# FedLESAM: 本地估计全局 SAM
# ============================================================================

class FedLESAMClient(BaselineClient):
    """
    FedLESAM 客户端。

    使用全局模型差作为扰动方向进行 SAM 优化。
    论文: Fan et al., "Locally Estimated Global Perturbations are Better than Local Perturbations for Federated SAM", ICML 2024
    """

    def __init__(
            self,
            client_id: int,
            dataloader: DataLoader,
            device: torch.device,
            num_classes: int,
            lr: float = 0.01,
            momentum: float = 0.9,
            weight_decay: float = 1e-4,
            optimizer_name: str = "sgd",
            rho: float = 0.5
    ):
        super().__init__(
            client_id, dataloader, device, num_classes,
            lr, momentum, weight_decay, optimizer_name, mu=0.0
        )
        self.rho = rho
        self.global_direction: Optional[List[torch.Tensor]] = None

    def set_global_direction(
            self,
            current_global: BaselineModel,
            prev_global: Optional[BaselineModel]
    ):
        """
        计算全局梯度方向: d = (w_t - w_{t-1}) / ||w_t - w_{t-1}||
        """
        if prev_global is None:
            self.global_direction = None
            return

        direction = []
        norm_sq = 0.0

        for p_curr, p_prev in zip(current_global.parameters(), prev_global.parameters()):
            diff = p_curr.data - p_prev.data
            direction.append(diff.clone().to(self.device))
            norm_sq += diff.pow(2).sum().item()

        norm = max(norm_sq ** 0.5, 1e-8)

        # 归一化
        for d in direction:
            d.div_(norm)

        self.global_direction = direction

    def train_one_epoch(self) -> Dict[str, float]:
        """训练一个本地轮次（带全局方向扰动）。"""
        self.model.train()

        loss_meter = AverageMeter("loss")
        acc_meter = AverageMeter("accuracy")

        criterion = nn.CrossEntropyLoss()

        for data, target in self.dataloader:
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()

            if self.global_direction is not None:
                # 沿全局方向扰动: w' = w + ρ * d
                for p, d in zip(self.model.parameters(), self.global_direction):
                    p.data.add_(d, alpha=self.rho)

                # 在扰动点计算梯度
                logits = self.model(data)
                loss = criterion(logits, target)
                loss.backward()

                # 恢复模型: w = w - ρ * d
                for p, d in zip(self.model.parameters(), self.global_direction):
                    p.data.sub_(d, alpha=self.rho)
            else:
                # 第一轮没有历史，使用普通 SGD
                logits = self.model(data)
                loss = criterion(logits, target)
                loss.backward()

            self.optimizer.step()

            # 更新指标
            batch_size = data.size(0)
            loss_meter.update(loss.item(), batch_size)

            pred = logits.argmax(dim=1)
            correct = pred.eq(target).sum().item()
            acc_meter.update(correct / batch_size, batch_size)

        return {
            "loss": loss_meter.avg,
            "accuracy": acc_meter.avg
        }


class FedLESAMServer(BaselineServer):
    """FedLESAM 服务器，维护历史全局模型。"""

    def __init__(
            self,
            model_config: Dict,
            device: torch.device,
            test_loader: Optional[DataLoader] = None
    ):
        super().__init__(model_config, device, test_loader)
        self.prev_global_model: Optional[BaselineModel] = None

    def get_prev_global_model(self) -> Optional[BaselineModel]:
        """获取上一轮全局模型。"""
        return self.prev_global_model

    def aggregate(self, client_models: List[BaselineModel], weights: List[float]):
        """使用加权平均聚合客户端模型。"""
        # 保存当前模型作为历史
        self.prev_global_model = copy.deepcopy(self.global_model)
        # 聚合
        self.global_model = average_models(client_models, weights)


# ============================================================================
# pFedHB: 层次贝叶斯个性化联邦学习
# ============================================================================

class pFedHBClient(BaselineClient):
    """
    pFedHB 客户端。

    使用层次贝叶斯框架实现个性化联邦学习。
    论文: Thapa et al., "Harnessing Heterogeneous Statistical Strength for Personalized Federated Learning via Hierarchical Bayesian Inference", ICML 2025
    """

    def __init__(
            self,
            client_id: int,
            dataloader: DataLoader,
            device: torch.device,
            num_classes: int,
            lr: float = 0.01,
            momentum: float = 0.9,
            weight_decay: float = 1e-4,
            optimizer_name: str = "sgd",
            prior_var: float = 1.0,
            posterior_var: float = 0.1,
            kl_weight: float = 0.01
    ):
        super().__init__(
            client_id, dataloader, device, num_classes,
            lr, momentum, weight_decay, optimizer_name, mu=0.0
        )
        self.prior_var = prior_var  # 先验方差
        self.posterior_var = posterior_var  # 后验方差
        self.kl_weight = kl_weight  # KL 散度权重
        self.global_prior: Optional[Dict[str, torch.Tensor]] = None

    def set_global_prior(self, model: BaselineModel):
        """设置全局先验（全局模型参数作为先验均值）。"""
        self.global_prior = {}
        for name, param in model.named_parameters():
            self.global_prior[name] = param.data.clone().to(self.device)

    def _compute_kl_divergence(self) -> torch.Tensor:
        """
        计算本地后验与全局先验之间的 KL 散度。

        假设高斯分布:
        KL(q(θ) || p(θ|θ_global)) = Σ [(μ_local - μ_global)² / (2 * σ²_prior)]
        """
        if self.global_prior is None:
            return torch.tensor(0.0, device=self.device)

        kl_div = torch.tensor(0.0, device=self.device)
        for name, param in self.model.named_parameters():
            if name in self.global_prior:
                diff = param - self.global_prior[name]
                kl_div = kl_div + (diff.pow(2).sum() / (2 * self.prior_var))

        return kl_div

    def train_one_epoch(self) -> Dict[str, float]:
        """训练一个本地轮次（带 KL 散度正则化）。"""
        self.model.train()

        loss_meter = AverageMeter("loss")
        acc_meter = AverageMeter("accuracy")

        criterion = nn.CrossEntropyLoss()

        for data, target in self.dataloader:
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()

            # 本地模型前向
            logits = self.model(data)

            # 任务损失（似然项）
            task_loss = criterion(logits, target)

            # KL 散度正则化（先验项）
            kl_loss = self._compute_kl_divergence()

            # ELBO = E[log p(D|θ)] - KL(q(θ) || p(θ|θ_global))
            # 最小化负 ELBO: L = L_task + λ_kl * KL
            loss = task_loss + self.kl_weight * kl_loss

            loss.backward()
            self.optimizer.step()

            # 更新指标
            batch_size = data.size(0)
            loss_meter.update(loss.item(), batch_size)

            pred = logits.argmax(dim=1)
            correct = pred.eq(target).sum().item()
            acc_meter.update(correct / batch_size, batch_size)

        return {
            "loss": loss_meter.avg,
            "accuracy": acc_meter.avg
        }

    def get_posterior_params(self) -> Dict[str, torch.Tensor]:
        """获取本地后验参数（模型参数作为后验均值）。"""
        posterior = {}
        for name, param in self.model.named_parameters():
            posterior[name] = param.data.clone()
        return posterior


class pFedHBServer(BaselineServer):
    """
    pFedHB 服务器。

    聚合客户端后验参数以更新全局先验。
    """

    def __init__(
            self,
            model_config: Dict,
            device: torch.device,
            test_loader: Optional[DataLoader] = None,
            prior_var: float = 1.0
    ):
        super().__init__(model_config, device, test_loader)
        self.prior_var = prior_var

    def aggregate(self, client_models: List[BaselineModel], weights: List[float]):
        """
        使用贝叶斯聚合更新全局模型。

        在层次贝叶斯框架下，全局后验是各客户端后验的加权平均。
        """
        self.global_model = average_models(client_models, weights)

    def aggregate_posteriors(
            self,
            client_posteriors: List[Dict[str, torch.Tensor]],
            weights: List[float]
    ):
        """
        聚合客户端后验参数。

        参数:
            client_posteriors: 客户端后验参数列表
            weights: 客户端权重（样本数）
        """
        if not client_posteriors:
            return

        # 归一化权重
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]

        # 加权平均后验参数
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                weighted_sum = torch.zeros_like(param)
                for posterior, weight in zip(client_posteriors, normalized_weights):
                    if name in posterior:
                        weighted_sum += posterior[name].to(self.device) * weight
                param.data.copy_(weighted_sum)


# ============================================================================
# FedMoE-DA: 基于域感知细粒度聚合的联邦 MoE
# Zhan et al., "FedMoE-DA: Federated Mixture of Experts via Domain Aware
# Fine-grained Aggregation", INFOCOM 2025
#
# 核心思想：
# 1. 每个客户端训练相同结构的 MoE 模型（backbone + softmax router + experts）
# 2. 训练后，客户端上传 MoE 模型 + router gate 权重（作为 expert proxy）
# 3. 服务器通过 expert proxy 的余弦相似度判断不同客户端的 expert 是否
#    专注于相似的域/类别
# 4. 按域相似度匹配 expert，加权聚合匹配的 expert 对
# 5. backbone 照常 FedAvg 聚合
# ============================================================================

class FedMoEDAClient(BaselineClient):
    """
    FedMoE-DA 客户端。

    使用标准 MoE 模型（softmax router），训练后提取 router gate 权重
    作为 expert 的 domain proxy 上传给服务器。
    """

    def __init__(
            self,
            client_id: int,
            dataloader: DataLoader,
            device: torch.device,
            num_classes: int,
            lr: float = 0.01,
            momentum: float = 0.9,
            weight_decay: float = 1e-4,
            optimizer_name: str = "sgd",
            moe_config: Optional[Dict] = None,
            **kwargs
    ):
        super().__init__(
            client_id, dataloader, device, num_classes,
            lr, momentum, weight_decay, optimizer_name, mu=0.0
        )
        from .moe import create_moe_model
        self.moe_config = moe_config or {}
        # 创建本地 MoE 模型
        self.moe_model: Optional[nn.Module] = None

    def set_moe_model(self, global_moe: nn.Module):
        """从全局 MoE 模型加载参数。"""
        self.moe_model = copy.deepcopy(global_moe).to(self.device)
        self.optimizer = create_optimizer(
            self.moe_model, self.optimizer_name,
            self.lr, self.momentum, self.weight_decay
        )

    def set_model(self, model, keep_global: bool = False):
        """兼容 BaselineClient 接口（不使用）。"""
        pass

    def get_moe_model(self) -> nn.Module:
        return copy.deepcopy(self.moe_model)

    def get_router_weights(self) -> torch.Tensor:
        """
        提取 router gate 的权重矩阵作为 expert domain proxy。

        返回: [num_experts, input_dim] 的权重张量
        """
        return self.moe_model.router.router_network.weight.data.clone().cpu()

    def train_one_epoch(self) -> Dict[str, float]:
        self.moe_model.train()
        loss_meter = AverageMeter("loss")
        acc_meter = AverageMeter("accuracy")
        criterion = nn.CrossEntropyLoss()

        for data, target in self.dataloader:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()

            output = self.moe_model(data)
            logits = output.logits
            loss = criterion(logits, target)

            # Switch Transformer 标准负载均衡损失 (Fedus et al., 2022)
            # L_aux = N * Σ(f_i · P_i)
            # f_i = 被路由到 expert i 的样本比例（硬分配）
            # P_i = expert i 的平均路由概率（软分配）
            # 当所有 expert 均匀使用时 L_aux 最小
            num_experts = output.router_output.expert_probs.size(1)
            expert_probs = output.router_output.expert_probs  # [B, E]
            top_k_indices = output.router_output.top_k_indices  # [B, K]

            # f_i: 每个 expert 被选中的样本比例
            one_hot = F.one_hot(top_k_indices, num_experts).float()  # [B, K, E]
            dispatch_fraction = one_hot.sum(dim=(0, 1))  # [E]
            dispatch_fraction = dispatch_fraction / (data.size(0) * top_k_indices.size(1))

            # P_i: 每个 expert 的平均路由概率
            mean_prob = expert_probs.mean(dim=0)  # [E]

            balance_loss = num_experts * (dispatch_fraction * mean_prob).sum()
            loss = loss + 0.1 * balance_loss

            loss.backward()
            self.optimizer.step()

            batch_size = data.size(0)
            loss_meter.update(loss.item(), batch_size)
            pred = logits.argmax(dim=1)
            acc_meter.update(pred.eq(target).sum().item() / batch_size, batch_size)

        return {"loss": loss_meter.avg, "accuracy": acc_meter.avg}

    def train(self, num_epochs: int) -> Dict[str, float]:
        if self.moe_model is None:
            raise ValueError("MoE 模型未设置。")
        metrics = {}
        for _ in range(num_epochs):
            metrics = self.train_one_epoch()
        return metrics


class FedMoEDAServer(BaselineServer):
    """
    FedMoE-DA 服务器。

    核心: Domain-Aware Fine-grained Aggregation
    - backbone: FedAvg（所有客户端的 backbone 加权平均）
    - router: FedAvg（所有客户端的 router 加权平均）
    - experts: Domain-aware 匹配聚合（按 proxy 余弦相似度匹配后加权平均）

    Expert 匹配算法 (贪心):
    1. 以第一个客户端的 expert 位置作为参考
    2. 对后续每个客户端，计算其所有 expert 与参考 expert 的余弦相似度
    3. 贪心分配：每个参考位置选择相似度最高的未分配 expert
    4. 按相似度加权聚合匹配的 expert
    """

    def __init__(
            self,
            model_config: Dict,
            device: torch.device,
            test_loader: Optional[DataLoader] = None,
            moe_config: Optional[Dict] = None,
    ):
        # 调用 BaselineServer.__init__ 以获取 sample_clients 等方法
        super().__init__(model_config, device, test_loader)

        from .moe import create_moe_model
        self.moe_config = moe_config or {}

        # 全局 MoE 模型（替代 BaselineServer 的 global_model 用于评估）
        self.global_moe = create_moe_model(
            device=device, **self.moe_config
        )

    def get_global_moe(self) -> nn.Module:
        return copy.deepcopy(self.global_moe)

    def get_global_model(self):
        return self.global_moe

    def aggregate_domain_aware(
            self,
            client_moe_models: List[nn.Module],
            sample_weights: List[float],
            router_weights_list: List[torch.Tensor]
    ):
        """
        Domain-Aware Fine-grained Aggregation。

        参数:
            client_moe_models: 客户端 MoE 模型列表
            sample_weights: 样本数权重
            router_weights_list: 每个客户端的 router gate 权重 [num_experts, input_dim]
        """
        if not client_moe_models:
            return

        num_clients = len(client_moe_models)
        total_weight = sum(sample_weights)
        nw = [w / total_weight for w in sample_weights]
        num_experts = self.global_moe.num_experts

        # ====== 1. Backbone: FedAvg（含 BN running stats）======
        with torch.no_grad():
            global_bb_sd = self.global_moe.backbone.state_dict()
            for name in global_bb_sd:
                weighted_sum = torch.zeros_like(global_bb_sd[name], dtype=torch.float32)
                for model, w in zip(client_moe_models, nw):
                    src = model.backbone.state_dict()[name]
                    weighted_sum += src.float().to(self.device) * w
                global_bb_sd[name] = weighted_sum.to(global_bb_sd[name].dtype)
            self.global_moe.backbone.load_state_dict(global_bb_sd)

        # ====== 2. Expert 域匹配 ======
        # 参考 proxy: 使用所有客户端的平均 router 权重（比单个客户端更稳定）
        avg_proxy = torch.stack(router_weights_list).mean(dim=0)  # [E, D]
        avg_proxy = F.normalize(avg_proxy.float(), dim=1)

        # 对每个客户端，计算其 experts 到参考位置的最佳匹配
        expert_assignments = []  # [num_clients][num_experts] -> 源 expert 索引

        for ci in range(num_clients):
            proxy_i = F.normalize(router_weights_list[ci].float(), dim=1)
            sim_matrix = torch.mm(avg_proxy, proxy_i.t())  # [E, E]

            # 贪心匹配（最大权重二部匹配的近似）
            assignment = [-1] * num_experts
            used = set()
            for _ in range(num_experts):
                best_val = -2.0
                best_ref = best_src = -1
                for r in range(num_experts):
                    if assignment[r] != -1:
                        continue
                    for s in range(num_experts):
                        if s in used:
                            continue
                        if sim_matrix[r, s].item() > best_val:
                            best_val = sim_matrix[r, s].item()
                            best_ref, best_src = r, s
                if best_ref >= 0:
                    assignment[best_ref] = best_src
                    used.add(best_src)
            expert_assignments.append(assignment)

        # ====== 3. Router + Experts: 按匹配结果联合聚合 ======
        # 关键: router row e 和 expert e 必须来自同一个客户端的同一个源位置，
        # 否则 router 指向的 expert 行为与实际参数不匹配。
        with torch.no_grad():
            # --- Router: 按匹配重排后加权平均 ---
            router_weight = self.global_moe.router.router_network.weight  # [E, D]
            router_bias = self.global_moe.router.router_network.bias  # [E]
            new_rw = torch.zeros_like(router_weight)
            new_rb = torch.zeros_like(router_bias)

            for e in range(num_experts):
                for ci, (model, w) in enumerate(zip(client_moe_models, nw)):
                    src_idx = expert_assignments[ci][e]
                    src_rw = model.router.router_network.weight.data  # [E, D]
                    src_rb = model.router.router_network.bias.data  # [E]
                    new_rw[e] += src_rw[src_idx].to(self.device) * w
                    new_rb[e] += src_rb[src_idx].to(self.device) * w

            router_weight.data.copy_(new_rw)
            router_bias.data.copy_(new_rb)

            # --- Experts: 按匹配重排后加权平均 ---
            for e in range(num_experts):
                target_expert = self.global_moe.experts.experts[e]
                for name, param in target_expert.named_parameters():
                    weighted_sum = torch.zeros_like(param)
                    for ci, (model, w) in enumerate(zip(client_moe_models, nw)):
                        src_idx = expert_assignments[ci][e]
                        src_expert = model.experts.experts[src_idx]
                        src_param = dict(src_expert.named_parameters())[name]
                        weighted_sum += src_param.data.to(self.device) * w
                    param.data.copy_(weighted_sum)

    def aggregate(self, client_models, weights):
        """兼容接口（不使用）。"""
        pass

    def evaluate(self, model=None) -> Dict[str, float]:
        """评估全局 MoE 模型。"""
        if self.test_loader is None:
            return {}

        self.global_moe.eval()
        correct = total = 0
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_moe(data)
                logits = output.logits
                loss = criterion(logits, target)
                total_loss += loss.item() * target.size(0)
                pred = logits.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)

        if total == 0:
            return {"accuracy": 0.0, "loss": 0.0, "avg_confidence": 0.0}

        return {
            "accuracy": correct / total,
            "loss": total_loss / total,
            "avg_confidence": 0.0
        }


# ============================================================================
# FedEvi: 基于证据不确定性的联邦学习聚合
# Chen et al., "FedEvi: Improving Federated Medical Image Segmentation
# via Evidential Weight Aggregation", MICCAI 2024
#
# 核心思想：
# - 客户端训练使用标准 CE 损失（与 FedAvg 相同）
# - 训练后，在本地数据上评估全局模型和本地模型的 uncertainty
# - 聚合权重基于两个因素调整：
#   (1) Generalization gap: 全局模型在该客户端数据上的不确定性
#   (2) Local reliability: 本地模型的可信度
# - 公式: w_i ∝ n_i × generalization_gap_i × local_reliability_i
# ============================================================================

class FedEviClient(BaselineClient):
    """
    FedEvi 客户端。

    使用标准 CE 损失训练（与 FedAvg 完全相同），训练后通过 post-hoc
    方式计算 uncertainty 指标用于服务器聚合权重调整。
    """

    def __init__(
            self,
            client_id: int,
            dataloader: DataLoader,
            device: torch.device,
            num_classes: int,
            lr: float = 0.01,
            momentum: float = 0.9,
            weight_decay: float = 1e-4,
            optimizer_name: str = "sgd",
            **kwargs
    ):
        super().__init__(
            client_id, dataloader, device, num_classes,
            lr, momentum, weight_decay, optimizer_name, mu=0.0
        )
        self.global_model_copy: Optional[BaselineModel] = None
        self.uncertainty_stats: Optional[Dict[str, float]] = None

    def set_model(self, model: BaselineModel, keep_global: bool = False):
        """设置模型，同时保存全局模型副本。"""
        self.global_model_copy = copy.deepcopy(model).to(self.device)
        for p in self.global_model_copy.parameters():
            p.requires_grad = False

        self.model = copy.deepcopy(model).to(self.device)
        self.optimizer = create_optimizer(
            self.model, self.optimizer_name,
            self.lr, self.momentum, self.weight_decay
        )

    def train_one_epoch(self) -> Dict[str, float]:
        """标准 CE 训练（与 FedAvg 完全相同）。"""
        self.model.train()
        loss_meter = AverageMeter("loss")
        acc_meter = AverageMeter("accuracy")
        criterion = nn.CrossEntropyLoss()

        for data, target in self.dataloader:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(data)
            loss = criterion(logits, target)
            loss.backward()
            self.optimizer.step()

            batch_size = data.size(0)
            loss_meter.update(loss.item(), batch_size)
            pred = logits.argmax(dim=1)
            acc_meter.update(pred.eq(target).sum().item() / batch_size, batch_size)

        return {"loss": loss_meter.avg, "accuracy": acc_meter.avg}

    @torch.no_grad()
    def _compute_model_uncertainty(self, model: nn.Module) -> float:
        """计算模型在本地数据上的归一化熵（uncertainty proxy）。"""
        model.eval()
        total_entropy = 0.0
        total_samples = 0

        for data, _ in self.dataloader:
            data = data.to(self.device)
            logits = model(data)
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
            max_entropy = torch.log(torch.tensor(
                float(self.num_classes), device=self.device))
            total_entropy += (entropy / max_entropy).sum().item()
            total_samples += data.size(0)

        return total_entropy / max(total_samples, 1)

    def compute_uncertainty(self) -> Dict[str, float]:
        """计算 generalization gap 和 local reliability。"""
        global_u = self._compute_model_uncertainty(self.global_model_copy)
        local_u = self._compute_model_uncertainty(self.model)

        self.uncertainty_stats = {
            "generalization_gap": global_u,
            "local_reliability": max(1.0 - local_u, 0.01),
        }
        return self.uncertainty_stats

    def get_uncertainty_stats(self) -> Optional[Dict[str, float]]:
        return self.uncertainty_stats


class FedEviServer(BaselineServer):
    """FedEvi 服务器：uncertainty-weighted 聚合。"""

    def aggregate_with_uncertainty(
            self,
            client_models: List[BaselineModel],
            sample_weights: List[float],
            uncertainty_stats: List[Dict[str, float]]
    ):
        if not client_models:
            return

        adjusted_weights = []
        for sw, stats in zip(sample_weights, uncertainty_stats):
            gap = stats.get("generalization_gap", 0.5)
            reliability = stats.get("local_reliability", 0.5)
            adjusted_weights.append(sw * gap * reliability)

        total = sum(adjusted_weights)
        if total > 0:
            adjusted_weights = [w / total for w in adjusted_weights]
        else:
            adjusted_weights = [1.0 / len(client_models)] * len(client_models)

        self.global_model = average_models(client_models, adjusted_weights)

    def aggregate(self, client_models: List[BaselineModel], weights: List[float]):
        self.global_model = average_models(client_models, weights)
