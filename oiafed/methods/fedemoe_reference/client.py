"""
FedEMoE 联邦客户端实现模块。
"""

import copy
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from .emoe import EMoE, EMoEOutput
from .edl_loss import EDLLoss
from .helpers import AverageMeter, create_optimizer


class FedEMoEClient:
    """
    带有 EMoE 模型的联邦学习客户端。
    
    每个客户端:
    1. 从服务器接收模型
    2. 在私有数据上进行本地训练
    3. 将更新后的模型返回给服务器
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
        edl_lambda1: float = 1.0,
        edl_lambda2: float = 0.1,
        annealing_epochs: int = 10
    ):
        """
        初始化客户端。
        
        参数:
            client_id: 唯一客户端标识符
            dataloader: 本地数据的 DataLoader
            device: 训练设备
            num_classes: 输出类别数
            lr: 学习率
            momentum: SGD 动量
            weight_decay: 权重衰减
            optimizer_name: 优化器类型
            edl_lambda1: EDL 损失权重
            edl_lambda2: KL 散度权重
            annealing_epochs: KL 退火轮数
        """
        self.client_id = client_id
        self.dataloader = dataloader
        self.device = device
        self.num_classes = num_classes
        
        # 训练超参数
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer_name
        
        # EDL 损失参数
        self.edl_lambda1 = edl_lambda1
        self.edl_lambda2 = edl_lambda2
        self.annealing_epochs = annealing_epochs
        
        # 模型和优化器（训练时设置）
        self.model: Optional[EMoE] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.loss_fn: Optional[EDLLoss] = None
        
        # 统计信息
        self.num_samples = len(dataloader.dataset)
    
    def set_model(self, model: EMoE):
        """
        设置该客户端的模型。
        
        参数:
            model: 要使用的 EMoE 模型
        """
        self.model = copy.deepcopy(model).to(self.device)
        
        # 创建优化器
        self.optimizer = create_optimizer(
            self.model,
            self.optimizer_name,
            self.lr,
            self.momentum,
            self.weight_decay
        )
        
        # 创建损失函数
        self.loss_fn = EDLLoss(
            num_classes=self.num_classes,
            lambda1=self.edl_lambda1,
            lambda2=self.edl_lambda2,
            annealing_epochs=self.annealing_epochs
        )
    
    def get_model(self) -> EMoE:
        """获取当前模型。"""
        return self.model
    
    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """
        训练一个本地轮次。
        
        参数:
            epoch: 当前轮次编号（用于退火）
        
        返回:
            训练指标字典
        """
        self.model.train()
        
        # 指标跟踪器
        loss_meter = AverageMeter("loss")
        loss_task_meter = AverageMeter("loss_task")
        loss_edl_meter = AverageMeter("loss_edl")
        acc_meter = AverageMeter("accuracy")
        uncertainty_meter = AverageMeter("uncertainty")
        k_meter = AverageMeter("dynamic_k")
        
        for batch_idx, (data, target) in enumerate(self.dataloader):
            data, target = data.to(self.device), target.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            output: EMoEOutput = self.model(data)
            
            # 计算损失
            loss_dict = self.loss_fn(
                logits=output.logits,
                evidence=output.router_output.evidence,
                target=target,
                epoch=epoch
            )
            
            loss = loss_dict["loss"]
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 更新指标
            batch_size = data.size(0)
            loss_meter.update(loss.item(), batch_size)
            loss_task_meter.update(loss_dict["loss_task"].item(), batch_size)
            loss_edl_meter.update(loss_dict["loss_edl"].item(), batch_size)
            
            # 准确率
            pred = output.logits.argmax(dim=1)
            correct = pred.eq(target).sum().item()
            acc_meter.update(correct / batch_size, batch_size)
            
            # 不确定性和动态 k
            uncertainty_meter.update(
                output.router_output.uncertainty.mean().item(), 
                batch_size
            )
            k_meter.update(
                output.router_output.dynamic_k.float().mean().item(), 
                batch_size
            )
        
        return {
            "loss": loss_meter.avg,
            "loss_task": loss_task_meter.avg,
            "loss_edl": loss_edl_meter.avg,
            "accuracy": acc_meter.avg,
            "uncertainty": uncertainty_meter.avg,
            "dynamic_k": k_meter.avg
        }
    
    def train(self, num_epochs: int, global_epoch: int = 0) -> Dict[str, float]:
        """
        执行本地训练。
        
        参数:
            num_epochs: 本地训练轮数
            global_epoch: 当前全局轮次（用于退火）
        
        返回:
            最终训练指标字典
        """
        if self.model is None:
            raise ValueError("模型未设置。请先调用 set_model()。")
        
        metrics = {}
        for epoch in range(num_epochs):
            # 使用 global_epoch + local_epoch 进行正确的退火
            effective_epoch = global_epoch * num_epochs + epoch
            metrics = self.train_one_epoch(effective_epoch)
        
        return metrics
    
    def evaluate(self) -> Dict[str, float]:
        """
        在本地数据上评估模型。
        
        返回:
            评估指标字典
        """
        if self.model is None:
            raise ValueError("模型未设置。")
        
        self.model.eval()
        
        correct = 0
        total = 0
        total_uncertainty = 0
        
        with torch.no_grad():
            for data, target in self.dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output: EMoEOutput = self.model(data)
                
                pred = output.logits.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
                total_uncertainty += output.router_output.uncertainty.sum().item()
        
        return {
            "accuracy": correct / total if total > 0 else 0.0,
            "avg_uncertainty": total_uncertainty / total if total > 0 else 0.0,
            "num_samples": total
        }
    
    def get_model_state(self) -> Dict:
        """获取模型状态字典。"""
        return self.model.state_dict()
    
    def set_model_state(self, state_dict: Dict):
        """设置模型状态字典。"""
        self.model.load_state_dict(state_dict)

    def compute_evidence_profile(self) -> Dict[int, torch.Tensor]:
        """
        计算每个 expert 的类级别平均证据概况（用于 CEGA 聚合）。

        遍历本地数据集，对每个被路由到 expert e 的样本:
        1. 获取 expert e 的输出 logits
        2. 对 logits 施加 Softplus 得到证据向量 e^(e) ∈ R^C
        3. 累加到 expert e 的证据统计中

        返回:
            {expert_id: avg_evidence_vector[C]}
            若 expert 未被路由任何样本，则其证据为零向量
        """
        import torch.nn.functional as F

        if self.model is None:
            raise ValueError("模型未设置。请先调用 set_model()。")

        self.model.eval()
        num_experts = self.model.num_experts
        num_classes = self.model.num_classes

        evidence_sum = torch.zeros(num_experts, num_classes, device=self.device)
        evidence_count = torch.zeros(num_experts, device=self.device)

        with torch.no_grad():
            for data, target in self.dataloader:
                data = data.to(self.device)

                # 提取特征
                features = self.model.backbone(data)

                # 路由决策
                router_output = self.model.router(features)
                top_k_indices = router_output.top_k_indices   # [batch, k_max]
                top_k_weights = router_output.top_k_weights   # [batch, k_max]

                # 对每个 top-k 位置、每个 expert 累加证据
                for k_pos in range(top_k_indices.shape[1]):
                    for expert_id in range(num_experts):
                        # 被路由到 expert_id 且权重 > 0 的样本
                        mask = (top_k_indices[:, k_pos] == expert_id) & (
                            top_k_weights[:, k_pos] > 0
                        )
                        if not mask.any():
                            continue
                        # expert 输出 logits → Softplus → 证据
                        expert_logits = self.model.experts.experts[expert_id](
                            features[mask]
                        )
                        expert_evidence = F.softplus(expert_logits)  # [n, C]
                        evidence_sum[expert_id] += expert_evidence.sum(dim=0)
                        evidence_count[expert_id] += mask.sum().float()

        # 计算平均证据
        profiles: Dict[int, torch.Tensor] = {}
        for e in range(num_experts):
            if evidence_count[e] > 0:
                profiles[e] = evidence_sum[e] / evidence_count[e]
            else:
                profiles[e] = torch.zeros(num_classes, device=self.device)

        return profiles


class ClientManager:
    """
    多个联邦客户端的管理器。
    """
    
    def __init__(
        self,
        dataloaders: list,
        device: torch.device,
        num_classes: int,
        **client_kwargs
    ):
        """
        初始化客户端管理器。
        
        参数:
            dataloaders: DataLoader 列表，每个客户端一个
            device: 训练设备
            num_classes: 类别数
            **client_kwargs: FedEMoEClient 的额外参数
        """
        self.num_clients = len(dataloaders)
        self.device = device
        
        # 创建客户端
        self.clients = []
        for client_id, dataloader in enumerate(dataloaders):
            client = FedEMoEClient(
                client_id=client_id,
                dataloader=dataloader,
                device=device,
                num_classes=num_classes,
                **client_kwargs
            )
            self.clients.append(client)
    
    def get_client(self, client_id: int) -> FedEMoEClient:
        """根据 ID 获取客户端。"""
        return self.clients[client_id]
    
    def get_clients(self, client_ids: list) -> list:
        """根据 ID 列表获取多个客户端。"""
        return [self.clients[i] for i in client_ids]
    
    def get_all_clients(self) -> list:
        """获取所有客户端。"""
        return self.clients
