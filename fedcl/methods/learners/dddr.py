"""
DDDR (Diffusion-Driven Data Replay) 学习器

基于FedCL框架实现DDDR方法，专注于本地客户端训练逻辑：
1. 本地分类器训练（当前任务数据 + 合成历史数据）
2. 本地生成器嵌入训练（用于类反演）
3. 模型权重管理

联邦协调逻辑在DDDRFederationTrainer中实现。
"""

import os
from copy import deepcopy
from tqdm import tqdm, trange
from glob import glob
from omegaconf import OmegaConf

import numpy as np
from typing import Dict, Any, Optional, List
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from einops import rearrange

from ...fl.base_learner import AbstractLearner
from ...models import (
    IncrementalNet, 
    SupConLoss, 
    GenDataset, 
    DatasetSplit, 
    DataIter,
    kd_loss
)
from ...api.decorators import learner


@learner("dddr", description="DDDR持续学习器")
class DDDRLearner(AbstractLearner):
    """
    DDDR学习器 - 专注于本地客户端训练逻辑
    
    职责：
    1. 本地分类器训练（当前任务 + 历史合成数据）
    2. 本地生成器嵌入训练（用于联邦类反演）
    3. 模型权重管理和更新
    
    不负责：
    - 联邦聚合逻辑（在Trainer中）
    - 全局模型管理（在Trainer中）
    - 客户端选择和协调（在Trainer中）
    """
    
    def __init__(self, client_id: str, config: Dict[str, Any]):
        super().__init__(client_id, config)
        
        # DDDR损失权重配置
        self.w_kd = config.get("w_kd", 10.0)      # 知识蒸馏权重
        self.w_ce_pre = config.get("w_ce_pre", 0.5)  # 历史数据交叉熵权重
        self.w_scl = config.get("w_scl", 1.0)     # 对比学习权重
        
        # 生成器训练配置
        self.g_local_train_steps = config.get("g_local_train_steps", 50)
        self.g_local_bs = config.get("g_local_bs", 12)
        
        # 初始化分类网络
        if "net" not in config:
            config["net"] = "resnet18"
        self._network = IncrementalNet(config, False)
        self.model = self._network  # 兼容AbstractLearner
        
        # 初始化扩散模型（用于生成器嵌入训练）
        self._generator = None
        self.generator_init_embedding = None
        self._init_generator()
        
        # 对比学习损失函数
        self.scl_criterion = SupConLoss() if self.w_scl > 0 else None
        
        # 任务状态
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self._old_network = None  # 用于知识蒸馏的旧模型
        
        # 数据加载器（由外部设置）
        self.train_loader = None
        self.gen_data_loader = None  # 用于生成器训练
        self.pre_syn_data_iter = None  # 历史合成数据迭代器
        
        # 类别信息
        self.current_classes = []
        self.min_class_id = 0
        self.max_class_id = 0
        
        self.logger.info(f"✅ DDDR学习器初始化完成 - 客户端: {client_id}")

    # ============ FedCL框架接口方法 ============
    
    def train_task(self, task_id: int, train_data: Any = None, **kwargs) -> Dict[str, Any]:
        """
        FedCL框架要求的train_task方法
        根据任务ID决定是首任务训练还是增量任务训练
        """
        self.logger.info(f"开始训练任务 {task_id}")
        
        # 更新任务状态
        if task_id > self._cur_task:
            self._cur_task = task_id
            self._known_classes = self._total_classes
            self._total_classes += len(self.current_classes)
            
            # 保存旧模型用于知识蒸馏
            if self._old_network is None:
                self._old_network = deepcopy(self._network)
            else:
                self._old_network.load_state_dict(self._network.state_dict())
        
        # 根据任务类型选择训练方法
        if task_id == 0:
            # 首任务：使用local_update
            return self.local_update(round_num=task_id)
        else:
            # 增量任务：使用local_finetune
            return self.local_finetune(round_num=task_id, known_classes=self._known_classes)
    
    def train_on_client(self, client_data: Any = None, **kwargs) -> Dict[str, Any]:
        """
        FedCL框架要求的train_on_client方法
        兼容性接口，调用train_task
        """
        task_id = kwargs.get('task_id', 0)
        return self.train_task(task_id, client_data, **kwargs)
        """
        首任务本地更新（对应ours.py的_local_update）
        仅使用当前任务数据和可选的SCL，不包含回放与KD。
        """
        if self.train_loader is None:
            raise ValueError("train_loader 未设置。请用户在本地构建并通过 set_train_data 传入。")

        self._network.train()
        self._network.to(self.device if device is None else device)

        optimizer = torch.optim.SGD(self._network.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        local_ep = int(self.config.get("local_ep", self.config.get("local_epochs", 1)))

        total_loss = 0.0
        total_samples = 0

        for _ in range(local_ep):
            for _, (_, images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                output = self._network(images)
                loss = F.cross_entropy(output["logits"], labels)
                if self.w_scl > 0 and self.scl_criterion is not None:
                    loss_scl = self.scl_criterion(output['scl_emb'], labels)
                    loss = loss + self.w_scl * loss_scl
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * images.size(0)
                total_samples += images.size(0)

        avg_loss = total_loss / max(1, total_samples)
        return {
            "model_weights": self.get_model_weights(),
            "num_samples": total_samples,
            "loss": avg_loss,
            "round_num": round_num,
        }

    def local_finetune(self, round_num: int = 0, device: str = None, known_classes: int = None) -> Dict[str, Any]:
        """
        增量任务微调（对应ours.py的_local_finetune）
        使用当前任务数据 + 历史合成回放 + 知识蒸馏 + 可选SCL。
        需要：self._old_network 和 self.pre_syn_data_iter
        """
        if self.train_loader is None:
            raise ValueError("train_loader 未设置。请用户在本地构建并通过 set_train_data 传入。")
        if self._old_network is None:
            self.logger.warning("_old_network 未设置，KD将被跳过。可在任务切换后调用 save_old_model。")

        self._network.train()
        self._network.to(self.device if device is None else device)
        if self._old_network is not None:
            self._old_network.eval()

        optimizer = torch.optim.SGD(self._network.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        local_ep = int(self.config.get("local_ep", self.config.get("local_epochs", 1)))

        total_loss = 0.0
        total_samples = 0

        for _ in range(local_ep):
            for _, (_, images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)

                # 当前任务的分类损失（对新类别部分）
                output = self._network(images)
                if known_classes is None:
                    known_classes = self._known_classes
                fake_targets = labels - known_classes
                loss = F.cross_entropy(output["logits"][:, known_classes:], fake_targets)

                # 历史合成数据回放的交叉熵
                if self.w_ce_pre > 0 and self.pre_syn_data_iter is not None:
                    try:
                        _, pre_imgs, pre_labels = self.pre_syn_data_iter.next()
                        pre_imgs, pre_labels = pre_imgs.to(self.device), pre_labels.to(self.device)
                        s_out = self._network(pre_imgs)
                        loss_ce_pre = F.cross_entropy(s_out["logits"][:, :known_classes], pre_labels)
                        loss = loss + self.w_ce_pre * loss_ce_pre
                    except StopIteration:
                        pass

                # 知识蒸馏损失
                if self.w_kd > 0 and self._old_network is not None and self.pre_syn_data_iter is not None:
                    with torch.no_grad():
                        t_out = self._old_network(pre_imgs.detach())["logits"]
                    loss_kd = kd_loss(
                        s_out["logits"][:, : known_classes],
                        t_out.detach(),
                        2,
                    )
                    loss = loss + self.w_kd * loss_kd

                # 对比学习损失
                if self.w_scl > 0 and self.scl_criterion is not None:
                    loss_scl = self.scl_criterion(output['scl_emb'], labels)
                    loss = loss + self.w_scl * loss_scl

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * images.size(0)
                total_samples += images.size(0)

        avg_loss = total_loss / max(1, total_samples)
        return {
            "model_weights": self.get_model_weights(),
            "num_samples": total_samples,
            "loss": avg_loss,
            "round_num": round_num,
        }

    def set_current_syn_data(self, syn_imgs_dir: str, task_id: int, cur_size: int = 50) -> Dict[str, Any]:
        """设置当前任务的合成数据并合并到训练集（尽力合并）"""
        try:
            from fedcl.models.data_utils import TaskSynImageDataset
            from torch.utils.data import ConcatDataset, DataLoader
            cur_syn_dataset = TaskSynImageDataset(syn_imgs_dir, task_id, cur_size, transform=None)
            if self.train_loader is not None and hasattr(self.train_loader, 'dataset'):
                concat_ds = ConcatDataset([self.train_loader.dataset, cur_syn_dataset])
                bs = getattr(self.train_loader, 'batch_size', self.config.get("batch_size", 32))
                self.train_loader = DataLoader(concat_ds, batch_size=bs, shuffle=True, num_workers=0)
            return {"status": "ok", "cur_added": len(cur_syn_dataset)}
        except Exception as e:
            self.logger.warning(f"设置当前任务合成数据失败: {e}")
            return {"status": "error", "error": str(e)}

    def set_replay_syn_data(self, syn_imgs_dir: str, current_task: int, pre_size: int = 200) -> Dict[str, Any]:
        """设置历史任务的合成数据回放迭代器"""
        try:
            from fedcl.models.data_utils import TaskSynImageDataset, DataIter
            from torch.utils.data import ConcatDataset, DataLoader
            pre_syn_datasets = []
            for i in range(current_task):
                ds = TaskSynImageDataset(syn_imgs_dir, i, pre_size, transform=None)
                if len(ds) > 0:
                    pre_syn_datasets.append(ds)
            if pre_syn_datasets:
                pre_syn_dataset = ConcatDataset(pre_syn_datasets)
                pre_loader = DataLoader(pre_syn_dataset, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
                self.pre_syn_data_iter = DataIter(pre_loader)
                return {"status": "ok", "replay_size": len(pre_syn_dataset)}
            else:
                self.pre_syn_data_iter = None
                return {"status": "ok", "replay_size": 0}
        except Exception as e:
            self.logger.warning(f"设置历史回放数据失败: {e}")
            return {"status": "error", "error": str(e)}
    
    def _init_generator(self):
        """初始化扩散模型生成器（用于类反演）"""
        try:
            # 加载LDM配置
            config_path = self.config.get("ldm_config", "config/ldm_dddr.yaml")
            ckpt_path = self.config.get("ldm_ckpt", "PM/ldm/text2img-large/model.ckpt")
            
            if not os.path.exists(config_path) or not os.path.exists(ckpt_path):
                self.logger.warning(f"LDM文件不存在: {config_path} 或 {ckpt_path}")
                return
            
            # 加载配置并创建生成器
            ldm_config = OmegaConf.load(config_path)
            ldm_config.model.params.ckpt_path = ckpt_path
            ldm_config.model.params.personalization_config.params.num_classes = \
                self.config.get('total_classes', 100)
            
            from ...models.ldm import LatentDiffusion
            self._generator = LatentDiffusion(**ldm_config.model.params)
            
            # 加载预训练权重
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint)
            self._generator.load_state_dict(state_dict, strict=False)
            
            # 保存初始嵌入状态（用于重置）
            if hasattr(self._generator, 'embedding_manager'):
                self.generator_init_embedding = deepcopy(
                    self._generator.embedding_manager.state_dict()
                )
            
            # 设置学习率
            batch_size = ldm_config.data.params.batch_size
            base_lr = ldm_config.model.base_learning_rate
            self._generator.learning_rate = batch_size * base_lr
            
            self.logger.info(f"✅ 生成器初始化完成，学习率: {self._generator.learning_rate:.2e}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ 生成器初始化失败: {e}")
            self._generator = None
    
    # ============ 简化的接口方法 ============
    
    async def train_epoch(self, **kwargs) -> Dict[str, Any]:
        """
        执行本地DDDR训练
        
        实现DDDR的本地训练逻辑：
        1. 当前任务数据训练
        2. 历史合成数据回放（如果有）
        3. 知识蒸馏（如果有旧模型）
        4. 对比学习损失
        """
        if self.train_loader is None:
            raise ValueError("训练数据加载器未设置，请先调用set_train_data")
        
        # 设置训练模式和设备
        self._network.train()
        self._network.to(self.device)
        
        # 获取训练参数
        round_num = kwargs.get("round_num", 0)
        local_epochs = kwargs.get("local_epochs", self.config.get("local_epochs", 1))
        learning_rate = kwargs.get("learning_rate", 0.01)
        
        # 创建优化器
        optimizer = torch.optim.SGD(
            self._network.parameters(), 
            lr=learning_rate, 
            momentum=0.9, 
            weight_decay=5e-4
        )
        
        total_loss = 0.0
        total_samples = 0
        
        # 本地训练轮次
        for epoch in range(local_epochs):
            epoch_loss = 0.0
            epoch_samples = 0
            
            for batch_idx, (_, images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                # 计算当前任务损失
                loss = self._compute_current_task_loss(images, labels)
                
                # 添加历史数据损失（如果有）
                if self.w_ce_pre > 0 and self.pre_syn_data_iter is not None:
                    loss += self._compute_replay_loss()
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * images.size(0)
                epoch_samples += images.size(0)
            
            total_loss += epoch_loss
            total_samples += epoch_samples
        
        # 计算平均损失
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        
        # 构建返回结果
        result = {
            'model_weights': self.get_model_weights(),
            'num_samples': total_samples,
            'loss': avg_loss,
            'local_epochs': local_epochs,
            'round_num': round_num,
            'task_id': self._cur_task
        }
        
        # 记录训练历史
        self.update_round(round_num)
        self.log_training_result(result)
        
        self.logger.info(
            f"✅ 本地训练完成 - 轮次: {round_num}, 任务: {self._cur_task}, "
            f"损失: {avg_loss:.4f}, 样本数: {total_samples}"
        )
        
        return result
    
    def _compute_current_task_loss(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """计算当前任务的损失"""
        # 前向传播
        output = self._network(images)
        
        # 交叉熵损失
        loss = F.cross_entropy(output["logits"], labels)
        
        # 对比学习损失
        if self.w_scl > 0 and self.scl_criterion is not None:
            loss_scl = self.scl_criterion(output['scl_emb'], labels)
            loss = loss + self.w_scl * loss_scl
        
        # 知识蒸馏损失（如果有旧模型）
        if self._old_network is not None and self._cur_task > 0:
            with torch.no_grad():
                old_output = self._old_network(images)
            
            if old_output["logits"].size(1) <= output["logits"].size(1):
                kd_loss_val = kd_loss(
                    output["logits"][:, :old_output["logits"].size(1)],
                    old_output["logits"].detach(),
                    2
                )
                loss = loss + self.w_kd * kd_loss_val
        
        return loss
    
    def _compute_replay_loss(self) -> torch.Tensor:
        """计算历史数据回放损失"""
        try:
            _, pre_imgs, pre_labels = self.pre_syn_data_iter.next()
            pre_imgs, pre_labels = pre_imgs.to(self.device), pre_labels.to(self.device)
            
            s_out = self._network(pre_imgs)
            loss_ce_pre = F.cross_entropy(
                s_out["logits"][:, :self._known_classes], 
                pre_labels
            )
            return self.w_ce_pre * loss_ce_pre
        except StopIteration:
            return torch.tensor(0.0, device=self.device)
    
    async def evaluate(self, **kwargs) -> Dict[str, Any]:
        """执行本地模型评估"""
        if self._network is None:
            raise ValueError("模型未初始化")
        
        test_loader = kwargs.get("test_loader")
        if test_loader is None:
            raise ValueError("测试数据加载器未提供")
        
        self._network.eval()
        metrics = kwargs.get("metrics", ["accuracy", "loss"])
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for _, (_, images, labels) in enumerate(test_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                output = self._network(images)
                
                if "loss" in metrics:
                    loss = F.cross_entropy(output["logits"], labels)
                    total_loss += loss.item() * images.size(0)
                
                if "accuracy" in metrics:
                    pred = output["logits"].argmax(dim=1)
                    correct += (pred == labels).sum().item()
                
                total += images.size(0)
        
        results = {"num_samples": total, "task_id": self._cur_task}
        
        if "loss" in metrics and total > 0:
            results["loss"] = total_loss / total
        
        if "accuracy" in metrics and total > 0:
            results["accuracy"] = correct / total
        
        self.logger.info(f"✅ 评估完成 - 任务: {self._cur_task}, 准确率: {results.get('accuracy', 0):.4f}")
        
        return results
    
    def get_model_weights(self) -> Dict[str, Any]:
        """获取模型权重"""
        if self._network is None:
            raise ValueError("模型未初始化")
        return {k: v.cpu().clone() for k, v in self._network.state_dict().items()}
    
    def set_model_weights(self, weights: Dict[str, Any]) -> None:
        """设置模型权重"""
        if self._network is None:
            raise ValueError("模型未初始化")
        
        # 将权重移动到正确的设备
        device_weights = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
            for k, v in weights.items()
        }
        
        try:
            self._network.load_state_dict(device_weights, strict=False)
            self.logger.debug("✅ 模型权重已更新")
        except Exception as e:
            # 尝试部分更新
            model_dict = self._network.state_dict()
            pretrained_dict = {k: v for k, v in device_weights.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self._network.load_state_dict(model_dict)
            self.logger.warning(f"⚠️ 模型权重部分更新: {e}")
    
    # ============ 数据设置方法 ============
    
    def set_train_data(self, train_loader: DataLoader) -> None:
        """设置训练数据加载器"""
        self.train_loader = train_loader
        self.logger.debug(f"✅ 训练数据加载器已设置，批次大小: {train_loader.batch_size}")
    
    def set_generator_data(self, gen_data_loader: DataLoader) -> None:
        """设置生成器训练数据加载器"""
        self.gen_data_loader = gen_data_loader
        self.logger.debug("✅ 生成器数据加载器已设置")
    
    def set_replay_data(self, replay_data_iter) -> None:
        """设置历史数据回放迭代器"""
        self.pre_syn_data_iter = replay_data_iter
        self.logger.debug("✅ 历史数据回放迭代器已设置")
    
    # ============ 生成器相关方法 ============
    
    def train_generator_embeddings(self) -> Optional[Dict[str, Any]]:
        """
        训练生成器嵌入（用于联邦类反演）
        
        这是DDDR的核心组件之一，本地训练生成器的嵌入参数，
        用于后续的联邦聚合和图像生成。
        """
        if self._generator is None or self.gen_data_loader is None:
            self.logger.warning("⚠️ 生成器或生成数据未初始化，跳过嵌入训练")
            return None
        
        try:
            # 重置嵌入到初始状态
            self._generator.to(self.device)
            if self.generator_init_embedding is not None:
                self._generator.embedding_manager.load_state_dict(self.generator_init_embedding)

            # 使用持久数据迭代器，避免每步重新从头取首个batch
            data_iter = iter(self.gen_data_loader)

            # 本地训练生成器嵌入
            for step in range(self.g_local_train_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.gen_data_loader)
                    batch = next(data_iter)

                batch["image"] = batch["image"].to(self.device)

                # 计算损失并更新
                loss, _ = self._generator.shared_step(batch)
                optim = self._generator.configure_optimizers()
                optim.zero_grad()
                loss.backward()
                optim.step()

                if step % 10 == 0:
                    self.logger.debug(
                        f"生成器训练步数 {step + 1}/{self.g_local_train_steps}, 损失: {loss.item():.4f}"
                    )
            
            # 返回训练后的嵌入权重
            embedding_weights = self._generator.embedding_manager.state_dict()
            self.logger.info("✅ 生成器嵌入训练完成")
            return embedding_weights
            
        except Exception as e:
            self.logger.error(f"❌ 生成器嵌入训练失败: {e}")
            return None
    
    def update_task_info(self, task_id: int, known_classes: int, total_classes: int, 
                        current_classes: List[int]) -> None:
        """更新任务信息"""
        self._cur_task = task_id
        self._known_classes = known_classes
        self._total_classes = total_classes
        self.current_classes = current_classes
        
        # 更新网络输出层
        self._network.update_fc(total_classes)
        
        if current_classes:
            self.min_class_id = min(current_classes)
            self.max_class_id = max(current_classes)
        
        self.logger.info(
            f"✅ 任务更新完成 - 任务: {task_id}, 已知类别: {known_classes}, "
            f"总类别: {total_classes}, 当前类别: {current_classes}"
        )
    
    def save_old_model(self) -> None:
        """保存当前模型作为旧模型（用于知识蒸馏）"""
        if self._network is not None:
            self._old_network = self._network.copy().freeze()
            self.logger.debug("✅ 旧模型已保存")
