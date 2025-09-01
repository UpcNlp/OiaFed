"""
DDDR (Diffusion-Driven Data Replay) 学习器

基于FedCL框架实现DDDR方法，只负责本地客户端训练逻辑。
"""

import os
from copy import deepcopy
from tqdm import tqdm, trange
from glob import glob
from omegaconf import OmegaConf

import numpy as np
from typing import Dict, Any
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from einops import rearrange

from ...execution.base_learner import AbstractLearner
from ...models import (
    IncrementalNet, 
    SupConLoss, 
    GenDataset, 
    DatasetSplit, 
    DataIter,
    kd_loss
)
from ...api.decorators import learner


@learner("contrastive", description="联邦对比学习器")
class DDDRLearner(AbstractLearner):
    """DDDR学习器 - 只负责本地客户端训练逻辑"""
    
    def __init__(self, client_id: str, config: Dict[str, Any]):
        super().__init__(client_id, config)
        
        # DDDR特定配置
        self.w_kd = config.get("w_kd", 10.0)
        self.w_ce_pre = config.get("w_ce_pre", 0.5)
        self.w_scl = config.get("w_scl", 1.0)
        
        # 扩散模型配置
        self.ldm_config_path = config.get("ldm_config", "ldm/ldm_dddr.yaml")
        self.ldm_ckpt_path = config.get("ldm_ckpt", "models/ldm/text2img-large/model.ckpt")
        
        # 数据生成配置
        self.pre_size = config.get("pre_size", 200)
        self.cur_size = config.get("cur_size", 50)
        self.n_iter = config.get("n_iter", 5)
        
        # 训练配置
        self.g_local_train_steps = config.get("g_local_train_steps", 50)
        self.g_local_bs = config.get("g_local_bs", 12)
        
        # 初始化网络
        self._network = IncrementalNet(config, False)
        
        # 初始化扩散模型生成器
        self.generator_init()
        
        # 对比学习损失
        if self.w_scl > 0:
            self.scl_criterion = SupConLoss()
        else:
            self.scl_criterion = None
        
        # 合成数据路径
        self.need_syn_imgs = config.get('syn_image_path') is None
        if config.get('syn_image_path') is not None:
            self.syn_imgs_dir = config['syn_image_path']
        else:
            self.syn_imgs_dir = os.path.join(config.get('save_dir', 'outputs'), "syn_imgs")
        
        # 任务相关状态
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self._old_network = None
        
        # 数据加载器
        self.local_train_dataset = None
        self.local_cur_loader = None
        self.gen_data_loader = None
        self.pre_syn_data_iter = None
        
        # 类别信息
        self.all_classes = []
        self.min_class_id = 0
        self.max_class_id = 0
        
        self.logger.info(f"DDDR学习器初始化完成 - 客户端: {client_id}")
    
    def generator_init(self):
        """初始化扩散模型生成器"""
        try:
            self.config = OmegaConf.load(self.ldm_config_path)
            self.config.model.params.ckpt_path = self.ldm_ckpt_path
            self.config['model']["params"]['personalization_config']["params"]['num_classes'] = \
                self.config.get('increment', 20)
            
            # 导入LDM模块
            from ldm import LatentDiffusion
            self._generator = LatentDiffusion(**self.config['model']["params"])
            self._generator.load_state_dict(
                torch.load(self.ldm_ckpt_path, map_location="cpu")["state_dict"], 
                strict=False
            )
            
            self.generator_init_embedding = deepcopy(self._generator.embedding_manager.state_dict())
            self._generator.learning_rate = (
                self.config.data.params.batch_size * 
                self.config.model.base_learning_rate
            )
            
            self.logger.info(f"扩散模型初始化完成，学习率: {self._generator.learning_rate:.2e}")
        except Exception as e:
            self.logger.warning(f"扩散模型初始化失败: {e}")
            self._generator = None
    
    def update_task(self, task_id: int, known_classes: int, total_classes: int):
        """更新任务信息"""
        self._cur_task = task_id
        self._known_classes = known_classes
        self._total_classes = total_classes
        
        # 更新网络输出层
        self._network.update_fc(self._total_classes)
        
        self.logger.info(f"任务更新: {task_id}, 已知类别: {known_classes}, 总类别: {total_classes}")
    
    def after_task(self):
        """任务完成后的处理"""
        self._known_classes = self._total_classes
        self._old_network = self._network.copy().freeze()
        self.logger.info(f"任务 {self._cur_task} 完成，已知类别数: {self._known_classes}")
    
    def set_train_data(self, train_dataset, class_ids):
        """设置训练数据"""
        self.local_train_dataset = train_dataset
        self.all_classes = class_ids
        self.min_class_id, self.max_class_id = np.min(self.all_classes), np.max(self.all_classes)
        
        # 创建生成数据加载器
        if self._generator is not None:
            gen_dataset = GenDataset(
                input_np_array=train_dataset.images,
                class_ids=train_dataset.labels
            )
            gen_dataset.min_class_id = self.min_class_id
            self.gen_data_loader = DataLoader(
                gen_dataset, 
                batch_size=self.g_local_bs,
                num_workers=4, 
                shuffle=True
            )
    
    def set_synthetic_data(self, syn_imgs_dir: str, task_id: int, transform=None):
        """设置合成数据"""
        from fedcl.models import TaskSynImageDataset
        
        # 当前任务的合成数据
        cur_syn_dataset = TaskSynImageDataset(
            syn_imgs_dir, task_id, self.cur_size,
            transform=transform
        )
        
        # 合并真实数据和合成数据
        if self.local_train_dataset is not None:
            combined_dataset = ConcatDataset([self.local_train_dataset, cur_syn_dataset])
            self.local_cur_loader = DataLoader(
                combined_dataset,
                batch_size=self.config.get("batch_size", 128),
                shuffle=True,
                num_workers=4
            )
        
        # 历史任务的合成数据
        if task_id > 0:
            pre_syn_datasets = []
            for i in range(task_id):
                pre_syn_dataset = TaskSynImageDataset(
                    syn_imgs_dir, i, self.pre_size,
                    transform=transform
                )
                pre_syn_datasets.append(pre_syn_dataset)
            
            pre_syn_dataset = ConcatDataset(pre_syn_datasets)
            pre_syn_data_loader = DataLoader(
                pre_syn_dataset,
                batch_size=128,
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )
            self.pre_syn_data_iter = DataIter(pre_syn_data_loader)
    
    def train_generator_embeddings(self):
        """训练生成器嵌入 - 本地客户端逻辑"""
        if self._generator is None or self.gen_data_loader is None:
            return None
        
        self._generator.cuda()
        self._generator.embedding_manager.load_state_dict(self.generator_init_embedding)
        
        # 本地训练生成器
        for _ in range(self.g_local_train_steps):
            batch = self.gen_data_loader.next()
            batch["image"] = batch["image"].cuda()
            loss, _ = self._generator.shared_step(batch)
            
            optim = self._generator.configure_optimizers()
            optim.zero_grad()
            loss.backward()
            optim.step()
        
        return self._generator.embedding_manager.state_dict()
    
    def generate_synthetic_images(self, class_ids: list, output_dir: str):
        """生成合成图像 - 本地客户端逻辑"""
        if self._generator is None:
            return
        
        # 导入DDIM采样器
        from ldm import DDIMSampler
        sampler = DDIMSampler(self._generator)
        
        os.makedirs(output_dir, exist_ok=True)
        
        prompt = "a photo of *"
        n_samples = 40
        scale = 10.0
        ddim_steps = 50
        ddim_eta = 0.0
        H = 256
        W = 256
        
        with torch.no_grad():
            for tmp_cls in class_ids:
                base_count = 0
                class_dir = os.path.join(output_dir, str(tmp_cls))
                os.makedirs(class_dir, exist_ok=True)
                
                with self._generator.ema_scope():
                    uc = None
                    tmp_cls_tensor = torch.LongTensor([tmp_cls - self.min_class_id,] * n_samples)
                    
                    if scale != 1.0:
                        uc = self._generator.get_learned_conditioning(n_samples * [""], tmp_cls_tensor)
                    
                    for _ in trange(self.n_iter, desc="Sampling"):
                        c = self._generator.get_learned_conditioning(n_samples * [prompt], tmp_cls_tensor)
                        shape = [4, H//8, W//8]
                        samples_ddim, _ = sampler.sample(
                            S=ddim_steps,
                            conditioning=c,
                            batch_size=n_samples,
                            shape=shape,
                            verbose=False,
                            unconditional_guidance_scale=scale,
                            unconditional_conditioning=uc,
                            eta=ddim_eta
                        )
                        
                        x_samples_ddim = self._generator.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
                        
                        for x_sample in x_samples_ddim:
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            Image.fromarray(x_sample.astype(np.uint8)).save(
                                os.path.join(class_dir, f"{tmp_cls}-{base_count}.jpg")
                            )
                            base_count += 1
    
    async def train_epoch(self, **kwargs) -> Dict[str, Any]:
        """执行一个epoch的训练 - FedCL框架要求"""
        if self.local_cur_loader is None:
            raise ValueError("训练数据未设置")
        
        self._network.train()
        self._network.cuda()
        
        # 获取训练参数
        round_num = kwargs.get("round_num", 0)
        local_epochs = kwargs.get("local_epochs", self.config.get("local_epochs", 5))
        
        total_loss = 0.0
        total_samples = 0
        
        for epoch in range(local_epochs):
            epoch_loss = 0.0
            epoch_samples = 0
            
            for batch_idx, (_, images, labels) in enumerate(self.local_cur_loader):
                images, labels = images.cuda(), labels.cuda()
                
                # 前向传播
                output = self._network(images)
                loss = F.cross_entropy(output["logits"], labels)
                
                # 对比学习损失
                if self.w_scl > 0 and self.scl_criterion is not None:
                    loss_scl = self.scl_criterion(output['scl_emb'], labels)
                    loss = loss + self.w_scl * loss_scl
                
                # 知识蒸馏损失（如果有旧模型）
                if self._old_network is not None and self._cur_task > 0:
                    with torch.no_grad():
                        old_output = self._old_network(images)
                    
                    # 只对已知类别进行蒸馏
                    if old_output["logits"].size(1) <= output["logits"].size(1):
                        kd_loss_val = kd_loss(
                            output["logits"][:, :old_output["logits"].size(1)],
                            old_output["logits"].detach(),
                            2
                        )
                        loss = loss + self.w_kd * kd_loss_val
                
                # 历史数据损失
                if self.w_ce_pre > 0 and self.pre_syn_data_iter is not None:
                    try:
                        _, pre_imgs, pre_labels = self.pre_syn_data_iter.next()
                        pre_imgs, pre_labels = pre_imgs.cuda(), pre_labels.cuda()
                        
                        s_out = self._network(pre_imgs)
                        loss_ce_pre = F.cross_entropy(s_out["logits"][:, :self._known_classes], pre_labels)
                        loss = loss + self.w_ce_pre * loss_ce_pre
                    except StopIteration:
                        pass
                
                # 反向传播
                optimizer = torch.optim.SGD(self._network.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * images.size(0)
                epoch_samples += images.size(0)
            
            total_loss += epoch_loss
            total_samples += epoch_samples
            
            self.logger.debug(
                f"轮次 {round_num}, Epoch {epoch+1}/{local_epochs}, "
                f"损失: {epoch_loss/epoch_samples:.4f}"
            )
        
        # 计算平均损失
        avg_loss = total_loss / (total_samples * local_epochs)
        
        # 准备返回结果
        result = {
            'model_weights': self.get_model_weights(),
            'num_samples': total_samples,
            'loss': avg_loss,
            'local_epochs': local_epochs,
            'round_num': round_num,
            'task_id': self._cur_task
        }
        
        # 记录训练结果
        self.update_round(round_num)
        self.log_training_result(result)
        
        self.logger.info(
            f"训练完成 - 轮次: {round_num}, 任务: {self._cur_task}, "
            f"损失: {avg_loss:.4f}, 样本数: {total_samples}"
        )
        
        return result
    
    async def evaluate(self, **kwargs) -> Dict[str, Any]:
        """执行评估 - FedCL框架要求"""
        if self._network is None:
            raise ValueError("模型未初始化")
        
        self._network.eval()
        
        # 获取测试数据
        test_loader = kwargs.get("test_loader")
        if test_loader is None:
            raise ValueError("测试数据加载器未提供")
        
        # 获取需要计算的指标
        metrics = kwargs.get("metrics", ["accuracy", "loss"])
        
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for _, (_, images, labels) in enumerate(test_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                output = self._network(images)
                
                # 计算损失
                if "loss" in metrics:
                    loss = F.cross_entropy(output["logits"], labels)
                    total_loss += loss.item() * images.size(0)
                
                # 计算准确率
                if "accuracy" in metrics:
                    pred = output["logits"].argmax(dim=1, keepdim=True)
                    correct_predictions += pred.eq(labels.view_as(pred)).sum().item()
                
                total_samples += images.size(0)
        
        # 构建结果字典
        results = {}
        
        if "loss" in metrics:
            results["loss"] = total_loss / total_samples
        
        if "accuracy" in metrics:
            results["accuracy"] = correct_predictions / total_samples
        
        # 添加样本数量
        results["num_samples"] = total_samples
        results["task_id"] = self._cur_task
        
        self.logger.info(f"评估完成 - 任务: {self._cur_task}, 结果: {results}")
        
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
        device_weights = {k: v.to(self.device) for k, v in weights.items()}
        self._network.load_state_dict(device_weights)
        self.logger.debug("模型权重已更新")
    
    def get_task_info(self) -> Dict[str, Any]:
        """获取任务信息"""
        return {
            "cur_task": self._cur_task,
            "known_classes": self._known_classes,
            "total_classes": self._total_classes,
            "client_id": self.client_id
        }
