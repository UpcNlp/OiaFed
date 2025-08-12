import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import copy
from typing import List, Dict, Tuple
from diffusers import UNet2DModel, DDPMScheduler
import warnings
warnings.filterwarnings('ignore')


# ===== 数据处理 =====

class ClassIncrementalDataset:
    """类增量数据集管理器"""
    def __init__(self, dataset_name='MNIST', num_clients=5, classes_per_task=2):
        self.dataset_name = dataset_name
        self.num_clients = num_clients
        self.classes_per_task = classes_per_task
        
        # 数据预处理
        if dataset_name == 'MNIST':
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            # 加载MNIST数据集
            self.train_dataset = torchvision.datasets.MNIST(
                root='./data', train=True, download=True, transform=self.transform
            )
            self.test_dataset = torchvision.datasets.MNIST(
                root='./data', train=False, download=True, transform=self.transform
            )
            self.num_classes = 10
            self.input_size = 28 * 28
            self.img_channels = 1
            self.img_size = 28
        
        # 生成类增量任务
        self.tasks = self._create_incremental_tasks()
        
        # 为每个客户端分配数据
        self.client_data = self._distribute_data_to_clients()
        
    def _create_incremental_tasks(self):
        """创建类增量任务"""
        tasks = []
        all_classes = list(range(self.num_classes))
        
        for i in range(0, self.num_classes, self.classes_per_task):
            task_classes = all_classes[i:i + self.classes_per_task]
            tasks.append(task_classes)
            
        return tasks
    
    def _distribute_data_to_clients(self):
        """将数据分配给客户端"""
        client_data = {i: {} for i in range(self.num_clients)}
        
        for task_id, task_classes in enumerate(self.tasks):
            # 获取当前任务的数据索引
            train_indices = []
            test_indices = []
            
            for class_id in task_classes:
                # 训练集
                class_train_indices = [i for i, (_, label) in enumerate(self.train_dataset) 
                                     if label == class_id]
                train_indices.extend(class_train_indices)
                
                # 测试集
                class_test_indices = [i for i, (_, label) in enumerate(self.test_dataset) 
                                    if label == class_id]
                test_indices.extend(class_test_indices)
            
            # 随机打乱
            np.random.shuffle(train_indices)
            np.random.shuffle(test_indices)
            
            # 分配给客户端（Non-IID分布）
            train_split = np.array_split(train_indices, self.num_clients)
            test_split = np.array_split(test_indices, self.num_clients)
            
            for client_id in range(self.num_clients):
                client_data[client_id][task_id] = {
                    'train': train_split[client_id].tolist(),
                    'test': test_split[client_id].tolist(),
                    'classes': task_classes
                }
        
        return client_data
    
    def get_client_dataloader(self, client_id: int, task_id: int, 
                            batch_size: int = 32, mode: str = 'train'):
        """获取客户端数据加载器"""
        indices = self.client_data[client_id][task_id][mode]
        dataset = self.train_dataset if mode == 'train' else self.test_dataset
        subset = Subset(dataset, indices)
        
        return DataLoader(subset, batch_size=batch_size, shuffle=(mode=='train'))
    
    def get_all_test_data(self, task_ids: List[int] = None):
        """获取所有测试数据（用于评估）"""
        if task_ids is None:
            task_ids = list(range(len(self.tasks)))
            
        all_indices = []
        for task_id in task_ids:
            task_classes = self.tasks[task_id]
            for class_id in task_classes:
                class_indices = [i for i, (_, label) in enumerate(self.test_dataset) 
                               if label == class_id]
                all_indices.extend(class_indices)
        
        subset = Subset(self.test_dataset, all_indices)
        return DataLoader(subset, batch_size=128, shuffle=False)


# ===== 客户端模型 =====

class Expert(nn.Module):
    """专家网络"""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, key_dim: int):
        super().__init__()
        self.key_embedding = nn.Parameter(torch.randn(key_dim))
        self.value_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x):
        return self.value_network(x)
    
    def get_key_embedding(self):
        return F.normalize(self.key_embedding, dim=0)


class MoEClassifier(nn.Module):
    """MoE分类器"""
    def __init__(self, input_dim: int = 784, hidden_dim: int = 256, 
                 num_classes: int = 10, key_dim: int = 64, 
                 num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_experts = num_experts
        self.top_k = top_k
        self.key_dim = key_dim
        
        # 特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 专家网络
        self.experts = nn.ModuleList([
            Expert(hidden_dim, hidden_dim, num_classes, key_dim) 
            for _ in range(num_experts)
        ])
        
        # 特征到key空间的投影
        self.feature_to_key = nn.Linear(hidden_dim, key_dim)
        
    def compute_expert_weights(self, features):
        """计算专家权重"""
        projected_features = F.normalize(self.feature_to_key(features), dim=-1)
        
        similarities = []
        for expert in self.experts:
            key = expert.get_key_embedding()
            sim = torch.cosine_similarity(projected_features, key.unsqueeze(0), dim=-1)
            similarities.append(sim)
        
        similarities = torch.stack(similarities, dim=-1)
        
        # Top-k selection
        top_k_values, top_k_indices = torch.topk(similarities, self.top_k, dim=-1)
        weights = torch.zeros_like(similarities)
        weights.scatter_(-1, top_k_indices, F.softmax(top_k_values, dim=-1))
        
        return weights, top_k_indices, similarities
    
    def forward(self, x):
        # 展平输入
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
            
        # 特征提取
        features = self.feature_extractor(x)
        
        # 计算专家权重
        weights, active_experts, all_similarities = self.compute_expert_weights(features)
        
        # 专家输出
        expert_outputs = []
        for expert in self.experts:
            output = expert(features)
            expert_outputs.append(output)
        
        expert_outputs = torch.stack(expert_outputs, dim=1)
        
        # 加权融合
        weights = weights.unsqueeze(-1)
        final_output = torch.sum(expert_outputs * weights, dim=1)
        
        return final_output, weights.squeeze(-1), active_experts, all_similarities
    
    def get_all_key_embeddings(self):
        """获取所有专家的key embeddings"""
        return torch.stack([expert.get_key_embedding() for expert in self.experts])


# ===== 服务端扩散模型 =====

class PretrainedConditionalDiffusionServer:
    """使用预训练条件扩散模型的服务端"""
    def __init__(self, key_dim: int = 64, img_size: int = 28, 
                 model_name: str = "runwayml/stable-diffusion-v1-5"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.key_dim = key_dim
        self.img_size = img_size
        self.model_name = model_name
        
        print(f"加载预训练条件扩散模型: {model_name}")
        
        # 加载预训练的条件UNet
        try:
            from diffusers import UNet2DConditionModel, DDPMScheduler
            self.unet = UNet2DConditionModel.from_pretrained(
                model_name, 
                subfolder="unet",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            ).to(self.device)
            print("✓ 成功加载预训练UNet")
        except Exception as e:
            print(f"警告: 无法加载预训练模型 ({e}), 使用备用方案")
            # 备用方案：使用DDPM的UNet
            self.unet = UNet2DModel.from_pretrained("google/ddpm-cat-256").to(self.device)
            self._use_conditional = False
        else:
            self._use_conditional = True
        
        # 调度器
        self.scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=1e-4,
            beta_end=2e-2,
            beta_schedule="linear"
        )
        
        # 获取模型配置
        if self._use_conditional:
            self.cross_attention_dim = self.unet.config.cross_attention_dim  # 通常是768或1024
            self.unet_channels = self.unet.config.in_channels  # 通常是4 (latent space)
        else:
            self.unet_channels = self.unet.config.in_channels  # 通常是3
        
        # Key embedding适配器
        if self._use_conditional:
            self.key_adapter = nn.Sequential(
                nn.Linear(key_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, self.cross_attention_dim)
            ).to(self.device)
        
        # MNIST适配层 (将1通道转换为预训练模型需要的通道数)
        if self._use_conditional:
            # 对于Stable Diffusion，需要处理latent space (4通道)
            self.mnist_encoder = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((32, 32)),  # 调整尺寸
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 4, 3, padding=1),  # 输出4通道 (latent channels)
                nn.Tanh()
            ).to(self.device)
            
            self.mnist_decoder = nn.Sequential(
                nn.Conv2d(4, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 32, 3, padding=1),
                nn.ReLU(),
                nn.Upsample(size=(28, 28), mode='bilinear', align_corners=False),
                nn.Conv2d(32, 16, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 1, 3, padding=1),
                nn.Tanh()
            ).to(self.device)
        else:
            # 对于DDPM，处理RGB图像
            self.mnist_encoder = nn.Sequential(
                nn.Conv2d(1, 3, 1),  # 1通道转3通道
                nn.Tanh()
            ).to(self.device)
            
            self.mnist_decoder = nn.Sequential(
                nn.Conv2d(3, 1, 1),  # 3通道转1通道
                nn.Tanh()
            ).to(self.device)
        
        # 冻结预训练参数
        self.freeze_pretrained_weights()
        
    def freeze_pretrained_weights(self):
        """冻结预训练模型权重"""
        for param in self.unet.parameters():
            param.requires_grad = False
        print("✓ 已冻结预训练UNet参数")
        
    def unfreeze_pretrained_weights(self):
        """解冻预训练模型权重"""
        for param in self.unet.parameters():
            param.requires_grad = True
        print("✓ 已解冻预训练UNet参数")
    
    def encode_mnist_to_latent(self, mnist_images):
        """将MNIST图像编码到潜在空间"""
        return self.mnist_encoder(mnist_images)
    
    def decode_latent_to_mnist(self, latent):
        """将潜在表示解码回MNIST图像"""
        return self.mnist_decoder(latent)
    
    def prepare_condition(self, key_embeddings, batch_size, sequence_length=77):
        """准备条件信息"""
        if not self._use_conditional:
            return None
            
        # 将key embedding转换为cross attention条件
        condition = self.key_adapter(key_embeddings)  # [batch_size, cross_attention_dim]
        
        # 扩展到序列长度 (CLIP text encoder通常输出77个token)
        condition = condition.unsqueeze(1).repeat(1, sequence_length, 1)
        
        return condition
    
    def add_noise(self, original_samples, noise, timesteps):
        """添加噪声"""
        return self.scheduler.add_noise(original_samples, noise, timesteps)
    
    def predict_noise(self, noisy_samples, timesteps, key_embeddings):
        """预测噪声"""
        batch_size = noisy_samples.size(0)
        
        if self._use_conditional:
            # 条件模型
            condition = self.prepare_condition(key_embeddings, batch_size)
            
            noise_pred = self.unet(
                noisy_samples,
                timesteps,
                encoder_hidden_states=condition,
                return_dict=False
            )[0]
        else:
            # 无条件模型
            noise_pred = self.unet(noisy_samples, timesteps).sample
        
        return noise_pred
    
    def generate_samples(self, key_embeddings, num_samples: int = 16, 
                        num_inference_steps: int = 20, guidance_scale: float = 7.5):
        """生成样本"""
        self.unet.eval()
        device = self.device
        
        # 确保key_embeddings格式正确
        if key_embeddings.dim() == 1:
            key_embeddings = key_embeddings.unsqueeze(0)
        if key_embeddings.size(0) == 1:
            key_embeddings = key_embeddings.repeat(num_samples, 1)
        
        with torch.no_grad():
            # 初始化噪声
            if self._use_conditional:
                # latent space通常是原图尺寸的1/8
                latent_size = 32  # 对应256x256图像的latent size
                shape = (num_samples, 4, latent_size, latent_size)
            else:
                shape = (num_samples, 3, 256, 256)  # DDPM cat model的尺寸
            
            sample = torch.randn(shape, device=device, dtype=self.unet.dtype)
            
            # 设置推理步骤
            self.scheduler.set_timesteps(num_inference_steps)
            
            # 条件引导（如果使用条件模型）
            if self._use_conditional and guidance_scale > 1.0:
                # 无条件key embedding (空向量)
                null_key_embeddings = torch.zeros_like(key_embeddings)
                
                # 合并条件和无条件
                combined_key_embeddings = torch.cat([null_key_embeddings, key_embeddings])
                sample = torch.cat([sample, sample])
            
            # 去噪过程
            for i, t in enumerate(self.scheduler.timesteps):
                timestep_tensor = t.unsqueeze(0).repeat(sample.size(0)).to(device)
                
                if self._use_conditional and guidance_scale > 1.0:
                    # 条件引导
                    noise_pred = self.predict_noise(sample, timestep_tensor, combined_key_embeddings)
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    sample = sample.chunk(2)[1]  # 只保留条件分支
                else:
                    # 常规预测
                    noise_pred = self.predict_noise(sample, timestep_tensor, key_embeddings)
                
                # 去噪步骤
                sample = self.scheduler.step(noise_pred, t, sample).prev_sample
            
            # 解码到MNIST空间
            if self._use_conditional:
                # 从latent space解码
                mnist_images = self.decode_latent_to_mnist(sample)
                # 调整尺寸到28x28
                mnist_images = F.interpolate(mnist_images, size=(28, 28), mode='bilinear', align_corners=False)
            else:
                # 直接解码
                mnist_images = self.decode_latent_to_mnist(sample)
                mnist_images = F.interpolate(mnist_images, size=(28, 28), mode='bilinear', align_corners=False)
            
            # 后处理
            mnist_images = (mnist_images + 1.0) / 2.0  # 归一化到[0,1]
            mnist_images = mnist_images.clamp(0, 1)
            
            return mnist_images
    
    def get_trainable_parameters(self):
        """获取可训练参数"""
        trainable_params = []
        
        # 适配器参数
        if self._use_conditional:
            trainable_params.extend(list(self.key_adapter.parameters()))
        
        # MNIST编码解码器参数
        trainable_params.extend(list(self.mnist_encoder.parameters()))
        trainable_params.extend(list(self.mnist_decoder.parameters()))
        
        return trainable_params


# ===== 联邦学习训练器 =====

class PseudoFederatedTrainer:
    """伪联邦训练器"""
    def __init__(self, dataset_manager, num_clients=5, device=None):
        self.dataset_manager = dataset_manager
        self.num_clients = num_clients
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 客户端模型
        self.client_models = {}
        self.client_optimizers = {}
        
        # 服务端模型 - 使用预训练条件扩散模型
        self.server_diffusion = PretrainedConditionalDiffusionServer(
            key_dim=64, 
            img_size=dataset_manager.img_size
        )
        
        # 优化器只优化可训练参数（适配器和编码解码器）
        trainable_params = self.server_diffusion.get_trainable_parameters()
        self.server_optimizer = torch.optim.AdamW(
            trainable_params,
            lr=1e-4, weight_decay=1e-5
        )
        
        print(f"可训练参数数量: {sum(p.numel() for p in trainable_params):,}")
        print(f"UNet总参数数量: {sum(p.numel() for p in self.server_diffusion.unet.parameters()):,}")
        
        # 全局模型（用于聚合）
        self.global_model = None
        
        # 历史记录
        self.history = {
            'client_losses': defaultdict(list),
            'server_losses': [],
            'task_accuracies': defaultdict(list),
            'forgetting_measures': []
        }
        
    def initialize_clients(self):
        """初始化客户端模型"""
        for client_id in range(self.num_clients):
            model = MoEClassifier(
                input_dim=self.dataset_manager.input_size,
                hidden_dim=256,
                num_classes=self.dataset_manager.num_classes,
                key_dim=64,
                num_experts=8,
                top_k=2
            ).to(self.device)
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            
            self.client_models[client_id] = model
            self.client_optimizers[client_id] = optimizer
            
        # 初始化全局模型
        self.global_model = MoEClassifier(
            input_dim=self.dataset_manager.input_size,
            hidden_dim=256,
            num_classes=self.dataset_manager.num_classes,
            key_dim=64,
            num_experts=8,
            top_k=2
        ).to(self.device)
    
    def client_train_step(self, client_id, task_id, replay_data=None, replay_weight=0.3):
        """客户端训练步骤"""
        model = self.client_models[client_id]
        optimizer = self.client_optimizers[client_id]
        
        # 获取数据
        dataloader = self.dataset_manager.get_client_dataloader(
            client_id, task_id, batch_size=32, mode='train'
        )
        
        model.train()
        epoch_losses = []
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            
            # 前向传播
            output, weights, active_experts, similarities = model(data)
            
            # 主要损失
            main_loss = F.cross_entropy(output, target)
            total_loss = main_loss
            
            # 回放数据损失
            if replay_data is not None:
                replay_images, replay_labels = replay_data
                if len(replay_images) > 0:
                    replay_output, _, _, _ = model(replay_images)
                    replay_loss = F.cross_entropy(replay_output, replay_labels)
                    total_loss = main_loss + replay_weight * replay_loss
            
            # 专家多样性正则化
            diversity_loss = self.compute_diversity_loss(weights)
            total_loss = total_loss + 0.01 * diversity_loss
            
            # 反向传播
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_losses.append(total_loss.item())
            
            if batch_idx >= 10:  # 限制每个epoch的batch数量
                break
        
        avg_loss = np.mean(epoch_losses)
        self.history['client_losses'][client_id].append(avg_loss)
        
        return avg_loss
    
    def compute_diversity_loss(self, weights):
        """计算专家多样性损失"""
        expert_usage = weights.mean(dim=0)
        uniform_dist = torch.ones_like(expert_usage) / len(expert_usage)
        diversity_loss = F.kl_div(expert_usage.log(), uniform_dist, reduction='batchmean')
        return diversity_loss
    
    def server_train_step(self, task_id):
        """服务端训练步骤"""
        # 设置训练模式
        if hasattr(self.server_diffusion, 'key_adapter'):
            self.server_diffusion.key_adapter.train()
        self.server_diffusion.mnist_encoder.train()
        self.server_diffusion.mnist_decoder.train()
        
        # 收集所有客户端的key embeddings和数据
        all_key_embeddings = []
        all_images = []
        
        for client_id in range(self.num_clients):
            # 获取key embeddings
            key_embeddings = self.client_models[client_id].get_all_key_embeddings()
            
            # 获取少量真实图像用于训练扩散模型
            dataloader = self.dataset_manager.get_client_dataloader(
                client_id, task_id, batch_size=4, mode='train'  # 减少batch size
            )
            
            for data, _ in dataloader:
                data = data.to(self.device)
                # 重复key embeddings以匹配batch size
                batch_keys = key_embeddings[:data.size(0)]
                
                all_key_embeddings.append(batch_keys)
                all_images.append(data)
                break  # 只取一个batch
        
        if not all_key_embeddings:
            return 0.0
            
        # 合并数据
        key_embeddings = torch.cat(all_key_embeddings, dim=0)
        images = torch.cat(all_images, dim=0)
        
        # 训练扩散模型
        self.server_optimizer.zero_grad()
        
        batch_size = images.size(0)
        device = images.device
        
        try:
            # 将MNIST图像编码到扩散模型的空间
            encoded_images = self.server_diffusion.encode_mnist_to_latent(images)
            
            # 随机时间步
            timesteps = torch.randint(
                0, self.server_diffusion.scheduler.config.num_train_timesteps,
                (batch_size,), device=device
            )
            
            # 添加噪声
            noise = torch.randn_like(encoded_images)
            noisy_images = self.server_diffusion.add_noise(encoded_images, noise, timesteps)
            
            # 预测噪声
            noise_pred = self.server_diffusion.predict_noise(noisy_images, timesteps, key_embeddings)
            
            # 计算损失
            loss = F.mse_loss(noise_pred, noise)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.server_diffusion.get_trainable_parameters(),
                max_norm=1.0
            )
            self.server_optimizer.step()
            
            self.history['server_losses'].append(loss.item())
            return loss.item()
            
        except Exception as e:
            print(f"服务端训练错误: {e}")
            return 0.0
    
    def generate_replay_data(self, client_id, task_id, num_samples=16):
        """生成回放数据"""
        model = self.client_models[client_id]
        key_embeddings = model.get_all_key_embeddings()
        
        try:
            # 生成图像
            generated_images = self.server_diffusion.generate_samples(
                key_embeddings[0],  # 使用第一个专家的key
                num_samples=num_samples,
                num_inference_steps=10,  # 减少推理步骤以加速
                guidance_scale=7.5 if hasattr(self.server_diffusion, '_use_conditional') 
                              and self.server_diffusion._use_conditional else 1.0
            )
            
            # 生成对应的标签（使用当前任务的类别）
            task_classes = self.dataset_manager.tasks[task_id]
            labels = torch.randint(0, len(task_classes), (num_samples,)).to(self.device)
            # 映射到实际类别标签
            labels = torch.tensor([task_classes[l] for l in labels]).to(self.device)
            
            # 应用数据集的预处理 (归一化)
            generated_images = generated_images * 2.0 - 1.0  # 归一化到[-1,1]
            
            return generated_images, labels
            
        except Exception as e:
            print(f"生成回放数据错误: {e}")
            # 返回空数据
            empty_images = torch.empty(0, 1, 28, 28).to(self.device)
            empty_labels = torch.empty(0, dtype=torch.long).to(self.device)
            return empty_images, empty_labels
    
    def federated_averaging(self):
        """联邦平均"""
        global_dict = self.global_model.state_dict()
        
        for key in global_dict.keys():
            global_dict[key] = torch.stack([
                self.client_models[client_id].state_dict()[key].float()
                for client_id in range(self.num_clients)
            ]).mean(dim=0)
        
        self.global_model.load_state_dict(global_dict)
        
        # 将全局模型参数分发给客户端
        for client_id in range(self.num_clients):
            self.client_models[client_id].load_state_dict(global_dict)
    
    def evaluate_task(self, task_id):
        """评估特定任务"""
        self.global_model.eval()
        
        dataloader = self.dataset_manager.get_all_test_data([task_id])
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output, _, _, _ = self.global_model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        accuracy = 100. * correct / total
        return accuracy
    
    def evaluate_all_tasks(self, current_task):
        """评估所有已学习的任务"""
        accuracies = {}
        for task_id in range(current_task + 1):
            acc = self.evaluate_task(task_id)
            accuracies[task_id] = acc
            self.history['task_accuracies'][task_id].append(acc)
        
        return accuracies
    
    def compute_forgetting(self, current_task):
        """计算遗忘度量"""
        if current_task == 0:
            return 0.0
            
        forgetting = 0.0
        count = 0
        
        for task_id in range(current_task):
            task_accs = self.history['task_accuracies'][task_id]
            if len(task_accs) > 1:
                max_acc = max(task_accs[:-1])  # 之前的最大准确率
                current_acc = task_accs[-1]    # 当前准确率
                forgetting += max(0, max_acc - current_acc)
                count += 1
        
        avg_forgetting = forgetting / count if count > 0 else 0.0
        self.history['forgetting_measures'].append(avg_forgetting)
        
        return avg_forgetting
    
    def train_incremental(self, num_rounds_per_task=5, local_epochs=1):
        """类增量训练主流程"""
        print("=== 开始伪联邦类增量学习 ===")
        print(f"数据集: {self.dataset_manager.dataset_name}")
        print(f"客户端数量: {self.num_clients}")
        print(f"任务数量: {len(self.dataset_manager.tasks)}")
        print(f"每个任务的类别: {self.dataset_manager.tasks}")
        
        self.initialize_clients()
        
        for task_id, task_classes in enumerate(self.dataset_manager.tasks):
            print(f"\n--- 任务 {task_id}: 类别 {task_classes} ---")
            
            for round_num in range(num_rounds_per_task):
                print(f"Round {round_num + 1}/{num_rounds_per_task}")
                
                # 客户端训练
                client_losses = []
                for client_id in range(self.num_clients):
                    # 生成回放数据（如果不是第一个任务）
                    replay_data = None
                    if task_id > 0:
                        replay_images, replay_labels = self.generate_replay_data(
                            client_id, task_id, num_samples=16
                        )
                        replay_data = (replay_images, replay_labels)
                    
                    # 本地训练
                    for epoch in range(local_epochs):
                        loss = self.client_train_step(client_id, task_id, replay_data)
                        client_losses.append(loss)
                
                # 联邦平均
                self.federated_averaging()
                
                # 服务端训练
                server_loss = self.server_train_step(task_id)
                
                print(f"  客户端平均损失: {np.mean(client_losses):.4f}")
                print(f"  服务端损失: {server_loss:.4f}")
            
            # 任务完成后评估
            print("\n--- 评估结果 ---")
            accuracies = self.evaluate_all_tasks(task_id)
            forgetting = self.compute_forgetting(task_id)
            
            for tid, acc in accuracies.items():
                print(f"  任务 {tid} 准确率: {acc:.2f}%")
            print(f"  平均遗忘度: {forgetting:.2f}%")
        
        print("\n=== 训练完成 ===")
        return self.history
    
    def visualize_results(self):
        """可视化结果"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 客户端损失
        axes[0, 0].set_title('Client Training Losses')
        for client_id in range(self.num_clients):
            axes[0, 0].plot(self.history['client_losses'][client_id], 
                          label=f'Client {client_id}')
        axes[0, 0].set_xlabel('Round')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 服务端损失
        axes[0, 1].set_title('Server Diffusion Loss')
        axes[0, 1].plot(self.history['server_losses'])
        axes[0, 1].set_xlabel('Round')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True)
        
        # 任务准确率
        axes[1, 0].set_title('Task Accuracies')
        for task_id in range(len(self.dataset_manager.tasks)):
            if task_id in self.history['task_accuracies']:
                axes[1, 0].plot(self.history['task_accuracies'][task_id], 
                              label=f'Task {task_id}')
        axes[1, 0].set_xlabel('Evaluation Round')
        axes[1, 0].set_ylabel('Accuracy (%)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 遗忘度量
        axes[1, 1].set_title('Forgetting Measure')
        axes[1, 1].plot(self.history['forgetting_measures'])
        axes[1, 1].set_xlabel('Task')
        axes[1, 1].set_ylabel('Average Forgetting (%)')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def generate_and_visualize_samples(self, num_samples=8):
        """生成并可视化样本"""
        # 获取第一个客户端的key embeddings
        key_embeddings = self.client_models[0].get_all_key_embeddings()
        
        try:
            # 生成样本
            generated_images = self.server_diffusion.generate_samples(
                key_embeddings[0], 
                num_samples=num_samples,
                num_inference_steps=20,
                guidance_scale=7.5 if hasattr(self.server_diffusion, '_use_conditional') 
                              and self.server_diffusion._use_conditional else 1.0
            )
            
            # 可视化
            fig, axes = plt.subplots(2, 4, figsize=(12, 6))
            axes = axes.flatten()
            
            for i in range(min(num_samples, 8)):
                img = generated_images[i].cpu().squeeze().numpy()
                axes[i].imshow(img, cmap='gray')
                axes[i].axis('off')
                axes[i].set_title(f'Generated {i+1}')
            
            plt.suptitle('Generated Replay Samples (Pretrained Conditional Diffusion)')
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"可视化生成样本错误: {e}")
            print("跳过样本可视化...")


# ===== 主函数 =====

def main():
    """主函数"""
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("=== 伪联邦类增量学习 - 预训练条件扩散模型 ===")
    
    # 创建数据集管理器
    dataset_manager = ClassIncrementalDataset(
        dataset_name='MNIST',
        num_clients=3,  # 减少客户端数量以加速实验
        classes_per_task=2
    )
    
    print("数据集信息:")
    print(f"总类别数: {dataset_manager.num_classes}")
    print(f"任务划分: {dataset_manager.tasks}")
    
    # 创建训练器
    trainer = PseudoFederatedTrainer(
        dataset_manager=dataset_manager,
        num_clients=3
    )
    
    print("\n模型架构:")
    print("✓ 客户端: MoE分类器 (8个专家)")
    print("✓ 服务端: 预训练条件扩散模型 + 适配器")
    print("✓ 联邦设置: 参数聚合 + 回放数据生成")
    
    try:
        # 开始训练
        history = trainer.train_incremental(
            num_rounds_per_task=2,  # 减少轮数以加速实验
            local_epochs=1
        )
        
        # 可视化结果
        trainer.visualize_results()
        
        # 生成并可视化样本
        trainer.generate_and_visualize_samples()
        
        return trainer, history
        
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        print("这可能是由于GPU内存不足或网络连接问题导致的。")
        print("建议:")
        print("1. 减少batch_size")
        print("2. 使用CPU模式")
        print("3. 检查网络连接以下载预训练模型")
        return None, None


if __name__ == "__main__":
    trainer, history = main()