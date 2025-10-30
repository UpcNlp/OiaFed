import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
import torchvision
import torchvision.transforms as transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
import numpy as np
from collections import defaultdict
import argparse
import sys
import copy
from pathlib import Path
from loguru import logger
from tqdm import tqdm

# ========== 日志配置 ==========

def setup_logger(log_level="INFO", log_file=None):
    """配置loguru日志系统（不使用emoji）"""
    logger.remove()
    
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=log_level,
        colorize=True
    )
    
    if log_file:
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
            level=log_level,
            rotation="100 MB"
        )
        logger.info(f"Logging to file: {log_file}")
    
    return logger

# ========== 模型定义 ==========

class TaskSpecificClassifier(nn.Module):
    """任务特定的分类头（Stage 1使用，用完即弃）"""
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        return self.classifier(x)

class ReconstructionModel(nn.Module):
    """重建模型：从分类头输出重建VIT特征"""
    def __init__(self, classifier_output_dim, vit_feature_dim, hidden_dim=512):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(classifier_output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim * 2, vit_feature_dim)
        )
        
    def forward(self, classifier_output):
        return self.model(classifier_output)

class SharedClassifier(nn.Module):
    """共享的最终分类头（所有任务共用）"""
    def __init__(self, input_dim, total_classes):
        super().__init__()
        self.classifier = nn.Linear(input_dim, total_classes)
        
    def forward(self, x):
        return self.classifier(x)

# ========== 持续学习系统 ==========

class ContinualLearningSystem:
    def __init__(self, device='cuda', total_classes=10, hidden_dim=512):
        """
        初始化持续学习系统
        
        Args:
            device: 训练设备 ('cuda' or 'cpu')
            total_classes: 总类别数量（MNIST=10, CIFAR-10=10, CIFAR-100=100）
            hidden_dim: 重建模型的隐藏层维度
        """
        self.device = device
        self.total_classes = total_classes
        self.hidden_dim = hidden_dim
        
        logger.info(f"Initializing Continual Learning System")
        logger.info(f"Total classes: {total_classes}")
        logger.info(f"Device: {device}")
        logger.info(f"Hidden dimension: {hidden_dim}")
        
        # 初始化预训练VIT（冻结）
        logger.info("Loading pretrained ViT-B/16 model...")
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.vit.heads = nn.Identity()
        
        # 冻结VIT
        for param in self.vit.parameters():
            param.requires_grad = False
        self.vit.eval()
        self.vit.to(device)
        logger.info("ViT model loaded and frozen")
        
        # VIT特征维度
        self.vit_feature_dim = 768
        
        # 初始化共享分类头
        self.shared_classifier = SharedClassifier(
            self.vit_feature_dim, total_classes
        ).to(device)
        
        # 存储各阶段的组件
        self.task_classifiers = {}
        self.reconstruction_models = {}
        
        # 回放缓冲区
        self.replay_buffer = []
        
        self.current_task = 0
        
    def extract_vit_features(self, x):
        """提取VIT特征（冻结，不参与训练）"""
        with torch.no_grad():
            features = self.vit(x)
            return features
    
    def extract_task_features(self, train_loader, task_id):
        """一次性提取整个任务的VIT特征（优化训练速度）"""
        logger.info("="*60)
        logger.info(f"Extracting VIT features for Task {task_id}")
        logger.info("="*60)
        
        self.vit.eval()
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(train_loader, desc=f"Extracting features Task {task_id}"):
                images = images.to(self.device)
                features = self.extract_vit_features(images)
                
                all_features.append(features.cpu())
                all_labels.append(labels)
        
        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        logger.success(f"Extracted {len(all_features)} features with dimension {all_features.size(1)}")
        logger.info(f"Feature extraction completed for Task {task_id}")
        
        return all_features, all_labels
    
    def stage1_train_task_classifier(self, features, labels, task_id, classes_in_task, 
                                     epochs=10, lr=0.001, batch_size=128):
        """Stage 1: 训练任务特定的分类头（使用预提取的特征）"""
        logger.info("="*60)
        logger.info(f"Stage 1: Training task-specific classifier for Task {task_id}")
        logger.info(f"Classes in this task: {classes_in_task}")
        logger.info("="*60)
        
        num_classes_in_task = len(classes_in_task)
        
        task_classifier = TaskSpecificClassifier(
            self.vit_feature_dim, num_classes_in_task
        ).to(self.device)
        
        optimizer = optim.Adam(task_classifier.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        # 创建数据集和dataloader（特征会在训练时移到设备上）
        dataset = TensorDataset(features, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        epoch_pbar = tqdm(range(epochs), desc=f"Stage 1 Task {task_id}")
        
        for epoch in epoch_pbar:
            task_classifier.train()
            total_loss = 0
            correct = 0
            total = 0
            
            batch_pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
            
            for batch_features, batch_labels in batch_pbar:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                # 将全局标签转换为任务内标签
                task_labels = torch.zeros_like(batch_labels)
                for i, cls in enumerate(classes_in_task):
                    task_labels[batch_labels == cls] = i
                
                outputs = task_classifier(batch_features)
                loss = criterion(outputs, task_labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += task_labels.size(0)
                correct += predicted.eq(task_labels).sum().item()
                
                batch_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
            
            acc = 100. * correct / total
            avg_loss = total_loss / len(dataloader)
            epoch_pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'acc': f'{acc:.2f}%'
            })
            
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Acc: {acc:.2f}%")
        
        self.task_classifiers[task_id] = task_classifier
        logger.success(f"Stage 1 completed for Task {task_id}")
        
        return task_classifier
    
    def stage2_train_reconstruction(self, features, labels, task_id, classes_in_task, 
                                    epochs=20, lr=0.001, samples_per_class=100,
                                    mse_weight=1.0, cosine_weight=0.1, cls_weight=0.5,
                                    batch_size=128):
        """Stage 2: 为整个任务训练一个重建网络（带分类一致性损失），并在结束后生成回放数据"""
        logger.info("="*60)
        logger.info(f"Stage 2: Training reconstruction model for Task {task_id}")
        logger.info(f"Classes in this task: {classes_in_task}")
        logger.info(f"Loss weights - MSE: {mse_weight}, Cosine: {cosine_weight}, Classification: {cls_weight}")
        logger.info("="*60)
        
        task_classifier = self.task_classifiers[task_id]
        task_classifier.eval()  # 冻结分类器
        
        num_classes_in_task = len(classes_in_task)
        
        # 使用预提取的特征，直接通过分类器获取输出
        logger.info("Generating classifier outputs for all features...")
        all_outputs = []
        
        # 将特征移到设备上（创建副本，不修改原始features）
        device_features = features.to(self.device)
        dataset = TensorDataset(device_features)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        with torch.no_grad():
            for (batch_features,) in tqdm(dataloader, desc="Generating outputs", leave=False):
                classifier_outputs = task_classifier(batch_features)
                all_outputs.append(classifier_outputs)
        
        all_outputs = torch.cat(all_outputs, dim=0)
        all_features = device_features  # 已经在GPU上
        
        logger.info(f"Collected {len(all_outputs)} samples for Task {task_id}")
        
        # 为整个任务训练一个重建网络
        recon_model = ReconstructionModel(
            num_classes_in_task, self.vit_feature_dim, self.hidden_dim
        ).to(self.device)
        
        optimizer = optim.Adam(recon_model.parameters(), lr=lr, weight_decay=1e-4)
        mse_criterion = nn.MSELoss()
        cosine_criterion = nn.CosineEmbeddingLoss()
        cls_criterion = nn.MSELoss()  # 分类一致性损失
        
        dataset = TensorDataset(all_outputs, all_features)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        best_loss = float('inf')
        
        logger.info(f"Training reconstruction model with {len(dataset)} samples...")
        logger.info(f"Using 3-way loss: Feature MSE + Cosine Similarity + Classification Consistency")
        
        epoch_pbar = tqdm(range(epochs), desc=f"Stage 2 Task {task_id}")
        
        for epoch in epoch_pbar:
            recon_model.train()
            
            # 统计各部分损失和准确率
            total_loss = 0
            total_mse_loss = 0
            total_cosine_loss = 0
            total_cls_loss = 0
            num_batches = 0
            correct_recon = 0
            total_samples = 0
            
            batch_pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
            
            for batch_outputs, batch_features in batch_pbar:
                # 重建特征
                reconstructed = recon_model(batch_outputs)
                
                # 1. 特征MSE损失
                mse_loss = mse_criterion(reconstructed, batch_features)
                
                # 2. 余弦相似度损失
                target = torch.ones(batch_features.size(0)).to(self.device)
                cosine_loss = cosine_criterion(reconstructed, batch_features, target)
                
                # 3. 分类一致性损失（冻结分类器）
                with torch.no_grad():
                    # 原始特征通过分类器的输出（作为目标）
                    original_cls_output = task_classifier(batch_features)
                
                # 重建特征通过分类器的输出（需要优化）
                reconstructed_cls_output = task_classifier(reconstructed)
                
                # 确保重建特征的分类输出接近原始特征的分类输出
                cls_loss = cls_criterion(reconstructed_cls_output, original_cls_output)
                
                # 总损失（加权组合）
                loss = (mse_weight * mse_loss + 
                       cosine_weight * cosine_loss + 
                       cls_weight * cls_loss)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # 累计损失
                total_loss += loss.item()
                total_mse_loss += mse_loss.item()
                total_cosine_loss += cosine_loss.item()
                total_cls_loss += cls_loss.item()
                num_batches += 1
                
                # 计算分类准确率
                with torch.no_grad():
                    # 获取真实标签（从classifier输出推断）
                    _, original_pred = original_cls_output.max(1)
                    _, recon_pred = reconstructed_cls_output.max(1)
                    
                    # 计算准确率（重建特征的分类准确率）
                    correct_recon += recon_pred.eq(original_pred).sum().item()
                    total_samples += batch_features.size(0)
                
                # 计算当前批次准确率
                batch_acc = 100. * recon_pred.eq(original_pred).sum().item() / batch_features.size(0)
                
                # 更新batch进度条
                batch_pbar.set_postfix({
                    'mse': f'{mse_loss.item():.4f}',
                    'cos': f'{cosine_loss.item():.4f}',
                    'cls': f'{cls_loss.item():.4f}',
                    'acc': f'{batch_acc:.1f}%',
                    'total': f'{loss.item():.4f}'
                })
            
            # 计算平均损失和准确率
            acc = 100. * correct_recon / total_samples
            avg_loss = total_loss / num_batches
            avg_mse = total_mse_loss / num_batches
            avg_cosine = total_cosine_loss / num_batches
            avg_cls = total_cls_loss / num_batches
            
            if avg_loss < best_loss:
                best_loss = avg_loss
            
            # 更新epoch进度条
            epoch_pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'mse': f'{avg_mse:.4f}',
                'cos': f'{avg_cosine:.4f}',
                'cls': f'{avg_cls:.4f}',
                'acc': f'{acc:.1f}%',
                'best': f'{best_loss:.4f}'
            })
            
            # 定期打印详细日志
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                logger.info(f"Epoch {epoch+1}/{epochs} - "
                          f"Total: {avg_loss:.4f}, "
                          f"MSE: {avg_mse:.4f}, "
                          f"Cosine: {avg_cosine:.4f}, "
                          f"Classification: {avg_cls:.4f}, "
                          f"Accuracy: {acc:.2f}%, "
                          f"Best: {best_loss:.4f}")
        
        self.reconstruction_models[task_id] = recon_model
        logger.success(f"Stage 2 reconstruction training completed for Task {task_id}")
        logger.info(f"Best total loss: {best_loss:.4f}")
        
        # Stage 2结束后立即生成回放数据
        logger.info("="*60)
        logger.info(f"Generating replay data for Task {task_id}")
        logger.info("="*60)
        self._generate_replay_data(task_id, classes_in_task, samples_per_class)
    
    def _generate_replay_data(self, task_id, classes_in_task, samples_per_class=100):
        """生成并保存回放数据（在Stage 2结束时调用）"""
        recon_model = self.reconstruction_models[task_id]
        recon_model.eval()
        
        all_replay_features = []
        all_replay_labels = []
        
        num_classes_in_task = len(classes_in_task)
        
        logger.info(f"Generating {samples_per_class} samples per class using single reconstruction model...")
        
        for class_id in tqdm(classes_in_task, desc="Generating replay", leave=False):
            class_idx = classes_in_task.index(class_id)
            
            with torch.no_grad():
                fake_outputs = torch.zeros(samples_per_class, num_classes_in_task).to(self.device)
                fake_outputs[:, class_idx] = 10.0
                
                reconstructed_features = recon_model(fake_outputs)
            
            all_replay_features.append(reconstructed_features.cpu())
            labels = torch.full((samples_per_class,), class_id, dtype=torch.long)
            all_replay_labels.append(labels)
            
            logger.info(f"Generated {samples_per_class} samples for class {class_id}")
        
        if all_replay_features:
            replay_features = torch.cat(all_replay_features, dim=0)
            replay_labels = torch.cat(all_replay_labels, dim=0)
            
            replay_buffer_entry = {
                'task_id': task_id,
                'features': replay_features.to(self.device),
                'labels': replay_labels.to(self.device),
                'classes': classes_in_task
            }
            
            self.replay_buffer.append(replay_buffer_entry)
            
            logger.success(f"Replay data saved: {len(replay_features)} samples, {replay_features.size(1)} features")
            logger.info(f"Current replay buffer size: {len(self.replay_buffer)} tasks")
        else:
            logger.warning(f"No replay data generated for Task {task_id}")
    
    def stage3_train_shared_classifier(self, current_features, current_labels, task_id, 
                                       classes_in_task, epochs=20, lr=0.001,
                                       batch_size=128, current_weight=1.5):
        """Stage 3: 训练共享分类头（使用预提取的特征+历史回放数据）"""
        logger.info("="*60)
        logger.info(f"Stage 3: Training shared classifier after Task {task_id}")
        logger.info(f"Total classes seen so far: {sum([len(buf['classes']) for buf in self.replay_buffer])}")
        logger.info("="*60)
        
        # 保存旧模型用于知识蒸馏
        if task_id > 0:
            old_classifier = copy.deepcopy(self.shared_classifier)
            old_classifier.eval()
        else:
            old_classifier = None
        
        # 将当前特征移到设备上
        current_features = current_features.to(self.device)
        current_labels = current_labels.to(self.device)
        
        # 准备混合数据集
        if len(self.replay_buffer) > 0:
            logger.info("Mixing current features with historical replay data...")
            all_replay_features = []
            all_replay_labels = []
            
            for buffer in self.replay_buffer:
                all_replay_features.append(buffer['features'])
                all_replay_labels.append(buffer['labels'])
                logger.info(f"Task {buffer['task_id']}: {len(buffer['features'])} replay samples")
            
            replay_features = torch.cat(all_replay_features, dim=0)
            replay_labels = torch.cat(all_replay_labels, dim=0)
            
            logger.info(f"Total historical replay: {len(replay_features)} samples")
            
            # 计算当前任务数据的重复次数
            repeat_times = max(1, int(len(replay_features) * current_weight / len(current_features)))
            logger.info(f"Repeating current task data {repeat_times}x to balance with replay data")
            
            # 重复当前任务数据
            repeated_current_features = current_features.repeat(repeat_times, 1)
            repeated_current_labels = current_labels.repeat(repeat_times)
            
            # 合并数据（现在都在同一设备上）
            mixed_features = torch.cat([repeated_current_features, replay_features], dim=0)
            mixed_labels = torch.cat([repeated_current_labels, replay_labels], dim=0)
            
            # 标记哪些是当前任务数据（用于加权损失）
            is_current = torch.cat([
                torch.ones(len(repeated_current_features), dtype=torch.bool),
                torch.zeros(len(replay_features), dtype=torch.bool)
            ])
        else:
            logger.info("No historical data (first task)")
            mixed_features = current_features
            mixed_labels = current_labels
            is_current = torch.ones(len(current_features), dtype=torch.bool)
        
        # 创建数据集
        dataset = TensorDataset(mixed_features, mixed_labels, is_current)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        logger.info(f"Mixed dataset size: {len(mixed_features)} samples")
        logger.info(f"Current weight multiplier: {current_weight}x")
        
        # 开始训练
        optimizer = optim.Adam(self.shared_classifier.parameters(), lr=lr, weight_decay=1e-4)
        ce_criterion = nn.CrossEntropyLoss(reduction='none')
        kd_criterion = nn.KLDivLoss(reduction='batchmean')
        
        epoch_pbar = tqdm(range(epochs), desc=f"Stage 3 Task {task_id}")
        
        for epoch in epoch_pbar:
            self.shared_classifier.train()
            total_loss = 0
            total_ce_loss = 0
            total_kd_loss = 0
            correct = 0
            total = 0
            current_correct = 0
            current_total = 0
            
            batch_pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
            
            for batch_features, batch_labels, batch_is_current in batch_pbar:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                batch_is_current = batch_is_current.to(self.device)
                
                # 前向传播
                outputs = self.shared_classifier(batch_features)
                
                # 交叉熵损失（带权重）
                ce_loss = ce_criterion(outputs, batch_labels)
                
                # 加权loss：当前任务权重更高
                weighted_loss = torch.where(
                    batch_is_current,
                    ce_loss * current_weight,
                    ce_loss
                )
                ce_loss_value = weighted_loss.mean()
                
                # 知识蒸馏损失
                kd_loss_value = 0.0
                if old_classifier is not None:
                    with torch.no_grad():
                        old_outputs = old_classifier(batch_features)
                    
                    T = 2.0
                    kd_loss_value = kd_criterion(
                        F.log_softmax(outputs / T, dim=1),
                        F.softmax(old_outputs / T, dim=1)
                    ) * (T * T)
                    
                    loss = ce_loss_value + 0.3 * kd_loss_value
                    total_kd_loss += kd_loss_value.item()
                else:
                    loss = ce_loss_value
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_ce_loss += ce_loss_value.item()
                _, predicted = outputs.max(1)
                total += batch_labels.size(0)
                correct += predicted.eq(batch_labels).sum().item()
                
                # 统计当前任务准确率
                current_mask = batch_is_current.bool()
                if current_mask.any():
                    current_total += current_mask.sum().item()
                    current_correct += predicted[current_mask].eq(batch_labels[current_mask]).sum().item()
                
                # 更新进度条
                current_acc = 100. * current_correct / max(current_total, 1)
                batch_info = {
                    'loss': f'{loss.item():.3f}',
                    'ce': f'{ce_loss_value.item():.3f}',
                    'acc': f'{100.*correct/total:.1f}%',
                    'cur': f'{current_acc:.1f}%'
                }
                if old_classifier is not None:
                    batch_info['kd'] = f'{kd_loss_value.item():.3f}'
                batch_pbar.set_postfix(batch_info)
            
            # 计算平均损失
            acc = 100. * correct / total
            current_acc = 100. * current_correct / max(current_total, 1)
            avg_loss = total_loss / len(dataloader)
            avg_ce = total_ce_loss / len(dataloader)
            
            epoch_info = {
                'loss': f'{avg_loss:.3f}',
                'ce': f'{avg_ce:.3f}',
                'acc': f'{acc:.1f}%',
                'cur': f'{current_acc:.1f}%'
            }
            if old_classifier is not None:
                avg_kd = total_kd_loss / len(dataloader)
                epoch_info['kd'] = f'{avg_kd:.3f}'
            epoch_pbar.set_postfix(epoch_info)
            
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                log_msg = (f"Epoch {epoch+1}/{epochs} - "
                          f"Loss: {avg_loss:.3f}, "
                          f"CE: {avg_ce:.3f}, ")
                if old_classifier is not None:
                    log_msg += f"KD: {avg_kd:.3f}, "
                log_msg += f"Overall Acc: {acc:.2f}%, Current Task Acc: {current_acc:.2f}%"
                logger.info(log_msg)
        
        logger.success(f"Stage 3 completed for Task {task_id}")
    
    def train_task(self, train_loader, task_id, classes_in_task, 
                   stage1_epochs=10, stage2_epochs=20, stage3_epochs=20,
                   stage1_lr=0.001, stage2_lr=0.001, stage3_lr=0.001,
                   replay_samples=100, stage3_batch_size=128, current_weight=1.5,
                   stage2_mse_weight=1.0, stage2_cosine_weight=0.1, stage2_cls_weight=0.5):
        """训练一个完整任务（三个阶段）"""
        logger.info("#"*60)
        logger.info(f"TRAINING TASK {task_id}")
        logger.info(f"Classes: {classes_in_task}")
        logger.info("#"*60)
        
        self.current_task = task_id
        
        # 首先提取整个任务的VIT特征（一次性完成，加速后续训练）
        features, labels = self.extract_task_features(train_loader, task_id)
        
        # Stage 1（使用预提取的特征）
        self.stage1_train_task_classifier(
            features, labels, task_id, classes_in_task, 
            epochs=stage1_epochs, lr=stage1_lr, batch_size=stage3_batch_size
        )
        
        # Stage 2（使用预提取的特征，包含回放数据生成）
        self.stage2_train_reconstruction(
            features, labels, task_id, classes_in_task,
            epochs=stage2_epochs, lr=stage2_lr,
            samples_per_class=replay_samples,
            mse_weight=stage2_mse_weight,
            cosine_weight=stage2_cosine_weight,
            cls_weight=stage2_cls_weight,
            batch_size=stage3_batch_size
        )
        
        # Stage 3（使用预提取的特征+回放数据）
        self.stage3_train_shared_classifier(
            features, labels, task_id, classes_in_task,
            epochs=stage3_epochs, lr=stage3_lr,
            batch_size=stage3_batch_size,
            current_weight=current_weight
        )
        
        logger.success(f"Task {task_id} training completed")
    
    def evaluate(self, test_loader):
        """评估共享分类头的性能"""
        self.shared_classifier.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Evaluating", leave=False):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                vit_features = self.extract_vit_features(images)
                outputs = self.shared_classifier(vit_features)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        accuracy = 100. * correct / total
        return accuracy

# ========== 数据加载函数 ==========

def load_mnist_tasks(num_tasks=5, data_root='./data'):
    """将MNIST分割为多个类增量任务"""
    
    class ToRGB:
        def __call__(self, x):
            return x.repeat(3, 1, 1)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        ToRGB()
    ])
    
    logger.info(f"Loading MNIST dataset from {data_root}")
    train_dataset = torchvision.datasets.MNIST(
        root=data_root, train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root=data_root, train=False, download=True, transform=transform
    )
    
    all_classes = list(range(10))
    classes_per_task = len(all_classes) // num_tasks
    
    tasks = []
    for task_id in range(num_tasks):
        start_idx = task_id * classes_per_task
        if task_id == num_tasks - 1:
            task_classes = all_classes[start_idx:]
        else:
            task_classes = all_classes[start_idx:start_idx + classes_per_task]
        
        train_indices = [i for i, (_, label) in enumerate(train_dataset) 
                        if label in task_classes]
        test_indices = [i for i, (_, label) in enumerate(test_dataset) 
                       if label in task_classes]
        
        train_subset = Subset(train_dataset, train_indices)
        test_subset = Subset(test_dataset, test_indices)
        
        tasks.append({
            'train': train_subset,
            'test': test_subset,
            'classes': task_classes
        })
    
    logger.info(f"MNIST split into {num_tasks} tasks")
    return tasks

def load_cifar10_tasks(num_tasks=5, data_root='./data'):
    """将CIFAR-10分割为多个类增量任务"""
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    logger.info(f"Loading CIFAR-10 dataset from {data_root}")
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=transform
    )
    
    all_classes = list(range(10))
    classes_per_task = len(all_classes) // num_tasks
    
    tasks = []
    for task_id in range(num_tasks):
        start_idx = task_id * classes_per_task
        if task_id == num_tasks - 1:
            task_classes = all_classes[start_idx:]
        else:
            task_classes = all_classes[start_idx:start_idx + classes_per_task]
        
        train_indices = [i for i, (_, label) in enumerate(train_dataset) 
                        if label in task_classes]
        test_indices = [i for i, (_, label) in enumerate(test_dataset) 
                       if label in task_classes]
        
        train_subset = Subset(train_dataset, train_indices)
        test_subset = Subset(test_dataset, test_indices)
        
        tasks.append({
            'train': train_subset,
            'test': test_subset,
            'classes': task_classes
        })
    
    logger.info(f"CIFAR-10 split into {num_tasks} tasks")
    return tasks

def load_cifar100_tasks(num_tasks=10, data_root='./data'):
    """将CIFAR-100分割为多个类增量任务"""
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    logger.info(f"Loading CIFAR-100 dataset from {data_root}")
    train_dataset = torchvision.datasets.CIFAR100(
        root=data_root, train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root=data_root, train=False, download=True, transform=transform
    )
    
    all_classes = list(range(100))
    classes_per_task = len(all_classes) // num_tasks
    
    tasks = []
    for task_id in range(num_tasks):
        start_idx = task_id * classes_per_task
        if task_id == num_tasks - 1:
            task_classes = all_classes[start_idx:]
        else:
            task_classes = all_classes[start_idx:start_idx + classes_per_task]
        
        train_indices = [i for i, (_, label) in enumerate(train_dataset) 
                        if label in task_classes]
        test_indices = [i for i, (_, label) in enumerate(test_dataset) 
                       if label in task_classes]
        
        train_subset = Subset(train_dataset, train_indices)
        test_subset = Subset(test_dataset, test_indices)
        
        tasks.append({
            'train': train_subset,
            'test': test_subset,
            'classes': task_classes
        })
    
    logger.info(f"CIFAR-100 split into {num_tasks} tasks")
    return tasks

# ========== 主训练函数 ==========

def run_continual_learning_experiment(args):
    """运行完整的持续学习实验"""
    
    logger.info("="*60)
    logger.info("Continual Learning Experiment")
    logger.info(f"Dataset: {args.dataset.upper()}")
    logger.info(f"Number of tasks: {args.num_tasks}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info("="*60)
    
    # 加载数据
    if args.dataset == 'mnist':
        tasks = load_mnist_tasks(args.num_tasks, args.data_root)
        total_classes = 10
    elif args.dataset == 'cifar10':
        tasks = load_cifar10_tasks(args.num_tasks, args.data_root)
        total_classes = 10
    elif args.dataset == 'cifar100':
        tasks = load_cifar100_tasks(args.num_tasks, args.data_root)
        total_classes = 100
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # 创建持续学习系统
    cl_system = ContinualLearningSystem(
        device=args.device,
        total_classes=total_classes,
        hidden_dim=args.hidden_dim
    )
    
    # 记录性能
    accuracy_matrix = []
    
    task_pbar = tqdm(enumerate(tasks), total=len(tasks), desc="Overall Progress")
    
    # 逐个训练任务
    for task_id, task_data in task_pbar:
        task_pbar.set_description(f"Training Task {task_id}/{len(tasks)-1}")
        
        train_loader = DataLoader(
            task_data['train'], 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=args.num_workers
        )
        
        # 训练任务
        cl_system.train_task(
            train_loader, task_id, task_data['classes'],
            stage1_epochs=args.stage1_epochs,
            stage2_epochs=args.stage2_epochs,
            stage3_epochs=args.stage3_epochs,
            stage1_lr=args.stage1_lr,
            stage2_lr=args.stage2_lr,
            stage3_lr=args.stage3_lr,
            replay_samples=args.replay_samples,
            stage3_batch_size=args.batch_size,
            current_weight=args.current_weight,
            stage2_mse_weight=args.stage2_mse_weight,
            stage2_cosine_weight=args.stage2_cosine_weight,
            stage2_cls_weight=args.stage2_cls_weight
        )
        
        # 评估所有已见任务
        logger.info("="*60)
        logger.info(f"Evaluation after Task {task_id}")
        logger.info("="*60)
        
        task_accuracies = []
        eval_pbar = tqdm(range(task_id + 1), desc="Evaluating", leave=False)
        
        for eval_task_id in eval_pbar:
            eval_pbar.set_description(f"Eval Task {eval_task_id}")
            eval_data = tasks[eval_task_id]
            test_loader = DataLoader(
                eval_data['test'], 
                batch_size=args.batch_size, 
                shuffle=False, 
                num_workers=args.num_workers
            )
            
            acc = cl_system.evaluate(test_loader)
            task_accuracies.append(acc)
            logger.info(f"Task {eval_task_id} (Classes {eval_data['classes']}): {acc:.2f}%")
        
        avg_acc = np.mean(task_accuracies)
        logger.info(f"Average Accuracy: {avg_acc:.2f}%")
        
        task_pbar.set_postfix({'avg_acc': f'{avg_acc:.2f}%'})
        
        accuracy_matrix.append(task_accuracies)
    
    # 计算最终指标
    logger.info("="*60)
    logger.info("FINAL RESULTS")
    logger.info("="*60)
    
    final_accs = accuracy_matrix[-1]
    final_avg = np.mean(final_accs)
    logger.info(f"Final Average Accuracy: {final_avg:.2f}%")
    
    # 遗忘率（修正版）
    logger.info("\nForgetting Analysis:")
    forgetting = []
    for task_id in range(len(tasks) - 1):
        # 使用该任务刚训练完时的准确率作为initial
        initial_acc = accuracy_matrix[task_id][task_id]
        # 使用最终的准确率作为final
        final_acc = accuracy_matrix[-1][task_id]
        forget = initial_acc - final_acc
        forgetting.append(forget)
        logger.info(f"Task {task_id}: Initial {initial_acc:.2f}% → Final {final_acc:.2f}% (Forgetting: {forget:.2f}%)")
    
    avg_forgetting = np.mean(forgetting) if forgetting else 0
    logger.info(f"Average Forgetting: {avg_forgetting:.2f}%")
    
    return cl_system, accuracy_matrix

# ========== 参数解析 ==========

def parse_args():
    parser = argparse.ArgumentParser(
        description='Continual Learning with Feature Reconstruction'
    )
    
    parser.add_argument('--dataset', type=str, default='mnist',
                       choices=['mnist', 'cifar10', 'cifar100'],
                       help='Dataset to use')
    parser.add_argument('--data-root', type=str, default='./data',
                       help='Root directory for dataset')
    parser.add_argument('--num-tasks', type=int, default=5,
                       help='Number of tasks to split the dataset into')
    
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size for training')
    parser.add_argument('--num-workers', type=int, default=2,
                       help='Number of workers for data loading')
    
    parser.add_argument('--stage1-epochs', type=int, default=10,
                       help='Number of epochs for Stage 1')
    parser.add_argument('--stage1-lr', type=float, default=0.001,
                       help='Learning rate for Stage 1')
    
    parser.add_argument('--stage2-epochs', type=int, default=20,
                       help='Number of epochs for Stage 2')
    parser.add_argument('--stage2-lr', type=float, default=0.001,
                       help='Learning rate for Stage 2')
    parser.add_argument('--stage2-mse-weight', type=float, default=1.0,
                       help='Weight for MSE loss in Stage 2')
    parser.add_argument('--stage2-cosine-weight', type=float, default=0.1,
                       help='Weight for cosine similarity loss in Stage 2')
    parser.add_argument('--stage2-cls-weight', type=float, default=0.5,
                       help='Weight for classification consistency loss in Stage 2')
    
    parser.add_argument('--stage3-epochs', type=int, default=20,
                       help='Number of epochs for Stage 3')
    parser.add_argument('--stage3-lr', type=float, default=0.001,
                       help='Learning rate for Stage 3')
    parser.add_argument('--replay-samples', type=int, default=100,
                       help='Number of replay samples per class')
    parser.add_argument('--current-weight', type=float, default=1.5,
                       help='Weight for current task data in Stage 3')
    
    parser.add_argument('--hidden-dim', type=int, default=512,
                       help='Hidden dimension for reconstruction model')
    
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use for training')
    
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--log-file', type=str, default=None,
                       help='Path to log file (optional)')
    
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU instead")
        args.device = 'cpu'
    
    return args

# ========== 主函数 ==========

def main():
    args = parse_args()
    
    setup_logger(args.log_level, args.log_file)
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    logger.info(f"Random seed set to {args.seed}")
    
    logger.info("Experiment Configuration:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    
    try:
        cl_system, accuracy_matrix = run_continual_learning_experiment(args)
        logger.success("Experiment completed successfully")
    except Exception as e:
        logger.error(f"Experiment failed with error: {e}")
        raise
    
    return cl_system, accuracy_matrix

if __name__ == "__main__":
    main()