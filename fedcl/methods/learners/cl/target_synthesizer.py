"""
GlobalSynthesizer for TARGET Algorithm
Implements server-side synthetic data generation for federated continual learning
"""
import os
import copy
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import transforms
from kornia import augmentation

from .target_generator import (
    Generator, DeepInversionHook, ImagePool, Normalizer,
    kldiv, reptile_grad, fomaml_grad, reset_l0_fun
)


class GlobalSynthesizer:
    """
    Global synthesizer for generating synthetic data on the server

    Implements the data generation algorithm from TARGET:
    1. Initialize generator and student model
    2. For each iteration:
       - Generate synthetic images from random noise
       - Optimize generator to match teacher's batch norm statistics
       - Optimize to maximize cross-entropy on teacher predictions
       - Optionally add adversarial loss (student != teacher)
    3. Use MAML or REPTILE meta-learning to update generator

    Args:
        teacher: Teacher model (current global model)
        student: Student model (randomly initialized for validation)
        generator: Generator network
        nz: Dimension of latent noise
        num_classes: Number of classes
        img_size: Image size tuple (C, H, W)
        iterations: Number of synthesis iterations per round
        lr_g: Learning rate for generator
        lr_z: Learning rate for latent noise
        synthesis_batch_size: Batch size for synthesis
        sample_batch_size: Batch size for sampling
        adv: Weight for adversarial loss
        bn: Weight for batch norm loss
        oh: Weight for one-hot cross entropy loss
        save_dir: Directory to save synthetic data
        transform: Data augmentation transform
        normalizer: Normalization function
        warmup: Number of warmup rounds before adversarial loss
        reset_l0: Whether to reset first layer at round 120
        reset_bn: Whether to reset batch norm
        bn_mmt: Batch norm momentum
        is_maml: Use MAML (True) or REPTILE (False)
    """

    def __init__(self, teacher, student, generator, nz, num_classes, img_size,
                 iterations=100, lr_g=0.1, lr_z=0.01,
                 synthesis_batch_size=128, sample_batch_size=128,
                 adv=0.0, bn=1.0, oh=1.0,
                 save_dir='run/fast', transform=None, normalizer=None,
                 warmup=10, reset_l0=0, reset_bn=0, bn_mmt=0,
                 is_maml=1, device='cuda'):

        self.teacher = teacher
        self.student = student
        self.save_dir = save_dir
        self.img_size = img_size
        self.iterations = iterations
        self.lr_g = lr_g
        self.lr_z = lr_z
        self.nz = nz
        self.adv = adv
        self.bn = bn
        self.oh = oh
        self.is_maml = is_maml
        self.device = device if torch.cuda.is_available() else 'cpu'

        self.num_classes = num_classes
        self.synthesis_batch_size = synthesis_batch_size
        self.sample_batch_size = sample_batch_size
        self.normalizer = normalizer

        self.data_pool = ImagePool(root=self.save_dir)
        self.transform = transform
        self.generator = generator.to(self.device).train()
        self.ep = 0
        self.ep_start = warmup
        self.reset_l0 = reset_l0
        self.reset_bn = reset_bn
        self.prev_z = None

        # Meta optimizer for generator
        if self.is_maml:
            self.meta_optimizer = Adam(self.generator.parameters(),
                                      self.lr_g * self.iterations,
                                      betas=[0.5, 0.999])
        else:
            self.meta_optimizer = Adam(self.generator.parameters(),
                                      self.lr_g * self.iterations,
                                      betas=[0.5, 0.999])

        # Data augmentation
        self.aug = transforms.Compose([
            augmentation.RandomCrop(size=[self.img_size[-2], self.img_size[-1]], padding=4),
            augmentation.RandomHorizontalFlip(),
            normalizer,
        ])

        # Batch normalization hooks
        self.bn_mmt = bn_mmt
        self.hooks = []
        for m in teacher.modules():
            if isinstance(m, nn.BatchNorm2d):
                self.hooks.append(DeepInversionHook(m, self.bn_mmt))

    def synthesize(self, targets=None):
        """
        Generate one batch of synthetic data

        Returns:
            Generated images are saved to self.data_pool
        """
        self.ep += 1
        self.student.eval()
        self.teacher.eval()
        best_cost = 1e6

        # Reset first layer at specific epoch
        if (self.ep == 120 + self.ep_start) and self.reset_l0:
            reset_l0_fun(self.generator)

        best_inputs = None
        # Initialize random latent noise
        z = torch.randn(size=(self.synthesis_batch_size, self.nz)).to(self.device)
        z.requires_grad = True

        # Initialize targets
        if targets is None:
            targets = torch.randint(low=0, high=self.num_classes,
                                  size=(self.synthesis_batch_size,))
        else:
            targets = targets.sort()[0]  # sort for better visualization
        targets = targets.to(self.device)

        # Clone generator for fast adaptation
        fast_generator = self.generator.clone()
        optimizer = Adam([
            {'params': fast_generator.parameters()},
            {'params': [z], 'lr': self.lr_z}
        ], lr=self.lr_g, betas=[0.5, 0.999])

        # Inner loop optimization
        for it in range(self.iterations):
            # Generate synthetic images
            inputs = fast_generator(z)
            inputs_aug = self.aug(inputs)  # crop and normalize

            # Teacher forward pass
            t_out = self.teacher(inputs_aug)
            if isinstance(t_out, dict):
                t_out = t_out["logits"]

            if targets is None:
                targets = torch.argmax(t_out, dim=-1)
                targets = targets.to(self.device)

            # Compute losses
            # 1. Batch normalization statistics matching loss
            loss_bn = sum([h.r_feature for h in self.hooks])

            # 2. Cross-entropy loss (one-hot)
            loss_oh = nn.functional.cross_entropy(t_out, targets)

            # 3. Adversarial distillation loss
            if self.adv > 0 and (self.ep >= self.ep_start):
                s_out = self.student(inputs_aug)
                if isinstance(s_out, dict):
                    s_out = s_out["logits"]
                # Only apply loss when student and teacher agree
                mask = (s_out.max(1)[1] == t_out.max(1)[1]).float()
                loss_adv = -(kldiv(s_out, t_out, reduction='none').sum(1) * mask).mean()
            else:
                loss_adv = loss_oh.new_zeros(1)

            # Total loss
            loss = self.bn * loss_bn + self.oh * loss_oh + self.adv * loss_adv

            # Track best inputs
            with torch.no_grad():
                if best_cost > loss.item() or best_inputs is None:
                    best_cost = loss.item()
                    best_inputs = inputs.data.cpu()

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()

            # MAML meta-gradient accumulation
            if self.is_maml:
                if it == 0:
                    self.meta_optimizer.zero_grad()
                fomaml_grad(self.generator, fast_generator)
                if it == (self.iterations - 1):
                    self.meta_optimizer.step()

            optimizer.step()

        # Update batch norm momentum
        if self.bn_mmt != 0:
            for h in self.hooks:
                h.update_mmt()

        # REPTILE meta-gradient (if not MAML)
        if not self.is_maml:
            self.meta_optimizer.zero_grad()
            reptile_grad(self.generator, fast_generator)
            self.meta_optimizer.step()

        self.student.train()
        self.prev_z = (z, targets)

        # Add best inputs to data pool
        self.data_pool.add(best_inputs)

        return best_inputs

    def remove_hooks(self):
        """Remove all batch norm hooks"""
        for h in self.hooks:
            h.remove()
