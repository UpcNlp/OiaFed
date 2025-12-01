"""
TARGET Generator and Data Synthesis Module
Based on paper implementation from ICCV 2023
"""
import os
import math
import time
import copy
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image
from typing import Optional, Tuple, List
import numpy as np


class Generator(nn.Module):
    """
    Generator network for synthesizing images

    Args:
        nz: Dimension of latent noise vector
        ngf: Number of generator features
        img_size: Output image size
        nc: Number of output channels (3 for RGB)
    """
    def __init__(self, nz=100, ngf=64, img_size=32, nc=3):
        super(Generator, self).__init__()
        self.params = (nz, ngf, img_size, nc)
        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(nz, ngf * 2 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(ngf * 2),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(ngf*2, ngf*2, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(ngf*2, ngf, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

    def clone(self):
        """Return a copy of itself"""
        clone = Generator(self.params[0], self.params[1], self.params[2], self.params[3])
        clone.load_state_dict(self.state_dict())
        if torch.cuda.is_available():
            return clone.cuda()
        return clone


class DeepInversionHook:
    """
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    """

    def __init__(self, module, mmt_rate):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = module
        self.mmt_rate = mmt_rate
        self.mmt = None
        self.tmp_val = None
        self.r_feature = None

    def hook_fn(self, module, input, output):
        # hook to compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)

        # forcing mean and variance to match between two distributions
        if self.mmt is None:
            r_feature = torch.norm(module.running_var.data - var, 2) + \
                        torch.norm(module.running_mean.data - mean, 2)
        else:
            mean_mmt, var_mmt = self.mmt
            r_feature = torch.norm(module.running_var.data - (1 - self.mmt_rate) * var - self.mmt_rate * var_mmt, 2) + \
                        torch.norm(module.running_mean.data - (1 - self.mmt_rate) * mean - self.mmt_rate * mean_mmt, 2)

        self.r_feature = r_feature
        self.tmp_val = (mean, var)

    def update_mmt(self):
        mean, var = self.tmp_val
        if self.mmt is None:
            self.mmt = (mean.data, var.data)
        else:
            mean_mmt, var_mmt = self.mmt
            self.mmt = (self.mmt_rate*mean_mmt+(1-self.mmt_rate)*mean.data,
                        self.mmt_rate*var_mmt+(1-self.mmt_rate)*var.data)

    def remove(self):
        self.hook.remove()


def normalize(tensor, mean, std, reverse=False):
    """Normalize or denormalize tensor"""
    if reverse:
        _mean = [-m / s for m, s in zip(mean, std)]
        _std = [1/s for s in std]
    else:
        _mean = mean
        _std = std

    _mean = torch.as_tensor(_mean, dtype=tensor.dtype, device=tensor.device)
    _std = torch.as_tensor(_std, dtype=tensor.dtype, device=tensor.device)
    tensor = (tensor - _mean[None, :, None, None]) / (_std[None, :, None, None])
    return tensor


class Normalizer:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x, reverse=False):
        return normalize(x, self.mean, self.std, reverse=reverse)


def kldiv(logits, targets, T=1.0, reduction='batchmean'):
    """KL Divergence loss for knowledge distillation"""
    q = F.log_softmax(logits/T, dim=1)
    p = F.softmax(targets/T, dim=1)
    return F.kl_div(q, p, reduction=reduction) * (T*T)


class KLDiv(nn.Module):
    def __init__(self, T=1.0, reduction='batchmean'):
        super().__init__()
        self.T = T
        self.reduction = reduction

    def forward(self, logits, targets):
        return kldiv(logits, targets, T=self.T, reduction=self.reduction)


def reptile_grad(src, tar):
    """REPTILE meta-learning gradient"""
    for p, tar_p in zip(src.parameters(), tar.parameters()):
        if p.grad is None:
            p.grad = Variable(torch.zeros(p.size())).cuda() if torch.cuda.is_available() else Variable(torch.zeros(p.size()))
        p.grad.data.add_(p.data - tar_p.data, alpha=67)


def fomaml_grad(src, tar):
    """First-Order MAML gradient"""
    for p, tar_p in zip(src.parameters(), tar.parameters()):
        if p.grad is None:
            p.grad = Variable(torch.zeros(p.size())).cuda() if torch.cuda.is_available() else Variable(torch.zeros(p.size()))
        p.grad.data.add_(tar_p.grad.data)


def reset_l0_fun(model):
    """Reset first layer weights"""
    for n, m in model.named_modules():
        if n == "l1.0" or n == "conv_blocks.0":
            nn.init.normal_(m.weight, 0.0, 0.02)
            nn.init.constant_(m.bias, 0)


def weight_init(m):
    """Initialize model weights"""
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


def pack_images(images, col=None, channel_last=False, padding=1):
    """Pack multiple images into a grid"""
    if isinstance(images, (list, tuple)):
        images = np.stack(images, 0)
    if channel_last:
        images = images.transpose(0,3,1,2)
    assert len(images.shape)==4
    assert isinstance(images, np.ndarray)

    N,C,H,W = images.shape
    if col is None:
        col = int(math.ceil(math.sqrt(N)))
    row = int(math.ceil(N / col))

    pack = np.zeros((C, H*row+padding*(row-1), W*col+padding*(col-1)), dtype=images.dtype)
    for idx, img in enumerate(images):
        h = (idx // col) * (H+padding)
        w = (idx % col) * (W+padding)
        pack[:, h:h+H, w:w+W] = img
    return pack


def save_image_batch(imgs, output, col=None, size=None, pack=True):
    """Save a batch of images to file

    Supports both RGB (3-channel) and grayscale (1-channel) images.
    """
    if isinstance(imgs, torch.Tensor):
        imgs = (imgs.detach().clamp(0, 1).cpu().numpy()*255).astype('uint8')
    base_dir = os.path.dirname(output)
    if base_dir!='':
        os.makedirs(base_dir, exist_ok=True)
    if pack:
        packed = pack_images(imgs, col=col).transpose(1, 2, 0)
        # Handle grayscale images (squeeze removes the channel dimension)
        if packed.shape[-1] == 1:
            packed = packed.squeeze(-1)
        imgs_pil = Image.fromarray(packed)
        if size is not None:
            if isinstance(size, (list,tuple)):
                imgs_pil = imgs_pil.resize(size)
            else:
                w, h = imgs_pil.size
                max_side = max(h, w)
                scale = float(size) / float(max_side)
                _w, _h = int(w*scale), int(h*scale)
                imgs_pil = imgs_pil.resize([_w, _h])
        imgs_pil.save(output)
    else:
        output_filename = output.strip('.png')
        for idx, img in enumerate(imgs):
            # img shape: (C, H, W) -> (H, W, C)
            img_transposed = img.transpose(1, 2, 0)
            # Handle grayscale images (1 channel)
            if img_transposed.shape[-1] == 1:
                img_transposed = img_transposed.squeeze(-1)
            img_pil = Image.fromarray(img_transposed)
            img_pil.save(output_filename+'-%d.png'%(idx))


def _collect_all_images(nums, root, postfix=['png', 'jpg', 'jpeg', 'JPEG']):
    """Collect all images from directory"""
    images = []
    if isinstance(postfix, str):
        postfix = [postfix]
    for dirpath, dirnames, files in os.walk(root):
        for pos in postfix:
            if nums is not None:
                files.sort()
                files = files[:nums]
            for f in files:
                if f.endswith(pos):
                    images.append(os.path.join(dirpath, f))
    return images


class UnlabeledImageDataset(torch.utils.data.Dataset):
    """Dataset for unlabeled images

    Supports both RGB and grayscale images. Images are loaded in their
    original mode (grayscale images stay grayscale).
    """
    def __init__(self, root, transform=None, nums=None):
        self.root = os.path.abspath(root)
        self.images = _collect_all_images(nums, self.root)
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.open(self.images[idx])
        # Keep grayscale images as-is (mode 'L'), convert others to RGB
        if img.mode not in ('L', 'RGB'):
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            elif img.mode == 'P':
                img = img.convert('RGB')
            # For other modes, try to keep as-is
        if self.transform:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.images)

    def __repr__(self):
        return 'Unlabeled data:\n\troot: %s\n\tdata mount: %d\n\ttransforms: %s'%(self.root, len(self), self.transform)


class ImagePool:
    """Image pool for storing generated synthetic images"""
    def __init__(self, root):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)
        self._idx = 0

    def add(self, imgs, targets=None):
        """Add images to the pool"""
        save_image_batch(imgs, os.path.join(self.root, "%d.png"%(self._idx)), pack=False)
        self._idx += 1

    def get_dataset(self, nums=None, transform=None):
        """Get dataset from pool"""
        return UnlabeledImageDataset(self.root, transform=transform, nums=nums)


class DataIter:
    """Infinite data iterator"""
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self._iter = iter(self.dataloader)

    def next(self):
        try:
            data = next(self._iter)
        except StopIteration:
            self._iter = iter(self.dataloader)
            data = next(self._iter)
        return data
