"""
FOT AlexNet — 带多头输出与激活捕获

移植自官方 FOT 实现:
https://github.com/duygunuryldz/Federated_Orthogonal_Training/blob/main/model/alexnet.py

核心特性:
1. 多头输出: 每个任务一个独立分类头 (self.last)
2. 激活捕获: forward 时记录每层输入到 self.act (GPSE 所需)
3. 与官方代码结构完全一致
"""

import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from copy import deepcopy

from ...registry.decorators import model


@model(
    name='fot_alexnet',
    description='FOT AlexNet — 多头输出 + 激活捕获 (ICLR 2024)',
    task='classification',
    version='1.0',
    input_shape=(3, 32, 32),
    output_shape=(10,)
)
class FOTAlexNet(nn.Module):
    """
    FOT 使用的 AlexNet 模型

    与官方实现完全一致:
    - 3 个卷积层 (64 → 128 → 256) + MaxPool + Dropout
    - 2 个全连接层 (2048 → 2048)
    - task_num 个独立分类头

    forward 返回 list[Tensor], 每个元素是一个任务头的输出。
    训练时用 output[task_id] 选择当前任务的头。

    self.act: OrderedDict, forward 时捕获每层输入激活
    self.ksize: 各卷积层的 kernel size (GPSE unfold 所需)
    self.map: 各层的特征图尺寸信息
    """

    # 需要做正交投影的层名列表 (与官方一致)
    ORTH_LAYER_NAMES = [
        'conv1.weight', 'conv2.weight', 'conv3.weight',
        'fc1.weight', 'fc2.weight',
    ]

    def __init__(self, num_classes: int = 10, task_num: int = 5):
        super().__init__()
        self.act = OrderedDict()
        self.map = []
        self.ksize = []
        self.in_channel = []

        # --- conv1 ---
        self.map.append(32)
        self.conv1 = nn.Conv2d(3, 64, 4, bias=False)
        s = self._compute_conv_output_size(32, 4)
        s = s // 2
        self.ksize.append(4)
        self.in_channel.append(3)

        # --- conv2 ---
        self.map.append(s)
        self.conv2 = nn.Conv2d(64, 128, 3, bias=False)
        s = self._compute_conv_output_size(s, 3)
        s = s // 2
        self.ksize.append(3)
        self.in_channel.append(64)

        # --- conv3 ---
        self.map.append(s)
        self.conv3 = nn.Conv2d(128, 256, 2, bias=False)
        s = self._compute_conv_output_size(s, 2)
        s = s // 2
        self.smid = s
        self.ksize.append(2)
        self.in_channel.append(128)

        self.map.append(256 * self.smid * self.smid)

        # --- shared layers ---
        self.maxpool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.drop1 = nn.Dropout(0.2)
        self.drop2 = nn.Dropout(0.5)

        # --- fc layers ---
        self.fc1 = nn.Linear(256 * self.smid * self.smid, 2048, bias=False)
        self.fc2 = nn.Linear(2048, 2048, bias=False)
        self.map.extend([2048])

        # --- 多头分类器 ---
        self.last = nn.ModuleList()
        for _ in range(task_num):
            self.last.append(nn.Linear(2048, num_classes, bias=False))

        self.n_outputs = num_classes
        self.task_num = task_num

    def forward(self, x):
        bsz = x.size(0)
        x = x.view(bsz, 3, 32, 32)

        self.act['conv1.weight'] = x
        x = self.conv1(x)
        x = self.maxpool(self.drop1(self.relu(x)))

        self.act['conv2.weight'] = x
        x = self.conv2(x)
        x = self.maxpool(self.drop1(self.relu(x)))

        self.act['conv3.weight'] = x
        x = self.conv3(x)
        x = self.maxpool(self.drop2(self.relu(x)))

        x = x.view(bsz, -1)
        self.act['fc1.weight'] = x
        x = self.fc1(x)
        x = self.drop2(self.relu(x))

        self.act['fc2.weight'] = x
        x = self.fc2(x)
        x = self.drop2(self.relu(x))

        y = []
        for t in range(len(self.last)):
            y.append(self.last[t](x))
        return y

    @staticmethod
    def _compute_conv_output_size(Lin, kernel_size, stride=1, padding=0, dilation=1):
        return int(np.floor(
            (Lin + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1
        ))


def get_fot_alexnet(num_classes=10, task_num=5):
    return FOTAlexNet(num_classes=num_classes, task_num=task_num)
