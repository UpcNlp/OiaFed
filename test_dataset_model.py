"""
简单的数据集和模型管理测试
"""
import torch
import torch.nn as nn
from torchvision import datasets, transforms

# Step 1: 设置路径
import sys
sys.path.insert(0, '/home/nlp/ct/projects/MOE-FedCL')

# Step 2: 直接导入需要的模块（避免触发 methods 的完整初始化）
from fedcl.api.decorators import dataset, model
from fedcl.api.registry import registry
from fedcl.methods.datasets.base import FederatedDataset
from fedcl.methods.models.base import FederatedModel

print("✅ 所有模块导入成功！\n")

# Step 3: 注册数据集
@dataset('mnist_test', description='MNIST测试数据集', num_classes=10)
class MNISTTest(FederatedDataset):
    def __init__(self, root='./data', train=True):
        super().__init__(root, train, download=True)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.dataset = datasets.MNIST(root, train=train, download=True, transform=transform)
        self.num_classes = 10
        self.input_shape = (1, 28, 28)

    def get_statistics(self):
        return {
            'num_samples': len(self.dataset),
            'num_classes': self.num_classes,
            'input_shape': self.input_shape
        }

print("✅ 数据集已注册！")

# Step 4: 注册模型
@model('simple_cnn_test', description='简单CNN测试', task='classification')
class SimpleCNNTest(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(32 * 14 * 14, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

print("✅ 模型已注册！\n")

# Step 5: 从注册表获取并使用
print("=" * 60)
print("从注册表获取组件")
print("=" * 60)

# 获取数据集
DatasetCls = registry.get_dataset('mnist_test')
fed_dataset = DatasetCls(root='./data', train=True)
print(f"\n数据集统计: {fed_dataset.get_statistics()}")

# 测试数据划分
print("\n--- IID划分测试 ---")
clients = fed_dataset.partition(num_clients=3, strategy='iid')
for cid, ds in clients.items():
    print(f"  Client {cid}: {len(ds)} 样本")

# 获取模型
ModelCls = registry.get_model('simple_cnn_test')
model = ModelCls(num_classes=10)
print(f"\n模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

# 查看注册表信息
print("\n" + "=" * 60)
print("注册表信息")
print("=" * 60)
components = registry.list_all_components()
for comp_type, names in components.items():
    if names:
        print(f"  {comp_type}: {names}")

print("\n✅ 测试完成！")
