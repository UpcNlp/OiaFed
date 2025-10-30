"""
数据集和模型管理系统使用示例
examples/dataset_model_demo.py

展示如何使用：
1. @dataset 装饰器注册数据集
2. @model 装饰器注册模型
3. 数据划分策略（IID/Non-IID）
4. 从注册表获取组件
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from typing import Dict, Any

# 导入装饰器和基类
from fedcl.api.decorators import dataset, model
from fedcl.api.registry import registry

# 直接导入，避免触发 methods 的完整初始化
import sys
sys.path.insert(0, '/home/nlp/ct/projects/MOE-FedCL')
from fedcl.methods.datasets.base import FederatedDataset
from fedcl.methods.models.base import FederatedModel, count_parameters, print_model_summary


# ==================== 1. 定义并注册数据集 ====================

@dataset(
    name='mnist_federated',
    description='MNIST联邦学习数据集',
    version='1.0',
    dataset_type='image_classification',
    num_classes=10
)
class MNISTFederated(FederatedDataset):
    """MNIST联邦数据集实现"""

    def __init__(self, root: str = './data', train: bool = True, download: bool = True):
        super().__init__(root, train, download)

        # 数据转换
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # 加载MNIST数据集
        self.dataset = datasets.MNIST(
            root=root,
            train=train,
            download=download,
            transform=transform
        )

        # 设置属性
        self.num_classes = 10
        self.input_shape = (1, 28, 28)

    def get_statistics(self) -> Dict[str, Any]:
        """获取数据集统计信息"""
        return {
            'dataset_name': 'MNIST',
            'num_samples': len(self.dataset),
            'num_classes': self.num_classes,
            'input_shape': self.input_shape,
            'train': self.train,
        }


# ==================== 2. 定义并注册模型 ====================

@model(
    name='simple_cnn',
    description='简单的CNN分类模型',
    version='1.0',
    task='classification',
    input_shape=(1, 28, 28),
    output_shape=(10,)
)
class SimpleCNN(nn.Module):
    """简单的CNN模型"""

    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


@model(
    name='advanced_cnn',
    description='带有FederatedModel功能的高级CNN',
    version='1.0',
    task='classification'
)
class AdvancedCNN(FederatedModel):
    """继承FederatedModel的高级CNN（可选）"""

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # 设置元数据
        self.set_metadata(
            task_type='classification',
            input_shape=(1, 28, 28),
            output_shape=(num_classes,)
        )

        # 定义网络结构
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# ==================== 3. 使用示例 ====================

def demo_dataset_partitioning():
    """演示数据集划分"""
    print("=" * 80)
    print("数据集划分演示")
    print("=" * 80)

    # 从注册表获取数据集类
    MNISTFedCls = registry.get_dataset('mnist_federated')
    fed_dataset = MNISTFedCls(root='./data', train=True, download=True)

    print(f"\n数据集信息: {fed_dataset.get_statistics()}")

    # 演示1：IID划分
    print("\n--- 1. IID划分（均匀随机）---")
    client_datasets_iid = fed_dataset.partition(num_clients=5, strategy='iid')

    for client_id, dataset in client_datasets_iid.items():
        print(f"  Client {client_id}: {len(dataset)} 样本")
        # 查看类别分布
        dist = fed_dataset.get_class_distribution(dataset.indices)
        print(f"    类别分布: {dist}")

    # 演示2：Label Skew Non-IID
    print("\n--- 2. Label Skew Non-IID（每客户端2个类别）---")
    client_datasets_label = fed_dataset.partition(
        num_clients=5,
        strategy='non_iid_label',
        labels_per_client=2
    )

    for client_id, dataset in client_datasets_label.items():
        print(f"  Client {client_id}: {len(dataset)} 样本")
        dist = fed_dataset.get_class_distribution(dataset.indices)
        print(f"    类别分布: {dist}")

    # 演示3：Dirichlet Non-IID
    print("\n--- 3. Dirichlet Non-IID (alpha=0.5) ---")
    client_datasets_dir = fed_dataset.partition(
        num_clients=5,
        strategy='dirichlet',
        alpha=0.5
    )

    for client_id, dataset in client_datasets_dir.items():
        print(f"  Client {client_id}: {len(dataset)} 样本")
        dist = fed_dataset.get_class_distribution(dataset.indices)
        print(f"    类别分布: {dist}")


def demo_model_usage():
    """演示模型使用"""
    print("\n" + "=" * 80)
    print("模型使用演示")
    print("=" * 80)

    # 从注册表获取模型类
    SimpleCNNCls = registry.get_model('simple_cnn')
    AdvancedCNNCls = registry.get_model('advanced_cnn')

    # 创建模型实例
    simple_model = SimpleCNNCls(num_classes=10)
    advanced_model = AdvancedCNNCls(num_classes=10)

    print("\n--- Simple CNN ---")
    print(f"参数数量: {count_parameters(simple_model):,}")
    print_model_summary(simple_model, input_shape=(1, 28, 28))

    print("\n--- Advanced CNN (FederatedModel) ---")
    print(f"参数数量: {advanced_model.get_param_count():,}")
    print(f"模型大小: {advanced_model.get_model_size('MB'):.2f} MB")
    print(f"权重范数: {advanced_model.get_weight_norm():.4f}")
    print(advanced_model.summary(input_shape=(1, 28, 28)))


def demo_registry_info():
    """演示注册表信息查询"""
    print("\n" + "=" * 80)
    print("注册表信息查询")
    print("=" * 80)

    # 列出所有组件
    all_components = registry.list_all_components()
    print("\n已注册的组件:")
    for comp_type, names in all_components.items():
        if names:  # 只显示非空的
            print(f"  {comp_type}: {names}")

    # 组件数量统计
    counts = registry.get_component_count()
    print("\n组件数量统计:")
    for comp_type, count in counts.items():
        print(f"  {comp_type}: {count}")

    # 检查组件是否存在
    print("\n组件存在性检查:")
    print(f"  mnist_federated (dataset): {registry.has_component('mnist_federated', 'dataset')}")
    print(f"  simple_cnn (model): {registry.has_component('simple_cnn', 'model')}")
    print(f"  non_existent (dataset): {registry.has_component('non_existent', 'dataset')}")


if __name__ == '__main__':
    print("\n🚀 MOE-FedCL 数据集和模型管理系统演示\n")

    # 1. 数据集划分演示
    demo_dataset_partitioning()

    # 2. 模型使用演示
    demo_model_usage()

    # 3. 注册表信息查询
    demo_registry_info()

    print("\n✅ 演示完成！")
