"""
数据变换工具模块

集中管理各种数据集的数据变换，避免在学习器和训练器中硬编码。
"""

from torchvision import transforms


def get_train_transform(dataset_name: str):
    """获取训练数据变换"""
    transforms_dict = {
        "cifar100": transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5071, 0.4867, 0.4408), 
                std=(0.2675, 0.2565, 0.2761)
            ),
        ]),
        "cifar10": transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465), 
                std=(0.2023, 0.1994, 0.2010)
            ),
        ]),
        "tiny_imagenet": transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ),
        ]),
        "imagenet": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ),
        ]),
        "mnist": transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]),
        "fashion_mnist": transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
    }
    
    if dataset_name not in transforms_dict:
        raise ValueError(f"不支持的数据集: {dataset_name}")
    
    return transforms_dict[dataset_name]


def get_test_transform(dataset_name: str):
    """获取测试数据变换"""
    transforms_dict = {
        "cifar100": transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5071, 0.4867, 0.4408), 
                std=(0.2675, 0.2565, 0.2761)
            ),
        ]),
        "cifar10": transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465), 
                std=(0.2023, 0.1994, 0.2010)
            ),
        ]),
        "tiny_imagenet": transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ),
        ]),
        "imagenet": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ),
        ]),
        "mnist": transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]),
        "fashion_mnist": transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
    }
    
    if dataset_name not in transforms_dict:
        raise ValueError(f"不支持的数据集: {dataset_name}")
    
    return transforms_dict[dataset_name]


def get_augmentation_transform(dataset_name: str, augmentation_type: str = "standard"):
    """获取数据增强变换"""
    if augmentation_type == "standard":
        return get_train_transform(dataset_name)
    elif augmentation_type == "strong":
        # 更强的数据增强
        if dataset_name in ["cifar100", "cifar10"]:
            return transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.5071, 0.4867, 0.4408) if dataset_name == "cifar100" else (0.4914, 0.4822, 0.4465), 
                    std=(0.2675, 0.2565, 0.2761) if dataset_name == "cifar100" else (0.2023, 0.1994, 0.2010)
                ),
            ])
        else:
            return get_train_transform(dataset_name)
    else:
        raise ValueError(f"不支持的数据增强类型: {augmentation_type}")


# 支持的数据集列表
SUPPORTED_DATASETS = [
    "cifar100", "cifar10", "tiny_imagenet", 
    "imagenet", "mnist", "fashion_mnist"
]
