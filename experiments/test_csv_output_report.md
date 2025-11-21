# 联邦学习实验结果报告

生成时间: 2025-11-20 20:22:53

## 1. 基本统计

- 总实验数: 3
- 成功: 3
- 失败: 0

## 2. 按数据集分组统计

| Dataset   |   ('Final_Accuracy', 'mean') |   ('Final_Accuracy', 'max') |   ('Final_Accuracy', 'min') |   ('Final_Accuracy', 'count') |
|:----------|-----------------------------:|----------------------------:|----------------------------:|------------------------------:|
| CIFAR10   |                       0.7234 |                      0.7234 |                      0.7234 |                             1 |
| FMNIST    |                       0.8567 |                      0.8567 |                      0.8567 |                             1 |
| MNIST     |                       0.9823 |                      0.9823 |                      0.9823 |                             1 |

## 3. 按算法分组统计

| Algorithm   |   ('Final_Accuracy', 'mean') |   ('Final_Accuracy', 'max') |   ('Final_Accuracy', 'min') |   ('Final_Accuracy', 'count') |
|:------------|-----------------------------:|----------------------------:|----------------------------:|------------------------------:|
| FedAvg      |                       0.9823 |                      0.9823 |                      0.9823 |                             1 |
| FedProx     |                       0.7234 |                      0.7234 |                      0.7234 |                             1 |
| SCAFFOLD    |                       0.8567 |                      0.8567 |                      0.8567 |                             1 |

## 4. Top 10 最佳实验

| Experiment_Name             | Dataset   | Algorithm   | NonIID_Type   |   Final_Accuracy |   Completed_Rounds |
|:----------------------------|:----------|:------------|:--------------|-----------------:|-------------------:|
| MNIST_IID_FedAvg            | MNIST     | FedAvg      | IID           |           0.9823 |                 12 |
| FMNIST_#C=2_SCAFFOLD        | FMNIST    | SCAFFOLD    | #C=2          |           0.8567 |                 18 |
| CIFAR10_pk~Dir(0.5)_FedProx | CIFAR10   | FedProx     | pk~Dir(0.5)   |           0.7234 |                 25 |

