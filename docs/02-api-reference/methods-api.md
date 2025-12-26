# 算法 API

内置算法实现的 API 参考。

---

## 聚合器

### FedAvgAggregator

```python
@register("aggregator.fedavg")
class FedAvgAggregator(Aggregator):
    def __init__(self, weighted: bool = True):
        """
        Args:
            weighted: 是否按样本数加权
        """
    
    def aggregate(self, updates, global_model=None):
        """加权平均聚合"""
```

### FedProxAggregator

```python
@register("aggregator.fedprox")
class FedProxAggregator(Aggregator):
    def __init__(self, mu: float = 0.01, weighted: bool = True):
        """
        Args:
            mu: 近端项系数
        """
```

### ScaffoldAggregator

```python
@register("aggregator.scaffold")
class ScaffoldAggregator(Aggregator):
    """
    SCAFFOLD 聚合器
    
    维护全局控制变量，修正客户端漂移
    """
    
    def aggregate(self, updates, global_model=None):
        """
        聚合更新和控制变量
        
        updates 需包含:
            - weights: 模型权重
            - control_variate: 客户端控制变量
        """
```

### FedNovaAggregator

```python
@register("aggregator.fednova")
class FedNovaAggregator(Aggregator):
    """
    FedNova 聚合器
    
    归一化本地更新步数
    """
```

### FedAdamAggregator

```python
@register("aggregator.fedadam")
class FedAdamAggregator(Aggregator):
    def __init__(
        self,
        lr: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        """服务端 Adam 优化器"""
```

### FedYogiAggregator

```python
@register("aggregator.fedyogi")
class FedYogiAggregator(Aggregator):
    def __init__(
        self,
        lr: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
    ):
        """服务端 Yogi 优化器"""
```

### FedBNAggregator

```python
@register("aggregator.fedbn")
class FedBNAggregator(Aggregator):
    """
    跳过 BatchNorm 层的聚合
    
    保留客户端本地 BN 统计信息
    """
    
    def aggregate(self, updates, global_model=None):
        """跳过 running_mean, running_var 等"""
```

### FedProtoAggregator

```python
@register("aggregator.fedproto")
class FedProtoAggregator(Aggregator):
    """
    原型聚合器
    
    聚合类原型而非模型参数
    """
```

---

## 学习器

### DefaultLearner

```python
@register("learner.default")
class DefaultLearner(Learner):
    def __init__(
        self,
        model,
        data=None,
        batch_size: int = 64,
        lr: float = 0.01,
        optimizer: str = "sgd",
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        device: str = "auto",
        **kwargs,
    )
```

### MOONLearner

```python
@register("learner.moon")
class MOONLearner(Learner):
    def __init__(
        self,
        model,
        data=None,
        temperature: float = 0.5,
        mu: float = 1.0,
        **kwargs,
    ):
        """
        MOON 对比学习
        
        Args:
            temperature: 对比温度
            mu: 对比损失权重
        """
```

### FedPerLearner

```python
@register("learner.fedper")
class FedPerLearner(Learner):
    def __init__(
        self,
        model,
        data=None,
        personal_layers: List[str] = None,
        **kwargs,
    ):
        """
        个性化层学习器
        
        Args:
            personal_layers: 不参与聚合的层名
        """
    
    def get_shared_weights(self):
        """只返回共享层权重"""
    
    def set_shared_weights(self, weights):
        """只设置共享层权重"""
```

### FedRepLearner

```python
@register("learner.fedrep")
class FedRepLearner(Learner):
    def __init__(
        self,
        model,
        data=None,
        head_epochs: int = 5,
        body_epochs: int = 1,
        **kwargs,
    ):
        """
        表示学习
        
        交替训练 head（分类器）和 body（特征提取器）
        """
```

### FedBABULearner

```python
@register("learner.fedbabu")
class FedBABULearner(Learner):
    def __init__(
        self,
        model,
        data=None,
        finetune_epochs: int = 5,
        freeze_body: bool = True,
        **kwargs,
    ):
        """
        冻结 body，微调 head
        """
```

### FedProtoLearner

```python
@register("learner.fedproto")
class FedProtoLearner(Learner):
    def __init__(
        self,
        model,
        data=None,
        proto_dim: int = 256,
        temperature: float = 0.5,
        **kwargs,
    ):
        """
        原型学习器
        
        计算并上传类原型
        """
    
    def compute_prototypes(self):
        """计算每个类的原型向量"""
```

---

## 持续学习

### TARGETLearner

```python
@register("learner.target")
class TARGETLearner(Learner):
    def __init__(
        self,
        model,
        data=None,
        generator_type: str = "cnn",
        synthesizer_type: str = "dfad",
        distill_weight: float = 1.0,
        **kwargs,
    )
```

### FedWEITLearner

```python
@register("learner.fedweit")
class FedWEITLearner(Learner):
    def __init__(
        self,
        model,
        data=None,
        decompose_ratio: float = 0.5,
        **kwargs,
    ):
        """
        权重分解
        
        分离任务相关和任务无关参数
        """
```

### FedKNOWLearner

```python
@register("learner.fedknow")
class FedKNOWLearner(Learner):
    def __init__(
        self,
        model,
        data=None,
        distill_weight: float = 1.0,
        temperature: float = 2.0,
        **kwargs,
    ):
        """知识蒸馏防遗忘"""
```

---

## 模型

### 内置模型

```python
@register("model.simple_cnn")
class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10): ...

@register("model.cifar10_cnn")
class Cifar10CNN(nn.Module):
    def __init__(self, num_classes: int = 10): ...

@register("model.mnist_cnn")
class MnistCNN(nn.Module):
    def __init__(self, num_classes: int = 10): ...

@register("model.resnet18")
class ResNet18(nn.Module):
    def __init__(self, num_classes: int = 10, pretrained: bool = False): ...

@register("model.mlp")
class MLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        num_classes: int,
    ): ...
```

---

## 数据集

### 内置数据集

```python
@register("dataset.mnist")
class MNISTDataset: ...

@register("dataset.cifar10")
class CIFAR10Dataset: ...

@register("dataset.cifar100")
class CIFAR100Dataset: ...

@register("dataset.fmnist")
class FashionMNISTDataset: ...
```

### 划分器

```python
@register("partitioner.iid")
class IIDPartitioner:
    def partition(self, dataset, targets) -> Dict[int, List[int]]: ...

@register("partitioner.dirichlet")
class DirichletPartitioner:
    def __init__(self, num_partitions, alpha, seed=42): ...

@register("partitioner.label_skew")
class LabelSkewPartitioner:
    def __init__(self, num_partitions, num_labels_per_client, seed=42): ...
```

---

## 下一步

- [核心 API](core-api.md)
- [自定义算法](../01-guides/custom-algorithm.md)
