# DDDR (Diffusion-Driven Data Replay) 实现说明

## 概述

本文档说明了基于FedCL框架实现的DDDR (Diffusion-Driven Data Replay) 联邦类持续学习方法。

## 核心思想

DDDR是一种联邦类持续学习方法，通过扩散模型生成历史数据来缓解灾难性遗忘问题：

1. **扩散模型数据生成**: 使用预训练的Latent Diffusion Model生成历史数据
2. **联邦类别反演**: 通过联邦训练学习每个类别的文本嵌入
3. **知识蒸馏**: 使用教师模型防止遗忘
4. **对比学习**: 使用SupConLoss增强表示能力

## 代码结构

### 1. 模型组件 (`fedcl/models/`)

从DDDR项目复制的核心模型代码：

- **`networks.py`**: 网络架构 (IncrementalNet, BaseNet)
- **`losses.py`**: 损失函数 (SupConLoss, kd_loss)
- **`data_utils.py`**: 数据工具 (GenDataset, TaskSynImageDataset, DataIter, DatasetSplit)
- **`transforms.py`**: 数据变换工具 (get_train_transform, get_test_transform, get_augmentation_transform)

### 2. 学习器 (`fedcl/methods/learners/dddr.py`)

**职责**: 只负责本地客户端训练逻辑

**主要功能**:
- 本地模型训练 (包含知识蒸馏、对比学习)
- 生成器嵌入训练
- 合成图像生成
- 数据管理

**关键方法**:
```python
async def train_epoch(self, **kwargs) -> Dict[str, Any]:
    """执行一个epoch的训练 - FedCL框架要求"""
    
def train_generator_embeddings(self):
    """训练生成器嵌入 - 本地客户端逻辑"""
    
def generate_synthetic_images(self, class_ids: list, output_dir: str):
    """生成合成图像 - 本地客户端逻辑"""
    
def set_train_data(self, train_dataset, class_ids):
    """设置训练数据"""
    
def set_synthetic_data(self, syn_imgs_dir: str, task_id: int, transform=None):
    """设置合成数据 - 数据变换由外部提供"""
```

### 3. 联邦训练器 (`fedcl/methods/trainers/dddr_federation_trainer.py`)

**职责**: 负责全局联邦训练流程

**主要功能**:
- 联邦类别反演训练
- 扩散模型数据生成
- 客户端协调
- 模型聚合

**关键方法**:
```python
async def train(self, **kwargs) -> Dict[str, Any]:
    """执行DDDR联邦训练"""

def _federated_class_inversion(self, task_id: int, class_ids: List[int]):
    """联邦类别反演训练"""

def _synthesis_images(self, inv_text_embeds, task_id: int, class_ids: List[int]):
    """合成图像"""
```

## 训练流程

### 1. 任务序列
```
任务0: 类别0-19
任务1: 类别20-39 (使用任务0的合成数据)
任务2: 类别40-59 (使用任务0,1的合成数据)
任务3: 类别60-79 (使用任务0,1,2的合成数据)
任务4: 类别80-99 (使用任务0,1,2,3的合成数据)
```

### 2. 每个任务的训练步骤

1. **准备任务数据**
   - 获取当前任务的真实数据
   - 检查是否需要生成合成数据

2. **联邦类别反演训练** (如果需要生成合成数据)
   - **直接使用现有的客户端学习器**进行生成器嵌入训练
   - 聚合嵌入权重
   - 生成合成图像

3. **联邦模型训练**
   - 各客户端使用真实数据+合成数据进行训练
   - 聚合模型权重
   - 评估全局模型

4. **任务完成处理**
   - 保存旧模型用于知识蒸馏
   - 更新已知类别数

## 架构设计原则

### 1. 职责分离
- **Learner**: 只负责本地客户端训练逻辑
- **Trainer**: 负责全局联邦训练流程
- **Data Manager**: 负责数据加载和变换
- **Transforms**: 负责数据变换逻辑

### 2. 直接使用现有客户端
在联邦类别反演训练中，**直接使用现有的客户端学习器**，而不是创建临时的：
```python
def _federated_class_inversion(self, task_id: int, class_ids: List[int]):
    """联邦类别反演训练"""
    for idx in idxs_users:
        # 直接使用现有的客户端学习器
        if idx < len(self.learners):
            learner = self.learners[idx]
            
            # 设置训练数据用于生成器训练
            if self.data_manager is not None:
                train_dataset = self.data_manager.get_client_data(idx, class_ids)
                learner.set_train_data(train_dataset, class_ids)
            
            # 训练生成器嵌入
            w = learner.train_generator_embeddings()
            if w is not None:
                local_weights.append(copy.deepcopy(w))
```

### 3. 学习器生命周期管理
- 在任务开始时创建学习器列表
- 在整个任务期间复用这些学习器
- 避免重复创建和销毁学习器

### 4. 数据变换职责分离
**改进**: 数据变换不再硬编码在学习器中，而是由数据管理器提供：

```python
# 在联邦训练器中
transform = self.data_manager.get_train_transform() if hasattr(self.data_manager, 'get_train_transform') else None
learner.set_synthetic_data(syn_imgs_dir, self._cur_task, transform)

# 在学习器中
def set_synthetic_data(self, syn_imgs_dir: str, task_id: int, transform=None):
    """设置合成数据 - 数据变换由外部提供"""
    cur_syn_dataset = TaskSynImageDataset(
        syn_imgs_dir, task_id, self.cur_size,
        transform=transform  # 使用外部提供的数据变换
    )
```

**优势**:
- 数据变换与学习器解耦
- 支持不同数据集的不同变换
- 便于扩展和维护
- 符合单一职责原则

## 配置参数

### 基础配置
```python
{
    "num_clients": 5,           # 客户端数量
    "num_tasks": 5,             # 任务数量
    "classes_per_task": 20,     # 每任务类别数
    "total_classes": 100,       # 总类别数
    "com_rounds": 100,          # 联邦轮数
    "local_epochs": 5,          # 本地训练轮数
    "batch_size": 128,          # 批处理大小
}
```

### DDDR特定配置
```python
{
    "w_kd": 10.0,               # 知识蒸馏权重
    "w_ce_pre": 0.5,            # 历史数据交叉熵权重
    "w_scl": 1.0,               # 对比学习权重
    "pre_size": 200,            # 历史数据大小
    "cur_size": 50,             # 当前数据大小
    "n_iter": 5,                # 生成迭代次数
    "com_rounds_gen": 10,       # 生成器通信轮数
    "g_local_train_steps": 50,  # 生成器本地训练步数
}
```

### 扩散模型配置
```python
{
    "ldm_config": "ldm/ldm_dddr.yaml",
    "ldm_ckpt": "models/ldm/text2img-large/model.ckpt",
}
```

## 使用方法

### 1. 安装依赖
```bash
# 安装DDDR项目的依赖
pip install -r DDDR-master/requirements.txt

# 安装CLIP
pip install git+https://github.com/openai/CLIP.git
```

### 2. 下载预训练模型
```bash
mkdir -p models/ldm/text2img-large
wget -O models/ldm/text2img-large/model.ckpt \
  https://ommer-lab.com/files/latent-diffusion/nitro/txt2img-f8-large/model.ckpt
```

### 3. 运行示例
```bash
python example_dddr_federation.py
```

## 技术特点

### 1. 联邦类别反演训练
- **直接使用现有客户端学习器**进行生成器嵌入训练
- 通过联邦聚合学习全局类别表示
- 支持差分隐私保护

### 2. 扩散模型数据生成
- 使用预训练的Latent Diffusion Model
- 通过类别反演生成高质量历史数据
- 支持多种生成参数配置

### 3. 知识蒸馏防遗忘
- 使用教师模型保存历史知识
- 通过知识蒸馏损失防止遗忘
- 支持温度参数调节

### 4. 监督对比学习
- 使用SupConLoss增强表示能力
- 提高生成数据的表示质量
- 支持多种对比学习模式

### 5. 数据变换管理
- 集中管理各种数据集的数据变换
- 支持训练、测试和增强变换
- 与学习器和训练器解耦

## 注意事项

1. **依赖要求**: 需要安装DDDR项目的完整依赖，包括LDM、CLIP等
2. **计算资源**: 扩散模型训练和生成需要大量计算资源
3. **存储空间**: 生成的合成图像需要大量存储空间
4. **数据隐私**: 联邦训练保护数据隐私，但生成的数据可能泄露信息
5. **数据变换**: 数据变换由数据管理器提供，确保与数据集匹配

## 扩展开发

### 1. 添加新的网络架构
在 `fedcl/models/networks.py` 中添加新的网络类，继承 `BaseNet`

### 2. 添加新的损失函数
在 `fedcl/models/losses.py` 中添加新的损失函数

### 3. 添加新的数据工具
在 `fedcl/models/data_utils.py` 中添加新的数据处理工具

### 4. 添加新的数据变换
在 `fedcl/models/transforms.py` 中添加新的数据集变换

### 5. 自定义训练策略
继承 `DDDRLearner` 或 `DDDRFederationTrainer` 实现自定义训练策略

## 总结

DDDR实现遵循了FedCL框架的设计原则：

- **Learner**: 只负责本地客户端训练逻辑
- **Trainer**: 负责全局联邦训练流程
- **直接使用现有客户端**: 避免创建临时学习器，提高效率
- **模型组件**: 从DDDR项目复制，保持代码一致性
- **数据变换分离**: 数据变换由专门模块管理，与学习器解耦
- **配置驱动**: 支持灵活的配置参数
- **扩展性**: 支持自定义组件和策略

这种设计使得DDDR方法能够很好地集成到FedCL框架中，同时保持了原有方法的完整性和有效性。**关键改进包括直接使用现有的客户端学习器进行生成器训练，以及将数据变换职责分离到专门的模块中**，这更符合软件工程的最佳实践。
