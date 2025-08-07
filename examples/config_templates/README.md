# FedCL 联邦学习框架配置模板

本目录提供基于 FedCL 联邦学习框架的配置模板和使用指南。FedCL 是一个功能强大的联邦持续学习框架，支持装饰器驱动的组件开发、灵活的配置管理和强大的Hook扩展系统。

## 📁 目录结构

```
config_templates/
├── README.md                           # 本文档
├── experiment_config.yaml              # 集中式配置模板（单文件包含所有配置）
└── server_client_configs/              # 分布式配置模板目录
    ├── README.md                       # 分布式配置详细说明
    ├── server_config.yaml              # 服务端配置模板
    ├── client_config_template.yaml     # 客户端配置模板（用于创建新客户端）
    ├── client_1_config.yaml            # 客户端1配置示例（默认learner）
    ├── client_2_config.yaml            # 客户端2配置示例（EWC learner）
    └── client_3_config.yaml            # 客户端3配置示例（多learner协作）
```

## 🚀 快速开始

### 1. 复制配置模板
```bash
# 复制模板到您的项目目录
cp examples/config_templates/experiment_config.yaml my_experiment_config.yaml

# 编辑配置文件
vim my_experiment_config.yaml
```

### 2. 运行实验
```bash
# 使用配置文件运行实验
python -m fedcl.experiment.experiment my_experiment_config.yaml

# 或使用Python API
python -c "
import fedcl
experiment = fedcl.FedCLExperiment('my_experiment_config.yaml')
results = experiment.run()
print(f'Final accuracy: {results.get(\"accuracy\", \"N/A\")}')
"
```

### 3. 查看结果
```bash
# 查看实验输出
ls experiments/fedcl_template_experiment/

# 查看日志
tail -f experiments/fedcl_template_experiment/logs/*.log
```

## 📖 配置文件详解

### 🔧 核心配置部分

#### 1. 实验配置 (`experiment`)
```yaml
experiment:
  name: "my_experiment"              # 实验名称（必须唯一）
  description: "实验描述"            # 实验描述
  seed: 42                          # 随机种子
  working_dir: "experiments/"       # 工作目录
  save_checkpoints: true            # 启用检查点
  checkpoint_frequency: 1           # 保存频率
```

#### 2. 数据配置 (`dataset`)
```yaml
dataset:
  name: "MNIST"                     # 数据集: MNIST/CIFAR10/自定义
  path: "data/MNIST"                # 数据路径
  type: "classification"            # 任务类型
  num_classes: 10                   # 类别数
  split_config:
    num_clients: 3                  # 客户端数量
    distribution: "iid"             # 数据分布: iid/non_iid
```

#### 3. 联邦学习配置 (`federation`)
```yaml
federation:
  num_rounds: 5                     # 训练轮次
  min_clients: 2                    # 最少参与客户端
  max_clients: 3                    # 最多参与客户端
  aggregation_strategy: "fedavg"    # 聚合策略
```

#### 4. 模型配置 (`model`)
```yaml
model:
  type: "SimpleMLP"                 # 模型类型
  input_size: 784                   # 输入维度
  hidden_sizes: [256, 128]          # 隐藏层
  num_classes: 10                   # 输出类别
```

#### 5. 训练配置 (`training`)
```yaml
training:
  local_epochs: 3                   # 本地训练轮次
  batch_size: 32                    # 批次大小
  optimizer:
    type: "SGD"                     # 优化器
    lr: 0.01                        # 学习率
    momentum: 0.9                   # 动量
```

## 🔌 Hook系统详解

FedCL 的 Hook 系统是框架的核心扩展机制，提供事件驱动的插件化架构，支持组件注册、多learner协调、特征交换等高级功能。

### 🎯 Hook系统特性

1. **事件驱动**: 基于训练生命周期的各个阶段触发
2. **组件注册**: 通过装饰器API注册learner、聚合器、评估器等组件
3. **多Learner支持**: 专门的多learner协调和特征交换机制
4. **优先级管理**: 支持Hook执行顺序控制
5. **灵活配置**: 通过YAML配置文件和装饰器双重管理
6. **内置Hook**: 提供检查点、指标收集、可视化等常用Hook
7. **自定义扩展**: 支持用户自定义Hook开发

### 📊 Hook执行阶段

#### 1. 基础执行阶段
```python
class HookPhase(Enum):
    BEFORE_EXPERIMENT = "before_experiment"    # 实验开始前
    AFTER_EXPERIMENT = "after_experiment"      # 实验结束后
    BEFORE_ROUND = "before_round"              # 联邦轮次开始前
    AFTER_ROUND = "after_round"                # 联邦轮次结束后
    BEFORE_TASK = "before_task"                # 任务开始前
    AFTER_TASK = "after_task"                  # 任务结束后
    BEFORE_EPOCH = "before_epoch"              # 训练轮开始前
    AFTER_EPOCH = "after_epoch"                # 训练轮结束后
    BEFORE_BATCH = "before_batch"              # 批次开始前
    AFTER_BATCH = "after_batch"                # 批次结束后
    ON_ERROR = "on_error"                      # 错误发生时
    ON_CHECKPOINT = "on_checkpoint"            # 检查点保存时
    ON_EVALUATION = "on_evaluation"            # 评估时
```

#### 2. 多Learner专用阶段
```python
class MultiLearnerHookPhase(Enum):
    # 初始化阶段
    MULTI_LEARNER_INIT = "multi_learner_init"
    LEARNERS_REGISTRATION = "learners_registration"
    LEARNERS_READY = "learners_ready"
    
    # 执行计划阶段
    EXECUTION_PLANNING = "execution_planning"
    PLAN_OPTIMIZATION = "plan_optimization" 
    RESOURCE_ALLOCATION = "resource_allocation"
    
    # 执行协调阶段
    BEFORE_EXECUTION_GROUP = "before_execution_group"
    AFTER_EXECUTION_GROUP = "after_execution_group"
    BEFORE_LEARNER_EXECUTION = "before_learner_execution"
    AFTER_LEARNER_EXECUTION = "after_learner_execution"
    
    # 特征交换阶段
    FEATURE_EXTRACTION = "feature_extraction"
    FEATURE_EXCHANGE = "feature_exchange"
    FEATURE_AGGREGATION = "feature_aggregation"
    FEATURE_DISTRIBUTION = "feature_distribution"
    
    # 完成阶段
    ALL_LEARNERS_COMPLETE = "all_learners_complete"
    MULTI_LEARNER_AGGREGATION = "multi_learner_aggregation"
    EXECUTION_SUMMARY = "execution_summary"
```

### 🏗️ 组件注册系统

#### 1. 装饰器API注册

**学习器注册**
```python
import fedcl

@fedcl.learner("ewc_mnist")
class EWCLearner(fedcl.BaseLearner):
    """弹性权重巩固学习器"""
    
    def __init__(self, context, config, **kwargs):
        super().__init__(context, config, **kwargs)
        self.fisher_information = {}
        self.old_params = {}
    
    def train_task(self, task_data, task_id):
        """任务训练逻辑"""
        # EWC特定的训练逻辑
        loss = self.compute_loss(predictions, targets)
        ewc_loss = self.compute_ewc_penalty()
        total_loss = loss + self.lambda_ewc * ewc_loss
        return total_loss
    
    def after_task_training(self, task_id):
        """任务训练后的处理"""
        self.compute_fisher_information()
        self.save_old_parameters()
```

**聚合器注册**
```python
@fedcl.aggregator("fedprox")
class FedProxAggregator(fedcl.BaseAggregator):
    """FedProx聚合器"""
    
    def __init__(self, context, config):
        super().__init__(context, config)
        self.mu = config.get('mu', 0.01)  # 正则化参数
    
    def aggregate(self, client_updates):
        """执行FedProx聚合"""
        aggregated_params = {}
        total_samples = sum(update['num_samples'] for update in client_updates)
        
        for param_name in client_updates[0]['params'].keys():
            weighted_sum = torch.zeros_like(client_updates[0]['params'][param_name])
            
            for update in client_updates:
                weight = update['num_samples'] / total_samples
                weighted_sum += weight * update['params'][param_name]
            
            aggregated_params[param_name] = weighted_sum
        
        return aggregated_params
```

**评估器注册**
```python
@fedcl.evaluator("continual_accuracy")
class ContinualAccuracyEvaluator(fedcl.BaseEvaluator):
    """持续学习准确率评估器"""
    
    def evaluate(self, model, test_data, context):
        """评估模型在所有已学任务上的性能"""
        task_accuracies = {}
        overall_accuracy = 0.0
        
        for task_id, task_test_data in test_data.items():
            accuracy = self.evaluate_task(model, task_test_data)
            task_accuracies[f'task_{task_id}_accuracy'] = accuracy
            overall_accuracy += accuracy
        
        overall_accuracy /= len(test_data)
        
        return {
            'overall_accuracy': overall_accuracy,
            'backward_transfer': self.compute_backward_transfer(task_accuracies),
            'forward_transfer': self.compute_forward_transfer(task_accuracies),
            **task_accuracies
        }
```

**损失函数注册**
```python
@fedcl.loss("distillation_loss")
def knowledge_distillation_loss(student_logits, teacher_logits, targets, temperature=3.0):
    """知识蒸馏损失函数"""
    soft_targets = F.softmax(teacher_logits / temperature, dim=1)
    soft_prob = F.log_softmax(student_logits / temperature, dim=1)
    
    # 软标签损失
    soft_loss = -torch.sum(soft_targets * soft_prob) / student_logits.size(0)
    
    # 硬标签损失
    hard_loss = F.cross_entropy(student_logits, targets)
    
    # 组合损失
    return 0.7 * soft_loss * (temperature ** 2) + 0.3 * hard_loss
```

**辅助模型注册**
```python
@fedcl.model("teacher_network")
class TeacherNetwork:
    """教师网络辅助模型"""
    
    def __init__(self, config=None, context=None):
        self.config = config or {}
        self.context = context
        
    def create_model(self):
        """创建预训练的教师模型"""
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        model.eval()
        
        # 返回模型和特征提取器
        feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
        
        return {
            'model': model,
            'feature_extractor': feature_extractor,
            'output_dim': model.fc.in_features
        }
```

#### 2. 配置文件注册

```yaml
# 在配置文件中注册组件
components:
  learners:
    - name: "ewc_mnist"
      class_path: "my_learners.EWCLearner"
      config:
        lambda_ewc: 0.4
        fisher_samples: 200
        
  aggregators:
    - name: "fedprox"
      class_path: "my_aggregators.FedProxAggregator"
      config:
        mu: 0.01
        
  hooks:
    - name: "distillation_hook"
      class_path: "my_hooks.DistillationHook"
      phase: "after_task"
      priority: 10
      config:
        teacher_model: "teacher_network"
        temperature: 3.0
```

### 🔄 多Learner协调机制

#### 1. 多Learner配置

```yaml
# 多learner实验配置
experiment:
  name: "multi_learner_continual"
  multi_learner:
    enabled: true
    coordination_strategy: "adaptive"
    feature_sharing: true
    
learners:
  - name: "ewc_learner"
    type: "EWCLearner" 
    tasks: [0, 1, 2]  # 负责的任务
    priority: 1
    
  - name: "si_learner"
    type: "SynapticIntelligenceLearner"
    tasks: [3, 4, 5]
    priority: 2
    
  - name: "replay_learner"
    type: "ExperienceReplayLearner"
    tasks: [0, 1, 2, 3, 4, 5]  # 所有任务
    priority: 0  # 最高优先级

# 多learner Hook配置
hooks:
  enabled: true
  
  # learner协调Hook
  learner_coordination_hook:
    enabled: true
    phase: "execution_planning"
    priority: 0
    config:
      strategy: "adaptive"
      
  # 特征交换Hook
  feature_exchange_hook:
    enabled: true
    phase: "feature_exchange"
    priority: 5
    config:
      exchange_strategy: "selective"
      feature_dependencies:
        ewc_learner: ["si_learner"]
        si_learner: ["replay_learner"]
```

#### 2. 多Learner Hook实现

**Learner协调Hook**
```python
@fedcl.hook("execution_planning", priority=0)
class LearnerCoordinationHook(Hook):
    """多learner执行协调Hook"""
    
    def __init__(self, coordination_strategy="adaptive"):
        super().__init__("execution_planning", 0)
        self.coordination_strategy = coordination_strategy
    
    def execute(self, context, **kwargs):
        learners = kwargs.get('learners', {})
        current_task = kwargs.get('current_task')
        
        if self.coordination_strategy == "priority_based":
            return self._optimize_by_priority(learners, current_task)
        elif self.coordination_strategy == "resource_based":
            return self._optimize_by_resources(learners, current_task)
        else:  # adaptive
            return self._adaptive_optimization(learners, current_task)
    
    def _adaptive_optimization(self, learners, current_task):
        """自适应优化执行计划"""
        execution_plan = {
            'primary_learner': None,
            'support_learners': [],
            'execution_order': [],
            'resource_allocation': {}
        }
        
        # 根据任务特性和learner能力决定执行计划
        for learner_id, learner in learners.items():
            if current_task in learner.config.get('tasks', []):
                if learner.config.get('priority', 0) == 0:
                    execution_plan['primary_learner'] = learner_id
                else:
                    execution_plan['support_learners'].append(learner_id)
        
        return execution_plan
```

**特征交换Hook**
```python
@fedcl.hook("feature_exchange", priority=5)
class FeatureExchangeHook(Hook):
    """learner间特征交换Hook"""
    
    def __init__(self, exchange_strategy="selective"):
        super().__init__("feature_exchange", 5)
        self.exchange_strategy = exchange_strategy
        self.feature_cache = {}
    
    def execute(self, context, **kwargs):
        learners = kwargs.get('learners', {})
        execution_results = kwargs.get('execution_results', {})
        
        # 收集各learner的特征
        features = self._collect_features(execution_results)
        
        # 执行特征交换
        if self.exchange_strategy == "selective":
            return self._selective_exchange(features, learners, context)
        else:
            return self._broadcast_exchange(features, learners, context)
    
    def _selective_exchange(self, features, learners, context):
        """选择性特征交换"""
        exchanges = []
        
        # 根据预定义的依赖关系交换特征
        dependencies = context.config.get('feature_dependencies', {})
        
        for source_learner, target_learners in dependencies.items():
            if source_learner in features:
                source_features = features[source_learner]
                
                for target_learner in target_learners:
                    if target_learner in learners:
                        # 共享特征到目标learner
                        context.share_features(
                            source_learner, 
                            source_features, 
                            target_learner
                        )
                        
                        exchanges.append({
                            'source': source_learner,
                            'target': target_learner,
                            'feature_type': 'intermediate_representations'
                        })
        
        return {'exchanges': exchanges, 'strategy': 'selective'}
```

### 🏗️ Hook配置示例

#### 1. 基础Hook配置
```yaml
hooks:
  enabled: true                              # 启用Hook系统
  
  # 检查点Hook - 用于模型状态保存
  checkpoint_hook:
    enabled: true                            # 启用检查点Hook
    phase: "after_round"                     # 在每轮结束后执行
    priority: 0                              # 最高优先级

  # 多learner协调Hook
  learner_coordination_hook:
    enabled: true
    phase: "execution_planning"
    priority: 0
    config:
      coordination_strategy: "adaptive"
      
  # 特征交换Hook
  feature_exchange_hook:
    enabled: true
    phase: "feature_exchange"
    priority: 5
    config:
      exchange_strategy: "selective"
```

#### 2. 内置Hook详解

**检查点Hook (CheckpointHook)**
```yaml
checkpoint:
  enabled: true                              # 启用检查点
  save_frequency: 1                          # 每轮保存
  save_model: true                           # 保存模型参数
  save_optimizer: true                       # 保存优化器状态
  checkpoint_dir: "checkpoints/"             # 保存目录
  max_checkpoints: 3                         # 最大保留数量
  naming_pattern: "checkpoint_round_{round}_epoch_{epoch}"
```

**指标收集Hook (MetricsHook)**
```yaml
metrics_hook:
  enabled: true                              # 启用指标收集
  phase: "after_evaluation"                  # 评估后执行
  priority: 5                                # 中等优先级
  config:
    track_loss: true                         # 跟踪损失
    track_accuracy: true                     # 跟踪准确率
    save_to_file: true                       # 保存到文件
    track_continual_metrics: true            # 跟踪持续学习指标
```

**TensorBoard Hook**
```yaml
tensorboard_hook:
  enabled: true                              # 启用TensorBoard
  phase: "after_epoch"                       # 每轮后执行
  priority: 10                               # 较低优先级
  config:
    log_dir: "runs/"                         # TensorBoard日志目录
    log_images: false                        # 是否记录图像
    log_histograms: true                     # 记录参数分布
    log_learner_metrics: true                # 记录各learner指标
```

**Weights & Biases Hook**
```yaml
wandb_hook:
  enabled: false                             # 默认关闭
  phase: "after_round"                       # 轮次后执行
  priority: 20                               # 低优先级
  config:
    project: "fedcl_continual_learning"      # WandB项目名
    entity: "your_team"                      # 团队名称
    tags: ["federated_learning", "continual", "multi_learner"]  # 实验标签
    log_multi_learner_metrics: true          # 记录多learner指标
```

#### 3. 自定义Hook开发

**定义自定义Hook**
```python
import fedcl
from fedcl.core.hook import Hook
from fedcl.core.execution_context import ExecutionContext

@fedcl.hook("after_task", priority=15)
class CustomAnalysisHook(Hook):
    """自定义分析Hook"""
    
    def __init__(self, config=None):
        super().__init__(
            phase="after_task",
            priority=15,
            name="CustomAnalysisHook"
        )
        self.config = config or {}
        self.analysis_results = {}
    
    def execute(self, context: ExecutionContext, **kwargs):
        """Hook执行逻辑"""
        # 获取当前任务信息
        task_id = kwargs.get('task_id')
        model = kwargs.get('model')
        task_results = kwargs.get('task_results')
        
        # 执行自定义分析
        analysis = self.analyze_task_performance(model, task_results)
        
        # 存储分析结果
        self.analysis_results[task_id] = analysis
        
        # 更新执行上下文
        context.set_analysis_results(self.analysis_results)
        
        logger.info(f"Task {task_id} analysis completed: {analysis}")
        
        return analysis
    
    def analyze_task_performance(self, model, task_results):
        """分析任务性能"""
        return {
            'accuracy': task_results.get('accuracy', 0),
            'loss': task_results.get('loss', float('inf')),
            'model_complexity': self.compute_model_complexity(model),
            'forgetting_measure': self.compute_forgetting(task_results)
        }
```

**多Learner专用Hook**
```python
@fedcl.hook("learners_registration", priority=0)
class LearnerRegistrationHook(Hook):
    """Learner注册管理Hook"""
    
    def execute(self, context, **kwargs):
        """管理learner注册过程"""
        learner_configs = kwargs.get('learner_configs', [])
        registered_learners = {}
        
        for config in learner_configs:
            learner_id = config['name']
            learner_type = config['type']
            
            # 从注册表获取learner类
            learner_class = context.registry.get_learner(learner_type)
            
            # 创建learner实例
            learner = learner_class(context, config.get('config', {}))
            
            # 注册到上下文
            registered_learners[learner_id] = learner
            
            logger.info(f"Registered learner: {learner_id} ({learner_type})")
        
        # 更新上下文
        context.set_learners(registered_learners)
        
        return registered_learners

@fedcl.hook("feature_aggregation", priority=5) 
class FeatureAggregationHook(Hook):
    """特征聚合Hook"""
    
    def execute(self, context, **kwargs):
        """聚合多个learner的特征"""
        learner_features = kwargs.get('learner_features', {})
        aggregation_strategy = self.config.get('strategy', 'average')
        
        if aggregation_strategy == 'average':
            return self._average_aggregation(learner_features)
        elif aggregation_strategy == 'weighted':
            return self._weighted_aggregation(learner_features, context)
        else:
            return self._attention_aggregation(learner_features)
    
    def _weighted_aggregation(self, learner_features, context):
        """加权特征聚合"""
        # 根据learner性能计算权重
        learner_weights = {}
        for learner_id in learner_features.keys():
            performance = context.get_learner_performance(learner_id)
            learner_weights[learner_id] = performance.get('accuracy', 0.5)
        
        # 归一化权重
        total_weight = sum(learner_weights.values())
        learner_weights = {k: v/total_weight for k, v in learner_weights.items()}
        
        # 加权聚合
        aggregated_features = None
        for learner_id, features in learner_features.items():
            weight = learner_weights[learner_id]
            if aggregated_features is None:
                aggregated_features = weight * features
            else:
                aggregated_features += weight * features
        
        return aggregated_features
```

**注册自定义Hook的多种方式**

**方法1：装饰器注册（推荐）**
```python
import fedcl

@fedcl.hook("after_evaluation", priority=15)
def custom_metrics_hook(context, **kwargs):
    """函数式Hook"""
    metrics = kwargs.get('metrics', {})
    
    # 计算自定义指标
    custom_metrics = {
        'accuracy_improvement': metrics.get('accuracy', 0) - context.get_previous_accuracy(),
        'loss_reduction': context.get_previous_loss() - metrics.get('loss', 0),
        'stability_score': compute_stability_score(metrics)
    }
    
    # 更新上下文
    context.update_metrics(custom_metrics)
    
    logger.info(f"Custom metrics: {custom_metrics}")
    return custom_metrics

# 类式Hook注册
@fedcl.hook("before_round", priority=5)
class DataAugmentationHook(Hook):
    """数据增强Hook"""
    
    def execute(self, context, **kwargs):
        training_data = kwargs.get('training_data')
        
        # 应用数据增强
        augmented_data = self.apply_augmentation(training_data)
        
        # 更新训练数据
        context.set_training_data(augmented_data)
        
        return augmented_data
```

**方法2：配置文件注册**
```yaml
# 在配置文件中添加自定义Hook
hooks:
  enabled: true
  
  custom_hooks:
    - name: "CustomAnalysisHook"
      class_path: "my_hooks.CustomAnalysisHook"
      enabled: true
      phase: "after_task"
      priority: 15
      config:
        analysis_type: "comprehensive"
        save_plots: true
        output_file: "analysis_results.json"
    
    - name: "LearnerCoordinationHook"
      class_path: "my_hooks.LearnerCoordinationHook"
      enabled: true
      phase: "execution_planning"
      priority: 0
      config:
        coordination_strategy: "adaptive"
        resource_constraints:
          max_memory: "8GB"
          max_compute_time: 300
```

**方法3：运行时动态注册**
```python
# 在Python代码中动态注册
import fedcl

def setup_custom_hooks(experiment):
    """设置自定义Hook"""
    
    # 注册分析Hook
    analysis_hook = CustomAnalysisHook({
        'analysis_type': 'detailed',
        'save_plots': True
    })
    experiment.register_hook(analysis_hook)
    
    # 注册多learner协调Hook
    coordination_hook = LearnerCoordinationHook({
        'strategy': 'resource_aware'
    })
    experiment.register_hook(coordination_hook)
    
    # 注册条件Hook
    def conditional_checkpoint_hook(context, **kwargs):
        # 只有在准确率提升时才保存检查点
        current_accuracy = kwargs.get('metrics', {}).get('accuracy', 0)
        previous_accuracy = context.get_previous_accuracy()
        
        if current_accuracy > previous_accuracy:
            context.save_checkpoint()
            logger.info("Checkpoint saved due to accuracy improvement")
    
    # 注册条件Hook
    experiment.register_hook(
        fedcl.Hook("after_evaluation", 5, "ConditionalCheckpointHook"),
        conditional_checkpoint_hook
    )
```

### 🔄 Hook执行流程

Hook系统的执行流程如下：

1. **Hook注册阶段**: 
   - 框架启动时扫描装饰器注册的Hook
   - 解析配置文件中的Hook定义
   - 创建Hook实例并注册到HookExecutor

2. **阶段触发阶段**:
   - 在相应的执行阶段触发对应的Hook
   - HookExecutor按优先级排序所有Hook
   - 依次执行每个Hook的execute方法

3. **上下文传递阶段**:
   - Hook通过ExecutionContext获取当前状态
   - Hook可以修改ExecutionContext中的数据
   - 修改后的数据传递给后续Hook和框架组件

4. **错误处理阶段**:
   - Hook执行失败时记录错误日志
   - 根据错误处理策略决定是否继续执行
   - 提供Hook级别的错误恢复机制

```python
# Hook执行流程示例
class HookExecutor:
    def execute_hooks(self, phase: str, context: ExecutionContext, **kwargs):
        """执行指定阶段的所有Hook"""
        hooks = self.get_hooks_for_phase(phase)
        
        # 按优先级排序（数字越小优先级越高）
        hooks.sort(key=lambda h: h.priority)
        
        results = []
        for hook in hooks:
            try:
                # 检查Hook是否应该执行
                if hook.should_execute(context, **kwargs):
                    # 执行Hook
                    result = hook.execute(context, **kwargs)
                    results.append(result)
                    
                    # 更新执行统计
                    hook.execution_count += 1
                    
            except Exception as e:
                # 错误处理
                logger.error(f"Hook {hook.name} failed: {e}")
                self.handle_hook_error(hook, e, context)
        
        return results
```

### 📈 Hook使用最佳实践

#### 1. 性能优化
```yaml
# 合理设置Hook优先级
hooks:
  checkpoint_hook:
    priority: 0        # 关键操作最高优先级
  
  learner_coordination_hook:
    priority: 1        # 协调操作次之
    
  feature_exchange_hook:  
    priority: 5        # 特征交换中等优先级
    
  tensorboard_hook:
    priority: 10       # 可视化较低优先级
    
  custom_analysis_hook:
    priority: 20       # 分析操作最低优先级
```

#### 2. 错误处理
```python
class RobustHook(Hook):
    """错误处理示例Hook"""
    
    def execute(self, context, **kwargs):
        try:
            # Hook核心逻辑
            result = self.core_logic(context, **kwargs)
            return result
            
        except CriticalError as e:
            # 关键错误，需要中断实验
            logger.error(f"Critical error in {self.name}: {e}")
            context.set_error_state(e)
            raise
            
        except RecoverableError as e:
            # 可恢复错误，记录但继续执行
            logger.warning(f"Recoverable error in {self.name}: {e}")
            return self.get_fallback_result()
            
        except Exception as e:
            # 未知错误，安全处理
            logger.error(f"Unexpected error in {self.name}: {e}")
            return None  # 返回安全默认值
```

#### 3. 资源管理
```python
class ResourceAwareHook(Hook):
    """资源感知Hook"""
    
    def execute(self, context, **kwargs):
        # 检查系统资源
        if not self.check_resources():
            logger.warning(f"Insufficient resources for {self.name}, skipping")
            return None
        
        # 检查时间限制
        if context.is_timeout_approaching():
            logger.info(f"Timeout approaching, executing lightweight version")
            return self.lightweight_execution(context, **kwargs)
        
        # 正常执行
        return self.full_execution(context, **kwargs)
    
    def check_resources(self):
        """检查系统资源是否充足"""
        import psutil
        
        # 检查内存使用率
        memory_usage = psutil.virtual_memory().percent
        if memory_usage > 90:
            return False
        
        # 检查CPU使用率  
        cpu_usage = psutil.cpu_percent(interval=1)
        if cpu_usage > 95:
            return False
            
        return True
```

#### 4. Hook间通信
```python
class CommunicatingHook(Hook):
    """Hook间通信示例"""
    
    def execute(self, context, **kwargs):
        # 从其他Hook获取数据
        analysis_results = context.get_hook_data('CustomAnalysisHook')
        coordination_plan = context.get_hook_data('LearnerCoordinationHook')
        
        # 基于其他Hook的结果执行逻辑
        if analysis_results and analysis_results.get('accuracy') > 0.9:
            # 高准确率时的特殊处理
            result = self.high_accuracy_processing(coordination_plan)
        else:
            # 常规处理
            result = self.normal_processing()
        
        # 共享数据给其他Hook
        context.set_hook_data(self.name, result)
        
        return result
```

### 🔄 Hook执行流程

1. **Hook注册**: 框架启动时注册所有启用的Hook
2. **阶段触发**: 在相应阶段触发对应的Hook
3. **优先级排序**: 按优先级顺序执行Hook（数字越小优先级越高）
4. **上下文传递**: Hook通过ExecutionContext获取和修改状态
5. **错误处理**: Hook执行失败时的错误恢复机制

### 📈 Hook使用最佳实践

#### 1. 性能优化
```yaml
# 合理设置Hook优先级
checkpoint_hook:
  priority: 0        # 关键操作优先
tensorboard_hook:
  priority: 10       # 可视化次之
custom_analysis_hook:
  priority: 20       # 分析最后
```

#### 2. 错误处理
```python
class RobustHook(Hook):
    def execute(self, context):
        try:
            # Hook逻辑
            pass
        except Exception as e:
            logger.error(f"Hook {self.name} failed: {e}")
            # 不要抛出异常，避免影响训练
```

#### 3. 资源管理
```python
class ResourceAwareHook(Hook):
    def execute(self, context):
        # 检查资源状态
        if context.should_save_checkpoint():
            # 执行资源密集操作
            pass
        else:
            # 跳过或简化操作
            pass
```

## 📊 实验监控与日志

### 日志系统特性

1. **分层日志**: 支持DEBUG/INFO/WARNING/ERROR多级别
2. **客户端标识**: 在日志中自动标记客户端ID，便于调试
3. **结构化输出**: JSON格式的结构化日志支持
4. **实时监控**: 支持实时日志流监控

### 日志配置示例

```yaml
experiment:
  logging:
    level: "INFO"                            # 全局日志级别
    log_client_training: true                # 客户端训练日志标识
    log_to_file: true                        # 文件输出
    log_to_console: true                     # 控制台输出
    log_dir: "logs/"                         # 日志目录
```

### 日志查看命令

```bash
# 实时查看所有日志
tail -f experiments/*/logs/*.log

# 过滤客户端训练日志
grep "客户端\[" experiments/*/logs/*.log

# 查看错误日志
grep "ERROR" experiments/*/logs/*.log

# 查看Hook执行日志
grep "Hook" experiments/*/logs/*.log
```

## 🔍 故障排查

### 常见问题

1. **配置文件格式错误**
   ```bash
   # 验证YAML格式
   python -c "import yaml; yaml.safe_load(open('config.yaml'))"
   ```

2. **路径问题**
   ```yaml
   # 使用绝对路径或相对于工作目录的路径
   paths:
     data_path: "data/MNIST"              # 相对路径
     output_path: "/absolute/path/output" # 绝对路径
   ```

3. **Hook执行失败**
   ```bash
   # 查看Hook相关日志
   grep -A 5 -B 5 "Hook.*failed" logs/*.log
   ```

4. **内存不足**
   ```yaml
   # 减少批次大小
   training:
     batch_size: 16    # 从32减少到16
   
   # 减少检查点保存频率
   checkpoint:
     save_frequency: 5  # 每5轮保存一次
   ```

### 调试技巧

1. **启用详细日志**
   ```yaml
   experiment:
     logging:
       level: "DEBUG"   # 开启DEBUG级别
   ```

2. **Hook调试**
   ```python
   # 在自定义Hook中添加调试信息
   def execute(self, context):
       logger.debug(f"Hook {self.name} executing with context: {context}")
   ```

3. **性能分析**
   ```bash
   # 查看Hook执行时间
   grep "execution_time" logs/*.log
   ```

## 📚 参考资源

### 官方文档
- [FedCL GitHub仓库](https://github.com/UPC518/MOE-FedCL)
- [API文档](docs/)
- [示例代码](examples/)

### 配置模板
- [集中式配置模板](experiment_config.yaml) - 单文件包含所有配置，适合快速实验
- [分布式配置模板](server_client_configs/) - 服务端-客户端分离，适合真实部署
  - [服务端配置](server_client_configs/server_config.yaml)
  - [客户端配置模板](server_client_configs/client_config_template.yaml)
  - [客户端1配置](server_client_configs/client_1_config.yaml) - 默认learner
  - [客户端2配置](server_client_configs/client_2_config.yaml) - EWC learner
  - [客户端3配置](server_client_configs/client_3_config.yaml) - 多learner协作
  - [分布式配置说明](server_client_configs/README.md)
- [实际测试配置](../../tests/configs/mnist_real_test/)

## 🏗️ 配置架构选择

### 集中式 vs 分布式配置

| 特性 | 集中式配置 | 分布式配置 |
|------|------------|------------|
| **配置文件** | `experiment_config.yaml` | `server_config.yaml` + `client_*_config.yaml` |
| **适用场景** | 快速原型、算法验证、单机测试 | 真实部署、异构环境、多learner协作 |
| **管理复杂度** | 低 | 中等 |
| **异构客户端** | ❌ | ✅ |
| **个性化配置** | ❌ | ✅ |
| **扩展性** | 低 | 高 |

**选择建议:**
- 🔬 **算法研究** → 使用集中式配置
- 🌐 **真实部署** → 使用分布式配置
- 🤝 **多Learner实验** → 使用分布式配置

## 📋 完整配置示例

### 单Learner持续学习实验
```yaml
# experiment_config.yaml
experiment:
  name: "single_learner_continual_mnist" 
  description: "单一learner的MNIST持续学习实验"
  output_dir: "experiments/single_learner_test"
  
  date_id: "20250805_120000"
  save_config: true
  save_logs: true

# 数据配置
data:
  dataset: "MNIST"
  dataset_path: "./data/MNIST"
  
  # 任务序列定义
  task_sequence:
    num_tasks: 5
    task_type: "split"
    split_method: "class_based"
    classes_per_task: 2
    
  # 数据预处理
  preprocessing:
    normalize: true
    augmentation:
      enabled: true
      methods: ["rotation", "translation"]

# 联邦学习配置
federation:
  num_clients: 3
  participation_rate: 1.0
  
  client_data:
    distribution: "iid"  # 或 "non_iid"
    samples_per_client: 1000
    
  communication:
    rounds: 10
    local_epochs: 5

# 模型配置
model:
  type: "simple_cnn"
  input_shape: [1, 28, 28]
  num_classes: 10
  
  architecture:
    conv_layers: 2
    hidden_dims: [128, 64]
    dropout: 0.2

# Learner配置  
learners:
  - name: "ewc_learner"
    type: "EWCLearner"
    config:
      lambda_ewc: 0.4
      sample_size: 200
      
# 训练配置
training:
  optimizer: "adam"
  learning_rate: 0.001
  batch_size: 32
  
  loss:
    type: "cross_entropy"
    
  metrics:
    - "accuracy"
    - "loss" 
    - "forgetting"

# 评估配置
evaluation:
  interval: 1  # 每轮评估
  metrics:
    - "accuracy"
    - "backward_transfer"
    - "forward_transfer"
    
  test_tasks: "all"  # 评估所有已学习任务

# 检查点配置
checkpoint:
  enabled: true
  save_interval: 5
  save_best: true
  save_last: true

# Hook配置
hooks:
  enabled: true
  
  # 基本Hook
  checkpoint_hook:
    enabled: true
    priority: 0
    
  evaluation_hook:
    enabled: true
    priority: 5
    
  tensorboard_hook:
    enabled: true
    priority: 10
    config:
      log_dir: "runs"
      
  # 自定义Hook
  custom_hooks:
    - name: "PerformanceAnalysisHook"
      class_path: "my_hooks.PerformanceAnalysisHook"
      enabled: true
      phase: "after_evaluation"
      priority: 15
      config:
        save_plots: true
        output_file: "performance_analysis.json"

# 日志配置
logging:
  level: "INFO"
  save_to_file: true
  include_client_id: true  # 区分客户端日志
  
  formatters:
    default: "[%(asctime)s][%(name)s][%(levelname)s] %(message)s"
    client: "[%(asctime)s][Client-%(client_id)s][%(name)s][%(levelname)s] %(message)s"
```

### 多Learner协作实验
```yaml
# experiment_config.yaml  
experiment:
  name: "multi_learner_collaboration_mnist"
  description: "多learner协作的MNIST持续学习实验"
  output_dir: "experiments/multi_learner_test"

# 数据配置（同上）
data:
  dataset: "MNIST"
  dataset_path: "./data/MNIST"
  task_sequence:
    num_tasks: 5
    task_type: "split"
    classes_per_task: 2

# 联邦学习配置（同上）
federation:
  num_clients: 5
  participation_rate: 0.8

# 模型配置 - 支持多种模型
model:
  models:
    - name: "cnn_model"
      type: "simple_cnn"
      input_shape: [1, 28, 28]
      num_classes: 10
      
    - name: "resnet_model" 
      type: "resnet18"
      input_shape: [1, 28, 28]
      num_classes: 10

# 多Learner配置
learners:
  - name: "ewc_learner"
    type: "EWCLearner"
    model: "cnn_model"
    config:
      lambda_ewc: 0.4
      sample_size: 200
      
  - name: "mas_learner"
    type: "MASLearner" 
    model: "resnet_model"
    config:
      lambda_mas: 0.1
      accumulate_gradients: true
      
  - name: "replay_learner"
    type: "ReplayLearner"
    model: "cnn_model"
    config:
      buffer_size: 500
      replay_batch_size: 16

# 多Learner协调配置
multi_learner:
  coordination:
    enabled: true
    strategy: "feature_exchange"  # 或 "ensemble", "distillation"
    
    # 特征交换配置
    feature_exchange:
      frequency: 2  # 每2轮交换一次
      layer_names: ["fc1", "fc2"]  # 交换的层
      aggregation: "weighted_average"  # 聚合方式
      
  # Learner权重分配
  learner_weights:
    ewc_learner: 0.4
    mas_learner: 0.4  
    replay_learner: 0.2

# Hook配置 - 包含多Learner专用Hook
hooks:
  enabled: true
  
  # 多Learner Hook
  learner_coordination_hook:
    enabled: true
    priority: 0
    config:
      coordination_strategy: "adaptive"
      resource_aware: true
      
  feature_exchange_hook:
    enabled: true
    priority: 1
    config:
      exchange_frequency: 2
      aggregation_method: "attention"
      
  ensemble_evaluation_hook:
    enabled: true
    priority: 5
    config:
      voting_strategy: "weighted"
      confidence_threshold: 0.8
      
  # 其他Hook
  checkpoint_hook:
    enabled: true
    priority: 0
    
  tensorboard_hook:
    enabled: true
    priority: 10
    config:
      log_multi_learner: true  # 记录每个learner的指标

# 日志配置 - 支持多Learner区分
logging:
  level: "INFO"
  save_to_file: true
  include_client_id: true
  include_learner_id: true  # 区分learner日志
  
  formatters:
    multi_learner: "[%(asctime)s][Client-%(client_id)s][Learner-%(learner_id)s][%(name)s][%(levelname)s] %(message)s"
```

## 🚀 快速开始指南

### 1. 环境准备
```bash
# 克隆项目
git clone <repository_url>
cd Moe-Fedcl

# 安装依赖
pip install -r requirements.txt

# 或使用uv（推荐）
uv sync
```

### 2. 运行基础实验
```bash
# 复制配置模板
cp examples/config_templates/experiment_config.yaml my_experiment_config.yaml

# 编辑配置文件（根据需要修改）
vim my_experiment_config.yaml

# 运行实验
python main.py --config my_experiment_config.yaml
```

### 3. 常见使用场景

#### 场景1：MNIST分类持续学习
```yaml
# 基础MNIST配置
data:
  dataset: "MNIST"
  task_sequence:
    num_tasks: 5
    classes_per_task: 2
    
learners:
  - name: "ewc_learner"
    type: "EWCLearner"
    config:
      lambda_ewc: 0.4
```

#### 场景2：多客户端联邦学习
```yaml
# 多客户端配置
federation:
  num_clients: 10
  participation_rate: 0.8
  
  client_data:
    distribution: "non_iid"
    alpha: 0.5  # Dirichlet分布参数
```

#### 场景3：多Learner集成
```yaml
# 多learner配置
learners:
  - name: "ewc_learner"
    type: "EWCLearner"
  - name: "replay_learner" 
    type: "ReplayLearner"
    
multi_learner:
  coordination:
    strategy: "ensemble"
```

### 4. 调试与监控

#### 启用详细日志
```yaml
logging:
  level: "DEBUG"
  include_client_id: true
  include_learner_id: true
```

#### 使用TensorBoard
```yaml
hooks:
  tensorboard_hook:
    enabled: true
    config:
      log_dir: "runs"
      log_multi_learner: true
```

#### 性能监控
```yaml
hooks:
  resource_monitoring_hook:
    enabled: true
    config:
      monitor_memory: true
      monitor_gpu: true
```

## 🔧 故障排除

### 常见问题

#### 1. 内存不足
**问题**: 运行多learner实验时内存溢出
**解决**: 
```yaml
# 减少batch size
training:
  batch_size: 16  # 从32减少到16

# 启用梯度检查点
model:
  gradient_checkpointing: true

# 限制并发learner
multi_learner:
  max_concurrent_learners: 2
```

#### 2. Hook执行错误
**问题**: 自定义Hook导致实验中断
**解决**:
```yaml
# 禁用有问题的Hook
hooks:
  custom_hooks:
    - name: "ProblematicHook"
      enabled: false

# 或设置错误处理策略
hooks:
  error_handling: "continue"  # 继续执行其他Hook
```

#### 3. 配置文件格式错误
**问题**: YAML格式错误导致配置解析失败
**解决**:
```bash
# 验证YAML格式
python -c "import yaml; yaml.safe_load(open('my_config.yaml'))"

# 使用配置验证工具
python -m fedcl.config.validator my_config.yaml
```

### 性能优化建议

#### 1. 数据加载优化
```yaml
data:
  dataloader:
    num_workers: 4
    pin_memory: true
    prefetch_factor: 2
```

#### 2. Hook优先级优化
```yaml
hooks:
  # 关键Hook使用低数字（高优先级）
  checkpoint_hook:
    priority: 0
    
  # 非关键Hook使用高数字（低优先级）  
  visualization_hook:
    priority: 20
```

#### 3. 模型并行化
```yaml
model:
  parallel:
    enabled: true
    devices: [0, 1]  # 使用多GPU
```

## 📖 总结

通过本文档，您应该能够：

1. ✅ **理解FedCL框架架构**：掌握联邦学习、持续学习、Hook系统的核心概念
2. ✅ **配置自己的实验**：根据需求选择合适的learner、模型、Hook组合
3. ✅ **开发自定义组件**：实现自己的learner、Hook、聚合器等组件
4. ✅ **监控和调试实验**：使用日志、TensorBoard、性能监控等工具
5. ✅ **优化实验性能**：合理配置资源、优先级、并行化等参数

### 社区资源
- 提交Issue: [GitHub Issues](https://github.com/UPC518/MOE-FedCL/issues)
- 讨论交流: [GitHub Discussions](https://github.com/UPC518/MOE-FedCL/discussions)

---

**💡 提示**: 建议从基础配置开始，逐步添加高级功能。Hook系统提供了强大的扩展能力，但应根据实际需求谨慎使用，避免过度复杂化。

如有任何问题，请参考项目文档或提交Issue。祝您实验顺利！ 🎉
