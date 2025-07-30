# AI代码实现Prompt模版（简化版）


```
你是一个专业的Python开发工程师，正在开发FedCL（联邦持续学习）框架。请根据以下详细规划实现指定的类。

## 项目背景
FedCL是一个分布式联邦学习框架，支持持续学习算法。项目采用模块化设计，使用依赖注入和配置驱动的架构。

当前项目结构：
执行tree fedcl 查看


！！！需要实现的功能：


### Step 8.1: 基础联邦组件

**文档输入:**
- `联邦学习核心.md` - 联邦组件设计
- `服务器聚合详细流程图.md` - 聚合流程

注意！！！需要支持实现伪联邦（单机多个客户端）
整个系统存在两种模式：真联邦和伪联邦

#### LocalTrainer 实现

**必须实现的接口:**
```python
class LocalTrainer:
    def __init__(self, learner: BaseLearner, config: DictConfig)
    
    def train_epoch(self, model: torch.nn.Module, task_data: DataLoader) -> Dict[str, float]:
        """训练一个轮次"""
        pass
    
    def evaluate_model(self, model: torch.nn.Module, test_data: DataLoader) -> Dict[str, float]:
        """评估模型"""
        pass
    
    def compute_model_update(self, old_model: torch.nn.Module, 
                            new_model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        """计算模型更新"""
        pass
    
    def apply_model_update(self, model: torch.nn.Module, 
                          update: Dict[str, torch.Tensor]) -> torch.nn.Module:
        """应用模型更新"""
        pass
    
    def get_model_parameters(self, model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        """获取模型参数"""
        pass
    
    def set_model_parameters(self, model: torch.nn.Module, 
                            parameters: Dict[str, torch.Tensor]) -> None:
        """设置模型参数"""
        pass
    
    def compute_gradient_norms(self, model: torch.nn.Module) -> Dict[str, float]:
        """计算梯度范数"""
        pass
    
    def get_training_stats(self) -> Dict[str, Any]:
        """获取训练统计"""
        pass
    
    def reset_optimizer(self) -> None:
        """重置优化器"""
        pass
    
    def save_training_state(self) -> Dict[str, Any]:
        """保存训练状态"""
        pass
    
    def load_training_state(self, state: Dict[str, Any]) -> None:
        """加载训练状态"""
        pass
```

#### ModelManager 实现

**必须实现的接口:**
```python
class ModelManager:
    def __init__(self, config: DictConfig, aggregator: BaseAggregator)
    
    def update_global_model(self, client_updates: List[Dict[str, torch.Tensor]]) -> torch.nn.Module:
        """更新全局模型"""
        pass
    
    def get_current_model(self) -> torch.nn.Module:
        """获取当前模型"""
        pass
    
    def set_global_model(self, model: torch.nn.Module) -> None:
        """设置全局模型"""
        pass
    
    def save_checkpoint(self, round_id: int, additional_info: Dict[str, Any] = None) -> Path:
        """保存检查点"""
        pass
    
    def load_checkpoint(self, checkpoint_path: Path) -> torch.nn.Module:
        """加载检查点"""
        pass
    
    def get_model_diff(self, old_model: torch.nn.Module, 
                      new_model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        """获取模型差异"""
        pass
    
    def apply_model_diff(self, model: torch.nn.Module, 
                        diff: Dict[str, torch.Tensor]) -> torch.nn.Module:
        """应用模型差异"""
        pass
    
    def get_model_size(self, model: torch.nn.Module) -> int:
        """获取模型大小"""
        pass
    
    def compress_model(self, model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        """压缩模型"""
        pass
    
    def decompress_model(self, compressed_model: Dict[str, torch.Tensor]) -> torch.nn.Module:
        """解压模型"""
        pass
    
    def get_model_history(self) -> List[Dict[str, Any]]:
        """获取模型历史"""
        pass
    
    def validate_model_update(self, update: Dict[str, torch.Tensor]) -> bool:
        """验证模型更新"""
        pass
```

#### ClientManager 实现

**必须实现的接口:**
```python
class ClientManager:
    def __init__(self, config: DictConfig)
    
    def register_client(self, client_id: str, client_info: Dict[str, Any]) -> bool:
        """注册客户端"""
        pass
    
    def unregister_client(self, client_id: str) -> bool:
        """注销客户端"""
        pass
    
    def get_active_clients(self) -> List[str]:
        """获取活跃客户端"""
        pass
    
    def select_clients_for_round(self, num_clients: int, round_id: int) -> List[str]:
        """选择参与轮次的客户端"""
        pass
    
    def update_client_status(self, client_id: str, status: ClientStatus) -> None:
        """更新客户端状态"""
        pass
    
    def get_client_info(self, client_id: str) -> Dict[str, Any]:
        """获取客户端信息"""
        pass
    
    def broadcast_to_clients(self, message: Any, targets: List[str]) -> Dict[str, bool]:
        """向客户端广播消息"""
        pass
    
    def collect_from_clients(self, data_type: str, sources: List[str], 
                            timeout: float = 60.0) -> Dict[str, Any]:
        """从客户端收集数据"""
        pass
    
    def get_client_statistics(self, client_id: str) -> Dict[str, Any]:
        """获取客户端统计信息"""
        pass
    
    def set_client_selection_strategy(self, strategy: str) -> None:
        """设置客户端选择策略"""
        pass
    
    def handle_client_failure(self, client_id: str, error: Exception) -> None:
        """处理客户端故障"""
        pass
```

**ClientStatus 枚举:**
```python
from enum import Enum

class ClientStatus(Enum):
    DISCONNECTED = "disconnected"
    CONNECTED = "connected"
    TRAINING = "training"
    READY = "ready"
    ERROR = "error"
    TIMEOUT = "timeout"
```



## 期望输出格式

### 1. 主实现文件
```python
# fedcl/{module_path}/{file_name}.py
"""
模块文档说明
"""
from typing import Union, Dict, Any
from pathlib import Path
from loguru import logger

class {ClassName}:
    """类文档说明"""
    
    def __init__(self, ...):
        """初始化方法"""
        pass
    
    def method_name(self, ...) -> ReturnType:
        """方法文档说明"""
        pass
```

### 2. 测试文件
```python
# tests/unit/{test_path}/test_{file_name}.py
"""
测试模块文档说明
"""
import pytest
from unittest.mock import Mock, patch
from fedcl.{module_path}.{file_name} import {ClassName}

class Test{ClassName}:
    """测试类文档说明"""
    
    def test_method_name_success(self):
        """测试正常情况"""
        pass
    
    def test_method_name_error(self):
        """测试异常情况"""
        pass
```

### 3. 配置示例（如需要，配置文件相关需要）
```yaml
# configs/{config_name}.yaml
# 相关配置示例
```

## 实现注意事项
- 如果是静态工具类，不需要依赖注入和日志记录
- 如果有依赖关系，严格按照依赖注入模式实现
- 所有约束条件都必须在代码中体现
- 验收标准将作为测试用例的设计依据
- 确保代码可以直接集成到FedCL项目中
- 如果需要查看已经实现了的类的定义，你可以停下生成，去读取相应类的定义

请开始实现，确保代码质量和完整性！
