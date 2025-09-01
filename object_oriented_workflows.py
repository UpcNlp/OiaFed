#!/usr/bin/env python3
"""
基于抽象类的自定义联邦学习工作流框架

使用抽象类和继承的方式，让用户能够：
1. 直接访问和操作客户端模型
2. 定义自己的具体业务逻辑
3. 更好地组织复杂的训练流程
4. 支持状态管理和生命周期控制
"""

import os
import sys
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from loguru import logger

# 添加项目路径
sys.path.insert(0, os.path.abspath('.'))

from fedcl.transparent.execution_engine import TrainingResult


# ============================================================================
# 抽象基类定义
# ============================================================================

class BaseFederatedWorkflow(ABC):
    """
    联邦学习工作流抽象基类
    
    用户通过继承这个类来定义自己的联邦学习业务逻辑
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.round_history: List[Dict[str, Any]] = []
        self.global_state: Dict[str, Any] = {}
        self.client_states: Dict[str, Dict[str, Any]] = {}
        
    @abstractmethod
    def setup_workflow(self, federation_context, **kwargs) -> None:
        """设置工作流"""
        pass
    
    @abstractmethod
    def setup_client_models(self, client_id: str, **kwargs) -> Dict[str, nn.Module]:
        """为客户端设置模型"""
        pass
    
    @abstractmethod
    def client_train_step(self, client_id: str, models: Dict[str, nn.Module], 
                         round_num: int, **kwargs) -> Dict[str, Any]:
        """客户端训练步骤"""
        pass
    
    @abstractmethod
    def server_aggregate_step(self, client_results: List[Dict[str, Any]], 
                            round_num: int, **kwargs) -> Dict[str, Any]:
        """服务器聚合步骤"""
        pass
    
    def before_round(self, round_num: int, **kwargs) -> None:
        """轮次开始前的钩子"""
        pass
    
    def after_round(self, round_num: int, round_result: Dict[str, Any], **kwargs) -> None:
        """轮次结束后的钩子"""
        pass
    
    def get_client_data(self, client_id: str, round_num: int, **kwargs) -> Any:
        """获取客户端数据（子类可重写）"""
        batch_size = self.config.get("batch_size", 32)
        data_dim = self.config.get("data_dim", 784)
        num_classes = self.config.get("num_classes", 10)
        
        data = torch.randn(batch_size, data_dim)
        target = torch.randint(0, num_classes, (batch_size,))
        
        return {"data": data, "target": target, "client_id": client_id}
    
    def update_global_state(self, key: str, value: Any) -> None:
        """更新全局状态"""
        self.global_state[key] = value
    
    def get_global_state(self, key: str, default: Any = None) -> Any:
        """获取全局状态"""
        return self.global_state.get(key, default)


class BaseClientModel(ABC):
    """客户端模型抽象基类"""
    
    def __init__(self, client_id: str, config: Dict[str, Any] = None):
        self.client_id = client_id
        self.config = config or {}
        self.models: Dict[str, nn.Module] = {}
        self.optimizers: Dict[str, torch.optim.Optimizer] = {}
    
    @abstractmethod
    def build_models(self) -> Dict[str, nn.Module]:
        """构建客户端模型"""
        pass
    
    @abstractmethod
    def forward_pass(self, data: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """前向传播"""
        pass
    
    @abstractmethod
    def compute_loss(self, outputs: Dict[str, torch.Tensor], 
                    targets: torch.Tensor, **kwargs) -> torch.Tensor:
        """计算损失"""
        pass
    
    def get_model_weights(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """获取所有模型的权重"""
        weights = {}
        for name, model in self.models.items():
            weights[name] = {k: v.clone().detach() for k, v in model.state_dict().items()}
        return weights


# ============================================================================
# 工作流执行器
# ============================================================================

class WorkflowExecutor:
    """工作流执行器"""
    
    def __init__(self, workflow: BaseFederatedWorkflow):
        self.workflow = workflow
        
    def execute(self, federation_context, num_rounds: int, **kwargs) -> TrainingResult:
        """执行工作流"""
        logger.info(f"🚀 开始执行自定义工作流: {self.workflow.__class__.__name__}")
        
        # 设置工作流
        self.workflow.setup_workflow(federation_context, **kwargs)
        
        # 获取客户端列表
        num_clients = kwargs.get("num_clients", 3)
        client_ids = [f"client_{i}" for i in range(num_clients)]
        
        round_history = []
        
        for round_num in range(1, num_rounds + 1):
            logger.info(f"🔄 执行轮次 {round_num}")
            
            # 轮次开始前钩子
            self.workflow.before_round(round_num, **kwargs)
            
            # 客户端训练
            client_results = []
            for client_id in client_ids:
                # 获取客户端模型
                models = self.workflow.setup_client_models(client_id, **kwargs)
                
                # 执行客户端训练
                client_result = self.workflow.client_train_step(
                    client_id, models, round_num, **kwargs
                )
                client_result["client_id"] = client_id
                client_results.append(client_result)
            
            # 服务器聚合
            aggregation_result = self.workflow.server_aggregate_step(
                client_results, round_num, **kwargs
            )
            
            # 记录轮次结果
            round_data = {
                "round": round_num,
                "num_participants": len(client_results),
                **aggregation_result
            }
            round_history.append(round_data)
            
            # 轮次结束后钩子
            self.workflow.after_round(round_num, round_data, **kwargs)
        
        # 构建结果
        final_metrics = {k: v for k, v in round_history[-1].items() 
                        if k not in ["round", "num_participants"]}
        
        return TrainingResult(
            total_rounds=num_rounds,
            final_metrics=final_metrics,
            round_history=round_history,
            client_results={},
            execution_mode=federation_context.mode.value,
            training_time=0.0,
            custom_results=self.workflow.global_state.copy()
        )


# ============================================================================
# 具体实现示例
# ============================================================================

class DiffusionFederatedWorkflow(BaseFederatedWorkflow):
    """扩散模型联邦学习工作流"""
    
    def setup_workflow(self, federation_context, **kwargs):
        """设置扩散模型工作流"""
        self.federation_context = federation_context
        
        # 初始化全局扩散模型
        diffusion_model = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 784)
        )
        self.update_global_state("diffusion_model", diffusion_model)
        
        logger.info("🎨 扩散模型联邦学习工作流设置完成")
    
    def setup_client_models(self, client_id: str, **kwargs) -> Dict[str, nn.Module]:
        """设置客户端模型"""
        models = {
            "classifier": nn.Sequential(
                nn.Linear(256, 128),  # 修复维度
                nn.ReLU(),
                nn.Linear(128, 10)
            ),
            "feature_extractor": nn.Sequential(
                nn.Linear(784, 256),
                nn.ReLU()
            )
        }
        return models
    
    def client_train_step(self, client_id: str, models: Dict[str, nn.Module], 
                         round_num: int, **kwargs) -> Dict[str, Any]:
        """客户端训练步骤"""
        # 获取数据
        data_batch = self.get_client_data(client_id, round_num)
        data, targets = data_batch["data"], data_batch["target"]
        
        # 前向传播
        features = models["feature_extractor"](data)
        logits = models["classifier"](features)
        
        # 计算分类损失
        classification_loss = nn.CrossEntropyLoss()(logits, targets)
        
        # 计算扩散损失
        diffusion_model = self.get_global_state("diffusion_model")
        noise = torch.randn_like(features)
        noisy_features = features + 0.1 * noise
        reconstructed = diffusion_model(noisy_features)
        diffusion_loss = nn.MSELoss()(reconstructed, features)
        
        # 总损失
        total_loss = classification_loss + 0.1 * diffusion_loss
        accuracy = (logits.argmax(dim=1) == targets).float().mean().item()
        
        return {
            "loss": total_loss.item(),
            "classification_loss": classification_loss.item(),
            "diffusion_loss": diffusion_loss.item(),
            "accuracy": accuracy,
            "num_samples": len(data),
            "features": features.clone().detach()
        }
    
    def server_aggregate_step(self, client_results: List[Dict[str, Any]], 
                            round_num: int, **kwargs) -> Dict[str, Any]:
        """服务器聚合步骤"""
        # 收集特征用于更新扩散模型
        all_features = torch.cat([result["features"] for result in client_results], dim=0)
        
        # 模拟扩散模型更新
        diffusion_fid = max(10.0, 50.0 - 3.0 * round_num)
        
        # 计算聚合指标
        total_samples = sum(result["num_samples"] for result in client_results)
        avg_accuracy = sum(
            result["accuracy"] * result["num_samples"] for result in client_results
        ) / total_samples
        avg_loss = sum(
            result["loss"] * result["num_samples"] for result in client_results
        ) / total_samples
        
        return {
            "accuracy": avg_accuracy,
            "loss": avg_loss,
            "diffusion_fid": diffusion_fid,
            "generation_quality": 0.6 + 0.1 * round_num
        }


# ============================================================================
# 测试代码
# ============================================================================

def test_object_oriented_workflows():
    """测试面向对象的自定义工作流"""
    logger.info("🧪 测试面向对象的自定义工作流")
    
    try:
        # 模拟联邦上下文
        class MockFederationContext:
            def __init__(self):
                self.mode = type('Mode', (), {'value': 'local_simulation'})()
        
        federation_context = MockFederationContext()
        
        # 测试扩散模型工作流
        logger.info("\n--- 测试扩散模型联邦学习工作流 ---")
        diffusion_workflow = DiffusionFederatedWorkflow({
            "batch_size": 32,
            "data_dim": 784,
            "num_classes": 10
        })
        
        diffusion_executor = WorkflowExecutor(diffusion_workflow)
        result = diffusion_executor.execute(federation_context, num_rounds=3, num_clients=2)
        
        logger.info(f"扩散模型结果 - 准确率: {result.final_metrics['accuracy']:.4f}")
        logger.info(f"扩散FID: {result.final_metrics['diffusion_fid']:.2f}")
        
        logger.info("\n✅ 面向对象工作流测试通过!")
        return True
        
    except Exception as e:
        logger.error(f"❌ 面向对象工作流测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行测试"""
    logger.info("🚀 开始面向对象自定义工作流演示")
    
    success = test_object_oriented_workflows()
    
    if success:
        logger.info("\n🎉 面向对象自定义工作流运行成功!")
        
        logger.info("\n📋 面向对象工作流优势:")
        print("1. ✅ 直接访问客户端模型")
        print("2. ✅ 清晰的抽象类结构")
        print("3. ✅ 灵活的状态管理")
        print("4. ✅ 生命周期钩子支持")
        print("5. ✅ 更好的代码组织")
        
    return success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)