# mnist_process_demo.py
import asyncio
import multiprocessing as mp
import os
import sys
import time
import json
from typing import Dict, Any, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# 添加路径（根据你的项目结构调整）
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fedcl.types import (
    TrainingResult,
    EvaluationResult,
    ModelData,
    RoundResult,
)
from fedcl.utils.auto_logger import setup_auto_logging
from fedcl.federation.server import FederationServer
from fedcl.federation.client import FederationClient
from fedcl.learner.base_learner import BaseLearner
from fedcl.trainer.base_trainer import BaseTrainer
from fedcl.federation.coordinator import FederationCoordinator
from fedcl.types import FederationConfig


# ───────────────────────────────────────
# 1. 定义模型（简单 CNN）
# ───────────────────────────────────────
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(8 * 14 * 14, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)


# 将 PyTorch 模型的参数（即 state_dict）转换为 ModelData（字典，值为嵌套列表）
def model_to_state_dict(model: nn.Module) -> ModelData:
    return {k: v.cpu().numpy().tolist() for k, v in model.state_dict().items()}

# 将一个 序列化后的模型参数字典（ ModelData）重新加载回一个 PyTorch 模型实例中
def state_dict_to_model(state_dict: ModelData, model_class=MNISTNet) -> nn.Module:
    model = model_class()   # 创建新模型实例（结构必须一致）
    new_state = {}
    for k, v in state_dict.items():
        new_state[k] = torch.tensor(v)   # 将 list 转回 Tensor
    model.load_state_dict(new_state)
    return model


# ───────────────────────────────────────
# 2. 客户端 Learner（真实训练 MNIST）
# ───────────────────────────────────────
class MNISTLearner(BaseLearner):
    def __init__(self, client_id: str, config: Dict[str, Any], logger=None):
        super().__init__(client_id, config, logger)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MNISTNet().to(self.device)
        self._local_model = model_to_state_dict(self.model)

        # 加载 MNIST 数据（模拟非 IID 划分：每个客户端只取一部分）
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        full_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)

        # 简单划分：client_1 用 0-29999，client_2 用 30000-59999（实际可更复杂）
        total = len(full_dataset)
        start = int(client_id.split('_')[-1]) - 1  # client_1 → 0, client_2 → 1
        indices = list(range(start * total // 2, (start + 1) * total // 2))
        self.train_dataset = Subset(full_dataset, indices)
        self.train_loader = DataLoader(self.train_dataset, batch_size=32, shuffle=True)
        self.local_samples = len(self.train_dataset)

        self.test_dataset = datasets.MNIST('./data', train=False, transform=transform)
        self.test_loader = DataLoader(self.test_dataset, batch_size=32, shuffle=False)

    async def train(self, training_params: Dict[str, Any]) -> TrainingResult:
        # 接收并加载全局模型
        if "global_model" in training_params:
            global_state = training_params["global_model"]
            self.model = state_dict_to_model(global_state).to(self.device)
            self._local_model = global_state

        # 解析训练超参数
        epochs = training_params.get("epochs", 1)
        lr = training_params.get("learning_rate", 0.01)

        # 设置训练模式与优化器
        self.model.train()  # 训练模式
        optimizer = optim.SGD(self.model.parameters(), lr=lr)
        criterion = nn.NLLLoss()

        # 本地训练循环
        total_loss = 0.0
        correct = 0
        start_time = time.time()  # 开始计时
        for _ in range(epochs):
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        training_time = time.time() - start_time    # 结束计时

        # 计算训练指标
        avg_loss = total_loss / (epochs * len(self.train_loader))
        accuracy = correct / self.local_samples

        #保存并返回训练结果
        self._local_model = model_to_state_dict(self.model)

        raw_json = json.dumps(self._local_model)
        print(f"[DEBUG] 模型 JSON 大小: {len(raw_json) / 1024:.1f} KB")

        return TrainingResult(
            client_id=self.client_id,
            success=True,
            loss=avg_loss,
            accuracy=accuracy,
            samples_count=self.local_samples,
            training_time=training_time,
            model_update=self._local_model
        )

    async def evaluate(self, evaluation_params: Dict[str, Any]) -> EvaluationResult:
        self.model.eval()   # 评估模式
        test_loss = 0
        correct = 0
        criterion = nn.NLLLoss()
        start_time = time.time()  # 开始计时
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        evaluation_time = time.time() - start_time

        avg_loss = test_loss / len(self.test_loader)
        accuracy = correct / len(self.test_dataset)

        return EvaluationResult(
            client_id=self.client_id,
            success=True,
            loss=avg_loss,
            accuracy=accuracy,
            samples_count=len(self.test_dataset),
            evaluation_time=evaluation_time
        )

    async def get_local_model(self) -> ModelData:
        return self._local_model

    async def set_local_model(self, model_data: ModelData) -> bool:
        self.model = state_dict_to_model(model_data).to(self.device)
        self._local_model = model_data
        return True


# ───────────────────────────────────────
# 3. 服务端 Trainer（FedAvg 聚合）
# ───────────────────────────────────────
class MNISTTrainer(BaseTrainer):
    def __init__(self, global_model=None, training_config=None, logger=None):
        super().__init__(global_model, training_config, logger)
        self.round = 0

    async def train_round(self, round_num: int, client_ids: List[str]) -> RoundResult:
        print(f"\n🔄 第 {round_num} 轮训练：客户端 {client_ids}")

        start_time = time.time()  # 开始计时

        client_results = {}     # 保存每个成功客户端的 TrainingResult
        successful_clients = []
        failed_clients = []

        # 构建异步任务列表（不立即执行）
        tasks = []
        for cid in client_ids:
            if cid in self.learner_proxies and self.is_client_ready(cid):
                proxy = self.learner_proxies[cid]
                task = proxy.train({
                    "global_model": self.global_model,
                    "epochs": 1,
                    "learning_rate": 0.01
                })
                tasks.append((cid, task))
            else:
                failed_clients.append(cid)

        # 并发等待所有任务完成
        for cid, task in tasks:
            try:
                result = await task
                client_results[cid] = result
                successful_clients.append(cid)
            except Exception as e:
                print(f"❌ 客户端 {cid} 训练失败: {e}")
                failed_clients.append(cid)

        # 聚合模型更新
        aggregated_model = await self.aggregate_models(client_results)
        self.global_model = aggregated_model

        # 计算本轮加权平均指标
        total_samples = sum(r.get("samples_count") for r in client_results.values())
        avg_loss = sum(r.get("loss") * r.get("samples_count") for r in client_results.values()) / max(total_samples, 1)
        avg_accuracy = sum(r.get("accuracy") * r.get("samples_count") for r in client_results.values()) / max(total_samples, 1)

        training_time = time.time() - start_time

        # 返回本轮结果
        return RoundResult(
            participants=client_ids,
            successful_clients=successful_clients,
            failed_clients=failed_clients,
            aggregated_model=aggregated_model,
            round_metrics={
                "avg_loss": avg_loss,
                "avg_accuracy": avg_accuracy
            },
            training_time=training_time
        )

    async def aggregate_models(self, client_results: Dict[str, Any]) -> ModelData:
        print("🧮 聚合模型（FedAvg）...")
        total_samples = sum(r.get("samples_count", 0) for r in client_results.values())
        if total_samples == 0:
            return self.global_model

        # 初始化聚合字典
        agg_state = None
        for result in client_results.values():
            weight = result.get("samples_count", 0) / total_samples
            client_state = result.get("model_update", None)
            if agg_state is None:
                agg_state = {k: torch.tensor(v) * weight for k, v in client_state.items()}
            else:
                for k in agg_state:
                    agg_state[k] += torch.tensor(client_state[k]) * weight

        # 转回 list（JSON serializable）
        return {k: v.tolist() for k, v in agg_state.items()}

    async def evaluate_global_model(self) -> EvaluationResult:
        # 可选：服务端评估（这里略，或用虚拟数据）
        return EvaluationResult(
            client_id="server",
            success=True,
            loss=0.0,
            accuracy=0.0,
            samples_count=0,
            evaluation_time=0.0
        )

    def should_stop_training(self, round_num: int, round_result: RoundResult) -> bool:
        return round_num >= 2  # 跑 2 轮


# ───────────────────────────────────────
# 4. 主函数（使用 process 模式）
# ───────────────────────────────────────
async def run_federation():

    setup_auto_logging()
    config = {"mode": "process", "timeout": 30.0}

    # 初始全局模型
    initial_model = model_to_state_dict(MNISTNet())

    # 启动服务端
    server = FederationServer(config)
    await server.initialize_with_trainer(
        trainer_class=MNISTTrainer,
        global_model=initial_model,
        trainer_config={}
    )
    await server.start_server()
    print("✅ 服务端启动")

    client1_config = {
        "mode": "process",
        "timeout": 30.0,
        "transport": {
            "specific_config": {
                'port': 0
            }
        }
    }
    # 启动客户端（使用多进程）
    client1 = FederationClient.create_client(client1_config, client_id="process_client_1")
    await client1.initialize_with_learner(MNISTLearner)
    await client1.start_client()
    print("✅ 客户端 1 启动")

    client2_config = {
        "mode": "process",
        "timeout": 30.0,
        "transport": {
            "specific_config": {
                'port': 0
            }
        }
    }
    client2 = FederationClient.create_client(client2_config, client_id="process_client_2")
    await client2.initialize_with_learner(MNISTLearner)
    await client2.start_client()
    print("✅ 客户端 2 启动")

    # 协调器
    coordinator = FederationCoordinator(
        federation_server=server,
        federation_config=FederationConfig(
            max_rounds=2,
            min_clients=2,
        )
    )

    result = await coordinator.start_federation()
    print(f"🎉 联邦训练完成！最终轮准确率: {result.final_accuracy:.4f}, 轮数: {result.completed_rounds}")

    await client1.stop_client()
    await client2.stop_client()
    await server.stop_server()


def main():
    # 在 Windows 或 Jupyter 中需保护入口
    mp.set_start_method("spawn", force=True)
    asyncio.run(run_federation())


if __name__ == "__main__":
    main()