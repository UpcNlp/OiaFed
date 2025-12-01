"""
持续学习训练器 - 支持多任务顺序学习
fedcl/methods/trainers/continual.py

支持任务调度、任务切换、遗忘度量等CL特性
"""
import asyncio
import os
import copy
from typing import Dict, Any, List, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms

from fedcl.api.decorators import trainer
from fedcl.methods.trainers.generic import GenericTrainer
from fedcl.methods.metrics.continual_metrics import ContinualLearningMetrics


@trainer('Continual',
         description='持续学习训练器 - 支持多任务顺序学习',
         version='1.0',
         author='MOE-FedCL',
         algorithms=['continual_learning', 'class_incremental'])
class ContinualTrainer(GenericTrainer):
    """持续学习训练器

    扩展GenericTrainer，添加：
    - 任务调度逻辑（基于round自动切换任务）
    - 传递task_id给learner
    - 计算CL指标（遗忘率、后向迁移等）

    配置示例:
    {
        "trainer": {
            "name": "Continual",
            "params": {
                "local_epochs": 1,
                "learning_rate": 0.01,
                "batch_size": 32,

                # CL特定参数
                "num_tasks": 2,                # 任务总数
                "rounds_per_task": 3,          # 每个任务训练的轮数
                "evaluate_all_tasks": true,    # 是否在所有已见任务上评估
                "compute_forgetting": true     # 是否计算遗忘度量
            }
        }
    }
    """

    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config, **kwargs)

        # CL特定参数
        self.num_tasks = config.get('num_tasks', 2)
        self.rounds_per_task = config.get('rounds_per_task', 3)
        self.evaluate_all_tasks = config.get('evaluate_all_tasks', True)
        self.compute_forgetting = config.get('compute_forgetting', True)

        # TARGET数据生成参数
        self.enable_data_generation = config.get('enable_data_generation', False)
        self.data_gen_config = config.get('data_generation', {})

        # 从dataset配置中获取数据集名称
        dataset_name = config.get('dataset', {}).get('name', 'MNIST')

        # 根据数据集设置默认参数
        if dataset_name.upper() == 'CIFAR100':
            default_gen_config = {
                'synthesis_batch_size': 256,
                'sample_batch_size': 256,
                'g_steps': 10,
                'is_maml': 1,
                'kd_steps': 400,
                'warmup': 20,
                'lr_g': 0.002,
                'lr_z': 0.01,
                'oh': 0.5,
                'T': 20.0,
                'act': 0.0,
                'adv': 1.0,
                'bn': 10.0,
                'reset_l0': 1,
                'reset_bn': 0,
                'bn_mmt': 0.9,
                'syn_round': 10,
                'tau': 1,
                'nz': 256,
                'img_size': 32,
            }
        else:  # MNIST或其他
            default_gen_config = {
                'synthesis_batch_size': 128,
                'sample_batch_size': 128,
                'g_steps': 10,
                'is_maml': 1,
                'kd_steps': 200,
                'warmup': 10,
                'lr_g': 0.002,
                'lr_z': 0.01,
                'oh': 0.5,
                'T': 10.0,
                'act': 0.0,
                'adv': 1.0,
                'bn': 5.0,
                'reset_l0': 1,
                'reset_bn': 0,
                'bn_mmt': 0.9,
                'syn_round': 5,
                'tau': 1,
                'nz': 100,
                'img_size': 32,
            }

        # 合并用户配置和默认配置
        for key, value in default_gen_config.items():
            if key not in self.data_gen_config:
                self.data_gen_config[key] = value

        self.save_dir = config.get('save_dir', 'run/target_synthetic_data')
        self.synthesizer = None  # 延迟初始化

        # 当前任务ID（从0开始）
        self.current_task_id = 0

        # CL指标追踪
        if self.compute_forgetting:
            self.cl_metrics = ContinualLearningMetrics(self.num_tasks)
        else:
            self.cl_metrics = None

        self.logger.info(
            f"ContinualTrainer初始化: num_tasks={self.num_tasks}, "
            f"rounds_per_task={self.rounds_per_task}, "
            f"data_generation={'enabled' if self.enable_data_generation else 'disabled'}"
        )

    def _get_current_task_id(self, round_num: int) -> int:
        """根据round数确定当前任务ID"""
        task_id = (round_num - 1) // self.rounds_per_task
        # 确保不超过最大任务数
        return min(task_id, self.num_tasks - 1)

    async def train_round(self, round_num: int, client_ids: List[str]) -> Dict[str, Any]:
        """执行一轮联邦训练（带任务调度）"""

        # 确定当前任务
        new_task_id = self._get_current_task_id(round_num)

        # 检查任务切换
        if new_task_id != self.current_task_id:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"任务切换: Task {self.current_task_id} -> Task {new_task_id}")
            self.logger.info(f"{'='*60}\n")
            self.current_task_id = new_task_id

        self.logger.info(f"\n--- Round {round_num} (Task {self.current_task_id}) ---")

        # 学习率衰减
        if self.lr_decay_enabled and round_num > 1:
            if (round_num - 1) % self.lr_decay_step == 0:
                self.current_lr *= self.lr_decay_rate
                self.logger.info(f"  学习率衰减: {self.current_lr:.6f}")

        self.logger.info(f"  Selected clients: {client_ids}")
        self.logger.info(f"  Current learning rate: {self.current_lr:.6f}")

        # 创建训练请求 - 关键：添加task_id
        training_params = {
            "round_number": round_num,
            "num_epochs": self.local_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.current_lr,
            "task_id": self.current_task_id  # ← CL关键参数
        }

        # 并行训练所有客户端
        tasks = []
        for client_id in client_ids:
            if self.is_client_ready(client_id):
                proxy = self._proxy_manager.get_proxy(client_id)
                if proxy:
                    self.logger.info(f"  [{client_id}] Starting training on Task {self.current_task_id}...")
                    task = proxy.train(training_params)
                    tasks.append((client_id, task))

        # 收集结果
        client_results = {}
        failed_clients = []

        for client_id, task in tasks:
            try:
                result = await task
                if result.success:
                    client_results[client_id] = result
                    self.logger.info(
                        f"  [{client_id}] Training succeeded: "
                        f"Loss={result.result['loss']:.4f}, Acc={result.result['accuracy']:.4f}"
                    )
                else:
                    self.logger.error(f"  [{client_id}] Training failed: {result}")
                    failed_clients.append(client_id)
            except Exception as e:
                self.logger.exception(f"  [{client_id}] Training failed: {e}")
                failed_clients.append(client_id)

        # 聚合模型
        aggregated_weights = None
        if client_results:
            aggregated_weights = await self.aggregate_models(client_results)
            if aggregated_weights:
                from fedcl.methods.models.base import set_weights_from_dict
                set_weights_from_dict(self.global_model_obj, aggregated_weights)

        # 计算轮次指标
        if client_results:
            avg_loss = np.mean([r.result['loss'] for r in client_results.values()])
            avg_accuracy = np.mean([r.result['accuracy'] for r in client_results.values()])
        else:
            avg_loss, avg_accuracy = 0.0, 0.0

        self.logger.info(f"  Round {round_num} summary: Loss={avg_loss:.4f}, Acc={avg_accuracy:.4f}")

        # 检查是否是任务结束轮（每个任务的最后一轮）
        is_task_end = (round_num % self.rounds_per_task == 0) or (round_num == self.max_rounds)

        # 在任务结束时进行多任务评估和计算CL指标
        cl_metrics_result = {}
        if is_task_end and self.evaluate_all_tasks and self.cl_metrics:
            cl_metrics_result = await self._evaluate_continual_learning(round_num)

        # TARGET数据生成：在任务结束时生成合成数据（除了最后一个任务）
        if (is_task_end and
            self.enable_data_generation and
            self.current_task_id < self.num_tasks - 1):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"开始为Task {self.current_task_id}生成合成数据...")
            self.logger.info(f"{'='*60}\n")
            await self._generate_synthetic_data(self.current_task_id)

        return {
            "round": round_num,
            "task_id": self.current_task_id,
            "participants": client_ids,
            "successful_clients": list(client_results.keys()),
            "failed_clients": failed_clients,
            "model_aggregated": aggregated_weights is not None,
            "round_metrics": {
                "avg_loss": avg_loss,
                "avg_accuracy": avg_accuracy,
                "successful_count": len(client_results)
            },
            "cl_metrics": cl_metrics_result  # CL指标
        }

    async def _evaluate_continual_learning(self, round_num: int) -> Dict[str, Any]:
        """
        在所有已见任务上评估，计算CL指标

        通过调用客户端的evaluate方法，在每个已见任务上评估当前模型性能

        Returns:
            包含遗忘率、后向迁移等指标的字典
        """
        current_task = self._get_current_task_id(round_num)

        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"持续学习评估 - 已完成Task {current_task}")
        self.logger.info(f"{'='*60}")

        # 获取可用客户端
        available_clients = self.get_available_clients()
        if not available_clients:
            self.logger.warning("没有可用客户端进行评估")
            return {}

        # 选择一个客户端进行评估（使用第一个可用客户端）
        eval_client_id = available_clients[0]
        proxy = self._proxy_manager.get_proxy(eval_client_id)

        if not proxy:
            self.logger.warning(f"无法获取客户端 {eval_client_id} 的代理")
            return {}

        # 确保客户端拥有最新的全局模型
        from fedcl.methods.models.base import get_weights_as_dict
        global_weights = get_weights_as_dict(self.global_model_obj)
        # ModelData是dict类型别名，直接使用字典
        model_data = {"model_data": global_weights}

        try:
            await proxy.set_local_model(model_data)
            self.logger.info(f"已将全局模型发送到客户端 {eval_client_id} 用于评估")
        except Exception as e:
            self.logger.error(f"发送全局模型到客户端失败: {e}")
            return {}

        # 在每个已见任务上评估
        task_accuracies = {}
        self.logger.info(f"在客户端 {eval_client_id} 上评估所有已见任务...")

        for task_id in range(current_task + 1):
            try:
                # 调用客户端的evaluate方法
                eval_params = {"task_id": task_id}
                eval_result = await proxy.evaluate(eval_params)

                if eval_result.success:
                    # eval_result.result 可能是嵌套结构:
                    # {'success': True, 'result': {'accuracy': ...}, 'metadata': ...}
                    result_data = eval_result.result
                    if isinstance(result_data, dict) and 'result' in result_data:
                        # 嵌套结构
                        accuracy = result_data['result'].get('accuracy', 0.0)
                    else:
                        # 直接结构
                        accuracy = result_data.get('accuracy', 0.0) if isinstance(result_data, dict) else 0.0
                    task_accuracies[task_id] = accuracy
                    self.logger.info(
                        f"  Task {task_id}: Accuracy = {accuracy:.4f}"
                    )
                else:
                    self.logger.warning(f"  Task {task_id}: 评估失败")
                    task_accuracies[task_id] = 0.0
            except Exception as e:
                self.logger.error(f"  Task {task_id}: 评估异常 - {e}")
                task_accuracies[task_id] = 0.0

        # 更新CL指标
        self.cl_metrics.update(current_task, task_accuracies)

        # 计算所有CL指标
        metrics = self.cl_metrics.get_all_metrics(up_to_task=current_task)

        # 打印CL指标
        self.logger.info(f"\n--- CL 指标汇总 ---")
        self.logger.info(f"平均准确率 (AA): {metrics['average_accuracy']:.4f}")
        self.logger.info(f"遗忘度量 (FM): {metrics['forgetting_measure']:.4f}")
        self.logger.info(f"后向迁移 (BWT): {metrics['backward_transfer']:.4f}")
        self.logger.info(f"前向迁移 (FWT): {metrics['forward_transfer']:.4f}")
        self.logger.info(f"{'='*60}\n")

        # 打印准确率矩阵
        if current_task > 0:
            self.cl_metrics.print_accuracy_matrix()

        return {
            "evaluated_tasks": list(range(current_task + 1)),
            "task_accuracies": task_accuracies,
            "metrics": metrics
        }

    async def _generate_synthetic_data(self, task_id: int) -> None:
        """
        为指定任务生成合成数据（TARGET算法）

        Args:
            task_id: 当前完成的任务ID

        流程：
        1. 初始化Generator和Student模型
        2. 循环syn_round次生成合成数据
        3. 每轮用合成数据训练Student验证质量
        4. 保存合成数据到磁盘供后续任务使用
        """
        try:
            # 导入必要的模块
            from fedcl.methods.learners.cl.target_generator import (
                Generator, Normalizer, DataIter, UnlabeledImageDataset,
                weight_init
            )
            from fedcl.methods.learners.cl.target_synthesizer import GlobalSynthesizer
            from fedcl.methods.models.base import get_weights_as_dict
            from torch.utils.data import DataLoader

            # 获取配置参数
            cfg = self.data_gen_config
            nz = cfg.get('nz', 100)
            img_size = cfg.get('img_size', 32)
            syn_round = cfg.get('syn_round', 5)
            kd_steps = cfg.get('kd_steps', 200)
            warmup = cfg.get('warmup', 10)
            T = cfg.get('T', 10.0)

            # 确定图像shape
            if img_size == 32:
                img_shape = (3, 32, 32)
                data_normalize = dict(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))
            elif img_size == 28:
                img_shape = (1, 28, 28)
                data_normalize = dict(mean=(0.1307,), std=(0.3081,))
            else:
                img_shape = (3, 64, 64)
                data_normalize = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            # 数据归一化器
            normalizer = Normalizer(**data_normalize)

            # 创建Generator
            self.logger.info(f"初始化Generator (nz={nz}, img_size={img_size})...")
            generator = Generator(nz=nz, ngf=64, img_size=img_size, nc=img_shape[0])

            # 创建Student模型（随机初始化，用于验证）
            self.logger.info("初始化Student模型...")
            student = copy.deepcopy(self.global_model_obj)
            student.apply(weight_init)

            # 计算已见类别数
            num_classes = (task_id + 1) * cfg.get('classes_per_task', 5)

            # 创建保存目录
            task_dir = os.path.join(self.save_dir, f"task_{task_id}")
            os.makedirs(task_dir, exist_ok=True)

            # 创建GlobalSynthesizer
            self.logger.info("创建GlobalSynthesizer...")
            synthesizer = GlobalSynthesizer(
                teacher=copy.deepcopy(self.global_model_obj),
                student=student,
                generator=generator,
                nz=nz,
                num_classes=num_classes,
                img_size=img_shape,
                save_dir=task_dir,
                transform=None,
                normalizer=normalizer,
                synthesis_batch_size=cfg.get('synthesis_batch_size', 128),
                sample_batch_size=cfg.get('sample_batch_size', 128),
                iterations=cfg.get('g_steps', 10),
                warmup=warmup,
                lr_g=cfg.get('lr_g', 0.002),
                lr_z=cfg.get('lr_z', 0.01),
                adv=cfg.get('adv', 1.0),
                bn=cfg.get('bn', 5.0),
                oh=cfg.get('oh', 0.5),
                reset_l0=cfg.get('reset_l0', 1),
                reset_bn=cfg.get('reset_bn', 0),
                bn_mmt=cfg.get('bn_mmt', 0.9),
                is_maml=cfg.get('is_maml', 1),
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )

            # KL散度损失
            from fedcl.methods.learners.cl.target_generator import KLDiv
            criterion = KLDiv(T=T)

            # Student优化器
            optimizer = SGD(student.parameters(), lr=0.2, weight_decay=0.0001, momentum=0.9)
            scheduler = CosineAnnealingLR(optimizer, 200, eta_min=2e-4)

            # 数据生成循环
            self.logger.info(f"开始生成合成数据 (syn_round={syn_round})...")
            for it in range(syn_round):
                # 生成一批合成数据
                synthesizer.synthesize()

                # 从warmup轮开始进行KD训练
                if it >= warmup:
                    # 创建合成数据的DataLoader
                    syn_dataset = UnlabeledImageDataset(
                        task_dir,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(**data_normalize)
                        ]),
                        nums=None
                    )

                    if len(syn_dataset) > 0:
                        syn_loader = DataLoader(
                            syn_dataset,
                            batch_size=cfg.get('sample_batch_size', 128),
                            shuffle=True,
                            num_workers=0
                        )

                        # KD训练Student
                        self._kd_train_student(
                            student=student,
                            teacher=self.global_model_obj,
                            criterion=criterion,
                            optimizer=optimizer,
                            data_loader=syn_loader,
                            kd_steps=kd_steps
                        )

                        # 学习率调整
                        scheduler.step()

                        self.logger.info(
                            f"Task {task_id}, Data Generation, Round {it + 1}/{syn_round} => "
                            f"Generated {len(syn_dataset)} synthetic samples"
                        )

            # 清理hooks
            synthesizer.remove_hooks()

            self.logger.info(
                f"Task {task_id} 数据生成完成！共生成 {len(os.listdir(task_dir))} 个数据文件"
            )
            self.logger.info(f"合成数据保存于: {task_dir}\n")

        except Exception as e:
            self.logger.error(f"数据生成失败: {e}", exc_info=True)

    def _kd_train_student(self, student, teacher, criterion, optimizer,
                         data_loader, kd_steps):
        """
        使用合成数据训练Student模型进行验证

        Args:
            student: Student模型
            teacher: Teacher模型（当前全局模型）
            criterion: KL散度损失
            optimizer: 优化器
            data_loader: 合成数据的DataLoader
            kd_steps: 训练步数
        """
        student.train()
        teacher.eval()

        data_iter = DataIter(data_loader)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        for i in range(kd_steps):
            images = data_iter.next().to(device)

            with torch.no_grad():
                t_out = teacher(images)
                if isinstance(t_out, dict):
                    t_out = t_out["logits"]

            s_out = student(images.detach())
            if isinstance(s_out, dict):
                s_out = s_out["logits"]

            loss_s = criterion(s_out, t_out.detach())

            optimizer.zero_grad()
            loss_s.backward()
            optimizer.step()

