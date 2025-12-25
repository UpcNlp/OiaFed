"""
持续学习训练器 - 支持多任务顺序学习

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

from ...registry import trainer
from ...core.types import ClientUpdate, TrainResult, RoundResult, RoundMetrics
from .default import DefaultTrainer


@trainer(
    name='Continual',
    description='持续学习训练器 - 支持多任务顺序学习',
    version='1.0',
    author='MOE-FedCL',
    algorithms=['continual_learning', 'class_incremental']
)
class ContinualTrainer(DefaultTrainer):
    """持续学习训练器

    扩展DefaultTrainer，添加：
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

    def __init__(
        self,
        learners: "ProxyCollection",
        aggregator: "Aggregator",
        dataset: Optional[Any] = None,
        datasets: Optional[Dict[str, List[Any]]] = None,
        model: Optional["Model"] = None,
        tracker: Optional["Tracker"] = None,
        callbacks: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        node_id: Optional[str] = None,
    ):
        super().__init__(learners, aggregator, dataset, datasets, model, tracker, callbacks, config, node_id)

        # CL特定参数
        self.num_tasks = self.config.get('num_tasks', 2)
        self.rounds_per_task = self.config.get('rounds_per_task', 3)
        self.evaluate_all_tasks = self.config.get('evaluate_all_tasks', True)
        self.compute_forgetting = self.config.get('compute_forgetting', True)

        # TARGET数据生成参数
        self.enable_data_generation = self.config.get('enable_data_generation', False)
        self.data_gen_config = self.config.get('data_generation', {})

        # 从dataset配置中获取数据集名称
        dataset_name = self.config.get('dataset', {}).get('name', 'MNIST')

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

        self.save_dir = self.config.get('save_dir', 'run/target_synthetic_data')
        self.synthesizer = None  # 延迟初始化

        # 当前任务ID（从0开始）
        self.current_task_id = 0

        # CL指标追踪
        if self.compute_forgetting:
            # 延迟导入（避免循环依赖）
            from methods.metrics.continual_metrics import ContinualLearningMetrics
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

    async def train_round(self, round_num: int) -> RoundResult:
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

        # 触发轮次开始回调
        if self.callbacks:
            await self.callbacks.on_round_begin(self, round_num, {"task_id": self.current_task_id})

        config = self.config
        client_fraction = config.get("client_fraction", 1.0)
        fit_config = config.get("fit_config", {})

        # 1. 选择学习器
        connected = self.get_connected_learners()
        num_selected = max(1, int(len(connected) * client_fraction))
        selected = self.select_learners(num_selected, strategy="random")

        selected_ids = [getattr(l, '_target_id', 'unknown') for l in selected]
        self.logger.info(f"  Selected clients: {selected_ids}")

        # 2. 创建训练配置 - 关键：添加task_id
        fit_config_with_task = fit_config.copy()
        fit_config_with_task.update({
            "round_number": round_num,
            "task_id": self.current_task_id  # ← CL关键参数
        })

        # 3. 收集训练结果
        self.logger.debug(f"[Round {round_num}] 开始训练，配置: {fit_config_with_task}")
        results = await self.collect_results(selected, "fit", fit_config_with_task)

        # 4. 过滤成功的结果并创建更新
        updates = []
        for i, (learner, result) in enumerate(zip(selected, results)):
            learner_id = getattr(learner, '_target_id', f'learner_{i}')

            if isinstance(result, Exception):
                self.logger.error(f"  [{learner_id}] Training failed: {type(result).__name__}: {result}")
                continue

            # 兼容性检查
            is_train_result = isinstance(result, TrainResult) or (
                type(result).__name__ == 'TrainResult' and
                hasattr(result, 'num_samples') and
                hasattr(result, 'metrics')
            )

            if is_train_result:
                loss_value = result.metrics.final_loss if hasattr(result.metrics, 'final_loss') else result.metrics.metrics.get('loss', 'N/A')
                self.logger.info(
                    f"  [{learner_id}] Training succeeded: "
                    f"Loss={loss_value:.4f}, samples={result.num_samples}"
                )
                updates.append(ClientUpdate.from_result(learner_id, result))

        if not updates:
            self.logger.error(f"轮次 {round_num}: 没有成功的更新")
            raise RuntimeError(f"轮次 {round_num}: 所有学习器都失败了")

        # 5. 聚合
        self.logger.debug(f"[Round {round_num}] 开始聚合，updates数量: {len(updates)}")
        new_weights = self.aggregator.aggregate(updates, self.model)
        if self.model:
            self.model.set_weights(new_weights)
        self.logger.info(f"轮次 {round_num}: 聚合完成")

        # 6. 广播新权重
        self.logger.info(f"[Round {round_num}] 开始广播新权重到 {len(selected)} 个学习器")
        await self.broadcast_to_learners("set_weights", new_weights)

        # 7. 计算轮次指标
        round_metrics = self._compute_round_metrics(updates, round_num)
        round_metrics.metrics['task_id'] = self.current_task_id

        # 检查是否是任务结束轮（每个任务的最后一轮）
        max_rounds = config.get("max_rounds", 100)
        is_task_end = (round_num % self.rounds_per_task == 0) or (round_num == max_rounds)

        # 8. 在任务结束时进行多任务评估和计算CL指标
        cl_metrics_result = {}
        if is_task_end and self.evaluate_all_tasks and self.cl_metrics:
            cl_metrics_result = await self._evaluate_continual_learning(round_num)
            # 将CL指标合并到轮次指标
            round_metrics.metrics.update(cl_metrics_result.get('metrics', {}))

        # 9. TARGET数据生成：在任务结束时生成合成数据（除了最后一个任务）
        if (is_task_end and
            self.enable_data_generation and
            self.current_task_id < self.num_tasks - 1):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"开始为Task {self.current_task_id}生成合成数据...")
            self.logger.info(f"{'='*60}\n")
            await self._generate_synthetic_data(self.current_task_id)

        self.logger.info(
            f"轮次 {round_num}: "
            f"客户端数量={round_metrics.num_clients}, "
            f"数据集数量={round_metrics.total_samples}"
        )

        result = RoundResult(
            round_num=round_num,
            updates=updates,
            aggregated_weights=new_weights,
            metrics=round_metrics,
            metadata={
                "task_id": self.current_task_id,
                "is_task_end": is_task_end,
                "cl_metrics": cl_metrics_result
            }
        )

        # 触发轮次结束回调
        if self.callbacks:
            await self.callbacks.on_round_end(self, round_num, {
                "metrics": round_metrics,
                "task_id": self.current_task_id,
                "cl_metrics": cl_metrics_result
            })

        return result

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
        available_clients = self.get_connected_learners()
        if not available_clients:
            self.logger.warning("没有可用客户端进行评估")
            return {}

        # 选择一个客户端进行评估（使用第一个可用客户端）
        eval_learner = available_clients[0]
        eval_learner_id = getattr(eval_learner, '_target_id', 'learner_0')

        # 确保客户端拥有最新的全局模型（已经通过broadcast_to_learners发送）
        self.logger.info(f"在客户端 {eval_learner_id} 上评估所有已见任务...")

        # 在每个已见任务上评估
        task_accuracies = {}

        for task_id in range(current_task + 1):
            try:
                # 调用客户端的evaluate方法
                eval_params = {"task_id": task_id}

                # 使用 collect_results 来收集评估结果
                eval_results = await self.collect_results([eval_learner], "evaluate", eval_params)
                eval_result = eval_results[0]

                if isinstance(eval_result, Exception):
                    self.logger.warning(f"  Task {task_id}: 评估失败 - {eval_result}")
                    task_accuracies[task_id] = 0.0
                    continue

                # 从 EvalResult 中提取准确率
                if hasattr(eval_result, 'metrics') and isinstance(eval_result.metrics, dict):
                    accuracy = eval_result.metrics.get('accuracy', 0.0)
                else:
                    accuracy = 0.0

                task_accuracies[task_id] = accuracy
                self.logger.info(
                    f"  Task {task_id}: Accuracy = {accuracy:.4f}"
                )
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
            # 导入必要的模块（使用旧路径，因为这些模块尚未迁移）
            from methods.learners.cl.target_generator import (
                Generator, Normalizer, DataIter, UnlabeledImageDataset,
                weight_init
            )
            from methods.learners.cl.target_synthesizer import GlobalSynthesizer
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

            # 获取全局模型（PyTorch模型）
            if hasattr(self.model, '_model'):
                # 如果是 Model 包装器
                global_model_obj = self.model._model
            else:
                global_model_obj = self.model

            # 创建Student模型（随机初始化，用于验证）
            self.logger.info("初始化Student模型...")
            student = copy.deepcopy(global_model_obj)
            student.apply(weight_init)

            # 计算已见类别数
            num_classes = (task_id + 1) * cfg.get('classes_per_task', 5)

            # 创建保存目录
            task_dir = os.path.join(self.save_dir, f"task_{task_id}")
            os.makedirs(task_dir, exist_ok=True)

            # 创建GlobalSynthesizer
            self.logger.info("创建GlobalSynthesizer...")
            synthesizer = GlobalSynthesizer(
                teacher=copy.deepcopy(global_model_obj),
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
            from methods.learners.cl.target_generator import KLDiv
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
                            teacher=global_model_obj,
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
