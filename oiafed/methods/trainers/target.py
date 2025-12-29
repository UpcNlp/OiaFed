"""
TARGET Trainer - Federated Class-Continual Learning
src/methods/trainers/target.py

TARGET: Federated Class-Continual Learning via Exemplar-Free Distillation (ICCV 2023)
Core Features:
1. Server-side synthetic data generation (using Generator + BatchNorm matching)
2. Automatic data generation triggered on task transitions
3. Client-side knowledge distillation with synthetic data
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

from ...core.trainer import Trainer
from ...core.types import ClientUpdate, TrainResult, RoundResult, RoundMetrics
from ...registry import trainer


@trainer(
    name='TARGET',
    description='TARGET: Federated continual learning with server-side data generation',
    version='2.0',
    author='MOE-FedCL',
    algorithms=['TARGET', 'federated_continual_learning']
)
class TARGETTrainer(Trainer):
    """TARGET Algorithm Trainer

    Extends Trainer with:
    - Task scheduling logic (automatic task switching based on rounds)
    - Server-side synthetic data generation (Generator + GlobalSynthesizer)
    - Passing task_id to learners
    - Computing CL metrics (forgetting rate, backward transfer, etc.)

    Configuration Example:
    {
        "trainer": {
            "name": "TARGET",
            "params": {
                "local_epochs": 1,
                "learning_rate": 0.01,
                "batch_size": 32,

                # CL-specific parameters
                "num_tasks": 2,
                "rounds_per_task": 3,
                "evaluate_all_tasks": true,
                "compute_forgetting": true,

                # TARGET data generation config
                "enable_data_generation": true,
                "save_dir": "run/target_synthetic_data",
                "data_generation": {
                    "synthesis_batch_size": 128,
                    "g_steps": 10,
                    "syn_round": 5,
                    "nz": 100,
                    ...
                }
            }
        }
    }
    """

    def __init__(
        self,
        learners,
        aggregator,
        dataset=None,
        datasets=None,
        model=None,
        tracker=None,
        callbacks=None,
        config=None,
        node_id=None
    ):
        """Initialize TARGET trainer

        Args:
            learners: Learner proxy collection
            aggregator: Aggregator instance
            dataset: Server-side dataset (optional, for evaluation)
            datasets: Dataset dictionary by split (optional)
            model: Global model
            tracker: Metrics tracker
            callbacks: Callback manager
            config: Trainer configuration
            node_id: Node ID (for logging)
        """
        super().__init__(
            learners=learners,
            aggregator=aggregator,
            dataset=dataset,
            datasets=datasets,
            model=model,
            tracker=tracker,
            callbacks=callbacks,
            config=config,
            node_id=node_id
        )

        # Extract parameters from config
        # CL-specific parameters
        self.num_tasks = config.get('num_tasks', 2)
        self.rounds_per_task = config.get('rounds_per_task', 3)
        self.evaluate_all_tasks = config.get('evaluate_all_tasks', True)
        self.compute_forgetting = config.get('compute_forgetting', True)

        # TARGET data generation parameters
        self.enable_data_generation = config.get('enable_data_generation', False)
        self.data_gen_config = config.get('data_generation', {})
        self.save_dir = config.get('save_dir', 'run/target_synthetic_data')

        # Get dataset name from config
        dataset_config = config.get('test_dataset', {})
        dataset_name = dataset_config.get('name', 'MNIST')

        # Set default parameters based on dataset
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
        else:  # MNIST or others
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

        # Merge user config with defaults
        for key, value in default_gen_config.items():
            if key not in self.data_gen_config:
                self.data_gen_config[key] = value

        self.synthesizer = None  # Lazy initialization

        # Current task ID (starting from 0)
        self.current_task_id = 0

        # CL metrics tracking
        if self.compute_forgetting:
            # Import here to avoid circular dependency
            from methods.metrics.continual_metrics import ContinualLearningMetrics
            self.cl_metrics = ContinualLearningMetrics(self.num_tasks)
        else:
            self.cl_metrics = None

        self.logger.info(
            f"TARGETTrainer initialized: num_tasks={self.num_tasks}, "
            f"rounds_per_task={self.rounds_per_task}, "
            f"data_generation={'enabled' if self.enable_data_generation else 'disabled'}"
        )

    def _get_current_task_id(self, round_num: int) -> int:
        """Determine current task ID based on round number"""
        task_id = (round_num - 1) // self.rounds_per_task
        # Ensure not exceeding max tasks
        return min(task_id, self.num_tasks - 1)

    async def train_round(self, round_num: int) -> RoundResult:
        """Execute one federated training round (with task scheduling and data generation)

        Args:
            round_num: Round number (starting from 1)

        Returns:
            RoundResult: Contains updates, weights, and metrics
        """
        # Trigger round begin callback
        if self.callbacks:
            await self.callbacks.on_round_begin(self, round_num, {})

        # Determine current task
        new_task_id = self._get_current_task_id(round_num)

        # Check for task transition
        if new_task_id != self.current_task_id:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Task transition: Task {self.current_task_id} -> Task {new_task_id}")
            self.logger.info(f"{'='*60}\n")
            self.current_task_id = new_task_id

        self.logger.info(f"\n--- Round {round_num} (Task {self.current_task_id}) ---")

        # Get configuration
        client_fraction = self.config.get('client_fraction', 1.0)
        local_epochs = self.config.get('local_epochs', 1)
        batch_size = self.config.get('batch_size', 32)
        learning_rate = self.config.get('learning_rate', 0.01)

        # 1. Select learners
        connected = self.get_connected_learners()
        num_selected = max(1, int(len(connected) * client_fraction))
        selected = self.select_learners(num_selected, strategy="random")

        selected_ids = [getattr(l, '_target_id', 'unknown') for l in selected]
        self.logger.info(f"Round {round_num}: Selected {len(selected)} learners: {selected_ids}")

        # 2. Create training configuration - KEY: add task_id
        fit_config = {
            "epochs": local_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "task_id": self.current_task_id  # CL key parameter
        }

        self.logger.info(f"Training config: {fit_config}")

        # 3. Collect training results
        results = await self.collect_results(selected, "fit", fit_config)

        # 4. Filter successful results and create updates
        updates = []
        for i, (learner, result) in enumerate(zip(selected, results)):
            learner_id = getattr(learner, '_target_id', f'learner_{i}')

            if isinstance(result, Exception):
                self.logger.error(f"[Round {round_num}] Learner {learner_id} failed: {result}")
                continue

            # Compatibility check
            is_train_result = isinstance(result, TrainResult) or (
                type(result).__name__ == 'TrainResult' and
                hasattr(result, 'num_samples') and
                hasattr(result, 'metrics')
            )

            if is_train_result:
                loss_value = result.metrics.final_loss if hasattr(result.metrics, 'final_loss') else result.metrics.metrics.get('loss', 'N/A')
                self.logger.info(f"[{learner_id}] Training succeeded: samples={result.num_samples}, loss={loss_value}")
                updates.append(ClientUpdate.from_result(learner_id, result))

        if not updates:
            self.logger.error(f"Round {round_num}: No successful updates")
            raise RuntimeError(f"Round {round_num}: All learners failed")

        # 5. Aggregate
        self.logger.info(f"Round {round_num}: Aggregating {len(updates)} updates")
        new_weights = self.aggregator.aggregate(updates, self.model)
        if self.model:
            self.model.set_weights(new_weights)
        self.logger.info(f"Round {round_num}: Aggregation complete")

        # 6. Broadcast new weights
        self.logger.info(f"Round {round_num}: Broadcasting new weights to {len(selected)} learners")
        await self.broadcast_to_learners("set_weights", new_weights)

        # 7. Compute round metrics
        round_metrics = self._compute_round_metrics(updates, round_num)

        self.logger.info(
            f"Round {round_num} summary: "
            f"clients={round_metrics.num_clients}, "
            f"samples={round_metrics.total_samples}, "
            f"metrics={round_metrics.metrics}"
        )

        # Check if this is the end of a task (last round of each task)
        is_task_end = (round_num % self.rounds_per_task == 0) or (round_num == self.config.get('max_rounds', 100))

        # Perform multi-task evaluation and compute CL metrics at task end
        cl_metrics_result = {}
        if is_task_end and self.evaluate_all_tasks and self.cl_metrics:
            cl_metrics_result = await self._evaluate_continual_learning(round_num)
            # Merge CL metrics into round metrics
            round_metrics.metrics.update(cl_metrics_result.get('metrics', {}))

        # TARGET data generation: generate synthetic data at task end (except for the last task)
        if (is_task_end and
            self.enable_data_generation and
            self.current_task_id < self.num_tasks - 1):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Starting synthetic data generation for Task {self.current_task_id}...")
            self.logger.info(f"{'='*60}\n")
            await self._generate_synthetic_data(self.current_task_id)

        # Create round result
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

        # Trigger round end callback
        if self.callbacks:
            await self.callbacks.on_round_end(self, round_num, {"metrics": round_metrics})

        return result

    def _compute_round_metrics(self, updates: List[ClientUpdate], round_num: int) -> RoundMetrics:
        """Compute round metrics"""
        if not updates:
            return RoundMetrics(
                round_num=round_num,
                num_clients=0,
                total_samples=0,
                metrics={}
            )

        total_samples = sum(u.num_samples for u in updates)

        # Aggregate client metrics
        all_metrics: Dict[str, List[float]] = {}
        for update in updates:
            for key, value in update.metrics.items():
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(value)

        # Compute averages
        aggregated_metrics = {}
        for key, values in all_metrics.items():
            aggregated_metrics[f"avg_{key}"] = sum(values) / len(values)

        return RoundMetrics(
            round_num=round_num,
            num_clients=len(updates),
            total_samples=total_samples,
            metrics=aggregated_metrics
        )

    async def _evaluate_continual_learning(self, round_num: int) -> Dict[str, Any]:
        """
        Evaluate on all seen tasks, compute CL metrics

        Calls learner's evaluate method on each seen task to assess current model performance

        Returns:
            Dictionary containing forgetting rate, backward transfer, and other metrics
        """
        current_task = self._get_current_task_id(round_num)

        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Continual Learning Evaluation - Completed Task {current_task}")
        self.logger.info(f"{'='*60}")

        # Get available learners
        available_learners = self.get_connected_learners()
        if not available_learners:
            self.logger.warning("No available learners for evaluation")
            return {}

        # Select one learner for evaluation (use first available)
        eval_learner = available_learners[0]
        eval_learner_id = getattr(eval_learner, '_target_id', 'unknown')

        self.logger.info(f"Using learner {eval_learner_id} for evaluation")

        # Ensure learner has the latest global model
        if self.model:
            global_weights = self.model.get_weights()
            try:
                await eval_learner.set_weights(global_weights)
                self.logger.info(f"Sent global model to learner {eval_learner_id} for evaluation")
            except Exception as e:
                self.logger.error(f"Failed to send global model to learner: {e}")
                return {}

        # Evaluate on each seen task
        task_accuracies = {}
        self.logger.info(f"Evaluating all seen tasks on learner {eval_learner_id}...")

        for task_id in range(current_task + 1):
            try:
                # Call learner's evaluate method
                eval_config = {"task_id": task_id}
                eval_result = await eval_learner.evaluate(eval_config)

                if not isinstance(eval_result, Exception):
                    # Extract accuracy from result
                    if hasattr(eval_result, 'metrics'):
                        accuracy = eval_result.metrics.get('accuracy', 0.0)
                    else:
                        accuracy = 0.0
                    task_accuracies[task_id] = accuracy
                    self.logger.info(f"  Task {task_id}: Accuracy = {accuracy:.4f}")
                else:
                    self.logger.warning(f"  Task {task_id}: Evaluation failed")
                    task_accuracies[task_id] = 0.0
            except Exception as e:
                self.logger.error(f"  Task {task_id}: Evaluation exception - {e}")
                task_accuracies[task_id] = 0.0

        # Update CL metrics
        self.cl_metrics.update(current_task, task_accuracies)

        # Compute all CL metrics
        metrics = self.cl_metrics.get_all_metrics(up_to_task=current_task)

        # Print CL metrics summary
        self.logger.info(f"\n--- CL Metrics Summary ---")
        self.logger.info(f"Average Accuracy (AA): {metrics['average_accuracy']:.4f}")
        self.logger.info(f"Forgetting Measure (FM): {metrics['forgetting_measure']:.4f}")
        self.logger.info(f"Backward Transfer (BWT): {metrics['backward_transfer']:.4f}")
        self.logger.info(f"Forward Transfer (FWT): {metrics['forward_transfer']:.4f}")
        self.logger.info(f"{'='*60}\n")

        # Print accuracy matrix
        if current_task > 0:
            self.cl_metrics.print_accuracy_matrix()

        return {
            "evaluated_tasks": list(range(current_task + 1)),
            "task_accuracies": task_accuracies,
            "metrics": metrics
        }

    async def _generate_synthetic_data(self, task_id: int) -> None:
        """
        Generate synthetic data for specified task (TARGET algorithm)

        Args:
            task_id: Current completed task ID

        Process:
        1. Initialize Generator and Student models
        2. Loop syn_round times to generate synthetic data
        3. Train Student with synthetic data each round to verify quality
        4. Save synthetic data to disk for use in subsequent tasks
        """
        try:
            # Import necessary modules
            from methods.learners.cl.target_generator import (
                Generator, Normalizer, DataIter, UnlabeledImageDataset,
                weight_init
            )
            from methods.learners.cl.target_synthesizer import GlobalSynthesizer
            from torch.utils.data import DataLoader

            # Get configuration parameters
            cfg = self.data_gen_config
            nz = cfg.get('nz', 100)
            img_size = cfg.get('img_size', 32)
            syn_round = cfg.get('syn_round', 5)
            kd_steps = cfg.get('kd_steps', 200)
            warmup = cfg.get('warmup', 10)
            T = cfg.get('T', 10.0)

            # Determine image shape
            if img_size == 32:
                img_shape = (3, 32, 32)
                data_normalize = dict(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))
            elif img_size == 28:
                img_shape = (1, 28, 28)
                data_normalize = dict(mean=(0.1307,), std=(0.3081,))
            else:
                img_shape = (3, 64, 64)
                data_normalize = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            # Data normalizer
            normalizer = Normalizer(**data_normalize)

            # Determine device
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            # Create Generator and move to device
            self.logger.info(f"Initializing Generator (nz={nz}, img_size={img_size})...")
            generator = Generator(nz=nz, ngf=64, img_size=img_size, nc=img_shape[0])
            generator = generator.to(device)

            # Create Student model (randomly initialized for verification) and move to device
            self.logger.info("Initializing Student model...")
            # Get the underlying PyTorch model from the Model wrapper
            if hasattr(self.model, '_model'):
                base_model = self.model._model
            else:
                base_model = self.model
            student = copy.deepcopy(base_model)
            student.apply(weight_init)
            student = student.to(device)

            # Teacher model also needs to be on device
            teacher = copy.deepcopy(base_model)
            teacher = teacher.to(device)

            # Compute number of seen classes
            num_classes = (task_id + 1) * cfg.get('classes_per_task', 5)

            # Create save directory
            task_dir = os.path.join(self.save_dir, f"task_{task_id}")
            os.makedirs(task_dir, exist_ok=True)

            # Create GlobalSynthesizer
            self.logger.info("Creating GlobalSynthesizer...")
            synthesizer = GlobalSynthesizer(
                teacher=teacher,
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
                device=device
            )

            # KL divergence loss
            from methods.learners.cl.target_generator import KLDiv
            criterion = KLDiv(T=T)

            # Student optimizer
            optimizer = SGD(student.parameters(), lr=0.2, weight_decay=0.0001, momentum=0.9)
            scheduler = CosineAnnealingLR(optimizer, 200, eta_min=2e-4)

            # Data generation loop
            self.logger.info(f"Starting synthetic data generation (syn_round={syn_round})...")
            for it in range(syn_round):
                # Generate a batch of synthetic data
                synthesizer.synthesize()

                # Start KD training from warmup round
                if it >= warmup:
                    # Create DataLoader for synthetic data
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

                        # KD train Student
                        self._kd_train_student(
                            student=student,
                            teacher=teacher,
                            criterion=criterion,
                            optimizer=optimizer,
                            data_loader=syn_loader,
                            kd_steps=kd_steps
                        )

                        # Learning rate adjustment
                        scheduler.step()

                        self.logger.info(
                            f"Task {task_id}, Data Generation, Round {it + 1}/{syn_round} => "
                            f"Generated {len(syn_dataset)} synthetic samples"
                        )

            # Clean up hooks
            synthesizer.remove_hooks()

            self.logger.info(
                f"Task {task_id} data generation complete! Generated {len(os.listdir(task_dir))} data files"
            )
            self.logger.info(f"Synthetic data saved at: {task_dir}\n")

        except Exception as e:
            self.logger.error(f"Data generation failed: {e}", exc_info=True)

    def _kd_train_student(self, student, teacher, criterion, optimizer,
                         data_loader, kd_steps):
        """
        Train Student model with synthetic data for verification

        Args:
            student: Student model
            teacher: Teacher model (current global model)
            criterion: KL divergence loss
            optimizer: Optimizer
            data_loader: Synthetic data DataLoader
            kd_steps: Training steps
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
