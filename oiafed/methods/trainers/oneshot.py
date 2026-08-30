"""Server orchestration for native one-shot federation baselines."""

from __future__ import annotations

import copy
import math
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ...core.trainer import Trainer
from ...core.types import ClientUpdate, RoundMetrics, RoundResult, TrainResult
from ...registry import trainer
from ..models.oneshot import (
    DataFreeGenerator,
    FAFIServerModel,
    FedCGSServerModel,
    FuseFLResNet18,
    OneShotEnsemble,
)


def _copy_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def _mean_states(states: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not states:
        raise ValueError("cannot average an empty state list")
    keys = states[0].keys()
    if any(state.keys() != states[0].keys() for state in states[1:]):
        raise ValueError("client state_dict keys do not match")
    result: dict[str, torch.Tensor] = {}
    for key in keys:
        values = [state[key].detach().cpu() for state in states]
        if values[0].is_floating_point() or values[0].is_complex():
            result[key] = torch.stack(values).mean(dim=0)
        else:
            result[key] = values[0].clone()
    return result


class _BNFeatureHook:
    def __init__(self, module: nn.BatchNorm2d):
        self.value = torch.tensor(0.0)
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, module: nn.BatchNorm2d, inputs: tuple[torch.Tensor, ...], output: Any) -> None:
        del output
        features = inputs[0]
        mean = features.mean(dim=(0, 2, 3))
        variance = features.var(dim=(0, 2, 3), unbiased=False)
        self.value = (mean - module.running_mean).norm(2) + (variance - module.running_var).norm(2)

    def close(self) -> None:
        self.handle.remove()


class _OneShotTrainer(Trainer):
    algorithm = "oneshot"

    async def run(self) -> dict[str, Any]:
        rounds = int(self.config.get("max_rounds", self.config.get("num_rounds", 1)))
        if rounds != 1:
            raise ValueError(f"{self.algorithm} uses one federation session; max_rounds must be 1")
        if float(self.config.get("client_fraction", 1.0)) != 1.0:
            raise ValueError(f"{self.algorithm} requires the complete configured client cohort")
        self._config["max_rounds"] = 1
        return await super().run()

    async def _collect_updates(
        self,
        selected: list[Any],
        fit_config: Optional[dict[str, Any]] = None,
    ) -> list[ClientUpdate]:
        config = dict(self.config.get("fit_config", {}))
        config.update(fit_config or {})
        config.setdefault("epochs", int(self.config.get("local_epochs", 100)))
        results = await self.collect_results(
            selected,
            "fit",
            config,
            timeout=self.config.get("fit_timeout"),
        )
        updates: list[ClientUpdate] = []
        failures: list[Exception] = []
        for index, (proxy, result) in enumerate(zip(selected, results)):
            client_id = getattr(proxy, "_target_id", f"learner_{index}")
            if isinstance(result, Exception):
                failures.append(result)
                continue
            if not isinstance(result, TrainResult) and type(result).__name__ == "TrainResult":
                result = TrainResult.from_dict(result.to_dict())
            if not isinstance(result, TrainResult):
                failures.append(TypeError(f"{client_id} returned {type(result).__name__}"))
                continue
            updates.append(ClientUpdate.from_result(client_id, result))
        if failures or len(updates) != len(selected):
            first = failures[0] if failures else "missing result"
            raise RuntimeError(f"{self.algorithm} requires every client; first failure: {first}")
        return updates

    async def _begin_round(self, round_num: int) -> None:
        if self.callbacks:
            await self.callbacks.on_round_begin(
                self,
                round_num,
                {"algorithm": self.algorithm},
            )

    async def _finalize_round(self, result: RoundResult) -> RoundResult:
        metrics = {
            key: value
            for key, value in result.metrics.metrics.items()
            if key.startswith("eval_")
        }
        if self.tracker and metrics:
            self.tracker.log_metrics(metrics, step=result.round_num)
        if self.callbacks:
            await self.callbacks.on_round_end(
                self,
                result.round_num,
                {"algorithm": self.algorithm, "round_result": result},
            )
        return result

    @staticmethod
    def _models_from_bundle(template: nn.Module, bundle: dict[str, Any]) -> list[nn.Module]:
        models = []
        for client in bundle["clients"]:
            local = copy.deepcopy(template).cpu()
            local.load_state_dict(client["state_dict"], strict=True)
            models.append(local)
        return models

    def _evaluate(self, model: nn.Module) -> dict[str, float]:
        if not self.has_global_test:
            return {}
        device_name = self.config.get("device")
        if device_name is None:
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        if str(device_name).startswith("cuda") and not torch.cuda.is_available():
            device_name = "cpu"
        device = torch.device(device_name)
        loader = DataLoader(
            self.test_dataset,
            batch_size=int(self.config.get("eval_batch_size", 256)),
            shuffle=False,
            num_workers=int(self.config.get("eval_num_workers", 0)),
            pin_memory=device.type == "cuda",
        )
        model.to(device).eval()
        total = correct = 0
        loss_sum = 0.0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True).long()
                logits = model(inputs)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                loss_sum += float(F.cross_entropy(logits, labels, reduction="sum"))
                correct += int((logits.argmax(1) == labels).sum())
                total += int(labels.numel())
        return {
            "eval_accuracy": correct / total if total else 0.0,
            "eval_loss": loss_sum / total if total else 0.0,
            "eval_samples": float(total),
        }

    def _round_result(
        self,
        round_num: int,
        updates: list[ClientUpdate],
        aggregated: Any,
        eval_metrics: dict[str, float],
        **metadata: Any,
    ) -> RoundResult:
        total = sum(update.num_samples for update in updates)
        train_loss = sum(
            float(update.metrics.get("loss", 0.0)) * update.num_samples for update in updates
        ) / max(total, 1)
        return RoundResult(
            round_num,
            updates,
            aggregated,
            RoundMetrics(round_num, len(updates), total, {"train_loss": train_loss, **eval_metrics}),
            {"algorithm": self.algorithm, **metadata},
        )


@trainer(name="ofedavg", description="One-shot FedAvg over independently trained clients", version="1.0")
class OFedAvgTrainer(_OneShotTrainer):
    algorithm = "ofedavg"

    async def train_round(self, round_num: int) -> RoundResult:
        await self._begin_round(round_num)
        selected = self.get_connected_learners()
        updates = await self._collect_updates(selected)
        bundle = self.aggregator.aggregate(updates, self.model)
        averaged = _mean_states([client["state_dict"] for client in bundle["clients"]])
        assert self.model is not None
        self.model.load_state_dict(averaged, strict=True)
        metrics = self._evaluate(self.model)
        return await self._finalize_round(
            self._round_result(round_num, updates, averaged, metrics, averaging="uniform")
        )


@trainer(name="ensemble", description="Direct uniform-logit one-shot ensemble", version="1.0")
class EnsembleTrainer(_OneShotTrainer):
    algorithm = "ensemble"

    async def train_round(self, round_num: int) -> RoundResult:
        await self._begin_round(round_num)
        selected = self.get_connected_learners()
        updates = await self._collect_updates(selected)
        bundle = self.aggregator.aggregate(updates, self.model)
        if self.model is None:
            raise ValueError("EnsembleTrainer requires a server model template")
        ensemble = OneShotEnsemble(self._models_from_bundle(self.model, bundle))
        self._model = ensemble
        metrics = self._evaluate(ensemble)
        return await self._finalize_round(
            self._round_result(round_num, updates, bundle, metrics, averaging="uniform_logits")
        )


@trainer(name="fafi", description="FAFI prototype and feature aggregation", version="1.0")
class FAFITrainer(_OneShotTrainer):
    algorithm = "fafi"

    async def train_round(self, round_num: int) -> RoundResult:
        await self._begin_round(round_num)
        selected = self.get_connected_learners()
        updates = await self._collect_updates(selected)
        bundle = self.aggregator.aggregate(updates, self.model)
        if self.model is None:
            raise ValueError("FAFITrainer requires a server model template")
        prototypes = [torch.as_tensor(client["metadata"]["prototypes"]).float() for client in bundle["clients"]]
        expected = prototypes[0].shape
        if any(proto.shape != expected for proto in prototypes[1:]):
            raise ValueError("FAFI client prototype shapes do not match")
        global_prototypes = torch.stack(prototypes).mean(dim=0)
        models = self._models_from_bundle(self.model, bundle)
        counts = [client["num_samples"] for client in bundle["clients"]]
        server_model = FAFIServerModel(models, counts, global_prototypes)
        self._model = server_model
        metrics = self._evaluate(server_model)
        return await self._finalize_round(
            self._round_result(
                round_num,
                updates,
                {**bundle, "global_prototypes": global_prototypes},
                metrics,
                feature_weighting="data_size",
                prototype_weighting="uniform",
            )
        )


@trainer(name="fedcgs", description="FedCGS pooled-statistics LDA server", version="1.0")
class FedCGSTrainer(_OneShotTrainer):
    algorithm = "fedcgs"

    @staticmethod
    def aggregate_statistics(
        metadata: list[dict[str, Any]],
        ridge: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        class_sums = sum((torch.as_tensor(item["class_sums"]).double() for item in metadata))
        class_counts = sum((torch.as_tensor(item["class_count_vector"]).double() for item in metadata))
        feature_sum = sum((torch.as_tensor(item["feature_sum"]).double() for item in metadata))
        second = sum((torch.as_tensor(item["second_moment"]).double() for item in metadata))
        if (class_counts <= 0).any():
            missing = (class_counts <= 0).nonzero().flatten().tolist()
            raise ValueError(f"FedCGS has no global samples for classes {missing}")
        count = class_counts.sum()
        if count <= 1:
            raise ValueError("FedCGS covariance requires at least two samples")
        mean = feature_sum / count
        covariance = (second - torch.outer(feature_sum, mean)) / (count - 1)
        covariance = (covariance + covariance.T) / 2
        covariance = covariance + float(ridge) * torch.eye(covariance.size(0), dtype=covariance.dtype)
        try:
            precision = torch.linalg.inv(covariance)
        except torch.linalg.LinAlgError:
            precision = torch.linalg.pinv(covariance)
        class_means = class_sums / class_counts[:, None]
        weights = class_means @ precision
        bias = -0.5 * (class_means * weights).sum(dim=1) + torch.log(class_counts / count)
        return weights.float(), bias.float(), covariance.float()

    async def train_round(self, round_num: int) -> RoundResult:
        await self._begin_round(round_num)
        selected = self.get_connected_learners()
        updates = await self._collect_updates(selected, {"epochs": 0})
        bundle = self.aggregator.aggregate(updates, self.model)
        weights, bias, covariance = self.aggregate_statistics(
            [client["metadata"] for client in bundle["clients"]],
            ridge=float(self.config.get("covariance_ridge", 0.0)),
        )
        if self.model is None:
            raise ValueError("FedCGSTrainer requires the shared feature backbone")
        server_model = FedCGSServerModel(copy.deepcopy(self.model).cpu(), weights, bias)
        self._model = server_model
        metrics = self._evaluate(server_model)
        aggregated = {"format": "oiafed.fedcgs.statistics", "weight": weights, "bias": bias, "covariance": covariance}
        return await self._finalize_round(
            self._round_result(round_num, updates, aggregated, metrics)
        )


@trainer(name="fusefl", description="Four-phase progressively expandable FuseFL", version="1.0")
class FuseFLTrainer(_OneShotTrainer):
    algorithm = "fusefl"

    async def train_round(self, round_num: int) -> RoundResult:
        await self._begin_round(round_num)
        selected = self.get_connected_learners()
        if not selected:
            raise RuntimeError("FuseFL has no connected learners")
        if not isinstance(self.model, FuseFLResNet18):
            raise TypeError("FuseFLTrainer requires model.type=fusefl_resnet18")
        stages = int(self.config.get("split_num", 4))
        if stages != 4:
            raise ValueError("the integrated ResNet-18 FuseFL definition uses split_num=4")

        all_updates: list[ClientUpdate] = []
        stage_states: list[list[dict[str, torch.Tensor]]] = []
        classifiers: list[dict[str, torch.Tensor]] = []
        previous: list[dict[str, torch.Tensor]] = []
        for stage in range(stages):
            contexts = await self.broadcast_to_selected(
                selected,
                "set_fusefl_context",
                {"stage": stage, "previous_branch_states": previous},
            )
            errors = [item for item in contexts if isinstance(item, Exception)]
            if errors:
                raise RuntimeError(f"FuseFL stage {stage} setup failed: {errors[0]}")
            updates = await self._collect_updates(
                selected,
                {"epochs": int(self.config.get("local_epochs_per_stage", self.config.get("local_epochs", 50)))},
            )
            if any(int(update.metadata.get("stage", -1)) != stage for update in updates):
                raise RuntimeError(f"FuseFL received an update for the wrong stage {stage}")
            previous = [_copy_state(update.weights) for update in updates]
            stage_states.append(previous)
            classifiers = [update.metadata["classifier_state"] for update in updates]
            all_updates = updates

        server_model = copy.deepcopy(self.model).cpu()
        for stage, branches in enumerate(stage_states):
            server_model.install_fused_stage(stage, branches)
        classifier_state = _mean_states(classifiers)
        server_model.classifier.load_state_dict(classifier_state, strict=True)
        self._model = server_model
        metrics = self._evaluate(server_model)
        aggregated = {
            "format": "oiafed.fusefl.progressive",
            "split_num": stages,
            "stage_branch_states": stage_states,
            "classifier_state": classifier_state,
        }
        return await self._finalize_round(
            self._round_result(
                round_num,
                all_updates,
                aggregated,
                metrics,
                communication_phases=stages + 1,
            )
        )


@trainer(name="coboosting", description="Co-Boosting data-free ensemble distillation", version="1.0")
class CoBoostingTrainer(_OneShotTrainer):
    algorithm = "coboosting"

    @staticmethod
    def _kl(student: torch.Tensor, teacher: torch.Tensor, temperature: float) -> torch.Tensor:
        return F.kl_div(
            F.log_softmax(student / temperature, dim=1),
            F.softmax(teacher / temperature, dim=1),
            reduction="batchmean",
        ) * temperature**2

    def _normalizer(self, inputs: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.config.get("normalization_mean", [0.5, 0.5, 0.5]), device=inputs.device).view(1, -1, 1, 1)
        std = torch.as_tensor(self.config.get("normalization_std", [0.5, 0.5, 0.5]), device=inputs.device).view(1, -1, 1, 1)
        return (inputs - mean) / std

    def _synthesize(
        self,
        teacher: OneShotEnsemble,
        student: nn.Module,
        generator: DataFreeGenerator,
        hooks: list[_BNFeatureHook],
        device: torch.device,
        epoch: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        generator.reset_parameters()
        generator.train()
        batch_size = int(self.config.get("synthesis_batch_size", self.config.get("batch_size", 128)))
        latent = int(self.config.get("latent_dim", 256))
        classes = int(self.config.get("num_classes", 10))
        noise = torch.randn(batch_size, latent, device=device, requires_grad=True)
        targets = torch.randint(classes, (batch_size,), device=device)
        optimizer = torch.optim.Adam(
            [{"params": generator.parameters()}, {"params": [noise]}],
            lr=float(self.config.get("generator_lr", 1e-3)),
            betas=(0.5, 0.999),
        )
        best_loss = math.inf
        best_inputs: torch.Tensor | None = None
        for _ in range(int(self.config.get("generator_steps", 30))):
            optimizer.zero_grad(set_to_none=True)
            inputs = self._normalizer(generator(noise))
            teacher_logits = teacher(inputs)
            probabilities = F.softmax(teacher_logits, dim=1)
            selected = probabilities.gather(1, targets[:, None]).squeeze(1)
            hard_power = float(self.config.get("hard_sample_power", 1.0))
            one_hot = ((1 - selected.detach()).pow(hard_power) * F.cross_entropy(teacher_logits, targets, reduction="none")).mean()
            student_logits = student(inputs)
            adversarial = -self._kl(student_logits, teacher_logits, 3.0)
            bn_loss = torch.stack([hook.value.to(device) for hook in hooks]).mean() if hooks else inputs.new_zeros(())
            loss = (
                float(self.config.get("bn_weight", 0.0)) * bn_loss
                + float(self.config.get("one_hot_weight", 1.0)) * one_hot
                + float(self.config.get("adversarial_weight", 1.0)) * adversarial
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 10.0)
            optimizer.step()
            if float(loss.detach()) < best_loss:
                best_loss = float(loss.detach())
                best_inputs = inputs.detach().clone()
        if best_inputs is None:
            raise RuntimeError("Co-Boosting synthesis produced no batch")
        return best_inputs, targets.detach()

    def _adjust_weights(
        self,
        teacher: OneShotEnsemble,
        images: torch.Tensor,
        labels: torch.Tensor,
        epoch: int,
    ) -> None:
        if epoch == 0:
            return
        batch_size = int(self.config.get("batch_size", 128))
        device = teacher.mixture_weights.device
        for start in range(0, images.size(0), batch_size):
            batch = images[start : start + batch_size].to(device)
            targets = labels[start : start + batch_size].to(device)
            original = teacher.mixture_weights.detach()
            mix = original.clone().requires_grad_(True)
            client_logits = torch.stack([model(batch) for model in teacher.models])
            normalized = mix.clamp_min(0) / mix.clamp_min(0).sum().clamp_min(1e-12)
            loss = F.cross_entropy((normalized[:, None, None] * client_logits).sum(0), targets)
            gradient = torch.autograd.grad(loss, mix)[0]
            step = float(self.config.get("weight_step", 0.01)) * float(self.config.get("weight_decay_factor", 0.99)) ** epoch
            updated = original - step * gradient.sign()
            updated = torch.clamp(original + torch.clamp(updated - original, -1, 1), 0, 1)
            teacher.mixture_weights.data.copy_(updated)

    def _distill(
        self,
        teacher: OneShotEnsemble,
        student: nn.Module,
        optimizer: torch.optim.Optimizer,
        pool: list[torch.Tensor],
        device: torch.device,
    ) -> None:
        images = torch.cat(pool, dim=0)
        batch_size = int(self.config.get("batch_size", 128))
        # cb_kd_train in the reference iterates once over the accumulated
        # synthetic ImagePool; kd_steps is an upper bound rather than a fixed
        # number of repeated updates.
        steps = min(
            int(self.config.get("kd_steps", 400)),
            math.ceil(images.size(0) / batch_size),
        )
        temperature = float(self.config.get("kd_temperature", 4.0))
        ods_eta = float(self.config.get("ods_eta", 8.0))
        student.train()
        for _ in range(steps):
            indices = torch.randint(images.size(0), (min(batch_size, images.size(0)),))
            batch = images[indices].to(device).detach().clone().requires_grad_(True)
            teacher_logits = teacher(batch)
            random_weights = torch.empty_like(teacher_logits).uniform_(-1, 1)
            ods_objective = (random_weights * F.softmax(teacher_logits / 4.0, dim=1)).sum()
            image_gradient = torch.autograd.grad(ods_objective, batch)[0]
            perturbed = (batch + ods_eta * image_gradient.sign()).detach()
            with torch.no_grad():
                targets = teacher(perturbed)
            optimizer.zero_grad(set_to_none=True)
            logits = student(perturbed)
            loss = self._kl(logits, targets, temperature)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 10.0)
            optimizer.step()

    def _run_server_distillation(self, models: list[nn.Module]) -> tuple[nn.Module, dict[str, float]]:
        if self.model is None:
            raise ValueError("CoBoostingTrainer requires a student model template")
        device_name = self.config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        if str(device_name).startswith("cuda") and not torch.cuda.is_available():
            device_name = "cpu"
        device = torch.device(device_name)
        models = [model.to(device).eval() for model in models]
        teacher = OneShotEnsemble(models, trainable_weights=True).to(device).eval()
        teacher.mixture_weights.requires_grad_(False)
        for model in teacher.models:
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        student = copy.deepcopy(self.model).to(device)
        generator = DataFreeGenerator(
            latent_dim=int(self.config.get("latent_dim", 256)),
            width=int(self.config.get("generator_width", 64)),
            image_size=int(self.config.get("image_size", 32)),
            channels=int(self.config.get("image_channels", 3)),
        ).to(device)
        hooks = [
            _BNFeatureHook(module)
            for model in teacher.models
            for module in model.modules()
            if isinstance(module, nn.BatchNorm2d)
        ]
        optimizer = torch.optim.SGD(
            student.parameters(),
            lr=float(self.config.get("kd_learning_rate", 0.01)),
            momentum=0.9,
            weight_decay=float(self.config.get("weight_decay", 1e-4)),
        )
        epochs = int(self.config.get("server_epochs", 200))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
        pool: list[torch.Tensor] = []
        pool_labels: list[torch.Tensor] = []
        best_state = _copy_state(student.state_dict())
        best_accuracy = -1.0
        try:
            for epoch in range(epochs):
                synthetic, labels = self._synthesize(teacher, student, generator, hooks, device, epoch)
                if pool:
                    previous_images = torch.cat(pool, dim=0)
                    previous_labels = torch.cat(pool_labels, dim=0)
                    self._adjust_weights(teacher, previous_images, previous_labels, epoch)
                pool.append(synthetic.detach().cpu())
                pool_labels.append(labels.detach().cpu())
                self._distill(teacher, student, optimizer, pool, device)
                scheduler.step()
                metrics = self._evaluate(student)
                accuracy = float(metrics.get("eval_accuracy", 0.0))
                if accuracy >= best_accuracy:
                    best_accuracy = accuracy
                    best_state = _copy_state(student.state_dict())
        finally:
            for hook in hooks:
                hook.close()
        student.load_state_dict(best_state, strict=True)
        return student.cpu(), {
            "server_best_accuracy": best_accuracy,
            "teacher_weights": teacher.normalized_weights().detach().cpu().tolist(),
            "synthetic_samples": float(sum(batch.size(0) for batch in pool)),
        }

    async def train_round(self, round_num: int) -> RoundResult:
        await self._begin_round(round_num)
        selected = self.get_connected_learners()
        updates = await self._collect_updates(selected)
        bundle = self.aggregator.aggregate(updates, self.model)
        if self.model is None:
            raise ValueError("CoBoostingTrainer requires a server model template")
        models = self._models_from_bundle(self.model, bundle)
        student, server_metadata = self._run_server_distillation(models)
        self._model = student
        metrics = self._evaluate(student)
        return await self._finalize_round(
            self._round_result(round_num, updates, bundle, metrics, **server_metadata)
        )


__all__ = [
    "OFedAvgTrainer",
    "EnsembleTrainer",
    "FAFITrainer",
    "FedCGSTrainer",
    "FuseFLTrainer",
    "CoBoostingTrainer",
]
