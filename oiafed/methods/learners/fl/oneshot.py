"""Client learners for one-shot federation baselines."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ....core.learner import Learner
from ....core.types import EvalResult, StepMetrics, TrainMetrics, TrainResult
from ....registry import learner
from ...fedsra import extract_class_counts
from ...models.oneshot import FuseFLResNet18


def _supcon_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Exact two-view supervised contrastive objective used in FAFI."""
    batch, views, dim = features.shape
    contrast = torch.cat(torch.unbind(features, dim=1), dim=0)
    logits = contrast @ contrast.T / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    label_mask = labels.view(-1, 1).eq(labels.view(1, -1)).float().repeat(views, views)
    self_mask = ~torch.eye(batch * views, dtype=torch.bool, device=features.device)
    label_mask = label_mask * self_mask.float()
    exp_logits = torch.exp(logits) * self_mask.float()
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    positive_count = label_mask.sum(dim=1).clamp_min(1.0)
    return -((label_mask * log_prob).sum(dim=1) / positive_count).mean()


class _SupervisedOneShotLearner(Learner):
    algorithm = "oneshot"

    def __init__(
        self,
        model: Any,
        datasets: Optional[dict[str, Any]] = None,
        tracker: Optional[Any] = None,
        callbacks: Optional[Any] = None,
        config: Optional[dict[str, Any]] = None,
        node_id: Optional[str] = None,
        **_: Any,
    ):
        super().__init__(model, None, tracker, callbacks, config, node_id)
        self._datasets = datasets or {}
        self._device_obj = torch.device("cpu")
        self._optimizer: torch.optim.Optimizer | None = None
        self._scheduler: Any = None
        self._train_dataloader: DataLoader | None = None
        self._num_classes = int(self.config.get("num_classes", getattr(model, "num_classes", 10)))
        train_dataset = self._first_dataset("train")
        self._class_counts = extract_class_counts(train_dataset, self._num_classes) if train_dataset is not None else {}

    def _first_dataset(self, split: str) -> Any | None:
        values = self._datasets.get(split, [])
        return values[0] if values else None

    def _checkpoint_path(self, config: dict[str, Any]) -> Path | None:
        checkpoint_dir = config.get("checkpoint_dir", self.config.get("checkpoint_dir"))
        if not checkpoint_dir:
            return None
        filename = "model.pt"
        if self.algorithm == "fusefl":
            filename = f"stage_{int(getattr(self, '_fusefl_stage', 0))}.pt"
        return Path(str(checkpoint_dir)) / self.node_id / filename

    async def fit(self, config: Optional[dict[str, Any]] = None) -> TrainResult:
        run_config = {**self.config, **(config or {})}
        checkpoint_path = self._checkpoint_path(run_config)
        resume = bool(run_config.get("resume", False))
        signature = run_config.get("checkpoint_signature")
        requested_epochs = int(run_config.get("epochs", 1))

        if resume and checkpoint_path is not None and checkpoint_path.exists():
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            signature_matches = signature is None or payload.get("signature") == signature
            epochs_match = int(payload.get("epochs", -1)) == requested_epochs
            if signature_matches and epochs_match:
                self.model.load_state_dict(payload["state_dict"], strict=True)
                final_loss = float(payload.get("final_loss", 0.0))
                num_samples = int(payload.get("num_samples", self.get_num_samples()))
                metadata = self.get_metadata()
                metadata["resumed_from_checkpoint"] = True
                return TrainResult(
                    weights=self.get_weights(),
                    num_samples=num_samples,
                    metrics=TrainMetrics(
                        total_epochs=0,
                        final_loss=final_loss,
                        total_samples=num_samples,
                        metrics={"loss": final_loss, "resumed": 1.0},
                    ),
                    metadata=metadata,
                )

        result = await super().fit(config)
        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = checkpoint_path.with_suffix(".pt.tmp")
            torch.save(
                {
                    "state_dict": {
                        key: value.detach().cpu()
                        for key, value in self.model.state_dict().items()
                    },
                    "final_loss": float(result.metrics.final_loss),
                    "num_samples": int(result.num_samples),
                    "epochs": requested_epochs,
                    "signature": signature,
                },
                temporary_path,
            )
            os.replace(temporary_path, checkpoint_path)
        return result

    async def setup(self, config: dict[str, Any]) -> None:
        device = config.get("device", self.config.get("device"))
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self._device_obj = torch.device(device)
        self.model.to(self._device_obj).train()

        dataset = self._first_dataset("train")
        if dataset is None:
            raise ValueError(f"{type(self).__name__} requires a train dataset")
        workers = int(config.get("num_workers", self.config.get("num_workers", 0)))
        self._train_dataloader = DataLoader(
            dataset,
            batch_size=int(config.get("batch_size", self.config.get("batch_size", 128))),
            shuffle=True,
            num_workers=workers,
            drop_last=bool(config.get("drop_last", self.config.get("drop_last", False))),
            pin_memory=self._device_obj.type == "cuda",
            persistent_workers=workers > 0 and bool(config.get("persistent_workers", False)),
        )
        self._optimizer = torch.optim.SGD(
            (parameter for parameter in self.model.parameters() if parameter.requires_grad),
            lr=float(config.get("learning_rate", self.config.get("learning_rate", 0.01))),
            momentum=float(config.get("momentum", self.config.get("momentum", 0.9))),
            weight_decay=float(config.get("weight_decay", self.config.get("weight_decay", 0.0))),
        )
        scheduler = str(config.get("scheduler", self.config.get("scheduler", "none"))).lower()
        epochs = max(1, int(config.get("epochs", 1)))
        self._scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(self._optimizer, T_max=epochs)
            if scheduler == "cosine"
            else None
        )

    async def train_step(self, batch: Any, batch_idx: int) -> StepMetrics:
        del batch_idx
        inputs, labels = batch[:2]
        inputs = inputs.to(self._device_obj, non_blocking=True)
        labels = labels.to(self._device_obj, non_blocking=True).long()
        assert self._optimizer is not None
        self._optimizer.zero_grad(set_to_none=True)
        outputs = self.model(inputs)
        logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        self._optimizer.step()
        accuracy = (logits.detach().argmax(dim=1) == labels).float().mean().item()
        return StepMetrics(float(loss.detach()), int(labels.size(0)), {"accuracy": accuracy})

    async def train_epoch(self, epoch_idx: int):
        metrics = await super().train_epoch(epoch_idx)
        if self._scheduler is not None:
            self._scheduler.step()
        return metrics

    async def evaluate(self, config: Optional[dict[str, Any]] = None) -> EvalResult:
        dataset = self._first_dataset("test")
        if dataset is None:
            return EvalResult(0, {})
        loader = DataLoader(dataset, batch_size=int((config or {}).get("batch_size", 256)), shuffle=False)
        self.model.eval()
        total = correct = 0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(self._device_obj)
                labels = labels.to(self._device_obj).long()
                outputs = self.model(inputs)
                logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
                correct += int((logits.argmax(1) == labels).sum())
                total += int(labels.numel())
        return EvalResult(total, {"accuracy": correct / total if total else 0.0})

    def get_num_samples(self) -> int:
        return sum(self._class_counts.values())

    def get_metadata(self) -> dict[str, Any]:
        metadata = super().get_metadata()
        metadata.update(
            {
                "algorithm": self.algorithm,
                "num_classes": self._num_classes,
                "class_counts": dict(self._class_counts),
            }
        )
        return metadata


@learner(name="ofedavg", description="Independent supervised client training for O-FedAvg", version="1.0")
class OFedAvgLearner(_SupervisedOneShotLearner):
    algorithm = "ofedavg"


@learner(name="ensemble", description="Independent supervised client training for direct Ensemble", version="1.0")
class EnsembleLearner(_SupervisedOneShotLearner):
    algorithm = "ensemble"


@learner(name="coboosting", description="Independent supervised teachers for Co-Boosting", version="1.0")
class CoBoostingLearner(_SupervisedOneShotLearner):
    algorithm = "coboosting"


@learner(name="fafi", description="FAFI two-view learnable-prototype client learner", version="1.0")
class FAFILearner(_SupervisedOneShotLearner):
    algorithm = "fafi"

    def _two_views(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # This mirrors get_supcon_transform in the FAFI artifact.  CIFAR-10 and
        # Tiny ImageNet arrive as raw tensors; CIFAR-100 intentionally arrives
        # pre-normalised because that is also what the reference loader does.
        try:
            from torchvision import transforms

            if self._num_classes == 100:
                mean = (0.5071, 0.4867, 0.4408)
                std = (0.2675, 0.2565, 0.2761)
            else:
                mean = (0.5, 0.5, 0.5)
                std = (0.5, 0.5, 0.5)
            transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(inputs.shape[-1], scale=(0.2, 1.0), antialias=False),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.Normalize(mean, std),
                ]
            )
            first = torch.stack([transform(image) for image in inputs])
            second = torch.stack([transform(image) for image in inputs])
        except Exception:
            first, second = inputs.clone(), inputs.clone()
        return first, second

    async def train_epoch(self, epoch_idx: int):
        if bool(self.config.get("reset_optimizer_each_epoch", True)):
            # The reference invokes ours_local_training(..., local_epochs=1)
            # once per pseudo-round, which recreates SGD and clears momentum.
            self._optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=float(self.config.get("learning_rate", 0.05)),
                momentum=float(self.config.get("momentum", 0.9)),
                weight_decay=float(self.config.get("weight_decay", 1e-4)),
            )
        return await super().train_epoch(epoch_idx)

    async def train_step(self, batch: Any, batch_idx: int) -> StepMetrics:
        del batch_idx
        inputs, labels = batch[:2]
        inputs = inputs.to(self._device_obj, non_blocking=True)
        labels = labels.to(self._device_obj, non_blocking=True).long()
        if bool(self.config.get("two_view_augmentation", True)):
            view1, view2 = self._two_views(inputs)
        else:
            view1, view2 = inputs, inputs
        augmented = torch.cat([view1, view2], dim=0)
        augmented_labels = torch.cat([labels, labels], dim=0)

        assert self._optimizer is not None
        self._optimizer.zero_grad(set_to_none=True)
        logits, features = self.model(augmented)
        batch_size = labels.size(0)
        first, second = features[:batch_size], features[batch_size:]
        pair_features = torch.stack([first, second], dim=1)
        prototypes = self.model.learnable_proto
        classification = F.cross_entropy(logits, augmented_labels)
        supervised_contrastive = _supcon_loss(pair_features, labels, temperature=0.07)
        prototype_feature = F.cross_entropy(features @ prototypes.T, augmented_labels)
        prototype_self = F.cross_entropy(
            prototypes @ prototypes.T,
            torch.arange(prototypes.size(0), device=prototypes.device),
        )
        loss = classification + supervised_contrastive + prototype_feature + prototype_self
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self._optimizer.step()
        accuracy = (logits.detach().argmax(1) == augmented_labels).float().mean().item()
        return StepMetrics(float(loss.detach()), int(augmented_labels.numel()), {"accuracy": accuracy})

    def get_metadata(self) -> dict[str, Any]:
        metadata = super().get_metadata()
        metadata["prototypes"] = self.model.get_proto().detach().cpu().clone()
        metadata["feature_dim"] = int(self.model.feature_dim)
        return metadata


@learner(name="fedcgs", description="FedCGS sufficient-statistics client", version="1.0")
class FedCGSLearner(_SupervisedOneShotLearner):
    algorithm = "fedcgs"

    async def fit(self, config: Optional[dict[str, Any]] = None) -> TrainResult:
        run_config = {**self.config, **(config or {})}
        await self.setup(run_config)
        self.model.eval()
        class_sums: torch.Tensor | None = None
        class_counts = torch.zeros(self._num_classes, dtype=torch.float64)
        feature_sum: torch.Tensor | None = None
        second_moment: torch.Tensor | None = None
        with torch.no_grad():
            for inputs, labels in self.get_dataloader():
                inputs = inputs.to(self._device_obj)
                labels = labels.long()
                features = self.model.forward_features(inputs).double().cpu()
                if class_sums is None:
                    dimension = features.size(1)
                    class_sums = torch.zeros(self._num_classes, dimension, dtype=torch.float64)
                    feature_sum = torch.zeros(dimension, dtype=torch.float64)
                    second_moment = torch.zeros(dimension, dimension, dtype=torch.float64)
                assert feature_sum is not None and second_moment is not None
                class_sums.index_add_(0, labels, features)
                class_counts.index_add_(0, labels, torch.ones(labels.numel(), dtype=torch.float64))
                feature_sum += features.sum(dim=0)
                second_moment += features.T @ features
        if class_sums is None or feature_sum is None or second_moment is None:
            raise ValueError("FedCGS cannot aggregate an empty client")
        samples = int(class_counts.sum().item())
        metadata = self.get_metadata()
        metadata.update(
            {
                "class_sums": class_sums,
                "class_count_vector": class_counts,
                "feature_sum": feature_sum,
                "second_moment": second_moment,
                "feature_dim": int(feature_sum.numel()),
            }
        )
        return TrainResult(
            weights={},
            num_samples=samples,
            metrics=TrainMetrics(0, 0.0, samples, {"statistics_only": 1.0}),
            metadata=metadata,
        )

    async def train_step(self, batch: Any, batch_idx: int) -> StepMetrics:
        raise RuntimeError("FedCGS uploads statistics and does not run SGD")


@learner(name="fusefl", description="Progressive four-stage FuseFL client", version="1.0")
class FuseFLLearner(_SupervisedOneShotLearner):
    algorithm = "fusefl"

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        if not isinstance(self.model, FuseFLResNet18):
            raise TypeError("FuseFLLearner requires model.type=fusefl_resnet18")
        self._fusefl_stage = 0

    def set_fusefl_context(self, context: dict[str, Any]) -> bool:
        stage = int(context["stage"])
        if stage > 0:
            self.model.install_fused_stage(stage - 1, context["previous_branch_states"])
        self._fusefl_stage = stage
        return True

    def get_weights(self) -> dict[str, torch.Tensor]:
        return self.model.export_stage_state(self._fusefl_stage)

    def get_metadata(self) -> dict[str, Any]:
        metadata = super().get_metadata()
        metadata.update(
            {
                "stage": self._fusefl_stage,
                "classifier_state": {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.classifier.state_dict().items()
                },
            }
        )
        return metadata


__all__ = [
    "OFedAvgLearner",
    "EnsembleLearner",
    "CoBoostingLearner",
    "FAFILearner",
    "FedCGSLearner",
    "FuseFLLearner",
]
