"""FedSRA one-shot client learner."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ....core.learner import Learner
from ....core.types import EvalResult, EpochMetrics, StepMetrics, TrainMetrics, TrainResult
from ....registry import learner
from ...fedsra import (
    extract_class_counts,
    generate_simplex_etf,
    joint_erl_loss,
)


@learner(
    name="fedsra",
    description="FedSRA one-shot learner with frozen ETF and ERL",
    version="1.0",
    author="FedSRA",
)
class FedSRALearner(Learner):
    def __init__(
        self,
        model: Any,
        datasets: Optional[Dict[str, Any]] = None,
        tracker: Optional[Any] = None,
        callbacks: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        node_id: Optional[str] = None,
        **_: Any,
    ):
        super().__init__(model, None, tracker, callbacks, config, node_id)
        self._datasets = datasets or {}
        self._device_obj = torch.device("cpu")
        self._optimizer = None
        self._scheduler = None
        self._train_dataloader = None

        self._num_classes = int(self._config.get("num_classes", getattr(model, "num_classes", 10)))
        self._feature_dim = int(self._config.get("feature_dim", getattr(model, "feature_dim", 256)))
        self._etf_seed = int(self._config.get("etf_seed", 42))
        self._loss_type = str(self._config.get("loss_type", "J")).upper()
        self._alignment_weight = float(self._config.get("alignment_weight", 0.5))
        self._temperature = float(self._config.get("temperature", 0.1))
        self._etf = generate_simplex_etf(
            self._num_classes,
            self._feature_dim,
            self._etf_seed,
        )

        train_dataset = self._first_dataset("train")
        self._class_counts = (
            extract_class_counts(train_dataset, self._num_classes)
            if train_dataset is not None
            else {}
        )

    def _first_dataset(self, split: str) -> Any | None:
        datasets = self._datasets.get(split, [])
        return datasets[0] if datasets else None

    def set_fedsra_context(self, context: Dict[str, Any]) -> bool:
        """Receive the shared frozen ETF from the server before local training."""
        etf = torch.as_tensor(context["etf"]).detach().clone().float()
        expected = (int(context["num_classes"]), int(context["feature_dim"]))
        if tuple(etf.shape) != expected:
            raise ValueError(f"ETF has shape {tuple(etf.shape)}, expected {expected}")
        if getattr(self.model, "feature_dim", expected[1]) != expected[1]:
            raise ValueError("Learner model feature dimension does not match the shared ETF")

        self._num_classes = expected[0]
        self._feature_dim = expected[1]
        self._etf_seed = int(context["etf_seed"])
        self._etf = etf
        train_dataset = self._first_dataset("train")
        if train_dataset is not None:
            self._class_counts = extract_class_counts(train_dataset, self._num_classes)
        return True

    async def setup(self, config: Dict[str, Any]) -> None:
        device_name = config.get("device", self._config.get("device"))
        if device_name is None:
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        if str(device_name).startswith("cuda") and not torch.cuda.is_available():
            self.logger.warning("CUDA is unavailable; falling back to CPU")
            device_name = "cpu"
        self._device_obj = torch.device(device_name)
        self.model.to(self._device_obj)
        self.model.train()
        self._etf = self._etf.to(self._device_obj)

        batch_size = int(config.get("batch_size", self._config.get("batch_size", 64)))
        train_dataset = self._first_dataset("train")
        if train_dataset is None:
            raise ValueError("FedSRALearner requires a training dataset")
        num_workers = int(config.get("num_workers", self._config.get("num_workers", 0)))
        self._train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=self._device_obj.type == "cuda",
            drop_last=bool(config.get("drop_last", self._config.get("drop_last", False))),
            persistent_workers=(
                num_workers > 0
                and bool(
                    config.get(
                        "persistent_workers",
                        self._config.get("persistent_workers", False),
                    )
                )
            ),
        )

        learning_rate = float(config.get("learning_rate", self._config.get("learning_rate", 1e-3)))
        weight_decay = float(config.get("weight_decay", self._config.get("weight_decay", 0.0)))
        self._optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        epochs = int(config.get("epochs", 1))
        self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self._optimizer,
            T_max=max(1, epochs),
        )
        self._loss_type = str(config.get("loss_type", self._loss_type)).upper()

    def _checkpoint_path(self, config: Dict[str, Any]) -> Path | None:
        checkpoint_dir = config.get(
            "checkpoint_dir",
            self._config.get("checkpoint_dir"),
        )
        if not checkpoint_dir:
            return None
        return Path(str(checkpoint_dir)) / self.node_id / "backbone.pt"

    async def fit(self, config: Optional[Dict[str, Any]] = None) -> TrainResult:
        run_config = {**self._config, **(config or {})}
        checkpoint_path = self._checkpoint_path(run_config)
        resume = bool(run_config.get("resume", False))

        if resume and checkpoint_path is not None and checkpoint_path.exists():
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
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
            state_dict = {
                key: value.detach().cpu()
                for key, value in self.model.state_dict().items()
            }
            torch.save(
                {
                    "state_dict": state_dict,
                    "final_loss": float(result.metrics.final_loss),
                    "num_samples": int(result.num_samples),
                    "metadata": result.metadata,
                },
                temporary_path,
            )
            os.replace(temporary_path, checkpoint_path)
        return result

    async def train_step(self, batch: Any, batch_idx: int) -> StepMetrics:
        del batch_idx
        inputs, labels = batch
        inputs = inputs.to(self._device_obj, non_blocking=True)
        labels = labels.to(self._device_obj, non_blocking=True).long()

        assert self._optimizer is not None
        self._optimizer.zero_grad(set_to_none=True)
        use_bf16 = bool(self._config.get("use_bf16", False)) and self._device_obj.type == "cuda"
        with torch.autocast(
            device_type=self._device_obj.type,
            dtype=torch.bfloat16,
            enabled=use_bf16,
        ):
            features = self.model(inputs)
            loss = joint_erl_loss(
                features,
                labels,
                self._etf,
                loss_type=self._loss_type,
                alignment_weight=self._alignment_weight,
                temperature=self._temperature,
                num_local_classes=len(self._class_counts),
            )
        loss.backward()
        self._optimizer.step()

        logits = F.normalize(features.detach(), dim=1) @ self._etf.T
        accuracy = (logits.argmax(dim=1) == labels).float().mean().item()
        return StepMetrics(
            loss=float(loss.detach().item()),
            batch_size=int(labels.size(0)),
            metrics={"accuracy": accuracy},
        )

    async def train_epoch(self, epoch_idx: int) -> EpochMetrics:
        metrics = await super().train_epoch(epoch_idx)
        if self._scheduler is not None:
            self._scheduler.step()
        return metrics

    async def evaluate(self, config: Optional[Dict[str, Any]] = None) -> EvalResult:
        dataset = self._first_dataset("test")
        if dataset is None:
            return EvalResult(num_samples=0, metrics={})
        batch_size = int((config or {}).get("batch_size", self._config.get("batch_size", 64)))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        self.model.eval()
        total = correct = 0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(self._device_obj)
                labels = labels.to(self._device_obj).long()
                logits = self.model(inputs) @ self._etf.T
                correct += int((logits.argmax(dim=1) == labels).sum().item())
                total += int(labels.size(0))
        return EvalResult(
            num_samples=total,
            metrics={"accuracy": correct / total if total else 0.0},
        )

    def get_dataloader(self) -> Any:
        return self._train_dataloader

    def get_num_samples(self) -> int:
        return sum(self._class_counts.values())

    def get_metadata(self) -> Dict[str, Any]:
        metadata = super().get_metadata()
        metadata.update(
            {
                "algorithm": "fedsra",
                "class_counts": dict(self._class_counts),
                "num_classes": self._num_classes,
                "feature_dim": self._feature_dim,
                "etf_seed": self._etf_seed,
                "loss_type": self._loss_type,
            }
        )
        return metadata


__all__ = ["FedSRALearner"]
