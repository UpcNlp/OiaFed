"""Native OiaFed learner adapter for the validated FedEMoE client."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

from ....core.learner import Learner
from ....core.types import EvalResult, TrainMetrics, TrainResult
from ....registry import learner
from ...fedemoe_reference.client import FedEMoEClient


def _client_number(node_id: str) -> int:
    try:
        return int(node_id.rsplit("_", 1)[-1])
    except (TypeError, ValueError):
        return 0


@learner(
    name="fedemoe",
    description="FedEMoE local EDL training and evidence-profile extraction",
    version="1.0",
    author="FedEMoE Authors",
    upstream="Stephen-Chow1/FedEMoE-CEGA",
)
class FedEMoELearner(Learner):
    """Expose the upstream ``FedEMoEClient`` through OiaFed's learner API."""

    def __init__(
        self,
        model: Any,
        datasets: Optional[Dict[str, Any]] = None,
        tracker: Optional[Any] = None,
        callbacks: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        node_id: Optional[str] = None,
        **_: Any,
    ) -> None:
        super().__init__(model, None, tracker, callbacks, config, node_id)
        self._datasets = datasets or {}

        train_datasets = self._datasets.get("train", [])
        if not train_datasets:
            raise ValueError("FedEMoELearner requires one training dataset")
        self._train_dataset = train_datasets[0]

        requested_device = str(self._config.get("device", "cuda"))
        if requested_device == "cuda" and torch.cuda.is_available():
            self._device_obj = torch.device("cuda")
        else:
            self._device_obj = torch.device("cpu")

        self._train_loader = DataLoader(
            self._train_dataset,
            batch_size=int(self._config.get("batch_size", 64)),
            shuffle=True,
            num_workers=int(self._config.get("num_workers", 4)),
            pin_memory=True,
        )
        self._reference_client = FedEMoEClient(
            client_id=_client_number(self.node_id),
            dataloader=self._train_loader,
            device=self._device_obj,
            num_classes=int(
                self._config.get("num_classes", getattr(model, "num_classes", 10))
            ),
            lr=float(
                self._config.get("learning_rate", self._config.get("lr", 0.01))
            ),
            momentum=float(self._config.get("momentum", 0.9)),
            weight_decay=float(self._config.get("weight_decay", 1e-4)),
            optimizer_name=str(self._config.get("optimizer", "sgd")),
            edl_lambda1=float(self._config.get("edl_lambda1", 1.0)),
            edl_lambda2=float(self._config.get("edl_lambda2", 0.1)),
            annealing_epochs=int(self._config.get("annealing_epochs", 10)),
        )

    def set_fedemoe_weights(self, weights: Dict[str, torch.Tensor]) -> bool:
        """Install one symbiosis-pool model with strict key checking."""
        self.model.load_state_dict(weights, strict=True)
        return True

    async def fit(self, config: Optional[Dict[str, Any]] = None) -> TrainResult:
        run_config = {**self._config, **(config or {})}
        local_epochs = int(run_config.get("epochs", run_config.get("local_epochs", 5)))
        global_round = int(run_config.get("global_round", 0))

        # set_model, train, and compute_evidence_profile are the validated
        # upstream implementation.  They intentionally execute in this order.
        self._reference_client.set_model(self.model)
        metrics = self._reference_client.train(
            num_epochs=local_epochs,
            global_epoch=global_round,
        )
        evidence_profile = self._reference_client.compute_evidence_profile()
        self._model = self._reference_client.get_model()

        num_samples = self._reference_client.num_samples
        return TrainResult(
            weights=self.get_weights(),
            num_samples=num_samples,
            metrics=TrainMetrics(
                total_epochs=local_epochs,
                final_loss=float(metrics.get("loss", 0.0)),
                total_samples=num_samples,
                metrics={key: float(value) for key, value in metrics.items()},
                epoch_history=[],
            ),
            metadata={
                "algorithm": "fedemoe",
                "client_id": self._reference_client.client_id,
                "global_round": global_round,
                "evidence_profile": evidence_profile,
            },
        )

    async def evaluate(self, config: Optional[Dict[str, Any]] = None) -> EvalResult:
        del config
        if self._reference_client.model is None:
            self._reference_client.set_model(self.model)
        metrics = self._reference_client.evaluate()
        return EvalResult(
            num_samples=int(metrics.pop("num_samples")),
            metrics={key: float(value) for key, value in metrics.items()},
        )

    async def train_step(self, batch: Any, batch_idx: int):
        del batch, batch_idx
        raise RuntimeError("FedEMoELearner.fit uses the validated upstream client loop")

    def get_dataloader(self) -> DataLoader:
        return self._train_loader

    def get_num_samples(self) -> int:
        return len(self._train_dataset)


__all__ = ["FedEMoELearner"]
