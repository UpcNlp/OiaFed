"""OiaFed learner adapters for the FedEMoE comparison-suite baselines."""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

from ....core.learner import Learner
from ....core.types import StepMetrics, TrainMetrics, TrainResult
from ....registry import learner
from ...fedemoe_baselines_reference.baselines import (
    BaselineClient,
    FedEviClient,
    FedLESAMClient,
    FedMoEDAClient,
    FedNTDClient,
    FedProcClient,
    FedProtoClient,
    FedSOLClient,
    pFedHBClient,
)


REFERENCE_METHODS = (
    "fedavg",
    "fedprox",
    "fedsym",
    "fedproto",
    "fedproc",
    "fedntd",
    "fedsol",
    "fedlesam",
    "pfedhb",
    "fedmoeda",
    "fedevi",
)


def _client_number(node_id: str) -> int:
    try:
        return int(node_id.rsplit("_", 1)[-1])
    except (TypeError, ValueError):
        return 0


def _cpu_clone(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_clone(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_clone(item) for item in value)
    if isinstance(value, list):
        return [_cpu_clone(item) for item in value]
    return value


class FedEMoEBaselineLearner(Learner):
    """Expose one unchanged artifact client through OiaFed's learner API."""

    REFERENCE_METHOD = ""

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
        if self.REFERENCE_METHOD not in REFERENCE_METHODS:
            raise ValueError(f"Unknown FedEMoE baseline: {self.REFERENCE_METHOD}")

        train_datasets = (datasets or {}).get("train", [])
        if not train_datasets:
            raise ValueError("FedEMoE baseline learner requires one training dataset")
        self._train_dataset = train_datasets[0]

        requested_device = str(self.config.get("device", "cuda"))
        self._device_obj = torch.device(
            "cuda" if requested_device == "cuda" and torch.cuda.is_available() else "cpu"
        )
        self._train_loader = DataLoader(
            self._train_dataset,
            batch_size=int(self.config.get("batch_size", 64)),
            shuffle=True,
            num_workers=int(self.config.get("num_workers", 4)),
            pin_memory=True,
        )
        self._reference_client = self._create_reference_client()
        self._prepared_round: Optional[int] = None

    def _common_client_args(self) -> Dict[str, Any]:
        return {
            "client_id": _client_number(self.node_id),
            "dataloader": self._train_loader,
            "device": self._device_obj,
            "num_classes": int(self.config.get("num_classes", 10)),
            "lr": float(self.config.get("learning_rate", self.config.get("lr", 0.01))),
            "momentum": float(self.config.get("momentum", 0.9)),
            "weight_decay": float(self.config.get("weight_decay", 1e-4)),
            "optimizer_name": str(self.config.get("optimizer", "sgd")),
        }

    def _create_reference_client(self) -> Any:
        method = self.REFERENCE_METHOD
        args = self._common_client_args()
        if method in {"fedavg", "fedprox", "fedsym"}:
            args["mu"] = float(self.config.get("fedprox_mu", 0.1)) if method == "fedprox" else 0.0
            return BaselineClient(**args)
        if method == "fedproto":
            return FedProtoClient(
                **args, proto_lambda=float(self.config.get("fedproto_lambda", 0.1))
            )
        if method == "fedproc":
            return FedProcClient(
                **args,
                proto_weight=float(self.config.get("proto_weight", 0.1)),
                temperature=float(self.config.get("proto_temperature", 0.5)),
            )
        if method == "fedntd":
            return FedNTDClient(
                **args,
                ntd_weight=float(self.config.get("ntd_weight", 1.0)),
                temperature=float(self.config.get("ntd_temperature", 1.0)),
            )
        if method == "fedsol":
            return FedSOLClient(**args, rho=float(self.config.get("sol_rho", 0.5)))
        if method == "fedlesam":
            return FedLESAMClient(**args, rho=float(self.config.get("lesam_rho", 0.5)))
        if method == "pfedhb":
            return pFedHBClient(
                **args,
                prior_var=float(self.config.get("pfedhb_prior_var", 1.0)),
                posterior_var=float(self.config.get("pfedhb_posterior_var", 0.1)),
                kl_weight=float(self.config.get("pfedhb_kl_weight", 0.01)),
            )
        if method == "fedmoeda":
            return FedMoEDAClient(
                **args,
                moe_config={
                    "num_classes": int(self.config.get("num_classes", 10)),
                    "num_experts": int(self.config.get("num_experts", 8)),
                    "backbone": str(self.config.get("backbone", "cnn")),
                    "input_channels": int(self.config.get("input_channels", 3)),
                    "input_size": int(self.config.get("input_size", 32)),
                    "expert_hidden_dim": int(self.config.get("expert_hidden_dim", 256)),
                    "top_k": int(self.config.get("fedmoeda_top_k", 2)),
                },
            )
        return FedEviClient(**args)

    def _model_from_weights(self, weights: Dict[str, torch.Tensor]) -> Any:
        model = copy.deepcopy(self.model).to(self._device_obj)
        model.load_state_dict(weights, strict=True)
        return model

    def prepare_reference_round(self, payload: Dict[str, Any]) -> bool:
        """Install exactly the state distributed by the artifact server."""
        method = self.REFERENCE_METHOD
        global_model = self._model_from_weights(payload["model_weights"])
        client = self._reference_client

        if method == "fedmoeda":
            client.set_moe_model(global_model)
        elif method == "fedproto":
            if client.model is None:
                client.set_model(global_model)
            client.set_global_prototypes(
                payload.get("global_prototypes"), payload.get("prototype_mask")
            )
        elif method == "fedproc":
            client.set_model(global_model)
            client.set_global_prototypes(payload.get("global_prototypes"))
        elif method == "fedntd":
            client.set_model(global_model)
            client.set_teacher_model(global_model)
        elif method == "fedsol":
            client.set_model(global_model, keep_global=True)
        elif method == "fedlesam":
            client.set_model(global_model)
            previous = payload.get("prev_global_weights")
            previous_model = (
                self._model_from_weights(previous) if previous is not None else None
            )
            client.set_global_direction(global_model, previous_model)
        elif method == "pfedhb":
            client.set_model(global_model)
            client.set_global_prior(global_model)
        else:
            client.set_model(global_model, keep_global=(method == "fedprox"))

        self._prepared_round = int(payload["round_num"])
        return True

    async def setup(self, config: Dict[str, Any]) -> None:
        del config

    async def fit(self, config: Optional[Dict[str, Any]] = None) -> TrainResult:
        if self._prepared_round is None:
            raise RuntimeError("prepare_reference_round must be called before fit")
        run_config = {**self.config, **(config or {})}
        epochs = int(run_config.get("epochs", run_config.get("local_epochs", 5)))
        metrics = self._reference_client.train(epochs)
        method = self.REFERENCE_METHOD
        metadata: Dict[str, Any] = {
            "algorithm": method,
            "round_num": self._prepared_round,
            "client_id": _client_number(self.node_id),
        }

        if method == "fedmoeda":
            trained_model = self._reference_client.get_moe_model()
            metadata["router_weights"] = _cpu_clone(
                self._reference_client.get_router_weights()
            )
        else:
            trained_model = self._reference_client.get_model()

        if method == "fedproto":
            proto, mask, counts = self._reference_client.compute_local_prototypes()
            metadata.update(
                local_prototypes=_cpu_clone(proto),
                prototype_mask=_cpu_clone(mask),
                class_counts=_cpu_clone(counts),
            )
        elif method == "fedproc":
            proto, mask = self._reference_client.compute_local_prototypes()
            metadata.update(
                local_prototypes=_cpu_clone(proto), prototype_mask=_cpu_clone(mask)
            )
        elif method == "pfedhb":
            metadata["posterior"] = _cpu_clone(
                self._reference_client.get_posterior_params()
            )
        elif method == "fedevi":
            metadata["uncertainty"] = self._reference_client.compute_uncertainty()

        self._model = trained_model
        result = TrainResult(
            weights=_cpu_clone(trained_model.state_dict()),
            num_samples=len(self._train_dataset),
            metrics=TrainMetrics(
                total_epochs=epochs,
                final_loss=float(metrics.get("loss", 0.0)),
                total_samples=len(self._train_dataset) * epochs,
                metrics={key: float(value) for key, value in metrics.items()},
            ),
            metadata=metadata,
        )
        self._prepared_round = None
        return result

    async def train_step(self, batch: Any, batch_idx: int) -> StepMetrics:
        del batch, batch_idx
        raise RuntimeError("The validated reference client owns its local loop")


def _register_reference_learners() -> None:
    for method in REFERENCE_METHODS:
        class_name = "FedEMoE" + "".join(part.title() for part in method.split("_")) + "Learner"
        component_name = f"fedemoe_{method}"
        cls = type(
            class_name,
            (FedEMoEBaselineLearner,),
            {"REFERENCE_METHOD": method, "__module__": __name__},
        )
        cls = learner(
            name=component_name,
            description=f"FedEMoE artifact {method} learner",
            version="1.0",
            author="FedEMoE Authors",
            upstream="Stephen-Chow1/FedEMoE-CEGA",
        )(cls)
        globals()[class_name] = cls


_register_reference_learners()

__all__ = ["FedEMoEBaselineLearner", "REFERENCE_METHODS"] + [
    "FedEMoE" + "".join(part.title() for part in method.split("_")) + "Learner"
    for method in REFERENCE_METHODS
]
