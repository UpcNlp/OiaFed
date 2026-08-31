"""Server orchestration for validated one-shot federation baselines."""

from __future__ import annotations

import copy
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ...core.trainer import Trainer
from ...core.types import ClientUpdate, RoundMetrics, RoundResult, TrainResult
from ...registry import trainer
from ..models.oneshot import FedCGSServerModel, OneShotEnsemble


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
            await self.callbacks.on_round_begin(self, round_num, {"algorithm": self.algorithm})

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
        aggregated = {
            "format": "oiafed.fedcgs.statistics",
            "weight": weights,
            "bias": bias,
            "covariance": covariance,
        }
        return await self._finalize_round(
            self._round_result(round_num, updates, aggregated, metrics)
        )


__all__ = ["OFedAvgTrainer", "EnsembleTrainer", "FedCGSTrainer"]
