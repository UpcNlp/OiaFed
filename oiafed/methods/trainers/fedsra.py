"""Dedicated one-shot trainer for FedSRA."""

from __future__ import annotations

import copy
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ...core.trainer import Trainer
from ...core.types import ClientUpdate, RoundMetrics, RoundResult, TrainResult
from ...registry import trainer
from ..fedsra import generate_simplex_etf
from ..models.fedsra import FedSRAEnsemble


@trainer(
    name="fedsra",
    description="FedSRA single-round orchestration and exact RGA evaluation",
    version="1.0",
    author="FedSRA",
    algorithms=["fedsra"],
)
class FedSRATrainer(Trainer):
    async def run(self) -> Dict[str, Any]:
        configured_rounds = int(self.config.get("max_rounds", self.config.get("num_rounds", 1)))
        if configured_rounds != 1:
            raise ValueError("FedSRA is one-shot: trainer.max_rounds must equal 1")
        self._config["max_rounds"] = 1
        return await super().run()

    async def train_round(self, round_num: int) -> RoundResult:
        if round_num != 1:
            raise RuntimeError("FedSRA supports exactly one communication round")
        if float(self.config.get("client_fraction", 1.0)) != 1.0:
            raise ValueError(
                "FedSRA requires client_fraction=1.0 for the configured one-shot cohort"
            )

        if self.callbacks:
            await self.callbacks.on_round_begin(self, round_num, {"algorithm": "fedsra"})

        selected = self.get_connected_learners()
        if not selected:
            raise RuntimeError("No connected learners are available for FedSRA")

        template = self.model
        if template is None:
            raise ValueError("FedSRATrainer requires a backbone model on the server")
        feature_dim = int(self.config.get("feature_dim", getattr(template, "feature_dim", 256)))
        num_classes = int(self.config.get("num_classes", getattr(template, "num_classes", 10)))
        etf_seed = int(self.config.get("etf_seed", 42))
        etf = generate_simplex_etf(num_classes, feature_dim, etf_seed)

        context = {
            "etf": etf,
            "num_classes": num_classes,
            "feature_dim": feature_dim,
            "etf_seed": etf_seed,
        }
        context_results = await self.broadcast_to_selected(
            selected,
            "set_fedsra_context",
            context,
        )
        context_errors = [result for result in context_results if isinstance(result, Exception)]
        if context_errors:
            raise RuntimeError(f"Failed to install shared ETF on a learner: {context_errors[0]}")

        fit_config = dict(self.config.get("fit_config", {}))
        fit_config.setdefault(
            "epochs",
            int(self.config.get("local_epochs", self.config.get("epochs", 1))),
        )
        results = await self.collect_results(
            selected,
            "fit",
            fit_config,
            timeout=self.config.get("fit_timeout"),
        )

        updates: List[ClientUpdate] = []
        failures: List[Exception] = []
        for index, (learner_proxy, result) in enumerate(zip(selected, results)):
            client_id = getattr(learner_proxy, "_target_id", f"learner_{index}")
            if isinstance(result, Exception):
                failures.append(result)
                continue
            if not isinstance(result, TrainResult) and type(result).__name__ == "TrainResult":
                result = TrainResult.from_dict(result.to_dict())
            if not isinstance(result, TrainResult):
                failures.append(TypeError(f"{client_id} returned {type(result).__name__}"))
                continue
            updates.append(ClientUpdate.from_result(client_id, result))

        if failures:
            raise RuntimeError(
                f"FedSRA requires the complete cohort; {len(failures)} learner(s) failed: "
                f"{failures[0]}"
            )

        bundle = self.aggregator.aggregate(updates, template)
        ensemble = self._build_ensemble(template, bundle, etf)
        self._model = ensemble

        eval_metrics = self._evaluate_ensemble(ensemble)
        total_samples = sum(update.num_samples for update in updates)
        weighted_train_loss = sum(
            float(update.metrics.get("loss", 0.0)) * update.num_samples
            for update in updates
        ) / max(total_samples, 1)
        metrics = RoundMetrics(
            round_num=round_num,
            num_clients=len(updates),
            total_samples=total_samples,
            metrics={"train_loss": weighted_train_loss, **eval_metrics},
        )
        result = RoundResult(
            round_num=round_num,
            updates=updates,
            aggregated_weights=bundle,
            metrics=metrics,
            metadata={
                "algorithm": "fedsra",
                "evaluation_scope": "full_test_set_zscore" if eval_metrics else "not_evaluated",
            },
        )

        if self.tracker and eval_metrics:
            self.tracker.log_metrics(eval_metrics, step=round_num)
        if self.callbacks:
            await self.callbacks.on_round_end(
                self,
                round_num,
                {"round_result": result, "algorithm": "fedsra"},
            )
        return result

    @staticmethod
    def _build_ensemble(
        template: torch.nn.Module,
        bundle: Dict[str, Any],
        etf: torch.Tensor,
    ) -> FedSRAEnsemble:
        backbones = []
        sample_counts = []
        client_ids = []
        class_counts = []
        for client in bundle["clients"]:
            backbone = copy.deepcopy(template).cpu()
            backbone.load_state_dict(client["state_dict"], strict=True)
            backbones.append(backbone)
            sample_counts.append(client["num_samples"])
            client_ids.append(client["client_id"])
            class_counts.append(client["class_counts"])
        return FedSRAEnsemble(
            backbones,
            etf,
            sample_counts,
            client_ids=client_ids,
            class_counts=class_counts,
        )

    def _evaluate_ensemble(self, ensemble: FedSRAEnsemble) -> Dict[str, float]:
        if not self.has_global_test:
            self.logger.warning("FedSRA server has no test dataset; skipping exact RGA evaluation")
            return {}

        loader = DataLoader(
            self.test_dataset,
            batch_size=int(self.config.get("eval_batch_size", 256)),
            shuffle=False,
            num_workers=int(self.config.get("eval_num_workers", 0)),
            pin_memory=str(self.config.get("device", "")).startswith("cuda"),
        )
        device = self.config.get("device")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"

        logits, labels = ensemble.predict_loader(loader, device=device)
        loss = F.cross_entropy(logits, labels).item()
        accuracy = (logits.argmax(dim=1) == labels).float().mean().item()
        return {
            "eval_accuracy": float(accuracy),
            "eval_loss": float(loss),
            "eval_samples": float(labels.numel()),
        }


__all__ = ["FedSRATrainer"]
