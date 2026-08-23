import asyncio
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset

from oiafed.methods.aggregators.fedsra import FedSRAAggregator
from oiafed.methods.fedsra import (
    etf_alignment_loss,
    etf_contrastive_loss,
    generate_simplex_etf,
    rga_aggregate,
)
from oiafed.methods.learners.fl.fedsra import FedSRALearner
from oiafed.methods.models.fedsra import FedSRAEnsemble, FedSRAResNet18Backbone
from oiafed.methods.trainers.fedsra import FedSRATrainer
from oiafed.core.types import ClientUpdate
from oiafed.infra.logging import setup_logging


class TinyBackbone(nn.Module):
    def __init__(self, feature_dim=4, num_classes=3):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.fc = nn.Linear(3 * 4 * 4, feature_dim)

    def forward(self, inputs):
        return F.normalize(self.fc(inputs.flatten(1)), dim=1)


class LocalLearnerProxy:
    def __init__(self, target_id, learner):
        self._target_id = target_id
        self.learner = learner

    async def set_weights(self, weights):
        return self.learner.set_weights(weights)

    async def set_fedsra_context(self, context):
        return self.learner.set_fedsra_context(context)

    async def fit(self, config):
        return await self.learner.fit(config)


class LocalProxyCollection:
    def __init__(self, proxies):
        self.proxies = list(proxies)

    def get_available_proxies(self):
        return list(self.proxies)

    def get_all_proxies(self):
        return list(self.proxies)

    def get_healthy_proxies(self):
        return list(self.proxies)

    def get_stats(self):
        ids = [proxy._target_id for proxy in self.proxies]
        return {
            "total": len(ids),
            "available": len(ids),
            "unavailable": 0,
            "available_ids": ids,
        }

    async def broadcast(self, method, *args, **kwargs):
        outputs = {}
        for proxy in self.proxies:
            outputs[proxy._target_id] = await getattr(proxy, method)(*args, **kwargs)
        return outputs


def _tiny_dataset(seed, labels):
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.randn(len(labels), 3, 4, 4, generator=generator)
    return TensorDataset(inputs, torch.tensor(labels, dtype=torch.long))


def test_simplex_etf_geometry_is_deterministic():
    first = generate_simplex_etf(10, 32, seed=42)
    second = generate_simplex_etf(10, 32, seed=42)
    assert torch.equal(first, second)
    assert torch.allclose(first.norm(dim=1), torch.ones(10), atol=1e-6)
    gram = first @ first.T
    off_diagonal = gram[~torch.eye(10, dtype=torch.bool)]
    assert torch.allclose(off_diagonal, torch.full_like(off_diagonal, -1 / 9), atol=1e-6)


def test_erl_losses_match_reference_formulas():
    torch.manual_seed(7)
    features = torch.randn(8, 12)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    etf = generate_simplex_etf(4, 12)

    normalized = F.normalize(features, dim=1)
    expected_alignment = (1 - (normalized * etf[labels]).sum(1)).mean()
    assert torch.allclose(etf_alignment_loss(features, labels, etf), expected_alignment)

    actual_contrastive = etf_contrastive_loss(features, labels, etf)
    assert torch.isfinite(actual_contrastive)
    assert actual_contrastive.item() > 0


def test_rga_matches_reference_loop():
    torch.manual_seed(3)
    raw = torch.randn(3, 17, 6)
    counts = [4, 9, 25]

    expected = torch.zeros(17, 6)
    total_weight = 0.0
    for client_index, count in enumerate(counts):
        features = raw[client_index]
        standardized = (features - features.mean(0, keepdim=True)) / (
            features.std(0, keepdim=True) + 1e-8
        )
        weight = math.sqrt(count)
        expected += standardized * weight
        total_weight += weight
    expected = F.normalize(expected / total_weight, dim=1)

    assert torch.allclose(rga_aggregate(raw, counts), expected, atol=1e-6)


def test_resnet_exposes_raw_rga_features_and_normalized_erl_features():
    model = FedSRAResNet18Backbone(feature_dim=16, num_classes=10).eval()
    inputs = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        raw = model.forward_raw(inputs)
        normalized = model(inputs)
    assert raw.shape == (2, 16)
    assert torch.allclose(normalized, F.normalize(raw, dim=1), atol=1e-7)


def test_aggregator_preserves_distinct_client_states():
    updates = []
    for index, samples in enumerate((5, 7)):
        updates.append(
            ClientUpdate(
                client_id=f"client_{index}",
                weights={"weight": torch.tensor([float(index)])},
                num_samples=samples,
                metadata={
                    "class_counts": {index: samples},
                    "num_classes": 3,
                    "feature_dim": 4,
                    "etf_seed": 42,
                },
            )
        )

    bundle = FedSRAAggregator().aggregate(updates)
    assert bundle["format"] == "oiafed.fedsra.bundle"
    assert len(bundle["clients"]) == 2
    assert bundle["clients"][0]["state_dict"]["weight"].item() == 0
    assert bundle["clients"][1]["state_dict"]["weight"].item() == 1


def test_one_shot_trainer_builds_rga_ensemble():
    for node_id in ("learner_0", "learner_1", "trainer"):
        setup_logging(
            node_id=node_id,
            console=False,
            log_dir="./.pytest_cache/fedsra-logs",
        )

    client_datasets = [
        _tiny_dataset(0, [0, 1, 0, 1, 0, 1]),
        _tiny_dataset(1, [1, 2, 1, 2, 1, 2]),
    ]
    proxies = []
    for index, dataset in enumerate(client_datasets):
        learner = FedSRALearner(
            model=TinyBackbone(),
            datasets={"train": [dataset]},
            config={
                "batch_size": 3,
                "learning_rate": 1e-3,
                "num_classes": 3,
                "feature_dim": 4,
                "etf_seed": 42,
                "device": "cpu",
            },
            node_id=f"learner_{index}",
        )
        proxies.append(LocalLearnerProxy(f"learner_{index}", learner))

    trainer = FedSRATrainer(
        learners=LocalProxyCollection(proxies),
        aggregator=FedSRAAggregator(),
        datasets={"test": [_tiny_dataset(9, [0, 1, 2, 0, 1, 2])]},
        model=TinyBackbone(),
        config={
            "max_rounds": 1,
            "local_epochs": 1,
            "client_fraction": 1.0,
            "eval_batch_size": 3,
            "device": "cpu",
            "shutdown_learners": False,
        },
        node_id="trainer",
    )

    result = asyncio.run(trainer.run())
    assert result["completed_rounds"] == 1
    assert result["final_round_metrics"]["eval_samples"] == 6.0
    assert isinstance(trainer.model, FedSRAEnsemble)
    assert len(trainer.model.backbones) == 2
