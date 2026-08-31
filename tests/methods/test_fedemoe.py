"""Parity and native-component tests for FedEMoE."""

from __future__ import annotations

import copy
import hashlib
import inspect
import random
from pathlib import Path

import numpy as np
import torch

from oiafed.core.types import ClientUpdate, RoundMetrics, RoundResult
from oiafed.data.partitioner import FedEMoEDirichletPartitioner
from oiafed.methods.aggregators.fedemoe import FedEMoEAggregator
from oiafed.methods.learners.fl.fedemoe import FedEMoELearner
from oiafed.methods.models.fedemoe import FedEMoEModel
from oiafed.methods.trainers.fedemoe import FedEMoETrainer
from oiafed.papers import get_registry


CORE_HASHES = {
    "backbones.py": "8b72c7d940f3e6a9efafed70777d84d84dd15da5e8ad84d8e80a98264d8e65b8",
    "edl_router.py": "9224143f8849bbc858caa81d35d197376919356a9ed91c124248e236147d5af4",
    "experts.py": "6b7445cd0e5d676064b094befe46b61e6e5934ce4c6c6074afe14dcf7dcd3808",
    "emoe.py": "3ec53ae36dbe767433c420f7bb86df7682936b144a6f720beda1f6e8ba60ca77",
    "edl_loss.py": "b703f85ae572f77106da642c7a73a70373d5b8d23ed9fdb77dd949888597d819",
    "helpers.py": "ac7270e4709b1d15e4a6cff0806f89e51b8a0a2007203c9666d6a7453e22397f",
    "metrics.py": "491b314b1953e627bb45291228701ec2262ec540b5ee948681c12777ce82044c",
    "evidence_symbiosis.py": "0aaa43ad9d5ab11bb7703075e526553d7740255cd5bbfa1ca19b3d380fdcdbb9",
}


def test_native_learner_satisfies_oiafed_lifecycle_contract():
    assert not inspect.isabstract(FedEMoELearner)


def test_trainer_releases_consumed_round_weights_without_losing_metrics():
    update = ClientUpdate("learner_0", {"weight": torch.ones(2)}, 2)
    metrics = RoundMetrics(1, 1, 2, {"eval_accuracy": 0.5})
    result = RoundResult(1, [update], {"round": 1}, metrics)

    FedEMoETrainer._release_consumed_updates(result)

    assert result.updates == []
    assert result.metrics.metrics == {"eval_accuracy": 0.5}
    assert result.aggregated_weights == {"round": 1}


def test_validated_core_hashes_are_unchanged():
    core_dir = (
        Path(__file__).parents[2] / "oiafed" / "methods" / "fedemoe_reference"
    )
    for filename, expected in CORE_HASHES.items():
        actual = hashlib.sha256((core_dir / filename).read_bytes()).hexdigest()
        assert actual == expected, filename


def test_registered_model_is_exact_seeded_template():
    kwargs = dict(
        num_classes=3,
        num_experts=2,
        backbone="cnn",
        input_channels=3,
        input_size=32,
        expert_hidden_dim=16,
        router_hidden_dim=16,
        k_min=1,
        k_max=2,
        dropout=0.1,
    )
    first = FedEMoEModel(seed=42, **kwargs)
    second = FedEMoEModel(seed=42, **kwargs)
    for key, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[key], rtol=0, atol=0)

    inputs = torch.linspace(-1, 1, 2 * 3 * 32 * 32).reshape(2, 3, 32, 32)
    first.eval()
    second.eval()
    first_output = first(inputs)
    second_output = second(inputs)
    torch.testing.assert_close(first_output.logits, second_output.logits, rtol=0, atol=0)
    torch.testing.assert_close(
        first_output.router_output.evidence,
        second_output.router_output.evidence,
        rtol=0,
        atol=0,
    )


def test_fedemoe_dirichlet_matches_reference_cumulative_rounding():
    labels = [class_id for class_id in range(4) for _ in range(13)]
    partitioner = FedEMoEDirichletPartitioner(alpha=0.5, seed=42)
    actual = partitioner.partition(len(labels), 5, labels)

    np.random.seed(42)
    label_array = np.asarray(labels)
    expected = {client_id: [] for client_id in range(5)}
    for class_id in range(4):
        indices = np.where(label_array == class_id)[0]
        np.random.shuffle(indices)
        proportions = np.random.dirichlet([0.5] * 5)
        endpoints = (np.cumsum(proportions / proportions.sum()) * len(indices)).astype(int)
        start = 0
        for client_id, end in enumerate(endpoints):
            expected[client_id].extend(indices[start:end].tolist())
            start = end

    assert actual == expected
    assert sum(map(len, actual.values())) == len(labels)


def test_aggregator_adapter_matches_direct_pool_update():
    template = FedEMoEModel(
        num_classes=3,
        num_experts=2,
        expert_hidden_dim=8,
        router_hidden_dim=8,
        k_min=1,
        k_max=2,
        seed=42,
    )
    aggregator = FedEMoEAggregator(
        pool_size=3,
        num_experts=2,
        num_classes=3,
        num_parents=2,
        symbiosis_mode="endo",
    )
    aggregator.initialize(template)
    direct = copy.deepcopy(aggregator.pool)
    assert direct is not None

    updates = []
    for client_id, pool_index, state in aggregator.distribute([0, 1, 2]):
        for value in state.values():
            if value.is_floating_point():
                value.add_(client_id * 0.001)
        signature = {
            expert_id: torch.tensor(
                [client_id + expert_id + 1.0, 2.0, 3.0], dtype=torch.float32
            )
            for expert_id in range(2)
        }
        updates.append(
            ClientUpdate(
                client_id=f"learner_{client_id}",
                weights=state,
                num_samples=4,
                metadata={
                    "pool_index": pool_index,
                    "global_round": 1,
                    "evidence_profile": signature,
                },
            )
        )

    direct.save_pool_snapshot()
    for update in updates:
        pool_index = update.metadata["pool_index"]
        model = copy.deepcopy(direct.get_model(pool_index))
        model.load_state_dict(update.weights, strict=True)
        model.eval()
        direct.set_model(
            pool_index,
            model,
            signature=update.metadata["evidence_profile"],
        )
    random.seed(17)
    direct.update_pool(direct.perform_symbiosis(1))

    random.seed(17)
    result = aggregator.aggregate(updates, template)
    assert result["round"] == 1
    assert aggregator.pool is not None
    for direct_model, adapted_model in zip(direct.models, aggregator.pool.models):
        for key, value in direct_model.state_dict().items():
            torch.testing.assert_close(
                value, adapted_model.state_dict()[key], rtol=0, atol=0
            )


def test_paper_definition_is_hfl_and_uses_native_components():
    paper = get_registry().get("fedemoe")
    assert paper is not None
    assert paper.category == "HFL"
    assert paper.components == {
        "learner": "fedemoe",
        "aggregator": "fedemoe",
        "trainer": "fedemoe",
        "model": "fedemoe",
        "dataset": "cifar10",
    }
    assert paper.defaults["trainer"]["num_rounds"] == 500
    assert paper.defaults["dataset"]["partition"]["strategy"] == "fedemoe_dirichlet"


def test_programmatic_configs_preserve_partition_and_dataset_roles(tmp_path):
    registry = get_registry()
    configs = registry.generate_node_configs(
        "fedemoe",
        num_clients=2,
        override={"dataset": {"data_dir": str(tmp_path)}},
    )

    trainer, learner_0, learner_1 = configs
    assert [dataset["split"] for dataset in trainer["datasets"]] == ["test"]
    for learner in (learner_0, learner_1):
        assert [dataset["split"] for dataset in learner["datasets"]] == ["train"]
        train_dataset = learner["datasets"][0]
        assert train_dataset["partition"]["strategy"] == "fedemoe_dirichlet"
        assert train_dataset["partition"]["alpha"] == 0.5
        assert train_dataset["args"]["shared_cache"] is True
        assert "server_test" not in train_dataset["args"]
        assert "learner_test" not in train_dataset["args"]
