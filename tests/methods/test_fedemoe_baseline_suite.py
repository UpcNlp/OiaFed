"""Parity and registration tests for the FedEMoE comparison suite."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import oiafed.methods  # noqa: F401 - populate the component registry
from oiafed.core.types import ClientUpdate
from oiafed.infra.logging import setup_logging
from oiafed.methods.fedemoe_baselines_reference.baselines import BaselineTrainer
from oiafed.methods.learners.fl.fedemoe_baselines import REFERENCE_METHODS
from oiafed.papers.loader import reload_registry
from oiafed.registry import registry


class _Logger:
    def info(self, *_args, **_kwargs):
        pass

    def log_metrics(self, *_args, **_kwargs):
        pass


def _seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _datasets() -> list[TensorDataset]:
    generator = torch.Generator().manual_seed(7)
    return [
        TensorDataset(
            torch.randn(4, 3, 32, 32, generator=generator),
            torch.tensor([0, 1, 0, 1]),
        ),
        TensorDataset(
            torch.randn(4, 3, 32, 32, generator=generator),
            torch.tensor([1, 2, 1, 2]),
        ),
    ]


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        num_classes=10,
        backbone="cnn",
        input_channels=3,
        input_size=32,
        expert_hidden_dim=256,
        num_experts=8,
        fedmoeda_top_k=2,
        symbiosis_pool_size=3,
        symbiosis_mode="endo",
        lr=0.01,
        momentum=0.9,
        weight_decay=1e-4,
        optimizer="sgd",
        fedprox_mu=0.1,
        fedproto_lambda=0.1,
        proto_weight=0.1,
        proto_temperature=0.5,
        ntd_weight=1.0,
        ntd_temperature=1.0,
        sol_rho=0.5,
        lesam_rho=0.5,
        pfedhb_prior_var=1.0,
        pfedhb_posterior_var=0.1,
        pfedhb_kl_weight=0.01,
        clients_per_round=2,
        local_epochs=1,
    )


def _adapter_args(method: str) -> tuple[dict, dict]:
    learner_args = {
        "learning_rate": 0.01,
        "batch_size": 4,
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "optimizer": "sgd",
        "num_classes": 10,
        "num_workers": 0,
        "device": "cpu",
        "fedprox_mu": 0.1,
        "fedproto_lambda": 0.1,
        "proto_weight": 0.1,
        "proto_temperature": 0.5,
        "ntd_weight": 1.0,
        "ntd_temperature": 1.0,
        "sol_rho": 0.5,
        "lesam_rho": 0.5,
        "pfedhb_prior_var": 1.0,
        "pfedhb_posterior_var": 0.1,
        "pfedhb_kl_weight": 0.01,
        "num_experts": 8,
        "backbone": "cnn",
        "input_channels": 3,
        "input_size": 32,
        "expert_hidden_dim": 256,
        "fedmoeda_top_k": 2,
    }
    aggregator_args = {
        "num_classes": 10,
        "backbone": "cnn",
        "input_channels": 3,
        "input_size": 32,
        "hidden_dim": 256,
        "device": "cpu",
        "symbiosis_pool_size": 3,
        "symbiosis_mode": "endo",
        "pfedhb_prior_var": 1.0,
        "num_experts": 8,
        "expert_hidden_dim": 256,
        "fedmoeda_top_k": 2,
    }
    return learner_args, aggregator_args


async def _run_adapter(method: str, datasets: list[TensorDataset]):
    model_name = "fedemoe_moeda" if method == "fedmoeda" else "fedemoe_baseline"
    model_args = {
        "num_classes": 10,
        "backbone": "cnn",
        "input_channels": 3,
        "input_size": 32,
        "seed": 42,
    }
    if method == "fedmoeda":
        model_args.update(num_experts=8, expert_hidden_dim=256, top_k=2)
    learner_args, aggregator_args = _adapter_args(method)
    learner_cls = registry.get(f"learner.fedemoe_{method}")
    learners = [
        learner_cls(
            model=registry.create(f"model.{model_name}", **model_args),
            datasets={"train": [dataset]},
            config=learner_args,
            node_id=f"learner_{index}",
        )
        for index, dataset in enumerate(datasets)
    ]
    adapter = registry.create(f"aggregator.fedemoe_{method}", **aggregator_args)

    _seed()
    adapter.initialize()
    selected = random.sample(range(2), 2)
    updates = []
    for position, client_id in enumerate(selected):
        payload = adapter.make_client_payload(client_id, position, 1)
        learners[client_id].prepare_reference_round(payload)
        result = await learners[client_id].fit({"epochs": 1})
        updates.append(ClientUpdate.from_result(f"learner_{client_id}", result))
    adapter.aggregate_round(updates, 1)
    adapter.set_test_loader(DataLoader(datasets[0], batch_size=4, shuffle=False))
    metrics = adapter.evaluate()
    return adapter.get_global_model().state_dict(), metrics


@pytest.mark.parametrize("method", REFERENCE_METHODS)
def test_one_round_matches_vendored_reference(method: str, tmp_path: Path):
    setup_logging(node_id="fedemoe_suite_test", console=False, log_dir=str(tmp_path))
    datasets = _datasets()
    config = _config()
    direct_loaders = [DataLoader(dataset, batch_size=4, shuffle=True) for dataset in datasets]
    test_loader = DataLoader(datasets[0], batch_size=4, shuffle=False)

    _seed()
    direct = BaselineTrainer(
        method=method,
        config=config,
        client_loaders=direct_loaders,
        test_loader=test_loader,
        device=torch.device("cpu"),
        logger=_Logger(),
    )
    direct.train_round(1)
    direct_metrics = direct.evaluate()
    direct_state = direct.server.get_global_model().state_dict()

    adapter_state, adapter_metrics = asyncio.run(_run_adapter(method, datasets))
    assert direct_state.keys() == adapter_state.keys()
    for key in direct_state:
        assert torch.equal(direct_state[key].cpu(), adapter_state[key].cpu()), key
    assert adapter_metrics == pytest.approx(direct_metrics, abs=0.0, rel=0.0)


def test_every_method_is_registered_and_catalogued():
    papers = reload_registry()
    for method in REFERENCE_METHODS:
        name = f"fedemoe_{method}"
        assert registry.exists(f"learner.{name}")
        assert registry.exists(f"aggregator.{name}")
        assert registry.exists(f"trainer.{name}")
        paper = papers.get(name)
        assert paper is not None
        assert paper.category == "HFL"


def test_vendored_manifest_matches_package_files():
    root = (
        Path(__file__).resolve().parents[2]
        / "oiafed"
        / "methods"
        / "fedemoe_baselines_reference"
    )
    for line in (root / "VENDORED_MANIFEST.sha256").read_text().splitlines():
        expected, relative = line.split(maxsplit=1)
        actual = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        assert actual == expected
