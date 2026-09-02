"""Validated baseline implementations from the FedEMoE artifact.

Only the import paths in :mod:`baselines` were made package-relative.  The
algorithm statements and all numerical operations remain the supplied source.
"""

from .baseline import BaselineModel, create_baseline_model
from .moe import MoE, create_moe_model
from .baselines import (
    BaselineClient,
    BaselineServer,
    FedAvgServer,
    FedProxServer,
    FedSymServer,
    FedProtoClient,
    FedProtoServer,
    FedProcClient,
    FedProcServer,
    FedNTDClient,
    FedNTDServer,
    FedSOLClient,
    FedSOLServer,
    FedLESAMClient,
    FedLESAMServer,
    pFedHBClient,
    pFedHBServer,
    FedMoEDAClient,
    FedMoEDAServer,
    FedEviClient,
    FedEviServer,
)

__all__ = [
    "BaselineModel",
    "create_baseline_model",
    "MoE",
    "create_moe_model",
    "BaselineClient",
    "BaselineServer",
    "FedAvgServer",
    "FedProxServer",
    "FedSymServer",
    "FedProtoClient",
    "FedProtoServer",
    "FedProcClient",
    "FedProcServer",
    "FedNTDClient",
    "FedNTDServer",
    "FedSOLClient",
    "FedSOLServer",
    "FedLESAMClient",
    "FedLESAMServer",
    "pFedHBClient",
    "pFedHBServer",
    "FedMoEDAClient",
    "FedMoEDAServer",
    "FedEviClient",
    "FedEviServer",
]
