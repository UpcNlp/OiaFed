"""Validated FedEMoE algorithm core vendored for the OiaFed integration.

The numerical implementation comes from Stephen-Chow1/FedEMoE-CEGA.  The
OiaFed-facing model, learner, aggregator and trainer live outside this package.
Only ``client.py`` has import-only changes so it can resolve this package.
"""

from .client import FedEMoEClient
from .edl_loss import EDLLoss
from .emoe import EMoE, EMoEOutput, create_emoe_model
from .evidence_symbiosis import EvidenceGuidedSymbiosisPool

__all__ = [
    "EDLLoss",
    "EMoE",
    "EMoEOutput",
    "EvidenceGuidedSymbiosisPool",
    "FedEMoEClient",
    "create_emoe_model",
]
