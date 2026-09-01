# FedEMoE comparison-suite reference code

This directory vendors the baseline implementation used by the supplied
`Stephen-Chow1/FedEMoE-CEGA` artifact (source archive SHA-256
`636118a6a96a309202ff093f64b89e4c88205a431e7a7867186580aa1d04dbcb`).

The numerical core is retained from the artifact.  In `baselines.py`, four
absolute imports were changed to package-relative imports so that the code can
be imported from OiaFed without creating conflicting top-level `models` and
`utils` packages.  `UPSTREAM_MANIFEST.sha256` records the unmodified upstream
files, and `VENDORED_MANIFEST.sha256` records the import-adapted package.

Supported methods: FedAvg, FedProx, FedSym, FedProto, FedProc, FedNTD,
FedSOL, FedLESAM, pFedHB, FedMoE-DA, and FedEvi.
