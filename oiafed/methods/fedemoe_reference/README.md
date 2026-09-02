# FedEMoE validated core

This directory vendors the numerical core used by the native OiaFed FedEMoE
integration.  It comes from
[`Stephen-Chow1/FedEMoE-CEGA`](https://github.com/Stephen-Chow1/FedEMoE-CEGA),
supplied as `FedEMoE-CEGA-master.zip` with SHA-256
`636118A6A96A309202FF093F64B89E4C88205A431E7A7867186580AA1D04DBCB`.

The following files are byte-for-byte copies of the validated source:

- `backbones.py`
- `edl_router.py`
- `experts.py`
- `emoe.py`
- `edl_loss.py`
- `helpers.py`
- `metrics.py`
- `evidence_symbiosis.py`

`client.py` changes only its three imports from repository-absolute paths to
package-relative paths.  The client training and evidence-profile logic is
unchanged.  OiaFed-specific orchestration is deliberately kept in the normal
`models`, `learners`, `aggregators`, and `trainers` component directories.

The upstream MIT license is included in `LICENSE`.
