# Changelog

All notable changes to MOE-FedCL will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive documentation system
- AI assistant guide for better AI integration
- Paper reproduction guide with 288 pre-configured experiments
- Example scripts and templates
- CONTRIBUTING.md for community contributions

### Changed
- Improved README with clearer structure
- Enhanced configuration system documentation

## [0.1.0] - 2025-12-25

### Added
- Initial public release
- Core federated learning framework with two-layer architecture
- Communication layer (node_comm) with Memory and gRPC transports
- 3 running modes: Serial, Parallel, Distributed
- 12+ Federated Learning algorithms:
  - Basic: FedAvg, FedProx, FedNova
  - Adaptive: FedAdam, FedYogi, FedAdaGrad
  - Personalized: FedPer, FedRep, FedBABU, FedRod
  - Advanced: MOON, SCAFFOLD, FedBN, FedProto, GPFL
- 7 Continual Learning methods:
  - TARGET, FedWEIT, FedKNOW, FedCPrompt, GLFC, LGA
- 8 built-in datasets:
  - MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100, SVHN, EMNIST, FEMNIST, CINIC-10
- Data partitioning strategies:
  - IID, Dirichlet, Label Skew, Quantity Skew
- Registry system for extensibility
- Callback mechanism for lifecycle hooks
- MLflow and Loguru integration for experiment tracking
- YAML-based configuration with inheritance support
- 288 pre-configured paper reproduction experiments
- Examples and tutorials

### Infrastructure
- Complete test suite
- Development tools (Black, isort, mypy)
- Documentation system
- CI/CD setup (placeholder)

---

## Release Notes

### v0.1.0 - Initial Release

This is the first public release of MOE-FedCL, a modular and extensible framework for federated learning and continual learning research.

**Key Features**:
- **Flexible Architecture**: Two-layer design separates communication from federation logic
- **Multiple Scenarios**: Support for FL, CL, and hybrid scenarios
- **Easy to Use**: Configuration-driven with minimal code
- **Easy to Extend**: Registry system for adding custom algorithms
- **Production Ready**: Multiple running modes, comprehensive tracking

**Getting Started**:
1. Install: `pip install -e .`
2. Run: `python examples/simple_fedavg.py`
3. Read: [Documentation](docs/README.md)

**For Researchers**:
- 288 pre-configured experiments for paper reproduction
- Support for 8 datasets and 20+ algorithms
- MLflow integration for result tracking

**For Developers**:
- Clean API with type hints
- Comprehensive test coverage
- Extensible registry system

---

## Upgrade Guide

### From Pre-release to v0.1.0

If you were using an early version of MOE-FedCL, here are the main changes:

1. **Configuration Format**: Now uses standardized YAML format
2. **Registry System**: Components now register with `@register` decorator
3. **Running Modes**: Specify mode explicitly (`--mode serial/parallel/distributed`)
4. **Imports**: Core classes moved to `src.core` module

---

## Breaking Changes

None yet (initial release).

---

## Deprecations

None yet (initial release).

---

## Known Issues

### v0.1.0
- WandB integration is incomplete (documented but not fully implemented)
- Plugin system is planned but not yet available
- Hierarchical federated learning not yet supported
- Some edge cases in distributed mode may require manual configuration

See [GitHub Issues](https://github.com/YOUR_USERNAME/MOE-FedCL/issues) for current bug reports and feature requests.

---

## Future Releases (Planned)

### v0.2.0 (Planned Q1 2026)
- WandB integration
- Web-based dashboard
- More datasets and models
- Performance optimizations
- Enhanced distributed mode

### v0.3.0 (Planned Q2 2026)
- Plugin system
- Community algorithm marketplace
- Hierarchical federated learning
- Advanced scheduling strategies

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## Version History

- **v0.1.0** (2025-12-25) - Initial public release

---

*For detailed commit history, see [GitHub Commits](https://github.com/YOUR_USERNAME/MOE-FedCL/commits).*
