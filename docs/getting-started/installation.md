# Installation Guide

This guide will help you install MOE-FedCL and its dependencies.

## Requirements

### System Requirements
- **Python**: 3.12 or higher
- **OS**: Linux, macOS, or Windows (Linux recommended for distributed mode)
- **GPU**: Optional but recommended for large-scale experiments

### Hardware Recommendations
- **For local testing**: 8GB RAM, 2+ CPU cores
- **For experiments**: 16GB+ RAM, GPU with 8GB+ VRAM
- **For distributed**: Network connectivity between machines

---

## Installation Methods

### Method 1: Using UV (Recommended)

[UV](https://github.com/astral-sh/uv) is a fast Python package manager.

```bash
# Install uv if you haven't
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/YOUR_USERNAME/MOE-FedCL.git
cd MOE-FedCL

# Install dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate  # On Linux/macOS
# Or: .venv\Scripts\activate  # On Windows
```

### Method 2: Using Pip

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/MOE-FedCL.git
cd MOE-FedCL

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Linux/macOS
# Or: venv\Scripts\activate  # On Windows

# Install in editable mode
pip install -e .
```

### Method 3: Using Conda

```bash
# Create conda environment
conda create -n fedcl python=3.12
conda activate fedcl

# Clone and install
git clone https://github.com/YOUR_USERNAME/MOE-FedCL.git
cd MOE-FedCL
pip install -e .
```

---

## Verifying Installation

```bash
# Check Python version
python --version  # Should be 3.12+

# Check PyTorch installation
python -c "import torch; print(torch.__version__)"

# Check MOE-FedCL import
python -c "from src.core import FederatedSystem; print('Installation successful!')"
```

---

## GPU Support

### CUDA Setup

If you have NVIDIA GPUs, install PyTorch with CUDA support:

```bash
# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Multiple GPUs

For multi-GPU experiments, ensure your CUDA drivers are properly configured:

```bash
# Check GPUs
nvidia-smi

# Check PyTorch can see all GPUs
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
```

---

## Optional Dependencies

### Development Tools

For contributing or development:

```bash
# Install dev dependencies
uv sync --group dev

# Or with pip
pip install -e ".[dev]"
```

This includes:
- `black` - Code formatting
- `isort` - Import sorting
- `mypy` - Type checking
- `pytest` - Testing

### Experiment Tracking

#### MLflow (Included by default)

MLflow is included in the base installation. No additional steps needed.

```bash
# Start MLflow UI to view experiments
mlflow ui --port 5000

# Open http://localhost:5000 in your browser
```

#### Weights & Biases (Optional)

```bash
pip install wandb

# Login
wandb login
```

---

## Dataset Preparation

MOE-FedCL will automatically download datasets on first use. However, you can pre-download them:

```bash
# Create data directory
mkdir -p data

# Datasets will be downloaded to ./data/ on first use
# MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100 are auto-downloaded
```

For large datasets (CINIC-10, FEMNIST), see [Data Preparation Guide](../user-guide/data-preparation.md).

---

## Troubleshooting

### Common Issues

#### Issue 1: PyTorch Installation Fails

**Solution**: Use pre-built wheels from PyTorch official:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### Issue 2: gRPC Installation Issues

**Solution**: Install build tools:
```bash
# On Ubuntu/Debian
sudo apt-get install build-essential

# On macOS
xcode-select --install

# Then reinstall
pip install --upgrade grpcio grpcio-tools
```

#### Issue 3: CUDA Version Mismatch

**Solution**: Check your CUDA version and install matching PyTorch:
```bash
nvidia-smi  # Check CUDA version
# Then install corresponding PyTorch version from https://pytorch.org
```

#### Issue 4: Permission Denied When Creating Directories

**Solution**: Create directories with appropriate permissions:
```bash
mkdir -p data logs mlruns
chmod 755 data logs mlruns
```

---

## Next Steps

✅ Installation complete! Now:

1. **Quick Start**: Run your first experiment → [Quick Start Tutorial](quickstart.md)
2. **Learn Concepts**: Understand the framework → [Core Concepts](concepts.md)
3. **Explore Examples**: Check out examples in `/examples` directory

---

## Uninstallation

To remove MOE-FedCL:

```bash
# If installed with pip -e
pip uninstall moe-fedcl

# Remove virtual environment
rm -rf venv  # or .venv for uv

# Remove downloaded data (optional)
rm -rf data logs mlruns
```

---

## Getting Help

If you encounter issues during installation:

1. Check [Troubleshooting](#troubleshooting) section above
2. Search [GitHub Issues](https://github.com/YOUR_USERNAME/MOE-FedCL/issues)
3. Ask in [GitHub Discussions](https://github.com/YOUR_USERNAME/MOE-FedCL/discussions)
4. Email: your.email@example.com
