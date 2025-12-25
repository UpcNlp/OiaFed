# Contributing to MOE-FedCL

Thank you for your interest in contributing to MOE-FedCL! We welcome contributions from the community.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Adding New Components](#adding-new-components)

---

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors, regardless of background or identity.

### Our Standards

**Positive behavior includes**:
- Using welcoming and inclusive language
- Respecting differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community

**Unacceptable behavior includes**:
- Harassment, trolling, or discriminatory language
- Publishing others' private information
- Other conduct that could reasonably be considered inappropriate

### Enforcement

Instances of unacceptable behavior may be reported to the project team. All complaints will be reviewed and investigated promptly and fairly.

---

## How to Contribute

There are many ways to contribute:

### 🐛 Reporting Bugs

**Before submitting a bug report**:
1. Check the [existing issues](https://github.com/YOUR_USERNAME/MOE-FedCL/issues)
2. Ensure you're using the latest version
3. Try to reproduce with a minimal example

**When reporting**:
- Use a clear, descriptive title
- Describe the exact steps to reproduce
- Provide your configuration files
- Include error messages and logs
- Specify your environment (OS, Python version, GPU, etc.)

### 💡 Suggesting Features

**Before suggesting**:
1. Check if it's already been suggested
2. Ensure it aligns with the project's scope

**When suggesting**:
- Explain the use case
- Describe the expected behavior
- Provide examples of how it would be used
- Consider implementation complexity

### 📝 Improving Documentation

Documentation improvements are always welcome:
- Fix typos or clarify confusing sections
- Add examples or tutorials
- Improve API documentation
- Translate documentation

### 🔧 Contributing Code

See the sections below for code contribution guidelines.

---

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/MOE-FedCL.git
cd MOE-FedCL

# Add upstream remote
git remote add upstream https://github.com/ORIGINAL_OWNER/MOE-FedCL.git
```

### 2. Set Up Environment

```bash
# Using uv (recommended)
uv sync --group dev

# Or using pip
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 4. Install Pre-commit Hooks (Optional)

```bash
pip install pre-commit
pre-commit install
```

---

## Coding Standards

### Code Style

We follow PEP 8 with some modifications:

- **Line length**: 88 characters (Black default)
- **Indentation**: 4 spaces
- **Quotes**: Double quotes for strings
- **Imports**: Organized with `isort`

### Formatting Tools

```bash
# Format code with Black
black src/ tests/

# Sort imports with isort
isort src/ tests/

# Type checking with mypy
mypy src/

# All at once
black src/ tests/ && isort src/ tests/ && mypy src/
```

### Naming Conventions

- **Files**: `snake_case.py`
- **Classes**: `PascalCase`
- **Functions/Methods**: `snake_case()`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private**: `_leading_underscore`

**Example**:
```python
class FedAvgAggregator:
    DEFAULT_TIMEOUT = 30.0

    def __init__(self):
        self._weights = None

    def aggregate(self, updates):
        return self._weighted_average(updates)

    def _weighted_average(self, updates):
        pass
```

### Docstring Style

Use Google-style docstrings:

```python
def aggregate(self, updates: List[ClientUpdate], global_model: Model = None) -> Dict:
    """Aggregate client updates into a global model.

    Args:
        updates: List of client updates containing weights and metadata.
        global_model: Optional global model for reference (used by some aggregators).

    Returns:
        Dictionary with aggregated weights and metadata.

    Raises:
        ValueError: If updates list is empty.

    Example:
        >>> aggregator = FedAvgAggregator()
        >>> global_weights = aggregator.aggregate(client_updates)
    """
    pass
```

### Type Hints

Always use type hints:

```python
from typing import List, Dict, Optional, Union

def train(
    self,
    model: Model,
    data: DataLoader,
    epochs: int = 5,
    config: Optional[Dict[str, Any]] = None
) -> TrainResult:
    pass
```

---

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_aggregators.py

# Run specific test
pytest tests/test_aggregators.py::test_fedavg_basic
```

### Writing Tests

**Test file structure**:
```
tests/
├── unit/
│   ├── test_aggregators.py
│   ├── test_learners.py
│   └── test_models.py
├── integration/
│   ├── test_serial_mode.py
│   └── test_parallel_mode.py
└── e2e/
    └── test_full_training.py
```

**Test example**:
```python
import pytest
from src.methods.aggregators import FedAvgAggregator
from src.core.types import ClientUpdate

class TestFedAvgAggregator:
    def test_basic_aggregation(self):
        """Test basic weighted averaging."""
        aggregator = FedAvgAggregator(weighted=True)

        # Create mock updates
        updates = [
            ClientUpdate(weights=[1.0, 2.0], num_samples=100),
            ClientUpdate(weights=[3.0, 4.0], num_samples=200),
        ]

        result = aggregator.aggregate(updates)

        # Expected: (1*100 + 3*200) / 300 = 2.33, (2*100 + 4*200) / 300 = 3.33
        assert abs(result["weights"][0] - 2.33) < 0.01
        assert abs(result["weights"][1] - 3.33) < 0.01

    def test_empty_updates_raises_error(self):
        """Test that empty updates list raises ValueError."""
        aggregator = FedAvgAggregator()

        with pytest.raises(ValueError):
            aggregator.aggregate([])
```

### Test Coverage

Aim for:
- **Unit tests**: 80%+ coverage
- **Integration tests**: Key workflows covered
- **E2E tests**: At least one per major feature

---

## Pull Request Process

### 1. Before Submitting

✅ **Checklist**:
- [ ] Code follows style guidelines (run Black, isort, mypy)
- [ ] Added/updated tests (coverage maintained or improved)
- [ ] All tests pass locally
- [ ] Documentation updated (if applicable)
- [ ] Commit messages are clear and descriptive
- [ ] Branch is up to date with main

### 2. Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (no logic change)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples**:
```
feat(aggregator): add FedAdam optimizer

Implements adaptive server-side optimization using Adam.
Closes #42

---

fix(config): handle missing optional fields gracefully

Previously crashed when optional tracker config was absent.
Now defaults to no tracking.

---

docs(readme): add installation troubleshooting section
```

### 3. Pull Request Template

When opening a PR, include:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
Describe how you tested:
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Tested locally

## Related Issues
Closes #issue_number

## Screenshots (if applicable)

## Checklist
- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] Documentation updated
- [ ] No breaking changes (or documented if unavoidable)
```

### 4. Review Process

1. **Automated checks**: CI/CD runs tests and linting
2. **Code review**: Maintainers review your code
3. **Feedback**: Address review comments
4. **Approval**: At least one maintainer approves
5. **Merge**: Maintainer merges your PR

**Expected timeline**: 1-7 days for initial review

---

## Adding New Components

### Adding a New Aggregator

**1. Create the file**: `src/methods/aggregators/my_aggregator.py`

```python
from src.core import Aggregator, register
from typing import List, Dict, Any

@register("aggregator", "my_aggregator")
class MyAggregator(Aggregator):
    """
    My custom aggregation algorithm.

    Args:
        weighted: Whether to weight by dataset size.
        my_param: Custom parameter for my algorithm.
    """

    def __init__(self, weighted: bool = True, my_param: float = 0.1):
        self.weighted = weighted
        self.my_param = my_param

    def aggregate(self, updates: List[ClientUpdate], global_model=None) -> Dict[str, Any]:
        """Aggregate client updates.

        Args:
            updates: Client updates with weights and metadata.
            global_model: Optional reference model.

        Returns:
            Dictionary with 'weights' key and aggregated parameters.
        """
        # Your aggregation logic here
        aggregated_weights = self._my_aggregation_logic(updates)

        return {
            "weights": aggregated_weights,
            "num_clients": len(updates),
        }

    def _my_aggregation_logic(self, updates):
        # Implementation details
        pass
```

**2. Add to `__init__.py`**: `src/methods/aggregators/__init__.py`

```python
from .my_aggregator import MyAggregator

__all__ = [..., "MyAggregator"]
```

**3. Add tests**: `tests/unit/test_my_aggregator.py`

**4. Add documentation**: Update `docs/user-guide/builtin-algorithms.md`

**5. Add config preset**: `configs/presets/algorithms/my_aggregator.yaml`

```yaml
aggregator:
  type: my_aggregator
  args:
    weighted: true
    my_param: 0.1
```

### Adding a New Learner

Similar process in `src/methods/learners/`:

```python
from src.core import Learner, register

@register("learner", "my_learner")
class MyLearner(Learner):
    async def fit(self, config):
        # Training logic
        return {"loss": 0.1, "accuracy": 0.95}

    async def evaluate(self, config):
        # Evaluation logic
        return {"accuracy": 0.96}
```

### Adding a New Dataset

In `src/data/datasets.py`:

```python
from src.core import Dataset, register

@register("dataset", "my_dataset")
class MyDataset(Dataset):
    def __init__(self, data_dir: str, download: bool = True):
        # Load dataset
        pass

    def get_data(self, partition_id: int):
        # Return data for specific partition
        pass
```

---

## Git Workflow

### Keeping Your Fork Updated

```bash
# Fetch upstream changes
git fetch upstream

# Merge into your main branch
git checkout main
git merge upstream/main

# Push to your fork
git push origin main
```

### Rebasing Your Feature Branch

```bash
# Update main first
git checkout main
git pull upstream main

# Rebase your feature branch
git checkout feature/your-feature
git rebase main

# Force push if already pushed
git push --force-with-lease origin feature/your-feature
```

---

## Release Process

(For maintainers)

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create release branch: `release/v0.2.0`
4. Tag: `git tag -a v0.2.0 -m "Release v0.2.0"`
5. Push: `git push origin v0.2.0`
6. Create GitHub Release with notes

---

## Getting Help

- **Questions**: [GitHub Discussions](https://github.com/YOUR_USERNAME/MOE-FedCL/discussions)
- **Bugs**: [GitHub Issues](https://github.com/YOUR_USERNAME/MOE-FedCL/issues)
- **Email**: your.email@example.com
- **Slack/Discord**: (if available)

---

## Recognition

Contributors will be recognized in:
- `CONTRIBUTORS.md` file
- Release notes
- GitHub contributors page

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to MOE-FedCL! 🎉
