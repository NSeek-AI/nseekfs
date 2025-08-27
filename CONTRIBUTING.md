# Contributing to NSeekFS

Thank you for your interest in contributing to NSeekFS! We welcome contributions from the community and are excited to work with you.

## 🚀 Project Overview

NSeekFS v1.0 focuses on **exact vector search** with high performance and reliability. Our roadmap includes:

- **v1.x**: Exact search optimization, multiple metrics, memory efficiency
- **v2.x**: Approximate nearest neighbor (ANN) search, hybrid modes

## 🛠️ Development Setup

### Prerequisites

- **Python 3.8+**
- **Rust 1.70+** (for development)
- **Git**

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/NSeek-AI/nseekfs.git
cd nseekfs

# Run setup script
chmod +x setup_dev.sh
./setup_dev.sh
```

### Manual Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Install maturin for Rust-Python bindings
pip install maturin

# Build in development mode
maturin develop --release

# Run tests
python test_nseekfs_v1.py
```

## 📋 How to Contribute

### 1. Types of Contributions

We welcome several types of contributions:

- **🐛 Bug Reports**: Report issues with detailed reproduction steps
- **✨ Feature Requests**: Suggest new features or improvements
- **📚 Documentation**: Improve README, docstrings, or examples
- **🔧 Code Improvements**: Performance optimizations, code quality
- **🧪 Tests**: Add test cases or improve test coverage
- **🌍 Examples**: Real-world usage examples

### 2. Getting Started

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Make** your changes
4. **Test** your changes: `python test_nseekfs_v1.py`
5. **Commit** your changes: `git commit -m 'Add amazing feature'`
6. **Push** to your branch: `git push origin feature/amazing-feature`
7. **Open** a Pull Request

### 3. Development Workflow

#### Code Structure

```
nseekfs/
├── src/                    # Rust source code
│   ├── lib.rs             # Main Python bindings
│   ├── engine.rs          # Core search engine
│   ├── prepare.rs         # Index creation
│   └── utils/             # Utility modules
├── nseekfs/               # Python package
│   ├── __init__.py        # Main API
│   ├── highlevel.py       # High-level interface
│   └── validation.py      # Input validation
├── tests/                 # Test suite
├── examples/              # Usage examples
└── docs/                  # Documentation
```

#### Rust Development

```bash
# Format code
cargo fmt

# Check for issues
cargo clippy

# Run Rust tests
cargo test

# Build release
cargo build --release
```

#### Python Development

```bash
# Format code
black nseekfs/
ruff check nseekfs/

# Type checking
mypy nseekfs/

# Build Python package
maturin develop

# Run tests
pytest tests/
python test_nseekfs_v1.py
```

## 🧪 Testing

### Running Tests

```bash
# Complete test suite
python test_nseekfs_v1.py

# Quick verification
python quick_example.py

# Python unit tests
pytest tests/ -v

# Rust tests
cargo test

# Benchmark tests
python test_nseekfs_v1.py --benchmark
```

### Test Categories

- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end functionality
- **Performance Tests**: Benchmark and timing
- **Edge Cases**: Boundary conditions and error handling

### Adding Tests

When adding new features, please include:

1. **Unit tests** for the core functionality
2. **Integration tests** for the Python API
3. **Documentation** with usage examples
4. **Performance benchmarks** if relevant

## 📝 Code Style

### Python

We use [Black](https://black.readthedocs.io/) for code formatting and [Ruff](https://docs.astral.sh/ruff/) for linting:

```bash
# Format code
black nseekfs/

# Check style
ruff check nseekfs/

# Fix issues automatically
ruff check --fix nseekfs/
```

### Rust

We follow standard Rust formatting and linting:

```bash
# Format code
cargo fmt

# Check for issues
cargo clippy

# Fix issues
cargo clippy --fix
```

### Documentation

- Use clear, concise language
- Include code examples for functions
- Add type hints to Python code
- Document parameters and return values

## 🐛 Bug Reports

When reporting bugs, please include:

### Required Information

- **NSeekFS version**: `nseekfs.__version__`
- **Python version**: `python --version`
- **Operating system**: Windows/macOS/Linux + version
- **NumPy version**: `numpy.__version__`

### Reproduction Steps

```python
import nseekfs
import numpy as np

# Minimal example that reproduces the bug
vectors = np.random.randn(100, 32).astype(np.float32)
index = nseekfs.from_embeddings(vectors)
# ... steps that cause the issue
```

### Expected vs Actual Behavior

- **Expected**: What should happen
- **Actual**: What actually happens
- **Error messages**: Full traceback if applicable

## ✨ Feature Requests

When suggesting features, please include:

### Use Case

- **Problem**: What problem does this solve?
- **Solution**: How would this feature help?
- **Alternatives**: What alternatives have you considered?

### Implementation Ideas

- **API Design**: How should the feature be exposed?
- **Performance**: Any performance considerations?
- **Compatibility**: Impact on existing functionality?

## 🔄 Pull Request Process

### Before Submitting

1. **Test** your changes thoroughly
2. **Update** documentation if needed
3. **Add** tests for new functionality
4. **Follow** code style guidelines
5. **Rebase** on latest main branch

### PR Checklist

- [ ] Code follows project style guidelines
- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] Documentation updated if needed
- [ ] No breaking changes (or clearly documented)
- [ ] Performance impact considered

### Review Process

1. **Automated checks** run on all PRs
2. **Maintainer review** for code quality and design
3. **Testing** on multiple platforms
4. **Documentation review** if applicable
5. **Merge** after approval

## 📚 Documentation

### Types of Documentation

- **API Documentation**: Docstrings and type hints
- **User Guide**: README and usage examples
- **Developer Guide**: This contributing guide
- **Examples**: Real-world usage patterns

### Writing Guidelines

- **Be clear and concise**
- **Include code examples**
- **Use consistent terminology**
- **Test all code examples**

## 🎯 Performance Considerations

### Optimization Guidelines

- **Profile first**: Use profiling tools to identify bottlenecks
- **Measure impact**: Benchmark before and after changes
- **Consider trade-offs**: Memory vs speed, accuracy vs performance
- **Test on realistic data**: Use representative datasets

### Performance Testing

```python
# Example benchmark
import time
import numpy as np
import nseekfs

def benchmark_feature():
    vectors = np.random.randn(10000, 384).astype(np.float32)
    query = np.random.randn(384).astype(np.float32)
    
    # Measure index creation
    start = time.time()
    index = nseekfs.from_embeddings(vectors)
    build_time = time.time() - start
    
    # Measure query time
    start = time.time()
    results = index.query(query, top_k=10)
    query_time = time.time() - start
    
    print(f"Build: {build_time:.3f}s, Query: {query_time*1000:.2f}ms")

if __name__ == "__main__":
    benchmark_feature()
```

## 🚦 Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

### Release Types

- **v1.x.x**: Exact search improvements
- **v2.0.0**: ANN functionality (future)
- **v2.x.x**: ANN improvements and new features

## 💬 Communication

### Getting Help

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and community chat
- **Documentation**: README and code examples

### Community Guidelines

- **Be respectful** and professional
- **Help others** when you can
- **Search existing issues** before creating new ones
- **Provide context** when asking questions

## 🏆 Recognition

Contributors will be:

- **Listed** in the CONTRIBUTORS file
- **Mentioned** in release notes for significant contributions
- **Invited** to join the maintainer team for ongoing contributors

## 📄 License

By contributing to NSeekFS, you agree that your contributions will be licensed under the MIT License.

---

## 🚀 Ready to Contribute?

1. **Set up** your development environment
2. **Pick an issue** from our GitHub issues
3. **Join the discussion** if you have questions
4. **Submit your PR** when ready

Thank you for helping make NSeekFS better! 🎉