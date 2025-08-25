# NSeekFS 🚀

[![PyPI version](https://badge.fury.io/py/nseekfs.svg)](https://badge.fury.io/py/nseekfs)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://pepy.tech/badge/nseekfs)](https://pepy.tech/project/nseekfs)
[![Build Status](https://github.com/NSeek-AI/nseekfs/workflows/CI/badge.svg)](https://github.com/NSeek-AI/nseekfs/actions)

**High-Performance Exact Vector Similarity Search**

NSeekFS delivers **production-ready vector search** with blazing performance. Built with Rust backend and optimized for real-world ML applications.

```bash
pip install nseekfs
```

## ⚡ Performance First

- **12x faster** than NumPy cosine similarity
- **100% exact results** - no approximations
- **SIMD optimizations** (AVX2, NEON)  
- **Memory efficient** - handles millions of vectors
- **Production ready** - thread-safe, battle-tested

## 🎯 Quick Start

```python
import nseekfs
import numpy as np

# Your embeddings (any size)
embeddings = np.random.randn(100000, 384).astype(np.float32)
query = np.random.randn(384).astype(np.float32)

# Build index and search
index = nseekfs.from_embeddings(embeddings)
results = index.query(query, top_k=10)

# Clean results: [{'idx': 0, 'score': 0.95}, ...]
print(f"Top result: index {results[0]['idx']} with score {results[0]['score']:.3f}")
```

## 🏆 Real-World Performance

| Dataset Size | NSeekFS | NumPy | Speedup |
|-------------|---------|-------|---------|
| 10K vectors | 2.1ms   | 24ms  | **11.4x** |
| 100K vectors| 18ms    | 240ms | **13.3x** |  
| 1M vectors  | 180ms   | 2.4s  | **13.3x** |

*Tested on MacBook Pro M2, 384-dim vectors*

## 🔥 Why Choose NSeekFS?

**Perfect for:**
- RAG systems and semantic search
- Recommendation engines  
- Document similarity
- Image/embedding matching
- Research and ML pipelines

**Built for Production:**
- Zero dependencies beyond NumPy
- Thread-safe operations
- Optimized memory usage
- Cross-platform wheels
- Professional support

## 📖 Documentation

- [Installation Guide](docs/installation.md)
- [Quick Start](docs/quickstart.md) 
- [API Reference](docs/api_reference.md)
- [Performance Tips](docs/performance.md)
- [Examples](examples/)

## 🛠️ Advanced Features

```python
# Performance monitoring
results, timing = index.query(query, return_timing=True)
print(f"Search took {timing['total_ms']:.2f}ms")

# Batch queries for maximum efficiency  
queries = np.random.randn(100, 384).astype(np.float32)
all_results = index.batch_query(queries, top_k=5)

# Save and load indices
index.save("my_index.nseek")
index2 = nseekfs.load_index("my_index.nseek")
```

## 🤝 Contributing

We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
```bash
git clone https://github.com/NSeek-AI/nseekfs
cd nseekfs  
pip install -e ".[dev]"
pytest tests/
```

## 📊 Benchmarks

Run your own performance tests:
```python
import nseekfs
nseekfs.benchmark_system(verbose=True)  # Compare vs NumPy
```

## 📜 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙋‍♂️ Support

- **Issues**: [GitHub Issues](https://github.com/NSeek-AI/nseekfs/issues)
- **Discussions**: [GitHub Discussions](https://github.com/NSeek-AI/nseekfs/discussions)  
- **Email**: support@nseek.ai

---

**🚀 Production Ready | 🧪 Battle Tested | 🔒 Memory Safe**

*Built with ❤️ by the [NSeek team](https://github.com/NSeek-AI)*

> 🔬 **Coming Soon**: NSeekFS is part of the larger **NSeek AI Platform** - comprehensive ML infrastructure tools for modern applications.