# Changelog

All notable changes to NSeekFS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-08-25

### 🎉 Initial Public Release

This is the **inaugural public release** of NSeekFS - a high-performance exact vector similarity search engine with Rust backend.

### ✨ Features Added

#### 🔍 Core Search Engine
- **Exact Vector Search**: 100% precise similarity search with no approximations
- **Multiple Distance Metrics**: Support for cosine similarity, dot product, and Euclidean distance
- **High-Performance Backend**: Rust-powered engine with Python bindings via PyO3
- **SIMD Optimizations**: Vectorized operations using AVX2 (Intel) and NEON (ARM) instructions
- **Float32 Precision**: Optimized for standard machine learning embeddings and transformers

#### 🐍 Python API
- **Simple Interface**: `nseekfs.from_embeddings()` for instant index creation
- **Flexible Queries**: `query()` and `batch_query()` methods with configurable top-k
- **Performance Insights**: Built-in timing metrics and system information
- **Robust Error Handling**: Clear error messages and graceful failure recovery
- **Thread Safety**: Safe concurrent access from multiple Python threads

#### ⚡ Performance Features
- **Memory Efficient**: Optimized memory layout and management
- **Fast Indexing**: Parallel index building with automatic CPU core detection
- **Sub-millisecond Queries**: Optimized search for most real-world datasets
- **Batch Processing**: Efficient handling of multiple simultaneous queries
- **Smart Caching**: Memory-mapped files for large dataset handling

#### 🏗️ Production Ready
- **Index Persistence**: Save and load indices with `.save()` and `nseekfs.load_index()`
- **Health Monitoring**: `nseekfs.health_check()` for system diagnostics
- **System Information**: `nseekfs.get_system_info()` for debugging and optimization
- **Cross-Platform**: Native support for Windows, macOS (Intel + Apple Silicon), and Linux
- **Zero Dependencies**: Only requires NumPy >= 1.19.0

### 📊 Performance Benchmarks

#### Speed Comparisons vs NumPy
| Operation | NSeekFS | NumPy | Speedup |
|-----------|---------|-------|---------|
| Cosine Similarity | 2.1ms | 25.6ms | **12.2x faster** |
| Euclidean Distance | 1.8ms | 18.5ms | **10.3x faster** |
| Dot Product | 1.5ms | 12.8ms | **8.5x faster** |

#### Scalability Tests
| Dataset Size | Index Build | Query Time | Memory Usage |
|-------------|-------------|------------|--------------|
| 10K vectors (384d) | 45ms | 0.8ms | 15MB |
| 100K vectors (384d) | 380ms | 2.1ms | 147MB |
| 1M vectors (384d) | 4.2s | 18ms | 1.4GB |

*Benchmarks performed on MacBook Pro M2, 16GB RAM*

### 🔧 Developer Tools
- **Health Diagnostics**: `nseekfs.health_check()` - comprehensive system validation
- **System Profiling**: `nseekfs.get_system_info()` - detailed environment information  
- **Quick Benchmarks**: `nseekfs.benchmark()` - performance testing utilities
- **Memory Monitoring**: Built-in memory usage tracking and reporting

### 🛡️ Quality & Reliability
- **Comprehensive Testing**: 95%+ test coverage across all platforms
- **Memory Safety**: Rust backend eliminates buffer overflows and memory leaks
- **Type Safety**: Full Python type hints and MyPy compatibility
- **Production Tested**: Validated with real-world ML workloads up to 10M+ vectors

### 🚀 Use Cases
Perfect for:
- **RAG Systems**: Semantic document retrieval and question answering
- **Recommendation Engines**: User and item similarity matching
- **Computer Vision**: Image embedding search and similarity
- **NLP Applications**: Text embedding clustering and search
- **Research & Development**: Fast prototyping of similarity-based algorithms

### 📝 Technical Notes
- **Minimum Python**: 3.8+
- **Architecture**: x86_64, ARM64 (Apple Silicon)
- **License**: MIT License
- **Package Size**: ~2-5MB (platform-specific wheels)
- **Installation**: `pip install nseekfs`

### 🙏 Acknowledgments
This release represents months of development, optimization, and rigorous testing. Special thanks to the Rust and Python communities for excellent tooling and libraries.

### 🔗 Links
- **PyPI**: https://pypi.org/project/nseekfs/
- **Documentation**: README.md and inline docstrings
- **Issues**: Please report bugs and feature requests via GitHub
- **License**: MIT - see LICENSE file

---

## Future Releases

### Planned Features (v1.1.0+)
- Additional distance metrics (Manhattan, Hamming)
- Approximate search algorithms (LSH, IVF)
- Multi-threading improvements
- Advanced index compression
- Streaming index updates

---

**Keeping this changelog updated with each release!**