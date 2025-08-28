# NSeekFS 🚀

[![PyPI version](https://badge.fury.io/py/nseekfs.svg)](https://badge.fury.io/py/nseekfs)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**High-Performance Vector Similarity Search with Rust Backend**

Fast and exact cosine similarity search for Python. Built with Rust for performance, designed for production use.

```bash
pip install nseekfs
```

## 🎯 Quick Start

```python
import nseekfs
import numpy as np

# Create some test vectors
embeddings = np.random.randn(10000, 384).astype(np.float32)
query = np.random.randn(384).astype(np.float32)

# Build index and search (cosine similarity)
index = nseekfs.from_embeddings(embeddings, normalize=True)
results = index.query(query, top_k=10)

# Results: [{'idx': 42, 'score': 0.95}, ...]
print(f"Found {len(results)} results")
print(f"Best match: vector {results[0]['idx']} (similarity: {results[0]['score']:.3f})")
```

## ⚡ Performance Benchmarks

*Benchmarks from TestPyPI installation (actual performance data)*

### Single Query Performance
| Dataset | Dimensions | Query Time | vs NumPy | Speedup |
|---------|------------|------------|----------|---------|
| 1K vectors | 128 | 0.8ms | 2.4ms | **3.0x** |
| 10K vectors | 384 | 8ms | 45ms | **5.6x** |
| 100K vectors | 512 | 85ms | 520ms | **6.1x** |

**Note**: Performance varies based on hardware, vector dimensions, and data distribution.

## 🛠️ Complete Feature Set

### Core Search (Cosine Similarity)
```python
# Basic search
index = nseekfs.from_embeddings(embeddings)
results = index.query(query, top_k=10)

# With normalization control
index = nseekfs.from_embeddings(embeddings, normalize=True)
results = index.query(query, top_k=10)

# Verbose mode for debugging
index = nseekfs.from_embeddings(embeddings, normalize=True, verbose=True)
results = index.query(query, top_k=10)
```

### Batch Processing
```python
# Process multiple queries efficiently
queries = np.random.randn(50, 384).astype(np.float32)
all_results = index.query_batch(queries, top_k=5)

print(f"Processed {len(queries)} queries in batch")
for i, results in enumerate(all_results):
    print(f"Query {i}: {len(results)} results")
```

### Index Persistence
```python
# Load existing index
index = nseekfs.from_bin("my_vectors.nseek")
print(f"Loaded index with {index.rows} vectors")

# Note: Save functionality exists in Rust backend but not exposed in Python API yet
```

### Performance Monitoring
```python
# Get detailed performance metrics
metrics = index.get_performance_metrics()
print(f"Total queries: {metrics['total_queries']}")
print(f"Average time: {metrics['avg_query_time_ms']:.2f}ms")
print(f"SIMD usage: {metrics['simd_queries']}/{metrics['total_queries']}")

# Query with timing information
results, timing = index.query(query, top_k=10, return_timing=True)
print(f"Query took {timing['query_time_ms']:.2f}ms")
```

## 🏗️ Architecture Highlights

### SIMD Optimizations
- **AVX2 support** for 8x parallelism on compatible CPUs
- **Automatic fallback** to scalar operations on older hardware  
- **Runtime detection** of CPU capabilities

### Memory Management
- **Memory mapping** for efficient data access
- **Thread-local buffers** for zero-allocation queries
- **Cache-aligned data structures** for optimal performance

## ✅ What Currently Works

### Confirmed Features
- **Exact cosine similarity search** - 100% accurate results
- **SIMD acceleration** - AVX2 optimization when available
- **Batch processing** - efficient handling of multiple queries  
- **Index persistence** - load pre-built indices (from_bin)
- **Performance monitoring** - detailed timing and usage metrics
- **Memory efficient** - scales to hundreds of thousands of vectors
- **Cross-platform** - Windows, macOS, Linux support (CI tested)
- **Float32 optimized** - designed for standard ML embeddings

### Current Limitations
- **Cosine similarity only** - no other distance metrics in Python API yet
- **No save method** - can load indices but not save from Python (Rust has it)
- **Single-threaded queries** - parallelism limited to SIMD level
- **Memory usage** - stores full vectors in memory (no compression)

## 🔧 System Requirements

- **Python 3.8+** (CI tested: 3.8, 3.9, 3.10, 3.11, 3.12)
- **NumPy** (automatically installed)
- **4GB+ RAM** recommended for large datasets
- **Modern CPU** (AVX2 support recommended but not required)

## 📊 When to Use NSeekFS

### Ideal For:
- **Research and prototyping** - exact results for ground truth
- **Medium-scale applications** - up to ~1M vectors  
- **Cosine similarity applications** - semantic search, recommendations
- **Quality-critical systems** - where accuracy matters more than speed
- **Cross-platform deployment** - consistent behavior across OS

### Consider Alternatives For:
- **Multiple similarity metrics** - currently only supports cosine
- **Very large datasets** (>10M vectors) - dedicated vector databases
- **Ultra-low latency** (<1ms) - specialized hardware solutions  
- **Approximate search** - Faiss, Annoy, or HNSW libraries
- **High write throughput** - this is optimized for read-heavy workloads

## 🧪 System Health Check

```python
import nseekfs

# Check system compatibility
health = nseekfs.health_check(verbose=True)
print(f"Status: {health['status']}")
print(f"SIMD available: {health['simd_available']}")

# Run built-in benchmark
nseekfs.benchmark(vectors=1000, dims=384, queries=100)
```

## 📈 Performance Tips

### For Maximum Speed:
```python
# Pre-normalize vectors if using cosine similarity
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
index = nseekfs.from_embeddings(embeddings, normalize=False)

# Use batch queries for multiple searches
results = index.query_batch(queries, top_k=10)

# Choose appropriate top_k (smaller = faster)
results = index.query(query, top_k=5)  # vs top_k=100
```

### Memory Optimization:
```python
# Always use float32 for embeddings
embeddings = embeddings.astype(np.float32)

# Load pre-built indices instead of rebuilding
index = nseekfs.from_bin("embeddings.nseek")
```

## 🎯 API Reference

### Index Creation
```python
# Basic usage
index = nseekfs.from_embeddings(embeddings)

# With options
index = nseekfs.from_embeddings(
    embeddings, 
    normalize=True,    # Normalize vectors (default: False)
    verbose=True       # Show progress (default: False)
)

# Load existing index
index = nseekfs.from_bin("path/to/index.nseek")
```

### Querying
```python
# Single query (simple format)
results = index.query(query_vector, top_k=10)
# Returns: [{'idx': int, 'score': float}, ...]

# Single query with timing
results, timing = index.query(query_vector, top_k=10, return_timing=True)

# Batch queries
all_results = index.query_batch(queries_array, top_k=10)
# Returns: List of result lists

# Detailed query (with metrics)
result_obj = index.query(query_vector, top_k=10, format="detailed")
# Returns: QueryResult object with metadata
```

## 🤝 Contributing

This is an active project with room for improvement.

**Priority areas for contribution:**
- Expose multiple similarity metrics to Python API
- Add save functionality to Python API  
- Multi-threading support for queries
- Memory usage optimization
- Documentation and examples

**Development setup:**
```bash
git clone https://github.com/NSeek/nseekfs
cd nseekfs
pip install maturin numpy
maturin develop --release
```

## 📜 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙋‍♂️ Support

- **Issues**: [GitHub Issues](https://github.com/NSeek/nseekfs/issues)
- **Source Code**: [GitHub Repository](https://github.com/NSeek/nseekfs)

## 🔄 Version History

**v1.0.0** - Initial release
- Exact cosine similarity search with Rust backend
- SIMD optimizations (AVX2)  
- Batch processing and performance monitoring
- Cross-platform support (Windows, macOS, Linux)
- Index loading (from_bin functionality)

---

**Fast, exact cosine similarity search for Python.**

*Built with Rust for performance, designed for Python developers.*