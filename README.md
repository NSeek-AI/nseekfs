# NSeekFS

[![PyPI version](https://badge.fury.io/py/nseekfs.svg)](https://pypi.org/project/nseekfs)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**High-Performance Exact Vector Search with Rust Backend**

Fast and exact cosine similarity search for Python. Built with Rust for performance, designed for production use.

---

NSeekFS combines the safety and performance of Rust with a clean Python API.  
This first release focuses on **exact cosine search with SIMD acceleration**, providing predictable and reproducible results for ML workloads.  

Upcoming releases will expand support to:
- Euclidean distance
- Approximate Nearest Neighbor (ANN) search
- Additional precision levels and memory optimizations

Our goal: deliver a **fast, reliable, and production-ready search engine** that evolves with your needs.

```bash
pip install nseekfs
```

## Quick Start

```python
import nseekfs
import numpy as np

# Create some test vectors
embeddings = np.random.randn(10000, 384).astype(np.float32)
query = np.random.randn(384).astype(np.float32)

# Normalize embeddings and query
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
query = query / np.linalg.norm(query)

# Build index and run a search
#By default, from_embeddings assumes vectors normalized (normalized=True). 
#If your vectors are not normalized, set normalized=False and NSeekFS will handle it internally
index = nseekfs.from_embeddings(embeddings, normalized=True)
results = index.query(query, top_k=10)

print(f"Found {len(results)} results")
print(f"Best match: idx={results[0]['idx']} score={results[0]['score']:.3f}")
```

## Core Features

### Exact Search

```python
# Basic query
results = index.query(query, top_k=10)

# Access results
for item in results:
    print(f"Vector {item['idx']}: {item['score']:.6f}")
```

### Batch Queries

```python
queries = np.random.randn(50, 384).astype(np.float32)
batch_results = index.query_batch(queries, top_k=5)
print(f"Processed {len(batch_results)} queries")
```

### Query Options

```python
# Simple query (alias for query with format="simple")
results = index.query_simple(query, top_k=10)

# Detailed query with timing and diagnostics
result = index.query_detailed(query, top_k=10)
print(f"Query took {result.query_time_ms:.2f} ms, top1 idx={result.results[0]['idx']}")
```

### Index Persistence

```python
# Build and save index
index = nseekfs.from_embeddings(embeddings, normalized=True)
print("Index saved at:", index.index_path)

# Later, reload from file
index2 = nseekfs.from_bin(index.index_path)
print(f"Reloaded index: {index2.rows} vectors x {index2.dims} dims")
```

### Performance Metrics

```python
metrics = index.get_performance_metrics()
print(f"Total queries: {metrics['total_queries']}")
print(f"Average time: {metrics['avg_query_time_ms']:.2f} ms")
```

### Built-in Benchmark

```python
nseekfs.benchmark(vectors=1000, dims=384, queries=100, verbose=True)
```

## API Reference

### Index

* `from_embeddings(embeddings, normalized=True, verbose=False)`
* `from_bin(path)`

### Queries

* `query(query_vector, top_k=10)`
* `query_simple(query_vector, top_k=10)`
* `query_detailed(query_vector, top_k=10)`
* `query_batch(queries, top_k=10)`

### Properties

* `index.rows`
* `index.dims`
* `index.config`

### Utilities

* `get_performance_metrics()`
* `benchmark(vectors=..., dims=..., queries=...)`

## Architecture Highlights

### SIMD Optimizations
- AVX2 support for 8x parallelism on compatible CPUs
- Automatic fallback to scalar operations on older hardware  
- Runtime detection of CPU capabilities

### Memory Management
- Memory mapping for efficient data access
- Thread-local buffers for zero-allocation queries
- Cache-aligned data structures for optimal performance

### Batch Processing
- Intelligent batching strategies based on query size
- SIMD vectorization across multiple queries
- Optimized memory access patterns

## Installation

```bash
# From PyPI
pip install nseekfs

# Verify installation
python -c "import nseekfs; print('NSeekFS installed successfully')"
```

## Technical Details

- **Precision**: Float32 optimized for standard ML embeddings
- **Memory**: Efficient memory usage with optimized data structures
- **Performance**: Rust backend with SIMD optimizations where available
- **Compatibility**: Python 3.8+ on Windows, macOS, and Linux
- **Thread Safety**: Safe concurrent access from multiple threads

## Performance Tips

```python
# Pre-normalize vectors if using cosine similarity
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
index = nseekfs.from_embeddings(embeddings, normalized=True)

# Use appropriate data types
embeddings = embeddings.astype(np.float32)

# Choose optimal top_k values
results = index.query(query, top_k=10)  # vs top_k=1000

# Use batch processing for multiple queries
batch_results = index.query_batch(queries, top_k=10)
```

## License

MIT License - see LICENSE file for details.

---

**Fast, exact cosine similarity search for Python.**

*Built with Rust for performance, designed for Python developers.*
Source: [github.com/NSeek-AI/nseekfs](https://github.com/NSeek-AI/nseekfs)
