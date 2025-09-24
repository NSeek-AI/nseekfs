# NSeekFS

[![PyPI version](https://badge.fury.io/py/nseekfs.svg)](https://pypi.org/project/nseekfs)  
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**High-Performance Exact Vector Search with Rust Backend**

Fast and exact vector similarity search for Python. Built with Rust for performance, designed for production use.

---

NSeekFS combines the safety and performance of Rust with a clean Python API.  
This release supports **exact vector search** with multiple similarity metrics:

- `cosine` (requires normalized vectors)  
- `dot`  
- `euclidean`  

Upcoming releases will expand support to:  
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

# Choose metric: "cosine", "dot", or "euclidean"
metric = "cosine"

# Normalize only if using cosine
if metric == "cosine":
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    query = query / np.linalg.norm(query)

# Build index and run a search
index = nseekfs.from_embeddings(embeddings, metric=metric, normalized=True)
results = index.query(query, top_k=10)

print(f"Found {len(results)} results")
print(f"Best match: idx={results[0]['idx']} score={results[0]['score']:.3f}")
```

## Core Features

### Exact Search

```python
# Simple query
results = index.query(query, top_k=10)

# Access results
for item in results:
    print(f"Vector {item['idx']}: {item['score']:.6f}")
```

### Batch Queries

```python
queries = np.random.randn(50, 384).astype(np.float32)
if metric == "cosine":
    queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)

batch_results = index.query_batch(queries, top_k=5)
print(f"Processed {len(batch_results)} queries")
```

### Query Options

```python
# Simple query
results = index.query_simple(query, top_k=10)

# Detailed query
result = index.query_detailed(query, top_k=10)
print(f"Query took {result.query_time_ms:.2f} ms, top1 idx={result.results[0]['idx']}")
```

### Index Persistence

```python
# Build and save index
index = nseekfs.from_embeddings(embeddings, metric=metric, normalized=True)
print("Index saved at:", index.index_path)

# Later, reload from file
index2 = nseekfs.from_bin(index.index_path)
print(f"Reloaded index: {index2.rows} vectors x {index2.dims} dims")
```

## API Reference

### Index

* `from_embeddings(embeddings, metric="cosine", normalized=True, verbose=False)`
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

## Metric Guide

| Metric     | Normalization required | Typical use case                         |
|------------|-------------------------|------------------------------------------|
| cosine     | Yes                     | Semantic embeddings (e.g. sentence-transformers, OpenAI) |
| dot        | No                      | Raw model outputs where scale carries meaning |
| euclidean  | No                      | Geometric distance or clustering tasks    |

## Architecture Highlights

- **Similarity Metrics**: cosine (with enforced normalization), dot product, and Euclidean (−L2²).  
- **Batch Query Engine**: adaptive selection between full matrix GEMM, chunked streaming, and parallel Rayon fallback.  
- **SIMD Acceleration**: custom `wide::f32x8` kernels for dot/L2, with `matrixmultiply::sgemm` for block GEMM.  
- **Memory Mapping**: compact binary format (header + float data) using `memmap2` for zero-copy loading.  
- **Streaming Index Build**: chunked writes, runtime memory estimation, safe normalization, and atomic file finalization.  
- **Cross-Platform Optimizations**: runtime SIMD detection (AVX2/AVX/SSE4.2/NEON) and environment-controlled tuning (`NSEEK_THREADS`, `NSEEK_QBLOCK`, `NSEEK_DBLOCK`).  


## Installation

```bash
# From PyPI
pip install nseekfs

# Verify installation
python -c "import nseekfs; print('NSeekFS installed successfully')"
```

## Technical Details

- **Similarity Metrics**: Cosine (with enforced normalization), Dot Product, Euclidean (−L2²).  
- **Precision**: Float32 core; future-ready hooks for f16, f8, f64 levels.  
- **Index Format**: Compact binary (12-byte header + contiguous float data), memory-mapped via `memmap2`.  
- **Batch Queries**: Adaptive engine with GEMM (matrixmultiply), SIMD kernels (`wide::f32x8`), or parallel Rayon fallback.  
- **Memory**: Streaming index build with chunking, safe normalization, and runtime memory estimation.  
- **Performance**: AVX2/AVX/SSE4.2/NEON runtime detection; env vars (`NSEEK_THREADS`, `NSEEK_QBLOCK`, `NSEEK_DBLOCK`) for tuning.  
- **Thread Safety**: Parallel query execution via Rayon; thread-safe index loading.  
- **Compatibility**: Python 3.8+ on Windows, macOS, Linux.  
  

## Performance Tips

```python
# Cosine similarity: normalize embeddings and queries
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
index = nseekfs.from_embeddings(embeddings, metric="cosine", normalized=True)

# Dot or Euclidean: use raw embeddings, no normalization
index = nseekfs.from_embeddings(embeddings, metric="dot", normalized=False)
```

## License

MIT License - see LICENSE file for details.

---

**Fast, exact vector similarity search for Python.**  

*Built with Rust for performance, designed for Python developers.*  
Source: [github.com/NSeek-AI/nseekfs](https://github.com/NSeek-AI/nseekfs)
