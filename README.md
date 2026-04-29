# NSeekFS

[![PyPI version](https://badge.fury.io/py/nseekfs.svg)](https://pypi.org/project/nseekfs)  
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**High-Performance Exact Vector Search with Rust Backend**

Fast and exact vector similarity search for Python. Built with Rust for performance, designed for production use, with a split between a low-overhead search path and a richer audit/detailed path.

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

### Result Layers

NSeekFS now exposes two result layers on top of the exact engine:

- `simple` for the lowest-overhead ranked results
- `detailed` for engineering, audit, margins, replay, and debugging

Use `simple` when you care about latency. Use `detailed` when you need traceability.

```python
# Fast path: minimal result shaping
results = index.query_simple(query, top_k=10)
raw = index.query_simple_arrays(query, top_k=10)

# Detailed path: ranking + audit metadata + margins
detail = index.query_detailed(query, top_k=10)
print(detail.results[0]["rank"], detail.results[0]["idx"], detail.results[0]["score"])
print(detail.margins)
print(detail.audit["index_hash"])

# Batch detailed path
batch_detail = index.query_batch(queries, top_k=10, format="detailed")
```

What the detailed layer adds:

- deterministic ranking with `rank`
- `score` on every result
- `margins` such as `margin_to_next` and `margin_top1_to_last`
- audit metadata: `engine_version`, `metric`, `dims`, `rows`, `index_hash`, `query_hash`
- optional replay support through exported audit JSON

### Query Options

```python
# Simple query
results = index.query_simple(query, top_k=10)

# Fast array query for engineers
raw = index.query_simple_arrays(query, top_k=10)
print(raw["indices"], raw["scores"])

# Detailed query
result = index.query_detailed(query, top_k=10)
print(f"Query took {result.query_time_ms:.2f} ms, top1 rank={result.results[0]['rank']}, idx={result.results[0]['idx']}")
print(result.margins["margin_top1_to_last"])
print(result.audit["query_hash"])
```

### Provably Optimal Exact Search (Certified)

```python
# Exact search with deterministic pruning and optimality certificate
certified = index.query_exact_certified(
    query,
    top_k=10,
    block_size=64,
    enable_pruning=True,
    return_certificate=True,
)

print(certified.results[0])
print(certified.certificate["safe"])  # must be True
print(certified.certificate["pruned_candidates"])
```

This mode keeps exact semantics while pruning candidates using safe bounds.
Every certified query returns metadata proving pruning safety for the final top-k.

#### Certified Guarantee

- Same ranking as brute force exact search for the same metric and inputs.
- Deterministic tie-break rule: score order + `idx` ascending as final tie-break.
- Certificate includes pruning counts, final `kth_score`, bound type, and pruned bound values.

#### Assumptions and Contracts

- Data type: `float32`.
- `cosine`:
- If `normalized=True`, embeddings are expected normalized by caller.
- If `normalized=False`, index build normalizes embeddings internally.
- Query vector can be any norm (ranking is invariant to positive scaling).
- Optional strict mode: `strict_query_normalized=True` in `query_exact_certified`.

#### When It Speeds Up

- More acceleration when early blocks are discriminative and `top_k` is small.
- Less acceleration on highly tied datasets or when all vectors are similarly close.
- Correctness does not depend on speedup; pruning may be low and still certified.

#### Audit and Replay

- `export_audit` stores certificate and replay metadata (`block_order`, `block_size`, `simd_path`, `compile_flags`, bounds).
- `replay` validates ranking and certificate invariants (counts and bound condition).

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
* `query_simple_arrays(query_vector, top_k=10)`
* `query_detailed(query_vector, top_k=10)`
* `query_batch(queries, top_k=10)`
* `query_batch_detailed(queries, top_k=10)`
* `query_batch_arrays(queries, top_k=10)`
* `query_exact_audit(query_vector, top_k=10)`
* `query_exact_certified(query_vector, top_k=10, block_size=64, enable_pruning=True, return_certificate=True)`

### Audit and Replay

```python
# Export an audit JSON from a detailed query
result = index.query_detailed(query, top_k=10)
audit = nseekfs.export_audit(result, "audit.json")

# Replay later using the saved audit
replayed = nseekfs.replay("audit.json")
print(replayed["ok"])
```

This layer is intentionally separate from `query_simple`. It is for:

- deterministic reruns
- offline verification
- debugging ranking differences
- chatbot confidence logic based on margins and ties

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

The primary optimized path in this release is `cosine` with already-normalized embeddings.  
That is the path to use when you want the lowest overhead and the simplest engineering contract.  
`dot` and `euclidean` remain available for compatibility and debugging.  
Ordering is deterministic: equal scores are broken by smaller `idx`.

## Architecture Highlights

- **Similarity Metrics**: cosine (with enforced normalization), dot product, and Euclidean (−L2²).  
- **Result Layers**: low-overhead `simple`, engineering-oriented `detailed`, and audit/replay metadata for deterministic validation.  
- **Batch Query Engine**: adaptive selection between full matrix GEMM, chunked streaming, and parallel Rayon fallback.  
- **SIMD Acceleration**: custom `wide::f32x8` kernels for dot/L2, with `matrixmultiply::sgemm` for block GEMM.  
- **Memory Mapping**: compact binary format (header + float data) using `memmap2` for zero-copy loading.  
- **Streaming Index Build**: chunked writes, runtime memory estimation, safe normalization, and atomic file finalization.  
- **Cross-Platform Optimizations**: runtime SIMD detection (AVX2/AVX/SSE4.2/NEON) and environment-controlled mode overrides (`NSEEK_THREADS`, `NSEEK_SINGLE_MODE`, `NSEEK_BATCH_MODE`).  


## Installation

```bash
# From PyPI
pip install nseekfs

# Verify installation
python -c "import nseekfs; print('NSeekFS installed successfully')"
```

## Technical Details

- **Similarity Metrics**: Cosine (with enforced normalization), Dot Product, Euclidean (−L2²).  
- **Result Layers**: `simple` for low overhead, `detailed` for margins/audit, plus replay support for deterministic validation.  
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

## Competitive Benchmarking

Use the competitive benchmark to compare NSeekFS with similar tools on the same data:

```bash
python bench/benchmark_competitors.py --rows 100000 --dims 384 --queries 200 --top-k 10 --metric cosine
```

Notes:

- Ground truth is exact brute force from NSeekFS.
- FAISS/HNSWlib are optional; script reports unavailable libraries gracefully.
- Report includes latency and recall@k vs exact ground truth.

Run the standardized scenario suite:

```bash
python bench/run_competitive_suite.py --scenarios bench/competitive_scenarios.json --out-dir benchmark_artifacts
```

Publication template and rules: `docs/COMPETITIVE_BENCHMARKS.md`

## Product Positioning

NSeekFS is optimized for exact, deterministic, auditable retrieval workflows.  
Positioning details: `docs/POSITIONING.md`

## License

MIT License - see LICENSE file for details.

---

**Fast, exact vector similarity search for Python.**  

*Built with Rust for performance, designed for Python developers.*  
Source: [github.com/NSeek-AI/nseekfs](https://github.com/NSeek-AI/nseekfs)
