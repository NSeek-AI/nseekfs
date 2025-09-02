# NSeekFS

[![PyPI version](https://badge.fury.io/py/nseekfs.svg)](https://badge.fury.io/py/nseekfs)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**High-Performance Vector Similarity Search with Rust Backend**

Fast and exact cosine similarity search for Python. Built with Rust for performance, designed for production use.

```bash
pip install nseekfs
```

## Quick Start

```python
import nseekfs
import numpy as np

# Create test vectors
embeddings = np.random.randn(10000, 384).astype(np.float32)
query = np.random.randn(384).astype(np.float32)

# Build index and search
index = nseekfs.from_embeddings(embeddings, normalized=True)
results = index.query(query, top_k=10)

# Results is a list of dictionaries
print(f"Found {len(results)} results")
print(f"Best match: vector {results[0]['idx']} (similarity: {results[0]['score']:.3f})")
```

## Core Features

### Exact Vector Search
```python
# Basic search
index = nseekfs.from_embeddings(embeddings, normalized=True)
results = index.query(query, top_k=10)

# Access results
for item in results:
    print(f"Vector {item['idx']}: {item['score']:.6f}")

# Index properties
print(f"Index contains {index.rows} vectors x {index.dims} dimensions")
```

### Batch Processing
```python
# Process multiple queries efficiently
queries = np.random.randn(50, 384).astype(np.float32)
batch_results = index.query_batch(queries, top_k=5)

print(f"Processed {len(queries)} queries")
for i, query_results in enumerate(batch_results):
    print(f"Query {i}: {len(query_results)} results")
    # Each query_results is a list of {'idx': int, 'score': float}
```

### Advanced Query Options
```python
# Simple format (default) - returns list of dicts
results = index.query(query, top_k=10, format='simple')

# Detailed format - returns QueryResult object with timing
result_obj = index.query(query, top_k=10, format='detailed')
print(f"Query took {result_obj.query_time_ms:.2f}ms")

# With timing tuple
results, timing = index.query(query, top_k=10, return_timing=True)
```

### Index Persistence
```python
# Load existing index
index = nseekfs.from_bin("my_vectors.nseek")
print(f"Loaded index with {index.rows} vectors x {index.dims} dimensions")
```

### Performance Monitoring
```python
# Get detailed performance metrics
metrics = index.get_performance_metrics()
print(f"Total queries: {metrics['total_queries']}")
print(f"Average time: {metrics['avg_query_time_ms']:.2f}ms")
print(f"SIMD queries: {metrics['simd_queries']}")
print(f"Queries per second: {metrics['queries_per_second']:.0f}")
```

### Built-in Benchmark
```python
# Run performance benchmark
nseekfs.benchmark(vectors=1000, dims=384, queries=100, verbose=True)
```

## API Reference

### Index Creation
```python
# Basic usage
index = nseekfs.from_embeddings(
    embeddings,        # numpy array of float32 vectors
    normalized=True,   # normalize vectors (default: False)
    verbose=False      # show progress (default: False)
)

# Load existing index
index = nseekfs.from_bin("path/to/index.nseek")
```

### Query Methods
```python
# Simple query (returns list of dicts)
results = index.query(query_vector, top_k=10)
# Returns: [{'idx': int, 'score': float}, ...]

# Detailed query (returns QueryResult object)
result = index.query_detailed(query_vector, top_k=10)

# Simple query explicitly
results = index.query_simple(query_vector, top_k=10)

# Batch queries
batch_results = index.query_batch(queries_array, top_k=10)
# Returns: List of lists of dicts
```

### Index Properties
```python
print(f"Vectors: {index.rows}")
print(f"Dimensions: {index.dims}")
print(f"Index path: {index.index_path}")
print(f"Config: {index.config}")
```

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
index = nseekfs.from_embeddings(embeddings, normalized=False)

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