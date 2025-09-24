#!/usr/bin/env python3
"""
Quick Test for nseekfs
======================

Run this script after installing nseekfs:
    python quick_example.py

It will test:
1. Index creation with cosine metric
2. Simple query
3. Batch queries
4. Detailed query
5. Index persistence
6. Performance metrics
7. Built-in benchmark

Note:
- Default metric is "cosine" (requires normalized vectors).
- You can also use "dot" or "euclidean" as metric, in that case
  do NOT normalize the embeddings or queries.
"""

import numpy as np
import nseekfs


def main():
    print("Testing NSeekFS...")

    # Example with cosine similarity (normalize vectors)
    metric = "cosine"  # or "dot" / "euclidean" (without normalization)
    embeddings = np.random.randn(5000, 384).astype(np.float32)
    query = np.random.randn(384).astype(np.float32)

    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    query = query / np.linalg.norm(query)

    index = nseekfs.from_embeddings(embeddings, metric=metric, normalized=True)
    print(f"Index built: {index.rows} vectors x {index.dims} dims (metric={metric})")

    # Simple query
    results = index.query(query, top_k=5)
    print("Simple query:", results[:3])

    # Batch queries
    queries = np.random.randn(10, 384).astype(np.float32)
    queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    batch_results = index.query_batch(queries, top_k=3)
    print(f"Batch queries: processed {len(batch_results)} queries")

    # Detailed query
    detailed = index.query_detailed(query, top_k=5)
    print(f"Detailed query took {detailed.query_time_ms:.2f} ms")

    # Index persistence
    path = index.index_path
    reloaded = nseekfs.from_bin(path) # Index persistence (reloaded from the .bin file)
    print(f"Reloaded index: {reloaded.rows} vectors")

    # Performance metrics
    metrics = index.get_performance_metrics()
    print(f"Metrics: {metrics}")


if __name__ == "__main__":
    main()
