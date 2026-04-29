"""
NSeekFS v1.0 - Exact Vector Search
==================================

Quick usage:
    import nseekfs
    index = nseekfs.from_embeddings(vectors, normalized=True, verbose=False)
    results = index.query(query_vector, top_k=10)
"""

import sys
import warnings
from importlib import import_module
from typing import Any, Dict, List, Optional, Union

__version__ = "1.1.0"
__author__ = "Diogo Novo"
__email__ = "contact@nseek.io"
__license__ = "MIT"
__description__ = "Exact vector similarity search with Rust backend"
__url__ = "https://github.com/NSeek-AI/nseekfs"

__version_info__ = tuple(map(int, __version__.split('.')))


def _check_compatibility():
    """Check Python version"""
    if sys.version_info < (3, 8):
        raise RuntimeError(f"Python 3.8+ required, got {sys.version_info.major}.{sys.version_info.minor}")


def _hl():
    """Import high-level module"""
    try:
        return import_module("nseekfs.highlevel")
    except ImportError as e:
        raise ImportError(f"Failed to import nseekfs.highlevel: {e}")


def _ll():
    """Import low-level module"""
    try:
        return import_module("nseekfs.nseekfs")
    except ImportError as e:
        raise ImportError(f"Failed to import nseekfs.nseekfs: {e}")


_check_compatibility()


def from_embeddings(vectors, metric="cosine", normalized=True, verbose=False):
    """Create index from embeddings"""
    return _hl().from_embeddings(vectors, metric=metric, normalized=normalized, verbose=verbose)


def from_bin(bin_file_path, verbose=False):
    """Load index from binary file"""
    return _hl().load_index(bin_file_path, verbose=verbose)


def health_check(quick=True, verbose=True):
    """Run system health check"""
    return _ll().health_check(quick=quick, verbose=verbose)


def benchmark(vectors=1000, dims=384, queries=100, verbose=True):
    """Run a benchmark with synthetic data"""
    import numpy as np
    import time

    print(f"NSeekFS Benchmark: {vectors} vectors, {dims}D, {queries} queries")

    data_start = time.time()
    test_vectors = np.random.randn(vectors, dims).astype(np.float32)
    test_queries = np.random.randn(queries, dims).astype(np.float32)
    test_vectors /= np.linalg.norm(test_vectors, axis=1, keepdims=True)
    test_queries /= np.linalg.norm(test_queries, axis=1, keepdims=True)
    data_gen_time = time.time() - data_start

    build_start = time.time()
    index = from_embeddings(test_vectors, verbose=verbose)
    build_time = time.time() - build_start

    query_times = []
    for i in range(queries):
        query_start = time.time()
        _ = index.query(test_queries[i], top_k=10)
        query_time = time.time() - query_start
        query_times.append(query_time)

    avg_query_time = np.mean(query_times)

    try:
        import psutil
        memory_usage = psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        memory_usage = 0.0

    results = {
        "data_generation_time_s": data_gen_time,
        "index_build_time_s": build_time,
        "avg_query_time_ms": avg_query_time * 1000,
        "queries_per_second": 1.0 / avg_query_time,
        "vectors_per_second": vectors / build_time,
        "memory_usage_mb": memory_usage,
    }

    if verbose:
        print(" Benchmark Results:")
        print(f"   Index creation: {build_time:.2f}s")
        print(f"   Avg query time: {avg_query_time * 1000:.2f}ms")
        print(f"   Queries/sec: {results['queries_per_second']:.0f}")
        print(f"   Memory usage: {memory_usage:.1f}MB")

    return results


def export_audit(result, path):
    """Export audit.json from a detailed result"""
    return _hl().export_audit(result, path)


def replay(audit_path):
    """Replay a query from audit.json and validate determinism"""
    return _hl().replay(audit_path)


def query_exact_certified(
    index,
    query_vector,
    top_k=10,
    block_size=64,
    enable_pruning=True,
    return_certificate=True,
    strict_query_normalized=False,
):
    """Run certified exact query with deterministic pruning"""
    if not hasattr(index, "query_exact_certified"):
        raise TypeError("index must be a SearchEngine instance")
    return index.query_exact_certified(
        query_vector,
        top_k=top_k,
        block_size=block_size,
        enable_pruning=enable_pruning,
        return_certificate=return_certificate,
        strict_query_normalized=strict_query_normalized,
    )


def query_batch_exact_certified(
    index,
    queries,
    top_k=10,
    block_size=64,
    enable_pruning=True,
    return_certificate=True,
):
    """Run certified exact query in batch mode"""
    if not hasattr(index, "query_batch_exact_certified"):
        raise TypeError("index must be a SearchEngine instance")
    return index.query_batch_exact_certified(
        queries,
        top_k=top_k,
        block_size=block_size,
        enable_pruning=enable_pruning,
        return_certificate=return_certificate,
    )


def _get_search_engine_class():
    return _hl().SearchEngine


def _get_search_config_class():
    return _hl().SearchConfig


def _get_pysearchengine_class():
    return _ll().PySearchEngine


class _LazyClass:
    def __init__(self, loader_func):
        self._loader = loader_func
        self._loaded = None

    def __call__(self, *args, **kwargs):
        if self._loaded is None:
            self._loaded = self._loader()
        return self._loaded(*args, **kwargs)

    def __getattr__(self, name):
        if self._loaded is None:
            self._loaded = self._loader()
        return getattr(self._loaded, name)


SearchEngine = _LazyClass(_get_search_engine_class)
SearchConfig = _LazyClass(_get_search_config_class)
PySearchEngine = _LazyClass(_get_pysearchengine_class)

build_from_embeddings = from_embeddings
build_index = from_embeddings
from_vectors = from_embeddings
fit_embeddings = from_embeddings
build_from_bin = from_bin
load_index = from_bin
read_index = from_bin

if hasattr(sys, 'ps1') or hasattr(sys, 'ps2'):
    try:
        _hl()
        print(f"NSeekFS v{__version__} loaded")
        print("   Quick start: nseekfs.from_embeddings(vectors)")
        print("   Simple query: index.query(vector, top_k=10)")
    except ImportError as e:
        print(f"NSeekFS v{__version__} loaded with issues:")
        print(f"   Error: {e}")


__all__ = [
    'from_embeddings',
    'from_bin',
    'health_check',
    'benchmark',
    'export_audit',
    'replay',
    'query_exact_certified',
    'query_batch_exact_certified',
    'SearchEngine',
    'SearchConfig',
    'PySearchEngine',
    'build_from_embeddings',
    'build_index',
    'from_vectors',
    'fit_embeddings',
    'build_from_bin',
    'load_index',
    'read_index',
    '__version__',
    '__version_info__',
    '__author__',
    '__email__',
    '__license__',
    '__description__',
    '__url__',
]
