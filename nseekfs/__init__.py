"""
NSeekFS v1.0 - High-Performance Exact Vector Search
==================================================

v1.0 Features:
- Simplified API (clean and efficient)
- Built-in verbose logging and timing

Simple Usage:
    import nseekfs
    
    # Simple API - CORRECTED
    index = nseekfs.from_embeddings(vectors, normalized=True, verbose=False)
    results = index.query(query_vector, top_k=10)
    # Returns: [{'idx': 0, 'score': 0.95}, ...]

Verbose Mode:
    # Enable detailed logging - CORRECTED
    index = nseekfs.from_embeddings(vectors, normalized=True, verbose=True)
    results = index.query(query_vector, top_k=10)  # Shows detailed logs

Control Normalization:
    # Disable normalization if vectors are already normalized
    index = nseekfs.from_embeddings(vectors, normalized=False, verbose=False)
"""

import sys
import warnings
from importlib import import_module
from typing import Any, Dict, List, Optional, Union


__version__ = "1.0.4"
__author__ = "Diogo Novo"
__email__ = "contact@nseek.io"
__license__ = "MIT"
__description__ = "High-performance exact vector similarity search with Rust backend"
__url__ = "https://github.com/NSeek-AI/nseekfs"


__version_info__ = tuple(map(int, __version__.split('.')))


# Version compatibility check
def _check_compatibility():
    """Check system compatibility and warn about changes"""
    if sys.version_info < (3, 8):
        raise RuntimeError(f"Python 3.8+ required, got {sys.version_info.major}.{sys.version_info.minor}")


def _hl():
    """Lazy import of highlevel module with error handling"""
    try:
        return import_module("nseekfs.highlevel")
    except ImportError as e:
        raise ImportError(f"Failed to import nseekfs.highlevel: {e}. Please ensure NSeekFS is properly installed.")

def _ll():
    """Lazy import of low-level module with error handling"""
    try:
        return import_module("nseekfs.nseekfs")
    except ImportError as e:
        raise ImportError(f"Failed to import nseekfs.nseekfs: {e}. Please ensure the Rust extension is compiled.")


# Check compatibility at import
_check_compatibility()


# MAIN API - The essentials
def from_embeddings(vectors, normalized=True, verbose=False):
    """Create index from embeddings - main entry point"""
    return _hl().from_embeddings(vectors, normalized=normalized, verbose=verbose)


def from_bin(bin_file_path, verbose=False):
    """Load index from binary file"""
    return _hl().from_bin(bin_file_path, verbose=verbose)


def health_check(quick=True, verbose=True):
    """Run system health check"""
    return _hl().health_check(quick=quick, verbose=verbose)


def benchmark(vectors=1000, dims=384, queries=100, verbose=True):
    """Run performance benchmark"""
    import numpy as np
    import time
    
    print(f"NSeekFS Benchmark: {vectors} vectors, {dims}D, {queries} queries")
    
    # Generate test data
    data_start = time.time()
    test_vectors = np.random.randn(vectors, dims).astype(np.float32)
    test_queries = np.random.randn(queries, dims).astype(np.float32)
    data_gen_time = time.time() - data_start
    
    # Build index
    build_start = time.time()
    index = from_embeddings(test_vectors, verbose=verbose)
    build_time = time.time() - build_start
    
    # Query benchmark
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
    
    benchmark_results = {
        "data_generation_time_s": data_gen_time,
        "index_build_time_s": build_time,
        "avg_query_time_ms": avg_query_time * 1000,
        "queries_per_second": 1.0 / avg_query_time,
        "vectors_per_second": vectors / build_time,
        "memory_usage_mb": memory_usage,
    }
    
    if verbose:
        print(f" Benchmark Results:")
        print(f"   Index creation: {build_time:.2f}s")
        print(f"   Avg query time: {avg_query_time * 1000:.2f}ms")
        print(f"   Queries/sec: {benchmark_results['queries_per_second']:.0f}")
        print(f"   Memory usage: {memory_usage:.1f}MB")
    
    return benchmark_results


# LAZY LOADING: Advanced classes for when needed
def _get_search_engine_class():
    """Get SearchEngine class (lazy)"""
    return _hl().SearchEngine

def _get_search_config_class():
    """Get SearchConfig class (lazy)"""
    return _hl().SearchConfig

def _get_pysearchengine_class():
    """Get PySearchEngine class (lazy)"""
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


# Advanced classes (lazy loaded)
SearchEngine = _LazyClass(_get_search_engine_class)
SearchConfig = _LazyClass(_get_search_config_class)
PySearchEngine = _LazyClass(_get_pysearchengine_class)

# Compatibility aliases
build_from_embeddings = from_embeddings  # v1.0 compat
build_from_bin = from_bin  # v1.0 compat

# Show version info in interactive sessions
if hasattr(sys, 'ps1') or hasattr(sys, 'ps2'):
    try:
        # Test basic import
        _hl()
        print(f" NSeekFS v{__version__} loaded successfully")
        print(f"    Quick start: nseekfs.from_embeddings(vectors)")
        print(f"    Simple query: index.query(vector, top_k=10)")
        print(f"    Health check: nseekfs.health_check()")
    except ImportError as e:
        print(f"  NSeekFS v{__version__} loaded with issues:")
        print(f"   Error: {e}")
        print(f"   Run nseekfs.health_check() for diagnosis")


# CLEAN API: Only essential exports
__all__ = [
    # Core functions
    'from_embeddings',
    'from_bin', 
    'health_check',
    'benchmark',
    
    # Advanced classes (lazy loaded)
    'SearchEngine',
    'SearchConfig', 
    'PySearchEngine',
    
    # Compatibility
    'build_from_embeddings',
    'build_from_bin',
    
    # Metadata
    '__version__',
    '__version_info__',
    '__author__',
    '__email__',
    '__license__',
    '__description__',
    '__url__',
]