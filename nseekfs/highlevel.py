#!/usr/bin/env python3
"""
NSeekFS v1.0 – High-level interface

Thin Python wrapper around the Rust core.
Provides a simple API for exact vector search with error handling and memory management.
"""

import os
import sys
import time
import warnings
import gc
import tempfile
from pathlib import Path
from typing import Union, List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import contextlib

try:
    import nseekfs.nseekfs as rust_engine
except ImportError:
    raise ImportError("NSeekFS Rust extension not found. Install with: pip install nseekfs")

__version__ = "1.0.7"

MAX_VECTORS_MEMORY_CHECK = 1_000_000
MAX_BATCH_SIZE = 10000
DEFAULT_CHUNK_SIZE = 5000


def get_memory_usage():
    """Return current process memory usage in MB"""
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


def estimate_memory_requirement(vectors: int, dims: int) -> float:
    """Estimate memory requirement in MB for a dataset"""
    base_size = vectors * dims * 4
    processing_overhead = base_size * 1.5
    return processing_overhead / (1024 * 1024)


def check_memory_availability(required_mb: float) -> bool:
    """Check if enough memory is available"""
    try:
        import psutil
        available_mb = psutil.virtual_memory().available / (1024 * 1024)
        return available_mb > required_mb * 1.5
    except ImportError:
        return required_mb < 16000


@dataclass
class SearchConfig:
    """Runtime configuration for the search engine"""
    metric: str = "cosine"
    normalized: bool = True
    verbose: bool = False
    enable_metrics: bool = False
    chunk_size: int = DEFAULT_CHUNK_SIZE


@dataclass
class QueryResult:
    """Result of a single query with optional timings"""
    results: List[Dict[str, Any]]
    query_time_ms: float
    method_used: str
    candidates_examined: int = 0
    simd_used: bool = False
    parse_time_ms: float = 0.0
    compute_time_ms: float = 0.0
    sort_time_ms: float = 0.0

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self):
        return iter(self.results)

    def __getitem__(self, key):
        return self.results[key]


class SearchEngine:
    """Loaded index ready for exact nearest-neighbour queries"""

    def __init__(self, index_path: Union[str, Path], config: Optional[SearchConfig] = None):
        self.index_path = Path(index_path)
        self.config = config or SearchConfig()
        self._engine = None
        self._initialized = False

        if not self.index_path.exists():
            raise FileNotFoundError(f"Index file not found: {self.index_path}")

        self._initialize_engine()

    def _initialize_engine(self):
        try:
            start_time = time.time()
            self._engine = rust_engine.PySearchEngine(str(self.index_path), ann=False)
            load_time = time.time() - start_time
            self._initialized = True

            if self.config.verbose:
                mem_usage = get_memory_usage()
                print(f"Engine loaded in {load_time:.3f}s")
                print(f"Index: {self.rows:,} vectors × {self.dims} dimensions")
                print(f"Memory usage: {mem_usage:.1f}MB")

        except Exception as e:
            raise RuntimeError(f"Failed to initialize search engine: {e}")

    @property
    def dims(self) -> int:
        if not self._initialized:
            raise RuntimeError("Engine not initialized")
        return self._engine.dims()

    @property
    def rows(self) -> int:
        if not self._initialized:
            raise RuntimeError("Engine not initialized")
        return self._engine.rows()

    def _validate_query_vector(self, query_vector: np.ndarray) -> np.ndarray:
        if not isinstance(query_vector, np.ndarray):
            query_vector = np.asarray(query_vector, dtype=np.float32)

        if query_vector.dtype != np.float32:
            query_vector = query_vector.astype(np.float32, copy=False)

        if query_vector.ndim != 1:
            raise ValueError("Query vector must be 1D")

        if len(query_vector) != self.dims:
            raise ValueError(f"Query dimensions {len(query_vector)} != index dimensions {self.dims}")

        if not np.all(np.isfinite(query_vector)):
            raise ValueError("Query vector contains non-finite values")

        return query_vector

    def query(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        format: str = "simple",
        return_timing: bool = False
    ) -> Union[List[Dict], QueryResult, Tuple]:
        if not self._initialized:
            raise RuntimeError("Engine not initialized")

        query_vector = self._validate_query_vector(query_vector)

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        effective_k = min(top_k, self.rows)

        start_time = time.time()
        try:
            rr = self._engine.query_exact(query_vector, int(effective_k))
            query_time = (time.time() - start_time) * 1000.0

            results_py = []
            method_used = getattr(rr, "method_used", "exact")
            candidates_generated = getattr(rr, "candidates_generated", 0)
            simd_used = bool(getattr(rr, "simd_used", False))
            parse_time_ms = float(getattr(rr, "parse_time_ms", 0.0))
            compute_time_ms = float(getattr(rr, "compute_time_ms", 0.0))
            sort_time_ms = float(getattr(rr, "sort_time_ms", 0.0))

            if hasattr(rr, "results"):
                for item in rr.results:
                    idx = getattr(item, "idx", None)
                    score = getattr(item, "score", None)
                    if idx is not None and score is not None:
                        if np.isfinite(score):
                            results_py.append({"idx": int(idx), "score": float(score)})

            if format == "simple":
                if return_timing:
                    return results_py, {"query_time_ms": query_time, "simd_used": simd_used}
                return results_py

            qr = QueryResult(
                results=results_py,
                query_time_ms=query_time,
                method_used=method_used,
                candidates_examined=candidates_generated,
                simd_used=simd_used,
                parse_time_ms=parse_time_ms,
                compute_time_ms=compute_time_ms,
                sort_time_ms=sort_time_ms,
            )

            if format == "detailed":
                if return_timing:
                    return qr, {
                        "query_time_ms": query_time,
                        "method_used": method_used,
                        "simd_used": simd_used,
                    }
                return qr

            raise ValueError(f"Unknown format '{format}'. Use 'simple' or 'detailed'")

        except Exception as e:
            error_msg = f"Query failed: {e}"
            if self.config.verbose:
                print(f"{error_msg}")
            raise RuntimeError(error_msg)

    def query_simple(self, query_vector: np.ndarray, top_k: int = 10) -> List[Dict]:
        return self.query(query_vector, top_k, format="simple")

    def query_detailed(self, query_vector: np.ndarray, top_k: int = 10) -> QueryResult:
        return self.query(query_vector, top_k, format="detailed")

    def query_batch(self, queries: np.ndarray, top_k: int = 10, format: str = "simple") -> List:
        if not self._initialized:
            raise RuntimeError("Engine not initialized")

        if not isinstance(queries, np.ndarray):
            queries = np.asarray(queries, dtype=np.float32)

        if queries.dtype != np.float32:
            queries = queries.astype(np.float32, copy=False)

        if queries.ndim != 2:
            raise ValueError("Queries must be a 2D array (N × dims)")

        if queries.shape[1] != self.dims:
            raise ValueError(f"Query dimensions {queries.shape[1]} != index dimensions {self.dims}")

        if queries.shape[0] == 0:
            return []

        if not np.all(np.isfinite(queries)):
            raise ValueError("Queries contain non-finite values")

        num_queries = queries.shape[0]

        should_chunk = False
        if num_queries > MAX_BATCH_SIZE:
            should_chunk = True
            reason = f"queries > {MAX_BATCH_SIZE}"
        else:
            current_mem = get_memory_usage()
            estimated_batch_mem = (num_queries * self.rows * 4) / (1024 * 1024)
            if estimated_batch_mem > 2000 and current_mem > 12000:
                should_chunk = True
                reason = f"estimated {estimated_batch_mem:.0f}MB + {current_mem:.0f}MB current"

        if should_chunk:
            if self.config.verbose:
                print(f"Chunking batch: {reason}")
            return self._query_batch_chunked(queries, top_k, format)

        old_env = {}
        performance_vars = {
            'NSEEK_QBLOCK': str(min(num_queries, 512)),
            'NSEEK_DBLOCK': str(min(self.rows, 32768)),
        }

        for key, value in performance_vars.items():
            old_env[key] = os.environ.get(key)
            os.environ[key] = value

        try:
            rust_results = self._engine.query_batch(queries, top_k)
            return self._process_batch_results(rust_results, format)
        except Exception as e:
            error_msg = f"Batch query failed: {e}"
            if self.config.verbose:
                print(f"{error_msg}, falling back to chunked processing")
            return self._query_batch_chunked(queries, top_k, format)
        finally:
            for key, old_value in old_env.items():
                if old_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = old_value

    def _query_batch_chunked(self, queries: np.ndarray, top_k: int, format: str) -> List:
        num_queries = queries.shape[0]

        if self.rows < 100_000:
            chunk_size = min(self.config.chunk_size * 2, num_queries)
        elif self.rows < 500_000:
            chunk_size = min(self.config.chunk_size, num_queries)
        else:
            chunk_size = min(self.config.chunk_size // 2, num_queries)

        chunk_size = max(chunk_size, min(100, num_queries))
        results = []

        if self.config.verbose:
            print(f"Processing {num_queries} queries in chunks of {chunk_size}")

        old_env = {}
        chunk_vars = {
            'NSEEK_QBLOCK': str(min(chunk_size, 256)),
            'NSEEK_DBLOCK': str(min(self.rows, 16384)),
        }

        for key, value in chunk_vars.items():
            old_env[key] = os.environ.get(key)
            os.environ[key] = value

        try:
            for i in range(0, num_queries, chunk_size):
                end_idx = min(i + chunk_size, num_queries)
                chunk = queries[i:end_idx]

                try:
                    chunk_results = self._engine.query_batch(chunk, top_k)
                    processed_chunk = self._process_batch_results(chunk_results, format)
                    results.extend(processed_chunk)

                    if i % (chunk_size * 10) == 0 and i > 0:
                        gc.collect()

                    if self.config.verbose and (i // chunk_size) % 20 == 0:
                        progress = min(100, (end_idx / num_queries) * 100)
                        mem_usage = get_memory_usage()
                        print(f"Progress: {progress:.1f}% (mem: {mem_usage:.1f}MB)")

                except Exception as e:
                    if self.config.verbose:
                        print(f"Chunk {i//chunk_size} failed: {e}")
                    chunk_size_actual = end_idx - i
                    empty_results = [[] if format == "simple" else {} for _ in range(chunk_size_actual)]
                    results.extend(empty_results)

        finally:
            for key, old_value in old_env.items():
                if old_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = old_value

        return results

    def _process_batch_results(self, rust_results, format: str) -> List:
        if format == "simple":
            return [
                [{"idx": item.idx, "score": item.score} for item in result.results
                 if hasattr(item, 'idx') and hasattr(item, 'score') and np.isfinite(item.score)]
                for result in rust_results
            ]

        if format == "detailed":
            processed = []
            for result in rust_results:
                valid_results = [
                    {"idx": item.idx, "score": item.score} for item in result.results
                    if hasattr(item, 'idx') and hasattr(item, 'score') and np.isfinite(item.score)
                ]

                processed.append({
                    "results": valid_results,
                    "query_time_ms": getattr(result, "query_time_ms", 0.0),
                    "method_used": getattr(result, "method_used", "unknown"),
                    "candidates_examined": getattr(result, "candidates_generated", 0),
                    "simd_used": getattr(result, "simd_used", False),
                    "parse_time_ms": getattr(result, "parse_time_ms", 0.0),
                    "compute_time_ms": getattr(result, "compute_time_ms", 0.0),
                    "sort_time_ms": getattr(result, "sort_time_ms", 0.0),
                })
            return processed

        raise ValueError("Unknown format. Use 'simple' or 'detailed'.")

    def get_performance_metrics(self) -> Dict[str, Any]:
        try:
            return self._engine.get_performance_metrics()
        except AttributeError:
            return {
                "total_queries": 0,
                "avg_query_time_ms": 0.0,
                "simd_queries": 0,
                "scalar_queries": 0,
                "queries_per_second": 0.0,
                "memory_usage_mb": get_memory_usage(),
            }

    def __repr__(self) -> str:
        if not self._initialized:
            return f"SearchEngine(uninitialized, path='{self.index_path}')"

        if self.config.verbose:
            mem_usage = get_memory_usage()
            return f"SearchEngine(path='{self.index_path}', vectors={self.rows:,}, dims={self.dims}, mem={mem_usage:.1f}MB)"
        return f"SearchEngine({self.rows:,} vectors × {self.dims}D)"

    def __del__(self):
        if hasattr(self, '_engine'):
            del self._engine
        gc.collect()


def from_embeddings(
    embeddings: np.ndarray,
    metric: str = "cosine",
    base_name: str = "nseekfs_index",
    output_dir: Optional[str] = None,
    normalized: bool = True,
    config: Optional[SearchConfig] = None,
    verbose: bool = False
) -> SearchEngine:
    import re
    try:
        from nseekfs.nseekfs import py_prepare_bin_from_embeddings
    except ImportError:
        raise RuntimeError("Rust engine not available. Make sure the package is installed.")

    x = np.asarray(embeddings, dtype=np.float32, order="C")
    if x.ndim != 2:
        raise ValueError("Embeddings must be 2D")
    rows, dims = x.shape

    if not np.all(np.isfinite(x)):
        raise ValueError("Embeddings contain NaN/Inf")

    clean_base_name = re.sub(r"[^\w\-_]", "_", base_name) or "nseekfs_index"

    if output_dir is None:
        output_dir = os.getcwd()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    normalize_flag = (metric == "cosine") and (not normalized)

    t0 = time.time()
    path = py_prepare_bin_from_embeddings(
        x,
        int(dims),
        int(rows),
        clean_base_name,
        "f32",
        bool(normalize_flag),
        False,
        0,
        metric,
        str(output_dir),
    )
    if verbose:
        print(f"Index created in {time.time()-t0:.2f}s → {path}")

    if config is None:
        config = SearchConfig(metric=metric, normalized=normalized, verbose=verbose)
    return SearchEngine(path, config)


def load_index(
    index_path: Union[str, Path],
    config: Optional[SearchConfig] = None,
    verbose: bool = False
) -> SearchEngine:
    index_path = Path(index_path)

    if not index_path.exists():
        raise FileNotFoundError(f"Index file not found: {index_path}")

    if not index_path.is_file():
        raise ValueError(f"Path is not a file: {index_path}")

    file_size = index_path.stat().st_size
    if file_size < 12:
        raise ValueError(f"Index file too small: {file_size} bytes")

    if config is None:
        config = SearchConfig(verbose=verbose)

    try:
        engine = SearchEngine(index_path, config)

        if verbose:
            print(f"Index loaded successfully")
            print(f"{engine.rows:,} vectors × {engine.dims} dimensions")

        return engine

    except Exception as e:
        raise RuntimeError(f"Failed to load index: {e}")


ValidationError = ValueError
IndexError = Exception


@contextlib.contextmanager
def temporary_index(embeddings: np.ndarray, **kwargs):
    """Context manager for creating temporary indices that are automatically cleaned"""
    with tempfile.TemporaryDirectory() as temp_dir:
        kwargs.setdefault('output_dir', temp_dir)
        kwargs.setdefault('base_name', 'temp_index')

        engine = from_embeddings(embeddings, **kwargs)
        try:
            yield engine
        finally:
            pass
