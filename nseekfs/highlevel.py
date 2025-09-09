#!/usr/bin/env python3
"""
NSeekFS v1.0 — High-level interface

Thin, predictable wrapper around the Rust core that provides an
easy API for exact vector search (build, load, and query).
"""

import os
import sys
import time
import warnings
from pathlib import Path
from typing import Union, List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass

try:
    import nseekfs.nseekfs as rust_engine
except ImportError:
    raise ImportError("NSeekFS Rust extension not found. Install with: pip install nseekfs")

__version__ = "1.0.3"


@dataclass
class SearchConfig:
    """
    Runtime configuration for the search engine.
    """
    metric: str = "cosine"
    normalized: bool = True
    verbose: bool = False
    enable_metrics: bool = False


@dataclass
class QueryResult:
    """
    Detailed result for a single query, including timings and flags.
    """
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
    """
    Loaded index ready for exact nearest-neighbour queries.
    """

    def __init__(self, index_path: Union[str, Path], config: Optional[SearchConfig] = None):
        self.index_path = Path(index_path)
        self.config = config or SearchConfig()

        if not self.index_path.exists():
            raise FileNotFoundError(f"Index file not found: {self.index_path}")

        # Load the Rust core
        start_time = time.time()
        self._engine = rust_engine.PySearchEngine(str(self.index_path), ann=False)
        load_time = time.time() - start_time

        if self.config.verbose:
            print(f"NSeekFS engine loaded in {load_time:.3f}s")
            print(f"Index: {self.rows:,} vectors × {self.dims} dimensions")

    @property
    def dims(self) -> int:
        """Vector dimensionality of the index."""
        return self._engine.dims()

    @property
    def rows(self) -> int:
        """Number of vectors stored in the index."""
        return self._engine.rows()

    def query(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        format: str = "simple",
        return_timing: bool = False
    ) -> Union[List[Dict], QueryResult, Tuple]:
        """
        Run an exact search for the top_k most similar vectors.

        Parameters
        ----------
        query_vector : np.ndarray
            1D float32 array with length == self.dims.
        top_k : int
            Number of neighbours to return (clamped to total rows).
        format : {"simple", "detailed"}
            - "simple": list of dicts: {"idx": int, "score": float}
            - "detailed": QueryResult with timings and flags
        return_timing : bool
            If True and format=="simple", also return a small timing dict.

        Returns
        -------
        simple:
            List[{"idx": int, "score": float}] or (list, timing_dict)
        detailed:
            QueryResult or (QueryResult, timing_dict)
        """
        # Basic validation and dtype
        if not isinstance(query_vector, np.ndarray):
            query_vector = np.asarray(query_vector, dtype=np.float32)

        if query_vector.dtype != np.float32:
            query_vector = query_vector.astype(np.float32, copy=False)

        if query_vector.ndim != 1:
            raise ValueError("Query vector must be 1D")

        if len(query_vector) != self.dims:
            raise ValueError(f"Query dimensions {len(query_vector)} != index dimensions {self.dims}")

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        if top_k > self.rows:
            top_k = self.rows

        start_time = time.time()
        try:
            rr = self._engine.query_exact(query_vector, int(top_k))
            query_time = (time.time() - start_time) * 1000.0

            # Convert to Python structures
            results_py = []
            method_used = getattr(rr, "method_used", "exact")
            candidates_generated = getattr(rr, "candidates_generated", 0)
            simd_used = bool(getattr(rr, "simd_used", False))
            parse_time_ms = float(getattr(rr, "parse_time_ms", 0.0))
            compute_time_ms = float(getattr(rr, "compute_time_ms", 0.0))
            sort_time_ms = float(getattr(rr, "sort_time_ms", 0.0))

            if hasattr(rr, "results"):
                for it in rr.results:
                    idx = getattr(it, "idx", None)
                    score = getattr(it, "score", None)
                    if idx is not None and score is not None:
                        results_py.append({"idx": int(idx), "score": float(score)})

            if format == "simple":
                if return_timing:
                    return results_py, {"query_time_ms": query_time, "simd_used": simd_used}
                return results_py

            # Detailed result
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

        except Exception as e:
            raise RuntimeError(f"Query failed: {e}")

    def query_simple(self, query_vector: np.ndarray, top_k: int = 10) -> List[Dict]:
        """Shorthand for query(..., format='simple')."""
        return self.query(query_vector, top_k, format="simple")

    def query_detailed(self, query_vector: np.ndarray, top_k: int = 10) -> QueryResult:
        """Shorthand for query(..., format='detailed')."""
        return self.query(query_vector, top_k, format="detailed")

    def query_batch(self, queries: np.ndarray, top_k: int = 10, format: str = "simple") -> List:
        """
        Batch exact search. The Rust core picks the best strategy internally.

        Parameters
        ----------
        queries : np.ndarray
            2D float32 array of shape (N, dims).
        top_k : int
            Neighbours per query.
        format : {"simple", "detailed"}
            Output layout for each query in the batch.

        Returns
        -------
        simple:
            List[List[{"idx": int, "score": float}]]
        detailed:
            List[Dict] with per-query timings and flags.
        """
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

        rust_results = self._engine.query_batch(queries, top_k)

        if format == "simple":
            return [
                [{"idx": item.idx, "score": item.score} for item in result.results]
                for result in rust_results
            ]

        if format == "detailed":
            return [
                {
                    "results": [{"idx": item.idx, "score": item.score} for item in result.results],
                    "query_time_ms": result.query_time_ms,
                    "method_used": result.method_used,
                    "candidates_examined": result.candidates_generated,
                    "simd_used": result.simd_used,
                    "parse_time_ms": result.parse_time_ms,
                    "compute_time_ms": result.compute_time_ms,
                    "sort_time_ms": result.sort_time_ms,
                }
                for result in rust_results
            ]

        raise ValueError("Unknown format. Use 'simple' or 'detailed'.")

    def get_batch_performance_summary(self) -> Dict[str, Any]:
        """
        Return aggregated metrics for batch processing if exposed by the core.
        """
        try:
            return self._engine.get_performance_metrics()
        except (AttributeError, Exception):
            return {
                "message": "Batch performance metrics not available",
                "suggestion": "Update to the latest NSeekFS version for detailed metrics",
            }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Return engine-wide counters and averages, if available.
        """
        try:
            return self._engine.get_performance_metrics()
        except AttributeError:
            return {
                "total_queries": 0,
                "avg_query_time_ms": 0.0,
                "simd_queries": 0,
                "scalar_queries": 0,
                "queries_per_second": 0.0,
            }

    def __repr__(self) -> str:
        if self.config.verbose:
            return f"SearchEngine(path='{self.index_path}', vectors={self.rows:,}, dims={self.dims})"
        return f"SearchEngine({self.rows:,} vectors × {self.dims}D)"


def from_embeddings(
    embeddings: np.ndarray,
    metric: str = "cosine",
    base_name: str = "nseekfs_index",
    output_dir: Optional[str] = None,
    normalized: bool = True,
    config: Optional[SearchConfig] = None,
    verbose: bool = False
) -> SearchEngine:
    """
    Build an index from a 2D array of embeddings and return a ready-to-query engine.

    Parameters
    ----------
    embeddings : np.ndarray
        2D float32 array (rows × dims).
    metric : str
        Similarity metric. Currently "cosine".
    base_name : str
        Base file name for the generated index.
    output_dir : str | Path | None
        Directory where the index file will be written. Defaults to CWD.
    normalized : bool
        Set True if embeddings are already L2-normalized. If False, the core will normalize.
    config : SearchConfig | None
        Optional runtime options for the loaded engine.
    verbose : bool
        Print build timings and paths.

    Returns
    -------
    SearchEngine
        Loaded engine bound to the newly created index.
    """
    if output_dir is None:
        output_dir = os.getcwd()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure ndarray/float32/2D
    if not isinstance(embeddings, np.ndarray):
        embeddings = np.asarray(embeddings, dtype=np.float32)

    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32, copy=False)

    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array")

    rows, dims = embeddings.shape
    if rows == 0 or dims == 0:
        raise ValueError("Embeddings cannot be empty")

    if config is None:
        config = SearchConfig(metric=metric, normalized=normalized, verbose=verbose)

    try:
        from nseekfs.nseekfs import py_prepare_bin_from_embeddings
    except ImportError:
        raise RuntimeError("Rust engine not available. Ensure NSeekFS is compiled and installed.")

    if verbose:
        print(f"Creating index for {rows:,} vectors × {dims}D...")
        start_time = time.time()

    try:
        result_path = py_prepare_bin_from_embeddings(
            embeddings,         # numpy array
            dims,               # dimensions
            rows,               # number of vectors
            base_name,          # base name
            str(output_dir),    # output directory
            "f32",              # precision level
            normalized,         # normalization flag
            False,              # ANN disabled (exact search)
            None,               # seed (unused)
        )

        if verbose:
            creation_time = time.time() - start_time
            print(f"Index created in {creation_time:.2f}s")
            print(f"Saved to: {result_path}")

        return SearchEngine(result_path, config)

    except Exception as e:
        raise RuntimeError(f"Failed to create index: {e}")


def load_index(
    index_path: Union[str, Path],
    config: Optional[SearchConfig] = None,
    verbose: bool = False
) -> SearchEngine:
    """
    Load an existing index file and return a SearchEngine.
    """
    if config is None:
        config = SearchConfig(verbose=verbose)
    return SearchEngine(index_path, config)


# Compatibility aliases for downstream code
ValidationError = ValueError
IndexError = Exception
