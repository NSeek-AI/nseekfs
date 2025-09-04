#!/usr/bin/env python3
"""
NSeekFS v1.0 - High-Level API
=============================

Vector search simplified API.
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
    raise ImportError("NSeekFS Rust extension not found. Please install with: pip install nseekfs")

__version__ = "1.0.0"

@dataclass
class SearchConfig:
    """Configuration for the search engine."""
    metric: str = "cosine"
    normalized: bool = True
    verbose: bool = False
    enable_metrics: bool = False

@dataclass 
class QueryResult:
    """Search result with timing and diagnostics."""
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
    """Vector search engine wrapper over the Rust core."""
    
    def __init__(self, index_path: Union[str, Path], config: Optional[SearchConfig] = None):
        self.index_path = Path(index_path)
        self.config = config or SearchConfig()
        
        if not self.index_path.exists():
            raise FileNotFoundError(f"Index file not found: {self.index_path}")
        
        # Load Rust engine
        start_time = time.time()
        self._engine = rust_engine.PySearchEngine(str(self.index_path), ann=False)
        load_time = time.time() - start_time
        
        if self.config.verbose:
            print(f"NSeekFS engine loaded in {load_time:.3f}s")
            print(f"Index: {self.rows:,} vectors × {self.dims} dimensions")
    
    @property
    def dims(self) -> int:
        """Number of dimensions of the indexed vectors."""
        return self._engine.dims()
    
    @property  
    def rows(self) -> int:
        """Number of vectors stored in the index."""
        return self._engine.rows()
    
    def query(self, 
              query_vector: np.ndarray,
              top_k: int = 10,
              format: str = "simple",
              return_timing: bool = False) -> Union[List[Dict], QueryResult, Tuple]:
        """Simplified API: search for the top_k most similar vectors.
        
        Parameters
        ----------
        query_vector : np.ndarray
            1D float32 vector with length == self.dims.
        top_k : int
            Number of neighbors to return (clamped to number of rows).
        format : {"simple","detailed","legacy"}
            Output format. "simple" returns a list of {idx, score}.
            "detailed"/"legacy" returns a QueryResult object.
        return_timing : bool
            If True, also returns a dict with timing flags.
        """
        
        # Validation and dtype enforcement
        if not isinstance(query_vector, np.ndarray):
            query_vector = np.asarray(query_vector, dtype=np.float32)
        
        if query_vector.dtype != np.float32:
            query_vector = query_vector.astype(np.float32, copy=False)
        
        if query_vector.ndim != 1:
            raise ValueError("Query vector must be 1D")
        
        if len(query_vector) != self.dims:
            raise ValueError(f"Query vector dimensions {len(query_vector)} != index dimensions {self.dims}")
        
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        
        if top_k > self.rows:
            top_k = self.rows
        
        # Execute exact query
        start_time = time.time()
        
        try:
            rr = self._engine.query_exact(query_vector, int(top_k))
            query_time = (time.time() - start_time) * 1000.0

            # Extract results and diagnostics
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

            # Default simplified format
            if format == "simple":
                if return_timing:
                    return results_py, {"query_time_ms": query_time, "simd_used": simd_used}
                else:
                    return results_py
            
            # Detailed format
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

            if format == "detailed" or format == "legacy":
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
        """Shortcut: simple query without metrics."""
        return self.query(query_vector, top_k, format="simple")
    
    def query_detailed(self, query_vector: np.ndarray, top_k: int = 10) -> QueryResult:
        """Shortcut: query with full metrics."""
        return self.query(query_vector, top_k, format="detailed")
    
    def query_batch(self, queries: np.ndarray, top_k: int = 10, format: str = "simple") -> List:
        """Batch querying convenience method (iterates per row)."""
        if not isinstance(queries, np.ndarray):
            queries = np.asarray(queries, dtype=np.float32)
            
        if queries.dtype != np.float32:
            queries = queries.astype(np.float32, copy=False)
            
        if queries.ndim != 2:
            raise ValueError("Queries must be 2D array (N × dims)")
            
        if queries.shape[1] != self.dims:
            raise ValueError(f"Query dimensions {queries.shape[1]} != index dimensions {self.dims}")
        
        # Fallback: process individually
        results = []
        for i in range(queries.shape[0]):
            result = self.query(queries[i], top_k, format=format)
            results.append(result)
        return results
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Return aggregated performance metrics if available."""
        try:
            return self._engine.get_performance_metrics()
        except AttributeError:
            return {
                'total_queries': 0,
                'avg_query_time_ms': 0.0,
                'simd_queries': 0,
                'scalar_queries': 0,
                'queries_per_second': 0.0
            }
    
    def __repr__(self) -> str:
        if self.config.verbose:
            return f"SearchEngine(path='{self.index_path}', vectors={self.rows:,}, dims={self.dims})"
        else:
            return f"SearchEngine({self.rows:,} vectors × {self.dims}D)"


def from_embeddings(embeddings: np.ndarray,
                    metric: str = "cosine",
                    base_name: str = "nseekfs_index",
                    output_dir: Optional[str] = None,
                    normalized: bool = False,
                    config: Optional[SearchConfig] = None,
                    verbose: bool = False) -> SearchEngine:
    """Create a search index from a 2D embeddings array and return a loaded engine."""
    
    if output_dir is None:
        output_dir = os.getcwd()
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate and convert embeddings
    if not isinstance(embeddings, np.ndarray):
        embeddings = np.asarray(embeddings, dtype=np.float32)
    
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32, copy=False)
    
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be 2D array")
    
    rows, dims = embeddings.shape
    
    if rows == 0 or dims == 0:
        raise ValueError("Embeddings cannot be empty")
    
    # Configuration (kept for future extensibility)
    if config is None:
        config = SearchConfig(metric=metric, normalized=normalized, verbose=verbose)
    
    # Prepare binary file using the Rust function
    try:
        from nseekfs.nseekfs import py_prepare_bin_from_embeddings
    except ImportError:
        raise RuntimeError("Rust engine not available. Please ensure NSeekFS is properly compiled.")
    
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
            False,              # ann flag (always False for exact search)
            None,               # seed (optional)
        )
        
        if verbose:
            creation_time = time.time() - start_time
            print(f"Index created in {creation_time:.2f}s")
            print(f"Saved to: {result_path}")
        
        return SearchEngine(result_path, config)
        
    except Exception as e:
        raise RuntimeError(f"Failed to create index: {e}")


def load_index(index_path: Union[str, Path], 
               config: Optional[SearchConfig] = None,
               verbose: bool = False) -> SearchEngine:
    """Load an existing index from disk."""
    if config is None:
        config = SearchConfig(verbose=verbose)
    
    return SearchEngine(index_path, config)


# Compatibility aliases for downstream code
ValidationError = ValueError
IndexError = Exception
