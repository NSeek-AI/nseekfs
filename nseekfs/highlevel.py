#!/usr/bin/env python3
"""NSeekFS v1.1 - High-level interface.

Compact compatibility wrapper around the Rust core.
It exposes the public Python surface used by the README and PyPI package.
"""

from __future__ import annotations

import base64
import contextlib
import gc
import json
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import nseekfs.nseekfs as rust_engine
except ImportError as exc:  # pragma: no cover - import-time guard
    raise ImportError("NSeekFS Rust extension not found. Install with: pip install nseekfs") from exc

__version__ = "1.1.0"

MAX_VECTORS_MEMORY_CHECK = 1_000_000
MAX_BATCH_SIZE = 10_000
DEFAULT_CHUNK_SIZE = 5_000
SCORE_EPS = 1e-6

_AUDIT_FIELDS = [
    "engine_version",
    "metric",
    "normalized",
    "dims",
    "rows",
    "index_path",
    "index_hash",
    "query_hash",
    "query_vector_bytes",
    "top_k",
]

_AUDIT_OPTIONAL_FIELDS = [
    "certificate",
    "block_order",
    "block_size",
    "pruning_count",
    "threshold_evolution",
    "simd_path",
    "compile_flags",
    "bound_kind",
    "pruned_bound_values",
    "cosine_contract",
    "query_norm",
]


def get_memory_usage() -> float:
    """Return current process memory usage in MB."""
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


def estimate_memory_requirement(vectors: int, dims: int) -> float:
    """Estimate memory requirement in MB for a dataset."""
    base_size = vectors * dims * 4
    processing_overhead = base_size * 1.5
    return processing_overhead / (1024 * 1024)


def check_memory_availability(required_mb: float) -> bool:
    """Check if enough memory is available."""
    try:
        import psutil

        available_mb = psutil.virtual_memory().available / (1024 * 1024)
        return available_mb > required_mb * 1.5
    except ImportError:
        return required_mb < 16_000


def _encode_query_vector_bytes(query_vector: np.ndarray) -> str:
    vec = np.asarray(query_vector, dtype="<f4", order="C")
    return base64.b64encode(vec.tobytes(order="C")).decode("ascii")


def _rank_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ranked: List[Dict[str, Any]] = []
    for rank, item in enumerate(results, start=1):
        ranked.append(
            {
                "rank": rank,
                "idx": int(item["idx"]),
                "score": float(item["score"]),
            }
        )
    return ranked


def _compute_margins(scores: List[float], metric: str) -> Dict[str, Any]:
    if not scores:
        return {"margin_to_next": [], "margin_top1_to_last": None, "ties_count": 0}

    margin_to_next: List[Optional[float]] = []
    ties_count = 0
    for i in range(len(scores) - 1):
        a = scores[i]
        b = scores[i + 1]
        margin = (b - a) if metric == "euclidean" else (a - b)
        margin_to_next.append(float(margin))
        if abs(a - b) <= SCORE_EPS:
            ties_count += 1
    margin_to_next.append(None)

    margin_top1_to_last = (scores[-1] - scores[0]) if metric == "euclidean" else (scores[0] - scores[-1])
    return {
        "margin_to_next": margin_to_next,
        "margin_top1_to_last": float(margin_top1_to_last),
        "ties_count": ties_count,
    }


def _build_audit_json(result: "QueryResult") -> Dict[str, Any]:
    if result.audit is None:
        raise ValueError("Missing audit data on QueryResult")

    payload = {field: result.audit.get(field) for field in _AUDIT_FIELDS}
    payload["top_k"] = result.audit.get("top_k", result.top_k)
    payload["query_vector_bytes"] = result.audit.get("query_vector_bytes")

    missing = [k for k, v in payload.items() if v is None]
    if missing:
        raise ValueError(f"Missing audit fields: {', '.join(missing)}")

    if result.results is not None:
        payload["ranking"] = _rank_results(result.results)
    if result.margins is not None:
        payload["margins"] = result.margins
    if result.stability is not None:
        payload["stability"] = result.stability
    if result.certificate is not None:
        payload["certificate"] = result.certificate
        payload["block_order"] = result.certificate.get("block_order")
        payload["block_size"] = result.certificate.get("block_size")
        payload["pruning_count"] = result.certificate.get("pruned_candidates")
        payload["threshold_evolution"] = result.certificate.get("threshold_evolution")
        payload["simd_path"] = result.certificate.get("simd_path")
        payload["compile_flags"] = result.certificate.get("compile_flags")
        payload["bound_kind"] = result.certificate.get("bound_kind")
        payload["pruned_bound_values"] = result.certificate.get("pruned_bound_values")
    if result.audit is not None:
        if "cosine_contract" in result.audit:
            payload["cosine_contract"] = result.audit.get("cosine_contract")
        if "query_norm" in result.audit:
            payload["query_norm"] = result.audit.get("query_norm")

    return payload


@dataclass
class SearchConfig:
    """Runtime configuration for the search engine."""

    metric: str = "cosine"
    normalized: bool = True
    verbose: bool = False
    enable_metrics: bool = False
    chunk_size: int = DEFAULT_CHUNK_SIZE


@dataclass
class QueryResult:
    """Result of a single query with optional timings and metadata."""

    results: List[Dict[str, Any]]
    query_time_ms: float
    method_used: str
    audit: Optional[Dict[str, Any]] = None
    margins: Optional[Dict[str, Any]] = None
    stability: Optional[Dict[str, Any]] = None
    top_k: Optional[int] = None
    candidates_examined: int = 0
    simd_used: bool = False
    parse_time_ms: float = 0.0
    compute_time_ms: float = 0.0
    sort_time_ms: float = 0.0
    certificate: Optional[Dict[str, Any]] = None

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self):
        return iter(self.results)

    def __getitem__(self, key):
        return self.results[key]

    def save_audit(self, path: Union[str, Path]) -> Dict[str, Any]:
        return export_audit(self, path)


def export_audit(result: QueryResult, path: Union[str, Path]) -> Dict[str, Any]:
    """Export audit.json using data already returned by Rust."""
    payload = _build_audit_json(result)
    path = Path(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return payload


def replay(audit_path: Union[str, Path]) -> Dict[str, Any]:
    """Replay a query from audit.json and validate determinism."""
    audit_path = Path(audit_path)
    with audit_path.open("r", encoding="utf-8") as handle:
        audit = json.load(handle)

    differences: List[Dict[str, Any]] = []
    missing_fields = [field for field in _AUDIT_FIELDS if field not in audit]
    if missing_fields:
        differences.append({"type": "missing_fields", "fields": missing_fields})
        return {"ok": False, "differences": differences}

    index_path = Path(audit["index_path"])
    if not index_path.exists():
        differences.append({"type": "index_path_missing", "path": str(index_path)})
        return {"ok": False, "differences": differences}

    try:
        raw = base64.b64decode(audit["query_vector_bytes"])
    except Exception as exc:
        differences.append({"type": "query_vector_bytes_invalid", "error": str(exc)})
        return {"ok": False, "differences": differences}

    query_vector = np.frombuffer(raw, dtype="<f4")
    expected_dims = int(audit["dims"])
    if query_vector.size != expected_dims:
        differences.append(
            {
                "type": "query_vector_dims_mismatch",
                "expected": expected_dims,
                "actual": int(query_vector.size),
            }
        )
        return {"ok": False, "differences": differences}

    try:
        engine = SearchEngine(index_path)
        expected_cert = audit.get("certificate")
        if expected_cert is not None:
            replay_result = engine.query_exact_certified(
                query_vector,
                top_k=int(audit["top_k"]),
                block_size=int(audit.get("block_size", expected_cert.get("block_size", 64))),
                enable_pruning=bool(expected_cert.get("enable_pruning", True)),
                return_certificate=True,
            )
        else:
            replay_result = engine.query(query_vector, top_k=int(audit["top_k"]), format="detailed")
    except Exception as exc:
        differences.append({"type": "replay_error", "error": str(exc)})
        return {"ok": False, "differences": differences}

    replay_audit = replay_result.audit or {}
    for field in ["index_hash", "query_hash", "metric", "normalized", "dims", "rows", "engine_version"]:
        if audit.get(field) != replay_audit.get(field):
            differences.append(
                {
                    "type": "field_mismatch",
                    "field": field,
                    "expected": audit.get(field),
                    "actual": replay_audit.get(field),
                }
            )

    expected_ranking = audit.get("ranking")
    if expected_ranking is not None:
        actual_ranking = replay_result.results or []
        if len(expected_ranking) != len(actual_ranking):
            differences.append(
                {
                    "type": "ranking_length_mismatch",
                    "expected": len(expected_ranking),
                    "actual": len(actual_ranking),
                }
            )
        else:
            for idx, (exp_item, act_item) in enumerate(zip(expected_ranking, actual_ranking)):
                exp_idx = int(exp_item["idx"])
                act_idx = int(act_item["idx"])
                exp_score = float(exp_item["score"])
                act_score = float(act_item["score"])
                if exp_idx != act_idx or abs(exp_score - act_score) > SCORE_EPS:
                    differences.append(
                        {
                            "type": "ranking_mismatch",
                            "position": idx,
                            "expected": {"idx": exp_idx, "score": exp_score},
                            "actual": {"idx": act_idx, "score": act_score},
                        }
                    )
                    break

    expected_cert = audit.get("certificate")
    if expected_cert is not None:
        actual_cert = replay_result.certificate or {}
        if not expected_cert.get("safe", False):
            differences.append(
                {"type": "certificate_not_safe", "expected": True, "actual": expected_cert.get("safe")}
            )
        total = int(expected_cert.get("total_candidates", 0))
        pruned = int(expected_cert.get("pruned_candidates", 0))
        full_eval = int(expected_cert.get("full_evaluated", 0))
        if total > 0 and (pruned + full_eval) != total:
            differences.append(
                {"type": "certificate_count_mismatch", "expected": total, "actual": pruned + full_eval}
            )
        for key in ["bound_kind", "block_size", "enable_pruning"]:
            if expected_cert.get(key) != actual_cert.get(key):
                differences.append(
                    {
                        "type": "certificate_field_mismatch",
                        "field": key,
                        "expected": expected_cert.get(key),
                        "actual": actual_cert.get(key),
                    }
                )

    return {"ok": len(differences) == 0, "differences": differences}


class SearchEngine:
    """Loaded index ready for exact nearest-neighbour queries."""

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

    @property
    def fast_path(self) -> str:
        if self.config.metric == "cosine" and self.config.normalized:
            return "normalized_cosine"
        if self.config.metric == "dot":
            return "dot_compat"
        return "euclidean_compat"

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

    def _result_list(self, raw_rr) -> List[Dict[str, Any]]:
        results_py: List[Dict[str, Any]] = []
        if hasattr(raw_rr, "results"):
            for item in raw_rr.results:
                idx = getattr(item, "idx", None)
                score = getattr(item, "score", None)
                if idx is not None and score is not None and np.isfinite(score):
                    results_py.append({"idx": int(idx), "score": float(score)})
        return results_py

    def _audit_from_rr(self, rr, query_vector: np.ndarray, top_k: int) -> Dict[str, Any]:
        audit = {
            "index_hash": getattr(rr, "index_hash", None),
            "query_hash": getattr(rr, "query_hash", None),
            "engine_version": getattr(rr, "engine_version", None),
            "metric": getattr(rr, "metric", None),
            "normalized": getattr(rr, "normalized", None),
            "dims": getattr(rr, "dims", None),
            "rows": getattr(rr, "rows", None),
            "index_path": getattr(rr, "index_path", None),
            "query_vector_bytes": _encode_query_vector_bytes(query_vector),
            "top_k": int(top_k),
        }
        if self.config.metric == "cosine":
            audit["cosine_contract"] = "caller_provided_normalized_embeddings" if self.config.normalized else "index_normalized_internally_from_raw_embeddings"
            audit["query_norm"] = float(np.linalg.norm(query_vector))
        return audit

    def query(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        format: str = "simple",
        return_timing: bool = False,
        stability: bool = False,
        stability_n: int = 16,
        stability_noise_eps: float = 1e-3,
        stability_seed: int = 1337,
    ) -> Union[List[Dict], QueryResult, Tuple]:
        if not self._initialized:
            raise RuntimeError("Engine not initialized")

        query_vector = self._validate_query_vector(query_vector)

        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if stability and (stability_n <= 0 or stability_noise_eps < 0.0):
            raise ValueError("Invalid stability configuration")

        effective_k = min(top_k, self.rows)
        start_time = time.time()

        try:
            rr = self._engine.query_exact_detailed(query_vector, int(effective_k)) if format == "detailed" else self._engine.query_exact(query_vector, int(effective_k))
            query_time = (time.time() - start_time) * 1000.0

            results_py = self._result_list(rr)
            method_used = getattr(rr, "method_used", "exact")
            candidates_generated = getattr(rr, "candidates_generated", 0)
            simd_used = bool(getattr(rr, "simd_used", False))
            parse_time_ms = float(getattr(rr, "parse_time_ms", 0.0))
            compute_time_ms = float(getattr(rr, "compute_time_ms", 0.0))
            sort_time_ms = float(getattr(rr, "sort_time_ms", 0.0))

            if format == "simple":
                if return_timing:
                    return results_py, {"query_time_ms": query_time, "simd_used": simd_used}
                return results_py

            ranked_results = _rank_results(results_py)
            audit = self._audit_from_rr(rr, query_vector, int(effective_k))
            scores_only = [float(item["score"]) for item in ranked_results]
            margins = _compute_margins(scores_only, str(audit.get("metric") or ""))

            stability_result = None
            if stability:
                stability_result = self._compute_stability(
                    query_vector,
                    ranked_results,
                    int(effective_k),
                    int(stability_n),
                    float(stability_noise_eps),
                    int(stability_seed),
                )

            qr = QueryResult(
                results=ranked_results,
                query_time_ms=query_time,
                method_used=method_used,
                audit=audit,
                margins=margins,
                stability=stability_result,
                top_k=int(effective_k),
                candidates_examined=candidates_generated,
                simd_used=simd_used,
                parse_time_ms=parse_time_ms,
                compute_time_ms=compute_time_ms,
                sort_time_ms=sort_time_ms,
            )

            if return_timing:
                return qr, {"query_time_ms": query_time, "method_used": method_used, "simd_used": simd_used}
            return qr
        except Exception as e:
            error_msg = f"Query failed: {e}"
            if self.config.verbose:
                print(error_msg)
            raise RuntimeError(error_msg)

    def query_simple(self, query_vector: np.ndarray, top_k: int = 10) -> List[Dict]:
        return self.query(query_vector, top_k, format="simple")

    def query_simple_arrays(self, query_vector: np.ndarray, top_k: int = 10) -> Dict[str, Any]:
        if not self._initialized:
            raise RuntimeError("Engine not initialized")
        query_vector = self._validate_query_vector(query_vector)
        effective_k = min(top_k, self.rows)
        try:
            return getattr(self._engine, "query_exact_arrays")(query_vector, int(effective_k))
        except AttributeError:
            rr = self._engine.query_exact(query_vector, int(effective_k))
            results = self._result_list(rr)
            return {
                "indices": np.asarray([item["idx"] for item in results], dtype=np.int64)[None, :],
                "scores": np.asarray([item["score"] for item in results], dtype=np.float32)[None, :],
            }
        except Exception as e:
            raise RuntimeError(f"Query failed: {e}")

    def query_detailed(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        stability: bool = False,
        stability_n: int = 16,
        stability_noise_eps: float = 1e-3,
        stability_seed: int = 1337,
    ) -> QueryResult:
        return self.query(
            query_vector,
            top_k,
            format="detailed",
            stability=stability,
            stability_n=stability_n,
            stability_noise_eps=stability_noise_eps,
            stability_seed=stability_seed,
        )

    def query_batch(self, queries: np.ndarray, top_k: int = 10, format: str = "simple") -> List:
        if not self._initialized:
            raise RuntimeError("Engine not initialized")

        queries = np.asarray(queries, dtype=np.float32, order="C")
        if queries.ndim != 2:
            raise ValueError("Queries must be a 2D array (N × dims)")
        if queries.shape[1] != self.dims:
            raise ValueError(f"Query dimensions {queries.shape[1]} != index dimensions {self.dims}")
        if queries.shape[0] == 0:
            return []
        if not np.all(np.isfinite(queries)):
            raise ValueError("Queries contain non-finite values")

        num_queries = queries.shape[0]
        should_chunk = num_queries > MAX_BATCH_SIZE
        if not should_chunk:
            estimated_batch_mem = (num_queries * self.rows * 4) / (1024 * 1024)
            should_chunk = estimated_batch_mem > 2000 and get_memory_usage() > 12_000

        if should_chunk:
            return self._query_batch_chunked(queries, top_k, format)

        try:
            if format == "detailed":
                rust_results = self._engine.query_batch_detailed(queries, top_k)
            else:
                rust_results = self._engine.query_batch(queries, top_k)
            return self._process_batch_results(rust_results, format)
        except Exception as e:
            if self.config.verbose:
                print(f"Batch query failed: {e}, falling back to chunked processing")
            return self._query_batch_chunked(queries, top_k, format)

    def query_batch_arrays(self, queries: np.ndarray, top_k: int = 10) -> Dict[str, Any]:
        if not self._initialized:
            raise RuntimeError("Engine not initialized")

        queries = np.asarray(queries, dtype=np.float32, order="C")
        if queries.ndim != 2:
            raise ValueError("Queries must be a 2D array (N × dims)")
        if queries.shape[1] != self.dims:
            raise ValueError(f"Query dimensions {queries.shape[1]} != index dimensions {self.dims}")
        if queries.shape[0] == 0:
            return {"indices": np.empty((0, 0), dtype=np.int64), "scores": np.empty((0, 0), dtype=np.float32)}
        if not np.all(np.isfinite(queries)):
            raise ValueError("Queries contain non-finite values")

        effective_k = min(top_k, self.rows)
        try:
            return getattr(self._engine, "query_batch_arrays")(queries, int(effective_k))
        except AttributeError:
            batch = self.query_batch(queries, top_k=effective_k, format="simple")
            indices = np.asarray([[item["idx"] for item in row] for row in batch], dtype=np.int64)
            scores = np.asarray([[item["score"] for item in row] for row in batch], dtype=np.float32)
            return {"indices": indices, "scores": scores}
        except Exception as e:
            raise RuntimeError(f"Batch query failed: {e}")

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
        for i in range(0, num_queries, chunk_size):
            end_idx = min(i + chunk_size, num_queries)
            chunk = queries[i:end_idx]
            try:
                if format == "detailed":
                    chunk_results = self._engine.query_batch_detailed(chunk, top_k)
                else:
                    chunk_results = self._engine.query_batch(chunk, top_k)
                results.extend(self._process_batch_results(chunk_results, format))
            except Exception:
                empty_results = [[] if format == "simple" else {} for _ in range(end_idx - i)]
                results.extend(empty_results)
        return results

    def _process_batch_results(self, rust_results, format: str) -> List:
        if format == "simple":
            return [
                [
                    {"idx": int(item.idx), "score": float(item.score)}
                    for item in result.results
                    if hasattr(item, "idx") and hasattr(item, "score") and np.isfinite(item.score)
                ]
                for result in rust_results
            ]

        if format == "detailed":
            processed = []
            for result in rust_results:
                valid_results = [
                    {"idx": int(item.idx), "score": float(item.score)}
                    for item in result.results
                    if hasattr(item, "idx") and hasattr(item, "score") and np.isfinite(item.score)
                ]
                ranked_results = _rank_results(valid_results)
                audit = {
                    "index_hash": getattr(result, "index_hash", None),
                    "query_hash": getattr(result, "query_hash", None),
                    "engine_version": getattr(result, "engine_version", None),
                    "metric": getattr(result, "metric", None),
                    "normalized": getattr(result, "normalized", None),
                    "dims": getattr(result, "dims", None),
                    "rows": getattr(result, "rows", None),
                    "index_path": getattr(result, "index_path", None),
                }
                processed.append(
                    {
                        "results": ranked_results,
                        "query_time_ms": getattr(result, "query_time_ms", 0.0),
                        "method_used": getattr(result, "method_used", "unknown"),
                        "audit": audit,
                        "margins": _compute_margins(
                            [float(item["score"]) for item in ranked_results],
                            str(audit.get("metric") or ""),
                        ),
                        "candidates_examined": getattr(result, "candidates_generated", 0),
                        "simd_used": getattr(result, "simd_used", False),
                        "parse_time_ms": getattr(result, "parse_time_ms", 0.0),
                        "compute_time_ms": getattr(result, "compute_time_ms", 0.0),
                        "sort_time_ms": getattr(result, "sort_time_ms", 0.0),
                    }
                )
            return processed

        raise ValueError("Unknown format. Use 'simple' or 'detailed'.")

    def query_exact_certified(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        block_size: int = 64,
        enable_pruning: bool = True,
        return_certificate: bool = True,
        strict_query_normalized: bool = False,
    ) -> Union[List[Dict[str, Any]], QueryResult]:
        if not self._initialized:
            raise RuntimeError("Engine not initialized")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if block_size <= 0:
            raise ValueError("block_size must be positive")

        query_vector = self._validate_query_vector(query_vector)
        query_norm = float(np.linalg.norm(query_vector))
        cosine_contract: Optional[str] = None
        if self.config.metric == "cosine":
            cosine_contract = (
                "caller_provided_normalized_embeddings" if self.config.normalized else "index_normalized_internally_from_raw_embeddings"
            )
            if strict_query_normalized and abs(query_norm - 1.0) > 1e-3:
                raise ValueError("Cosine query must be normalized when strict_query_normalized=True")

        effective_k = min(top_k, self.rows)
        try:
            raw = self._engine.query_exact_certified(
                query_vector,
                int(effective_k),
                int(block_size),
                bool(enable_pruning),
            )
        except Exception as e:
            raise RuntimeError(f"Certified query failed: {e}")

        results_py = [item for item in raw.get("results", []) if item.get("idx") is not None and item.get("score") is not None and np.isfinite(item.get("score"))]
        ranked_results = _rank_results(results_py)
        certificate = raw.get("certificate") or {}
        audit = {
            "index_hash": raw.get("index_hash"),
            "query_hash": raw.get("query_hash"),
            "engine_version": raw.get("engine_version"),
            "metric": raw.get("metric"),
            "normalized": raw.get("normalized"),
            "dims": raw.get("dims"),
            "rows": raw.get("rows"),
            "index_path": raw.get("index_path"),
            "query_vector_bytes": _encode_query_vector_bytes(query_vector),
            "top_k": int(effective_k),
            "query_norm": query_norm,
        }
        if cosine_contract is not None:
            audit["cosine_contract"] = cosine_contract

        qr = QueryResult(
            results=ranked_results,
            query_time_ms=float(raw.get("query_time_ms", 0.0)),
            method_used=str(raw.get("method_used", "exact-certified")),
            audit=audit,
            margins=_compute_margins([float(x["score"]) for x in ranked_results], str(audit.get("metric") or "")),
            stability=None,
            top_k=int(effective_k),
            candidates_examined=int(certificate.get("full_evaluated", 0)),
            simd_used=bool(certificate.get("simd_path", "scalar") != "scalar"),
            parse_time_ms=0.0,
            compute_time_ms=0.0,
            sort_time_ms=0.0,
            certificate=certificate if return_certificate else None,
        )
        return qr if return_certificate else qr.results

    def query_batch_exact_certified(
        self,
        queries: np.ndarray,
        top_k: int = 10,
        block_size: int = 64,
        enable_pruning: bool = True,
        return_certificate: bool = True,
    ) -> List[Union[List[Dict[str, Any]], QueryResult]]:
        if not self._initialized:
            raise RuntimeError("Engine not initialized")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if block_size <= 0:
            raise ValueError("block_size must be positive")

        queries = np.asarray(queries, dtype=np.float32, order="C")
        if queries.ndim != 2:
            raise ValueError("Queries must be a 2D array (N x dims)")
        if queries.shape[1] != self.dims:
            raise ValueError(f"Query dimensions {queries.shape[1]} != index dimensions {self.dims}")
        if not np.all(np.isfinite(queries)):
            raise ValueError("Queries contain non-finite values")
        if queries.shape[0] == 0:
            return []

        effective_k = min(top_k, self.rows)
        try:
            raw_batch = self._engine.query_batch_exact_certified(
                queries,
                int(effective_k),
                int(block_size),
                bool(enable_pruning),
            )
        except Exception as e:
            raise RuntimeError(f"Batch certified query failed: {e}")

        out: List[Union[List[Dict[str, Any]], QueryResult]] = []
        for i, raw in enumerate(raw_batch):
            results_py = [item for item in raw.get("results", []) if item.get("idx") is not None and item.get("score") is not None and np.isfinite(item.get("score"))]
            ranked_results = _rank_results(results_py)
            certificate = raw.get("certificate") or {}
            q = queries[i]
            audit = {
                "index_hash": raw.get("index_hash"),
                "query_hash": raw.get("query_hash"),
                "engine_version": raw.get("engine_version"),
                "metric": raw.get("metric"),
                "normalized": raw.get("normalized"),
                "dims": raw.get("dims"),
                "rows": raw.get("rows"),
                "index_path": raw.get("index_path"),
                "query_vector_bytes": _encode_query_vector_bytes(q),
                "top_k": int(effective_k),
                "query_norm": float(np.linalg.norm(q)),
            }
            qr = QueryResult(
                results=ranked_results,
                query_time_ms=float(raw.get("query_time_ms", 0.0)),
                method_used=str(raw.get("method_used", "exact-certified")),
                audit=audit,
                margins=_compute_margins([float(x["score"]) for x in ranked_results], str(audit.get("metric") or "")),
                stability=None,
                top_k=int(effective_k),
                candidates_examined=int(certificate.get("full_evaluated", 0)),
                simd_used=bool(certificate.get("simd_path", "scalar") != "scalar"),
                parse_time_ms=0.0,
                compute_time_ms=0.0,
                sort_time_ms=0.0,
                certificate=certificate if return_certificate else None,
            )
            out.append(qr if return_certificate else qr.results)
        return out

    def compare(
        self,
        query_vector: np.ndarray,
        idx_a: int,
        idx_b: int,
        block_size: int = 64,
        top_blocks: int = 5,
    ) -> Dict[str, Any]:
        if not self._initialized:
            raise RuntimeError("Engine not initialized")

        query_vector = self._validate_query_vector(query_vector)
        if idx_a < 0 or idx_b < 0:
            raise ValueError("idx_a and idx_b must be non-negative")
        if idx_a >= self.rows or idx_b >= self.rows:
            raise ValueError(f"idx_a or idx_b out of bounds (rows={self.rows})")
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if top_blocks <= 0:
            raise ValueError("top_blocks must be positive")

        return self._engine.compare(query_vector, int(idx_a), int(idx_b), int(block_size), int(top_blocks))

    def explain(
        self,
        query_vector: np.ndarray,
        idx: int,
        block_size: int = 64,
        top_blocks: int = 5,
    ) -> Dict[str, Any]:
        if not self._initialized:
            raise RuntimeError("Engine not initialized")

        query_vector = self._validate_query_vector(query_vector)
        if idx < 0:
            raise ValueError("idx must be non-negative")
        if idx >= self.rows:
            raise ValueError(f"idx out of bounds (rows={self.rows})")
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if top_blocks <= 0:
            raise ValueError("top_blocks must be positive")

        return self._engine.explain(query_vector, int(idx), int(block_size), int(top_blocks))

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
            return (
                f"SearchEngine(path='{self.index_path}', vectors={self.rows:,}, dims={self.dims}, "
                f"fast_path='{self.fast_path}', mem={mem_usage:.1f}MB)"
            )
        return f"SearchEngine({self.rows:,} vectors × {self.dims}D, fast_path='{self.fast_path}')"

    def __del__(self):
        if hasattr(self, "_engine"):
            del self._engine
        gc.collect()


def from_embeddings(
    embeddings: np.ndarray,
    metric: str = "cosine",
    base_name: str = "nseekfs_index",
    output_dir: Optional[str] = None,
    normalized: bool = True,
    config: Optional[SearchConfig] = None,
    verbose: bool = False,
) -> SearchEngine:
    try:
        from nseekfs.nseekfs import py_prepare_bin_from_embeddings
    except ImportError as exc:
        raise RuntimeError("Rust engine not available. Make sure the package is installed.") from exc

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
        print(f"Index created in {time.time() - t0:.2f}s → {path}")

    if config is None:
        config = SearchConfig(metric=metric, normalized=normalized, verbose=verbose)
    return SearchEngine(path, config)


def load_index(
    index_path: Union[str, Path],
    config: Optional[SearchConfig] = None,
    verbose: bool = False,
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
            print("Index loaded successfully")
            print(f"{engine.rows:,} vectors × {engine.dims} dimensions")
        return engine
    except Exception as e:
        raise RuntimeError(f"Failed to load index: {e}")


ValidationError = ValueError
IndexError = Exception


@contextlib.contextmanager
def temporary_index(embeddings: np.ndarray, **kwargs):
    """Context manager for creating temporary indices that are automatically cleaned."""
    with tempfile.TemporaryDirectory() as temp_dir:
        kwargs.setdefault("output_dir", temp_dir)
        kwargs.setdefault("base_name", "temp_index")

        engine = from_embeddings(embeddings, **kwargs)
        try:
            yield engine
        finally:
            pass
