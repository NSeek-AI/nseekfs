"""Input validation utilities for NSeekFS v1.0 - Exact Search Focus"""

import numpy as np
from typing import Union, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


MIN_DIMENSIONS = 8
MAX_DIMENSIONS = 10000  
MAX_VECTORS = 100_000_000  
MIN_VECTORS = 1
MAX_FILE_SIZE = 10 * 1024 * 1024 * 1024  
NORMALIZATION_TOLERANCE = 0.1  

class ValidationError(ValueError):
    """Custom exception for validation errors with context"""
    pass

def validate_embeddings(embeddings: Union[np.ndarray, List[List[float]], str]) -> np.ndarray:
    """
    Validate and convert embeddings to proper format for NSeekFS v1.0.
    
    Args:
        embeddings: Input embeddings in various formats
        
    Returns:
        np.ndarray: Validated embeddings as f32 array
        
    Raises:
        ValidationError: If embeddings are invalid
        FileNotFoundError: If file path doesn't exist
        MemoryError: If embeddings are too large
    """
    
    
    if isinstance(embeddings, (str, Path)):
        path = Path(embeddings)
        if not path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {embeddings}")
        
        
        file_size = path.stat().st_size
        if file_size > MAX_FILE_SIZE:
            raise ValidationError(f"File too large: {file_size / (1024**3):.1f}GB (max: {MAX_FILE_SIZE / (1024**3):.1f}GB)")
        
        if file_size == 0:
            raise ValidationError("File is empty")
        
        logger.info(f"Loading embeddings from file: {path} ({file_size / (1024**2):.1f}MB)")
        
        try:
            if str(path).endswith(".npy"):
                embeddings = np.load(path)
            elif str(path).endswith((".csv", ".txt")):
                try:
                    embeddings = np.loadtxt(path, delimiter=",")
                except ValueError as e:
                    
                    for delimiter in [";", "\t", " "]:
                        try:
                            embeddings = np.loadtxt(path, delimiter=delimiter)
                            logger.info(f"Successfully loaded CSV with delimiter '{delimiter}'")
                            break
                        except ValueError:
                            continue
                    else:
                        raise ValidationError(f"Failed to parse CSV file: {e}")
            else:
                raise ValidationError(f"Unsupported file format: {path.suffix}. Only .npy, .csv, .txt are supported.")
        except MemoryError:
            raise MemoryError(f"File too large to load into memory: {path}")
        except Exception as e:
            raise ValidationError(f"Failed to load file {path}: {e}")
    
    
    try:
        embeddings = np.asarray(embeddings, dtype=np.float32)
    except (ValueError, TypeError) as e:
        raise ValidationError(f"Cannot convert embeddings to numpy array: {e}")
    except MemoryError:
        raise MemoryError("Embeddings too large for available memory")
    
    
    if embeddings.ndim == 0:
        raise ValidationError("Embeddings cannot be a scalar")
    elif embeddings.ndim == 1:
        
        if len(embeddings) < MIN_DIMENSIONS:
            raise ValidationError(f"Single vector too short: {len(embeddings)} dimensions (minimum: {MIN_DIMENSIONS})")
        embeddings = embeddings.reshape(1, -1)
        logger.info("Reshaped single vector to 2D array")
    elif embeddings.ndim != 2:
        raise ValidationError(f"Embeddings must be 1D or 2D array, got {embeddings.ndim}D")
    
    n_samples, dims = embeddings.shape
    
    
    if n_samples < MIN_VECTORS:
        raise ValidationError(f"Too few vectors: {n_samples} (minimum: {MIN_VECTORS})")
    
    if n_samples > MAX_VECTORS:
        raise ValidationError(f"Too many vectors: {n_samples} (maximum: {MAX_VECTORS:,})")
    
    if dims < MIN_DIMENSIONS:
        raise ValidationError(f"Embedding dimension too small: {dims} (minimum: {MIN_DIMENSIONS})")
    
    if dims > MAX_DIMENSIONS:
        raise ValidationError(f"Embedding dimension too large: {dims} (maximum: {MAX_DIMENSIONS})")
    
    
    logger.debug("Validating embedding values...")
    
    
    if np.any(np.isnan(embeddings)):
        nan_count = np.sum(np.isnan(embeddings))
        nan_rows = np.any(np.isnan(embeddings), axis=1)
        nan_row_indices = np.where(nan_rows)[0]
        raise ValidationError(f"NaN values detected: {nan_count} total NaNs in {np.sum(nan_rows)} vectors "
                            f"(first few: {nan_row_indices[:5].tolist()})")
    
    
    if np.any(np.isinf(embeddings)):
        inf_count = np.sum(np.isinf(embeddings))
        inf_rows = np.any(np.isinf(embeddings), axis=1)
        inf_row_indices = np.where(inf_rows)[0]
        raise ValidationError(f"Infinite values detected: {inf_count} total infinities in {np.sum(inf_rows)} vectors "
                            f"(first few: {inf_row_indices[:5].tolist()})")
    
    
    norms = np.linalg.norm(embeddings, axis=1)
    zero_vectors = norms == 0.0
    if np.any(zero_vectors):
        zero_indices = np.where(zero_vectors)[0]
        raise ValidationError(f"Zero vectors found at indices: {zero_indices[:10].tolist()}")
    
    
    tiny_vectors = norms < 1e-10
    if np.any(tiny_vectors):
        tiny_indices = np.where(tiny_vectors)[0]
        logger.warning(f"Very small vectors detected at indices: {tiny_indices[:5].tolist()} "
                      f"(may cause numerical instability)")
    
    
    large_norms = norms > 1e6
    if np.any(large_norms):
        large_indices = np.where(large_norms)[0]
        max_norm = np.max(norms)
        logger.warning(f"Very large vectors detected at indices: {large_indices[:5].tolist()} "
                      f"(max norm: {max_norm:.2e})")
    
    
    memory_usage = embeddings.nbytes
    if memory_usage > 1024**3:  
        logger.warning(f"Large embeddings array: {memory_usage / (1024**3):.2f}GB memory usage")
    
    
    logger.debug("Computing basic statistics...")
    mean_norm = np.mean(norms)
    std_norm = np.std(norms)
    
    logger.info(f"Embeddings validated: {n_samples:,} vectors × {dims} dims")
    logger.info(f"Vector norms: mean={mean_norm:.3f}, std={std_norm:.3f}, range=[{np.min(norms):.3f}, {np.max(norms):.3f}]")
    
    
    normalized_check = np.abs(norms - 1.0) < NORMALIZATION_TOLERANCE
    if np.all(normalized_check):
        logger.info("Vectors appear to be normalized")
    elif np.mean(normalized_check) > 0.8:
        logger.info(f"Most vectors appear normalized ({np.mean(normalized_check)*100:.1f}%)")
    else:
        logger.info("Vectors do not appear to be normalized")
    
    return embeddings

def validate_query_vector(query_vector: Union[np.ndarray, List[float]], expected_dims: int) -> np.ndarray:
    """
    Validate query vector format and dimensions.
    
    Args:
        query_vector: Input query vector
        expected_dims: Expected number of dimensions
        
    Returns:
        np.ndarray: Validated query vector
        
    Raises:
        ValidationError: If query vector is invalid
    """
    
    if query_vector is None:
        raise ValidationError("Query vector cannot be None")
    
    if not isinstance(query_vector, (list, np.ndarray, tuple)):
        raise ValidationError(f"Query vector must be a list, tuple, or numpy array, got {type(query_vector)}")
    
    
    try:
        query_vector = np.asarray(query_vector, dtype=np.float32)
    except (ValueError, TypeError) as e:
        raise ValidationError(f"Cannot convert query vector to numpy array: {e}")
    
    
    if query_vector.ndim == 0:
        raise ValidationError("Query vector cannot be a scalar")
    elif query_vector.ndim > 2:
        raise ValidationError(f"Query vector must be 1D or 2D, got {query_vector.ndim}D")
    elif query_vector.ndim == 2:
        if query_vector.shape[0] != 1:
            raise ValidationError(f"2D query vector must have shape (1, dims), got {query_vector.shape}")
        query_vector = query_vector.flatten()
    
    
    if len(query_vector) != expected_dims:
        raise ValidationError(f"Query vector dimension mismatch: expected {expected_dims}, got {len(query_vector)}")
    
    
    if np.any(np.isnan(query_vector)):
        nan_indices = np.where(np.isnan(query_vector))[0]
        raise ValidationError(f"Query vector contains NaN values at indices: {nan_indices[:5].tolist()}")
    
    if np.any(np.isinf(query_vector)):
        inf_indices = np.where(np.isinf(query_vector))[0]
        raise ValidationError(f"Query vector contains infinite values at indices: {inf_indices[:5].tolist()}")
    
    
    norm = np.linalg.norm(query_vector)
    if norm == 0.0:
        raise ValidationError("Query vector cannot be zero (norm = 0)")
    
    
    if norm < 1e-10:
        logger.warning(f"Query vector has very small norm: {norm:.2e} (may cause numerical instability)")
    
    
    if norm > 1e6:
        logger.warning(f"Query vector has very large norm: {norm:.2e}")
    
    return query_vector

def validate_level(level: str) -> str:
    """
    Validate precision level.
    
    Args:
        level: Precision level string
        
    Returns:
        str: Validated level
        
    Raises:
        ValidationError: If level is invalid
    """
    if not isinstance(level, str):
        raise ValidationError(f"Level must be a string, got {type(level)}")
    
    level = level.strip().lower()
    
    if not level:
        raise ValidationError("Level cannot be empty")
    
    valid_levels = {"f8", "f16", "f32", "f64"}
    
    if level not in valid_levels:
        
        suggestions = []
        if level in {"float8", "8bit", "8-bit"}:
            suggestions.append("f8")
        elif level in {"float16", "half", "16bit", "16-bit"}:
            suggestions.append("f16")
        elif level in {"float32", "float", "32bit", "32-bit"}:
            suggestions.append("f32")
        elif level in {"float64", "double", "64bit", "64-bit"}:
            suggestions.append("f64")
        
        suggestion_text = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        raise ValidationError(f"Invalid level '{level}'. Must be one of: {', '.join(sorted(valid_levels))}.{suggestion_text}")
    
    return level

def validate_top_k(top_k: int, max_rows: int) -> int:
    """
    Validate top_k parameter.
    
    Args:
        top_k: Number of results to return
        max_rows: Maximum available rows
        
    Returns:
        int: Validated top_k
        
    Raises:
        ValidationError: If top_k is invalid
    """
    if not isinstance(top_k, (int, np.integer)):
        raise ValidationError(f"top_k must be an integer, got {type(top_k)}")
    
    top_k = int(top_k)  
    
    if top_k <= 0:
        raise ValidationError(f"top_k must be greater than 0, got {top_k}")
    
    if top_k > max_rows:
        raise ValidationError(f"top_k ({top_k}) cannot be greater than number of vectors ({max_rows})")
    
    
    if top_k > 10000:
        logger.warning(f"Large top_k value: {top_k} (may impact performance)")
    
    return top_k

def validate_similarity(similarity: str) -> str:
    """
    Validate similarity metric.
    
    Args:
        similarity: Similarity metric string
        
    Returns:
        str: Validated similarity metric
        
    Raises:
        ValidationError: If similarity is invalid
    """
    if not isinstance(similarity, str):
        raise ValidationError(f"Similarity must be a string, got {type(similarity)}")
    
    similarity = similarity.strip().lower()
    
    if not similarity:
        raise ValidationError("Similarity cannot be empty")
    
    
    similarity_mapping = {
        "cosine": "cosine",
        "cos": "cosine",
        "cosine_similarity": "cosine",
        "euclidean": "euclidean",
        "euclidean_distance": "euclidean",
        "l2": "euclidean",
        "l2_distance": "euclidean",
        "dot_product": "dot_product",
        "dot": "dot_product",
        "inner_product": "dot_product",
        "ip": "dot_product",
    }
    
    if similarity in similarity_mapping:
        return similarity_mapping[similarity]
    
    valid_similarities = {"cosine", "euclidean", "dot_product"}
    
    
    suggestions = []
    if similarity in {"manhattan", "l1", "l1_distance"}:
        suggestions.append("euclidean (L2)")
    elif similarity in {"correlation", "pearson"}:
        suggestions.append("cosine")
    elif similarity in {"jaccard", "hamming"}:
        suggestions.append("cosine (for continuous vectors)")
    
    suggestion_text = f" Available metrics: {', '.join(sorted(valid_similarities))}."
    if suggestions:
        suggestion_text += f" Similar to what you want: {', '.join(suggestions)}."
    
    raise ValidationError(f"Invalid similarity '{similarity}'.{suggestion_text}")

def validate_file_path(path: Union[str, Path], must_exist: bool = True, check_writable: bool = False) -> Path:
    """
    Validate file path.
    
    Args:
        path: File path to validate
        must_exist: Whether file must already exist
        check_writable: Whether to check if directory is writable
        
    Returns:
        Path: Validated Path object
        
    Raises:
        ValidationError: If path is invalid
        FileNotFoundError: If file doesn't exist when required
    """
    if path is None:
        raise ValidationError("Path cannot be None")
    
    if not isinstance(path, (str, Path)):
        raise ValidationError(f"Path must be string or Path object, got {type(path)}")
    
    try:
        path = Path(path)
    except Exception as e:
        raise ValidationError(f"Invalid path format: {e}")
    
    if must_exist and not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    if path.exists():
        if not path.is_file():
            raise ValidationError(f"Path exists but is not a file: {path}")
        
        
        if not path.is_readable():
            raise ValidationError(f"File is not readable: {path}")
    
    if check_writable:
        parent_dir = path.parent
        if not parent_dir.exists():
            try:
                parent_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise ValidationError(f"Cannot create directory {parent_dir}: {e}")
        
        if not parent_dir.is_writable():
            raise ValidationError(f"Directory is not writable: {parent_dir}")
    
    return path

def validate_memory_usage(n_vectors: int, dimensions: int, level: str = "f32") -> None:
    """
    Validate estimated memory usage and warn if excessive.
    
    Args:
        n_vectors: Number of vectors
        dimensions: Vector dimensions
        level: Precision level
        
    Raises:
        ValidationError: If memory usage would be excessive
    """
    
    bytes_per_element = {
        "f8": 1,
        "f16": 2,
        "f32": 4,
        "f64": 8,
    }
    
    if level not in bytes_per_element:
        raise ValidationError(f"Unknown precision level for memory calculation: {level}")
    
    
    base_memory = n_vectors * dimensions * bytes_per_element[level]
    
    
    overhead_factor = 1.5  
    total_memory = base_memory * overhead_factor
    
    
    WARNING_THRESHOLD = 2 * 1024**3  
    ERROR_THRESHOLD = 8 * 1024**3    
    
    if total_memory > ERROR_THRESHOLD:
        raise ValidationError(
            f"Estimated memory usage too high: {total_memory / (1024**3):.1f}GB "
            f"(limit: {ERROR_THRESHOLD / (1024**3):.1f}GB). "
            f"Consider using lower precision level or smaller dataset."
        )
    
    if total_memory > WARNING_THRESHOLD:
        logger.warning(
            f"High estimated memory usage: {total_memory / (1024**3):.1f}GB "
            f"({n_vectors:,} vectors × {dimensions} dims × {level})"
        )
    else:
        logger.info(
            f"Estimated memory usage: {total_memory / (1024**2):.1f}MB "
            f"({n_vectors:,} vectors × {dimensions} dims × {level})"
        )



def check_normalization(vectors: np.ndarray, tolerance: float = NORMALIZATION_TOLERANCE) -> tuple:
    """
    Check if vectors are normalized.
    
    Args:
        vectors: Input vectors
        tolerance: Tolerance for normalization check
        
    Returns:
        tuple: (is_normalized, percentage_normalized, mean_norm, std_norm)
    """
    norms = np.linalg.norm(vectors, axis=1)
    normalized_mask = np.abs(norms - 1.0) < tolerance
    percentage_normalized = np.mean(normalized_mask)
    
    return (
        percentage_normalized > 0.95,  
        percentage_normalized,
        np.mean(norms),
        np.std(norms)
    )

def validate_config_dict(config: dict, required_keys: List[str] = None, 
                        optional_keys: List[str] = None) -> dict:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary
        required_keys: Keys that must be present
        optional_keys: Keys that are allowed but optional
        
    Returns:
        dict: Validated configuration
        
    Raises:
        ValidationError: If configuration is invalid
    """
    if not isinstance(config, dict):
        raise ValidationError(f"Config must be a dictionary, got {type(config)}")
    
    required_keys = required_keys or []
    optional_keys = optional_keys or []
    allowed_keys = set(required_keys + optional_keys)
    
    
    missing_keys = set(required_keys) - set(config.keys())
    if missing_keys:
        raise ValidationError(f"Missing required config keys: {sorted(missing_keys)}")
    
    
    unknown_keys = set(config.keys()) - allowed_keys
    if unknown_keys:
        logger.warning(f"Unknown config keys (will be ignored): {sorted(unknown_keys)}")
    
    return config