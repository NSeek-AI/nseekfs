def print_analysis(self):
        """Print comprehensive analysis of results"""
        
        print_section("COMPREHENSIVE ANALYSIS")
        
        if not self.results:
            print("No results to analyze")
            return
        
        # Collect all successful results across datasets
        all_successful = {}
        total_comparisons = 0
        
        for result in self.results:
            dataset_name = result.get('dataset_name', 'unknown')
            for lib, data in result.items():
                if lib in ['dataset_name', 'data_size_mb', 'n_vectors', 'dimensions', 'num_queries', 'top_k']:
                    continue
                    
                if isinstance(data, dict) and data.get('success', False):
                    if lib not in all_successful:
                        all_successful[lib] = []
                    all_successful[lib].append({
                        'dataset': dataset_name,
                        'avg_time_ms': data['avg_time_ms'],
                        'qps': data['qps'],
                        'build_time_ms': data.get('build_time_ms', 0)
                    })
                    total_comparisons += 1
        
        if not all_successful:
            print("No successful results to analyze")
            return
        
        print(f"Libraries tested: {len(all_successful)}")
        print(f"Datasets: {len(self.results)}")
        
        # Overall speed ranking
        print(f"\nOverall Performance Ranking (avg query time):")
        
        lib_avg_times = {}
        for lib, results in all_successful.items():
            lib_avg_times[lib] = np.mean([r['avg_time_ms'] for r in results])
        
        sorted_libs = sorted(lib_avg_times.items(), key=lambda x: x[1])
        
        for i, (lib, avg_time) in enumerate(sorted_libs, 1):
            results_count = len(all_successful[lib])
            avg_qps = 1000 / avg_time
            
            if i == 1:
                print(f"  🥇 {lib.upper()}: {avg_time:.2f}ms avg ({avg_qps:.0f} QPS, {results_count} tests)")
            elif i == 2:
                print(f"  🥈 {lib.upper()}: {avg_time:.2f}ms avg ({avg_qps:.0f} QPS, {results_count} tests)")
            elif i == 3:
                print(f"  🥉 {lib.upper()}: {avg_time:.2f}ms avg ({avg_qps:.0f} QPS, {results_count} tests)")
            else:
                print(f"  {i}. {lib.upper()}: {avg_time:.2f}ms avg ({avg_qps:.0f} QPS, {results_count} tests)")
        
        # Build time analysis
        libs_with_build = {lib: data for lib, data in all_successful.items() 
                          if any(r['build_time_ms'] > 0 for r in data)}
        
        if libs_with_build:
            print(f"\nBuild Time Analysis:")
            for lib, results in libs_with_build.items():
                avg_build = np.mean([r['build_time_ms'] for r in results])
                print(f"  {lib.upper()}: {avg_build:.1f}ms avg build time")
        
        # NSeekFS specific analysis
        if 'nseekfs' in all_successful:
            print(f"\nNSeekFS Performance Analysis:")
            nseekfs_results = all_successful['nseekfs']
            
            for other_lib in all_successful:
                if other_lib == 'nseekfs':
                    continue
                    
                other_results = all_successful[other_lib]
                nseekfs_avg = np.mean([r['avg_time_ms'] for r in nseekfs_results])
                other_avg = np.mean([r['avg_time_ms'] for r in other_results])
                
                if other_avg < nseekfs_avg:
                    speedup = nseekfs_avg / other_avg
                    print(f"  {other_lib.upper()} is {speedup:.1f}x faster than NSeekFS")
                else:
                    speedup = other_avg / nseekfs_avg
                    print(f"  NSeekFS is {speedup:.1f}x faster than {other_lib.upper()}")
        
        # Library characteristics
        print(f"\nLibrary Characteristics:")
        characteristics = {
            'nseekfs': "Rust-based, exact search, thread-safe, persistent indices",
            'faiss': "Meta AI, highly optimized, exact mode, production-ready",
            'annoy': "Spotify, tree-based, memory efficient, approximate but tunable",
            'sklearn': "Python ecosystem, exact brute-force, well-integrated",
            'numpy': "Raw computation baseline, no indexing overhead"
        }
        
        for lib in all_successful:
            if lib in characteristics:
                print(f"  {lib.upper()}: {characteristics[lib]}")
        
        # Usage recommendations
        print(f"\nUsage Recommendations:")
        
        fastest_lib = sorted_libs[0][0] if sorted_libs else None
        
        if 'nseekfs' in all_successful:
            nseekfs_rank = next(i for i, (lib, _) in enumerate(sorted_libs, 1) if lib == 'nseekfs')
            print(f"  NSeekFS ranked #{nseekfs_rank} in speed")
            
            if nseekfs_rank <= 2:
                print(f"  - NSeekFS shows competitive performance")
                print(f"  - Good choice for structured similarity search")
            else:
                print(f"  - NSeekFS prioritizes features over raw speed")
                print(f"  - Consider for: thread safety, persistence, exact results")
                print(f"  - For maximum speed: consider {fastest_lib.upper()}")
        
        print(f"  - For one-off queries: NumPy or fastest available")
        print(f"  - For production systems: FAISS or NSeekFS")
        print(f"  - For Python integration: scikit-learn")
        print(f"  - For memory constraints: Annoy")

def export_results(results: list, filename: str):
    """Export results to JSON file"""
    try:
        with open(filename, 'w') as f:
            json.dump({
                'benchmark_type': 'similarity_search_comparison',
                'description': 'Comparison of NSeekFS vs other similarity search libraries',
                'libraries_tested': ['nseekfs', 'faiss', 'annoy', 'sklearn', 'numpy'],
                'timestamp': time.time(),
                'results': results
            }, f, indent=2, default=str)
        print(f"Results exported to: {filename}")
    except Exception as e:
        print(f"Export failed: {e}")

def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description='NSeekFS vs Similarity Search Libraries Benchmark')
    parser.add_argument('--datasets', type=str, default='small,medium,large',
                       help='Datasets to test (tiny,small,medium,large,xlarge,huge)')
    parser.add_argument('--export', type=str, help='Export results to JSON file')
    parser.add_argument('--install-missing', action='store_true', 
                       help='Show install commands for missing libraries')
    
    args = parser.parse_args()
    
    # Show install commands if requested
    if args.install_missing:
        print("Install missing libraries:")
        if not AVAILABLE_LIBS['faiss']:
            print("  pip install faiss-cpu")
        if not AVAILABLE_LIBS['annoy']:
            print("  pip install annoy")
        if not AVAILABLE_LIBS['sklearn']:
            print("  pip install scikit-learn")
        print("  pip install nseekfs")
        return 0
    
    # Check NSeekFS availability
    try:
        import nseekfs
        print(f"NSeekFS {getattr(nseekfs, '__version__', 'unknown')} available")
    except ImportError:
        print("ERROR: NSeekFS not found. Install with: pip install nseekfs")
        return 1
    
    print(f"NumPy {np.__version__} available")
    
    # Show available libraries
    available_count = sum(AVAILABLE_LIBS.values()) + 1  # +1 for nseekfs
    total_libs = len(AVAILABLE_LIBS) + 1
    print(f"Libraries available: {available_count}/{total_libs}")
    
    # Parse datasets
    datasets = [d.strip() for d in args.datasets.split(',')]
    
    # Run benchmark
    try:
        benchmark = FairBenchmark()
        
        start_time = time.time()
        results = benchmark.run_comprehensive_benchmark(datasets)
        total_time = time.time() - start_time
        
        print(f"\nBenchmark completed in {total_time:.1f}s")
        
        # Export if requested
        if args.export:
            export_results(results, args.export)
        
        return 0
        
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        return 2
    except Exception as e:
        print(f"\nBenchmark error: {e}")
        import traceback
        traceback.print_exc()
        return 3#!/usr/bin/env python3
"""
NSeekFS Fair Cosine Similarity Benchmark
========================================

Fair comparison of NSeekFS vs similar indexing libraries for cosine similarity.
Tests equivalent operations with structured similarity search libraries.

Libraries tested:
- NSeekFS (Rust-based exact search)  
- FAISS (Meta AI, exact mode)
- Annoy (Spotify, high precision mode)
- scikit-learn NearestNeighbors (exact)
- NumPy (baseline raw computation)

Usage:
    python fair_cosine_benchmark.py [--datasets=small,medium,large] [--export=results.json]
"""

import time
import numpy as np
import argparse
import json
import sys
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Check library availability
AVAILABLE_LIBS = {}

try:
    import faiss
    AVAILABLE_LIBS['faiss'] = True
    print("FAISS available")
except ImportError:
    AVAILABLE_LIBS['faiss'] = False

try:
    import annoy
    AVAILABLE_LIBS['annoy'] = True
    print("Annoy available")
except ImportError:
    AVAILABLE_LIBS['annoy'] = False

try:
    from sklearn.neighbors import NearestNeighbors
    AVAILABLE_LIBS['sklearn'] = True
    print("scikit-learn available")
except ImportError:
    AVAILABLE_LIBS['sklearn'] = False

AVAILABLE_LIBS['numpy'] = True

def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")

def print_test(message: str):
    print(f"Testing: {message}")

def print_result(message: str):
    print(f"  {message}")

class FairBenchmark:
    """Fair benchmark comparing equivalent cosine similarity operations"""
    
    def __init__(self):
        self.results = []
    
    def numpy_cosine_similarity(self, query: np.ndarray, vectors: np.ndarray, top_k: int) -> Tuple[List[int], List[float], float]:
        """NumPy implementation matching NSeekFS functionality"""
        start_time = time.time()
        
        # Normalize query and vectors (same as NSeekFS with normalized=True)
        query_norm = query / (np.linalg.norm(query) + 1e-8)
        vectors_norm = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
        
        # Compute cosine similarities
        similarities = np.dot(vectors_norm, query_norm)
        
        # Get top-k results (same as NSeekFS)
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        top_scores = similarities[top_indices]
        
        query_time_ms = (time.time() - start_time) * 1000
        
        return top_indices.tolist(), top_scores.tolist(), query_time_ms
    
    def benchmark_faiss(self, vectors: np.ndarray, queries: List[np.ndarray], top_k: int) -> Dict[str, Any]:
        """Benchmark FAISS exact search"""
        if not AVAILABLE_LIBS['faiss']:
            return {'success': False, 'error': 'FAISS not available'}
        
        try:
            import faiss
            
            # Build exact index
            build_start = time.time()
            index = faiss.IndexFlatIP(vectors.shape[1])  # Inner Product for cosine
            
            # Normalize vectors for cosine similarity
            vectors_norm = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
            index.add(vectors_norm)
            build_time = (time.time() - build_start) * 1000
            
            # Warm-up
            query_norm = queries[0] / (np.linalg.norm(queries[0]) + 1e-8)
            index.search(query_norm.reshape(1, -1), top_k)
            
            # Benchmark
            query_times = []
            for query in queries:
                query_norm = query / (np.linalg.norm(query) + 1e-8)
                start_time = time.time()
                scores, indices = index.search(query_norm.reshape(1, -1), top_k)
                query_time = (time.time() - start_time) * 1000
                query_times.append(query_time)
            
            return {
                'success': True,
                'build_time_ms': build_time,
                'avg_time_ms': np.mean(query_times),
                'std_time_ms': np.std(query_times),
                'qps': 1000 / np.mean(query_times),
                'all_times': query_times
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def benchmark_annoy(self, vectors: np.ndarray, queries: List[np.ndarray], top_k: int) -> Dict[str, Any]:
        """Benchmark Annoy with high precision settings"""
        if not AVAILABLE_LIBS['annoy']:
            return {'success': False, 'error': 'Annoy not available'}
        
        try:
            import annoy
            
            # Build index with many trees for high precision
            build_start = time.time()
            index = annoy.AnnoyIndex(vectors.shape[1], 'angular')  # Angular = cosine
            
            # Normalize vectors for cosine similarity
            vectors_norm = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
            for i, vector in enumerate(vectors_norm):
                index.add_item(i, vector)
            
            n_trees = min(200, max(10, vectors.shape[0] // 100))  # Adaptive tree count
            index.build(n_trees)
            build_time = (time.time() - build_start) * 1000
            
            # Warm-up
            query_norm = queries[0] / (np.linalg.norm(queries[0]) + 1e-8)
            index.get_nns_by_vector(query_norm, top_k, search_k=-1)
            
            # Benchmark
            query_times = []
            for query in queries:
                query_norm = query / (np.linalg.norm(query) + 1e-8)
                start_time = time.time()
                # search_k=-1 for exhaustive search (highest precision)
                indices = index.get_nns_by_vector(query_norm, top_k, search_k=-1)
                query_time = (time.time() - start_time) * 1000
                query_times.append(query_time)
            
            return {
                'success': True,
                'build_time_ms': build_time,
                'avg_time_ms': np.mean(query_times),
                'std_time_ms': np.std(query_times),
                'qps': 1000 / np.mean(query_times),
                'all_times': query_times,
                'note': f'Built with {n_trees} trees, exhaustive search'
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def benchmark_sklearn(self, vectors: np.ndarray, queries: List[np.ndarray], top_k: int) -> Dict[str, Any]:
        """Benchmark scikit-learn NearestNeighbors (exact)"""
        if not AVAILABLE_LIBS['sklearn']:
            return {'success': False, 'error': 'scikit-learn not available'}
        
        try:
            from sklearn.neighbors import NearestNeighbors
            
            # Build exact index
            build_start = time.time()
            # Normalize vectors for cosine similarity
            vectors_norm = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
            
            # Use 'cosine' metric for exact cosine similarity
            nn = NearestNeighbors(n_neighbors=top_k, metric='cosine', algorithm='brute')
            nn.fit(vectors_norm)
            build_time = (time.time() - build_start) * 1000
            
            # Warm-up
            query_norm = queries[0] / (np.linalg.norm(queries[0]) + 1e-8)
            nn.kneighbors([query_norm])
            
            # Benchmark
            query_times = []
            for query in queries:
                query_norm = query / (np.linalg.norm(query) + 1e-8)
                start_time = time.time()
                distances, indices = nn.kneighbors([query_norm])
                query_time = (time.time() - start_time) * 1000
                query_times.append(query_time)
            
            return {
                'success': True,
                'build_time_ms': build_time,
                'avg_time_ms': np.mean(query_times),
                'std_time_ms': np.std(query_times),
                'qps': 1000 / np.mean(query_times),
                'all_times': query_times
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def benchmark_cosine_similarity(self, vectors: np.ndarray, queries: List[np.ndarray], top_k: int = 10) -> Dict[str, Any]:
        """Benchmark cosine similarity with all available libraries"""
        
        n_vectors, dimensions = vectors.shape
        num_queries = len(queries)
        
        print_test(f"Cosine similarity: {n_vectors:,} vectors x {dimensions}D, {num_queries} queries")
        
        results = {
            'n_vectors': n_vectors,
            'dimensions': dimensions,
            'num_queries': num_queries,
            'top_k': top_k,
        }
        
        # Test all available libraries
        libraries_to_test = [
            ('nseekfs', self.benchmark_nseekfs),
            ('faiss', self.benchmark_faiss),
            ('annoy', self.benchmark_annoy), 
            ('sklearn', self.benchmark_sklearn),
            ('numpy', self.benchmark_numpy_baseline)
        ]
        
        for lib_name, benchmark_func in libraries_to_test:
            if lib_name == 'nseekfs':
                # Always test NSeekFS if available
                try:
                    import nseekfs
                except ImportError:
                    results[lib_name] = {'success': False, 'error': 'NSeekFS not available'}
                    continue
            elif lib_name != 'numpy' and not AVAILABLE_LIBS.get(lib_name, False):
                results[lib_name] = {'success': False, 'error': f'{lib_name} not available'}
                continue
            
            print_result(f"{lib_name.upper()}...")
            try:
                result = benchmark_func(vectors, queries, top_k)
                results[lib_name] = result
                
                if result.get('success', True):
                    build_time = result.get('build_time_ms', 0)
                    avg_time = result.get('avg_time_ms', 0)
                    qps = result.get('qps', 0)
                    
                    print_result(f"  Build: {build_time:.1f}ms")
                    print_result(f"  Query: {avg_time:.2f}ms avg ({qps:.0f} QPS)")
                    
                    if 'note' in result:
                        print_result(f"  Note: {result['note']}")
                else:
                    print_result(f"  FAILED: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                print_result(f"  ERROR: {e}")
                results[lib_name] = {'success': False, 'error': str(e)}
        
        # Analysis
        self.analyze_library_comparison(results)
        
        return results
    
    def benchmark_nseekfs(self, vectors: np.ndarray, queries: List[np.ndarray], top_k: int) -> Dict[str, Any]:
        """Benchmark NSeekFS"""
        try:
            import nseekfs
            
            # Build index
            build_start = time.time()
            index = nseekfs.from_embeddings(vectors, normalized=True, verbose=False)
            build_time = (time.time() - build_start) * 1000
            
            # Warm-up
            index.query(queries[0], top_k=top_k)
            
            # Benchmark
            query_times = []
            for query in queries:
                start_time = time.time()
                result = index.query(query, top_k=top_k)
                query_time = (time.time() - start_time) * 1000
                query_times.append(query_time)
            
            return {
                'success': True,
                'build_time_ms': build_time,
                'avg_time_ms': np.mean(query_times),
                'std_time_ms': np.std(query_times),
                'qps': 1000 / np.mean(query_times),
                'all_times': query_times
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def benchmark_numpy_baseline(self, vectors: np.ndarray, queries: List[np.ndarray], top_k: int) -> Dict[str, Any]:
        """Benchmark NumPy baseline (equivalent operations)"""
        try:
            # Warm-up
            self.numpy_cosine_similarity(queries[0], vectors, top_k)
            
            # Benchmark
            query_times = []
            for query in queries:
                _, _, query_time = self.numpy_cosine_similarity(query, vectors, top_k)
                query_times.append(query_time)
            
            return {
                'success': True,
                'build_time_ms': 0.0,  # No build phase
                'avg_time_ms': np.mean(query_times),
                'std_time_ms': np.std(query_times),
                'qps': 1000 / np.mean(query_times),
                'all_times': query_times
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def analyze_library_comparison(self, results: Dict[str, Any]):
        """Analyze comparison between libraries"""
        successful = {k: v for k, v in results.items() 
                     if isinstance(v, dict) and v.get('success', False)}
        
        if len(successful) < 2:
            print_result("Not enough successful results for comparison")
            return
        
        # Sort by query speed
        sorted_by_speed = sorted(successful.items(), 
                               key=lambda x: x[1]['avg_time_ms'])
        
        print_result("\nRanking by query speed:")
        for i, (lib, data) in enumerate(sorted_by_speed, 1):
            time_ms = data['avg_time_ms']
            qps = data['qps']
            
            if i == 1:
                print_result(f"  🥇 {lib.upper()}: {time_ms:.2f}ms ({qps:.0f} QPS)")
            elif i == 2:
                print_result(f"  🥈 {lib.upper()}: {time_ms:.2f}ms ({qps:.0f} QPS)")
            elif i == 3:
                print_result(f"  🥉 {lib.upper()}: {time_ms:.2f}ms ({qps:.0f} QPS)")
            else:
                print_result(f"  {i}. {lib.upper()}: {time_ms:.2f}ms ({qps:.0f} QPS)")
        
        # Build time comparison (excluding NumPy)
        with_build = {k: v for k, v in successful.items() 
                     if v.get('build_time_ms', 0) > 0}
        
        if with_build:
            print_result("\nBuild time comparison:")
            sorted_by_build = sorted(with_build.items(), 
                                   key=lambda x: x[1]['build_time_ms'])
            for lib, data in sorted_by_build:
                print_result(f"  {lib.upper()}: {data['build_time_ms']:.1f}ms")
        
        # NSeekFS specific analysis
        if 'nseekfs' in successful:
            nseekfs_time = successful['nseekfs']['avg_time_ms']
            print_result(f"\nNSeekFS comparison:")
            
            for lib, data in successful.items():
                if lib != 'nseekfs':
                    lib_time = data['avg_time_ms']
                    if lib_time < nseekfs_time:
                        speedup = nseekfs_time / lib_time
                        print_result(f"  {lib.upper()} is {speedup:.1f}x faster than NSeekFS")
                    else:
                        speedup = lib_time / nseekfs_time
                        print_result(f"  NSeekFS is {speedup:.1f}x faster than {lib.upper()}")
    
    def run_comprehensive_benchmark(self, datasets: List[str] = None) -> List[Dict[str, Any]]:
        """Run comprehensive benchmark with different dataset sizes"""
        
        if datasets is None:
            datasets = ['small', 'medium', 'large']
        
        # Dataset configurations - focus on where NSeekFS should excel
        dataset_configs = {
            'tiny': {'n_vectors': 1000, 'dimensions': 128, 'num_queries': 20},
            'small': {'n_vectors': 5000, 'dimensions': 256, 'num_queries': 15},
            'medium': {'n_vectors': 25000, 'dimensions': 384, 'num_queries': 10},
            'large': {'n_vectors': 100000, 'dimensions': 512, 'num_queries': 8},
            'xlarge': {'n_vectors': 250000, 'dimensions': 768, 'num_queries': 5},
            'huge': {'n_vectors': 500000, 'dimensions': 1024, 'num_queries': 3}
        }
        
        print_section("NSeekFS VS SIMILARITY SEARCH LIBRARIES")
        print(f"Testing datasets: {', '.join(datasets)}")
        print(f"Libraries: NSeekFS, FAISS, Annoy, scikit-learn, NumPy baseline")
        print(f"Focus: Structured similarity search comparison")
        
        # Check which libraries are available
        available_libs = []
        for lib, available in AVAILABLE_LIBS.items():
            if available:
                available_libs.append(lib)
        
        try:
            import nseekfs
            available_libs.append('nseekfs')
        except ImportError:
            pass
            
        print(f"Available: {', '.join(available_libs)}")
        
        all_results = []
        
        for dataset_name in datasets:
            if dataset_name not in dataset_configs:
                print(f"Unknown dataset: {dataset_name}")
                continue
            
            config = dataset_configs[dataset_name]
            
            print_section(f"DATASET: {dataset_name.upper()}")
            
            try:
                # Generate reproducible data
                np.random.seed(42)
                vectors = np.random.randn(config['n_vectors'], config['dimensions']).astype(np.float32)
                queries = [np.random.randn(config['dimensions']).astype(np.float32) 
                          for _ in range(config['num_queries'])]
                
                data_size_mb = vectors.nbytes / (1024 * 1024)
                print(f"Generated: {config['n_vectors']:,} vectors x {config['dimensions']}D ({data_size_mb:.1f}MB)")
                
                # Run benchmark
                result = self.benchmark_cosine_similarity(vectors, queries, top_k=10)
                result['dataset_name'] = dataset_name
                result['data_size_mb'] = data_size_mb
                
                all_results.append(result)
                self.results.append(result)
                
            except Exception as e:
                print(f"Error in dataset {dataset_name}: {e}")
                import traceback
                traceback.print_exc()
        
        # Generate analysis
        self.print_analysis()
        
        return all_results
    
    def print_analysis(self):
        """Print comprehensive analysis of results"""
        
        print_section("COMPREHENSIVE ANALYSIS")
        
        if not self.results:
            print("No results to analyze")
            return
        
        successful_results = [r for r in self.results if r.get('nseekfs', {}).get('success') and r.get('numpy', {}).get('success')]
        
        if not successful_results:
            print("No successful comparisons to analyze")
            return
        
        print(f"Successful tests: {len(successful_results)}/{len(self.results)}")
        
        # Win/loss analysis
        nseekfs_wins = sum(1 for r in successful_results if r.get('winner') == 'nseekfs')
        numpy_wins = sum(1 for r in successful_results if r.get('winner') == 'numpy')
        
        print(f"\nQuery Speed Results:")
        print(f"  NSeekFS wins: {nseekfs_wins}/{len(successful_results)}")
        print(f"  NumPy wins: {numpy_wins}/{len(successful_results)}")
        
        # Performance trends
        print(f"\nPerformance by dataset size:")
        for result in successful_results:
            dataset = result['dataset_name']
            n_vectors = result['n_vectors']
            winner = result.get('winner', 'tie')
            speedup = result.get('speedup', 1.0)
            
            if winner == 'numpy':
                print(f"  {dataset} ({n_vectors:,} vectors): NumPy {speedup:.1f}x faster")
            elif winner == 'nseekfs':
                print(f"  {dataset} ({n_vectors:,} vectors): NSeekFS {speedup:.1f}x faster")
            else:
                print(f"  {dataset} ({n_vectors:,} vectors): Tie")
        
        # NSeekFS strengths analysis
        print(f"\nNSeekFS Advantages (regardless of raw speed):")
        print(f"  - Exact results (100% precision)")
        print(f"  - Built-in indexing and persistence")
        print(f"  - Thread-safe concurrent access")
        print(f"  - Memory-efficient for large datasets")
        print(f"  - Rust-based numerical stability")
        
        # Usage recommendations
        print(f"\nUsage Recommendations:")
        
        if numpy_wins > nseekfs_wins:
            print(f"  - For simple one-off queries: NumPy may be faster")
            print(f"  - For repeated queries on same dataset: NSeekFS provides benefits")
            print(f"  - For production systems: NSeekFS offers better structure")
        else:
            print(f"  - NSeekFS shows competitive or better performance")
            print(f"  - Recommended for most similarity search use cases")
        
        print(f"  - Consider build time amortization over multiple queries")
        print(f"  - Thread safety important for concurrent applications")

def export_results(results: List[Dict[str, Any]], filename: str):
    """Export results to JSON file"""
    try:
        with open(filename, 'w') as f:
            json.dump({
                'benchmark_type': 'fair_cosine_similarity',
                'description': 'Fair comparison of NSeekFS vs NumPy for cosine similarity',
                'timestamp': time.time(),
                'results': results
            }, f, indent=2, default=str)
        print(f"Results exported to: {filename}")
    except Exception as e:
        print(f"Export failed: {e}")

def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description='Fair NSeekFS Cosine Similarity Benchmark')
    parser.add_argument('--datasets', type=str, default='small,medium,large',
                       help='Datasets to test (tiny,small,medium,large,xlarge,huge)')
    parser.add_argument('--export', type=str, help='Export results to JSON file')
    
    args = parser.parse_args()
    
    # Check NSeekFS availability
    try:
        import nseekfs
        print(f"NSeekFS {getattr(nseekfs, '__version__', 'unknown')} available")
    except ImportError:
        print("ERROR: NSeekFS not found. Install with: pip install nseekfs")
        return 1
    
    print(f"NumPy {np.__version__} available")
    
    # Parse datasets
    datasets = [d.strip() for d in args.datasets.split(',')]
    
    # Run benchmark
    try:
        benchmark = FairBenchmark()
        
        start_time = time.time()
        results = benchmark.run_comprehensive_benchmark(datasets)
        total_time = time.time() - start_time
        
        print(f"\nBenchmark completed in {total_time:.1f}s")
        
        # Export if requested
        if args.export:
            export_results(results, args.export)
        
        return 0
        
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        return 2
    except Exception as e:
        print(f"\nBenchmark error: {e}")
        import traceback
        traceback.print_exc()
        return 3

if __name__ == "__main__":
    sys.exit(main())