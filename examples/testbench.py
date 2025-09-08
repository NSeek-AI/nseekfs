#!/usr/bin/env python3
"""
NSeekFS LinkedIn Benchmark
====================================================

Fair and representative benchmarks for LinkedIn publication:
- Exact Cosine Similarity Search
- Realistic datasets: 25K, 50K, 100K, 200K, 500K vectors
- Single Query vs Batch Query
- Most common Top-K values: 5, 10, 50
- Comparison with FAISS, scikit-learn, NumPy
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import psutil
import os
import sys

import nseekfs
import faiss
from sklearn.neighbors import NearestNeighbors

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def memory_mb():
    """Return current memory usage in MB"""
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

def timer_robust(func, *args, repeat=5, warmup=1, **kwargs):
    """Robust timer with warmup and multiple executions"""

    for _ in range(warmup):
        try:
            func(*args, **kwargs)
        except:
            pass
    
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            elapsed = (time.perf_counter() - start) * 1000  # ms
            times.append(elapsed)
        except Exception as e:
            return None, str(e)
    
    if len(times) >= 3:
        mean = np.mean(times)
        std = np.std(times)
        times = [t for t in times if abs(t - mean) <= 2 * std]
    
    return np.mean(times) if times else None, None

def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """Normalize vectors for cosine similarity"""
    if vectors.ndim == 1:
        return vectors / np.linalg.norm(vectors)
    else:
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

class BenchmarkEngine:
    """Main engine for professional benchmarks"""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.results = []
        
    def log(self, message):
        if self.verbose:
            print(f"📊 {message}")
    
    def benchmark_nseekfs(self, vectors: np.ndarray, queries: np.ndarray, top_k: int) -> Dict:
        """Benchmark NSeekFS with detailed info"""
        try:
            self.log("Testing NSeekFS...")
            
            build_start = time.perf_counter()
            index = nseekfs.from_embeddings(vectors, normalized=True, verbose=False)
            build_time = (time.perf_counter() - build_start) * 1000
            
            single_query = queries[0]
            single_time, error = timer_robust(lambda: index.query(single_query, top_k=top_k))
            if error:
                return {'success': False, 'error': f"Single query: {error}"}
            
            detailed_result = index.query_detailed(single_query, top_k=top_k)
            
            batch_time, error = timer_robust(lambda: index.query_batch(queries, top_k=top_k, format="simple"))
            if error:
                return {'success': False, 'error': f"Batch query: {error}"}
            
            return {
                'success': True,
                'library': 'NSeekFS',
                'build_time_ms': build_time,
                'single_query_ms': single_time,
                'batch_total_ms': batch_time,
                'batch_per_query_ms': batch_time / len(queries),
                'method_used': detailed_result.method_used,
                'simd_used': detailed_result.simd_used,
                'memory_mb': memory_mb()
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def benchmark_faiss(self, vectors: np.ndarray, queries: np.ndarray, top_k: int) -> Dict:
        """Benchmark FAISS"""
        try:
            self.log("Testing FAISS...")
            
            build_start = time.perf_counter()
            index = faiss.IndexFlatIP(vectors.shape[1])
            index.add(vectors.astype(np.float32))
            build_time = (time.perf_counter() - build_start) * 1000
            
            single_query = queries[0:1]  # FAISS expects 2D
            single_time, error = timer_robust(lambda: index.search(single_query, top_k))
            if error:
                return {'success': False, 'error': f"Single query: {error}"}
            
            batch_time, error = timer_robust(lambda: index.search(queries, top_k))
            if error:
                return {'success': False, 'error': f"Batch query: {error}"}
            
            return {
                'success': True,
                'library': 'FAISS',
                'build_time_ms': build_time,
                'single_query_ms': single_time,
                'batch_total_ms': batch_time,
                'batch_per_query_ms': batch_time / len(queries),
                'memory_mb': memory_mb()
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def benchmark_sklearn(self, vectors: np.ndarray, queries: np.ndarray, top_k: int) -> Dict:
        """Benchmark scikit-learn"""
        try:
            self.log("Testing scikit-learn...")
            
            build_start = time.perf_counter()
            nn = NearestNeighbors(n_neighbors=top_k, algorithm="brute", metric="cosine")
            nn.fit(vectors)
            build_time = (time.perf_counter() - build_start) * 1000
            
            single_query = queries[0:1]  # sklearn expects 2D
            single_time, error = timer_robust(lambda: nn.kneighbors(single_query))
            if error:
                return {'success': False, 'error': f"Single query: {error}"}
            
            batch_time, error = timer_robust(lambda: nn.kneighbors(queries))
            if error:
                return {'success': False, 'error': f"Batch query: {error}"}
            
            return {
                'success': True,
                'library': 'scikit-learn',
                'build_time_ms': build_time,
                'single_query_ms': single_time,
                'batch_total_ms': batch_time,
                'batch_per_query_ms': batch_time / len(queries),
                'memory_mb': memory_mb()
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def benchmark_numpy(self, vectors: np.ndarray, queries: np.ndarray, top_k: int) -> Dict:
        """Benchmark NumPy baseline"""
        try:
            self.log("Testing NumPy...")
            
            build_time = 0
            
            def numpy_single_search(query):
                similarities = vectors @ query
                top_indices = np.argpartition(-similarities, top_k)[:top_k]
                return top_indices[np.argsort(-similarities[top_indices])]
            
            def numpy_batch_search(queries):
                results = []
                for query in queries:
                    similarities = vectors @ query
                    top_indices = np.argpartition(-similarities, top_k)[:top_k]
                    results.append(top_indices[np.argsort(-similarities[top_indices])])
                return results
            
            single_time, error = timer_robust(lambda: numpy_single_search(queries[0]))
            if error:
                return {'success': False, 'error': f"Single query: {error}"}
            
            batch_time, error = timer_robust(lambda: numpy_batch_search(queries))
            if error:
                return {'success': False, 'error': f"Batch query: {error}"}
            
            return {
                'success': True,
                'library': 'NumPy',
                'build_time_ms': build_time,
                'single_query_ms': single_time,
                'batch_total_ms': batch_time,
                'batch_per_query_ms': batch_time / len(queries),
                'memory_mb': memory_mb()
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def run_scenario(self, n_vectors: int, dimensions: int, n_queries: int, top_k: int) -> Dict:
        """Run a full benchmark scenario"""
        
        print(f"\n🚀 BENCHMARK: {n_vectors:,} vectors × {dimensions}D, {n_queries} queries, top_k={top_k}")
        print("=" * 80)
        
        self.log("Generating normalized test data...")
        vectors = normalize_vectors(np.random.randn(n_vectors, dimensions).astype(np.float32))
        queries = normalize_vectors(np.random.randn(n_queries, dimensions).astype(np.float32))
        
        libraries = [
            ('NSeekFS', self.benchmark_nseekfs),
            ('FAISS', self.benchmark_faiss),
            ('scikit-learn', self.benchmark_sklearn),
            ('NumPy', self.benchmark_numpy)
        ]
        
        scenario_results = {
            'n_vectors': n_vectors,
            'dimensions': dimensions,
            'n_queries': n_queries,
            'top_k': top_k,
            'libraries': {}
        }
        
        for lib_name, benchmark_func in libraries:
            result = benchmark_func(vectors, queries, top_k)
            scenario_results['libraries'][lib_name] = result
            
            if result['success']:
                print(f"✅ {lib_name:12s}: Single={result['single_query_ms']:6.2f}ms | "
                      f"Batch={result['batch_per_query_ms']:6.2f}ms/q | "
                      f"Build={result['build_time_ms']:6.0f}ms | "
                      f"Mem={result['memory_mb']:5.0f}MB")
                
                if lib_name == 'NSeekFS' and 'method_used' in result:
                    print(f"    └─ Algorithm: {result['method_used']}, SIMD: {result['simd_used']}")
            else:
                print(f"❌ {lib_name:12s}: {result['error']}")
        
        self.results.append(scenario_results)
        return scenario_results
    
    def generate_summary_table(self) -> pd.DataFrame:
        """Generate a summary table of results"""
        data = []
        
        for scenario in self.results:
            for lib_name, result in scenario['libraries'].items():
                if result['success']:
                    data.append({
                        'Dataset': f"{scenario['n_vectors']/1000:.0f}K × {scenario['dimensions']}D",
                        'Library': lib_name,
                        'Top-K': scenario['top_k'],
                        'Single Query (ms)': result['single_query_ms'],
                        'Batch per Query (ms)': result['batch_per_query_ms'],
                        'Build Time (ms)': result['build_time_ms'],
                        'Memory (MB)': result['memory_mb'],
                        'Algorithm': result.get('method_used', 'N/A'),
                        'SIMD': result.get('simd_used', 'N/A')
                    })
        
        return pd.DataFrame(data)
    
    def create_visualization(self, save_path: str = "nseekfs_benchmark.png"):
        """Create professional visualization of results"""
        
        df = self.generate_summary_table()
        
        viz_df = df[df['Top-K'].isin([10, 50])].copy()
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('NSeekFS vs Popular Vector Search Libraries\nExact Cosine Similarity Search', 
                     fontsize=16, fontweight='bold')
        
        single_df = viz_df[viz_df['Top-K'] == 10]
        sns.barplot(data=single_df, x='Dataset', y='Single Query (ms)', 
                   hue='Library', ax=ax1)
        ax1.set_title('Single Query Performance (Top-K=10)', fontweight='bold')
        ax1.set_yscale('log')
        ax1.set_ylabel('Time (ms) - Log Scale')
        ax1.tick_params(axis='x', rotation=45)
        
        sns.barplot(data=single_df, x='Dataset', y='Batch per Query (ms)', 
                   hue='Library', ax=ax2)
        ax2.set_title('Batch Query Performance (Top-K=10)', fontweight='bold')
        ax2.set_yscale('log')
        ax2.set_ylabel('Time per Query (ms) - Log Scale')
        ax2.tick_params(axis='x', rotation=45)
        
        speedup_data = []
        for dataset in single_df['Dataset'].unique():
            dataset_data = single_df[single_df['Dataset'] == dataset]
            numpy_time = dataset_data[dataset_data['Library'] == 'NumPy']['Single Query (ms)'].iloc[0]
            
            for lib in ['NSeekFS', 'FAISS', 'scikit-learn']:
                lib_data = dataset_data[dataset_data['Library'] == lib]
                if not lib_data.empty:
                    lib_time = lib_data['Single Query (ms)'].iloc[0]
                    speedup = numpy_time / lib_time
                    speedup_data.append({
                        'Dataset': dataset,
                        'Library': lib,
                        'Speedup': speedup
                    })
        
        speedup_df = pd.DataFrame(speedup_data)
        sns.barplot(data=speedup_df, x='Dataset', y='Speedup', hue='Library', ax=ax3)
        ax3.set_title('Speedup vs NumPy (Single Query)', fontweight='bold')
        ax3.set_ylabel('Speedup Factor')
        ax3.tick_params(axis='x', rotation=45)
        ax3.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='NumPy Baseline')
        
        memory_df = viz_df.groupby(['Dataset', 'Library'])['Memory (MB)'].mean().reset_index()
        sns.barplot(data=memory_df, x='Dataset', y='Memory (MB)', hue='Library', ax=ax4)
        ax4.set_title('Memory Usage Comparison', fontweight='bold')
        ax4.set_ylabel('Memory (MB)')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 Visualization saved as: {save_path}")
        
        return fig
    
    def print_linkedin_summary(self):
        """Print formatted summary for LinkedIn"""
        print("\n" + "="*80)
        print("📈 LINKEDIN SUMMARY - NSeekFS Performance Results")
        print("="*80)
        
        df = self.generate_summary_table()
        
        categories = ['Single Query (ms)', 'Batch per Query (ms)']
        
        for category in categories:
            print(f"\n🏆 {category.replace(' (ms)', '')} Champions:")
            
            for dataset in df['Dataset'].unique():
                dataset_df = df[(df['Dataset'] == dataset) & (df['Top-K'] == 10)]
                if not dataset_df.empty:
                    best = dataset_df.loc[dataset_df[category].idxmin()]
                    print(f"  {dataset}: {best['Library']} ({best[category]:.2f}ms)")
        
        nseekfs_df = df[df['Library'] == 'NSeekFS']
        print(f"\n🚀 NSeekFS Highlights:")
        print(f"  • Algorithm Adaptivity: {nseekfs_df['Algorithm'].nunique()} different optimization paths")
        print(f"  • SIMD Acceleration: Active in {nseekfs_df['SIMD'].sum()}/{len(nseekfs_df)} tests")
        print(f"  • Avg Single Query: {nseekfs_df['Single Query (ms)'].mean():.2f}ms")
        print(f"  • Avg Batch per Query: {nseekfs_df['Batch per Query (ms)'].mean():.2f}ms")
        
        numpy_df = df[df['Library'] == 'NumPy']
        speedups = []
        for _, nseek_row in nseekfs_df.iterrows():
            numpy_row = numpy_df[
                (numpy_df['Dataset'] == nseek_row['Dataset']) & 
                (numpy_df['Top-K'] == nseek_row['Top-K'])
            ]
            if not numpy_row.empty:
                speedup = numpy_row['Single Query (ms)'].iloc[0] / nseek_row['Single Query (ms)']
                speedups.append(speedup)
        
        if speedups:
            print(f"  • Avg Speedup vs NumPy: {np.mean(speedups):.1f}x")

def main():
    """Main function to run LinkedIn benchmarks"""
    
    print("🚀 NSeekFS LinkedIn Benchmark Suite")
    print("====================================")
    print("Exact Cosine Similarity Search Comparison")
    print("Libraries: NSeekFS, FAISS, scikit-learn, NumPy\n")
    
    vector_sizes = [25_000, 50_000, 100_000, 200_000, 500_000]
    dimensions = 384  # Typical embedding dimension (BERT-like)
    n_queries = 50    # Reasonable number for stable timings
    top_k_values = [5, 10, 50]  # Most common values
    
    benchmark = BenchmarkEngine(verbose=True)
    
    for n_vectors in vector_sizes:
        for top_k in top_k_values:
            try:
                benchmark.run_scenario(n_vectors, dimensions, n_queries, top_k)
            except Exception as e:
                print(f"❌ Error in scenario {n_vectors}K, top_k={top_k}: {e}")
                continue
    
    print("\n" + "="*80)
    print("📊 GENERATING FINAL OUTPUTS...")
    print("="*80)
    
    df = benchmark.generate_summary_table()
    csv_path = "nseekfs_benchmark_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"✅ Results saved to: {csv_path}")
    
    try:
        benchmark.create_visualization("nseekfs_linkedin_benchmark.png")
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
    
    benchmark.print_linkedin_summary()
    
    print(f"\n🎉 Benchmark complete! Check the generated files for LinkedIn content.")

if __name__ == "__main__":
    main()
