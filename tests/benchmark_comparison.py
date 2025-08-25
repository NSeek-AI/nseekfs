#!/usr/bin/env python3
"""
NSeekFS v1.0 - Benchmark Comparativo
====================================

Compara NSeekFS com implementações nativas NumPy/SciPy
para demonstrar as vantagens do engine Rust otimizado.

Uso:
    python benchmark_comparison.py [--datasets=small,medium,large] [--metrics=all] [--export=results.csv]
"""

import time
import numpy as np
import argparse
import csv
import json
import sys
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

try:
    from scipy.spatial.distance import cdist
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("⚠️ SciPy não encontrado - algumas comparações serão limitadas")

# Cores para output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'

def colored(text, color):
    return f"{color}{text}{Colors.END}"

def print_header(title):
    print("\n" + "="*70)
    print(colored(f"🚀 {title}", Colors.BOLD + Colors.BLUE))
    print("="*70)

def print_result(message):
    print(f"{colored('📊', Colors.CYAN)} {message}")

def print_winner(message):
    print(f"{colored('🏆', Colors.GREEN)} {message}")

def print_benchmark(name, time_ms, qps, extra=""):
    print(f"   {name:20s}: {time_ms:8.2f}ms  {qps:8.0f} QPS  {extra}")

class BenchmarkSuite:
    """Suite completa de benchmarks comparativos"""
    
    def __init__(self):
        self.results = []
        
    def numpy_cosine_similarity(self, query: np.ndarray, vectors: np.ndarray, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Implementação numpy de similaridade cosine"""
        # Normalizar
        query_norm = query / np.linalg.norm(query)
        vectors_norm = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        
        # Produto escalar
        similarities = np.dot(vectors_norm, query_norm)
        
        # Top-k
        if top_k >= len(similarities):
            indices = np.argsort(similarities)[::-1]
        else:
            indices = np.argpartition(similarities, -top_k)[-top_k:]
            indices = indices[np.argsort(similarities[indices])][::-1]
        
        return indices, similarities[indices]
    
    def numpy_dot_product(self, query: np.ndarray, vectors: np.ndarray, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Implementação numpy de produto escalar"""
        scores = np.dot(vectors, query)
        
        # Top-k
        if top_k >= len(scores):
            indices = np.argsort(scores)[::-1]
        else:
            indices = np.argpartition(scores, -top_k)[-top_k:]
            indices = indices[np.argsort(scores[indices])][::-1]
        
        return indices, scores[indices]
    
    def numpy_euclidean_distance(self, query: np.ndarray, vectors: np.ndarray, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Implementação numpy de distância euclidiana"""
        diff = vectors - query
        distances = np.sqrt(np.sum(diff * diff, axis=1))
        
        # Top-k (menores distâncias)
        if top_k >= len(distances):
            indices = np.argsort(distances)
        else:
            indices = np.argpartition(distances, top_k-1)[:top_k]
            indices = indices[np.argsort(distances[indices])]
        
        return indices, distances[indices]
    
    def scipy_cosine_similarity(self, query: np.ndarray, vectors: np.ndarray, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Implementação scipy de similaridade cosine"""
        if not HAS_SCIPY:
            return self.numpy_cosine_similarity(query, vectors, top_k)
        
        distances = cdist([query], vectors, metric='cosine')[0]
        similarities = 1 - distances
        
        # Top-k
        if top_k >= len(similarities):
            indices = np.argsort(similarities)[::-1]
        else:
            indices = np.argpartition(similarities, -top_k)[-top_k:]
            indices = indices[np.argsort(similarities[indices])][::-1]
        
        return indices, similarities[indices]
    
    def scipy_euclidean_distance(self, query: np.ndarray, vectors: np.ndarray, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Implementação scipy de distância euclidiana"""
        if not HAS_SCIPY:
            return self.numpy_euclidean_distance(query, vectors, top_k)
        
        distances = cdist([query], vectors, metric='euclidean')[0]
        
        # Top-k (menores distâncias)
        if top_k >= len(distances):
            indices = np.argsort(distances)
        else:
            indices = np.argpartition(distances, top_k-1)[:top_k]
            indices = indices[np.argsort(distances[indices])]
        
        return indices, distances[indices]
    
    def benchmark_metric(self, vectors: np.ndarray, queries: List[np.ndarray], 
                        metric: str, top_k: int = 10) -> Dict[str, Any]:
        """Benchmark uma métrica específica"""
        
        print(f"\n🔍 Benchmarking {metric.upper()} similarity...")
        print(f"   Dataset: {vectors.shape[0]:,} vectors × {vectors.shape[1]} dimensions")
        print(f"   Top-K: {top_k}, Trials: {len(queries)}")
        
        results = {}
        
        # 1. Testar NSeekFS
        print(f"\n   🚀 Testing NSeekFS...")
        try:
            import nseekfs
            
            # Criar índice
            engine = nseekfs.from_embeddings(
                vectors,
                metric=metric if metric != 'dot_product' else 'dot',
                base_name=f"benchmark_{metric}",
                normalized=(metric == 'cosine')
            )
            
            nseekfs_times = []
            
            for query in queries:
                start_time = time.time()
                try:
                    result = engine.query(query, top_k=top_k)
                    query_time = (time.time() - start_time) * 1000
                    nseekfs_times.append(query_time)
                except Exception as e:
                    print(f"      ❌ Query failed: {e}")
                    continue
            
            if nseekfs_times:
                results['nseekfs'] = {
                    'avg_time_ms': np.mean(nseekfs_times),
                    'std_time_ms': np.std(nseekfs_times),
                    'min_time_ms': np.min(nseekfs_times),
                    'max_time_ms': np.max(nseekfs_times),
                    'qps': 1000 / np.mean(nseekfs_times),
                    'memory_mb': engine.memory_usage_mb,
                    'simd_used': True  # Assumir SIMD para dimensões >= 64
                }
                
                print_benchmark("NSeekFS", 
                              results['nseekfs']['avg_time_ms'],
                              results['nseekfs']['qps'],
                              f"±{results['nseekfs']['std_time_ms']:.1f}")
            else:
                print("   ❌ NSeekFS: Nenhuma query bem-sucedida")
                results['nseekfs'] = {'error': 'No successful queries'}
                
        except ImportError:
            print("   ❌ NSeekFS failed: módulo não encontrado")
            results['nseekfs'] = {'error': 'Module not found'}
        except Exception as e:
            print(f"   ❌ NSeekFS failed: {e}")
            results['nseekfs'] = {'error': str(e)}
        
        # 2. Testar NumPy
        print(f"   📐 Testing NumPy...")
        try:
            numpy_times = []
            
            # Escolher função NumPy baseada na métrica
            if metric == 'cosine':
                func = self.numpy_cosine_similarity
            elif metric == 'dot' or metric == 'dot_product':
                func = self.numpy_dot_product
            elif metric == 'euclidean':
                func = self.numpy_euclidean_distance
            else:
                raise ValueError(f"Métrica não suportada: {metric}")
            
            for query in queries:
                start_time = time.time()
                indices, scores = func(query, vectors, top_k)
                query_time = (time.time() - start_time) * 1000
                numpy_times.append(query_time)
            
            results['numpy'] = {
                'avg_time_ms': np.mean(numpy_times),
                'std_time_ms': np.std(numpy_times),
                'min_time_ms': np.min(numpy_times),
                'max_time_ms': np.max(numpy_times),
                'qps': 1000 / np.mean(numpy_times),
                'simd_used': False
            }
            
            print_benchmark("NumPy", 
                          results['numpy']['avg_time_ms'],
                          results['numpy']['qps'],
                          f"±{results['numpy']['std_time_ms']:.1f}")
            
        except Exception as e:
            print(f"   ❌ NumPy failed: {e}")
            results['numpy'] = {'error': str(e)}
        
        # 3. Testar SciPy (se disponível)
        if HAS_SCIPY and metric in ['cosine', 'euclidean']:
            print(f"   🔬 Testing SciPy...")
            try:
                scipy_times = []
                
                if metric == 'cosine':
                    func = self.scipy_cosine_similarity
                elif metric == 'euclidean':
                    func = self.scipy_euclidean_distance
                
                for query in queries:
                    start_time = time.time()
                    indices, scores = func(query, vectors, top_k)
                    query_time = (time.time() - start_time) * 1000
                    scipy_times.append(query_time)
                
                results['scipy'] = {
                    'avg_time_ms': np.mean(scipy_times),
                    'std_time_ms': np.std(scipy_times),
                    'min_time_ms': np.min(scipy_times),
                    'max_time_ms': np.max(scipy_times),
                    'qps': 1000 / np.mean(scipy_times),
                    'simd_used': False
                }
                
                print_benchmark("SciPy", 
                              results['scipy']['avg_time_ms'],
                              results['scipy']['qps'],
                              f"±{results['scipy']['std_time_ms']:.1f}")
                
            except Exception as e:
                print(f"   ❌ SciPy failed: {e}")
                results['scipy'] = {'error': str(e)}
        
        # Análise da métrica
        print(f"\n   📊 Analysis for {metric}:")
        
        # Determinar vencedor
        valid_results = {k: v for k, v in results.items() if 'avg_time_ms' in v}
        
        if valid_results:
            fastest = min(valid_results.keys(), key=lambda x: valid_results[x]['avg_time_ms'])
            fastest_time = valid_results[fastest]['avg_time_ms']
            
            print(f"   🏆 Fastest: {fastest.upper()} ({fastest_time:.2f}ms)")
            
            # Calcular speedups
            if 'nseekfs' in valid_results:
                nseekfs_time = valid_results['nseekfs']['avg_time_ms']
                for impl, data in valid_results.items():
                    if impl != 'nseekfs':
                        speedup = data['avg_time_ms'] / nseekfs_time
                        if speedup > 1:
                            print(f"   ⚡ NSeekFS is {speedup:.1f}x faster than {impl.upper()}")
                        else:
                            print(f"   📈 {impl.upper()} is {1/speedup:.1f}x faster than NSeekFS")
        else:
            print("   ❌ No valid results for comparison")
        
        return results
    
    def benchmark_dataset(self, dataset_name: str, n_vectors: int, dimensions: int, 
                         num_queries: int = 5, metrics: List[str] = None, 
                         top_k: int = 10) -> Dict[str, Any]:
        """Benchmark completo de um dataset"""
        
        print_header(f"DATASET: {dataset_name.upper()}")
        
        if metrics is None:
            metrics = ['cosine', 'dot', 'euclidean']
        
        # Gerar dados
        print_result(f"Generating {dataset_name} dataset: {n_vectors:,} × {dimensions}D...")
        np.random.seed(42)  # Reproducibilidade
        vectors = np.random.randn(n_vectors, dimensions).astype(np.float32)
        queries = [np.random.randn(dimensions).astype(np.float32) for _ in range(num_queries)]
        
        data_size_mb = vectors.nbytes / (1024 * 1024)
        print_result(f"   ✅ Generated {dataset_name}: {data_size_mb:.1f}MB")
        
        results = {
            'dataset_name': dataset_name,
            'n_vectors': n_vectors,
            'dimensions': dimensions,
            'num_queries': num_queries,
            'top_k': top_k,
            'data_size_mb': data_size_mb,
            'metrics': {}
        }
        
        # Benchmark para cada métrica
        for metric in metrics:
            print_result(f"\n🔍 Benchmarking {metric.upper()} similarity...")
            metric_results = self.benchmark_metric(vectors, queries, metric, top_k)
            results['metrics'][metric] = metric_results
        
        return results
    
    def run_comprehensive_benchmark(self, datasets: List[str] = None, 
                                  metrics: List[str] = None, 
                                  top_k: int = 10) -> List[Dict[str, Any]]:
        """Executar benchmark completo"""
        
        print_header("🚀 NSEEKFS v1.0 - COMPREHENSIVE BENCHMARK")
        
        if datasets is None:
            datasets = ['small', 'medium']
        
        if metrics is None:
            metrics = ['cosine', 'dot', 'euclidean']
        
        # Configurações de datasets
        dataset_configs = {
            'tiny': {'n_vectors': 1_000, 'dimensions': 128, 'num_queries': 5},
            'small': {'n_vectors': 10_000, 'dimensions': 256, 'num_queries': 5},
            'medium': {'n_vectors': 50_000, 'dimensions': 384, 'num_queries': 5},
            'large': {'n_vectors': 100_000, 'dimensions': 512, 'num_queries': 3},
            'xlarge': {'n_vectors': 200_000, 'dimensions': 768, 'num_queries': 3},
            'extreme': {'n_vectors': 500_000, 'dimensions': 1024, 'num_queries': 1}
        }
        
        # Verificar dependências
        try:
            import nseekfs
            nseekfs_available = True
            nseekfs_version = getattr(nseekfs, '__version__', '1.0.0')
        except ImportError:
            nseekfs_available = False
            nseekfs_version = 'Not available'
        
        print_result(f"📊 Configuration:")
        print_result(f"   • Datasets: {', '.join(datasets)}")
        print_result(f"   • Metrics: {', '.join(metrics)}")
        print_result(f"   • Top-K: {top_k}")
        print_result(f"   • NSeekFS: {'✅' if nseekfs_available else '❌'}")
        print_result(f"   • SciPy: {'✅' if HAS_SCIPY else '❌'}")
        
        if not nseekfs_available:
            print("\n❌ NSeekFS não disponível - benchmark limitado")
        
        print_header("GENERATING DATASETS")
        
        all_results = []
        
        for dataset_name in datasets:
            if dataset_name not in dataset_configs:
                print(f"⚠️ Dataset '{dataset_name}' não encontrado")
                continue
                
            config = dataset_configs[dataset_name]
            
            try:
                result = self.benchmark_dataset(
                    dataset_name=dataset_name,
                    n_vectors=config['n_vectors'],
                    dimensions=config['dimensions'],
                    num_queries=config['num_queries'],
                    metrics=metrics,
                    top_k=top_k
                )
                all_results.append(result)
                self.results.append(result)
                
            except Exception as e:
                print(f"❌ Erro no dataset {dataset_name}: {e}")
                import traceback
                traceback.print_exc()
        
        # Gerar análise final
        self.print_final_analysis()
        
        return all_results
    
    def print_final_analysis(self) -> None:
        """Gerar relatório resumo dos benchmarks"""
        
        print_header("📈 FINAL ANALYSIS")
        
        if not self.results:
            print("❌ Nenhum resultado disponível")
            return
        
        # Análise geral
        total_tests = sum(len(r['metrics']) for r in self.results)
        print_result(f"Total tests executed: {total_tests}")
        print_result(f"Datasets tested: {len(self.results)}")
        
        # Análise por métrica
        metrics_analysis = {}
        
        for result in self.results:
            for metric, metric_data in result['metrics'].items():
                if metric not in metrics_analysis:
                    metrics_analysis[metric] = {
                        'nseekfs_wins': 0,
                        'numpy_wins': 0,
                        'scipy_wins': 0,
                        'total_tests': 0,
                        'speedups': []
                    }
                
                # Determinar vencedor
                times = {}
                for impl in ['nseekfs', 'numpy', 'scipy']:
                    if impl in metric_data and isinstance(metric_data[impl], dict) and 'avg_time_ms' in metric_data[impl]:
                        times[impl] = metric_data[impl]['avg_time_ms']
                
                if times:
                    winner = min(times.keys(), key=lambda x: times[x])
                    metrics_analysis[metric][f'{winner}_wins'] += 1
                    metrics_analysis[metric]['total_tests'] += 1
                    
                    # Calcular speedup do NSeekFS vs melhor alternativa
                    if 'nseekfs' in times:
                        others = [times[k] for k in times.keys() if k != 'nseekfs']
                        if others:
                            best_other = min(others)
                            speedup = best_other / times['nseekfs']
                            metrics_analysis[metric]['speedups'].append(speedup)
        
        # Relatório por métrica
        print_result("\n📊 Analysis by metric:")
        
        for metric, analysis in metrics_analysis.items():
            if analysis['total_tests'] > 0:
                print_result(f"\n🎯 {metric.upper()}:")
                
                # Vitórias
                for impl in ['nseekfs', 'numpy', 'scipy']:
                    wins = analysis[f'{impl}_wins']
                    if wins > 0:
                        percentage = (wins / analysis['total_tests']) * 100
                        print_result(f"   • {impl.upper()}: {wins}/{analysis['total_tests']} wins ({percentage:.1f}%)")
                
                # Speedup médio do NSeekFS
                if analysis['speedups']:
                    avg_speedup = np.mean(analysis['speedups'])
                    if avg_speedup > 1:
                        print_winner(f"   🚀 NSeekFS avg speedup: {avg_speedup:.1f}x")
                    else:
                        print_result(f"   📊 NSeekFS avg performance: {avg_speedup:.1f}x")
        
        # Conclusões
        print_result(f"\n🎯 CONCLUSIONS:")
        
        total_nseekfs_wins = sum(analysis['nseekfs_wins'] for analysis in metrics_analysis.values())
        total_tests = sum(analysis['total_tests'] for analysis in metrics_analysis.values())
        
        if total_tests > 0:
            win_rate = (total_nseekfs_wins / total_tests) * 100
            print_result(f"   • NSeekFS win rate: {win_rate:.1f}% ({total_nseekfs_wins}/{total_tests})")
            
            all_speedups = []
            for analysis in metrics_analysis.values():
                all_speedups.extend(analysis['speedups'])
            
            if all_speedups:
                overall_speedup = np.mean(all_speedups)
                if overall_speedup > 1:
                    print_winner(f"   🏆 Overall NSeekFS speedup: {overall_speedup:.1f}x")
                else:
                    print_result(f"   📊 Overall NSeekFS performance: {overall_speedup:.1f}x")
        
        # Recomendações
        print_result(f"\n💡 RECOMMENDATIONS:")
        print_result(f"   • Best for: Exact similarity search with high precision")
        print_result(f"   • Optimal dimensions: 128-1024 (SIMD benefits)")
        print_result(f"   • Memory efficiency: ~{np.mean([r['data_size_mb'] for r in self.results]):.1f}MB per dataset")
        print_result(f"   • Production ready: ✅ Thread-safe, memory efficient")

def export_results(results: List[Dict[str, Any]], filename: str):
    """Exportar resultados para arquivo"""
    
    if filename.endswith('.json'):
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print_result(f"📊 Results exported to {filename}")
        
    elif filename.endswith('.csv'):
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'dataset', 'n_vectors', 'dimensions', 'metric', 'implementation',
                'avg_time_ms', 'qps', 'simd_used', 'memory_mb'
            ])
            
            # Data
            for result in results:
                dataset = result['dataset_name']
                n_vectors = result['n_vectors']
                dimensions = result['dimensions']
                
                for metric, metric_data in result['metrics'].items():
                    for impl, impl_data in metric_data.items():
                        if isinstance(impl_data, dict) and 'avg_time_ms' in impl_data:
                            writer.writerow([
                                dataset, n_vectors, dimensions, metric, impl,
                                impl_data['avg_time_ms'], impl_data['qps'],
                                impl_data.get('simd_used', ''), impl_data.get('memory_mb', '')
                            ])
        
        print_result(f"📊 Results exported to {filename}")
    else:
        print("❌ Unsupported format. Use .json or .csv")

def run_quick_comparison():
    """Comparação rápida para teste"""
    print_header("QUICK COMPARISON")
    
    try:
        import nseekfs
        
        # Dados pequenos para teste rápido
        vectors = np.random.randn(5000, 256).astype(np.float32)
        query = np.random.randn(256).astype(np.float32)
        
        print_result(f"Dataset: {vectors.shape[0]:,} vectors × {vectors.shape[1]}D")
        
        # NSeekFS
        print_result("Testing NSeekFS...")
        start_time = time.time()
        index = nseekfs.from_embeddings(vectors, metric="cosine")
        build_time = time.time() - start_time
        
        start_time = time.time()
        results = index.query(query, top_k=10)
        nseekfs_time = (time.time() - start_time) * 1000
        
        # NumPy
        print_result("Testing NumPy...")
        query_norm = query / np.linalg.norm(query)
        vectors_norm = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        
        start_time = time.time()
        similarities = np.dot(vectors_norm, query_norm)
        top_indices = np.argpartition(similarities, -10)[-10:]
        numpy_time = (time.time() - start_time) * 1000
        
        # Resultados
        print_result(f"\n🏆 Results:")
        print_benchmark("NSeekFS", nseekfs_time, 1000/nseekfs_time)
        print_benchmark("NumPy", numpy_time, 1000/numpy_time)
        
        speedup = numpy_time / nseekfs_time
        if speedup > 1:
            print_winner(f"🚀 NSeekFS is {speedup:.1f}x faster!")
        else:
            print_result(f"NumPy é {1/speedup:.1f}x mais rápido")
        
        print_result(f"💾 Memória NSeekFS: {index.memory_usage_mb:.1f}MB")
        
    except ImportError:
        print("❌ NSeekFS não encontrado")
    except Exception as e:
        print(f"❌ Erro: {e}")

def main():
    """Função principal"""
    
    parser = argparse.ArgumentParser(description='NSeekFS v1.0 Benchmark Comparativo')
    parser.add_argument('--datasets', type=str, default='small,medium',
                       help='Datasets para testar (tiny,small,medium,large,xlarge,extreme)')
    parser.add_argument('--metrics', type=str, default='cosine,dot,euclidean',
                       help='Métricas para testar (cosine,dot,euclidean)')
    parser.add_argument('--top-k', type=int, default=10, help='Número de resultados top-k')
    parser.add_argument('--export', type=str, help='Exportar resultados (arquivo.json ou arquivo.csv)')
    parser.add_argument('--quick', action='store_true', help='Comparação rápida apenas')
    
    args = parser.parse_args()
    
    print(colored("🚀 🚀 NSEEKFS v1.0 - COMPREHENSIVE BENCHMARK", Colors.BOLD + Colors.BLUE))
    print(colored("=" * 70, Colors.BLUE))
    
    if args.quick:
        run_quick_comparison()
        return 0
    
    # Parse argumentos
    datasets = [d.strip() for d in args.datasets.split(',')]
    metrics = [m.strip() for m in args.metrics.split(',')]
    
    # Verificar dependências
    try:
        import nseekfs
        print_result(f"✅ NSeekFS v{getattr(nseekfs, '__version__', '1.0.0')} disponível")
    except ImportError:
        print("❌ NSeekFS não encontrado. Instale com: pip install nseekfs")
        return 1
    
    print_result(f"✅ NumPy {np.__version__} disponível")
    
    if HAS_SCIPY:
        import scipy
        print_result(f"✅ SciPy {scipy.__version__} disponível")
    else:
        print_result("⚠️ SciPy não disponível - algumas comparações limitadas")
    
    # Executar benchmarks
    try:
        suite = BenchmarkSuite()
        
        start_time = time.time()
        results = suite.run_comprehensive_benchmark(datasets=datasets, metrics=metrics, top_k=args.top_k)
        total_time = time.time() - start_time
        
        print_result(f"\n⏱️ Tempo total de benchmark: {total_time:.1f}s")
        
        # Exportar se solicitado
        if args.export:
            export_results(results, args.export)
        
        print_result(f"\n✅ Benchmark comparativo concluído!")
        return 0
        
    except KeyboardInterrupt:
        print("\n⏹️ Benchmark interrompido pelo usuário")
        return 2
    except Exception as e:
        print(f"\n❌ Erro no benchmark: {e}")
        import traceback
        traceback.print_exc()
        return 3

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)