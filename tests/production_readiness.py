#!/usr/bin/env python3
"""
NSeekFS v1.0 - Production Readiness Test
========================================

This script performs comprehensive production readiness testing for NSeekFS v1.0:
- ✅ Production dataset simulation
- ✅ Concurrent load testing
- ✅ Long-term stability testing
- ✅ Resource monitoring and limits
- ✅ Real-world scenario validation
- ✅ Performance benchmarking under load

Usage:
    python production_readiness.py [--duration=300] [--load=medium] [--scenario=medium_ecommerce]
"""

import sys
import time
import psutil
import numpy as np
import threading
import argparse
import json
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Colors for output
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

@dataclass
class ProductionScenario:
    """Production scenario configuration"""
    name: str
    num_vectors: int
    dimensions: int
    description: str
    expected_qps: int
    memory_limit_gb: float

# Production scenarios
PRODUCTION_SCENARIOS = {
    'small_company': ProductionScenario(
        name="Small Company",
        num_vectors=10000,
        dimensions=384,
        description="Small company with 10K document embeddings",
        expected_qps=100,
        memory_limit_gb=1.0
    ),
    'medium_ecommerce': ProductionScenario(
        name="Medium E-commerce",
        num_vectors=100000,
        dimensions=512,
        description="E-commerce platform with 100K product embeddings",
        expected_qps=500,
        memory_limit_gb=4.0
    ),
    'large_content': ProductionScenario(
        name="Large Content Platform",
        num_vectors=500000,
        dimensions=768,
        description="Large content platform with 500K article embeddings",
        expected_qps=1000,
        memory_limit_gb=8.0
    ),
    'enterprise': ProductionScenario(
        name="Enterprise",
        num_vectors=1000000,
        dimensions=1024,
        description="Enterprise with 1M+ knowledge base embeddings",
        expected_qps=2000,
        memory_limit_gb=16.0
    )
}

class ProductionReadinessTest:
    """Comprehensive production readiness testing suite"""
    
    def __init__(self):
        self.start_time = time.time()
        self.system_info = self._gather_system_info()
        
    def _gather_system_info(self) -> Dict[str, Any]:
        """Gather system information"""
        return {
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq().current if psutil.cpu_freq() else 0,
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'memory_available_gb': psutil.virtual_memory().available / (1024**3),
            'platform': sys.platform,
            'python_version': sys.version.split()[0]
        }
    
    def create_production_dataset(self, scenario_name: str) -> Tuple[np.ndarray, int]:
        """Create realistic production dataset"""
        
        if scenario_name not in PRODUCTION_SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario_name}")
        
        scenario = PRODUCTION_SCENARIOS[scenario_name]
        logger.info(f"Creating dataset for scenario: {scenario.description}")
        
        # Check memory requirements
        estimated_memory_gb = (scenario.num_vectors * scenario.dimensions * 4) / (1024**3)
        available_memory_gb = self.system_info['memory_available_gb']
        
        if estimated_memory_gb > available_memory_gb * 0.8:
            logger.warning(f"Dataset may exceed available memory: {estimated_memory_gb:.1f}GB needed, {available_memory_gb:.1f}GB available")
        
        # Generate realistic embeddings (with some structure)
        np.random.seed(42)  # Reproducible
        
        # Create clustered embeddings to simulate real-world data
        num_clusters = min(100, scenario.num_vectors // 100)
        cluster_centers = np.random.randn(num_clusters, scenario.dimensions).astype(np.float32)
        
        vectors = []
        cluster_size = scenario.num_vectors // num_clusters
        
        for i in range(num_clusters):
            # Add noise around cluster centers
            cluster_vectors = cluster_centers[i] + np.random.randn(cluster_size, scenario.dimensions) * 0.1
            vectors.append(cluster_vectors.astype(np.float32))
        
        # Handle remainder
        remainder = scenario.num_vectors - (num_clusters * cluster_size)
        if remainder > 0:
            remainder_vectors = np.random.randn(remainder, scenario.dimensions).astype(np.float32)
            vectors.append(remainder_vectors)
        
        dataset = np.vstack(vectors)
        
        logger.info(f"Generated dataset: {dataset.shape[0]:,} vectors × {dataset.shape[1]}D ({estimated_memory_gb:.1f}GB)")
        
        return dataset, scenario.dimensions
    
    def test_startup_performance(self, vectors: np.ndarray) -> Tuple[Dict[str, Any], Any]:
        """Test index creation performance"""
        
        logger.info("Testing startup performance...")
        
        try:
            import nseekfs
        except ImportError:
            raise RuntimeError("NSeekFS not available")
        
        memory_before = psutil.virtual_memory().used / (1024**3)
        
        start_time = time.time()
        index = nseekfs.from_embeddings(vectors, normalized=True, verbose=False)
        creation_time = time.time() - start_time
        
        memory_after = psutil.virtual_memory().used / (1024**3)
        memory_used = memory_after - memory_before
        
        # Test initial queries to ensure index is working
        test_query = np.random.randn(vectors.shape[1]).astype(np.float32)
        
        query_start = time.time()
        initial_results = index.query(test_query, top_k=10)
        initial_query_time = (time.time() - query_start) * 1000
        
        results = {
            'creation_time_s': creation_time,
            'memory_used_gb': memory_used,
            'initial_query_time_ms': initial_query_time,
            'vectors_per_second': len(vectors) / creation_time,
            'success': len(initial_results) == 10
        }
        
        logger.info(f"Startup completed in {creation_time:.2f}s, using {memory_used:.2f}GB")
        
        return results, index
    
    def test_concurrent_load(self, index: Any, duration_seconds: int, load_level: str) -> Dict[str, Any]:
        """Test concurrent load performance"""
        
        logger.info(f"Testing concurrent load for {duration_seconds}s at {load_level} level...")
        
        # Load level configurations
        load_configs = {
            'light': {'num_threads': 2, 'queries_per_second': 10},
            'medium': {'num_threads': 4, 'queries_per_second': 50},
            'heavy': {'num_threads': 8, 'queries_per_second': 200}
        }
        
        if load_level not in load_configs:
            raise ValueError(f"Unknown load level: {load_level}")
        
        config = load_configs[load_level]
        query_interval = 1.0 / config['queries_per_second']
        
        # Metrics collection
        metrics = {
            'query_times': [],
            'errors': 0,
            'successful_queries': 0,
            'threads_completed': 0,
            'memory_samples': [],
            'cpu_samples': []
        }
        
        metrics_lock = threading.Lock()
        stop_event = threading.Event()
        
        def query_worker(worker_id: int, queries_per_thread: int):
            """Worker thread for concurrent queries"""
            local_times = []
            local_errors = 0
            
            try:
                for i in range(queries_per_thread):
                    if stop_event.is_set():
                        break
                    
                    # Generate random query
                    query = np.random.randn(index.dims).astype(np.float32)
                    
                    query_start = time.time()
                    try:
                        results = index.query(query, top_k=10)
                        query_time = (time.time() - query_start) * 1000
                        local_times.append(query_time)
                        
                        # Validate results
                        if not isinstance(results, list) or len(results) != 10:
                            local_errors += 1
                            
                    except Exception as e:
                        local_errors += 1
                        logger.error(f"Query error in worker {worker_id}: {e}")
                    
                    # Rate limiting
                    time.sleep(query_interval)
                
            except Exception as e:
                logger.error(f"Worker {worker_id} failed: {e}")
                local_errors += 1
            
            # Update global metrics
            with metrics_lock:
                metrics['query_times'].extend(local_times)
                metrics['errors'] += local_errors
                metrics['successful_queries'] += len(local_times)
                metrics['threads_completed'] += 1
        
        def monitor_resources():
            """Monitor system resources during test"""
            while not stop_event.is_set():
                try:
                    memory_percent = psutil.virtual_memory().percent
                    cpu_percent = psutil.cpu_percent()
                    
                    with metrics_lock:
                        metrics['memory_samples'].append(memory_percent)
                        metrics['cpu_samples'].append(cpu_percent)
                    
                    time.sleep(1)
                except Exception as e:
                    logger.error(f"Resource monitoring error: {e}")
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
        monitor_thread.start()
        
        # Calculate queries per thread
        total_expected_queries = config['queries_per_second'] * duration_seconds
        queries_per_thread = total_expected_queries // config['num_threads']
        
        # Start load test
        test_start = time.time()
        
        with ThreadPoolExecutor(max_workers=config['num_threads']) as executor:
            futures = []
            for i in range(config['num_threads']):
                future = executor.submit(query_worker, i, queries_per_thread)
                futures.append(future)
            
            # Wait for duration or completion
            time.sleep(duration_seconds)
            stop_event.set()
            
            # Wait for threads to finish
            for future in as_completed(futures, timeout=30):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Thread execution error: {e}")
        
        test_duration = time.time() - test_start
        
        # Calculate results
        total_queries = metrics['successful_queries'] + metrics['errors']
        success_rate = (metrics['successful_queries'] / total_queries * 100) if total_queries > 0 else 0
        
        avg_query_time = np.mean(metrics['query_times']) if metrics['query_times'] else 0
        p95_query_time = np.percentile(metrics['query_times'], 95) if metrics['query_times'] else 0
        
        actual_qps = metrics['successful_queries'] / test_duration
        
        avg_memory = np.mean(metrics['memory_samples']) if metrics['memory_samples'] else 0
        avg_cpu = np.mean(metrics['cpu_samples']) if metrics['cpu_samples'] else 0
        
        results = {
            'test_duration_s': test_duration,
            'total_queries': total_queries,
            'successful_queries': metrics['successful_queries'],
            'errors': metrics['errors'],
            'success_rate': success_rate,
            'avg_query_time_ms': avg_query_time,
            'p95_query_time_ms': p95_query_time,
            'actual_qps': actual_qps,
            'target_qps': config['queries_per_second'],
            'qps_achievement': (actual_qps / config['queries_per_second']) * 100,
            'avg_memory_percent': avg_memory,
            'avg_cpu_percent': avg_cpu,
            'threads_completed': metrics['threads_completed']
        }
        
        logger.info(f"Load test completed: {actual_qps:.1f} QPS, {success_rate:.1f}% success rate")
        
        return results
    
    def test_stability_over_time(self, index: Any, duration_seconds: int) -> Dict[str, Any]:
        """Test long-term stability"""
        
        logger.info(f"Testing stability for {duration_seconds}s...")
        
        # Stability metrics
        stability_metrics = {
            'query_times_by_minute': [],
            'memory_usage_by_minute': [],
            'error_count_by_minute': [],
            'total_queries': 0,
            'total_errors': 0
        }
        
        start_time = time.time()
        minute_queries = []
        minute_errors = 0
        last_minute_mark = 0
        
        query_count = 0
        
        while time.time() - start_time < duration_seconds:
            # Execute query
            query = np.random.randn(index.dims).astype(np.float32)
            
            query_start = time.time()
            try:
                results = index.query(query, top_k=10)
                query_time = (time.time() - query_start) * 1000
                minute_queries.append(query_time)
                query_count += 1
                
                # Validate results
                if not isinstance(results, list) or len(results) != 10:
                    minute_errors += 1
                    
            except Exception as e:
                minute_errors += 1
                logger.error(f"Stability test query error: {e}")
            
            # Check if a minute has passed
            current_minute = int((time.time() - start_time) / 60)
            if current_minute > last_minute_mark:
                # Record minute metrics
                if minute_queries:
                    stability_metrics['query_times_by_minute'].append({
                        'minute': current_minute,
                        'avg_time_ms': np.mean(minute_queries),
                        'queries_count': len(minute_queries)
                    })
                
                stability_metrics['error_count_by_minute'].append({
                    'minute': current_minute,
                    'errors': minute_errors
                })
                
                # Record memory usage
                memory_percent = psutil.virtual_memory().percent
                stability_metrics['memory_usage_by_minute'].append({
                    'minute': current_minute,
                    'memory_percent': memory_percent
                })
                
                # Reset for next minute
                minute_queries = []
                minute_errors = 0
                last_minute_mark = current_minute
                
                if current_minute % 5 == 0:  # Log every 5 minutes
                    logger.info(f"Stability test: {current_minute} minutes completed")
            
            # Small delay to prevent overwhelming
            time.sleep(0.1)
        
        total_duration = time.time() - start_time
        stability_metrics['total_queries'] = query_count
        
        # Analyze stability
        if stability_metrics['query_times_by_minute']:
            first_minute_avg = stability_metrics['query_times_by_minute'][0]['avg_time_ms']
            last_minute_avg = stability_metrics['query_times_by_minute'][-1]['avg_time_ms']
            
            performance_drift = ((last_minute_avg - first_minute_avg) / first_minute_avg) * 100
        else:
            performance_drift = 0
        
        total_errors = sum(m['errors'] for m in stability_metrics['error_count_by_minute'])
        error_rate = (total_errors / query_count) * 100 if query_count > 0 else 0
        
        results = {
            'test_duration_s': total_duration,
            'total_queries': query_count,
            'total_errors': total_errors,
            'error_rate': error_rate,
            'performance_drift_percent': performance_drift,
            'minutes_tested': len(stability_metrics['query_times_by_minute']),
            'stability_metrics': stability_metrics,
            'stable': abs(performance_drift) < 20 and error_rate < 5
        }
        
        logger.info(f"Stability test completed: {performance_drift:+.1f}% drift, {error_rate:.2f}% error rate")
        
        return results
    
    def generate_production_assessment(self, all_results: Dict[str, Any], scenario_name: str) -> None:
        """Generate comprehensive production readiness assessment"""
        
        scenario = PRODUCTION_SCENARIOS[scenario_name]
        
        print("\n" + "="*80)
        print(colored("🎯 PRODUCTION READINESS ASSESSMENT", Colors.BOLD + Colors.BLUE))
        print("="*80)
        
        print(f"\n{colored('📋 SCENARIO:', Colors.BOLD)} {scenario.description}")
        print(f"   Dataset: {scenario.num_vectors:,} vectors × {scenario.dimensions}D")
        print(f"   Target QPS: {scenario.expected_qps}")
        print(f"   Memory Limit: {scenario.memory_limit_gb}GB")
        
        # Startup Assessment
        print(f"\n{colored('🚀 STARTUP PERFORMANCE:', Colors.BOLD)}")
        startup = all_results.get('startup', {})
        if startup.get('success'):
            print(colored(f"   ✅ Index creation: {startup['creation_time_s']:.2f}s", Colors.GREEN))
            print(colored(f"   ✅ Memory usage: {startup['memory_used_gb']:.2f}GB", Colors.GREEN))
            print(colored(f"   ✅ Initial query: {startup['initial_query_time_ms']:.2f}ms", Colors.GREEN))
        else:
            print(colored("   ❌ Startup failed", Colors.RED))
        
        # Load Test Assessment
        print(f"\n{colored('⚡ LOAD PERFORMANCE:', Colors.BOLD)}")
        load = all_results.get('concurrent_load', {})
        if load:
            qps_achievement = load.get('qps_achievement', 0)
            success_rate = load.get('success_rate', 0)
            
            if qps_achievement >= 80 and success_rate >= 95:
                print(colored(f"   ✅ QPS Achievement: {qps_achievement:.1f}%", Colors.GREEN))
                print(colored(f"   ✅ Success Rate: {success_rate:.1f}%", Colors.GREEN))
            elif qps_achievement >= 60 and success_rate >= 90:
                print(colored(f"   ⚠️ QPS Achievement: {qps_achievement:.1f}%", Colors.YELLOW))
                print(colored(f"   ⚠️ Success Rate: {success_rate:.1f}%", Colors.YELLOW))
            else:
                print(colored(f"   ❌ QPS Achievement: {qps_achievement:.1f}%", Colors.RED))
                print(colored(f"   ❌ Success Rate: {success_rate:.1f}%", Colors.RED))
            
            print(f"   Average Query Time: {load.get('avg_query_time_ms', 0):.2f}ms")
            print(f"   P95 Query Time: {load.get('p95_query_time_ms', 0):.2f}ms")
            print(f"   Resource Usage: {load.get('avg_cpu_percent', 0):.1f}% CPU, {load.get('avg_memory_percent', 0):.1f}% Memory")
        
        # Stability Assessment
        print(f"\n{colored('🔄 STABILITY:', Colors.BOLD)}")
        stability = all_results.get('stability', {})
        if stability:
            drift = stability.get('performance_drift_percent', 0)
            error_rate = stability.get('error_rate', 0)
            
            if stability.get('stable'):
                print(colored(f"   ✅ Performance Drift: {drift:+.1f}%", Colors.GREEN))
                print(colored(f"   ✅ Error Rate: {error_rate:.2f}%", Colors.GREEN))
            else:
                print(colored(f"   ❌ Performance Drift: {drift:+.1f}%", Colors.RED))
                print(colored(f"   ❌ Error Rate: {error_rate:.2f}%", Colors.RED))
            
            print(f"   Test Duration: {stability.get('minutes_tested', 0)} minutes")
            print(f"   Total Queries: {stability.get('total_queries', 0):,}")
        
        # Overall Score Calculation
        scores = []
        
        # Startup score (25%)
        if startup.get('success') and startup.get('creation_time_s', float('inf')) < 30:
            scores.append(25)
        elif startup.get('success'):
            scores.append(15)
        else:
            scores.append(0)
        
        # Load score (40%)
        if load:
            load_score = 0
            if load.get('qps_achievement', 0) >= 80:
                load_score += 20
            elif load.get('qps_achievement', 0) >= 60:
                load_score += 15
            elif load.get('qps_achievement', 0) >= 40:
                load_score += 10
            
            if load.get('success_rate', 0) >= 95:
                load_score += 20
            elif load.get('success_rate', 0) >= 90:
                load_score += 15
            elif load.get('success_rate', 0) >= 80:
                load_score += 10
            
            scores.append(load_score)
        else:
            scores.append(0)
        
        # Stability score (35%)
        if stability.get('stable'):
            scores.append(35)
        elif abs(stability.get('performance_drift_percent', 50)) < 30:
            scores.append(20)
        else:
            scores.append(10)
        
        overall_score = sum(scores)
        
        # Final Assessment
        print(f"\n{colored('📊 OVERALL ASSESSMENT:', Colors.BOLD)}")
        print(f"   Production Readiness Score: {overall_score}/100")
        
        if overall_score >= 85:
            print(colored("   🎉 NSeekFS v1.0 is READY for production", Colors.BOLD + Colors.GREEN))
            print("   Recommended for deployment with confidence")
        elif overall_score >= 75:
            print(colored("   ✅ NSeekFS v1.0 is SUITABLE for production", Colors.GREEN))
            print("   Implement additional monitoring")
        elif overall_score >= 60:
            print(colored("   ⚠️ NSeekFS v1.0 needs IMPROVEMENTS", Colors.YELLOW))
            print("   Address identified issues first")
        else:
            print(colored("   ❌ NSeekFS v1.0 is NOT ready for production", Colors.RED))
            print("   Significant optimizations needed")
        
        print(colored("="*80, Colors.BOLD))

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='NSeekFS v1.0 - Production Readiness Test')
    parser.add_argument('--duration', type=int, default=300, help='Test duration (seconds)')
    parser.add_argument('--load', choices=['light', 'medium', 'heavy'], default='medium', help='Load level')
    parser.add_argument('--scenario', choices=['small_company', 'medium_ecommerce', 'large_content', 'enterprise'], 
                       default='medium_ecommerce', help='Dataset scenario')
    parser.add_argument('--stability-duration', type=int, default=900, help='Stability test duration (seconds)')
    parser.add_argument('--skip-stability', action='store_true', help='Skip long stability test')
    parser.add_argument('--export', type=str, help='Export results to JSON file')
    
    args = parser.parse_args()
    
    print(colored("NSeekFS v1.0 - PRODUCTION READINESS TEST", Colors.BOLD + Colors.BLUE))
    print(colored("=" * 70, Colors.BLUE))
    
    # Check NSeekFS
    try:
        import nseekfs
        logger.info(f"NSeekFS v{getattr(nseekfs, '__version__', '1.0.0')} available")
    except ImportError:
        logger.error("NSeekFS not found. Install with: pip install nseekfs")
        return 1
    
    # Create test
    test = ProductionReadinessTest()
    all_results = {}
    
    try:
        # 1. Create production dataset
        logger.info(f"Creating dataset for scenario: {args.scenario}")
        vectors, dimensions = test.create_production_dataset(args.scenario)
        
        # 2. Startup performance test
        startup_results, index = test.test_startup_performance(vectors)
        all_results['startup'] = startup_results
        
        # 3. Concurrent load test
        load_results = test.test_concurrent_load(index, args.duration, args.load)
        all_results['concurrent_load'] = load_results
        
        # 4. Stability test (optional)
        if not args.skip_stability:
            stability_results = test.test_stability_over_time(index, args.stability_duration)
            all_results['stability'] = stability_results
        
        # 5. Generate assessment
        test.generate_production_assessment(all_results, args.scenario)
        
        # 6. Export results if requested
        if args.export:
            with open(args.export, 'w') as f:
                json.dump(all_results, f, indent=2, default=str)
            logger.info(f"Results exported to {args.export}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("Test interrupted by user")
        return 2
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 3

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)