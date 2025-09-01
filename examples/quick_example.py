#!/usr/bin/env python3
"""
NSeekFS v1.0 - Quick Example
============================

Minimal demonstration of NSeekFS v1.0 core functionality.
Based on actual API testing and production usage patterns.

Usage:
    python quick_example.py
"""

import numpy as np
import time
import sys

def main():
    print("NSeekFS v1.0 - Quick Example")
    print("=" * 40)
    
    try:
        # Import and verify NSeekFS installation
        import nseekfs
        version = getattr(nseekfs, '__version__', 'unknown')
        print(f"NSeekFS imported successfully - version: {version}")
        
        # 1. Generate sample embeddings
        print("\nGenerating sample embeddings...")
        n_vectors = 5000
        dimensions = 256
        
        # Create realistic embedding data (similar to BERT/sentence transformers)
        vectors = np.random.randn(n_vectors, dimensions).astype(np.float32)
        query_vector = np.random.randn(dimensions).astype(np.float32)
        
        data_size_mb = vectors.nbytes / 1024 / 1024
        print(f"Created {n_vectors:,} vectors x {dimensions}D ({data_size_mb:.1f}MB)")
        
        # 2. Build search index
        print("\nBuilding search index...")
        start_time = time.time()
        
        # Use actual API signature based on current implementation
        index = nseekfs.from_embeddings(
            vectors,
            normalized=True,    # Automatically normalize vectors
            verbose=True        # Show build progress
        )
        
        build_time = time.time() - start_time
        print(f"Index built in {build_time:.3f} seconds")
        
        # Display index properties
        print(f"Index contains {index.rows:,} vectors with {index.dims} dimensions")
        
        # 3. Perform similarity search
        print("\nPerforming similarity search...")
        
        # Basic search operation
        start_time = time.time()
        results = index.query(query_vector, top_k=5)
        query_time = (time.time() - start_time) * 1000
        
        print(f"Search completed in {query_time:.2f}ms")
        print(f"Found {len(results)} results")
        
        # Display top results
        print("\nTop 3 most similar vectors:")
        for i, result in enumerate(results[:3], 1):
            idx = result['idx']
            score = result['score']
            print(f"  {i}. Vector {idx:,} (similarity: {score:.6f})")
        
        # 4. Test advanced features (with error handling)
        print("\nTesting advanced features...")
        
        # Try query with timing information
        try:
            results_timed, timing_info = index.query(
                query_vector, 
                top_k=10, 
                return_timing=True
            )
            
            available_timing = list(timing_info.keys())
            print(f"Query timing available: {len(available_timing)} metrics")
            
            if 'query_time_ms' in timing_info:
                print(f"Detailed timing: {timing_info['query_time_ms']:.2f}ms")
                
        except TypeError:
            print("Advanced timing not available in this build")
        except Exception as e:
            print(f"Timing feature unavailable: {type(e).__name__}")
        
        # Try batch queries
        try:
            batch_queries = np.random.randn(10, dimensions).astype(np.float32)
            
            start_time = time.time()
            batch_results = index.query_batch(batch_queries, top_k=3)
            batch_time = (time.time() - start_time) * 1000
            
            avg_time_per_query = batch_time / len(batch_queries)
            print(f"Batch search: {len(batch_queries)} queries in {batch_time:.2f}ms")
            print(f"Average per query: {avg_time_per_query:.2f}ms")
            
        except AttributeError:
            print("Batch query functionality not available")
        except Exception as e:
            print(f"Batch search failed: {type(e).__name__}")
        
        # 5. Performance metrics
        print("\nRetrieving performance metrics...")
        try:
            metrics = index.get_performance_metrics()
            
            total_queries = metrics.get('total_queries', 0)
            avg_time = metrics.get('avg_query_time_ms', 0)
            qps = metrics.get('queries_per_second', 0)
            
            print(f"Total queries processed: {total_queries}")
            if avg_time > 0:
                print(f"Average query time: {avg_time:.2f}ms")
            if qps > 0:
                print(f"Queries per second: {qps:.0f}")
                
        except AttributeError:
            print("Performance metrics not available")
        except Exception as e:
            print(f"Metrics unavailable: {type(e).__name__}")
        
        # 6. System health check
        print("\nRunning system health check...")
        try:
            health_status = nseekfs.health_check(verbose=False)
            
            status = health_status.get('status', 'unknown')
            compatible = health_status.get('system_compatible', False)
            simd_available = health_status.get('simd_available', False)
            
            print(f"System status: {status}")
            print(f"System compatible: {compatible}")
            print(f"SIMD acceleration: {simd_available}")
            
        except AttributeError:
            print("Health check not available")
        except Exception as e:
            print(f"Health check failed: {type(e).__name__}")
        
        # 7. Quick benchmark
        print("\nExecuting quick benchmark...")
        try:
            # Try built-in benchmark if available
            nseekfs.benchmark(vectors=1000, dims=128, queries=50, verbose=False)
            print("Built-in benchmark completed successfully")
            
        except (AttributeError, TypeError):
            # Fallback to manual benchmark
            print("Running manual benchmark...")
            
            benchmark_vectors = np.random.randn(1000, 128).astype(np.float32)
            benchmark_queries = np.random.randn(20, 128).astype(np.float32)
            
            # Measure build time
            build_start = time.time()
            bench_index = nseekfs.from_embeddings(
                benchmark_vectors, 
                normalized=True, 
                verbose=False
            )
            build_duration = time.time() - build_start
            
            # Measure query times
            query_times = []
            for query in benchmark_queries:
                query_start = time.time()
                bench_index.query(query, top_k=10)
                query_duration = (time.time() - query_start) * 1000
                query_times.append(query_duration)
            
            avg_query_time = np.mean(query_times)
            queries_per_second = 1000 / avg_query_time if avg_query_time > 0 else 0
            
            print(f"Manual benchmark results:")
            print(f"  Build time: {build_duration:.3f}s")
            print(f"  Average query time: {avg_query_time:.2f}ms")
            print(f"  Estimated QPS: {queries_per_second:.0f}")
        
        # 8. Environment information
        print("\nEnvironment information:")
        python_version = '.'.join(map(str, sys.version_info[:3]))
        numpy_version = np.__version__
        nseekfs_version = getattr(nseekfs, '__version__', 'unknown')
        
        print(f"  Python: {python_version}")
        print(f"  NumPy: {numpy_version}")
        print(f"  NSeekFS: {nseekfs_version}")
        
        # Check available modules
        try:
            import pkgutil
            available_modules = []
            for _, module_name, _ in pkgutil.iter_modules(nseekfs.__path__):
                available_modules.append(module_name)
            if available_modules:
                print(f"  NSeekFS modules: {', '.join(available_modules)}")
        except:
            print("  NSeekFS modules: detection failed")
        
        print("\n" + "="*50)
        print("Example completed successfully!")
        print("NSeekFS v1.0 is working correctly.")
        print("="*50)
        
        return True
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Install NSeekFS with: pip install nseekfs")
        return False
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)