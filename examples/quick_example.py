#!/usr/bin/env python3
"""
NSeekFS v1.0 - Exemplo Rápido
=============================

Demonstração minimalista do NSeekFS v1.0 para usar em demos,
documentação ou como primeiro teste após instalação.

Uso:
    python quick_example.py
"""

import numpy as np
import time

def main():
    print("🚀 NSeekFS v1.0 - Exemplo Rápido")
    print("=" * 40)
    
    try:
        # Importar NSeekFS
        import nseekfs
        print(f"✅ NSeekFS importado - versão: {getattr(nseekfs, '__version__', 'desconhecida')}")
        
        # 1. Criar dados de exemplo
        print("\n📊 Criando dados de exemplo...")
        n_vectors = 5000
        dimensions = 256
        
        # Simular embeddings (ex: de um modelo BERT)
        vectors = np.random.randn(n_vectors, dimensions).astype(np.float32)
        query_vector = np.random.randn(dimensions).astype(np.float32)
        
        print(f"   • {n_vectors:,} vetores de {dimensions} dimensões")
        print(f"   • Tamanho dos dados: {vectors.nbytes / 1024 / 1024:.1f}MB")
        
        # 2. Criar índice
        print("\n🔧 Criando índice exato...")
        start_time = time.time()
        
        index = nseekfs.from_embeddings(
            vectors,
            metric="cosine",           # Métrica de similaridade
            base_name="quick_demo",    # Nome do arquivo
            normalized=True            # Normalizar vetores automaticamente
        )
        
        build_time = time.time() - start_time
        print(f"   • Índice criado em {build_time:.3f}s")
        print(f"   • Uso de memória: {index.memory_usage_mb:.1f}MB")
        
        # 3. Executar buscas
        print("\n🔍 Executando buscas...")
        
        # Busca simples
        start_time = time.time()
        results = index.query(query_vector, top_k=5)
        query_time = (time.time() - start_time) * 1000
        
        print(f"   • Busca executada em {query_time:.2f}ms")
        print(f"   • {len(results)} resultados encontrados")
        
        # Mostrar top 3 resultados
        print("\n📋 Top 3 resultados:")
        for i, result in enumerate(results[:3]):
            print(f"   {i+1}. Vetor {result['idx']:,} (similaridade: {result['score']:.6f})")
        
        # 4. Busca com timing detalhado
        print("\n⏱️  Busca com timing detalhado...")
        results_with_timing, timing = index.query(
            query_vector, 
            top_k=10, 
            return_timing=True
        )
        
        print(f"   • Tempo: {timing['query_time_ms']:.2f}ms")
        print(f"   • Método: {timing['method_used']}")
        print(f"   • SIMD ativo: {timing['simd_used']}")
        print(f"   • Vetores examinados: {timing.get('candidates_generated', 'todos')}")
        
        # 5. Múltiplas queries (batch)
        print("\n📦 Testando batch de queries...")
        batch_queries = np.random.randn(3, dimensions).astype(np.float32)
        
        start_time = time.time()
        batch_results = index.query_batch(batch_queries, top_k=3)
        batch_time = (time.time() - start_time) * 1000
        
        print(f"   • 3 queries em batch: {batch_time:.2f}ms")
        print(f"   • Tempo médio por query: {batch_time/3:.2f}ms")
        
        # 6. Estatísticas do índice
        print("\n📈 Estatísticas do índice:")
        stats = index.stats
        print(f"   • Total de queries: {stats['total_queries']}")
        print(f"   • Tempo médio: {stats['avg_query_time_ms']:.2f}ms")
        print(f"   • Queries exatas: {stats['exact_queries']}")
        print(f"   • Queries SIMD: {stats['simd_queries']}")
        
        # 7. Health check
        print("\n🏥 Verificação de saúde:")
        health = index.health_check()
        print(f"   • Status: {health['status']}")
        if health['status'] == 'healthy':
            print(f"   • Teste básico: {health['basic_test_time_ms']:.2f}ms")
        
        # 8. Comparar métricas
        print("\n🔄 Comparando métricas de similaridade...")
        metrics = ["cosine", "dot_product", "euclidean"]
        
        for metric in metrics:
            # Criar mini-índice para comparação
            mini_vectors = vectors[:100]  # Apenas 100 vetores para rapidez
            mini_index = nseekfs.from_embeddings(
                mini_vectors, 
                metric=metric, 
                base_name=f"demo_{metric}"
            )
            
            start_time = time.time()
            mini_results = mini_index.query(query_vector, top_k=3)
            mini_time = (time.time() - start_time) * 1000
            
            print(f"   • {metric:12}: {mini_time:5.2f}ms (top: idx={mini_results[0]['idx']}, score={mini_results[0]['score']:.4f})")
        
        # 9. Benchmark rápido
        print("\n🏃 Benchmark rápido...")
        benchmark = nseekfs.benchmark(vectors_count=2000, dimensions=128)
        
        print(f"   • Build time: {benchmark['index_time_seconds']:.2f}s")
        print(f"   • Query time: {benchmark['avg_query_time_ms']:.2f}ms")
        print(f"   • Queries/segundo: {benchmark['queries_per_second']:,.0f}")
        print(f"   • Throughput: {benchmark['queries_per_second'] * 2000:,.0f} vetores/segundo")
        
        # 10. Informações do sistema
        print("\n🖥️  Informações do sistema:")
        info = nseekfs.get_system_info()
        print(f"   • Plataforma: {info['platform']}")
        print(f"   • Python: {info['python_version']}")
        print(f"   • NumPy: {info.get('numpy_version', 'N/A')}")
        print(f"   • Engine Rust: {info['rust_engine']}")
        
        print("\n" + "="*50)
        print("🎉 Exemplo concluído com sucesso!")
        print("🚀 NSeekFS v1.0 está funcionando perfeitamente!")
        print("="*50)
        
        return True
        
    except ImportError:
        print("❌ Erro: NSeekFS não está instalado")
        print("💡 Instale com: pip install nseekfs")
        return False
        
    except Exception as e:
        print(f"❌ Erro inesperado: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)