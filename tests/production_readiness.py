#!/usr/bin/env python3
"""
NSeekFS v1.0 - Validação para Produção
======================================

Testa cenários realistas de produção para validar se o NSeekFS v1.0
está pronto para deployment em ambiente real.

Cenários testados:
- Cargas de trabalho realistas
- Simulação de aplicações reais
- Testes de resistência temporal
- Cenários de falha e recuperação
- Monitoramento de recursos

Uso:
    python production_readiness.py [--duration=3600] [--load=medium] [--scenarios=all]
"""

import time
import threading
import queue
import psutil
import logging
import argparse
import json
import sys
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import tempfile
import gc
import warnings
warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nseekfs_production_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
    WHITE = '\033[97m'   

def colored(text, color):
    return f"{color}{text}{Colors.END}"

@dataclass
class SystemMetrics:
    """Métricas do sistema"""
    timestamp: float
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    disk_io_read: int
    disk_io_write: int

@dataclass
class QueryMetrics:
    """Métricas de query"""
    timestamp: float
    query_time_ms: float
    top_k: int
    results_count: int
    success: bool
    error: Optional[str] = None

class SystemMonitor:
    """Monitor de sistema em tempo real"""
    
    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.metrics: List[SystemMetrics] = []
        self.monitoring = False
        self.monitor_thread = None
        
    def start(self):
        """Iniciar monitoramento"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Sistema de monitoramento iniciado")
        
    def stop(self):
        """Parar monitoramento"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        logger.info("Sistema de monitoramento parado")
        
    def _monitor_loop(self):
        """Loop principal de monitoramento"""
        while self.monitoring:
            try:
                # CPU
                cpu_percent = psutil.cpu_percent(interval=None)
                
                # Memória
                memory = psutil.virtual_memory()
                memory_mb = memory.used / (1024 * 1024)
                memory_percent = memory.percent
                
                # Disk I/O
                disk_io = psutil.disk_io_counters()
                disk_read = disk_io.read_bytes if disk_io else 0
                disk_write = disk_io.write_bytes if disk_io else 0
                
                metrics = SystemMetrics(
                    timestamp=time.time(),
                    cpu_percent=cpu_percent,
                    memory_mb=memory_mb,
                    memory_percent=memory_percent,
                    disk_io_read=disk_read,
                    disk_io_write=disk_write
                )
                
                self.metrics.append(metrics)
                
                # Manter apenas últimas 1000 métricas
                if len(self.metrics) > 1000:
                    self.metrics = self.metrics[-1000:]
                    
            except Exception as e:
                logger.error(f"Erro no monitoramento: {e}")
                
            time.sleep(self.interval)
    
    def get_current_stats(self) -> Dict[str, float]:
        """Obter estatísticas atuais"""
        if not self.metrics:
            return {}
            
        recent = self.metrics[-10:]  # Últimas 10 medições
        
        return {
            'avg_cpu_percent': np.mean([m.cpu_percent for m in recent]),
            'avg_memory_mb': np.mean([m.memory_mb for m in recent]),
            'avg_memory_percent': np.mean([m.memory_percent for m in recent]),
            'peak_memory_mb': max(m.memory_mb for m in self.metrics),
            'peak_cpu_percent': max(m.cpu_percent for m in self.metrics)
        }

class ApplicationSimulator:
    """Simulador de aplicações reais usando NSeekFS"""
    
    def __init__(self, index, name: str):
        self.index = index
        self.name = name
        self.query_metrics: List[QueryMetrics] = []
        self.running = False
        
    def simulate_search_application(self, duration_seconds: int, queries_per_second: float):
        """Simular aplicação de busca semântica"""
        logger.info(f"Iniciando simulação: {self.name} por {duration_seconds}s a {queries_per_second} QPS")
        
        self.running = True
        start_time = time.time()
        query_interval = 1.0 / queries_per_second
        
        dimensions = self.index.dims
        
        while self.running and (time.time() - start_time) < duration_seconds:
            try:
                # Gerar query realística
                query_vector = np.random.randn(dimensions).astype(np.float32)
                query_vector = query_vector / np.linalg.norm(query_vector)
                
                # Executar query
                query_start = time.time()
                try:
                    results = self.index.query(query_vector, top_k=10)
                    query_time = (time.time() - query_start) * 1000
                    
                    metrics = QueryMetrics(
                        timestamp=time.time(),
                        query_time_ms=query_time,
                        top_k=10,
                        results_count=len(results),
                        success=True
                    )
                    
                except Exception as e:
                    query_time = (time.time() - query_start) * 1000
                    metrics = QueryMetrics(
                        timestamp=time.time(),
                        query_time_ms=query_time,
                        top_k=10,
                        results_count=0,
                        success=False,
                        error=str(e)
                    )
                
                self.query_metrics.append(metrics)
                
                # Aguardar próxima query
                time.sleep(max(0, query_interval - (time.time() - query_start)))
                
            except Exception as e:
                logger.error(f"Erro na simulação {self.name}: {e}")
                break
        
        self.running = False
        logger.info(f"Simulação {self.name} concluída: {len(self.query_metrics)} queries")
    
    def simulate_recommendation_system(self, duration_seconds: int, users_per_second: float):
        """Simular sistema de recomendação"""
        logger.info(f"Simulando sistema de recomendação: {self.name}")
        
        self.running = True
        start_time = time.time()
        user_interval = 1.0 / users_per_second
        dimensions = self.index.dims
        
        while self.running and (time.time() - start_time) < duration_seconds:
            try:
                # Usuário faz múltiplas queries (perfil + histórico)
                for _ in range(np.random.randint(1, 4)):  # 1-3 queries por usuário
                    query_vector = np.random.randn(dimensions).astype(np.float32)
                    query_vector = query_vector / np.linalg.norm(query_vector)
                    
                    query_start = time.time()
                    try:
                        results = self.index.query(query_vector, top_k=20)  # Mais resultados para recomendação
                        query_time = (time.time() - query_start) * 1000
                        
                        metrics = QueryMetrics(
                            timestamp=time.time(),
                            query_time_ms=query_time,
                            top_k=20,
                            results_count=len(results),
                            success=True
                        )
                        
                    except Exception as e:
                        query_time = (time.time() - query_start) * 1000
                        metrics = QueryMetrics(
                            timestamp=time.time(),
                            query_time_ms=query_time,
                            top_k=20,
                            results_count=0,
                            success=False,
                            error=str(e)
                        )
                    
                    self.query_metrics.append(metrics)
                
                time.sleep(user_interval)
                
            except Exception as e:
                logger.error(f"Erro na simulação de recomendação {self.name}: {e}")
                break
        
        self.running = False
        logger.info(f"Sistema de recomendação {self.name} concluído")
    
    def simulate_batch_processing(self, duration_seconds: int, batch_size: int, batches_per_minute: float):
        """Simular processamento em lote"""
        logger.info(f"Simulando processamento batch: {self.name}")
        
        self.running = True
        start_time = time.time()
        batch_interval = 60.0 / batches_per_minute
        dimensions = self.index.dims
        
        while self.running and (time.time() - start_time) < duration_seconds:
            try:
                # Processar lote de queries
                batch_start = time.time()
                batch_queries = []
                
                for _ in range(batch_size):
                    query_vector = np.random.randn(dimensions).astype(np.float32)
                    query_vector = query_vector / np.linalg.norm(query_vector)
                    batch_queries.append(query_vector)
                
                # Executar batch
                for query_vector in batch_queries:
                    query_start = time.time()
                    try:
                        results = self.index.query(query_vector, top_k=5)
                        query_time = (time.time() - query_start) * 1000
                        
                        metrics = QueryMetrics(
                            timestamp=time.time(),
                            query_time_ms=query_time,
                            top_k=5,
                            results_count=len(results),
                            success=True
                        )
                        
                    except Exception as e:
                        query_time = (time.time() - query_start) * 1000
                        metrics = QueryMetrics(
                            timestamp=time.time(),
                            query_time_ms=query_time,
                            top_k=5,
                            results_count=0,
                            success=False,
                            error=str(e)
                        )
                    
                    self.query_metrics.append(metrics)
                
                batch_time = time.time() - batch_start
                logger.info(f"Batch processado: {batch_size} queries em {batch_time:.2f}s")
                
                # Aguardar próximo batch
                time.sleep(max(0, batch_interval - batch_time))
                
            except Exception as e:
                logger.error(f"Erro no processamento batch {self.name}: {e}")
                break
        
        self.running = False
        logger.info(f"Processamento batch {self.name} concluído")
    
    def stop(self):
        """Parar simulação"""
        self.running = False
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Obter estatísticas de performance"""
        if not self.query_metrics:
            return {}
        
        successful_queries = [m for m in self.query_metrics if m.success]
        failed_queries = [m for m in self.query_metrics if not m.success]
        
        if not successful_queries:
            return {
                'total_queries': len(self.query_metrics),
                'successful_queries': 0,
                'failed_queries': len(failed_queries),
                'success_rate': 0.0
            }
        
        query_times = [m.query_time_ms for m in successful_queries]
        
        # Calcular duração total
        if len(self.query_metrics) >= 2:
            total_duration = self.query_metrics[-1].timestamp - self.query_metrics[0].timestamp
            qps = len(successful_queries) / total_duration if total_duration > 0 else 0
        else:
            qps = 0
        
        return {
            'total_queries': len(self.query_metrics),
            'successful_queries': len(successful_queries),
            'failed_queries': len(failed_queries),
            'success_rate': len(successful_queries) / len(self.query_metrics) * 100,
            'avg_query_time_ms': np.mean(query_times),
            'median_query_time_ms': np.median(query_times),
            'p95_query_time_ms': np.percentile(query_times, 95),
            'p99_query_time_ms': np.percentile(query_times, 99),
            'min_query_time_ms': np.min(query_times),
            'max_query_time_ms': np.max(query_times),
            'qps': qps
        }

class ProductionReadinessTest:
    """Teste abrangente de prontidão para produção"""
    
    def __init__(self):
        self.system_monitor = SystemMonitor(interval=1.0)
        self.applications: List[ApplicationSimulator] = []
        
    def create_production_dataset(self, scenario: str) -> Tuple[np.ndarray, int]:
        """Criar datasets realistas baseados em cenários"""
        scenarios = {
            'small_company': {
                'vectors': 5000,
                'dimensions': 384,
                'description': 'Pequena empresa - documentos internos'
            },
            'medium_ecommerce': {
                'vectors': 50000,
                'dimensions': 512,
                'description': 'E-commerce médio - catálogo de produtos'
            },
            'large_content': {
                'vectors': 200000,
                'dimensions': 768,
                'description': 'Plataforma de conteúdo - artigos e posts'
            },
            'enterprise': {
                'vectors': 1000000,
                'dimensions': 1024,
                'description': 'Empresa grande - base de conhecimento'
            }
        }
        
        if scenario not in scenarios:
            raise ValueError(f"Cenário inválido: {scenario}")
        
        config = scenarios[scenario]
        logger.info(f"Criando dataset: {config['description']}")
        logger.info(f"Vetores: {config['vectors']:,}, Dimensões: {config['dimensions']}")
        
        # Gerar dados realísticos (não completamente aleatórios)
        vectors = np.random.randn(config['vectors'], config['dimensions']).astype(np.float32)
        
        # Simular clusters realísticos
        num_clusters = max(10, config['vectors'] // 1000)
        cluster_centers = np.random.randn(num_clusters, config['dimensions']).astype(np.float32)
        
        for i in range(config['vectors']):
            cluster_id = i % num_clusters
            noise_level = 0.3
            vectors[i] = (1 - noise_level) * cluster_centers[cluster_id] + noise_level * vectors[i]
        
        # Normalizar
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        vectors = vectors / norms
        
        memory_usage = vectors.nbytes / (1024 * 1024)
        logger.info(f"Dataset criado: {memory_usage:.1f}MB")
        
        return vectors, config['dimensions']
    
    def test_startup_performance(self, vectors: np.ndarray) -> Tuple[Dict[str, Any], Any]:
        """Testar performance de inicialização"""
        logger.info("Testando performance de startup")
        
        # Limpar cache
        gc.collect()
        
        # Medir tempo de construção do índice
        start_time = time.time()
        
        import nseekfs
        index = nseekfs.from_embeddings(
            vectors,
            metric="cosine",
            base_name="production_test",
            normalized=True
        )
        
        build_time = time.time() - start_time
        
        # Estatísticas
        memory_usage = index.memory_usage_mb
        vectors_count = index.rows
        dimensions = index.dims
        
        startup_results = {
            'build_time_s': build_time,
            'memory_usage_mb': memory_usage,
            'vectors_count': vectors_count,
            'dimensions': dimensions,
            'throughput_vectors_per_sec': vectors_count / build_time if build_time > 0 else 0.0,
            'memory_efficiency_mb_per_1k_vectors': (memory_usage / vectors_count) * 1000 if vectors_count > 0 else 0.0
        }
        
        logger.info(f"Índice construído em {build_time:.2f}s")
        logger.info(f"{vectors_count:,} vetores × {dimensions} dim = {memory_usage:.1f}MB")
        
        return startup_results, index
    
    def test_concurrent_load(self, index, duration: int, load_level: str) -> Dict[str, Any]:
        """Testar carga concorrente com diferentes aplicações"""
        logger.info(f"Testando carga concorrente: {load_level} por {duration}s")
        
        # Configurações de carga
        load_configs = {
            'light': {
                'search_apps': 2,
                'search_qps': 5,
                'rec_apps': 1,
                'rec_users_per_sec': 2,
                'batch_apps': 1,
                'batch_per_min': 6
            },
            'medium': {
                'search_apps': 4,
                'search_qps': 10,
                'rec_apps': 2,
                'rec_users_per_sec': 5,
                'batch_apps': 2,
                'batch_per_min': 12
            },
            'heavy': {
                'search_apps': 8,
                'search_qps': 20,
                'rec_apps': 4,
                'rec_users_per_sec': 10,
                'batch_apps': 3,
                'batch_per_min': 24
            }
        }
        
        config = load_configs[load_level]
        
        # Iniciar monitoramento
        self.system_monitor.start()
        self.applications = []
        threads = []
        
        # Aplicações de busca
        for i in range(config['search_apps']):
            app = ApplicationSimulator(index, f"search_app_{i}")
            self.applications.append(app)
            
            thread = threading.Thread(
                target=app.simulate_search_application,
                args=(duration, config['search_qps'])
            )
            threads.append(thread)
        
        # Sistemas de recomendação
        for i in range(config['rec_apps']):
            app = ApplicationSimulator(index, f"recommendation_{i}")
            self.applications.append(app)
            
            thread = threading.Thread(
                target=app.simulate_recommendation_system,
                args=(duration, config['rec_users_per_sec'])
            )
            threads.append(thread)
        
        # Processamento batch
        for i in range(config['batch_apps']):
            app = ApplicationSimulator(index, f"batch_processor_{i}")
            self.applications.append(app)
            
            thread = threading.Thread(
                target=app.simulate_batch_processing,
                args=(duration, 50, config['batch_per_min'])  # 50 queries por batch
            )
            threads.append(thread)
        
        # Iniciar todas as aplicações
        logger.info(f"Iniciando {len(threads)} aplicações simuladas")
        start_time = time.time()
        
        for thread in threads:
            thread.start()
        
        # Aguardar conclusão
        for thread in threads:
            thread.join()
        
        total_time = time.time() - start_time
        
        # Parar monitoramento
        self.system_monitor.stop()
        
        # Coletar resultados
        results = {
            'duration_s': total_time,
            'load_level': load_level,
            'applications': len(self.applications),
            'system_stats': self.system_monitor.get_current_stats(),
            'app_stats': {}
        }
        
        total_queries = 0
        total_successful = 0
        all_query_times = []
        
        for app in self.applications:
            app_stats = app.get_performance_stats()
            results['app_stats'][app.name] = app_stats
            
            if 'total_queries' in app_stats:
                total_queries += app_stats['total_queries']
                total_successful += app_stats['successful_queries']
                
                # Coletar tempos de query
                successful_metrics = [m for m in app.query_metrics if m.success]
                all_query_times.extend([m.query_time_ms for m in successful_metrics])
        
        # Estatísticas globais
        if all_query_times and total_time > 0:
            results['global_stats'] = {
                'total_queries': total_queries,
                'successful_queries': total_successful,
                'global_success_rate': (total_successful / total_queries * 100) if total_queries > 0 else 0,
                'global_qps': total_successful / total_time,
                'global_avg_time_ms': np.mean(all_query_times),
                'global_p95_time_ms': np.percentile(all_query_times, 95),
                'global_p99_time_ms': np.percentile(all_query_times, 99)
            }
        
        logger.info(f"Teste de carga concluído: {total_successful}/{total_queries} queries bem-sucedidas")
        
        return results
    
    def test_stability_over_time(self, index, duration: int = 3600) -> Dict[str, Any]:
        """Testar estabilidade ao longo do tempo"""
        logger.info(f"Testando estabilidade por {duration}s ({duration//3600}h)")
        
        self.system_monitor.start()
        
        # Carga leve mas constante
        app = ApplicationSimulator(index, "stability_test")
        
        # Thread de simulação
        thread = threading.Thread(
            target=app.simulate_search_application,
            args=(duration, 5)  # 5 QPS constante
        )
        
        start_time = time.time()
        thread.start()
        
        # Monitorar progresso
        last_report = start_time
        report_interval = 300  # Relatório a cada 5 minutos
        
        while thread.is_alive():
            time.sleep(10)
            
            current_time = time.time()
            if current_time - last_report >= report_interval:
                elapsed = current_time - start_time
                progress = (elapsed / duration) * 100
                
                system_stats = self.system_monitor.get_current_stats()
                app_stats = app.get_performance_stats()
                
                logger.info(f"Progresso: {progress:.1f}% - "
                          f"CPU: {system_stats.get('avg_cpu_percent', 0):.1f}% - "
                          f"Mem: {system_stats.get('avg_memory_percent', 0):.1f}% - "
                          f"QPS: {app_stats.get('qps', 0):.1f}")
                
                last_report = current_time
        
        thread.join()
        self.system_monitor.stop()
        
        # Análise de estabilidade
        app_stats = app.get_performance_stats()
        system_stats = self.system_monitor.get_current_stats()
        
        # Analisar degradação ao longo do tempo
        successful_metrics = [m for m in app.query_metrics if m.success]
        
        if len(successful_metrics) > 100:
            # Dividir em períodos para análise de tendência
            period_size = len(successful_metrics) // 10
            periods = [successful_metrics[i:i+period_size] for i in range(0, len(successful_metrics), period_size)]
            
            period_avg_times = [np.mean([m.query_time_ms for m in period]) for period in periods if period]
            
            # Detectar degradação
            if len(period_avg_times) >= 2:
                degradation_rate = (period_avg_times[-1] - period_avg_times[0]) / period_avg_times[0] * 100
            else:
                degradation_rate = 0
        else:
            degradation_rate = 0
        
        stability_results = {
            'duration_s': duration,
            'app_stats': app_stats,
            'system_stats': system_stats,
            'degradation_rate_percent': degradation_rate,
            'stable': abs(degradation_rate) < 10  # Considerado estável se degradação < 10%
        }
        
        logger.info(f"Teste de estabilidade concluído - Degradação: {degradation_rate:.1f}%")
        
        return stability_results
    
    def test_error_recovery(self, index) -> Dict[str, Any]:
        """Testar recuperação de erros"""
        logger.info("Testando recuperação de erros")
        
        error_tests = []
        
        # Teste 1: Queries com dimensões incorretas
        try:
            wrong_vector = np.random.randn(64).astype(np.float32)  # Dimensão errada
            index.query(wrong_vector, top_k=5)
            error_tests.append({'test': 'wrong_dimensions', 'handled': False})
        except Exception as e:
            error_tests.append({'test': 'wrong_dimensions', 'handled': True, 'error': str(e)})
        
        # Teste 2: Queries com valores inválidos
        try:
            nan_vector = np.full(index.dims, np.nan, dtype=np.float32)
            index.query(nan_vector, top_k=5)
            error_tests.append({'test': 'nan_values', 'handled': False})
        except Exception as e:
            error_tests.append({'test': 'nan_values', 'handled': True, 'error': str(e)})
        
        # Teste 3: Queries com top_k inválido
        try:
            query_vector = np.random.randn(index.dims).astype(np.float32)
            index.query(query_vector, top_k=-1)
            error_tests.append({'test': 'invalid_top_k', 'handled': False})
        except Exception as e:
            error_tests.append({'test': 'invalid_top_k', 'handled': True, 'error': str(e)})
        
        # Teste 4: Stress de memória (consulta válida)
        try:
            query_vector = np.random.randn(index.dims).astype(np.float32)
            results = index.query(query_vector, top_k=min(1000, index.rows))
            error_tests.append({'test': 'large_top_k', 'handled': True, 'results': len(results)})
        except Exception as e:
            error_tests.append({'test': 'large_top_k', 'handled': False, 'error': str(e)})
        
        handled_errors = sum(1 for test in error_tests if test['handled'])
        total_tests = len(error_tests)
        
        error_recovery_results = {
            'tests_run': total_tests,
            'errors_handled': handled_errors,
            'error_handling_rate': handled_errors / total_tests * 100 if total_tests > 0 else 0.0,
            'test_details': error_tests
        }
        
        logger.info(f"Recuperação de erros: {handled_errors}/{total_tests} tratados adequadamente")
        
        return error_recovery_results
    
    def generate_production_report(self, all_results: Dict[str, Any]):
        """Gerar relatório abrangente de prontidão"""
        print("\n" + "="*80)
        print(colored("NSeekFS v1.0 - RELATÓRIO DE PRONTIDÃO PARA PRODUÇÃO", Colors.BOLD + Colors.BLUE))
        print("="*80)
        
        # Análise de startup
        startup = all_results.get('startup', {})
        if startup:
            print(colored("\nPERFORMANCE DE STARTUP:", Colors.BOLD + Colors.GREEN))
            print(f"   Tempo de construção: {startup['build_time_s']:.2f}s")
            print(f"   Uso de memória: {startup['memory_usage_mb']:.1f}MB")
            print(f"   Throughput: {startup['throughput_vectors_per_sec']:.0f} vetores/s")
            print(f"   Eficiência: {startup['memory_efficiency_mb_per_1k_vectors']:.2f}MB/1k vetores")
        
        # Análise de carga concorrente
        concurrent = all_results.get('concurrent_load', {})
        if concurrent:
            print(colored("\nTESTE DE CARGA CONCORRENTE:", Colors.BOLD + Colors.YELLOW))
            print(f"   Nível de carga: {concurrent['load_level']}")
            print(f"   Aplicações simuladas: {concurrent['applications']}")
            
            global_stats = concurrent.get('global_stats', {})
            if global_stats:
                print(f"   QPS global: {global_stats['global_qps']:.1f}")
                print(f"   Taxa de sucesso: {global_stats['global_success_rate']:.1f}%")
                print(f"   Tempo médio: {global_stats['global_avg_time_ms']:.2f}ms")
                print(f"   P95: {global_stats['global_p95_time_ms']:.2f}ms")
                print(f"   P99: {global_stats['global_p99_time_ms']:.2f}ms")
            
            system_stats = concurrent.get('system_stats', {})
            if system_stats:
                print(f"   CPU médio: {system_stats.get('avg_cpu_percent', 0):.1f}%")
                print(f"   Memória média: {system_stats.get('avg_memory_percent', 0):.1f}%")
                print(f"   Pico de CPU: {system_stats.get('peak_cpu_percent', 0):.1f}%")
        
        # Análise de estabilidade
        stability = all_results.get('stability', {})
        if stability:
            print(colored("\nTESTE DE ESTABILIDADE:", Colors.BOLD + Colors.CYAN))
            print(f"   Duração: {stability['duration_s']//3600}h {(stability['duration_s']%3600)//60}m")
            print(f"   Degradação: {stability['degradation_rate_percent']:.1f}%")
            
            if stability['stable']:
                print(colored("   Sistema ESTÁVEL ao longo do tempo", Colors.GREEN))
            else:
                print(colored("   Degradação detectada", Colors.YELLOW))
            
            app_stats = stability.get('app_stats', {})
            if app_stats:
                print(f"   QPS médio: {app_stats.get('qps', 0):.1f}")
                print(f"   Taxa de sucesso: {app_stats.get('success_rate', 0):.1f}%")
        
        # Análise de recuperação de erros
        error_recovery = all_results.get('error_recovery', {})
        if error_recovery:
            print(colored("\nRECUPERAÇÃO DE ERROS:", Colors.BOLD + Colors.MAGENTA))
            print(f"   Testes executados: {error_recovery['tests_run']}")
            print(f"   Erros tratados: {error_recovery['errors_handled']}")
            print(f"   Taxa de tratamento: {error_recovery['error_handling_rate']:.1f}%")
            
            if error_recovery['error_handling_rate'] >= 75:
                print(colored("   Tratamento de erros ROBUSTO", Colors.GREEN))
            else:
                print(colored("   Melhorar tratamento de erros", Colors.YELLOW))
        
        # Calcular pontuação geral
        scores = []
        
        # Pontuação de startup (0-25 pontos)
        if startup:
            startup_score = min(25, max(0, 25 - (startup['build_time_s'] - 1) * 5))
            scores.append(startup_score)
        
        # Pontuação de carga (0-35 pontos)
        if concurrent and concurrent.get('global_stats'):
            global_stats = concurrent['global_stats']
            success_score = (global_stats['global_success_rate'] / 100) * 20
            performance_score = min(15, max(0, 15 - (global_stats['global_avg_time_ms'] - 10) * 0.5))
            scores.extend([success_score, performance_score])
        
        # Pontuação de estabilidade (0-25 pontos)
        if stability:
            if stability['stable']:
                stability_score = 25
            else:
                degradation = abs(stability['degradation_rate_percent'])
                stability_score = max(0, 25 - degradation)
            scores.append(stability_score)
        
        # Pontuação de recuperação (0-15 pontos)
        if error_recovery:
            error_score = (error_recovery['error_handling_rate'] / 100) * 15
            scores.append(error_score)
        
        overall_score = sum(scores)
        max_possible = 100
        
        print(colored(f"\nPONTUAÇÃO GERAL DE PRONTIDÃO:", Colors.BOLD + Colors.WHITE))
        print(f"   Pontuação: {overall_score:.1f}/{max_possible}")
        print(f"   Percentual: {overall_score/max_possible*100:.1f}%")
        
        # Veredito final
        print(colored("\nVEREDITO FINAL:", Colors.BOLD + Colors.WHITE))
        
        if overall_score >= 90:
            print(colored("   NSeekFS v1.0 está PRONTO para produção!", Colors.BOLD + Colors.GREEN))
            print("   Deploy recomendado com confiança")
        elif overall_score >= 75:
            print(colored("   NSeekFS v1.0 está ADEQUADO para produção", Colors.GREEN))
            print("   Implementar monitoramento adicional")
        elif overall_score >= 60:
            print(colored("   NSeekFS v1.0 precisa de MELHORIAS", Colors.YELLOW))
            print("   Resolver problemas identificados primeiro")
        else:
            print(colored("   NSeekFS v1.0 NÃO está pronto para produção", Colors.RED))
            print("   Otimizações significativas necessárias")
        
        print(colored("="*80, Colors.BOLD))

def main():
    """Função principal"""
    parser = argparse.ArgumentParser(description='NSeekFS v1.0 - Teste de Prontidão para Produção')
    parser.add_argument('--duration', type=int, default=300, help='Duração dos testes (segundos)')
    parser.add_argument('--load', choices=['light', 'medium', 'heavy'], default='medium', help='Nível de carga')
    parser.add_argument('--scenario', choices=['small_company', 'medium_ecommerce', 'large_content', 'enterprise'], 
                       default='medium_ecommerce', help='Cenário de dataset')
    parser.add_argument('--stability-duration', type=int, default=900, help='Duração do teste de estabilidade (segundos)')
    parser.add_argument('--skip-stability', action='store_true', help='Pular teste de estabilidade longa')
    parser.add_argument('--export', type=str, help='Exportar resultados para arquivo JSON')
    
    args = parser.parse_args()
    
    print(colored("NSeekFS v1.0 - TESTE DE PRONTIDÃO PARA PRODUÇÃO", Colors.BOLD + Colors.BLUE))
    print(colored("=" * 70, Colors.BLUE))
    
    # Verificar NSeekFS
    try:
        import nseekfs
        logger.info(f"NSeekFS v{getattr(nseekfs, '__version__', '1.0.0')} disponível")
    except ImportError:
        logger.error("NSeekFS não encontrado. Instale com: pip install nseekfs")
        return 1
    
    # Criar teste
    test = ProductionReadinessTest()
    all_results = {}
    
    try:
        # 1. Criar dataset de produção
        logger.info(f"Criando dataset para cenário: {args.scenario}")
        vectors, dimensions = test.create_production_dataset(args.scenario)
        
        # 2. Teste de startup
        startup_results, index = test.test_startup_performance(vectors)
        all_results['startup'] = startup_results
        
        # 3. Teste de carga concorrente
        load_results = test.test_concurrent_load(index, args.duration, args.load)
        all_results['concurrent_load'] = load_results
        
        # 4. Teste de estabilidade (opcional)
        if not args.skip_stability:
            stability_results = test.test_stability_over_time(index, args.stability_duration)
            all_results['stability'] = stability_results
        
        # 5. Teste de recuperação de erros
        error_results = test.test_error_recovery(index)
        all_results['error_recovery'] = error_results
        
        # 6. Gerar relatório
        test.generate_production_report(all_results)
        
        # 7. Exportar se solicitado
        if args.export:
            with open(args.export, 'w') as f:
                json.dump(all_results, f, indent=2, default=str)
            logger.info(f"Resultados exportados para {args.export}")
        
        logger.info("Teste de prontidão para produção concluído!")
        return 0
        
    except KeyboardInterrupt:
        logger.warning("Teste interrompido pelo usuário")
        return 2
    except Exception as e:
        logger.error(f"Erro no teste de produção: {e}")
        import traceback
        traceback.print_exc()
        return 3
    finally:
        # Cleanup
        if hasattr(test, 'system_monitor'):
            test.system_monitor.stop()
        for app in test.applications:
            app.stop()

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
