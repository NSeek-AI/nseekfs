# Competitive Benchmarks

This page defines the standard format for publishing benchmark results against similar tools.

## Standard Scenario Set

Scenarios are versioned in:

- `bench/competitive_scenarios.json`

Current matrix includes small/medium/large sizes and multiple metrics.

## How to Run

```bash
python bench/run_competitive_suite.py --scenarios bench/competitive_scenarios.json --out-dir benchmark_artifacts
```

Outputs:

- `benchmark_artifacts/competitive_suite_summary.json`
- `benchmark_artifacts/competitive_suite_summary.md`

## Publication Rules

- Always report hardware/OS/Python info.
- Keep the same scenario matrix when comparing commits.
- Use NSeekFS brute force as exact ground truth.
- For certified mode, require recall@k = 1.0 vs exact.
- ANN tool recall should be shown explicitly.

## Result Template

| scenario | metric | rows | dims | queries | top_k | nseek_gt_ms | nseek_cert_ms | nseek_recall | nseek_pruned_pct | faiss_ms | faiss_recall | hnsw_ms | hnsw_recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| s1_small_cosine | cosine | 10000 | 128 | 100 | 10 | ... | ... | ... | ... | ... | ... | ... | ... |
| s2_medium_dot | dot | 50000 | 384 | 200 | 10 | ... | ... | ... | ... | ... | ... | ... | ... |
| s3_medium_euclidean | euclidean | 50000 | 384 | 200 | 10 | ... | ... | ... | ... | ... | ... | ... | ... |
| s4_large_cosine | cosine | 100000 | 384 | 200 | 10 | ... | ... | ... | ... | ... | ... | ... | ... |

## Interpretation

- `nseek_recall` should stay at `1.0`.
- `nseek_pruned_pct` indicates how much certified pruning reduced work.
- FAISS/HNSW numbers are comparative references, not correctness authorities.
