# Positioning: Where NSeekFS Is Strong

## Core positioning

NSeekFS is best positioned as:

- exact vector search engine for Python
- deterministic and auditable ranking pipeline
- certified pruning with correctness guarantees

It is not positioned as a generic ANN-first platform.

## Where it creates most value

- regulated or high-trust domains (finance, legal, healthcare)
- model evaluation pipelines where ranking reproducibility matters
- forensic/debug workflows requiring replay and audit trails
- workloads that need exact top-k but still benefit from safe pruning

## Where competitors may be better

- ultra-large scale ANN-first serving where small recall loss is acceptable
- teams needing mature ANN ecosystem integrations out-of-the-box
- use cases prioritizing peak throughput over auditability

## Decision guide

Choose NSeekFS when you need:

- exact semantics
- deterministic tie-breaking
- replayable and inspectable results
- certificate-backed pruning safety

Choose ANN engines when you need:

- approximate recall tradeoffs at extreme scale
- extensive ANN ecosystem features

## Benchmark policy

- Correctness gate is mandatory (build fails on divergence).
- External benchmark tracking is informative (latency/recall), not a correctness gate.
- Competitive benchmark script: `bench/benchmark_competitors.py`.
