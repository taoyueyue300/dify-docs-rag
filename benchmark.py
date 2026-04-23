# -*- coding: utf-8 -*-
"""dify-docs-rag retrieval latency benchmark.

Measures HybridRetriever latency at scale:
  - cold start (first query)
  - steady-state P50 / P95 / P99 across N repeats

Usage: python benchmark.py [--repeats 50]
Output: benchmark_report.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))


TEST_QUERIES = [
    "How to deploy Dify locally with Docker?",
    "Which vector databases does Dify support?",
    "How does Dify RAG knowledge base work?",
    "How to call Dify API from external apps?",
    "What file formats can Dify ingest?",
    "How to build a chat assistant in Dify?",
    "How to configure environment variables in Dify?",
    "Dify workflow node design and orchestration?",
    "How to contribute code to the Dify project?",
    "Dify system architecture overview frontend backend",
]


def run_bench(repeats):
    from retriever import HybridRetriever

    print("[init] loading HybridRetriever ...")
    t0 = time.perf_counter()
    retriever = HybridRetriever(use_reranker=False)
    init_time = time.perf_counter() - t0
    print("[init] done in %.2fs" % init_time)

    n_chunks = len(retriever.all_docs)
    print("[init] index size: %d chunks" % n_chunks)

    # cold start
    cold_start = time.perf_counter()
    _ = retriever.retrieve(TEST_QUERIES[0], top_k=5)
    cold_ms = (time.perf_counter() - cold_start) * 1000
    print("[cold] first query: %.2f ms" % cold_ms)

    # warm up
    for q in TEST_QUERIES:
        retriever.retrieve(q, top_k=5)

    # steady-state
    samples = []
    for _ in range(repeats):
        for q in TEST_QUERIES:
            t = time.perf_counter()
            retriever.retrieve(q, top_k=5)
            samples.append((time.perf_counter() - t) * 1000)

    samples.sort()
    n = len(samples)
    return {
        "init_seconds": round(init_time, 2),
        "index_chunks": n_chunks,
        "cold_start_ms": round(cold_ms, 2),
        "samples": n,
        "p50_ms": round(samples[n // 2], 3),
        "p90_ms": round(samples[int(n * 0.9)], 3),
        "p95_ms": round(samples[int(n * 0.95)], 3),
        "p99_ms": round(samples[int(n * 0.99)] if n > 100 else samples[-1], 3),
        "max_ms": round(samples[-1], 3),
        "mean_ms": round(statistics.mean(samples), 3),
        "qps_steady": round(1000.0 / statistics.mean(samples), 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=20,
                        help="repeat the test query set this many times")
    parser.add_argument("--report", default="benchmark_report.json")
    args = parser.parse_args()

    print("=" * 70)
    print("  dify-docs-rag benchmark  " + datetime.now().isoformat(timespec="seconds"))
    print("=" * 70)

    result = run_bench(args.repeats)

    print("\n[result]")
    for k, v in result.items():
        print("  %-18s : %s" % (k, v))

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": vars(args),
        "retrieval": result,
        "notes": "End-to-end latency for HybridRetriever (FAISS vector + BM25, no reranker). "
                 "Hardware: local CPU. Embedding model: BAAI/bge-small-zh-v1.5.",
    }
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\nOK report written to: " + args.report)


if __name__ == "__main__":
    main()
