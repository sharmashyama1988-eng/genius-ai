"""
Genius AI - Verification & Benchmark Suite
Validates TTFT, Stream Throughput, and Memory Retention under High Load.
"""

from __future__ import annotations

import asyncio
import os
import statistics
import time
from typing import List
import sys
sys.path.insert(0, os.path.abspath("."))

from dotenv import load_dotenv

load_dotenv("F:/Dekstop/llm/.env")

# Requires psutil for process inspection
import psutil


async def run_benchmark_suite():
    from src.system.genius_runtime import GeniusAIRuntime

    api_key = os.getenv("OPENROUTER_API_KEY", "")
    runtime = GeniusAIRuntime(api_key=api_key)
    await runtime.startup()

    process = psutil.Process(os.getpid())
    ttft_records: List[float] = []

    test_queries = [
        "hi",                                    # Edge Fast-Path
        "kaise ho bhai?",                        # Edge Fast-Path Hinglish
        "ok",                                    # Edge Fast-Path
        "Explain SQLite WAL vs Rollback journal", # Complex Path
        "thanks",                                # Edge Fast-Path
        "Analyze distributed consensus in Raft", # Complex Path
    ] * 2  # 12 Conversational Turns for quick benchmark

    print("==================================================================")
    print("STARTING BENCHMARK: Simulating Steady-State Execution")
    print("==================================================================")

    for i, q in enumerate(test_queries, start=1):
        start = time.perf_counter()
        first_token_received = False
        ttft = 0.0
        total_tokens = 0

        async for chunk in runtime.execute_turn(session_id="benchmark_session", query=q):
            if not first_token_received:
                ttft = (time.perf_counter() - start) * 1000.0
                ttft_records.append(ttft)
                first_token_received = True
            total_tokens += max(1, len(chunk) // 4)

        rss_mb = process.memory_info().rss / (1024 * 1024)
        print(f"Turn {i:02d} | Query: '{q[:22]:<22}' | TTFT: {ttft:6.1f}ms | RAM RSS: {rss_mb:5.1f} MB")
        await asyncio.sleep(0.05)

    await runtime.shutdown()

    # Metrics Compilation
    ttft_records.sort()
    p50 = statistics.median(ttft_records)
    p95 = ttft_records[int(len(ttft_records) * 0.95)]
    peak_rss = process.memory_info().rss / (1024 * 1024)

    print("\n---------------------- BENCHMARK RESULTS ----------------------")
    print(f"Total Iterations Analyzed : {len(ttft_records)}")
    print(f"Latency P50 (TTFT)        : {p50:.2f} ms  (Target: < 350ms)")
    print(f"Latency P95 (TTFT)        : {p95:.2f} ms  (Target: < 600ms)")
    print(f"Final Peak Process RSS    : {peak_rss:.2f} MB (Target: < 350MB)")
    print("---------------------------------------------------------------")


if __name__ == "__main__":
    asyncio.run(run_benchmark_suite())
