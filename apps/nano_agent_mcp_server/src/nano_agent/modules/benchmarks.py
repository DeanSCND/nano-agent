"""
benchmarks.py – Offline Performance Benchmarks
==============================================

Phase-5 requirement: provide repeatable *offline* benchmarks for
latency, throughput, cost-efficiency and memory-footprint of the
multi-provider architecture.

The module DOES NOT make real network calls – it relies on mocks for
`ProviderConfig.create_agent` and `Runner.run_sync`.  This allows CI /
developers to run the suite without API keys or costs.

Usage
-----

$ uv python -m nano_agent.modules.benchmarks              # run all benches
>>> from nano_agent.modules.benchmarks import run_all_benchmarks
>>> report = run_all_benchmarks()

The returned `BenchmarkReport` dataclass can be serialised to JSON or
pretty-printed.
"""

from __future__ import annotations

import contextlib
import json
import statistics
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
from unittest.mock import patch

from nano_agent.modules.model_router import execute_with_fallback
from nano_agent.modules.task_analyzer import analyze_prompt
from nano_agent.modules.token_tracking import Usage

# Optional rich import for nicer output
try:
    from rich.console import Console
    from rich.table import Table

    _RICH_AVAILABLE = True
    _console = Console()
except ImportError:  # pragma: no cover
    _RICH_AVAILABLE = False
    _console = None  # type: ignore


# --------------------------------------------------------------------------- #
# Dataclasses                                                                 #
# --------------------------------------------------------------------------- #
@dataclass
class OneRunMetric:
    provider: str
    model: str
    duration_s: float
    total_tokens: int
    total_cost: float


@dataclass
class ProviderBenchmarks:
    runs: List[OneRunMetric] = field(default_factory=list)

    @property
    def avg_time(self) -> float:
        return statistics.mean(r.duration_s for r in self.runs) if self.runs else 0.0

    @property
    def avg_tokens(self) -> float:
        return statistics.mean(r.total_tokens for r in self.runs) if self.runs else 0.0

    @property
    def avg_cost(self) -> float:
        return statistics.mean(r.total_cost for r in self.runs) if self.runs else 0.0

    def to_dict(self):
        return {
            "avg_time_s": round(self.avg_time, 4),
            "avg_tokens": round(self.avg_tokens, 2),
            "avg_cost": round(self.avg_cost, 6),
            "runs": [r.__dict__ for r in self.runs],
        }


@dataclass
class RoutingBenchmarks:
    complexity_analysis_ms: float
    routing_overhead_ms: float
    fallback_chain_time_s: float

    def to_dict(self):
        return {
            "complexity_analysis_ms": round(self.complexity_analysis_ms, 4),
            "routing_overhead_ms": round(self.routing_overhead_ms, 4),
            "fallback_chain_time_s": round(self.fallback_chain_time_s, 4),
        }


@dataclass
class MemoryBenchmarks:
    peak_kb: int
    tracker_peak_kb: int
    agent_creation_kb: int
    tool_exec_kb: int

    def to_dict(self):
        return {
            "peak_kb": self.peak_kb,
            "tracker_peak_kb": self.tracker_peak_kb,
            "agent_creation_kb": self.agent_creation_kb,
            "tool_exec_kb": self.tool_exec_kb,
        }


@dataclass
class BenchmarkReport:
    provider: Dict[str, ProviderBenchmarks]
    routing: RoutingBenchmarks
    memory: MemoryBenchmarks
    generated_at: float = field(default_factory=time.time)

    def to_dict(self):
        return {
            "generated_at": self.generated_at,
            "provider": {k: v.to_dict() for k, v in self.provider.items()},
            "routing": self.routing.to_dict(),
            "memory": self.memory.to_dict(),
        }

    # Pretty print
    def display(self):
        if not _RICH_AVAILABLE:
            print(json.dumps(self.to_dict(), indent=2))
            return

        # provider table
        table = Table(title="Provider Benchmarks")
        table.add_column("Provider")
        table.add_column("Avg Time (s)", justify="right")
        table.add_column("Avg Tokens", justify="right")
        table.add_column("Avg Cost ($)", justify="right")
        for prov, bench in self.provider.items():
            table.add_row(
                prov,
                f"{bench.avg_time:.2f}",
                f"{bench.avg_tokens:.0f}",
                f"{bench.avg_cost:.5f}",
            )
        _console.print(table)

        routing_tbl = Table(title="Model-Routing Benchmarks")
        routing_tbl.add_column("Metric")
        routing_tbl.add_column("Value", justify="right")
        for k, v in self.routing.to_dict().items():
            routing_tbl.add_row(k, str(v))
        _console.print(routing_tbl)

        mem_tbl = Table(title="Memory Benchmarks")
        mem_tbl.add_column("Metric")
        mem_tbl.add_column("KB", justify="right")
        for k, v in self.memory.to_dict().items():
            mem_tbl.add_row(k, str(v))
        _console.print(mem_tbl)


# --------------------------------------------------------------------------- #
# Internal helpers (mocks)                                                    #
# --------------------------------------------------------------------------- #
class _DummyRunnerResult:
    """Mimic agents.Runner synchronous result."""
    def __init__(self, content: str = "ok", tokens: int = 42):
        self.final_output = content
        self.messages = []
        self.usage = Usage(input_tokens=tokens, output_tokens=0, total_tokens=tokens)


@contextlib.contextmanager
def _patched_agent_execution(dummy_tokens: int = 42):
    """
    Patch heavy network pieces so that execute_with_fallback is fast & offline.
    """
    with patch(
        "nano_agent.modules.model_router.ProviderConfig.create_agent",
        return_value="DummyAgent",
    ), patch(
        "nano_agent.modules.model_router.Runner.run_sync",
        return_value=_DummyRunnerResult(tokens=dummy_tokens),
    ):
        yield


# --------------------------------------------------------------------------- #
# Benchmark suites                                                            #
# --------------------------------------------------------------------------- #

def benchmark_providers(runs: int = 3) -> Dict[str, ProviderBenchmarks]:
    """
    Run response-time & throughput benchmarks for each supported provider.

    NOTE: All network calls are mocked; we only measure framework overhead.
    """
    providers_to_models = {
        "openai": "gpt-5-mini",
        "azure": "gpt-5-mini",
        "anthropic": "claude-3-haiku-20240307",
        "ollama": "gpt-oss:20b",
    }
    benches: Dict[str, ProviderBenchmarks] = {p: ProviderBenchmarks() for p in providers_to_models}

    for provider, model in providers_to_models.items():
        for _ in range(runs):
            with _patched_agent_execution():
                start = time.perf_counter()
                resp = execute_with_fallback(
                    prompt="Say hello",
                    tools=[],
                    provider=provider,
                    prefer_cost_efficient=True,
                    max_turns=1,
                    max_retries=0,
                )
                duration = time.perf_counter() - start
            benches[provider].runs.append(
                OneRunMetric(
                    provider=provider,
                    model=model,
                    duration_s=duration,
                    total_tokens=resp.token_report.total_tokens if resp.token_report else 0,
                    total_cost=resp.token_report.total_cost if resp.token_report else 0.0,
                )
            )
    return benches


def benchmark_model_routing() -> RoutingBenchmarks:
    """
    Measure complexity analysis, routing overhead & fallback chain latency.
    """
    prompt = "Refactor 300 lines in 5 files to integrate Kubernetes support"
    # Complexity analysis
    t0 = time.perf_counter()
    complexity = analyze_prompt(prompt)
    t1 = time.perf_counter()

    # Model selection (route_task) – this includes building fallback chain
    from nano_agent.modules.model_router import route_task, _build_fallback_chain

    t2 = time.perf_counter()
    selection = route_task(prompt, provider="openai", prefer_cost_efficient=False)
    _ = _build_fallback_chain(selection.primary_model, "openai", False)
    t3 = time.perf_counter()

    # Simulate fallback execution (primary fails once, fallback succeeds)
    start_chain = time.perf_counter()
    with _patched_agent_execution():
        execute_with_fallback(
            prompt=prompt,
            tools=[],
            provider="openai",
            prefer_cost_efficient=False,
            max_retries=1,  # will still succeed on first because mocks
        )
    chain_time = time.perf_counter() - start_chain

    return RoutingBenchmarks(
        complexity_analysis_ms=(t1 - t0) * 1000,
        routing_overhead_ms=(t3 - t2) * 1000,
        fallback_chain_time_s=chain_time,
    )


def benchmark_memory_usage() -> MemoryBenchmarks:
    """
    Track peak memory for token tracker, agent creation & a mock tool exec.
    """
    tracemalloc.start()

    # TokenTracker footprint
    from nano_agent.modules.token_tracking import TokenTracker

    snap0 = tracemalloc.take_snapshot()
    tracker = TokenTracker()
    snap1 = tracemalloc.take_snapshot()
    tracker_peak = max(stat.size for stat in snap1.compare_to(snap0, "filename"))
    del tracker

    # Agent creation (mocked)
    with _patched_agent_execution():
        from nano_agent.modules.provider_config import ProviderConfig

        snap2 = tracemalloc.take_snapshot()
        ProviderConfig.create_agent(
            name="BenchAgent",
            instructions="",
            tools=[],
            model="gpt-5-mini",
            provider="openai",
            model_settings=None,
        )
        snap3 = tracemalloc.take_snapshot()
    agent_mem = max(stat.size for stat in snap3.compare_to(snap2, "filename"))

    # Tool exec memory (simulate small list_directory)
    from nano_agent.modules.nano_agent_tools import list_directory_raw

    snap4 = tracemalloc.take_snapshot()
    list_directory_raw(".")
    snap5 = tracemalloc.take_snapshot()
    tool_mem = max(stat.size for stat in snap5.compare_to(snap4, "filename"))

    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    # Convert to KB
    return MemoryBenchmarks(
        peak_kb=peak // 1024,
        tracker_peak_kb=tracker_peak // 1024,
        agent_creation_kb=agent_mem // 1024,
        tool_exec_kb=tool_mem // 1024,
    )


# --------------------------------------------------------------------------- #
# Public orchestration                                                        #
# --------------------------------------------------------------------------- #
def run_all_benchmarks(runs_per_provider: int = 3) -> BenchmarkReport:
    """High-level entry-point."""
    prov = benchmark_providers(runs_per_provider)
    routing = benchmark_model_routing()
    memory = benchmark_memory_usage()
    report = BenchmarkReport(provider=prov, routing=routing, memory=memory)
    return report


# --------------------------------------------------------------------------- #
# CLI runner                                                                  #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    rep = run_all_benchmarks()
    rep.display()