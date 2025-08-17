"""
Model Router – Dynamic Model Selection & Fallback Logic
=======================================================

This module forms Phase-4 of the multi-provider roadmap.  It analyses a task
prompt, chooses an appropriate model tier (simple / medium / complex), and
executes the task through the existing ProviderConfig + OpenAI Agent-SDK
pipeline.  If the primary model call fails it can transparently fall back to
one or more cheaper / more available alternatives while still exposing a
single coherent response.

Public API
----------
route_task(prompt: str, provider: str = "openai", prefer_cost_efficient: bool = False) -> ModelSelection
    Analyse the prompt and return the recommended model list.

execute_with_fallback(prompt: str,
                       tools: list,
                       provider: str = "openai",
                       prefer_cost_efficient: bool = False,
                       max_turns: int | None = None,
                       max_retries: int = 2) -> AgentResponse
    Execute using the first recommended model, falling back if necessary.

get_routing_metrics() -> RoutingMetrics
    Retrieve aggregated runtime metrics for observability / optimisation.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agents import Runner, RunConfig
from agents.lifecycle import RunHooksBase

# Internal modules
from .task_analyzer import (
    analyze_prompt,
    get_recommended_model,
    TaskComplexity,
)
from .provider_config import ProviderConfig
from .token_tracking import TokenTracker, TokenUsageReport

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Dataclasses                                                                 #
# --------------------------------------------------------------------------- #

@dataclass
class ModelSelection:
    """Information about the chosen model and fallbacks."""
    provider: str
    complexity: TaskComplexity
    primary_model: str
    fallbacks: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "primary_model": self.primary_model,
            "fallbacks": self.fallbacks,
            "complexity": self.complexity.to_dict(),
        }


@dataclass
class RoutingMetrics:
    """Aggregated runtime metrics useful for optimisation dashboards."""
    total_requests: int = 0
    success_count: int = 0
    failure_count: int = 0
    per_model_stats: Dict[str, Dict[str, int]] = field(default_factory=dict)  # {model: {"success": x, "failure": y}}
    total_tokens: int = 0
    total_cost: float = 0.0

    def record(self, model: str, success: bool, tracker: Optional[TokenTracker] = None) -> None:
        self.total_requests += 1
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1

        stats = self.per_model_stats.setdefault(model, {"success": 0, "failure": 0})
        key = "success" if success else "failure"
        stats[key] += 1

        if tracker:
            self.total_tokens += tracker.total_usage.total_tokens
            self.total_cost += tracker.calculate_cost()[3]  # total cost

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "success": self.success_count,
            "failure": self.failure_count,
            "per_model": self.per_model_stats,
            "total_tokens": self.total_tokens,
            "total_cost": round(self.total_cost, 4),
        }


@dataclass
class AgentResponse:
    """Unified wrapper around the Agent SDK response with extra metadata."""
    success: bool
    result: Optional[str] = None
    error: Optional[str] = None
    model_used: Optional[str] = None
    provider: Optional[str] = None
    retries: int = 0
    token_report: Optional[TokenUsageReport] = None
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "model_used": self.model_used,
            "provider": self.provider,
            "retries": self.retries,
            "duration_seconds": round(self.duration_seconds, 2),
            "token_report": self.token_report.to_dict() if self.token_report else None,
        }


# --------------------------------------------------------------------------- #
# Global Metrics Singleton                                                    #
# --------------------------------------------------------------------------- #

_ROUTING_METRICS = RoutingMetrics()


def get_routing_metrics() -> RoutingMetrics:
    """Return live routing metrics (read-only)."""
    return _ROUTING_METRICS


# --------------------------------------------------------------------------- #
# Core Routing Logic                                                          #
# --------------------------------------------------------------------------- #

def _build_fallback_chain(primary: str, provider: str, prefer_cost_efficient: bool) -> List[str]:
    """
    Very simple fallback strategy:

    • Start with the primary recommended model.
    • If prefer_cost_efficient=True we order fallbacks cheapest→expensive.
      Otherwise expensive→cheapest (to maximise quality).

    For now we hard-code known model families.  This can be replaced by a
    pricing-aware sort later.
    """
    if provider.lower() in ("openai", "azure"):
        chain = ["gpt-5-nano", "gpt-5-mini", "gpt-5"]
        if not prefer_cost_efficient:
            chain = list(reversed(chain))
    elif provider.lower() == "anthropic":
        chain = [
            "claude-3-haiku-20240307",
            "claude-sonnet-4-20250514",
            "claude-opus-4-20250514",
        ]
        if not prefer_cost_efficient:
            chain = list(reversed(chain))
    else:
        chain = [primary]

    # Ensure primary is first then unique fallbacks
    chain = [primary] + [m for m in chain if m != primary]
    return chain


def route_task(
    prompt: str,
    provider: str = "openai",
    prefer_cost_efficient: bool = False,
) -> ModelSelection:
    """
    Analyse the prompt and return a `ModelSelection` with primary + fallbacks.
    """
    complexity = analyze_prompt(prompt)
    primary_model = get_recommended_model(complexity, provider)
    fallbacks = _build_fallback_chain(primary_model, provider, prefer_cost_efficient)[1:]

    selection = ModelSelection(
        provider=provider,
        complexity=complexity,
        primary_model=primary_model,
        fallbacks=fallbacks,
    )
    logger.debug("Model routing decision: %s", selection.to_dict())
    return selection


# --------------------------------------------------------------------------- #
# Execution with Fallback                                                     #
# --------------------------------------------------------------------------- #

class _SilentHooks(RunHooksBase):
    """Hooks that swallow tool output (useful when router is called programmatically)."""
    pass


def _execute_agent_once(
    prompt: str,
    tools: list,
    model: str,
    provider: str,
    max_turns: int | None,
) -> tuple[bool, str, TokenTracker]:
    """
    Helper that creates an Agent via ProviderConfig and executes synchronously.
    """
    from .nano_agent import NANO_AGENT_SYSTEM_PROMPT  # local import to avoid circular
    from .constants import DEFAULT_TEMPERATURE, MAX_TOKENS

    token_tracker = TokenTracker(model=model, provider=provider)
    hooks = _SilentHooks(token_tracker=token_tracker)

    # Build model settings
    model_settings = ProviderConfig.get_model_settings(
        model=model,
        provider=provider,
        base_settings={"temperature": DEFAULT_TEMPERATURE, "max_tokens": MAX_TOKENS},
    )
    # Create agent
    agent = ProviderConfig.create_agent(
        name="RouterExecutionAgent",
        instructions=NANO_AGENT_SYSTEM_PROMPT,
        tools=tools,
        model=model,
        provider=provider,
        model_settings=model_settings,
    )

    try:
        result = Runner.run_sync(
            agent,
            prompt,
            max_turns=max_turns or 3,
            run_config=RunConfig(workflow_name="router_execution"),
            hooks=hooks,
        )
        final_output = result.final_output if hasattr(result, "final_output") else str(result)
        # Update tracker with final usage if provided
        if hasattr(result, "usage"):
            token_tracker.update(result.usage)
        return True, final_output, token_tracker
    except Exception as exc:  # noqa: BLE001
        logger.warning("Model %s/%s failed: %s", provider, model, exc, exc_info=True)
        return False, str(exc), token_tracker


def execute_with_fallback(
    prompt: str,
    tools: list,
    provider: str = "openai",
    prefer_cost_efficient: bool = False,
    max_turns: int | None = None,
    max_retries: int = 2,
) -> AgentResponse:
    """
    Execute a task via the selected model; retry with fallbacks if needed.

    Returns an `AgentResponse` containing the first successful result or the
    last encountered error.
    """
    selection = route_task(prompt, provider, prefer_cost_efficient)
    models_to_try = [selection.primary_model] + selection.fallbacks

    start_ts = time.time()
    retries = 0
    last_error: Optional[str] = None
    tracker_for_return: Optional[TokenTracker] = None

    for model in models_to_try:
        success, result_or_err, tracker = _execute_agent_once(
            prompt=prompt,
            tools=tools,
            model=model,
            provider=provider,
            max_turns=max_turns,
        )
        tracker_for_return = tracker
        _ROUTING_METRICS.record(model, success, tracker)

        if success:
            return AgentResponse(
                success=True,
                result=result_or_err,
                model_used=model,
                provider=provider,
                retries=retries,
                token_report=tracker.generate_report() if tracker else None,
                duration_seconds=time.time() - start_ts,
            )

        # Failure pathway
        retries += 1
        last_error = result_or_err
        if retries > max_retries:
            break

    # All attempts failed
    return AgentResponse(
        success=False,
        error=last_error or "Unknown failure",
        provider=provider,
        retries=retries,
        model_used=models_to_try[min(retries, len(models_to_try)-1)],
        token_report=tracker_for_return.generate_report() if tracker_for_return else None,
        duration_seconds=time.time() - start_ts,
    )


# --------------------------------------------------------------------------- #
# __all__                                                                     #
# --------------------------------------------------------------------------- #

__all__ = [
    "ModelSelection",
    "RoutingMetrics",
    "AgentResponse",
    "route_task",
    "execute_with_fallback",
    "get_routing_metrics",
]