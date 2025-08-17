"""
Tests for Phase-4 Model Routing components:

1. task_analyzer – complexity heuristics & model recommendation
2. model_router  – routing / fallback logic & metrics tracking

No real network or OpenAI/Anthropic traffic is generated; all heavy
dependencies are stub-mocked.
"""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

# --------------------------------------------------------------------------- #
# task_analyzer tests                                                         #
# --------------------------------------------------------------------------- #
from nano_agent.modules.task_analyzer import (
    analyze_prompt,
    get_recommended_model,
)

def test_task_analyzer_simple_prompt():
    prompt = "Fix a typo in README"
    complexity = analyze_prompt(prompt)
    # Very low complexity
    assert complexity.score < 3
    # Should pick the simple tier model
    assert get_recommended_model(complexity, "openai") == "gpt-5-nano"

def test_task_analyzer_medium_prompt():
    prompt = "Add 300 lines of code across 5 files to implement user login"
    complexity = analyze_prompt(prompt)
    assert 3 <= complexity.score <= 7
    assert get_recommended_model(complexity, "openai") == "gpt-5-mini"

def test_task_analyzer_complex_prompt():
    prompt = (
        "Refactor the entire architecture and integrate with Kubernetes; "
        "update 1200 lines in 20 files"
    )
    complexity = analyze_prompt(prompt)
    assert complexity.score > 7
    assert get_recommended_model(complexity, "openai") == "gpt-5"

def test_loc_and_file_estimation():
    prompt = "Create 120 lines over 3 files"
    complexity = analyze_prompt(prompt)
    # Heuristics should capture at least the literal numbers
    assert complexity.code_volume_loc >= 120
    assert complexity.file_count >= 3

def test_keyword_indicator_extraction():
    prompt = "Improve performance and integration testing"
    complexity = analyze_prompt(prompt)
    indicators = complexity.complexity_indicators
    assert "performance" in indicators
    assert "integration" in indicators

# --------------------------------------------------------------------------- #
# model_router tests                                                          #
# --------------------------------------------------------------------------- #

from nano_agent.modules.model_router import (
    route_task,
    execute_with_fallback,
    get_routing_metrics,
    _build_fallback_chain,   # internal helper – safe for testing
)

# Dummy result object emulating Runner.run_sync return structure
class _DummyRunResult:
    def __init__(self, content: str = "ok", tokens: int = 10):
        from nano_agent.modules.token_tracking import Usage
        self.final_output = content
        # Minimal Usage object
        self.usage = Usage(
            input_tokens=tokens,
            output_tokens=0,
            total_tokens=tokens
        )
        self.messages = []  # Needed by some downstream code

# Patch ProviderConfig.create_agent to avoid heavy construction
@patch("nano_agent.modules.model_router.ProviderConfig.create_agent", return_value="DummyAgent")
def test_route_task_and_fallback_chain(mock_create_agent):
    sel = route_task("Quick edit task", provider="openai", prefer_cost_efficient=True)
    # Primary model for a quick task should be nano or mini
    assert sel.primary_model in ("gpt-5-nano", "gpt-5-mini")
    # Build explicit chain for inspection
    chain = _build_fallback_chain(sel.primary_model, "openai", True)
    # Cost-efficient chain is nano -> mini -> full
    assert chain[0] == sel.primary_model
    assert chain[-1] == "gpt-5"

# --------------------------------------------------------------------------- #
# execute_with_fallback success on first attempt                              #
# --------------------------------------------------------------------------- #
@patch("nano_agent.modules.model_router.ProviderConfig.create_agent", return_value="DummyAgent")
@patch("nano_agent.modules.model_router.Runner.run_sync", return_value=_DummyRunResult("all good"))
def test_execute_with_fallback_first_try_success(mock_run_sync, mock_create_agent):
    resp = execute_with_fallback(
        prompt="Say hello",
        tools=[],
        provider="openai",
        prefer_cost_efficient=False,
        max_turns=1,
    )
    assert resp.success is True
    assert resp.retries == 0
    assert resp.result == "all good"
    # Metrics should show at least one successful request
    metrics = get_routing_metrics()
    assert metrics.total_requests >= 1
    assert metrics.success_count >= 1

# --------------------------------------------------------------------------- #
# execute_with_fallback needs a retry                                         #
# --------------------------------------------------------------------------- #
@patch("nano_agent.modules.model_router.ProviderConfig.create_agent", return_value="DummyAgent")
def test_execute_with_fallback_retry_path(mock_create_agent):
    # First Runner.run_sync call raises, second succeeds
    with patch(
        "nano_agent.modules.model_router.Runner.run_sync",
        side_effect=[Exception("boom"), _DummyRunResult("recovered")]
    ):
        resp = execute_with_fallback(
            prompt="Complex refactor task",
            tools=[],
            provider="openai",
            prefer_cost_efficient=False,
            max_retries=2,
            max_turns=1,
        )
    assert resp.success is True
    assert resp.retries == 1
    assert resp.result == "recovered"

# --------------------------------------------------------------------------- #
# cost-efficient vs quality fallback ordering                                 #
# --------------------------------------------------------------------------- #
def test_fallback_order_cost_vs_quality():
    cheap = _build_fallback_chain("gpt-5-mini", "openai", prefer_cost_efficient=True)
    quality = _build_fallback_chain("gpt-5-mini", "openai", prefer_cost_efficient=False)
    # Second item differs (nano vs full model)
    assert cheap[1] == "gpt-5-nano"
    assert quality[1] == "gpt-5"

# --------------------------------------------------------------------------- #
# Ensure routing metrics aggregate correctly                                   #
# --------------------------------------------------------------------------- #
def test_routing_metrics_aggregation():
    metrics = get_routing_metrics().to_dict()
    # After previous tests there should be at least one request recorded
    assert metrics["total_requests"] >= 1
    # success + failure counts add up
    assert metrics["success"] + metrics["failure"] == metrics["total_requests"]