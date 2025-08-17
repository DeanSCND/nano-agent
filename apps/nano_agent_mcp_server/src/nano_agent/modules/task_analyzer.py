"""
Task Analyzer – Model-Routing Heuristics
=======================================

Phase-4 feature: decide which model tier should handle a given task
based on a lightweight static analysis of the user prompt.

Public API
----------
• analyze_prompt(prompt: str) -> TaskComplexity
• get_recommended_model(complexity: TaskComplexity, provider: str = "openai") -> str
• estimate_token_usage(complexity: TaskComplexity) -> TokenEstimate

The logic is intentionally *simple & explainable* but written to be
easily extended later (e.g. plugin registry, ML classifier, etc.).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import List

# --------------------------------------------------------------------------- #
# Data-classes                                                                #
# --------------------------------------------------------------------------- #


@dataclass
class TaskComplexity:
    """Structured representation of a task’s complexity assessment."""
    prompt: str
    code_volume_loc: int = 0
    file_count: int = 0
    complexity_indicators: List[str] = field(default_factory=list)
    score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "code_volume_loc": self.code_volume_loc,
            "file_count": self.file_count,
            "complexity_indicators": self.complexity_indicators,
            "score": self.score,
        }


@dataclass
class TokenEstimate:
    """Rough token usage estimation derived from complexity metrics."""
    max_input_tokens: int
    max_output_tokens: int
    estimated_total_tokens: int

    def to_dict(self) -> dict:
        return {
            "max_input_tokens": self.max_input_tokens,
            "max_output_tokens": self.max_output_tokens,
            "estimated_total_tokens": self.estimated_total_tokens,
        }


# --------------------------------------------------------------------------- #
# Heuristic parameters                                                        #
# --------------------------------------------------------------------------- #

# Keywords that indicate *high* complexity / architectural work
_COMPLEXITY_KEYWORDS = [
    "architecture",
    "architectural",
    "refactor",
    "integration",
    "multi-provider",
    "migration",
    "performance",
    "optimiz",
    "database",
    "docker",
    "kubernetes",
    "microservice",
]

# --------------------------------------------------------------------------- #
# Core helpers                                                                #
# --------------------------------------------------------------------------- #

_PROMPT_LOC_RE = re.compile(r"\b(\d{1,4})\s*(?:lines|loc)\b", re.IGNORECASE)
_PROMPT_FILE_RE = re.compile(r"\b(\d{1,3})\s*(?:files?)\b", re.IGNORECASE)


def _estimate_loc(prompt: str) -> int:
    """Extract LOC hints from prompt (e.g. 'add 200 lines of code')."""
    matches = _PROMPT_LOC_RE.findall(prompt)
    if matches:
        return max(int(m) for m in matches)
    # fallback: simple heuristic – every 5 words ~ 1 LOC requested
    return max(0, (len(prompt.split()) // 5))


def _estimate_file_count(prompt: str) -> int:
    """Estimate how many files are mentioned/required."""
    matches = _PROMPT_FILE_RE.findall(prompt)
    if matches:
        return max(int(m) for m in matches)

    # crude heuristic: count keyword occurrences
    file_keywords = ["file", "files", ".py", ".js", ".ts", ".md", ".html"]
    count = sum(prompt.lower().count(k) for k in file_keywords)
    return max(1, count) if count else 1


def _extract_complexity_indicators(prompt: str) -> List[str]:
    indicators = [kw for kw in _COMPLEXITY_KEYWORDS if kw in prompt.lower()]
    return indicators


def _score_complexity(loc: int, files: int, indicators: List[str]) -> float:
    """
    Combine metrics into a single complexity score.

    – code_volume_loc     : logarithmic contribution
    – file_count          : logarithmic contribution
    – indicators          : +2 per indicator hit
    """
    score = 0.0
    if loc:
        score += math.log1p(loc)  # diminishing returns
    if files:
        score += math.log1p(files)  # diminishing returns
    score += 2.0 * len(indicators)
    return round(score, 2)


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #

def analyze_prompt(prompt: str) -> TaskComplexity:
    """
    Analyse a natural-language task prompt and return a TaskComplexity object.
    """
    loc = _estimate_loc(prompt)
    files = _estimate_file_count(prompt)
    indicators = _extract_complexity_indicators(prompt)
    score = _score_complexity(loc, files, indicators)

    return TaskComplexity(
        prompt=prompt,
        code_volume_loc=loc,
        file_count=files,
        complexity_indicators=indicators,
        score=score,
    )


def get_recommended_model(complexity: TaskComplexity, provider: str = "openai") -> str:
    """
    Map a complexity score to the appropriate model tier for a given provider.

    Providers supported:
    – openai
    – anthropic
    – azure  (same mapping as openai but with Azure deployment names)
    """
    provider = provider.lower()

    if complexity.score < 3:
        tier = "simple"
    elif complexity.score <= 7:
        tier = "medium"
    else:
        tier = "complex"

    # Mapping table
    mapping = {
        "openai": {
            "simple": "gpt-5-nano",
            "medium": "gpt-5-mini",
            "complex": "gpt-5",
        },
        "azure": {  # use same IDs, assuming deployments mirror OpenAI IDs
            "simple": "gpt-5-nano",
            "medium": "gpt-5-mini",
            "complex": "gpt-5",
        },
        "anthropic": {
            "simple": "claude-3-haiku-20240307",
            "medium": "claude-sonnet-4-20250514",
            "complex": "claude-opus-4-20250514",
        },
    }

    return mapping.get(provider, mapping["openai"]).get(tier)


def estimate_token_usage(complexity: TaskComplexity) -> TokenEstimate:
    """
    Roughly guess token requirements from complexity metrics.
    Purely heuristic – useful for sizing `max_tokens` before the first call.
    """
    # Assume ~1.3 tokens per LOC and baseline of 200 input tokens
    input_tokens = 200 + int(complexity.code_volume_loc * 1.3)
    # Assume output roughly equals code_volume_loc * 2 (code + explanations)
    output_tokens = int(complexity.code_volume_loc * 2.0) + 100
    total = input_tokens + output_tokens

    # Cap to reasonable defaults for safety
    input_tokens = min(input_tokens, 16_000)
    output_tokens = min(output_tokens, 16_000)
    total = min(total, 32_000)

    return TokenEstimate(
        max_input_tokens=input_tokens,
        max_output_tokens=output_tokens,
        estimated_total_tokens=total,
    )


# --------------------------------------------------------------------------- #
# __all__                                                                     #
# --------------------------------------------------------------------------- #

__all__ = [
    "TaskComplexity",
    "TokenEstimate",
    "analyze_prompt",
    "get_recommended_model",
    "estimate_token_usage",
]