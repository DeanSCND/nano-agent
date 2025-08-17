"""
Tool Translator Module for Multi-Provider Support
================================================

Purpose
-------
OpenAI's function-calling / tool specification JSON is becoming a de-facto
standard, but a few providers (Anthropic, Google, etc.) still expect slightly
different shapes or wrapper keys.  This module centralises translation logic
so the rest of the nano-agent codebase can continue working with the canonical
OpenAI format regardless of which provider is ultimately executing the call.

Design
------
• Public helpers:
    - convert_tools_for_provider(tools, provider)
    - convert_response_to_openai(response, provider)

  Both perform **deep-copies** so callers can pass in shared objects without
  worrying about mutation.

• Internals:
    - _TO_TOOL_CONVERTERS : dict[str, Callable[[list[dict]], list[dict]]]
    - _FROM_RESULT_CONVERTERS : dict[str, Callable[[dict], dict]]

  Adding support for a new provider is as easy as registering two lambdas or
  functions.

• Currently implemented providers:
    - openai      (identity / no-op)
    - anthropic   (Anthropic’s Claude "tools" schema, July-2025 preview)
      see https://docs.anthropic.com/claude/docs/tools for format details.

The anthropic mapping in practice
---------------------------------
OpenAI:
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file",
            "parameters": {...JSONSchema...}
        }
    }

Anthropic (as of 2025-07):
    {
        "name": "read_file",
        "description": "Read the contents of a file",
        "input_schema": {...JSONSchema...}
    }

i.e. `type=function` wrapper is removed, and `parameters` key becomes
`input_schema`.

Tool-call *responses* also differ: Anthropic returns
`{"function_name": "...", "arguments": {...}}`.
We normalise that back into OpenAI’s
`{"name": "...", "arguments": JSON-STRING}` structure.

This keeps the higher-level nano-agent logic provider-agnostic.

"""

from __future__ import annotations

import copy
import logging
from typing import Callable, Dict, List

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Helper conversion functions                                                 #
# --------------------------------------------------------------------------- #

# ---------- Anthropic ------------------------------------------------------ #
def _openai_to_anthropic(tools: List[dict]) -> List[dict]:
    """
    Convert a list of OpenAI-formatted tools to Anthropic's format.

    OpenAI (input):
        [{
            "type": "function",
            "function": {
                "name": "...",
                "description": "...",
                "parameters": { ... }
            }
        }]

    Anthropic (output):
        [{
            "name": "...",
            "description": "...",
            "input_schema": { ... }
        }]
    """
    converted: List[dict] = []
    for tool in tools:
        if tool.get("type") != "function" or "function" not in tool:
            logger.debug("Tool already appears to be Anthropic-style or unrecognised "
                         "– passing through unchanged: %s", tool)
            converted.append(copy.deepcopy(tool))
            continue

        fn = tool["function"]
        anthropic_tool = {
            "name": fn.get("name"),
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters", {}),
        }
        converted.append(anthropic_tool)
    return converted


def _anthropic_to_openai_response(response: dict) -> dict:
    """
    Convert an Anthropic tool-call *response* back to OpenAI-style.

    Anthropic sample:
        {
            "name": "read_file",
            "arguments": {...dict...}
        }

    OpenAI expected by nano-agent:
        {
            "function": {
                "name": "read_file",
                "arguments": "{\"path\":\"...\"}"   # (!!) note JSON string
            },
            "type": "function"
        }
    """
    if "name" not in response:
        logger.debug("Anthropic response missing 'name' – returning unchanged: %s", response)
        return response

    # arguments must be JSON-encoded string for OpenAI format
    import json

    openai_response = {
        "type": "function",
        "function": {
            "name": response["name"],
            "arguments": json.dumps(response.get("arguments", {})),
        }
    }
    return openai_response


# ---------- Identity helpers (OpenAI / default) ---------------------------- #
def _identity_tools(tools: List[dict]) -> List[dict]:
    """Return a deep-copy of tools list (no conversion)."""
    return copy.deepcopy(tools)


def _identity_response(response: dict) -> dict:
    """Return a deep-copy of response dict (no conversion)."""
    return copy.deepcopy(response)


# --------------------------------------------------------------------------- #
# Registry dictionaries                                                       #
# --------------------------------------------------------------------------- #

_TO_TOOL_CONVERTERS: Dict[str, Callable[[List[dict]], List[dict]]] = {
    "openai": _identity_tools,
    "anthropic": _openai_to_anthropic,
    "azure": _identity_tools,     # Azure is OpenAI-compatible
    "ollama": _identity_tools,    # Uses OpenAI schema via compat layer
    # Add more providers here ...
}

_FROM_RESULT_CONVERTERS: Dict[str, Callable[[dict], dict]] = {
    "openai": _identity_response,
    "anthropic": _anthropic_to_openai_response,
    "azure": _identity_response,
    "ollama": _identity_response,
    # Add more providers here ...
}


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #
def convert_tools_for_provider(tools: List[dict], provider: str) -> List[dict]:
    """
    Convert an OpenAI-formatted tools array into the provider-specific
    representation expected by LiteLLM / underlying API.

    If the provider is unknown, the function returns a deep-copy of the original
    list and logs a warning.
    """
    provider = provider.lower()
    converter = _TO_TOOL_CONVERTERS.get(provider)
    if converter is None:
        logger.warning("No tool converter registered for provider '%s' – "
                       "passing through tools unchanged", provider)
        return copy.deepcopy(tools)
    try:
        return converter(tools)
    except Exception as e:
        logger.error("Error converting tools for provider '%s': %s", provider, e, exc_info=True)
        # Fallback: return original (deep-copied)
        return copy.deepcopy(tools)


def convert_response_to_openai(response: dict, provider: str) -> dict:
    """
    Convert a provider-specific tool-call response back into canonical OpenAI
    format so the rest of the agent pipeline can consume it transparently.

    For unknown providers the response is shallow-copied unchanged.
    """
    provider = provider.lower()
    converter = _FROM_RESULT_CONVERTERS.get(provider)
    if converter is None:
        logger.warning("No response converter registered for provider '%s' – "
                       "passing through response unchanged", provider)
        return copy.deepcopy(response)
    try:
        return converter(response)
    except Exception as e:
        logger.error("Error converting response from provider '%s': %s", provider, e, exc_info=True)
        return copy.deepcopy(response)


# --------------------------------------------------------------------------- #
# Convenience registration decorator for new providers                        #
# --------------------------------------------------------------------------- #
def register_provider(
    provider: str,
    to_tool_fn: Callable[[List[dict]], List[dict]],
    from_response_fn: Callable[[dict], dict],
) -> None:
    """
    Dynamically register conversion helpers for a new provider.

    Example:
        @register_provider("my_provider", to_my_fmt, from_my_fmt)
        def _(): pass
    """
    provider = provider.lower()
    if provider in _TO_TOOL_CONVERTERS or provider in _FROM_RESULT_CONVERTERS:
        logger.warning("Overwriting existing converters for provider '%s'", provider)
    _TO_TOOL_CONVERTERS[provider] = to_tool_fn
    _FROM_RESULT_CONVERTERS[provider] = from_response_fn
    logger.debug("Registered custom tool/response converters for provider '%s'", provider)


# --------------------------------------------------------------------------- #
# __all__                                                                     #
# --------------------------------------------------------------------------- #
__all__ = [
    "convert_tools_for_provider",
    "convert_response_to_openai",
    "register_provider",
]