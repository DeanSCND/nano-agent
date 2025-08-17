"""
Tests for LiteLLM integration layer.

Coverage:
1. nano_agent.modules.litellm_adapter
2. nano_agent.modules.tool_translator
3. nano_agent.modules.provider_config (Anthropic branch)

The real LiteLLM package or remote APIs are NOT required – everything is
fully stub-mocked.
"""

from __future__ import annotations

import asyncio
import types
from typing import Any, Dict, List

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

# --------------------------------------------------------------------------- #
# Helper fixtures / stubs                                                     #
# --------------------------------------------------------------------------- #


@pytest.fixture
def dummy_litellm_module(monkeypatch):
    """
    Inject a fake 'litellm' module into sys.modules with the minimal surface
    used by our adapter.
    """
    fake = types.ModuleType("litellm")

    # Storage to assert call parameters
    fake.calls: List[Dict[str, Any]] = []

    def _fake_completion(model: str, messages: list, **kwargs):
        fake.calls.append(
            {"type": "sync", "model": model, "messages": messages, "kwargs": kwargs}
        )
        # Return OpenAI-style dummy response
        return {
            "choices": [
                {"message": {"content": "stub-response"}}
            ],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
        }

    async def _fake_acompletion(model: str, messages: list, **kwargs):
        fake.calls.append(
            {"type": "async", "model": model, "messages": messages, "kwargs": kwargs}
        )
        return {
            "choices": [
                {"message": {"content": "stub-async-response"}}
            ],
            "usage": {
                "prompt_tokens": 2,
                "completion_tokens": 2,
                "total_tokens": 4,
            },
        }

    # Attributes required by adapter
    fake.completion = _fake_completion
    fake.acompletion = _fake_acompletion
    fake.exceptions = types.SimpleNamespace(
        APIError=Exception,
        AuthenticationError=Exception,
        RateLimitError=Exception,
        ServiceUnavailableError=Exception,
        Timeout=Exception,
    )
    fake.enable_cache = False
    fake.cache_duration = 0
    fake.set_verbose = False

    # Register in sys.modules
    import sys

    sys.modules["litellm"] = fake
    yield fake
    # Clean up after the test session
    sys.modules.pop("litellm", None)


# --------------------------------------------------------------------------- #
# 1. Tests for litellm_adapter                                                #
# --------------------------------------------------------------------------- #

@pytest.mark.usefixtures("dummy_litellm_module")
class TestLiteLLMAdapter:
    """Validate core logic of LiteLLMAdapter without real network calls."""

    @pytest.fixture(autouse=True)
    def _import_adapter(self, dummy_litellm_module):  # noqa: D401
        # Re-import adapter so it picks up the dummy module & sets flag
        import importlib
        from nano_agent.modules import litellm_adapter as adapter

        importlib.reload(adapter)
        self.adapter_mod = adapter

    def test_model_string_formatting(self):
        """Provider prefixes & passthrough logic."""
        adapter = self.adapter_mod.create_litellm_adapter(
            provider="anthropic",
            model="claude-3-haiku-20240307",
            api_key="test",
        )
        assert adapter.get_model_string() == "claude-3-haiku-20240307"

        adapter_openai = self.adapter_mod.create_litellm_adapter(
            provider="openai",
            model="gpt-5-mini",
            api_key="test",
        )
        assert adapter_openai.get_model_string() == "gpt-5-mini"

        adapter_azure = self.adapter_mod.create_litellm_adapter(
            provider="azure",
            model="gpt-5-mini",
            api_key="test",
        )
        assert adapter_azure.get_model_string() == "azure/gpt-5-mini"

    def test_sync_completion_calls_dummy_litellm(self, dummy_litellm_module):
        adapter = self.adapter_mod.create_litellm_adapter(
            provider="anthropic",
            model="claude-3-haiku-20240307",
            api_key="test",
        )
        messages = [{"role": "user", "content": "Hello"}]
        resp = adapter.completion(messages)

        # Verify payload & response
        call = dummy_litellm_module.calls[-1]
        assert call["type"] == "sync"
        assert call["model"] == "claude-3-haiku-20240307"
        assert resp["choices"][0]["message"]["content"] == "stub-response"

    @pytest.mark.asyncio
    async def test_async_completion_calls_dummy_litellm(self, dummy_litellm_module):
        adapter = self.adapter_mod.create_litellm_adapter(
            provider="anthropic",
            model="claude-3-haiku-20240307",
            api_key="test",
        )
        messages = [{"role": "user", "content": "Hi"}]
        resp = await adapter.acompletion(messages)

        call = dummy_litellm_module.calls[-1]
        assert call["type"] == "async"
        assert call["model"] == "claude-3-haiku-20240307"
        assert resp["choices"][0]["message"]["content"] == "stub-async-response"


# --------------------------------------------------------------------------- #
# 2. Tests for tool_translator                                                #
# --------------------------------------------------------------------------- #

from nano_agent.modules.tool_translator import (
    convert_tools_for_provider,
    convert_response_to_openai,
)


class TestToolTranslator:
    """Validate OpenAI ↔ Anthropic tool/response conversions."""

    def _sample_openai_tools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read file contents",
                    "parameters": {
                        "type": "object",
                        "properties": {"file_path": {"type": "string"}},
                        "required": ["file_path"],
                    },
                },
            }
        ]

    def test_openai_to_anthropic_conversion(self):
        tools_openai = self._sample_openai_tools()
        tools_anthropic = convert_tools_for_provider(tools_openai, "anthropic")

        assert tools_anthropic[0]["name"] == "read_file"
        assert tools_anthropic[0]["description"] == "Read file contents"
        assert "input_schema" in tools_anthropic[0]
        assert tools_anthropic[0]["input_schema"]["properties"]["file_path"]["type"] == "string"

    def test_anthropic_response_to_openai(self):
        anthropic_resp = {
            "name": "read_file",
            "arguments": {"file_path": "demo.txt"},
        }
        openai_resp = convert_response_to_openai(anthropic_resp, "anthropic")
        assert openai_resp["type"] == "function"
        assert openai_resp["function"]["name"] == "read_file"
        # arguments should be JSON-encoded string
        import json

        assert json.loads(openai_resp["function"]["arguments"]) == {"file_path": "demo.txt"}


# --------------------------------------------------------------------------- #
# 3. provider_config integration                                              #
# --------------------------------------------------------------------------- #

class DummyAdapter:
    """Stub adapter returned by create_litellm_adapter inside ProviderConfig."""

    def __init__(self):
        self.calls = []

    def completion(self, messages, **kwargs):
        self.calls.append(("sync", messages))
        return {"choices": [{"message": {"content": "ok"}}]}

    async def acompletion(self, messages, **kwargs):
        self.calls.append(("async", messages))
        return {"choices": [{"message": {"content": "ok-async"}}]}


class TestProviderConfigAnthropicIntegration:
    """Ensure ProviderConfig routes Anthropic provider through LiteLLM."""

    @pytest.fixture(autouse=True)
    def _patch_provider_config(self, dummy_litellm_module):  # noqa: D401
        # Patch internals used by create_agent
        patches = {
            # Ensure LiteLLM path is selected
            "nano_agent.modules.provider_config.LITELLM_AVAILABLE": True,
            # Stub adapter factory
            "nano_agent.modules.provider_config.create_litellm_adapter": lambda *_, **__: DummyAdapter(),
            # Spy on tool translation
            "nano_agent.modules.provider_config.convert_tools_for_provider": MagicMock(
                side_effect=lambda tools, provider: tools
            ),
            # Replace heavy Agent / Model classes with simple mocks
            "nano_agent.modules.provider_config.Agent": MagicMock(return_value="DummyAgent"),
            "nano_agent.modules.provider_config.OpenAIChatCompletionsModel": MagicMock(
                return_value="DummyModel"
            ),
        }
        self.mocks = {}
        for path, new_val in patches.items():
            patcher = patch(path, new_val)
            self.mocks[path] = patcher.start()
        yield
        for patcher in self.mocks.values():
            patcher.stop()

    def test_create_agent_uses_litellm_path(self):
        from nano_agent.modules.provider_config import ProviderConfig

        agent = ProviderConfig.create_agent(
            name="TestAgent",
            instructions="Do things",
            tools=[],
            model="claude-3-haiku-20240307",
            provider="anthropic",
            model_settings=None,
        )

        # Should return stubbed agent
        assert agent == "DummyAgent"
        # convert_tools_for_provider must have been invoked
        self.mocks[
            "nano_agent.modules.provider_config.convert_tools_for_provider"
        ].assert_called_once_with([], "anthropic")

    def test_fallback_to_async_openai_when_adapter_fails(self):
        # Patch create_litellm_adapter to raise and ensure fallback path executes
        with patch(
            "nano_agent.modules.provider_config.create_litellm_adapter",
            side_effect=RuntimeError("boom"),
        ), patch(
            "nano_agent.modules.provider_config.AsyncOpenAI", MagicMock(return_value="AsyncClient")
        ), patch(
            "nano_agent.modules.provider_config.OpenAIChatCompletionsModel",
            MagicMock(return_value="DummyModel"),
        ), patch(
            "nano_agent.modules.provider_config.Agent",
            MagicMock(return_value="FallbackAgent"),
        ):
            from nano_agent.modules.provider_config import ProviderConfig

            agent = ProviderConfig.create_agent(
                name="TestAgent",
                instructions="Do things",
                tools=[],
                model="claude-3-haiku-20240307",
                provider="anthropic",
                model_settings=None,
            )

            # Should hit fallback path and still return an agent
            assert agent == "FallbackAgent"