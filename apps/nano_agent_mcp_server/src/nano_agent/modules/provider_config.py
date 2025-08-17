"""
Provider Configuration for Multi-Model Support.

This module provides a thin abstraction layer for creating agents
with different model providers (OpenAI, Anthropic, Ollama).
"""

# Apply OpenAI compatibility patches BEFORE any OpenAI imports
from . import openai_compat

from typing import Optional, Union
import os
import logging
from openai import AsyncOpenAI
from agents import Agent, OpenAIChatCompletionsModel, ModelSettings, set_tracing_disabled
import requests

# LiteLLM + tool translation
from .litellm_adapter import create_litellm_adapter, LITELLM_AVAILABLE
from .tool_translator import convert_tools_for_provider

# Apply typing fixes for Python 3.12+ compatibility
from . import typing_fix

logger = logging.getLogger(__name__)


class ProviderConfig:
    """Configuration for different model providers."""
    
    @staticmethod
    def get_model_settings(model: str, provider: str, base_settings: dict) -> ModelSettings:
        """Get appropriate model settings for a given model and provider.
        
        Args:
            model: Model identifier
            provider: Provider name
            base_settings: Base settings dictionary with temperature, max_tokens, etc.
            
        Returns:
            ModelSettings configured appropriately for the model
        """
        # Filter settings based on model capabilities
        filtered_settings = {}
        
        # GPT-5 models have special requirements
        if model.startswith("gpt-5"):
            logger.debug(f"Configuring GPT-5 model {model} - using max_completion_tokens")
            # GPT-5 uses max_completion_tokens instead of max_tokens
            if "max_tokens" in base_settings:
                filtered_settings["max_completion_tokens"] = base_settings["max_tokens"]
            # GPT-5 models only support temperature=1 (default)
            # Don't include temperature in settings
        else:
            # Other models support all settings
            filtered_settings = base_settings.copy()
        
        # Anthropic models use the same parameters via OpenAI-compatible endpoint
        if provider == "anthropic":
            pass
        
        logger.debug(f"Model settings for {model}: {filtered_settings}")
        return ModelSettings(**filtered_settings)
    
    @staticmethod
    def create_agent(
        name: str,
        instructions: str,
        tools: list,
        model: str,
        provider: str,
        model_settings: Optional[ModelSettings] = None
    ) -> Agent:
        """Create an agent with the appropriate provider configuration.
        
        Args:
            name: Agent name
            instructions: System instructions for the agent
            tools: List of tool functions
            model: Model identifier
            provider: Provider name ('openai', 'anthropic', 'ollama')
            model_settings: Optional model settings
            
        Returns:
            Configured Agent instance
            
        Raises:
            ValueError: If provider is not supported
        """
        
        if provider == "openai":
            # Default OpenAI configuration
            logger.debug(f"Creating OpenAI agent with model: {model}")
            return Agent(
                name=name,
                instructions=instructions,
                tools=tools,
                model=model,
                model_settings=model_settings
            )
        
        elif provider == "anthropic":
            """
            Anthropic integration now uses LiteLLM for proper Claude support.
            We attempt to build an OpenAI-compatible wrapper around a LiteLLM
            adapter so the rest of the Agent SDK can interact with it
            transparently.  If LiteLLM is unavailable we gracefully fall back
            to the previous (often-broken) AsyncOpenAI endpoint strategy.
            """
            logger.debug(f"Creating Anthropic agent with model: {model} via LiteLLM")

            if LITELLM_AVAILABLE:
                try:
                    # ------------------------------------------------------------------ #
                    # 1. Instantiate LiteLLM adapter
                    # ------------------------------------------------------------------ #
                    adapter = create_litellm_adapter(
                        provider="anthropic",
                        model=model,
                        api_key=os.getenv("ANTHROPIC_API_KEY")
                    )

                    # ------------------------------------------------------------------ #
                    # 2. Convert tools to Anthropic format
                    # ------------------------------------------------------------------ #
                    translated_tools = convert_tools_for_provider(tools, "anthropic")

                    # ------------------------------------------------------------------ #
                    # 3. Build a minimal OpenAI-compatible wrapper
                    # ------------------------------------------------------------------ #
                    class _LiteLLMChatCompletions:
                        """Expose create / acreate the way OpenAI SDK expects."""
                        def __init__(self, _adapter):
                            self._adapter = _adapter

                        def create(self, model: str, messages: list, **kwargs):
                            # LiteLLM handles the model string itself; we pass messages onwards
                            return self._adapter.completion(messages=messages, **kwargs)

                        async def acreate(self, model: str, messages: list, **kwargs):
                            return await self._adapter.acompletion(messages=messages, **kwargs)

                    class _LiteLLMClient:
                        """Mimic the minimal subset of the OpenAI client used by Agent SDK."""
                        def __init__(self, _adapter):
                            self.chat = type("obj", (), {})()  # empty simple namespace object
                            self.chat.completions = _LiteLLMChatCompletions(_adapter)

                    lite_client = _LiteLLMClient(adapter)

                    # ------------------------------------------------------------------ #
                    # 4. Return fully configured Agent
                    # ------------------------------------------------------------------ #
                    return Agent(
                        name=name,
                        instructions=instructions,
                        tools=translated_tools,
                        model=OpenAIChatCompletionsModel(
                            model=model,
                            openai_client=lite_client  # our wrapper
                        ),
                        model_settings=model_settings
                    )

                except Exception as e:
                    logger.warning(
                        "LiteLLM integration for Anthropic failed (%s). "
                        "Falling back to AsyncOpenAI path.", e, exc_info=True
                    )
                    # Intentional fall-through to fallback

            # ------------------------- Fallback Path ------------------------------ #
            logger.debug("Falling back to AsyncOpenAI Anthropic endpoint (may be limited).")
            anthropic_client = AsyncOpenAI(
                base_url="https://api.anthropic.com/v1/",
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )
            return Agent(
                name=name,
                instructions=instructions,
                tools=tools,
                model=OpenAIChatCompletionsModel(
                    model=model,
                    openai_client=anthropic_client
                ),
                model_settings=model_settings
            )
        
        elif provider == "ollama":
            # Use OpenAI-compatible endpoint for Ollama
            logger.debug(f"Creating Ollama agent with model: {model}")
            ollama_client = AsyncOpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama"  # Dummy key required by client
            )
            return Agent(
                name=name,
                instructions=instructions,
                tools=tools,
                model=OpenAIChatCompletionsModel(
                    model=model,
                    openai_client=ollama_client
                ),
                model_settings=model_settings
            )
        
        elif provider == "azure":
            # Use OpenAI SDK with Azure OpenAI endpoint
            logger.debug(f"Creating Azure OpenAI agent with model: {model}")
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            if not azure_endpoint:
                raise ValueError("AZURE_OPENAI_ENDPOINT environment variable is required for Azure provider")
            
            # Azure OpenAI uses a different URL structure
            # Format: https://YOUR_RESOURCE_NAME.openai.azure.com/
            from openai import AsyncAzureOpenAI
            azure_client = AsyncAzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
                azure_deployment=model  # The deployment name in Azure
            )
            return Agent(
                name=name,
                instructions=instructions,
                tools=tools,
                model=OpenAIChatCompletionsModel(
                    model=model,  # This will be the deployment name
                    openai_client=azure_client
                ),
                model_settings=model_settings
            )
        
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    @staticmethod
    def setup_provider(provider: str) -> None:
        """Setup provider-specific configurations.
        
        Args:
            provider: Provider name
        """
        if provider != "openai":
            # Disable tracing for non-OpenAI providers by default
            # unless an OpenAI key is available for tracing
            if not os.getenv("OPENAI_API_KEY"):
                logger.info(f"Disabling tracing for {provider} provider (no OpenAI API key for tracing)")
                set_tracing_disabled(True)
            else:
                logger.debug(f"Tracing enabled for {provider} provider using OpenAI API key")
    
    @staticmethod
    def validate_provider_setup(provider: str, model: str, available_models: dict, provider_requirements: dict) -> tuple[bool, Optional[str]]:
        """Validate that provider is properly configured.
        
        Args:
            provider: Provider name
            model: Model identifier
            available_models: Dictionary of available models per provider
            provider_requirements: Dictionary of API key requirements
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        
        # Check model availability
        if provider not in available_models:
            return False, f"Unknown provider: {provider}"
        
        if model not in available_models[provider]:
            return False, f"Model {model} not available for {provider}. Available models: {', '.join(available_models[provider])}"
        
        # Check API keys
        required_key = provider_requirements.get(provider)
        if required_key and not os.getenv(required_key):
            return False, f"Missing environment variable: {required_key}"
        
        # Check Azure-specific requirements
        if provider == "azure":
            if not os.getenv("AZURE_OPENAI_ENDPOINT"):
                return False, "Missing environment variable: AZURE_OPENAI_ENDPOINT"
            # Model parameter for Azure is actually the deployment name
            logger.debug(f"Azure provider validated with deployment: {model}")
        
        # Check Ollama availability
        if provider == "ollama":
            try:
                response = requests.get("http://localhost:11434/api/tags", timeout=1)
                models = [m["name"] for m in response.json().get("models", [])]
                if model not in models:
                    return False, f"Model {model} not pulled in Ollama. Run: ollama pull {model}"
            except requests.ConnectionError:
                return False, "Ollama service not running. Start with: ollama serve"
            except requests.Timeout:
                return False, "Ollama service timeout. Check if service is running: ollama serve"
            except Exception as e:
                return False, f"Error checking Ollama availability: {str(e)}"
        
        return True, None