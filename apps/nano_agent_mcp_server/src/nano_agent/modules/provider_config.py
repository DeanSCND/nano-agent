"""
Provider Configuration for Multi-Model Support.

This module provides a thin abstraction layer for creating agents
with different model providers (OpenAI, Anthropic, Ollama).
"""

from typing import Optional, Union
import os
import logging
from openai import AsyncOpenAI
from agents import Agent, OpenAIChatCompletionsModel, ModelSettings, set_tracing_disabled
import requests

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
        
        # GPT-5 and o3 models have special requirements
        if model.startswith("gpt-5") or model.startswith("o3") or model.startswith("o1"):
            logger.debug(f"Configuring {model} - using max_completion_tokens via extra_args")
            # These models use max_completion_tokens instead of max_tokens
            # Pass it through extra_args which gets unpacked as **kwargs
            if "max_tokens" in base_settings:
                filtered_settings["extra_args"] = {"max_completion_tokens": base_settings["max_tokens"]}
            # Explicitly set max_tokens to None to prevent it from being passed
            filtered_settings["max_tokens"] = None
            # These models only support temperature=1 (default)
            # Don't include temperature in settings
            filtered_settings["temperature"] = None
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
            # Use OpenAI SDK with Anthropic's OpenAI-compatible endpoint
            logger.debug(f"Creating Anthropic agent with model: {model}")
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
            
            # For o3/o1 models, use latest API version
            api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
            if model.startswith("o3") or model.startswith("o1"):
                api_version = "2024-12-01-preview"  # Latest version for o3 models
                logger.debug(f"Using API version {api_version} for {model}")
            
            azure_client = AsyncAzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=api_version,
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