"""
LiteLLM Adapter for Multi-Provider Support.

This module provides a unified interface for multiple LLM providers through LiteLLM,
handling provider-specific configurations and format translations.
"""

import os
import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass

try:
    import litellm
    from litellm import completion, acompletion
    from litellm.exceptions import (
        APIError,
        AuthenticationError,
        RateLimitError,
        ServiceUnavailableError,
        Timeout
    )
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    litellm = None
    
from .constants import (
    ERROR_NO_API_KEY,
    ERROR_PROVIDER_NOT_SUPPORTED,
)

# Initialize logger
logger = logging.getLogger(__name__)


@dataclass
class LiteLLMConfig:
    """Configuration for LiteLLM provider."""
    
    provider: str
    model: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    api_version: Optional[str] = None
    extra_params: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize extra_params if not provided."""
        if self.extra_params is None:
            self.extra_params = {}


class LiteLLMAdapter:
    """Adapter for handling multiple providers through LiteLLM."""
    
    # Map provider names to LiteLLM model prefixes
    PROVIDER_PREFIXES = {
        "anthropic": "claude",
        "google": "gemini",
        "cohere": "command",
        "replicate": "replicate",
        "bedrock": "bedrock",
        "vertex_ai": "vertex_ai",
        "azure": "azure",
        "openai": "",  # OpenAI doesn't need a prefix
        "ollama": "ollama",
    }
    
    # Provider-specific environment variable mappings
    ENV_VAR_MAPPINGS = {
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
        "cohere": "COHERE_API_KEY",
        "replicate": "REPLICATE_API_KEY",
        "bedrock": ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION_NAME"],
        "vertex_ai": ["GOOGLE_APPLICATION_CREDENTIALS", "GOOGLE_PROJECT_ID"],
        "azure": ["AZURE_API_KEY", "AZURE_API_BASE", "AZURE_API_VERSION"],
        "openai": "OPENAI_API_KEY",
        "ollama": None,  # Ollama doesn't need an API key
    }
    
    def __init__(self, config: LiteLLMConfig):
        """Initialize the LiteLLM adapter.
        
        Args:
            config: LiteLLM configuration
        """
        if not LITELLM_AVAILABLE:
            raise ImportError("LiteLLM is not installed. Please install it with: pip install litellm")
        
        self.config = config
        self._setup_provider()
    
    def _setup_provider(self):
        """Setup provider-specific configurations."""
        # Set up environment variables if not already set
        self._setup_environment()
        
        # Configure LiteLLM settings
        if self.config.provider == "anthropic":
            # Enable Claude's prompt caching
            litellm.enable_cache = True
            litellm.cache_duration = 3600  # 1 hour cache
        
        # Set custom API base if provided
        if self.config.api_base:
            if self.config.provider == "ollama":
                os.environ["OLLAMA_API_BASE"] = self.config.api_base
            elif self.config.provider == "azure":
                os.environ["AZURE_API_BASE"] = self.config.api_base
        
        # Enable verbose logging for debugging
        if logger.level == logging.DEBUG:
            litellm.set_verbose = True
    
    def _setup_environment(self):
        """Setup environment variables for the provider."""
        env_vars = self.ENV_VAR_MAPPINGS.get(self.config.provider)
        
        if env_vars is None:
            # Provider doesn't need API keys (e.g., Ollama)
            return
        
        if isinstance(env_vars, str):
            # Single environment variable
            if self.config.api_key and not os.getenv(env_vars):
                os.environ[env_vars] = self.config.api_key
            elif not os.getenv(env_vars) and self.config.provider != "ollama":
                logger.warning(f"No API key found for {self.config.provider}. Set {env_vars} environment variable.")
        elif isinstance(env_vars, list):
            # Multiple environment variables (e.g., AWS, Azure)
            for var in env_vars:
                value = self.config.extra_params.get(var.lower())
                if value and not os.getenv(var):
                    os.environ[var] = value
    
    def get_model_string(self) -> str:
        """Get the properly formatted model string for LiteLLM.
        
        Returns:
            Formatted model string
        """
        prefix = self.PROVIDER_PREFIXES.get(self.config.provider, "")
        
        if self.config.provider == "azure":
            # Azure uses a special format
            return f"azure/{self.config.model}"
        elif self.config.provider == "bedrock":
            # Bedrock format
            return f"bedrock/{self.config.model}"
        elif self.config.provider == "vertex_ai":
            # Vertex AI format
            return f"vertex_ai/{self.config.model}"
        elif self.config.provider == "ollama":
            # Ollama format
            return f"ollama/{self.config.model}"
        elif prefix:
            # For providers that just need a prefix
            # Check if model already has the prefix
            if self.config.model.startswith(prefix):
                return self.config.model
            return f"{prefix}-{self.config.model}" if not self.config.model.startswith(f"{prefix}-") else self.config.model
        else:
            # OpenAI and others that don't need a prefix
            return self.config.model
    
    def completion(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Dict[str, Any]:
        """Synchronous completion using LiteLLM.
        
        Args:
            messages: List of message dictionaries
            **kwargs: Additional parameters for the completion
            
        Returns:
            Completion response
        """
        model_string = self.get_model_string()
        
        # Merge with extra params from config
        params = {**self.config.extra_params, **kwargs}
        
        try:
            response = completion(
                model=model_string,
                messages=messages,
                **params
            )
            return response
        except AuthenticationError as e:
            logger.error(f"Authentication failed for {self.config.provider}: {e}")
            raise
        except RateLimitError as e:
            logger.warning(f"Rate limit hit for {self.config.provider}: {e}")
            raise
        except ServiceUnavailableError as e:
            logger.error(f"Service unavailable for {self.config.provider}: {e}")
            raise
        except Timeout as e:
            logger.error(f"Request timed out for {self.config.provider}: {e}")
            raise
        except APIError as e:
            logger.error(f"API error for {self.config.provider}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in LiteLLM completion: {e}")
            raise
    
    async def acompletion(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Dict[str, Any]:
        """Asynchronous completion using LiteLLM.
        
        Args:
            messages: List of message dictionaries
            **kwargs: Additional parameters for the completion
            
        Returns:
            Completion response
        """
        model_string = self.get_model_string()
        
        # Merge with extra params from config
        params = {**self.config.extra_params, **kwargs}
        
        try:
            response = await acompletion(
                model=model_string,
                messages=messages,
                **params
            )
            return response
        except AuthenticationError as e:
            logger.error(f"Authentication failed for {self.config.provider}: {e}")
            raise
        except RateLimitError as e:
            logger.warning(f"Rate limit hit for {self.config.provider}: {e}")
            raise
        except ServiceUnavailableError as e:
            logger.error(f"Service unavailable for {self.config.provider}: {e}")
            raise
        except Timeout as e:
            logger.error(f"Request timed out for {self.config.provider}: {e}")
            raise
        except APIError as e:
            logger.error(f"API error for {self.config.provider}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in LiteLLM acompletion: {e}")
            raise
    
    @classmethod
    def validate_provider(cls, provider: str) -> bool:
        """Validate if a provider is supported.
        
        Args:
            provider: Provider name to validate
            
        Returns:
            True if provider is supported
        """
        return provider.lower() in cls.PROVIDER_PREFIXES
    
    @classmethod
    def get_required_env_vars(cls, provider: str) -> Union[str, List[str], None]:
        """Get required environment variables for a provider.
        
        Args:
            provider: Provider name
            
        Returns:
            Required environment variable name(s) or None
        """
        return cls.ENV_VAR_MAPPINGS.get(provider.lower())
    
    @staticmethod
    def convert_tool_to_litellm_format(tool: Dict[str, Any]) -> Dict[str, Any]:
        """Convert OpenAI tool format to LiteLLM format.
        
        Most providers use OpenAI's format, but this method can handle
        provider-specific conversions if needed.
        
        Args:
            tool: Tool in OpenAI format
            
        Returns:
            Tool in LiteLLM-compatible format
        """
        # Most providers use OpenAI format through LiteLLM
        # Add provider-specific conversions here if needed
        return tool
    
    @staticmethod
    def convert_response_to_openai_format(response: Dict[str, Any]) -> Dict[str, Any]:
        """Convert LiteLLM response to OpenAI format.
        
        Ensures consistent response format regardless of provider.
        
        Args:
            response: LiteLLM response
            
        Returns:
            Response in OpenAI format
        """
        # LiteLLM already normalizes responses to OpenAI format
        # Add any additional normalization here if needed
        return response


def create_litellm_adapter(
    provider: str,
    model: str,
    api_key: Optional[str] = None,
    **extra_params
) -> LiteLLMAdapter:
    """Factory function to create a LiteLLM adapter.
    
    Args:
        provider: Provider name
        model: Model identifier
        api_key: Optional API key
        **extra_params: Additional provider-specific parameters
        
    Returns:
        Configured LiteLLMAdapter instance
    """
    config = LiteLLMConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        extra_params=extra_params
    )
    
    return LiteLLMAdapter(config)


# Export key components
__all__ = [
    "LiteLLMAdapter",
    "LiteLLMConfig",
    "create_litellm_adapter",
    "LITELLM_AVAILABLE",
]