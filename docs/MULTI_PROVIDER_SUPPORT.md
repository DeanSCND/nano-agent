# Multi-Provider Support for Nano-Agent

## Overview

This document describes the multi-provider LLM support implementation for nano-agent, enabling seamless integration with OpenAI, Anthropic, Azure OpenAI, Google Vertex AI, AWS Bedrock, and local Ollama models.

## Architecture

### Core Components

#### 1. LiteLLM Integration (`litellm_adapter.py`)
- Provides unified interface for multiple LLM providers
- Handles provider-specific authentication and configuration
- Manages API key validation and environment setup
- Supports automatic model string formatting

#### 2. Tool Format Translation (`tool_translator.py`)
- Converts between OpenAI and provider-specific tool formats
- Handles Anthropic's unique tool calling structure
- Preserves tool functionality across providers
- Provides fallback for unsupported providers

#### 3. Task Complexity Analysis (`task_analyzer.py`)
- Analyzes natural language prompts for complexity indicators
- Estimates lines of code from task descriptions
- Identifies architectural complexity markers
- Calculates normalized complexity scores (0-10 scale)

#### 4. Intelligent Model Routing (`model_router.py`)
- Routes tasks to appropriate models based on complexity
- Implements fallback chains for reliability
- Tracks routing metrics and performance
- Supports custom routing strategies

#### 5. Provider Configuration (`provider_config.py`)
- Creates provider-specific agent configurations
- Manages model settings and parameters
- Handles GPT-5 specific requirements
- Integrates LiteLLM for Anthropic support

#### 6. Token Tracking & Cost Calculation (`token_tracking.py`)
- Tracks token usage across all providers
- Calculates costs using current pricing models
- Supports cached and reasoning tokens
- Generates detailed usage reports

## Supported Providers

### OpenAI
- **Models**: gpt-5-mini, gpt-5-nano, gpt-5, gpt-4o, gpt-4o-mini
- **Authentication**: `OPENAI_API_KEY`
- **Features**: Full tool support, streaming, GPT-5 reasoning tokens

### Anthropic (via LiteLLM)
- **Models**: claude-3-haiku, claude-3-sonnet, claude-3-opus, claude-3.5-sonnet
- **Authentication**: `ANTHROPIC_API_KEY`
- **Features**: Tool calling via format translation, streaming

### Azure OpenAI
- **Models**: Custom deployments
- **Authentication**: `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`
- **Features**: Enterprise features, regional endpoints

### Google Vertex AI
- **Models**: gemini-pro, gemini-pro-vision
- **Authentication**: `GOOGLE_API_KEY` or service account
- **Features**: Multimodal support

### AWS Bedrock
- **Models**: Claude, Llama, and other Bedrock models
- **Authentication**: AWS credentials
- **Features**: Enterprise security, VPC endpoints

### Ollama (Local)
- **Models**: llama3, mistral, phi, custom models
- **Authentication**: None (local)
- **Features**: Privacy, no API costs, custom models

## Usage

### Basic Usage

```python
from nano_agent.modules import prompt_nano_agent

# Use OpenAI (default)
result = await prompt_nano_agent(
    agentic_prompt="Create a REST API with CRUD operations",
    model="gpt-5-mini",
    provider="openai"
)

# Use Anthropic
result = await prompt_nano_agent(
    agentic_prompt="Analyze this codebase and suggest improvements",
    model="claude-3-haiku-20240307",
    provider="anthropic"
)

# Use local Ollama
result = await prompt_nano_agent(
    agentic_prompt="Generate unit tests for the data module",
    model="llama3:8b",
    provider="ollama"
)
```

### With Task Routing

```python
from nano_agent.modules.model_router import execute_with_fallback

# Automatically routes to appropriate model based on complexity
response = await execute_with_fallback(
    prompt="Build a microservices architecture with 10 services",
    tools=[...],
    provider="openai"
)
# Complex task → Routes to gpt-5
```

### Configuration

#### Environment Variables

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# Azure OpenAI
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_API_VERSION="2024-08-01-preview"

# Google Vertex AI
export GOOGLE_API_KEY="..."
# Or use service account authentication

# AWS Bedrock
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_REGION="us-east-1"
```

#### MCP Configuration

```json
{
  "mcpServers": {
    "nano-agent": {
      "command": "node",
      "args": ["/path/to/nano-agent/mcp-nano-agent.js"],
      "env": {
        "OPENAI_API_KEY": "your-key",
        "ANTHROPIC_API_KEY": "your-key",
        "AZURE_OPENAI_API_KEY": "your-key",
        "AZURE_OPENAI_ENDPOINT": "your-endpoint"
      }
    }
  }
}
```

## Model Selection Strategy

### Complexity-Based Routing

| Complexity Score | Model Selection | Use Case |
|-----------------|-----------------|----------|
| 0-3 | gpt-5-nano, claude-3-haiku | Simple tasks, typo fixes, basic queries |
| 3-7 | gpt-5-mini, claude-3-sonnet | Standard features, bug fixes, refactoring |
| 7-10 | gpt-5, claude-3-opus | Complex architecture, system design |

### Fallback Chains

```
Primary Model → Secondary Model → Fallback Model
gpt-5 → gpt-4o → gpt-5-mini
claude-3-opus → claude-3-sonnet → claude-3-haiku
```

## Cost Optimization

### Token Pricing (per 1M tokens)

| Model | Input | Output | Cached Input |
|-------|-------|--------|--------------|
| gpt-5-mini | $0.30 | $1.20 | $0.15 |
| gpt-5 | $1.20 | $5.00 | $0.60 |
| claude-3-haiku | $0.25 | $1.25 | - |
| claude-3-sonnet | $3.00 | $15.00 | - |
| llama3 (Ollama) | Free | Free | Free |

### Cost Optimization Strategies

1. **Use complexity routing** - Automatically selects cheaper models for simple tasks
2. **Enable caching** - Reduces costs for repeated context
3. **Use local models** - Ollama for development and testing
4. **Batch operations** - Combine multiple small tasks
5. **Set token limits** - Configure max_tokens appropriately

## Known Issues and Limitations

### OpenAI SDK Compatibility
- The openai-agents library expects types that don't exist in OpenAI SDK v1.75.0
- Compatibility patches are provided in `openai_compat.py`
- Some import errors may occur with certain OpenAI SDK versions

### Provider-Specific Limitations

#### Anthropic
- Tool calling requires format translation
- No native support for OpenAI function format
- Maximum context window varies by model

#### Ollama
- Requires local installation and model downloads
- No streaming support in some configurations
- Performance depends on local hardware

#### Azure OpenAI
- Requires deployment names instead of model names
- Regional availability varies
- Rate limits depend on subscription

## Testing

### Unit Tests
```bash
# Test individual components
pytest tests/test_litellm_integration.py -v
pytest tests/test_model_routing.py -v
pytest tests/test_token_tracking.py -v
```

### Integration Tests
```bash
# Test multi-provider functionality
pytest tests/test_multi_provider_integration.py -v
```

### Manual Testing
```python
# Test each provider
python -m nano_agent.test_providers
```

## Performance Benchmarks

### Latency Comparison

| Provider | Model | First Token (ms) | Total Time (s) |
|----------|-------|------------------|----------------|
| OpenAI | gpt-5-mini | 250 | 2.5 |
| OpenAI | gpt-5 | 450 | 4.2 |
| Anthropic | claude-3-haiku | 180 | 1.8 |
| Ollama | llama3:8b | 50 | 3.0 |

### Throughput

- **OpenAI**: 100-150 requests/minute (with rate limiting)
- **Anthropic**: 50-100 requests/minute
- **Ollama**: Limited by local hardware (typically 5-20 requests/minute)

## Troubleshooting

### Common Issues

#### Import Errors
```
ImportError: cannot import name 'Variables' from 'openai.types.responses'
```
**Solution**: Ensure openai_compat.py is imported before any OpenAI imports

#### API Key Not Found
```
ValueError: Missing environment variable: ANTHROPIC_API_KEY
```
**Solution**: Set the appropriate environment variable or configure in MCP

#### Ollama Connection Error
```
ConnectionError: Ollama service not running
```
**Solution**: Start Ollama with `ollama serve`

#### Model Not Available
```
ValueError: Model claude-3-opus not available for anthropic
```
**Solution**: Check available models in constants.py

## Future Enhancements

1. **Additional Providers**
   - Cohere support
   - Hugging Face Inference API
   - Custom provider plugins

2. **Advanced Routing**
   - ML-based complexity prediction
   - Cost-aware routing
   - Latency-optimized routing

3. **Enhanced Monitoring**
   - Prometheus metrics export
   - Real-time cost tracking dashboard
   - Provider health monitoring

4. **Caching Layer**
   - Redis-based response caching
   - Semantic similarity matching
   - Automatic cache invalidation

## Contributing

To add a new provider:

1. Add provider configuration to `constants.py`
2. Implement adapter in `litellm_adapter.py`
3. Add tool translation if needed in `tool_translator.py`
4. Update provider configuration in `provider_config.py`
5. Add tests in `tests/test_<provider>_integration.py`
6. Update this documentation

## License

This implementation is part of the nano-agent project and follows the same license terms.