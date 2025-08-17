# Product Requirements Document: Multi-Provider Support for Nano-Agent

**Version:** 1.0  
**Date:** January 2025  
**Author:** EleventySeven Architecture Team  
**Status:** Draft

## Executive Summary

The nano-agent currently relies on OpenAI's Agent SDK, limiting it to OpenAI models despite attempts to support other providers. With Claude Sonnet 4.1 outperforming OpenAI's o3 on coding benchmarks (72.5% vs 69.1% on SWE-bench) while being 3-4x cheaper, and RepoPrompt being Mac-only, there's a critical need to enable true multi-provider support in nano-agent for Windows users and cost optimization.

## Problem Statement

### Current State
1. **Fake Anthropic Support**: The current implementation attempts to use Anthropic through an OpenAI-compatible endpoint, which doesn't work because:
   - Anthropic's API is not OpenAI-compatible
   - Tool calling formats are incompatible
   - Response structures differ significantly

2. **Performance Issues**: 
   - GPT-5 models take excessive time without visibility into progress
   - No real-time monitoring of agent execution
   - Black-box execution with no feedback until completion

3. **Platform Limitations**:
   - RepoPrompt only works on Mac, leaving Windows users dependent on nano-agent
   - No viable alternative for Claude models on Windows

4. **Cost Inefficiency**:
   - Forced to use expensive OpenAI models (o3: $10/$40 per 1M tokens)
   - Cannot leverage cheaper, better-performing alternatives (Claude Sonnet 4.1: $3/$15)

### Impact
- **Development Speed**: Slower iterations due to GPT-5 thinking time
- **Cost**: 3-4x higher costs than necessary
- **Quality**: Missing out on superior coding performance from Claude models
- **Developer Experience**: No visibility into long-running operations

## Requirements

### Functional Requirements

#### Must Have (P0)
1. **True Multi-Provider Support**
   - Native support for OpenAI models (GPT-5, o3, o3-mini)
   - Native support for Anthropic models (Claude Sonnet 4.1, Opus 4)
   - Proper tool calling translation between providers
   - Consistent response handling across providers

2. **Real-Time Monitoring**
   - Progress indicators during execution
   - Tool call visibility with arguments
   - Token usage tracking
   - Cost tracking per execution
   - Model/provider display

3. **Intelligent Model Routing**
   - Simple tasks → Fast models (GPT-5-nano, o3-mini, Claude Haiku)
   - Complex tasks → Powerful models (o3, Claude Sonnet 4.1)
   - Cost-aware routing options

#### Should Have (P1)
1. **Fallback Mechanism**
   - Automatic fallback to alternative provider on failure
   - Timeout handling with model switching
   - Error recovery strategies

2. **Performance Optimization**
   - Context caching to avoid re-reading
   - Minimal token usage through smart prompting
   - Parallel tool execution where possible

3. **Observatory Integration**
   - Send execution events to Observatory dashboard
   - Resource usage reporting
   - Pattern analysis for optimization

#### Nice to Have (P2)
1. **Streaming Responses**
   - Stream token-by-token for long responses
   - Progressive status updates
   - Early cancellation support

2. **Custom Provider Support**
   - Plugin architecture for new providers
   - Support for local models (Ollama)
   - Azure OpenAI proper integration

### Non-Functional Requirements

1. **Performance**
   - < 2 second latency for simple tasks with fast models
   - < 30 second timeout for complex tasks
   - Graceful degradation under load

2. **Reliability**
   - 99% success rate for supported operations
   - Automatic retry with exponential backoff
   - Clear error messages on failure

3. **Compatibility**
   - Maintain backward compatibility with existing API
   - Cross-platform support (Windows, Mac, Linux)
   - Python 3.9+ compatibility

4. **Observability**
   - Detailed logging at multiple levels
   - Metrics collection for analysis
   - Debug mode for troubleshooting

## Proposed Solution

### Architecture Overview

```
┌─────────────────────────────────────────────────┐
│            Nano-Agent MCP Interface             │
│                 (Unchanged API)                  │
└─────────────────────┬───────────────────────────┘
                      ▼
         ┌────────────────────────────┐
         │    Provider Abstraction    │
         │         Layer              │
         └────────────┬───────────────┘
                      ▼
        ┌─────────────┴─────────────┐
        ▼                           ▼
┌──────────────┐           ┌──────────────┐
│   OpenAI     │           │  Anthropic   │
│   Provider   │           │   Provider   │
├──────────────┤           ├──────────────┤
│ Agent SDK    │           │ Native SDK   │
│ (existing)   │           │   (new)      │
└──────────────┘           └──────────────┘
        │                           │
        ▼                           ▼
   OpenAI Models              Claude Models
   - GPT-5 family             - Sonnet 4.1
   - o3/o3-mini               - Opus 4
                              - Haiku
```

### Implementation Approach

#### Phase 1: Monitoring & Visibility (Week 1)
1. Add progress reporting hooks to existing system
2. Implement simple console progress indicators
3. Add token/cost tracking and reporting
4. Create debugging output mode

#### Phase 2: LiteLLM Integration (Week 2-3)
1. Integrate LiteLLM for provider abstraction
2. Implement tool format translation layer
3. Add response normalization
4. Test with Claude models via LiteLLM

#### Phase 3: Native Provider Support (Week 4-6)
1. Build provider-agnostic agent loop
2. Implement native Anthropic SDK integration
3. Add intelligent model routing logic
4. Comprehensive testing across providers

### Technical Design

#### Provider Abstraction Layer
```python
class ProviderAdapter:
    """Abstract base for provider adapters"""
    async def complete(self, messages, tools, **kwargs):
        raise NotImplementedError
    
    def translate_tools(self, tools):
        raise NotImplementedError
    
    def normalize_response(self, response):
        raise NotImplementedError

class OpenAIAdapter(ProviderAdapter):
    """OpenAI-specific implementation"""
    # Uses existing Agent SDK

class AnthropicAdapter(ProviderAdapter):
    """Anthropic-specific implementation"""
    # Direct SDK usage with tool translation
```

#### Tool Format Translation
```python
class ToolTranslator:
    @staticmethod
    def openai_to_anthropic(tool):
        """Convert OpenAI tool format to Anthropic"""
        return {
            "name": tool["function"]["name"],
            "description": tool["function"]["description"],
            "input_schema": tool["function"]["parameters"]
        }
    
    @staticmethod
    def anthropic_to_openai(tool_use):
        """Convert Anthropic tool use to OpenAI format"""
        return {
            "type": "function",
            "function": {
                "name": tool_use["name"],
                "arguments": json.dumps(tool_use["input"])
            }
        }
```

#### Model Router
```python
class ModelRouter:
    def select_model(self, task_complexity, cost_preference):
        if task_complexity == "simple":
            if cost_preference == "cheapest":
                return ("anthropic", "claude-3-haiku")
            else:
                return ("openai", "o3-mini")
        elif task_complexity == "complex":
            if cost_preference == "performance":
                return ("anthropic", "claude-sonnet-4.1")
            else:
                return ("openai", "o3")
```

## Success Metrics

### Quantitative Metrics
1. **Cost Reduction**: 50-70% reduction in API costs through Claude usage
2. **Performance**: 2-3x faster execution for simple tasks
3. **Success Rate**: >95% task completion rate across providers
4. **Coverage**: Support for 5+ models across 2+ providers

### Qualitative Metrics
1. **Developer Satisfaction**: Improved visibility and control
2. **Cross-Platform**: Full functionality on Windows
3. **Flexibility**: Easy model switching based on task needs

## Implementation Timeline

### Phase 1: Quick Wins (Week 1)
- [x] Research and document current limitations
- [ ] Add basic progress reporting
- [ ] Implement token/cost tracking
- [ ] Create debug output mode

### Phase 2: LiteLLM Bridge (Week 2-3)
- [ ] Integrate LiteLLM library
- [ ] Build tool translation layer
- [ ] Test Claude models via LiteLLM
- [ ] Add fallback mechanisms

### Phase 3: Native Support (Week 4-6)
- [ ] Design provider abstraction
- [ ] Implement Anthropic adapter
- [ ] Build model routing logic
- [ ] Comprehensive testing
- [ ] Documentation update

### Phase 4: Polish (Week 7-8)
- [ ] Observatory integration
- [ ] Performance optimization
- [ ] Error handling improvements
- [ ] User documentation

## Risks and Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Tool calling incompatibility | High | Medium | Build comprehensive translation layer |
| Breaking API changes | High | Low | Maintain backward compatibility layer |
| Performance regression | Medium | Low | Extensive benchmarking before release |
| Anthropic API changes | Medium | Low | Version lock dependencies |

## Dependencies

### External Dependencies
- OpenAI SDK (existing)
- Anthropic SDK (new)
- LiteLLM (new)
- Rich console library (existing)

### Internal Dependencies
- MCP server framework
- File operation tools
- Token tracking module

## Open Questions

1. **Streaming Support**: Should we prioritize streaming responses in Phase 1?
2. **Model Selection UI**: Should model selection be automatic or user-controlled?
3. **Cost Limits**: Should we implement spending limits per execution?
4. **Caching Strategy**: How aggressive should context caching be?

## Appendix

### A. Model Comparison Table

| Model | Provider | SWE-bench | Cost (per 1M) | Speed | Best For |
|-------|----------|-----------|---------------|-------|----------|
| Claude Sonnet 4.1 | Anthropic | 72.5% | $3/$15 | Fast | Most coding |
| o3 | OpenAI | 69.1% | $10/$40 | Slow | Complex reasoning |
| o3-mini | OpenAI | ~65% | $2/$8 | Medium | Balanced tasks |
| GPT-5-mini | OpenAI | ~60% | $1/$4 | Fast | Simple edits |

### B. Tool Calling Format Examples

**OpenAI Format:**
```json
{
  "type": "function",
  "function": {
    "name": "read_file",
    "description": "Read a file",
    "parameters": {
      "type": "object",
      "properties": {
        "path": {"type": "string"}
      }
    }
  }
}
```

**Anthropic Format:**
```json
{
  "name": "read_file",
  "description": "Read a file",
  "input_schema": {
    "type": "object",
    "properties": {
      "path": {"type": "string"}
    }
  }
}
```

### C. References
- [OpenAI Agent SDK Documentation](https://github.com/openai/agent-sdk)
- [Anthropic Tool Use Guide](https://docs.anthropic.com/claude/docs/tool-use)
- [LiteLLM Documentation](https://docs.litellm.ai)
- [SWE-bench Leaderboard](https://www.swebench.com)

---

*This PRD is a living document and will be updated as requirements evolve and new information becomes available.*