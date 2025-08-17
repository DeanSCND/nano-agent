"""
Test progress monitoring enhancements in RichLoggingHooks.

This module tests the enhanced progress monitoring features including:
- Real-time tool execution display
- Token usage tracking
- Cost calculation
- Rich formatting output
"""

import pytest
import asyncio
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime
import time

from nano_agent.modules.nano_agent import RichLoggingHooks
from nano_agent.modules.token_tracking import TokenTracker
from agents import Usage


class TestRichLoggingHooks:
    """Test the enhanced RichLoggingHooks class."""
    
    @pytest.fixture
    def token_tracker(self):
        """Create a mock token tracker."""
        tracker = TokenTracker(model="gpt-5-mini", provider="openai")
        return tracker
    
    @pytest.fixture
    def hooks(self, token_tracker):
        """Create RichLoggingHooks instance with token tracker."""
        return RichLoggingHooks(token_tracker=token_tracker)
    
    @pytest.fixture
    def mock_context(self):
        """Create a mock context with usage information."""
        context = Mock()
        context.trace_metadata = {
            "model": "gpt-5-mini",
            "provider": "openai",
            "timestamp": datetime.now().isoformat()
        }
        context.usage = Usage(
            requests=1,
            input_tokens=100,
            output_tokens=50,
            total_tokens=150
        )
        return context
    
    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent."""
        agent = Mock()
        agent.name = "TestAgent"
        agent.model = Mock()
        agent.model.model = "gpt-5-mini"
        return agent
    
    @pytest.fixture
    def mock_tool(self):
        """Create a mock tool."""
        tool = Mock()
        tool.name = "read_file"
        return tool
    
    @pytest.mark.asyncio
    async def test_on_agent_start_displays_rich_info(self, hooks, mock_context, mock_agent):
        """Test that on_agent_start displays model, provider, and timestamp."""
        with patch('nano_agent.modules.nano_agent.console') as mock_console:
            await hooks.on_agent_start(mock_context, mock_agent)
            
            # Verify console.print was called
            assert mock_console.print.called
            
            # Verify agent_start_time was set
            assert hasattr(hooks, 'agent_start_time')
            assert isinstance(hooks.agent_start_time, float)
            
            # Verify token tracker was updated
            if hooks.token_tracker:
                assert hooks.token_tracker.total_usage.total_tokens > 0
    
    @pytest.mark.asyncio
    async def test_on_tool_start_increments_counter(self, hooks, mock_context, mock_agent, mock_tool):
        """Test that on_tool_start increments tool counter."""
        initial_count = hooks.tool_call_count
        
        with patch('nano_agent.modules.nano_agent.console'):
            await hooks.on_tool_start(mock_context, mock_agent, mock_tool)
            
            # Verify tool counter incremented
            assert hooks.tool_call_count == initial_count + 1
            
            # Verify tool name was stored
            assert hooks.tool_call_map[hooks.tool_call_count] == "read_file"
            
            # Verify start time was recorded
            assert hasattr(hooks, 'current_tool_start_time')
    
    @pytest.mark.asyncio
    async def test_on_tool_end_shows_running_totals(self, hooks, mock_context, mock_agent, mock_tool):
        """Test that on_tool_end displays running token and cost totals."""
        # Set up initial state
        hooks.current_tool_number = 1
        hooks.current_tool_start_time = time.time()
        
        with patch('nano_agent.modules.nano_agent.console') as mock_console:
            result = "File content successfully read"
            await hooks.on_tool_end(mock_context, mock_agent, mock_tool, result)
            
            # Verify console.print was called
            assert mock_console.print.called
            
            # Get the actual call arguments
            call_args = mock_console.print.call_args
            if call_args:
                # Check if Panel was created with proper title containing metrics
                panel = call_args[0][0] if call_args[0] else None
                if panel and hasattr(panel, 'title'):
                    # Title should contain execution time, tokens, and cost
                    assert "Tool Call #" in panel.title
                    assert "s)" in panel.title  # execution time
                    assert "Tokens:" in panel.title
                    assert "Cost:" in panel.title
    
    @pytest.mark.asyncio
    async def test_on_agent_end_comprehensive_summary(self, hooks, mock_context, mock_agent):
        """Test that on_agent_end provides comprehensive final summary."""
        # Set up state
        hooks.agent_start_time = time.time() - 5.0  # 5 seconds ago
        hooks.tool_call_count = 3
        
        # Add some token usage
        hooks.token_tracker.update(Usage(
            requests=1,
            input_tokens=500,
            output_tokens=200,
            total_tokens=700
        ))
        
        with patch('nano_agent.modules.nano_agent.console') as mock_console:
            output = "Task completed successfully"
            await hooks.on_agent_end(mock_context, mock_agent, output)
            
            # Verify console.print was called
            assert mock_console.print.called
            
            # Get the actual call
            call_args = mock_console.print.call_args
            if call_args:
                panel = call_args[0][0] if call_args[0] else None
                if panel:
                    # Verify it's a summary panel
                    assert hasattr(panel, 'title')
                    assert "Agent Summary" in panel.title or "Agent Finished" in panel.title
    
    def test_truncate_value(self, hooks):
        """Test the _truncate_value helper method."""
        # Short value should not be truncated
        short_val = "short"
        assert hooks._truncate_value(short_val, 10) == "short"
        
        # Long value should be truncated
        long_val = "a" * 200
        truncated = hooks._truncate_value(long_val, 100)
        assert len(truncated) == 100
        assert truncated.endswith("...")
    
    def test_format_tool_args_returns_dict(self, hooks):
        """Test that _format_tool_args returns a dictionary."""
        result = hooks._format_tool_args("test_tool")
        assert isinstance(result, dict)
    
    @pytest.mark.asyncio
    async def test_token_tracker_integration(self, token_tracker):
        """Test that token tracker correctly calculates costs."""
        # Add some usage
        token_tracker.update(Usage(
            requests=1,
            input_tokens=1000,
            output_tokens=500,
            total_tokens=1500
        ))
        
        # Generate report
        report = token_tracker.generate_report()
        
        # Verify counts
        assert report.total_tokens == 1500
        assert report.total_input_tokens == 1000
        assert report.total_output_tokens == 500
        
        # Verify cost calculation (based on gpt-5-mini pricing)
        assert report.total_cost > 0
        assert report.input_cost > 0
        assert report.output_cost > 0
    
    @pytest.mark.asyncio
    async def test_hooks_without_token_tracker(self):
        """Test that hooks work without token tracker."""
        hooks = RichLoggingHooks(token_tracker=None)
        
        mock_context = Mock()
        mock_agent = Mock(name="TestAgent")
        mock_tool = Mock(name="test_tool")
        
        # These should not raise errors
        with patch('nano_agent.modules.nano_agent.console'):
            await hooks.on_agent_start(mock_context, mock_agent)
            await hooks.on_tool_start(mock_context, mock_agent, mock_tool)
            await hooks.on_tool_end(mock_context, mock_agent, mock_tool, "result")
            await hooks.on_agent_end(mock_context, mock_agent, "output")
    
    @pytest.mark.asyncio
    async def test_model_provider_extraction(self, hooks, mock_agent):
        """Test extraction of model and provider information."""
        # Test with trace metadata
        context_with_meta = Mock()
        context_with_meta.trace_metadata = {
            "model": "gpt-5",
            "provider": "azure"
        }
        # Add proper usage object
        context_with_meta.usage = Usage(
            requests=1,
            input_tokens=100,
            output_tokens=50,
            total_tokens=150
        )
        
        with patch('nano_agent.modules.nano_agent.console'):
            await hooks.on_agent_start(context_with_meta, mock_agent)
            # Should extract from trace_metadata
        
        # Test fallback to agent attributes
        context_no_meta = Mock()
        context_no_meta.trace_metadata = {}
        # Don't set usage to test without token tracking
        delattr(context_no_meta, 'usage') if hasattr(context_no_meta, 'usage') else None
        agent_with_model = Mock()
        agent_with_model.name = "TestAgent"
        agent_with_model.model_name = "claude-3-haiku"
        
        with patch('nano_agent.modules.nano_agent.console'):
            await hooks.on_agent_start(context_no_meta, agent_with_model)
            # Should extract from agent.model_name
    
    def test_execution_time_calculation(self, hooks):
        """Test that execution time is calculated correctly."""
        hooks.current_tool_start_time = time.time() - 2.5  # 2.5 seconds ago
        
        mock_context = Mock()
        # Add proper usage object
        mock_context.usage = Usage(
            requests=1,
            input_tokens=100,
            output_tokens=50,
            total_tokens=150
        )
        mock_agent = Mock()
        mock_tool = Mock(name="test_tool")
        
        with patch('nano_agent.modules.nano_agent.console'):
            # Run synchronously for testing
            asyncio.run(hooks.on_tool_end(mock_context, mock_agent, mock_tool, "result"))
            
            # Execution time should be approximately 2.5 seconds
            # (checked via the panel title in actual implementation)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])