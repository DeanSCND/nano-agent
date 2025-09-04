"""
Test case for nano-agent empty output validation.

This test ensures that the nano-agent properly handles cases where
the agent completes execution but produces no meaningful output.
"""

import pytest
from unittest.mock import Mock, patch
from nano_agent.modules.nano_agent import _execute_nano_agent
from nano_agent.modules.data_types import PromptNanoAgentRequest


def test_empty_output_returns_failure():
    """Test that empty output from agent returns success=False with appropriate error."""
    
    # Create a mock result with empty final_output
    mock_result = Mock()
    mock_result.final_output = ""  # Empty string output
    mock_result.messages = []
    
    # Create a test request
    request = PromptNanoAgentRequest(
        agentic_prompt="Test task that produces no output",
        model="gpt-5-mini",
        provider="openai"
    )
    
    # Mock the Runner.run to return our mock result
    with patch('nano_agent.modules.nano_agent.Runner') as MockRunner:
        mock_runner_instance = MockRunner.return_value
        mock_runner_instance.run.return_value = mock_result
        
        # Also mock the validation to pass
        with patch('nano_agent.modules.nano_agent.ProviderConfig.validate_provider_setup') as mock_validate:
            mock_validate.return_value = (True, None)
            
            # Execute the agent
            response = _execute_nano_agent(request, enable_rich_logging=False)
    
    # Assertions
    assert response.success is False, "Should return success=False for empty output"
    assert response.error is not None, "Should provide an error message"
    assert "no output" in response.error.lower(), "Error should mention no output"
    assert response.metadata.get("error_type") == "empty_output", "Should indicate empty_output error type"


def test_none_output_returns_failure():
    """Test that None output from agent returns success=False with appropriate error."""
    
    # Create a mock result with None final_output
    mock_result = Mock()
    mock_result.final_output = None  # None output
    mock_result.messages = []
    
    # Mock str(mock_result) to return empty string when final_output is None
    mock_result.__str__ = Mock(return_value="")
    
    # Create a test request
    request = PromptNanoAgentRequest(
        agentic_prompt="Test task that produces None output",
        model="gpt-5-mini", 
        provider="openai"
    )
    
    # Mock the Runner.run to return our mock result
    with patch('nano_agent.modules.nano_agent.Runner') as MockRunner:
        mock_runner_instance = MockRunner.return_value
        mock_runner_instance.run.return_value = mock_result
        
        # Also mock the validation to pass
        with patch('nano_agent.modules.nano_agent.ProviderConfig.validate_provider_setup') as mock_validate:
            mock_validate.return_value = (True, None)
            
            # Execute the agent
            response = _execute_nano_agent(request, enable_rich_logging=False)
    
    # Assertions
    assert response.success is False, "Should return success=False for None output"
    assert response.error is not None, "Should provide an error message"
    assert "no output" in response.error.lower(), "Error should mention no output"
    assert response.metadata.get("error_type") == "empty_output", "Should indicate empty_output error type"


def test_whitespace_only_output_returns_failure():
    """Test that whitespace-only output returns success=False."""
    
    # Create a mock result with whitespace-only output
    mock_result = Mock()
    mock_result.final_output = "   \n\t  "  # Only whitespace
    mock_result.messages = []
    
    # Create a test request
    request = PromptNanoAgentRequest(
        agentic_prompt="Test task that produces only whitespace",
        model="gpt-5-mini",
        provider="openai"
    )
    
    # Mock the Runner.run to return our mock result
    with patch('nano_agent.modules.nano_agent.Runner') as MockRunner:
        mock_runner_instance = MockRunner.return_value
        mock_runner_instance.run.return_value = mock_result
        
        # Also mock the validation to pass
        with patch('nano_agent.modules.nano_agent.ProviderConfig.validate_provider_setup') as mock_validate:
            mock_validate.return_value = (True, None)
            
            # Execute the agent
            response = _execute_nano_agent(request, enable_rich_logging=False)
    
    # Assertions
    assert response.success is False, "Should return success=False for whitespace-only output"
    assert response.error is not None, "Should provide an error message"
    assert "no output" in response.error.lower(), "Error should mention no output"


def test_valid_output_returns_success():
    """Test that valid output returns success=True."""
    
    # Create a mock result with valid output
    mock_result = Mock()
    mock_result.final_output = "Task completed successfully with this output"
    mock_result.messages = []
    
    # Create a test request  
    request = PromptNanoAgentRequest(
        agentic_prompt="Test task that produces valid output",
        model="gpt-5-mini",
        provider="openai"
    )
    
    # Mock the Runner.run to return our mock result
    with patch('nano_agent.modules.nano_agent.Runner') as MockRunner:
        mock_runner_instance = MockRunner.return_value
        mock_runner_instance.run.return_value = mock_result
        
        # Also mock the validation to pass
        with patch('nano_agent.modules.nano_agent.ProviderConfig.validate_provider_setup') as mock_validate:
            mock_validate.return_value = (True, None)
            
            # Execute the agent
            response = _execute_nano_agent(request, enable_rich_logging=False)
    
    # Assertions
    assert response.success is True, "Should return success=True for valid output"
    assert response.result == "Task completed successfully with this output", "Should return the actual output"
    assert response.error is None, "Should not have an error for valid output"