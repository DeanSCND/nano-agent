"""
Early OpenAI compatibility patches to be applied before importing OpenAI.

This module creates stub types that openai-agents expects but which don't
exist in certain versions of the OpenAI SDK.
"""

import sys
import types
import logging

logger = logging.getLogger(__name__)

def patch_existing_modules():
    """Monkey-patch missing types into existing OpenAI modules."""
    
    # Patch missing types in response_prompt_param module
    try:
        import openai.types.responses.response_prompt_param as prompt_module
        
        if not hasattr(prompt_module, 'Variables'):
            class Variables:
                """Stub for Variables."""
                pass
            prompt_module.Variables = Variables
            logger.debug("Added Variables to existing module")
            
    except ImportError:
        # Module doesn't exist, will be created later
        pass
    
    # Try to import the existing module
    try:
        import openai.types.responses.response_input_item_param as input_item_module
        
        # Add missing types if they don't exist
        if not hasattr(input_item_module, 'LocalShellCallOutput'):
            class LocalShellCallOutput:
                """Stub for LocalShellCallOutput."""
                pass
            input_item_module.LocalShellCallOutput = LocalShellCallOutput
            logger.debug("Added LocalShellCallOutput to existing module")
        
        if not hasattr(input_item_module, 'McpApprovalResponse'):
            class McpApprovalResponse:
                """Stub for McpApprovalResponse."""
                pass
            input_item_module.McpApprovalResponse = McpApprovalResponse
            logger.debug("Added McpApprovalResponse to existing module")
            
    except ImportError:
        # Module doesn't exist, will be created later
        pass

    # Patch missing ImageGenerationCall in response_output_item module
    try:
        import openai.types.responses.response_output_item as output_item_module

        if not hasattr(output_item_module, 'ImageGenerationCall'):
            class ImageGenerationCall:
                """Stub for ImageGenerationCall."""
                pass
            output_item_module.ImageGenerationCall = ImageGenerationCall
            logger.debug("Added ImageGenerationCall to existing module")
            
        # Add LocalShellCall if missing
        if not hasattr(output_item_module, 'LocalShellCall'):
            class LocalShellCall:
                """Stub for LocalShellCall."""
                pass
            output_item_module.LocalShellCall = LocalShellCall
            logger.debug("Added LocalShellCall to existing module")
        
        # Add ResponseOutputItemParam stub if missing
        if not hasattr(output_item_module, 'ResponseOutputItemParam'):
            class ResponseOutputItemParam:
                """Stub for ResponseOutputItemParam."""
                pass
            output_item_module.ResponseOutputItemParam = ResponseOutputItemParam
            logger.debug("Added ResponseOutputItemParam to existing module")
        
        # Add McpApprovalRequest stub if missing
        if not hasattr(output_item_module, 'McpApprovalRequest'):
            class McpApprovalRequest:
                """Stub for McpApprovalRequest."""
                pass
            output_item_module.McpApprovalRequest = McpApprovalRequest
            logger.debug("Added McpApprovalRequest to existing module")
        
        # Add McpCall stub if missing
        if not hasattr(output_item_module, 'McpCall'):
            class McpCall:
                """Stub for McpCall."""
                pass
            output_item_module.McpCall = McpCall
            logger.debug("Added McpCall to existing module")

        # Add McpListTools stub if missing
        if not hasattr(output_item_module, 'McpListTools'):
            class McpListTools:
                """Stub for McpListTools."""
                pass
            output_item_module.McpListTools = McpListTools
            logger.debug("Added McpListTools to existing module")

    except ImportError:
        # Module doesn't exist, will be created later
        pass
    
    # Patch missing types in tool_param module
    try:
        import openai.types.responses.tool_param as tool_param_module
        
        if not hasattr(tool_param_module, 'CodeInterpreter'):
            class CodeInterpreter:
                """Stub for CodeInterpreter."""
                pass
            tool_param_module.CodeInterpreter = CodeInterpreter
            logger.debug("Added CodeInterpreter to existing module")
        
        if not hasattr(tool_param_module, 'ImageGeneration'):
            class ImageGeneration:
                """Stub for ImageGeneration."""
                pass
            tool_param_module.ImageGeneration = ImageGeneration
            logger.debug("Added ImageGeneration to existing module")
        
        if not hasattr(tool_param_module, 'Mcp'):
            class Mcp:
                """Stub for Mcp."""
                pass
            tool_param_module.Mcp = Mcp
            logger.debug("Added Mcp to existing module")
            
    except ImportError:
        # Module doesn't exist, will be created later
        pass

def create_response_stubs():
    """Create all necessary response stub modules and types."""
    
    # Check if we need to create stubs
    try:
        # Try importing to see if real module exists
        import openai.types.responses
        # Module exists, we're good
        return
    except (ImportError, ModuleNotFoundError):
        pass
    
    # Create the responses package structure
    responses_pkg = types.ModuleType('openai.types.responses')
    responses_pkg.__path__ = []  # Make it a package
    sys.modules['openai.types.responses'] = responses_pkg
    
    # Create response_prompt_param module
    response_prompt_module = types.ModuleType('openai.types.responses.response_prompt_param')
    
    class ResponsePromptParam:
        """Stub for ResponsePromptParam."""
        pass
    
    class Variables:
        """Stub for Variables."""
        pass
    
    response_prompt_module.ResponsePromptParam = ResponsePromptParam
    response_prompt_module.Variables = Variables
    sys.modules['openai.types.responses.response_prompt_param'] = response_prompt_module
    responses_pkg.response_prompt_param = response_prompt_module
    
    # Create response_input_item_param module
    input_item_module = types.ModuleType('openai.types.responses.response_input_item_param')
    
    class LocalShellCallOutput:
        """Stub for LocalShellCallOutput."""
        pass
    
    class McpApprovalResponse:
        """Stub for McpApprovalResponse."""
        pass
    
    class ResponseInputItemParam:
        """Stub for ResponseInputItemParam."""
        pass
    
    input_item_module.LocalShellCallOutput = LocalShellCallOutput
    input_item_module.McpApprovalResponse = McpApprovalResponse
    input_item_module.ResponseInputItemParam = ResponseInputItemParam
    sys.modules['openai.types.responses.response_input_item_param'] = input_item_module
    responses_pkg.response_input_item_param = input_item_module
    
    # Create function_tool_param module
    function_tool_module = types.ModuleType('openai.types.responses.function_tool_param')
    
    class FunctionToolParam:
        """Stub for FunctionToolParam."""
        pass
    
    function_tool_module.FunctionToolParam = FunctionToolParam
    sys.modules['openai.types.responses.function_tool_param'] = function_tool_module
    responses_pkg.function_tool_param = function_tool_module
    
    # Create tool_param module
    tool_param_module = types.ModuleType('openai.types.responses.tool_param')
    
    class CodeInterpreter:
        """Stub for CodeInterpreter."""
        pass
    
    class ImageGeneration:
        """Stub for ImageGeneration."""
        pass
    
    class Mcp:
        """Stub for Mcp."""
        pass
    
    tool_param_module.CodeInterpreter = CodeInterpreter
    tool_param_module.ImageGeneration = ImageGeneration
    tool_param_module.Mcp = Mcp
    sys.modules['openai.types.responses.tool_param'] = tool_param_module
    responses_pkg.tool_param = tool_param_module

    # Create response_output_item module
    output_item_module = types.ModuleType('openai.types.responses.response_output_item')
    
    class ImageGenerationCall:
        """Stub for ImageGenerationCall."""
        pass

    class LocalShellCall:
        """Stub for LocalShellCall."""
        pass

    # Optional generic stub for ResponseOutputItemParam (added for completeness)
    class ResponseOutputItemParam:
        """Stub for ResponseOutputItemParam."""
        pass

    class McpApprovalRequest:
        """Stub for McpApprovalRequest."""
        pass
    
    class McpCall:
        """Stub for McpCall."""
        pass
    
    class McpListTools:
        """Stub for McpListTools."""
        pass
    
    output_item_module.ImageGenerationCall = ImageGenerationCall
    output_item_module.LocalShellCall = LocalShellCall
    output_item_module.ResponseOutputItemParam = ResponseOutputItemParam
    output_item_module.McpApprovalRequest = McpApprovalRequest
    output_item_module.McpCall = McpCall
    output_item_module.McpListTools = McpListTools
    sys.modules['openai.types.responses.response_output_item'] = output_item_module
    responses_pkg.response_output_item = output_item_module
    
    # Now inject into openai.types
    try:
        import openai.types as types_module
        types_module.responses = responses_pkg
    except ImportError:
        # OpenAI not yet imported, will be linked when it is
        pass
    
    logger.debug("Created OpenAI response type stubs")

# Apply patches immediately
patch_existing_modules()
create_response_stubs()