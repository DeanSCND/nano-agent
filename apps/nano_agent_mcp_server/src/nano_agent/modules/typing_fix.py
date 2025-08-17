"""Comprehensive typing compatibility fix for OpenAI SDK with openai-agents library.

This fix is necessary because:
1. OpenAI SDK >=1.99.2 changed several TypedDict types to Union types
2. The openai-agents library tries to instantiate these Union types directly
3. Union types cannot be instantiated in Python

This will be unnecessary once openai-agents updates to handle the new type structure.
"""

import sys
import logging

logger = logging.getLogger(__name__)

def apply_missing_modules():
    """Create stub modules for missing OpenAI types that openai-agents expects."""
    
    import types
    
    # Ensure responses module exists
    if 'openai.types.responses' not in sys.modules:
        responses_module = types.ModuleType('openai.types.responses')
        sys.modules['openai.types.responses'] = responses_module
        
        # Add to parent module
        import openai.types
        openai.types.responses = responses_module
    
    # Create stub for response_prompt_param if it doesn't exist
    try:
        from openai.types.responses import response_prompt_param
    except (ImportError, ModuleNotFoundError):
        # Create response_prompt_param module with stub
        response_prompt_module = types.ModuleType('openai.types.responses.response_prompt_param')
        
        # Add a stub ResponsePromptParam class
        class ResponsePromptParam:
            """Stub for missing ResponsePromptParam type."""
            pass
        
        response_prompt_module.ResponsePromptParam = ResponsePromptParam
        sys.modules['openai.types.responses.response_prompt_param'] = response_prompt_module
        
        # Link to parent
        sys.modules['openai.types.responses'].response_prompt_param = response_prompt_module
        
        logger.debug("Created stub for missing openai.types.responses.response_prompt_param")
    
    # Fix the response_input_item_param import issue
    try:
        from openai.types.responses.response_input_item_param import LocalShellCallOutput
    except (ImportError, AttributeError):
        # Check if the module exists
        if 'openai.types.responses.response_input_item_param' in sys.modules:
            # Module exists but LocalShellCallOutput is missing
            module = sys.modules['openai.types.responses.response_input_item_param']
            
            # Create stub class
            class LocalShellCallOutput:
                """Stub for missing LocalShellCallOutput type."""
                pass
            
            # Add to module
            module.LocalShellCallOutput = LocalShellCallOutput
            logger.debug("Added stub LocalShellCallOutput to existing module")
        else:
            # Create the entire module
            input_item_module = types.ModuleType('openai.types.responses.response_input_item_param')
            
            # Add stub classes
            class LocalShellCallOutput:
                """Stub for missing LocalShellCallOutput type."""
                pass
            
            input_item_module.LocalShellCallOutput = LocalShellCallOutput
            sys.modules['openai.types.responses.response_input_item_param'] = input_item_module
            
            # Link to parent
            if 'openai.types.responses' in sys.modules:
                sys.modules['openai.types.responses'].response_input_item_param = input_item_module
            
            logger.debug("Created stub module for response_input_item_param")

def apply_patches():
    """Replace problematic Union types with concrete types for compatibility."""
    
    # Only apply once
    if hasattr(sys, '_openai_typing_patched'):
        return
    
    try:
        # Import the chat module and typing utilities
        import openai.types.chat as chat_module
        import openai.types as types_module
        from typing import get_origin, Union
        
        # Import concrete types to use as replacements
        from openai.types.chat import (
            ChatCompletionMessageFunctionToolCallParam,
            ChatCompletionAssistantMessageParam,
            ChatCompletionFunctionToolParam,
        )
        
        # List of patches to apply (Union type name -> concrete type to use)
        patches = {
            'ChatCompletionMessageToolCallParam': ChatCompletionMessageFunctionToolCallParam,
            # Add more patches here if other Union types cause issues
        }
        
        # Apply patches
        for attr_name, replacement in patches.items():
            if hasattr(chat_module, attr_name):
                original = getattr(chat_module, attr_name)
                # Only patch if it's actually a Union type
                if get_origin(original) is Union:
                    setattr(chat_module, attr_name, replacement)
                    # Also update in parent module's namespace
                    if hasattr(types_module, 'chat'):
                        setattr(types_module.chat, attr_name, replacement)
                    logger.debug(f"Patched {attr_name} from Union to {replacement.__name__}")
        
        # Mark as patched
        sys._openai_typing_patched = True
        logger.debug("OpenAI typing patches applied successfully")
        
    except ImportError as e:
        # OpenAI SDK not installed or different version structure
        logger.debug(f"Could not apply OpenAI typing patches: {e}")
    except Exception as e:
        # Log but don't fail - the patches are a workaround
        logger.debug(f"Error applying OpenAI typing patches: {e}")

# Auto-apply patches on import
apply_missing_modules()
apply_patches()