# Apply OpenAI compatibility patches first, before any OpenAI imports
from .modules import openai_compat

# Then apply typing fixes
from .modules import typing_fix

def hello() -> str:
    return "Hello from nano-agent!"
