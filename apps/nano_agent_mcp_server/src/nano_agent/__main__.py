#!/usr/bin/env python
"""Nano Agent MCP Server - Main entry point."""

# Apply typing fixes FIRST before any other imports that might use OpenAI SDK
from .modules import typing_fix

import logging
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# Load environment variables from .env file
load_dotenv()

# Import our nano agent tool
from .modules.nano_agent import prompt_nano_agent

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the MCP server instance
mcp = FastMCP(
    name="nano-agent",
    instructions="""
    A powerful MCP server that bridges Model Context Protocol with OpenAI's Agent SDK.
    
    This server enables autonomous agent execution through natural language prompts,
    allowing clients to describe work in plain English and have it completed by
    an AI agent with access to file system tools.
    
    The agent can read files, create files, and perform complex multi-step tasks
    autonomously, making it ideal for code generation, data processing, and
    automation workflows.
    
    Main tool:
    - prompt_nano_agent: Execute an autonomous agent with a natural language task description
    """
)

# Register the nano agent tool
mcp.tool()(prompt_nano_agent)


def run():
    """Entry point for the nano-agent command."""
    try:
        logger.info("Starting Nano Agent MCP Server...")
        # FastMCP.run() handles its own async context with anyio
        # Don't wrap it in asyncio.run()
        mcp.run()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise


if __name__ == "__main__":
    run()