#!/usr/bin/env python
"""
Nano Agent CLI - Direct command-line interface for testing the nano agent.

This provides a simple command-line interface to test the nano agent functionality
with various commands and interactive modes.
"""

import asyncio
import typer
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.progress import Progress, SpinnerColumn, TextColumn
from pathlib import Path
import os
import sys
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from .modules.nano_agent import prompt_nano_agent, _execute_nano_agent
from .modules.data_types import PromptNanoAgentRequest
from .modules.constants import (
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    ERROR_NO_API_KEY,
    DEMO_PROMPTS
)

app = typer.Typer()
console = Console()

def check_api_key():
    """Check if OpenAI API key is set."""
    if not os.getenv("OPENAI_API_KEY"):
        console.print(f"[red]Error: {ERROR_NO_API_KEY}[/red]")
        console.print("Please set it with: export OPENAI_API_KEY=your-api-key")
        sys.exit(1)

@app.command()
def test_tools():
    """Test individual tool functions."""
    # Import the raw tool functions from nano_agent_tools
    from .modules.nano_agent_tools import (
        read_file_raw,
        list_directory_raw,
        write_file_raw,
        get_file_info_raw,
        edit_file_raw
    )
    
    console.print(Panel("[cyan]Testing Nano Agent Tools[/cyan]", expand=False))
    
    # Test list_directory (call the raw function, not the FunctionTool)
    console.print("\n[yellow]1. Testing list_directory:[/yellow]")
    result = list_directory_raw(".")
    console.print(result[:500] + "..." if len(result) > 500 else result)
    
    # Test write_file
    console.print("\n[yellow]2. Testing write_file:[/yellow]")
    test_file = "test_nano_agent.txt"
    result = write_file_raw(test_file, "Hello from Nano Agent CLI!\nThis is line 2\nThis is line 3")
    console.print(result)
    
    # Test read_file
    console.print("\n[yellow]3. Testing read_file:[/yellow]")
    result = read_file_raw(test_file)
    console.print(f"Content: {result}")
    
    # Test edit_file
    console.print("\n[yellow]4. Testing edit_file:[/yellow]")
    result = edit_file_raw(test_file, "This is line 2", "This is the EDITED line 2")
    console.print(f"Edit result: {result}")
    result = read_file_raw(test_file)
    console.print(f"Content after edit: {result}")
    
    # Test get_file_info
    console.print("\n[yellow]5. Testing get_file_info:[/yellow]")
    result = get_file_info_raw(test_file)
    info = json.loads(result)
    console.print(json.dumps(info, indent=2))
    
    # Clean up
    Path(test_file).unlink(missing_ok=True)
    console.print("\n[green]✓ All tool tests completed successfully![/green]")

@app.command()
def run(
    prompt: str,
    model: str = typer.Option(DEFAULT_MODEL, help="Model to use (default: gpt-5-mini)"),
    provider: str = typer.Option(DEFAULT_PROVIDER, help="Provider to use"),
    max_turns: Optional[int] = typer.Option(None, help="Maximum agent turns (default: 20, max: 100)"),
    verbose: bool = typer.Option(False, help="Show detailed output")
):
    """Run the nano agent with a prompt."""
    check_api_key()
    
    console.print(Panel(f"[cyan]Running Nano Agent[/cyan]\nModel: {model}\nProvider: {provider}", expand=False))
    console.print(f"\n[yellow]Prompt:[/yellow] {prompt}\n")
    
    # Create request
    request = PromptNanoAgentRequest(
        agentic_prompt=prompt,
        model=model,
        provider=provider,
        max_turns=max_turns
    )
    
    # Execute agent without progress spinner (rich logging will show progress)
    response = _execute_nano_agent(request)
    
    # Display results in panels
    if response.success:
        console.print(Panel(
            f"[green]{response.result}[/green]",
            title="📋 Agent Result",
            border_style="green",
            expand=False
        ))
        
        if verbose:
            # Format metadata as a single JSON object
            metadata_display = response.metadata.copy()
            
            # Add execution time to metadata
            metadata_display["execution_time_seconds"] = round(response.execution_time_seconds, 2)
            
            # Format token usage fields if present
            if "token_usage" in metadata_display:
                usage = metadata_display["token_usage"]
                # Flatten key metrics for better display
                metadata_display["token_usage"] = {
                    "total_tokens": f"{usage['total_tokens']:,}",
                    "input_tokens": f"{usage['input_tokens']:,}",
                    "output_tokens": f"{usage['output_tokens']:,}",
                    "cached_tokens": f"{usage['cached_tokens']:,}",
                    "total_cost": f"${usage['total_cost']:.4f}"
                }
            
            # Pretty print the combined metadata
            metadata_json = json.dumps(metadata_display, indent=2)
            
            console.print(Panel(
                Syntax(metadata_json, "json", theme="monokai", line_numbers=False),
                title="🔍 Metadata & Usage",
                border_style="dim",
                expand=False
            ))
    else:
        console.print(Panel(
            f"[red]{response.error}[/red]",
            title="❌ Agent Failed",
            border_style="red",
            expand=False
        ))
        if verbose and response.metadata:
            console.print(Panel(
                json.dumps(response.metadata, indent=2),
                title="🔍 Error Details",
                border_style="dim",
                expand=False
            ))

@app.command()
def demo():
    """Run a demo showing various agent capabilities."""
    check_api_key()
    
    console.print(Panel("[cyan]Nano Agent Demo[/cyan]", expand=False))
    
    for i, (prompt, model) in enumerate(DEMO_PROMPTS, 1):
        console.print(f"\n[yellow]Demo {i}:[/yellow] {prompt}")
        
        request = PromptNanoAgentRequest(
            agentic_prompt=prompt,
            model=model,
            provider=DEFAULT_PROVIDER
        )
        
        # Execute without progress spinner
        response = _execute_nano_agent(request)
        
        if response.success:
            console.print(f"[green]✓[/green] {response.result[:200]}...")
        else:
            console.print(f"[red]✗[/red] {response.error}")
    
    # Clean up
    Path("demo.txt").unlink(missing_ok=True)
    console.print("\n[green]✓ Demo completed![/green]")

@app.command()
def interactive():
    """Run the agent in interactive mode."""
    check_api_key()
    
    console.print(Panel("[cyan]Nano Agent Interactive Mode[/cyan]\nType 'exit' to quit", expand=False))
    
    model = typer.prompt("Model to use", default=DEFAULT_MODEL)
    
    while True:
        try:
            prompt = typer.prompt("\n[yellow]Enter prompt[/yellow]")
            
            if prompt.lower() in ["exit", "quit", "q"]:
                console.print("[dim]Goodbye![/dim]")
                break
            
            request = PromptNanoAgentRequest(
                agentic_prompt=prompt,
                model=model,
                provider=DEFAULT_PROVIDER
            )
            
            # Execute without progress spinner
            response = _execute_nano_agent(request)
            
            if response.success:
                console.print(Panel(
                    response.result,
                    title="💬 Agent Response",
                    border_style="cyan",
                    expand=False
                ))
            else:
                console.print(Panel(
                    response.error,
                    title="❌ Error",
                    border_style="red",
                    expand=False
                ))
                
        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted. Type 'exit' to quit.[/dim]")
        except Exception as e:
            console.print(f"\n[red]Error:[/red] {str(e)}")

def main():
    """Main entry point for the CLI."""
    app()

if __name__ == "__main__":
    main()