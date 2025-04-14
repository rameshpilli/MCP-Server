import argparse
from test_llm import TestLLMClient
import json
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

console = Console()

def setup_client():
    """Initialize and setup the LLM client"""
    client = TestLLMClient()
    
    # Setup authentication and model
    if not client.api_key:
        console.print("🔑 Getting API key...")
        client.get_api_key()
    
    if not client.model_id:
        console.print("📝 Registering model...")
        client.register_model()
    
    return client

def interactive_mode():
    """Run an interactive chat session with the LLM"""
    client = setup_client()
    console.print(Panel.fit("🤖 Test LLM Interactive Mode", style="bold blue"))
    console.print("Type 'exit' to quit, 'help' for commands\n")

    while True:
        try:
            # Get user input
            prompt = console.input("[bold blue]You:[/bold blue] ")
            
            if prompt.lower() in ['exit', 'quit']:
                break
            elif prompt.lower() == 'help':
                console.print(Markdown("""
                Commands:
                - exit/quit: Exit the program
                - help: Show this help message
                - clear: Clear the screen
                - context: Show current context data
                """))
                continue
            elif prompt.lower() == 'clear':
                console.clear()
                continue
            elif prompt.lower() == 'context':
                # Query latest context data
                context_data = client.query_data("SELECT * FROM cat_jobs LIMIT 5")
                console.print(Panel(json.dumps(context_data, indent=2), title="Current Context"))
                continue

            # Get context data
            context_data = client.query_data("SELECT * FROM cat_jobs LIMIT 5")
            
            # Generate response
            response = client.generate_response(prompt, context_data)
            
            # Display response
            console.print("\n[bold green]LLM:[/bold green]", style="bold green")
            console.print(response["generated_text"])
            console.print(f"\n[dim]Tokens used: {response['tokens_used']}[/dim]")
            console.print("\n" + "─" * 50 + "\n")

        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")

def single_query(prompt: str):
    """Run a single query and exit"""
    client = setup_client()
    
    try:
        # Get context data
        context_data = client.query_data("SELECT * FROM cat_jobs LIMIT 5")
        
        # Generate response
        response = client.generate_response(prompt, context_data)
        
        # Display response
        console.print("\n[bold green]LLM:[/bold green]", style="bold green")
        console.print(response["generated_text"])
        console.print(f"\n[dim]Tokens used: {response['tokens_used']}[/dim]\n")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Test LLM CLI")
    parser.add_argument(
        "--prompt", "-p",
        help="Single prompt to query (if not provided, runs in interactive mode)"
    )
    
    args = parser.parse_args()
    
    if args.prompt:
        single_query(args.prompt)
    else:
        interactive_mode()

if __name__ == "__main__":
    main() 