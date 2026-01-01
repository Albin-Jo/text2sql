# text2sql_mvp/main.py
"""
Text-to-SQL MVP Command Line Interface.
"""

import asyncio
import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax

# Initialize CLI
app = typer.Typer(
    name="text2sql",
    help="Text-to-SQL MVP - Convert natural language to SQL queries",
    add_completion=False,
)
console = Console()


def get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent


@app.command()
def info():
    """Display system information and configuration."""
    from config.settings import get_settings
    from strategies.registry import list_strategies, list_strategies_by_type
    from strategies.base import StrategyType
    
    settings = get_settings()
    
    console.print(Panel.fit(
        "[bold blue]Text-to-SQL MVP[/bold blue]\n"
        "Convert natural language to SQL queries",
        title="System Information"
    ))
    
    # Configuration
    table = Table(title="Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Database Mode", settings.database_mode.value)
    table.add_row("Default Domain", settings.default_domain)
    table.add_row("Gemini Model", settings.gemini.model)
    table.add_row("Log Level", settings.log_level.value)
    table.add_row("Cache Enabled", str(settings.cache.enabled))
    
    console.print(table)
    
    # Strategies
    console.print("\n[bold]Registered Strategies:[/bold]")
    
    for stype in StrategyType:
        strategies = list_strategies_by_type(stype)
        if strategies:
            console.print(f"  [cyan]{stype.value}:[/cyan] {', '.join(strategies)}")
    
    # Domains
    domains_dir = settings.domains_dir
    if domains_dir.exists():
        domains = [d.name for d in domains_dir.iterdir() if d.is_dir()]
        console.print(f"\n[bold]Available Domains:[/bold] {', '.join(domains)}")


@app.command()
def generate(
    question: str = typer.Argument(..., help="Natural language question"),
    domain: str = typer.Option("finance", "--domain", "-d", help="Domain to use"),
    strategy: str = typer.Option("zero_shot", "--strategy", "-s", help="Strategy to use"),
    execute: bool = typer.Option(False, "--execute", "-e", help="Execute the generated SQL"),
    temperature: float = typer.Option(0.0, "--temp", "-t", help="Generation temperature"),
):
    """Generate SQL from a natural language question."""
    asyncio.run(_generate_async(question, domain, strategy, execute, temperature))


async def _generate_async(
    question: str,
    domain: str,
    strategy_name: str,
    execute: bool,
    temperature: float,
):
    """Async implementation of generate command."""
    from config.settings import get_settings
    from core.gemini_client import GeminiClient
    from schema.manager import SchemaManager
    from strategies.registry import get_strategy
    from strategies.base import GenerationContext
    
    settings = get_settings()
    
    # Check API key
    if not settings.google_api_key:
        console.print("[red]Error: GOOGLE_API_KEY not set[/red]")
        console.print("Set it in .env file or environment variable")
        raise typer.Exit(1)
    
    # Load schema
    schema_path = settings.domains_dir / domain / "schema.sql"
    if not schema_path.exists():
        console.print(f"[red]Error: Schema not found at {schema_path}[/red]")
        raise typer.Exit(1)
    
    console.print(f"[dim]Loading schema from {schema_path}...[/dim]")
    schema_manager = SchemaManager()
    schema = schema_manager.load_from_sql(schema_path, schema_name=domain)
    
    # Get strategy
    console.print(f"[dim]Using strategy: {strategy_name}[/dim]")
    
    # Create client
    client = GeminiClient(api_key=settings.google_api_key)
    
    # Get strategy with client
    strategy = get_strategy(strategy_name, llm_client=client, temperature=temperature)
    if not strategy:
        console.print(f"[red]Error: Strategy '{strategy_name}' not found[/red]")
        console.print(f"Available: {', '.join(list_strategies())}")
        raise typer.Exit(1)
    
    # Create context
    context = GenerationContext(
        schema=schema,
        question=question,
        dialect="bigquery",
    )
    
    # Generate
    console.print(f"\n[bold]Question:[/bold] {question}\n")
    
    with console.status("[bold green]Generating SQL..."):
        result = await strategy.generate(question, context)
    
    # Display results
    if result.success:
        console.print("[bold green]Generated SQL:[/bold green]")
        syntax = Syntax(result.sql, "sql", theme="monokai", line_numbers=True)
        console.print(syntax)
        
        # Show metadata
        console.print(f"\n[dim]Latency: {result.latency_ms:.0f}ms | "
                     f"Tokens: {result.total_tokens} | "
                     f"Model: {result.model}[/dim]")
        
        if execute:
            console.print("\n[yellow]Execution not yet implemented[/yellow]")
    else:
        console.print(f"[red]Generation failed: {result.error}[/red]")
        if result.raw_response:
            console.print(f"\n[dim]Raw response:[/dim]\n{result.raw_response}")


@app.command()
def validate(
    domain: str = typer.Option("finance", "--domain", "-d", help="Domain to validate"),
):
    """Validate domain schema and configuration."""
    from config.settings import get_settings
    from schema.manager import SchemaManager
    from data.test_cases import TestCaseLoader
    from data.few_shot_pool import FewShotPool
    
    settings = get_settings()
    domain_path = settings.domains_dir / domain
    
    console.print(f"[bold]Validating domain: {domain}[/bold]\n")
    
    errors = []
    warnings = []
    
    # Check schema
    schema_path = domain_path / "schema.sql"
    if schema_path.exists():
        console.print(f"✓ Schema file found: {schema_path}")
        try:
            manager = SchemaManager()
            schema = manager.load_from_sql(schema_path)
            console.print(f"  Tables: {', '.join(schema.table_names())}")
            
            # Validate schema
            issues = manager.validate_schema(schema)
            for issue in issues:
                warnings.append(f"Schema: {issue}")
        except Exception as e:
            errors.append(f"Schema parse error: {e}")
    else:
        errors.append(f"Schema file not found: {schema_path}")
    
    # Check test cases
    test_cases_path = domain_path / "test_cases.json"
    if test_cases_path.exists():
        console.print(f"✓ Test cases found: {test_cases_path}")
        try:
            loader = TestCaseLoader()
            loader.load_from_json(test_cases_path)
            stats = loader.get_stats()
            console.print(f"  Total: {stats['total']} | By difficulty: {stats['by_difficulty']}")
        except Exception as e:
            errors.append(f"Test cases error: {e}")
    else:
        warnings.append(f"Test cases not found: {test_cases_path}")
    
    # Check few-shot examples
    examples_path = domain_path / "few_shot_examples.json"
    if examples_path.exists():
        console.print(f"✓ Few-shot examples found: {examples_path}")
        try:
            pool = FewShotPool()
            pool.load_from_json(examples_path)
            console.print(f"  Examples: {pool.size}")
        except Exception as e:
            errors.append(f"Few-shot examples error: {e}")
    else:
        warnings.append(f"Few-shot examples not found: {examples_path}")
    
    # Summary
    console.print()
    if errors:
        console.print("[red]Errors:[/red]")
        for e in errors:
            console.print(f"  ✗ {e}")
    
    if warnings:
        console.print("[yellow]Warnings:[/yellow]")
        for w in warnings:
            console.print(f"  ⚠ {w}")
    
    if not errors and not warnings:
        console.print("[green]✓ All validations passed![/green]")
    
    raise typer.Exit(1 if errors else 0)


@app.command("test-connection")
def test_connection():
    """Test connection to Gemini API."""
    asyncio.run(_test_connection_async())


async def _test_connection_async():
    """Async implementation of connection test."""
    from config.settings import get_settings
    from core.gemini_client import GeminiClient
    
    settings = get_settings()
    
    if not settings.google_api_key:
        console.print("[red]Error: GOOGLE_API_KEY not set[/red]")
        raise typer.Exit(1)
    
    console.print("[bold]Testing Gemini API connection...[/bold]\n")
    
    try:
        client = GeminiClient(api_key=settings.google_api_key)
        
        with console.status("[bold green]Sending test request..."):
            response = await client.generate(
                prompt="Say 'Hello' and nothing else.",
                temperature=0.0,
                max_tokens=10,
            )
        
        if response.success:
            console.print(f"[green]✓ Connection successful![/green]")
            console.print(f"  Model: {response.metrics.model}")
            console.print(f"  Response: {response.text.strip()}")
            console.print(f"  Latency: {response.metrics.latency_ms:.0f}ms")
        else:
            console.print(f"[red]✗ Request failed: {response.error}[/red]")
            raise typer.Exit(1)
            
    except Exception as e:
        console.print(f"[red]✗ Connection error: {e}[/red]")
        raise typer.Exit(1)


@app.command("list-strategies")
def list_strategies_cmd():
    """List all available generation strategies."""
    from strategies.registry import get_registry
    from strategies.base import StrategyType
    
    # Import strategies to register them
    import strategies.prompting  # noqa
    
    registry = get_registry()
    
    table = Table(title="Available Strategies")
    table.add_column("Name", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Description")
    
    for name in sorted(registry.list_strategies()):
        info = registry.get_info(name)
        if info:
            table.add_row(
                info.get("name", name),
                info.get("type", "unknown"),
                info.get("description", "")
            )
    
    console.print(table)


# Experiment commands
experiment_app = typer.Typer(help="Experiment management commands")
app.add_typer(experiment_app, name="experiment")


@experiment_app.command("run")
def run_experiment(
    name: str = typer.Argument(..., help="Experiment name"),
    domain: str = typer.Option("finance", "--domain", "-d"),
    strategies: str = typer.Option("zero_shot,few_shot_static", "--strategies", "-s"),
    limit: int = typer.Option(10, "--limit", "-l", help="Max test cases"),
):
    """Run an experiment with specified strategies."""
    console.print(f"[yellow]Experiment runner not yet implemented[/yellow]")
    console.print(f"Would run: {name} on {domain} with strategies: {strategies}")


@experiment_app.command("list")
def list_experiments():
    """List previous experiments."""
    console.print("[yellow]Experiment listing not yet implemented[/yellow]")


def main():
    """Main entry point."""
    # Import strategies to register them
    try:
        import strategies.prompting  # noqa
    except ImportError:
        pass
    
    app()


if __name__ == "__main__":
    main()
