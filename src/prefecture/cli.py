"""Command-line interface. `play` and `world show` arrive with the world engine (step 1)."""

from __future__ import annotations

import typer
from rich.console import Console

from prefecture.llm import load_config, ollama_status

app = typer.Typer(help="Préfecture: an absurd bureaucracy benchmark for LLM agents.")
console = Console()


@app.callback()
def main() -> None:
    """Préfecture command-line interface."""


@app.command()
def doctor() -> None:
    """Check that Ollama is reachable and that the configured models are pulled."""
    config = load_config()
    for name, role in config.roles.items():
        console.print(f"[bold]{name}[/]: {role.model} (temperature={role.temperature})")
    status = ollama_status(config)
    if not status["reachable"]:
        console.print(f"[red]Ollama unreachable at {config.ollama_url}[/]: {status['error']}")
        console.print("Start it with: [bold]ollama serve[/]")
        raise typer.Exit(1)
    if status["missing"]:
        for model in status["missing"]:
            console.print(f"[yellow]Missing model[/]: ollama pull {model}")
        raise typer.Exit(1)
    console.print("[green]Ollama is up and all configured models are available.[/]")


if __name__ == "__main__":
    app()
