"""Entry point for ``python -m cta``; runs the Typer command-line app."""

from cta.cli import app

if __name__ == "__main__":
    app.main()
