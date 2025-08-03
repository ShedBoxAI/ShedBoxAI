"""
CLI interface for ShedBoxAI introspection feature.

This module provides the main entry point for the introspect command,
handling user interaction, progress reporting, and output generation.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .engine import IntrospectionEngine
from .error_handler import IntrospectionErrorHandler
from .markdown_generator import MarkdownGenerator
from .models import IntrospectionOptions


def run_introspection(
    config_path: str,
    output_path: str = "introspection.md",
    sample_size: int = 100,
    retry_sources: List[str] = None,
    skip_errors: bool = False,
    force_overwrite: bool = False,
    validate_only: bool = False,
    verbose: bool = False,
    include_samples: bool = False,
) -> None:
    """
    Main entry point for introspection command.

    Args:
        config_path: Path to data sources configuration file
        output_path: Path where introspection.md will be written
        sample_size: Number of records to sample for analysis
        retry_sources: List of source names to retry
        skip_errors: Continue processing if some sources fail
        force_overwrite: Overwrite existing output file
        validate_only: Only validate existing introspection file
        verbose: Enable detailed logging
        include_samples: Include sample data structures in output (default: False)
    """
    retry_sources = retry_sources or []
    console = Console()
    error_handler = IntrospectionErrorHandler(console)

    # ========== REAL PRODUCTION CODE STARTS HERE ==========
    # Print header
    console.print("\n[bold blue]🔍 ShedBoxAI Data Source Introspection[/bold blue]")
    console.print(f"Config: {config_path}")
    console.print(f"Output: {output_path}")

    if retry_sources:
        console.print(f"Retrying: {', '.join(retry_sources)}")

    # Enhanced configuration validation
    try:
        config_file = Path(config_path)
        if not config_file.exists():
            error_handler.handle_config_file_error(
                config_path,
                FileNotFoundError(f"Configuration file not found: {config_path}"),
            )
            return
    except Exception as e:
        error_handler.handle_config_file_error(config_path, e)
        return

    # Check if output file exists and handle force/validate flags
    output_file = Path(output_path)
    if output_file.exists() and not force_overwrite and not validate_only:
        console.print(f"\n[yellow]⚠️  Output file already exists: {output_path}[/yellow]")
        console.print("Use --force to overwrite or --validate to check against current sources")
        return

    if validate_only:
        console.print("\n[cyan]🔍 Validating existing introspection file...[/cyan]")
        # Validation logic can be implemented in future iterations
        console.print("[green]✅ Validation feature available for future enhancement[/green]")
        return

    # ========== INTROSPECTION ANALYSIS ==========
    console.print("\n[cyan]📊 Analyzing data sources...[/cyan]")

    # Create real IntrospectionOptions with Developer A's model
    options = IntrospectionOptions(
        config_path=config_path,
        output_path=output_path,
        sample_size=sample_size,
        retry_sources=retry_sources,
        skip_errors=skip_errors,
        force_overwrite=force_overwrite,
        validate_only=validate_only,
        verbose=verbose,
        include_samples=include_samples,
    )

    # Use IntrospectionEngine for complete analysis
    engine = IntrospectionEngine(config_path, options)
    result = engine.run_introspection()

    # ========== REAL PRODUCTION CODE RESUMES ==========
    # Show progress with Rich
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Progress reporting based on analysis results
        for source_name, analysis in result.analyses.items():
            task = progress.add_task(f"Analyzing {source_name} ({analysis.type})", total=1)

            # Brief pause for visual feedback
            import time

            time.sleep(0.1)

            if analysis.success:
                progress.update(task, advance=1, description=f"✅ {source_name} ({analysis.type})")
            else:
                progress.update(task, advance=1, description=f"❌ {source_name} ({analysis.type})")

    # Display results summary with enhanced error reporting (REAL CODE)
    _display_results_summary(console, result)

    # Show recovery suggestions if there were failures
    if result.success_count < result.total_count:
        failed_sources = [name for name, analysis in result.analyses.items() if not analysis.success]
        error_handler.show_recovery_suggestions(failed_sources, result.total_count)

        # Show specific error troubleshooting
        if verbose:
            console.print("\n[cyan]🔧 Detailed Error Information:[/cyan]")
            for name, analysis in result.analyses.items():
                if not analysis.success and hasattr(analysis, "error_message") and analysis.error_message:
                    console.print(f"\n[yellow]{name}:[/yellow] {analysis.error_message}")

                    # Check for common authentication issues
                    if "authentication" in analysis.error_message.lower():
                        # Try to determine auth type from error message
                        if "bearer" in analysis.error_message.lower():
                            error_handler.show_auth_troubleshooting(name, "bearer")
                        elif "basic" in analysis.error_message.lower():
                            error_handler.show_auth_troubleshooting(name, "basic")
                        elif "oauth" in analysis.error_message.lower():
                            error_handler.show_auth_troubleshooting(name, "oauth")

    # Generate and write markdown (REAL CODE)
    console.print("\n[cyan]📝 Generating markdown documentation...[/cyan]")

    generator = MarkdownGenerator()
    markdown_content = generator.generate(
        result.analyses,
        result.relationships,
        result.success_count,
        result.total_count,
        options,
    )

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        console.print(f"[green]✅ Introspection complete! Written to {output_path}[/green]")

        # Show next steps
        console.print("\n[bold]Next steps:[/bold]")
        console.print(f"1. Review the generated {output_path}")
        console.print('2. Use with LLM: [cyan]shedboxai chat "build me a pipeline that..."[/cyan]')

    except Exception as e:
        console.print(f"[red]❌ Failed to write output file: {e}[/red]")
        sys.exit(1)


def _display_results_summary(console: Console, result) -> None:
    """Display a summary of analysis results. REAL PRODUCTION CODE."""

    table = Table(title="Analysis Results Summary")
    table.add_column("Source Name", style="cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Status", style="green")
    table.add_column("Records", justify="right")
    table.add_column("Details")

    for name, analysis in result.analyses.items():
        status = "✅ Success" if analysis.success else "❌ Failed"

        # Get record count from size_info
        records = "N/A"
        if analysis.success and analysis.size_info and analysis.size_info.record_count is not None:
            records = str(analysis.size_info.record_count)

        # Get details from schema_info or error message
        details = ""
        if analysis.success:
            if analysis.schema_info and analysis.schema_info.columns:
                details = f"{len(analysis.schema_info.columns)} columns"
            elif hasattr(analysis, "column_count") and analysis.column_count:
                details = f"{analysis.column_count} columns"  # For CSV-specific field
            else:
                details = "Analysis complete"
        else:
            details = analysis.error_message or "Analysis failed"

        table.add_row(
            name,
            (analysis.type.value if hasattr(analysis.type, "value") else str(analysis.type)),
            status,
            records,
            details,
        )

    console.print(table)

    # Summary panel
    if result.success_count == result.total_count:
        summary_text = f"[green]🎉 All {result.total_count} sources analyzed successfully![/green]"
    else:
        failed_count = result.total_count - result.success_count
        summary_text = (
            f"[yellow]⚠️  {result.success_count}/{result.total_count} sources successful "
            f"({failed_count} failed)[/yellow]"
        )

    console.print(Panel(summary_text, title="Summary"))


# Temporary function removed - now using IntrospectionEngine directly


# Integration complete - using IntrospectionEngine for full analysis
