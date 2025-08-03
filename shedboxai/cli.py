"""
Command-line interface for ShedBoxAI.

Provides a simple CLI for running ShedBoxAI pipelines from configuration files.
"""

from __future__ import annotations

# Disable urllib3 warnings before importing any modules that might use it
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import NoReturn

from .core.exceptions import (
    AuthenticationError,
    CyclicDependencyError,
    EnvironmentVariableError,
    GraphExecutionError,
    MissingDependencyError,
    NetworkError,
    ShedBoxAIError,
    UnknownOperationError,
)
from .pipeline import Pipeline


class CLIError(Exception):
    """Base exception for CLI-specific errors."""

    pass


class ConfigFileError(CLIError):
    """Raised when configuration file issues are detected."""

    pass


class OutputFileError(CLIError):
    """Raised when output file issues are detected."""

    pass


def validate_config_file(config_path: str) -> Path:
    """
    Validate configuration file exists and is readable.

    Args:
        config_path: Path to configuration file

    Returns:
        Path object for the configuration file

    Raises:
        ConfigFileError: If file doesn't exist, isn't readable, or has wrong extension
    """
    try:
        config_file = Path(config_path)
    except (TypeError, ValueError) as e:
        raise ConfigFileError(f"Invalid config path '{config_path}': {e}")

    # Check if file exists
    if not config_file.exists():
        raise ConfigFileError(f"Configuration file not found: {config_file}")

    # Check if it's actually a file (not a directory)
    if not config_file.is_file():
        raise ConfigFileError(f"Configuration path is not a file: {config_file}")

    # Check file extension
    valid_extensions = {".yaml", ".yml", ".json"}
    if config_file.suffix.lower() not in valid_extensions:
        raise ConfigFileError(
            f"Invalid configuration file extension '{config_file.suffix}'. "
            f"Supported extensions: {', '.join(valid_extensions)}"
        )

    # Check if file is readable
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            # Try to read first byte to check permissions
            f.read(1)
    except PermissionError:
        raise ConfigFileError(f"Permission denied reading configuration file: {config_file}")
    except UnicodeDecodeError:
        raise ConfigFileError(f"Configuration file is not valid UTF-8: {config_file}")
    except OSError as e:
        raise ConfigFileError(f"Cannot read configuration file {config_file}: {e}")

    return config_file


def validate_output_file(output_path: str) -> Path:
    """
    Validate output file path and permissions.

    Args:
        output_path: Path where output should be written

    Returns:
        Path object for the output file

    Raises:
        OutputFileError: If output path is invalid or not writable
    """
    try:
        output_file = Path(output_path)
    except (TypeError, ValueError) as e:
        raise OutputFileError(f"Invalid output path '{output_path}': {e}")

    # Check if parent directory exists or can be created
    parent_dir = output_file.parent
    if not parent_dir.exists():
        try:
            parent_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            raise OutputFileError(f"Permission denied creating output directory: {parent_dir}")
        except OSError as e:
            raise OutputFileError(f"Cannot create output directory {parent_dir}: {e}")

    # Check if parent is actually a directory
    if not parent_dir.is_dir():
        raise OutputFileError(f"Output parent path is not a directory: {parent_dir}")

    # Check write permissions on parent directory
    if not parent_dir.is_dir() or not parent_dir.stat().st_mode & 0o200:
        raise OutputFileError(f"No write permission for output directory: {parent_dir}")

    # If output file already exists, check if it's writable
    if output_file.exists():
        if not output_file.is_file():
            raise OutputFileError(f"Output path exists but is not a file: {output_file}")

        try:
            # Test write access by opening in append mode
            with open(output_file, "a", encoding="utf-8"):
                pass
        except PermissionError:
            raise OutputFileError(f"Permission denied writing to output file: {output_file}")
        except OSError as e:
            raise OutputFileError(f"Cannot write to output file {output_file}: {e}")

    return output_file


def print_error(message: str, verbose: bool = False) -> None:
    """Print error message to stderr with optional details."""
    print(f"Error: {message}", file=sys.stderr)
    if verbose:
        print(traceback.format_exc(), file=sys.stderr)


def exit_with_error(message: str, verbose: bool = False, exit_code: int = 1) -> NoReturn:
    """Print error and exit with specified code."""
    print_error(message, verbose)
    sys.exit(exit_code)


def format_shedboxai_error(error: ShedBoxAIError) -> str:
    """Format a ShedBoxAI error for display."""
    message = str(error)

    # For certain error types, add extra information
    if isinstance(error, EnvironmentVariableError):
        message += "\n\nHint: Make sure to create a .env file " "with required environment variables."

    elif isinstance(error, NetworkError):
        message += "\n\nHint: Check your network connection and API endpoint availability."
    elif isinstance(error, AuthenticationError):
        message += "\n\nHint: Verify your API credentials and permissions."
    elif isinstance(error, UnknownOperationError):
        message += "\n\nHint: Check the operation name in your graph configuration."
    elif isinstance(error, MissingDependencyError):
        message += "\n\nHint: Ensure all node IDs referenced " "in dependencies exist in your graph."

    elif isinstance(error, CyclicDependencyError):
        message += "\n\nHint: Restructure your graph to eliminate circular dependencies."
    elif isinstance(error, GraphExecutionError) and hasattr(error, "example_yaml") and error.example_yaml:
        # If it's a GraphExecutionError with an example, format it nicely
        message += f"\n\nExample:\n```yaml\n{error.example_yaml}\n```"

    # Add additional suggestions if available
    if hasattr(error, "suggestion") and error.suggestion:
        message += f"\n\nSuggestion: {error.suggestion}"

    # Add config path if available
    if hasattr(error, "config_path") and error.config_path:
        message += f"\n\nConfiguration path: {error.config_path}"

    return message


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ShedBoxAI - AI-powered applications through configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  run          Execute ShedBoxAI pipeline with processing operations
    Common options: --output, --verbose, --quiet

  introspect   Analyze data sources and generate documentation
    Common options: --output, --verbose, --quiet
    Specific options: --retry, --skip-errors, --force, --validate, --include-samples

Examples:
  shedboxai run config.yaml                    # Run pipeline with config
  shedboxai run config.yaml -o results.json   # Run and save results
  shedboxai run config.yaml --quiet           # Run without logging output

  shedboxai introspect sources.yaml            # Analyze data sources
  shedboxai introspect sources.yaml --force    # Overwrite existing output
  shedboxai introspect sources.yaml --include-samples  # Include sample data
        """,
    )
    parser.add_argument("command", choices=["run", "introspect"], help="Command to execute")
    parser.add_argument("config", help="Path to configuration file")
    parser.add_argument("--output", "-o", help="Output file path (optional)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress log messages")

    # Introspect-specific arguments
    parser.add_argument(
        "--retry",
        help="[introspect only] Retry specific failed sources (comma-separated)",
    )
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="[introspect only] Continue with partial results if some sources fail",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="[introspect only] Overwrite existing introspection.md",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="[introspect only] Re-validate existing introspection against current sources",
    )
    parser.add_argument(
        "--include-samples",
        action="store_true",
        help="[introspect only] Include sample data structures in output (default: off)",
    )

    # Parse arguments with error handling
    try:
        args = parser.parse_args()
    except SystemExit as e:
        # argparse calls sys.exit() on error, but we want to handle it
        if e.code != 0:
            sys.exit(e.code)
        return

    # Configure logging based on command line arguments
    root_logger = logging.getLogger()

    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    if args.quiet:
        # Suppress all logging
        root_logger.setLevel(logging.CRITICAL + 1)  # Above CRITICAL to suppress all
        # Also disable urllib3 warnings
        logging.getLogger("urllib3").setLevel(logging.CRITICAL + 1)
        # Disable all warnings
        warnings.filterwarnings("ignore")
    elif args.verbose:
        # Enable debug logging
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.DEBUG)
    else:
        # Default to only showing warnings and errors
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.WARNING)
        # Disable urllib3 warnings
        logging.getLogger("urllib3").setLevel(logging.ERROR)

    # Validate inputs early
    try:
        config_file = validate_config_file(args.config)

        output_file = None
        if args.output:
            output_file = validate_output_file(args.output)

    except CLIError as e:
        exit_with_error(str(e), args.verbose)

    # Run the pipeline
    if args.command == "run":
        try:
            if args.verbose:
                print(f"Loading configuration from: {config_file}")

            pipeline = Pipeline(str(config_file))
            result = pipeline.run()

            if output_file:
                if args.verbose:
                    print(f"Saving results to: {output_file}")

                try:
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(result, f, indent=2, ensure_ascii=False)

                    if args.verbose:
                        print(f"Results successfully saved to: {output_file}")

                except (OSError, IOError) as e:
                    exit_with_error(f"Failed to write output file: {e}", args.verbose)
                except (TypeError, ValueError) as e:
                    exit_with_error(f"Failed to serialize results to JSON: {e}", args.verbose)
            else:
                # Print results to stdout if no output file specified
                try:
                    # Print a summary instead of the full JSON
                    print("\n✅ ShedBoxAI Pipeline Execution Completed\n")

                    # Print key counts and statistics
                    print("📊 Data Summary:")
                    for source_name, source_data in result.items():
                        if source_name == "customer_analysis" or not isinstance(source_data, dict):
                            continue

                        print(f"  • {source_name}: ", end="")
                        if isinstance(source_data, dict):
                            items = []
                            for key, value in source_data.items():
                                if isinstance(value, list):
                                    items.append(f"{len(value)} {key}")
                            if items:
                                print(", ".join(items))
                            else:
                                print("processed successfully")
                        else:
                            print("processed successfully")

                    # Print AI analysis if available
                    if "customer_analysis" in result:
                        print("\n🤖 AI Analysis:")
                        print(f"  {result['customer_analysis']}")

                    # Print success message with hint for full JSON
                    print("\n💡 Tip: Use the --output option to save the full result as JSON")

                except (TypeError, ValueError) as e:
                    exit_with_error(f"Failed to format results: {e}", args.verbose)

        except KeyboardInterrupt:
            exit_with_error("Operation cancelled by user", args.verbose, exit_code=130)
        except ShedBoxAIError as e:
            # Format and print ShedBoxAI-specific errors
            exit_with_error(format_shedboxai_error(e), args.verbose)
        except Exception as e:
            # For unexpected errors, provide more context with verbose flag
            if args.verbose:
                exit_with_error(f"Unexpected error: {e}", True)
            else:
                exit_with_error(
                    f"Unexpected error: {e}\n\nRun with --verbose for more details",
                    False,
                )

    # Run introspection
    elif args.command == "introspect":
        from .core.introspection.introspect_cli import run_introspection

        try:
            run_introspection(
                config_path=str(config_file),
                output_path=args.output or "introspection.md",
                sample_size=100,  # Fixed default value since we removed the flag
                retry_sources=args.retry.split(",") if args.retry else [],
                skip_errors=args.skip_errors,
                force_overwrite=args.force,
                validate_only=args.validate,
                verbose=args.verbose,
                include_samples=args.include_samples,
            )

        except KeyboardInterrupt:
            exit_with_error("Operation cancelled by user", args.verbose, exit_code=130)
        except ShedBoxAIError as e:
            # Format and print ShedBoxAI-specific errors
            exit_with_error(format_shedboxai_error(e), args.verbose)
        except Exception as e:
            # For unexpected errors, provide more context with verbose flag
            if args.verbose:
                exit_with_error(f"Unexpected error: {e}", True)
            else:
                exit_with_error(
                    f"Unexpected error: {e}\n\nRun with --verbose for more details",
                    False,
                )


if __name__ == "__main__":
    main()
