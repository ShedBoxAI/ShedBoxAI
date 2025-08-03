"""
Enhanced error handling and user guidance for introspection feature.

This module provides comprehensive error handling with helpful hints
and recovery suggestions for common introspection failures.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.text import Text


@dataclass
class ErrorContext:
    """Context information for error reporting."""

    source_name: str
    source_type: str
    file_path: Optional[str] = None
    url: Optional[str] = None
    config_section: Optional[str] = None


class IntrospectionErrorHandler:
    """Handles introspection errors with helpful user guidance."""

    def __init__(self, console: Console):
        self.console = console

    def handle_config_file_error(self, config_path: str, error: Exception) -> None:
        """Handle configuration file errors with helpful guidance."""

        if isinstance(error, FileNotFoundError):
            self._show_file_not_found_error(config_path, "configuration file")
        elif isinstance(error, PermissionError):
            self._show_permission_error(config_path, "read")
        elif "yaml" in str(error).lower() or "json" in str(error).lower():
            self._show_config_syntax_error(config_path, error)
        else:
            self._show_generic_config_error(config_path, error)

    def handle_source_analysis_error(self, context: ErrorContext, error: Exception) -> str:
        """
        Handle data source analysis errors and return helpful error message.

        Returns:
            User-friendly error message with recovery suggestions
        """

        if isinstance(error, FileNotFoundError):
            return self._get_file_not_found_message(context)
        elif isinstance(error, PermissionError):
            return self._get_permission_error_message(context)
        elif "authentication" in str(error).lower() or "unauthorized" in str(error).lower():
            return self._get_auth_error_message(context, error)
        elif "connection" in str(error).lower() or "network" in str(error).lower():
            return self._get_network_error_message(context, error)
        elif "json" in str(error).lower() or "yaml" in str(error).lower():
            return self._get_parsing_error_message(context, error)
        else:
            return self._get_generic_analysis_error_message(context, error)

    def show_validation_errors(self, errors: List[Tuple[str, str]]) -> None:
        """Show configuration validation errors before analysis starts."""

        if not errors:
            return

        self.console.print("\n[red]❌ Configuration Validation Errors:[/red]")

        for source_name, error_message in errors:
            self.console.print(f"  • [yellow]{source_name}[/yellow]: {error_message}")

        self.console.print("\n[cyan]💡 Fix these issues before running introspection:[/cyan]")
        self.console.print("  1. Check file paths exist and are readable")
        self.console.print("  2. Verify API URLs are accessible")
        self.console.print("  3. Ensure environment variables are set in .env file")
        self.console.print("  4. Use --skip-errors to analyze only working sources")

    def show_environment_variable_guidance(self, missing_vars: List[str]) -> None:
        """Show guidance for missing environment variables."""

        self.console.print("\n[yellow]⚠️  Missing Environment Variables:[/yellow]")

        for var in missing_vars:
            self.console.print(f"  • {var}")

        env_file_path = Path.cwd() / ".env"

        guidance = Text()
        guidance.append("💡 Create a .env file in your project root:\n\n", style="cyan")

        for var in missing_vars:
            if "API_KEY" in var:
                guidance.append(f"{var}=your_api_key_here\n", style="green")
            elif "CLIENT_ID" in var:
                guidance.append(f"{var}=your_client_id\n", style="green")
            elif "CLIENT_SECRET" in var:
                guidance.append(f"{var}=your_client_secret\n", style="green")
            elif "USERNAME" in var:
                guidance.append(f"{var}=your_username\n", style="green")
            elif "PASSWORD" in var:
                guidance.append(f"{var}=your_password\n", style="green")
            else:
                guidance.append(f"{var}=your_value_here\n", style="green")

        guidance.append(f"\nSave as: {env_file_path}", style="dim")

        self.console.print(Panel(guidance, title="Environment Setup", border_style="cyan"))

    def show_auth_troubleshooting(self, source_name: str, auth_type: str) -> None:
        """Show authentication troubleshooting guidance."""

        auth_guides = {
            "bearer": [
                "1. Verify your API key is correct and active",
                "2. Check API key permissions and scopes",
                "3. Test manually: curl -H 'Authorization: Bearer $API_KEY' [URL]",
                "4. Check for rate limiting or quota exceeded",
            ],
            "basic": [
                "1. Verify username and password are correct",
                "2. Check if account is locked or expired",
                "3. Test manually: curl -u 'username:password' [URL]",
                "4. Ensure basic auth is enabled on the endpoint",
            ],
            "oauth": [
                "1. Verify client_id and client_secret are correct",
                "2. Check OAuth scopes and permissions",
                "3. Ensure token endpoint URL is correct",
                "4. Check if OAuth application is approved/active",
            ],
        }

        steps = auth_guides.get(auth_type.lower(), auth_guides["bearer"])

        self.console.print(f"\n[yellow]🔐 Authentication Troubleshooting for {source_name}:[/yellow]")
        for step in steps:
            self.console.print(f"  {step}")

    def show_recovery_suggestions(self, failed_sources: List[str], total_sources: int) -> None:
        """Show recovery suggestions based on failure patterns."""

        if not failed_sources:
            return

        self.console.print("\n[yellow]🔧 Recovery Suggestions:[/yellow]")

        if len(failed_sources) == total_sources:
            self.console.print("  • All sources failed - check your configuration file format")
            self.console.print("  • Verify file paths and network connectivity")
            self.console.print("  • Run with --verbose for detailed error information")
        elif len(failed_sources) < total_sources / 2:
            self.console.print(
                f"  • {len(failed_sources)} sources failed, {total_sources - len(failed_sources)} succeeded"
            )
            self.console.print("  • Use --skip-errors to generate partial introspection")
            self.console.print("  • Fix failed sources and run --retry with specific source names")
        else:
            self.console.print("  • Multiple sources failed - check common issues:")
            self.console.print("    - Network connectivity")
            self.console.print("    - Environment variables in .env file")
            self.console.print("    - File permissions and paths")

        # Show retry command suggestion
        retry_list = ",".join(failed_sources[:3])  # Show first 3 for brevity
        if len(failed_sources) > 3:
            retry_list += "..."

        self.console.print("\n[cyan]💡 To retry specific sources:[/cyan]")
        self.console.print(f"  shedboxai introspect config.yaml --retry {retry_list}")

    def _show_file_not_found_error(self, file_path: str, file_type: str) -> None:
        """Show file not found error with helpful guidance."""

        self.console.print(f"\n[red]❌ {file_type.title()} not found: {file_path}[/red]")

        # Check if it's a relative path issue
        abs_path = Path(file_path).resolve()
        current_dir = Path.cwd()

        suggestions = [
            f"Current directory: {current_dir}",
            f"Looking for: {abs_path}",
            "",
            "💡 Suggestions:",
            "  1. Check the file path spelling",
            "  2. Use absolute paths to avoid confusion",
            "  3. Ensure you're in the correct working directory",
        ]

        # Look for similar files
        if file_path.endswith((".yaml", ".yml")):
            similar_files = list(Path.cwd().glob("*.yaml")) + list(Path.cwd().glob("*.yml"))
            if similar_files:
                suggestions.extend(
                    [
                        "",
                        "📁 YAML files found in current directory:",
                        *[f"    {f.name}" for f in similar_files[:5]],
                    ]
                )

        for suggestion in suggestions:
            self.console.print(suggestion)

    def _show_permission_error(self, file_path: str, operation: str) -> None:
        """Show permission error with helpful guidance."""

        self.console.print(f"\n[red]❌ Permission denied: Cannot {operation} {file_path}[/red]")

        self.console.print("\n💡 Solutions:")
        self.console.print("  1. Check file permissions with: ls -la")
        self.console.print(f"  2. Fix permissions with: chmod 644 {file_path}")
        self.console.print("  3. Ensure you have access to the parent directory")
        self.console.print("  4. Run with appropriate user permissions")

    def _show_config_syntax_error(self, config_path: str, error: Exception) -> None:
        """Show configuration syntax error with helpful guidance."""

        self.console.print("\n[red]❌ Configuration file syntax error:[/red]")
        self.console.print(f"  File: {config_path}")
        self.console.print(f"  Error: {str(error)}")

        self.console.print("\n💡 Common YAML/JSON issues:")
        self.console.print("  1. Check indentation (YAML uses spaces, not tabs)")
        self.console.print("  2. Ensure proper quoting of strings")
        self.console.print("  3. Verify all brackets/braces are balanced")
        self.console.print("  4. Use a YAML/JSON validator online")

        # Suggest online validators
        self.console.print("\n🔗 Online validators:")
        self.console.print("  • YAML: https://yaml-online-parser.appspot.com/")
        self.console.print("  • JSON: https://jsonlint.com/")

    def _show_generic_config_error(self, config_path: str, error: Exception) -> None:
        """Show generic configuration error."""

        self.console.print("\n[red]❌ Configuration error:[/red]")
        self.console.print(f"  File: {config_path}")
        self.console.print(f"  Error: {str(error)}")

        self.console.print("\n💡 Try:")
        self.console.print("  1. Check the configuration file format")
        self.console.print("  2. Compare with working examples in tests/fixtures/")
        self.console.print("  3. Run with --verbose for more details")

    def _get_file_not_found_message(self, context: ErrorContext) -> str:
        """Get file not found error message."""

        return f"File not found: {context.file_path}\n" f"💡 Check file path and permissions"

    def _get_permission_error_message(self, context: ErrorContext) -> str:
        """Get permission error message."""

        return f"Permission denied accessing: {context.file_path}\n" f"💡 Fix with: chmod 644 {context.file_path}"

    def _get_auth_error_message(self, context: ErrorContext, error: Exception) -> str:
        """Get authentication error message."""

        error_str = str(error).lower()

        if "unauthorized" in error_str or "401" in error_str:
            return "Authentication failed: Invalid credentials\n" "💡 Check API key in environment variables"
        elif "forbidden" in error_str or "403" in error_str:
            return "Authentication failed: Insufficient permissions\n" "💡 Verify API key has required scopes"
        elif "rate limit" in error_str or "429" in error_str:
            return "Rate limit exceeded\n" "💡 Wait before retrying or reduce --sample-size"
        else:
            return f"Authentication error: {str(error)}\n" f"💡 Check credentials and API endpoint"

    def _get_network_error_message(self, context: ErrorContext, error: Exception) -> str:
        """Get network error message."""

        return (
            f"Network error: Cannot connect to {context.url}\n"
            f"💡 Check internet connection and API endpoint availability"
        )

    def _get_parsing_error_message(self, context: ErrorContext, error: Exception) -> str:
        """Get data parsing error message."""

        return f"Data parsing error: {str(error)}\n" f"💡 File may be corrupted or in unexpected format"

    def _get_generic_analysis_error_message(self, context: ErrorContext, error: Exception) -> str:
        """Get generic analysis error message."""

        return f"Analysis failed: {str(error)}\n" f"💡 Run with --verbose for detailed error information"
