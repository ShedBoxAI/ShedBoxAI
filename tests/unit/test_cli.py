"""
Comprehensive unit tests for the CLI module with error handling.
"""

import argparse
import json
import os
import sys
import tempfile
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, Mock, mock_open, patch

import pytest

# Import the new CLI functions and exceptions
from shedboxai.cli import (
    CLIError,
    ConfigFileError,
    OutputFileError,
    exit_with_error,
    format_shedboxai_error,
    main,
    print_error,
    validate_config_file,
    validate_output_file,
)

# Remove global pipeline module mock - use targeted patches instead


class MockPipeline:
    def __init__(self, config_path):
        self.config_path = config_path
        self.run_called = False

    def run(self):
        self.run_called = True
        return {"result": "success", "data": [1, 2, 3]}


class TestConfigFileValidation:
    """Test configuration file validation scenarios."""

    def test_nonexistent_file(self):
        """Test error when config file doesn't exist."""
        with pytest.raises(ConfigFileError, match="Configuration file not found"):
            validate_config_file("nonexistent.yaml")

    def test_directory_instead_of_file(self, tmp_path):
        """Test error when config path points to directory."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()

        with pytest.raises(ConfigFileError, match="Configuration path is not a file"):
            validate_config_file(str(config_dir))

    def test_invalid_file_extension(self, tmp_path):
        """Test error with unsupported file extension."""
        config_file = tmp_path / "config.txt"
        config_file.write_text("content")

        with pytest.raises(ConfigFileError, match="Invalid configuration file extension"):
            validate_config_file(str(config_file))

    def test_unreadable_file(self, tmp_path):
        """Test error when file permissions deny reading."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: content")
        config_file.chmod(0o000)  # No permissions

        try:
            with pytest.raises(ConfigFileError, match="Permission denied reading"):
                validate_config_file(str(config_file))
        finally:
            # Cleanup - restore permissions so pytest can clean up
            config_file.chmod(0o644)

    def test_binary_file(self, tmp_path):
        """Test error with non-UTF8 file."""
        config_file = tmp_path / "config.yaml"
        config_file.write_bytes(b"\xff\xfe\x00\x00")  # Invalid UTF-8

        with pytest.raises(ConfigFileError, match="not valid UTF-8"):
            validate_config_file(str(config_file))

    def test_valid_yaml_file(self, tmp_path):
        """Test successful validation of valid YAML file."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: value")

        result = validate_config_file(str(config_file))
        assert result == config_file

    def test_valid_yml_file(self, tmp_path):
        """Test successful validation of valid YML file."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("test: value")

        result = validate_config_file(str(config_file))
        assert result == config_file

    def test_valid_json_file(self, tmp_path):
        """Test successful validation of valid JSON file."""
        config_file = tmp_path / "config.json"
        config_file.write_text('{"test": "value"}')

        result = validate_config_file(str(config_file))
        assert result == config_file

    def test_invalid_path_type(self):
        """Test error with invalid path type."""
        with pytest.raises(ConfigFileError, match="Invalid config path"):
            validate_config_file(None)


class TestOutputFileValidation:
    """Test output file validation scenarios."""

    def test_readonly_parent_directory(self, tmp_path):
        """Test error when parent directory is readonly."""
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        readonly_dir.chmod(0o444)  # Read-only

        output_file = readonly_dir / "output.json"

        try:
            with pytest.raises(OutputFileError, match="No write permission"):
                validate_output_file(str(output_file))
        finally:
            # Cleanup
            readonly_dir.chmod(0o755)

    def test_parent_is_file_not_directory(self, tmp_path):
        """Test error when parent path is a file, not directory."""
        parent_file = tmp_path / "parent.txt"
        parent_file.write_text("content")

        output_file = parent_file / "output.json"

        with pytest.raises(OutputFileError, match="Output parent path is not a directory"):
            validate_output_file(str(output_file))

    def test_readonly_existing_output_file(self, tmp_path):
        """Test error when existing output file is readonly."""
        output_file = tmp_path / "readonly_output.json"
        output_file.write_text('{"test": "data"}')
        output_file.chmod(0o444)  # Read-only

        try:
            with pytest.raises(OutputFileError, match="Permission denied writing"):
                validate_output_file(str(output_file))
        finally:
            # Cleanup
            output_file.chmod(0o644)

    def test_output_path_is_directory(self, tmp_path):
        """Test error when output path exists as directory."""
        output_dir = tmp_path / "output.json"
        output_dir.mkdir()

        with pytest.raises(OutputFileError, match="Output path exists but is not a file"):
            validate_output_file(str(output_dir))

    def test_valid_new_output_file(self, tmp_path):
        """Test successful validation of new output file."""
        output_file = tmp_path / "output.json"

        result = validate_output_file(str(output_file))
        assert result == output_file

    def test_valid_existing_writable_file(self, tmp_path):
        """Test successful validation of existing writable file."""
        output_file = tmp_path / "existing.json"
        output_file.write_text('{"old": "data"}')

        result = validate_output_file(str(output_file))
        assert result == output_file

    def test_create_nested_directories(self, tmp_path):
        """Test creation of nested output directories."""
        output_file = tmp_path / "nested" / "deep" / "output.json"

        result = validate_output_file(str(output_file))
        assert result == output_file
        assert output_file.parent.exists()
        assert output_file.parent.is_dir()

    def test_invalid_path_type(self):
        """Test error with invalid path type."""
        with pytest.raises(OutputFileError, match="Invalid output path"):
            validate_output_file(None)


class TestCLIHelperFunctions:
    """Test CLI helper functions."""

    def test_print_error(self, capsys):
        """Test error printing function."""
        print_error("Test error message")
        captured = capsys.readouterr()
        assert "Error: Test error message" in captured.err

    def test_print_error_verbose(self, capsys):
        """Test error printing with verbose traceback."""
        try:
            raise ValueError("Test exception")
        except ValueError:
            print_error("Test error", verbose=True)

        captured = capsys.readouterr()
        assert "Error: Test error" in captured.err
        assert "ValueError: Test exception" in captured.err

    def test_exit_with_error(self):
        """Test exit with error function."""
        with pytest.raises(SystemExit) as excinfo:
            exit_with_error("Test exit message")
        assert excinfo.value.code == 1

    def test_exit_with_error_custom_code(self):
        """Test exit with custom error code."""
        with pytest.raises(SystemExit) as excinfo:
            exit_with_error("Test exit message", exit_code=42)
        assert excinfo.value.code == 42

    def test_format_shedboxai_error(self):
        """Test formatting of ShedBoxAIError."""
        # Mock exceptions import
        mock_exceptions = Mock()
        mock_exceptions.ShedBoxAIError = type("ShedBoxAIError", (Exception,), {})
        mock_exceptions.EnvironmentVariableError = type(
            "EnvironmentVariableError", (mock_exceptions.ShedBoxAIError,), {}
        )
        mock_exceptions.NetworkError = type("NetworkError", (mock_exceptions.ShedBoxAIError,), {})
        mock_exceptions.AuthenticationError = type("AuthenticationError", (mock_exceptions.ShedBoxAIError,), {})
        mock_exceptions.UnknownOperationError = type("UnknownOperationError", (mock_exceptions.ShedBoxAIError,), {})

        # Create error instances
        base_error = mock_exceptions.ShedBoxAIError("Base error message")
        env_error = mock_exceptions.EnvironmentVariableError("Missing env var")
        network_error = mock_exceptions.NetworkError("Connection failed")

        # Add additional attributes that our formatter expects
        base_error.suggestion = "Try this fix"
        base_error.config_path = "section.key"

        # Test formatting
        with patch(
            "shedboxai.cli.EnvironmentVariableError",
            mock_exceptions.EnvironmentVariableError,
        ):
            with patch("shedboxai.cli.NetworkError", mock_exceptions.NetworkError):
                with patch(
                    "shedboxai.cli.AuthenticationError",
                    mock_exceptions.AuthenticationError,
                ):
                    with patch("shedboxai.cli.ShedBoxAIError", mock_exceptions.ShedBoxAIError):
                        # Test base error with attributes
                        formatted = format_shedboxai_error(base_error)
                        assert "Base error message" in formatted
                        assert "Suggestion: Try this fix" in formatted
                        assert "Configuration path: section.key" in formatted

                        # Test environment variable error
                        formatted = format_shedboxai_error(env_error)
                        assert "Missing env var" in formatted
                        assert ".env file" in formatted

                        # Test network error
                        formatted = format_shedboxai_error(network_error)
                        assert "Connection failed" in formatted
                        assert "network connection" in formatted


class TestCLIArgumentParsing:
    """Test CLI argument parsing functionality."""

    @patch("sys.argv", ["cli.py", "run", "config.yaml"])
    @patch("shedboxai.cli.validate_config_file")
    @patch("shedboxai.cli.Pipeline")
    def test_parse_basic_run_command(self, mock_pipeline_class, mock_validate_config):
        """Test parsing basic run command."""
        mock_validate_config.return_value = Path("config.yaml")
        mock_pipeline = Mock()
        mock_pipeline.run.return_value = {"test": "data"}
        mock_pipeline_class.return_value = mock_pipeline

        with patch("sys.stdout", new_callable=StringIO):
            main()

        mock_validate_config.assert_called_once_with("config.yaml")
        mock_pipeline_class.assert_called_once_with("config.yaml")

    @patch("sys.argv", ["cli.py", "run", "config.yaml", "--output", "results.json"])
    @patch("shedboxai.cli.validate_config_file")
    @patch("shedboxai.cli.validate_output_file")
    @patch("shedboxai.cli.Pipeline")
    @patch("builtins.open", new_callable=mock_open)
    def test_parse_run_with_output(self, mock_file, mock_pipeline_class, mock_validate_output, mock_validate_config):
        """Test parsing run command with output option."""
        mock_validate_config.return_value = Path("config.yaml")
        mock_validate_output.return_value = Path("results.json")
        mock_pipeline = Mock()
        mock_pipeline.run.return_value = {"test": "data"}
        mock_pipeline_class.return_value = mock_pipeline

        with patch("sys.stdout", new_callable=StringIO):
            main()

        mock_validate_config.assert_called_once_with("config.yaml")
        mock_validate_output.assert_called_once_with("results.json")

    @patch("sys.argv", ["cli.py", "run", "config.yaml", "-v"])
    @patch("shedboxai.cli.validate_config_file")
    @patch("shedboxai.cli.Pipeline")
    @patch("shedboxai.cli.logging")
    def test_parse_run_with_verbose(self, mock_logging, mock_pipeline_class, mock_validate_config):
        """Test parsing run command with verbose option."""
        mock_validate_config.return_value = Path("config.yaml")
        mock_pipeline = Mock()
        mock_pipeline.run.return_value = {"test": "data"}
        mock_pipeline_class.return_value = mock_pipeline

        # Set up handlers attribute on the mock logger
        mock_logger = Mock()
        mock_logger.handlers = []
        mock_logging.getLogger.return_value = mock_logger

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            main()

        # Verify the command ran successfully
        output = mock_stdout.getvalue()
        assert "ShedBoxAI Pipeline Execution Completed" in output

    @patch("sys.argv", ["cli.py", "run", "config.yaml", "-q"])
    @patch("shedboxai.cli.validate_config_file")
    @patch("shedboxai.cli.Pipeline")
    @patch("warnings.filterwarnings")
    @patch("shedboxai.cli.logging")
    def test_parse_run_with_quiet(
        self,
        mock_logging,
        mock_filter_warnings,
        mock_pipeline_class,
        mock_validate_config,
    ):
        """Test parsing run command with quiet option."""
        mock_validate_config.return_value = Path("config.yaml")
        mock_pipeline = Mock()
        mock_pipeline.run.return_value = {"test": "data"}
        mock_pipeline_class.return_value = mock_pipeline

        # Set up handlers attribute on the mock logger
        mock_logger = Mock()
        mock_logger.handlers = []
        mock_logging.getLogger.return_value = mock_logger

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            main()

        # Verify warnings were suppressed
        mock_filter_warnings.assert_called_with("ignore")

        # Verify the command ran successfully
        output = mock_stdout.getvalue()
        assert "ShedBoxAI Pipeline Execution Completed" in output


class TestCLIExecution:
    """Test CLI execution functionality with new error handling."""

    @patch("sys.argv", ["cli.py", "run", "config.yaml"])
    @patch("shedboxai.cli.validate_config_file")
    @patch("shedboxai.cli.Pipeline")
    def test_main_basic_run(self, mock_pipeline_class, mock_validate_config):
        """Test basic run command execution."""
        mock_validate_config.return_value = Path("config.yaml")
        mock_pipeline = Mock()
        mock_pipeline.run.return_value = {"success": True}
        mock_pipeline_class.return_value = mock_pipeline

        with patch("sys.stdout", new_callable=StringIO):
            main()

        mock_validate_config.assert_called_once_with("config.yaml")
        mock_pipeline_class.assert_called_once_with("config.yaml")
        mock_pipeline.run.assert_called_once()

    @patch("sys.argv", ["cli.py", "run", "config.yaml", "-o", "output.json"])
    @patch("shedboxai.cli.validate_config_file")
    @patch("shedboxai.cli.validate_output_file")
    @patch("shedboxai.cli.Pipeline")
    @patch("builtins.open", new_callable=mock_open)
    def test_main_run_with_output(self, mock_file, mock_pipeline_class, mock_validate_output, mock_validate_config):
        """Test run command with output file."""
        test_result = {"data": "test", "status": "success"}
        mock_validate_config.return_value = Path("config.yaml")
        mock_validate_output.return_value = Path("output.json")
        mock_pipeline = Mock()
        mock_pipeline.run.return_value = test_result
        mock_pipeline_class.return_value = mock_pipeline

        with patch("sys.stdout", new_callable=StringIO):
            main()

        mock_file.assert_called_once_with(Path("output.json"), "w", encoding="utf-8")

        # Verify JSON was written
        handle = mock_file.return_value.__enter__.return_value
        written_data = "".join(call.args[0] for call in handle.write.call_args_list)
        assert '"data": "test"' in written_data

    @patch("sys.argv", ["cli.py", "run", "config.yaml"])
    @patch(
        "shedboxai.cli.validate_config_file",
        side_effect=ConfigFileError("Config not found"),
    )
    def test_config_file_error_handling(self, mock_validate_config, capsys):
        """Test handling of configuration file errors."""
        with pytest.raises(SystemExit) as excinfo:
            main()

        assert excinfo.value.code == 1
        captured = capsys.readouterr()
        assert "Error: Config not found" in captured.err

    @patch("sys.argv", ["cli.py", "run", "config.yaml", "-o", "invalid/output.json"])
    @patch("shedboxai.cli.validate_config_file")
    @patch(
        "shedboxai.cli.validate_output_file",
        side_effect=OutputFileError("Cannot write output"),
    )
    def test_output_file_error_handling(self, mock_validate_output, mock_validate_config, capsys):
        """Test handling of output file errors."""
        mock_validate_config.return_value = Path("config.yaml")

        with pytest.raises(SystemExit) as excinfo:
            main()

        assert excinfo.value.code == 1
        captured = capsys.readouterr()
        assert "Error: Cannot write output" in captured.err

    @patch("sys.argv", ["cli.py", "run", "config.yaml"])
    @patch("shedboxai.cli.validate_config_file")
    @patch("shedboxai.cli.Pipeline", side_effect=Exception("Pipeline error"))
    def test_pipeline_exception_handling(self, mock_pipeline_class, mock_validate_config, capsys):
        """Test handling of pipeline exceptions."""
        mock_validate_config.return_value = Path("config.yaml")

        with pytest.raises(SystemExit) as excinfo:
            main()

        assert excinfo.value.code == 1
        captured = capsys.readouterr()
        assert "Error: Unexpected error: Pipeline error" in captured.err

    @patch("sys.argv", ["cli.py", "run", "config.yaml", "-v"])
    @patch("shedboxai.cli.validate_config_file")
    @patch("shedboxai.cli.Pipeline", side_effect=Exception("Pipeline error"))
    def test_pipeline_exception_verbose(self, mock_pipeline_class, mock_validate_config, capsys):
        """Test handling of pipeline exceptions with verbose mode."""
        mock_validate_config.return_value = Path("config.yaml")

        with pytest.raises(SystemExit) as excinfo:
            main()

        assert excinfo.value.code == 1
        captured = capsys.readouterr()
        assert "Error: Unexpected error: Pipeline error" in captured.err
        assert "Traceback" in captured.err  # Verbose mode shows traceback

    @patch("sys.argv", ["cli.py", "run", "config.yaml"])
    @patch("shedboxai.cli.validate_config_file")
    @patch("shedboxai.cli.Pipeline")
    def test_keyboard_interrupt_handling(self, mock_pipeline_class, mock_validate_config, capsys):
        """Test handling of keyboard interrupt (Ctrl+C)."""
        mock_validate_config.return_value = Path("config.yaml")
        mock_pipeline_class.side_effect = KeyboardInterrupt()

        with pytest.raises(SystemExit) as excinfo:
            main()

        assert excinfo.value.code == 130  # Standard exit code for SIGINT
        captured = capsys.readouterr()
        assert "Error: Operation cancelled by user" in captured.err


class TestCLIFileOperations:
    """Test CLI file operation functionality."""

    @patch("sys.argv", ["cli.py", "run", "config.yaml", "-o", "output.json"])
    @patch("shedboxai.cli.validate_config_file")
    @patch("shedboxai.cli.validate_output_file")
    @patch("shedboxai.cli.Pipeline")
    @patch("builtins.open", new_callable=mock_open)
    def test_json_serialization(self, mock_file, mock_pipeline_class, mock_validate_output, mock_validate_config):
        """Test JSON serialization of results."""
        test_result = {
            "string": "test",
            "number": 42,
            "list": [1, 2, 3],
            "nested": {"key": "value"},
        }

        mock_validate_config.return_value = Path("config.yaml")
        mock_validate_output.return_value = Path("output.json")
        mock_pipeline = Mock()
        mock_pipeline.run.return_value = test_result
        mock_pipeline_class.return_value = mock_pipeline

        with patch("json.dump") as mock_json_dump:
            with patch("sys.stdout", new_callable=StringIO):
                main()

            # Verify json.dump was called with correct parameters
            mock_json_dump.assert_called_once()
            call_args = mock_json_dump.call_args
            assert call_args[0][0] == test_result  # First arg is the data
            assert call_args[1]["indent"] == 2
            assert call_args[1]["ensure_ascii"] is False

    @patch("sys.argv", ["cli.py", "run", "config.yaml", "-o", "output.json"])
    @patch("shedboxai.cli.validate_config_file")
    @patch("shedboxai.cli.validate_output_file")
    @patch("shedboxai.cli.Pipeline")
    @patch("builtins.open", side_effect=OSError("Disk full"))
    def test_file_write_error_handling(
        self,
        mock_file,
        mock_pipeline_class,
        mock_validate_output,
        mock_validate_config,
        capsys,
    ):
        """Test handling of file write errors."""
        test_result = {"data": "test"}
        mock_validate_config.return_value = Path("config.yaml")
        mock_validate_output.return_value = Path("output.json")
        mock_pipeline = Mock()
        mock_pipeline.run.return_value = test_result
        mock_pipeline_class.return_value = mock_pipeline

        with pytest.raises(SystemExit) as excinfo:
            main()

        assert excinfo.value.code == 1
        captured = capsys.readouterr()
        assert "Error: Failed to write output file: Disk full" in captured.err

    @patch("sys.argv", ["cli.py", "run", "config.yaml", "-o", "output.json"])
    @patch("shedboxai.cli.validate_config_file")
    @patch("shedboxai.cli.validate_output_file")
    @patch("shedboxai.cli.Pipeline")
    def test_json_serialization_error(self, mock_pipeline_class, mock_validate_output, mock_validate_config, capsys):
        """Test handling of JSON serialization errors."""

        # Create non-serializable object
        class NonSerializable:
            pass

        test_result = {"valid": "data", "invalid": NonSerializable()}
        mock_validate_config.return_value = Path("config.yaml")
        mock_validate_output.return_value = Path("output.json")
        mock_pipeline = Mock()
        mock_pipeline.run.return_value = test_result
        mock_pipeline_class.return_value = mock_pipeline

        with pytest.raises(SystemExit) as excinfo:
            main()

        assert excinfo.value.code == 1
        captured = capsys.readouterr()
        assert "Error: Failed to serialize results to JSON" in captured.err

    @patch("sys.argv", ["cli.py", "run", "config.yaml"])
    @patch("shedboxai.cli.validate_config_file")
    @patch("shedboxai.cli.Pipeline")
    def test_stdout_output_when_no_file(self, mock_pipeline_class, mock_validate_config):
        """Test that results are printed to stdout when no output file specified."""
        test_result = {
            "customer_data": {
                "customers": [{"id": 1}, {"id": 2}],
                "products": [{"id": "p1"}, {"id": "p2"}, {"id": "p3"}],
            },
            "customer_analysis": "This is an AI analysis.",
        }
        mock_validate_config.return_value = Path("config.yaml")
        mock_pipeline = Mock()
        mock_pipeline.run.return_value = test_result
        mock_pipeline_class.return_value = mock_pipeline

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            main()

        output = mock_stdout.getvalue()
        # Check for summary format rather than raw JSON
        assert "ShedBoxAI Pipeline Execution Completed" in output
        assert "Data Summary:" in output
        assert "customer_data" in output
        assert "2 customers" in output
        assert "3 products" in output
        assert "AI Analysis:" in output
        assert "This is an AI analysis." in output
        assert "Tip: Use the --output option" in output


class TestCLIIntegration:
    """Integration-style tests for CLI functionality."""

    @patch("sys.argv", ["cli.py", "run", "test_config.yaml", "-o", "results.json", "-v"])
    @patch("shedboxai.cli.validate_config_file")
    @patch("shedboxai.cli.validate_output_file")
    @patch("shedboxai.cli.Pipeline")
    @patch("builtins.open", new_callable=mock_open)
    @patch("shedboxai.cli.logging")
    def test_complete_successful_run(
        self,
        mock_logging,
        mock_file,
        mock_pipeline_class,
        mock_validate_output,
        mock_validate_config,
    ):
        """Test complete successful CLI run with all features."""
        complex_result = {
            "customer_data": {
                "customers": [{"id": 1}, {"id": 2}],
                "products": [{"id": "p1"}, {"id": "p2"}, {"id": "p3"}],
            },
            "customer_analysis": "This is an AI analysis.",
        }

        mock_validate_config.return_value = Path("test_config.yaml")
        mock_validate_output.return_value = Path("results.json")
        mock_pipeline = Mock()
        mock_pipeline.run.return_value = complex_result
        mock_pipeline_class.return_value = mock_pipeline

        # Set up handlers attribute on the mock logger
        mock_logger = Mock()
        mock_logger.handlers = []
        mock_logging.getLogger.return_value = mock_logger

        # Mock the sys.stdout to capture output
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            # Run the CLI
            main()

        # Verify all components worked
        mock_validate_config.assert_called_once_with("test_config.yaml")
        mock_validate_output.assert_called_once_with("results.json")
        mock_pipeline_class.assert_called_once_with("test_config.yaml")
        mock_pipeline.run.assert_called_once()
        mock_file.assert_called_once()

        # Verify output format - with verbose mode, the output format is different
        output = mock_stdout.getvalue()
        # In verbose mode, the old style output is used
        assert "Loading configuration from:" in output
        assert "Saving results to:" in output
        assert "Results successfully saved to:" in output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
