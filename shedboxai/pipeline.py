"""
Pipeline orchestrator for ShedBoxAI.
Handles the entire flow of data through the system based on configuration.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd
import yaml
from pydantic import BaseModel, Field
from pydantic import ValidationError as PydanticValidationError

from .connector import DataSourceConnector
from .core.ai import AIInterface
from .core.exceptions import AIInterfaceError, ConfigurationError, InvalidSectionError, OutputError, PromptError
from .core.processor import DataProcessor
from .core.utils.error_formatting import extract_yaml_context, format_config_error, load_yaml_safely

# Valid output types and formats
VALID_OUTPUT_TYPES = {"file", "print"}
VALID_OUTPUT_FORMATS = {"json", "yaml"}


class PipelineConfig(BaseModel):
    """Configuration model for the entire pipeline."""

    data_sources: Dict[str, Any] = Field(..., description="Data source configurations")
    processing: Optional[Dict[str, Any]] = Field(None, description="Data processing configuration")
    output: Optional[Dict[str, Any]] = Field(None, description="Output configuration")
    ai_interface: Optional[Dict[str, Any]] = Field(None, description="AI interface configuration")


class Pipeline:
    """Main pipeline orchestrator for ShedBoxAI."""

    def __init__(self, config_path: Union[str, Path]):
        """
        Initialize the pipeline with a configuration file.

        Args:
            config_path: Path to the YAML/JSON configuration file

        Raises:
            ConfigurationError: If configuration file can't be read or parsed
            InvalidSectionError: If required configuration sections are missing
        """
        self.config_path = Path(config_path)
        self.raw_config = None
        self.config = self._load_config(self.config_path)
        self.connector = DataSourceConnector(self.config_path)
        self.processor = None  # Will be initialized when needed
        self.ai_interface = None  # Will be initialized when needed
        self.data = {}
        self.logger = logging.getLogger(__name__)

    def run(self) -> Any:
        """
        Run the entire pipeline.

        Returns:
            The final processed result

        Raises:
            Various exceptions depending on the pipeline stage that fails
        """
        # Step 1: Load all data sources
        self.logger.info("Loading data from configured sources...")
        self.data = self.connector.get_data(self.config.data_sources)

        # Step 2: Process data if configured
        if self.config.processing:
            self.logger.info("Processing data...")
            if not self.processor:
                self.processor = DataProcessor(self.config.processing)
            self.data = self.processor.process(self.data)

        # Step 3: Handle AI interface if configured
        if self.config.ai_interface:
            self.logger.info("Processing data through AI interface...")
            self.data = self._handle_ai_interface(self.data)

        # Step 4: Handle output if configured
        if self.config.output:
            self.logger.info("Handling output...")
            result = self._handle_output(self.data)
            return result

        return self.data

    def _handle_ai_interface(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process data through the AI interface.

        Args:
            data: Current pipeline data

        Returns:
            Updated data with AI processing results

        Raises:
            AIInterfaceError: If AI interface initialization fails
            PromptError: If prompt configuration is invalid
        """
        if not self.ai_interface:
            try:
                from .core.config.ai_config import AIInterfaceConfig

                self.ai_interface = AIInterface(AIInterfaceConfig(**self.config.ai_interface))
            except Exception as e:
                raise AIInterfaceError(
                    f"Failed to initialize AI interface: {str(e)}",
                    config_path="ai_interface",
                ) from e

        result = data.copy()

        # Check if prompts section exists
        if "prompts" not in self.config.ai_interface:
            raise PromptError(
                "Missing 'prompts' section in AI interface configuration",
                config_path="ai_interface",
                suggestion="Add a 'prompts' section with at least one prompt configuration",
            )

        # Process each configured prompt (with fan-out support)
        for prompt_name in self.config.ai_interface["prompts"]:
            try:
                # Use fan-out processing which handles both single and multiple prompts
                result[prompt_name] = self.ai_interface.process_prompt_with_fanout(
                    prompt_name,
                    data,  # Pass all data sources for fan-out
                    {},  # Empty base context (data sources will be added)
                )
            except KeyError as e:
                # Specific handling for missing prompt configuration
                if str(e).strip("'") == prompt_name:
                    example_yaml = (
                        "ai_interface:\n"
                        "  prompts:\n"
                        f"    {prompt_name}:  # ← Add this section\n"
                        '      system: "System prompt here"\n'
                        '      user_template: "User prompt template here"\n'
                        "      response_format: markdown"
                    )

                    error_msg = format_config_error(
                        "AI_INTERFACE",
                        "Referenced prompt not found",
                        f"ai_interface.prompts.{prompt_name}",
                        "All referenced prompts must be defined in the ai_interface.prompts section",
                        f"Add a '{prompt_name}' section to your prompts configuration",
                        example_yaml,
                    )
                    self.logger.error(error_msg)
                    result[prompt_name] = {"error": error_msg}
                else:
                    self.logger.error(f"Error processing AI prompt '{prompt_name}': {str(e)}")
                    result[prompt_name] = {"error": str(e)}
            except Exception as e:
                self.logger.error(f"Error processing AI prompt '{prompt_name}': {str(e)}")
                result[prompt_name] = {"error": str(e)}

        return result

    def _load_config(self, config_path: Path) -> PipelineConfig:
        """
        Load and validate the pipeline configuration.

        Args:
            config_path: Path to the configuration file

        Returns:
            Validated PipelineConfig object

        Raises:
            ConfigurationError: If configuration file can't be read or has invalid format
            InvalidSectionError: If required configuration sections are missing
        """
        # Check file extension
        valid_extensions = [".yaml", ".yml", ".json"]
        if config_path.suffix.lower() not in valid_extensions:
            example_yaml = (
                "# Example YAML configuration\n"
                "data_sources:\n"
                "  users:\n"
                "    type: csv\n"
                "    path: data/users.csv"
            )

            raise ConfigurationError(
                format_config_error(
                    "PIPELINE",
                    "Invalid configuration file format",
                    f"Configuration file '{config_path}'",
                    f"Configuration file must have one of these extensions: {', '.join(valid_extensions)}",
                    "Rename your file to have a .yaml extension, or convert it to YAML format",
                    example_yaml,
                )
            )

        # Read configuration file
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                file_content = f.read()

            # Parse YAML/JSON content
            if config_path.suffix.lower() in [".yaml", ".yml"]:
                config_dict = load_yaml_safely(file_content)
            else:  # JSON
                config_dict = json.loads(file_content)

            self.raw_config = config_dict
        except (FileNotFoundError, PermissionError) as e:
            raise ConfigurationError(
                format_config_error(
                    "PIPELINE",
                    "Configuration file access error",
                    f"Configuration file '{config_path}'",
                    "Configuration file must exist and be readable",
                    f"Check that the file exists and you have permission to read it: {str(e)}",
                )
            ) from e
        except json.JSONDecodeError as e:
            # Get line and column info from JSON error
            line_info = f" at line {e.lineno}, column {e.colno}"
            context = ""
            if file_content:
                context = "\n\n" + extract_yaml_context(file_content, e.lineno)

            raise ConfigurationError(
                format_config_error(
                    "PIPELINE",
                    "Invalid JSON in configuration file",
                    f"Configuration file '{config_path}'{line_info}",
                    "Configuration file must contain valid JSON",
                    f"Fix the JSON syntax error: {e.msg}{context}",
                )
            ) from e

        # Validate configuration structure
        if not isinstance(config_dict, dict):
            raise ConfigurationError(
                format_config_error(
                    "PIPELINE",
                    "Invalid configuration format",
                    f"Configuration file '{config_path}'",
                    "Configuration must be a YAML/JSON object (dictionary)",
                    "Ensure your configuration file contains a valid YAML/JSON object",
                )
            )

        # Check for required sections
        if "data_sources" not in config_dict:
            example_yaml = (
                "data_sources:  # ← Required section\n"
                "  users:\n"
                "    type: csv\n"
                "    path: data/users.csv\n"
                "\n"
                "# Optional sections\n"
                "processing:\n"
                "  contextual_filtering:\n"
                "    users:\n"
                "      - field: age\n"
                '        condition: "> 18"\n'
                "\n"
                "output:\n"
                "  type: file\n"
                "  path: output/result.json"
            )

            raise InvalidSectionError(
                format_config_error(
                    "PIPELINE",
                    "Missing required configuration section",
                    "Root level of configuration",
                    "Configuration must contain a 'data_sources' section",
                    "Add a 'data_sources' section to your configuration",
                    example_yaml,
                )
            )

        # Validate with Pydantic
        try:
            return PipelineConfig(**config_dict)
        except PydanticValidationError as e:
            # Extract field information from validation error
            field_errors = []
            for error in e.errors():
                location = ".".join(str(loc) for loc in error["loc"])
                field_errors.append(f"Field '{location}': {error['msg']}")

            fields_info = "\n".join(field_errors)

            raise ConfigurationError(
                format_config_error(
                    "PIPELINE",
                    "Configuration validation failed",
                    "Multiple fields",
                    "All fields must match their expected types and constraints",
                    f"Fix the following validation errors:\n{fields_info}",
                )
            ) from e

    def _handle_output(self, data: Any) -> Any:
        """
        Handle the output configuration.

        Args:
            data: Data to output

        Returns:
            Processed data

        Raises:
            OutputError: If output configuration is invalid
        """
        output_config = self.config.output
        output_type = output_config.get("type")

        if output_type not in VALID_OUTPUT_TYPES:
            valid_types_str = ", ".join(f"'{t}'" for t in VALID_OUTPUT_TYPES)
            example_yaml = (
                "output:\n" "  type: file  # ← Use 'file' or 'print'\n" "  path: output/result.json\n" "  format: json"
            )

            raise OutputError(
                format_config_error(
                    "PIPELINE",
                    "Invalid output type",
                    "output.type",
                    f"Output type must be one of: {valid_types_str}",
                    "Change your output type to a supported value",
                    example_yaml,
                )
            )

        if output_type == "file":
            return self._save_to_file(data)
        elif output_type == "print":
            print(data)
            return data

    def _save_to_file(self, data: Any) -> Any:
        """
        Save data to a file based on output configuration.

        Args:
            data: Data to save

        Returns:
            Processed data

        Raises:
            OutputError: If output file configuration is invalid or file can't be written
        """
        output_config = self.config.output

        # Check for required path field
        if "path" not in output_config:
            example_yaml = (
                "output:\n" "  type: file\n" "  path: output/result.json  # ← Add this field\n" "  format: json"
            )

            raise OutputError(
                format_config_error(
                    "PIPELINE",
                    "Missing output path",
                    "output.path",
                    "File output requires a 'path' field",
                    "Add a 'path' field to your output configuration",
                    example_yaml,
                )
            )

        output_path = Path(output_config.get("path"))

        # Create directory if it doesn't exist
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        except (PermissionError, OSError) as e:
            raise OutputError(
                format_config_error(
                    "PIPELINE",
                    "Output directory creation failed",
                    f"Directory: {output_path.parent}",
                    "Output directory must be writable",
                    f"Check directory permissions and path validity: {str(e)}",
                )
            ) from e

        output_format = output_config.get("format", "json")

        # Validate output format
        if output_format not in VALID_OUTPUT_FORMATS:
            valid_formats_str = ", ".join(f"'{f}'" for f in VALID_OUTPUT_FORMATS)
            example_yaml = (
                "output:\n"
                "  type: file\n"
                "  path: output/result.json\n"
                f"  format: json  # ← Use {valid_formats_str}"
            )

            raise OutputError(
                format_config_error(
                    "PIPELINE",
                    "Invalid output format",
                    "output.format",
                    f"Output format must be one of: {valid_formats_str}",
                    "Change your output format to a supported value",
                    example_yaml,
                )
            )

        # Convert data to serializable format
        try:
            if isinstance(data, dict):
                serializable_data = {
                    k: v.to_dict(orient="records") if isinstance(v, pd.DataFrame) else v for k, v in data.items()
                }
            else:
                serializable_data = data.to_dict(orient="records") if isinstance(data, pd.DataFrame) else data

            # Write to file based on format
            try:
                if output_format == "json":
                    output_path.write_text(json.dumps(serializable_data, indent=2))
                elif output_format == "yaml":
                    output_path.write_text(yaml.dump(serializable_data))
            except (PermissionError, OSError) as e:
                raise OutputError(
                    format_config_error(
                        "PIPELINE",
                        "Failed to write output file",
                        f"File: {output_path}",
                        "Output file must be writable",
                        f"Check file permissions and path validity: {str(e)}",
                    )
                ) from e

        except (TypeError, ValueError) as e:
            raise OutputError(
                format_config_error(
                    "PIPELINE",
                    "Data serialization failed",
                    "output",
                    "Data must be serializable to the chosen format",
                    f"Check data structure or change output format: {str(e)}",
                )
            ) from e

        return data
