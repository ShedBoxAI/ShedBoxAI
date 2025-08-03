"""
Data Source Connector for ShedBoxAI.
Handles connecting to and retrieving data from various data sources.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic import ValidationError as PydanticValidationError

from .core.exceptions import (
    AuthenticationError,
    ConfigurationError,
    DataSourceError,
    EnvironmentVariableError,
    FileAccessError,
    InvalidFieldError,
    NetworkError,
)
from .core.utils.error_formatting import format_config_error, get_env_var_references

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Valid data source types
VALID_DATA_SOURCE_TYPES = {"csv", "json", "yaml", "rest", "text"}

# Required fields for each data source type
REQUIRED_FIELDS = {
    "csv": {"path"},
    "json": {"path"},
    "yaml": {"path"},
    "rest": {"url"},
    "text": {"path"},
}


class DataSourceConfig(BaseModel):
    """Configuration model for a data source."""

    type: str = Field(..., description="Type of data source (csv, json, yaml, rest, text)")
    path: Optional[str] = Field(None, description="Path to local file for csv/json sources")
    url: Optional[str] = Field(None, description="URL for REST API sources")
    method: str = Field("GET", description="HTTP method for REST API sources")
    headers: Optional[Dict[str, str]] = Field(default_factory=dict)
    options: Optional[Dict[str, Any]] = Field(default_factory=dict)
    data: Optional[Any] = Field(None, description="Direct data for csv, yaml, or json sources")
    response_path: Optional[str] = Field(None, description="Extract data from a given field in the response")
    # New fields for token handling
    is_token_source: bool = Field(False, description="Whether this endpoint provides a token")
    token_for: Optional[List[str]] = Field(None, description="List of data sources that should use this token")
    requires_token: bool = Field(False, description="Whether this endpoint requires a token")
    token_source: Optional[str] = Field(None, description="Name of the data source that provides the token")


class DataSourceConnector:
    """Connector for retrieving data from various sources."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the data source connector.

        Args:
            config_path: Path to the configuration file (used to locate .env file)
        """
        logger.debug(f"Current working directory: {os.getcwd()}")

        # Try to load .env from the config directory first
        if config_path:
            config_dir = os.path.dirname(os.path.abspath(config_path))
            env_path = os.path.join(config_dir, ".env")
            logger.debug(f"Looking for .env file in: {env_path}")
            if os.path.exists(env_path):
                logger.debug(f"Found .env file at: {env_path}")
                load_dotenv(env_path)
            else:
                logger.debug("No .env file found in config directory, trying current directory")
                load_dotenv()
        else:
            logger.debug("No config path provided, loading .env from current directory")
            load_dotenv()

        logger.debug("Environment variables loaded")
        self._token_cache = {}  # Cache for storing tokens
        self._raw_config = None  # Store raw configuration for error context

    def get_data(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieve data from configured sources.

        Args:
            config: Dictionary of data source configurations

        Returns:
            Dictionary containing data from all configured sources
        """
        self._raw_config = config
        result = {}

        # Check if config is empty
        if not config:
            raise ConfigurationError(
                format_config_error(
                    "DATA_SOURCE",
                    "Empty data sources configuration",
                    "data_sources",
                    "At least one data source must be defined",
                    "Add at least one data source to your configuration",
                    "data_sources:\n  users:\n    type: csv\n    path: data/users.csv",
                )
            )

        # First pass: collect all token sources and validate configurations
        token_sources = {}

        # Check for environment variables used in the configuration
        all_env_vars = get_env_var_references(config)
        missing_env_vars = []

        for var_name in all_env_vars:
            if not os.getenv(var_name):
                missing_env_vars.append(var_name)

        if missing_env_vars:
            paths = []
            for var in missing_env_vars:
                paths.extend(all_env_vars[var])

            paths_str = "\n".join(f"- {p}" for p in paths[:5])
            if len(paths) > 5:
                paths_str += f"\n- and {len(paths) - 5} more..."

            raise EnvironmentVariableError(
                format_config_error(
                    "DATA_SOURCE",
                    "Missing environment variables",
                    "data_sources (multiple locations)",
                    "All referenced environment variables must be defined",
                    f"Add the following environment variables to your .env file: {', '.join(missing_env_vars)}",
                    "# .env file example\n"
                    + "\n".join(f"{var}=your_{var.lower()}_value" for var in missing_env_vars[:5]),
                )
            )

        # First pass: validate all configurations and collect token sources
        for source_name, source_config in config.items():
            try:
                # Validate with Pydantic
                source = DataSourceConfig(**source_config)

                # Check source type is valid
                if source.type not in VALID_DATA_SOURCE_TYPES:
                    valid_types_str = ", ".join(f"'{t}'" for t in VALID_DATA_SOURCE_TYPES)
                    example_yaml = (
                        f"data_sources:\n"
                        f"  {source_name}:\n"
                        f"    type: csv  # ← Use one of: {valid_types_str}\n"
                        f"    path: data/{source_name}.csv"
                    )

                    raise DataSourceError(
                        format_config_error(
                            "DATA_SOURCE",
                            f"Invalid data source type: '{source.type}'",
                            f"data_sources.{source_name}.type",
                            f"Data source type must be one of: {valid_types_str}",
                            "Change the type to one of the supported values",
                            example_yaml,
                        )
                    )

                # Check required fields for this source type (skip if direct data is provided)
                if not source.data:
                    required_fields = REQUIRED_FIELDS.get(source.type, set())
                    for field in required_fields:
                        if not getattr(source, field):
                            example_field_value = (
                                f"data/{source_name}.{source.type}"
                                if field == "path"
                                else "https://api.example.com/data"
                            )
                            example_yaml = (
                                f"data_sources:\n"
                                f"  {source_name}:\n"
                                f"    type: {source.type}\n"
                                f"    {field}: {example_field_value}  # ← Add this field"
                            )

                            raise InvalidFieldError(
                                format_config_error(
                                    "DATA_SOURCE",
                                    f"Missing required field for {source.type} data source",
                                    f"data_sources.{source_name}.{field}",
                                    f"{source.type.upper()} data sources require a '{field}' field",
                                    f"Add a '{field}' field to your {source.type} data source configuration",
                                    example_yaml,
                                )
                            )

                # If this is a token source, collect it
                if source.is_token_source:
                    if not source.token_for:
                        example_yaml = (
                            f"data_sources:\n"
                            f"  {source_name}:\n"
                            f"    type: {source.type}\n"
                            f"    url: https://api.example.com/token\n"
                            f"    is_token_source: true\n"
                            f"    token_for:  # ← Add this field\n"
                            f"      - protected_endpoint"
                        )

                        raise InvalidFieldError(
                            format_config_error(
                                "DATA_SOURCE",
                                "Token source missing 'token_for' field",
                                f"data_sources.{source_name}.token_for",
                                "Token sources must specify which endpoints they provide tokens for",
                                "Add a 'token_for' list with endpoint names",
                                example_yaml,
                            )
                        )

                    token_sources[source_name] = source
                    logger.debug(f"Found token source: {source_name}")

                # If this source requires a token, validate token configuration
                if source.requires_token:
                    if not source.token_source:
                        example_yaml = (
                            f"data_sources:\n"
                            f"  auth_api:  # Define token source\n"
                            f"    type: rest\n"
                            f"    url: https://api.example.com/token\n"
                            f"    is_token_source: true\n"
                            f"    token_for:\n"
                            f"      - {source_name}\n\n"
                            f"  {source_name}:\n"
                            f"    type: {source.type}\n"
                            f"    url: https://api.example.com/data\n"
                            f"    requires_token: true\n"
                            f"    token_source: auth_api  # ← Add this field"
                        )

                        raise InvalidFieldError(
                            format_config_error(
                                "DATA_SOURCE",
                                "Missing token source reference",
                                f"data_sources.{source_name}.token_source",
                                "Sources that require a token must specify which source provides it",
                                ("Add a 'token_source' field " "with the name of the token source"),
                                example_yaml,
                            )
                        )

            except PydanticValidationError as e:
                # Format validation errors
                field_errors = []
                for error in e.errors():
                    location = ".".join(str(loc) for loc in error["loc"])
                    field_errors.append(f"- {location}: {error['msg']}")

                fields_info = "\n".join(field_errors)
                raise ConfigurationError(
                    format_config_error(
                        "DATA_SOURCE",
                        f"Invalid configuration for data source '{source_name}'",
                        f"data_sources.{source_name}",
                        "Data source configuration must be valid",
                        f"Fix the following validation errors:\n{fields_info}",
                    )
                )

        # Second pass: process all sources
        for source_name, source_config in config.items():
            try:
                logger.debug(f"Processing data source: {source_name}")
                source = DataSourceConfig(**source_config)

                # If this source requires a token, get it from the token source
                if source.requires_token and source.token_source:
                    self._handle_token_requirement(source, source_name, token_sources)

                data = self._fetch_data(source, source_name)
                # Only include non-token sources in the results
                if not source.is_token_source:
                    result[source_name] = data
            except (
                DataSourceError,
                FileAccessError,
                NetworkError,
                AuthenticationError,
                EnvironmentVariableError,
            ) as e:
                # Pass through our custom exceptions
                logger.error(f"Error fetching data from {source_name}: {str(e)}")
                if not source.is_token_source:
                    result[source_name] = {"error": str(e)}
            except Exception as e:
                # Wrap other exceptions with helpful messages
                logger.error(f"Error fetching data from {source_name}: {str(e)}")
                if not source.is_token_source:
                    result[source_name] = {"error": f"Failed to fetch data: {str(e)}"}

        return result

    def _handle_token_requirement(
        self,
        source: DataSourceConfig,
        source_name: str,
        token_sources: Dict[str, DataSourceConfig],
    ):
        """Handle token requirements for a source that needs authentication."""
        if source.token_source not in token_sources:
            example_yaml = (
                f"data_sources:\n"
                f"  {source.token_source}:  # ← Define this token source\n"
                f"    type: rest\n"
                f"    url: https://api.example.com/token\n"
                f"    method: POST\n"
                f"    options:\n"
                f"      json:\n"
                f"        username: ${{API_USERNAME}}\n"
                f"        password: ${{API_PASSWORD}}\n"
                f"    is_token_source: true\n"
                f"    token_for:\n"
                f"      - {source_name}\n\n"
                f"  {source_name}:\n"
                f"    type: {source.type}\n"
                f"    url: https://api.example.com/protected\n"
                f"    requires_token: true\n"
                f"    token_source: {source.token_source}  # ← Reference to the defined token source"
            )

            raise AuthenticationError(
                format_config_error(
                    "DATA_SOURCE",
                    "Referenced token source not found",
                    f"data_sources.{source_name}.token_source",
                    "Token source must be defined and configured as a token provider",
                    f"Define '{source.token_source}' as a token source or correct the reference name",
                    example_yaml,
                )
            )

        token_source = token_sources[source.token_source]
        if source_name not in token_source.token_for:
            example_yaml = (
                f"data_sources:\n"
                f"  {source.token_source}:\n"
                f"    type: rest\n"
                f"    url: https://api.example.com/token\n"
                f"    is_token_source: true\n"
                f"    token_for:\n"
                f"      - {source_name}  # ← Add this source to token_for list\n\n"
                f"  {source_name}:\n"
                f"    type: {source.type}\n"
                f"    url: https://api.example.com/protected\n"
                f"    requires_token: true\n"
                f"    token_source: {source.token_source}"
            )

            raise AuthenticationError(
                format_config_error(
                    "DATA_SOURCE",
                    "Token source not configured for this endpoint",
                    f"data_sources.{source.token_source}.token_for",
                    f"Token source must list '{source_name}' in its token_for configuration",
                    f"Add '{source_name}' to the token_for list in the token source configuration",
                    example_yaml,
                )
            )

        # Get token if not in cache
        if source.token_source not in self._token_cache:
            logger.debug(f"Getting token from {source.token_source}")
            try:
                token_data = self._fetch_data(token_source, source.token_source)
                if isinstance(token_data, dict) and "token" in token_data:
                    self._token_cache[source.token_source] = token_data["token"]
                    logger.debug("Token cached successfully")
                else:
                    raise AuthenticationError(
                        format_config_error(
                            "DATA_SOURCE",
                            "Invalid token response format",
                            f"Response from {source.token_source}",
                            "Token response must be a JSON object with a 'token' field",
                            "Check the token endpoint response format and adjust your configuration",
                            '# Expected token response format:\n{\n  "token": "your-auth-token"\n}',
                        )
                    )
            except Exception as e:
                raise AuthenticationError(
                    format_config_error(
                        "DATA_SOURCE",
                        "Failed to retrieve authentication token",
                        f"data_sources.{source.token_source}",
                        "Token endpoint must return a valid token response",
                        f"Check the token endpoint configuration and credentials: {str(e)}",
                    )
                ) from e

        # Add token to headers
        source.headers["Authorization"] = f"Bearer {self._token_cache[source.token_source]}"
        logger.debug("Added bearer token to headers")

    def _fetch_data(self, config: DataSourceConfig, source_name: str) -> Any:
        """
        Fetch data from a single configured source.

        Args:
            config: Data source configuration
            source_name: Name of the data source (for error messages)

        Returns:
            Data from the source

        Raises:
            Various exceptions depending on the data source type and error condition
        """
        logger.debug(f"Fetching data of type: {config.type}")

        # If data is provided directly in config, use it
        if hasattr(config, "data") and config.data is not None:
            if config.type == "csv":
                try:
                    # Ensure DataFrame has correct column order
                    df = pd.DataFrame(config.data)
                    if set(["name", "age", "city"]).issubset(df.columns):
                        return df[["name", "age", "city"]]  # Reorder columns
                    return df
                except Exception as e:
                    raise DataSourceError(
                        format_config_error(
                            "DATA_SOURCE",
                            "Invalid CSV data format",
                            f"data_sources.{source_name}.data",
                            "CSV data must be convertible to a pandas DataFrame",
                            f"Fix the data format: {str(e)}",
                        )
                    ) from e
            elif config.type == "yaml" or config.type == "json":
                return config.data
            elif config.type == "text":
                return str(config.data)
            else:
                example_yaml = (
                    f"data_sources:\n"
                    f"  {source_name}:\n"
                    f"    type: csv  # ← Direct data is only supported for csv, yaml, and json\n"
                    f"    data:\n"
                    f"      - name: John\n"
                    f"        age: 30\n"
                    f"        city: New York"
                )

                raise DataSourceError(
                    format_config_error(
                        "DATA_SOURCE",
                        f"Direct data not supported for type: {config.type}",
                        f"data_sources.{source_name}.data",
                        "Direct data is only supported for csv, yaml, json, and text source types",
                        "Change the source type or use path/url instead of direct data",
                        example_yaml,
                    )
                )

        # Otherwise, try to read from file or API
        if config.type == "csv":
            if not config.path and not config.data:
                raise InvalidFieldError(
                    format_config_error(
                        "DATA_SOURCE",
                        "Missing required field for CSV data source",
                        f"data_sources.{source_name}.path",
                        "CSV data sources require a 'path' field",
                        "Add a 'path' field to your CSV data source configuration",
                        (
                            f"data_sources:\n  {source_name}:\n    type: csv\n"
                            f"    path: data/{source_name}.csv  # ← Add this field"
                        ),
                    )
                )
            return self._fetch_csv(config, source_name)
        elif config.type == "json":
            if not config.path and not config.data:
                raise InvalidFieldError(
                    format_config_error(
                        "DATA_SOURCE",
                        "Missing required field for JSON data source",
                        f"data_sources.{source_name}.path",
                        "JSON data sources require a 'path' field",
                        "Add a 'path' field to your JSON data source configuration",
                        (
                            f"data_sources:\n  {source_name}:\n    type: json\n"
                            f"    path: data/{source_name}.json  # ← Add this field"
                        ),
                    )
                )
            return self._fetch_json(config, source_name)
        elif config.type == "yaml":
            if not config.path and not config.data:
                raise InvalidFieldError(
                    format_config_error(
                        "DATA_SOURCE",
                        "Missing required field for YAML data source",
                        f"data_sources.{source_name}.path",
                        "YAML data sources require a 'path' field",
                        "Add a 'path' field to your YAML data source configuration",
                        (
                            f"data_sources:\n  {source_name}:\n    type: yaml\n"
                            f"    path: data/{source_name}.yaml  # ← Add this field"
                        ),
                    )
                )
            return self._fetch_yaml(config, source_name)
        elif config.type == "text":
            if not config.path and not config.data:
                raise InvalidFieldError(
                    format_config_error(
                        "DATA_SOURCE",
                        "Missing required field for text data source",
                        f"data_sources.{source_name}.path",
                        "Text data sources require a 'path' field",
                        "Add a 'path' field to your text data source configuration",
                        (
                            f"data_sources:\n  {source_name}:\n    type: text\n"
                            f"    path: data/{source_name}.txt  # ← Add this field"
                        ),
                    )
                )
            return self._fetch_text(config, source_name)
        elif config.type == "rest":
            if not config.url:
                raise InvalidFieldError(
                    format_config_error(
                        "DATA_SOURCE",
                        "Missing required field for REST API data source",
                        f"data_sources.{source_name}.url",
                        "REST API data sources require a 'url' field",
                        "Add a 'url' field to your REST API data source configuration",
                        (
                            f"data_sources:\n  {source_name}:\n    type: rest\n"
                            f"    url: https://api.example.com/data  # ← Add this field"
                        ),
                    )
                )
            return self._fetch_rest(config, source_name)
        else:
            raise DataSourceError(
                format_config_error(
                    "DATA_SOURCE",
                    f"Unsupported data source type: {config.type}",
                    f"data_sources.{source_name}.type",
                    f"Data source type must be one of: {', '.join(VALID_DATA_SOURCE_TYPES)}",
                    "Change the type to one of the supported values",
                    (
                        f"data_sources:\n  {source_name}:\n    type: csv  "
                        f"# ← Use one of: {', '.join(VALID_DATA_SOURCE_TYPES)}\n"
                        f"    path: data/{source_name}.csv"
                    ),
                )
            )

    def _fetch_csv(self, config: DataSourceConfig, source_name: str) -> pd.DataFrame:
        """
        Fetch data from a CSV file.

        Args:
            config: Data source configuration
            source_name: Name of the data source (for error messages)

        Returns:
            Pandas DataFrame with CSV data

        Raises:
            FileAccessError: If file cannot be accessed
            DataSourceError: If CSV parsing fails
        """
        file_path = Path(config.path)
        options = config.options or {}

        try:
            return pd.read_csv(file_path, **options)
        except FileNotFoundError:
            raise FileAccessError(
                format_config_error(
                    "DATA_SOURCE",
                    "CSV file not found",
                    f"data_sources.{source_name}.path",
                    "CSV file must exist at the specified path",
                    f"Check that the file exists at: {file_path}",
                    f"data_sources:\n  {source_name}:\n    type: csv\n    path: data/existing_file.csv",
                )
            )
        except PermissionError:
            raise FileAccessError(
                format_config_error(
                    "DATA_SOURCE",
                    "Permission denied reading CSV file",
                    f"data_sources.{source_name}.path",
                    "CSV file must be readable",
                    f"Check file permissions for: {file_path}",
                )
            )
        except pd.errors.ParserError as e:
            raise DataSourceError(
                format_config_error(
                    "DATA_SOURCE",
                    "CSV parsing error",
                    f"data_sources.{source_name}.path",
                    "CSV file must be properly formatted",
                    f"Fix the CSV format: {str(e)}",
                )
            ) from e
        except Exception as e:
            raise DataSourceError(
                format_config_error(
                    "DATA_SOURCE",
                    "Error reading CSV file",
                    f"data_sources.{source_name}.path",
                    "CSV file must be readable and properly formatted",
                    f"Check the file and configuration: {str(e)}",
                )
            ) from e

    def _fetch_json(self, config: DataSourceConfig, source_name: str) -> Dict[str, Any]:
        """
        Fetch data from a JSON file.

        Args:
            config: Data source configuration
            source_name: Name of the data source (for error messages)

        Returns:
            Parsed JSON data

        Raises:
            FileAccessError: If file cannot be accessed
            DataSourceError: If JSON parsing fails
        """
        file_path = Path(config.path)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileAccessError(
                format_config_error(
                    "DATA_SOURCE",
                    "JSON file not found",
                    f"data_sources.{source_name}.path",
                    "JSON file must exist at the specified path",
                    f"Check that the file exists at: {file_path}",
                    f"data_sources:\n  {source_name}:\n    type: json\n    path: data/existing_file.json",
                )
            )
        except PermissionError:
            raise FileAccessError(
                format_config_error(
                    "DATA_SOURCE",
                    "Permission denied reading JSON file",
                    f"data_sources.{source_name}.path",
                    "JSON file must be readable",
                    f"Check file permissions for: {file_path}",
                )
            )
        except json.JSONDecodeError as e:
            raise DataSourceError(
                format_config_error(
                    "DATA_SOURCE",
                    "JSON parsing error",
                    f"data_sources.{source_name}.path",
                    "JSON file must contain valid JSON",
                    f"Fix the JSON syntax at line {e.lineno}, column {e.colno}: {e.msg}",
                )
            ) from e
        except Exception as e:
            raise DataSourceError(
                format_config_error(
                    "DATA_SOURCE",
                    "Error reading JSON file",
                    f"data_sources.{source_name}.path",
                    "JSON file must be readable and contain valid JSON",
                    f"Check the file and configuration: {str(e)}",
                )
            ) from e

    def _fetch_yaml(self, config: DataSourceConfig, source_name: str) -> Dict[str, Any]:
        """
        Fetch data from a YAML file.

        Args:
            config: Data source configuration
            source_name: Name of the data source (for error messages)

        Returns:
            Parsed YAML data

        Raises:
            FileAccessError: If file cannot be accessed
            DataSourceError: If YAML parsing fails
        """
        file_path = Path(config.path)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                return yaml.safe_load(content)
        except FileNotFoundError:
            raise FileAccessError(
                format_config_error(
                    "DATA_SOURCE",
                    "YAML file not found",
                    f"data_sources.{source_name}.path",
                    "YAML file must exist at the specified path",
                    f"Check that the file exists at: {file_path}",
                    f"data_sources:\n  {source_name}:\n    type: yaml\n    path: data/existing_file.yaml",
                )
            )
        except PermissionError:
            raise FileAccessError(
                format_config_error(
                    "DATA_SOURCE",
                    "Permission denied reading YAML file",
                    f"data_sources.{source_name}.path",
                    "YAML file must be readable",
                    f"Check file permissions for: {file_path}",
                )
            )
        except yaml.YAMLError as e:
            line = e.problem_mark.line + 1 if hasattr(e, "problem_mark") else None
            column = e.problem_mark.column + 1 if hasattr(e, "problem_mark") else None
            location = "unknown"
            if line:
                location = f"line {line}"
                if column:
                    location += f", column {column}"

            raise DataSourceError(
                format_config_error(
                    "DATA_SOURCE",
                    "YAML parsing error",
                    f"data_sources.{source_name}.path",
                    "YAML file must contain valid YAML",
                    f"Fix the YAML syntax at {location}: {str(e)}",
                )
            ) from e
        except Exception as e:
            raise DataSourceError(
                format_config_error(
                    "DATA_SOURCE",
                    "Error reading YAML file",
                    f"data_sources.{source_name}.path",
                    "YAML file must be readable and contain valid YAML",
                    f"Check the file and configuration: {str(e)}",
                )
            ) from e

    def _fetch_rest(self, config: DataSourceConfig, source_name: str) -> Dict[str, Any]:
        """
        Fetch data from a REST API.

        Args:
            config: Data source configuration
            source_name: Name of the data source (for error messages)

        Returns:
            Parsed JSON response

        Raises:
            NetworkError: If API connection fails
            AuthenticationError: If authentication fails
            EnvironmentVariableError: If environment variables are missing
            DataSourceError: For other API errors
        """
        try:
            # Process headers to replace environment variables
            headers = {}
            for key, value in config.headers.items():
                if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                    env_var = value[2:-1]
                    env_value = os.getenv(env_var)
                    if env_value:
                        headers[key] = env_value
                        logger.debug(f"Substituted environment variable {env_var} in header")
                    else:
                        raise EnvironmentVariableError(
                            format_config_error(
                                "DATA_SOURCE",
                                f"Missing environment variable: {env_var}",
                                f"data_sources.{source_name}.headers.{key}",
                                "Environment variables must be defined in the .env file",
                                f"Add {env_var}=your_value to your .env file",
                                f"# In .env file\n{env_var}=your_{env_var.lower()}_value",
                            )
                        )
                else:
                    headers[key] = value

            logger.debug(f"Final headers: {headers}")

            # Handle basic auth if specified in options
            auth = None
            if config.options and "auth" in config.options:
                auth_values = config.options["auth"]
                if not isinstance(auth_values, list) or len(auth_values) != 2:
                    example_yaml = (
                        f"data_sources:\n"
                        f"  {source_name}:\n"
                        f"    type: rest\n"
                        f"    url: {config.url}\n"
                        f"    options:\n"
                        f"      auth:  # ← Must be a list with exactly 2 values\n"
                        f"        - username\n"
                        f"        - password"
                    )

                    raise DataSourceError(
                        format_config_error(
                            "DATA_SOURCE",
                            "Invalid basic authentication configuration",
                            f"data_sources.{source_name}.options.auth",
                            "Basic authentication requires exactly 2 values (username and password)",
                            "Provide a list with username and password for basic auth",
                            example_yaml,
                        )
                    )

                # Substitute environment variables in auth values
                username = auth_values[0]
                password = auth_values[1]

                if isinstance(username, str) and username.startswith("${") and username.endswith("}"):
                    env_var = username[2:-1]
                    username = os.getenv(env_var)
                    if not username:
                        raise EnvironmentVariableError(
                            format_config_error(
                                "DATA_SOURCE",
                                f"Missing environment variable for username: {env_var}",
                                f"data_sources.{source_name}.options.auth[0]",
                                "Environment variable must be defined in the .env file",
                                f"Add {env_var}=your_username to your .env file",
                                f"# In .env file\n{env_var}=your_api_username",
                            )
                        )

                if isinstance(password, str) and password.startswith("${") and password.endswith("}"):
                    env_var = password[2:-1]
                    password = os.getenv(env_var)
                    if not password:
                        raise EnvironmentVariableError(
                            format_config_error(
                                "DATA_SOURCE",
                                f"Missing environment variable for password: {env_var}",
                                f"data_sources.{source_name}.options.auth[1]",
                                "Environment variable must be defined in the .env file",
                                f"Add {env_var}=your_password to your .env file",
                                f"# In .env file\n{env_var}=your_api_password",
                            )
                        )

                auth = (username, password)
                logger.debug(f"Using basic auth with username: {username}")

            # Handle bearer token if specified in headers
            if "Authorization" in headers:
                auth_header = headers["Authorization"]
                if auth_header.startswith("Bearer ${") and auth_header.endswith("}"):
                    # Extract variable name: "Bearer ${API_TOKEN}" -> "API_TOKEN"
                    token_var = auth_header[9:-1]  # Remove 'Bearer ${' (9 chars) and '}' (1 char)
                    token_value = os.getenv(token_var)
                    if token_value:
                        headers["Authorization"] = f"Bearer {token_value}"
                        logger.debug("Bearer token substituted in Authorization header")
                    else:
                        raise EnvironmentVariableError(
                            format_config_error(
                                "DATA_SOURCE",
                                f"Missing environment variable for token: {token_var}",
                                f"data_sources.{source_name}.headers.Authorization",
                                "Environment variable must be defined in the .env file",
                                f"Add {token_var}=your_token to your .env file",
                                f"# In .env file\n{token_var}=your_api_token",
                            )
                        )

            # Process options to replace environment variables in JSON body
            processed_options = {}
            for key, value in config.options.items():
                if key == "auth":  # Skip auth as it's handled separately
                    continue
                if key == "json" and isinstance(value, dict):
                    processed_json = {}
                    for json_key, json_value in value.items():
                        if isinstance(json_value, str) and json_value.startswith("${") and json_value.endswith("}"):
                            env_var = json_value[2:-1]
                            env_value = os.getenv(env_var)
                            if env_value:
                                processed_json[json_key] = env_value
                                logger.debug(f"Substituted environment variable {env_var} in JSON body")
                            else:
                                raise EnvironmentVariableError(
                                    format_config_error(
                                        "DATA_SOURCE",
                                        f"Missing environment variable for JSON body: {env_var}",
                                        f"data_sources.{source_name}.options.json.{json_key}",
                                        "Environment variable must be defined in the .env file",
                                        f"Add {env_var}=your_value to your .env file",
                                        f"# In .env file\n{env_var}=your_{env_var.lower()}_value",
                                    )
                                )
                        else:
                            processed_json[json_key] = json_value
                    processed_options[key] = processed_json
                else:
                    processed_options[key] = value

            # Make the API request
            try:
                response = requests.request(
                    method=config.method,
                    url=config.url,
                    headers=headers,
                    auth=auth,
                    **processed_options,
                )
                response.raise_for_status()
            except requests.exceptions.ConnectionError as e:
                raise NetworkError(
                    format_config_error(
                        "DATA_SOURCE",
                        "API connection error",
                        f"data_sources.{source_name}.url",
                        "API endpoint must be reachable",
                        f"Check network connectivity and URL: {config.url}",
                        (
                            f"data_sources:\n  {source_name}:\n    type: rest\n"
                            f"    url: https://api.example.com/data  # ← Check this URL"
                        ),
                    )
                ) from e
            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code

                # Handle specific HTTP status codes
                if status_code == 401:
                    raise AuthenticationError(
                        format_config_error(
                            "DATA_SOURCE",
                            "API authentication failed",
                            f"data_sources.{source_name}",
                            "API credentials must be valid",
                            "Check your API credentials and authentication configuration",
                            (
                                "# Common authentication configurations:\n\n"
                                "# Option 1: Bearer token\nheaders:\n  Authorization: Bearer ${API_TOKEN}\n\n"
                                "# Option 2: Basic auth\noptions:\n  auth:\n"
                                "    - ${API_USERNAME}\n    - ${API_PASSWORD}"
                            ),
                        )
                    ) from e
                elif status_code == 403:
                    raise AuthenticationError(
                        format_config_error(
                            "DATA_SOURCE",
                            "API access forbidden",
                            f"data_sources.{source_name}",
                            "API credentials must have permission to access the endpoint",
                            "Check that your API credentials have the correct permissions",
                        )
                    ) from e
                elif status_code == 404:
                    raise NetworkError(
                        format_config_error(
                            "DATA_SOURCE",
                            "API endpoint not found",
                            f"data_sources.{source_name}.url",
                            "API endpoint URL must be valid",
                            f"Check that the URL is correct: {config.url}",
                        )
                    ) from e
                elif status_code == 429:
                    raise NetworkError(
                        format_config_error(
                            "DATA_SOURCE",
                            "API rate limit exceeded",
                            f"data_sources.{source_name}",
                            "API request rate must be within limits",
                            "Reduce request frequency or implement rate limiting",
                        )
                    ) from e
                else:
                    raise DataSourceError(
                        format_config_error(
                            "DATA_SOURCE",
                            f"API request failed with status {status_code}",
                            f"data_sources.{source_name}",
                            "API request must succeed",
                            f"Check API configuration and parameters: {e.response.text[:200]}",
                        )
                    ) from e
            except requests.exceptions.Timeout as e:
                raise NetworkError(
                    format_config_error(
                        "DATA_SOURCE",
                        "API request timed out",
                        f"data_sources.{source_name}.url",
                        "API must respond within the timeout period",
                        "Check API endpoint performance or increase timeout in options",
                    )
                ) from e
            except requests.exceptions.RequestException as e:
                raise NetworkError(
                    format_config_error(
                        "DATA_SOURCE",
                        "API request failed",
                        f"data_sources.{source_name}",
                        "API request must succeed",
                        f"Check API configuration: {str(e)}",
                    )
                ) from e

            # Parse the response
            try:
                json_data = response.json()
            except json.JSONDecodeError as e:
                raise DataSourceError(
                    format_config_error(
                        "DATA_SOURCE",
                        "Invalid JSON response from API",
                        f"data_sources.{source_name}",
                        "API response must be valid JSON",
                        f"Check API endpoint response format: {str(e)}",
                    )
                ) from e

            # Extract nested data if response_path is specified
            if config.response_path:
                try:
                    path_parts = config.response_path.split(".")
                    current_data = json_data

                    for part in path_parts:
                        if isinstance(current_data, dict) and part in current_data:
                            current_data = current_data[part]
                        else:
                            # If part is not found, log a warning and break the extraction
                            missing_parts = []
                            for test_part in path_parts:
                                if isinstance(current_data, dict) and test_part in current_data:
                                    current_data = current_data[test_part]
                                else:
                                    missing_parts.append(test_part)
                                    break

                            available_keys = []
                            if isinstance(current_data, dict):
                                available_keys = list(current_data.keys())

                            keys_str = ", ".join(f"'{k}'" for k in available_keys[:5])
                            if len(available_keys) > 5:
                                keys_str += f", and {len(available_keys) - 5} more"

                            raise DataSourceError(
                                format_config_error(
                                    "DATA_SOURCE",
                                    "Response path not found in API response",
                                    f"data_sources.{source_name}.response_path",
                                    f"Response path '{config.response_path}' must exist in the API response",
                                    f"Path part '{part}' not found. Available keys: {keys_str if keys_str else 'none'}",
                                )
                            )

                    return current_data
                except Exception as e:
                    if not isinstance(e, DataSourceError):
                        raise DataSourceError(
                            format_config_error(
                                "DATA_SOURCE",
                                "Error extracting data from response path",
                                f"data_sources.{source_name}.response_path",
                                "Response path must be valid for the API response structure",
                                f"Check the response path configuration: {str(e)}",
                            )
                        ) from e
                    raise

            return json_data

        except (
            NetworkError,
            AuthenticationError,
            EnvironmentVariableError,
            DataSourceError,
        ):
            # Pass through our custom exceptions
            raise
        except Exception as e:
            raise DataSourceError(
                format_config_error(
                    "DATA_SOURCE",
                    "Error fetching data from REST API",
                    f"data_sources.{source_name}",
                    "REST API request must succeed",
                    f"Check the API configuration: {str(e)}",
                )
            ) from e

    def _fetch_text(self, config: DataSourceConfig, source_name: str) -> str:
        """
        Fetch raw text content from a file.

        Args:
            config: Data source configuration
            source_name: Name of the data source (for error messages)

        Returns:
            Raw text content as string

        Raises:
            FileNotFoundError: If the file doesn't exist
            PermissionError: If the file cannot be read
            DataSourceError: For other file reading errors
        """
        try:
            path = Path(config.path)

            # Check if file exists
            if not path.exists():
                raise FileNotFoundError(
                    format_config_error(
                        "DATA_SOURCE",
                        "Text file not found",
                        f"data_sources.{source_name}.path",
                        "Text file must exist and be readable",
                        f"Check that the file exists: {config.path}",
                        (
                            f"data_sources:\n  {source_name}:\n    type: text\n"
                            f"    path: {config.path}  # ← Check this path"
                        ),
                    )
                )

            # Try to read file with UTF-8 encoding first
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                logger.debug(f"Successfully loaded text file: {config.path} ({len(content)} characters)")
                return content

            except UnicodeDecodeError:
                # If UTF-8 fails, try other common encodings
                encodings = ["latin-1", "cp1252", "iso-8859-1"]

                for encoding in encodings:
                    try:
                        with open(path, "r", encoding=encoding) as f:
                            content = f.read()
                        logger.debug(f"Successfully loaded text file with {encoding} encoding: {config.path}")
                        return content
                    except UnicodeDecodeError:
                        continue

                # If all encodings fail, read as binary and decode with errors='replace'
                try:
                    with open(path, "rb") as f:
                        raw_content = f.read()
                    content = raw_content.decode("utf-8", errors="replace")
                    logger.warning(f"Text file contains non-UTF-8 characters, some may be replaced: {config.path}")
                    return content
                except Exception as e:
                    raise DataSourceError(
                        format_config_error(
                            "DATA_SOURCE",
                            "Text file encoding error",
                            f"data_sources.{source_name}.path",
                            "Text file must be readable with standard encodings",
                            f"Check file encoding or convert to UTF-8: {str(e)}",
                        )
                    ) from e

        except FileNotFoundError:
            # Pass through our formatted FileNotFoundError
            raise
        except PermissionError as e:
            raise PermissionError(
                format_config_error(
                    "DATA_SOURCE",
                    "Permission denied reading text file",
                    f"data_sources.{source_name}.path",
                    "Text file must have read permissions",
                    f"Check file permissions: {config.path}",
                )
            ) from e
        except Exception as e:
            raise DataSourceError(
                format_config_error(
                    "DATA_SOURCE",
                    "Error reading text file",
                    f"data_sources.{source_name}.path",
                    "Text file must be readable",
                    f"Check the file and configuration: {str(e)}",
                )
            ) from e
