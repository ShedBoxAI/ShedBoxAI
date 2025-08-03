"""
Exception hierarchy for ShedBoxAI.

This module provides a comprehensive set of exceptions used throughout the
ShedBoxAI framework to provide clear, actionable error messages for users.
"""

from typing import Optional


class ShedBoxAIError(Exception):
    """Base exception for ShedBoxAI framework."""

    def __init__(
        self,
        message: str,
        config_path: Optional[str] = None,
        suggestion: Optional[str] = None,
    ):
        self.config_path = config_path
        self.suggestion = suggestion
        self.message = message
        super().__init__(message)


# Configuration errors
class ConfigurationError(ShedBoxAIError):
    """Configuration-related errors."""

    pass


class InvalidSectionError(ConfigurationError):
    """Invalid or missing section in configuration."""

    pass


class InvalidFieldError(ConfigurationError):
    """Invalid or missing field in configuration section."""

    pass


class ValidationError(ConfigurationError):
    """Configuration validation errors."""

    pass


# Data source errors
class DataSourceError(ShedBoxAIError):
    """Data source-related errors."""

    pass


class FileAccessError(DataSourceError):
    """File access or permission errors."""

    pass


class NetworkError(DataSourceError):
    """Network connectivity issues."""

    pass


class AuthenticationError(DataSourceError):
    """Authentication or credential errors."""

    pass


class EnvironmentVariableError(DataSourceError):
    """Missing or invalid environment variable errors."""

    pass


# Processing errors
class ProcessingError(ShedBoxAIError):
    """Processing-related errors."""

    pass


class OperationError(ProcessingError):
    """Invalid operation or operation configuration."""

    pass


class DependencyError(ProcessingError):
    """Dependency resolution errors."""

    pass


class FieldReferenceError(ProcessingError):
    """Invalid field reference errors."""

    pass


class NormalizationError(ProcessingError):
    """Configuration normalization errors."""

    pass


# AI interface errors
class AIInterfaceError(ShedBoxAIError):
    """AI interface-related errors."""

    pass


class PromptError(AIInterfaceError):
    """Prompt configuration or rendering errors."""

    pass


class TemplateError(AIInterfaceError):
    """Template rendering errors."""

    pass


class ModelConfigError(AIInterfaceError):
    """Model configuration errors."""

    pass


class APIError(AIInterfaceError):
    """API communication errors."""

    pass


class RateLimitError(APIError):
    """API rate limit or quota errors."""

    pass


class ResponseParsingError(APIError):
    """API response parsing errors."""

    pass


# Pipeline errors
class PipelineError(ShedBoxAIError):
    """Pipeline-related errors."""

    pass


class OutputError(PipelineError):
    """Output handling errors."""

    pass


# Graph execution errors
class GraphExecutionError(ShedBoxAIError):
    """Graph execution-related errors."""

    def __init__(
        self,
        message: str,
        config_path: Optional[str] = None,
        suggestion: Optional[str] = None,
        example_yaml: Optional[str] = None,
    ):
        self.example_yaml = example_yaml
        super().__init__(message, config_path, suggestion)


class UnknownOperationError(GraphExecutionError):
    """Unknown operation referenced in graph configuration."""

    pass


class MissingDependencyError(GraphExecutionError):
    """Missing dependency referenced in graph configuration."""

    pass


class CyclicDependencyError(GraphExecutionError):
    """Cyclic dependency detected in graph configuration."""

    pass


class OperationExecutionError(GraphExecutionError):
    """Error during operation execution."""

    pass


class InvalidConfigurationError(GraphExecutionError):
    """Invalid configuration for operation."""

    pass
