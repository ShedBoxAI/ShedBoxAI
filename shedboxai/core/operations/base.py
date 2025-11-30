"""
Base classes for processing operations.

This module provides the abstract base class that all processing
operation handlers must implement.
"""

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict


class Severity(str, Enum):
    """Severity levels for processing issues."""

    ERROR = "error"  # Critical failure - operation failed completely
    WARNING = "warning"  # Non-critical issue - operation completed but with issues


class ProcessingIssue:
    """Represents an issue (error or warning) that occurred during processing."""

    def __init__(
        self,
        source: str,
        operation: str,
        message: str,
        severity: Severity = Severity.ERROR,
        field: str = None,
        expression: str = None,
    ):
        """
        Initialize a processing issue.

        Args:
            source: The data source where the issue occurred
            operation: The operation that was being performed
            message: The issue message
            severity: Severity level (ERROR or WARNING)
            field: Optional field name related to the issue
            expression: Optional expression that caused the issue
        """
        self.source = source
        self.operation = operation
        self.message = message
        self.severity = severity
        self.field = field
        self.expression = expression

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "severity": self.severity.value,
            "source": self.source,
            "operation": self.operation,
            "message": self.message,
        }
        if self.field:
            result["field"] = self.field
        if self.expression:
            result["expression"] = self.expression
        return result


# Backward compatibility alias
ProcessingError = ProcessingIssue


class OperationHandler(ABC):
    """
    Abstract base class for all processing operation handlers.

    Each operation handler is responsible for:
    - Processing data according to its specific operation type
    - Handling its own error cases and edge conditions
    - Maintaining consistent input/output interfaces
    - Collecting errors that occur during processing
    """

    def __init__(self, engine=None):
        """
        Initialize the operation handler.

        Args:
            engine: Optional expression engine for operations that need it
        """
        self.engine = engine
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._processing_issues: list = []

    @abstractmethod
    def process(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process data according to the operation configuration.

        Args:
            data: Input data dictionary with source data
            config: Normalized configuration for this operation

        Returns:
            Modified data dictionary with operation results

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement process()")

    @property
    @abstractmethod
    def operation_name(self) -> str:
        """
        Get the name of this operation.

        Returns:
            Operation name string
        """
        raise NotImplementedError("Subclasses must implement operation_name")

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate the configuration for this operation.

        Args:
            config: Configuration to validate

        Returns:
            True if configuration is valid, False otherwise
        """
        # Default implementation - subclasses can override for specific validation
        return config is not None and isinstance(config, dict)

    def _log_debug(self, message: str) -> None:
        """
        Log a debug message.

        Args:
            message: Debug message to log
        """
        self.logger.debug(message)

    def _log_info(self, message: str) -> None:
        """
        Log an info message.

        Args:
            message: Info message to log
        """
        self.logger.info(message)

    def _log_warning(self, message: str) -> None:
        """
        Log a warning message.

        Args:
            message: Warning message to log
        """
        self.logger.warning(message)

    def _log_error(self, message: str) -> None:
        """
        Log an error message.

        Args:
            message: Error message to log
        """
        self.logger.error(message)

    def _collect_issue(
        self,
        source: str,
        message: str,
        severity: Severity = Severity.ERROR,
        field: str = None,
        expression: str = None,
    ) -> None:
        """
        Collect a processing issue for later retrieval.

        Args:
            source: The data source where the issue occurred
            message: The issue message
            severity: Severity level (ERROR or WARNING)
            field: Optional field name related to the issue
            expression: Optional expression that caused the issue
        """
        issue = ProcessingIssue(
            source=source,
            operation=self.operation_name,
            message=message,
            severity=severity,
            field=field,
            expression=expression,
        )
        self._processing_issues.append(issue)

    def _collect_error(self, source: str, message: str, field: str = None, expression: str = None) -> None:
        """
        Collect a processing error (severity=ERROR) for later retrieval.

        Args:
            source: The data source where the error occurred
            message: The error message
            field: Optional field name related to the error
            expression: Optional expression that caused the error
        """
        self._collect_issue(source, message, Severity.ERROR, field, expression)

    def _collect_warning(self, source: str, message: str, field: str = None, expression: str = None) -> None:
        """
        Collect a processing warning (severity=WARNING) for later retrieval.

        Args:
            source: The data source where the warning occurred
            message: The warning message
            field: Optional field name related to the warning
            expression: Optional expression that caused the warning
        """
        self._collect_issue(source, message, Severity.WARNING, field, expression)

    def get_issues(self, severity: Severity = None) -> list:
        """
        Get collected processing issues, optionally filtered by severity.

        Args:
            severity: Optional severity filter (ERROR or WARNING)

        Returns:
            List of ProcessingIssue objects
        """
        if severity is None:
            return self._processing_issues
        return [i for i in self._processing_issues if i.severity == severity]

    def get_errors(self) -> list:
        """
        Get all collected processing errors (severity=ERROR).

        Returns:
            List of ProcessingIssue objects with ERROR severity
        """
        return self.get_issues(Severity.ERROR)

    def get_warnings(self) -> list:
        """
        Get all collected processing warnings (severity=WARNING).

        Returns:
            List of ProcessingIssue objects with WARNING severity
        """
        return self.get_issues(Severity.WARNING)

    def get_issues_as_dicts(self, severity: Severity = None) -> list:
        """
        Get collected processing issues as dictionaries.

        Args:
            severity: Optional severity filter (ERROR or WARNING)

        Returns:
            List of issue dictionaries suitable for JSON serialization
        """
        return [issue.to_dict() for issue in self.get_issues(severity)]

    def get_errors_as_dicts(self) -> list:
        """
        Get all collected processing errors as dictionaries.

        Returns:
            List of error dictionaries suitable for JSON serialization
        """
        return self.get_issues_as_dicts(Severity.ERROR)

    def get_warnings_as_dicts(self) -> list:
        """
        Get all collected processing warnings as dictionaries.

        Returns:
            List of warning dictionaries suitable for JSON serialization
        """
        return self.get_issues_as_dicts(Severity.WARNING)

    def clear_issues(self) -> None:
        """Clear all collected processing issues."""
        self._processing_issues = []

    def clear_errors(self) -> None:
        """Clear all collected processing issues (alias for clear_issues)."""
        self.clear_issues()

    def has_issues(self, severity: Severity = None) -> bool:
        """
        Check if any issues were collected, optionally filtered by severity.

        Args:
            severity: Optional severity filter (ERROR or WARNING)

        Returns:
            True if issues were collected, False otherwise
        """
        return len(self.get_issues(severity)) > 0

    def has_errors(self) -> bool:
        """
        Check if any errors were collected.

        Returns:
            True if errors were collected, False otherwise
        """
        return self.has_issues(Severity.ERROR)

    def has_warnings(self) -> bool:
        """
        Check if any warnings were collected.

        Returns:
            True if warnings were collected, False otherwise
        """
        return self.has_issues(Severity.WARNING)
