"""
Base classes for processing operations.

This module provides the abstract base class that all processing
operation handlers must implement.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict


class OperationHandler(ABC):
    """
    Abstract base class for all processing operation handlers.

    Each operation handler is responsible for:
    - Processing data according to its specific operation type
    - Handling its own error cases and edge conditions
    - Maintaining consistent input/output interfaces
    """

    def __init__(self, engine=None):
        """
        Initialize the operation handler.

        Args:
            engine: Optional expression engine for operations that need it
        """
        self.engine = engine
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

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
