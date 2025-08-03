"""
Refactored data processor with modular architecture.

This module provides the main DataProcessor class that orchestrates
all data processing operations using a clean, modular architecture.
"""

from typing import Any, Dict

from .config import ProcessorConfig
from .expression import ExpressionEngine
from .graph import GraphExecutor


class DataProcessor:
    """
    Main data processor with modular, clean architecture.

    This refactored processor delegates operations to specialized handlers
    and uses a graph executor for managing complex processing workflows.
    """

    def __init__(self, config: Dict[str, Any], ai_enabled: bool = False):
        """
        Initialize the data processor.

        Args:
            config: Processor configuration dictionary
            ai_enabled: Whether to enable AI features
        """
        self.config = ProcessorConfig(**config)
        self.engine = ExpressionEngine(ai_enabled=ai_enabled)
        self._ai_enabled = ai_enabled
        self.graph_executor = GraphExecutor(self.engine)

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process data using either graph-based or linear pipeline execution.

        Args:
            data: Input data dictionary with source data

        Returns:
            Processed data dictionary

        Raises:
            CyclicDependencyError: If processing graph contains cycles
            GraphExecutionError: If other graph execution errors occur
        """
        # Start with a copy of the input data
        result = data.copy()

        # Choose execution strategy based on configuration
        if hasattr(self.config, "graph") and self.config.graph:
            # Use graph-based execution for complex workflows
            result = self.graph_executor.execute_graph(result, self.config.graph, self.config)
        else:
            # Use linear pipeline execution for simple workflows
            result = self.graph_executor.execute_linear_pipeline(result, self.config)

        return result

    @property
    def ai_enabled(self) -> bool:
        """Check if AI features are enabled."""
        return self._ai_enabled

    @property
    def expression_engine(self) -> ExpressionEngine:
        """Get the expression engine instance."""
        return self.engine

    def register_custom_function(self, name: str, func) -> None:
        """
        Register a custom function with the expression engine.

        Args:
            name: Function name
            func: Function to register
        """
        self.engine.register_function(name, func)

    def register_custom_operator(self, symbol: str, func) -> None:
        """
        Register a custom operator with the expression engine.

        Args:
            symbol: Operator symbol
            func: Function to register
        """
        self.engine.register_operator(symbol, func)

    def validate_configuration(self) -> bool:
        """
        Validate the processor configuration.

        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Basic validation - check if config is valid Pydantic model
            return isinstance(self.config, ProcessorConfig)
        except Exception:
            return False

    def get_operation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of configured operations.

        Returns:
            Dictionary summarizing configured operations
        """
        summary = {
            "ai_enabled": self._ai_enabled,
            "execution_mode": ("graph" if (hasattr(self.config, "graph") and self.config.graph) else "linear"),
            "configured_operations": [],
            "graph_nodes": (len(self.config.graph) if (hasattr(self.config, "graph") and self.config.graph) else 0),
        }

        # Check which operations are configured
        operations = [
            "contextual_filtering",
            "format_conversion",
            "content_summarization",
            "relationship_highlighting",
            "advanced_operations",
            "template_matching",
        ]

        for op in operations:
            if getattr(self.config, op) is not None:
                summary["configured_operations"].append(op)

        return summary
