"""
Graph execution engine for processing pipelines.

This module provides the GraphExecutor class that manages
the execution of processing operations in dependency order
using directed acyclic graphs.
"""

from typing import Any, Dict, List

import networkx as nx

from ..config.models import GraphNode
from ..exceptions import (
    CyclicDependencyError,
    InvalidConfigurationError,
    MissingDependencyError,
    OperationExecutionError,
    UnknownOperationError,
)
from ..operations import (
    AdvancedOperationsHandler,
    ContentSummarizationHandler,
    ContextualFilteringHandler,
    FormatConversionHandler,
    RelationshipHighlightingHandler,
    TemplateMatchingHandler,
)


class GraphExecutor:
    """
    Executes processing operations in dependency order using a directed graph.

    Supports both linear pipeline execution and complex dependency graphs
    with topological sorting for correct execution order.
    """

    def __init__(self, engine=None):
        """
        Initialize the graph executor.

        Args:
            engine: Optional expression engine to pass to handlers
        """
        self.engine = engine

        # Map of operation names to handler classes
        self.operation_handlers = {
            "contextual_filtering": ContextualFilteringHandler,
            "format_conversion": FormatConversionHandler,
            "content_summarization": ContentSummarizationHandler,
            "relationship_highlighting": RelationshipHighlightingHandler,
            "advanced_operations": AdvancedOperationsHandler,
            "template_matching": TemplateMatchingHandler,
        }

    def execute_graph(
        self, data: Dict[str, Any], graph_nodes: List[GraphNode], processor_config: Any
    ) -> Dict[str, Any]:
        """
        Execute a processing graph.

        Args:
            data: Input data dictionary
            graph_nodes: List of graph nodes defining the processing steps
            processor_config: Processor configuration containing operation configs

        Returns:
            Processed data dictionary

        Raises:
            UnknownOperationError: If an operation is not recognized
            MissingDependencyError: If a dependency is missing
            CyclicDependencyError: If the graph contains cycles
            InvalidConfigurationError: If operation configuration is invalid
            OperationExecutionError: If an operation fails during execution
        """
        # Create directed graph
        graph = nx.DiGraph()
        nodes = {}

        # Add nodes and build dependency graph
        for node in graph_nodes:
            if node.operation not in self.operation_handlers:
                example_yaml = """
graph:
  - id: filter_node
    operation: contextual_filtering  # Valid operations include: contextual_filtering, format_conversion, etc.
    config_key: primary_filter
                """
                raise UnknownOperationError(
                    f"Unknown operation '{node.operation}' in graph node '{node.id}'",
                    suggestion=f"Use one of the supported operations: {', '.join(self.operation_handlers.keys())}",
                    example_yaml=example_yaml,
                )

            graph.add_node(node.id)
            nodes[node.id] = {
                "operation": node.operation,
                "config_key": node.config_key,
                "handler_class": self.operation_handlers[node.operation],
            }

            # Add dependencies
            node_ids = [n.id for n in graph_nodes]
            for dependency in node.depends_on:
                if dependency not in node_ids:
                    example_yaml = """
graph:
  - id: filter_node
    operation: contextual_filtering

  - id: summarize_node
    operation: content_summarization
    depends_on:
      - filter_node  # Must reference an existing node id
                    """
                    raise MissingDependencyError(
                        f"Dependency '{dependency}' referenced by node '{node.id}' does not exist",
                        suggestion=f"Ensure all dependencies refer to valid node IDs: {', '.join(node_ids)}",
                        example_yaml=example_yaml,
                    )
                graph.add_edge(dependency, node.id)

        # Check for cycles
        if not nx.is_directed_acyclic_graph(graph):
            cycles = list(nx.simple_cycles(graph))
            cycle_str = ", ".join([" → ".join(cycle) for cycle in cycles])
            example_yaml = """
# Good graph (no cycles)
graph:
  - id: filter_node
    operation: contextual_filtering

  - id: summarize_node
    operation: content_summarization
    depends_on:
      - filter_node
            """
            raise CyclicDependencyError(
                f"The processing graph contains cycles: {cycle_str}",
                suggestion="Ensure the graph is acyclic by removing circular dependencies",
                example_yaml=example_yaml,
            )

        # Determine execution order
        execution_order = list(nx.topological_sort(graph))

        # Execute nodes in order
        result = data.copy()

        for node_id in execution_order:
            node_info = nodes[node_id]
            operation = node_info["operation"]
            config_key = node_info["config_key"]
            handler_class = node_info["handler_class"]

            # Get the appropriate config for this node
            operation_config = getattr(processor_config, operation, None)
            if operation_config is None:
                continue

            # Extract specific configuration if config_key is specified
            if config_key and isinstance(operation_config, dict):
                if config_key in operation_config:
                    specific_config = operation_config[config_key]
                    if not specific_config:
                        error_msg = f"No configuration found for node '{node_id}' with config_key '{config_key}'"
                        example_yaml = f"""
{operation}:
  {config_key}:  # This section is empty or missing
    # Add your configuration here
                        """
                        raise InvalidConfigurationError(
                            error_msg,
                            config_path=f"{operation}.{config_key}",
                            suggestion=(
                                f"Provide a valid configuration for '{config_key}' under " f"the '{operation}' section"
                            ),
                            example_yaml=example_yaml,
                        )
                else:
                    error_msg = f"Config key '{config_key}' not found in '{operation}' configuration"
                    example_yaml = f"""
{operation}:
  {config_key}:  # Add this section
    # Your configuration here
                    """
                    raise InvalidConfigurationError(
                        error_msg,
                        config_path=operation,
                        suggestion=f"Add a '{config_key}' section to the '{operation}' configuration",
                        example_yaml=example_yaml,
                    )
            else:
                specific_config = operation_config

            # Create handler and process
            handler = handler_class(self.engine)

            # Get the appropriate normalization function
            try:
                normalized_config = self._normalize_config(operation, specific_config)
            except Exception as e:
                raise InvalidConfigurationError(
                    f"Failed to normalize configuration for operation '{operation}' in node '{node_id}': {str(e)}",
                    config_path=f"{operation}{f'.{config_key}' if config_key else ''}",
                    suggestion="Check that your configuration follows the required format for this operation",
                ) from e

            # Process the data
            try:
                result = handler.process(result, normalized_config)
            except Exception as e:
                raise OperationExecutionError(
                    f"Error executing operation '{operation}' in node '{node_id}': {str(e)}",
                    suggestion="Check your input data and operation configuration for compatibility issues",
                ) from e

        return result

    def execute_linear_pipeline(self, data: Dict[str, Any], processor_config: Any) -> Dict[str, Any]:
        """
        Execute a linear processing pipeline.

        Args:
            data: Input data dictionary
            processor_config: Processor configuration

        Returns:
            Processed data dictionary

        Raises:
            InvalidConfigurationError: If operation configuration is invalid
            OperationExecutionError: If an operation fails during execution
        """
        import logging

        result = data.copy()
        logger = logging.getLogger(__name__)

        # Default linear execution order
        default_order = [
            "contextual_filtering",
            "format_conversion",
            "content_summarization",
            "relationship_highlighting",
            "advanced_operations",
            "template_matching",
        ]

        # Count configured operations
        configured_ops = [op for op in default_order if getattr(processor_config, op, None)]

        if configured_ops:
            logger.info("=" * 60)
            logger.info(f"PROCESSING PIPELINE ({len(configured_ops)} operations)")
            logger.info("=" * 60)

        stage_num = 0
        for operation in default_order:
            config = getattr(processor_config, operation, None)
            if config is None:
                continue

            handler_class = self.operation_handlers.get(operation)
            if not handler_class:
                continue

            stage_num += 1

            # Log stage start
            logger.info("")
            logger.info(f"Stage {stage_num}/{len(configured_ops)}: {operation}")
            logger.info("-" * 60)

            # Track what exists before processing
            before_keys = set(result.keys())

            # Create handler and process
            handler = handler_class(self.engine)

            # Normalize configuration with error handling
            try:
                normalized_config = self._normalize_config(operation, config)
            except Exception as e:
                raise InvalidConfigurationError(
                    f"Failed to normalize configuration for operation '{operation}': {str(e)}",
                    config_path=operation,
                    suggestion="Check that your configuration follows the required format for this operation",
                ) from e

            # Process with error handling
            try:
                result = handler.process(result, normalized_config)
            except Exception as e:
                raise OperationExecutionError(
                    f"Error executing operation '{operation}': {str(e)}",
                    suggestion="Check your input data and operation configuration for compatibility issues",
                ) from e

            # Track what was created
            after_keys = set(result.keys())
            new_keys = after_keys - before_keys

            if new_keys:
                for key in sorted(new_keys):
                    if isinstance(result[key], list):
                        logger.info(f"  → Created '{key}': {len(result[key])} records")
                    elif isinstance(result[key], dict):
                        logger.info(f"  → Created '{key}': dict with {len(result[key])} keys")
                    else:
                        logger.info(f"  → Created '{key}': {type(result[key]).__name__}")
            else:
                logger.debug("  No new variables created")

        if configured_ops:
            logger.info("")
            logger.info("=" * 60)
            logger.info("PROCESSING COMPLETE")
            logger.info("=" * 60)

        return result

    def _normalize_config(self, operation: str, config: Any) -> Dict[str, Any]:
        """
        Normalize configuration for an operation.

        Args:
            operation: Operation name
            config: Raw configuration

        Returns:
            Normalized configuration dictionary

        Raises:
            InvalidConfigurationError: If the configuration cannot be normalized
        """
        # Import normalization functions
        from ..config.normalizers import (
            normalize_advanced_operations_config,
            normalize_content_summarization_config,
            normalize_contextual_filtering_config,
            normalize_format_conversion_config,
            normalize_relationship_highlighting_config,
            normalize_template_matching_config,
        )

        normalization_functions = {
            "contextual_filtering": normalize_contextual_filtering_config,
            "format_conversion": normalize_format_conversion_config,
            "content_summarization": normalize_content_summarization_config,
            "relationship_highlighting": normalize_relationship_highlighting_config,
            "advanced_operations": normalize_advanced_operations_config,
            "template_matching": normalize_template_matching_config,
        }

        normalizer = normalization_functions.get(operation)
        if normalizer is None:
            # This should never happen as we check earlier if the operation is supported
            raise InvalidConfigurationError(
                f"No normalization function found for operation '{operation}'",
                suggestion=f"Use one of the supported operations: {', '.join(normalization_functions.keys())}",
            )

        if not config:
            return {}

        try:
            return normalizer(config)
        except Exception as e:
            example_yaml = ""
            if operation == "contextual_filtering":
                example_yaml = """
contextual_filtering:
  patterns:
    - pattern: "important information"
      weight: 2.0
                """
            elif operation == "format_conversion":
                example_yaml = """
format_conversion:
  output_format: "json"
  schema:
    type: "object"
    properties:
      title:
        type: "string"
      content:
        type: "string"
                """

            raise InvalidConfigurationError(
                f"Failed to normalize configuration for '{operation}': {str(e)}",
                config_path=operation,
                suggestion="Check that your configuration follows the required format for this operation",
                example_yaml=example_yaml,
            ) from e
