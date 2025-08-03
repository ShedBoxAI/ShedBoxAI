"""
Contextual filtering operation handler.

This module provides filtering capabilities for data sources based on
field conditions with AI-enhanced evaluation support.
"""

from typing import Any, Dict

from ..config.models import ContextualFilterConfig
from .base import OperationHandler


class ContextualFilteringHandler(OperationHandler):
    """Handler for contextual filtering operations."""

    @property
    def operation_name(self) -> str:
        return "contextual_filtering"

    def process(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply contextual filtering to the data.

        Args:
            data: Input data dictionary
            config: Normalized filtering configuration (source_name -> filters)

        Returns:
            Data dictionary with filtered results
        """
        result = data.copy()

        # Process each source in the config
        for source_name, filters in config.items():
            if source_name not in result:
                continue

            source_data = result[source_name]

            if isinstance(source_data, list):
                # Ensure filters is a list of ContextualFilterConfig
                if not isinstance(filters, list):
                    self._log_warning(
                        f"Invalid filter configuration for source '{source_name}': expected list, got {type(filters)}"
                    )
                    continue

                # Start with all data and apply filters sequentially (AND logic)
                filtered_data = source_data[:]  # Create a copy

                for filter_config in filters:
                    # Ensure filter_config is a ContextualFilterConfig
                    if not isinstance(filter_config, ContextualFilterConfig):
                        if isinstance(filter_config, dict):
                            try:
                                filter_config = ContextualFilterConfig(**filter_config)
                            except Exception as e:
                                self._log_warning(f"Invalid filter configuration: {e}")
                                continue
                        else:
                            self._log_warning(
                                f"Invalid filter configuration: expected dict or "
                                f"ContextualFilterConfig, got {type(filter_config)}"
                            )
                            continue

                    # Apply this filter to the current filtered_data
                    newly_filtered = []
                    for item in filtered_data:
                        if self._evaluate_filter_condition(item, filter_config):
                            newly_filtered.append(item)

                    # Update filtered_data for next iteration
                    filtered_data = newly_filtered

                # Determine target name for results
                # Use new_name from the last filter that has it, or source_name
                target_name = source_name
                for filter_config in filters:
                    if hasattr(filter_config, "new_name") and filter_config.new_name:
                        target_name = filter_config.new_name
                        break  # Use the first new_name found

                result[target_name] = filtered_data

        return result

    def _evaluate_filter_condition(self, item: Dict[str, Any], filter_config: ContextualFilterConfig) -> bool:
        """
        Evaluate filter condition for a single item.

        Args:
            item: Data item to evaluate
            filter_config: Filter configuration

        Returns:
            True if item passes the filter, False otherwise
        """
        field = filter_config.field
        condition = filter_config.condition

        # Simple equality check for non-comparison conditions
        if not any(op in condition for op in ["<=", ">=", "<", ">", "==", "!="]):
            return item.get(field) == condition

        # Get the field value and convert to float if it's a number
        field_value = item.get(field)
        try:
            # Try to convert to float if it's a string that looks like a number
            if isinstance(field_value, str):
                field_value = float(field_value)
            elif isinstance(field_value, (int, float)):
                field_value = float(field_value)
        except (ValueError, TypeError):
            # If conversion fails, keep the original value
            pass

        # Create a proper expression for the engine to evaluate
        expr = f"{field_value} {condition}"

        try:
            # Let the expression engine handle the comparison
            if self.engine:
                return self.engine.evaluate(expr, {})
            else:
                # Fallback evaluation without engine
                return self._simple_comparison_eval(field_value, condition)
        except Exception:
            # FIXED BUG 2: Fall back to simple comparison when engine fails
            try:
                return self._simple_comparison_eval(field_value, condition)
            except Exception:
                return False

    def _simple_comparison_eval(self, field_value: Any, condition: str) -> bool:
        """
        Simple comparison evaluation without expression engine.

        Args:
            field_value: Value to compare
            condition: Condition string (e.g., "> 100")

        Returns:
            Comparison result
        """
        try:
            # Parse the condition
            condition = condition.strip()
            if condition.startswith(">="):
                return field_value >= float(condition[2:].strip())
            elif condition.startswith("<="):
                return field_value <= float(condition[2:].strip())
            elif condition.startswith(">"):
                return field_value > float(condition[1:].strip())
            elif condition.startswith("<"):
                return field_value < float(condition[1:].strip())
            elif condition.startswith("=="):
                return field_value == float(condition[2:].strip())
            elif condition.startswith("!="):
                return field_value != float(condition[2:].strip())
            else:
                return False
        except Exception:
            return False
