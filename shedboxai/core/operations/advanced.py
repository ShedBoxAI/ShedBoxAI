"""
Advanced operations handler.

This module provides advanced data operations including
grouping, aggregation, sorting, and limiting.
"""

import re
from difflib import get_close_matches
from typing import Any, Dict, List

from ..config.models import AdvancedOperationConfig
from ..exceptions import ConfigurationError
from .base import OperationHandler

# Allowed aggregation functions
ALLOWED_AGG_FUNCTIONS = ["SUM", "AVG", "COUNT", "MIN", "MAX", "MEDIAN", "STD"]
# Pattern for simple aggregations: FUNCTION(field) or COUNT(*) - allows whitespace
ALLOWED_AGG_PATTERN = r"^(SUM|AVG|COUNT|MIN|MAX|MEDIAN|STD)\(\s*[\w*]+\s*\)$"


def validate_aggregation_expression(expr: str) -> None:
    """
    Validate that aggregation expression is simple and supported.

    Args:
        expr: Aggregation expression to validate

    Raises:
        ConfigurationError: If expression is too complex or invalid
    """
    expr = expr.strip()

    # Check if it matches allowed pattern
    if not re.match(ALLOWED_AGG_PATTERN, expr):
        raise ConfigurationError(
            f"Invalid aggregation expression: '{expr}'\n"
            f"Only simple aggregations are supported: SUM(field), AVG(field), "
            f"COUNT(*), MIN(field), MAX(field), MEDIAN(field), STD(field)\n"
            f"Complex expressions like arithmetic operations (e.g., 'SUM(x)/100'), "
            f"CASE statements, or DISTINCT (e.g., 'COUNT(DISTINCT x)') are not supported.\n\n"
            f"Workaround: Use derived fields in relationship_highlighting first, then aggregate:\n"
            f"  relationship_highlighting:\n"
            f"    my_data:\n"
            f"      derived_fields:\n"
            f"        - calculated_value = item.original_value / 100\n"
            f"  advanced_operations:\n"
            f"    summary:\n"
            f"      source: my_data\n"
            f"      aggregate:\n"
            f"        total: 'SUM(calculated_value)'"
        )


def validate_field_exists(field_name: str, available_fields: List[str], source_name: str) -> None:
    """
    Validate that a field exists in the available fields, provide suggestions if not.

    Args:
        field_name: The field name to validate
        available_fields: List of available field names
        source_name: Name of the data source for error messages

    Raises:
        ConfigurationError: If field doesn't exist
    """
    if field_name not in available_fields:
        # Try to find similar field names
        suggestions = get_close_matches(field_name, available_fields, n=1, cutoff=0.6)

        error_msg = (
            f"Field '{field_name}' not found in '{source_name}'\n" f"Available fields: {', '.join(available_fields)}"
        )

        if suggestions:
            error_msg += f"\n\nDid you mean: '{suggestions[0]}'?"

        raise ConfigurationError(error_msg)


class AdvancedOperationsHandler(OperationHandler):
    """Handler for advanced data operations."""

    @property
    def operation_name(self) -> str:
        return "advanced_operations"

    def process(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply advanced operations to the data.

        Args:
            data: Input data dictionary
            config: Normalized advanced operations configuration (result_name -> AdvancedOperationConfig)

        Returns:
            Data dictionary with advanced operation results
        """
        result = data.copy()

        # Process each operation in the config
        for result_name, op_config in config.items():
            # Ensure op_config is an AdvancedOperationConfig
            if not isinstance(op_config, AdvancedOperationConfig):
                if isinstance(op_config, dict):
                    try:
                        op_config = AdvancedOperationConfig(**op_config)
                    except Exception as e:
                        self._log_warning(f"Invalid advanced operation configuration for '{result_name}': {e}")
                        continue
                else:
                    self._log_warning(
                        (
                            f"Invalid advanced operation configuration for '{result_name}': "
                            f"expected dict or AdvancedOperationConfig, got {type(op_config)}"
                        )
                    )
                    continue

            source = op_config.source
            if source not in result:
                continue

            source_data = result[source]

            # Convert DataFrames to list of dicts for processing
            if hasattr(source_data, "to_dict") and callable(source_data.to_dict):
                source_data = source_data.to_dict("records")

            # BUG 2 FIX: Handle non-list data for sorting/limiting operations
            if not isinstance(source_data, list):
                # For non-list data, create result if any operation is specified
                if op_config.sort or op_config.limit is not None:
                    result[result_name] = source_data  # Non-list data unchanged
                continue

            # BUG 1 & 3 FIX: Apply operations in sequence, ensuring result exists for sorting/limiting
            processed_data = self._apply_grouping_and_aggregation(source_data, op_config, result_name)
            if processed_data is not None:
                result[result_name] = processed_data
            else:
                # No grouping - use original source data for sorting/limiting
                if op_config.sort or op_config.limit is not None:  # BUG 3 FIX: Check limit is not None
                    result[result_name] = source_data[:]  # Copy the original data

            # Apply sorting if specified
            if op_config.sort and result_name in result:
                result[result_name] = self._apply_sorting(result[result_name], op_config.sort)

            # Apply limiting if specified
            if op_config.limit and result_name in result:
                result[result_name] = self._apply_limiting(result[result_name], op_config.limit)

        return result

    def _apply_grouping_and_aggregation(
        self, source_data: list, config: AdvancedOperationConfig, result_name: str
    ) -> list:
        """
        Apply grouping and aggregation operations.

        Args:
            source_data: Source data list
            config: Operation configuration
            result_name: Name for the result

        Returns:
            Processed data list or None if no grouping
        """
        if not config.group_by:
            return None

        group_by_field = config.group_by
        grouped_data = {}

        # Group data by field
        for item in source_data:
            group_key = item.get(group_by_field)
            if group_key is not None:
                grouped_data.setdefault(str(group_key), []).append(item)

        # Apply aggregations if specified
        if config.aggregate:
            return self._apply_aggregations(grouped_data, group_by_field, config.aggregate)
        else:
            # Return grouped data without aggregation
            return [
                {group_by_field: group_key, "items": group_items} for group_key, group_items in grouped_data.items()
            ]

    def _apply_aggregations(
        self,
        grouped_data: Dict[str, list],
        group_by_field: str,
        aggregates: Dict[str, str],
    ) -> list:
        """
        Apply aggregation functions to grouped data.

        Args:
            grouped_data: Data grouped by field values
            group_by_field: Field used for grouping
            aggregates: Aggregation specifications

        Returns:
            List of aggregated results
        """
        # Validate all aggregation expressions first
        for agg_field, agg_expr in aggregates.items():
            validate_aggregation_expression(agg_expr)

        aggregated_data = []

        for group_key, group_items in grouped_data.items():
            result_item = {group_by_field: group_key}

            for agg_field, agg_expr in aggregates.items():
                agg_result = self._evaluate_aggregation(group_items, agg_expr)
                result_item[agg_field] = agg_result

            aggregated_data.append(result_item)

        return aggregated_data

    def _evaluate_aggregation(self, group_items: list, agg_expr: str) -> Any:
        """
        Evaluate a single aggregation expression.

        Args:
            group_items: Items in the group
            agg_expr: Aggregation expression (e.g., "SUM(price)", "COUNT(*)")

        Returns:
            Aggregation result
        """
        match = re.match(r"(\w+)\((.*?)\)", agg_expr)
        if not match:
            self._log_warning(f"Invalid aggregation expression: {agg_expr}")
            return None

        agg_func = match.group(1).upper()
        agg_target = match.group(2)

        if agg_func == "COUNT":
            if agg_target == "*":
                return len(group_items)
            else:
                return sum(1 for item in group_items if agg_target in item and item[agg_target] is not None)

        # For other functions, extract numeric values
        if agg_target == "*":
            self._log_warning(f"Aggregation function {agg_func} cannot use '*' target")
            return None

        values = []
        for item in group_items:
            if agg_target in item and item[agg_target] is not None:
                try:
                    val = float(item[agg_target])
                    values.append(val)
                except (ValueError, TypeError):
                    continue

        if not values:
            return None

        if agg_func == "SUM":
            return sum(values)
        elif agg_func == "AVG":
            return sum(values) / len(values)
        elif agg_func == "MIN":
            return min(values)
        elif agg_func == "MAX":
            return max(values)
        elif agg_func == "MEDIAN":
            sorted_values = sorted(values)
            n = len(sorted_values)
            if n % 2 == 0:
                return (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2
            else:
                return sorted_values[n // 2]
        elif agg_func == "STD":
            if len(values) > 1:
                mean = sum(values) / len(values)
                variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
                return variance**0.5
            else:
                return 0
        else:
            self._log_warning(f"Unknown aggregation function: {agg_func}")
            return None

    def _apply_sorting(self, data: list, sort_spec: str) -> list:
        """
        Apply sorting to data.

        Args:
            data: Data to sort
            sort_spec: Sort specification (e.g., "field" or "-field" for descending)

        Returns:
            Sorted data list
        """
        if not isinstance(data, list):
            return data

        sort_field = sort_spec
        reverse = False

        if sort_field.startswith("-"):
            sort_field = sort_field[1:]
            reverse = True

        try:
            return sorted(
                data,
                key=lambda x: x.get(sort_field, 0) if isinstance(x, dict) else 0,
                reverse=reverse,
            )
        except (TypeError, ValueError) as e:
            self._log_warning(f"Error sorting by field '{sort_field}': {e}")
            return data

    def _apply_limiting(self, data: list, limit: int) -> list:
        """
        Apply limit to data.

        Args:
            data: Data to limit
            limit: Maximum number of items to return

        Returns:
            Limited data list
        """
        if not isinstance(data, list):
            return data

        return data[:limit] if limit > 0 else data
