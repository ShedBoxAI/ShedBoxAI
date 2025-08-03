"""
Advanced operations handler.

This module provides advanced data operations including
grouping, aggregation, sorting, and limiting.
"""

import re
from typing import Any, Dict

from ..config.models import AdvancedOperationConfig
from .base import OperationHandler


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
