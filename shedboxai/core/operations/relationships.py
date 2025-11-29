"""
Relationship highlighting operation handler.

This module provides advanced relationship detection and highlighting
capabilities including JSONPath links, pattern detection, and conditional highlighting.
"""

from typing import Any, Dict

import jsonpath_ng.ext as jsonpath

from ..config.models import RelationshipConfig
from .base import OperationHandler


class RelationshipHighlightingHandler(OperationHandler):
    """Handler for relationship highlighting operations."""

    @property
    def operation_name(self) -> str:
        return "relationship_highlighting"

    def _convert_dataframes(self, result: Dict[str, Any]) -> None:
        """Convert any DataFrame values in result to list of dicts in place."""
        for key, value in result.items():
            if hasattr(value, "to_dict") and callable(value.to_dict):
                result[key] = value.to_dict("records")

    def process(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply relationship highlighting to the data.

        Args:
            data: Input data dictionary
            config: Normalized relationship configuration (source_name -> RelationshipConfig)

        Returns:
            Data dictionary with relationship analysis results
        """
        result = data.copy()

        # Convert any DataFrames to list of dicts for processing
        self._convert_dataframes(result)

        # Process each source in the config
        for source_name, relationship_config in config.items():
            # Ensure relationship_config is a RelationshipConfig
            if not isinstance(relationship_config, RelationshipConfig):
                if isinstance(relationship_config, dict):
                    # BUG 2 FIX: Validate dict contains only valid RelationshipConfig fields
                    valid_fields = {
                        "link_fields",
                        "jsonpath_links",
                        "pattern_detection",
                        "conditional_highlighting",
                        "context_additions",
                        "derived_fields",
                    }
                    invalid_fields = set(relationship_config.keys()) - valid_fields
                    if invalid_fields:
                        self._log_warning(
                            f"Invalid relationship configuration for '{source_name}': Unknown fields {invalid_fields}"
                        )
                        continue
                    try:
                        relationship_config = RelationshipConfig(**relationship_config)
                    except Exception as e:
                        self._log_warning(f"Invalid relationship configuration for '{source_name}': {e}")
                        continue
                else:
                    self._log_warning(
                        f"Invalid relationship configuration for '{source_name}': "
                        f"expected dict or RelationshipConfig, got {type(relationship_config)}"
                    )
                    continue

            # Process different types of relationship operations
            self._process_link_fields(result, relationship_config)
            self._process_jsonpath_links(result, relationship_config)
            self._process_pattern_detection(result, relationship_config)
            self._process_conditional_highlighting(result, relationship_config)
            self._process_context_additions(result, relationship_config)
            self._process_derived_fields(result, relationship_config, source_name)

        return result

    def _process_link_fields(self, result: Dict[str, Any], config: RelationshipConfig) -> None:
        """Process original link fields (joins)."""
        if not config.link_fields:
            return

        for link in config.link_fields:
            source = link.get("source")
            to = link.get("to")

            # BUG 1 FIX: Handle both old format (single match_on) and new format (separate fields)
            if "source_field" in link and "target_field" in link:
                # New format with separate field names
                source_field = link.get("source_field")
                target_field = link.get("target_field")
            else:
                # Old format with single match_on field (same field name in both)
                match_on = link.get("match_on")
                source_field = match_on
                target_field = match_on

            if not all([source, source_field, to, target_field]) or source not in result or to not in result:
                continue

            source_data = result[source]
            target_data = result[to]

            if isinstance(source_data, list) and isinstance(target_data, list):
                # Create lookup dictionary for target data
                target_lookup = {item.get(target_field): item for item in target_data if item.get(target_field)}

                # Enhance source data with matching target data
                for item in source_data:
                    match_value = item.get(source_field)
                    if match_value and match_value in target_lookup:
                        item[f"{to}_info"] = target_lookup[match_value]

    def _process_jsonpath_links(self, result: Dict[str, Any], config: RelationshipConfig) -> None:
        """Process JSONPath-based links."""
        if not config.jsonpath_links:
            return

        for link in config.jsonpath_links:
            source = link.get("source")
            target = link.get("target")
            source_path = link.get("source_path")
            target_path = link.get("target_path")
            result_field = link.get("result_field", "related_data")

            if not all([source, target, source_path, target_path]) or source not in result or target not in result:
                continue

            try:
                # Parse JSONPath expressions
                if not source_path.startswith("$"):
                    source_path = "$." + source_path
                if not target_path.startswith("$"):
                    target_path = "$." + target_path

                source_expr = jsonpath.parse(source_path)
                target_expr = jsonpath.parse(target_path)

                # Extract values from source data
                source_values = []
                if isinstance(result[source], list):
                    for item in result[source]:
                        matches = source_expr.find(item)
                        source_values.extend([match.value for match in matches])
                else:
                    matches = source_expr.find(result[source])
                    source_values = [match.value for match in matches]

                # For each item in the target, find matching entries
                if isinstance(result[target], list):
                    for target_item in result[target]:
                        target_values = [match.value for match in target_expr.find(target_item)]

                        # Check for matching values
                        for source_val in source_values:
                            if source_val in target_values:
                                if result_field not in target_item:
                                    target_item[result_field] = []

                                # Add related source data
                                if isinstance(result[source], list):
                                    matching_sources = [
                                        item
                                        for item in result[source]
                                        if any(source_val == match.value for match in source_expr.find(item))
                                    ]
                                    target_item[result_field].extend(matching_sources)
                                else:
                                    target_item[result_field] = result[source]
            except Exception as e:
                self._log_error(f"Error processing JSONPath link: {e}")

    def _process_pattern_detection(self, result: Dict[str, Any], config: RelationshipConfig) -> None:
        """Process pattern detection."""
        if not config.pattern_detection:
            return

        for pattern_name, pattern in config.pattern_detection.items():
            pattern_type = pattern.get("type")
            source = pattern.get("source")

            if not source or source not in result:
                continue

            if pattern_type == "frequency":
                self._detect_frequency_patterns(result, source, pattern)
            elif pattern_type == "sequence":
                self._detect_sequence_patterns(result, source, pattern)

    def _detect_frequency_patterns(self, result: Dict[str, Any], source: str, pattern: Dict[str, Any]) -> None:
        """Detect frequency patterns in data."""
        field = pattern.get("field")
        threshold = pattern.get("threshold", 2)

        if field and isinstance(result[source], list):
            values = {}
            for item in result[source]:
                value = item.get(field)
                if value:
                    values[value] = values.get(value, 0) + 1

            # Filter patterns that meet threshold
            patterns = {k: v for k, v in values.items() if v >= threshold}

            # Store detected patterns
            if patterns:
                result[f"{source}_patterns"] = {
                    "type": "frequency",
                    "field": field,
                    "patterns": patterns,
                }

    def _detect_sequence_patterns(self, result: Dict[str, Any], source: str, pattern: Dict[str, Any]) -> None:
        """Detect sequential patterns in data."""
        field = pattern.get("field")
        sequence_length = pattern.get("length", 3)

        if field and isinstance(result[source], list):
            # Sort by field if available
            try:
                sorted_data = sorted(result[source], key=lambda x: x.get(field, 0))
            except (TypeError, ValueError):
                return

            # Find sequences
            sequences = []
            for i in range(len(sorted_data) - sequence_length + 1):
                sequence = sorted_data[i : i + sequence_length]
                # Check if it's a valid sequence
                values = [item.get(field) for item in sequence]
                try:
                    if all(values[i + 1] - values[i] == 1 for i in range(len(values) - 1)):
                        sequences.append(sequence)
                except (TypeError, ValueError):
                    continue

            if sequences:
                result[f"{source}_sequences"] = sequences

    def _process_conditional_highlighting(self, result: Dict[str, Any], config: RelationshipConfig) -> None:
        """Process conditional highlighting."""
        if not config.conditional_highlighting:
            return

        for highlight in config.conditional_highlighting:
            source = highlight.get("source")
            condition = highlight.get("condition")
            insight_name = highlight.get("insight_name", "highlight")
            insight_desc = highlight.get("context", "")

            if not source or not condition or source not in result:
                continue

            if isinstance(result[source], list):
                highlighted_items = []

                for item in result[source]:
                    # Use expression engine to evaluate condition
                    try:
                        if self.engine and self.engine.evaluate(condition, {"item": item}):
                            highlighted_item = item.copy()
                            highlighted_item["_highlight"] = {
                                "name": insight_name,
                                "description": insight_desc,
                            }
                            highlighted_items.append(highlighted_item)
                    except Exception as e:
                        self._log_error(f"Error evaluating condition: {condition} - {str(e)}")

                if highlighted_items:
                    result[f"{source}_highlights"] = highlighted_items

    def _process_context_additions(self, result: Dict[str, Any], config: RelationshipConfig) -> None:
        """Process context additions."""
        if not config.context_additions:
            return

        for target, context_template in config.context_additions.items():
            if target not in result:
                continue

            if isinstance(result[target], list):
                for item in result[target]:
                    try:
                        # Process template with item context
                        if self.engine:
                            context = self.engine.substitute_variables(context_template, {"item": item, "data": result})
                        else:
                            context = context_template  # Fallback to template as-is
                        item["_context"] = context
                    except Exception as e:
                        self._log_error(f"Error adding context: {str(e)}")

    def _process_derived_fields(self, result: Dict[str, Any], config: RelationshipConfig, target_source: str) -> None:
        """Process derived fields.

        Args:
            result: The result dictionary containing all data sources
            config: The relationship configuration
            target_source: The specific source name to apply derived fields to
        """
        if not config.derived_fields:
            return

        # Only apply derived fields to the target source, not all sources
        source_data = result.get(target_source)
        if not isinstance(source_data, list):
            return

        for derived_field in config.derived_fields:
            parts = derived_field.split("=", 1)
            if len(parts) != 2:
                continue
            field_name = parts[0].strip()
            expression = parts[1].strip()

            for item in source_data:
                try:
                    if self.engine:
                        item[field_name] = self.engine.evaluate(expression, {"item": item})
                    else:
                        # Fallback: just store the expression
                        item[field_name] = f"EXPR: {expression}"
                except Exception as e:
                    self._log_error(f"Error evaluating derived field on '{target_source}': {derived_field} - {str(e)}")
