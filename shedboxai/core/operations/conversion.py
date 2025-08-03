"""
Format conversion operation handler.

This module provides data format conversion capabilities including
field extraction and template-based transformation.
"""

from typing import Any, Dict

from ..config.models import FormatConversionConfig
from .base import OperationHandler


class FormatConversionHandler(OperationHandler):
    """Handler for format conversion operations."""

    @property
    def operation_name(self) -> str:
        return "format_conversion"

    def process(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply format conversion to the data.

        Args:
            data: Input data dictionary
            config: Normalized conversion configuration (source_name -> FormatConversionConfig)

        Returns:
            Data dictionary with converted formats
        """
        result = data.copy()

        # If config is None or empty, return the data as is
        if not config:
            return result

        # Process each source in the config
        for source_name, conversion_config in config.items():
            if source_name not in result:
                continue

            source_data = result[source_name]

            # Ensure conversion_config is a FormatConversionConfig
            if not isinstance(conversion_config, FormatConversionConfig):
                if isinstance(conversion_config, dict):
                    # BUG 2 FIX: Validate that dict contains only valid FormatConversionConfig fields
                    valid_fields = {"extract_fields", "template"}
                    invalid_fields = set(conversion_config.keys()) - valid_fields
                    if invalid_fields:
                        self._log_warning(
                            (
                                f"Invalid format conversion configuration for '{source_name}': "
                                f"Unknown fields {invalid_fields}"
                            )
                        )
                        continue
                    try:
                        conversion_config = FormatConversionConfig(**conversion_config)
                    except Exception as e:
                        self._log_warning(f"Invalid format conversion configuration for '{source_name}': {e}")
                        continue
                else:
                    self._log_warning(
                        (
                            f"Invalid format conversion configuration for '{source_name}': "
                            f"expected dict or FormatConversionConfig, got {type(conversion_config)}"
                        )
                    )
                    continue

            # Check for invalid configuration
            if conversion_config.extract_fields and conversion_config.template:
                raise ValueError(
                    f"Invalid format conversion configuration for '{source_name}': "
                    "Cannot specify both 'extract_fields' and 'template' options. "
                    "Please choose one or the other."
                )

            # Field extraction with nested field support
            if conversion_config.extract_fields:
                result[source_name] = self._extract_fields(source_data, conversion_config.extract_fields)
                continue

            # Template processing
            if conversion_config.template:
                if not isinstance(source_data, list):
                    raise ValueError(
                        f"Template processing only supports list data types. "
                        f"Source '{source_name}' has type {type(source_data)}"
                    )
                result[source_name] = [
                    self._process_template(conversion_config.template, {"item": item}) for item in source_data
                ]

        return result

    def _extract_fields(self, source_data: Any, extract_fields: list) -> Any:
        """
        Extract specified fields from source data.

        Args:
            source_data: Source data to extract from
            extract_fields: List of field names to extract

        Returns:
            Extracted data
        """
        if isinstance(source_data, list):
            converted_data = []
            for item in source_data:
                extracted_item = {}
                for field in extract_fields:
                    if isinstance(field, str) and "{{" in field and "}}" in field:
                        # This is a template field - use _process_template
                        value = self._process_template(field, {"item": item})
                        # Extract the last part of the path for the field name
                        field_name = field.replace("{{", "").replace("}}", "").strip().split(".")[-1]
                        extracted_item[field_name] = value
                    else:
                        # BUG 1 FIX: Regular field extraction with type checking
                        if isinstance(item, dict):
                            extracted_item[field] = item.get(field)
                        else:
                            # Handle non-dict items gracefully
                            extracted_item[field] = None
                converted_data.append(extracted_item)
            return converted_data

        elif isinstance(source_data, dict):
            result = {}
            for field in extract_fields:
                if isinstance(field, str) and "{{" in field and "}}" in field:
                    # Template field in a dictionary
                    value = self._process_template(field, {"item": source_data})
                    field_name = field.replace("{{", "").replace("}}", "").strip().split(".")[-1]
                    result[field_name] = value
                else:
                    # Regular field
                    result[field] = source_data.get(field)
            return result

        return source_data

    def _process_template(self, template: str, context: dict) -> str:
        """
        Process template with AI-enhanced substitution.

        Args:
            template: Template string with placeholders
            context: Context variables for substitution

        Returns:
            Processed template string
        """
        if self.engine:
            return self.engine.substitute_variables(template, context)
        else:
            # Fallback to simple substitution
            return self._simple_template_substitution(template, context)

    def _simple_template_substitution(self, template: str, context: dict) -> str:
        """
        Simple template substitution without expression engine.

        Args:
            template: Template string
            context: Context variables

        Returns:
            Template with simple variable substitution
        """
        import re

        def replace_match(match):
            var_name = match.group(1).strip()
            # Handle nested access like item.field
            parts = var_name.split(".")
            current = context
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return f"{{ERROR: {var_name} not found}}"
            return str(current)

        return re.sub(r"\{\{(.+?)\}\}", replace_match, template)
