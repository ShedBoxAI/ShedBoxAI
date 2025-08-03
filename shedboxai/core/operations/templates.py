"""
Template matching operation handler.

This module provides Jinja2-based template processing with
AI-enhanced variable substitution capabilities.
"""

import logging
import os
from typing import Any, Dict

import jinja2

from ..config.models import TemplateMatchingConfig
from .base import OperationHandler


class TemplateMatchingHandler(OperationHandler):
    """Handler for template matching operations."""

    def __init__(self, engine=None):
        """
        Initialize template handler.

        Args:
            engine: Optional expression engine for variable substitution
        """
        super().__init__(engine)
        self.logger = logging.getLogger(__name__)

        # BUG 1 FIX: Create Jinja2 environment without aggressive whitespace trimming
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(os.getcwd()),
            autoescape=jinja2.select_autoescape(["html", "xml"]),
            trim_blocks=False,  # Don't trim blocks
            lstrip_blocks=False,  # Don't strip leading whitespace
        )

        # Register custom filters if needed
        self._register_custom_filters()

    @property
    def operation_name(self) -> str:
        return "template_matching"

    def process(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply template matching to the data.

        Args:
            data: Input data dictionary
            config: Normalized template configuration (result_name -> TemplateMatchingConfig)

        Returns:
            Data dictionary with template processing results
        """
        result = data.copy()

        # Process each template in the config
        for result_name, template_config in config.items():
            # Ensure template_config is a TemplateMatchingConfig
            if not isinstance(template_config, TemplateMatchingConfig):
                if isinstance(template_config, dict):
                    # BUG 2 FIX: Validate dict contains only valid TemplateMatchingConfig fields
                    valid_fields = {"template", "template_id", "variables"}
                    invalid_fields = set(template_config.keys()) - valid_fields
                    if invalid_fields:
                        self._log_warning(
                            f"Invalid template matching configuration for '{result_name}': "
                            f"Unknown fields {invalid_fields}"
                        )
                        continue
                    try:
                        template_config = TemplateMatchingConfig(**template_config)
                    except Exception as e:
                        # BUG 2 FIX: Use _log_warning instead of self.logger.warning
                        self._log_warning(f"Invalid template matching configuration for '{result_name}': {e}")
                        continue
                else:
                    # BUG 3 FIX: Use _log_warning instead of self.logger.warning
                    self._log_warning(
                        f"Invalid template matching configuration for '{result_name}': "
                        f"expected dict or TemplateMatchingConfig, got {type(template_config)}"
                    )
                    continue

            # Process the template
            try:
                processed_result = self._process_template(template_config, data)
                result[result_name] = processed_result
            except Exception as e:
                self.logger.error(f"Error processing template '{result_name}': {e}")
                result[result_name] = f"ERROR: {str(e)}"

        return result

    def _process_template(self, config: TemplateMatchingConfig, data: Dict[str, Any]) -> str:
        """
        Process a single template configuration.

        Args:
            config: Template configuration
            data: Input data for template context

        Returns:
            Processed template string
        """
        # Get template content
        template_content = self._get_template_content(config)
        if not template_content:
            raise ValueError("No template content available")

        # Prepare context data
        context_data = data.copy()

        # Add any extra variables from config
        if config.variables:
            context_data.update(config.variables)

        # Process the template using Jinja2
        template = self.jinja_env.from_string(template_content)
        processed_template = template.render(**context_data)

        # Additional processing with expression engine for any {{variable}} substitutions
        # that may be used outside of Jinja2 syntax
        if self.engine:
            processed_template = self.engine.substitute_variables(processed_template, context_data)

        return processed_template

    def _get_template_content(self, config: TemplateMatchingConfig) -> str:
        """
        Get template content from configuration.

        Args:
            config: Template configuration

        Returns:
            Template content string
        """
        if config.template:
            # Use template string directly
            return config.template
        elif config.template_id:
            # Load template from file
            return self._load_template_file(config.template_id)
        else:
            raise ValueError("Either template or template_id must be provided")

    def _load_template_file(self, template_id: str) -> str:
        """
        Load template from file.

        Args:
            template_id: Template identifier/filename

        Returns:
            Template file content
        """
        try:
            template_path = os.path.join("templates", f"{template_id}.j2")
            if os.path.exists(template_path):
                with open(template_path, "r", encoding="utf-8") as f:
                    return f.read()
            else:
                # Try without .j2 extension
                template_path = os.path.join("templates", template_id)
                if os.path.exists(template_path):
                    with open(template_path, "r", encoding="utf-8") as f:
                        return f.read()
                else:
                    raise FileNotFoundError(f"Template file not found: {template_path}")
        except Exception as e:
            raise ValueError(f"Error loading template file '{template_id}': {e}")

    def _register_custom_filters(self) -> None:
        """Register custom Jinja2 filters."""
        # Example custom filters - can be extended as needed

        def currency_filter(value, currency="$"):
            """Format value as currency."""
            try:
                return f"{currency}{float(value):,.2f}"
            except (ValueError, TypeError):
                return str(value)

        def percentage_filter(value):
            """Format value as percentage."""
            try:
                return f"{float(value):.1f}%"
            except (ValueError, TypeError):
                return str(value)

        def safe_get_filter(obj, key, default=""):
            """Safely get value from object."""
            if isinstance(obj, dict):
                return obj.get(key, default)
            elif hasattr(obj, key):
                return getattr(obj, key, default)
            else:
                return default

        def length_filter(obj):
            """Get length of object."""
            try:
                return len(obj)
            except (TypeError, AttributeError):
                return 0

        def first_filter(obj):
            """Get first item from sequence."""
            try:
                return obj[0] if obj else None
            except (TypeError, IndexError):
                return None

        def last_filter(obj):
            """Get last item from sequence."""
            try:
                return obj[-1] if obj else None
            except (TypeError, IndexError):
                return None

        # Register filters
        self.jinja_env.filters["currency"] = currency_filter
        self.jinja_env.filters["percentage"] = percentage_filter
        self.jinja_env.filters["safe_get"] = safe_get_filter
        self.jinja_env.filters["length"] = length_filter
        self.jinja_env.filters["first"] = first_filter
        self.jinja_env.filters["last"] = last_filter
