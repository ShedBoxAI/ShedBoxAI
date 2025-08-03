"""
Configuration normalization functions.

This module provides functions to normalize various configuration formats
into standardized structures for processing operations.
"""

import hashlib
import logging
from typing import Any, Dict, List, Union

from .models import (
    AdvancedOperationConfig,
    ContentSummarizationConfig,
    ContextualFilterConfig,
    FormatConversionConfig,
    RelationshipConfig,
    TemplateMatchingConfig,
)


def normalize_contextual_filtering_config(
    config: Dict[str, Any],
) -> Dict[str, List[ContextualFilterConfig]]:
    """
    Normalize contextual filtering configuration.

    Handles both direct source mappings and named configurations.
    Works for both graph node (with config_key) and linear pipeline scenarios.

    Args:
        config: Raw configuration dictionary

    Returns:
        Normalized mapping of source_name -> list of filter configs
    """
    if not config:
        return {}

    result = {}

    for key, value in config.items():
        # Case 1: Direct source mapping - key is source name, value is filter list
        if isinstance(value, list):
            result[key] = value

        # Case 2: Named configuration - key is config name, value maps sources to filters
        elif isinstance(value, dict):
            for source_name, filters in value.items():
                result[source_name] = filters

    return result


def normalize_format_conversion_config(
    config: Dict[str, Any],
) -> Dict[str, FormatConversionConfig]:
    """
    Normalize format conversion configuration.

    Handles both direct source mappings and named configurations.
    Works for both graph node (with config_key) and linear pipeline scenarios.

    Args:
        config: Raw configuration dictionary

    Returns:
        Normalized mapping of source_name -> FormatConversionConfig
    """
    if not config:
        return {}

    result = {}

    for key, value in config.items():
        # Case 1: Direct source mapping - value is FormatConversionConfig or simple dict
        if isinstance(value, FormatConversionConfig):
            result[key] = value
        elif isinstance(value, dict) and _is_direct_config(value):
            try:
                result[key] = FormatConversionConfig(**value)
            except Exception as e:
                print(f"Warning: Invalid format conversion config for '{key}': {e}")
                continue

        # Case 2: Named configuration - value is dict mapping sources to configs
        elif isinstance(value, dict):
            for source_name, conv_config in value.items():
                if isinstance(conv_config, FormatConversionConfig):
                    result[source_name] = conv_config
                elif isinstance(conv_config, dict):
                    try:
                        result[source_name] = FormatConversionConfig(**conv_config)
                    except Exception as e:
                        print(f"Warning: Invalid format conversion config for '{source_name}': {e}")
                        continue
                else:
                    print(f"Warning: Invalid format conversion config type for '{source_name}': {type(conv_config)}")

    return result


def normalize_content_summarization_config(
    config: Dict[str, Any],
) -> Dict[str, ContentSummarizationConfig]:
    """
    Normalize content summarization configuration.

    Handles both direct source mappings and named configurations.
    Works for both graph node (with config_key) and linear pipeline scenarios.

    Args:
        config: Raw configuration dictionary

    Returns:
        Normalized mapping of source_name -> ContentSummarizationConfig
    """
    if not config:
        return {}

    result = {}

    for key, value in config.items():
        # Case 1: Direct source mapping
        if isinstance(value, ContentSummarizationConfig):
            result[key] = value
        elif isinstance(value, dict) and _is_direct_config(value):
            try:
                result[key] = ContentSummarizationConfig(**value)
            except Exception as e:
                print(f"Warning: Invalid content summarization config for '{key}': {e}")
                continue

        # Case 2: Named configuration
        elif isinstance(value, dict):
            for source_name, summary_config in value.items():
                if isinstance(summary_config, ContentSummarizationConfig):
                    result[source_name] = summary_config
                elif isinstance(summary_config, dict):
                    try:
                        result[source_name] = ContentSummarizationConfig(**summary_config)
                    except Exception as e:
                        print(f"Warning: Invalid content summarization config for '{source_name}': {e}")
                        continue
                else:
                    print(
                        (
                            f"Warning: Invalid content summarization config type for '{source_name}': "
                            f"{type(summary_config)}"
                        )
                    )

    return result


def normalize_relationship_highlighting_config(
    config: Union[Dict[str, Any], Any],
) -> Dict[str, RelationshipConfig]:
    """
    Normalize relationship highlighting configuration.

    Handles both direct source mappings and named configurations.
    Works for both graph node (with config_key) and linear pipeline scenarios.

    Args:
        config: Raw configuration (dict or RelationshipConfig)

    Returns:
        Normalized mapping of source_name -> RelationshipConfig
    """
    if not config:
        print("[DEBUG] normalize_relationship_highlighting_config - Empty config")
        return {}

    result = {}

    # Special case: If we received a RelationshipConfig directly
    # This happens in graph mode when config_key is used
    if isinstance(config, RelationshipConfig):
        result["data"] = config
        return result

    # Handle dictionary configs (used in linear mode or named configs)
    if isinstance(config, dict):
        for key, value in config.items():
            # If value is already a RelationshipConfig, use it directly
            if isinstance(value, RelationshipConfig):
                result[key] = value
            # If value is a dict that needs to be converted to RelationshipConfig
            elif isinstance(value, dict):
                try:
                    result[key] = RelationshipConfig(**value)
                except Exception as e:
                    print(f"Warning: Invalid relationship config for '{key}': {e}")
            else:
                print(f"Warning: Unsupported value type for '{key}': {type(value)}")
    else:
        print(f"[DEBUG] Unrecognized config type: {type(config)}")
        return {}

    return result


def normalize_advanced_operations_config(
    config: Dict[str, Any],
) -> Dict[str, AdvancedOperationConfig]:
    """
    Normalize advanced operations configuration.

    Handles both direct source mappings and named configurations.
    Works for both graph node (with config_key) and linear pipeline scenarios.

    Args:
        config: Raw configuration dictionary

    Returns:
        Normalized mapping of result_name -> AdvancedOperationConfig
    """
    if not config:
        return {}

    result = {}

    for key, value in config.items():
        # Case 1: Direct source mapping
        if isinstance(value, AdvancedOperationConfig):
            result[key] = value
        elif isinstance(value, dict) and _is_direct_config(value):
            try:
                result[key] = AdvancedOperationConfig(**value)
            except Exception as e:
                print(f"Warning: Invalid advanced operation config for '{key}': {e}")
                continue

        # Case 2: Named configuration
        elif isinstance(value, dict):
            for source_name, op_config in value.items():
                if isinstance(op_config, AdvancedOperationConfig):
                    result[source_name] = op_config
                elif isinstance(op_config, dict):
                    try:
                        result[source_name] = AdvancedOperationConfig(**op_config)
                    except Exception as e:
                        print(f"Warning: Invalid advanced operation config for '{source_name}': {e}")
                        continue
                else:
                    print(f"Warning: Invalid advanced operation config type for '{source_name}': {type(op_config)}")

    return result


def normalize_template_matching_config(
    config: Dict[str, Any],
) -> Dict[str, TemplateMatchingConfig]:
    """
    Normalize template matching configuration.

    Handles both direct source mappings and named configurations.
    Works for both graph node (with config_key) and linear pipeline scenarios.

    Args:
        config: Raw configuration dictionary

    Returns:
        Normalized mapping of result_name -> TemplateMatchingConfig
    """
    if not config:
        logging.warning("Empty template_matching configuration received")
        return {}

    result = {}

    # Special case: If we received a TemplateMatchingConfig directly
    # This happens in graph mode when config_key is used
    if isinstance(config, TemplateMatchingConfig):
        # Generate a unique key based on the template content
        config_key = _generate_template_key(config)
        result[config_key] = config
        return result

    for key, value in config.items():
        # Case 1: Direct source mapping
        if isinstance(value, TemplateMatchingConfig):
            result[key] = value
        elif isinstance(value, dict) and _is_direct_config(value):
            try:
                result[key] = TemplateMatchingConfig(**value)
            except Exception as e:
                logging.error(f"Error converting to TemplateMatchingConfig for '{key}': {e}")
                logging.error(f"Value that caused the error: {value}")
                continue

        # Case 2: Named configuration
        elif isinstance(value, dict):
            for source_name, template_config in value.items():
                if isinstance(template_config, TemplateMatchingConfig):
                    result[source_name] = template_config
                elif isinstance(template_config, dict):
                    try:
                        result[source_name] = TemplateMatchingConfig(**template_config)
                    except Exception as e:
                        logging.error(f"Error converting to TemplateMatchingConfig for '{source_name}': {e}")
                        logging.error(f"Value that caused the error: {template_config}")
                        continue
                else:
                    logging.warning(f"Unsupported template config type for '{source_name}': {type(template_config)}")
        else:
            logging.warning(f"Unsupported value type for key '{key}': {type(value)}")

    return result


def _is_direct_config(value: dict) -> bool:
    """Determine if a dictionary represents a direct configuration object.

    Rather than a named configuration mapping.
    A direct config has configuration fields, not nested source mappings.

    Args:
        value: Dictionary to check

    Returns:
        True if this appears to be a direct config object
    """
    if not isinstance(value, dict):
        return False

    # Common configuration field names that indicate direct config
    config_fields = {
        "extract_fields",
        "template",
        "source",
        "method",
        "fields",
        "summarize",
        "link_fields",
        "jsonpath_links",
        "pattern_detection",
        "conditional_highlighting",
        "context_additions",
        "derived_fields",
        "group_by",
        "aggregate",
        "sort",
        "limit",
        "template_id",
        "variables",
    }

    # If any key matches known config fields, likely a direct config
    if any(key in config_fields for key in value.keys()):
        return True

    # If all values are primitive types or lists, likely a direct config
    # If any value is a dict, likely a named configuration
    for v in value.values():
        if isinstance(v, dict):
            return False

    return True


def _generate_template_key(config: TemplateMatchingConfig) -> str:
    """
    Generate a unique key for a TemplateMatchingConfig.

    Args:
        config: TemplateMatchingConfig instance

    Returns:
        Unique key string
    """
    # Try to extract a sensible name from the template content or template_id
    if hasattr(config, "template_id") and config.template_id:
        return config.template_id
    elif hasattr(config, "template") and config.template:
        # Try to extract a title from the first line of the template
        first_line = config.template.split("\n")[0].strip("# ")
        if first_line:
            return first_line.lower().replace(" ", "_")[:20]  # Use first 20 chars of title

    # Generate a unique key based on the template content
    template_hash = hashlib.md5((config.template or "").encode(), usedforsecurity=False).hexdigest()[:8]
    return f"template_{template_hash}"
