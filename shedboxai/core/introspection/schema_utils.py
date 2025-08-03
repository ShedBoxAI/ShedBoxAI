"""
Shared utilities for JSON schema generation across analyzers.

This module provides common functionality for generating JSON schemas
from data using the genson library, promoting code reuse between
REST and JSON analyzers.
"""

import logging
from typing import Any, Dict, Optional

from genson import SchemaBuilder

logger = logging.getLogger(__name__)


def generate_json_schema(data: Any, max_items: int = 100) -> Optional[Dict[str, Any]]:
    """
    Generate JSON schema from data using genson library.

    Args:
        data: The data to analyze (can be dict, list, or primitive)
        max_items: Maximum number of items to sample from arrays

    Returns:
        Dict containing the generated JSON schema, or None if generation fails
    """
    try:
        schema_builder = SchemaBuilder()

        if isinstance(data, list):
            # Array of items - sample up to max_items
            items_to_process = min(len(data), max_items)
            for item in data[:items_to_process]:
                schema_builder.add_object(item)
        else:
            # Single object or primitive
            schema_builder.add_object(data)

        return schema_builder.to_schema()

    except Exception as e:
        logger.error(f"Failed to generate JSON schema: {str(e)}")
        return None


def extract_main_data(response_data: Any, response_path: Optional[str] = None) -> Any:
    """
    Extract main data from response using optional response_path.

    Args:
        response_data: The full response data
        response_path: Optional dot-notation path to extract specific data

    Returns:
        The extracted data, or original data if path not found/invalid
    """
    if not response_path:
        return response_data

    try:
        # Simple dot notation path handling
        path_parts = response_path.split(".")
        current_data = response_data

        for part in path_parts:
            if isinstance(current_data, dict) and part in current_data:
                current_data = current_data[part]
            else:
                # Path not found, return original data
                logger.warning(f"Response path '{response_path}' not found, using full response")
                return response_data

        return current_data

    except Exception as e:
        logger.warning(f"Error extracting response path '{response_path}': {str(e)}")
        return response_data


def calculate_nesting_depth(data: Any, current_depth: int = 0, max_depth: int = 10) -> int:
    """
    Calculate maximum nesting depth of data structure.

    Args:
        data: The data to analyze
        current_depth: Current depth level
        max_depth: Maximum depth to prevent infinite recursion

    Returns:
        Maximum nesting depth found
    """
    if current_depth >= max_depth:
        return current_depth

    if isinstance(data, dict):
        if not data:
            return current_depth
        return max(calculate_nesting_depth(value, current_depth + 1, max_depth) for value in data.values())
    elif isinstance(data, list):
        if not data:
            return current_depth
        return max(
            calculate_nesting_depth(item, current_depth + 1, max_depth) for item in data[:10]  # Sample first 10 items
        )
    else:
        return current_depth


def has_arrays(data: Any) -> bool:
    """
    Check if data structure contains arrays.

    Args:
        data: The data to check

    Returns:
        True if arrays are found, False otherwise
    """
    if isinstance(data, list):
        return True
    elif isinstance(data, dict):
        return any(has_arrays(value) for value in data.values())
    return False


def has_objects(data: Any) -> bool:
    """
    Check if data structure contains objects.

    Args:
        data: The data to check

    Returns:
        True if objects are found, False otherwise
    """
    if isinstance(data, dict):
        return True
    elif isinstance(data, list):
        return any(has_objects(item) for item in data[:10])
    return False
