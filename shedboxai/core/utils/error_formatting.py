"""
Error formatting utilities for ShedBoxAI.

This module provides utilities for formatting error messages in a consistent,
user-friendly way that provides actionable information for configuration authors.
"""

from typing import Any, Dict, List, Optional

import yaml


def format_config_error(
    component: str,
    problem: str,
    location: str,
    expected: str,
    fix: str,
    example_yaml: Optional[str] = None,
) -> str:
    """
    Format consistent error messages for configuration errors.

    Args:
        component: Component name (e.g., 'PIPELINE', 'DATA_SOURCE')
        problem: Brief description of the problem
        location: Configuration path where the error occurred
        expected: What was expected in the configuration
        fix: Suggested fix for the problem
        example_yaml: Optional YAML example of correct configuration

    Returns:
        Formatted error message
    """
    message = f"[{component.upper()}] Configuration Error: {problem}\n\n"
    message += f"Problem: {problem}\n"
    message += f"Location: {location}\n"
    message += f"Expected: {expected}\n"
    message += f"Fix: {fix}\n"

    if example_yaml:
        message += "\nExample:\n```yaml\n"
        message += example_yaml
        message += "\n```"

    return message


def get_config_path(config_dict: Dict[str, Any], target_key: str, parent_path: str = "") -> Optional[str]:
    """
    Get the path to a specific key in a nested configuration dictionary.

    Args:
        config_dict: Configuration dictionary to search
        target_key: Key to find
        parent_path: Path to current position in the dictionary

    Returns:
        Dot-notation path to the key (e.g., "data_sources.source1.type")
        or None if key not found
    """
    if not isinstance(config_dict, dict):
        return None

    if target_key in config_dict:
        return f"{parent_path}.{target_key}" if parent_path else target_key

    for key, value in config_dict.items():
        if isinstance(value, dict):
            new_path = f"{parent_path}.{key}" if parent_path else key
            result = get_config_path(value, target_key, new_path)
            if result:
                return result

    # Search in list items that are dictionaries
    for key, value in config_dict.items():
        if isinstance(value, list):
            new_path = f"{parent_path}.{key}" if parent_path else key
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    item_path = f"{new_path}[{i}]"
                    if target_key in item:
                        return f"{item_path}.{target_key}"
                    result = get_config_path(item, target_key, item_path)
                    if result:
                        return result

    return None


def extract_yaml_context(yaml_content: str, line_number: int, context_lines: int = 3) -> str:
    """
    Extract context from YAML content around a specific line.

    Args:
        yaml_content: Full YAML content string
        line_number: Line number to focus on (1-based)
        context_lines: Number of lines of context before and after

    Returns:
        String with line numbers and content around the specified line
    """
    lines = yaml_content.splitlines()

    # Adjust line_number to 0-based index
    line_index = line_number - 1

    # Determine start and end lines for context
    start_line = max(0, line_index - context_lines)
    end_line = min(len(lines) - 1, line_index + context_lines)

    # Build context with line numbers
    context = []
    for i in range(start_line, end_line + 1):
        prefix = "→ " if i == line_index else "  "
        context.append(f"{prefix}{i+1}: {lines[i]}")

    return "\n".join(context)


def get_operation_section_path(config: Dict[str, Any], operation_name: str) -> Optional[str]:
    """
    Find the path to an operation section in the configuration.

    Args:
        config: Full configuration dictionary
        operation_name: Name of the operation to find

    Returns:
        Path to the operation section or None if not found
    """
    if "processing" not in config:
        return None

    processing = config["processing"]

    # Check if operation exists directly in processing section
    if operation_name in processing:
        return f"processing.{operation_name}"

    # Check in graph nodes
    if "graph" in processing and isinstance(processing["graph"], list):
        for i, node in enumerate(processing["graph"]):
            if "operation" in node and node["operation"] == operation_name:
                return f"processing.graph[{i}].operation"

    return None


def find_field_reference(config: Dict[str, Any], field_name: str, section: Optional[str] = None) -> List[str]:
    """
    Find all references to a field in the configuration.

    Args:
        config: Configuration dictionary to search
        field_name: Field name to find
        section: Optional section to limit search to

    Returns:
        List of paths to the field references
    """
    paths = []

    def search_dict(d, path=""):
        if not isinstance(d, dict):
            return

        for k, v in d.items():
            current_path = f"{path}.{k}" if path else k

            if k == field_name:
                paths.append(current_path)

            if isinstance(v, dict):
                search_dict(v, current_path)
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    item_path = f"{current_path}[{i}]"
                    if isinstance(item, dict):
                        search_dict(item, item_path)

    if section and section in config:
        search_dict(config[section], section)
    else:
        search_dict(config)

    return paths


def format_field_path(path: str) -> str:
    """
    Format a field path for display in error messages.

    Args:
        path: Raw field path with dots

    Returns:
        Formatted path for display
    """
    # Replace dots with indentation to show nesting
    parts = path.split(".")
    if len(parts) <= 2:
        return path

    indented = parts[0] + ":\n"
    for i in range(1, len(parts) - 1):
        indented += "  " * i + parts[i] + ":\n"
    indented += "  " * (len(parts) - 1) + parts[-1]

    return indented


def load_yaml_safely(yaml_content: str) -> Dict[str, Any]:
    """
    Load YAML content with error handling.

    Args:
        yaml_content: YAML content string

    Returns:
        Parsed YAML as dictionary

    Raises:
        ValidationError: If YAML cannot be parsed
    """
    from ..exceptions import ValidationError

    try:
        return yaml.safe_load(yaml_content)
    except yaml.YAMLError as e:
        line = e.problem_mark.line + 1 if hasattr(e, "problem_mark") else None
        column = e.problem_mark.column + 1 if hasattr(e, "problem_mark") else None

        location = "unknown"
        if line:
            location = f"line {line}"
            if column:
                location += f", column {column}"

        error_context = ""
        if line and yaml_content:
            error_context = "\n\n" + extract_yaml_context(yaml_content, line)

        message = f"Invalid YAML syntax at {location}: {str(e)}{error_context}"
        raise ValidationError(message)


def get_env_var_references(config: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Find all environment variable references in the configuration.

    Args:
        config: Configuration dictionary to search

    Returns:
        Dictionary mapping environment variable names to their reference paths
    """
    env_vars = {}

    def search_for_env_vars(obj, path=""):
        if isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
            var_name = obj[2:-1]
            if var_name not in env_vars:
                env_vars[var_name] = []
            env_vars[var_name].append(path)
        elif isinstance(obj, dict):
            for k, v in obj.items():
                new_path = f"{path}.{k}" if path else k
                search_for_env_vars(v, new_path)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                new_path = f"{path}[{i}]"
                search_for_env_vars(item, new_path)

    search_for_env_vars(config)
    return env_vars


def format_graph_error(
    error_type: str,
    node_id: str,
    problem: str,
    suggestion: str,
    example_yaml: Optional[str] = None,
) -> str:
    """
    Format consistent error messages for graph execution errors.

    Args:
        error_type: Type of error (e.g., 'UNKNOWN_OPERATION', 'CYCLIC_DEPENDENCY')
        node_id: ID of the graph node where the error occurred
        problem: Brief description of the problem
        suggestion: Suggested fix for the problem
        example_yaml: Optional YAML example of correct configuration

    Returns:
        Formatted error message
    """
    message = f"[GRAPH] {error_type}: {problem}\n\n"
    message += f"Node: {node_id}\n"
    message += f"Problem: {problem}\n"
    message += f"Suggestion: {suggestion}\n"

    if example_yaml:
        message += "\nExample:\n```yaml\n"
        message += example_yaml
        message += "\n```"

    return message
