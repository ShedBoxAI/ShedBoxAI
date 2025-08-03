"""
Configuration fixtures for ShedBoxAI processor testing.

This module provides reusable configuration fixtures for testing various
processor operations and scenarios.
"""

from typing import Any, Dict

import pytest


@pytest.fixture
def basic_filtering_config():
    """Basic contextual filtering configuration."""
    return {"contextual_filtering": {"users": [{"field": "age", "condition": "> 25", "new_name": "adult_users"}]}}


@pytest.fixture
def format_conversion_config():
    """Format conversion configuration."""
    return {
        "format_conversion": {
            "users": {
                "extract_fields": ["name", "age", "salary"],
                "new_name": "user_summary",
            }
        }
    }


@pytest.fixture
def content_summarization_config():
    """Content summarization configuration."""
    return {
        "content_summarization": {
            "users": {
                "method": "group",
                "fields": ["department", "salary", "name"],
                "summarize": ["department"],
            }
        }
    }


@pytest.fixture
def multi_operation_config():
    """Configuration with multiple operations."""
    return {
        "contextual_filtering": {"users": [{"field": "age", "condition": "> 25"}]},
        "format_conversion": {"users": {"extract_fields": ["name", "age", "department"]}},
    }


@pytest.fixture
def graph_execution_config():
    """Configuration with graph-based execution."""
    return {
        "contextual_filtering": {"filter_adults": {"users": [{"field": "age", "condition": "> 25"}]}},
        "format_conversion": {"extract_info": {"users": {"extract_fields": ["name", "age", "salary"]}}},
        "graph": [
            {
                "id": "filter_step",
                "operation": "contextual_filtering",
                "config_key": "filter_adults",
                "depends_on": [],
            },
            {
                "id": "convert_step",
                "operation": "format_conversion",
                "config_key": "extract_info",
                "depends_on": ["filter_step"],
            },
        ],
    }


@pytest.fixture
def invalid_config():
    """Invalid configuration for error testing."""
    return {
        "contextual_filtering": {
            "users": [
                {
                    # Missing required fields
                    "condition": "> 25"
                }
            ]
        }
    }


@pytest.fixture
def empty_config():
    """Empty configuration for testing."""
    return {}


@pytest.fixture
def relationship_config():
    """Relationship highlighting configuration."""
    return {
        "relationship_highlighting": {
            "user_products": {
                "left_source": "users",
                "right_source": "products",
                "join_conditions": [
                    {
                        "left_field": "department",
                        "right_field": "category",
                        "condition": "==",
                    }
                ],
                "new_name": "user_product_matches",
            }
        }
    }
