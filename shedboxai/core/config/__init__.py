"""
Configuration package for data processing operations.

This package contains:
- Pydantic models for configuration validation
- Normalization functions for consistent configuration structure

Main exports:
- All configuration model classes
- All normalization functions
- ProcessorConfig: Main configuration class
"""

from .models import (
    AdvancedOperationConfig,
    ContentSummarizationConfig,
    ContextualFilterConfig,
    FormatConversionConfig,
    GraphNode,
    ProcessorConfig,
    RelationshipConfig,
    TemplateMatchingConfig,
)
from .normalizers import (
    normalize_advanced_operations_config,
    normalize_content_summarization_config,
    normalize_contextual_filtering_config,
    normalize_format_conversion_config,
    normalize_relationship_highlighting_config,
    normalize_template_matching_config,
)

__all__ = [
    # Models
    "ProcessorConfig",
    "ContextualFilterConfig",
    "FormatConversionConfig",
    "ContentSummarizationConfig",
    "RelationshipConfig",
    "AdvancedOperationConfig",
    "TemplateMatchingConfig",
    "GraphNode",
    # Normalizers
    "normalize_contextual_filtering_config",
    "normalize_format_conversion_config",
    "normalize_content_summarization_config",
    "normalize_relationship_highlighting_config",
    "normalize_advanced_operations_config",
    "normalize_template_matching_config",
]
