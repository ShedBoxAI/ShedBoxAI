"""
Refactored data processor - clean interface to modular components.

This module provides backward compatibility while using the new
modular architecture underneath.
"""

# Import everything from the new modular structure
from .config import (
    AdvancedOperationConfig,
    ContentSummarizationConfig,
    ContextualFilterConfig,
    FormatConversionConfig,
    GraphNode,
    ProcessorConfig,
    RelationshipConfig,
    TemplateMatchingConfig,
)
from .processor_store import DataProcessor

# Maintain backward compatibility by re-exporting key classes
__all__ = [
    "DataProcessor",
    "ProcessorConfig",
    "ContextualFilterConfig",
    "FormatConversionConfig",
    "ContentSummarizationConfig",
    "RelationshipConfig",
    "AdvancedOperationConfig",
    "TemplateMatchingConfig",
    "GraphNode",
]
