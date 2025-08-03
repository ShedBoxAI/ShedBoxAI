"""
Core processing package for ShedBoxAI.

This package provides a modular, clean architecture for data processing
with the following components:

- Expression engine: Lexing, parsing, and evaluation of expressions
- Configuration: Pydantic models and normalization functions
- Operations: Specialized handlers for different processing operations
- Graph execution: Dependency-aware processing workflow management

Main exports:
- DataProcessor: Main data processing orchestrator
- ExpressionEngine: Expression evaluation engine
- ProcessorConfig: Main configuration class
- All operation handlers for custom usage
"""

# Import the new modular components
from .config import ProcessorConfig
from .expression import ExpressionEngine
from .graph import GraphExecutor
from .operations import (
    AdvancedOperationsHandler,
    ContentSummarizationHandler,
    ContextualFilteringHandler,
    FormatConversionHandler,
    OperationHandler,
    RelationshipHighlightingHandler,
    TemplateMatchingHandler,
)

# Import the refactored processor
try:
    from .processor_store import DataProcessor
except ImportError:
    # Fallback to original processor if new one fails
    from .processor import DataProcessor

__all__ = [
    # Main classes
    "DataProcessor",
    "ExpressionEngine",
    "ProcessorConfig",
    "GraphExecutor",
    # Operation handlers
    "OperationHandler",
    "ContextualFilteringHandler",
    "FormatConversionHandler",
    "ContentSummarizationHandler",
    "RelationshipHighlightingHandler",
    "AdvancedOperationsHandler",
    "TemplateMatchingHandler",
]
