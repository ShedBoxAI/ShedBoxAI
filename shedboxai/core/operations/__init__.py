"""
Operations package for data processing handlers.

This package contains handlers for different types of data processing operations:
- Filtering: Contextual data filtering
- Conversion: Format conversion and field extraction
- Summarization: Statistical and AI-powered summarization
- Relationships: Relationship detection and highlighting
- Advanced: Grouping, aggregation, sorting, limiting
- Templates: Jinja2-based template processing

Main exports:
- All operation handler classes
- Base OperationHandler class
"""

from .advanced import AdvancedOperationsHandler
from .base import OperationHandler
from .conversion import FormatConversionHandler
from .filtering import ContextualFilteringHandler
from .relationships import RelationshipHighlightingHandler
from .summarization import ContentSummarizationHandler
from .templates import TemplateMatchingHandler

__all__ = [
    "OperationHandler",
    "ContextualFilteringHandler",
    "FormatConversionHandler",
    "ContentSummarizationHandler",
    "RelationshipHighlightingHandler",
    "AdvancedOperationsHandler",
    "TemplateMatchingHandler",
]
