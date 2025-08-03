"""
ShedBoxAI Introspection Module

This module provides data source introspection capabilities for generating
LLM-optimized schema documentation.
"""

from .engine import IntrospectionEngine
from .models import (
    ColumnInfo,
    CSVAnalysis,
    EndpointInfo,
    IntrospectionOptions,
    IntrospectionResult,
    JSONAnalysis,
    PaginationInfo,
    Relationship,
    RESTAnalysis,
    SchemaInfo,
    SizeInfo,
    SourceAnalysis,
    TextAnalysis,
    YAMLAnalysis,
)

__all__ = [
    "IntrospectionOptions",
    "SourceAnalysis",
    "CSVAnalysis",
    "RESTAnalysis",
    "JSONAnalysis",
    "YAMLAnalysis",
    "TextAnalysis",
    "SizeInfo",
    "SchemaInfo",
    "ColumnInfo",
    "Relationship",
    "PaginationInfo",
    "EndpointInfo",
    "IntrospectionResult",
    "IntrospectionEngine",
]
