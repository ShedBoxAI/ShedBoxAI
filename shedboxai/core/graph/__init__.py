"""
Graph execution package for processing pipelines.

This package provides graph-based execution capabilities for
complex data processing workflows with dependency management.

Main exports:
- GraphExecutor: Main execution engine for processing graphs
"""

from .executor import GraphExecutor

__all__ = [
    "GraphExecutor",
]
