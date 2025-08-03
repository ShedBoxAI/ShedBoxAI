"""
Refactored expression engine - clean interface to modular components.

This module provides backward compatibility while using the new
modular expression engine architecture underneath.
"""

# Import everything from the new modular structure
from .expression import ExpressionEngine, ExpressionPlugin, Lexer, Parser, PluginManager, Token

# The new modular structure provides all the same classes
# with better organization and cleaner code
__all__ = [
    "ExpressionEngine",
    "ExpressionPlugin",
    "PluginManager",
    "Token",
    "Lexer",
    "Parser",
]
