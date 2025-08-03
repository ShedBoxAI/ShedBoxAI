"""
Expression engine package.

This package provides a complete expression evaluation system with:
- Lexical analysis and parsing
- AST-based evaluation
- Plugin system for extensibility
- AI-powered enhancements

Main exports:
- ExpressionEngine: The main engine class
- ExpressionPlugin: Base class for plugins
- Token, Lexer, Parser: For advanced usage
"""

from .evaluator import ExpressionEngine
from .lexer import Lexer, Token
from .parser import Parser
from .plugins import ExpressionPlugin, PluginManager

__all__ = [
    "ExpressionEngine",
    "ExpressionPlugin",
    "PluginManager",
    "Lexer",
    "Token",
    "Parser",
]
