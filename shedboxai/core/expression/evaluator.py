"""
Expression evaluation engine with AI enhancements.

This module provides the main ExpressionEngine class that coordinates
lexing, parsing, and evaluation of expressions with support for
custom functions, operators, and AI-powered features.
"""

from __future__ import annotations

import datetime
import operator
import re
from typing import Any, Callable, Dict, Optional

from .parser import Parser
from .plugins import PluginManager


class ExpressionEngine:
    """
    Complete expression evaluation engine.

    Provides comprehensive expression evaluation capabilities including:
    - Mathematical, logical, and string operations
    - Custom function and operator registration
    - Template variable substitution
    - Plugin system for extensibility
    """

    def __init__(self, ai_enabled: bool = False):
        """
        Initialize the expression engine.

        Args:
            ai_enabled: Deprecated parameter (AI features removed)
        """
        self._functions = {}
        self._operators = {}
        self._parser = Parser()
        self._ai = None  # AI service removed
        self._plugin_manager = PluginManager(self)

        self._register_core_functions()
        self._register_core_operators()

    @property
    def plugin_manager(self) -> PluginManager:
        """Access the plugin manager."""
        return self._plugin_manager

    def register_function(self, name: str, func: Callable) -> None:
        """
        Register a function with the engine.

        Args:
            name: Function name (case-insensitive)
            func: Callable function to register
        """
        self._functions[name.lower()] = func

    def register_operator(self, symbol: str, func: Callable) -> None:
        """
        Register an operator with the engine.

        Args:
            symbol: Operator symbol (e.g., '+', '==')
            func: Callable function that takes two arguments
        """
        self._operators[symbol] = func

    def unregister_function(self, name: str) -> None:
        """
        Remove a registered function.

        Args:
            name: Function name to remove
        """
        self._functions.pop(name.lower(), None)

    def unregister_operator(self, symbol: str) -> None:
        """
        Remove a registered operator.

        Args:
            symbol: Operator symbol to remove
        """
        self._operators.pop(symbol, None)

    def evaluate(self, expression: str, context: Optional[Dict[str, Any]] = None) -> Any:
        """
        Evaluate an expression with full AI preprocessing.

        Args:
            expression: The expression string to evaluate
            context: Variable context for evaluation

        Returns:
            The evaluated result

        Raises:
            ValueError: If expression is invalid or evaluation fails
        """
        context = context or {}

        # Handle empty expressions
        if not expression.strip():
            raise ValueError("Empty expression")

        # Skip AI natural language processing (removed)

        # Prepare evaluation context
        eval_context = {
            **context,
            "_functions": self._functions,
            "_operators": self._operators,
        }

        try:
            ast = self._parser.parse(expression)
            return ast.evaluate(eval_context)
        except Exception as e:
            raise ValueError(f"Evaluation error: {str(e)}") from e

    def substitute_variables(self, template: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Template variable substitution with expression evaluation.

        Supports {{variable}} syntax with expression evaluation.

        Args:
            template: Template string with {{variable}} placeholders
            context: Variable context for substitution

        Returns:
            Template with variables substituted
        """
        context = context or {}

        def replace_match(match):
            expr = match.group(1).strip()

            try:
                result = self.evaluate(expr, context)
                return str(result)
            except Exception as e:
                return f"{{ERROR: {str(e)}}}"

        return re.sub(r"\{\{(.+?)\}\}", replace_match, template)

    def _register_core_functions(self) -> None:
        """Register all core functions."""
        # Math functions
        self.register_function("sum", sum)
        self.register_function("min", min)
        self.register_function("max", max)
        self.register_function("avg", lambda *args: sum(args) / len(args) if args else 0)
        self.register_function("round", round)
        self.register_function("abs", abs)

        # String functions
        self.register_function("concat", lambda *args: "".join(str(arg) for arg in args))
        self.register_function("upper", lambda s: str(s).upper())
        self.register_function("lower", lambda s: str(s).lower())
        self.register_function("trim", lambda s: str(s).strip())
        self.register_function("length", lambda s: len(str(s)))
        self.register_function(
            "substring",
            lambda s, start, end=None: str(s)[start:end] if end else str(s)[start:],
        )
        self.register_function("replace", lambda s, old, new: str(s).replace(old, new))

        # Date functions
        self.register_function("today", lambda: datetime.date.today())
        self.register_function("now", lambda: datetime.datetime.now())
        self.register_function("year", lambda d: d.year if hasattr(d, "year") else None)
        self.register_function("month", lambda d: d.month if hasattr(d, "month") else None)
        self.register_function("day", lambda d: d.day if hasattr(d, "day") else None)

        # Logical functions
        self.register_function("if", lambda cond, true_val, false_val: true_val if cond else false_val)
        self.register_function("and", lambda *args: all(args))
        self.register_function("or", lambda *args: any(args))
        self.register_function("not", lambda x: not x)

        # Collection functions
        self.register_function("count", len)
        self.register_function(
            "filter",
            lambda arr, key, value: [item for item in arr if item.get(key) == value],
        )
        self.register_function("map", lambda arr, key: [item.get(key) for item in arr])
        self.register_function("first", lambda arr: arr[0] if arr else None)
        self.register_function("last", lambda arr: arr[-1] if arr else None)

        # Type conversion
        self.register_function("to_string", str)
        self.register_function("to_number", float)
        self.register_function("to_int", int)
        self.register_function("to_bool", bool)

    def _register_core_operators(self) -> None:
        """Register all core operators."""
        # Arithmetic operators
        self.register_operator("+", operator.add)
        self.register_operator("-", operator.sub)
        self.register_operator("*", operator.mul)
        self.register_operator("/", operator.truediv)
        self.register_operator("%", operator.mod)
        self.register_operator("**", operator.pow)

        # Comparison operators
        self.register_operator("==", operator.eq)
        self.register_operator("!=", operator.ne)
        self.register_operator(">", operator.gt)
        self.register_operator(">=", operator.ge)
        self.register_operator("<", operator.lt)
        self.register_operator("<=", operator.le)

        # Logical operators
        self.register_operator("&&", lambda a, b: a and b)
        self.register_operator("||", lambda a, b: a or b)
        self.register_operator("!", operator.not_)
