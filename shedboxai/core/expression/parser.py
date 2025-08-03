"""
Expression parsing and Abstract Syntax Tree (AST) nodes.

This module provides parsing functionality that converts tokens into
an Abstract Syntax Tree for evaluation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .lexer import Lexer, Token


class ASTNode:
    """Base class for all AST nodes."""

    def evaluate(self, context: Dict[str, Any]) -> Any:
        """
        Evaluate this node in the given context.

        Args:
            context: Variable and function context for evaluation

        Returns:
            The evaluated result

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement evaluate()")


class LiteralNode(ASTNode):
    """Node representing a literal value (number, string, etc.)."""

    def __init__(self, value: Any):
        """
        Initialize with a literal value.

        Args:
            value: The literal value to store
        """
        self.value = value

    def evaluate(self, context: Dict[str, Any]) -> Any:
        """Return the literal value."""
        return self.value

    def __repr__(self) -> str:
        return f"Literal({self.value})"


class VariableNode(ASTNode):
    """Node representing a variable reference with dot notation support."""

    def __init__(self, name: str):
        """
        Initialize with a variable name.

        Args:
            name: Variable name, potentially with dot notation (e.g., 'item.field')
        """
        self.name = name

    def evaluate(self, context: Dict[str, Any]) -> Any:
        """
        Resolve variable reference in context.

        Supports dot notation for nested property access.
        """
        parts = self.name.split(".")
        current = context

        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current

    def __repr__(self) -> str:
        return f"Variable({self.name})"


class BinaryOpNode(ASTNode):
    """Node representing a binary operation (e.g., +, -, ==, etc.)."""

    def __init__(self, left: ASTNode, operator: str, right: ASTNode):
        """
        Initialize with left/right operands and operator.

        Args:
            left: Left operand AST node
            operator: Operator symbol
            right: Right operand AST node
        """
        self.left = left
        self.operator = operator
        self.right = right

    def evaluate(self, context: Dict[str, Any]) -> Any:
        """
        Evaluate binary operation.

        Looks up operator function in context and applies it to operands.
        """
        left_val = self.left.evaluate(context)
        right_val = self.right.evaluate(context)
        op_func = context.get("_operators", {}).get(self.operator)
        if not op_func:
            raise ValueError(f"Unknown operator: {self.operator}")
        return op_func(left_val, right_val)

    def __repr__(self) -> str:
        return f"BinaryOp({self.left}, {self.operator}, {self.right})"


class FunctionCallNode(ASTNode):
    """Node representing a function call with arguments."""

    def __init__(self, name: str, args: List[ASTNode]):
        """
        Initialize with function name and arguments.

        Args:
            name: Function name
            args: List of argument AST nodes
        """
        self.name = name
        self.args = args

    def evaluate(self, context: Dict[str, Any]) -> Any:
        """
        Evaluate function call.

        Looks up function in context and calls it with evaluated arguments.
        """
        func = context.get("_functions", {}).get(self.name.lower())
        if not func:
            raise ValueError(f"Unknown function: {self.name}")
        evaluated_args = [arg.evaluate(context) for arg in self.args]
        return func(*evaluated_args)

    def __repr__(self) -> str:
        return f"FunctionCall({self.name}, {self.args})"


class Parser:
    """
    Recursive descent parser that converts tokens into an AST.

    Handles operator precedence and creates appropriate AST nodes
    for evaluation.
    """

    OPERATOR_PRECEDENCE = {
        "||": 10,
        "&&": 20,
        "==": 30,
        "!=": 30,
        "<": 40,
        "<=": 40,
        ">": 40,
        ">=": 40,
        "+": 50,
        "-": 50,
        "*": 60,
        "/": 60,
        "%": 60,
        "**": 70,
        "!": 80,
    }

    def __init__(self, lexer: Optional[Lexer] = None):
        """
        Initialize parser with optional lexer.

        Args:
            lexer: Lexer instance, creates default if None
        """
        self.lexer = lexer or Lexer()

    def parse(self, expression: str) -> ASTNode:
        """
        Parse an expression string into an AST.

        Args:
            expression: The expression string to parse

        Returns:
            Root AST node

        Raises:
            ValueError: If expression has syntax errors
        """
        tokens = self.lexer.tokenize(expression)
        self.tokens = tokens
        self.position = 0
        return self._parse_expression()

    def _parse_expression(self, precedence: int = 0) -> ASTNode:
        """Parse expression with operator precedence."""
        left = self._parse_term()

        while (token := self._current_token()) and token.type == "OPERATOR":
            op = token.value
            op_precedence = self.OPERATOR_PRECEDENCE.get(op, 0)

            if op_precedence <= precedence:
                break

            self._next_token()
            right = self._parse_expression(op_precedence)
            left = BinaryOpNode(left, op, right)

        return left

    def _current_token(self) -> Optional[Token]:
        """Get current token without advancing position."""
        if self.position < len(self.tokens):
            return self.tokens[self.position]
        return None

    def _next_token(self) -> Optional[Token]:
        """Advance to next token and return it."""
        self.position += 1
        if self.position < len(self.tokens):
            return self.tokens[self.position]
        return None

    def _parse_term(self) -> ASTNode:
        """Parse arithmetic terms (* / %)."""
        left = self._parse_factor()
        while self._current_token() and self._current_token().type == "OPERATOR":
            op = self._current_token().value
            op_precedence = self.OPERATOR_PRECEDENCE.get(op, 0)

            if op_precedence < 50:  # Only handle */% at this level
                break

            self._next_token()  # Consume operator
            right = self._parse_factor()
            left = BinaryOpNode(left, op, right)
        return left

    def _parse_factor(self) -> ASTNode:
        """Parse primary expressions (literals, variables, function calls, parentheses)."""
        token = self._current_token()
        if not token:
            raise ValueError("Unexpected end of expression")

        if token.type == "NUMBER":
            self._next_token()
            return LiteralNode(token.value)

        elif token.type == "STRING":
            self._next_token()
            return LiteralNode(token.value)

        elif token.type == "IDENTIFIER":
            name = token.value
            self._next_token()

            # Check for property access (DOT)
            if self._current_token() and self._current_token().type == "DOT":
                property_path = [name]
                while self._current_token() and self._current_token().type == "DOT":
                    self._next_token()  # Consume DOT
                    if not self._current_token() or self._current_token().type != "IDENTIFIER":
                        raise ValueError(f"Expected identifier after '.' but got {self._current_token()}")
                    property_path.append(self._current_token().value)
                    self._next_token()  # Consume identifier

                full_path = ".".join(property_path)
                return VariableNode(full_path)

            # Check for function call
            if self._current_token() and self._current_token().type == "LEFT_PAREN":
                self._next_token()  # Consume opening parenthesis
                args = self._parse_arguments()
                if not self._current_token() or self._current_token().type != "RIGHT_PAREN":
                    raise ValueError("Expected closing parenthesis")
                self._next_token()  # Consume closing parenthesis
                return FunctionCallNode(name, args)

            return VariableNode(name)

        elif token.type == "LEFT_PAREN":
            self._next_token()  # Consume opening parenthesis
            expr = self._parse_expression()
            if not self._current_token() or self._current_token().type != "RIGHT_PAREN":
                raise ValueError("Expected closing parenthesis")
            self._next_token()  # Consume closing parenthesis
            return expr

        else:
            raise ValueError(f"Unexpected token: {token}")

    def _parse_arguments(self) -> List[ASTNode]:
        """Parse function call arguments."""
        args = []

        # Empty argument list
        if self._current_token() and self._current_token().type == "RIGHT_PAREN":
            return args

        # Parse first argument
        args.append(self._parse_expression())

        # Parse remaining arguments
        while self._current_token() and self._current_token().type == "COMMA":
            self._next_token()  # Consume comma
            args.append(self._parse_expression())

        return args
