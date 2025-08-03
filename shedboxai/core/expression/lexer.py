"""
Lexical analysis for expression parsing.

This module provides tokenization functionality for the expression engine,
breaking down expression strings into structured tokens for parsing.
"""

from __future__ import annotations

import re
from typing import Any, List


class Token:
    """Represents a token in an expression."""

    def __init__(self, type_: str, value: Any, position: int = 0):
        """
        Initialize a token.

        Args:
            type_: The token type (e.g., 'NUMBER', 'IDENTIFIER', 'OPERATOR')
            value: The token value
            position: Position in the original expression
        """
        self.type = type_
        self.value = value
        self.position = position

    def __repr__(self) -> str:
        return f"Token({self.type}, {self.value}, {self.position})"


class Lexer:
    """Tokenizes expression strings into structured tokens."""

    TOKEN_TYPES = {
        "NUMBER": r"\d+(\.\d+)?",
        "STRING": r"\"([^\"\\]*(\\.[^\"\\]*)*)\"",
        "IDENTIFIER": r"[a-zA-Z_][a-zA-Z0-9_]*",
        "DOT": r"\.",
        "OPERATOR": r"[\+\-\*\/\%\=\<\>\!\&\|\^]+",
        "LEFT_PAREN": r"\(",
        "RIGHT_PAREN": r"\)",
        "LEFT_BRACKET": r"\[",
        "RIGHT_BRACKET": r"\]",
        "COMMA": r",",
        "COLON": r":",
        "WHITESPACE": r"\s+",
    }

    def __init__(self):
        """Initialize the lexer with compiled regex patterns."""
        self.pattern = "|".join(f"(?P<{name}>{pattern})" for name, pattern in self.TOKEN_TYPES.items())
        self.regex = re.compile(self.pattern)

    def tokenize(self, expression: str) -> List[Token]:
        """
        Tokenize an expression string into a list of tokens.

        Args:
            expression: The expression string to tokenize

        Returns:
            List of Token objects

        Raises:
            ValueError: If the expression contains invalid syntax
        """
        tokens = []
        position = 0

        while position < len(expression):
            match = self.regex.match(expression, position)
            if not match:
                raise ValueError(f"Invalid syntax at position {position}")

            token_type = match.lastgroup
            token_value = match.group()

            if token_type == "WHITESPACE":
                position = match.end()
                continue

            # Process token values (numbers, strings, etc.)
            if token_type == "NUMBER":
                token_value = float(token_value) if "." in token_value else int(token_value)
            elif token_type == "STRING":
                token_value = token_value[1:-1].replace('\\"', '"')

            tokens.append(Token(token_type, token_value, position))
            position = match.end()

        return tokens
