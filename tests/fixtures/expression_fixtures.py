"""
Expression engine test fixtures.

This module provides reusable fixtures for testing the expression engine
components including lexer, parser, evaluator, and plugins.
"""

from typing import Any, Dict, List

import pytest

from shedboxai.core.expression.evaluator import ExpressionEngine
from shedboxai.core.expression.lexer import Lexer, Token
from shedboxai.core.expression.parser import Parser


# Lexer Fixtures
@pytest.fixture
def fresh_lexer():
    """Create a fresh lexer instance for each test."""
    return Lexer()


@pytest.fixture
def sample_tokens():
    """Sample tokens for parser testing."""
    return [
        Token("NUMBER", 42, 0),
        Token("OPERATOR", "+", 3),
        Token("NUMBER", 3.14, 5),
        Token("OPERATOR", "*", 10),
        Token("IDENTIFIER", "variable", 12),
    ]


@pytest.fixture
def complex_tokens():
    """Complex token sequence for advanced parsing tests."""
    return [
        Token("IDENTIFIER", "if", 0),
        Token("LEFT_PAREN", "(", 2),
        Token("IDENTIFIER", "age", 3),
        Token("OPERATOR", ">", 7),
        Token("NUMBER", 25, 9),
        Token("COMMA", ",", 11),
        Token("STRING", "adult", 13),
        Token("COMMA", ",", 20),
        Token("STRING", "minor", 22),
        Token("RIGHT_PAREN", ")", 29),
    ]


# Parser Fixtures
@pytest.fixture
def fresh_parser():
    """Create a fresh parser instance for each test."""
    return Parser()


@pytest.fixture
def parser_with_lexer(fresh_lexer):
    """Create parser with specific lexer instance."""
    return Parser(fresh_lexer)


# Expression Test Data Fixtures
@pytest.fixture
def simple_expressions():
    """Simple expressions for basic testing."""
    return {
        "arithmetic": ["2 + 3", "10 - 5", "4 * 6", "15 / 3", "17 % 5", "2 ** 3"],
        "comparison": ["5 > 3", "10 <= 20", 'name == "John"', 'status != "inactive"'],
        "logical": ["true && false", 'age > 18 || status == "verified"', "!(disabled)"],
    }


@pytest.fixture
def complex_expressions():
    """Complex expressions for advanced testing."""
    return {
        "nested": [
            "(2 + 3) * (4 - 1)",
            "((a + b) * c) / (d - e)",
            "if(condition, true_value, false_value)",
        ],
        "function_calls": [
            "sum(1, 2, 3, 4)",
            'concat("Hello", " ", "World")',
            'if(age > 18, "adult", "minor")',
            "max(score1, score2, score3)",
        ],
        "property_access": [
            "user.name",
            "user.profile.email",
            "data.results.items.first",
        ],
        "mixed": [
            'user.age > 25 && user.status == "active"',
            "sum(scores) / count(scores) > threshold",
            'concat(user.first_name, " ", user.last_name)',
        ],
    }


@pytest.fixture
def invalid_expressions():
    """Invalid expressions for error testing."""
    return [
        "",  # Empty expression
        "2 +",  # Incomplete expression
        "((2 + 3)",  # Unmatched parentheses
        "2 + + 3",  # Invalid operator sequence
        "unknown_function()",  # Unknown function
        "2 $ 3",  # Invalid operator
        '"unclosed string',  # Unclosed string
        "variable.",  # Incomplete property access
        "func(1, 2,)",  # Trailing comma
    ]


# Evaluation Context Fixtures
@pytest.fixture
def basic_context():
    """Basic evaluation context with common variables."""
    return {
        "age": 30,
        "name": "John Doe",
        "status": "active",
        "salary": 75000,
        "active": True,
        "score": 85.5,
    }


@pytest.fixture
def nested_context():
    """Context with nested objects for property access testing."""
    return {
        "user": {
            "id": 123,
            "name": "Jane Smith",
            "email": "jane@example.com",
            "profile": {
                "age": 28,
                "department": "Engineering",
                "skills": ["Python", "JavaScript", "SQL"],
            },
        },
        "settings": {"theme": "dark", "notifications": {"email": True, "push": False}},
    }


@pytest.fixture
def array_context():
    """Context with arrays for collection function testing."""
    return {
        "numbers": [1, 2, 3, 4, 5],
        "names": ["Alice", "Bob", "Charlie"],
        "scores": [95.5, 87.2, 92.1, 78.9],
        "users": [
            {"name": "John", "age": 25, "active": True},
            {"name": "Jane", "age": 30, "active": False},
            {"name": "Bob", "age": 35, "active": True},
        ],
    }


# Engine Fixtures
@pytest.fixture
def basic_engine():
    """Basic expression engine."""
    return ExpressionEngine(ai_enabled=False)


@pytest.fixture
def engine_with_custom_functions(basic_engine):
    """Engine with custom functions registered."""
    # Register custom functions
    basic_engine.register_function("double", lambda x: x * 2)
    basic_engine.register_function("greet", lambda name: f"Hello, {name}!")
    basic_engine.register_function("is_even", lambda x: x % 2 == 0)
    return basic_engine


@pytest.fixture
def engine_with_custom_operators(basic_engine):
    """Engine with custom operators registered."""
    # Register custom operators
    basic_engine.register_operator("~", lambda a, b: f"{a}~{b}")  # String concat with ~
    basic_engine.register_operator("%%", lambda a, b: (a + b) % 2)  # Custom modulo
    return basic_engine


# Expected Results Fixtures
@pytest.fixture
def expected_arithmetic_results():
    """Expected results for arithmetic expressions."""
    return {
        "2 + 3": 5,
        "10 - 5": 5,
        "4 * 6": 24,
        "15 / 3": 5.0,
        "17 % 5": 2,
        "2 ** 3": 8,
    }


@pytest.fixture
def expected_comparison_results():
    """Expected results for comparison expressions with basic context."""
    return {
        "age > 25": True,  # age = 30 in basic_context
        "age <= 25": False,
        'name == "John Doe"': True,
        'status != "inactive"': True,
    }


@pytest.fixture
def expected_function_results():
    """Expected results for function calls."""
    return {
        "sum(1, 2, 3, 4)": 10,
        "max(1, 5, 3)": 5,
        "min(1, 5, 3)": 1,
        'concat("Hello", " ", "World")': "Hello World",
        'upper("hello")': "HELLO",
        'lower("WORLD")': "world",
    }


# Error Test Fixtures
@pytest.fixture
def tokenization_error_cases():
    """Test cases that should cause tokenization errors."""
    return [
        ("Invalid char @", "valid + @invalid"),
        ("Unclosed string", '"unclosed string'),
        ("Multiple decimals", "1.2.3"),
    ]


@pytest.fixture
def parsing_error_cases():
    """Test cases that should cause parsing errors."""
    return [
        ("Missing operand", "2 +"),
        ("Unmatched parentheses", "((2 + 3)"),
        ("Invalid operator sequence", "2 + + 3"),
        ("Empty function args", "func(,)"),
        ("Missing function closing", "func(1, 2"),
    ]


@pytest.fixture
def evaluation_error_cases():
    """Test cases that should cause evaluation errors."""
    return [
        ("Unknown function", "unknown_func(1)"),
        ("Unknown variable", "undefined_var"),
        ("Division by zero", "10 / 0"),
        ("Unknown operator", "2 $ 3"),
        ("Type mismatch", '"string" + true'),
    ]


# Performance Test Fixtures
@pytest.fixture
def performance_expressions():
    """Expressions for performance testing."""
    return {
        "simple": "2 + 2",
        "medium": "sum(1, 2, 3, 4, 5) * max(10, 20, 30)",
        "complex": 'if(user.age > 25 && user.status == "active", ' "sum(user.scores), avg(user.fallback_scores))",
        "very_complex": (
            "concat(user.first_name, "
            '" ", user.last_name, " - ", if(user.premium, "Premium User", "Standard User"), '
            '" (Score: ", round(avg(user.monthly_scores)), ")")'
        ),
    }


@pytest.fixture
def large_context():
    """Large context for performance testing."""
    context = {}

    # Add many variables
    for i in range(1000):
        context[f"var_{i}"] = i

    # Add nested structures
    context["user"] = {
        "scores": list(range(100)),
        "monthly_scores": [i * 1.5 for i in range(12)],
        "first_name": "Performance",
        "last_name": "Test",
        "premium": True,
        "age": 30,
        "status": "active",
    }

    return context


# Template Testing Fixtures
@pytest.fixture
def template_test_cases():
    """Template strings for substitution testing."""
    return {
        "simple": "Hello {{name}}!",
        "multiple": "{{name}} is {{age}} years old and works in {{department}}",
        "expressions": ("Total: {{price + tax}} ({{if(discount > 0, " '"with discount", "no discount")}})'),
        "nested": "User: {{user.name}} ({{user.profile.department}})",
        "mixed": "Welcome {{user.name}}! Your score is {{round(user.score * 100)}}%",
    }


@pytest.fixture
def template_contexts():
    """Contexts for template testing."""
    return {
        "basic": {"name": "Alice", "age": 28, "department": "Engineering"},
        "with_calculations": {"price": 100, "tax": 15, "discount": 10},
        "nested": {"user": {"name": "Bob", "score": 0.875, "profile": {"department": "Sales"}}},
    }
