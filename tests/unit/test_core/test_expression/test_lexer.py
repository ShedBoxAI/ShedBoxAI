"""
Comprehensive tests for the expression lexer module.

Tests tokenization functionality including all token types,
edge cases, error conditions, and proper token attributes.
"""

import pytest

from shedboxai.core.expression.lexer import Lexer, Token


class TestToken:
    """Test the Token class functionality."""

    def test_token_creation(self):
        """Test basic token creation with all parameters."""
        token = Token("NUMBER", 42, 10)
        assert token.type == "NUMBER"
        assert token.value == 42
        assert token.position == 10

    def test_token_repr(self):
        """Test token string representation."""
        token = Token("IDENTIFIER", "variable", 5)
        expected = "Token(IDENTIFIER, variable, 5)"
        assert repr(token) == expected

    def test_token_default_position(self):
        """Test token creation with default position."""
        token = Token("STRING", "hello")
        assert token.position == 0


class TestLexerInitialization:
    """Test lexer initialization and setup."""

    def test_lexer_initialization(self, fresh_lexer):
        """Test that lexer initializes with proper patterns."""
        assert fresh_lexer.TOKEN_TYPES is not None
        assert fresh_lexer.pattern is not None
        assert fresh_lexer.regex is not None

    def test_token_types_defined(self, fresh_lexer):
        """Test that all expected token types are defined."""
        expected_types = {
            "NUMBER",
            "STRING",
            "IDENTIFIER",
            "DOT",
            "OPERATOR",
            "LEFT_PAREN",
            "RIGHT_PAREN",
            "LEFT_BRACKET",
            "RIGHT_BRACKET",
            "COMMA",
            "COLON",
            "WHITESPACE",
        }
        assert set(fresh_lexer.TOKEN_TYPES.keys()) == expected_types


class TestBasicTokenization:
    """Test basic tokenization functionality."""

    def test_empty_expression(self, fresh_lexer):
        """Test tokenizing empty expression."""
        tokens = fresh_lexer.tokenize("")
        assert tokens == []

    def test_whitespace_only(self, fresh_lexer):
        """Test that whitespace is properly filtered out."""
        tokens = fresh_lexer.tokenize("   \t\n  ")
        assert tokens == []

    def test_single_number_integer(self, fresh_lexer):
        """Test tokenizing single integer."""
        tokens = fresh_lexer.tokenize("42")
        assert len(tokens) == 1
        assert tokens[0].type == "NUMBER"
        assert tokens[0].value == 42

    def test_single_number_float(self, fresh_lexer):
        """Test tokenizing single float."""
        tokens = fresh_lexer.tokenize("3.14")
        assert len(tokens) == 1
        assert tokens[0].type == "NUMBER"
        assert tokens[0].value == 3.14

    def test_single_string(self, fresh_lexer):
        """Test tokenizing single string literal."""
        tokens = fresh_lexer.tokenize('"hello world"')
        assert len(tokens) == 1
        assert tokens[0].type == "STRING"
        assert tokens[0].value == "hello world"

    def test_single_identifier(self, fresh_lexer):
        """Test tokenizing single identifier."""
        tokens = fresh_lexer.tokenize("variable_name")
        assert len(tokens) == 1
        assert tokens[0].type == "IDENTIFIER"
        assert tokens[0].value == "variable_name"


class TestNumberTokenization:
    """Test number tokenization in detail."""

    @pytest.mark.parametrize(
        "number_str,expected_value,expected_type",
        [
            ("0", 0, int),
            ("123", 123, int),
            ("456789", 456789, int),
            ("0.0", 0.0, float),
            ("3.14159", 3.14159, float),
            ("0.5", 0.5, float),
            ("100.25", 100.25, float),
        ],
    )
    def test_number_parsing(self, fresh_lexer, number_str, expected_value, expected_type):
        """Test various number formats are parsed correctly."""
        tokens = fresh_lexer.tokenize(number_str)
        assert len(tokens) == 1
        assert tokens[0].type == "NUMBER"
        assert tokens[0].value == expected_value
        assert type(tokens[0].value) == expected_type


class TestStringTokenization:
    """Test string tokenization including edge cases."""

    def test_empty_string(self, fresh_lexer):
        """Test tokenizing empty string literal."""
        tokens = fresh_lexer.tokenize('""')
        assert len(tokens) == 1
        assert tokens[0].type == "STRING"
        assert tokens[0].value == ""

    def test_string_with_spaces(self, fresh_lexer):
        """Test string containing spaces."""
        tokens = fresh_lexer.tokenize('"hello world test"')
        assert len(tokens) == 1
        assert tokens[0].value == "hello world test"

    def test_string_with_escaped_quotes(self, fresh_lexer):
        """Test string containing escaped quotes."""
        tokens = fresh_lexer.tokenize(r'"He said \"hello\" to me"')
        assert len(tokens) == 1
        assert tokens[0].value == 'He said "hello" to me'

    def test_string_with_special_characters(self, fresh_lexer):
        """Test string with various special characters."""
        test_string = '"Special: !@#$%^&*()_+-=[]{}|;:,.<>?`~"'
        tokens = fresh_lexer.tokenize(test_string)
        assert len(tokens) == 1
        assert tokens[0].value == "Special: !@#$%^&*()_+-=[]{}|;:,.<>?`~"


class TestIdentifierTokenization:
    """Test identifier tokenization rules."""

    @pytest.mark.parametrize(
        "identifier",
        [
            "simple",
            "with_underscore",
            "camelCase",
            "PascalCase",
            "var123",
            "_private",
            "__dunder__",
            "a",
            "very_long_variable_name_with_many_parts",
        ],
    )
    def test_valid_identifiers(self, fresh_lexer, identifier):
        """Test various valid identifier formats."""
        tokens = fresh_lexer.tokenize(identifier)
        assert len(tokens) == 1
        assert tokens[0].type == "IDENTIFIER"
        assert tokens[0].value == identifier


class TestOperatorTokenization:
    """Test operator tokenization."""

    @pytest.mark.parametrize(
        "operator",
        [
            "+",
            "-",
            "*",
            "/",
            "%",
            "=",
            "<",
            ">",
            "!",
            "&",
            "|",
            "^",
            "==",
            "!=",
            "<=",
            ">=",
            "&&",
            "||",
            "**",
        ],
    )
    def test_operators(self, fresh_lexer, operator):
        """Test various operators are tokenized correctly."""
        tokens = fresh_lexer.tokenize(operator)
        assert len(tokens) == 1
        assert tokens[0].type == "OPERATOR"
        assert tokens[0].value == operator


class TestPunctuationTokenization:
    """Test punctuation and delimiter tokenization."""

    @pytest.mark.parametrize(
        "symbol,expected_type",
        [
            (".", "DOT"),
            ("(", "LEFT_PAREN"),
            (")", "RIGHT_PAREN"),
            ("[", "LEFT_BRACKET"),
            ("]", "RIGHT_BRACKET"),
            (",", "COMMA"),
            (":", "COLON"),
        ],
    )
    def test_punctuation(self, fresh_lexer, symbol, expected_type):
        """Test punctuation symbols are correctly categorized."""
        tokens = fresh_lexer.tokenize(symbol)
        assert len(tokens) == 1
        assert tokens[0].type == expected_type
        assert tokens[0].value == symbol


class TestComplexExpressions:
    """Test tokenization of complex multi-token expressions."""

    def test_arithmetic_expression(self, fresh_lexer):
        """Test simple arithmetic expression."""
        tokens = fresh_lexer.tokenize("2 + 3 * 4")
        expected = [
            ("NUMBER", 2),
            ("OPERATOR", "+"),
            ("NUMBER", 3),
            ("OPERATOR", "*"),
            ("NUMBER", 4),
        ]
        assert len(tokens) == 5
        for i, (exp_type, exp_value) in enumerate(expected):
            assert tokens[i].type == exp_type
            assert tokens[i].value == exp_value

    def test_function_call_expression(self, fresh_lexer):
        """Test function call tokenization."""
        tokens = fresh_lexer.tokenize('sum(a, b, "test")')
        expected_types = [
            "IDENTIFIER",
            "LEFT_PAREN",
            "IDENTIFIER",
            "COMMA",
            "IDENTIFIER",
            "COMMA",
            "STRING",
            "RIGHT_PAREN",
        ]
        assert len(tokens) == 8
        for i, exp_type in enumerate(expected_types):
            assert tokens[i].type == exp_type

    def test_dot_notation_expression(self, fresh_lexer):
        """Test property access with dot notation."""
        tokens = fresh_lexer.tokenize("user.profile.name")
        expected = [
            ("IDENTIFIER", "user"),
            ("DOT", "."),
            ("IDENTIFIER", "profile"),
            ("DOT", "."),
            ("IDENTIFIER", "name"),
        ]
        assert len(tokens) == 5
        for i, (exp_type, exp_value) in enumerate(expected):
            assert tokens[i].type == exp_type
            assert tokens[i].value == exp_value

    def test_comparison_expression(self, fresh_lexer):
        """Test comparison expression."""
        tokens = fresh_lexer.tokenize('age >= 21 && status == "active"')
        expected_types = [
            "IDENTIFIER",
            "OPERATOR",
            "NUMBER",
            "OPERATOR",
            "IDENTIFIER",
            "OPERATOR",
            "STRING",
        ]
        assert len(tokens) == 7
        for i, exp_type in enumerate(expected_types):
            assert tokens[i].type == exp_type


class TestTokenPositions:
    """Test that token positions are correctly tracked."""

    def test_position_tracking(self, fresh_lexer):
        """Test that token positions are accurately tracked."""
        expression = "var + 123"
        tokens = fresh_lexer.tokenize(expression)

        # Expected positions based on the expression
        expected_positions = [0, 4, 6]  # 'var' at 0, '+' at 4, '123' at 6

        assert len(tokens) == 3
        for i, expected_pos in enumerate(expected_positions):
            assert tokens[i].position == expected_pos

    def test_position_with_whitespace(self, fresh_lexer):
        """Test position tracking with various whitespace."""
        expression = "  a   +   b  "
        tokens = fresh_lexer.tokenize(expression)

        # Positions should skip whitespace
        expected_positions = [2, 6, 10]  # After leading spaces

        assert len(tokens) == 3
        for i, expected_pos in enumerate(expected_positions):
            assert tokens[i].position == expected_pos


class TestErrorConditions:
    """Test lexer error handling and edge cases."""

    def test_error_cases(self, fresh_lexer):
        """Test various tokenization error conditions."""
        # Test specific cases that should fail
        with pytest.raises(ValueError):
            fresh_lexer.tokenize("valid + @invalid")  # @ not in token patterns
        with pytest.raises(ValueError):
            fresh_lexer.tokenize('"unclosed string')  # Unclosed string

    def test_position_in_error_message(self, fresh_lexer):
        """Test that error messages include position information."""
        try:
            fresh_lexer.tokenize("good + #bad")
        except ValueError as e:
            assert "position" in str(e)
            assert "7" in str(e)  # Position of invalid character


@pytest.mark.integration
class TestLexerIntegration:
    """Integration tests for lexer with realistic expressions."""

    def test_realistic_filtering_expression(self, fresh_lexer):
        """Test lexing a realistic data filtering expression."""
        expr = 'age > 25 && (status == "active" || priority >= 3)'
        tokens = fresh_lexer.tokenize(expr)

        # Should tokenize without errors
        assert len(tokens) > 0

        # Check key tokens are present
        token_values = [t.value for t in tokens]
        assert "age" in token_values
        assert 25 in token_values
        assert ">" in token_values
        assert "active" in token_values

    def test_complex_function_call(self, fresh_lexer):
        """Test lexing complex nested function calls."""
        expr = 'if(sum(values) > avg(benchmarks), "high", lower(concat("low_", type)))'
        tokens = fresh_lexer.tokenize(expr)

        # Should parse without errors
        assert len(tokens) > 0

        # Verify function names are identified
        identifiers = [t.value for t in tokens if t.type == "IDENTIFIER"]
        assert "if" in identifiers
        assert "sum" in identifiers
        assert "avg" in identifiers

    def test_template_variable_expression(self, fresh_lexer):
        """Test lexing template-style expressions."""
        expr = 'user.profile.name + " (" + user.profile.email + ")"'
        tokens = fresh_lexer.tokenize(expr)

        # Should handle dot notation and string concatenation
        assert len(tokens) > 0

        # Check for proper dot notation tokenization
        dot_tokens = [t for t in tokens if t.type == "DOT"]
        assert len(dot_tokens) == 4  # Two dot notations = 4 dots total
