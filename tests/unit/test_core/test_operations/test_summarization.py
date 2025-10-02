"""
Comprehensive unit tests for content summarization operations.

Tests cover statistical summarization, field validation, edge cases,
and error conditions to uncover production bugs.

EXPECTED BUGS TO FIND:
1. Division by zero in statistical calculations (mean, std, median)
2. Type conversion issues with mixed data types
3. Invalid field handling in summarization
4. Empty data handling edge cases
"""

from unittest.mock import Mock

import pytest

from shedboxai.core.config.models import ContentSummarizationConfig
from shedboxai.core.operations.summarization import ContentSummarizationHandler


class TestContentSummarizationHandler:
    """Test suite for ContentSummarizationHandler operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = ContentSummarizationHandler()
        self.mock_engine = Mock()
        self.handler_with_engine = ContentSummarizationHandler(engine=self.mock_engine)

    # Basic Configuration Tests
    def test_operation_name(self):
        """Test operation name property."""
        assert self.handler.operation_name == "content_summarization"

    def test_empty_config_returns_data_unchanged(self):
        """Test that empty config returns data unchanged."""
        data = {"source": [{"value": 10, "amount": 20}]}
        result = self.handler.process(data, {})
        assert result == data

    def test_missing_source_data(self):
        """Test processing when source data doesn't exist."""
        data = {"other_data": [{"value": 10}]}
        config = {
            "missing_source": ContentSummarizationConfig(
                method="statistical", fields=["value"], summarize=["mean", "count"]
            )
        }

        result = self.handler.process(data, config)
        assert result == data  # Data unchanged

    def test_non_list_source_data(self):
        """Test processing when source data is not a list."""
        data = {"source": {"value": 10}}  # Dict instead of list
        config = {"source": ContentSummarizationConfig(method="statistical", fields=["value"], summarize=["mean"])}

        result = self.handler.process(data, config)
        assert result == data  # Data unchanged due to non-list

    # Statistical Summarization Tests
    def test_statistical_mean_calculation(self):
        """Test basic mean calculation."""
        data = {"numbers": [{"value": 10}, {"value": 20}, {"value": 30}]}
        config = {"numbers": ContentSummarizationConfig(method="statistical", fields=["value"], summarize=["mean"])}

        result = self.handler.process(data, config)
        assert result["numbers_summary"]["value_mean"] == 20.0

    def test_statistical_multiple_operations(self):
        """Test multiple statistical operations."""
        data = {
            "sales": [
                {"amount": 100, "quantity": 5},
                {"amount": 200, "quantity": 10},
                {"amount": 300, "quantity": 15},
            ]
        }
        config = {
            "sales": ContentSummarizationConfig(
                method="statistical",
                fields=["amount", "quantity"],
                summarize=["mean", "min", "max", "count", "sum"],
            )
        }

        result = self.handler.process(data, config)
        summary = result["sales_summary"]

        # Amount statistics
        assert summary["amount_mean"] == 200.0
        assert summary["amount_min"] == 100.0
        assert summary["amount_max"] == 300.0
        assert summary["amount_count"] == 3
        assert summary["amount_sum"] == 600.0

        # Quantity statistics
        assert summary["quantity_mean"] == 10.0
        assert summary["quantity_min"] == 5.0
        assert summary["quantity_max"] == 15.0
        assert summary["quantity_count"] == 3
        assert summary["quantity_sum"] == 30.0

    def test_statistical_median_odd_count(self):
        """Test median calculation with odd number of values."""
        data = {
            "values": [
                {"score": 10},
                {"score": 20},
                {"score": 30},
                {"score": 40},
                {"score": 50},
            ]
        }
        config = {"values": ContentSummarizationConfig(method="statistical", fields=["score"], summarize=["median"])}

        result = self.handler.process(data, config)
        assert result["values_summary"]["score_median"] == 30.0

    def test_statistical_median_even_count(self):
        """Test median calculation with even number of values."""
        data = {"values": [{"score": 10}, {"score": 20}, {"score": 30}, {"score": 40}]}
        config = {"values": ContentSummarizationConfig(method="statistical", fields=["score"], summarize=["median"])}

        result = self.handler.process(data, config)
        assert result["values_summary"]["score_median"] == 25.0  # (20 + 30) / 2

    def test_statistical_standard_deviation(self):
        """Test standard deviation calculation."""
        data = {
            "values": [
                {"score": 2},
                {"score": 4},
                {"score": 4},
                {"score": 4},
                {"score": 5},
                {"score": 5},
                {"score": 7},
                {"score": 9},
            ]
        }
        config = {"values": ContentSummarizationConfig(method="statistical", fields=["score"], summarize=["std"])}

        result = self.handler.process(data, config)
        # Standard deviation should be approximately 2.138 (sample std dev)
        assert abs(result["values_summary"]["score_std"] - 2.138) < 0.01

    def test_statistical_unique_count(self):
        """Test unique value counting."""
        data = {
            "categories": [
                {"type": "A"},
                {"type": "B"},
                {"type": "A"},
                {"type": "C"},
                {"type": "B"},
                {"type": "A"},
            ]
        }
        config = {
            "categories": ContentSummarizationConfig(
                method="statistical", fields=["type"], summarize=["unique", "count"]
            )
        }

        result = self.handler.process(data, config)
        summary = result["categories_summary"]
        assert summary["type_unique"] == 3  # A, B, C
        assert summary["type_count"] == 6  # Total count

    # Type Conversion and Mixed Data Tests
    def test_string_numeric_conversion(self):
        """Test conversion of string numbers to numeric values."""
        data = {"mixed": [{"price": "10.50"}, {"price": "20.75"}, {"price": "15.25"}]}
        config = {
            "mixed": ContentSummarizationConfig(method="statistical", fields=["price"], summarize=["mean", "sum"])
        }

        result = self.handler.process(data, config)
        summary = result["mixed_summary"]
        assert summary["price_mean"] == 15.5  # (10.5 + 20.75 + 15.25) / 3
        assert summary["price_sum"] == 46.5

    def test_mixed_numeric_and_string_data(self):
        """Test handling mixed numeric and non-numeric data."""
        data = {
            "mixed": [
                {"value": 10},
                {"value": "20"},
                {"value": "not_a_number"},
                {"value": 30.5},
                {"value": "invalid"},
                {"value": "40"},
            ]
        }
        config = {
            "mixed": ContentSummarizationConfig(
                method="statistical",
                fields=["value"],
                summarize=["mean", "count", "sum"],
            )
        }

        result = self.handler.process(data, config)
        summary = result["mixed_summary"]
        # Should only process valid numbers: 10, 20, 30.5, 40
        assert summary["value_mean"] == 25.125  # (10 + 20 + 30.5 + 40) / 4
        assert summary["value_sum"] == 100.5
        assert summary["value_count"] == 6  # Total count includes non-numeric

    def test_none_and_missing_values(self):
        """Test handling of None and missing values."""
        data = {
            "sparse": [
                {"value": 10, "other": "data"},
                {"value": None, "other": "data"},
                {"other": "data"},  # Missing value field
                {"value": 20, "other": "data"},
                {"value": 30},
            ]
        }
        config = {
            "sparse": ContentSummarizationConfig(
                method="statistical",
                fields=["value"],
                summarize=["mean", "count", "sum"],
            )
        }

        result = self.handler.process(data, config)
        summary = result["sparse_summary"]
        # Should only count non-None values: 10, 20, 30
        assert summary["value_mean"] == 20.0  # (10 + 20 + 30) / 3
        assert summary["value_sum"] == 60.0
        assert summary["value_count"] == 3  # Only non-None values

    # Edge Cases and Error Conditions
    def test_empty_data_list(self):
        """Test summarization with empty data list."""
        data = {"empty": []}
        config = {
            "empty": ContentSummarizationConfig(method="statistical", fields=["value"], summarize=["mean", "count"])
        }

        result = self.handler.process(data, config)
        assert "empty_summary" in result
        # Should handle empty data gracefully

    def test_field_not_in_any_items(self):
        """Test summarization when field doesn't exist in any items."""
        data = {"items": [{"name": "item1"}, {"name": "item2"}, {"name": "item3"}]}
        config = {
            "items": ContentSummarizationConfig(
                method="statistical",
                fields=["missing_field"],
                summarize=["mean", "count"],
            )
        }

        result = self.handler.process(data, config)
        # Should handle missing fields gracefully
        assert "items_summary" in result

    def test_single_value_standard_deviation(self):
        """Test standard deviation with only one value - should not cause division by zero."""
        data = {"single": [{"value": 42}]}
        config = {"single": ContentSummarizationConfig(method="statistical", fields=["value"], summarize=["std"])}

        result = self.handler.process(data, config)
        summary = result["single_summary"]
        # Standard deviation of single value should be 0
        assert summary["value_std"] == 0

    def test_all_same_values_standard_deviation(self):
        """Test standard deviation when all values are the same."""
        data = {"same": [{"value": 5}, {"value": 5}, {"value": 5}, {"value": 5}]}
        config = {"same": ContentSummarizationConfig(method="statistical", fields=["value"], summarize=["std"])}

        result = self.handler.process(data, config)
        summary = result["same_summary"]
        # Standard deviation should be 0 when all values are the same
        assert summary["value_std"] == 0.0

    # Configuration Validation Tests
    def test_dict_config_conversion(self):
        """Test that dict configs are converted to ContentSummarizationConfig objects."""
        data = {"values": [{"score": 10}, {"score": 20}]}
        config = {
            "values": {
                "method": "statistical",
                "fields": ["score"],
                "summarize": ["mean"],
            }
        }

        result = self.handler.process(data, config)
        assert "values_summary" in result
        assert result["values_summary"]["score_mean"] == 15.0

    def test_invalid_dict_config_logs_warning(self, caplog):
        """Test that invalid dict config logs warning."""
        data = {"values": [{"score": 10}]}
        config = {"values": {"invalid_field": "value"}}

        result = self.handler.process(data, config)
        assert "Invalid summarization configuration" in caplog.text
        assert result == data

    def test_invalid_config_type_logs_warning(self, caplog):
        """Test that invalid config type logs warning."""
        data = {"values": [{"score": 10}]}
        config = {"values": "invalid_string_config"}

        result = self.handler.process(data, config)
        assert "Invalid summarization configuration" in caplog.text
        assert "expected dict or ContentSummarizationConfig" in caplog.text
        assert result == data

    def test_unsupported_method_logs_warning(self, caplog):
        """Test that unsupported summarization method logs warning."""
        data = {"values": [{"score": 10}]}
        config = {
            "values": ContentSummarizationConfig(method="unsupported_method", fields=["score"], summarize=["mean"])
        }

        result = self.handler.process(data, config)
        assert "Unknown or unsupported summarization method" in caplog.text

    def test_ai_method_logs_warning(self, caplog):
        """Test that AI summarization method logs warning about being unsupported."""
        data = {"values": [{"score": 10}]}
        config = {"values": ContentSummarizationConfig(method="ai", fields=["score"], summarize=["mean"])}

        result = self.handler.process(data, config)
        assert "AI summarization method is no longer supported" in caplog.text

    # Multiple Sources Tests
    def test_multiple_sources_processing(self):
        """Test processing multiple sources in single operation."""
        data = {
            "sales": [{"amount": 100}, {"amount": 200}],
            "inventory": [{"count": 50}, {"count": 75}],
        }
        config = {
            "sales": ContentSummarizationConfig(method="statistical", fields=["amount"], summarize=["mean"]),
            "inventory": ContentSummarizationConfig(method="statistical", fields=["count"], summarize=["sum"]),
        }

        result = self.handler.process(data, config)
        assert result["sales_summary"]["amount_mean"] == 150.0
        assert result["inventory_summary"]["count_sum"] == 125.0

    # Bug Detection Tests - These should expose real bugs
    def test_division_by_zero_edge_cases(self):
        """Test potential division by zero scenarios."""
        data = {"empty_values": [{"value": None}, {"value": None}, {"other": "data"}]}
        config = {
            "empty_values": ContentSummarizationConfig(
                method="statistical", fields=["value"], summarize=["mean", "std"]
            )
        }

        # Should handle all None/missing values without crashing
        result = self.handler.process(data, config)
        assert "empty_values_summary" in result

    def test_non_dict_items_in_source_list(self):
        """Test summarization when source list contains non-dict items."""
        data = {"mixed_types": [{"value": 10}, "string_item", 42, {"value": 20}, None]}
        config = {
            "mixed_types": ContentSummarizationConfig(
                method="statistical", fields=["value"], summarize=["mean", "count"]
            )
        }

        # Should handle non-dict items gracefully
        result = self.handler.process(data, config)
        assert "mixed_types_summary" in result
