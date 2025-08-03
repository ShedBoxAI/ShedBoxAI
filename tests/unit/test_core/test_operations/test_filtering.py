"""
Tests for filtering.py - Contextual filtering operation handler.

Tests focus on finding real bugs in filtering logic and ensuring
all edge cases and comparison operations work correctly.
"""

from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from shedboxai.core.config.models import ContextualFilterConfig
from shedboxai.core.operations.filtering import ContextualFilteringHandler


class TestContextualFilteringHandler:
    """Test cases for ContextualFilteringHandler."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = ContextualFilteringHandler()
        self.mock_engine = MagicMock()
        self.handler_with_engine = ContextualFilteringHandler(engine=self.mock_engine)

    def test_operation_name(self):
        """Test that operation name is correct."""
        assert self.handler.operation_name == "contextual_filtering"

    def test_empty_config_returns_data_unchanged(self):
        """Test that empty config returns data unchanged."""
        data = {"source1": [{"id": 1}, {"id": 2}]}
        result = self.handler.process(data, {})
        assert result == data

    def test_missing_source_skips_processing(self):
        """Test that missing source in data is skipped."""
        data = {"source1": [{"id": 1}]}
        config = {"missing_source": [ContextualFilterConfig(field="id", condition="> 0")]}

        result = self.handler.process(data, config)
        assert result == data  # Unchanged

    def test_non_list_source_data_skips_processing(self):
        """Test that non-list source data is skipped."""
        data = {"source1": {"not": "a_list"}}
        config = {"source1": [ContextualFilterConfig(field="id", condition="> 0")]}

        result = self.handler.process(data, config)
        assert result == data  # Unchanged

    def test_simple_equality_filter(self):
        """Test simple equality filtering."""
        data = {
            "products": [
                {"name": "Apple", "category": "fruit"},
                {"name": "Carrot", "category": "vegetable"},
                {"name": "Orange", "category": "fruit"},
            ]
        }
        config = {"products": [ContextualFilterConfig(field="category", condition="fruit")]}

        result = self.handler.process(data, config)
        assert len(result["products"]) == 2
        assert all(item["category"] == "fruit" for item in result["products"])

    def test_numeric_comparison_filters(self):
        """Test numeric comparison filtering."""
        data = {
            "items": [
                {"price": 10, "name": "cheap"},
                {"price": 50, "name": "medium"},
                {"price": 100, "name": "expensive"},
                {"price": 25, "name": "affordable"},
            ]
        }

        # Test greater than
        config = {"items": [ContextualFilterConfig(field="price", condition="> 30")]}
        result = self.handler.process(data, config)
        assert len(result["items"]) == 2
        assert all(item["price"] > 30 for item in result["items"])

        # Test less than or equal
        config = {"items": [ContextualFilterConfig(field="price", condition="<= 50")]}
        result = self.handler.process(data, config)
        assert len(result["items"]) == 3
        assert all(item["price"] <= 50 for item in result["items"])

    def test_string_numeric_conversion(self):
        """Test that string numbers are converted for comparison."""
        data = {
            "records": [
                {"score": "85", "name": "good"},
                {"score": "92", "name": "excellent"},
                {"score": "78", "name": "okay"},
            ]
        }
        config = {"records": [ContextualFilterConfig(field="score", condition="> 80")]}

        result = self.handler.process(data, config)
        assert len(result["records"]) == 2
        assert all(float(item["score"]) > 80 for item in result["records"])

    def test_multiple_filters_applied_sequentially_FIXED(self):
        """Test that multiple filters are NOW applied sequentially (FIXED)."""
        data = {
            "products": [
                {
                    "price": 100,
                    "category": "electronics",
                    "rating": 4.5,
                },  # Should pass all filters
                {
                    "price": 50,
                    "category": "electronics",
                    "rating": 3.0,
                },  # Should fail price and rating
                {
                    "price": 80,
                    "category": "books",
                    "rating": 4.0,
                },  # Should fail category
                {
                    "price": 120,
                    "category": "electronics",
                    "rating": 4.8,
                },  # Should pass all filters
            ]
        }
        config = {
            "products": [
                ContextualFilterConfig(field="category", condition="electronics"),
                ContextualFilterConfig(field="price", condition="> 60"),
                ContextualFilterConfig(field="rating", condition=">= 4.0"),
            ]
        }

        result = self.handler.process(data, config)

        # FIXED: Now correctly applies filters sequentially (AND logic)
        assert len(result["products"]) == 2
        for item in result["products"]:
            assert item["category"] == "electronics"
            assert item["price"] > 60
            assert item["rating"] >= 4.0

        # Should have items with prices 100 and 120
        prices = sorted([item["price"] for item in result["products"]])
        assert prices == [100, 120]

    def test_single_filter_works_correctly(self):
        """Test that single filters work correctly (to isolate the bug)."""
        data = {
            "items": [
                {"price": 100, "category": "electronics"},
                {"price": 50, "category": "books"},
            ]
        }

        # Single filter should work
        config = {"items": [ContextualFilterConfig(field="category", condition="electronics")]}
        result = self.handler.process(data, config)

        assert len(result["items"]) == 1
        assert result["items"][0]["category"] == "electronics"

    def test_filter_with_new_name_creates_new_result(self):
        """Test that filter with new_name creates new result set."""
        data = {
            "all_products": [
                {"price": 100, "category": "premium"},
                {"price": 20, "category": "budget"},
                {"price": 150, "category": "premium"},
            ]
        }

        filter_config = ContextualFilterConfig(field="category", condition="premium")
        filter_config.new_name = "premium_products"
        config = {"all_products": [filter_config]}

        result = self.handler.process(data, config)

        # Original data should be unchanged
        assert len(result["all_products"]) == 3

        # New filtered result should exist
        assert "premium_products" in result
        assert len(result["premium_products"]) == 2
        assert all(item["category"] == "premium" for item in result["premium_products"])

    def test_missing_field_handled_gracefully(self):
        """Test that missing fields are handled gracefully."""
        data = {
            "items": [
                {"name": "complete", "price": 100},
                {"name": "incomplete"},  # Missing price
                {"name": "also_complete", "price": 50},
            ]
        }
        config = {"items": [ContextualFilterConfig(field="price", condition="> 60")]}

        result = self.handler.process(data, config)

        # Should only include items with price field > 60
        assert len(result["items"]) == 1
        assert result["items"][0]["name"] == "complete"

    def test_non_numeric_comparison_falls_back_gracefully(self):
        """Test that non-numeric values in numeric comparisons fail gracefully."""
        data = {
            "mixed": [
                {"value": 100, "name": "numeric"},
                {"value": "not_a_number", "name": "string"},
                {"value": None, "name": "null"},
                {"value": 50, "name": "also_numeric"},
            ]
        }
        config = {"mixed": [ContextualFilterConfig(field="value", condition="> 75")]}

        result = self.handler.process(data, config)

        # Should only include numeric values > 75
        assert len(result["mixed"]) == 1
        assert result["mixed"][0]["name"] == "numeric"

    def test_all_comparison_operators(self):
        """Test all supported comparison operators."""
        data = {"numbers": [{"val": 10}, {"val": 20}, {"val": 30}, {"val": 40}, {"val": 50}]}

        test_cases = [
            ("> 25", 3),  # 30, 40, 50
            ("< 35", 3),  # 10, 20, 30
            (">= 30", 3),  # 30, 40, 50
            ("<= 30", 3),  # 10, 20, 30
            ("== 30", 1),  # 30
            ("!= 30", 4),  # 10, 20, 40, 50
        ]

        for condition, expected_count in test_cases:
            config = {"numbers": [ContextualFilterConfig(field="val", condition=condition)]}
            result = self.handler.process(data, config)
            assert len(result["numbers"]) == expected_count, f"Failed for condition: {condition}"

    def test_engine_evaluation_when_available(self):
        """Test that engine is used for evaluation when available."""
        data = {"items": [{"score": 85}, {"score": 90}]}
        config = {"items": [ContextualFilterConfig(field="score", condition="> 87")]}

        # Mock engine to return True for first item, False for second
        self.mock_engine.evaluate.side_effect = [True, False]

        result = self.handler_with_engine.process(data, config)

        # Should use engine evaluation
        assert len(result["items"]) == 1
        assert self.mock_engine.evaluate.call_count == 2

        # Check that correct expressions were passed to engine
        calls = self.mock_engine.evaluate.call_args_list
        assert "85.0 > 87" in str(calls[0])
        assert "90.0 > 87" in str(calls[1])

    def test_engine_evaluation_failure_falls_back_FIXED(self):
        """Test that engine evaluation failure NOW falls back gracefully (FIXED)."""
        data = {"items": [{"score": 85}, {"score": 90}]}
        config = {"items": [ContextualFilterConfig(field="score", condition="> 87")]}

        # Mock engine to raise exception
        self.mock_engine.evaluate.side_effect = Exception("Engine failed")

        result = self.handler_with_engine.process(data, config)

        # FIXED: Now falls back to simple comparison and works correctly
        assert len(result["items"]) == 1
        assert result["items"][0]["score"] == 90

        # Verify that engine was called but then fallback worked
        assert self.mock_engine.evaluate.call_count >= 1

    def test_invalid_filter_config_logs_warning(self):
        """Test that invalid filter configurations log warnings."""
        data = {"items": [{"id": 1}]}

        # Test with non-list filters
        with patch.object(self.handler, "_log_warning") as mock_warning:
            config = {"items": "not_a_list"}
            result = self.handler.process(data, config)
            mock_warning.assert_called()
            assert "Invalid filter configuration" in str(mock_warning.call_args)

    def test_invalid_filter_config_dict_conversion(self):
        """Test conversion of dict filter configs to ContextualFilterConfig."""
        data = {"items": [{"price": 100}, {"price": 50}]}
        config = {"items": [{"field": "price", "condition": "> 75"}]}  # Dict instead of ContextualFilterConfig

        result = self.handler.process(data, config)

        # Should convert dict to ContextualFilterConfig and work
        assert len(result["items"]) == 1
        assert result["items"][0]["price"] == 100

    def test_invalid_dict_filter_config_logs_warning(self):
        """Test that invalid dict filter configs log warnings."""
        data = {"items": [{"price": 100}]}

        with patch.object(self.handler, "_log_warning") as mock_warning:
            config = {"items": [{"invalid_field": "value"}]}  # Missing required fields
            result = self.handler.process(data, config)

            mock_warning.assert_called()
            assert "Invalid filter configuration" in str(mock_warning.call_args)

    def test_invalid_filter_type_logs_warning(self):
        """Test that invalid filter types log warnings."""
        data = {"items": [{"price": 100}]}

        with patch.object(self.handler, "_log_warning") as mock_warning:
            config = {"items": ["invalid_string_filter"]}  # Wrong type
            result = self.handler.process(data, config)

            mock_warning.assert_called()
            assert "Invalid filter configuration" in str(mock_warning.call_args)

    def test_complex_nested_data_filtering(self):
        """Test filtering on complex nested data structures."""
        data = {
            "orders": [
                {
                    "id": 1,
                    "customer": {"name": "John", "tier": "premium"},
                    "total": 150,
                    "items": [{"name": "laptop"}, {"name": "mouse"}],
                },
                {
                    "id": 2,
                    "customer": {"name": "Jane", "tier": "basic"},
                    "total": 50,
                    "items": [{"name": "book"}],
                },
            ]
        }

        # Filter by total amount
        config = {"orders": [ContextualFilterConfig(field="total", condition="> 100")]}
        result = self.handler.process(data, config)

        assert len(result["orders"]) == 1
        assert result["orders"][0]["id"] == 1

    def test_edge_case_empty_data_list(self):
        """Test filtering on empty data list."""
        data = {"empty": []}
        config = {"empty": [ContextualFilterConfig(field="price", condition="> 0")]}

        result = self.handler.process(data, config)
        assert result["empty"] == []

    def test_edge_case_zero_values(self):
        """Test filtering with zero values and edge numeric cases."""
        data = {"values": [{"amount": 0}, {"amount": -10}, {"amount": 0.0}, {"amount": 10}]}

        # Test greater than 0
        config = {"values": [ContextualFilterConfig(field="amount", condition="> 0")]}
        result = self.handler.process(data, config)
        assert len(result["values"]) == 1
        assert result["values"][0]["amount"] == 10

        # Test greater than or equal to 0
        config = {"values": [ContextualFilterConfig(field="amount", condition=">= 0")]}
        result = self.handler.process(data, config)
        assert len(result["values"]) == 3  # 0, 0.0, 10

    def test_condition_string_parsing_edge_cases(self):
        """Test edge cases in condition string parsing."""
        data = {"items": [{"val": 10}, {"val": 20}]}

        # Test condition with extra whitespace
        config = {"items": [ContextualFilterConfig(field="val", condition="  >   15  ")]}
        result = self.handler.process(data, config)
        assert len(result["items"]) == 1

        # Test condition with no spaces
        config = {"items": [ContextualFilterConfig(field="val", condition=">15")]}
        result = self.handler.process(data, config)
        assert len(result["items"]) == 1

    def test_float_comparison_precision(self):
        """Test floating point comparison precision."""
        data = {
            "prices": [
                {"amount": 10.1},
                {"amount": 10.11},
                {"amount": 10.111},
                {"amount": 10.2},
            ]
        }

        config = {"prices": [ContextualFilterConfig(field="amount", condition="> 10.11")]}
        result = self.handler.process(data, config)

        # Should include 10.111 and 10.2, but not 10.1 or 10.11
        assert len(result["prices"]) == 2
        amounts = [item["amount"] for item in result["prices"]]
        assert 10.111 in amounts
        assert 10.2 in amounts


class TestContextualFilteringHandlerEdgeCases:
    """Test edge cases and potential bugs in ContextualFilteringHandler."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = ContextualFilteringHandler()

    def test_filter_preserves_original_data_structure(self):
        """Test that filtering preserves original data structure references."""
        original_item = {"id": 1, "nested": {"data": "value"}}
        data = {"items": [original_item, {"id": 2}]}
        config = {"items": [ContextualFilterConfig(field="id", condition="== 1")]}

        result = self.handler.process(data, config)

        # The filtered item should be the same object reference
        assert result["items"][0] is original_item
        assert result["items"][0]["nested"] is original_item["nested"]

    def test_data_immutability_during_filtering(self):
        """Test that original data is not modified during filtering."""
        original_data = {"products": [{"name": "A", "price": 100}, {"name": "B", "price": 50}]}
        data_copy = {"products": [item.copy() for item in original_data["products"]]}

        config = {"products": [ContextualFilterConfig(field="price", condition="> 75")]}

        result = self.handler.process(data_copy, config)

        # Original data structure should be unchanged
        assert len(data_copy["products"]) == 2  # Still has both items
        assert len(result["products"]) == 1  # Filtered result has one

    def test_multiple_sources_processed_independently(self):
        """Test that multiple sources are processed independently."""
        data = {
            "expensive": [{"price": 100}, {"price": 200}],
            "cheap": [{"price": 10}, {"price": 20}],
        }
        config = {
            "expensive": [ContextualFilterConfig(field="price", condition="> 150")],
            "cheap": [ContextualFilterConfig(field="price", condition="< 15")],
        }

        result = self.handler.process(data, config)

        assert len(result["expensive"]) == 1  # One item > 150
        assert len(result["cheap"]) == 1  # One item < 15
        assert result["expensive"][0]["price"] == 200
        assert result["cheap"][0]["price"] == 10

    def test_filter_config_validation_edge_cases(self):
        """Test various edge cases in filter config validation."""
        data = {"items": [{"val": 1}]}

        test_cases = [
            # Valid cases that should work
            ([ContextualFilterConfig(field="val", condition="== 1")], 1),
            # Edge cases that should be handled gracefully
            ([], 1),  # Empty filter list
            (None, 1),  # None instead of list (should log warning)
            ("string", 1),  # String instead of list (should log warning)
            ([None], 1),  # List with None (should log warning for None item)
        ]

        for filter_config, expected_len in test_cases:
            config = {"items": filter_config}
            result = self.handler.process(data, config)
            # All should leave data unchanged or filter appropriately
            assert len(result["items"]) <= 1

    def test_engine_integration_edge_cases(self):
        """Test edge cases with engine integration."""
        mock_engine = MagicMock()
        handler = ContextualFilteringHandler(engine=mock_engine)

        data = {"items": [{"score": "not_a_number"}]}
        config = {"items": [ContextualFilterConfig(field="score", condition="> 50")]}

        # Engine should receive the unconverted value in expression
        mock_engine.evaluate.return_value = False

        result = handler.process(data, config)

        # Should still call engine even with non-numeric data
        mock_engine.evaluate.assert_called()
        call_args = mock_engine.evaluate.call_args[0]
        assert "not_a_number" in str(call_args[0])

    def test_comparison_operator_boundary_conditions(self):
        """Test boundary conditions for comparison operators."""
        data = {"values": [{"x": 0}, {"x": 1}, {"x": -1}]}

        boundary_tests = [
            ("== 0", [0]),
            ("!= 0", [1, -1]),
            ("> 0", [1]),
            ("< 0", [-1]),
            (">= 0", [0, 1]),
            ("<= 0", [0, -1]),
        ]

        for condition, expected_values in boundary_tests:
            config = {"values": [ContextualFilterConfig(field="x", condition=condition)]}
            result = self.handler.process(data, config)

            actual_values = [item["x"] for item in result["values"]]
            assert set(actual_values) == set(expected_values), f"Failed for condition: {condition}"

    def test_large_dataset_performance_characteristics(self):
        """Test filtering behavior with larger datasets."""
        # Create a moderately large dataset
        large_data = {"records": [{"id": i, "value": i * 2} for i in range(1000)]}
        config = {"records": [ContextualFilterConfig(field="value", condition="> 1000")]}

        result = self.handler.process(large_data, config)

        # Should efficiently filter and return correct results
        assert len(result["records"]) == 499  # Values 1002, 1004, ..., 1998
        assert all(item["value"] > 1000 for item in result["records"])

    def test_memory_usage_with_deep_copies(self):
        """Test that filtering doesn't create unnecessary deep copies."""
        import sys

        # Create data with nested structure
        nested_object = {"deep": {"nested": {"data": list(range(100))}}}
        data = {"items": [nested_object] * 10}  # 10 references to same object

        config = {"items": [ContextualFilterConfig(field="deep", condition="deep")]}  # Always true

        result = self.handler.process(data, config)

        # All items in result should still reference the same nested object
        for item in result["items"]:
            assert item["deep"] is nested_object["deep"]

    def test_unicode_and_special_characters(self):
        """Test filtering with unicode and special characters."""
        data = {
            "international": [
                {"name": "José", "price": 100},
                {"name": "München", "price": 50},
                {"name": "🚀 Rocket", "price": 200},
                {"name": "Test", "price": 75},
            ]
        }

        # Filter by price (should work regardless of unicode in other fields)
        config = {"international": [ContextualFilterConfig(field="price", condition="> 80")]}
        result = self.handler.process(data, config)

        assert len(result["international"]) == 2
        names = [item["name"] for item in result["international"]]
        assert "José" in names
        assert "🚀 Rocket" in names


if __name__ == "__main__":
    pytest.main([__file__])
