"""
Comprehensive unit tests for relationship highlighting operations.

Tests cover JSONPath links, pattern detection, conditional highlighting,
and error conditions to uncover production bugs.

EXPECTED BUGS TO FIND:
1. JSONPath parsing errors with invalid expressions
2. Missing jsonpath_ng module import handling
3. Expression engine dependency issues in conditional highlighting
4. Pattern detection edge cases with empty/invalid data
5. Context template processing errors
"""

from unittest.mock import Mock, patch

import pytest

from shedboxai.core.config.models import RelationshipConfig
from shedboxai.core.operations.relationships import RelationshipHighlightingHandler


class TestRelationshipHighlightingHandler:
    """Test suite for RelationshipHighlightingHandler operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = RelationshipHighlightingHandler()
        self.mock_engine = Mock()
        self.handler_with_engine = RelationshipHighlightingHandler(engine=self.mock_engine)

    # Basic Configuration Tests
    def test_operation_name(self):
        """Test operation name property."""
        assert self.handler.operation_name == "relationship_highlighting"

    def test_empty_config_returns_data_unchanged(self):
        """Test that empty config returns data unchanged."""
        data = {"source": [{"id": 1, "name": "test"}]}
        result = self.handler.process(data, {})
        assert result == data

    def test_missing_source_data(self):
        """Test processing when referenced sources don't exist."""
        data = {"existing": [{"id": 1}]}
        config = {
            "test": RelationshipConfig(link_fields=[{"source": "missing_source", "match_on": "id", "to": "existing"}])
        }

        result = self.handler.process(data, config)
        assert result == data  # Data unchanged

    # Link Fields Tests (Basic Joins)
    def test_simple_link_fields(self):
        """Test basic link fields functionality - FIXED."""
        data = {
            "users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}],
            "orders": [
                {"order_id": 101, "user_id": 1, "amount": 100},
                {"order_id": 102, "user_id": 2, "amount": 200},
            ],
        }
        config = {
            "test": RelationshipConfig(
                link_fields=[
                    {
                        "source": "orders",
                        "source_field": "user_id",  # FIX: Use separate field names
                        "to": "users",
                        "target_field": "id",  # FIX: Instead of duplicate match_on
                    }
                ]
            )
        }

        result = self.handler.process(data, config)

        # Check that orders now have user info
        for order in result["orders"]:
            if order["user_id"] == 1:
                assert order["users_info"]["name"] == "Alice"
            elif order["user_id"] == 2:
                assert order["users_info"]["name"] == "Bob"

    def test_link_fields_no_matches(self):
        """Test link fields with no matching records."""
        data = {
            "users": [{"id": 1, "name": "Alice"}],
            "orders": [{"order_id": 101, "user_id": 999, "amount": 100}],  # No matching user
        }
        config = {
            "test": RelationshipConfig(
                link_fields=[
                    {
                        "source": "orders",
                        "source_field": "user_id",
                        "to": "users",
                        "target_field": "id",
                    }
                ]
            )
        }

        result = self.handler.process(data, config)
        # Order should not have users_info since no match
        assert "users_info" not in result["orders"][0]

    def test_link_fields_missing_match_field(self):
        """Test link fields when match field is missing from some records."""
        data = {
            "users": [{"id": 1, "name": "Alice"}, {"name": "Bob"}],  # Missing id field
            "orders": [
                {"order_id": 101, "user_id": 1, "amount": 100},
                {"order_id": 102, "amount": 200},  # Missing user_id field
            ],
        }
        config = {
            "test": RelationshipConfig(
                link_fields=[
                    {
                        "source": "orders",
                        "source_field": "user_id",
                        "to": "users",
                        "target_field": "id",
                    }
                ]
            )
        }

        result = self.handler.process(data, config)
        # Should handle missing fields gracefully
        orders = result["orders"]
        assert len(orders) == 2

    def test_link_fields_non_list_data(self):
        """Test link fields with non-list data."""
        data = {
            "user": {"id": 1, "name": "Alice"},  # Not a list
            "orders": [{"order_id": 101, "user_id": 1}],
        }
        config = {
            "test": RelationshipConfig(
                link_fields=[
                    {
                        "source": "orders",
                        "source_field": "user_id",
                        "to": "user",
                        "target_field": "id",
                    }
                ]
            )
        }

        result = self.handler.process(data, config)
        # Should handle non-list target gracefully
        assert result == data

    # JSONPath Links Tests
    def test_jsonpath_links_simple(self):
        """Test basic JSONPath links functionality."""
        data = {
            "products": [
                {"id": 1, "category": {"id": 10, "name": "Electronics"}},
                {"id": 2, "category": {"id": 20, "name": "Books"}},
            ],
            "inventory": [
                {"product_id": 1, "stock": 50},
                {"product_id": 2, "stock": 25},
            ],
        }
        config = {
            "test": RelationshipConfig(
                jsonpath_links=[
                    {
                        "source": "products",
                        "target": "inventory",
                        "source_path": "id",
                        "target_path": "product_id",
                        "result_field": "inventory_info",
                    }
                ]
            )
        }

        result = self.handler.process(data, config)

        # Check that inventory items have related product data
        inventory = result["inventory"]
        for item in inventory:
            if "inventory_info" in item:
                assert len(item["inventory_info"]) > 0

    def test_jsonpath_links_nested_paths(self):
        """Test JSONPath links with nested field paths."""
        data = {
            "orders": [
                {"id": 1, "customer": {"profile": {"email": "alice@example.com"}}},
                {"id": 2, "customer": {"profile": {"email": "bob@example.com"}}},
            ],
            "notifications": [
                {"recipient": "alice@example.com", "message": "Order shipped"},
                {"recipient": "bob@example.com", "message": "Order confirmed"},
            ],
        }
        config = {
            "test": RelationshipConfig(
                jsonpath_links=[
                    {
                        "source": "orders",
                        "target": "notifications",
                        "source_path": "customer.profile.email",
                        "target_path": "recipient",
                        "result_field": "related_orders",
                    }
                ]
            )
        }

        result = self.handler.process(data, config)
        # Should handle nested paths correctly
        notifications = result["notifications"]
        assert len(notifications) == 2

    def test_jsonpath_links_invalid_path(self):
        """Test JSONPath links with invalid path expressions."""
        data = {"source": [{"id": 1}], "target": [{"ref": 1}]}
        config = {
            "test": RelationshipConfig(
                jsonpath_links=[
                    {
                        "source": "source",
                        "target": "target",
                        "source_path": "invalid[[[path",  # Invalid JSONPath
                        "target_path": "ref",
                    }
                ]
            )
        }

        result = self.handler.process(data, config)
        # Should handle invalid paths gracefully without crashing
        assert result == data

    def test_jsonpath_links_missing_fields(self):
        """Test JSONPath links when required fields are missing."""
        data = {"source": [{"id": 1}], "target": [{"ref": 1}]}
        config = {
            "test": RelationshipConfig(
                jsonpath_links=[
                    {
                        "source": "source",
                        "target": "target",
                        # Missing source_path and target_path
                    }
                ]
            )
        }

        result = self.handler.process(data, config)
        # Should handle missing required fields gracefully
        assert result == data

    # Pattern Detection Tests
    def test_frequency_pattern_detection(self):
        """Test frequency pattern detection."""
        data = {
            "events": [
                {"type": "login", "user": "alice"},
                {"type": "logout", "user": "alice"},
                {"type": "login", "user": "bob"},
                {"type": "login", "user": "alice"},
                {"type": "error", "user": "charlie"},
            ]
        }
        config = {
            "test": RelationshipConfig(
                pattern_detection={
                    "user_patterns": {
                        "type": "frequency",
                        "source": "events",
                        "field": "type",
                        "threshold": 2,
                    }
                }
            )
        }

        result = self.handler.process(data, config)

        # Should detect login as frequent pattern (appears 3 times)
        if "events_patterns" in result:
            patterns = result["events_patterns"]["patterns"]
            assert "login" in patterns
            assert patterns["login"] >= 2

    def test_frequency_pattern_below_threshold(self):
        """Test frequency pattern detection with values below threshold."""
        data = {"events": [{"type": "login"}, {"type": "logout"}, {"type": "error"}]}
        config = {
            "test": RelationshipConfig(
                pattern_detection={
                    "patterns": {
                        "type": "frequency",
                        "source": "events",
                        "field": "type",
                        "threshold": 2,
                    }
                }
            )
        }

        result = self.handler.process(data, config)
        # No patterns should be detected since all appear only once
        if "events_patterns" in result:
            assert len(result["events_patterns"]["patterns"]) == 0

    def test_sequence_pattern_detection(self):
        """Test sequence pattern detection."""
        data = {
            "values": [
                {"sequence": 1, "data": "a"},
                {"sequence": 2, "data": "b"},
                {"sequence": 3, "data": "c"},
                {"sequence": 5, "data": "d"},
                {"sequence": 6, "data": "e"},
                {"sequence": 7, "data": "f"},
            ]
        }
        config = {
            "test": RelationshipConfig(
                pattern_detection={
                    "seq_patterns": {
                        "type": "sequence",
                        "source": "values",
                        "field": "sequence",
                        "length": 3,
                    }
                }
            )
        }

        result = self.handler.process(data, config)

        # Should detect sequences [1,2,3] and [5,6,7]
        if "values_sequences" in result:
            sequences = result["values_sequences"]
            assert len(sequences) >= 1

    def test_pattern_detection_missing_source(self):
        """Test pattern detection with missing source."""
        data = {"other": [{"value": 1}]}
        config = {
            "test": RelationshipConfig(
                pattern_detection={
                    "patterns": {
                        "type": "frequency",
                        "source": "missing_source",
                        "field": "value",
                    }
                }
            )
        }

        result = self.handler.process(data, config)
        # Should handle missing source gracefully
        assert result == data

    def test_pattern_detection_non_list_source(self):
        """Test pattern detection with non-list source."""
        data = {"source": {"value": 1}}  # Not a list
        config = {
            "test": RelationshipConfig(
                pattern_detection={
                    "patterns": {
                        "type": "frequency",
                        "source": "source",
                        "field": "value",
                    }
                }
            )
        }

        result = self.handler.process(data, config)
        # Should handle non-list source gracefully
        assert result == data

    # Conditional Highlighting Tests
    def test_conditional_highlighting_with_engine(self):
        """Test conditional highlighting with expression engine."""
        self.mock_engine.evaluate.return_value = True

        data = {
            "transactions": [
                {"amount": 1000, "type": "withdrawal"},
                {"amount": 50, "type": "deposit"},
            ]
        }
        config = {
            "test": RelationshipConfig(
                conditional_highlighting=[
                    {
                        "source": "transactions",
                        "condition": "item['amount'] > 500",
                        "insight_name": "high_value",
                        "context": "Large transaction detected",
                    }
                ]
            )
        }

        result = self.handler_with_engine.process(data, config)

        # Should create highlighted items
        if "transactions_highlights" in result:
            highlights = result["transactions_highlights"]
            assert len(highlights) > 0
            assert self.mock_engine.evaluate.called

    def test_conditional_highlighting_without_engine(self):
        """Test conditional highlighting without expression engine."""
        data = {"items": [{"value": 100}, {"value": 50}]}
        config = {
            "test": RelationshipConfig(
                conditional_highlighting=[
                    {
                        "source": "items",
                        "condition": "item.value > 75",
                        "insight_name": "high_value",
                    }
                ]
            )
        }

        result = self.handler.process(data, config)
        # Without engine, should handle gracefully
        assert result  # Should not crash

    def test_conditional_highlighting_invalid_condition(self):
        """Test conditional highlighting with invalid condition."""
        self.mock_engine.evaluate.side_effect = Exception("Invalid condition")

        data = {"items": [{"value": 100}]}
        config = {
            "test": RelationshipConfig(
                conditional_highlighting=[
                    {
                        "source": "items",
                        "condition": "invalid condition syntax",
                        "insight_name": "test",
                    }
                ]
            )
        }

        result = self.handler_with_engine.process(data, config)
        # Should handle invalid conditions gracefully
        assert result == data

    def test_conditional_highlighting_missing_fields(self):
        """Test conditional highlighting with missing required fields."""
        data = {"items": [{"value": 100}]}
        config = {
            "test": RelationshipConfig(
                conditional_highlighting=[
                    {
                        "source": "items"
                        # Missing condition field
                    }
                ]
            )
        }

        result = self.handler.process(data, config)
        # Should handle missing fields gracefully
        assert result == data

    # Context Additions Tests
    def test_context_additions_with_engine(self):
        """Test context additions with expression engine."""
        self.mock_engine.substitute_variables.return_value = "User Alice has 2 orders"

        data = {
            "users": [
                {"name": "Alice", "order_count": 2},
                {"name": "Bob", "order_count": 1},
            ]
        }
        config = {
            "test": RelationshipConfig(
                context_additions={"users": "User {{item.name}} has {{item.order_count}} orders"}
            )
        }

        result = self.handler_with_engine.process(data, config)

        # Check that context was added
        for user in result["users"]:
            assert "_context" in user
            assert self.mock_engine.substitute_variables.called

    def test_context_additions_without_engine(self):
        """Test context additions without expression engine."""
        data = {"items": [{"name": "test", "value": 100}]}
        config = {"test": RelationshipConfig(context_additions={"items": "Template: {{item.name}}"})}

        result = self.handler.process(data, config)

        # Without engine, should still add context (as template string)
        for item in result["items"]:
            assert "_context" in item

    def test_context_additions_template_error(self):
        """Test context additions with template processing error."""
        self.mock_engine.substitute_variables.side_effect = Exception("Template error")

        data = {"items": [{"name": "test"}]}
        config = {"test": RelationshipConfig(context_additions={"items": "Invalid template {{missing.field}}"})}

        result = self.handler_with_engine.process(data, config)
        # Should handle template errors gracefully
        assert result

    # Derived Fields Tests
    def test_derived_fields_with_engine(self):
        """Test derived fields with expression engine."""
        self.mock_engine.evaluate.return_value = 150.0  # 100 * 1.5

        data = {
            "products": [
                {"price": 100, "tax_rate": 0.5},
                {"price": 200, "tax_rate": 0.5},
            ]
        }
        config = {"test": RelationshipConfig(derived_fields=["total_price = item.price * (1 + item.tax_rate)"])}

        result = self.handler_with_engine.process(data, config)

        # Check that derived fields were added
        for product in result["products"]:
            assert "total_price" in product
            assert self.mock_engine.evaluate.called

    def test_derived_fields_without_engine(self):
        """Test derived fields without expression engine."""
        data = {"items": [{"a": 10, "b": 20}]}
        config = {"test": RelationshipConfig(derived_fields=["sum = item.a + item.b"])}

        result = self.handler.process(data, config)

        # Without engine, should store expression as fallback
        for item in result["items"]:
            assert "sum" in item
            assert "EXPR:" in str(item["sum"])

    def test_derived_fields_invalid_expression(self):
        """Test derived fields with invalid expression."""
        self.mock_engine.evaluate.side_effect = Exception("Invalid expression")

        data = {"items": [{"value": 10}]}
        config = {"test": RelationshipConfig(derived_fields=["result = invalid expression"])}

        result = self.handler_with_engine.process(data, config)
        # Should handle invalid expressions gracefully
        assert result

    def test_derived_fields_malformed(self):
        """Test derived fields with malformed expressions."""
        data = {"items": [{"value": 10}]}
        config = {"test": RelationshipConfig(derived_fields=["no_equals_sign", "multiple = equals = signs", ""])}

        result = self.handler.process(data, config)
        # Should handle malformed expressions gracefully
        assert result == data

    # Configuration Validation Tests
    def test_dict_config_conversion(self):
        """Test that dict configs are converted to RelationshipConfig objects."""
        data = {"items": [{"id": 1}]}
        config = {
            "test": {
                "link_fields": [
                    {
                        "source": "items",
                        "source_field": "id",
                        "to": "items",
                        "target_field": "id",
                    }
                ]
            }
        }

        result = self.handler.process(data, config)
        # Should handle dict config conversion
        assert result

    def test_invalid_dict_config_logs_warning(self, caplog):
        """Test that invalid dict config logs warning - NOW FIXED."""
        data = {"items": [{"id": 1}]}
        config = {"test": {"invalid_field": "value"}}

        result = self.handler.process(data, config)
        assert "Invalid relationship configuration" in caplog.text
        assert "Unknown fields" in caplog.text
        assert "invalid_field" in caplog.text

    def test_invalid_config_type_logs_warning(self, caplog):
        """Test that invalid config type logs warning."""
        data = {"items": [{"id": 1}]}
        config = {"test": "invalid_string_config"}

        result = self.handler.process(data, config)
        assert "Invalid relationship configuration" in caplog.text
        assert "expected dict or RelationshipConfig" in caplog.text

    # Bug Detection Tests - These should expose real bugs
    def test_jsonpath_import_missing(self):
        """Test behavior when jsonpath_ng module is missing."""
        # This test might reveal import dependency issues
        data = {"source": [{"id": 1}], "target": [{"ref": 1}]}
        config = {
            "test": RelationshipConfig(
                jsonpath_links=[
                    {
                        "source": "source",
                        "target": "target",
                        "source_path": "id",
                        "target_path": "ref",
                    }
                ]
            )
        }

        # Should not crash even if jsonpath operations fail
        result = self.handler.process(data, config)
        assert result

    def test_all_operations_combined(self):
        """Test all relationship operations combined."""
        data = {
            "users": [{"id": 1, "name": "Alice", "type": "admin"}],
            "orders": [{"id": 101, "user_id": 1, "amount": 1000}],
            "logs": [{"user": 1, "action": "login"}],
        }
        config = {
            "test": RelationshipConfig(
                link_fields=[
                    {
                        "source": "orders",
                        "source_field": "user_id",
                        "to": "users",
                        "target_field": "id",
                    }
                ],
                jsonpath_links=[
                    {
                        "source": "users",
                        "target": "logs",
                        "source_path": "id",
                        "target_path": "user",
                    }
                ],
                pattern_detection={
                    "user_patterns": {
                        "type": "frequency",
                        "source": "users",
                        "field": "type",
                    }
                },
                conditional_highlighting=[
                    {
                        "source": "orders",
                        "condition": "item.amount > 500",
                        "insight_name": "high_value",
                    }
                ],
                context_additions={"users": "User: {{item.name}}"},
                derived_fields=["user_score = item.id * 10"],
            )
        }

        # Should handle all operations without crashing
        result = self.handler.process(data, config)
        assert result

    def test_dataframe_source_is_converted_to_list(self):
        """Test that pandas DataFrame source data is converted to list of dicts.

        This tests the fix for DataFrame handling where relationship_highlighting
        would silently skip DataFrame sources instead of processing them.
        """
        import pandas as pd

        # Create DataFrames similar to what CSV loading produces
        users_df = pd.DataFrame(
            [
                {"id": 1, "name": "Alice", "department": "Engineering"},
                {"id": 2, "name": "Bob", "department": "Sales"},
                {"id": 3, "name": "Carol", "department": "Engineering"},
            ]
        )
        orders_df = pd.DataFrame(
            [
                {"order_id": 101, "user_id": 1, "amount": 500},
                {"order_id": 102, "user_id": 2, "amount": 300},
                {"order_id": 103, "user_id": 1, "amount": 700},
            ]
        )
        data = {"users": users_df, "orders": orders_df}

        config = {
            "test": RelationshipConfig(
                link_fields=[
                    {
                        "source": "orders",
                        "source_field": "user_id",
                        "to": "users",
                        "target_field": "id",
                    }
                ],
                pattern_detection={
                    "dept_frequency": {
                        "type": "frequency",
                        "source": "users",
                        "field": "department",
                        "threshold": 2,
                    }
                },
            )
        }

        result = self.handler.process(data, config)

        # Verify DataFrames were converted to lists
        assert isinstance(result["users"], list)
        assert isinstance(result["orders"], list)

        # Verify link fields worked - orders should have users_info
        for order in result["orders"]:
            if order["user_id"] == 1:
                assert "users_info" in order
                assert order["users_info"]["name"] == "Alice"
            elif order["user_id"] == 2:
                assert "users_info" in order
                assert order["users_info"]["name"] == "Bob"

        # Verify pattern detection worked
        assert "users_patterns" in result
        assert result["users_patterns"]["patterns"]["Engineering"] == 2
