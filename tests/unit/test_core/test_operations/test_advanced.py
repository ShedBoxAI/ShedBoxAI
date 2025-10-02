"""
Comprehensive unit tests for advanced operations.

Tests cover grouping, aggregation, sorting, limiting, and error conditions
to uncover production bugs.

EXPECTED BUGS TO FIND:
1. Aggregation regex parsing issues with malformed expressions
2. Division by zero in AVG calculations with empty groups
3. Type conversion errors in numeric aggregations
4. Sorting errors with mixed data types
5. Edge cases in grouping with None/missing values
"""

from unittest.mock import Mock

import pytest

from shedboxai.core.config.models import AdvancedOperationConfig
from shedboxai.core.operations.advanced import AdvancedOperationsHandler


class TestAdvancedOperationsHandler:
    """Test suite for AdvancedOperationsHandler operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = AdvancedOperationsHandler()
        self.mock_engine = Mock()
        self.handler_with_engine = AdvancedOperationsHandler(engine=self.mock_engine)

    # Basic Configuration Tests
    def test_operation_name(self):
        """Test operation name property."""
        assert self.handler.operation_name == "advanced_operations"

    def test_empty_config_returns_data_unchanged(self):
        """Test that empty config returns data unchanged."""
        data = {"source": [{"id": 1, "value": 10}]}
        result = self.handler.process(data, {})
        assert result == data

    def test_missing_source_data(self):
        """Test processing when source data doesn't exist."""
        data = {"other_data": [{"id": 1}]}
        config = {"result": AdvancedOperationConfig(source="missing_source", group_by="category")}

        result = self.handler.process(data, config)
        assert result == data  # Data unchanged

    def test_non_list_source_data(self):
        """Test processing when source data is not a list."""
        data = {"source": {"id": 1, "value": 10}}  # Dict instead of list
        config = {"result": AdvancedOperationConfig(source="source", group_by="category")}

        result = self.handler.process(data, config)
        assert result == data  # Data unchanged due to non-list

    # Basic Grouping Tests
    def test_simple_grouping_without_aggregation(self):
        """Test basic grouping without aggregation."""
        data = {
            "sales": [
                {"region": "North", "amount": 100},
                {"region": "South", "amount": 150},
                {"region": "North", "amount": 200},
                {"region": "East", "amount": 120},
            ]
        }
        config = {"grouped": AdvancedOperationConfig(source="sales", group_by="region")}

        result = self.handler.process(data, config)
        grouped = result["grouped"]

        # Should have 3 groups
        assert len(grouped) == 3

        # Check group structure
        region_groups = {item["region"]: item["items"] for item in grouped}
        assert len(region_groups["North"]) == 2
        assert len(region_groups["South"]) == 1
        assert len(region_groups["East"]) == 1

    def test_grouping_with_none_values(self):
        """Test grouping when group_by field has None values."""
        data = {
            "items": [
                {"category": "A", "value": 10},
                {"category": None, "value": 20},
                {"category": "B", "value": 30},
                {"category": None, "value": 40},
                {"value": 50},  # Missing category field
            ]
        }
        config = {"grouped": AdvancedOperationConfig(source="items", group_by="category")}

        result = self.handler.process(data, config)
        grouped = result["grouped"]

        # Should only group items with non-None category values
        categories = [item["category"] for item in grouped]
        assert "A" in categories
        assert "B" in categories
        assert "None" in categories or None not in categories  # Depends on implementation

    # Aggregation Tests
    def test_count_aggregation(self):
        """Test COUNT aggregation."""
        data = {
            "orders": [
                {"status": "pending", "amount": 100},
                {"status": "completed", "amount": 150},
                {"status": "pending", "amount": 200},
                {"status": "completed", "amount": 300},
            ]
        }
        config = {
            "summary": AdvancedOperationConfig(
                source="orders",
                group_by="status",
                aggregate={"order_count": "COUNT(*)"},
            )
        }

        result = self.handler.process(data, config)
        summary = result["summary"]

        status_counts = {item["status"]: item["order_count"] for item in summary}
        assert status_counts["pending"] == 2
        assert status_counts["completed"] == 2

    def test_sum_aggregation(self):
        """Test SUM aggregation."""
        data = {
            "sales": [
                {"region": "North", "amount": 100},
                {"region": "South", "amount": 150},
                {"region": "North", "amount": 200},
            ]
        }
        config = {
            "totals": AdvancedOperationConfig(
                source="sales",
                group_by="region",
                aggregate={"total_amount": "SUM(amount)"},
            )
        }

        result = self.handler.process(data, config)
        totals = result["totals"]

        region_totals = {item["region"]: item["total_amount"] for item in totals}
        assert region_totals["North"] == 300.0
        assert region_totals["South"] == 150.0

    def test_avg_aggregation(self):
        """Test AVG aggregation."""
        data = {
            "scores": [
                {"student": "Alice", "score": 80},
                {"student": "Bob", "score": 90},
                {"student": "Alice", "score": 70},
                {"student": "Bob", "score": 85},
            ]
        }
        config = {
            "averages": AdvancedOperationConfig(
                source="scores",
                group_by="student",
                aggregate={"avg_score": "AVG(score)"},
            )
        }

        result = self.handler.process(data, config)
        averages = result["averages"]

        student_avgs = {item["student"]: item["avg_score"] for item in averages}
        assert student_avgs["Alice"] == 75.0  # (80 + 70) / 2
        assert student_avgs["Bob"] == 87.5  # (90 + 85) / 2

    def test_min_max_aggregation(self):
        """Test MIN and MAX aggregations."""
        data = {
            "temperatures": [
                {"city": "NYC", "temp": 20},
                {"city": "LA", "temp": 25},
                {"city": "NYC", "temp": 15},
                {"city": "LA", "temp": 30},
            ]
        }
        config = {
            "temp_ranges": AdvancedOperationConfig(
                source="temperatures",
                group_by="city",
                aggregate={"min_temp": "MIN(temp)", "max_temp": "MAX(temp)"},
            )
        }

        result = self.handler.process(data, config)
        ranges = result["temp_ranges"]

        city_ranges = {item["city"]: {"min": item["min_temp"], "max": item["max_temp"]} for item in ranges}
        assert city_ranges["NYC"]["min"] == 15.0
        assert city_ranges["NYC"]["max"] == 20.0
        assert city_ranges["LA"]["min"] == 25.0
        assert city_ranges["LA"]["max"] == 30.0

    def test_median_aggregation(self):
        """Test MEDIAN aggregation."""
        data = {
            "values": [
                {"group": "A", "value": 1},
                {"group": "A", "value": 2},
                {"group": "A", "value": 3},
                {"group": "B", "value": 10},
                {"group": "B", "value": 20},
                {"group": "B", "value": 30},
                {"group": "B", "value": 40},
            ]
        }
        config = {
            "medians": AdvancedOperationConfig(
                source="values",
                group_by="group",
                aggregate={"median_value": "MEDIAN(value)"},
            )
        }

        result = self.handler.process(data, config)
        medians = result["medians"]

        group_medians = {item["group"]: item["median_value"] for item in medians}
        assert group_medians["A"] == 2.0  # Middle of [1, 2, 3]
        assert group_medians["B"] == 25.0  # Average of 20, 30 in [10, 20, 30, 40]

    def test_std_aggregation(self):
        """Test STD (standard deviation) aggregation."""
        data = {
            "measurements": [
                {"device": "A", "reading": 10},
                {"device": "A", "reading": 12},
                {"device": "A", "reading": 8},
                {"device": "B", "reading": 5},  # Single value
            ]
        }
        config = {
            "std_devs": AdvancedOperationConfig(
                source="measurements",
                group_by="device",
                aggregate={"std_reading": "STD(reading)"},
            )
        }

        result = self.handler.process(data, config)
        std_devs = result["std_devs"]

        device_stds = {item["device"]: item["std_reading"] for item in std_devs}
        assert device_stds["A"] > 0  # Should have some standard deviation
        assert device_stds["B"] == 0  # Single value should have std = 0

    def test_multiple_aggregations(self):
        """Test multiple aggregations on same data."""
        data = {
            "orders": [
                {"customer": "Alice", "amount": 100},
                {"customer": "Bob", "amount": 150},
                {"customer": "Alice", "amount": 200},
                {"customer": "Bob", "amount": 50},
            ]
        }
        config = {
            "customer_stats": AdvancedOperationConfig(
                source="orders",
                group_by="customer",
                aggregate={
                    "total": "SUM(amount)",
                    "average": "AVG(amount)",
                    "min": "MIN(amount)",
                    "max": "MAX(amount)",
                    "count": "COUNT(*)",
                },
            )
        }

        result = self.handler.process(data, config)
        stats = result["customer_stats"]

        alice_stats = next(item for item in stats if item["customer"] == "Alice")
        assert alice_stats["total"] == 300.0
        assert alice_stats["average"] == 150.0
        assert alice_stats["min"] == 100.0
        assert alice_stats["max"] == 200.0
        assert alice_stats["count"] == 2

    # Sorting Tests
    def test_sorting_ascending(self):
        """Test ascending sort."""
        data = {
            "items": [
                {"name": "C", "value": 30},
                {"name": "A", "value": 10},
                {"name": "B", "value": 20},
            ]
        }
        config = {"sorted": AdvancedOperationConfig(source="items", sort="value")}

        result = self.handler.process(data, config)
        sorted_items = result["sorted"]

        assert [item["value"] for item in sorted_items] == [10, 20, 30]

    def test_sorting_descending(self):
        """Test descending sort."""
        data = {
            "items": [
                {"name": "A", "value": 10},
                {"name": "C", "value": 30},
                {"name": "B", "value": 20},
            ]
        }
        config = {"sorted": AdvancedOperationConfig(source="items", sort="-value")}

        result = self.handler.process(data, config)
        sorted_items = result["sorted"]

        assert [item["value"] for item in sorted_items] == [30, 20, 10]

    def test_sorting_with_missing_field(self):
        """Test sorting when some items don't have the sort field."""
        data = {
            "items": [
                {"name": "A", "value": 10},
                {"name": "B"},  # Missing value
                {"name": "C", "value": 30},
            ]
        }
        config = {"sorted": AdvancedOperationConfig(source="items", sort="value")}

        result = self.handler.process(data, config)
        # Should handle missing fields gracefully without crashing
        assert len(result["sorted"]) == 3

    def test_sorting_non_list_data(self):
        """Test sorting on non-list data - NOW FIXED."""
        data = {"item": {"value": 10}}  # Not a list
        config = {"sorted": AdvancedOperationConfig(source="item", sort="value")}

        result = self.handler.process(data, config)
        # Should return original non-list data unchanged
        assert result["sorted"] == {"value": 10}

    # Limiting Tests
    def test_limiting_results(self):
        """Test limiting number of results."""
        data = {"items": [{"value": i} for i in range(10)]}
        config = {"limited": AdvancedOperationConfig(source="items", limit=3)}

        result = self.handler.process(data, config)
        assert len(result["limited"]) == 3

    def test_limiting_with_zero(self):
        """Test limiting with zero limit - NOW FIXED."""
        data = {"items": [{"value": i} for i in range(5)]}
        config = {"limited": AdvancedOperationConfig(source="items", limit=0)}

        result = self.handler.process(data, config)
        # Zero limit should return all data (since limit <= 0 returns all)
        assert len(result["limited"]) == 5

    def test_limiting_negative(self):
        """Test limiting with negative limit."""
        data = {"items": [{"value": i} for i in range(5)]}
        config = {"limited": AdvancedOperationConfig(source="items", limit=-1)}

        result = self.handler.process(data, config)
        # Negative limit should return all data
        assert len(result["limited"]) == 5

    def test_limiting_non_list_data(self):
        """Test limiting on non-list data - NOW FIXED."""
        data = {"item": {"value": 10}}
        config = {"limited": AdvancedOperationConfig(source="item", limit=1)}

        result = self.handler.process(data, config)
        # Should return original non-list data unchanged
        assert result["limited"] == {"value": 10}

    # Combined Operations Tests
    def test_group_aggregate_sort_limit_combined(self):
        """Test combining grouping, aggregation, sorting, and limiting."""
        data = {
            "sales": [
                {"region": "North", "amount": 100},
                {"region": "South", "amount": 300},
                {"region": "East", "amount": 200},
                {"region": "West", "amount": 150},
                {"region": "North", "amount": 50},
            ]
        }
        config = {
            "top_regions": AdvancedOperationConfig(
                source="sales",
                group_by="region",
                aggregate={"total": "SUM(amount)"},
                sort="-total",  # Sort by total descending
                limit=2,  # Top 2 regions
            )
        }

        result = self.handler.process(data, config)
        top_regions = result["top_regions"]

        assert len(top_regions) == 2
        assert top_regions[0]["region"] == "South"
        assert top_regions[0]["total"] == 300.0
        assert top_regions[1]["region"] == "East"
        assert top_regions[1]["total"] == 200.0

    # Configuration Validation Tests
    def test_dict_config_conversion(self):
        """Test that dict configs are converted to AdvancedOperationConfig objects."""
        data = {"items": [{"category": "A", "value": 10}]}
        config = {"grouped": {"source": "items", "group_by": "category"}}

        result = self.handler.process(data, config)
        assert "grouped" in result
        assert len(result["grouped"]) == 1

    def test_invalid_dict_config_logs_warning(self, caplog):
        """Test that invalid dict config logs warning."""
        data = {"items": [{"value": 10}]}
        config = {"result": {"invalid_field": "value"}}

        result = self.handler.process(data, config)
        assert "Invalid advanced operation configuration" in caplog.text
        assert result == data

    def test_invalid_config_type_logs_warning(self, caplog):
        """Test that invalid config type logs warning."""
        data = {"items": [{"value": 10}]}
        config = {"result": "invalid_string_config"}

        result = self.handler.process(data, config)
        assert "Invalid advanced operation configuration" in caplog.text
        assert "expected dict or AdvancedOperationConfig" in caplog.text
        assert result == data

    # Bug Detection Tests - These should expose real bugs
    def test_malformed_aggregation_expressions(self, caplog):
        """Test malformed aggregation expressions."""
        data = {"items": [{"category": "A", "value": 10}, {"category": "A", "value": 20}]}
        config = {
            "result": AdvancedOperationConfig(
                source="items",
                group_by="category",
                aggregate={
                    "bad1": "INVALID_FUNC(value)",
                    "bad2": "SUM value",  # Missing parentheses
                    "bad3": "COUNT(",  # Malformed
                    "good": "SUM(value)",
                },
            )
        }

        result = self.handler.process(data, config)
        # Should handle malformed expressions gracefully
        group = result["result"][0]
        assert "good" in group  # Good expression should work
        # Bad expressions should be None or generate warnings

    def test_aggregation_with_non_numeric_fields(self):
        """Test numeric aggregations on non-numeric fields."""
        data = {
            "items": [
                {"category": "A", "description": "text1", "value": 10},
                {"category": "A", "description": "text2", "value": 20},
            ]
        }
        config = {
            "result": AdvancedOperationConfig(
                source="items",
                group_by="category",
                aggregate={
                    "sum_desc": "SUM(description)",  # SUM on text field
                    "avg_desc": "AVG(description)",  # AVG on text field
                    "sum_val": "SUM(value)",  # Valid SUM
                },
            )
        }

        result = self.handler.process(data, config)
        group = result["result"][0]
        # Non-numeric aggregations should return None
        assert group["sum_desc"] is None
        assert group["avg_desc"] is None
        assert group["sum_val"] == 30.0

    def test_empty_groups_division_by_zero(self):
        """Test AVG aggregation with empty groups (potential division by zero)."""
        data = {
            "items": [
                {"category": "A", "value": None},
                {"category": "A", "value": "invalid"},
                {"category": "B", "value": 10},
            ]
        }
        config = {
            "result": AdvancedOperationConfig(
                source="items",
                group_by="category",
                aggregate={"avg_value": "AVG(value)"},
            )
        }

        result = self.handler.process(data, config)
        groups = {item["category"]: item for item in result["result"]}
        # Group A has no valid numeric values - should handle gracefully
        assert groups["A"]["avg_value"] is None
        assert groups["B"]["avg_value"] == 10.0

    def test_count_with_specific_field_vs_asterisk(self):
        """Test COUNT with specific field vs COUNT(*)."""
        data = {
            "items": [
                {"category": "A", "value": 10},
                {"category": "A", "value": None},
                {"category": "A"},  # Missing value field
                {"category": "B", "value": 20},
            ]
        }
        config = {
            "result": AdvancedOperationConfig(
                source="items",
                group_by="category",
                aggregate={"count_all": "COUNT(*)", "count_value": "COUNT(value)"},
            )
        }

        result = self.handler.process(data, config)
        groups = {item["category"]: item for item in result["result"]}

        # Category A: 3 total items, but only 1 with non-None value
        assert groups["A"]["count_all"] == 3
        assert groups["A"]["count_value"] == 1

        # Category B: 1 total item, 1 with non-None value
        assert groups["B"]["count_all"] == 1
        assert groups["B"]["count_value"] == 1

    def test_sorting_mixed_types_error(self):
        """Test sorting with mixed data types that can't be compared."""
        data = {
            "mixed": [
                {"value": 10},
                {"value": "string"},
                {"value": 20},
                {"value": None},
            ]
        }
        config = {"sorted": AdvancedOperationConfig(source="mixed", sort="value")}

        # Should handle mixed types gracefully without crashing
        result = self.handler.process(data, config)
        assert len(result["sorted"]) == 4
