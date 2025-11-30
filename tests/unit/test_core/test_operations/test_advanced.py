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
from shedboxai.core.exceptions import ConfigurationError
from shedboxai.core.operations.advanced import (
    AdvancedOperationsHandler,
    get_nested_value,
    validate_aggregation_expression,
    validate_field_exists,
)


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

    # Nested Path Group By Tests (Issue #7 - Issue 2)
    def test_group_by_nested_path(self):
        """Test group_by with nested path like 'customers_info.membership_level'.

        This is a regression test for Issue #7 - Issue 2: group_by didn't support
        nested paths, which are common after joins that create nested structures.
        """
        data = {
            "sales": [
                {"id": 1, "amount": 100, "customers_info": {"membership_level": "Gold"}},
                {"id": 2, "amount": 150, "customers_info": {"membership_level": "Gold"}},
                {"id": 3, "amount": 200, "customers_info": {"membership_level": "Silver"}},
            ]
        }
        config = {
            "by_membership": AdvancedOperationConfig(
                source="sales",
                group_by="customers_info.membership_level",
                aggregate={"total": "SUM(amount)", "count": "COUNT(*)"},
            )
        }

        result = self.handler.process(data, config)
        summary = result["by_membership"]

        # Should have 2 groups
        assert len(summary) == 2

        groups = {item["customers_info.membership_level"]: item for item in summary}
        assert groups["Gold"]["total"] == 250.0
        assert groups["Gold"]["count"] == 2
        assert groups["Silver"]["total"] == 200.0
        assert groups["Silver"]["count"] == 1

    def test_group_by_deeply_nested_path(self):
        """Test group_by with deeply nested path."""
        data = {
            "orders": [
                {"id": 1, "amount": 100, "user": {"profile": {"tier": "premium"}}},
                {"id": 2, "amount": 200, "user": {"profile": {"tier": "premium"}}},
                {"id": 3, "amount": 50, "user": {"profile": {"tier": "basic"}}},
            ]
        }
        config = {
            "by_tier": AdvancedOperationConfig(
                source="orders",
                group_by="user.profile.tier",
                aggregate={"total": "SUM(amount)"},
            )
        }

        result = self.handler.process(data, config)
        groups = {item["user.profile.tier"]: item for item in result["by_tier"]}

        assert groups["premium"]["total"] == 300.0
        assert groups["basic"]["total"] == 50.0

    def test_group_by_nested_path_with_missing_intermediate(self):
        """Test group_by with nested path when some items don't have the intermediate path."""
        data = {
            "items": [
                {"id": 1, "value": 100, "meta": {"category": "A"}},
                {"id": 2, "value": 200, "meta": {"category": "A"}},
                {"id": 3, "value": 50},  # Missing 'meta' entirely
                {"id": 4, "value": 75, "meta": None},  # meta is None
            ]
        }
        config = {
            "by_category": AdvancedOperationConfig(
                source="items",
                group_by="meta.category",
                aggregate={"total": "SUM(value)", "count": "COUNT(*)"},
            )
        }

        result = self.handler.process(data, config)
        # Only items with valid nested path should be grouped
        assert len(result["by_category"]) == 1
        assert result["by_category"][0]["meta.category"] == "A"
        assert result["by_category"][0]["total"] == 300.0
        assert result["by_category"][0]["count"] == 2

    def test_group_by_simple_field_still_works(self):
        """Test that simple (non-nested) group_by still works after the nested path fix."""
        data = {
            "sales": [
                {"region": "North", "amount": 100},
                {"region": "South", "amount": 150},
                {"region": "North", "amount": 200},
            ]
        }
        config = {
            "by_region": AdvancedOperationConfig(
                source="sales",
                group_by="region",
                aggregate={"total": "SUM(amount)"},
            )
        }

        result = self.handler.process(data, config)
        groups = {item["region"]: item for item in result["by_region"]}

        assert groups["North"]["total"] == 300.0
        assert groups["South"]["total"] == 150.0

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
    def test_malformed_aggregation_expressions(self):
        """Test malformed aggregation expressions - NOW PROPERLY VALIDATED."""
        data = {"items": [{"category": "A", "value": 10}, {"category": "A", "value": 20}]}

        # Test with invalid function name
        config = {
            "result": AdvancedOperationConfig(
                source="items",
                group_by="category",
                aggregate={"bad1": "INVALID_FUNC(value)"},
            )
        }
        with pytest.raises(ConfigurationError, match="Invalid aggregation expression"):
            self.handler.process(data, config)

        # Test with missing parentheses
        config = {
            "result": AdvancedOperationConfig(
                source="items",
                group_by="category",
                aggregate={"bad2": "SUM value"},
            )
        }
        with pytest.raises(ConfigurationError, match="Invalid aggregation expression"):
            self.handler.process(data, config)

        # Test with malformed expression
        config = {
            "result": AdvancedOperationConfig(
                source="items",
                group_by="category",
                aggregate={"bad3": "COUNT("},
            )
        }
        with pytest.raises(ConfigurationError, match="Invalid aggregation expression"):
            self.handler.process(data, config)

        # Test with valid expression - should work
        config = {
            "result": AdvancedOperationConfig(
                source="items",
                group_by="category",
                aggregate={"good": "SUM(value)"},
            )
        }
        result = self.handler.process(data, config)
        assert "result" in result
        assert result["result"][0]["good"] == 30.0

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

    def test_dataframe_source_is_converted_to_list(self):
        """Test that pandas DataFrame source data is converted to list of dicts.

        This tests the fix for GitHub issue #4 where advanced_operations
        would return raw source data instead of aggregated results when
        the input was a DataFrame (as loaded from CSV files via pandas).
        """
        import pandas as pd

        # Create DataFrame similar to what CSV loading produces
        df = pd.DataFrame(
            [
                {"category": "Electronics", "amount": 100.0},
                {"category": "Electronics", "amount": 200.0},
                {"category": "Clothing", "amount": 50.0},
                {"category": "Clothing", "amount": 75.0},
                {"category": "Home", "amount": 150.0},
            ]
        )
        data = {"sales": df}

        config = {
            "category_totals": AdvancedOperationConfig(
                source="sales",
                group_by="category",
                aggregate={"total": "SUM(amount)", "count": "COUNT(*)"},
                sort="-total",
            )
        }

        result = self.handler.process(data, config)

        # Should have 3 aggregated groups, not 5 raw rows
        assert "category_totals" in result
        assert len(result["category_totals"]) == 3

        # Verify aggregation was performed correctly
        groups = {item["category"]: item for item in result["category_totals"]}
        assert groups["Electronics"]["total"] == 300.0
        assert groups["Electronics"]["count"] == 2
        assert groups["Clothing"]["total"] == 125.0
        assert groups["Clothing"]["count"] == 2
        assert groups["Home"]["total"] == 150.0
        assert groups["Home"]["count"] == 1

        # Verify sorting (descending by total)
        totals = [item["total"] for item in result["category_totals"]]
        assert totals == sorted(totals, reverse=True)

    def test_dataframe_source_sort_only(self):
        """Test DataFrame source with sort-only operation (no grouping)."""
        import pandas as pd

        df = pd.DataFrame(
            [
                {"name": "Alice", "score": 85},
                {"name": "Bob", "score": 92},
                {"name": "Carol", "score": 78},
            ]
        )
        data = {"students": df}

        config = {
            "sorted_students": AdvancedOperationConfig(
                source="students",
                sort="-score",
            )
        }

        result = self.handler.process(data, config)

        assert "sorted_students" in result
        assert len(result["sorted_students"]) == 3
        # Verify sorted by score descending
        assert result["sorted_students"][0]["name"] == "Bob"
        assert result["sorted_students"][1]["name"] == "Alice"
        assert result["sorted_students"][2]["name"] == "Carol"


class TestAggregationValidation:
    """Tests for aggregation expression validation (Feedback 3 - Issue 2)."""

    def test_simple_sum_aggregation_passes(self):
        """Test that simple SUM aggregation passes validation."""
        # Should not raise exception
        validate_aggregation_expression("SUM(amount)")

    def test_simple_avg_aggregation_passes(self):
        """Test that simple AVG aggregation passes validation."""
        validate_aggregation_expression("AVG(price)")

    def test_simple_count_star_passes(self):
        """Test that COUNT(*) passes validation."""
        validate_aggregation_expression("COUNT(*)")

    def test_simple_count_field_passes(self):
        """Test that COUNT(field) passes validation."""
        validate_aggregation_expression("COUNT(user_id)")

    def test_simple_min_aggregation_passes(self):
        """Test that simple MIN aggregation passes validation."""
        validate_aggregation_expression("MIN(created_date)")

    def test_simple_max_aggregation_passes(self):
        """Test that simple MAX aggregation passes validation."""
        validate_aggregation_expression("MAX(quantity)")

    def test_simple_median_aggregation_passes(self):
        """Test that simple MEDIAN aggregation passes validation."""
        validate_aggregation_expression("MEDIAN(value)")

    def test_simple_std_aggregation_passes(self):
        """Test that simple STD aggregation passes validation."""
        validate_aggregation_expression("STD(measurement)")

    def test_arithmetic_expression_fails(self):
        """Test that arithmetic expressions are rejected."""
        with pytest.raises(ConfigurationError, match="Invalid aggregation expression"):
            validate_aggregation_expression("SUM(amount) / 100")

    def test_multiple_arithmetic_fails(self):
        """Test that complex arithmetic expressions are rejected."""
        with pytest.raises(ConfigurationError, match="Invalid aggregation expression"):
            validate_aggregation_expression("SUM(price) * SUM(quantity)")

    def test_case_when_statement_fails(self):
        """Test that CASE WHEN statements are rejected."""
        with pytest.raises(ConfigurationError, match="Invalid aggregation expression"):
            validate_aggregation_expression("CASE WHEN amount > 100 THEN 'high' ELSE 'low' END")

    def test_count_distinct_fails(self):
        """Test that COUNT DISTINCT is rejected."""
        with pytest.raises(ConfigurationError, match="Invalid aggregation expression"):
            validate_aggregation_expression("COUNT(DISTINCT user_id)")

    def test_nested_functions_fail(self):
        """Test that nested functions are rejected."""
        with pytest.raises(ConfigurationError, match="Invalid aggregation expression"):
            validate_aggregation_expression("AVG(SUM(amount))")

    def test_whitespace_is_handled(self):
        """Test that whitespace doesn't break validation."""
        # These should pass
        validate_aggregation_expression("  SUM(amount)  ")
        validate_aggregation_expression("COUNT(  *  )")

    def test_error_message_suggests_workaround(self):
        """Test that error message includes workaround suggestion."""
        try:
            validate_aggregation_expression("SUM(x)/100")
            pytest.fail("Expected ConfigurationError")
        except ConfigurationError as e:
            error_msg = str(e)
            assert "SUM(x)/100" in error_msg
            assert "Only simple aggregations are supported" in error_msg
            assert "Workaround" in error_msg
            assert "derived fields" in error_msg
            assert "relationship_highlighting" in error_msg

    def test_error_message_lists_allowed_functions(self):
        """Test that error message lists allowed aggregation functions."""
        try:
            validate_aggregation_expression("INVALID(field)")
            pytest.fail("Expected ConfigurationError")
        except ConfigurationError as e:
            error_msg = str(e)
            assert "SUM(field)" in error_msg
            assert "AVG(field)" in error_msg
            assert "COUNT(*)" in error_msg
            assert "MIN(field)" in error_msg
            assert "MAX(field)" in error_msg

    def test_validation_integrated_in_handler(self):
        """Test that validation is triggered during handler processing."""
        handler = AdvancedOperationsHandler()
        data = {
            "transactions": [
                {"category": "A", "amount": 100},
                {"category": "A", "amount": 200},
            ]
        }
        config = {
            "summary": AdvancedOperationConfig(
                source="transactions",
                group_by="category",
                aggregate={"calculated": "SUM(amount) / 100"},  # Invalid expression
            )
        }

        # Should raise ConfigurationError when trying to process
        with pytest.raises(ConfigurationError, match="Invalid aggregation expression"):
            handler.process(data, config)

    def test_multiple_simple_aggregations_pass(self):
        """Test that multiple simple aggregations all pass validation."""
        handler = AdvancedOperationsHandler()
        data = {
            "sales": [
                {"region": "North", "amount": 100, "quantity": 5},
                {"region": "North", "amount": 200, "quantity": 10},
            ]
        }
        config = {
            "summary": AdvancedOperationConfig(
                source="sales",
                group_by="region",
                aggregate={
                    "total_amount": "SUM(amount)",
                    "avg_amount": "AVG(amount)",
                    "total_quantity": "SUM(quantity)",
                    "count": "COUNT(*)",
                },
            )
        }

        # Should not raise any exception
        result = handler.process(data, config)
        assert "summary" in result
        assert len(result["summary"]) == 1
        assert result["summary"][0]["total_amount"] == 300.0
        assert result["summary"][0]["avg_amount"] == 150.0


class TestFieldValidation:
    """Tests for field validation with suggestions (Feedback 3 - Issue 3)."""

    def test_field_exists_validation_passes(self):
        """Test that validation passes when field exists."""
        available_fields = ["id", "name", "email", "age"]
        # Should not raise exception
        validate_field_exists("name", available_fields, "users")

    def test_field_not_found_raises_error(self):
        """Test that missing field raises ConfigurationError."""
        available_fields = ["id", "name", "email"]
        with pytest.raises(ConfigurationError, match="Field 'username' not found"):
            validate_field_exists("username", available_fields, "users")

    def test_error_message_lists_available_fields(self):
        """Test that error message lists all available fields."""
        available_fields = ["id", "name", "email"]
        try:
            validate_field_exists("username", available_fields, "users")
            pytest.fail("Expected ConfigurationError")
        except ConfigurationError as e:
            error_msg = str(e)
            assert "Available fields: id, name, email" in error_msg

    def test_error_message_suggests_similar_field(self):
        """Test that error message suggests similar field name."""
        available_fields = ["id", "username", "email", "age"]
        try:
            validate_field_exists("user_name", available_fields, "users")
            pytest.fail("Expected ConfigurationError")
        except ConfigurationError as e:
            error_msg = str(e)
            assert "Did you mean: 'username'?" in error_msg

    def test_no_suggestion_when_no_similar_field(self):
        """Test that no suggestion is made when no similar field exists."""
        available_fields = ["id", "name", "email"]
        try:
            validate_field_exists("completely_different", available_fields, "users")
            pytest.fail("Expected ConfigurationError")
        except ConfigurationError as e:
            error_msg = str(e)
            assert "Did you mean" not in error_msg
            assert "Available fields:" in error_msg

    def test_case_sensitive_validation(self):
        """Test that field validation is case-sensitive."""
        available_fields = ["id", "Name", "email"]
        # "name" doesn't exist (only "Name" exists)
        with pytest.raises(ConfigurationError, match="Field 'name' not found"):
            validate_field_exists("name", available_fields, "users")

    def test_suggestion_with_typo(self):
        """Test that typos trigger helpful suggestions."""
        available_fields = ["transaction_id", "amount", "timestamp"]
        try:
            validate_field_exists("transaction_id", available_fields, "transactions")
            # Should pass without error
        except ConfigurationError:
            pytest.fail("Should not raise error for exact match")

        try:
            validate_field_exists("transactoin_id", available_fields, "transactions")  # Typo
            pytest.fail("Expected ConfigurationError")
        except ConfigurationError as e:
            error_msg = str(e)
            # Should suggest the correct field
            assert "Did you mean: 'transaction_id'?" in error_msg

    def test_source_name_in_error_message(self):
        """Test that source name appears in error message."""
        available_fields = ["id", "value"]
        try:
            validate_field_exists("missing_field", available_fields, "my_source")
            pytest.fail("Expected ConfigurationError")
        except ConfigurationError as e:
            error_msg = str(e)
            assert "my_source" in error_msg

    def test_empty_available_fields(self):
        """Test validation with empty available fields list."""
        available_fields = []
        with pytest.raises(ConfigurationError, match="Field 'any_field' not found"):
            validate_field_exists("any_field", available_fields, "empty_source")

    def test_special_characters_in_field_names(self):
        """Test that fields with special characters are handled correctly."""
        available_fields = ["id", "user_name", "email-address", "created.at"]
        # Exact match should work
        validate_field_exists("email-address", available_fields, "users")

        # Typo should still suggest
        try:
            validate_field_exists("email_address", available_fields, "users")  # Different separator
            pytest.fail("Expected ConfigurationError")
        except ConfigurationError as e:
            error_msg = str(e)
            # May or may not suggest depending on similarity threshold
            assert "Field 'email_address' not found" in error_msg


class TestGetNestedValue:
    """Tests for get_nested_value helper function (Issue #7 - Issue 2)."""

    def test_simple_top_level_field(self):
        """Test getting a simple top-level field."""
        item = {"id": 1, "name": "Alice"}
        assert get_nested_value(item, "id") == 1
        assert get_nested_value(item, "name") == "Alice"

    def test_single_level_nested(self):
        """Test getting a single-level nested field."""
        item = {"id": 1, "info": {"level": "Gold"}}
        assert get_nested_value(item, "info.level") == "Gold"

    def test_multi_level_nested(self):
        """Test getting a deeply nested field."""
        item = {"user": {"profile": {"settings": {"theme": "dark"}}}}
        assert get_nested_value(item, "user.profile.settings.theme") == "dark"

    def test_missing_top_level_field(self):
        """Test getting a missing top-level field returns None."""
        item = {"id": 1}
        assert get_nested_value(item, "nonexistent") is None

    def test_missing_intermediate_path(self):
        """Test getting a nested field with missing intermediate path returns None."""
        item = {"id": 1, "info": {"level": "Gold"}}
        assert get_nested_value(item, "meta.category") is None

    def test_none_intermediate_value(self):
        """Test getting a nested field when intermediate value is None."""
        item = {"id": 1, "info": None}
        assert get_nested_value(item, "info.level") is None

    def test_non_dict_intermediate_value(self):
        """Test getting a nested field when intermediate value is not a dict."""
        item = {"id": 1, "info": "not_a_dict"}
        assert get_nested_value(item, "info.level") is None

    def test_empty_dict(self):
        """Test getting value from empty dict."""
        item = {}
        assert get_nested_value(item, "any.path") is None

    def test_numeric_value(self):
        """Test getting numeric values."""
        item = {"stats": {"count": 42, "ratio": 0.75}}
        assert get_nested_value(item, "stats.count") == 42
        assert get_nested_value(item, "stats.ratio") == 0.75

    def test_list_value(self):
        """Test getting list values."""
        item = {"data": {"items": [1, 2, 3]}}
        assert get_nested_value(item, "data.items") == [1, 2, 3]
