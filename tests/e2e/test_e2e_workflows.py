"""
End-to-End tests for ShedBoxAI workflows.

These tests run complete workflows using the Flask test server and verify
that all operations work correctly in real scenarios.
"""

import pytest

from shedboxai.core.exceptions import ConfigurationError, OperationExecutionError
from shedboxai.pipeline import Pipeline


class TestBasicWorkflows:
    """E2E tests for basic ShedBoxAI workflows using existing test configs."""

    def test_basic_sources(self, market_analyser_server, config_file, run_pipeline):
        """Test 1: Basic data source loading from REST APIs."""
        config = config_file("test_1_basic_sources.yaml")
        result = run_pipeline(config)

        # Verify both data sources loaded
        assert "transactions" in result
        assert "demographics" in result

        # Verify transactions structure
        assert isinstance(result["transactions"], list)
        assert len(result["transactions"]) > 0
        assert "transaction_id" in result["transactions"][0]
        assert "amount" in result["transactions"][0]

        # Verify demographics structure (API returns a single dict, not a list)
        assert isinstance(result["demographics"], dict)
        assert len(result["demographics"]) > 0

    def test_contextual_filtering(self, market_analyser_server, config_file, run_pipeline):
        """Test 2: Contextual filtering operation."""
        config = config_file("test_2_contextual_filtering.yaml")
        result = run_pipeline(config)

        # Should have filtered results
        assert "high_value_transactions" in result or "large_transactions" in result

        # Verify all filtered transactions meet the condition
        filtered_key = "high_value_transactions" if "high_value_transactions" in result else "large_transactions"
        for transaction in result[filtered_key]:
            # High value transactions should have amount > some threshold
            assert "amount" in transaction

    def test_format_conversion(self, market_analyser_server, config_file, run_pipeline):
        """Test 3: Format conversion operation."""
        config = config_file("test_3_format_conversion.yaml")
        result = run_pipeline(config)

        # Should have converted data
        assert len(result) > 0

    def test_content_summarization(self, market_analyser_server, config_file, run_pipeline):
        """Test 4: Content summarization operation."""
        config = config_file("test_4_content_summarization.yaml")
        result = run_pipeline(config)

        # Should have summary data with _summary suffix
        summary_keys = [k for k in result.keys() if "_summary" in k]
        assert len(summary_keys) > 0

        # Verify summary structure (should have statistical metrics)
        summary = result[summary_keys[0]]
        assert isinstance(summary, dict)

    def test_advanced_operations(self, market_analyser_server, config_file, run_pipeline):
        """Test 5: Advanced operations (group, aggregate, sort, limit)."""
        config = config_file("test_5_advanced_operations.yaml")
        result = run_pipeline(config)

        # Should have grouped/aggregated results
        assert "transaction_summary" in result
        summary = result["transaction_summary"]

        # Verify structure
        assert isinstance(summary, list)
        assert len(summary) > 0

        # Verify aggregation fields
        first_row = summary[0]
        assert "transaction_type" in first_row  # group_by field
        assert "total_amount" in first_row  # aggregated field
        assert "avg_amount" in first_row  # aggregated field
        assert "transaction_count" in first_row  # aggregated field

        # Verify sorting (should be descending by total_amount)
        if len(summary) > 1:
            assert summary[0]["total_amount"] >= summary[1]["total_amount"]

        # Verify limit (should be at most 5)
        assert len(summary) <= 5

    def test_relationship_highlighting(self, market_analyser_server, config_file, run_pipeline):
        """Test 6: Relationship highlighting operation."""
        config = config_file("test_6_relationship_highlighting.yaml")
        result = run_pipeline(config)

        # Should have relationship data
        assert len(result) > 0

    def test_template_matching(self, market_analyser_server, config_file, run_pipeline):
        """Test 7: Template matching operation."""
        config = config_file("test_7_template_matching.yaml")
        result = run_pipeline(config)

        # Should have template-rendered content
        assert len(result) > 0

    def test_complex_pipeline(self, market_analyser_server, config_file, run_pipeline):
        """Test 8: Complex pipeline with multiple operations."""
        config = config_file("test_8_complex_pipeline.yaml")
        result = run_pipeline(config)

        # Should have multiple intermediate and final results
        assert len(result) > 0

    def test_graph_processing(self, market_analyser_server, config_file, run_pipeline):
        """Test 9: Graph-based processing with dependencies."""
        config = config_file("test_9_graph_processing.yaml")
        result = run_pipeline(config)

        # Should have results from graph execution
        assert len(result) > 0

    def test_csv_inline(self, market_analyser_server, config_file, run_pipeline):
        """Test 12: CSV inline data processing."""
        config = config_file("test_12_csv_inline.yaml")
        result = run_pipeline(config)

        # Should have inline CSV data loaded
        assert len(result) > 0


class TestAggregationValidation:
    """E2E tests for aggregation validation (Feedback 3 - Part 1, Issue 2)."""

    def test_simple_aggregations_work(self, market_analyser_server, tmp_path):
        """Test that all simple aggregations work in E2E scenario."""
        config = tmp_path / "test_simple_agg.yaml"
        config.write_text(
            """
data_sources:
  transactions:
    type: rest
    url: http://localhost:5000/api/transactions
    method: GET

processing:
  advanced_operations:
    summary:
      source: transactions
      group_by: transaction_type
      aggregate:
        total: SUM(amount)
        average: AVG(amount)
        count: COUNT(*)
        min_amount: MIN(amount)
        max_amount: MAX(amount)

output:
  type: print
  format: json
"""
        )

        pipeline = Pipeline(str(config))
        result = pipeline.run()

        # Verify all aggregations worked
        assert "summary" in result
        assert len(result["summary"]) > 0

        first_row = result["summary"][0]
        assert "total" in first_row
        assert "average" in first_row
        assert "count" in first_row
        assert "min_amount" in first_row
        assert "max_amount" in first_row

    def test_complex_aggregation_fails(self, market_analyser_server, tmp_path):
        """Test that complex aggregations are rejected with clear error."""
        config = tmp_path / "test_complex_agg.yaml"
        config.write_text(
            """
data_sources:
  transactions:
    type: rest
    url: http://localhost:5000/api/transactions
    method: GET

processing:
  advanced_operations:
    summary:
      source: transactions
      group_by: transaction_type
      aggregate:
        total_dollars: "SUM(amount) / 100"  # ❌ Complex expression

output:
  type: print
  format: json
"""
        )

        pipeline = Pipeline(str(config))

        # Should raise OperationExecutionError (wraps ConfigurationError)
        with pytest.raises(OperationExecutionError) as exc_info:
            pipeline.run()

        # Verify error message mentions the issue
        error_msg = str(exc_info.value)
        assert "Invalid aggregation expression" in error_msg
        assert "SUM(amount) / 100" in error_msg
        assert "Workaround" in error_msg
        assert "derived fields" in error_msg

    def test_count_distinct_fails(self, market_analyser_server, tmp_path):
        """Test that COUNT DISTINCT is rejected."""
        config = tmp_path / "test_count_distinct.yaml"
        config.write_text(
            """
data_sources:
  transactions:
    type: rest
    url: http://localhost:5000/api/transactions
    method: GET

processing:
  advanced_operations:
    summary:
      source: transactions
      group_by: transaction_type
      aggregate:
        unique_customers: "COUNT(DISTINCT customer_id)"  # ❌ DISTINCT not supported

output:
  type: print
  format: json
"""
        )

        pipeline = Pipeline(str(config))

        with pytest.raises(OperationExecutionError) as exc_info:
            pipeline.run()

        error_msg = str(exc_info.value)
        assert "Invalid aggregation expression" in error_msg
        assert "COUNT(DISTINCT customer_id)" in error_msg

    def test_derived_field_workaround(self, market_analyser_server, tmp_path):
        """Test that the derived field workaround works for complex calculations."""
        config = tmp_path / "test_derived_workaround.yaml"
        config.write_text(
            """
data_sources:
  transactions:
    type: rest
    url: http://localhost:5000/api/transactions
    method: GET

processing:
  relationship_highlighting:
    transactions:
      derived_fields:
        - amount_dollars = item.amount / 100

  advanced_operations:
    summary:
      source: transactions
      group_by: transaction_type
      aggregate:
        total_dollars: SUM(amount_dollars)  # ✅ Works with derived field!

output:
  type: print
  format: json
"""
        )

        pipeline = Pipeline(str(config))
        result = pipeline.run()

        # Should work successfully
        assert "summary" in result
        assert len(result["summary"]) > 0
        assert "total_dollars" in result["summary"][0]


class TestDerivedFieldsIsolation:
    """E2E tests for Issue #7: derived fields must only apply to target source."""

    def test_derived_fields_do_not_corrupt_other_sources(self, market_analyser_server, tmp_path):
        """Test that derived fields are ONLY applied to the configured source.

        This is a regression test for Issue #7: derived fields were being applied
        to ALL data sources, causing fields in other sources (like customers.membership_level)
        to be overwritten with None.
        """
        config = tmp_path / "test_derived_isolation.yaml"
        config.write_text(
            """
data_sources:
  sales:
    type: csv
    data:
      - id: 1
        customer_id: CUST001
        product_id: PROD001
        quantity: 2
      - id: 2
        customer_id: CUST002
        product_id: PROD002
        quantity: 3

  customers:
    type: csv
    data:
      - customer_id: CUST001
        name: Alice
        membership_level: Gold
      - customer_id: CUST002
        name: Bob
        membership_level: Silver

  products:
    type: csv
    data:
      - product_id: PROD001
        name: Widget
        price: 10.00
      - product_id: PROD002
        name: Gadget
        price: 20.00

processing:
  # First, link sales to customers
  relationship_highlighting:
    sales:
      link_fields:
        - source: sales
          source_field: customer_id
          to: customers
          target_field: customer_id
      # This derived field extracts membership_level from the linked customers_info
      # It should ONLY be applied to 'sales', NOT to 'customers' or 'products'
      derived_fields:
        - membership_level = item.customers_info.membership_level

output:
  type: print
  format: json
"""
        )

        pipeline = Pipeline(str(config))
        result = pipeline.run()

        # Convert DataFrames to lists for easier assertion
        import pandas as pd

        def to_list(data):
            if isinstance(data, pd.DataFrame):
                return data.to_dict("records")
            return data

        sales = to_list(result["sales"])
        customers = to_list(result["customers"])
        products = to_list(result["products"])

        # Sales should have the derived membership_level field
        assert "sales" in result
        assert sales[0]["membership_level"] == "Gold"
        assert sales[1]["membership_level"] == "Silver"

        # Customers should STILL have their original membership_level (not overwritten to None)
        assert "customers" in result
        assert customers[0]["membership_level"] == "Gold"
        assert customers[1]["membership_level"] == "Silver"

        # Products should NOT have the membership_level field at all
        assert "products" in result
        assert "membership_level" not in products[0]
        assert "membership_level" not in products[1]

    def test_join_then_aggregate_workflow(self, market_analyser_server, tmp_path):
        """Test the complete workflow from Issue #7: join + derived field + group_by.

        This tests the scenario where:
        1. Sales are linked to customers (join)
        2. A derived field extracts customer membership
        3. Group by the derived membership field

        Before the fix, step 2 would corrupt the customers source, causing step 3 to fail.
        """
        config = tmp_path / "test_join_aggregate.yaml"
        config.write_text(
            """
data_sources:
  sales:
    type: csv
    data:
      - id: 1
        customer_id: CUST001
        amount: 100
      - id: 2
        customer_id: CUST001
        amount: 150
      - id: 3
        customer_id: CUST002
        amount: 200

  customers:
    type: csv
    data:
      - customer_id: CUST001
        membership_level: Gold
      - customer_id: CUST002
        membership_level: Silver

processing:
  relationship_highlighting:
    sales:
      link_fields:
        - source: sales
          source_field: customer_id
          to: customers
          target_field: customer_id
      derived_fields:
        - customer_membership = item.customers_info.membership_level

  advanced_operations:
    sales_by_membership:
      source: sales
      group_by: customer_membership
      aggregate:
        total_amount: SUM(amount)
        count: COUNT(*)
      sort: -total_amount

output:
  type: print
  format: json
"""
        )

        pipeline = Pipeline(str(config))
        result = pipeline.run()

        # Convert DataFrames to lists for easier assertion
        import pandas as pd

        def to_list(data):
            if isinstance(data, pd.DataFrame):
                return data.to_dict("records")
            return data

        # Should have aggregated results by membership
        assert "sales_by_membership" in result
        summary = to_list(result["sales_by_membership"])
        assert len(summary) == 2  # Gold and Silver

        # Verify Gold has 2 sales totaling 250
        gold_row = next((r for r in summary if r["customer_membership"] == "Gold"), None)
        assert gold_row is not None
        assert gold_row["total_amount"] == 250
        assert gold_row["count"] == 2

        # Verify Silver has 1 sale totaling 200
        silver_row = next((r for r in summary if r["customer_membership"] == "Silver"), None)
        assert silver_row is not None
        assert silver_row["total_amount"] == 200
        assert silver_row["count"] == 1


class TestDataFrameHandling:
    """E2E tests for DataFrame handling in templates (Feedback 3 - Part 1, Issue 1)."""

    def test_dataframe_in_template_with_has_data(self, market_analyser_server, tmp_path):
        """Test that DataFrames work in templates with 'is has_data' test."""
        config = tmp_path / "test_dataframe_template.yaml"
        config.write_text(
            """
data_sources:
  transactions:
    type: rest
    url: http://localhost:5000/api/transactions
    method: GET

processing:
  template_matching:
    report:
      template: |
        {% if transactions is defined and transactions is has_data %}
        Found {{ transactions|length }} transactions
        {% else %}
        No transactions found
        {% endif %}

output:
  type: print
  format: json
"""
        )

        pipeline = Pipeline(str(config))
        result = pipeline.run()

        # Should render successfully without "truth value is ambiguous" error
        assert "report" in result
        assert "Found" in result["report"] or "No transactions" in result["report"]

    def test_dataframe_length_filter(self, market_analyser_server, tmp_path):
        """Test that length filter works with DataFrames."""
        config = tmp_path / "test_dataframe_length.yaml"
        config.write_text(
            """
data_sources:
  transactions:
    type: rest
    url: http://localhost:5000/api/transactions
    method: GET

processing:
  template_matching:
    count_report:
      template: "Transaction count: {{ transactions|length }}"

output:
  type: print
  format: json
"""
        )

        pipeline = Pipeline(str(config))
        result = pipeline.run()

        # Should work and show count
        assert "count_report" in result
        assert "Transaction count:" in result["count_report"]
        assert result["count_report"] != "Transaction count: "


@pytest.mark.slow
class TestFullWorkflows:
    """Comprehensive E2E tests for complete workflows (marked slow)."""

    def test_relationship_complex(self, market_analyser_server, config_file, run_pipeline):
        """Test 10: Complex relationship highlighting."""
        config = config_file("test_10_relationship_complex.yaml")
        result = run_pipeline(config)

        # Should successfully execute complex relationships
        assert len(result) > 0
