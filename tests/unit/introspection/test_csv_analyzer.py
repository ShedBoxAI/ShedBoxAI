"""
Test suite for CSV analyzer functionality.

This module contains comprehensive tests for the CSVAnalyzer class,
covering all analysis capabilities and edge cases.
"""

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from shedboxai.core.introspection.analyzers.csv_analyzer import CSVAnalyzer
from shedboxai.core.introspection.models import AnalysisStatus, SourceType


class TestCSVAnalyzer:
    def setup_method(self):
        self.analyzer = CSVAnalyzer()

    def test_analyze_sample_csv(self):
        """Test analysis with sample CSV fixture"""
        config = {
            "name": "test_customers",
            "type": "csv",
            "path": "tests/fixtures/introspection/sample_customers.csv",
        }

        result = self.analyzer.analyze(config, sample_size=10)

        assert result.success
        assert result.name == "test_customers"
        assert result.type == SourceType.CSV
        assert result.column_count > 0
        assert result.size_info is not None
        assert result.schema_info is not None
        assert len(result.sample_data) > 0
        assert len(result.llm_recommendations) > 0

    def test_analyze_inline_data(self):
        """Test analysis with inline CSV data"""
        config = {
            "name": "inline_test",
            "type": "csv",
            "data": [
                {"id": 1, "name": "Alice", "age": 25},
                {"id": 2, "name": "Bob", "age": 30},
                {"id": 3, "name": "Charlie", "age": 35},
            ],
        }

        result = self.analyzer.analyze(config)

        assert result.success
        assert result.column_count == 3
        assert len(result.sample_data) == 3
        assert result.size_info.record_count == 3

    def test_column_type_detection(self):
        """Test column type detection"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("id,name,age,salary,active,created_date\n")
            f.write("1,Alice,25,50000.5,true,2023-01-15\n")
            f.write("2,Bob,30,60000.0,false,2023-02-20\n")
            f.write("3,Charlie,35,70000.0,true,2023-03-10\n")
            temp_path = f.name

        try:
            config = {"name": "test", "path": temp_path}
            result = self.analyzer.analyze(config)

            assert result.success

            # Check column types
            columns_by_name = {col.name: col for col in result.schema_info.columns}

            assert columns_by_name["id"].type == "integer"
            assert columns_by_name["name"].type == "string"
            assert columns_by_name["age"].type == "integer"
            assert columns_by_name["salary"].type == "float"
            assert columns_by_name["active"].type == "boolean"

        finally:
            os.unlink(temp_path)

    def test_large_dataset_detection(self):
        """Test large dataset detection and recommendations"""
        # Create large dataset
        large_data = [{"id": i, "value": f"item_{i}"} for i in range(60000)]

        config = {"name": "large_test", "data": large_data}

        result = self.analyzer.analyze(config)

        assert result.success
        assert result.size_info.is_large_dataset

        # Check for appropriate recommendations
        recommendations = " ".join(result.llm_recommendations)
        assert "large dataset" in recommendations.lower()
        assert "filtering" in recommendations.lower() or "aggregation" in recommendations.lower()

    def test_primary_key_detection(self):
        """Test primary key detection"""
        config = {
            "name": "pk_test",
            "data": [
                {"user_id": "USER_001", "name": "Alice"},
                {"user_id": "USER_002", "name": "Bob"},
                {"user_id": "USER_003", "name": "Charlie"},
            ],
        }

        result = self.analyzer.analyze(config)

        assert result.success

        # Find user_id column
        user_id_col = next(col for col in result.schema_info.columns if col.name == "user_id")
        assert user_id_col.is_likely_primary_key
        assert user_id_col.uniqueness_ratio == 1.0
        assert user_id_col.null_percentage == 0.0

    def test_statistical_analysis(self):
        """Test statistical analysis for numeric columns"""
        config = {
            "name": "stats_test",
            "data": [
                {"id": 1, "score": 85.5},
                {"id": 2, "score": 92.0},
                {"id": 3, "score": 78.5},
                {"id": 4, "score": 95.0},
                {"id": 5, "score": 88.0},
            ],
        }

        result = self.analyzer.analyze(config)

        assert result.success

        # Find score column
        score_col = next(col for col in result.schema_info.columns if col.name == "score")

        assert score_col.type == "float"
        assert score_col.min_value == 78.5
        assert score_col.max_value == 95.0
        assert score_col.mean is not None
        assert score_col.std is not None
        assert score_col.quartiles is not None
        assert len(score_col.quartiles) == 3

    def test_categorical_detection(self):
        """Test categorical column detection"""
        config = {
            "name": "cat_test",
            "data": [
                {"id": 1, "category": "A", "status": "active"},
                {"id": 2, "category": "B", "status": "inactive"},
                {"id": 3, "category": "A", "status": "active"},
                {"id": 4, "category": "C", "status": "active"},
                {"id": 5, "category": "B", "status": "inactive"},
            ],
        }

        result = self.analyzer.analyze(config)

        assert result.success

        # Check categorical columns
        cat_col = next(col for col in result.schema_info.columns if col.name == "category")
        status_col = next(col for col in result.schema_info.columns if col.name == "status")

        assert cat_col.is_categorical
        assert status_col.is_categorical
        assert cat_col.value_counts is not None
        assert status_col.value_counts is not None

    def test_error_handling(self):
        """Test error handling for invalid files"""
        # Test missing file
        config = {"name": "missing", "path": "/nonexistent/file.csv"}
        result = self.analyzer.analyze(config)

        assert not result.success
        assert result.status == AnalysisStatus.FAILED
        assert "no such file" in result.error_message.lower() or "not found" in result.error_message.lower()
        assert result.error_hint is not None

    def test_encoding_detection(self):
        """Test encoding detection and handling"""
        # This would require creating files with different encodings
        # For now, test the detection method
        encoding = self.analyzer._detect_encoding("tests/fixtures/introspection/sample_customers.csv")
        assert encoding is not None
        assert isinstance(encoding, str)

    def test_sample_data_generation(self):
        """Test sample data generation with different strategies"""
        # Large dataset for stratified sampling
        config = {
            "name": "sample_test",
            "data": [
                {
                    "id": i,
                    "category": "A" if i % 3 == 0 else "B" if i % 3 == 1 else "C",
                    "value": i * 10,
                }
                for i in range(100)
            ],
        }

        result = self.analyzer.analyze(config, sample_size=15)

        assert result.success
        assert len(result.sample_data) == 15

        # Check that sample represents different categories
        categories = set(item["category"] for item in result.sample_data)
        assert len(categories) > 1  # Should have multiple categories

    def test_delimiter_detection(self):
        """Test CSV delimiter detection"""
        # Test semicolon delimiter
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("id;name;value\n")
            f.write("1;Alice;100\n")
            f.write("2;Bob;200\n")
            temp_path = f.name

        try:
            detected_delimiter = self.analyzer._detect_delimiter(temp_path)
            assert detected_delimiter == ";"
        finally:
            os.unlink(temp_path)

    def test_string_pattern_detection(self):
        """Test string pattern detection"""
        config = {
            "name": "pattern_test",
            "data": [
                {"email": "alice@example.com", "user_id": "USER_001"},
                {"email": "bob@example.com", "user_id": "USER_002"},
                {"email": "charlie@example.com", "user_id": "USER_003"},
            ],
        }

        result = self.analyzer.analyze(config)

        assert result.success

        # Check pattern detection
        columns_by_name = {col.name: col for col in result.schema_info.columns}
        assert columns_by_name["email"].pattern == "email"
        assert columns_by_name["user_id"].pattern == "structured_id"

    def test_date_pattern_detection_iso_format(self):
        """Test that ISO date format is detected as date type or date pattern, not phone"""
        config = {
            "name": "date_pattern_test",
            "data": [
                {"date": "2024-01-15", "signup_date": "2023-06-20"},
                {"date": "2024-01-16", "signup_date": "2023-07-15"},
                {"date": "2024-01-17", "signup_date": "2023-08-10"},
            ],
        }

        result = self.analyzer.analyze(config)

        assert result.success

        columns_by_name = {col.name: col for col in result.schema_info.columns}
        # Dates should be detected as date type OR date pattern, NOT phone
        # The type detection may catch it as date_string before pattern detection runs
        assert columns_by_name["date"].type == "date_string" or columns_by_name["date"].pattern == "date"
        assert columns_by_name["signup_date"].type == "date_string" or columns_by_name["signup_date"].pattern == "date"
        # Must NOT be detected as phone
        assert columns_by_name["date"].pattern != "phone"
        assert columns_by_name["signup_date"].pattern != "phone"

    def test_date_pattern_not_confused_with_phone(self):
        """Test that date patterns like 2024-01-15 are NOT matched as phone numbers"""
        # This is a regression test for the bug where dates were incorrectly
        # classified as phone numbers due to overly permissive regex
        config = {
            "name": "date_vs_phone_test",
            "data": [
                {"date": "2024-01-15", "phone": "(555) 123-4567"},
                {"date": "2024-02-20", "phone": "(555) 234-5678"},
                {"date": "2024-03-10", "phone": "(555) 345-6789"},
            ],
        }

        result = self.analyzer.analyze(config)

        assert result.success

        columns_by_name = {col.name: col for col in result.schema_info.columns}
        # Date column should be date type or pattern, NOT phone
        assert columns_by_name["date"].type == "date_string" or columns_by_name["date"].pattern == "date"
        assert columns_by_name["date"].pattern != "phone"
        # Phone column should be "phone" pattern
        assert columns_by_name["phone"].pattern == "phone"

    def test_phone_pattern_detection(self):
        """Test that various phone formats are correctly detected"""
        config = {
            "name": "phone_pattern_test",
            "data": [
                {"phone": "(555) 123-4567"},
                {"phone": "(555) 234-5678"},
                {"phone": "(555) 345-6789"},
            ],
        }

        result = self.analyzer.analyze(config)

        assert result.success

        columns_by_name = {col.name: col for col in result.schema_info.columns}
        assert columns_by_name["phone"].pattern == "phone"

    def test_us_date_format_detection(self):
        """Test that US date format (MM/DD/YYYY) is detected as date"""
        config = {
            "name": "us_date_test",
            "data": [
                {"date": "01/15/2024"},
                {"date": "02/20/2024"},
                {"date": "03/10/2024"},
            ],
        }

        result = self.analyzer.analyze(config)

        assert result.success

        columns_by_name = {col.name: col for col in result.schema_info.columns}
        # Should be date type or pattern, not phone
        assert columns_by_name["date"].type == "date_string" or columns_by_name["date"].pattern == "date"
        assert columns_by_name["date"].pattern != "phone"

    def test_identifier_detection(self):
        """Test identifier column detection"""
        config = {
            "name": "id_test",
            "data": [
                {"customer_id": "CUST_001", "order_key": "ORD_123", "name": "Alice"},
                {"customer_id": "CUST_002", "order_key": "ORD_124", "name": "Bob"},
            ],
        }

        result = self.analyzer.analyze(config)

        assert result.success

        # Check identifier detection
        columns_by_name = {col.name: col for col in result.schema_info.columns}
        assert columns_by_name["customer_id"].is_identifier
        assert columns_by_name["order_key"].is_identifier
        assert not columns_by_name["name"].is_identifier

    def test_null_handling(self):
        """Test handling of null values"""
        config = {
            "name": "null_test",
            "data": [
                {"id": 1, "name": "Alice", "age": 25, "email": "alice@example.com"},
                {"id": 2, "name": None, "age": 30, "email": None},
                {
                    "id": 3,
                    "name": "Charlie",
                    "age": None,
                    "email": "charlie@example.com",
                },
            ],
        }

        result = self.analyzer.analyze(config)

        assert result.success

        # Check null percentage calculations
        columns_by_name = {col.name: col for col in result.schema_info.columns}
        assert columns_by_name["id"].null_percentage == 0.0
        assert columns_by_name["name"].null_percentage == pytest.approx(33.33, rel=1e-1)
        assert columns_by_name["age"].null_percentage == pytest.approx(33.33, rel=1e-1)
        assert columns_by_name["email"].null_percentage == pytest.approx(33.33, rel=1e-1)

    def test_mixed_numeric_types(self):
        """Test handling of mixed numeric types"""
        config = {
            "name": "mixed_numeric",
            "data": [{"value": 1}, {"value": 2.5}, {"value": 3}, {"value": 4.7}],
        }

        result = self.analyzer.analyze(config)

        assert result.success

        # Should detect as float due to mixed integer/float values
        value_col = next(col for col in result.schema_info.columns if col.name == "value")
        assert value_col.type == "float"
        assert value_col.min_value == 1.0
        assert value_col.max_value == 4.7

    def test_empty_dataset(self):
        """Test handling of empty datasets"""
        config = {"name": "empty_test", "data": []}

        result = self.analyzer.analyze(config)

        assert not result.success
        assert result.status == AnalysisStatus.FAILED
        assert "empty" in result.error_message.lower()

    def test_single_row_dataset(self):
        """Test handling of single row datasets"""
        config = {
            "name": "single_row",
            "data": [{"id": 1, "name": "Alice", "value": 100.5}],
        }

        result = self.analyzer.analyze(config)

        assert result.success
        assert result.size_info.record_count == 1
        assert len(result.sample_data) == 1

        # Check that statistics are still calculated correctly
        value_col = next(col for col in result.schema_info.columns if col.name == "value")
        assert value_col.min_value == 100.5
        assert value_col.max_value == 100.5
        assert value_col.mean == 100.5
        assert value_col.std == 0.0  # Single value should have std=0

    def test_unicode_handling(self):
        """Test handling of Unicode characters"""
        config = {
            "name": "unicode_test",
            "data": [
                {"name": "Αλίκη", "city": "Αθήνα", "description": "Test with Greek"},
                {"name": "张三", "city": "北京", "description": "Test with Chinese"},
                {"name": "José", "city": "Madrid", "description": "Test with accents"},
            ],
        }

        result = self.analyzer.analyze(config)

        assert result.success
        assert len(result.sample_data) == 3

        # Check that Unicode characters are preserved
        for item in result.sample_data:
            assert len(item["name"]) > 0
            assert len(item["city"]) > 0

    def test_date_detection(self):
        """Test date column detection and handling"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("id,created_date,modified_time\n")
            f.write("1,2023-01-15,2023-01-15 10:30:00\n")
            f.write("2,2023-02-20,2023-02-20 14:45:30\n")
            f.write("3,2023-03-10,2023-03-10 09:15:45\n")
            temp_path = f.name

        try:
            config = {"name": "date_test", "path": temp_path}
            result = self.analyzer.analyze(config)

            assert result.success

            # Check if dates are detected (pandas might parse them as datetime)
            columns_by_name = {col.name: col for col in result.schema_info.columns}
            # Note: Date detection depends on pandas parsing, so we check for either date or datetime types
            date_types = ["date", "datetime", "date_string"]
            assert any(columns_by_name["created_date"].type in date_types for _ in [1])

        finally:
            os.unlink(temp_path)

    def test_llm_recommendations_generation(self):
        """Test LLM recommendation generation"""
        config = {
            "name": "recommendation_test",
            "data": [
                {
                    "customer_id": f"CUST_{i:03d}",
                    "category": "A" if i % 2 == 0 else "B",
                    "revenue": i * 100.5,
                    "count": i,
                }
                for i in range(1, 21)
            ],
        }

        result = self.analyzer.analyze(config)

        assert result.success
        assert len(result.llm_recommendations) > 0

        recommendations_text = " ".join(result.llm_recommendations).lower()

        # Should recommend key columns
        assert "key columns" in recommendations_text or "customer_id" in recommendations_text

        # Should recommend numeric operations
        assert "numeric" in recommendations_text or "statistical" in recommendations_text

        # Should recommend categorical operations
        assert "categorical" in recommendations_text or "grouping" in recommendations_text

    def test_memory_estimation(self):
        """Test memory usage estimation"""
        config = {
            "name": "memory_test",
            "data": [{"text_field": "A" * 1000, "number": i} for i in range(100)],  # Large text fields
        }

        result = self.analyzer.analyze(config)

        assert result.success
        assert result.size_info.memory_size_mb > 0
        assert result.size_info.estimated_tokens > 0

        # For this dataset, should detect significant memory usage
        assert result.size_info.memory_size_mb > 0.01  # At least 10KB

    def test_statistical_summary_generation(self):
        """Test comprehensive statistical summary generation"""
        config = {
            "name": "stats_summary_test",
            "data": [
                {
                    "numeric_col": i,
                    "text_col": f"item_{i}",
                    "category": "A" if i % 2 == 0 else "B",
                }
                for i in range(1, 101)
            ],
        }

        result = self.analyzer.analyze(config)

        assert result.success
        assert result.schema_info.statistical_summary is not None

        summary = result.schema_info.statistical_summary
        assert "total_rows" in summary
        assert "total_columns" in summary
        assert "memory_usage_mb" in summary
        assert summary["total_rows"] == 100
        assert summary["total_columns"] == 3

        # Should have numeric summary
        if "numeric_summary" in summary:
            assert "numeric_col" in summary["numeric_summary"]

        # Should have categorical summary
        if "categorical_summary" in summary:
            assert len(summary["categorical_summary"]) > 0
