"""
Comprehensive tests for the MarkdownGenerator.

Tests the LLM-optimized markdown generation including:
- Syntax reference section
- Actionable relationship YAML
- Field access patterns per data source
"""

from unittest.mock import MagicMock, Mock

import pytest

from shedboxai.core.introspection.markdown_generator import MarkdownGenerator
from shedboxai.core.introspection.models import (
    AnalysisStatus,
    ColumnInfo,
    Relationship,
    SchemaInfo,
    SourceAnalysis,
    SourceType,
)


class TestMarkdownGenerator:
    """Tests for basic MarkdownGenerator functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.generator = MarkdownGenerator()

    def test_generator_initialization(self):
        """Test that generator initializes correctly."""
        assert self.generator is not None

    def test_generate_returns_string(self):
        """Test that generate returns a non-empty string."""
        result = self.generator.generate(
            analyses={},
            relationships=[],
            success_count=0,
            total_count=0,
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_includes_header(self):
        """Test that generated markdown includes header."""
        result = self.generator.generate(
            analyses={},
            relationships=[],
            success_count=0,
            total_count=0,
        )
        assert "# Data Source Introspection for LLM" in result


class TestSyntaxReferenceSection:
    """Tests for the ShedBoxAI YAML syntax reference section."""

    def setup_method(self):
        """Set up test fixtures."""
        self.generator = MarkdownGenerator()

    def test_syntax_reference_section_included(self):
        """Test that syntax reference section is included in output."""
        result = self.generator.generate(
            analyses={},
            relationships=[],
            success_count=0,
            total_count=0,
        )
        assert "## ShedBoxAI YAML Syntax Reference" in result

    def test_field_access_patterns_documented(self):
        """Test that field access patterns are documented."""
        result = self.generator.generate(
            analyses={},
            relationships=[],
            success_count=0,
            total_count=0,
        )
        # Check for the critical joined field pattern
        assert "item.{target}_info.{field}" in result
        assert "item.products_info.unit_price" in result

    def test_aggregate_functions_documented(self):
        """Test that aggregate functions are documented."""
        result = self.generator.generate(
            analyses={},
            relationships=[],
            success_count=0,
            total_count=0,
        )
        assert "SUM(field)" in result
        assert "COUNT(*)" in result
        assert "AVG(field)" in result
        assert "MIN(field)" in result
        assert "MAX(field)" in result

    def test_expression_syntax_documented(self):
        """Test that expression syntax rules are documented."""
        result = self.generator.generate(
            analyses={},
            relationships=[],
            success_count=0,
            total_count=0,
        )
        # Supported operations
        assert "Arithmetic:" in result or "arithmetic" in result.lower()
        # Unsupported operations
        assert ".get()" in result
        assert "NOT Supported" in result

    def test_common_mistakes_section(self):
        """Test that common mistakes section is included."""
        result = self.generator.generate(
            analyses={},
            relationships=[],
            success_count=0,
            total_count=0,
        )
        assert "WRONG" in result
        assert "CORRECT" in result


class TestActionableRelationships:
    """Tests for actionable relationship YAML generation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.generator = MarkdownGenerator()

        # Create mock analyses
        self.mock_analyses = self._create_mock_analyses()
        self.mock_relationships = self._create_mock_relationships()

    def _create_mock_analyses(self):
        """Create mock source analyses."""
        # Sales analysis
        sales_schema = SchemaInfo(
            columns=[
                ColumnInfo(name="id", type="integer", null_percentage=0, unique_count=100, total_count=100),
                ColumnInfo(name="product_id", type="integer", null_percentage=0, unique_count=50, total_count=100),
                ColumnInfo(name="customer_id", type="integer", null_percentage=0, unique_count=30, total_count=100),
                ColumnInfo(name="quantity", type="integer", null_percentage=0, unique_count=10, total_count=100),
                ColumnInfo(name="amount", type="float", null_percentage=0, unique_count=80, total_count=100),
            ]
        )
        sales = SourceAnalysis(
            name="sales",
            type=SourceType.CSV,
            status=AnalysisStatus.SUCCESS,
            schema_info=sales_schema,
        )
        # Add columns property for backward compatibility
        sales.columns = ["id", "product_id", "customer_id", "quantity", "amount"]

        # Products analysis
        products_schema = SchemaInfo(
            columns=[
                ColumnInfo(name="id", type="integer", null_percentage=0, unique_count=50, total_count=50),
                ColumnInfo(name="name", type="string", null_percentage=0, unique_count=50, total_count=50),
                ColumnInfo(name="unit_price", type="float", null_percentage=0, unique_count=40, total_count=50),
                ColumnInfo(name="cost_price", type="float", null_percentage=0, unique_count=35, total_count=50),
            ]
        )
        products = SourceAnalysis(
            name="products",
            type=SourceType.CSV,
            status=AnalysisStatus.SUCCESS,
            schema_info=products_schema,
        )
        products.columns = ["id", "name", "unit_price", "cost_price"]

        # Customers analysis
        customers_schema = SchemaInfo(
            columns=[
                ColumnInfo(name="customer_id", type="integer", null_percentage=0, unique_count=30, total_count=30),
                ColumnInfo(name="name", type="string", null_percentage=0, unique_count=30, total_count=30),
                ColumnInfo(name="membership_level", type="string", null_percentage=0, unique_count=3, total_count=30),
            ]
        )
        customers = SourceAnalysis(
            name="customers",
            type=SourceType.CSV,
            status=AnalysisStatus.SUCCESS,
            schema_info=customers_schema,
        )
        customers.columns = ["customer_id", "name", "membership_level"]

        return {
            "sales": sales,
            "products": products,
            "customers": customers,
        }

    def _create_mock_relationships(self):
        """Create mock relationships."""
        return [
            Relationship(
                source_a="sales",
                source_b="products",
                type="foreign_key",
                confidence=0.95,
                field_a="product_id",
                field_b="id",
                description="Primary-foreign key relationship with 95% confidence",
            ),
            Relationship(
                source_a="sales",
                source_b="customers",
                type="foreign_key",
                confidence=0.90,
                field_a="customer_id",
                field_b="customer_id",
                description="Exact field name match with 90% confidence",
            ),
        ]

    def test_relationships_section_included(self):
        """Test that relationships section is included."""
        result = self.generator.generate(
            analyses=self.mock_analyses,
            relationships=self.mock_relationships,
            success_count=3,
            total_count=3,
        )
        assert "## Detected Relationships" in result

    def test_link_fields_yaml_generated(self):
        """Test that actionable link_fields YAML is generated."""
        result = self.generator.generate(
            analyses=self.mock_analyses,
            relationships=self.mock_relationships,
            success_count=3,
            total_count=3,
        )
        # Check for link_fields YAML structure
        assert "link_fields:" in result
        assert "source: sales" in result
        assert "source_field: product_id" in result
        assert "to: products" in result
        assert "target_field: id" in result

    def test_field_access_examples_generated(self):
        """Test that field access examples are shown for joined tables."""
        result = self.generator.generate(
            analyses=self.mock_analyses,
            relationships=self.mock_relationships,
            success_count=3,
            total_count=3,
        )
        # Should show how to access products fields after join
        assert "item.products_info." in result

    def test_confidence_percentage_shown(self):
        """Test that confidence percentage is shown."""
        result = self.generator.generate(
            analyses=self.mock_analyses,
            relationships=self.mock_relationships,
            success_count=3,
            total_count=3,
        )
        # Should show confidence as percentage
        assert "95%" in result or "90%" in result

    def test_no_relationships_message(self):
        """Test message when no relationships detected."""
        result = self.generator.generate(
            analyses=self.mock_analyses,
            relationships=[],
            success_count=3,
            total_count=3,
        )
        assert "No relationships detected" in result


class TestFieldAccessPatterns:
    """Tests for per-source field access patterns."""

    def setup_method(self):
        """Set up test fixtures."""
        self.generator = MarkdownGenerator()

    def _create_source_with_columns(self, name, columns):
        """Helper to create a source analysis with columns."""
        schema = SchemaInfo(
            columns=[
                ColumnInfo(name=col, type="string", null_percentage=0, unique_count=10, total_count=10)
                for col in columns
            ]
        )
        source = SourceAnalysis(
            name=name,
            type=SourceType.CSV,
            status=AnalysisStatus.SUCCESS,
            schema_info=schema,
        )
        source.columns = columns
        return source

    def test_direct_field_access_shown(self):
        """Test that direct field access patterns are shown."""
        analyses = {
            "users": self._create_source_with_columns("users", ["id", "name", "email", "age"]),
        }

        result = self.generator.generate(
            analyses=analyses,
            relationships=[],
            success_count=1,
            total_count=1,
        )

        # Should show direct field access examples
        assert "Field Access Patterns" in result
        assert "item.id" in result or "item.name" in result

    def test_joined_field_access_shown_with_relationships(self):
        """Test that joined field access is shown when relationships exist."""
        analyses = {
            "orders": self._create_source_with_columns("orders", ["id", "customer_id", "total"]),
            "customers": self._create_source_with_columns("customers", ["id", "name", "email"]),
        }

        relationships = [
            Relationship(
                source_a="orders",
                source_b="customers",
                type="foreign_key",
                confidence=0.9,
                field_a="customer_id",
                field_b="id",
                description="FK relationship",
            ),
        ]

        result = self.generator.generate(
            analyses=analyses,
            relationships=relationships,
            success_count=2,
            total_count=2,
        )

        # Should show joined field access pattern
        assert "item.customers_info" in result


class TestBackwardCompatibility:
    """Tests to ensure backward compatibility."""

    def setup_method(self):
        """Set up test fixtures."""
        self.generator = MarkdownGenerator()

    def test_generate_without_options(self):
        """Test that generate works without options parameter."""
        result = self.generator.generate(
            analyses={},
            relationships=[],
            success_count=0,
            total_count=0,
        )
        assert isinstance(result, str)

    def test_generate_with_empty_relationships(self):
        """Test that generate handles empty relationships list."""
        result = self.generator.generate(
            analyses={},
            relationships=[],
            success_count=0,
            total_count=0,
        )
        assert "## Detected Relationships" in result

    def test_generate_with_none_options(self):
        """Test that generate handles None options."""
        result = self.generator.generate(
            analyses={},
            relationships=[],
            success_count=0,
            total_count=0,
            options=None,
        )
        assert isinstance(result, str)

    def test_existing_sections_preserved(self):
        """Test that existing sections are still present."""
        result = self.generator.generate(
            analyses={},
            relationships=[],
            success_count=0,
            total_count=0,
        )
        # Verify existing sections are still there
        assert "## LLM Processing Notes" in result
        assert "## Data Sources" in result
        assert "## Detected Relationships" in result
        assert "## Recommended ShedBoxAI Operations" in result
        assert "## LLM Context Optimization Tips" in result


class TestPIISanitization:
    """Tests for PII sanitization in sample data."""

    def setup_method(self):
        """Set up test fixtures."""
        self.generator = MarkdownGenerator()

    def test_date_values_not_sanitized_as_phone(self):
        """Test that ISO date values are NOT replaced with phone numbers."""
        # This is a regression test for the bug where dates like "2024-01-15"
        # were incorrectly sanitized to "(555) 123-4567"
        values = ["2024-01-15", "2024-02-20", "2024-03-10"]
        sanitized = self.generator._sanitize_sample_values(values)

        # Dates should remain as dates, not become phone numbers
        assert sanitized[0] == "2024-01-15"
        assert sanitized[1] == "2024-02-20"
        assert sanitized[2] == "2024-03-10"
        assert "(555)" not in str(sanitized)

    def test_phone_values_are_sanitized(self):
        """Test that actual phone numbers are sanitized."""
        values = ["(555) 123-4567", "(555) 234-5678"]
        sanitized = self.generator._sanitize_sample_values(values)

        assert sanitized[0] == "(555) 123-4567"
        assert sanitized[1] == "(555) 123-4567"

    def test_email_values_are_sanitized(self):
        """Test that email addresses are sanitized."""
        values = ["alice@example.com", "bob@company.org"]
        sanitized = self.generator._sanitize_sample_values(values)

        assert sanitized[0] == "user@example.com"
        assert sanitized[1] == "user@example.com"

    def test_date_record_not_sanitized_as_phone(self):
        """Test that date fields in records are NOT replaced with phone numbers."""
        record = {
            "id": 1,
            "date": "2024-01-15",
            "signup_date": "2023-06-20",
            "amount": 100.50
        }
        sanitized = self.generator._sanitize_record(record)

        # Date values should remain unchanged
        assert sanitized["date"] == "2024-01-15"
        assert sanitized["signup_date"] == "2023-06-20"
        assert "(555)" not in str(sanitized.values())

    def test_date_field_name_detection(self):
        """Test that fields with 'date' in name are recognized as date fields."""
        assert self.generator._is_date_field("date") is True
        assert self.generator._is_date_field("signup_date") is True
        assert self.generator._is_date_field("created_at") is True
        assert self.generator._is_date_field("updated_at") is True
        assert self.generator._is_date_field("timestamp") is True
        assert self.generator._is_date_field("name") is False
        assert self.generator._is_date_field("phone") is False

    def test_date_value_detection(self):
        """Test that date values are correctly identified."""
        # ISO format
        assert self.generator._is_date_value("2024-01-15") is True
        assert self.generator._is_date_value("2024-01-15T10:30:00") is True
        # US format
        assert self.generator._is_date_value("01/15/2024") is True
        assert self.generator._is_date_value("1/5/24") is True
        # Not dates
        assert self.generator._is_date_value("(555) 123-4567") is False
        assert self.generator._is_date_value("hello world") is False
        assert self.generator._is_date_value("12345") is False

    def test_phone_value_detection(self):
        """Test that phone values are correctly identified."""
        # Valid phone formats
        assert self.generator._is_phone_value("(555) 123-4567") is True
        assert self.generator._is_phone_value("555-123-4567") is True
        assert self.generator._is_phone_value("555.123.4567") is True
        assert self.generator._is_phone_value("+1-555-123-4567") is True
        # Not phones (including dates that were previously misclassified)
        assert self.generator._is_phone_value("2024-01-15") is False
        assert self.generator._is_phone_value("hello world") is False
        assert self.generator._is_phone_value("user@example.com") is False

    def test_mixed_data_sanitization(self):
        """Test sanitization of records with mixed data types."""
        record = {
            "customer_id": "CUST_001",
            "name": "John Smith",
            "email": "john@example.com",
            "phone": "(555) 123-4567",
            "signup_date": "2024-01-15",
            "amount": 150.00,
            "category": "Premium"
        }
        sanitized = self.generator._sanitize_record(record)

        # IDs and categories should remain unchanged
        assert sanitized["customer_id"] == "CUST_001"
        assert sanitized["category"] == "Premium"
        # PII should be sanitized
        assert sanitized["name"] == "John Doe"
        assert sanitized["email"] == "user@example.com"
        assert sanitized["phone"] == "(555) 123-4567"
        # Dates should remain as dates
        assert sanitized["signup_date"] == "2024-01-15"
        # Numbers should remain unchanged
        assert sanitized["amount"] == 150.00

    def test_name_field_exact_matching(self):
        """Test that name sanitization uses exact field matching."""
        record = {
            "name": "John Smith",
            "first_name": "John",
            "last_name": "Smith",
            "username": "jsmith123",  # Should NOT be sanitized
            "filename": "report.pdf",  # Should NOT be sanitized
        }
        sanitized = self.generator._sanitize_record(record)

        assert sanitized["name"] == "John Doe"
        assert sanitized["first_name"] == "John"
        assert sanitized["last_name"] == "Doe"
        # These should NOT be sanitized as names
        assert sanitized["username"] == "jsmith123"
        assert sanitized["filename"] == "report.pdf"
