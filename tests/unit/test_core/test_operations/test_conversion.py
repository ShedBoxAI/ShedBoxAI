"""
Comprehensive unit tests for format conversion operations.

Tests cover field extraction, template processing, nested field handling,
and error conditions to uncover production bugs.

EXPECTED BUGS TO FIND:
1. Missing 'import re' at module level but uses re.sub()
2. Template field name extraction logic is fragile and inconsistent
3. _extract_fields doesn't handle non-dict items in lists properly
4. Template processing with missing nested paths fails ungracefully
"""

from unittest.mock import Mock

import pytest

from shedboxai.core.config.models import FormatConversionConfig
from shedboxai.core.operations.conversion import FormatConversionHandler


class TestFormatConversionHandler:
    """Test suite for FormatConversionHandler operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = FormatConversionHandler()
        self.mock_engine = Mock()
        self.handler_with_engine = FormatConversionHandler(engine=self.mock_engine)

    # Basic Configuration Tests
    def test_operation_name(self):
        """Test operation name property."""
        assert self.handler.operation_name == "format_conversion"

    def test_empty_config_returns_data_unchanged(self):
        """Test that empty config returns data unchanged."""
        data = {"source": [{"id": 1, "name": "test"}]}
        result = self.handler.process(data, {})
        assert result == data

    def test_none_config_returns_data_unchanged(self):
        """Test that None config returns data unchanged."""
        data = {"source": [{"id": 1, "name": "test"}]}
        result = self.handler.process(data, None)
        assert result == data

    # Field Extraction Tests
    def test_extract_fields_from_list_data(self):
        """Test extracting specific fields from list data."""
        data = {
            "users": [
                {"id": 1, "name": "Alice", "email": "alice@example.com", "age": 30},
                {"id": 2, "name": "Bob", "email": "bob@example.com", "age": 25},
            ]
        }
        config = {"users": FormatConversionConfig(extract_fields=["id", "name"])}

        result = self.handler.process(data, config)
        expected = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
        assert result["users"] == expected

    def test_extract_fields_from_dict_data(self):
        """Test extracting fields from dictionary data."""
        data = {
            "profile": {
                "id": 1,
                "name": "Alice",
                "email": "alice@example.com",
                "age": 30,
            }
        }
        config = {"profile": FormatConversionConfig(extract_fields=["id", "name"])}

        result = self.handler.process(data, config)
        expected = {"id": 1, "name": "Alice"}
        assert result["profile"] == expected

    def test_extract_missing_fields(self):
        """Test extracting fields that don't exist."""
        data = {"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]}
        config = {"users": FormatConversionConfig(extract_fields=["id", "missing_field"])}

        result = self.handler.process(data, config)
        expected = [{"id": 1, "missing_field": None}, {"id": 2, "missing_field": None}]
        assert result["users"] == expected

    def test_extract_template_fields_from_list(self):
        """Test extracting template fields from list data."""
        data = {
            "users": [
                {
                    "profile": {"name": "Alice"},
                    "contact": {"email": "alice@example.com"},
                },
                {"profile": {"name": "Bob"}, "contact": {"email": "bob@example.com"}},
            ]
        }
        config = {"users": FormatConversionConfig(extract_fields=["{{item.profile.name}}", "{{item.contact.email}}"])}

        result = self.handler.process(data, config)
        expected = [
            {"name": "Alice", "email": "alice@example.com"},
            {"name": "Bob", "email": "bob@example.com"},
        ]
        assert result["users"] == expected

    def test_extract_template_fields_from_dict(self):
        """Test extracting template fields from dict data."""
        data = {
            "user": {
                "profile": {"name": "Alice"},
                "contact": {"email": "alice@example.com"},
            }
        }
        config = {"user": FormatConversionConfig(extract_fields=["{{item.profile.name}}", "{{item.contact.email}}"])}

        result = self.handler.process(data, config)
        expected = {"name": "Alice", "email": "alice@example.com"}
        assert result["user"] == expected

    # Template Processing Tests
    def test_template_processing_basic(self):
        """Test basic template processing."""
        data = {"users": [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]}
        config = {"users": FormatConversionConfig(template="Name: {{item.name}}, Age: {{item.age}}")}

        result = self.handler.process(data, config)
        expected = ["Name: Alice, Age: 30", "Name: Bob, Age: 25"]
        assert result["users"] == expected

    def test_template_processing_with_engine(self):
        """Test template processing with expression engine."""
        self.mock_engine.substitute_variables.side_effect = [
            "Name: Alice, Age: 30",
            "Name: Bob, Age: 25",
        ]

        data = {"users": [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]}
        config = {"users": FormatConversionConfig(template="Name: {{item.name}}, Age: {{item.age}}")}

        result = self.handler_with_engine.process(data, config)
        expected = ["Name: Alice, Age: 30", "Name: Bob, Age: 25"]
        assert result["users"] == expected
        assert self.mock_engine.substitute_variables.call_count == 2

    def test_template_processing_non_list_data_raises_error(self):
        """Test that template processing on non-list data raises ValueError."""
        data = {"user": {"name": "Alice", "age": 30}}
        config = {"user": FormatConversionConfig(template="Name: {{item.name}}")}

        with pytest.raises(ValueError, match="Template processing only supports list data types"):
            self.handler.process(data, config)

    def test_simple_template_substitution_nested_access(self):
        """Test simple template substitution with nested field access."""
        template = "Hello {{item.profile.name}}, you are {{item.age}} years old"
        context = {"item": {"profile": {"name": "Alice"}, "age": 30}}

        result = self.handler._simple_template_substitution(template, context)
        assert result == "Hello Alice, you are 30 years old"

    def test_simple_template_substitution_missing_field(self):
        """Test simple template substitution with missing field."""
        template = "Hello {{item.missing.field}}"
        context = {"item": {"name": "Alice"}}

        result = self.handler._simple_template_substitution(template, context)
        assert "ERROR: item.missing.field not found" in result

    def test_simple_template_substitution_multiple_placeholders(self):
        """Test simple template substitution with multiple placeholders."""
        template = "{{item.name}} is {{item.age}} years old and lives in {{item.city}}"
        context = {"item": {"name": "Alice", "age": 30, "city": "New York"}}

        result = self.handler._simple_template_substitution(template, context)
        assert result == "Alice is 30 years old and lives in New York"

    # Configuration Validation Tests
    def test_invalid_config_both_extract_and_template(self):
        """Test that having both extract_fields and template raises ValueError."""
        data = {"users": [{"id": 1, "name": "Alice"}]}
        config = {"users": FormatConversionConfig(extract_fields=["id", "name"], template="{{item.name}}")}

        with pytest.raises(
            ValueError,
            match="Cannot specify both 'extract_fields' and 'template' options",
        ):
            self.handler.process(data, config)

    def test_dict_config_conversion(self):
        """Test that dict configs are converted to FormatConversionConfig objects."""
        data = {"users": [{"id": 1, "name": "Alice"}]}
        config = {"users": {"extract_fields": ["id", "name"]}}

        result = self.handler.process(data, config)
        expected = [{"id": 1, "name": "Alice"}]
        assert result["users"] == expected

    def test_invalid_dict_config_logs_warning(self, caplog):
        """Test that invalid dict config logs warning - NOW FIXED."""
        data = {"users": [{"id": 1, "name": "Alice"}]}
        config = {"users": {"invalid_field": "value"}}

        result = self.handler.process(data, config)
        assert "Invalid format conversion configuration" in caplog.text
        assert "Unknown fields" in caplog.text
        assert "invalid_field" in caplog.text
        assert result == data  # Data unchanged due to invalid config

    def test_invalid_config_type_logs_warning(self, caplog):
        """Test that invalid config type logs warning."""
        data = {"users": [{"id": 1, "name": "Alice"}]}
        config = {"users": "invalid_string_config"}

        result = self.handler.process(data, config)
        assert "Invalid format conversion configuration" in caplog.text
        assert "expected dict or FormatConversionConfig" in caplog.text
        assert result == data

    # Edge Cases and Error Handling
    def test_missing_source_data(self):
        """Test processing when source data doesn't exist."""
        data = {"other_data": [{"id": 1}]}
        config = {"missing_source": FormatConversionConfig(extract_fields=["id"])}

        result = self.handler.process(data, config)
        assert result == data  # Data unchanged

    def test_extract_fields_from_non_dict_list_items(self):
        """Test extracting fields when list contains non-dict items - NOW FIXED."""
        data = {
            "mixed_data": [
                {"id": 1, "name": "Alice"},
                "string_item",
                42,
                {"id": 2, "name": "Bob"},
            ]
        }
        config = {"mixed_data": FormatConversionConfig(extract_fields=["id", "name"])}

        result = self.handler.process(data, config)
        expected = [
            {"id": 1, "name": "Alice"},
            {"id": None, "name": None},  # string_item - now handled gracefully
            {"id": None, "name": None},  # 42 - now handled gracefully
            {"id": 2, "name": "Bob"},
        ]
        assert result["mixed_data"] == expected

    def test_template_processing_with_non_dict_items(self):
        """Test template processing when list contains non-dict items."""
        data = {"mixed_data": [{"name": "Alice"}, "string_item", {"name": "Bob"}]}
        config = {"mixed_data": FormatConversionConfig(template="Hello {{item.name}}")}

        result = self.handler.process(data, config)
        # Should handle non-dict items gracefully
        assert len(result["mixed_data"]) == 3

    def test_extract_fields_empty_list(self):
        """Test extracting fields from empty list."""
        data = {"empty": []}
        config = {"empty": FormatConversionConfig(extract_fields=["id", "name"])}

        result = self.handler.process(data, config)
        assert result["empty"] == []

    def test_extract_fields_with_none_values(self):
        """Test extracting fields with None values in data."""
        data = {
            "users": [
                {"id": 1, "name": None, "email": "alice@example.com"},
                {"id": None, "name": "Bob", "email": None},
            ]
        }
        config = {"users": FormatConversionConfig(extract_fields=["id", "name", "email"])}

        result = self.handler.process(data, config)
        expected = [
            {"id": 1, "name": None, "email": "alice@example.com"},
            {"id": None, "name": "Bob", "email": None},
        ]
        assert result["users"] == expected

    def test_template_field_extraction_with_complex_nesting(self):
        """Test template field extraction with deeply nested data."""
        data = {
            "complex": [
                {
                    "user": {
                        "profile": {"personal": {"name": "Alice"}},
                        "settings": {"preferences": {"theme": "dark"}},
                    }
                }
            ]
        }
        config = {
            "complex": FormatConversionConfig(
                extract_fields=[
                    "{{item.user.profile.personal.name}}",
                    "{{item.user.settings.preferences.theme}}",
                ]
            )
        }

        result = self.handler.process(data, config)
        expected = [{"name": "Alice", "theme": "dark"}]
        assert result["complex"] == expected

    def test_template_field_with_missing_nested_path(self):
        """Test template field extraction when nested path doesn't exist."""
        data = {
            "users": [
                {"profile": {"name": "Alice"}},  # Missing contact.email
                {"contact": {"email": "bob@example.com"}},  # Missing profile.name
            ]
        }
        config = {"users": FormatConversionConfig(extract_fields=["{{item.profile.name}}", "{{item.contact.email}}"])}

        result = self.handler.process(data, config)
        # Should handle missing paths gracefully
        assert len(result["users"]) == 2
        assert "name" in result["users"][0]
        assert "email" in result["users"][0]

    # Multiple Source Processing Tests
    def test_multiple_sources_processing(self):
        """Test processing multiple sources in single operation."""
        data = {
            "users": [{"id": 1, "name": "Alice", "email": "alice@example.com"}],
            "products": [{"id": 1, "title": "Widget", "price": 10.99}],
        }
        config = {
            "users": FormatConversionConfig(extract_fields=["id", "name"]),
            "products": FormatConversionConfig(extract_fields=["id", "title"]),
        }

        result = self.handler.process(data, config)
        assert result["users"] == [{"id": 1, "name": "Alice"}]
        assert result["products"] == [{"id": 1, "title": "Widget"}]

    def test_mixed_extract_and_template_operations(self):
        """Test mixing field extraction and template operations on different sources."""
        data = {
            "users": [{"id": 1, "name": "Alice", "age": 30}],
            "messages": [{"user": "Alice", "text": "Hello world"}],
        }
        config = {
            "users": FormatConversionConfig(extract_fields=["id", "name"]),
            "messages": FormatConversionConfig(template="{{item.user}}: {{item.text}}"),
        }

        result = self.handler.process(data, config)
        assert result["users"] == [{"id": 1, "name": "Alice"}]
        assert result["messages"] == ["Alice: Hello world"]

    # Data Type Preservation Tests
    def test_extract_fields_preserves_data_types(self):
        """Test that field extraction preserves original data types."""
        data = {
            "records": [
                {
                    "id": 1,
                    "price": 10.99,
                    "active": True,
                    "tags": ["tag1", "tag2"],
                    "metadata": {"key": "value"},
                }
            ]
        }
        config = {"records": FormatConversionConfig(extract_fields=["id", "price", "active", "tags", "metadata"])}

        result = self.handler.process(data, config)
        extracted = result["records"][0]

        assert isinstance(extracted["id"], int)
        assert isinstance(extracted["price"], float)
        assert isinstance(extracted["active"], bool)
        assert isinstance(extracted["tags"], list)
        assert isinstance(extracted["metadata"], dict)

    def test_original_data_not_modified(self):
        """Test that original data is not modified during processing."""
        original_data = {"users": [{"id": 1, "name": "Alice", "email": "alice@example.com"}]}
        data = {"users": [{"id": 1, "name": "Alice", "email": "alice@example.com"}]}
        config = {"users": FormatConversionConfig(extract_fields=["id", "name"])}

        result = self.handler.process(data, config)

        # Original data should not be modified
        assert data == original_data
        # But result should be different
        assert result["users"] != data["users"]
        assert len(result["users"][0]) == 2  # Only id and name
        assert len(data["users"][0]) == 3  # id, name, and email

    def test_dataframe_source_extract_fields(self):
        """Test that pandas DataFrame source data is converted for field extraction.

        This tests the fix for DataFrame handling where format_conversion
        would fail on DataFrame sources loaded from CSV files.
        """
        import pandas as pd

        # Create DataFrame similar to what CSV loading produces
        df = pd.DataFrame(
            [
                {"id": 1, "name": "Alice", "email": "alice@example.com", "age": 30},
                {"id": 2, "name": "Bob", "email": "bob@example.com", "age": 25},
                {"id": 3, "name": "Carol", "email": "carol@example.com", "age": 35},
            ]
        )
        data = {"users": df}

        config = {"users": FormatConversionConfig(extract_fields=["id", "name", "age"])}

        result = self.handler.process(data, config)

        # Verify DataFrame was converted and fields were extracted
        assert isinstance(result["users"], list)
        assert len(result["users"]) == 3

        # Verify only specified fields are present
        for user in result["users"]:
            assert set(user.keys()) == {"id", "name", "age"}
            assert "email" not in user

        # Verify data integrity
        assert result["users"][0]["name"] == "Alice"
        assert result["users"][1]["name"] == "Bob"
        assert result["users"][2]["name"] == "Carol"

    def test_dataframe_source_template_processing(self):
        """Test that pandas DataFrame source data works with template processing."""
        import pandas as pd

        df = pd.DataFrame(
            [
                {"name": "Alice", "role": "Engineer"},
                {"name": "Bob", "role": "Manager"},
            ]
        )
        data = {"employees": df}

        config = {"employees": FormatConversionConfig(template="{{item.name}} is a {{item.role}}")}

        result = self.handler.process(data, config)

        # Verify DataFrame was converted and templates were processed
        assert isinstance(result["employees"], list)
        assert len(result["employees"]) == 2
        assert result["employees"][0] == "Alice is a Engineer"
        assert result["employees"][1] == "Bob is a Manager"
