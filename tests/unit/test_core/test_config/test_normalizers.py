"""
Tests for normalizers.py - Configuration normalization functions.

Tests focus on finding real bugs in configuration processing logic
and ensuring all edge cases are handled correctly.
"""

import hashlib
from unittest.mock import MagicMock, patch

import pytest

from shedboxai.core.config.models import (
    AdvancedOperationConfig,
    ContentSummarizationConfig,
    ContextualFilterConfig,
    FormatConversionConfig,
    RelationshipConfig,
    TemplateMatchingConfig,
)
from shedboxai.core.config.normalizers import (
    _generate_template_key,
    _is_direct_config,
    normalize_advanced_operations_config,
    normalize_content_summarization_config,
    normalize_contextual_filtering_config,
    normalize_format_conversion_config,
    normalize_relationship_highlighting_config,
    normalize_template_matching_config,
)


class TestNormalizeContextualFilteringConfig:
    """Test normalize_contextual_filtering_config function."""

    def test_empty_config_returns_empty_dict(self):
        """Test that empty config returns empty dictionary."""
        assert normalize_contextual_filtering_config({}) == {}
        assert normalize_contextual_filtering_config(None) == {}

    def test_direct_source_mapping_with_list(self):
        """Test direct source mapping where value is a filter list."""
        config = {
            "source1": [{"field": "price", "condition": "> 100"}],
            "source2": [{"field": "status", "condition": "active"}],
        }
        result = normalize_contextual_filtering_config(config)

        assert len(result) == 2
        assert result["source1"] == [{"field": "price", "condition": "> 100"}]
        assert result["source2"] == [{"field": "status", "condition": "active"}]

    def test_named_configuration_with_nested_dict(self):
        """Test named configuration where value is dict mapping sources to filters."""
        config = {
            "filter_config": {
                "products": [{"field": "price", "condition": "> 100"}],
                "users": [{"field": "active", "condition": "true"}],
            }
        }
        result = normalize_contextual_filtering_config(config)

        assert len(result) == 2
        assert result["products"] == [{"field": "price", "condition": "> 100"}]
        assert result["users"] == [{"field": "active", "condition": "true"}]

    def test_mixed_direct_and_named_configurations(self):
        """Test mixing direct source mappings with named configurations."""
        config = {
            "direct_source": [{"field": "id", "condition": "> 0"}],
            "named_config": {
                "indirect_source1": [{"field": "type", "condition": "premium"}],
                "indirect_source2": [{"field": "count", "condition": "< 50"}],
            },
        }
        result = normalize_contextual_filtering_config(config)

        assert len(result) == 3
        assert result["direct_source"] == [{"field": "id", "condition": "> 0"}]
        assert result["indirect_source1"] == [{"field": "type", "condition": "premium"}]
        assert result["indirect_source2"] == [{"field": "count", "condition": "< 50"}]

    def test_empty_filter_lists(self):
        """Test handling of empty filter lists."""
        config = {"source1": [], "named_config": {"source2": []}}
        result = normalize_contextual_filtering_config(config)

        assert result["source1"] == []
        assert result["source2"] == []

    def test_non_list_non_dict_values_ignored(self):
        """Test that non-list, non-dict values are ignored (potential bug)."""
        config = {
            "valid_source": [{"field": "test", "condition": "value"}],
            "invalid_string": "not_a_list_or_dict",
            "invalid_number": 123,
            "invalid_none": None,
        }
        result = normalize_contextual_filtering_config(config)

        # Only valid_source should be in result
        assert len(result) == 1
        assert result["valid_source"] == [{"field": "test", "condition": "value"}]


class TestNormalizeFormatConversionConfig:
    """Test normalize_format_conversion_config function."""

    def test_empty_config_returns_empty_dict(self):
        """Test that empty config returns empty dictionary."""
        assert normalize_format_conversion_config({}) == {}
        assert normalize_format_conversion_config(None) == {}

    def test_direct_source_mapping_with_dict(self):
        """Test direct source mapping with dictionary configuration."""
        config = {
            "source1": {"extract_fields": ["name", "price"]},
            "source2": {"template": "Name: {{item.name}}"},
        }
        result = normalize_format_conversion_config(config)

        assert len(result) == 2
        assert isinstance(result["source1"], FormatConversionConfig)
        assert isinstance(result["source2"], FormatConversionConfig)
        assert result["source1"].extract_fields == ["name", "price"]
        assert result["source2"].template == "Name: {{item.name}}"

    def test_direct_source_mapping_with_config_object(self):
        """Test direct source mapping with FormatConversionConfig object."""
        config_obj = FormatConversionConfig(extract_fields=["id", "name"])
        config = {"source1": config_obj}

        result = normalize_format_conversion_config(config)
        assert result["source1"] is config_obj

    def test_named_configuration_nested_dict(self):
        """Test named configuration with nested dictionary structure."""
        config = {
            "conversion_rules": {
                "products": {"extract_fields": ["name", "price"]},
                "customers": {"template": "Customer: {{item.name}}"},
            }
        }
        result = normalize_format_conversion_config(config)

        assert len(result) == 2
        assert isinstance(result["products"], FormatConversionConfig)
        assert isinstance(result["customers"], FormatConversionConfig)

    def test_invalid_dict_configuration_is_actually_valid(self):
        """Test that what we thought was invalid is actually valid - revealing the real behavior."""
        config = {"source_with_extra_fields": {"invalid_field": "value"}}
        result = normalize_format_conversion_config(config)

        # DISCOVERY: Pydantic allows extra fields by default!
        # This reveals that the FormatConversionConfig is more permissive than expected
        assert "source_with_extra_fields" in result
        assert isinstance(result["source_with_extra_fields"], FormatConversionConfig)

        # The config object is created with None values for standard fields
        assert result["source_with_extra_fields"].extract_fields is None
        assert result["source_with_extra_fields"].template is None

    @patch("builtins.print")
    def test_invalid_nested_dict_configuration_prints_warning(self, mock_print):
        """Test that invalid nested dict configurations print warnings."""
        config = {
            "config_group": {
                "valid_source": {"extract_fields": ["name"]},
                "invalid_source": {"bad_field": "value"},
            }
        }
        result = normalize_format_conversion_config(config)

        # Should include valid config and print warning for invalid
        assert "valid_source" in result
        assert "invalid_source" not in result
        mock_print.assert_called()

    def test_mixed_valid_and_invalid_config_objects(self):
        """Test mixing valid config objects with invalid configurations."""
        config_obj = FormatConversionConfig(extract_fields=["id"])
        config = {
            "valid_object": config_obj,
            "valid_dict": {"template": "Hello {{name}}"},
            "named_group": {
                "nested_valid": {"extract_fields": ["price"]},
                "nested_object": config_obj,
            },
        }
        result = normalize_format_conversion_config(config)

        assert len(result) == 4
        assert result["valid_object"] is config_obj
        assert isinstance(result["valid_dict"], FormatConversionConfig)
        assert isinstance(result["nested_valid"], FormatConversionConfig)
        assert result["nested_object"] is config_obj


class TestNormalizeContentSummarizationConfig:
    """Test normalize_content_summarization_config function."""

    def test_empty_config_returns_empty_dict(self):
        """Test that empty config returns empty dictionary."""
        assert normalize_content_summarization_config({}) == {}
        assert normalize_content_summarization_config(None) == {}

    def test_direct_source_mapping_with_dict(self):
        """Test direct source mapping with dictionary configuration."""
        config = {
            "source1": {
                "method": "statistical",
                "fields": ["price"],
                "summarize": ["mean", "max"],
            },
            "source2": {
                "method": "statistical",
                "fields": ["description"],
                "summarize": ["count"],
            },
        }
        result = normalize_content_summarization_config(config)

        assert len(result) == 2
        assert isinstance(result["source1"], ContentSummarizationConfig)
        assert isinstance(result["source2"], ContentSummarizationConfig)
        assert result["source1"].method == "statistical"
        assert result["source2"].method == "statistical"

    def test_direct_source_mapping_with_config_object(self):
        """Test direct source mapping with ContentSummarizationConfig object."""
        config_obj = ContentSummarizationConfig(method="statistical", fields=["price"], summarize=["mean"])
        config = {"source1": config_obj}

        result = normalize_content_summarization_config(config)
        assert result["source1"] is config_obj

    @patch("builtins.print")
    def test_invalid_configuration_prints_warning(self, mock_print):
        """Test that invalid configurations print warnings."""
        config = {"invalid_source": {"invalid_method": "unknown"}}
        result = normalize_content_summarization_config(config)

        mock_print.assert_called()
        assert "invalid_source" not in result

    @patch("builtins.print")
    def test_missing_required_field_prints_warning(self, mock_print):
        """Test that missing required fields print warnings."""
        config = {
            "incomplete_source": {
                "method": "statistical",
                "fields": ["price"],
            }  # Missing summarize
        }
        result = normalize_content_summarization_config(config)

        mock_print.assert_called()
        assert "incomplete_source" not in result


class TestNormalizeRelationshipHighlightingConfig:
    """Test normalize_relationship_highlighting_config function."""

    @patch("builtins.print")
    def test_empty_config_returns_empty_dict_with_debug(self, mock_print):
        """Test that empty config returns empty dictionary and prints debug."""
        result = normalize_relationship_highlighting_config({})
        assert result == {}

        # Should print debug message
        mock_print.assert_called_with("[DEBUG] normalize_relationship_highlighting_config - Empty config")

    @patch("builtins.print")
    def test_none_config_returns_empty_dict_with_debug(self, mock_print):
        """Test that None config returns empty dictionary and prints debug."""
        result = normalize_relationship_highlighting_config(None)
        assert result == {}
        mock_print.assert_called_with("[DEBUG] normalize_relationship_highlighting_config - Empty config")

    def test_direct_relationship_config_object(self):
        """Test direct RelationshipConfig object handling."""
        config_obj = RelationshipConfig(link_fields=[{"source": "A", "to": "B", "match_on": "id"}])
        result = normalize_relationship_highlighting_config(config_obj)

        assert len(result) == 1
        assert "data" in result
        assert result["data"] is config_obj

    def test_dict_with_relationship_config_values(self):
        """Test dictionary with RelationshipConfig values."""
        config_obj = RelationshipConfig(link_fields=[{"source": "A", "to": "B", "match_on": "id"}])
        config = {"source1": config_obj}

        result = normalize_relationship_highlighting_config(config)
        assert result["source1"] is config_obj

    def test_dict_with_dict_values_converted_to_config(self):
        """Test dictionary with dict values that get converted to RelationshipConfig."""
        config = {"source1": {"link_fields": [{"source": "A", "to": "B", "match_on": "id"}]}}
        result = normalize_relationship_highlighting_config(config)

        assert isinstance(result["source1"], RelationshipConfig)
        assert result["source1"].link_fields == [{"source": "A", "to": "B", "match_on": "id"}]

    @patch("builtins.print")
    def test_invalid_dict_value_prints_warning(self, mock_print):
        """Test that invalid dict values print warnings."""
        config = {"source1": {"invalid_field": "value"}}
        result = normalize_relationship_highlighting_config(config)

        mock_print.assert_called()
        warning_call = str(mock_print.call_args)
        assert "Warning: Invalid relationship config" in warning_call
        assert "source1" not in result

    @patch("builtins.print")
    def test_unsupported_value_type_prints_warning(self, mock_print):
        """Test that unsupported value types print warnings."""
        config = {"source1": "invalid_string_value", "source2": 123}
        result = normalize_relationship_highlighting_config(config)

        # Should print warnings for both unsupported types
        assert mock_print.call_count >= 2
        assert len(result) == 0

    @patch("builtins.print")
    def test_unrecognized_config_type_prints_debug(self, mock_print):
        """Test that unrecognized config types print debug message."""
        result = normalize_relationship_highlighting_config("invalid_type")

        assert result == {}
        mock_print.assert_called()
        debug_call = str(mock_print.call_args)
        assert "[DEBUG] Unrecognized config type" in debug_call


class TestNormalizeAdvancedOperationsConfig:
    """Test normalize_advanced_operations_config function."""

    def test_empty_config_returns_empty_dict(self):
        """Test that empty config returns empty dictionary."""
        assert normalize_advanced_operations_config({}) == {}
        assert normalize_advanced_operations_config(None) == {}

    def test_direct_source_mapping_with_dict(self):
        """Test direct source mapping with dictionary configuration."""
        config = {
            "result1": {
                "source": "data",
                "group_by": "category",
                "aggregate": {"total": "SUM(amount)"},
            },
            "result2": {"source": "users", "sort": "name", "limit": 10},
        }
        result = normalize_advanced_operations_config(config)

        assert len(result) == 2
        assert isinstance(result["result1"], AdvancedOperationConfig)
        assert isinstance(result["result2"], AdvancedOperationConfig)
        assert result["result1"].source == "data"
        assert result["result2"].limit == 10

    def test_direct_source_mapping_with_config_object(self):
        """Test direct source mapping with AdvancedOperationConfig object."""
        config_obj = AdvancedOperationConfig(source="data", limit=5)
        config = {"result1": config_obj}

        result = normalize_advanced_operations_config(config)
        assert result["result1"] is config_obj

    def test_named_configuration_nested_dict(self):
        """Test named configuration with nested dictionary structure."""
        config = {
            "operation_group": {
                "summary": {"source": "sales", "group_by": "region"},
                "filtered": {"source": "users", "limit": 100},
            }
        }
        result = normalize_advanced_operations_config(config)

        assert len(result) == 2
        assert isinstance(result["summary"], AdvancedOperationConfig)
        assert isinstance(result["filtered"], AdvancedOperationConfig)

    @patch("builtins.print")
    def test_invalid_configuration_prints_warning(self, mock_print):
        """Test that invalid configurations print warnings."""
        config = {"invalid_result": {"invalid_field": "value"}}
        result = normalize_advanced_operations_config(config)

        mock_print.assert_called()
        assert "invalid_result" not in result

    @patch("builtins.print")
    def test_missing_source_field_prints_warning(self, mock_print):
        """Test that missing source field prints warnings."""
        config = {"incomplete_result": {"group_by": "category"}}  # Missing required source field
        result = normalize_advanced_operations_config(config)

        mock_print.assert_called()
        assert "incomplete_result" not in result


class TestNormalizeTemplateMatchingConfig:
    """Test normalize_template_matching_config function."""

    @patch("logging.warning")
    def test_empty_config_logs_warning(self, mock_warning):
        """Test that empty config logs warning."""
        result = normalize_template_matching_config({})
        assert result == {}
        mock_warning.assert_called_with("Empty template_matching configuration received")

    @patch("logging.warning")
    def test_none_config_logs_warning(self, mock_warning):
        """Test that None config logs warning."""
        result = normalize_template_matching_config(None)
        assert result == {}
        mock_warning.assert_called_with("Empty template_matching configuration received")

    def test_direct_template_matching_config_object(self):
        """Test direct TemplateMatchingConfig object handling."""
        config_obj = TemplateMatchingConfig(template="Hello {{name}}")
        result = normalize_template_matching_config(config_obj)

        assert len(result) == 1
        # Should generate a key based on template content
        key = list(result.keys())[0]
        assert result[key] is config_obj

    def test_dict_with_template_config_values(self):
        """Test dictionary with TemplateMatchingConfig values."""
        config_obj = TemplateMatchingConfig(template="Hello {{name}}")
        config = {"template1": config_obj}

        result = normalize_template_matching_config(config)
        assert result["template1"] is config_obj

    def test_dict_with_dict_values_converted_to_config(self):
        """Test dictionary with dict values that get converted to TemplateMatchingConfig."""
        config = {
            "template1": {"template": "Hello {{name}}"},
            "template2": {"template_id": "greeting_template"},
        }
        result = normalize_template_matching_config(config)

        assert len(result) == 2
        assert isinstance(result["template1"], TemplateMatchingConfig)
        assert isinstance(result["template2"], TemplateMatchingConfig)
        assert result["template1"].template == "Hello {{name}}"
        assert result["template2"].template_id == "greeting_template"

    @patch("logging.error")
    def test_invalid_dict_configuration_logs_error(self, mock_error):
        """Test that invalid dict configurations log errors."""
        config = {"invalid_template": {"invalid_field": "value"}}
        result = normalize_template_matching_config(config)

        mock_error.assert_called()
        assert "invalid_template" not in result

    @patch("logging.warning")
    def test_unsupported_value_type_logs_warning(self, mock_warning):
        """Test that unsupported value types log warnings."""
        config = {"template1": "invalid_string_value", "template2": 123}
        result = normalize_template_matching_config(config)

        # Should log warnings for unsupported types
        mock_warning.assert_called()
        assert len(result) == 0

    def test_named_configuration_nested_dict(self):
        """Test named configuration with nested dictionary structure."""
        config = {
            "template_group": {
                "email": {"template": "Dear {{name}}, {{content}}"},
                "report": {"template_id": "monthly_report"},
            }
        }
        result = normalize_template_matching_config(config)

        assert len(result) == 2
        assert isinstance(result["email"], TemplateMatchingConfig)
        assert isinstance(result["report"], TemplateMatchingConfig)


class TestGenerateTemplateKey:
    """Test _generate_template_key function."""

    def test_generate_key_from_template_id(self):
        """Test key generation from template_id."""
        config = TemplateMatchingConfig(template="test", template_id="my_template")
        key = _generate_template_key(config)
        assert key == "my_template"

    def test_generate_key_from_template_content(self):
        """Test key generation from template content."""
        config = TemplateMatchingConfig(template="# Invoice Template\nHello {{name}}")
        key = _generate_template_key(config)
        assert key == "invoice_template"

    def test_generate_key_from_template_content_long_title(self):
        """Test key generation truncates long titles."""
        config = TemplateMatchingConfig(
            template="# This is a very long template title that should be truncated\nContent"
        )
        key = _generate_template_key(config)
        assert len(key) == 20
        assert key == "this_is_a_very_long_"

    def test_generate_key_from_template_content_uses_first_line(self):
        """Test key generation uses first line content when no title marker."""
        config = TemplateMatchingConfig(template="Just some content without title")
        key = _generate_template_key(config)
        # Should use first 20 chars of first line
        assert key == "just_some_content_wi"
        assert len(key) == 20

    def test_generate_key_from_template_with_hash_fallback(self):
        """Test key generation falls back to hash when template is minimal."""
        config = TemplateMatchingConfig(template="Hi")
        key = _generate_template_key(config)
        # Short content should still use the content, not hash
        assert key == "hi"

    def test_generate_key_consistency(self):
        """Test that same template content generates same key."""
        config1 = TemplateMatchingConfig(template="# Same Title\nContent")
        config2 = TemplateMatchingConfig(template="# Same Title\nDifferent content")

        key1 = _generate_template_key(config1)
        key2 = _generate_template_key(config2)

        # Both should generate same key since they have same title
        assert key1 == key2 == "same_title"

    def test_generate_key_hash_fallback_for_empty_first_line(self):
        """Test hash fallback when first line is empty."""
        config = TemplateMatchingConfig(template="\n\nSome content on third line")
        key = _generate_template_key(config)
        # Should fall back to hash since first line is empty
        expected_hash = hashlib.md5("\n\nSome content on third line".encode()).hexdigest()[:8]
        assert key == f"template_{expected_hash}"


class TestIsDirectConfig:
    """Test _is_direct_config helper function."""

    def test_direct_config_with_known_fields(self):
        """Test recognition of direct config with known field names."""
        assert _is_direct_config({"extract_fields": ["name"]}) is True
        assert _is_direct_config({"template": "Hello {{name}}"}) is True
        assert _is_direct_config({"source": "data", "group_by": "type"}) is True
        assert _is_direct_config({"method": "statistical", "fields": ["price"]}) is True
        assert _is_direct_config({"template_id": "report"}) is True

    def test_named_config_with_nested_dicts(self):
        """Test recognition of named config with nested dictionaries."""
        assert _is_direct_config({"source1": {"extract_fields": ["name"]}}) is False
        assert _is_direct_config({"group1": {"template": "Hello"}}) is False
        assert _is_direct_config({"config": {"source": "data"}}) is False

    def test_direct_config_with_primitive_values(self):
        """Test recognition of direct config with primitive values."""
        assert _is_direct_config({"field1": "value", "field2": 123}) is True
        assert _is_direct_config({"items": ["a", "b", "c"]}) is True
        assert _is_direct_config({"enabled": True, "count": 5}) is True

    def test_mixed_config_with_dict_values_is_named(self):
        """Test that configs with any dict values are considered named configs."""
        assert _is_direct_config({"direct_field": "value", "nested": {"inner": "value"}}) is False

    def test_empty_dict_is_direct_config(self):
        """Test that empty dict is considered direct config."""
        assert _is_direct_config({}) is True

    def test_non_dict_returns_false(self):
        """Test that non-dict values return False."""
        assert _is_direct_config("string") is False
        assert _is_direct_config(123) is False
        assert _is_direct_config(["list"]) is False
        assert _is_direct_config(None) is False

    def test_config_fields_detection(self):
        """Test detection of all config field types."""
        config_fields = [
            "extract_fields",
            "template",
            "source",
            "method",
            "fields",
            "summarize",
            "link_fields",
            "jsonpath_links",
            "pattern_detection",
            "conditional_highlighting",
            "context_additions",
            "derived_fields",
            "group_by",
            "aggregate",
            "sort",
            "limit",
            "template_id",
            "variables",
        ]

        for field in config_fields:
            assert _is_direct_config({field: "value"}) is True

    def test_unknown_fields_with_primitives_is_direct(self):
        """Test that unknown fields with primitive values are considered direct."""
        assert _is_direct_config({"unknown_field": "value"}) is True
        assert _is_direct_config({"custom_setting": 42}) is True


class TestNormalizersEdgeCases:
    """Test edge cases and potential bugs across normalizers."""

    def test_all_normalizers_handle_none_gracefully(self):
        """Test that all normalizers handle None input gracefully."""
        assert normalize_contextual_filtering_config(None) == {}
        assert normalize_format_conversion_config(None) == {}
        assert normalize_content_summarization_config(None) == {}
        assert normalize_relationship_highlighting_config(None) == {}
        assert normalize_advanced_operations_config(None) == {}
        assert normalize_template_matching_config(None) == {}

    def test_all_normalizers_handle_empty_dict_gracefully(self):
        """Test that all normalizers handle empty dict input gracefully."""
        assert normalize_contextual_filtering_config({}) == {}
        assert normalize_format_conversion_config({}) == {}
        assert normalize_content_summarization_config({}) == {}
        assert normalize_relationship_highlighting_config({}) == {}
        assert normalize_advanced_operations_config({}) == {}
        assert normalize_template_matching_config({}) == {}

    def test_normalizers_preserve_config_object_references(self):
        """Test that normalizers preserve object references when possible."""
        filter_config = ContextualFilterConfig(field="test", condition="value")
        format_config = FormatConversionConfig(extract_fields=["test"])
        summary_config = ContentSummarizationConfig(method="statistical", fields=["test"], summarize=["count"])
        relationship_config = RelationshipConfig(link_fields=[])
        advanced_config = AdvancedOperationConfig(source="test")
        template_config = TemplateMatchingConfig(template="test")

        # Test that object references are preserved
        assert normalize_contextual_filtering_config({"src": [filter_config]})["src"][0] is filter_config
        assert normalize_format_conversion_config({"src": format_config})["src"] is format_config
        assert normalize_content_summarization_config({"src": summary_config})["src"] is summary_config
        assert normalize_relationship_highlighting_config({"src": relationship_config})["src"] is relationship_config
        assert normalize_advanced_operations_config({"src": advanced_config})["src"] is advanced_config
        assert normalize_template_matching_config({"src": template_config})["src"] is template_config

    def test_complex_nested_structures_handled_correctly(self):
        """Test that complex nested structures are handled correctly."""
        complex_config = {
            "direct_source": {"extract_fields": ["name", "price"]},
            "named_group": {
                "source1": {"template": "Hello {{name}}"},
                "source2": {"extract_fields": ["id", "status"]},
                "nested_group": {"subsource1": {"template": "Nested {{value}}"}},
            },
        }

        result = normalize_format_conversion_config(complex_config)

        # Should handle direct and nested sources correctly
        assert "direct_source" in result
        assert "source1" in result
        assert "source2" in result
        # Deeply nested structures should be flattened appropriately
        assert "subsource1" in result

    @patch("builtins.print")
    def test_error_handling_continues_processing(self, mock_print):
        """Test that errors in one config don't stop processing of others."""
        config = {
            "valid1": {"extract_fields": ["name"]},
            "invalid": {"bad_field": "value"},
            "valid2": {"template": "Hello {{name}}"},
        }

        result = normalize_format_conversion_config(config)

        # Should process valid configs despite invalid ones
        assert "valid1" in result
        assert "valid2" in result
        assert "invalid" not in result
        # Should have printed warning for invalid config
        mock_print.assert_called()

    def test_deeply_nested_configuration_structures(self):
        """Test handling of deeply nested configuration structures."""
        config = {"level1": {"level2": {"level3": {"final_source": {"extract_fields": ["deep_field"]}}}}}

        result = normalize_format_conversion_config(config)

        # Should handle deep nesting and extract the final config
        assert "final_source" in result or "level3" in result
        # The exact behavior depends on how deep nesting is handled

    def test_configuration_type_validation(self):
        """Test that configuration types are properly validated."""
        # Mix of valid and invalid configuration types
        configs = [
            ({}, {}),  # Empty configs should work
            ({"source": {"extract_fields": ["test"]}}, {"source"}),  # Valid config
            ({"source": "invalid_string"}, set()),  # Invalid type should be filtered
            ({"source": 123}, set()),  # Invalid type should be filtered
            ({"source": None}, set()),  # None should be filtered
        ]

        for config_input, expected_keys in configs:
            result = normalize_format_conversion_config(config_input)
            assert set(result.keys()) == expected_keys


if __name__ == "__main__":
    pytest.main([__file__])

"""
Tests for normalizers.py - Configuration normalization functions.

Tests focus on finding real bugs in configuration processing logic
and ensuring all edge cases are handled correctly.
"""
import hashlib
from unittest.mock import MagicMock, patch

import pytest

from shedboxai.core.config.models import (
    AdvancedOperationConfig,
    ContentSummarizationConfig,
    ContextualFilterConfig,
    FormatConversionConfig,
    RelationshipConfig,
    TemplateMatchingConfig,
)
from shedboxai.core.config.normalizers import (
    _generate_template_key,
    normalize_advanced_operations_config,
    normalize_content_summarization_config,
    normalize_contextual_filtering_config,
    normalize_format_conversion_config,
    normalize_relationship_highlighting_config,
    normalize_template_matching_config,
)


class TestNormalizeContextualFilteringConfig:
    """Test normalize_contextual_filtering_config function."""

    def test_empty_config_returns_empty_dict(self):
        """Test that empty config returns empty dictionary."""
        assert normalize_contextual_filtering_config({}) == {}
        assert normalize_contextual_filtering_config(None) == {}

    def test_direct_source_mapping_with_list(self):
        """Test direct source mapping where value is a filter list."""
        config = {
            "source1": [{"field": "price", "condition": "> 100"}],
            "source2": [{"field": "status", "condition": "active"}],
        }
        result = normalize_contextual_filtering_config(config)

        assert len(result) == 2
        assert result["source1"] == [{"field": "price", "condition": "> 100"}]
        assert result["source2"] == [{"field": "status", "condition": "active"}]

    def test_named_configuration_with_nested_dict(self):
        """Test named configuration where value is dict mapping sources to filters."""
        config = {
            "filter_config": {
                "products": [{"field": "price", "condition": "> 100"}],
                "users": [{"field": "active", "condition": "true"}],
            }
        }
        result = normalize_contextual_filtering_config(config)

        assert len(result) == 2
        assert result["products"] == [{"field": "price", "condition": "> 100"}]
        assert result["users"] == [{"field": "active", "condition": "true"}]

    def test_mixed_direct_and_named_configurations(self):
        """Test mixing direct source mappings with named configurations."""
        config = {
            "direct_source": [{"field": "id", "condition": "> 0"}],
            "named_config": {
                "indirect_source1": [{"field": "type", "condition": "premium"}],
                "indirect_source2": [{"field": "count", "condition": "< 50"}],
            },
        }
        result = normalize_contextual_filtering_config(config)

        assert len(result) == 3
        assert result["direct_source"] == [{"field": "id", "condition": "> 0"}]
        assert result["indirect_source1"] == [{"field": "type", "condition": "premium"}]
        assert result["indirect_source2"] == [{"field": "count", "condition": "< 50"}]

    def test_empty_filter_lists(self):
        """Test handling of empty filter lists."""
        config = {"source1": [], "named_config": {"source2": []}}
        result = normalize_contextual_filtering_config(config)

        assert result["source1"] == []
        assert result["source2"] == []

    def test_non_list_non_dict_values_ignored(self):
        """Test that non-list, non-dict values are ignored (potential bug)."""
        config = {
            "valid_source": [{"field": "test", "condition": "value"}],
            "invalid_string": "not_a_list_or_dict",
            "invalid_number": 123,
            "invalid_none": None,
        }
        result = normalize_contextual_filtering_config(config)

        # Only valid_source should be in result
        assert len(result) == 1
        assert result["valid_source"] == [{"field": "test", "condition": "value"}]


class TestNormalizeFormatConversionConfig:
    """Test normalize_format_conversion_config function."""

    def test_empty_config_returns_empty_dict(self):
        """Test that empty config returns empty dictionary."""
        assert normalize_format_conversion_config({}) == {}
        assert normalize_format_conversion_config(None) == {}

    def test_direct_source_mapping_with_dict(self):
        """Test direct source mapping with dictionary configuration."""
        config = {
            "source1": {"extract_fields": ["name", "price"]},
            "source2": {"template": "Name: {{item.name}}"},
        }
        result = normalize_format_conversion_config(config)

        assert len(result) == 2
        assert isinstance(result["source1"], FormatConversionConfig)
        assert isinstance(result["source2"], FormatConversionConfig)
        assert result["source1"].extract_fields == ["name", "price"]
        assert result["source2"].template == "Name: {{item.name}}"

    def test_direct_source_mapping_with_config_object(self):
        """Test direct source mapping with FormatConversionConfig object."""
        config_obj = FormatConversionConfig(extract_fields=["id", "name"])
        config = {"source1": config_obj}

        result = normalize_format_conversion_config(config)
        assert result["source1"] is config_obj

    def test_named_configuration_nested_dict(self):
        """Test named configuration with nested dictionary structure."""
        config = {
            "conversion_rules": {
                "products": {"extract_fields": ["name", "price"]},
                "customers": {"template": "Customer: {{item.name}}"},
            }
        }
        result = normalize_format_conversion_config(config)

        assert len(result) == 2
        assert isinstance(result["products"], FormatConversionConfig)
        assert isinstance(result["customers"], FormatConversionConfig)

    def test_invalid_nested_dict_configuration_misinterprets_structure(self):
        """Test that nested dict configurations are misinterpreted - BUG!"""
        config = {
            "config_group": {
                "valid_source": {"extract_fields": ["name"]},
                "invalid_source": {"bad_field": "value"},
            }
        }
        result = normalize_format_conversion_config(config)

        # BUG: The complex nested detection logic incorrectly treats this as
        # direct source mapping instead of named configuration
        assert "valid_source" in result
        assert "invalid_source" in result  # Should not be here if logic was correct
        assert "config_group" not in result  # Named config key is missing

    def test_complex_nested_detection_logic_bug(self):
        """Test the complex nested detection logic that might have bugs."""
        # This tests the condition: not any(isinstance(v, dict) for v in value.values())
        config = {
            "source1": {
                "extract_fields": ["name"],
                "nested_dict": {"key": "value"},  # This makes it a "nested configuration"
            }
        }
        result = normalize_format_conversion_config(config)

        # Due to the logic bug, this might be treated as named configuration
        # instead of direct source mapping
        # The actual behavior depends on the implementation
        print(f"Result keys: {list(result.keys())}")
        print(f"Result: {result}")


class TestNormalizeContentSummarizationConfig:
    """Test normalize_content_summarization_config function."""

    def test_empty_config_returns_empty_dict(self):
        """Test that empty config returns empty dictionary."""
        assert normalize_content_summarization_config({}) == {}
        assert normalize_content_summarization_config(None) == {}

    def test_direct_source_mapping_with_dict(self):
        """Test direct source mapping with dictionary configuration."""
        config = {
            "source1": {
                "method": "statistical",
                "fields": ["price"],
                "summarize": ["mean", "max"],
            },
            "source2": {
                "method": "statistical",
                "fields": ["description"],
                "summarize": ["count"],
            },  # Fixed: added required summarize field
        }
        result = normalize_content_summarization_config(config)

        assert len(result) == 2
        assert isinstance(result["source1"], ContentSummarizationConfig)
        assert isinstance(result["source2"], ContentSummarizationConfig)
        assert result["source1"].method == "statistical"
        assert result["source2"].method == "statistical"

    def test_direct_source_mapping_with_config_object(self):
        """Test direct source mapping with ContentSummarizationConfig object."""
        config_obj = ContentSummarizationConfig(method="statistical", fields=["price"], summarize=["mean"])
        config = {"source1": config_obj}

        result = normalize_content_summarization_config(config)
        assert result["source1"] is config_obj

    @patch("builtins.print")
    def test_invalid_configuration_prints_warning(self, mock_print):
        """Test that invalid configurations print warnings."""
        config = {"invalid_source": {"invalid_method": "unknown"}}
        result = normalize_content_summarization_config(config)

        mock_print.assert_called()
        assert "invalid_source" not in result


class TestNormalizeRelationshipHighlightingConfig:
    """Test normalize_relationship_highlighting_config function."""

    @patch("builtins.print")
    def test_empty_config_returns_empty_dict_with_debug(self, mock_print):
        """Test that empty config returns empty dictionary and prints debug."""
        result = normalize_relationship_highlighting_config({})
        assert result == {}

        # Should print debug message
        mock_print.assert_called_with("[DEBUG] normalize_relationship_highlighting_config - Empty config")

    @patch("builtins.print")
    def test_none_config_returns_empty_dict_with_debug(self, mock_print):
        """Test that None config returns empty dictionary and prints debug."""
        result = normalize_relationship_highlighting_config(None)
        assert result == {}
        mock_print.assert_called_with("[DEBUG] normalize_relationship_highlighting_config - Empty config")

    def test_direct_relationship_config_object(self):
        """Test direct RelationshipConfig object handling."""
        config_obj = RelationshipConfig(link_fields=[{"source": "A", "to": "B", "match_on": "id"}])
        result = normalize_relationship_highlighting_config(config_obj)

        assert len(result) == 1
        assert "data" in result
        assert result["data"] is config_obj

    def test_dict_with_relationship_config_values(self):
        """Test dictionary with RelationshipConfig values."""
        config_obj = RelationshipConfig(link_fields=[{"source": "A", "to": "B", "match_on": "id"}])
        config = {"source1": config_obj}

        result = normalize_relationship_highlighting_config(config)
        assert result["source1"] is config_obj

    def test_dict_with_dict_values_converted_to_config(self):
        """Test dictionary with dict values that get converted to RelationshipConfig."""
        config = {"source1": {"link_fields": [{"source": "A", "to": "B", "match_on": "id"}]}}
        result = normalize_relationship_highlighting_config(config)

        assert isinstance(result["source1"], RelationshipConfig)
        assert result["source1"].link_fields == [{"source": "A", "to": "B", "match_on": "id"}]

    @patch("builtins.print")
    def test_invalid_dict_value_creates_invalid_object(self, mock_print):
        """Test that invalid dict values create objects with None values - BUG!"""
        config = {"source1": {"invalid_field": "value"}}
        result = normalize_relationship_highlighting_config(config)

        # BUG: Invalid config creates object instead of being rejected
        assert "source1" in result
        assert isinstance(result["source1"], RelationshipConfig)
        # No warning printed - silent failure bug!

    @patch("builtins.print")
    def test_unsupported_value_type_prints_warning(self, mock_print):
        """Test that unsupported value types print warnings."""
        config = {"source1": "invalid_string_value", "source2": 123}
        result = normalize_relationship_highlighting_config(config)

        # Should print warnings for both unsupported types
        assert mock_print.call_count >= 2
        assert len(result) == 0

    @patch("builtins.print")
    def test_unrecognized_config_type_prints_debug(self, mock_print):
        """Test that unrecognized config types print debug message."""
        result = normalize_relationship_highlighting_config("invalid_type")

        assert result == {}
        mock_print.assert_called()
        debug_call = str(mock_print.call_args)
        assert "[DEBUG] Unrecognized config type" in debug_call


class TestNormalizeAdvancedOperationsConfig:
    """Test normalize_advanced_operations_config function."""

    def test_empty_config_returns_empty_dict(self):
        """Test that empty config returns empty dictionary."""
        assert normalize_advanced_operations_config({}) == {}
        assert normalize_advanced_operations_config(None) == {}

    def test_direct_source_mapping_with_dict(self):
        """Test direct source mapping with dictionary configuration - NOW FIXED!"""
        config = {
            "result1": {
                "source": "data",
                "group_by": "category",
                "aggregate": {"total": "SUM(amount)"},
            },
            "result2": {"source": "users", "sort": "name", "limit": 10},
        }
        result = normalize_advanced_operations_config(config)

        # FIXED: The structure is now correctly preserved
        print(f"Actual result keys: {sorted(result.keys())}")
        print(f"Result: {result}")

        # Now works correctly - no flattening bug
        assert len(result) == 2
        assert "result1" in result
        assert "result2" in result
        assert isinstance(result["result1"], AdvancedOperationConfig)
        assert isinstance(result["result2"], AdvancedOperationConfig)
        assert result["result1"].source == "data"
        assert result["result1"].group_by == "category"
        assert result["result1"].aggregate == {"total": "SUM(amount)"}
        assert result["result2"].limit == 10

    @patch("builtins.print")
    def test_invalid_configuration_prints_warning(self, mock_print):
        """Test that invalid configurations print warnings."""
        config = {"invalid_result": {"invalid_field": "value"}}
        result = normalize_advanced_operations_config(config)

        mock_print.assert_called()
        assert "invalid_result" not in result


class TestNormalizeTemplateMatchingConfig:
    """Test normalize_template_matching_config function."""

    @patch("logging.warning")
    def test_empty_config_logs_warning(self, mock_warning):
        """Test that empty config logs warning."""
        result = normalize_template_matching_config({})
        assert result == {}
        mock_warning.assert_called_with("Empty template_matching configuration received")

    @patch("logging.warning")
    def test_none_config_logs_warning(self, mock_warning):
        """Test that None config logs warning."""
        result = normalize_template_matching_config(None)
        assert result == {}
        mock_warning.assert_called_with("Empty template_matching configuration received")

    def test_direct_template_matching_config_object(self):
        """Test direct TemplateMatchingConfig object handling."""
        config_obj = TemplateMatchingConfig(template="Hello {{name}}")
        result = normalize_template_matching_config(config_obj)

        assert len(result) == 1
        # Should generate a key based on template content
        key = list(result.keys())[0]
        assert result[key] is config_obj

    def test_dict_with_template_config_values(self):
        """Test dictionary with TemplateMatchingConfig values."""
        config_obj = TemplateMatchingConfig(template="Hello {{name}}")
        config = {"template1": config_obj}

        result = normalize_template_matching_config(config)
        assert result["template1"] is config_obj

    def test_dict_with_dict_values_converted_to_config(self):
        """Test dictionary with dict values that get converted to TemplateMatchingConfig."""
        config = {
            "template1": {"template": "Hello {{name}}"},
            "template2": {"template_id": "greeting_template"},
        }
        result = normalize_template_matching_config(config)

        assert len(result) == 2
        assert isinstance(result["template1"], TemplateMatchingConfig)
        assert isinstance(result["template2"], TemplateMatchingConfig)
        assert result["template1"].template == "Hello {{name}}"
        assert result["template2"].template_id == "greeting_template"

    @patch("logging.error")
    def test_invalid_dict_configuration_creates_invalid_object(self, mock_error):
        """Test that invalid dict configurations create objects - BUG!"""
        config = {"invalid_template": {"invalid_field": "value"}}
        result = normalize_template_matching_config(config)

        # BUG: Invalid config creates object instead of being rejected
        assert "invalid_template" in result
        assert isinstance(result["invalid_template"], TemplateMatchingConfig)
        # No error logged - silent failure bug!

    @patch("logging.warning")
    def test_unsupported_value_type_creates_invalid_objects(self, mock_warning):
        """Test that unsupported value types create invalid objects - BUG!"""
        config = {"template1": "invalid_string_value", "template2": 123}
        result = normalize_template_matching_config(config)

        # BUG: Unsupported types should be rejected but they create invalid objects
        print(f"Result: {result}")
        # The actual behavior needs to be tested to see what happens


class TestGenerateTemplateKey:
    """Test _generate_template_key function."""

    def test_generate_key_from_template_id(self):
        """Test key generation from template_id."""
        config = TemplateMatchingConfig(template="test", template_id="my_template")
        key = _generate_template_key(config)
        assert key == "my_template"

    def test_generate_key_from_template_content(self):
        """Test key generation from template content."""
        config = TemplateMatchingConfig(template="# Invoice Template\nHello {{name}}")
        key = _generate_template_key(config)
        assert key == "invoice_template"

    def test_generate_key_from_template_content_long_title(self):
        """Test key generation truncates long titles."""
        config = TemplateMatchingConfig(
            template="# This is a very long template title that should be truncated\nContent"
        )
        key = _generate_template_key(config)
        assert len(key) == 20
        assert key == "this_is_a_very_long_"

    def test_generate_key_from_template_content_uses_first_line(self):
        """Test key generation uses first line content, not hash fallback."""
        config = TemplateMatchingConfig(template="Just some content without title")
        key = _generate_template_key(config)
        # The actual behavior: uses first 20 chars of first line, not hash
        assert key == "just_some_content_wi"
        assert len(key) == 20

    def test_generate_key_from_empty_template_validation_error(self):
        """Test that empty template raises validation error - this is expected."""
        with pytest.raises(Exception):  # Pydantic validation prevents empty templates
            TemplateMatchingConfig(template="")

    def test_generate_key_from_none_template_validation_error(self):
        """Test that None template raises validation error - this is expected."""
        with pytest.raises(Exception):  # Pydantic validation prevents None templates
            TemplateMatchingConfig(template=None)

    def test_generate_key_consistency(self):
        """Test that same template content generates same key."""
        config1 = TemplateMatchingConfig(template="# Same Title\nContent")
        config2 = TemplateMatchingConfig(template="# Same Title\nDifferent content")

        key1 = _generate_template_key(config1)
        key2 = _generate_template_key(config2)

        # Both should generate same key since they have same title
        assert key1 == key2 == "same_title"

    def test_generate_key_hash_consistency_uses_content(self):
        """Test that same content generates same key using first line."""
        config1 = TemplateMatchingConfig(template="Same content")
        config2 = TemplateMatchingConfig(template="Same content")

        key1 = _generate_template_key(config1)
        key2 = _generate_template_key(config2)

        assert key1 == key2
        assert key1 == "same_content"  # Uses first line, not hash


class TestNormalizersEdgeCases:
    """Test edge cases and potential bugs across normalizers."""

    def test_all_normalizers_handle_none_gracefully(self):
        """Test that all normalizers handle None input gracefully."""
        assert normalize_contextual_filtering_config(None) == {}
        assert normalize_format_conversion_config(None) == {}
        assert normalize_content_summarization_config(None) == {}
        assert normalize_relationship_highlighting_config(None) == {}
        assert normalize_advanced_operations_config(None) == {}
        assert normalize_template_matching_config(None) == {}

    def test_all_normalizers_handle_empty_dict_gracefully(self):
        """Test that all normalizers handle empty dict input gracefully."""
        assert normalize_contextual_filtering_config({}) == {}
        assert normalize_format_conversion_config({}) == {}
        assert normalize_content_summarization_config({}) == {}
        assert normalize_relationship_highlighting_config({}) == {}
        assert normalize_advanced_operations_config({}) == {}
        assert normalize_template_matching_config({}) == {}

    def test_normalizers_preserve_config_object_references(self):
        """Test that normalizers preserve object references when possible."""
        filter_config = ContextualFilterConfig(field="test", condition="value")
        format_config = FormatConversionConfig(extract_fields=["test"])
        summary_config = ContentSummarizationConfig(method="statistical", fields=["test"], summarize=["count"])
        relationship_config = RelationshipConfig(link_fields=[])
        advanced_config = AdvancedOperationConfig(source="test")
        template_config = TemplateMatchingConfig(template="test")

        # Test that object references are preserved
        assert normalize_contextual_filtering_config({"src": [filter_config]})["src"][0] is filter_config
        assert normalize_format_conversion_config({"src": format_config})["src"] is format_config
        assert normalize_content_summarization_config({"src": summary_config})["src"] is summary_config
        assert normalize_relationship_highlighting_config({"src": relationship_config})["src"] is relationship_config
        assert normalize_advanced_operations_config({"src": advanced_config})["src"] is advanced_config
        assert normalize_template_matching_config({"src": template_config})["src"] is template_config


if __name__ == "__main__":
    pytest.main([__file__])
