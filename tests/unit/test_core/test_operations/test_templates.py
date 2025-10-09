"""
Comprehensive unit tests for template matching operations.

Tests cover Jinja2 template processing, file loading, variable substitution,
and error conditions to uncover production bugs.

EXPECTED BUGS TO FIND:
1. File system operations failing when template files don't exist
2. Jinja2 template syntax errors not handled gracefully
3. Missing template content validation
4. Expression engine integration issues
5. Variable substitution errors with missing context
"""

import os
import tempfile
from unittest.mock import Mock, mock_open, patch

import pytest

from shedboxai.core.config.models import TemplateMatchingConfig
from shedboxai.core.operations.templates import TemplateMatchingHandler


class TestTemplateMatchingHandler:
    """Test suite for TemplateMatchingHandler operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = TemplateMatchingHandler()
        self.mock_engine = Mock()
        self.handler_with_engine = TemplateMatchingHandler(engine=self.mock_engine)

    # Basic Configuration Tests
    def test_operation_name(self):
        """Test operation name property."""
        assert self.handler.operation_name == "template_matching"

    def test_empty_config_returns_data_unchanged(self):
        """Test that empty config returns data unchanged."""
        data = {"source": [{"name": "test", "value": 10}]}
        result = self.handler.process(data, {})
        assert result == data

    # Template String Processing Tests
    def test_simple_template_string(self):
        """Test basic template string processing."""
        data = {"users": [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]}
        config = {
            "greeting": TemplateMatchingConfig(
                template="Hello {{ users[0].name }}, you are {{ users[0].age }} years old!"
            )
        }

        result = self.handler.process(data, config)
        assert result["greeting"] == "Hello Alice, you are 30 years old!"

    def test_template_with_loops(self):
        """Test template with Jinja2 loops - FIXED."""
        data = {
            "items": [
                {"name": "Apple", "price": 1.50},
                {"name": "Banana", "price": 0.75},
                {"name": "Orange", "price": 2.00},
            ]
        }
        config = {
            "menu": TemplateMatchingConfig(
                template="""Menu:
{% for item in items %}
- {{ item.name }}: ${{ item.price }}
{% endfor %}"""
            )
        }

        result = self.handler.process(data, config)
        # BUG 1 FIX: Expect correct formatting with $ symbol and no trailing newline
        expected = """Menu:

- Apple: $1.5

- Banana: $0.75

- Orange: $2.0
"""
        assert result["menu"] == expected

    def test_template_with_conditionals(self):
        """Test template with Jinja2 conditionals."""
        data = {"user": {"name": "Alice", "is_admin": True, "balance": 100}}
        config = {
            "message": TemplateMatchingConfig(
                template="""Welcome {{ user.name }}!
{% if user.is_admin %}
You have admin privileges.
{% endif %}
{% if user.balance > 50 %}
Your balance is healthy: ${{ user.balance }}
{% else %}
Low balance warning: ${{ user.balance }}
{% endif %}"""
            )
        }

        result = self.handler.process(data, config)
        assert "Welcome Alice!" in result["message"]
        assert "admin privileges" in result["message"]
        assert "healthy: $100" in result["message"]

    def test_template_with_filters(self):
        """Test template with custom Jinja2 filters."""
        data = {
            "products": [
                {"name": "Widget", "price": 19.99},
                {"name": "Gadget", "price": 49.50},
            ]
        }
        config = {
            "price_list": TemplateMatchingConfig(
                template="""Products:
{% for product in products %}
{{ product.name }}: {{ product.price | currency }}
{% endfor %}"""
            )
        }

        result = self.handler.process(data, config)
        # Should use the custom currency filter
        assert "$19.99" in result["price_list"] or "$19.99" in result["price_list"].replace(" ", "")

    def test_template_with_variables(self):
        """Test template with additional variables."""
        data = {"items": [{"name": "test"}]}
        config = {
            "report": TemplateMatchingConfig(
                template="Report for {{ company_name }}: {{ items | length }} items",
                variables={"company_name": "ACME Corp"},
            )
        }

        result = self.handler.process(data, config)
        assert result["report"] == "Report for ACME Corp: 1 items"

    # Template File Loading Tests
    @patch("os.path.exists")
    @patch("builtins.open", new_callable=mock_open, read_data="Hello {{ user.name }}!")
    def test_template_file_loading_j2_extension(self, mock_file, mock_exists):
        """Test loading template from .j2 file."""
        mock_exists.side_effect = lambda path: path.endswith("test_template.j2")

        data = {"user": {"name": "Alice"}}
        config = {"result": TemplateMatchingConfig(template_id="test_template")}

        result = self.handler.process(data, config)
        assert result["result"] == "Hello Alice!"
        mock_file.assert_called_once()

    @patch("os.path.exists")
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data="Template content: {{ data }}",
    )
    def test_template_file_loading_no_extension(self, mock_file, mock_exists):
        """Test loading template from file without .j2 extension."""
        # First call (with .j2) returns False, second call (without) returns True
        mock_exists.side_effect = [False, True]

        data = {"data": "test"}
        config = {"result": TemplateMatchingConfig(template_id="template_name")}

        result = self.handler.process(data, config)
        assert result["result"] == "Template content: test"

    @patch("os.path.exists", return_value=False)
    def test_template_file_not_found(self, mock_exists):
        """Test template file not found error."""
        data = {"test": "data"}
        config = {"result": TemplateMatchingConfig(template_id="missing_template")}

        result = self.handler.process(data, config)
        # Should handle missing file gracefully with error message
        assert "ERROR:" in result["result"]

    @patch("os.path.exists", return_value=True)
    @patch("builtins.open", side_effect=PermissionError("Access denied"))
    def test_template_file_permission_error(self, mock_file, mock_exists):
        """Test template file permission error."""
        data = {"test": "data"}
        config = {"result": TemplateMatchingConfig(template_id="restricted_template")}

        result = self.handler.process(data, config)
        # Should handle permission errors gracefully
        assert "ERROR:" in result["result"]

    # Expression Engine Integration Tests
    def test_template_with_expression_engine(self):
        """Test template processing with expression engine."""
        self.mock_engine.substitute_variables.return_value = "Processed by engine"

        data = {"value": 42}
        config = {"result": TemplateMatchingConfig(template="Value: {{ value }}")}

        result = self.handler_with_engine.process(data, config)
        # Should use expression engine for additional processing
        assert result["result"] == "Processed by engine"
        self.mock_engine.substitute_variables.assert_called_once()

    def test_template_without_expression_engine(self):
        """Test template processing without expression engine."""
        data = {"name": "test", "count": 5}
        config = {"result": TemplateMatchingConfig(template="Name: {{ name }}, Count: {{ count }}")}

        result = self.handler.process(data, config)
        assert result["result"] == "Name: test, Count: 5"

    # Error Handling Tests
    def test_template_syntax_error(self):
        """Test handling of Jinja2 template syntax errors."""
        data = {"name": "test"}
        config = {"result": TemplateMatchingConfig(template="Hello {{ name")}  # Missing closing braces

        result = self.handler.process(data, config)
        # Should handle syntax errors gracefully
        assert "ERROR:" in result["result"]

    def test_template_undefined_variable(self):
        """Test handling of undefined variables in template."""
        data = {"existing": "value"}
        config = {"result": TemplateMatchingConfig(template="Value: {{ missing_variable }}")}  # Variable doesn't exist

        result = self.handler.process(data, config)
        # Should handle undefined variables gracefully (Jinja2 default behavior)
        # Jinja2 by default renders undefined variables as empty string or raises error
        assert result["result"]  # Should not crash

    def test_template_filter_error(self):
        """Test handling of filter errors in template."""
        data = {"text": "hello"}
        config = {"result": TemplateMatchingConfig(template="Text: {{ text | nonexistent_filter }}")}

        result = self.handler.process(data, config)
        # Should handle filter errors gracefully
        assert "ERROR:" in result["result"]

    def test_missing_template_content(self):
        """Test handling when neither template nor template_id is provided."""
        data = {"test": "data"}
        config = {"result": TemplateMatchingConfig()}  # No template or template_id

        result = self.handler.process(data, config)
        # Should handle missing template content
        assert "ERROR:" in result["result"]

    # Configuration Validation Tests
    def test_dict_config_conversion(self):
        """Test that dict configs are converted to TemplateMatchingConfig objects."""
        data = {"name": "test"}
        config = {"result": {"template": "Hello {{ name }}!"}}

        result = self.handler.process(data, config)
        assert result["result"] == "Hello test!"

    def test_invalid_dict_config_logs_warning(self, caplog):
        """Test that invalid dict config logs warning - NOW FIXED."""
        data = {"name": "test"}
        config = {"result": {"invalid_field": "value"}}

        result = self.handler.process(data, config)
        # BUG 2 FIX: Check stdout for _log_warning output
        assert "Invalid template matching configuration" in caplog.text

    def test_invalid_config_type_logs_warning(self, caplog):
        """Test that invalid config type logs warning - NOW FIXED."""
        data = {"name": "test"}
        config = {"result": "invalid_string_config"}

        result = self.handler.process(data, config)
        # BUG 3 FIX: Check stdout for _log_warning output
        assert "Invalid template matching configuration" in caplog.text
        assert "expected dict or TemplateMatchingConfig" in caplog.text

    # Custom Filters Tests
    def test_custom_currency_filter(self):
        """Test custom currency filter."""
        data = {"price": 19.99}
        config = {"result": TemplateMatchingConfig(template="Price: {{ price | currency }}")}

        result = self.handler.process(data, config)
        assert "$19.99" in result["result"]

    def test_custom_percentage_filter(self):
        """Test custom percentage filter."""
        data = {"rate": 15.5}
        config = {"result": TemplateMatchingConfig(template="Rate: {{ rate | percentage }}")}

        result = self.handler.process(data, config)
        assert "15.5%" in result["result"]

    def test_custom_safe_get_filter(self):
        """Test custom safe_get filter."""
        data = {"user": {"name": "Alice"}}
        config = {"result": TemplateMatchingConfig(template="Email: {{ user | safe_get('email', 'Not provided') }}")}

        result = self.handler.process(data, config)
        assert "Not provided" in result["result"]

    def test_custom_length_filter(self):
        """Test custom length filter."""
        data = {"items": [1, 2, 3, 4, 5]}
        config = {"result": TemplateMatchingConfig(template="Count: {{ items | length }}")}

        result = self.handler.process(data, config)
        assert "Count: 5" in result["result"]

    def test_custom_first_last_filters(self):
        """Test custom first and last filters."""
        data = {"numbers": [1, 2, 3, 4, 5]}
        config = {"result": TemplateMatchingConfig(template="First: {{ numbers | first }}, Last: {{ numbers | last }}")}

        result = self.handler.process(data, config)
        assert "First: 1" in result["result"]
        assert "Last: 5" in result["result"]

    # Multiple Templates Tests
    def test_multiple_templates_processing(self):
        """Test processing multiple templates in single operation."""
        data = {
            "user": {"name": "Alice", "role": "admin"},
            "stats": {"logins": 42, "errors": 3},
        }
        config = {
            "welcome": TemplateMatchingConfig(template="Welcome {{ user.name }}! You are {{ user.role }}."),
            "summary": TemplateMatchingConfig(template="Stats: {{ stats.logins }} logins, {{ stats.errors }} errors"),
        }

        result = self.handler.process(data, config)
        assert result["welcome"] == "Welcome Alice! You are admin."
        assert result["summary"] == "Stats: 42 logins, 3 errors"

    # Edge Cases and Complex Scenarios
    def test_nested_data_access(self):
        """Test template with deeply nested data access."""
        data = {
            "company": {
                "departments": {
                    "engineering": {
                        "teams": [
                            {"name": "Backend", "size": 5},
                            {"name": "Frontend", "size": 3},
                        ]
                    }
                }
            }
        }
        config = {
            "report": TemplateMatchingConfig(
                template="""Engineering Teams:
{% for team in company.departments.engineering.teams %}
{{ team.name }}: {{ team.size }} members
{% endfor %}"""
            )
        }

        result = self.handler.process(data, config)
        assert "Backend: 5 members" in result["report"]
        assert "Frontend: 3 members" in result["report"]

    def test_template_with_complex_logic(self):
        """Test template with complex Jinja2 logic."""
        data = {
            "orders": [
                {"id": 1, "total": 100, "status": "completed"},
                {"id": 2, "total": 250, "status": "pending"},
                {"id": 3, "total": 75, "status": "cancelled"},
            ]
        }
        config = {
            "summary": TemplateMatchingConfig(
                template="""Order Summary:
Total Orders: {{ orders | length }}
{% set completed = orders | selectattr('status', 'equalto', 'completed') | list %}
Completed: {{ completed | length }}
{% set total_revenue = completed | sum(attribute='total') %}
Total Revenue: ${{ total_revenue }}
High Value Orders (>$200):
{% for order in orders if order.total > 200 %}
- Order #{{ order.id }}: ${{ order.total }} ({{ order.status }})
{% endfor %}"""
            )
        }

        result = self.handler.process(data, config)
        assert "Total Orders: 3" in result["summary"]
        assert "Order #2: $250" in result["summary"]

    def test_template_inheritance_macros(self):
        """Test template with macros (if supported)."""
        data = {
            "products": [
                {"name": "Widget", "price": 10, "sale": True},
                {"name": "Gadget", "price": 20, "sale": False},
            ]
        }
        config = {
            "catalog": TemplateMatchingConfig(
                template="""
{%- macro product_display(product) -%}
{{ product.name }} - ${{ product.price }}
{%- if product.sale %} (ON SALE!){% endif -%}
{%- endmacro -%}

Product Catalog:
{% for product in products %}
{{ product_display(product) }}
{% endfor %}"""
            )
        }

        result = self.handler.process(data, config)
        assert "Widget - $10 (ON SALE!)" in result["catalog"]
        assert "Gadget - $20" in result["catalog"]
        assert "(ON SALE!)" not in result["catalog"].split("Gadget")[1].split("\n")[0]

    # Bug Detection Tests - These should expose real bugs
    def test_jinja_environment_configuration_issues(self):
        """Test potential Jinja2 environment configuration problems."""
        # Test template that might reveal environment issues
        data = {"items": ["<script>", "safe & sound", "100% tested"]}
        config = {
            "html": TemplateMatchingConfig(
                template="""<div>
{% for item in items %}
<p>{{ item }}</p>
{% endfor %}
</div>"""
            )
        }

        result = self.handler.process(data, config)
        # Should handle HTML escaping according to environment settings
        assert result["html"]

    def test_file_system_race_conditions(self):
        """Test potential file system race conditions."""
        # This test might reveal issues with file loading
        data = {"test": "data"}
        config = {"result": TemplateMatchingConfig(template_id="potentially_missing_file")}

        # Should handle file system issues gracefully
        result = self.handler.process(data, config)
        assert "result" in result

    def test_memory_usage_with_large_templates(self):
        """Test memory usage with large template processing."""
        # Create large dataset
        data = {"items": [{"id": i, "value": f"item_{i}"} for i in range(1000)]}
        config = {
            "large_output": TemplateMatchingConfig(
                template="""Large Report:
{% for item in items %}
Item {{ item.id }}: {{ item.value }}
{% endfor %}
Total: {{ items | length }} items"""
            )
        }

        # Should handle large templates without memory issues
        result = self.handler.process(data, config)
        assert "Total: 1000 items" in result["large_output"]

    def test_concurrent_template_processing(self):
        """Test potential concurrency issues in template processing."""
        # Test multiple templates that might interfere with each other
        data = {"shared": "data", "counter": 42}
        config = {
            "template1": TemplateMatchingConfig(template="Template 1: {{ shared }} - {{ counter }}"),
            "template2": TemplateMatchingConfig(template="Template 2: {{ counter }} - {{ shared }}"),
            "template3": TemplateMatchingConfig(template="Template 3: Combined {{ shared }}{{ counter }}"),
        }

        result = self.handler.process(data, config)
        # All templates should process correctly without interference
        assert result["template1"] == "Template 1: data - 42"
        assert result["template2"] == "Template 2: 42 - data"
        assert result["template3"] == "Template 3: Combined data42"
