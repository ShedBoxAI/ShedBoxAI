"""
Tests for base.py - OperationHandler abstract base class.

Tests focus on finding real bugs in the abstract base implementation
and ensuring concrete implementations behave correctly.
"""

import io
import sys
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from shedboxai.core.operations.base import OperationHandler


class ConcreteHandler(OperationHandler):
    """Concrete implementation for testing the abstract base class."""

    def __init__(self, engine=None, operation_name="test_operation"):
        super().__init__(engine)
        self._operation_name = operation_name

    @property
    def operation_name(self) -> str:
        return self._operation_name

    def process(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        return data.copy()


class TestOperationHandler:
    """Test cases for OperationHandler base class."""

    def test_abstract_class_cannot_be_instantiated(self):
        """Test that OperationHandler cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            OperationHandler()

    def test_concrete_handler_initialization_without_engine(self):
        """Test concrete handler initializes correctly without engine."""
        handler = ConcreteHandler()
        assert handler.engine is None
        assert handler.operation_name == "test_operation"

    def test_concrete_handler_initialization_with_engine(self):
        """Test concrete handler initializes correctly with engine."""
        mock_engine = MagicMock()
        handler = ConcreteHandler(engine=mock_engine)
        assert handler.engine is mock_engine
        assert handler.operation_name == "test_operation"

    def test_operation_name_property_required(self):
        """Test that operation_name property must be implemented."""

        class IncompleteHandler(OperationHandler):
            def process(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
                return data

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteHandler()

    def test_process_method_required(self):
        """Test that process method must be implemented."""

        class IncompleteHandler(OperationHandler):
            @property
            def operation_name(self) -> str:
                return "incomplete"

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteHandler()

    def test_validate_config_default_implementation_valid_dict(self):
        """Test default validate_config accepts valid dictionary."""
        handler = ConcreteHandler()
        config = {"key": "value", "nested": {"data": 123}}
        assert handler.validate_config(config) is True

    def test_validate_config_default_implementation_empty_dict(self):
        """Test default validate_config accepts empty dictionary."""
        handler = ConcreteHandler()
        assert handler.validate_config({}) is True

    def test_validate_config_default_implementation_none(self):
        """Test default validate_config rejects None."""
        handler = ConcreteHandler()
        assert handler.validate_config(None) is False

    def test_validate_config_default_implementation_non_dict(self):
        """Test default validate_config rejects non-dictionary types."""
        handler = ConcreteHandler()
        assert handler.validate_config("string") is False
        assert handler.validate_config(123) is False
        assert handler.validate_config([]) is False
        assert handler.validate_config(set()) is False

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_log_warning_output_format(self, mock_stdout):
        """Test _log_warning outputs correct format."""
        handler = ConcreteHandler()
        handler._log_warning("Test warning message")
        output = mock_stdout.getvalue()
        assert "Warning [test_operation]: Test warning message\n" == output

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_log_error_output_format(self, mock_stdout):
        """Test _log_error outputs correct format."""
        handler = ConcreteHandler()
        handler._log_error("Test error message")
        output = mock_stdout.getvalue()
        assert "Error [test_operation]: Test error message\n" == output

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_log_warning_with_special_characters(self, mock_stdout):
        """Test _log_warning handles special characters correctly."""
        handler = ConcreteHandler()
        handler._log_warning("Warning with unicode: ñáéíóú and symbols: @#$%")
        output = mock_stdout.getvalue()
        assert "Warning [test_operation]: Warning with unicode: ñáéíóú and symbols: @#$%" in output

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_log_error_with_special_characters(self, mock_stdout):
        """Test _log_error handles special characters correctly."""
        handler = ConcreteHandler()
        handler._log_error("Error with unicode: ñáéíóú and symbols: @#$%")
        output = mock_stdout.getvalue()
        assert "Error [test_operation]: Error with unicode: ñáéíóú and symbols: @#$%" in output

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_log_warning_empty_message(self, mock_stdout):
        """Test _log_warning with empty message."""
        handler = ConcreteHandler()
        handler._log_warning("")
        output = mock_stdout.getvalue()
        assert "Warning [test_operation]: \n" == output

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_log_error_empty_message(self, mock_stdout):
        """Test _log_error with empty message."""
        handler = ConcreteHandler()
        handler._log_error("")
        output = mock_stdout.getvalue()
        assert "Error [test_operation]: \n" == output

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_log_methods_with_different_operation_names(self, mock_stdout):
        """Test logging methods use correct operation name."""
        handler1 = ConcreteHandler(operation_name="handler_one")
        handler2 = ConcreteHandler(operation_name="handler_two")

        handler1._log_warning("First warning")
        handler2._log_error("First error")

        output = mock_stdout.getvalue()
        assert "Warning [handler_one]: First warning\n" in output
        assert "Error [handler_two]: First error\n" in output

    def test_concrete_process_method_preserves_data(self):
        """Test that concrete process method works as expected."""
        handler = ConcreteHandler()
        original_data = {"source1": [{"id": 1}], "source2": [{"id": 2}]}
        config = {"some": "config"}

        result = handler.process(original_data, config)

        # Should return a copy, not the same object
        assert result == original_data
        assert result is not original_data

    def test_concrete_process_method_data_modification_independence(self):
        """Test that modifications to returned data don't affect original."""
        handler = ConcreteHandler()
        original_data = {"source1": [{"id": 1}], "source2": [{"id": 2}]}
        config = {}

        result = handler.process(original_data, config)
        result["new_key"] = "new_value"

        assert "new_key" not in original_data

    def test_engine_access_after_initialization(self):
        """Test that engine can be accessed and used after initialization."""
        mock_engine = MagicMock()
        mock_engine.some_method.return_value = "engine_result"

        handler = ConcreteHandler(engine=mock_engine)

        # Test that we can access and use the engine
        result = handler.engine.some_method("test_input")
        assert result == "engine_result"
        mock_engine.some_method.assert_called_once_with("test_input")

    def test_engine_none_handling(self):
        """Test handler behavior when engine is None."""
        handler = ConcreteHandler(engine=None)

        # Should not raise an error when engine is None
        assert handler.engine is None

        # Should handle engine being None gracefully in operations
        # (this would be implementation-specific in real handlers)

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_multiple_log_calls_order(self, mock_stdout):
        """Test that multiple log calls maintain correct order."""
        handler = ConcreteHandler()

        handler._log_warning("First warning")
        handler._log_error("First error")
        handler._log_warning("Second warning")

        output = mock_stdout.getvalue().split("\n")
        assert "Warning [test_operation]: First warning" == output[0]
        assert "Error [test_operation]: First error" == output[1]
        assert "Warning [test_operation]: Second warning" == output[2]

    def test_validate_config_with_complex_nested_structures(self):
        """Test validate_config with complex nested dictionary structures."""
        handler = ConcreteHandler()

        complex_config = {
            "level1": {
                "level2": {
                    "level3": {
                        "data": [1, 2, 3],
                        "metadata": {"type": "test", "version": 1.0},
                    }
                }
            },
            "arrays": [{"item": 1}, {"item": 2}],
            "mixed": None,
        }

        assert handler.validate_config(complex_config) is True

    def test_operation_name_property_consistency(self):
        """Test that operation_name property returns consistent values."""
        handler = ConcreteHandler(operation_name="consistent_name")

        # Should return the same value multiple times
        assert handler.operation_name == "consistent_name"
        assert handler.operation_name == "consistent_name"
        assert handler.operation_name == "consistent_name"


class TestOperationHandlerEdgeCases:
    """Test edge cases and potential bugs in OperationHandler."""

    def test_subclass_with_dynamic_operation_name(self):
        """Test subclass that changes operation_name dynamically."""

        class DynamicHandler(OperationHandler):
            def __init__(self, engine=None):
                super().__init__(engine)
                self._name_counter = 0

            @property
            def operation_name(self) -> str:
                self._name_counter += 1
                return f"dynamic_{self._name_counter}"

            def process(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
                return data

        handler = DynamicHandler()

        # Operation name should change each time it's accessed
        first_call = handler.operation_name
        second_call = handler.operation_name

        assert first_call != second_call
        assert first_call == "dynamic_1"
        assert second_call == "dynamic_2"

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_log_with_dynamic_operation_name(self, mock_stdout):
        """Test logging with dynamically changing operation name."""

        class DynamicHandler(OperationHandler):
            def __init__(self, engine=None):
                super().__init__(engine)
                self._name_counter = 0

            @property
            def operation_name(self) -> str:
                self._name_counter += 1
                return f"dynamic_{self._name_counter}"

            def process(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
                return data

        handler = DynamicHandler()
        handler._log_warning("Test message")

        output = mock_stdout.getvalue()
        assert "Warning [dynamic_1]: Test message\n" == output

    def test_validate_config_override_in_subclass(self):
        """Test that subclasses can override validate_config."""

        class StrictHandler(ConcreteHandler):
            def validate_config(self, config: Dict[str, Any]) -> bool:
                # Only allow configs with 'required_field'
                return isinstance(config, dict) and "required_field" in config

        handler = StrictHandler()

        assert handler.validate_config({"required_field": "value"}) is True
        assert handler.validate_config({"other_field": "value"}) is False
        assert handler.validate_config({}) is False

    def test_process_method_with_mutable_data_mutation(self):
        """Test process method behavior with mutable nested data."""

        class MutatingHandler(OperationHandler):
            @property
            def operation_name(self) -> str:
                return "mutating"

            def process(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
                # This handler modifies the original data (bad practice)
                result = data  # No .copy()!
                if "items" in result:
                    result["items"].append({"new": "item"})
                return result

        handler = MutatingHandler()
        original_data = {"items": [{"id": 1}]}

        result = handler.process(original_data, {})

        # This reveals the bug - original data was modified
        assert len(original_data["items"]) == 2
        assert original_data["items"][1] == {"new": "item"}
        assert result is original_data  # Same object reference

    def test_engine_attribute_modification_after_init(self):
        """Test what happens if engine attribute is modified after initialization."""
        handler = ConcreteHandler(engine=None)
        assert handler.engine is None

        # Modify engine attribute directly
        new_engine = MagicMock()
        handler.engine = new_engine

        assert handler.engine is new_engine

        # Test that the new engine works
        handler.engine.test_method.return_value = "success"
        result = handler.engine.test_method()
        assert result == "success"


if __name__ == "__main__":
    pytest.main([__file__])
