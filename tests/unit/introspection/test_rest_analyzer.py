"""
Tests for REST API analyzer.

Tests the REST analyzer functionality including authentication,
schema generation, and LLM optimization features.
"""

import json
from unittest.mock import MagicMock, mock_open, patch

import pytest

from shedboxai.core.introspection.analyzers.rest_analyzer import RESTAnalyzer
from shedboxai.core.introspection.models import AnalysisStatus, SourceType


class TestRESTAnalyzer:
    def setup_method(self):
        self.analyzer = RESTAnalyzer()

    def test_supported_type(self):
        """Test that analyzer supports REST type"""
        assert self.analyzer.supported_type == SourceType.REST

    def test_successful_api_analysis(self):
        """Test successful API analysis with authentication"""
        # Mock the connector
        mock_connector = MagicMock()
        self.analyzer.connector = mock_connector

        # Mock API response
        api_response = {
            "data": [
                {"id": 1, "name": "Item 1", "value": 100},
                {"id": 2, "name": "Item 2", "value": 200},
            ],
            "pagination": {"total": 50, "page": 1, "per_page": 2},
        }

        mock_connector.get_data.return_value = {"test_api": api_response}

        config = {
            "name": "test_api",
            "type": "rest",
            "url": "https://api.example.com/data",
            "headers": {"Authorization": "Bearer test_token"},
            "response_path": "data",
        }

        result = self.analyzer.analyze(config)

        assert result.success
        assert result.authentication_success
        assert result.schema_info is not None
        assert result.schema_info.json_schema is not None
        assert result.schema_info.pagination_info is not None
        assert len(result.sample_data) == 2
        assert result.size_info.record_count == 2

    def test_authentication_failure(self):
        """Test handling of authentication failure"""
        # Mock the connector to return an error
        mock_connector = MagicMock()
        self.analyzer.connector = mock_connector

        mock_connector.get_data.return_value = {"secure_api": {"error": "API authentication failed"}}

        config = {
            "name": "secure_api",
            "type": "rest",
            "url": "https://api.example.com/secure",
            "headers": {"Authorization": "Bearer invalid_token"},
        }

        result = self.analyzer.analyze(config)

        assert not result.success
        assert not result.authentication_success
        assert "authentication" in result.error_message.lower()

    def test_missing_data_source(self):
        """Test handling when data source is not in response"""
        mock_connector = MagicMock()
        self.analyzer.connector = mock_connector

        # Return empty response (missing the expected source)
        mock_connector.get_data.return_value = {}

        config = {
            "name": "missing_api",
            "type": "rest",
            "url": "https://api.example.com/data",
        }

        result = self.analyzer.analyze(config)

        assert not result.success
        assert (
            "no data returned" in result.error_message.lower() or "api request failed" in result.error_message.lower()
        )

    def test_response_path_extraction(self):
        """Test response path extraction logic"""
        response_data = {
            "status": "success",
            "data": {"items": [{"id": 1}, {"id": 2}], "meta": {"count": 2}},
        }

        # Test simple path
        extracted = self.analyzer._extract_main_data(response_data, "data.items")
        assert extracted == [{"id": 1}, {"id": 2}]

        # Test non-existent path
        extracted = self.analyzer._extract_main_data(response_data, "nonexistent.path")
        assert extracted == response_data  # Should return original data

    def test_pagination_detection(self):
        """Test pagination detection in responses"""
        # Response with pagination
        paginated_response = {
            "data": [{"id": 1}],
            "pagination": {"total": 100, "page": 1},
        }

        has_pagination = self.analyzer._detect_pagination_in_response(paginated_response)
        assert has_pagination

        # Response without pagination
        simple_response = {"items": [{"id": 1}]}
        has_pagination = self.analyzer._detect_pagination_in_response(simple_response)
        assert not has_pagination

    def test_response_path_finding(self):
        """Test automatic response path discovery"""
        complex_response = {
            "status": "ok",
            "result": {"users": [{"id": 1}, {"id": 2}], "meta": {"total": 2}},
            "data": {"items": [{"id": 3}, {"id": 4}]},
        }

        paths = self.analyzer._find_response_paths(complex_response)

        # Should find both data arrays
        assert "result.users" in paths
        assert "data.items" in paths

    def test_schema_generation(self):
        """Test JSON schema generation with genson"""
        sample_data = [
            {"id": 1, "name": "Alice", "active": True},
            {"id": 2, "name": "Bob", "active": False},
        ]

        schema_info = self.analyzer._analyze_api_schema(sample_data, {"data": sample_data})

        assert schema_info.json_schema is not None
        schema = schema_info.json_schema

        # Check that schema was generated correctly
        assert "type" in schema
        # When we pass the sample_data directly, genson might create an object schema
        # Let's check for either array or object with items
        assert schema["type"] in ["array", "object"]
        if schema["type"] == "array":
            assert "items" in schema
            assert "properties" in schema["items"]
            properties = schema["items"]["properties"]
        else:
            # For object type, check properties directly
            assert "properties" in schema
            properties = schema["properties"]
        assert "id" in properties
        assert "name" in properties
        assert "active" in properties

    def test_api_version_extraction(self):
        """Test API version extraction"""
        # From response data
        headers = {}
        response_data = {"meta": {"api_version": "1.5"}}

        version = self.analyzer._extract_api_version(headers, response_data)
        assert version == "1.5"

        # From top-level response
        response_data = {"version": "2.0"}
        version = self.analyzer._extract_api_version(headers, response_data)
        assert version == "2.0"

    def test_nesting_depth_calculation(self):
        """Test nesting depth calculation"""
        from shedboxai.core.introspection.schema_utils import calculate_nesting_depth

        # Simple flat object
        flat_data = {"id": 1, "name": "test"}
        depth = calculate_nesting_depth(flat_data)
        assert depth == 1

        # Nested object
        nested_data = {"user": {"profile": {"details": {"name": "test"}}}}
        depth = calculate_nesting_depth(nested_data)
        assert depth == 4

        # Array with objects
        array_data = [{"nested": {"value": 1}}, {"nested": {"value": 2}}]
        depth = calculate_nesting_depth(array_data)
        assert depth == 3

    def test_has_arrays_detection(self):
        """Test array detection in data structures"""
        from shedboxai.core.introspection.schema_utils import has_arrays

        # Data with arrays
        with_arrays = {"items": [1, 2, 3], "meta": {"count": 3}}
        assert has_arrays(with_arrays)

        # Data without arrays
        without_arrays = {"id": 1, "meta": {"count": 3}}
        assert not has_arrays(without_arrays)

        # Root level array
        root_array = [{"id": 1}, {"id": 2}]
        assert has_arrays(root_array)

    def test_has_objects_detection(self):
        """Test object detection in data structures"""
        from shedboxai.core.introspection.schema_utils import has_objects

        # Data with objects
        with_objects = {"user": {"name": "test"}}
        assert has_objects(with_objects)

        # Array with objects
        array_with_objects = [{"id": 1}, {"id": 2}]
        assert has_objects(array_with_objects)

        # Primitive data
        primitive_data = "simple string"
        assert not has_objects(primitive_data)

    def test_token_count_estimation(self):
        """Test token count estimation"""
        simple_data = {"message": "hello world"}
        tokens = self.analyzer._estimate_api_token_count(simple_data)
        assert tokens > 0

        # Larger data should have more tokens
        large_data = {"items": [{"id": i, "name": f"item_{i}"} for i in range(100)]}
        large_tokens = self.analyzer._estimate_api_token_count(large_data)
        assert large_tokens > tokens

    def test_sample_data_generation(self):
        """Test sample data generation from API responses"""
        # Array response
        array_data = [
            {"id": 1, "name": "Item 1"},
            {"id": 2, "name": "Item 2"},
            {"id": 3, "name": "Item 3"},
        ]
        samples = self.analyzer._generate_api_sample_data(array_data, 2)
        assert len(samples) == 2
        assert samples[0]["id"] == 1
        assert samples[1]["id"] == 2

        # Object response
        object_data = {"id": 1, "name": "Single Item"}
        samples = self.analyzer._generate_api_sample_data(object_data, 5)
        assert len(samples) == 1
        assert samples[0] == object_data

        # Primitive response
        primitive_data = "simple string"
        samples = self.analyzer._generate_api_sample_data(primitive_data, 5)
        assert len(samples) == 1
        assert samples[0]["value"] == "simple string"

    def test_size_analysis(self):
        """Test API response size analysis"""
        mock_connector = MagicMock()
        self.analyzer.connector = mock_connector

        large_response = {"data": [{"id": i, "value": f"item_{i}"} for i in range(1000)]}
        mock_connector.get_data.return_value = {"large_api": large_response}

        config = {
            "name": "large_api",
            "type": "rest",
            "url": "https://api.example.com/large",
            "response_path": "data",
        }

        result = self.analyzer.analyze(config)

        assert result.success
        assert result.size_info.record_count == 1000
        assert result.size_info.estimated_tokens > 0
        assert result.size_info.memory_size_mb > 0

    def test_error_classification(self):
        """Test API error classification"""
        # Authentication error
        auth_error = Exception("Authentication failed")
        error_type = self.analyzer._classify_api_error(auth_error)
        assert error_type == "auth"

        # Network error
        network_error = Exception("Connection failed")
        error_type = self.analyzer._classify_api_error(network_error)
        assert error_type == "network"

        # Timeout error
        timeout_error = Exception("Request timeout occurred")
        error_type = self.analyzer._classify_api_error(timeout_error)
        assert error_type == "timeout"

        # Rate limit error
        rate_error = Exception("Rate limit exceeded")
        error_type = self.analyzer._classify_api_error(rate_error)
        assert error_type == "rate_limit"

        # JSON parsing error
        json_error = Exception("Invalid JSON response")
        error_type = self.analyzer._classify_api_error(json_error)
        assert error_type == "parsing"

        # Generic error
        generic_error = Exception("Something went wrong")
        error_type = self.analyzer._classify_api_error(generic_error)
        assert error_type == "api_error"

    def test_error_hint_generation(self):
        """Test error hint generation"""
        config = {"url": "https://api.example.com/data"}

        # Authentication error hint
        auth_error = Exception("Authentication failed")
        hint = self.analyzer._generate_api_error_hint(auth_error, config)
        assert "credentials" in hint.lower()

        # Network error hint
        network_error = Exception("Connection failed")
        hint = self.analyzer._generate_api_error_hint(network_error, config)
        assert config["url"] in hint

        # Timeout error hint
        timeout_error = Exception("Request timeout occurred")
        hint = self.analyzer._generate_api_error_hint(timeout_error, config)
        assert "timeout" in hint.lower()

    def test_llm_recommendations(self):
        """Test LLM recommendation generation"""
        mock_connector = MagicMock()
        self.analyzer.connector = mock_connector

        api_response = {
            "data": [
                {"id": 1, "name": "Item 1", "category": "A"},
                {"id": 2, "name": "Item 2", "category": "B"},
            ],
            "pagination": {"total": 100, "page": 1, "per_page": 2},
        }

        mock_connector.get_data.return_value = {"test_api": api_response}

        config = {
            "name": "test_api",
            "type": "rest",
            "url": "https://api.example.com/data",
            "response_path": "data",
        }

        result = self.analyzer.analyze(config)

        assert result.success
        recommendations = " ".join(result.llm_recommendations).lower()

        # Should recommend response_path usage
        assert "response_path" in recommendations

        # Should suggest field extraction for object arrays
        assert "format_conversion" in recommendations

        # Should mention pagination
        processing_notes = " ".join(result.processing_notes).lower()
        assert "pagination" in processing_notes

    def test_oauth_token_handling(self):
        """Test OAuth token flow handling"""
        mock_connector = MagicMock()
        self.analyzer.connector = mock_connector

        # Mock successful response
        mock_connector.get_data.return_value = {"protected_api": {"data": [{"id": 1}]}}

        config = {
            "name": "protected_api",
            "type": "rest",
            "url": "https://api.example.com/protected",
            "requires_token": True,
            "token_source": "auth_endpoint",
            "_token_sources": {
                "auth_endpoint": {
                    "type": "rest",
                    "url": "https://api.example.com/token",
                    "is_token_source": True,
                    "token_for": ["protected_api"],
                }
            },
        }

        result = self.analyzer.analyze(config)

        # Should handle token sources in the connector config
        assert mock_connector.get_data.called
        call_args = mock_connector.get_data.call_args[0][0]
        assert "auth_endpoint" in call_args
        assert "protected_api" in call_args
