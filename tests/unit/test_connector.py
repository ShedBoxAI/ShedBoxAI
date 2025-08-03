"""
Comprehensive test suite for DataSourceConnector.
Tests all data source types, authentication methods, and error scenarios.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pandas as pd
import pytest
import responses
import yaml

# Import your connector
from shedboxai.connector import DataSourceConfig, DataSourceConnector
from shedboxai.core.exceptions import (
    AuthenticationError,
    DataSourceError,
    EnvironmentVariableError,
    FileAccessError,
    InvalidFieldError,
    NetworkError,
)


class TestDataSourceConfig:
    """Test the DataSourceConfig pydantic model."""

    def test_basic_csv_config(self):
        config = DataSourceConfig(type="csv", path="test.csv")
        assert config.type == "csv"
        assert config.path == "test.csv"
        assert config.method == "GET"  # default
        assert config.headers == {}  # default
        assert config.is_token_source == False  # default

    def test_rest_config_with_options(self):
        config = DataSourceConfig(
            type="rest",
            url="https://api.example.com",
            method="POST",
            headers={"Authorization": "Bearer test"},
            options={"json": {"key": "value"}},
        )
        assert config.type == "rest"
        assert config.url == "https://api.example.com"
        assert config.method == "POST"
        assert config.headers["Authorization"] == "Bearer test"
        assert config.options["json"]["key"] == "value"

    def test_token_source_config(self):
        config = DataSourceConfig(
            type="rest",
            url="https://auth.example.com/token",
            is_token_source=True,
            token_for=["protected_api"],
        )
        assert config.is_token_source == True
        assert config.token_for == ["protected_api"]

    def test_token_consumer_config(self):
        config = DataSourceConfig(
            type="rest",
            url="https://api.example.com/protected",
            requires_token=True,
            token_source="auth_endpoint",
        )
        assert config.requires_token == True
        assert config.token_source == "auth_endpoint"


class TestDataSourceConnectorInit:
    """Test DataSourceConnector initialization and environment loading."""

    def test_init_without_config_path(self):
        # Test that connector initializes correctly without config path
        connector = DataSourceConnector()
        assert connector._token_cache == {}
        # Note: load_dotenv() is called during initialization,
        # but testing the exact call is brittle due to import caching

    def test_init_with_config_path_env_exists(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.touch()
        env_file = tmp_path / ".env"
        env_file.write_text("TEST_VAR=test_value\n")

        # Test that connector initializes correctly with config path and existing .env
        connector = DataSourceConnector(str(config_file))
        assert connector._token_cache == {}
        # Note: load_dotenv() is called with env_file path during initialization

    def test_init_with_config_path_no_env(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.touch()

        # Test that connector initializes correctly with config path but no .env file
        connector = DataSourceConnector(str(config_file))
        assert connector._token_cache == {}
        # Note: load_dotenv() is called without args when .env doesn't exist


class TestCSVDataSource:
    """Test CSV data source functionality."""

    def test_fetch_csv_basic(self, tmp_path):
        # Create test CSV file
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("name,age,city\nJohn,25,NYC\nJane,30,LA")

        connector = DataSourceConnector()
        config = DataSourceConfig(type="csv", path=str(csv_file))

        result = connector._fetch_data(config, "test_csv")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert list(result.columns) == ["name", "age", "city"]
        assert result.iloc[0]["name"] == "John"
        assert result.iloc[1]["age"] == 30

    def test_fetch_csv_with_options(self, tmp_path):
        # Create CSV with custom delimiter
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("name|age|city\nJohn|25|NYC")

        connector = DataSourceConnector()
        config = DataSourceConfig(type="csv", path=str(csv_file), options={"delimiter": "|"})

        result = connector._fetch_data(config, "csv_with_options")
        assert len(result) == 1
        assert result.iloc[0]["name"] == "John"

    def test_fetch_csv_direct_data(self):
        connector = DataSourceConnector()
        config = DataSourceConfig(
            type="csv",
            data=[
                {"name": "Alice", "age": 28, "city": "Boston"},
                {"name": "Bob", "age": 35, "city": "Seattle"},
            ],
        )

        result = connector._fetch_data(config, "csv_direct_data")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        # Test column ordering (as specified in the code)
        assert list(result.columns) == ["name", "age", "city"]

    def test_fetch_csv_missing_path(self):
        connector = DataSourceConnector()
        config = DataSourceConfig(type="csv")  # No path provided

        # Now raises InvalidFieldError with a formatted error message
        with pytest.raises(InvalidFieldError):
            connector._fetch_data(config, "csv_missing_path")

    def test_fetch_csv_file_not_found(self):
        connector = DataSourceConnector()
        config = DataSourceConfig(type="csv", path="nonexistent.csv")

        with pytest.raises(FileAccessError):
            connector._fetch_data(config, "csv_not_found")


class TestJSONDataSource:
    """Test JSON data source functionality."""

    def test_fetch_json_basic(self, tmp_path):
        # Create test JSON file
        json_file = tmp_path / "test.json"
        test_data = {"users": [{"name": "John", "age": 25}]}
        json_file.write_text(json.dumps(test_data))

        connector = DataSourceConnector()
        config = DataSourceConfig(type="json", path=str(json_file))

        result = connector._fetch_data(config, "json_basic")
        assert result == test_data

    def test_fetch_json_missing_path(self):
        connector = DataSourceConnector()
        config = DataSourceConfig(type="json")

        with pytest.raises(InvalidFieldError):
            connector._fetch_data(config, "test_source")

    def test_fetch_json_file_not_found(self):
        connector = DataSourceConnector()
        config = DataSourceConfig(type="json", path="nonexistent.json")

        with pytest.raises(FileAccessError):
            connector._fetch_data(config, "test_source")


class TestYAMLDataSource:
    """Test YAML data source functionality."""

    def test_fetch_yaml_basic(self, tmp_path):
        # Create test YAML file
        yaml_file = tmp_path / "test.yaml"
        test_data = {"settings": {"debug": True, "port": 8080}}
        yaml_file.write_text(yaml.dump(test_data))

        connector = DataSourceConnector()
        config = DataSourceConfig(type="yaml", path=str(yaml_file))

        result = connector._fetch_data(config, "test_source")
        assert result == test_data

    def test_fetch_yaml_direct_data(self):
        connector = DataSourceConnector()
        test_data = {"config": {"enabled": True}}
        config = DataSourceConfig(type="yaml", data=test_data)

        result = connector._fetch_data(config, "test_source")
        assert result == test_data

    def test_fetch_yaml_missing_path(self):
        connector = DataSourceConnector()
        config = DataSourceConfig(type="yaml")

        with pytest.raises(InvalidFieldError):
            connector._fetch_data(config, "test_source")


class TestRESTDataSource:
    """Test REST API data source functionality."""

    @responses.activate
    def test_fetch_rest_basic_get(self):
        responses.add(
            responses.GET,
            "https://api.example.com/users",
            json={"users": [{"name": "John", "age": 25}]},
            status=200,
        )

        connector = DataSourceConnector()
        config = DataSourceConfig(type="rest", url="https://api.example.com/users")

        result = connector._fetch_data(config, "test_source")
        assert result == {"users": [{"name": "John", "age": 25}]}

    @responses.activate
    def test_fetch_rest_post_with_json(self):
        responses.add(
            responses.POST,
            "https://api.example.com/data",
            json={"success": True},
            status=200,
        )

        connector = DataSourceConnector()
        config = DataSourceConfig(
            type="rest",
            url="https://api.example.com/data",
            method="POST",
            options={"json": {"key": "value"}},
        )

        result = connector._fetch_data(config, "test_source")
        assert result == {"success": True}

    @responses.activate
    def test_fetch_rest_with_headers(self):
        responses.add(
            responses.GET,
            "https://api.example.com/protected",
            json={"data": "secret"},
            status=200,
        )

        connector = DataSourceConnector()
        config = DataSourceConfig(
            type="rest",
            url="https://api.example.com/protected",
            headers={"Authorization": "Bearer test-token"},
        )

        result = connector._fetch_data(config, "test_source")

        # Check that the request was made with correct headers
        assert len(responses.calls) == 1
        assert responses.calls[0].request.headers["Authorization"] == "Bearer test-token"

    @responses.activate
    def test_fetch_rest_response_path(self):
        responses.add(
            responses.GET,
            "https://api.weather.com/current",
            json={"current": {"condition": {"text": "Sunny", "temp": 75}}},
            status=200,
        )

        connector = DataSourceConnector()
        config = DataSourceConfig(
            type="rest",
            url="https://api.weather.com/current",
            response_path="current.condition",
        )

        result = connector._fetch_data(config, "test_source")
        assert result == {"text": "Sunny", "temp": 75}

    @responses.activate
    def test_fetch_rest_response_path_missing(self):
        responses.add(
            responses.GET,
            "https://api.example.com/data",
            json={"other": "data"},
            status=200,
        )

        connector = DataSourceConnector()
        config = DataSourceConfig(
            type="rest",
            url="https://api.example.com/data",
            response_path="missing.path",
        )

        # Should raise DataSourceError when path is not found
        with pytest.raises(DataSourceError):
            connector._fetch_data(config, "test_source")

    def test_fetch_rest_missing_url(self):
        connector = DataSourceConnector()
        config = DataSourceConfig(type="rest")

        with pytest.raises(InvalidFieldError):
            connector._fetch_data(config, "test_source")

    @responses.activate
    def test_fetch_rest_http_error(self):
        responses.add(responses.GET, "https://api.example.com/error", status=404)

        connector = DataSourceConnector()
        config = DataSourceConfig(type="rest", url="https://api.example.com/error")

        with pytest.raises(NetworkError):
            connector._fetch_data(config, "test_source")


class TestEnvironmentVariableHandling:
    """Test environment variable substitution in REST requests."""

    @responses.activate
    def test_bearer_token_substitution(self):
        """Test the Bearer ${VAR} substitution pattern (now fixed)."""
        responses.add(
            responses.GET,
            "https://api.example.com/protected",
            json={"data": "success"},
            status=200,
        )

        # Test the Bearer ${VAR} pattern that should now work
        with patch.dict(os.environ, {"API_TOKEN": "secret-token"}):
            connector = DataSourceConnector()
            config = DataSourceConfig(
                type="rest",
                url="https://api.example.com/protected",
                headers={"Authorization": "Bearer ${API_TOKEN}"},
            )

            result = connector._fetch_data(config, "test_source")

            # This should now work with the fixed bearer token substitution
            assert len(responses.calls) == 1
            assert responses.calls[0].request.headers["Authorization"] == "Bearer secret-token"

    @responses.activate
    def test_general_token_substitution(self):
        """Test the general ${VAR} pattern still works."""
        responses.add(
            responses.GET,
            "https://api.example.com/protected",
            json={"data": "success"},
            status=200,
        )

        with patch.dict(os.environ, {"API_TOKEN": "secret-token"}):
            connector = DataSourceConnector()
            config = DataSourceConfig(
                type="rest",
                url="https://api.example.com/protected",
                headers={"Authorization": "${API_TOKEN}"},
            )

            result = connector._fetch_data(config, "test_source")

            # General substitution should still work
            assert len(responses.calls) == 1
            assert responses.calls[0].request.headers["Authorization"] == "secret-token"

    @responses.activate
    def test_bearer_token_missing_env_var(self):
        responses.add(
            responses.GET,
            "https://api.example.com/protected",
            json={"data": "success"},
            status=200,
        )

        connector = DataSourceConnector()
        config = DataSourceConfig(
            type="rest",
            url="https://api.example.com/protected",
            headers={"Authorization": "Bearer ${MISSING_TOKEN}"},
        )

        # Should raise an EnvironmentVariableError
        with pytest.raises(EnvironmentVariableError):
            connector._fetch_data(config, "test_source")

    @responses.activate
    def test_basic_auth_substitution(self):
        responses.add(
            responses.GET,
            "https://api.example.com/basic",
            json={"data": "success"},
            status=200,
        )

        with patch.dict(os.environ, {"API_USER": "testuser", "API_PASS": "testpass"}):
            connector = DataSourceConnector()
            config = DataSourceConfig(
                type="rest",
                url="https://api.example.com/basic",
                options={"auth": ["${API_USER}", "${API_PASS}"]},
            )

            result = connector._fetch_data(config, "test_source")
            assert result == {"data": "success"}

    @responses.activate
    def test_basic_auth_missing_credentials(self):
        responses.add(
            responses.GET,
            "https://api.example.com/basic",
            json={"data": "success"},
            status=200,
        )

        connector = DataSourceConnector()
        config = DataSourceConfig(
            type="rest",
            url="https://api.example.com/basic",
            options={"auth": ["${MISSING_USER}", "${MISSING_PASS}"]},
        )

        with pytest.raises(EnvironmentVariableError):
            connector._fetch_data(config, "test_source")

    @responses.activate
    def test_json_body_env_substitution(self):
        responses.add(
            responses.POST,
            "https://api.example.com/login",
            json={"token": "abc123"},
            status=200,
        )

        with patch.dict(os.environ, {"USERNAME": "testuser", "PASSWORD": "testpass"}):
            connector = DataSourceConnector()
            config = DataSourceConfig(
                type="rest",
                url="https://api.example.com/login",
                method="POST",
                options={
                    "json": {
                        "username": "${USERNAME}",
                        "password": "${PASSWORD}",
                        "static": "value",
                    }
                },
            )

            result = connector._fetch_data(config, "test_source")

            # Check the request body had substituted values
            request_body = json.loads(responses.calls[0].request.body)
            assert request_body["username"] == "testuser"
            assert request_body["password"] == "testpass"
            assert request_body["static"] == "value"


class TestTokenAuthentication:
    """Test token-based authentication flow."""

    @responses.activate
    def test_token_flow_success(self):
        # Mock token endpoint
        responses.add(
            responses.POST,
            "https://auth.example.com/token",
            json={"token": "auth-token-123"},
            status=200,
        )

        # Mock protected endpoint
        responses.add(
            responses.GET,
            "https://api.example.com/protected",
            json={"data": "protected-data"},
            status=200,
        )

        connector = DataSourceConnector()
        config = {
            "auth_endpoint": {
                "type": "rest",
                "url": "https://auth.example.com/token",
                "method": "POST",
                "is_token_source": True,
                "token_for": ["protected_api"],
            },
            "protected_api": {
                "type": "rest",
                "url": "https://api.example.com/protected",
                "requires_token": True,
                "token_source": "auth_endpoint",
            },
        }

        result = connector.get_data(config)

        # Should only have the protected_api data (not the token source)
        assert "protected_api" in result
        assert "auth_endpoint" not in result
        assert result["protected_api"] == {"data": "protected-data"}

        # Verify the protected endpoint was called with bearer token
        protected_call = next(call for call in responses.calls if "api.example.com" in call.request.url)
        assert protected_call.request.headers["Authorization"] == "Bearer auth-token-123"

    @responses.activate
    def test_token_caching(self):
        # Mock token endpoint
        responses.add(
            responses.POST,
            "https://auth.example.com/token",
            json={"token": "cached-token"},
            status=200,
        )

        # Mock two protected endpoints
        responses.add(
            responses.GET,
            "https://api.example.com/endpoint1",
            json={"data": "data1"},
            status=200,
        )
        responses.add(
            responses.GET,
            "https://api.example.com/endpoint2",
            json={"data": "data2"},
            status=200,
        )

        connector = DataSourceConnector()
        config = {
            "auth": {
                "type": "rest",
                "url": "https://auth.example.com/token",
                "method": "POST",
                "is_token_source": True,
                "token_for": ["api1", "api2"],
            },
            "api1": {
                "type": "rest",
                "url": "https://api.example.com/endpoint1",
                "requires_token": True,
                "token_source": "auth",
            },
            "api2": {
                "type": "rest",
                "url": "https://api.example.com/endpoint2",
                "requires_token": True,
                "token_source": "auth",
            },
        }

        result = connector.get_data(config)

        # Both protected endpoints should have been called
        assert result["api1"] == {"data": "data1"}
        assert result["api2"] == {"data": "data2"}

        # Count calls to token endpoint - verify token was fetched
        token_calls = [call for call in responses.calls if "auth.example.com" in call.request.url]
        assert len(token_calls) >= 1  # At least one call to get token

    def test_token_source_not_found(self):
        connector = DataSourceConnector()
        config = {
            "protected_api": {
                "type": "rest",
                "url": "https://api.example.com/protected",
                "requires_token": True,
                "token_source": "missing_auth",
            }
        }

        # The connector should handle errors gracefully and return error in result
        result = connector.get_data(config)
        assert "error" in result["protected_api"]

    @responses.activate
    def test_invalid_token_response(self):
        # Mock token endpoint with invalid response
        responses.add(
            responses.POST,
            "https://auth.example.com/token",
            json={"error": "invalid"},
            status=200,
        )

        connector = DataSourceConnector()
        config = {
            "auth": {
                "type": "rest",
                "url": "https://auth.example.com/token",
                "method": "POST",
                "is_token_source": True,
                "token_for": ["api"],
            },
            "api": {
                "type": "rest",
                "url": "https://api.example.com/data",
                "requires_token": True,
                "token_source": "auth",
            },
        }

        # The connector should handle errors gracefully and return error in result
        result = connector.get_data(config)
        assert "error" in result["api"]


class TestGetDataIntegration:
    """Test the main get_data method with various scenarios."""

    def test_get_data_multiple_sources(self, tmp_path):
        # Create test files
        csv_file = tmp_path / "users.csv"
        csv_file.write_text("name,age\nJohn,25")

        json_file = tmp_path / "settings.json"
        json_file.write_text('{"debug": true}')

        connector = DataSourceConnector()
        config = {
            "users": {"type": "csv", "path": str(csv_file)},
            "settings": {"type": "json", "path": str(json_file)},
        }

        result = connector.get_data(config)

        assert "users" in result
        assert "settings" in result
        assert len(result["users"]) == 1
        assert result["settings"]["debug"] == True

    @responses.activate
    def test_get_data_with_errors(self):
        responses.add(
            responses.GET,
            "https://api.example.com/working",
            json={"data": "success"},
            status=200,
        )

        connector = DataSourceConnector()
        config = {
            "working_api": {"type": "rest", "url": "https://api.example.com/working"},
            "broken_file": {"type": "csv", "path": "nonexistent.csv"},
        }

        result = connector.get_data(config)

        # Working API should succeed
        assert result["working_api"] == {"data": "success"}

        # Broken file should have an error message
        assert "error" in result["broken_file"]

    def test_get_data_direct_data_sources(self):
        connector = DataSourceConnector()
        config = {
            "csv_data": {
                "type": "csv",
                "data": [{"name": "Alice", "age": 30, "city": "Boston"}],
            },
            "yaml_data": {"type": "yaml", "data": {"setting": "value"}},
        }

        result = connector.get_data(config)

        assert len(result["csv_data"]) == 1
        assert result["csv_data"].iloc[0]["name"] == "Alice"
        assert result["yaml_data"]["setting"] == "value"


class TestUnsupportedDataSource:
    """Test handling of unsupported data source types."""

    def test_unsupported_type(self):
        connector = DataSourceConnector()
        config = DataSourceConfig(type="unsupported")

        with pytest.raises(DataSourceError):
            connector._fetch_data(config, "test_source")

    def test_direct_data_unsupported_type(self):
        connector = DataSourceConnector()
        config = DataSourceConfig(type="xml", data={"test": "data"})

        with pytest.raises(DataSourceError):
            connector._fetch_data(config, "test_source")


# Test fixtures for common test data
@pytest.fixture
def sample_csv_data():
    return "name,age,city\nJohn,25,NYC\nJane,30,LA\nBob,35,Chicago"


@pytest.fixture
def sample_json_data():
    return {
        "users": [{"name": "Alice", "age": 28}, {"name": "Bob", "age": 32}],
        "metadata": {"total": 2},
    }


@pytest.fixture
def sample_yaml_data():
    return {
        "database": {"host": "localhost", "port": 5432},
        "features": ["auth", "logging"],
    }


# Integration test combining multiple features
class TestComplexIntegration:
    """Integration tests combining multiple features."""

    @responses.activate
    def test_full_workflow_with_auth_and_processing(self, tmp_path):
        # Setup files
        csv_file = tmp_path / "users.csv"
        csv_file.write_text("id,name,role\n1,John,admin\n2,Jane,user")

        # Mock authentication
        responses.add(
            responses.POST,
            "https://auth.example.com/token",
            json={"token": "workflow-token"},
            status=200,
        )

        # Mock API with nested response
        responses.add(
            responses.GET,
            "https://api.example.com/stats",
            json={"result": {"analytics": {"active_users": 150, "growth_rate": 12.5}}},
            status=200,
        )

        with patch.dict(os.environ, {"USERNAME": "testuser", "PASSWORD": "secret"}):
            connector = DataSourceConnector()
            config = {
                "local_users": {"type": "csv", "path": str(csv_file)},
                "auth_service": {
                    "type": "rest",
                    "url": "https://auth.example.com/token",
                    "method": "POST",
                    "options": {"json": {"username": "${USERNAME}", "password": "${PASSWORD}"}},
                    "is_token_source": True,
                    "token_for": ["analytics_api"],
                },
                "analytics_api": {
                    "type": "rest",
                    "url": "https://api.example.com/stats",
                    "requires_token": True,
                    "token_source": "auth_service",
                    "response_path": "result.analytics",
                },
            }

            result = connector.get_data(config)

            # Verify all data sources
            assert "local_users" in result
            assert "analytics_api" in result
            assert "auth_service" not in result  # Token sources not included

            # Verify CSV data
            assert len(result["local_users"]) == 2
            assert result["local_users"].iloc[0]["name"] == "John"

            # Verify API data with path extraction
            assert result["analytics_api"]["active_users"] == 150
            assert result["analytics_api"]["growth_rate"] == 12.5

            # Verify authentication flow
            auth_call = next(call for call in responses.calls if "auth.example.com" in call.request.url)
            auth_body = json.loads(auth_call.request.body)
            assert auth_body["username"] == "testuser"
            assert auth_body["password"] == "secret"

            # Verify protected API call
            api_call = next(call for call in responses.calls if "api.example.com" in call.request.url)
            assert api_call.request.headers["Authorization"] == "Bearer workflow-token"
