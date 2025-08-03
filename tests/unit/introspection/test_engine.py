"""
Comprehensive tests for the IntrospectionEngine.

Tests the main orchestrator functionality including configuration loading,
analyzer coordination, error handling, and result aggregation.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
import yaml

from shedboxai.core.introspection.engine import IntrospectionEngine
from shedboxai.core.introspection.models import (
    AnalysisStatus,
    CSVAnalysis,
    IntrospectionOptions,
    IntrospectionResult,
    Relationship,
    RESTAnalysis,
    SizeInfo,
    SourceAnalysis,
    SourceType,
)


class TestIntrospectionEngine:
    def setup_method(self):
        """Set up test fixtures"""
        # Create temporary config file
        self.config_data = {
            "data_sources": {
                "users": {"type": "csv", "path": "test_users.csv"},
                "api_data": {
                    "type": "rest",
                    "url": "https://api.example.com/data",
                    "headers": {"Authorization": "Bearer ${API_TOKEN}"},
                },
            }
        }

        self.temp_config = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        yaml.dump(self.config_data, self.temp_config)
        self.temp_config.close()

        self.options = IntrospectionOptions(config_path=self.temp_config.name)
        self.engine = IntrospectionEngine(self.temp_config.name, self.options)

    def teardown_method(self):
        """Clean up test fixtures"""
        Path(self.temp_config.name).unlink()

    def test_config_loading_success(self):
        """Test successful configuration loading and validation"""
        self.engine._load_configuration()

        assert self.engine.config is not None
        assert "data_sources" in self.engine.config
        assert len(self.engine.config["data_sources"]) == 2
        assert "users" in self.engine.config["data_sources"]
        assert "api_data" in self.engine.config["data_sources"]

    def test_config_loading_file_not_found(self):
        """Test configuration loading with missing file"""
        options = IntrospectionOptions(config_path="nonexistent.yaml")
        engine = IntrospectionEngine("nonexistent.yaml", options)

        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            engine._load_configuration()

    def test_config_loading_invalid_yaml(self):
        """Test configuration loading with invalid YAML"""
        invalid_config = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        invalid_config.write("invalid: yaml: content: {")
        invalid_config.close()

        try:
            options = IntrospectionOptions(config_path=invalid_config.name)
            engine = IntrospectionEngine(invalid_config.name, options)

            with pytest.raises(ValueError, match="Invalid YAML"):
                engine._load_configuration()
        finally:
            Path(invalid_config.name).unlink()

    def test_config_loading_empty_file(self):
        """Test configuration loading with empty file"""
        empty_config = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        empty_config.write("")
        empty_config.close()

        try:
            options = IntrospectionOptions(config_path=empty_config.name)
            engine = IntrospectionEngine(empty_config.name, options)

            with pytest.raises(ValueError, match="Configuration file is empty"):
                engine._load_configuration()
        finally:
            Path(empty_config.name).unlink()

    def test_config_loading_missing_data_sources(self):
        """Test configuration loading with missing data_sources section"""
        invalid_config_data = {"other_section": "value"}
        invalid_config = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        yaml.dump(invalid_config_data, invalid_config)
        invalid_config.close()

        try:
            options = IntrospectionOptions(config_path=invalid_config.name)
            engine = IntrospectionEngine(invalid_config.name, options)

            with pytest.raises(ValueError, match="Configuration must contain 'data_sources' section"):
                engine._load_configuration()
        finally:
            Path(invalid_config.name).unlink()

    def test_data_source_preparation_basic(self):
        """Test basic data source preparation and categorization"""
        self.engine._load_configuration()
        self.engine._prepare_data_sources()

        assert len(self.engine.data_sources) == 2
        assert "users" in self.engine.data_sources
        assert "api_data" in self.engine.data_sources

        # Check that names are added
        assert self.engine.data_sources["users"]["name"] == "users"
        assert self.engine.data_sources["api_data"]["name"] == "api_data"

    def test_data_source_preparation_missing_type(self):
        """Test data source preparation with missing type field"""
        config_data = {
            "data_sources": {
                "invalid_source": {
                    "path": "test.csv"
                    # Missing 'type' field
                }
            }
        }

        temp_config = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        yaml.dump(config_data, temp_config)
        temp_config.close()

        try:
            options = IntrospectionOptions(config_path=temp_config.name)
            engine = IntrospectionEngine(temp_config.name, options)

            engine._load_configuration()

            with pytest.raises(ValueError, match="Data source 'invalid_source' missing 'type' field"):
                engine._prepare_data_sources()
        finally:
            Path(temp_config.name).unlink()

    def test_data_source_preparation_invalid_type(self):
        """Test data source preparation with invalid type"""
        config_data = {"data_sources": {"invalid_source": {"type": "invalid_type", "path": "test.csv"}}}

        temp_config = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        yaml.dump(config_data, temp_config)
        temp_config.close()

        try:
            options = IntrospectionOptions(config_path=temp_config.name)
            engine = IntrospectionEngine(temp_config.name, options)

            engine._load_configuration()

            with pytest.raises(ValueError, match="Unsupported source type 'invalid_type'"):
                engine._prepare_data_sources()
        finally:
            Path(temp_config.name).unlink()

    def test_token_source_identification(self):
        """Test identification and validation of token sources"""
        config_with_oauth = {
            "data_sources": {
                "auth_endpoint": {
                    "type": "rest",
                    "url": "https://auth.example.com/token",
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
        }

        temp_oauth_config = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        yaml.dump(config_with_oauth, temp_oauth_config)
        temp_oauth_config.close()

        try:
            options = IntrospectionOptions(config_path=temp_oauth_config.name)
            engine = IntrospectionEngine(temp_oauth_config.name, options)

            engine._load_configuration()
            engine._prepare_data_sources()

            assert len(engine.token_sources) == 1
            assert "auth_endpoint" in engine.token_sources
            assert len(engine.data_sources) == 2

        finally:
            Path(temp_oauth_config.name).unlink()

    def test_token_dependency_validation_missing_token_source(self):
        """Test token dependency validation with missing token source"""
        config_with_missing_token = {
            "data_sources": {
                "protected_api": {
                    "type": "rest",
                    "url": "https://api.example.com/protected",
                    "requires_token": True,
                    # Missing 'token_source' field
                }
            }
        }

        temp_config = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        yaml.dump(config_with_missing_token, temp_config)
        temp_config.close()

        try:
            options = IntrospectionOptions(config_path=temp_config.name)
            engine = IntrospectionEngine(temp_config.name, options)

            engine._load_configuration()

            with pytest.raises(ValueError, match="requires token but no token_source specified"):
                engine._prepare_data_sources()
        finally:
            Path(temp_config.name).unlink()

    def test_token_dependency_validation_nonexistent_token_source(self):
        """Test token dependency validation with nonexistent token source"""
        config_with_invalid_token = {
            "data_sources": {
                "protected_api": {
                    "type": "rest",
                    "url": "https://api.example.com/protected",
                    "requires_token": True,
                    "token_source": "nonexistent_source",
                }
            }
        }

        temp_config = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        yaml.dump(config_with_invalid_token, temp_config)
        temp_config.close()

        try:
            options = IntrospectionOptions(config_path=temp_config.name)
            engine = IntrospectionEngine(temp_config.name, options)

            engine._load_configuration()

            with pytest.raises(ValueError, match="Token source 'nonexistent_source' not found"):
                engine._prepare_data_sources()
        finally:
            Path(temp_config.name).unlink()

    @patch("shedboxai.core.introspection.analyzers.CSVAnalyzer._safe_analyze")
    @patch("shedboxai.core.introspection.analyzers.RESTAnalyzer._safe_analyze")
    def test_analysis_orchestration_success(self, mock_rest_analyzer, mock_csv_analyzer):
        """Test successful orchestration of multiple analyzers"""
        # Mock successful analyses
        csv_analysis = CSVAnalysis(name="users", type=SourceType.CSV, status=AnalysisStatus.SUCCESS)

        rest_analysis = RESTAnalysis(name="api_data", type=SourceType.REST, status=AnalysisStatus.SUCCESS)

        mock_csv_analyzer.return_value = csv_analysis
        mock_rest_analyzer.return_value = rest_analysis

        # Mock relationship detector
        with patch.object(self.engine.relationship_detector, "detect_relationships") as mock_relationships:
            mock_relationships.return_value = []

            result = self.engine.run_introspection()

            assert result.success_count == 2
            assert result.failure_count == 0
            assert "users" in result.analyses
            assert "api_data" in result.analyses
            assert result.analyses["users"].success
            assert result.analyses["api_data"].success

    @patch("shedboxai.core.introspection.analyzers.CSVAnalyzer._safe_analyze")
    @patch("shedboxai.core.introspection.analyzers.RESTAnalyzer._safe_analyze")
    def test_analysis_orchestration_partial_failure(self, mock_rest_analyzer, mock_csv_analyzer):
        """Test orchestration with partial failures"""
        # Mock failed analysis for CSV
        failed_analysis = CSVAnalysis(
            name="users",
            type=SourceType.CSV,
            status=AnalysisStatus.FAILED,
            error_message="File not found",
            error_type="file_not_found",
        )

        # Mock successful analysis for REST
        success_analysis = RESTAnalysis(name="api_data", type=SourceType.REST, status=AnalysisStatus.SUCCESS)

        mock_csv_analyzer.return_value = failed_analysis
        mock_rest_analyzer.return_value = success_analysis

        with patch.object(self.engine.relationship_detector, "detect_relationships") as mock_relationships:
            mock_relationships.return_value = []

            result = self.engine.run_introspection()

            assert result.success_count == 1
            assert result.failure_count == 1
            assert not result.analyses["users"].success
            assert result.analyses["api_data"].success

    def test_retry_functionality(self):
        """Test retry functionality for specific sources"""
        self.engine._load_configuration()
        self.engine._prepare_data_sources()

        # Test retry source validation
        assert self.engine.can_retry_source("users")
        assert self.engine.can_retry_source("api_data")
        assert not self.engine.can_retry_source("nonexistent")

        # Test with retry options
        retry_options = IntrospectionOptions(config_path=self.temp_config.name, retry_sources=["users"])
        retry_engine = IntrospectionEngine(self.temp_config.name, retry_options)

        with patch.object(retry_engine.analyzers[SourceType.CSV], "_safe_analyze") as mock_analyzer:
            mock_analyzer.return_value = CSVAnalysis(name="users", type=SourceType.CSV, status=AnalysisStatus.SUCCESS)

            with patch.object(retry_engine.relationship_detector, "detect_relationships") as mock_relationships:
                mock_relationships.return_value = []

                retry_engine._load_configuration()
                retry_engine._prepare_data_sources()
                analyses = retry_engine._analyze_all_sources()

                # Should only analyze the retry source
                assert len(analyses) == 1
                assert "users" in analyses
                assert "api_data" not in analyses

    def test_relationship_integration(self):
        """Test integration with relationship detector"""
        self.engine._load_configuration()
        self.engine._prepare_data_sources()

        # Mock analyses
        analyses = {
            "users": CSVAnalysis(name="users", type=SourceType.CSV, status=AnalysisStatus.SUCCESS),
            "api_data": RESTAnalysis(name="api_data", type=SourceType.REST, status=AnalysisStatus.SUCCESS),
        }

        mock_relationship = Relationship(
            source_a="users",
            source_b="api_data",
            type="foreign_key",
            confidence=0.9,
            field_a="user_id",
            field_b="user_id",
            description="High confidence relationship",
        )

        with patch.object(self.engine.relationship_detector, "detect_relationships") as mock_detector:
            mock_detector.return_value = [mock_relationship]

            relationships = self.engine._detect_relationships(analyses)

            assert len(relationships) == 1
            assert relationships[0].confidence == 0.9
            assert relationships[0].type == "foreign_key"

    def test_global_recommendations_failed_sources(self):
        """Test global recommendation generation for failed sources"""
        analyses = {
            "failed_source": SourceAnalysis(
                name="failed_source",
                type=SourceType.REST,
                status=AnalysisStatus.FAILED,
                error_type="auth",
            ),
            "success_source": SourceAnalysis(
                name="success_source",
                type=SourceType.CSV,
                status=AnalysisStatus.SUCCESS,
            ),
        }

        result = IntrospectionResult(analyses=analyses, relationships=[])

        self.engine._add_global_recommendations(result, analyses, [])

        recommendations = " ".join(result.global_recommendations).lower()

        # Should have recommendations for failed sources
        assert "failed" in recommendations
        assert "check authentication" in recommendations

    def test_global_recommendations_large_datasets(self):
        """Test global recommendation generation for large datasets"""
        analyses = {
            "large_source": SourceAnalysis(name="large_source", type=SourceType.CSV, status=AnalysisStatus.SUCCESS)
        }

        # Add size info to trigger large dataset recommendation
        analyses["large_source"].size_info = SizeInfo(record_count=100000, is_large_dataset=True)

        result = IntrospectionResult(analyses=analyses, relationships=[])

        self.engine._add_global_recommendations(result, analyses, [])

        recommendations = " ".join(result.global_recommendations).lower()

        # Should have recommendations for large datasets
        assert "large" in recommendations
        assert "sampling" in recommendations or "aggregation" in recommendations

    def test_global_recommendations_context_warnings(self):
        """Test global recommendation generation for context window warnings"""
        analyses = {
            "context_warning_source": SourceAnalysis(
                name="context_warning_source",
                type=SourceType.JSON,
                status=AnalysisStatus.SUCCESS,
            )
        }

        # Add size info to trigger context window warning
        analyses["context_warning_source"].size_info = SizeInfo(record_count=10000, context_window_warning=True)

        result = IntrospectionResult(analyses=analyses, relationships=[])

        self.engine._add_global_recommendations(result, analyses, [])

        recommendations = " ".join(result.global_recommendations).lower()

        # Should have recommendations for context window
        assert "context window" in recommendations
        assert "contextual_filtering" in recommendations

    def test_global_recommendations_multiple_source_types(self):
        """Test global recommendation generation for multiple source types"""
        analyses = {
            "csv_source": SourceAnalysis(name="csv_source", type=SourceType.CSV, status=AnalysisStatus.SUCCESS),
            "rest_source": SourceAnalysis(name="rest_source", type=SourceType.REST, status=AnalysisStatus.SUCCESS),
            "json_source": SourceAnalysis(name="json_source", type=SourceType.JSON, status=AnalysisStatus.SUCCESS),
        }

        result = IntrospectionResult(analyses=analyses, relationships=[])

        self.engine._add_global_recommendations(result, analyses, [])

        recommendations = " ".join(result.global_recommendations).lower()

        # Should have recommendations for multiple source types
        assert "multiple data source types" in recommendations
        assert "unify processing" in recommendations

    def test_prerequisite_checking_success(self):
        """Test prerequisite checking with valid configuration"""
        # Mock environment variables
        with patch.dict("os.environ", {"API_TOKEN": "test_token"}, clear=True):
            # Create test file
            test_file = tempfile.NamedTemporaryFile(delete=False)
            test_file.close()

            try:
                # Update config to use existing file
                self.config_data["data_sources"]["users"]["path"] = test_file.name

                temp_config = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
                yaml.dump(self.config_data, temp_config)
                temp_config.close()

                options = IntrospectionOptions(config_path=temp_config.name)
                engine = IntrospectionEngine(temp_config.name, options)

                prereqs = engine.check_prerequisites()

                assert prereqs["valid"]
                assert len(prereqs["issues"]) == 0

            finally:
                Path(test_file.name).unlink()
                Path(temp_config.name).unlink()

    def test_prerequisite_checking_missing_env_vars(self):
        """Test prerequisite checking with missing environment variables"""
        # Create a fresh config without using the existing test setup
        config_with_env_vars = {
            "data_sources": {
                "test_api": {
                    "type": "rest",
                    "url": "https://api.test.com/data",
                    "headers": {"Authorization": "Bearer ${TEST_API_TOKEN}"},
                }
            }
        }

        temp_config = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        yaml.dump(config_with_env_vars, temp_config)
        temp_config.close()

        try:
            options = IntrospectionOptions(config_path=temp_config.name)
            engine = IntrospectionEngine(temp_config.name, options)

            # Use a unique environment variable name that won't conflict
            with patch.dict("os.environ", {}, clear=True):
                prereqs = engine.check_prerequisites()

                assert not prereqs["valid"]
                assert "TEST_API_TOKEN" in prereqs["missing_env_vars"]
                assert any("Missing environment variables" in issue for issue in prereqs["issues"])
        finally:
            Path(temp_config.name).unlink()

    def test_prerequisite_checking_missing_files(self):
        """Test prerequisite checking with missing files"""
        with patch.dict("os.environ", {"API_TOKEN": "test_token"}, clear=True):
            prereqs = self.engine.check_prerequisites()

            assert not prereqs["valid"]
            assert any("File not found" in issue for issue in prereqs["issues"])

    def test_analysis_summary_generation(self):
        """Test analysis summary statistics generation"""
        analyses = {
            "csv_success": CSVAnalysis(name="csv_success", type=SourceType.CSV, status=AnalysisStatus.SUCCESS),
            "rest_failed": RESTAnalysis(
                name="rest_failed",
                type=SourceType.REST,
                status=AnalysisStatus.FAILED,
                error_type="auth",
            ),
            "json_success": SourceAnalysis(name="json_success", type=SourceType.JSON, status=AnalysisStatus.SUCCESS),
        }

        summary = self.engine.get_analysis_summary(analyses)

        assert summary["total_sources"] == 3
        assert summary["successful"] == 2
        assert summary["failed"] == 1
        assert summary["success_rate"] == pytest.approx(66.67, rel=1e-2)
        assert summary["by_type"]["csv"]["successful"] == 1
        assert summary["by_type"]["rest"]["failed"] == 1
        assert summary["error_types"]["auth"] == 1

    def test_time_estimation(self):
        """Test analysis time estimation"""
        self.engine._load_configuration()
        self.engine._prepare_data_sources()

        estimated_time = self.engine.estimate_analysis_time()

        # Should be reasonable time (few seconds for 2 sources)
        assert 0.5 <= estimated_time <= 10.0

        # REST APIs should take longer than CSV
        assert estimated_time > 1.0  # Due to REST API multiplier

    def test_failed_sources_identification(self):
        """Test identification of failed sources"""
        analyses = {
            "success1": SourceAnalysis(name="success1", type=SourceType.CSV, status=AnalysisStatus.SUCCESS),
            "failed1": SourceAnalysis(name="failed1", type=SourceType.REST, status=AnalysisStatus.FAILED),
            "success2": SourceAnalysis(name="success2", type=SourceType.JSON, status=AnalysisStatus.SUCCESS),
            "failed2": SourceAnalysis(name="failed2", type=SourceType.YAML, status=AnalysisStatus.FAILED),
        }

        failed_sources = self.engine.get_failed_sources(analyses)

        assert len(failed_sources) == 2
        assert "failed1" in failed_sources
        assert "failed2" in failed_sources
        assert "success1" not in failed_sources
        assert "success2" not in failed_sources

    def test_configuration_improvement_suggestions(self):
        """Test configuration improvement suggestions"""
        # Create mock analyses with schema info
        from shedboxai.core.introspection.models import SchemaInfo

        analyses = {
            "api_with_complex_response": RESTAnalysis(
                name="api_with_complex_response",
                type=SourceType.REST,
                status=AnalysisStatus.SUCCESS,
            ),
            "large_dataset": SourceAnalysis(name="large_dataset", type=SourceType.CSV, status=AnalysisStatus.SUCCESS),
        }

        # Add schema info for API
        analyses["api_with_complex_response"].schema_info = SchemaInfo(response_paths=["data.items", "results"])

        # Add size info for large dataset
        analyses["large_dataset"].size_info = SizeInfo(record_count=100000, is_large_dataset=True)

        suggestions = self.engine.suggest_configuration_improvements(analyses)

        assert len(suggestions) >= 2
        assert any("response_path" in suggestion for suggestion in suggestions)
        assert any("sampling" in suggestion for suggestion in suggestions)

    def test_recover_from_partial_failure(self):
        """Test recovery suggestions for partial failures"""
        analyses = {
            "success1": SourceAnalysis(name="success1", type=SourceType.CSV, status=AnalysisStatus.SUCCESS),
            "success2": SourceAnalysis(name="success2", type=SourceType.JSON, status=AnalysisStatus.SUCCESS),
            "env_failure": SourceAnalysis(
                name="env_failure",
                type=SourceType.REST,
                status=AnalysisStatus.FAILED,
                error_type="missing_env_var",
            ),
            "auth_failure": SourceAnalysis(
                name="auth_failure",
                type=SourceType.REST,
                status=AnalysisStatus.FAILED,
                error_type="auth",
            ),
        }

        result = IntrospectionResult(analyses=analyses, relationships=[])

        recovered_result = self.engine.recover_from_partial_failure(result)

        recommendations = " ".join(recovered_result.global_recommendations).lower()

        # Should have specific recovery recommendations
        assert "environment variables" in recommendations
        assert "authentication" in recommendations
        assert "retry" in recommendations

    def test_full_introspection_workflow_error_handling(self):
        """Test full introspection workflow with error handling"""
        # Test with configuration that will cause errors
        with patch.object(self.engine, "_load_configuration") as mock_load:
            mock_load.side_effect = Exception("Configuration error")

            result = self.engine.run_introspection()

            assert result.success_count == 0
            assert result.failure_count == 0  # No sources were analyzed
            assert len(result.global_recommendations) > 0
            assert any("failed" in rec.lower() for rec in result.global_recommendations)

    @patch("shedboxai.core.introspection.analyzers.CSVAnalyzer._safe_analyze")
    def test_analysis_orchestration_token_source_first(self, mock_csv_analyzer):
        """Test that token sources are analyzed before regular sources"""
        # Create config with token source
        config_with_token = {
            "data_sources": {
                "token_source": {
                    "type": "rest",
                    "url": "https://auth.example.com/token",
                    "is_token_source": True,
                    "token_for": ["protected_api"],
                },
                "protected_api": {
                    "type": "rest",
                    "url": "https://api.example.com/protected",
                    "requires_token": True,
                    "token_source": "token_source",
                },
                "regular_csv": {"type": "csv", "path": "test.csv"},
            }
        }

        temp_config = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        yaml.dump(config_with_token, temp_config)
        temp_config.close()

        try:
            options = IntrospectionOptions(config_path=temp_config.name)
            engine = IntrospectionEngine(temp_config.name, options)

            # Track analysis order
            analysis_order = []

            def track_analysis(config, sample_size):
                analysis_order.append(config["name"])
                return SourceAnalysis(
                    name=config["name"],
                    type=SourceType(config["type"]),
                    status=AnalysisStatus.SUCCESS,
                )

            # Mock all analyzers to track order
            for analyzer in engine.analyzers.values():
                analyzer._safe_analyze = Mock(side_effect=track_analysis)

            with patch.object(engine.relationship_detector, "detect_relationships") as mock_rel:
                mock_rel.return_value = []

                result = engine.run_introspection()

                # Token source should be analyzed first
                assert analysis_order[0] == "token_source"
                assert "protected_api" in analysis_order[1:]
                assert "regular_csv" in analysis_order[1:]

        finally:
            Path(temp_config.name).unlink()
