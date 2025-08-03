"""
Unit tests for base analyzer classes.

These tests validate the base functionality that all analyzers inherit,
including error handling, timing, and utility methods.
"""

from unittest.mock import Mock, patch

import pytest

from shedboxai.core.introspection.analyzers.base import APIAnalyzer, FileAnalyzer, SourceAnalyzer
from shedboxai.core.introspection.models import AnalysisStatus, SourceAnalysis, SourceType


class TestSourceAnalyzer:
    """Test SourceAnalyzer base class"""

    def test_cannot_instantiate_abstract_class(self):
        """Test that SourceAnalyzer cannot be instantiated directly"""
        with pytest.raises(TypeError):
            SourceAnalyzer()

    def test_create_base_analysis(self):
        """Test creating base analysis object"""

        # Create a concrete implementation for testing
        class TestAnalyzer(SourceAnalyzer):
            @property
            def supported_type(self):
                return SourceType.CSV

            def analyze(self, source_config, sample_size=100):
                return self._create_base_analysis("test", SourceType.CSV)

        analyzer = TestAnalyzer()
        analysis = analyzer._create_base_analysis("test_source", SourceType.CSV)

        assert analysis.name == "test_source"
        assert analysis.type == SourceType.CSV
        assert analysis.status == AnalysisStatus.SUCCESS

    def test_error_classification(self):
        """Test error classification logic"""

        class TestAnalyzer(SourceAnalyzer):
            @property
            def supported_type(self):
                return SourceType.CSV

            def analyze(self, source_config, sample_size=100):
                pass

        analyzer = TestAnalyzer()

        # Test file not found error
        file_error = FileNotFoundError("File not found")
        assert analyzer._classify_error(file_error) == "file_not_found"

        # Test permission error
        perm_error = PermissionError("Access denied")
        assert analyzer._classify_error(perm_error) == "permission_denied"

        # Test network error
        from requests.exceptions import ConnectionError

        net_error = ConnectionError("Connection failed")
        assert analyzer._classify_error(net_error) == "network"

        # Test unknown error
        unknown_error = ValueError("Some other error")
        assert analyzer._classify_error(unknown_error) == "unknown"

    def test_error_hint_generation(self):
        """Test error hint generation"""

        class TestAnalyzer(SourceAnalyzer):
            @property
            def supported_type(self):
                return SourceType.CSV

            def analyze(self, source_config, sample_size=100):
                pass

        analyzer = TestAnalyzer()

        # Test file not found hint
        config = {"path": "/path/to/file.csv"}
        file_error = FileNotFoundError("File not found")
        hint = analyzer._generate_error_hint(file_error, config)
        assert "/path/to/file.csv" in hint
        assert "Check that the file path exists" in hint

        # Test network error hint
        config = {"url": "https://api.example.com"}
        net_error = ConnectionError("Connection failed")
        hint = analyzer._generate_error_hint(net_error, config)
        assert "https://api.example.com" in hint
        assert "Verify URL is accessible" in hint

    def test_safe_analyze_success(self):
        """Test successful analysis with timing"""

        class TestAnalyzer(SourceAnalyzer):
            @property
            def supported_type(self):
                return SourceType.CSV

            def analyze(self, source_config, sample_size=100):
                analysis = self._create_base_analysis("test", SourceType.CSV)
                analysis.llm_recommendations.append("Test recommendation")
                return analysis

        analyzer = TestAnalyzer()
        config = {"name": "test_source", "path": "/test.csv"}

        result = analyzer._safe_analyze(config)

        assert result.success is True
        assert result.name == "test"
        assert result.analysis_duration_ms is not None
        assert result.analysis_duration_ms >= 0
        assert "Test recommendation" in result.llm_recommendations

    def test_safe_analyze_failure(self):
        """Test failed analysis with error handling"""

        class TestAnalyzer(SourceAnalyzer):
            @property
            def supported_type(self):
                return SourceType.CSV

            def analyze(self, source_config, sample_size=100):
                raise FileNotFoundError("File not found")

        analyzer = TestAnalyzer()
        config = {"name": "test_source", "path": "/nonexistent.csv"}

        result = analyzer._safe_analyze(config)

        assert result.success is False
        assert result.status == AnalysisStatus.FAILED
        assert result.error_message == "File not found"
        assert result.error_type == "file_not_found"
        assert "/nonexistent.csv" in result.error_hint
        assert result.analysis_duration_ms is not None


class TestFileAnalyzer:
    """Test FileAnalyzer base class"""

    def test_cannot_instantiate_abstract_class(self):
        """Test that FileAnalyzer cannot be instantiated directly"""
        with pytest.raises(TypeError):
            FileAnalyzer()

    @patch("os.path.getsize")
    def test_get_file_size_info(self, mock_getsize):
        """Test file size calculation"""

        # Create a concrete implementation
        class TestFileAnalyzer(FileAnalyzer):
            @property
            def supported_type(self):
                return SourceType.CSV

            def analyze(self, source_config, sample_size=100):
                pass

        analyzer = TestFileAnalyzer()

        # Mock file size of 1MB (1024 * 1024 bytes)
        mock_getsize.return_value = 1024 * 1024

        size_bytes, size_mb = analyzer._get_file_size_info("/test.csv")

        assert size_bytes == 1024 * 1024
        assert size_mb == 1.0
        mock_getsize.assert_called_once_with("/test.csv")

    @patch("os.path.getsize")
    def test_get_file_size_info_error(self, mock_getsize):
        """Test file size calculation with error"""

        class TestFileAnalyzer(FileAnalyzer):
            @property
            def supported_type(self):
                return SourceType.CSV

            def analyze(self, source_config, sample_size=100):
                pass

        analyzer = TestFileAnalyzer()

        # Mock file not found error
        mock_getsize.side_effect = FileNotFoundError("File not found")

        size_bytes, size_mb = analyzer._get_file_size_info("/nonexistent.csv")

        assert size_bytes is None
        assert size_mb is None

    @patch("chardet.detect")
    @patch("builtins.open")
    def test_detect_encoding(self, mock_open, mock_detect):
        """Test encoding detection"""

        class TestFileAnalyzer(FileAnalyzer):
            @property
            def supported_type(self):
                return SourceType.CSV

            def analyze(self, source_config, sample_size=100):
                pass

        analyzer = TestFileAnalyzer()

        # Mock file reading and chardet detection
        mock_file = Mock()
        mock_file.read.return_value = b"test data"
        mock_open.return_value.__enter__.return_value = mock_file
        mock_detect.return_value = {"encoding": "utf-8"}

        encoding = analyzer._detect_encoding("/test.csv")

        assert encoding == "utf-8"
        mock_open.assert_called_once()
        mock_detect.assert_called_once()

    @patch("chardet.detect")
    @patch("builtins.open")
    def test_detect_encoding_error(self, mock_open, mock_detect):
        """Test encoding detection with error"""

        class TestFileAnalyzer(FileAnalyzer):
            @property
            def supported_type(self):
                return SourceType.CSV

            def analyze(self, source_config, sample_size=100):
                pass

        analyzer = TestFileAnalyzer()

        # Mock file error
        mock_open.side_effect = IOError("Cannot read file")

        encoding = analyzer._detect_encoding("/test.csv")

        # Should fall back to utf-8
        assert encoding == "utf-8"


class TestAPIAnalyzer:
    """Test APIAnalyzer base class"""

    def test_cannot_instantiate_abstract_class(self):
        """Test that APIAnalyzer cannot be instantiated directly"""
        with pytest.raises(TypeError):
            APIAnalyzer()

    def test_estimate_response_size(self):
        """Test response size estimation"""

        class TestAPIAnalyzer(APIAnalyzer):
            @property
            def supported_type(self):
                return SourceType.REST

            def analyze(self, source_config, sample_size=100):
                pass

        analyzer = TestAPIAnalyzer()

        # Test with simple response data
        response_data = {
            "data": [{"id": 1, "name": "test1"}, {"id": 2, "name": "test2"}],
            "total": 2,
        }

        size_bytes, size_mb, memory_size = analyzer._estimate_response_size(response_data)

        assert size_bytes is not None
        assert size_bytes > 0
        assert size_mb is not None
        assert size_mb >= 0
        assert memory_size is not None
        assert memory_size >= 0

        # Size should be reasonable for this small object
        assert size_bytes < 1000  # Less than 1KB
        assert size_mb < 0.001  # Less than 1MB

    def test_estimate_response_size_error(self):
        """Test response size estimation with error"""

        class TestAPIAnalyzer(APIAnalyzer):
            @property
            def supported_type(self):
                return SourceType.REST

            def analyze(self, source_config, sample_size=100):
                pass

        analyzer = TestAPIAnalyzer()

        # Test with non-serializable data
        class NonSerializable:
            pass

        response_data = {"obj": NonSerializable()}

        size_bytes, size_mb, memory_size = analyzer._estimate_response_size(response_data)

        # Should handle error gracefully
        assert size_bytes is None
        assert size_mb is None
        assert memory_size is None
