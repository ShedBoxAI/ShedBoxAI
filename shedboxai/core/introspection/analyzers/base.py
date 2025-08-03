"""
Base classes for data source analyzers.

This module provides the abstract base class and utility classes that all
specific analyzers (CSV, REST, JSON, etc.) inherit from.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict

from ..models import AnalysisStatus, SourceAnalysis, SourceType


class SourceAnalyzer(ABC):
    """Base class for all data source analyzers"""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def analyze(self, source_config: Dict[str, Any], sample_size: int = 100) -> SourceAnalysis:
        """
        Analyze a data source and return analysis results.

        Args:
            source_config: Configuration dictionary for the data source
            sample_size: Number of records to sample for analysis

        Returns:
            SourceAnalysis: Complete analysis results
        """

    @property
    @abstractmethod
    def supported_type(self) -> SourceType:
        """Return the SourceType this analyzer supports"""

    def _create_base_analysis(self, name: str, source_type: SourceType) -> SourceAnalysis:
        """Create a base SourceAnalysis object with common fields"""
        return SourceAnalysis(
            name=name,
            type=source_type,
            status=AnalysisStatus.SUCCESS,  # Will be updated if errors occur
        )

    def _measure_analysis_time(self, analysis_func):
        """Decorator to measure analysis time"""

        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = analysis_func(*args, **kwargs)
                end_time = time.time()
                result.analysis_duration_ms = (end_time - start_time) * 1000
                return result
            except Exception as e:
                end_time = time.time()
                # Create error analysis result
                error_result = self._create_base_analysis("unknown", self.supported_type)
                error_result.status = AnalysisStatus.FAILED
                error_result.error_type = self._classify_error(e)

                # Generate user-friendly error messages
                if error_result.error_type == "recursion_depth":
                    error_result.error_message = (
                        "Data structure analysis failed due to excessive nesting or" "circular references"
                    )
                elif error_result.error_type == "timeout":
                    error_result.error_message = "Request timed out while analyzing data source"
                elif error_result.error_type == "memory":
                    error_result.error_message = "Insufficient memory to analyze large dataset"
                elif error_result.error_type == "network":
                    error_result.error_message = "Network connection failed while accessing data source"
                elif error_result.error_type == "auth":
                    error_result.error_message = "Authentication failed - check credentials and permissions"
                else:
                    error_result.error_message = str(e)

                error_result.analysis_duration_ms = (end_time - start_time) * 1000
                return error_result

        return wrapper

    def _safe_analyze(self, source_config: Dict[str, Any], sample_size: int = 100) -> SourceAnalysis:
        """
        Safely analyze a source with error handling.
        This method wraps the analyze() call with timing and error handling.
        """
        start_time = time.time()
        source_name = source_config.get("name", "unknown")

        try:
            self.logger.info(f"Starting analysis of {source_name} ({self.supported_type.value})")
            result = self.analyze(source_config, sample_size)
            end_time = time.time()

            result.analysis_duration_ms = (end_time - start_time) * 1000
            self.logger.info(f"Completed analysis of {source_name} in {result.analysis_duration_ms:.1f}ms")
            return result

        except Exception as e:
            end_time = time.time()
            self.logger.error(f"Analysis failed for {source_name}: {str(e)}")

            # Check if this is already a properly formatted SourceAnalysis error from
            # child analyzer
            if isinstance(e, Exception) and hasattr(e, "__dict__"):
                # Try to call the child analyzer's analyze method to get proper error
                # handling
                try:
                    error_result = self.analyze(source_config, sample_size)
                    error_result.analysis_duration_ms = (end_time - start_time) * 1000
                    return error_result
                except Exception:
                    pass  # Fall through to generic error handling

            # Create generic error result only as fallback
            error_result = self._create_base_analysis(source_name, self.supported_type)
            error_result.status = AnalysisStatus.FAILED
            error_result.error_type = self._classify_error(e)
            error_result.error_hint = self._generate_error_hint(e, source_config)

            # Generate user-friendly error messages based on actual error content
            error_str = str(e).lower()
            if "environment" in error_str and "variable" in error_str:
                error_result.error_message = "Missing required environment variables"
            elif "connection" in error_str or "network" in error_str:
                error_result.error_message = "Cannot connect to data source"
            elif "authentication" in error_str or "unauthorized" in error_str:
                error_result.error_message = "Authentication failed"
            elif "file not found" in error_str or "no such file" in error_str:
                error_result.error_message = "File not found"
            elif "permission denied" in error_str:
                error_result.error_message = "Permission denied accessing file"
            elif "recursion" in error_str or "RecursionError" in str(type(e)):
                error_result.error_message = "Data structure too complex to analyze"
            elif "timeout" in error_str:
                error_result.error_message = "Request timed out"
            elif "memory" in error_str:
                error_result.error_message = "Not enough memory to process data"
            else:
                error_result.error_message = str(e)
            error_result.analysis_duration_ms = (end_time - start_time) * 1000

            return error_result

    def _classify_error(self, error: Exception) -> str:
        """Classify the type of error that occurred"""
        error_type = type(error).__name__
        error_str = str(error).lower()

        if "FileNotFoundError" in error_type:
            return "file_not_found"
        elif "PermissionError" in error_type:
            return "permission_denied"
        elif "ConnectionError" in error_type or "HTTPError" in error_type:
            return "network"
        elif "AuthenticationError" in error_type or "unauthorized" in error_str:
            return "auth"
        elif "ParseError" in error_type or "JSONDecodeError" in error_type:
            return "parsing"
        elif "RecursionError" in error_type or "maximum recursion depth" in error_str:
            return "recursion_depth"
        elif "timeout" in error_str:
            return "timeout"
        elif "memory" in error_str or "out of memory" in error_str:
            return "memory"
        else:
            return "unknown"

    def _generate_error_hint(self, error: Exception, source_config: Dict[str, Any]) -> str:
        """Generate a helpful hint for resolving the error"""
        error_type = self._classify_error(error)

        hints = {
            "file_not_found": (f"Check that the file path exists: " f"{source_config.get('path', 'unknown')}"),
            "permission_denied": "Check file permissions and access rights",
            "network": (f"Verify URL is accessible: {source_config.get('url', 'unknown')}"),
            "auth": "Check authentication credentials and API key permissions",
            "parsing": "Verify file format and encoding are correct",
            "recursion_depth": (
                "Data structure too deeply nested or contains circular references. "
                "Try simplifying the data structure."
            ),
            "timeout": "Request timed out - check network connectivity or increase timeout settings",
            "memory": (
                "Insufficient memory to process large dataset - consider reducing " "sample size or data complexity"
            ),
            "unknown": "Check logs for more details",
        }

        return hints.get(error_type, hints["unknown"])


class FileAnalyzer(SourceAnalyzer):
    """Base class for file-based analyzers (CSV, JSON, YAML, Text)"""

    def _get_file_size_info(self, file_path: str) -> tuple:
        """Get file size information"""
        try:
            import os

            file_size_bytes = os.path.getsize(file_path)
            file_size_mb = file_size_bytes / (1024 * 1024)
            return file_size_bytes, file_size_mb
        except Exception:
            return None, None

    def _detect_encoding(self, file_path: str) -> str:
        """Detect file encoding"""
        try:
            import chardet

            with open(file_path, "rb") as f:
                raw_data = f.read(10000)  # Read first 10KB for detection
                result = chardet.detect(raw_data)
                return result.get("encoding", "utf-8")
        except Exception:
            return "utf-8"


class APIAnalyzer(SourceAnalyzer):
    """Base class for API-based analyzers (REST)"""

    def _estimate_response_size(self, response_data: Any) -> tuple:
        """Estimate response size in bytes and MB"""
        try:
            import json
            import sys

            # Serialize to JSON to estimate size
            json_str = json.dumps(response_data)
            size_bytes = len(json_str.encode("utf-8"))
            size_mb = size_bytes / (1024 * 1024)

            # Also estimate memory size
            memory_size = sys.getsizeof(response_data) / (1024 * 1024)

            return size_bytes, size_mb, memory_size
        except Exception:
            return None, None, None
