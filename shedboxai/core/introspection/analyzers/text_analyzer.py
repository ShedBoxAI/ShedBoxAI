"""
Text analyzer for introspection feature.

This analyzer handles text files (logs, documents, etc.) and provides
content analysis optimized for LLM consumption.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from shedboxai.connector import DataSourceConnector

from ..models import AnalysisStatus, SchemaInfo, SizeInfo, SourceType, TextAnalysis
from .base import FileAnalyzer


class TextAnalyzer(FileAnalyzer):
    """Analyzer for text files and unstructured content"""

    def __init__(self):
        super().__init__()
        self.connector = DataSourceConnector()

    @property
    def supported_type(self) -> SourceType:
        return SourceType.TEXT

    def analyze(self, source_config: Dict[str, Any], sample_size: int = 100) -> TextAnalysis:
        """Analyze text data source"""
        source_name = source_config.get("name", "unknown")

        # Create analysis object
        analysis = TextAnalysis(name=source_name, type=SourceType.TEXT, status=AnalysisStatus.SUCCESS)

        try:
            # Use existing connector to load text data
            connector_config = {source_name: source_config}
            response_data = self.connector.get_data(connector_config)

            # Check if we got data successfully
            if source_name not in response_data:
                raise ValueError(f"No data returned for source: {source_name}")

            text_data = response_data[source_name]

            # Check for errors from connector
            if isinstance(text_data, dict) and "error" in text_data:
                analysis.status = AnalysisStatus.FAILED
                analysis.error_message = text_data["error"]
                analysis.error_type = "text_error"
                analysis.error_hint = "Check text file format and accessibility"
                return analysis

            # Ensure we have string data
            if not isinstance(text_data, str):
                text_data = str(text_data)

            # Detect content type and format
            analysis.content_type = self._detect_content_type(text_data)
            analysis.format_detected = self._detect_text_format(text_data)

            # Detect encoding (approximate)
            analysis.encoding = "utf-8"  # DataSourceConnector handles encoding

            # Extract patterns
            analysis.detected_patterns = self._detect_patterns(text_data)

            # Generate size and schema info
            analysis.size_info = self._analyze_text_size(text_data, source_config.get("path"))
            analysis.schema_info = self._analyze_text_schema(text_data)
            analysis.sample_data = self._generate_text_sample_data(text_data, sample_size)

            # Add recommendations
            self._add_text_recommendations(analysis, text_data)

        except Exception as e:
            self.logger.error(f"Text analysis failed for {source_name}: {str(e)}")
            analysis.status = AnalysisStatus.FAILED
            analysis.error_message = str(e)
            analysis.error_type = self._classify_text_error(e)
            analysis.error_hint = self._generate_text_error_hint(e, source_config)

        return analysis

    def _detect_content_type(self, text_data: str) -> str:
        """Detect the type of content in the text"""
        text_lower = text_data.lower()

        # Log file patterns
        if any(pattern in text_lower for pattern in ["error", "warning", "info", "debug", "trace"]):
            if re.search(r"\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}", text_data):
                return "log"

        # Configuration file patterns
        if any(pattern in text_lower for pattern in ["config", "setting", "option", "="]):
            return "config"

        # Code file patterns
        if any(pattern in text_lower for pattern in ["function", "class", "import", "def ", "var "]):
            return "code"

        # Data file patterns
        if re.search(r"\d+[,\t]\d+", text_data):  # Comma or tab separated numbers
            return "data"

        # Documentation patterns
        if any(pattern in text_lower for pattern in ["# ", "## ", "### ", "* ", "- "]):
            return "documentation"

        return "text"

    def _detect_text_format(self, text_data: str) -> str:
        """Detect specific text format"""
        # Check for structured log formats
        if re.search(r"\[\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}", text_data):
            return "structured_log"

        if re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", text_data):
            return "timestamp_log"

        # Check for CSV-like format
        if re.search(r"^[^,\n]+,[^,\n]+,[^,\n]+", text_data, re.MULTILINE):
            return "csv_like"

        # Check for TSV-like format
        if re.search(r"^[^\t\n]+\t[^\t\n]+\t[^\t\n]+", text_data, re.MULTILINE):
            return "tsv_like"

        # Check for key-value pairs
        if re.search(r"^\w+\s*[=:]\s*.+", text_data, re.MULTILINE):
            return "key_value"

        # Check for JSON-like content
        if "{" in text_data and "}" in text_data and '"' in text_data:
            return "json_like"

        return "unstructured"

    def _detect_patterns(self, text_data: str) -> List[str]:
        """Detect common patterns in text"""
        patterns = []

        # Timestamp patterns
        if re.search(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}", text_data):
            patterns.append("timestamps")

        # IP addresses
        if re.search(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", text_data):
            patterns.append("ip_addresses")

        # URLs
        if re.search(r"https?://[^\s]+", text_data):
            patterns.append("urls")

        # Email addresses
        if re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", text_data):
            patterns.append("email_addresses")

        # Error/exception patterns
        if re.search(r"\b(error|exception|failed|failure)\b", text_data, re.IGNORECASE):
            patterns.append("errors")

        # Numeric values
        if re.search(r"\b\d+\.\d+\b", text_data):
            patterns.append("decimal_numbers")

        if re.search(r"\b\d+\b", text_data):
            patterns.append("integers")

        # HTTP status codes
        if re.search(r"\b[1-5]\d{2}\b", text_data):
            patterns.append("http_status_codes")

        # File paths
        if re.search(r"[/\\][\w\-_./\\]+", text_data):
            patterns.append("file_paths")

        return patterns

    def _analyze_text_size(self, text_data: str, file_path: Optional[str]) -> SizeInfo:
        """Analyze text size characteristics"""
        lines = text_data.split("\n")
        character_count = len(text_data)
        word_count = len(text_data.split())

        file_size_mb = None
        if file_path:
            try:
                file_size_mb = Path(file_path).stat().st_size / (1024 * 1024)
            except Exception:
                pass

        # Token estimation for text
        estimated_tokens = max(word_count // 0.75, character_count // 4)  # More conservative for text

        return SizeInfo(
            record_count=len(lines),
            file_size_mb=file_size_mb,
            memory_size_mb=character_count / (1024 * 1024),
            estimated_tokens=int(estimated_tokens),
        )

    def _analyze_text_schema(self, text_data: str) -> SchemaInfo:
        """Analyze text structure"""
        lines = text_data.split("\n")

        return SchemaInfo(
            content_type=self._detect_content_type(text_data),
            encoding="utf-8",  # Assumed from connector
            line_count=len(lines),
            has_structure=self._has_structure(text_data),
            nested_levels=1,  # Text is generally flat
        )

    def _has_structure(self, text_data: str) -> bool:
        """Check if text has detectable structure"""
        # Check for consistent patterns across lines
        lines = text_data.split("\n")[:20]  # Check first 20 lines

        if len(lines) < 3:
            return False

        # Check for consistent delimiters
        comma_count = [line.count(",") for line in lines if line.strip()]
        tab_count = [line.count("\t") for line in lines if line.strip()]

        # If most lines have similar delimiter counts, it's structured
        if comma_count and len(set(comma_count)) <= 2 and max(comma_count) > 0:
            return True

        if tab_count and len(set(tab_count)) <= 2 and max(tab_count) > 0:
            return True

        # Check for timestamp patterns
        timestamp_pattern = r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}"
        timestamp_lines = sum(1 for line in lines if re.match(timestamp_pattern, line.strip()))

        return timestamp_lines > len(lines) * 0.5  # More than half have timestamps

    def _generate_text_sample_data(self, text_data: str, sample_size: int) -> List[Dict[str, Any]]:
        """Generate sample data from text"""
        lines = text_data.split("\n")
        sample_lines = lines[: min(sample_size, 20)]  # Limit sample for text

        return [
            {"line_number": i + 1, "content": line.strip(), "length": len(line)}
            for i, line in enumerate(sample_lines)
            if line.strip()  # Skip empty lines
        ]

    def _add_text_recommendations(self, analysis: TextAnalysis, text_data: str):
        """Add text-specific recommendations"""
        # General text recommendations
        if analysis.content_type == "log":
            analysis.add_llm_recommendation(
                "Log file detected - use contextual_filtering to extract specific log levels or time ranges"
            )

            if "error" in analysis.detected_patterns:
                analysis.add_llm_recommendation(
                    "Error patterns detected - suitable for error analysis and troubleshooting"
                )

        elif analysis.content_type == "config":
            analysis.add_llm_recommendation(
                "Configuration file detected - use format_conversion to extract key-value pairs"
            )

        elif analysis.content_type == "data":
            analysis.add_llm_recommendation("Structured data detected - consider parsing as CSV or delimited format")

        elif analysis.content_type == "documentation":
            analysis.add_llm_recommendation("Documentation detected - suitable for content summarization operations")

        # Pattern-based recommendations
        if "timestamps" in analysis.detected_patterns:
            analysis.add_llm_recommendation(
                "Timestamp patterns detected - suitable for time-based filtering and analysis"
            )

        if "ip_addresses" in analysis.detected_patterns:
            analysis.add_llm_recommendation("IP addresses detected - suitable for network analysis and filtering")

        if "urls" in analysis.detected_patterns:
            analysis.add_llm_recommendation("URLs detected - suitable for web traffic analysis")

        # Size-based recommendations
        if analysis.size_info:
            if analysis.size_info.is_large_dataset:
                analysis.add_llm_recommendation("Large text file - consider using contextual_filtering or sampling")

            if analysis.size_info.context_window_warning:
                analysis.add_llm_recommendation(
                    "Text may exceed LLM context window - use content_summarization for large files"
                )

        # Format-specific recommendations
        if analysis.format_detected in ["csv_like", "tsv_like"]:
            analysis.add_llm_recommendation(
                "Delimiter-separated content detected - consider processing as structured data"
            )

        if analysis.format_detected == "key_value":
            analysis.add_llm_recommendation(
                "Key-value pairs detected - use format_conversion to extract configuration values"
            )

    def _classify_text_error(self, error: Exception) -> str:
        """Classify text error type"""
        error_str = str(error).lower()

        if "file not found" in error_str or "no such file" in error_str:
            return "file_not_found"
        elif "permission" in error_str:
            return "permission_denied"
        elif "encoding" in error_str or "decode" in error_str:
            return "encoding"
        else:
            return "text_error"

    def _generate_text_error_hint(self, error: Exception, config: Dict[str, Any]) -> str:
        """Generate helpful hint for text errors"""
        error_type = self._classify_text_error(error)

        hints = {
            "file_not_found": f"Check that text file exists: {config.get('path')}",
            "permission_denied": f"Check file permissions for: {config.get('path')}",
            "encoding": "File encoding issues - try converting file to UTF-8",
            "text_error": "Check text file format and accessibility",
        }

        return hints.get(error_type, hints["text_error"])
