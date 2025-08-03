"""
YAML analyzer for introspection feature.

This analyzer handles YAML configuration files and provides schema analysis
optimized for LLM consumption.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from shedboxai.connector import DataSourceConnector

from ..models import AnalysisStatus, SchemaInfo, SizeInfo, SourceType, YAMLAnalysis
from .base import FileAnalyzer


class YAMLAnalyzer(FileAnalyzer):
    """Analyzer for YAML configuration files"""

    def __init__(self):
        super().__init__()
        self.connector = DataSourceConnector()

    @property
    def supported_type(self) -> SourceType:
        return SourceType.YAML

    def analyze(self, source_config: Dict[str, Any], sample_size: int = 100) -> YAMLAnalysis:
        """Analyze YAML data source"""
        source_name = source_config.get("name", "unknown")

        # Create analysis object
        analysis = YAMLAnalysis(name=source_name, type=SourceType.YAML, status=AnalysisStatus.SUCCESS)

        try:
            # Use existing connector to load YAML data
            connector_config = {source_name: source_config}
            response_data = self.connector.get_data(connector_config)

            # Check if we got data successfully
            if source_name not in response_data:
                raise ValueError(f"No data returned for source: {source_name}")

            yaml_data = response_data[source_name]

            # Check for errors from connector
            if isinstance(yaml_data, dict) and "error" in yaml_data:
                analysis.status = AnalysisStatus.FAILED
                analysis.error_message = yaml_data["error"]
                analysis.error_type = "yaml_error"
                analysis.error_hint = "Check YAML file format and accessibility"
                return analysis

            # Extract top-level keys
            if isinstance(yaml_data, dict):
                analysis.top_level_keys = list(yaml_data.keys())

            # Detect environment variables
            (
                analysis.env_variables,
                analysis.has_env_variables,
            ) = self._detect_env_variables(yaml_data)

            # Generate size and schema info
            analysis.size_info = self._analyze_yaml_size(yaml_data, source_config.get("path"))
            analysis.schema_info = self._analyze_yaml_schema(yaml_data)
            analysis.sample_data = self._generate_yaml_sample_data(yaml_data)

            # Add recommendations
            self._add_yaml_recommendations(analysis, yaml_data)

        except Exception as e:
            self.logger.error(f"YAML analysis failed for {source_name}: {str(e)}")
            analysis.status = AnalysisStatus.FAILED
            analysis.error_message = str(e)
            analysis.error_type = self._classify_yaml_error(e)
            analysis.error_hint = self._generate_yaml_error_hint(e, source_config)

        return analysis

    def _detect_env_variables(self, data: Any) -> tuple:
        """Detect environment variable references in YAML"""
        env_vars = set()

        def find_env_vars(obj: Any):
            if isinstance(obj, str):
                # Pattern: ${VAR_NAME}
                matches = re.findall(r"\$\{([^}]+)\}", obj)
                env_vars.update(matches)
            elif isinstance(obj, dict):
                for value in obj.values():
                    find_env_vars(value)
            elif isinstance(obj, list):
                for item in obj:
                    find_env_vars(item)

        find_env_vars(data)
        return list(env_vars), len(env_vars) > 0

    def _analyze_yaml_size(self, yaml_data: Any, file_path: Optional[str]) -> SizeInfo:
        """Analyze YAML size characteristics"""
        # For YAML, size is usually not a concern for LLM context
        record_count = self._count_yaml_items(yaml_data)

        file_size_mb = None
        if file_path:
            try:
                file_size_mb = Path(file_path).stat().st_size / (1024 * 1024)
            except Exception:
                pass

        # YAML is typically small and suitable for full LLM processing
        return SizeInfo(
            record_count=record_count,
            file_size_mb=file_size_mb,
            memory_size_mb=0.01,  # Typically very small
            estimated_tokens=len(str(yaml_data)) // 4,
        )

    def _count_yaml_items(self, data: Any) -> int:
        """Count items in YAML structure"""
        if isinstance(data, dict):
            return len(data)
        elif isinstance(data, list):
            return len(data)
        else:
            return 1

    def _analyze_yaml_schema(self, yaml_data: Any) -> SchemaInfo:
        """Analyze YAML structure"""
        # YAML is typically configuration, so analyze as nested structure
        nesting_depth = self._calculate_yaml_nesting(yaml_data)

        return SchemaInfo(
            nested_levels=nesting_depth,
            has_arrays=self._has_yaml_arrays(yaml_data),
            has_objects=isinstance(yaml_data, dict),
        )

    def _calculate_yaml_nesting(self, data: Any, depth: int = 0) -> int:
        """Calculate YAML nesting depth"""
        if isinstance(data, dict):
            if not data:
                return depth
            return max(self._calculate_yaml_nesting(v, depth + 1) for v in data.values())
        elif isinstance(data, list):
            if not data:
                return depth
            return max(self._calculate_yaml_nesting(item, depth + 1) for item in data[:5])
        return depth

    def _has_yaml_arrays(self, data: Any) -> bool:
        """Check if YAML contains arrays"""
        if isinstance(data, list):
            return True
        elif isinstance(data, dict):
            return any(self._has_yaml_arrays(v) for v in data.values())
        return False

    def _generate_yaml_sample_data(self, yaml_data: Any) -> List[Dict[str, Any]]:
        """Generate sample data representation"""
        if isinstance(yaml_data, dict):
            return [yaml_data]
        elif isinstance(yaml_data, list):
            return yaml_data[:10]  # First 10 items
        else:
            return [{"value": yaml_data}]

    def _add_yaml_recommendations(self, analysis: YAMLAnalysis, yaml_data: Any):
        """Add YAML-specific recommendations"""
        analysis.add_llm_recommendation(
            "YAML configuration suitable for direct processing - small size and structured format"
        )

        if analysis.has_env_variables:
            analysis.add_processing_note(f"Environment variables detected: {', '.join(analysis.env_variables)}")
            analysis.add_llm_recommendation(
                "Environment variables present - ensure they're set for proper configuration loading"
            )

        if analysis.schema_info.nested_levels > 2:
            analysis.add_llm_recommendation(
                "Nested configuration structure - use format_conversion to extract specific sections"
            )

        # Configuration-specific recommendations
        if isinstance(yaml_data, dict):
            if "database" in yaml_data or "db" in yaml_data:
                analysis.add_llm_recommendation(
                    "Database configuration detected - suitable for connection string extraction"
                )

            if "api" in yaml_data or "endpoints" in yaml_data:
                analysis.add_llm_recommendation("API configuration detected - suitable for endpoint URL extraction")

            if "logging" in yaml_data or "logs" in yaml_data:
                analysis.add_llm_recommendation(
                    "Logging configuration detected - suitable for log level and format analysis"
                )

    def _classify_yaml_error(self, error: Exception) -> str:
        """Classify YAML error type"""
        error_str = str(error).lower()

        if "file not found" in error_str or "no such file" in error_str:
            return "file_not_found"
        elif "permission" in error_str:
            return "permission_denied"
        elif "yaml" in error_str or "parsing" in error_str:
            return "parsing"
        else:
            return "yaml_error"

    def _generate_yaml_error_hint(self, error: Exception, config: Dict[str, Any]) -> str:
        """Generate helpful hint for YAML errors"""
        error_type = self._classify_yaml_error(error)

        hints = {
            "file_not_found": f"Check that YAML file exists: {config.get('path')}",
            "permission_denied": f"Check file permissions for: {config.get('path')}",
            "parsing": "Check YAML syntax and formatting",
            "yaml_error": "Check YAML file format and configuration",
        }

        return hints.get(error_type, hints["yaml_error"])
