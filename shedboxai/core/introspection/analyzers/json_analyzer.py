"""
JSON data source analyzer with nested structure analysis.

This module provides JSON file analysis capabilities including nested structure detection,
schema generation, and LLM-optimized recommendations.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..models import AnalysisStatus, JSONAnalysis, SchemaInfo, SizeInfo, SourceType
from ..schema_utils import calculate_nesting_depth, generate_json_schema, has_arrays, has_objects
from .base import FileAnalyzer


class JSONAnalyzer(FileAnalyzer):
    """Analyzer for JSON data sources with nested structure analysis"""

    @property
    def supported_type(self) -> SourceType:
        return SourceType.JSON

    def analyze(self, source_config: Dict[str, Any], sample_size: int = 100) -> JSONAnalysis:
        """
        Analyze JSON data source with nested structure analysis.

        Args:
            source_config: Source configuration containing 'path' or 'data'
            sample_size: Number of records to sample for analysis

        Returns:
            JSONAnalysis: Complete analysis results
        """
        source_name = source_config.get("name", "unknown")

        try:
            # Handle inline data vs file path
            if "data" in source_config:
                json_data = source_config["data"]
                file_path = None
            else:
                file_path = source_config.get("path")
                if not file_path:
                    raise ValueError("JSON source must have 'path' or 'data' field")
                json_data = self._load_json_file(file_path)

            # Create base analysis object
            analysis = JSONAnalysis(name=source_name, type=SourceType.JSON, status=AnalysisStatus.SUCCESS)

            # Determine if data is array or object
            analysis.is_array = isinstance(json_data, list)

            # Get top-level keys
            if isinstance(json_data, dict):
                analysis.top_level_keys = list(json_data.keys())
            elif isinstance(json_data, list) and len(json_data) > 0 and isinstance(json_data[0], dict):
                # Array of objects - get keys from first object
                analysis.top_level_keys = list(json_data[0].keys())

            # Generate size information
            analysis.size_info = self._analyze_size(json_data, file_path)

            # Generate schema information
            analysis.schema_info = self._analyze_schema(json_data)

            # Generate sample data
            analysis.sample_data = self._generate_sample_data(json_data, sample_size)

            # Add LLM recommendations
            self._add_llm_recommendations(analysis, json_data)

            return analysis

        except Exception as e:
            return self._handle_analysis_error(source_name, e)

    def _load_json_file(self, file_path: str) -> Union[Dict, List]:
        """Load JSON file with proper error handling"""
        try:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"JSON file not found: {file_path}")

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            return data

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {str(e)}")
        except UnicodeDecodeError:
            # Try alternative encodings
            fallback_encodings = ["latin-1", "cp1252"]
            for encoding in fallback_encodings:
                try:
                    with open(file_path, "r", encoding=encoding) as f:
                        return json.load(f)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
            raise ValueError("Could not decode JSON file with any encoding")

    def _analyze_size(self, json_data: Union[Dict, List], file_path: Optional[str] = None) -> SizeInfo:
        """Analyze size characteristics"""
        # Record count
        if isinstance(json_data, list):
            record_count = len(json_data)
        elif isinstance(json_data, dict):
            # For objects, count nested arrays or treat as single record
            record_count = self._count_nested_records(json_data)
        else:
            record_count = 1

        # File size
        file_size_mb = None
        if file_path:
            try:
                file_size_bytes = Path(file_path).stat().st_size
                file_size_mb = file_size_bytes / (1024 * 1024)
            except Exception:
                pass

        # Memory size
        memory_size_mb = sys.getsizeof(json_data) / (1024 * 1024)

        # Token estimation
        estimated_tokens = self._estimate_token_count(json_data)

        return SizeInfo(
            record_count=record_count,
            file_size_mb=file_size_mb,
            memory_size_mb=memory_size_mb,
            estimated_tokens=estimated_tokens,
        )

    def _count_nested_records(self, data: Union[Dict, List], max_depth: int = 3) -> int:
        """Count records in nested JSON structure"""
        if max_depth <= 0:
            return 1

        if isinstance(data, list):
            return len(data)
        elif isinstance(data, dict):
            # Look for arrays in the dictionary
            max_records = 1
            for value in data.values():
                if isinstance(value, list):
                    max_records = max(max_records, len(value))
                elif isinstance(value, dict):
                    nested_count = self._count_nested_records(value, max_depth - 1)
                    max_records = max(max_records, nested_count)
            return max_records
        else:
            return 1

    def _estimate_token_count(self, json_data: Union[Dict, List]) -> int:
        """Estimate token count for LLM context planning"""
        try:
            # Convert to JSON string to estimate size
            json_str = json.dumps(json_data, ensure_ascii=False)
            # Rough estimation: 1 token per 4 characters
            return len(json_str) // 4
        except Exception:
            # Fallback estimation based on memory size
            return sys.getsizeof(json_data) // 16  # Very rough estimate

    def _analyze_schema(self, json_data: Union[Dict, List]) -> SchemaInfo:
        """Analyze JSON schema structure using genson"""
        try:
            # Generate JSON schema using genson
            json_schema = generate_json_schema(json_data, max_items=100)

            # Calculate nesting depth
            nested_levels = calculate_nesting_depth(json_data)

            return SchemaInfo(
                json_schema=json_schema,
                nested_levels=nested_levels,
                has_arrays=has_arrays(json_data),
                has_objects=has_objects(json_data),
            )

        except Exception as e:
            self.logger.error(f"Schema analysis failed: {str(e)}")
            return SchemaInfo(json_schema=None, nested_levels=0, has_arrays=False, has_objects=False)

    def _generate_sample_data(self, json_data: Union[Dict, List], sample_size: int) -> List[Dict[str, Any]]:
        """Generate sample data for LLM context"""
        if isinstance(json_data, list):
            # Array data
            if not json_data:
                return []

            sample_size = min(sample_size, len(json_data))
            if all(isinstance(item, dict) for item in json_data[:10]):
                # Array of objects
                return json_data[:sample_size]
            else:
                # Array of primitives - convert to dict format
                return [{"value": item, "index": i} for i, item in enumerate(json_data[:sample_size])]

        elif isinstance(json_data, dict):
            # Single object
            return [json_data]

        else:
            # Primitive value
            return [{"value": json_data}]

    def _add_llm_recommendations(self, analysis: JSONAnalysis, json_data: Union[Dict, List]):
        """Add LLM-specific recommendations"""
        # Size-based recommendations
        if analysis.size_info.is_large_dataset:
            analysis.add_llm_recommendation(
                "Large JSON dataset - consider extracting specific fields with format_conversion"
            )

        # Structure-based recommendations
        if analysis.schema_info.nested_levels > 3:
            analysis.add_llm_recommendation("Deep nesting detected - use JSONPath expressions for data extraction")

        if analysis.is_array:
            analysis.add_llm_recommendation("Array structure detected - suitable for direct processing or iteration")

            if analysis.top_level_keys:
                analysis.add_llm_recommendation(f"Object array with fields: {', '.join(analysis.top_level_keys[:5])}")
        else:
            analysis.add_llm_recommendation(
                "Object structure detected - use format_conversion to extract nested fields"
            )

            if analysis.top_level_keys:
                analysis.add_processing_note(f"Top-level keys: {', '.join(analysis.top_level_keys)}")

        # Field-specific recommendations
        if analysis.schema_info.columns:
            # Look for identifier fields
            id_fields = [col.name for col in analysis.schema_info.columns if "id" in col.name.lower()]
            if id_fields:
                analysis.add_llm_recommendation(
                    f"ID fields detected ({', '.join(id_fields)}) - suitable for relationships"
                )

            # Look for nested objects/arrays
            complex_fields = [col.name for col in analysis.schema_info.columns if col.type in ["object", "array"]]
            if complex_fields:
                analysis.add_processing_note(f"Complex fields requiring extraction: {', '.join(complex_fields)}")

    def _handle_analysis_error(self, source_name: str, error: Exception) -> JSONAnalysis:
        """Handle analysis errors and return failed analysis"""
        error_type = self._classify_error(error)
        error_message = str(error)
        error_hint = self._generate_error_hint(error, {"name": source_name})

        return JSONAnalysis(
            name=source_name,
            type=SourceType.JSON,
            status=AnalysisStatus.FAILED,
            error_message=error_message,
            error_hint=error_hint,
            error_type=error_type,
        )
