"""
REST API analyzer for introspection feature.

This analyzer leverages the existing DataSourceConnector for authentication and data fetching,
then performs schema analysis, pagination detection, and LLM optimization.
"""

import json
import sys
import time
from typing import Any, Dict, List, Optional

from shedboxai.connector import DataSourceConnector

from ..models import AnalysisStatus, EndpointInfo, PaginationInfo, RESTAnalysis, SchemaInfo, SizeInfo, SourceType
from ..schema_utils import calculate_nesting_depth, generate_json_schema, has_arrays, has_objects
from .base import APIAnalyzer


class RESTAnalyzer(APIAnalyzer):
    """Analyzer for REST API data sources with authentication and schema generation"""

    def __init__(self):
        super().__init__()
        self.connector = DataSourceConnector()

    @property
    def supported_type(self) -> SourceType:
        return SourceType.REST

    def analyze(self, source_config: Dict[str, Any], sample_size: int = 100) -> RESTAnalysis:
        """
        Analyze REST API data source with authentication and schema generation.

        Args:
            source_config: Source configuration with URL, auth, etc.
            sample_size: Number of records to sample for analysis

        Returns:
            RESTAnalysis: Complete analysis results
        """
        source_name = source_config.get("name", "unknown")

        # Create base analysis object
        analysis = RESTAnalysis(name=source_name, type=SourceType.REST, status=AnalysisStatus.SUCCESS)

        start_time = time.time()

        try:
            # Use existing connector to fetch data (handles authentication)
            self.logger.info(f"Analyzing REST API: {source_name}")

            # Prepare config for DataSourceConnector
            connector_config = {source_name: source_config}

            # Handle token sources if present
            token_sources = source_config.get("_token_sources", {})
            if token_sources:
                connector_config.update(token_sources)

            # Fetch data using existing connector
            response_data = self.connector.get_data(connector_config)

            # Check if we got data successfully
            if source_name not in response_data:
                raise ValueError(f"No data returned for source: {source_name}")

            api_data = response_data[source_name]

            # Check for errors from connector
            if isinstance(api_data, dict) and "error" in api_data:
                analysis.status = AnalysisStatus.FAILED
                analysis.error_message = api_data["error"]
                analysis.error_type = "api_error"
                analysis.error_hint = "Check API configuration and credentials"
                analysis.authentication_success = False
                return analysis

            # Authentication was successful if we got here
            analysis.authentication_success = True

            # Extract main data based on response_path
            main_data = self._extract_main_data(api_data, source_config.get("response_path"))

            # Generate endpoint information
            analysis.endpoint_info = self._create_endpoint_info(source_config, api_data)

            # Generate size information
            analysis.size_info = self._analyze_api_size(main_data, api_data)

            # Generate schema information
            analysis.schema_info = self._analyze_api_schema(main_data, api_data)

            # Store sample responses
            analysis.sample_responses = [api_data] if api_data else []

            # Generate sample data for LLM
            analysis.sample_data = self._generate_api_sample_data(main_data, sample_size)

            # Add LLM recommendations
            self._add_api_llm_recommendations(analysis, main_data, api_data, source_config)

            analysis.analysis_duration = time.time() - start_time

        except Exception as e:
            analysis.status = AnalysisStatus.FAILED
            analysis.error_type = self._classify_api_error(e)
            analysis.error_hint = self._generate_api_error_hint(e, source_config)
            analysis.authentication_success = False
            analysis.analysis_duration = time.time() - start_time

            # Generate user-friendly error messages based on actual issue
            error_str = str(e).lower()
            if "environment" in error_str and "variable" in error_str:
                analysis.error_message = "Missing required environment variables"
                self.logger.error(f"API analysis failed for {source_name}: Missing environment variables")
            elif "connection" in error_str or "network" in error_str:
                analysis.error_message = "Cannot connect to API endpoint"
                self.logger.error(f"API analysis failed for {source_name}: Connection failed")
            elif "unauthorized" in error_str or "authentication" in error_str:
                analysis.error_message = "API authentication failed"
                self.logger.error(f"API analysis failed for {source_name}: Authentication failed")
            elif "timeout" in error_str:
                analysis.error_message = "API request timed out"
                self.logger.error(f"API analysis failed for {source_name}: Request timeout")
            elif "recursion" in error_str or "RecursionError" in str(type(e)):
                analysis.error_message = "API response too complex to analyze"
                self.logger.error(f"API analysis failed for {source_name}: Response too complex")
            elif "memory" in error_str:
                analysis.error_message = "API response too large to process"
                self.logger.error(f"API analysis failed for {source_name}: Response too large")
            else:
                analysis.error_message = "API request failed"
                self.logger.error(f"API analysis failed for {source_name}: {str(e)}")

        return analysis

    def _extract_main_data(self, response_data: Any, response_path: Optional[str]) -> Any:
        """Extract main data from API response using response_path"""
        if not response_path:
            return response_data

        try:
            # Simple dot notation path handling
            path_parts = response_path.split(".")
            current_data = response_data

            for part in path_parts:
                if isinstance(current_data, dict) and part in current_data:
                    current_data = current_data[part]
                else:
                    # Path not found, return original data
                    self.logger.warning(f"Response path '{response_path}' not found, using full response")
                    return response_data

            return current_data

        except Exception as e:
            self.logger.warning(f"Error extracting response path '{response_path}': {str(e)}")
            return response_data

    def _create_endpoint_info(self, config: Dict[str, Any], response_data: Any) -> EndpointInfo:
        """Create endpoint information from config and response"""
        return EndpointInfo(
            url=config.get("url", ""),
            method=config.get("method", "GET").upper(),
            requires_auth=bool(config.get("headers") or config.get("requires_token")),
            pagination_detected=self._detect_pagination_in_response(response_data),
            rate_limit_headers=None,  # Would need actual response headers
            response_time_ms=None,  # Would need actual timing
            content_type="application/json",  # Assumed for REST APIs
            api_version=self._extract_api_version({}, response_data),
        )

    def _analyze_api_size(self, main_data: Any, full_response: Any) -> SizeInfo:
        """Analyze size characteristics of API response"""
        # Count records
        if isinstance(main_data, list):
            record_count = len(main_data)
        elif isinstance(main_data, dict):
            # Look for array fields in the object
            record_count = max(
                (len(v) if isinstance(v, list) else 1 for v in main_data.values()),
                default=1,
            )
        else:
            record_count = 1

        # Estimate sizes
        (
            response_size_bytes,
            response_size_mb,
            memory_size_mb,
        ) = self._estimate_response_size(full_response)

        # Token estimation
        estimated_tokens = self._estimate_api_token_count(main_data)

        return SizeInfo(
            record_count=record_count,
            file_size_mb=response_size_mb,  # Using response size as "file" size
            memory_size_mb=memory_size_mb,
            estimated_tokens=estimated_tokens,
        )

    def _analyze_api_schema(self, main_data: Any, full_response: Any) -> SchemaInfo:
        """Analyze API response schema using genson"""
        try:
            # Generate JSON schema using shared utility
            json_schema = generate_json_schema(main_data, max_items=100)

            # Detect response paths for nested data
            response_paths = self._find_response_paths(full_response)

            # Detect pagination
            pagination_info = PaginationInfo.detect_pagination(full_response)

            # Calculate nesting depth using shared utility
            nested_levels = calculate_nesting_depth(main_data)

            return SchemaInfo(
                json_schema=json_schema,
                response_paths=response_paths,
                pagination_info=pagination_info,
                nested_levels=nested_levels,
                has_arrays=has_arrays(main_data),
                has_objects=has_objects(main_data),
            )

        except Exception as e:
            self.logger.error(f"Schema analysis failed: {str(e)}")
            return SchemaInfo(
                json_schema=None,
                response_paths=[],
                nested_levels=0,
                has_arrays=False,
                has_objects=False,
            )

    def _find_response_paths(self, response_data: Any, max_depth: int = 3) -> List[str]:
        """Find useful response paths for data extraction"""
        paths = []

        def find_arrays(data: Any, current_path: str = "", depth: int = 0):
            if depth > max_depth:
                return

            if isinstance(data, dict):
                for key, value in data.items():
                    new_path = f"{current_path}.{key}" if current_path else key

                    if isinstance(value, list) and value and isinstance(value[0], dict):
                        # Found array of objects
                        paths.append(new_path)
                    elif isinstance(value, (dict, list)):
                        find_arrays(value, new_path, depth + 1)

            elif isinstance(data, list) and data:
                if isinstance(data[0], dict):
                    # This is an array of objects at the root
                    if current_path:
                        paths.append(current_path)

                    # Continue searching in first object
                    find_arrays(data[0], current_path, depth + 1)

        find_arrays(response_data)

        # Sort by usefulness (shorter paths first, data/results/items preferred)
        def path_priority(path: str) -> int:
            parts = path.split(".")
            score = len(parts) * 10  # Prefer shorter paths

            # Boost common data field names
            common_names = ["data", "results", "items", "records", "rows"]
            if any(name in parts for name in common_names):
                score -= 50

            return score

        paths.sort(key=path_priority)
        return paths[:5]  # Return top 5 most useful paths

    def _detect_pagination_in_response(self, response_data: Any) -> bool:
        """Detect if response contains pagination information"""
        pagination_indicators = [
            "pagination",
            "paging",
            "page",
            "offset",
            "limit",
            "total",
            "next",
            "previous",
            "has_more",
            "total_pages",
            "per_page",
        ]

        def check_for_pagination(data: Any, depth: int = 0) -> bool:
            if depth > 2:  # Limit search depth
                return False

            if isinstance(data, dict):
                # Check if any pagination indicators are present as keys
                for indicator in pagination_indicators:
                    if indicator in data:
                        return True

                # Check nested objects
                for value in data.values():
                    if check_for_pagination(value, depth + 1):
                        return True

            return False

        return check_for_pagination(response_data)

    def _extract_api_version(self, headers: Dict[str, str], response_data: Any) -> Optional[str]:
        """Extract API version from headers or response"""
        # Check headers first (would need actual response headers)
        version_headers = ["api-version", "x-api-version", "version"]
        for header in version_headers:
            if header in headers:
                return headers[header]

        # Check response data
        if isinstance(response_data, dict):
            version_fields = ["version", "api_version", "apiVersion"]
            for field in version_fields:
                if field in response_data:
                    return str(response_data[field])

            # Check metadata or meta objects
            meta_objects = ["meta", "metadata", "_meta"]
            for meta in meta_objects:
                if meta in response_data and isinstance(response_data[meta], dict):
                    for field in version_fields:
                        if field in response_data[meta]:
                            return str(response_data[meta][field])

        return None

    def _estimate_api_token_count(self, data: Any) -> int:
        """Estimate token count for API response data"""
        try:
            # Convert to JSON string and estimate
            json_str = json.dumps(data, ensure_ascii=False)
            return len(json_str) // 4  # Rough estimation
        except Exception:
            return sys.getsizeof(data) // 16

    def _generate_api_sample_data(self, main_data: Any, sample_size: int) -> List[Dict[str, Any]]:
        """Generate sample data from API response"""
        if isinstance(main_data, list):
            # Array data - return sample
            sample_count = min(sample_size, len(main_data))
            if sample_count > 0 and all(isinstance(item, dict) for item in main_data[:10]):
                return main_data[:sample_count]
            else:
                # Array of primitives
                return [{"value": item, "index": i} for i, item in enumerate(main_data[:sample_count])]

        elif isinstance(main_data, dict):
            return [main_data]

        else:
            return [{"value": main_data}]

    def _add_api_llm_recommendations(
        self,
        analysis: RESTAnalysis,
        main_data: Any,
        full_response: Any,
        config: Dict[str, Any],
    ):
        """Add API-specific LLM recommendations"""
        # Response path recommendations
        if analysis.schema_info and analysis.schema_info.response_paths:
            best_path = analysis.schema_info.response_paths[0]
            analysis.add_llm_recommendation(f"Use response_path: '{best_path}' to extract main data array")

        # Pagination recommendations
        if analysis.schema_info and analysis.schema_info.pagination_info:
            pag_info = analysis.schema_info.pagination_info
            if pag_info.total_records:
                analysis.add_processing_note(f"API contains {pag_info.total_records:,} total records with pagination")
                analysis.add_llm_recommendation("Consider pagination handling for complete dataset access")

        # Authentication recommendations
        if analysis.authentication_success:
            analysis.add_processing_note("Authentication successful")

        # Data structure recommendations
        if isinstance(main_data, list):
            analysis.add_llm_recommendation("Array response suitable for direct processing with ShedBoxAI operations")

            if len(main_data) > 0 and isinstance(main_data[0], dict):
                # Object array - suggest field extraction
                sample_fields = list(main_data[0].keys())[:5]
                analysis.add_llm_recommendation(
                    f"Extract specific fields with format_conversion: {', '.join(sample_fields)}"
                )

        elif isinstance(main_data, dict):
            analysis.add_llm_recommendation("Object response - use format_conversion to extract nested data")

        # Schema complexity recommendations
        if analysis.schema_info and analysis.schema_info.nested_levels > 3:
            analysis.add_llm_recommendation("Deep nesting detected - consider flattening data structure")

        # Size-based recommendations
        if analysis.size_info:
            if analysis.size_info.is_large_dataset:
                analysis.add_llm_recommendation(
                    "Large dataset detected - consider using contextual_filtering for data reduction"
                )

            if analysis.size_info.context_window_warning:
                analysis.add_llm_recommendation(
                    "Dataset may exceed LLM context window - use sampling or aggregation operations"
                )

    def _classify_api_error(self, error: Exception) -> str:
        """Classify API error type"""
        error_str = str(error).lower()
        error_type = type(error).__name__

        if "authentication" in error_str or "unauthorized" in error_str:
            return "auth"
        elif "connection" in error_str or "network" in error_str:
            return "network"
        elif "timeout" in error_str:
            return "timeout"
        elif "rate limit" in error_str:
            return "rate_limit"
        elif "json" in error_str:
            return "parsing"
        elif "RecursionError" in error_type or "maximum recursion depth" in error_str:
            return "recursion_depth"
        elif "memory" in error_str or "out of memory" in error_str:
            return "memory"
        else:
            return "api_error"

    def _generate_api_error_hint(self, error: Exception, config: Dict[str, Any]) -> str:
        """Generate helpful hint for API errors"""
        error_type = self._classify_api_error(error)

        hints = {
            "auth": "Check API credentials and permissions",
            "network": f"Verify API endpoint is accessible: {config.get('url')}",
            "timeout": "API response too slow - consider increasing timeout or reducing sample size",
            "rate_limit": "Reduce sample_size parameter or wait before retrying",
            "parsing": "API returned non-JSON response - check endpoint and parameters",
            "recursion_depth": (
                "Data structure too deeply nested or contains circular references. "
                "Try simplifying the data structure."
            ),
            "memory": "Insufficient memory to process large dataset - consider reducing sample size or data complexity",
            "api_error": "Check API documentation and endpoint configuration",
        }

        return hints.get(error_type, hints["api_error"])
