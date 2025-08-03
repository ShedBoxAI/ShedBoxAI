"""
Data models for ShedBoxAI introspection system.

This module defines all the data structures used throughout the introspection
process, from configuration options to analysis results.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class SourceType(Enum):
    """Supported data source types"""

    CSV = "csv"
    JSON = "json"
    YAML = "yaml"
    TEXT = "text"
    REST = "rest"


class AnalysisStatus(Enum):
    """Analysis status for each source"""

    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"  # Some analysis completed but with warnings


@dataclass
class IntrospectionOptions:
    """Configuration options for introspection process"""

    config_path: str
    output_path: str = "introspection.md"
    sample_size: int = 100
    retry_sources: List[str] = field(default_factory=list)
    skip_errors: bool = False
    force_overwrite: bool = False
    validate_only: bool = False
    verbose: bool = False
    include_samples: bool = False

    def __post_init__(self):
        """Validate options after initialization"""
        if self.sample_size <= 0:
            raise ValueError("sample_size must be positive")
        if not self.config_path:
            raise ValueError("config_path is required")


@dataclass
class SizeInfo:
    """Information about data source size and memory implications"""

    record_count: Optional[int] = None
    file_size_mb: Optional[float] = None
    memory_size_mb: Optional[float] = None
    is_large_dataset: bool = False
    context_window_warning: bool = False
    estimated_tokens: Optional[int] = None  # For LLM context estimation

    def __post_init__(self):
        """Calculate derived fields"""
        # Determine if dataset is large (>10MB or >50k records or >100k tokens)
        if (
            (self.file_size_mb and self.file_size_mb > 10)
            or (self.record_count and self.record_count > 50000)
            or (self.estimated_tokens and self.estimated_tokens > 100000)
        ):
            self.is_large_dataset = True

        # Context window warning for datasets that might exceed typical LLM limits
        if (self.estimated_tokens and self.estimated_tokens > 50000) or (
            self.record_count and self.record_count > 10000
        ):
            self.context_window_warning = True


@dataclass
class ColumnInfo:
    """Detailed information about a single column/field"""

    name: str
    type: str  # 'string', 'integer', 'float', 'boolean', 'date', 'object'
    null_percentage: float
    unique_count: int
    total_count: int
    sample_values: List[Any] = field(default_factory=list)

    # Statistical info for numeric columns
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    median: Optional[float] = None
    quartiles: Optional[List[float]] = None  # [Q1, Q2, Q3]

    # String-specific info
    avg_length: Optional[float] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None  # Detected patterns like "CUST_###"

    # Categorical info
    value_counts: Optional[Dict[str, int]] = None  # Top values with counts
    is_categorical: bool = False
    is_identifier: bool = False  # Likely primary/foreign key

    @property
    def uniqueness_ratio(self) -> float:
        """Ratio of unique values to total values"""
        if self.total_count == 0:
            return 0.0
        return self.unique_count / self.total_count

    @property
    def is_likely_primary_key(self) -> bool:
        """Heuristic to detect primary keys"""
        return self.uniqueness_ratio == 1.0 and self.null_percentage == 0.0 and self.name.lower().endswith("_id")

    @property
    def is_likely_foreign_key(self) -> bool:
        """Heuristic to detect foreign keys"""
        return (
            self.name.lower().endswith("_id")
            and self.uniqueness_ratio < 1.0
            and self.uniqueness_ratio > 0.1  # Not too few unique values
        )


@dataclass
class PaginationInfo:
    """Information about API pagination"""

    type: str  # 'offset', 'cursor', 'page', 'none'
    total_records: Optional[int] = None
    page_size: Optional[int] = None
    has_more: bool = False
    next_page_info: Optional[Dict[str, Any]] = None

    @classmethod
    def detect_pagination(cls, response_data: Dict[str, Any]) -> "PaginationInfo":
        """Auto-detect pagination from API response"""
        # Common pagination patterns
        if "pagination" in response_data:
            pag = response_data["pagination"]
            return cls(
                type="page",
                total_records=pag.get("total"),
                page_size=pag.get("per_page"),
                has_more=pag.get("has_more", False),
            )
        elif "next" in response_data or "has_more" in response_data:
            return cls(type="cursor", has_more=response_data.get("has_more", False))
        elif "offset" in response_data or "limit" in response_data:
            return cls(type="offset")
        else:
            return cls(type="none")


@dataclass
class EndpointInfo:
    """Information about REST API endpoint"""

    url: str
    method: str
    requires_auth: bool
    pagination_detected: bool
    rate_limit_headers: Optional[Dict[str, str]] = None
    response_time_ms: Optional[float] = None
    content_type: Optional[str] = None
    api_version: Optional[str] = None


@dataclass
class SchemaInfo:
    """Schema information for different source types"""

    # For CSV/JSON/YAML
    columns: Optional[List[ColumnInfo]] = None
    statistical_summary: Optional[Dict[str, Any]] = None

    # For REST APIs
    json_schema: Optional[Dict[str, Any]] = None
    response_paths: Optional[List[str]] = None
    pagination_info: Optional[PaginationInfo] = None

    # For Text files
    content_type: Optional[str] = None  # 'logs', 'documentation', 'data', 'unknown'
    encoding: Optional[str] = None
    line_count: Optional[int] = None
    has_structure: bool = False  # Whether text has detectable patterns/structure

    # Common metadata
    nested_levels: int = 0  # Depth of nesting in JSON/YAML
    has_arrays: bool = False
    has_objects: bool = False


@dataclass
class SourceAnalysis:
    """Base analysis result for any data source"""

    name: str
    type: SourceType
    status: AnalysisStatus

    # Error information
    error_message: Optional[str] = None
    error_hint: Optional[str] = None
    error_type: Optional[str] = None  # 'auth', 'network', 'parsing', 'file_not_found'

    # Analysis results
    size_info: Optional[SizeInfo] = None
    schema_info: Optional[SchemaInfo] = None
    sample_data: List[Dict[str, Any]] = field(default_factory=list)

    # LLM optimization recommendations
    llm_recommendations: List[str] = field(default_factory=list)
    processing_notes: List[str] = field(default_factory=list)

    # Metadata
    analysis_timestamp: datetime = field(default_factory=datetime.now)
    analysis_duration_ms: Optional[float] = None

    @property
    def success(self) -> bool:
        """Convenience property for success check"""
        return self.status == AnalysisStatus.SUCCESS

    def add_llm_recommendation(self, recommendation: str):
        """Add an LLM processing recommendation"""
        if recommendation not in self.llm_recommendations:
            self.llm_recommendations.append(recommendation)

    def add_processing_note(self, note: str):
        """Add a processing note"""
        if note not in self.processing_notes:
            self.processing_notes.append(note)


# Type-specific analysis classes
@dataclass
class CSVAnalysis(SourceAnalysis):
    """CSV-specific analysis results"""

    delimiter: Optional[str] = None
    encoding: Optional[str] = None
    has_header: bool = True
    column_count: int = 0

    def __post_init__(self):
        """Add CSV-specific recommendations"""
        if self.size_info and self.size_info.is_large_dataset:
            self.add_llm_recommendation("Use contextual_filtering or advanced_operations for sampling/aggregation")
        if self.schema_info and self.schema_info.columns:
            # Find potential key columns
            key_cols = [col for col in self.schema_info.columns if col.is_likely_primary_key]
            if key_cols:
                self.add_processing_note(f"Primary key detected: {', '.join(col.name for col in key_cols)}")


@dataclass
class RESTAnalysis(SourceAnalysis):
    """REST API-specific analysis results"""

    authentication_success: bool = False
    endpoint_info: Optional[EndpointInfo] = None
    sample_responses: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        """Add REST-specific recommendations"""
        if self.schema_info and self.schema_info.response_paths:
            main_path = self.schema_info.response_paths[0]
            self.add_llm_recommendation(f"Use response_path: '{main_path}' to extract main data array")
        if self.schema_info and self.schema_info.pagination_info:
            if self.schema_info.pagination_info.total_records:
                total = self.schema_info.pagination_info.total_records
                self.add_processing_note(f"API contains {total:,} total records - consider pagination for full dataset")


@dataclass
class JSONAnalysis(SourceAnalysis):
    """JSON file-specific analysis results"""

    is_array: bool = False
    top_level_keys: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Add JSON-specific recommendations"""
        if self.schema_info and self.schema_info.nested_levels > 3:
            self.add_llm_recommendation("Deep nesting detected - consider using JSONPath for data extraction")


@dataclass
class YAMLAnalysis(SourceAnalysis):
    """YAML file-specific analysis results"""

    top_level_keys: List[str] = field(default_factory=list)
    has_env_variables: bool = False
    env_variables: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Add YAML-specific recommendations"""
        if self.has_env_variables:
            self.add_processing_note(f"Environment variables detected: {', '.join(self.env_variables)}")


@dataclass
class TextAnalysis(SourceAnalysis):
    """Text file-specific analysis results"""

    detected_format: Optional[str] = None  # 'logs', 'csv', 'json', 'plain'
    line_count: int = 0
    charset: Optional[str] = None

    def __post_init__(self):
        """Add text-specific recommendations"""
        if self.detected_format == "logs":
            self.add_llm_recommendation("Log format detected - consider parsing with regex patterns")


@dataclass
class Relationship:
    """Detected relationship between data sources"""

    source_a: str
    source_b: str
    type: str  # 'foreign_key', 'name_overlap', 'potential_join'
    confidence: float  # 0.0 to 1.0
    field_a: str
    field_b: str
    description: str

    # Additional metadata
    sample_matching_values: List[Any] = field(default_factory=list)
    match_percentage: Optional[float] = None

    @property
    def is_high_confidence(self) -> bool:
        """Check if this is a high-confidence relationship"""
        return self.confidence >= 0.8

    @property
    def relationship_strength(self) -> str:
        """Human-readable relationship strength"""
        if self.confidence >= 0.9:
            return "Very Strong"
        elif self.confidence >= 0.7:
            return "Strong"
        elif self.confidence >= 0.5:
            return "Moderate"
        else:
            return "Weak"


@dataclass
class IntrospectionResult:
    """Complete result of introspection process"""

    analyses: Dict[str, SourceAnalysis]
    relationships: List[Relationship]

    # Summary statistics
    success_count: int = 0
    total_count: int = 0
    failure_count: int = 0

    # Processing metadata
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    total_duration_ms: Optional[float] = None

    # Global recommendations
    global_recommendations: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Calculate summary statistics"""
        self.total_count = len(self.analyses)
        self.success_count = sum(1 for a in self.analyses.values() if a.success)
        self.failure_count = self.total_count - self.success_count

    @property
    def success_rate(self) -> float:
        """Percentage of successful analyses"""
        if self.total_count == 0:
            return 0.0
        return (self.success_count / self.total_count) * 100

    @property
    def has_relationships(self) -> bool:
        """Check if any relationships were detected"""
        return len(self.relationships) > 0

    @property
    def successful_analyses(self) -> Dict[str, SourceAnalysis]:
        """Get only successful analyses"""
        return {name: analysis for name, analysis in self.analyses.items() if analysis.success}

    @property
    def failed_analyses(self) -> Dict[str, SourceAnalysis]:
        """Get only failed analyses"""
        return {name: analysis for name, analysis in self.analyses.items() if not analysis.success}

    def add_global_recommendation(self, recommendation: str):
        """Add a global processing recommendation"""
        if recommendation not in self.global_recommendations:
            self.global_recommendations.append(recommendation)

    def finalize(self):
        """Finalize the result (call when introspection is complete)"""
        self.end_time = datetime.now()
        if self.start_time and self.end_time:
            self.total_duration_ms = (self.end_time - self.start_time).total_seconds() * 1000

        # Add global recommendations based on analysis results
        large_datasets = [
            name
            for name, analysis in self.successful_analyses.items()
            if analysis.size_info and analysis.size_info.is_large_dataset
        ]

        if large_datasets:
            self.add_global_recommendation(
                f"Large datasets detected ({', '.join(large_datasets)}) - " "use sampling and aggregation operations"
            )

        if self.has_relationships:
            self.add_global_recommendation(
                "Data relationships detected - consider using relationship_highlighting operations"
            )
