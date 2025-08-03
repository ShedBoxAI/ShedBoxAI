"""
Unit tests for introspection data models.

These tests validate the data models, their properties, validation logic,
and post-initialization behaviors.
"""

from datetime import datetime

import pytest

from shedboxai.core.introspection.models import (
    AnalysisStatus,
    ColumnInfo,
    CSVAnalysis,
    EndpointInfo,
    IntrospectionOptions,
    IntrospectionResult,
    JSONAnalysis,
    PaginationInfo,
    Relationship,
    RESTAnalysis,
    SchemaInfo,
    SizeInfo,
    SourceAnalysis,
    SourceType,
    TextAnalysis,
    YAMLAnalysis,
)


class TestIntrospectionOptions:
    """Test IntrospectionOptions dataclass"""

    def test_valid_options(self):
        """Test creating valid options"""
        options = IntrospectionOptions(config_path="test.yaml", sample_size=50)
        assert options.config_path == "test.yaml"
        assert options.sample_size == 50
        assert options.output_path == "introspection.md"  # default
        assert options.skip_errors is False  # default
        assert options.verbose is False  # default

    def test_all_options(self):
        """Test creating options with all parameters"""
        options = IntrospectionOptions(
            config_path="sources.yaml",
            output_path="custom.md",
            sample_size=200,
            retry_sources=["api1", "api2"],
            skip_errors=True,
            force_overwrite=True,
            validate_only=True,
            verbose=True,
        )
        assert options.config_path == "sources.yaml"
        assert options.output_path == "custom.md"
        assert options.sample_size == 200
        assert options.retry_sources == ["api1", "api2"]
        assert options.skip_errors is True
        assert options.force_overwrite is True
        assert options.validate_only is True
        assert options.verbose is True

    def test_invalid_sample_size(self):
        """Test validation of sample_size"""
        with pytest.raises(ValueError, match="sample_size must be positive"):
            IntrospectionOptions(config_path="test.yaml", sample_size=-1)

        with pytest.raises(ValueError, match="sample_size must be positive"):
            IntrospectionOptions(config_path="test.yaml", sample_size=0)

    def test_missing_config_path(self):
        """Test validation of config_path"""
        with pytest.raises(ValueError, match="config_path is required"):
            IntrospectionOptions(config_path="")


class TestSizeInfo:
    """Test SizeInfo dataclass and calculations"""

    def test_basic_size_info(self):
        """Test basic size info creation"""
        size_info = SizeInfo(record_count=1000, file_size_mb=5.0, memory_size_mb=8.0)
        assert size_info.record_count == 1000
        assert size_info.file_size_mb == 5.0
        assert size_info.memory_size_mb == 8.0
        assert size_info.is_large_dataset is False  # Not large enough
        assert size_info.context_window_warning is False

    def test_large_dataset_detection_by_file_size(self):
        """Test large dataset detection by file size"""
        size_info = SizeInfo(file_size_mb=15.0)
        assert size_info.is_large_dataset is True

        size_info = SizeInfo(file_size_mb=5.0)
        assert size_info.is_large_dataset is False

    def test_large_dataset_detection_by_record_count(self):
        """Test large dataset detection by record count"""
        size_info = SizeInfo(record_count=60000)
        assert size_info.is_large_dataset is True

        size_info = SizeInfo(record_count=30000)
        assert size_info.is_large_dataset is False

    def test_large_dataset_detection_by_tokens(self):
        """Test large dataset detection by token count"""
        size_info = SizeInfo(estimated_tokens=150000)
        assert size_info.is_large_dataset is True

        size_info = SizeInfo(estimated_tokens=50000)
        assert size_info.is_large_dataset is False

    def test_context_window_warning_by_tokens(self):
        """Test context window warning by token count"""
        size_info = SizeInfo(estimated_tokens=75000)
        assert size_info.context_window_warning is True

        size_info = SizeInfo(estimated_tokens=25000)
        assert size_info.context_window_warning is False

    def test_context_window_warning_by_records(self):
        """Test context window warning by record count"""
        size_info = SizeInfo(record_count=15000)
        assert size_info.context_window_warning is True

        size_info = SizeInfo(record_count=5000)
        assert size_info.context_window_warning is False


class TestColumnInfo:
    """Test ColumnInfo dataclass and properties"""

    def test_basic_column_info(self):
        """Test basic column info creation"""
        col = ColumnInfo(
            name="user_name",
            type="string",
            null_percentage=5.0,
            unique_count=95,
            total_count=100,
            sample_values=["Alice", "Bob", "Charlie"],
        )
        assert col.name == "user_name"
        assert col.type == "string"
        assert col.null_percentage == 5.0
        assert col.unique_count == 95
        assert col.total_count == 100
        assert col.uniqueness_ratio == 0.95

    def test_uniqueness_ratio_calculation(self):
        """Test uniqueness ratio calculation"""
        col = ColumnInfo("test", "string", 0.0, 50, 100)
        assert col.uniqueness_ratio == 0.5

        col = ColumnInfo("test", "string", 0.0, 100, 100)
        assert col.uniqueness_ratio == 1.0

        col = ColumnInfo("test", "string", 0.0, 0, 0)
        assert col.uniqueness_ratio == 0.0  # Handle division by zero

    def test_primary_key_detection(self):
        """Test primary key detection heuristics"""
        # Perfect primary key
        col = ColumnInfo(
            name="user_id",
            type="string",
            null_percentage=0.0,
            unique_count=100,
            total_count=100,
        )
        assert col.is_likely_primary_key is True
        assert col.uniqueness_ratio == 1.0

        # Not a primary key - has nulls
        col = ColumnInfo(
            name="user_id",
            type="string",
            null_percentage=5.0,
            unique_count=100,
            total_count=100,
        )
        assert col.is_likely_primary_key is False

        # Not a primary key - not unique
        col = ColumnInfo(
            name="user_id",
            type="string",
            null_percentage=0.0,
            unique_count=95,
            total_count=100,
        )
        assert col.is_likely_primary_key is False

        # Not a primary key - doesn't end with _id
        col = ColumnInfo(
            name="username",
            type="string",
            null_percentage=0.0,
            unique_count=100,
            total_count=100,
        )
        assert col.is_likely_primary_key is False

    def test_foreign_key_detection(self):
        """Test foreign key detection heuristics"""
        # Good foreign key candidate
        col = ColumnInfo(
            name="customer_id",
            type="string",
            null_percentage=0.0,
            unique_count=50,
            total_count=100,
        )
        assert col.is_likely_foreign_key is True
        assert col.uniqueness_ratio == 0.5

        # Too unique to be foreign key
        col = ColumnInfo(
            name="customer_id",
            type="string",
            null_percentage=0.0,
            unique_count=100,
            total_count=100,
        )
        assert col.is_likely_foreign_key is False

        # Too few unique values
        col = ColumnInfo(
            name="customer_id",
            type="string",
            null_percentage=0.0,
            unique_count=5,
            total_count=100,
        )
        assert col.is_likely_foreign_key is False

        # Doesn't end with _id
        col = ColumnInfo(
            name="customer_name",
            type="string",
            null_percentage=0.0,
            unique_count=50,
            total_count=100,
        )
        assert col.is_likely_foreign_key is False


class TestPaginationInfo:
    """Test PaginationInfo and detection logic"""

    def test_basic_pagination_info(self):
        """Test basic pagination info creation"""
        pag = PaginationInfo(type="page", total_records=1000, page_size=25, has_more=True)
        assert pag.type == "page"
        assert pag.total_records == 1000
        assert pag.page_size == 25
        assert pag.has_more is True

    def test_detect_page_pagination(self):
        """Test detection of page-based pagination"""
        response = {
            "data": [...],
            "pagination": {"total": 1000, "per_page": 50, "has_more": True},
        }
        pag = PaginationInfo.detect_pagination(response)
        assert pag.type == "page"
        assert pag.total_records == 1000
        assert pag.page_size == 50
        assert pag.has_more is True

    def test_detect_cursor_pagination(self):
        """Test detection of cursor-based pagination"""
        response = {"data": [...], "next": "cursor_token_123", "has_more": True}
        pag = PaginationInfo.detect_pagination(response)
        assert pag.type == "cursor"
        assert pag.has_more is True

    def test_detect_offset_pagination(self):
        """Test detection of offset-based pagination"""
        response = {"data": [...], "offset": 0, "limit": 100}
        pag = PaginationInfo.detect_pagination(response)
        assert pag.type == "offset"

    def test_detect_no_pagination(self):
        """Test detection when no pagination is present"""
        response = {"data": [...]}
        pag = PaginationInfo.detect_pagination(response)
        assert pag.type == "none"


class TestRelationship:
    """Test Relationship dataclass and properties"""

    def test_basic_relationship(self):
        """Test basic relationship creation"""
        rel = Relationship(
            source_a="users",
            source_b="orders",
            type="foreign_key",
            confidence=0.85,
            field_a="user_id",
            field_b="customer_id",
            description="Strong foreign key relationship",
        )
        assert rel.source_a == "users"
        assert rel.source_b == "orders"
        assert rel.type == "foreign_key"
        assert rel.confidence == 0.85
        assert rel.field_a == "user_id"
        assert rel.field_b == "customer_id"

    def test_high_confidence_property(self):
        """Test high confidence detection"""
        rel = Relationship("a", "b", "fk", 0.9, "id", "id", "test")
        assert rel.is_high_confidence is True

        rel = Relationship("a", "b", "fk", 0.7, "id", "id", "test")
        assert rel.is_high_confidence is False

    def test_relationship_strength(self):
        """Test relationship strength categorization"""
        rel = Relationship("a", "b", "fk", 0.95, "id", "id", "test")
        assert rel.relationship_strength == "Very Strong"

        rel = Relationship("a", "b", "fk", 0.8, "id", "id", "test")
        assert rel.relationship_strength == "Strong"

        rel = Relationship("a", "b", "fk", 0.6, "id", "id", "test")
        assert rel.relationship_strength == "Moderate"

        rel = Relationship("a", "b", "fk", 0.3, "id", "id", "test")
        assert rel.relationship_strength == "Weak"


class TestSourceAnalysis:
    """Test SourceAnalysis base class"""

    def test_basic_source_analysis(self):
        """Test basic source analysis creation"""
        analysis = SourceAnalysis(name="test_source", type=SourceType.CSV, status=AnalysisStatus.SUCCESS)
        assert analysis.name == "test_source"
        assert analysis.type == SourceType.CSV
        assert analysis.status == AnalysisStatus.SUCCESS
        assert analysis.success is True
        assert isinstance(analysis.analysis_timestamp, datetime)

    def test_failed_analysis(self):
        """Test failed analysis"""
        analysis = SourceAnalysis(
            name="failed_source",
            type=SourceType.REST,
            status=AnalysisStatus.FAILED,
            error_message="Connection timeout",
            error_type="network",
            error_hint="Check network connectivity",
        )
        assert analysis.success is False
        assert analysis.error_message == "Connection timeout"
        assert analysis.error_type == "network"
        assert analysis.error_hint == "Check network connectivity"

    def test_add_recommendations(self):
        """Test adding LLM recommendations and processing notes"""
        analysis = SourceAnalysis("test", SourceType.CSV, AnalysisStatus.SUCCESS)

        analysis.add_llm_recommendation("Use sampling for large datasets")
        analysis.add_llm_recommendation("Consider indexing key columns")
        analysis.add_llm_recommendation("Use sampling for large datasets")  # Duplicate

        assert len(analysis.llm_recommendations) == 2
        assert "Use sampling for large datasets" in analysis.llm_recommendations
        assert "Consider indexing key columns" in analysis.llm_recommendations

        analysis.add_processing_note("Primary key detected")
        analysis.add_processing_note("Foreign key relationships found")
        analysis.add_processing_note("Primary key detected")  # Duplicate

        assert len(analysis.processing_notes) == 2
        assert "Primary key detected" in analysis.processing_notes
        assert "Foreign key relationships found" in analysis.processing_notes


class TestCSVAnalysis:
    """Test CSV-specific analysis"""

    def test_csv_analysis_post_init(self):
        """Test CSV post-initialization logic"""
        # Create a CSV analysis with large dataset
        size_info = SizeInfo(file_size_mb=15.0)  # Large dataset
        columns = [
            ColumnInfo("user_id", "string", 0.0, 100, 100),  # Primary key
            ColumnInfo("name", "string", 5.0, 95, 100),
        ]
        schema_info = SchemaInfo(columns=columns)

        analysis = CSVAnalysis(
            name="large_csv",
            type=SourceType.CSV,
            status=AnalysisStatus.SUCCESS,
            size_info=size_info,
            schema_info=schema_info,
        )

        # Should have large dataset recommendation
        assert any("contextual_filtering" in rec for rec in analysis.llm_recommendations)

        # Should have primary key note
        assert any("Primary key detected: user_id" in note for note in analysis.processing_notes)


class TestRESTAnalysis:
    """Test REST API-specific analysis"""

    def test_rest_analysis_post_init(self):
        """Test REST post-initialization logic"""
        # Create REST analysis with response paths and pagination
        pagination_info = PaginationInfo("page", 1000, 50, True)
        schema_info = SchemaInfo(response_paths=["data", "results.items"], pagination_info=pagination_info)

        analysis = RESTAnalysis(
            name="api_source",
            type=SourceType.REST,
            status=AnalysisStatus.SUCCESS,
            schema_info=schema_info,
            authentication_success=True,
        )

        # Should have response_path recommendation
        assert any("response_path: 'data'" in rec for rec in analysis.llm_recommendations)

        # Should have pagination note
        assert any("1,000 total records" in note for note in analysis.processing_notes)


class TestIntrospectionResult:
    """Test IntrospectionResult aggregation logic"""

    def test_result_calculations(self):
        """Test result summary calculations"""
        # Create mock analyses
        analyses = {
            "source1": SourceAnalysis("source1", SourceType.CSV, AnalysisStatus.SUCCESS),
            "source2": SourceAnalysis("source2", SourceType.REST, AnalysisStatus.FAILED),
            "source3": SourceAnalysis("source3", SourceType.JSON, AnalysisStatus.SUCCESS),
        }

        result = IntrospectionResult(analyses=analyses, relationships=[])

        assert result.total_count == 3
        assert result.success_count == 2
        assert result.failure_count == 1
        assert abs(result.success_rate - 66.67) < 0.1  # Approximately 66.67%

        assert len(result.successful_analyses) == 2
        assert len(result.failed_analyses) == 1
        assert result.has_relationships is False

    def test_result_with_relationships(self):
        """Test result with relationships"""
        analyses = {
            "users": SourceAnalysis("users", SourceType.CSV, AnalysisStatus.SUCCESS),
            "orders": SourceAnalysis("orders", SourceType.REST, AnalysisStatus.SUCCESS),
        }
        relationships = [
            Relationship(
                "users",
                "orders",
                "foreign_key",
                0.9,
                "user_id",
                "customer_id",
                "FK relationship",
            )
        ]

        result = IntrospectionResult(analyses=analyses, relationships=relationships)
        assert result.has_relationships is True
        assert len(result.relationships) == 1

    def test_add_global_recommendations(self):
        """Test adding global recommendations"""
        result = IntrospectionResult(analyses={}, relationships=[])

        result.add_global_recommendation("Use sampling for performance")
        result.add_global_recommendation("Consider caching API responses")
        result.add_global_recommendation("Use sampling for performance")  # Duplicate

        assert len(result.global_recommendations) == 2
        assert "Use sampling for performance" in result.global_recommendations
        assert "Consider caching API responses" in result.global_recommendations

    def test_finalize_result(self):
        """Test finalizing result with global recommendations"""
        # Create analyses with large datasets
        large_size = SizeInfo(file_size_mb=20.0)  # Large
        analyses = {
            "large_csv": SourceAnalysis(
                "large_csv",
                SourceType.CSV,
                AnalysisStatus.SUCCESS,
                size_info=large_size,
            ),
            "small_json": SourceAnalysis("small_json", SourceType.JSON, AnalysisStatus.SUCCESS),
        }
        relationships = [Relationship("large_csv", "small_json", "fk", 0.8, "id", "csv_id", "test")]

        result = IntrospectionResult(analyses=analyses, relationships=relationships)
        result.finalize()

        # Should have timing information
        assert result.end_time is not None
        assert result.total_duration_ms is not None
        assert result.total_duration_ms >= 0

        # Should have global recommendations
        assert any("Large datasets detected" in rec for rec in result.global_recommendations)
        assert any("relationship_highlighting" in rec for rec in result.global_recommendations)
