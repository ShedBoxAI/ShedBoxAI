"""
CSV data source analyzer with comprehensive statistical analysis.

This module provides CSV file analysis capabilities including schema detection,
statistical summaries, and LLM-optimized recommendations.
"""

import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..models import AnalysisStatus, ColumnInfo, CSVAnalysis, SchemaInfo, SizeInfo, SourceType
from .base import FileAnalyzer


class CSVAnalyzer(FileAnalyzer):
    """Analyzer for CSV data sources with comprehensive statistical analysis"""

    @property
    def supported_type(self) -> SourceType:
        return SourceType.CSV

    def analyze(self, source_config: Dict[str, Any], sample_size: int = 100) -> CSVAnalysis:
        """
        Analyze CSV data source with statistical summaries and LLM optimization.

        Args:
            source_config: Source configuration containing 'path' or 'data'
            sample_size: Number of records to sample for analysis

        Returns:
            CSVAnalysis: Complete analysis results
        """
        source_name = source_config.get("name", "unknown")

        try:
            # Handle inline data vs file path
            if "data" in source_config:
                df = self._load_inline_data(source_config["data"])
                file_path = None
            else:
                file_path = source_config.get("path")
                if not file_path:
                    raise ValueError("CSV source must have 'path' or 'data' field")
                df = self._load_csv_file(file_path, source_config.get("options", {}))

            # Create base analysis object
            analysis = CSVAnalysis(name=source_name, type=SourceType.CSV, status=AnalysisStatus.SUCCESS)

            # Set CSV-specific metadata
            analysis.has_header = True  # Assume headers by default
            analysis.column_count = len(df.columns)

            if file_path:
                analysis.delimiter = self._detect_delimiter(file_path)
                analysis.encoding = self._detect_encoding(file_path)

            # Generate size information
            analysis.size_info = self._analyze_size(df, file_path)

            # Generate schema information
            analysis.schema_info = self._analyze_schema(df)

            # Generate sample data for LLM context
            analysis.sample_data = self._generate_sample_data(df, sample_size)

            # Add LLM-specific recommendations
            self._add_llm_recommendations(analysis, df)

            return analysis

        except Exception as e:
            return self._handle_analysis_error(source_name, e)

    def _load_csv_file(self, file_path: str, options: Dict[str, Any]) -> pd.DataFrame:
        """Load CSV file with proper error handling and options"""
        try:
            # Default pandas options
            default_options = {
                "encoding": "utf-8",
                "delimiter": ",",
                "low_memory": False,  # Better type inference
                "parse_dates": True,  # Auto-detect dates
            }

            # Merge with user options
            pandas_options = {**default_options, **options}

            # Special handling for encoding detection
            if "encoding" not in options:
                pandas_options["encoding"] = self._detect_encoding(file_path)

            df = pd.read_csv(file_path, **pandas_options)

            if df.empty:
                raise ValueError("CSV file is empty")

            return df

        except UnicodeDecodeError as e:
            # Try alternative encodings
            fallback_encodings = ["latin-1", "cp1252", "iso-8859-1"]
            for encoding in fallback_encodings:
                try:
                    options_copy = pandas_options.copy()
                    options_copy["encoding"] = encoding
                    return pd.read_csv(file_path, **options_copy)
                except UnicodeDecodeError:
                    continue
            raise ValueError(f"Could not decode CSV file with any encoding: {str(e)}")

        except pd.errors.EmptyDataError:
            raise ValueError("CSV file is empty or has no data")
        except pd.errors.ParserError as e:
            raise ValueError(f"CSV parsing error: {str(e)}")

    def _load_inline_data(self, data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Load data from inline configuration"""
        if not data:
            raise ValueError("Inline data is empty")

        try:
            df = pd.DataFrame(data)
            return df
        except Exception as e:
            raise ValueError(f"Could not create DataFrame from inline data: {str(e)}")

    def _detect_delimiter(self, file_path: str) -> str:
        """Detect CSV delimiter by reading first few lines"""
        try:
            import csv

            with open(file_path, "r", encoding="utf-8") as f:
                first_line = f.readline()
                # Use csv.Sniffer to detect delimiter
                sniffer = csv.Sniffer()
                delimiter = sniffer.sniff(first_line).delimiter
                return delimiter
        except Exception:
            return ","  # Default to comma

    def _analyze_size(self, df: pd.DataFrame, file_path: Optional[str] = None) -> SizeInfo:
        """Analyze size characteristics for LLM optimization"""
        record_count = len(df)

        # File size information
        file_size_mb = None
        if file_path:
            try:
                file_size_bytes = os.path.getsize(file_path)
                file_size_mb = file_size_bytes / (1024 * 1024)
            except Exception:
                pass

        # Memory size estimation
        memory_size_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

        # Token estimation for LLM context
        estimated_tokens = self._estimate_token_count(df)

        return SizeInfo(
            record_count=record_count,
            file_size_mb=file_size_mb,
            memory_size_mb=memory_size_mb,
            estimated_tokens=estimated_tokens,
        )

    def _estimate_token_count(self, df: pd.DataFrame) -> int:
        """Estimate token count for LLM context planning"""
        # Rough estimation: average 1 token per 4 characters
        total_chars = 0

        # Sample first 100 rows for estimation
        sample_df = df.head(100)

        for col in sample_df.columns:
            # Column name tokens
            total_chars += len(str(col)) * len(df)

            # Data tokens (estimate from sample)
            col_data = sample_df[col].astype(str)
            avg_length = col_data.str.len().mean()
            total_chars += avg_length * len(df)

        # Add structure overhead (JSON-like format)
        structure_overhead = len(df) * 20  # Rough estimate for JSON structure
        total_chars += structure_overhead

        return int(total_chars / 4)  # 4 chars per token approximation

    def _analyze_schema(self, df: pd.DataFrame) -> SchemaInfo:
        """Perform comprehensive schema analysis"""
        columns = []

        for col_name in df.columns:
            col_info = self._analyze_column(df, col_name)
            columns.append(col_info)

        # Generate statistical summary using pandas describe
        statistical_summary = self._generate_statistical_summary(df)

        return SchemaInfo(
            columns=columns,
            statistical_summary=statistical_summary,
            nested_levels=0,  # CSV is flat
            has_arrays=False,
            has_objects=False,
        )

    def _analyze_column(self, df: pd.DataFrame, col_name: str) -> ColumnInfo:
        """Analyze individual column characteristics"""
        series = df[col_name]
        total_count = len(series)
        null_count = series.isnull().sum()
        null_percentage = (null_count / total_count) * 100 if total_count > 0 else 0

        # Enhanced type detection
        detected_type = self._detect_enhanced_type(series)

        # Unique value analysis
        unique_values = series.dropna().unique()
        unique_count = len(unique_values)

        # Sample values (up to 5, diverse selection)
        sample_values = self._get_representative_samples(series, max_samples=5)

        # Create base column info
        col_info = ColumnInfo(
            name=col_name,
            type=detected_type,
            null_percentage=null_percentage,
            unique_count=unique_count,
            total_count=total_count,
            sample_values=sample_values,
        )

        # Add type-specific analysis
        if detected_type in ["integer", "float"]:
            self._add_numeric_analysis(col_info, series)
        elif detected_type == "string":
            self._add_string_analysis(col_info, series)

        # Categorical analysis
        self._add_categorical_analysis(col_info, series)

        # Identifier detection
        col_info.is_identifier = self._detect_identifier_pattern(col_name, series)

        return col_info

    def _detect_enhanced_type(self, series: pd.Series) -> str:
        """Enhanced type detection beyond pandas dtypes"""
        dtype_str = str(series.dtype)

        # Handle pandas dtypes
        if "int" in dtype_str:
            return "integer"
        elif "float" in dtype_str:
            return "float"
        elif "bool" in dtype_str:
            return "boolean"
        elif "datetime" in dtype_str:
            return "date"
        elif dtype_str == "object":
            # Further analysis for object type
            non_null_series = series.dropna()
            if len(non_null_series) == 0:
                return "string"

            # Check if it's actually numeric but stored as string
            try:
                pd.to_numeric(non_null_series.head(100))
                return "numeric_string"  # Numbers stored as strings
            except (ValueError, TypeError):
                pass

            # Check for date patterns
            try:
                import warnings

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    parsed = pd.to_datetime(non_null_series.head(10), format="mixed", errors="coerce")
                    # Only consider it a date if most values parsed successfully (not NaT)
                    valid_count = parsed.notna().sum()
                    if valid_count >= len(parsed) * 0.8:  # At least 80% valid dates
                        return "date_string"
            except (ValueError, TypeError):
                pass

            return "string"
        else:
            return "unknown"

    def _add_numeric_analysis(self, col_info: ColumnInfo, series: pd.Series):
        """Add statistical analysis for numeric columns"""
        numeric_series = pd.to_numeric(series, errors="coerce").dropna()

        if len(numeric_series) > 0:
            col_info.min_value = float(numeric_series.min())
            col_info.max_value = float(numeric_series.max())
            col_info.mean = float(numeric_series.mean())
            col_info.std = float(numeric_series.std()) if len(numeric_series) > 1 else 0.0
            col_info.median = float(numeric_series.median())

            # Quartiles
            quartiles = numeric_series.quantile([0.25, 0.5, 0.75]).tolist()
            col_info.quartiles = [float(q) for q in quartiles]

    def _add_string_analysis(self, col_info: ColumnInfo, series: pd.Series):
        """Add analysis for string columns"""
        string_series = series.dropna().astype(str)

        if len(string_series) > 0:
            lengths = string_series.str.len()
            col_info.avg_length = float(lengths.mean())
            col_info.max_length = int(lengths.max())

            # Pattern detection for IDs, codes, etc.
            col_info.pattern = self._detect_string_pattern(string_series)

    def _add_categorical_analysis(self, col_info: ColumnInfo, series: pd.Series):
        """Determine if column is categorical and analyze value distribution"""
        unique_ratio = col_info.unique_count / col_info.total_count if col_info.total_count > 0 else 0

        # Consider categorical if:
        # 1. Low unique ratio (< 0.1) OR
        # 2. Small number of unique values (< 20) for strings
        is_categorical = (unique_ratio < 0.1) or (col_info.unique_count < 20 and col_info.type == "string")

        if is_categorical:
            col_info.is_categorical = True
            # Get value counts for top values
            value_counts = series.value_counts().head(10)
            col_info.value_counts = value_counts.to_dict()

    def _detect_identifier_pattern(self, col_name: str, series: pd.Series) -> bool:
        """Detect if column is likely an identifier"""
        name_lower = col_name.lower()

        # Check name patterns
        id_patterns = ["_id", "id_", "key", "uuid", "guid"]
        if any(pattern in name_lower for pattern in id_patterns):
            return True

        # Check data patterns (if string)
        if series.dtype == "object":
            sample_values = series.dropna().head(10).astype(str)
            if len(sample_values) > 0:
                # Look for common ID patterns
                first_value = sample_values.iloc[0]
                if any(pattern in first_value.upper() for pattern in ["ID", "KEY", "UUID"]):
                    return True

        return False

    def _detect_string_pattern(self, string_series: pd.Series) -> Optional[str]:
        """Detect common string patterns"""
        if len(string_series) == 0:
            return None

        sample = string_series.head(10)

        # Email pattern
        email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        if sample.str.match(email_pattern).all():
            return "email"

        # Date patterns - check BEFORE phone to avoid false positives
        # ISO format: 2024-01-15, 2024-01-15T10:30:00
        iso_date_pattern = r"^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}:\d{2})?.*$"
        if sample.str.match(iso_date_pattern).all():
            return "date"

        # US date format: 01/15/2024, 1/15/24
        us_date_pattern = r"^\d{1,2}/\d{1,2}/\d{2,4}$"
        if sample.str.match(us_date_pattern).all():
            return "date"

        # Phone pattern - more restrictive to avoid matching dates
        # Requires parentheses OR starts with + OR has specific phone structure
        phone_pattern = r"^(\+\d{1,3}[\s-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}$"
        if sample.str.match(phone_pattern).all():
            return "phone"

        # ID patterns like "CUST_001", "USER_123"
        id_pattern = r"^[A-Z]+_\d+$"
        if sample.str.match(id_pattern).all():
            return "structured_id"

        # UUID pattern
        uuid_pattern = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
        if sample.str.match(uuid_pattern, case=False).all():
            return "uuid"

        return None

    def _get_representative_samples(self, series: pd.Series, max_samples: int = 5) -> List[Any]:
        """Get diverse, representative sample values"""
        non_null_series = series.dropna()

        if len(non_null_series) == 0:
            return []

        if len(non_null_series) <= max_samples:
            return non_null_series.tolist()

        # For categorical data, get most common values
        if len(non_null_series.unique()) < max_samples * 2:
            return non_null_series.value_counts().head(max_samples).index.tolist()

        # For continuous data, get diverse samples
        # Use quantiles to get spread across the range
        if pd.api.types.is_numeric_dtype(non_null_series):
            quantiles = [0, 0.25, 0.5, 0.75, 1.0][:max_samples]
            samples = non_null_series.quantile(quantiles).tolist()
            return [float(x) if pd.api.types.is_numeric_dtype(type(x)) else x for x in samples]

        # For strings, get samples from different parts of the dataset
        indices = np.linspace(0, len(non_null_series) - 1, max_samples, dtype=int)
        return non_null_series.iloc[indices].tolist()

    def _generate_statistical_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive statistical summary"""
        summary = {}

        # Overall dataset statistics
        summary["total_rows"] = len(df)
        summary["total_columns"] = len(df.columns)
        summary["memory_usage_mb"] = df.memory_usage(deep=True).sum() / (1024 * 1024)

        # Numeric columns summary
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            numeric_summary = df[numeric_cols].describe()
            summary["numeric_summary"] = numeric_summary.to_dict()

        # Categorical columns summary
        categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
        categorical_summary = {}
        for col in categorical_cols[:10]:  # Limit to avoid huge summaries
            value_counts = df[col].value_counts().head(5)
            categorical_summary[col] = {
                "unique_count": df[col].nunique(),
                "top_values": value_counts.to_dict(),
            }
        summary["categorical_summary"] = categorical_summary

        # Missing data summary
        missing_data = df.isnull().sum()
        missing_summary = missing_data[missing_data > 0].to_dict()
        summary["missing_data"] = missing_summary

        # Data type distribution
        dtype_counts = df.dtypes.value_counts().to_dict()
        summary["dtype_distribution"] = {str(k): v for k, v in dtype_counts.items()}

        return summary

    def _generate_sample_data(self, df: pd.DataFrame, sample_size: int) -> List[Dict[str, Any]]:
        """Generate representative sample data for LLM context"""
        if len(df) == 0:
            return []

        # For small datasets, return everything (up to sample_size)
        if len(df) <= sample_size:
            sample_df = df
        else:
            # For large datasets, use stratified sampling if possible
            sample_df = self._stratified_sample(df, sample_size)

        # Convert to list of dictionaries
        try:
            # Handle different data types properly
            sample_data = []
            for _, row in sample_df.iterrows():
                row_dict = {}
                for col, value in row.items():
                    # Convert pandas/numpy types to JSON-serializable types
                    if pd.isna(value):
                        row_dict[col] = None
                    elif isinstance(value, (np.integer, np.int64)):
                        row_dict[col] = int(value)
                    elif isinstance(value, (np.floating, np.float64)):
                        row_dict[col] = float(value)
                    elif isinstance(value, np.bool_):
                        row_dict[col] = bool(value)
                    elif isinstance(value, pd.Timestamp):
                        row_dict[col] = value.isoformat()
                    else:
                        row_dict[col] = str(value)

                sample_data.append(row_dict)

            return sample_data
        except Exception as e:
            self.logger.warning(f"Could not generate sample data: {str(e)}")
            return []

    def _stratified_sample(self, df: pd.DataFrame, sample_size: int) -> pd.DataFrame:
        """Attempt stratified sampling for better representation"""
        try:
            # Find a good categorical column for stratification
            categorical_cols = df.select_dtypes(include=["object"]).columns
            stratify_col = None

            for col in categorical_cols:
                unique_count = df[col].nunique()
                # Good stratification column: 2-20 unique values
                if 2 <= unique_count <= 20:
                    stratify_col = col
                    break

            if stratify_col:
                # Stratified sampling
                sample_dfs = []
                per_group_size = max(1, sample_size // df[stratify_col].nunique())

                for group_value in df[stratify_col].unique():
                    group_df = df[df[stratify_col] == group_value]
                    group_sample_size = min(len(group_df), per_group_size)
                    group_sample = group_df.sample(n=group_sample_size, random_state=42)
                    sample_dfs.append(group_sample)

                stratified_sample = pd.concat(sample_dfs, ignore_index=True)

                # If we don't have enough samples, add random samples
                if len(stratified_sample) < sample_size:
                    remaining_size = sample_size - len(stratified_sample)
                    remaining_df = df.drop(stratified_sample.index)
                    if len(remaining_df) > 0:
                        additional_sample = remaining_df.sample(
                            n=min(len(remaining_df), remaining_size), random_state=42
                        )
                        stratified_sample = pd.concat([stratified_sample, additional_sample], ignore_index=True)

                return stratified_sample.head(sample_size)

        except Exception:
            pass  # Fall back to random sampling

        # Random sampling fallback
        return df.sample(n=min(len(df), sample_size), random_state=42)

    def _add_llm_recommendations(self, analysis: CSVAnalysis, df: pd.DataFrame):
        """Add LLM-specific processing recommendations"""
        # Size-based recommendations
        if analysis.size_info.is_large_dataset:
            analysis.add_llm_recommendation("Large dataset detected - use contextual_filtering for data reduction")
            analysis.add_llm_recommendation(
                "Consider advanced_operations for aggregation instead of processing all records"
            )

        if analysis.size_info.context_window_warning:
            analysis.add_llm_recommendation("Dataset may exceed LLM context window - use sampling operations")

        # Schema-based recommendations
        if analysis.schema_info and analysis.schema_info.columns:
            # Find potential key columns
            key_columns = [
                col for col in analysis.schema_info.columns if col.is_likely_primary_key or col.is_likely_foreign_key
            ]

            if key_columns:
                key_names = [col.name for col in key_columns]
                analysis.add_llm_recommendation(
                    f"Key columns detected ({', '.join(key_names)}) - suitable for relationship operations"
                )

            # Find numeric columns for aggregation
            numeric_columns = [col for col in analysis.schema_info.columns if col.type in ["integer", "float"]]

            if numeric_columns:
                numeric_names = [col.name for col in numeric_columns]
                analysis.add_llm_recommendation(
                    f"Numeric columns available ({', '.join(numeric_names)}) - suitable for statistical operations"
                )

            # Find categorical columns for grouping
            categorical_columns = [col for col in analysis.schema_info.columns if col.is_categorical]

            if categorical_columns:
                cat_names = [col.name for col in categorical_columns]
                analysis.add_llm_recommendation(
                    f"Categorical columns available ({', '.join(cat_names)}) - suitable for grouping operations"
                )

        # Processing suggestions
        analysis.add_processing_note(f"CSV contains {len(df)} records with {len(df.columns)} columns")

        if analysis.delimiter and analysis.delimiter != ",":
            analysis.add_processing_note(f"Custom delimiter detected: '{analysis.delimiter}'")

        if analysis.encoding and analysis.encoding.lower() != "utf-8":
            analysis.add_processing_note(f"Non-UTF-8 encoding detected: {analysis.encoding}")

    def _handle_analysis_error(self, source_name: str, error: Exception) -> CSVAnalysis:
        """Handle analysis errors and return failed analysis"""
        error_type = self._classify_error(error)
        error_message = str(error)
        error_hint = self._generate_error_hint(error, {"name": source_name})

        return CSVAnalysis(
            name=source_name,
            type=SourceType.CSV,
            status=AnalysisStatus.FAILED,
            error_message=error_message,
            error_hint=error_hint,
            error_type=error_type,
        )
