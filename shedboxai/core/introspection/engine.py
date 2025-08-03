"""
Main orchestrator for data source introspection process.

This module provides the IntrospectionEngine class that coordinates all analyzers,
handles authentication flows, manages errors, and produces the final IntrospectionResult.
"""

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import yaml

from .analyzers import CSVAnalyzer, JSONAnalyzer, RESTAnalyzer, TextAnalyzer, YAMLAnalyzer
from .models import AnalysisStatus, IntrospectionOptions, IntrospectionResult, Relationship, SourceAnalysis, SourceType
from .relationship_detector import RelationshipDetector


class IntrospectionEngine:
    """Main orchestrator for data source introspection process"""

    def __init__(self, config_path: str, options: IntrospectionOptions):
        self.config_path = config_path
        self.options = options
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize analyzers
        self.analyzers = {
            SourceType.CSV: CSVAnalyzer(),
            SourceType.JSON: JSONAnalyzer(),
            SourceType.YAML: YAMLAnalyzer(),
            SourceType.TEXT: TextAnalyzer(),
            SourceType.REST: RESTAnalyzer(),
        }

        # Initialize relationship detector
        self.relationship_detector = RelationshipDetector()

        # Loaded configuration
        self.config = None
        self.data_sources = {}
        self.token_sources = {}

    def run_introspection(self) -> IntrospectionResult:
        """
        Main entry point for introspection process.

        Returns:
            IntrospectionResult: Complete introspection results
        """
        start_time = time.time()

        try:
            self.logger.info(f"Starting introspection process with config: {self.config_path}")

            # Load and validate configuration
            self._load_configuration()

            # Prepare data sources and identify token flows
            self._prepare_data_sources()

            # Analyze all data sources
            analyses = self._analyze_all_sources()

            # Detect relationships between sources
            relationships = self._detect_relationships(analyses)

            # Create final result
            result = IntrospectionResult(analyses=analyses, relationships=relationships)

            # Add global recommendations
            self._add_global_recommendations(result, analyses, relationships)

            # Finalize result
            result.finalize()

            end_time = time.time()
            duration = (end_time - start_time) * 1000

            self.logger.info(
                f"Introspection completed in {duration:.1f}ms. "
                f"Success rate: {result.success_rate:.1f}% "
                f"({result.success_count}/{result.total_count})"
            )

            return result

        except Exception as e:
            self.logger.error(f"Introspection failed: {str(e)}")

            # Return error result
            error_result = IntrospectionResult(analyses={}, relationships=[])
            error_result.add_global_recommendation(f"Introspection failed: {str(e)}")
            error_result.finalize()

            return error_result

    def _load_configuration(self):
        """Load and validate the configuration file"""
        try:
            config_path = Path(self.config_path)

            if not config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

            with open(config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)

            if not self.config:
                raise ValueError("Configuration file is empty")

            if "data_sources" not in self.config:
                raise ValueError("Configuration must contain 'data_sources' section")

            self.logger.info(f"Loaded configuration with {len(self.config['data_sources'])} data sources")

        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in configuration file: {str(e)}")

    def _prepare_data_sources(self):
        """Prepare data sources and identify authentication flows"""
        raw_sources = self.config["data_sources"]

        # Process each data source and add metadata
        for name, source_config in raw_sources.items():
            # Add source name to config
            source_config["name"] = name

            # Validate required fields
            source_type = source_config.get("type")
            if not source_type:
                raise ValueError(f"Data source '{name}' missing 'type' field")

            try:
                SourceType(source_type)
            except ValueError:
                raise ValueError(f"Unsupported source type '{source_type}' for source '{name}'")

            # Categorize sources
            if source_config.get("is_token_source"):
                self.token_sources[name] = source_config
                self.logger.debug(f"Identified token source: {name}")

            self.data_sources[name] = source_config

        # Validate token dependencies
        self._validate_token_dependencies()

        self.logger.info(
            f"Prepared {len(self.data_sources)} data sources " f"({len(self.token_sources)} token sources)"
        )

    def _validate_token_dependencies(self):
        """Validate that token dependencies are correctly configured"""
        for name, source_config in self.data_sources.items():
            if source_config.get("requires_token"):
                token_source = source_config.get("token_source")

                if not token_source:
                    raise ValueError(f"Source '{name}' requires token but no token_source specified")

                if token_source not in self.token_sources:
                    raise ValueError(f"Token source '{token_source}' not found for source '{name}'")

            if source_config.get("is_token_source"):
                token_for = source_config.get("token_for", [])
                for dependent in token_for:
                    if dependent not in self.data_sources:
                        self.logger.warning(f"Token source '{name}' references non-existent source '{dependent}'")

    def _analyze_all_sources(self) -> Dict[str, SourceAnalysis]:
        """Analyze all data sources using appropriate analyzers"""
        analyses = {}

        # Handle retry logic
        sources_to_analyze = list(self.data_sources.keys())
        if self.options.retry_sources:
            sources_to_analyze = self.options.retry_sources
            self.logger.info(f"Retrying analysis for sources: {', '.join(sources_to_analyze)}")

        # First pass: Analyze token sources
        for name in sources_to_analyze:
            if name in self.token_sources:
                self.logger.info(f"Analyzing token source: {name}")
                analysis = self._analyze_single_source(name, self.data_sources[name])
                analyses[name] = analysis

        # Second pass: Analyze regular sources (with token context)
        for name in sources_to_analyze:
            if name not in self.token_sources:
                self.logger.info(f"Analyzing data source: {name}")
                analysis = self._analyze_single_source(name, self.data_sources[name])
                analyses[name] = analysis

        # Handle skip_errors option
        if not self.options.skip_errors:
            failed_sources = [name for name, analysis in analyses.items() if not analysis.success]
            if failed_sources:
                self.logger.warning(f"Failed sources: {', '.join(failed_sources)}")
                if not self.options.force_overwrite:
                    self.logger.info("Use --skip-errors to continue with partial results")

        return analyses

    def _analyze_single_source(self, name: str, source_config: Dict[str, Any]) -> SourceAnalysis:
        """Analyze a single data source with appropriate analyzer"""
        try:
            source_type = SourceType(source_config["type"])
            analyzer = self.analyzers[source_type]

            # Add token sources context for REST APIs (excluding self to avoid circular references)
            if source_type == SourceType.REST:
                # Don't include the current source in its own token sources to avoid circular references
                filtered_token_sources = {k: v for k, v in self.token_sources.items() if k != name}
                source_config["_token_sources"] = filtered_token_sources

            # Perform analysis
            self.logger.debug(f"Starting analysis of {name} ({source_type.value})")

            analysis = analyzer._safe_analyze(source_config, self.options.sample_size)

            if analysis.success:
                self.logger.info(f"✅ Successfully analyzed {name}")
            else:
                # Use user-friendly error message
                error_msg = analysis.error_message or "Analysis failed"
                self.logger.error(f"❌ Failed to analyze {name}: {error_msg}")

                if self.options.verbose and analysis.error_hint:
                    self.logger.info(f"💡 Hint: {analysis.error_hint}")

            return analysis

        except Exception as e:
            self.logger.error(f"Unexpected error analyzing {name}: {str(e)}")

            # Create error analysis
            error_analysis = SourceAnalysis(
                name=name,
                type=source_type,
                status=AnalysisStatus.FAILED,
                error_message=str(e),
                error_type="unexpected_error",
                error_hint="Check logs for more details",
            )

            return error_analysis

    def _detect_relationships(self, analyses: Dict[str, SourceAnalysis]) -> List[Relationship]:
        """Detect relationships between successfully analyzed sources"""
        try:
            self.logger.info("Detecting relationships between data sources...")

            relationships = self.relationship_detector.detect_relationships(analyses)

            if relationships:
                self.logger.info(f"Found {len(relationships)} relationships")

                # Log high-confidence relationships
                high_conf_rels = [r for r in relationships if r.confidence >= 0.8]
                for rel in high_conf_rels:
                    self.logger.info(
                        f"🔗 {rel.source_a}.{rel.field_a} ↔ {rel.source_b}.{rel.field_b} "
                        f"({rel.confidence:.0%} confidence, {rel.type})"
                    )
            else:
                self.logger.info("No relationships detected")

            return relationships

        except Exception as e:
            self.logger.error(f"Relationship detection failed: {str(e)}")
            return []

    def _add_global_recommendations(
        self,
        result: IntrospectionResult,
        analyses: Dict[str, SourceAnalysis],
        relationships: List[Relationship],
    ):
        """Add global recommendations based on overall analysis"""

        # Success rate recommendations
        if result.success_rate < 100:
            failed_count = result.failure_count
            result.add_global_recommendation(
                f"{failed_count} data source{'s' if failed_count > 1 else ''} failed analysis - "
                "check authentication and configuration"
            )

        # Authentication recommendations
        auth_failures = [
            name
            for name, analysis in analyses.items()
            if hasattr(analysis, "authentication_success") and not analysis.authentication_success
        ]

        if auth_failures:
            result.add_global_recommendation(
                f"Authentication failed for: {', '.join(auth_failures)} - " "verify API credentials and permissions"
            )

        # Data size recommendations
        large_datasets = [
            name
            for name, analysis in analyses.items()
            if analysis.success and analysis.size_info and analysis.size_info.is_large_dataset
        ]

        if large_datasets:
            result.add_global_recommendation(
                f"Large datasets detected ({', '.join(large_datasets)}) - "
                "use sampling and aggregation operations for LLM processing"
            )

        # Context window recommendations
        context_warnings = [
            name
            for name, analysis in analyses.items()
            if analysis.success and analysis.size_info and analysis.size_info.context_window_warning
        ]

        if context_warnings:
            result.add_global_recommendation(
                f"Datasets may exceed LLM context window ({', '.join(context_warnings)}) - "
                "use contextual_filtering to reduce data size"
            )

        # Relationship recommendations
        if relationships:
            rel_summary = self.relationship_detector.generate_relationship_summary(relationships)
            result.add_global_recommendation(
                f"{len(relationships)} data relationships detected - "
                "consider using relationship_highlighting operations"
            )

            # Add specific relationship recommendations
            for rec in rel_summary["recommendations"]:
                result.add_global_recommendation(rec)

        # Data type diversity recommendations
        successful_analyses = [a for a in analyses.values() if a.success]

        if len(successful_analyses) >= 2:
            source_types = set(a.type for a in successful_analyses)
            if len(source_types) > 1:
                result.add_global_recommendation(
                    f"Multiple data source types detected ({', '.join(t.value for t in source_types)}) - "
                    "ShedBoxAI can unify processing across different formats"
                )

        # API-specific recommendations
        api_sources = [
            name for name, analysis in analyses.items() if analysis.success and analysis.type == SourceType.REST
        ]

        if api_sources:
            result.add_global_recommendation(
                f"API sources detected ({', '.join(api_sources)}) - "
                "monitor rate limits and consider caching for production use"
            )

        # Configuration recommendations
        env_var_sources = []
        for name, analysis in analyses.items():
            if (hasattr(analysis, "has_env_variables") and analysis.has_env_variables) or (
                analysis.error_type == "missing_env_var"
            ):
                env_var_sources.append(name)

        if env_var_sources:
            result.add_global_recommendation(
                f"Environment variables required for: {', '.join(env_var_sources)} - "
                "ensure all variables are set before running pipelines"
            )

    def validate_existing_introspection(self, introspection_path: str) -> Dict[str, Any]:
        """Validate existing introspection file against current sources"""
        try:
            self.logger.info(f"Validating existing introspection: {introspection_path}")

            if not Path(introspection_path).exists():
                return {"valid": False, "reason": "Introspection file not found"}

            # Load and parse existing introspection
            with open(introspection_path, "r", encoding="utf-8") as f:
                existing_content = f.read()

            # Simple validation - check if sources mentioned in introspection
            # match current configuration
            self._load_configuration()
            current_sources = set(self.config["data_sources"].keys())

            # Extract source names from markdown (simple regex approach)
            import re

            source_pattern = r"### (\w+) \("
            mentioned_sources = set(re.findall(source_pattern, existing_content))

            missing_sources = current_sources - mentioned_sources
            extra_sources = mentioned_sources - current_sources

            validation_result = {
                "valid": len(missing_sources) == 0 and len(extra_sources) == 0,
                "current_sources": list(current_sources),
                "mentioned_sources": list(mentioned_sources),
                "missing_sources": list(missing_sources),
                "extra_sources": list(extra_sources),
            }

            if validation_result["valid"]:
                self.logger.info("✅ Existing introspection is up to date")
            else:
                self.logger.warning("⚠️ Existing introspection is outdated")
                if missing_sources:
                    self.logger.warning(f"Missing sources: {', '.join(missing_sources)}")
                if extra_sources:
                    self.logger.warning(f"Extra sources: {', '.join(extra_sources)}")

            return validation_result

        except Exception as e:
            self.logger.error(f"Validation failed: {str(e)}")
            return {"valid": False, "reason": str(e)}

    def get_analysis_summary(self, analyses: Dict[str, SourceAnalysis]) -> Dict[str, Any]:
        """Generate summary statistics for analyses"""
        total = len(analyses)
        successful = sum(1 for a in analyses.values() if a.success)
        failed = total - successful

        # Categorize by type
        by_type = {}
        for analysis in analyses.values():
            type_name = analysis.type.value
            if type_name not in by_type:
                by_type[type_name] = {"total": 0, "successful": 0, "failed": 0}

            by_type[type_name]["total"] += 1
            if analysis.success:
                by_type[type_name]["successful"] += 1
            else:
                by_type[type_name]["failed"] += 1

        # Error categorization
        error_types = {}
        for analysis in analyses.values():
            if not analysis.success and analysis.error_type:
                error_types[analysis.error_type] = error_types.get(analysis.error_type, 0) + 1

        # Size statistics
        size_stats = {
            "large_datasets": 0,
            "context_warnings": 0,
            "total_records": 0,
            "total_estimated_tokens": 0,
        }

        for analysis in analyses.values():
            if analysis.success and analysis.size_info:
                if analysis.size_info.is_large_dataset:
                    size_stats["large_datasets"] += 1
                if analysis.size_info.context_window_warning:
                    size_stats["context_warnings"] += 1
                if analysis.size_info.record_count:
                    size_stats["total_records"] += analysis.size_info.record_count
                if analysis.size_info.estimated_tokens:
                    size_stats["total_estimated_tokens"] += analysis.size_info.estimated_tokens

        return {
            "total_sources": total,
            "successful": successful,
            "failed": failed,
            "success_rate": (successful / total * 100) if total > 0 else 0,
            "by_type": by_type,
            "error_types": error_types,
            "size_statistics": size_stats,
        }

    def get_failed_sources(self, analyses: Dict[str, SourceAnalysis]) -> List[str]:
        """Get list of failed source names for retry logic"""
        return [name for name, analysis in analyses.items() if not analysis.success]

    def can_retry_source(self, source_name: str) -> bool:
        """Check if a source can be retried"""
        if source_name not in self.data_sources:
            return False

        source_config = self.data_sources[source_name]

        # Can't retry if fundamental config issues
        required_fields = {
            "csv": ["path"],
            "json": ["path"],
            "yaml": ["path"],
            "text": ["path"],
            "rest": ["url"],
        }

        source_type = source_config.get("type")
        if source_type in required_fields:
            for field in required_fields[source_type]:
                if field not in source_config and "data" not in source_config:
                    return False

        return True

    def estimate_analysis_time(self) -> float:
        """Estimate total analysis time in seconds"""
        base_time_per_source = 0.5  # Base time in seconds

        time_multipliers = {
            "csv": 1.0,
            "json": 0.8,
            "yaml": 0.5,
            "text": 0.7,
            "rest": 2.0,  # REST APIs take longer due to network calls
        }

        total_time = 0
        for source_config in self.data_sources.values():
            source_type = source_config.get("type", "csv")
            multiplier = time_multipliers.get(source_type, 1.0)
            total_time += base_time_per_source * multiplier

        # Add relationship detection time (scales with number of source pairs)
        n_sources = len(self.data_sources)
        if n_sources > 1:
            relationship_time = (n_sources * (n_sources - 1) / 2) * 0.1
            total_time += relationship_time

        return total_time

    def check_prerequisites(self) -> Dict[str, Any]:
        """Check prerequisites before running introspection"""
        issues = []
        warnings = []

        # Check configuration file
        try:
            self._load_configuration()
        except Exception as e:
            issues.append(f"Configuration error: {str(e)}")
            return {"valid": False, "issues": issues, "warnings": warnings}

        # Check environment variables
        missing_env_vars = set()

        def check_env_var_in_value(value):
            """Helper to check for environment variables in a value"""
            if isinstance(value, str):
                # Look for ${VAR_NAME} patterns anywhere in the string
                import re

                env_var_pattern = r"\$\{([^}]+)\}"
                matches = re.findall(env_var_pattern, value)
                for env_var in matches:
                    if not os.getenv(env_var):
                        missing_env_vars.add(env_var)
            elif isinstance(value, dict):
                for v in value.values():
                    check_env_var_in_value(v)
            elif isinstance(value, list):
                for v in value:
                    check_env_var_in_value(v)

        for name, source_config in self.config["data_sources"].items():
            # Check headers for env vars
            headers = source_config.get("headers", {})
            check_env_var_in_value(headers)

            # Check auth options
            options = source_config.get("options", {})
            check_env_var_in_value(options)

            # Check other fields that might contain env vars
            for field in ["url", "path", "username", "password"]:
                if field in source_config:
                    check_env_var_in_value(source_config[field])

        if missing_env_vars:
            issues.append(f"Missing environment variables: {', '.join(missing_env_vars)}")

        # Check file paths
        for name, source_config in self.config["data_sources"].items():
            if "path" in source_config:
                file_path = Path(source_config["path"])
                if not file_path.exists():
                    issues.append(f"File not found for source '{name}': {source_config['path']}")

        # Check for potential issues
        rest_sources = [name for name, config in self.config["data_sources"].items() if config.get("type") == "rest"]
        if len(rest_sources) > 5:
            warnings.append(f"Many REST API sources ({len(rest_sources)}) - consider rate limiting")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "missing_env_vars": list(missing_env_vars),
        }

    def recover_from_partial_failure(self, result: IntrospectionResult) -> IntrospectionResult:
        """Attempt to recover from partial failures"""
        if result.success_rate >= 50:  # At least half succeeded
            # Add recovery recommendations
            failed_sources = [name for name, analysis in result.analyses.items() if not analysis.success]

            for source_name in failed_sources:
                analysis = result.analyses[source_name]

                if hasattr(analysis, "error_type") and analysis.error_type:
                    if analysis.error_type == "missing_env_var":
                        result.add_global_recommendation(
                            f"Set missing environment variables for {source_name} and retry"
                        )
                    elif analysis.error_type == "auth":
                        result.add_global_recommendation(
                            f"Fix authentication for {source_name} and use --retry {source_name}"
                        )
                    elif analysis.error_type == "file_not_found":
                        result.add_global_recommendation(f"Check file path for {source_name} and update configuration")

        return result

    def suggest_configuration_improvements(self, analyses: Dict[str, SourceAnalysis]) -> List[str]:
        """Suggest configuration improvements based on analysis results"""
        suggestions = []

        # Suggest response_path for APIs with complex responses
        for name, analysis in analyses.items():
            if (
                analysis.success
                and analysis.type == SourceType.REST
                and hasattr(analysis, "schema_info")
                and analysis.schema_info
                and analysis.schema_info.response_paths
            ):
                best_path = analysis.schema_info.response_paths[0]
                suggestions.append(f"Add 'response_path: \"{best_path}\"' to {name} for better data extraction")

        # Suggest sampling for large datasets
        for name, analysis in analyses.items():
            if analysis.success and analysis.size_info and analysis.size_info.is_large_dataset:
                suggestions.append(
                    f"Consider adding sampling parameters to {name} configuration for better performance"
                )

        return suggestions
