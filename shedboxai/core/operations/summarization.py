"""
Content summarization operation handler.

This module provides statistical and AI-powered summarization
capabilities for data analysis.
"""

from typing import Any, Dict

from ..config.models import ContentSummarizationConfig
from .base import OperationHandler


class ContentSummarizationHandler(OperationHandler):
    """Handler for content summarization operations."""

    @property
    def operation_name(self) -> str:
        return "content_summarization"

    def process(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply content summarization to the data.

        Args:
            data: Input data dictionary
            config: Normalized summarization configuration (source_name -> ContentSummarizationConfig)

        Returns:
            Data dictionary with summarization results
        """
        result = data.copy()

        # Process each source in the config
        for source_name, summary_config in config.items():
            if source_name not in result:
                continue

            source_data = result[source_name]
            if not isinstance(source_data, list):
                continue

            # Ensure summary_config is a ContentSummarizationConfig
            if not isinstance(summary_config, ContentSummarizationConfig):
                if isinstance(summary_config, dict):
                    try:
                        summary_config = ContentSummarizationConfig(**summary_config)
                    except Exception as e:
                        self._log_warning(f"Invalid summarization configuration for '{source_name}': {e}")
                        continue
                else:
                    self._log_warning(
                        f"Invalid summarization configuration for '{source_name}': "
                        f"expected dict or ContentSummarizationConfig, got {type(summary_config)}"
                    )
                    continue

            # Apply summarization based on method
            if summary_config.method == "statistical":
                summary = self._statistical_summarization(source_data, summary_config)
                result[f"{source_name}_summary"] = summary
            elif summary_config.method == "ai":
                self._log_warning(f"AI summarization method is no longer supported for '{source_name}'")
            else:
                self._log_warning(f"Unknown or unsupported summarization method: {summary_config.method}")

        return result

    def _statistical_summarization(self, source_data: list, config: ContentSummarizationConfig) -> Dict[str, Any]:
        """
        Perform statistical summarization on data.

        Args:
            source_data: List of data items to summarize
            config: Summarization configuration

        Returns:
            Statistical summary dictionary
        """
        summary = {}

        for field in config.fields:
            values = [item.get(field) for item in source_data if isinstance(item, dict) and item.get(field) is not None]
            if not values:
                continue

            # Convert to numeric values where possible
            numeric_values = []
            for value in values:
                try:
                    if isinstance(value, (int, float)):
                        numeric_values.append(float(value))
                    elif isinstance(value, str):
                        numeric_values.append(float(value))
                except (ValueError, TypeError):
                    continue

            # Calculate statistics
            for stat in config.summarize:
                if stat == "mean" and numeric_values:
                    summary[f"{field}_mean"] = sum(numeric_values) / len(numeric_values)
                elif stat == "min" and numeric_values:
                    summary[f"{field}_min"] = min(numeric_values)
                elif stat == "max" and numeric_values:
                    summary[f"{field}_max"] = max(numeric_values)
                elif stat == "count":
                    summary[f"{field}_count"] = len(values)
                elif stat == "sum" and numeric_values:
                    summary[f"{field}_sum"] = sum(numeric_values)
                elif stat == "median" and numeric_values:
                    sorted_values = sorted(numeric_values)
                    n = len(sorted_values)
                    if n % 2 == 0:
                        summary[f"{field}_median"] = (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2
                    else:
                        summary[f"{field}_median"] = sorted_values[n // 2]
                elif stat == "std" and numeric_values:
                    # BUG 1 & 2 FIX: Handle single value case and ensure correct calculation
                    if len(numeric_values) == 1:
                        # Standard deviation of single value is 0
                        summary[f"{field}_std"] = 0.0
                    elif len(numeric_values) > 1:
                        mean = sum(numeric_values) / len(numeric_values)
                        variance = sum((x - mean) ** 2 for x in numeric_values) / (len(numeric_values) - 1)
                        summary[f"{field}_std"] = variance**0.5
                elif stat == "unique":
                    summary[f"{field}_unique"] = len(set(values))

        return summary
