"""
Configuration models for data processing operations.

This module contains all Pydantic models that define the structure
and validation rules for different processing operation configurations.
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator

from .ai_config import AIInterfaceConfig


class ContextualFilterConfig(BaseModel):
    """Configuration for contextual filtering operations."""

    field: str
    condition: str
    new_name: Optional[str] = None


class FormatConversionConfig(BaseModel):
    """Configuration for format conversion operations."""

    source: Optional[str] = None
    extract_fields: Optional[List[str]] = None
    template: Optional[str] = None

    def __str__(self) -> str:
        """String representation for debugging."""
        return (
            f"FormatConversionConfig(source={self.source}, extract_fields={self.extract_fields}, "
            f"template={self.template})"
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FormatConversionConfig":
        """Create instance from dictionary."""
        return cls(**data)


class ContentSummarizationConfig(BaseModel):
    """Configuration for content summarization operations."""

    method: str
    fields: List[str]
    summarize: List[str]


class RelationshipConfig(BaseModel):
    """Configuration for relationship highlighting operations."""

    link_fields: Optional[List[Dict[str, str]]] = None
    derived_fields: Optional[List[str]] = None
    jsonpath_links: Optional[List[Dict[str, str]]] = None
    pattern_detection: Optional[Dict[str, Any]] = None
    conditional_highlighting: Optional[List[Dict[str, Any]]] = None
    context_additions: Optional[Dict[str, str]] = None


class AdvancedOperationConfig(BaseModel):
    """Configuration for advanced data operations."""

    source: str
    group_by: Optional[str] = None
    aggregate: Optional[Dict[str, str]] = None
    sort: Optional[str] = None
    limit: Optional[int] = None


class TemplateMatchingConfig(BaseModel):
    """Configuration for template matching operations."""

    template: Optional[str] = None
    template_id: Optional[str] = None
    variables: Optional[Dict[str, Any]] = None

    @validator("template", "template_id")
    def check_template_or_id_required(cls, v, values):
        """Ensure either template or template_id is provided."""
        if not v and "template_id" not in values and "template" not in values:
            raise ValueError("Either template or template_id must be provided")
        return v


class GraphNode(BaseModel):
    """Configuration for a graph node in the processing pipeline."""

    id: str
    operation: str
    depends_on: List[str] = Field(default_factory=list)
    config_key: Optional[str] = None


class ProcessorConfig(BaseModel):
    """
    Main configuration for the data processor.

    Supports both direct source mapping and named configurations for all operations.
    Can be used with either linear pipeline or directed graph execution.
    """

    # Configuration for each operation type
    contextual_filtering: Optional[
        Dict[
            str,
            Union[List[ContextualFilterConfig], Dict[str, List[ContextualFilterConfig]]],
        ]
    ] = None
    format_conversion: Optional[
        Dict[
            str,
            Union[
                FormatConversionConfig,
                Dict[str, FormatConversionConfig],
                Dict[str, Any],
            ],
        ]
    ] = None
    content_summarization: Optional[
        Dict[
            str,
            Union[ContentSummarizationConfig, Dict[str, ContentSummarizationConfig]],
        ]
    ] = None
    relationship_highlighting: Optional[Dict[str, Union[RelationshipConfig, Dict[str, RelationshipConfig]]]] = None
    advanced_operations: Optional[Dict[str, Union[AdvancedOperationConfig, Dict[str, AdvancedOperationConfig]]]] = None
    template_matching: Optional[Dict[str, Union[TemplateMatchingConfig, Dict[str, TemplateMatchingConfig]]]] = None

    # AI Interface configuration
    ai_interface: Optional[AIInterfaceConfig] = None

    # Optional graph configuration for complex workflows
    graph: Optional[List[GraphNode]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProcessorConfig":
        """
        Create a ProcessorConfig from a dictionary.

        Handles special processing for format conversion configuration.
        """
        # Handle format conversion config separately
        if "format_conversion" in data:
            format_conversion = data["format_conversion"]
            if isinstance(format_conversion, dict):
                # Convert nested configs to FormatConversionConfig objects
                for key, value in format_conversion.items():
                    if isinstance(value, dict):
                        format_conversion[key] = FormatConversionConfig.from_dict(value)
            data["format_conversion"] = format_conversion
        return cls(**data)
