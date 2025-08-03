"""
Configuration models for AI interface operations.

This module contains Pydantic models that define the structure and validation
rules for AI interface configurations, including model settings and prompts.
"""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, validator


class AIModelConfig(BaseModel):
    """Configuration for LLM API connection."""

    type: str = "rest"
    url: str
    method: str = "POST"
    headers: Dict[str, str]
    options: Dict[str, Any] = Field(default_factory=dict)

    @validator("type")
    def validate_type(cls, v):
        """Validate model type."""
        allowed_types = {"rest", "websocket", "grpc"}
        if v not in allowed_types:
            raise ValueError(f"Model type must be one of {allowed_types}")
        return v


class PromptConfig(BaseModel):
    """Configuration for a single prompt."""

    system: Optional[str] = None
    user_template: str
    response_format: Optional[str] = "text"
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    for_each: Optional[str] = None  # Data source to iterate over for fan-out
    parallel: Optional[bool] = False  # Whether to process iterations in parallel

    @validator("response_format")
    def validate_response_format(cls, v):
        """Validate response format."""
        allowed_formats = {"text", "json", "markdown", "html"}
        if v not in allowed_formats:
            raise ValueError(f"Response format must be one of {allowed_formats}")
        return v


class PromptStorageConfig(BaseModel):
    """Configuration for prompt storage functionality."""

    enabled: bool = False
    directory: str = "./prompts"
    store_only: bool = False  # If True, only store prompts without making LLM calls
    file_format: str = "{prompt_name}_{timestamp}.txt"
    include_metadata: bool = True  # Include context and config in stored files

    @validator("directory")
    def validate_directory(cls, v):
        """Validate directory path."""
        if not v.strip():
            raise ValueError("Directory path cannot be empty")
        return v.strip()


class AIInterfaceConfig(BaseModel):
    """Top-level AI interface configuration."""

    model: AIModelConfig
    prompts: Dict[str, PromptConfig]
    default_context: Optional[Dict[str, Any]] = Field(default_factory=dict)
    retry_config: Optional[Dict[str, Any]] = Field(default_factory=dict)
    prompt_storage: Optional[PromptStorageConfig] = Field(default_factory=PromptStorageConfig)
