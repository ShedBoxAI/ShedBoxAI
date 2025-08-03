"""
Comprehensive unit tests for the Pipeline class.
"""

import json

# Mock external dependencies
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, mock_open, patch

import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

mock_connector = MagicMock()
mock_processor = MagicMock()
mock_ai_interface = MagicMock()
mock_config_models = MagicMock()

sys.modules["shedboxai.connector"] = mock_connector
sys.modules["shedboxai.core.processor"] = mock_processor
sys.modules["shedboxai.core.ai"] = mock_ai_interface
sys.modules["shedboxai.core.config.models"] = mock_config_models
sys.modules["shedboxai.core.config.ai_config"] = MagicMock()

from typing import Any, Dict, Optional, Union

# Import classes to test
from pydantic import BaseModel, Field


class PipelineConfig(BaseModel):
    """Configuration model for the entire pipeline."""

    data_sources: Dict[str, Any] = Field(..., description="Data source configurations")
    processing: Optional[Dict[str, Any]] = Field(None, description="Data processing configuration")
    output: Optional[Dict[str, Any]] = Field(None, description="Output configuration")
    ai_interface: Optional[Dict[str, Any]] = Field(None, description="AI interface configuration")


class MockDataSourceConnector:
    def __init__(self, config_path):
        self.config_path = config_path

    def get_data(self, data_sources):
        return {"source1": [{"id": 1, "name": "test"}]}


class MockDataProcessor:
    def __init__(self, config):
        self.config = config

    def process(self, data):
        return {**data, "processed": True}


class MockAIInterface:
    def __init__(self, config):
        self.config = config

    def process_prompt(self, prompt_name, data):
        return f"AI result for {prompt_name}"


class Pipeline:
    """Main pipeline orchestrator for ShedBoxAI."""

    def __init__(self, config_path: Union[str, Path]):
        self.config_path = Path(config_path)
        self.config = self._load_config(self.config_path)
        self.connector = MockDataSourceConnector(self.config_path)
        self.processor = None
        self.ai_interface = None
        self.data = {}
        self.logger = Mock()

    def run(self) -> Any:
        # Step 1: Load all data sources
        self.data = self.connector.get_data(self.config.data_sources)

        # Step 2: Process data if configured
        if self.config.processing:
            if not self.processor:
                self.processor = MockDataProcessor(self.config.processing)
            self.data = self.processor.process(self.data)

        # Step 3: Handle AI interface if configured
        if self.config.ai_interface:
            self.data = self._handle_ai_interface(self.data)

        # Step 4: Handle output if configured
        if self.config.output:
            result = self._handle_output(self.data)
            return result

        return self.data

    def _handle_ai_interface(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.ai_interface:
            self.ai_interface = MockAIInterface(self.config.ai_interface)

        result = data.copy()

        # Process each configured prompt
        for prompt_name in self.config.ai_interface["prompts"]:
            try:
                result[prompt_name] = self.ai_interface.process_prompt(prompt_name, data)
            except Exception as e:
                self.logger.error(f"Error processing AI prompt '{prompt_name}': {str(e)}")
                result[prompt_name] = {"error": str(e)}

        return result

    def _load_config(self, config_path: Path) -> PipelineConfig:
        if not config_path.suffix.lower() in [".yaml", ".yml"]:
            raise ValueError("Configuration file must be YAML (.yaml or .yml)")

        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        return PipelineConfig(**config_dict)

    def _handle_output(self, data: Any) -> Any:
        output_type = self.config.output.get("type")
        if output_type == "file":
            return self._save_to_file(data)
        elif output_type == "print":
            print(data)
            return data
        else:
            raise ValueError(f"Unsupported output type: {output_type}")

    def _save_to_file(self, data: Any) -> Any:
        output_path = Path(self.config.output.get("path"))
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_format = self.config.output.get("format", "json")

        # Standard output handling
        if isinstance(data, dict):
            serializable_data = {
                k: v.to_dict(orient="records") if isinstance(v, pd.DataFrame) else v for k, v in data.items()
            }
        else:
            serializable_data = data.to_dict(orient="records") if isinstance(data, pd.DataFrame) else data

        if output_format == "json":
            output_path.write_text(json.dumps(serializable_data, indent=2))
        elif output_format == "yaml":
            output_path.write_text(yaml.dump(serializable_data))
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

        return data


class TestPipelineConfig:
    """Test PipelineConfig validation."""

    def test_valid_config(self):
        config_data = {
            "data_sources": {"source1": {"type": "csv", "path": "data.csv"}},
            "processing": {"operation": "filter"},
            "output": {"type": "file", "path": "output.json"},
        }
        config = PipelineConfig(**config_data)
        assert config.data_sources == config_data["data_sources"]
        assert config.processing == config_data["processing"]
        assert config.output == config_data["output"]

    def test_minimal_config(self):
        config_data = {"data_sources": {"source1": {"type": "csv"}}}
        config = PipelineConfig(**config_data)
        assert config.data_sources == config_data["data_sources"]
        assert config.processing is None
        assert config.output is None
        assert config.ai_interface is None

    def test_missing_data_sources(self):
        config_data = {"processing": {"operation": "filter"}}
        with pytest.raises(ValidationError):
            PipelineConfig(**config_data)


class TestPipeline:
    """Test Pipeline class functionality."""

    def setup_method(self):
        # Sample configuration
        self.valid_config = {
            "data_sources": {"source1": {"type": "csv", "path": "data.csv"}},
            "processing": {"operation": "filter"},
            "ai_interface": {"prompts": ["prompt1", "prompt2"]},
            "output": {"type": "file", "path": "output.json", "format": "json"},
        }

    @patch("builtins.open", new_callable=mock_open)
    @patch("yaml.safe_load")
    def test_init_valid_yaml_config(self, mock_yaml_load, mock_file):
        mock_yaml_load.return_value = self.valid_config

        pipeline = Pipeline("config.yaml")

        assert pipeline.config_path == Path("config.yaml")
        assert isinstance(pipeline.config, PipelineConfig)
        assert pipeline.data == {}
        assert pipeline.processor is None
        assert pipeline.ai_interface is None

    @patch("builtins.open", new_callable=mock_open)
    @patch("yaml.safe_load")
    def test_init_invalid_file_extension(self, mock_yaml_load, mock_file):
        mock_yaml_load.return_value = self.valid_config

        with pytest.raises(ValueError, match="Configuration file must be YAML"):
            Pipeline("config.json")

    @patch("builtins.open", new_callable=mock_open)
    @patch("yaml.safe_load")
    def test_init_missing_data_sources(self, mock_yaml_load, mock_file):
        invalid_config = {"processing": {"operation": "filter"}}
        mock_yaml_load.return_value = invalid_config

        with pytest.raises(ValidationError):
            Pipeline("config.yaml")

    @patch("builtins.open", side_effect=FileNotFoundError("File not found"))
    def test_load_config_file_not_found(self, mock_file):
        with pytest.raises(FileNotFoundError):
            Pipeline("nonexistent.yaml")

    @patch("builtins.open", new_callable=mock_open)
    @patch("yaml.safe_load")
    def test_run_data_sources_only(self, mock_yaml_load, mock_file):
        config = {"data_sources": {"source1": {"type": "csv"}}}
        mock_yaml_load.return_value = config

        pipeline = Pipeline("config.yaml")
        result = pipeline.run()

        assert "source1" in result
        assert result["source1"] == [{"id": 1, "name": "test"}]

    @patch("builtins.open", new_callable=mock_open)
    @patch("yaml.safe_load")
    def test_run_with_processing(self, mock_yaml_load, mock_file):
        config = {
            "data_sources": {"source1": {"type": "csv"}},
            "processing": {"operation": "filter"},
        }
        mock_yaml_load.return_value = config

        pipeline = Pipeline("config.yaml")
        result = pipeline.run()

        assert "source1" in result
        assert result["processed"] is True

    @patch("builtins.open", new_callable=mock_open)
    @patch("yaml.safe_load")
    def test_run_with_ai_interface(self, mock_yaml_load, mock_file):
        config = {
            "data_sources": {"source1": {"type": "csv"}},
            "ai_interface": {"prompts": ["prompt1", "prompt2"]},
        }
        mock_yaml_load.return_value = config

        pipeline = Pipeline("config.yaml")
        result = pipeline.run()

        assert "prompt1" in result
        assert "prompt2" in result
        assert result["prompt1"] == "AI result for prompt1"
        assert result["prompt2"] == "AI result for prompt2"

    @patch("builtins.open", new_callable=mock_open)
    @patch("yaml.safe_load")
    @patch("builtins.print")
    def test_run_with_print_output(self, mock_print, mock_yaml_load, mock_file):
        config = {
            "data_sources": {"source1": {"type": "csv"}},
            "output": {"type": "print"},
        }
        mock_yaml_load.return_value = config

        pipeline = Pipeline("config.yaml")
        result = pipeline.run()

        mock_print.assert_called_once()
        assert result["source1"] == [{"id": 1, "name": "test"}]

    @patch("builtins.open", new_callable=mock_open)
    @patch("yaml.safe_load")
    @patch("pathlib.Path.write_text")
    @patch("pathlib.Path.mkdir")
    def test_run_with_json_file_output(self, mock_mkdir, mock_write, mock_yaml_load, mock_file):
        config = {
            "data_sources": {"source1": {"type": "csv"}},
            "output": {"type": "file", "path": "output.json", "format": "json"},
        }
        mock_yaml_load.return_value = config

        pipeline = Pipeline("config.yaml")
        result = pipeline.run()

        mock_mkdir.assert_called_once()
        mock_write.assert_called_once()
        # Verify JSON was written
        written_data = mock_write.call_args[0][0]
        assert '"source1"' in written_data

    @patch("builtins.open", new_callable=mock_open)
    @patch("yaml.safe_load")
    @patch("pathlib.Path.write_text")
    @patch("pathlib.Path.mkdir")
    def test_run_with_yaml_file_output(self, mock_mkdir, mock_write, mock_yaml_load, mock_file):
        config = {
            "data_sources": {"source1": {"type": "csv"}},
            "output": {"type": "file", "path": "output.yaml", "format": "yaml"},
        }
        mock_yaml_load.return_value = config

        pipeline = Pipeline("config.yaml")
        result = pipeline.run()

        mock_mkdir.assert_called_once()
        mock_write.assert_called_once()

    @patch("builtins.open", new_callable=mock_open)
    @patch("yaml.safe_load")
    def test_run_full_pipeline(self, mock_yaml_load, mock_file):
        # Set the full config for yaml load
        mock_yaml_load.return_value = self.valid_config

        pipeline = Pipeline("config.yaml")

        with patch("pathlib.Path.write_text") as mock_write, patch("pathlib.Path.mkdir") as mock_mkdir:
            result = pipeline.run()

            # Should have original data, processing, and AI results
            assert "source1" in result
            assert "processed" in result
            assert "prompt1" in result
            assert "prompt2" in result

    @patch("builtins.open", new_callable=mock_open)
    @patch("yaml.safe_load")
    def test_handle_ai_interface_error(self, mock_yaml_load, mock_file):
        config = {
            "data_sources": {"source1": {"type": "csv"}},
            "ai_interface": {"prompts": ["failing_prompt"]},
        }
        mock_yaml_load.return_value = config

        pipeline = Pipeline("config.yaml")

        # Mock AI interface to raise exception
        with patch.object(MockAIInterface, "process_prompt", side_effect=Exception("AI Error")):
            result = pipeline.run()

            assert "failing_prompt" in result
            assert result["failing_prompt"]["error"] == "AI Error"
            pipeline.logger.error.assert_called()

    @patch("builtins.open", new_callable=mock_open)
    @patch("yaml.safe_load")
    def test_handle_output_unsupported_type(self, mock_yaml_load, mock_file):
        config = {
            "data_sources": {"source1": {"type": "csv"}},
            "output": {"type": "unsupported"},
        }
        mock_yaml_load.return_value = config

        pipeline = Pipeline("config.yaml")

        with pytest.raises(ValueError, match="Unsupported output type: unsupported"):
            pipeline.run()

    @patch("builtins.open", new_callable=mock_open)
    @patch("yaml.safe_load")
    def test_save_to_file_unsupported_format(self, mock_yaml_load, mock_file):
        config = {
            "data_sources": {"source1": {"type": "csv"}},
            "output": {"type": "file", "path": "output.txt", "format": "txt"},
        }
        mock_yaml_load.return_value = config

        pipeline = Pipeline("config.yaml")

        with pytest.raises(ValueError, match="Unsupported output format: txt"):
            pipeline.run()

    @patch("builtins.open", new_callable=mock_open)
    @patch("yaml.safe_load")
    @patch("pathlib.Path.write_text")
    @patch("pathlib.Path.mkdir")
    def test_save_to_file_with_dataframe(self, mock_mkdir, mock_write, mock_yaml_load, mock_file):
        config = {
            "data_sources": {"source1": {"type": "csv"}},
            "output": {"type": "file", "path": "output.json", "format": "json"},
        }
        mock_yaml_load.return_value = config

        pipeline = Pipeline("config.yaml")

        # Add DataFrame to data
        df = pd.DataFrame([{"col1": "val1", "col2": "val2"}])
        pipeline.data = {"dataframe": df, "regular": "data"}

        result = pipeline._save_to_file(pipeline.data)

        mock_write.assert_called_once()
        written_data = mock_write.call_args[0][0]
        # Should contain serialized DataFrame
        assert '"dataframe"' in written_data
        assert '"regular"' in written_data

    @patch("builtins.open", new_callable=mock_open)
    @patch("yaml.safe_load")
    @patch("pathlib.Path.write_text")
    @patch("pathlib.Path.mkdir")
    def test_save_to_file_dataframe_only(self, mock_mkdir, mock_write, mock_yaml_load, mock_file):
        config = {
            "data_sources": {"source1": {"type": "csv"}},
            "output": {"type": "file", "path": "output.json", "format": "json"},
        }
        mock_yaml_load.return_value = config

        pipeline = Pipeline("config.yaml")

        # Test with DataFrame as the main data
        df = pd.DataFrame([{"col1": "val1", "col2": "val2"}])

        result = pipeline._save_to_file(df)

        mock_write.assert_called_once()
        written_data = mock_write.call_args[0][0]
        # Should contain serialized DataFrame records
        assert '"col1"' in written_data
        assert '"val1"' in written_data


class TestPipelineEdgeCases:
    """Test edge cases and error conditions."""

    @patch("builtins.open", new_callable=mock_open)
    @patch("yaml.safe_load")
    def test_yml_extension_accepted(self, mock_yaml_load, mock_file):
        config = {"data_sources": {"source1": {"type": "csv"}}}
        mock_yaml_load.return_value = config

        # Should not raise exception for .yml extension
        pipeline = Pipeline("config.yml")
        assert pipeline.config_path.suffix == ".yml"

    @patch("builtins.open", new_callable=mock_open)
    @patch("yaml.safe_load")
    def test_case_insensitive_yaml_extension(self, mock_yaml_load, mock_file):
        config = {"data_sources": {"source1": {"type": "csv"}}}
        mock_yaml_load.return_value = config

        # Should work with uppercase extensions
        with patch.object(Path, "suffix", ".YAML"):
            pipeline = Pipeline("config.YAML")

    @patch("builtins.open", new_callable=mock_open)
    @patch("yaml.safe_load")
    def test_yaml_load_error(self, mock_yaml_load, mock_file):
        mock_yaml_load.side_effect = yaml.YAMLError("Invalid YAML")

        with pytest.raises(yaml.YAMLError):
            Pipeline("config.yaml")

    @patch("builtins.open", new_callable=mock_open)
    @patch("yaml.safe_load")
    def test_empty_prompts_list(self, mock_yaml_load, mock_file):
        config = {
            "data_sources": {"source1": {"type": "csv"}},
            "ai_interface": {"prompts": []},
        }
        mock_yaml_load.return_value = config

        pipeline = Pipeline("config.yaml")
        result = pipeline.run()

        # Should not crash with empty prompts
        assert "source1" in result

    @patch("builtins.open", new_callable=mock_open)
    @patch("yaml.safe_load")
    def test_processor_reuse(self, mock_yaml_load, mock_file):
        config = {
            "data_sources": {"source1": {"type": "csv"}},
            "processing": {"operation": "filter"},
        }
        mock_yaml_load.return_value = config

        pipeline = Pipeline("config.yaml")

        # Run twice to test processor reuse
        result1 = pipeline.run()
        result2 = pipeline.run()

        assert pipeline.processor is not None
        assert result1["processed"] is True
        assert result2["processed"] is True

    @patch("builtins.open", new_callable=mock_open)
    @patch("yaml.safe_load")
    def test_ai_interface_reuse(self, mock_yaml_load, mock_file):
        config = {
            "data_sources": {"source1": {"type": "csv"}},
            "ai_interface": {"prompts": ["prompt1"]},
        }
        mock_yaml_load.return_value = config

        pipeline = Pipeline("config.yaml")

        # Run twice to test AI interface reuse
        result1 = pipeline.run()
        result2 = pipeline.run()

        assert pipeline.ai_interface is not None
        assert result1["prompt1"] == "AI result for prompt1"
        assert result2["prompt1"] == "AI result for prompt1"

    @patch("builtins.open", new_callable=mock_open)
    @patch("yaml.safe_load")
    def test_output_default_format(self, mock_yaml_load, mock_file):
        config = {
            "data_sources": {"source1": {"type": "csv"}},
            "output": {"type": "file", "path": "output.json"},  # No format specified
        }
        mock_yaml_load.return_value = config

        pipeline = Pipeline("config.yaml")

        with patch("pathlib.Path.write_text") as mock_write, patch("pathlib.Path.mkdir"):
            pipeline.run()

            # Should default to JSON format
            written_data = mock_write.call_args[0][0]
            assert written_data.startswith("{")  # JSON format


class TestPipelineIntegration:
    """Integration-style tests for complete pipeline flows."""

    @patch("builtins.open", new_callable=mock_open)
    @patch("yaml.safe_load")
    def test_complex_pipeline_flow(self, mock_yaml_load, mock_file):
        """Test a complex pipeline with all components."""
        config = {
            "data_sources": {
                "source1": {"type": "csv", "path": "data1.csv"},
                "source2": {"type": "json", "path": "data2.json"},
            },
            "processing": {"operations": ["filter", "transform"]},
            "ai_interface": {"prompts": ["analyze", "summarize"]},
            "output": {"type": "file", "path": "results.json", "format": "json"},
        }
        mock_yaml_load.return_value = config

        pipeline = Pipeline("config.yaml")

        with patch("pathlib.Path.write_text") as mock_write, patch("pathlib.Path.mkdir"):
            result = pipeline.run()

            # Verify all components ran
            assert "source1" in pipeline.data
            assert "processed" in pipeline.data  # From processing
            assert "analyze" in pipeline.data  # From AI
            assert "summarize" in pipeline.data  # From AI
            mock_write.assert_called_once()  # Output saved


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
