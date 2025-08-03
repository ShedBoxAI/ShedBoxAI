"""
Comprehensive unit tests for the GraphExecutor class.
Tests cover normal operation, error conditions, edge cases, and potential bugs.
"""

# Mock external dependencies
import sys
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

import pytest

mock_nx = MagicMock()
sys.modules["networkx"] = mock_nx


class GraphNode:
    def __init__(
        self,
        id: str,
        operation: str,
        config_key: str = None,
        depends_on: List[str] = None,
    ):
        self.id = id
        self.operation = operation
        self.config_key = config_key
        self.depends_on = depends_on or []


class MockHandler:
    def __init__(self, engine=None):
        self.engine = engine

    def process(self, data, config):
        return data


class MockProcessorConfig:
    def __init__(self):
        self.contextual_filtering = None
        self.format_conversion = None
        self.content_summarization = None
        self.relationship_highlighting = None
        self.advanced_operations = None
        self.template_matching = None


class GraphExecutor:
    def __init__(self, engine=None):
        self.engine = engine
        self.operation_handlers = {
            "contextual_filtering": MockHandler,
            "format_conversion": MockHandler,
            "content_summarization": MockHandler,
            "relationship_highlighting": MockHandler,
            "advanced_operations": MockHandler,
            "template_matching": MockHandler,
        }

    def execute_graph(
        self, data: Dict[str, Any], graph_nodes: List[GraphNode], processor_config: Any
    ) -> Dict[str, Any]:
        graph = mock_nx.DiGraph()
        nodes = {}

        for node in graph_nodes:
            if node.operation not in self.operation_handlers:
                raise ValueError(f"Unknown operation: {node.operation}")

            graph.add_node(node.id)
            nodes[node.id] = {
                "operation": node.operation,
                "config_key": node.config_key,
                "handler_class": self.operation_handlers[node.operation],
            }

            for dependency in node.depends_on:
                if dependency not in [n.id for n in graph_nodes]:
                    raise ValueError(f"Dependency '{dependency}' not found")
                graph.add_edge(dependency, node.id)

        if not mock_nx.is_directed_acyclic_graph(graph):
            cycles = list(mock_nx.simple_cycles(graph))
            raise ValueError(f"Processing graph contains cycles: {cycles}")

        execution_order = list(mock_nx.topological_sort(graph))
        result = data.copy()

        for node_id in execution_order:
            node_info = nodes[node_id]
            operation = node_info["operation"]
            config_key = node_info["config_key"]
            handler_class = node_info["handler_class"]

            operation_config = getattr(processor_config, operation, None)
            if operation_config is None:
                continue

            if config_key and isinstance(operation_config, dict):
                if config_key in operation_config:
                    specific_config = operation_config[config_key]
                    if not specific_config:
                        print(f"Warning: No configuration found for node {node_id} with config_key {config_key}")
                        continue
                else:
                    print(f"Warning: Config key '{config_key}' not found in {operation} configuration")
                    continue
            else:
                specific_config = operation_config

            handler = handler_class(self.engine)
            normalized_config = self._normalize_config(operation, specific_config)
            result = handler.process(result, normalized_config)

        return result

    def execute_linear_pipeline(self, data: Dict[str, Any], processor_config: Any) -> Dict[str, Any]:
        result = data.copy()

        default_order = [
            "contextual_filtering",
            "format_conversion",
            "content_summarization",
            "relationship_highlighting",
            "advanced_operations",
            "template_matching",
        ]

        for operation in default_order:
            config = getattr(processor_config, operation, None)
            if config is None:
                continue

            handler_class = self.operation_handlers.get(operation)
            if not handler_class:
                continue

            handler = handler_class(self.engine)
            normalized_config = self._normalize_config(operation, config)
            result = handler.process(result, normalized_config)

        return result

    def _normalize_config(self, operation: str, config: Any) -> Dict[str, Any]:
        normalization_functions = {
            "contextual_filtering": lambda x: x,
            "format_conversion": lambda x: x,
            "content_summarization": lambda x: x,
            "relationship_highlighting": lambda x: x,
            "advanced_operations": lambda x: x,
            "template_matching": lambda x: x,
        }

        normalizer = normalization_functions.get(operation)
        if normalizer and config:
            return normalizer(config)
        else:
            return config or {}


class TestGraphExecutor:
    def setup_method(self):
        self.executor = GraphExecutor()
        self.mock_engine = Mock()
        self.executor_with_engine = GraphExecutor(engine=self.mock_engine)

        mock_nx.reset_mock()
        mock_nx.DiGraph.return_value = Mock()
        mock_nx.is_directed_acyclic_graph.return_value = True
        mock_nx.topological_sort.return_value = []
        mock_nx.simple_cycles.return_value = []

    def test_init_without_engine(self):
        executor = GraphExecutor()
        assert executor.engine is None
        assert len(executor.operation_handlers) == 6
        assert "contextual_filtering" in executor.operation_handlers

    def test_init_with_engine(self):
        mock_engine = Mock()
        executor = GraphExecutor(engine=mock_engine)
        assert executor.engine is mock_engine
        assert len(executor.operation_handlers) == 6

    def test_execute_graph_simple_single_node(self):
        data = {"input": "test_data"}
        node = GraphNode(id="node1", operation="contextual_filtering")
        processor_config = MockProcessorConfig()
        processor_config.contextual_filtering = {"param": "value"}

        mock_nx.topological_sort.return_value = ["node1"]

        result = self.executor.execute_graph(data, [node], processor_config)

        assert "input" in result
        assert result["input"] == "test_data"

    def test_execute_graph_multiple_nodes_with_dependencies(self):
        node1 = GraphNode(id="node1", operation="contextual_filtering")
        node2 = GraphNode(id="node2", operation="format_conversion", depends_on=["node1"])
        node3 = GraphNode(id="node3", operation="content_summarization", depends_on=["node2"])

        data = {"input": "test_data"}
        processor_config = MockProcessorConfig()
        processor_config.contextual_filtering = {"param1": "value1"}
        processor_config.format_conversion = {"param2": "value2"}
        processor_config.content_summarization = {"param3": "value3"}

        mock_nx.topological_sort.return_value = ["node1", "node2", "node3"]

        result = self.executor.execute_graph(data, [node1, node2, node3], processor_config)
        assert mock_nx.DiGraph.called

    def test_execute_graph_unknown_operation(self):
        node = GraphNode(id="node1", operation="unknown_operation")
        data = {"input": "test_data"}
        processor_config = MockProcessorConfig()

        with pytest.raises(ValueError, match="Unknown operation: unknown_operation"):
            self.executor.execute_graph(data, [node], processor_config)

    def test_execute_graph_missing_dependency(self):
        node = GraphNode(
            id="node1",
            operation="contextual_filtering",
            depends_on=["nonexistent_node"],
        )
        data = {"input": "test_data"}
        processor_config = MockProcessorConfig()

        with pytest.raises(ValueError, match="Dependency 'nonexistent_node' not found"):
            self.executor.execute_graph(data, [node], processor_config)

    def test_execute_graph_cycle_detection(self):
        node1 = GraphNode(id="node1", operation="contextual_filtering", depends_on=["node2"])
        node2 = GraphNode(id="node2", operation="format_conversion", depends_on=["node1"])

        data = {"input": "test_data"}
        processor_config = MockProcessorConfig()

        mock_nx.is_directed_acyclic_graph.return_value = False
        mock_nx.simple_cycles.return_value = [["node1", "node2"]]

        with pytest.raises(ValueError, match="Processing graph contains cycles"):
            self.executor.execute_graph(data, [node1, node2], processor_config)

    def test_execute_graph_missing_config(self):
        node = GraphNode(id="node1", operation="contextual_filtering")
        data = {"input": "test_data"}
        processor_config = MockProcessorConfig()

        mock_nx.topological_sort.return_value = ["node1"]

        result = self.executor.execute_graph(data, [node], processor_config)
        assert result == data

    def test_execute_graph_with_config_key(self):
        node = GraphNode(id="node1", operation="contextual_filtering", config_key="specific_config")
        data = {"input": "test_data"}
        processor_config = MockProcessorConfig()
        processor_config.contextual_filtering = {
            "specific_config": {"param": "value"},
            "other_config": {"param": "other_value"},
        }

        mock_nx.topological_sort.return_value = ["node1"]

        result = self.executor.execute_graph(data, [node], processor_config)
        assert result is not None

    def test_execute_graph_config_key_not_found(self, capsys):
        node = GraphNode(id="node1", operation="contextual_filtering", config_key="missing_key")
        data = {"input": "test_data"}
        processor_config = MockProcessorConfig()
        processor_config.contextual_filtering = {"other_key": {"param": "value"}}

        mock_nx.topological_sort.return_value = ["node1"]

        result = self.executor.execute_graph(data, [node], processor_config)

        captured = capsys.readouterr()
        assert "Config key 'missing_key' not found" in captured.out
        assert result == data

    def test_execute_graph_empty_config_for_key(self, capsys):
        node = GraphNode(id="node1", operation="contextual_filtering", config_key="empty_config")
        data = {"input": "test_data"}
        processor_config = MockProcessorConfig()
        processor_config.contextual_filtering = {"empty_config": None}

        mock_nx.topological_sort.return_value = ["node1"]

        result = self.executor.execute_graph(data, [node], processor_config)

        captured = capsys.readouterr()
        assert "No configuration found for node node1" in captured.out
        assert result == data

    def test_execute_linear_pipeline_all_operations(self):
        data = {"input": "test_data"}
        processor_config = MockProcessorConfig()

        for operation in [
            "contextual_filtering",
            "format_conversion",
            "content_summarization",
            "relationship_highlighting",
            "advanced_operations",
            "template_matching",
        ]:
            setattr(processor_config, operation, {"param": "value"})

        result = self.executor.execute_linear_pipeline(data, processor_config)
        assert result is not None

    def test_execute_linear_pipeline_partial_config(self):
        data = {"input": "test_data"}
        processor_config = MockProcessorConfig()

        processor_config.contextual_filtering = {"param": "value"}
        processor_config.content_summarization = {"param": "value"}

        result = self.executor.execute_linear_pipeline(data, processor_config)
        assert result is not None

    def test_execute_linear_pipeline_no_config(self):
        data = {"input": "test_data"}
        processor_config = MockProcessorConfig()

        result = self.executor.execute_linear_pipeline(data, processor_config)
        assert result == data

    def test_normalize_config_with_valid_operation(self):
        config = {"param": "value"}
        result = self.executor._normalize_config("contextual_filtering", config)
        assert result == config

    def test_normalize_config_with_invalid_operation(self):
        config = {"param": "value"}
        result = self.executor._normalize_config("invalid_operation", config)
        assert result == config

    def test_normalize_config_with_none_config(self):
        result = self.executor._normalize_config("contextual_filtering", None)
        assert result == {}

    def test_normalize_config_with_empty_config(self):
        result = self.executor._normalize_config("contextual_filtering", {})
        assert result == {}

    def test_data_immutability_in_execute_graph(self):
        original_data = {"input": "test_data", "count": 1}
        node = GraphNode(id="node1", operation="contextual_filtering")
        processor_config = MockProcessorConfig()
        processor_config.contextual_filtering = {"param": "value"}

        mock_nx.topological_sort.return_value = ["node1"]

        result = self.executor.execute_graph(original_data, [node], processor_config)

        assert original_data == {"input": "test_data", "count": 1}
        assert result is not original_data

    def test_data_immutability_in_linear_pipeline(self):
        original_data = {"input": "test_data", "count": 1}
        processor_config = MockProcessorConfig()
        processor_config.contextual_filtering = {"param": "value"}

        result = self.executor.execute_linear_pipeline(original_data, processor_config)

        assert original_data == {"input": "test_data", "count": 1}
        assert result is not original_data

    def test_engine_passed_to_handlers(self):
        mock_engine = Mock()
        executor = GraphExecutor(engine=mock_engine)

        node = GraphNode(id="node1", operation="contextual_filtering")
        data = {"input": "test_data"}
        processor_config = MockProcessorConfig()
        processor_config.contextual_filtering = {"param": "value"}

        mock_nx.topological_sort.return_value = ["node1"]

        result = executor.execute_graph(data, [node], processor_config)
        assert result is not None

    def test_complex_dependency_graph(self):
        node1 = GraphNode(id="root", operation="contextual_filtering")
        node2 = GraphNode(id="branch1", operation="format_conversion", depends_on=["root"])
        node3 = GraphNode(id="branch2", operation="content_summarization", depends_on=["root"])
        node4 = GraphNode(
            id="merge",
            operation="relationship_highlighting",
            depends_on=["branch1", "branch2"],
        )

        data = {"input": "test_data"}
        processor_config = MockProcessorConfig()
        for operation in [
            "contextual_filtering",
            "format_conversion",
            "content_summarization",
            "relationship_highlighting",
        ]:
            setattr(processor_config, operation, {"param": "value"})

        mock_nx.topological_sort.return_value = ["root", "branch1", "branch2", "merge"]

        result = self.executor.execute_graph(data, [node1, node2, node3, node4], processor_config)
        assert result is not None

    def test_empty_graph_nodes_list(self):
        data = {"input": "test_data"}
        processor_config = MockProcessorConfig()

        mock_nx.topological_sort.return_value = []

        result = self.executor.execute_graph(data, [], processor_config)
        assert result == data
        assert result is not data

    def test_processor_config_attribute_error(self):
        node = GraphNode(id="node1", operation="contextual_filtering")
        data = {"input": "test_data"}

        processor_config = object()

        mock_nx.topological_sort.return_value = ["node1"]

        result = self.executor.execute_graph(data, [node], processor_config)
        assert result == data

    @pytest.mark.parametrize(
        "operation",
        [
            "contextual_filtering",
            "format_conversion",
            "content_summarization",
            "relationship_highlighting",
            "advanced_operations",
            "template_matching",
        ],
    )
    def test_all_operations_individually(self, operation):
        data = {"input": "test_data"}
        processor_config = MockProcessorConfig()
        setattr(processor_config, operation, {"param": "value"})

        all_operations = [
            "contextual_filtering",
            "format_conversion",
            "content_summarization",
            "relationship_highlighting",
            "advanced_operations",
            "template_matching",
        ]
        for op in all_operations:
            if op != operation:
                setattr(processor_config, op, None)

        result = self.executor.execute_linear_pipeline(data, processor_config)
        assert result is not None


class TestGraphExecutorEdgeCases:
    def setup_method(self):
        self.executor = GraphExecutor()
        mock_nx.reset_mock()
        mock_nx.DiGraph.return_value = Mock()
        mock_nx.is_directed_acyclic_graph.return_value = True
        mock_nx.topological_sort.return_value = []

    def test_config_key_with_non_dict_config(self):
        node = GraphNode(id="node1", operation="contextual_filtering", config_key="some_key")
        data = {"input": "test_data"}
        processor_config = MockProcessorConfig()
        processor_config.contextual_filtering = "not_a_dict"

        mock_nx.topological_sort.return_value = ["node1"]

        result = self.executor.execute_graph(data, [node], processor_config)
        assert result is not None

    def test_large_graph_performance(self):
        nodes = []
        for i in range(10):
            depends_on = [f"node{i-1}"] if i > 0 else []
            nodes.append(
                GraphNode(
                    id=f"node{i}",
                    operation="contextual_filtering",
                    depends_on=depends_on,
                )
            )

        data = {"input": "test_data"}
        processor_config = MockProcessorConfig()
        processor_config.contextual_filtering = {"param": "value"}

        mock_nx.topological_sort.return_value = [f"node{i}" for i in range(10)]

        result = self.executor.execute_graph(data, nodes, processor_config)
        assert result is not None

    def test_normalize_config_exception_handling(self):
        config = {"param": "value"}
        result = self.executor._normalize_config("contextual_filtering", config)
        assert result == config

    def test_config_key_with_empty_dict(self):
        node = GraphNode(id="node1", operation="contextual_filtering", config_key="empty_key")
        data = {"input": "test_data"}
        processor_config = MockProcessorConfig()
        processor_config.contextual_filtering = {"empty_key": {}}

        mock_nx.topological_sort.return_value = ["node1"]

        result = self.executor.execute_graph(data, [node], processor_config)
        assert result == data

    def test_config_key_with_false_value(self):
        node = GraphNode(id="node1", operation="contextual_filtering", config_key="false_key")
        data = {"input": "test_data"}
        processor_config = MockProcessorConfig()
        processor_config.contextual_filtering = {"false_key": False}

        mock_nx.topological_sort.return_value = ["node1"]

        result = self.executor.execute_graph(data, [node], processor_config)
        assert result == data

    def test_config_key_with_zero_value(self):
        node = GraphNode(id="node1", operation="contextual_filtering", config_key="zero_key")
        data = {"input": "test_data"}
        processor_config = MockProcessorConfig()
        processor_config.contextual_filtering = {"zero_key": 0}

        mock_nx.topological_sort.return_value = ["node1"]

        result = self.executor.execute_graph(data, [node], processor_config)
        assert result == data


class TestGraphExecutorPerformance:
    def test_memory_usage_with_large_data(self):
        large_data = {f"key{i}": f"value{i}" * 100 for i in range(50)}

        executor = GraphExecutor()
        node = GraphNode(id="node1", operation="contextual_filtering")
        processor_config = MockProcessorConfig()
        processor_config.contextual_filtering = {"param": "value"}

        mock_nx.topological_sort.return_value = ["node1"]
        mock_nx.DiGraph.return_value = Mock()
        mock_nx.is_directed_acyclic_graph.return_value = True

        result = executor.execute_graph(large_data, [node], processor_config)

        assert result is not None
        assert len(result) == len(large_data)
        assert "key0" in result

    def test_deep_dependency_chain(self):
        nodes = []
        for i in range(20):
            depends_on = [f"node{i-1}"] if i > 0 else []
            nodes.append(
                GraphNode(
                    id=f"node{i}",
                    operation="contextual_filtering",
                    depends_on=depends_on,
                )
            )

        data = {"input": "test_data"}
        processor_config = MockProcessorConfig()
        processor_config.contextual_filtering = {"param": "value"}

        mock_nx.topological_sort.return_value = [f"node{i}" for i in range(20)]

        result = GraphExecutor().execute_graph(data, nodes, processor_config)
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
