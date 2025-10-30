import json
from pathlib import Path

import pytest
import yaml

from experiments.build_graph import build_graph
from experiments.io_schemas import OutputGraph

BASE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_HISTORY = BASE_DIR / "sample_history.json"
DEFAULT_CONFIG = BASE_DIR / "config.yaml"
MOCK_HISTORY = PROJECT_ROOT / "input_data" / "mock_data.json"


def test_pipeline_smoke(tmp_path):
    output_path = tmp_path / "graph.json"
    build_graph(SAMPLE_HISTORY, output_path, DEFAULT_CONFIG)

    assert output_path.exists()
    output = OutputGraph.model_validate_json(output_path.read_text(encoding="utf-8"))
    input_data = json.loads(SAMPLE_HISTORY.read_text(encoding="utf-8"))

    unique_conversations = 0
    if isinstance(input_data, list) and input_data:
        if "role" in input_data[0]:
            prefixes = set()
            for message in input_data:
                message_id = message.get("id", "")
                if "_" in message_id:
                    prefixes.add(message_id.rsplit("_", 1)[0])
                else:
                    prefixes.add("conv_0")
            unique_conversations = len(prefixes)
        else:
            unique_conversations = len(
                [
                    item
                    for item in input_data
                    if isinstance(item, dict) and "mapping" in item
                ]
            )
    elif isinstance(input_data, dict) and "mapping" in input_data:
        unique_conversations = 1

    assert len(output.nodes) == unique_conversations
    top_n = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))["keyword"][
        "top_n"
    ]
    for node in output.nodes:
        assert node.role == "conversation"
        assert node.num_messages > 0
        assert node.message_ids
        assert len(node.keywords) <= top_n
    assert output.metadata.counts.nodes == len(output.nodes)
    assert output.metadata.counts.edges == len(output.edges)


def test_graph_density_with_lower_topk(tmp_path):
    config_data = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    config_data["graph"]["sim_top_k"] = 3
    config_data["graph"]["sim_threshold"] = None
    temp_config = tmp_path / "config.yaml"
    temp_config.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    output_path = tmp_path / "graph_top3.json"
    build_graph(SAMPLE_HISTORY, output_path, temp_config)

    graph = OutputGraph.model_validate_json(output_path.read_text(encoding="utf-8"))
    if len(graph.nodes) > 1:
        expected_min_edges = int(len(graph.nodes) * 1.5)
        assert len(graph.edges) >= expected_min_edges
    else:
        assert len(graph.edges) == 0
    assert graph.metadata.counts.nodes == len(graph.nodes)
    assert graph.metadata.counts.edges == len(graph.edges)


@pytest.mark.skipif(not MOCK_HISTORY.exists(), reason="mock history file missing")
def test_pipeline_handles_chatgpt_export(tmp_path):
    output_path = tmp_path / "mock_graph.json"
    build_graph(MOCK_HISTORY, output_path, DEFAULT_CONFIG)

    graph = OutputGraph.model_validate_json(output_path.read_text(encoding="utf-8"))

    mock_data = json.loads(MOCK_HISTORY.read_text(encoding="utf-8"))
    expected_conversations = len([item for item in mock_data if "mapping" in item])

    assert graph.metadata.counts.nodes == expected_conversations
    assert len(graph.nodes) == expected_conversations
    assert graph.metadata.counts.edges == len(graph.edges)
    for node in graph.nodes:
        assert node.role == "conversation"
        assert node.num_messages >= 1
        assert node.message_ids
