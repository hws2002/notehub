import json
from pathlib import Path

import pytest
import yaml

from chat_graph.build_graph import build_graph
from chat_graph.io_schemas import OutputGraph

BASE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_HISTORY = BASE_DIR / "sample_history.json"
DEFAULT_CONFIG = BASE_DIR / "config.yaml"
MOCK_HISTORY = PROJECT_ROOT / "input_data" / "mock_data.json"


def load_input_messages(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_pipeline_smoke(tmp_path):
    output_path = tmp_path / "graph.json"
    build_graph(SAMPLE_HISTORY, output_path, DEFAULT_CONFIG)

    assert output_path.exists()
    output = OutputGraph.model_validate_json(output_path.read_text(encoding="utf-8"))
    input_messages = load_input_messages(SAMPLE_HISTORY)

    assert len(output.nodes) == len(input_messages)
    top_n = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))["keyword"]["top_n"]
    for node in output.nodes:
        assert len(node.keywords) <= top_n
    assert output.metadata.counts.nodes == len(output.nodes)
    assert output.metadata.counts.edges == len(output.edges)
    assert any(node.cluster != -1 for node in output.nodes)


def test_graph_density_with_lower_topk(tmp_path):
    config_data = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    config_data["graph"]["sim_top_k"] = 3
    config_data["graph"]["sim_threshold"] = None
    temp_config = tmp_path / "config.yaml"
    temp_config.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    output_path = tmp_path / "graph_top3.json"
    build_graph(SAMPLE_HISTORY, output_path, temp_config)

    graph = OutputGraph.model_validate_json(output_path.read_text(encoding="utf-8"))
    expected_min_edges = int(len(graph.nodes) * 1.5)
    assert len(graph.edges) >= expected_min_edges
    assert graph.metadata.counts.edges == len(graph.edges)


@pytest.mark.skipif(not MOCK_HISTORY.exists(), reason="mock history file missing")
def test_pipeline_handles_chatgpt_export(tmp_path):
    output_path = tmp_path / "mock_graph.json"
    build_graph(MOCK_HISTORY, output_path, DEFAULT_CONFIG)

    graph = OutputGraph.model_validate_json(output_path.read_text(encoding="utf-8"))
    assert graph.metadata.counts.nodes == len(graph.nodes)
    assert len(graph.nodes) > 0
    assert any(node.cluster != -1 for node in graph.nodes)
