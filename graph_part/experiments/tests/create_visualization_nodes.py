"""Utility script to build visualization-ready JSON from clustering results.

This script reads:
  - Cluster assignments from results/cluster/clustered_output_full_k=3.json
  - Keyword extraction output from results/keyword/test_output.json
  - Conversation metadata (titles) from input_data/conversations.json

It merges them into a single JSON document under results/visualization/
with a structure similar to the clustering output but containing a `nodes`
array (instead of `assignments`) where each node includes the cluster info,
conversation title, and keyword terms with scores.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def ensure_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def build_title_map(conversations_path: Path) -> Dict[str, str]:
    conversations = json.loads(conversations_path.read_text(encoding="utf-8"))
    title_map: Dict[str, str] = {}
    for idx, convo in enumerate(conversations):
        orig_id = f"conv_{idx}"
        title = convo.get("title") or ""
        title_map[orig_id] = title
    return title_map


def build_keywords_map(keyword_path: Path) -> Dict[int, Dict[str, Any]]:
    keyword_payload = load_json(keyword_path)
    conversations = keyword_payload.get("conversations", [])
    keywords_map: Dict[int, Dict[str, Any]] = {}
    for item in conversations:
        conv_id = item.get("id")
        if conv_id is None:
            continue
        keywords_map[int(conv_id)] = {
            "keywords": item.get("keywords", []),
            "num_messages": item.get("num_messages"),
            "timestamp": item.get("timestamp"),
        }
    return keywords_map


def build_nodes(
    assignments: List[Dict[str, Any]],
    titles: Dict[str, str],
    keywords_map: Dict[int, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    nodes: List[Dict[str, Any]] = []
    for assignment in assignments:
        conv_id = assignment.get("conversation_id")
        if conv_id is None:
            continue

        orig_id = assignment.get("orig_id", f"conv_{conv_id}")
        title = titles.get(orig_id)

        keyword_info = keywords_map.get(int(conv_id), {})
        node = {
            "id": conv_id,
            "orig_id": orig_id,
            "title": title,
            "cluster_id": assignment.get("cluster_id"),
            "confidence": assignment.get("confidence"),
            "keywords": keyword_info.get("keywords", []),
            "num_messages": keyword_info.get("num_messages"),
            "timestamp": keyword_info.get("timestamp"),
        }

        # Preserve any existing top keywords for quick reference
        if "top_keywords" in assignment:
            node["top_keywords"] = assignment["top_keywords"]

        nodes.append(node)
    return nodes


def main() -> None:
    experiments_dir = Path(__file__).resolve().parents[1]
    repo_root = experiments_dir.parent

    default_cluster = experiments_dir / "results/cluster/clustered_output_cosine.json"
    default_keywords = experiments_dir / "results/keyword/test_output_2.json"
    default_conversations = repo_root / "input_data/conversations.json"
    default_output = (
        experiments_dir / "results/visualization/clustered_nodes_cosine.json"
    )

    parser = argparse.ArgumentParser(
        description="Create visualization JSON combining clusters, keywords, and titles."
    )
    parser.add_argument(
        "--cluster",
        type=Path,
        default=default_cluster,
        help="Path to clustered output JSON (assignments + clusters).",
    )
    parser.add_argument(
        "--keywords",
        type=Path,
        default=default_keywords,
        help="Path to keyword extraction JSON.",
    )
    parser.add_argument(
        "--conversations",
        type=Path,
        default=default_conversations,
        help="Path to original conversations.json containing titles.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="Destination visualization JSON path.",
    )

    args = parser.parse_args()

    cluster_payload = load_json(args.cluster)
    assignments = cluster_payload.get("assignments", [])

    titles = build_title_map(args.conversations)
    keywords_map = build_keywords_map(args.keywords)
    nodes = build_nodes(assignments, titles, keywords_map)

    output_payload: Dict[str, Any] = {
        "nodes": nodes,
        "clusters": cluster_payload.get("clusters", []),
        "metadata": {
            **cluster_payload.get("metadata", {}),
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "source_files": {
                "cluster": str(args.cluster),
                "keywords": str(args.keywords),
                "conversations": str(args.conversations),
            },
        },
    }

    ensure_directory(args.output)
    with open(args.output, "w", encoding="utf-8") as fp:
        json.dump(output_payload, fp, ensure_ascii=False, indent=2)

    print(f"✅ Visualization JSON written to {args.output}")


if __name__ == "__main__":  # pragma: no cover
    main()
