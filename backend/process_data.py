import os
import json
import re
from itertools import combinations
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- Constants ---
# SIMILARITY_THRESHOLD: 두 대화가 의미적으로 관련 있다고 판단하는 기준 임계값입니다.
# 이 값을 높이면(예: 0.6) 더 직접적으로 관련된 대화만 연결되고,
# 낮추면(예: 0.4) 더 넓은 범위의 관련성까지 포함하게 됩니다.
# 값의 범위는 0에서 1 사이입니다.
SIMILARITY_THRESHOLD = 0.5

# STOP_WORDS: 키워드 분석 시 무시할 일반적인 단어 목록입니다.
# 분석 결과에 불필요하다고 생각되는 단어를 이 목록에 추가할 수 있습니다.
STOP_WORDS = set(
    [
        "the",
        "a",
        "an",
        "in",
        "is",
        "it",
        "of",
        "for",
        "on",
        "with",
        "to",
        "and",
        "what",
        "how",
        "why",
        "i",
        "you",
        "he",
        "she",
        "they",
        "we",
        "my",
        "your",
        "our",
        "his",
        "her",
        "their",
        "was",
        "were",
        "are",
        "be",
        "been",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "but",
        "if",
        "or",
        "as",
        "at",
        "by",
        "from",
        "so",
        "then",
        "that",
        "this",
        "these",
        "those",
    ]
)

# --- Helper Functions ---


def process_text(text):
    """
    Normalizes and tokenizes text for keyword analysis.

    Args:
        text (str): The input string to process.

    Returns:
        list[str]: A list of normalized and filtered words.
    """
    text = text.lower()
    words = re.findall(r"\b\w+\b", text)
    # Filter out stop words and short words.
    return [word for word in words if word not in STOP_WORDS and len(word) >= 4]


# --- Core Logic Functions ---


def create_nodes_and_full_text(data):
    """
    Parses the raw conversation data to create node objects.
    A temporary 'fullText' field is added for processing.

    Args:
        data (list[dict]): A list of raw conversation objects from the input JSON.

    Returns:
        list[dict]: A list of node objects, each with 'id', 'label', and 'fullText'.
    """
    nodes = []
    for i, conversation in enumerate(data):
        parts_list = []
        for m in conversation.get("mapping", {}).values():
            if m.get("message"):
                content = m.get("message", {}).get("content", {})
                if content:
                    parts = content.get("parts", [])
                    if parts and isinstance(parts[0], str):
                        parts_list.append(parts[0])
                    elif parts and isinstance(parts[0], dict):
                        # Safely handle cases where 'parts' contains a dictionary.
                        parts_list.append(json.dumps(parts[0]))

        full_text = " ".join(filter(None, parts_list))

        nodes.append(
            {
                "id": f"convo_{i+1}",
                "label": conversation.get("title", f"Conversation {i+1}"),
                "fullText": full_text,
                # Category will be added by a separate process if needed.
            }
        )
    return nodes


def create_keyword_links(nodes):
    """
    Phase 2: Creates links based on shared keywords.
    Returns a dictionary of links for the hybrid model.

    Args:
        nodes (list[dict]): A list of node objects, each containing 'id' and 'fullText'.

    Returns:
        dict: A dictionary where keys are tuples of (source_id, target_id) and
              values are link information dictionaries.
    """
    print("Phase 2: Analyzing keyword-based links...")
    word_map = {}
    for node in nodes:
        words = process_text(node["fullText"])
        for word in words:
            if word not in word_map:
                word_map[word] = []
            word_map[word].append(node["id"])

    keyword_links = {}
    for word, node_ids in word_map.items():
        # 너무 흔하거나(10개 이상 노드에 등장) 너무 희귀한(1개 노드에만 등장) 키워드는 제외합니다.
        # 이 범위를 조절하여 링크의 수를 제어할 수 있습니다.
        # 예를 들어, `if 1 < len(node_ids) < 5:`로 바꾸면 더 핵심적인 키워드만 사용하게 됩니다.
        if 1 < len(node_ids) < 10:
            for source_id, target_id in combinations(sorted(node_ids), 2):
                link_key = tuple(sorted((source_id, target_id)))
                if link_key not in keyword_links:
                    keyword_links[link_key] = {
                        "relationship": f"Shared keyword: {word}",
                        # 키워드 기반 링크의 기본 강도입니다.
                        "strength": 0.6,
                    }
    print(f"Found {len(keyword_links)} potential links based on keywords.")
    return keyword_links


def create_semantic_links(nodes):
    """
    Phase 3: Creates links based on semantic similarity of conversations.
    Returns a dictionary of links for the hybrid model.

    Args:
        nodes (list[dict]): A list of node objects, each containing 'id' and 'fullText'.

    Returns:
        dict: A dictionary where keys are tuples of (source_id, target_id) and
              values are link information dictionaries with similarity strength.
    """
    print("Phase 3: Analyzing semantic links...")
    # Using a pre-trained model suitable for multilingual semantic similarity.
    # The model will be downloaded automatically on the first run.
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Prepare texts for embedding.
    corpus = [node["fullText"] for node in nodes]
    node_ids = [node["id"] for node in nodes]

    print("Generating text embeddings (this may take a moment)...")
    embeddings = model.encode(corpus, show_progress_bar=True)

    print("Calculating cosine similarity...")
    # Calculate cosine similarity between all pairs of embeddings.
    similarity_matrix = cosine_similarity(embeddings)

    semantic_links = {}
    # Iterate through the upper triangle of the matrix to avoid duplicates.
    for i in range(len(similarity_matrix)):
        for j in range(i + 1, len(similarity_matrix)):
            similarity = similarity_matrix[i][j]
            if similarity > SIMILARITY_THRESHOLD:
                source_id = node_ids[i]
                target_id = node_ids[j]
                link_key = tuple(sorted((source_id, target_id)))
                semantic_links[link_key] = {
                    "relationship": "Semantically similar topics",
                    "strength": round(float(similarity), 2),
                }
    print(f"Found {len(semantic_links)} potential links based on semantics.")
    return semantic_links


def create_hybrid_links(keyword_links, semantic_links):
    """
    Merges keyword and semantic links using the Intelligent Hybrid strategy.

    Args:
        keyword_links (dict): Links generated from keyword analysis.
        semantic_links (dict): Links generated from semantic analysis.

    Returns:
        list[dict]: A list of final, merged link objects to be used in the graph.
    """
    print("Merging links with Intelligent Hybrid strategy...")
    final_links = []

    # Use a set of all unique link keys from both methods.
    all_link_keys = set(keyword_links.keys()) | set(semantic_links.keys())

    for source_id, target_id in all_link_keys:
        link_key = (source_id, target_id)
        is_keyword_link = link_key in keyword_links
        is_semantic_link = link_key in semantic_links

        link_info = {"source": source_id, "target": target_id}

        if is_keyword_link and is_semantic_link:
            # Case 1: Strongest link - found by both methods.
            # Use the more informative semantic strength and a combined relationship.
            keyword = keyword_links[link_key]["relationship"].split(": ")[1]
            link_info["relationship"] = (
                f"Semantically similar and share keyword: {keyword}"
            )
            link_info["strength"] = semantic_links[link_key]["strength"]
            final_links.append(link_info)
        elif is_semantic_link:
            # Case 2: Found only by semantic analysis.
            link_info.update(semantic_links[link_key])
            final_links.append(link_info)

    return final_links


def main():
    """Main function to process data and generate final graph data."""
    # Define input and output paths.
    # Make sure to create these directories if they don't exist.
    input_path = "input_data/conversations.json"
    output_path = "graph_data/graph_data.json"

    # Delete the existing graph_data.json file if it exists
    if os.path.exists(output_path):
        os.remove(output_path)

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            conversations = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        print(
            "Please place your 'conversations.json' file in the same directory as this script."
        )
        return
    except json.JSONDecodeError:
        print(
            f"Error: Could not decode JSON from {input_path}. Please check the file format."
        )
        return

    # --- Run Processing Pipeline ---
    # 1. Create base nodes and extract text.
    print("Phase 1: Creating nodes from conversations...")
    nodes = create_nodes_and_full_text(conversations)
    print(f"Found {len(nodes)} nodes.")

    # 2. Generate links from both methods.
    keyword_links = create_keyword_links(nodes)
    semantic_links = create_semantic_links(nodes)

    # 3. Merge links using the hybrid strategy.
    final_links = create_hybrid_links(keyword_links, semantic_links)
    print(f"Created {len(final_links)} final merged links.")

    # 4. Clean up temporary fullText field before final output.
    for node in nodes:
        del node["fullText"]

    # 5. Prepare final graph data object.
    graph_data = {"nodes": nodes, "links": final_links}

    # 6. Write the output file.
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, indent=2, ensure_ascii=False)

    print("\n--- Processing Complete ---")
    print(
        f"Successfully generated graph data with {len(nodes)} nodes and {len(final_links)} links."
    )
    print(f"Output file saved to: {output_path}")


if __name__ == "__main__":
    main()
