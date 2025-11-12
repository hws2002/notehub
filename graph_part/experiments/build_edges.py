"""Edge generation module for conversation graph using similarity-based and LLM verification."""

import argparse
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from openai import OpenAI


@dataclass
class EdgeCandidate:
    """Represents a potential edge between two nodes."""

    source_id: int
    target_id: int
    similarity: float
    source_cluster: int
    target_cluster: int


@dataclass
class Edge:
    """Final edge with metadata."""

    source: int
    target: int
    weight: float
    type: str = "semantic"
    is_intra_cluster: bool = False  # True if both nodes in same cluster
    confidence: str = "high"  # "high" or "llm_verified"


def compute_cosine_similarity(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity matrix from embeddings.

    Args:
        embeddings: Normalized embedding matrix of shape (n_conversations, embedding_dim)

    Returns:
        Similarity matrix of shape (n_conversations, n_conversations) with diagonal set to -inf
    """
    # Compute cosine similarity (embeddings should already be normalized)
    similarity = embeddings @ embeddings.T

    # Set diagonal to -inf to ignore self-similarity
    np.fill_diagonal(similarity, -np.inf)

    return similarity


def generate_edge_candidates(
    similarity_matrix: np.ndarray,
    cluster_assignments: Dict[int, int],
    high_threshold: float = 0.8,
    medium_threshold: float = 0.6,
) -> Tuple[List[Edge], List[EdgeCandidate]]:
    """
    Generate edge candidates based on similarity thresholds.

    Args:
        similarity_matrix: Cosine similarity matrix
        cluster_assignments: Dict mapping conversation_id to cluster_id
        high_threshold: Threshold for high-confidence edges (default: 0.8)
        medium_threshold: Threshold for medium-confidence edges requiring LLM check (default: 0.6)

    Returns:
        Tuple of (confirmed_edges, candidates_for_llm_check)
    """
    n_nodes = similarity_matrix.shape[0]
    confirmed_edges: List[Edge] = []
    candidates_for_llm: List[EdgeCandidate] = []

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):  # Only upper triangle (i < j)
            similarity = float(similarity_matrix[i, j])

            # Skip if below medium threshold
            if similarity < medium_threshold:
                continue

            source_cluster = cluster_assignments.get(i, -1)
            target_cluster = cluster_assignments.get(j, -1)
            is_intra_cluster = source_cluster == target_cluster and source_cluster != -1

            if similarity >= high_threshold:
                # High confidence edge
                edge = Edge(
                    source=i,
                    target=j,
                    weight=similarity,
                    type="semantic",
                    is_intra_cluster=is_intra_cluster,
                    confidence="high",
                )
                confirmed_edges.append(edge)
            else:
                # Medium confidence - needs LLM verification
                candidate = EdgeCandidate(
                    source_id=i,
                    target_id=j,
                    similarity=similarity,
                    source_cluster=source_cluster,
                    target_cluster=target_cluster,
                )
                candidates_for_llm.append(candidate)

    return confirmed_edges, candidates_for_llm


def llm_verify_edge(
    source_keywords: List[str],
    target_keywords: List[str],
    similarity_score: float,
    llm_client: OpenAI,
) -> bool:
    """
    Verify edge using LLM judgment.

    Args:
        source_keywords: Keywords from source conversation
        target_keywords: Keywords from target conversation
        similarity_score: Cosine similarity score
        llm_client: OpenAI client instance

    Returns:
        True if LLM confirms the edge should exist, False otherwise
    """
    # Format keywords for prompt
    source_kw_str = ", ".join(source_keywords) if source_keywords else "(no keywords)"
    target_kw_str = ", ".join(target_keywords) if target_keywords else "(no keywords)"

    prompt = f"""Given two conversations with the following keywords:
Conversation A: {source_kw_str}
Conversation B: {target_kw_str}
Cosine similarity: {similarity_score:.3f}

Should these conversations be connected in a knowledge graph?
Consider: topical overlap, semantic relationship, meaningful connection.
Respond with only "YES" or "NO"."""

    try:
        response = llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0,
        )

        answer = response.choices[0].message.content.strip().upper()
        return "YES" in answer

    except Exception as exc:
        # If LLM call fails, default to rejecting the edge
        print(f"Warning: LLM verification failed: {exc}")
        return False


def verify_candidates_with_llm(
    candidates: List[EdgeCandidate],
    conversations: List[Dict[str, Any]],
    llm_client: OpenAI,
    batch_size: int = 10,
    verbose: bool = True,
) -> List[Edge]:
    """
    Verify edge candidates using LLM.

    Args:
        candidates: List of edge candidates to verify
        conversations: List of conversation dictionaries with keywords
        llm_client: OpenAI client instance
        batch_size: Number of candidates to process in each batch (for progress reporting)
        verbose: Whether to print progress

    Returns:
        List of verified edges
    """
    verified_edges: List[Edge] = []
    total = len(candidates)

    for idx, candidate in enumerate(candidates):
        # Extract keywords from conversations
        source_conv = conversations[candidate.source_id]
        target_conv = conversations[candidate.target_id]

        source_keywords = [kw["term"] for kw in source_conv.get("keywords", [])]
        target_keywords = [kw["term"] for kw in target_conv.get("keywords", [])]

        # Verify with LLM
        is_verified = llm_verify_edge(
            source_keywords, target_keywords, candidate.similarity, llm_client
        )

        if is_verified:
            is_intra_cluster = (
                candidate.source_cluster == candidate.target_cluster
                and candidate.source_cluster != -1
            )
            edge = Edge(
                source=candidate.source_id,
                target=candidate.target_id,
                weight=candidate.similarity,
                type="semantic",
                is_intra_cluster=is_intra_cluster,
                confidence="llm_verified",
            )
            verified_edges.append(edge)

        # Print progress
        if verbose and (idx + 1) % batch_size == 0:
            print(f"Verified {idx + 1}/{total} edge candidates")

    if verbose and total > 0:
        print(f"Verified {total}/{total} edge candidates")
        print(f"LLM approved {len(verified_edges)}/{total} edges")

    return verified_edges


def build_edges(
    intermediate_path: Path,
    cluster_path: Path,
    output_path: Path,
    high_threshold: float = 0.8,
    medium_threshold: float = 0.6,
    use_llm_verification: bool = True,
    verbose: bool = True,
) -> None:
    """
    Build edges from intermediate results and cluster assignments.

    Args:
        intermediate_path: Path to intermediate JSON (embeddings + conversations)
        cluster_path: Path to cluster assignments JSON
        output_path: Path to output edges JSON
        high_threshold: High confidence threshold (default: 0.8)
        medium_threshold: Medium confidence threshold (default: 0.6)
        use_llm_verification: Whether to use LLM verification for medium-confidence edges
        verbose: Whether to print progress information
    """
    if verbose:
        print(f"Loading intermediate results from {intermediate_path}")

    # Load intermediate results
    with open(intermediate_path, "r", encoding="utf-8") as f:
        intermediate_data = json.load(f)

    conversations = intermediate_data["conversations"]
    embeddings = np.array(intermediate_data["embeddings"], dtype=np.float32)

    if verbose:
        print(f"Loaded {len(conversations)} conversations with embeddings")

    # Load cluster assignments
    if verbose:
        print(f"Loading cluster assignments from {cluster_path}")

    with open(cluster_path, "r", encoding="utf-8") as f:
        cluster_data = json.load(f)

    # Create cluster_assignments dict: {conversation_id: cluster_id}
    cluster_assignments: Dict[int, int] = {}
    for cluster in cluster_data.get("clusters", []):
        cluster_id = cluster["cluster_id"]
        for conv_id in cluster["conversation_ids"]:
            cluster_assignments[conv_id] = cluster_id

    if verbose:
        print(f"Loaded cluster assignments for {len(cluster_assignments)} conversations")

    # Compute similarity matrix
    if verbose:
        print("Computing cosine similarity matrix...")

    similarity_matrix = compute_cosine_similarity(embeddings)

    # Generate edge candidates
    if verbose:
        print(
            f"Generating edge candidates (high_threshold={high_threshold}, medium_threshold={medium_threshold})..."
        )

    confirmed_edges, candidates_for_llm = generate_edge_candidates(
        similarity_matrix, cluster_assignments, high_threshold, medium_threshold
    )

    if verbose:
        print(f"Generated {len(confirmed_edges)} high-confidence edges")
        print(f"Generated {len(candidates_for_llm)} medium-confidence candidates")

    # Verify medium-confidence candidates with LLM
    verified_edges: List[Edge] = []
    if use_llm_verification and candidates_for_llm:
        if verbose:
            print("Verifying medium-confidence candidates with LLM...")

        # Initialize OpenAI client
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print(
                "Warning: OPENAI_API_KEY not found in environment. Skipping LLM verification."
            )
        else:
            llm_client = OpenAI(api_key=api_key)
            verified_edges = verify_candidates_with_llm(
                candidates_for_llm, conversations, llm_client, verbose=verbose
            )
    elif not use_llm_verification and verbose:
        print("LLM verification disabled, skipping medium-confidence candidates")

    # Combine all edges
    all_edges = confirmed_edges + verified_edges

    # Compute statistics
    total_edges = len(all_edges)
    intra_count = sum(1 for edge in all_edges if edge.is_intra_cluster)
    inter_count = total_edges - intra_count
    high_count = sum(1 for edge in all_edges if edge.confidence == "high")
    llm_count = sum(1 for edge in all_edges if edge.confidence == "llm_verified")

    if verbose:
        print("\n=== Edge Statistics ===")
        print(f"Total edges: {total_edges}")
        print(f"Intra-cluster edges: {intra_count} ({100*intra_count/total_edges:.1f}%)" if total_edges > 0 else "Intra-cluster edges: 0")
        print(f"Inter-cluster edges: {inter_count} ({100*inter_count/total_edges:.1f}%)" if total_edges > 0 else "Inter-cluster edges: 0")
        print(f"High-confidence edges: {high_count}")
        print(f"LLM-verified edges: {llm_count}")

    # Save to output JSON
    output_data = {
        "edges": [asdict(edge) for edge in all_edges],
        "metadata": {
            "total_edges": total_edges,
            "intra_cluster_edges": intra_count,
            "inter_cluster_edges": inter_count,
            "high_confidence_edges": high_count,
            "llm_verified_edges": llm_count,
            "thresholds": {"high": high_threshold, "medium": medium_threshold},
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    if verbose:
        print(f"\nEdges saved to {output_path.resolve()}")


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point for edge generation."""
    parser = argparse.ArgumentParser(
        description="Generate edges from embeddings and cluster assignments with optional LLM verification."
    )
    parser.add_argument(
        "--intermediate",
        type=Path,
        required=True,
        help="Path to intermediate JSON (embeddings + conversations)",
    )
    parser.add_argument(
        "--clusters",
        type=Path,
        required=True,
        help="Path to cluster assignments JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to output edges JSON",
    )
    parser.add_argument(
        "--high-threshold",
        type=float,
        default=0.8,
        help="High confidence threshold (default: 0.8)",
    )
    parser.add_argument(
        "--medium-threshold",
        type=float,
        default=0.6,
        help="Medium confidence threshold (default: 0.6)",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM verification for medium-confidence edges",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Enable verbose output",
    )

    args = parser.parse_args(argv)

    build_edges(
        intermediate_path=args.intermediate,
        cluster_path=args.clusters,
        output_path=args.output,
        high_threshold=args.high_threshold,
        medium_threshold=args.medium_threshold,
        use_llm_verification=not args.no_llm,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
