"""Hybrid conversation clustering: LLM for cluster generation, embeddings for assignment."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from llm_clients import BaseLLMClient, create_llm_client


@dataclass
class Cluster:
    """Represents a topic cluster."""

    id: str
    name: str
    description: str
    key_themes: List[str]


@dataclass
class Assignment:
    """Represents a conversation-to-cluster assignment."""

    conversation_id: int
    orig_id: str
    cluster_id: str
    confidence: float
    top_keywords: List[str]


class HybridClusteringClient:
    """Hybrid clustering: LLM for cluster generation, embeddings for assignment."""

    def __init__(self, llm_client: BaseLLMClient):
        """Initialize the clustering client.

        Args:
            llm_client: An instance of BaseLLMClient for LLM operations
        """
        self.llm_client = llm_client
        self.call_count = 0
        self.embedding_model = None

    def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4000,
        temperature: float = 0.7,
    ) -> str:
        """Make a call to the LLM.

        Args:
            system_prompt: System-level instructions
            user_prompt: User query/request
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature

        Returns:
            LLM response text
        """
        self.call_count += 1
        response = self.llm_client.call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if response is None:
            raise ValueError("LLM returned no content")
        if not isinstance(response, str):
            raise TypeError(
                f"LLM client returned unexpected type {type(response)!r}; "
                "expected string response."
            )
        return response

    def generate_clusters(
        self,
        conversations: List[Dict[str, Any]],
        num_clusters: Optional[int] = None,
        min_clusters: int = 3,
        max_clusters: int = 5,
        verbose: bool = False,
    ) -> List[Cluster]:
        """Phase 1: Generate topic clusters using LLM.

        Args:
            conversations: List of conversation dictionaries with keywords
            num_clusters: Fixed number of clusters (if None, LLM chooses)
            min_clusters: Minimum clusters when num_clusters is None
            max_clusters: Maximum clusters when num_clusters is None
            verbose: Whether to show detailed progress

        Returns:
            List of Cluster objects
        """
        if num_clusters is None:
            print(
                f"\n📊 Phase 1: Letting LLM choose {min_clusters}-{max_clusters} topic clusters..."
            )
        else:
            print(f"\n📊 Phase 1: Generating {num_clusters} topic clusters...")

        # Format conversation data for the prompt
        conversation_summaries = []
        for conv in conversations:
            keywords_str = ", ".join(
                f"{kw['term']} ({kw['score']:.2f})" for kw in conv["keywords"][:5]
            )
            conversation_summaries.append(f"ID {conv['id']}: {keywords_str}")

        conversations_text = "\n".join(conversation_summaries)

        system_prompt = (
            "You are an expert at analyzing conversation topics and creating meaningful clusters. "
            "Return ONLY valid JSON without any markdown formatting or code blocks."
        )

        # Build prompt based on whether num_clusters is fixed
        if num_clusters is None:
            cluster_instruction = (
                f"Determine the optimal number of clusters between {min_clusters} and {max_clusters}. "
                'Provide the chosen number as "num_clusters" and create that many clusters.'
            )
            return_format = """{
  "num_clusters": 3,
  "clusters": [
    {
      "id": "cluster_1",
      "name": "Descriptive Name",
      "description": "One sentence description",
      "key_themes": ["theme1", "theme2", "theme3"]
    }
  ]
}"""
        else:
            cluster_instruction = f"Create {num_clusters} major topic clusters."
            return_format = """{
  "clusters": [
    {
      "id": "cluster_1",
      "name": "Descriptive Name",
      "description": "One sentence description",
      "key_themes": ["theme1", "theme2", "theme3"]
    }
  ]
}"""

        user_prompt = f"""Analyze these {len(conversations)} conversations and their extracted keywords.

{cluster_instruction}

Conversations:
{conversations_text}

Requirements:
- Each cluster should represent a distinct topic/domain
- Cluster names should be clear and descriptive (2-5 words)
- Provide a brief description for each cluster
- Identify 3-5 key themes/terms for each cluster

Return ONLY valid JSON in this exact format:
{return_format}"""

        if verbose:
            print(f"  Analyzing {len(conversations)} conversations...")

        response = self._call_llm(
            system_prompt, user_prompt, max_tokens=2000, temperature=0.7
        )

        # Parse JSON response
        try:
            # Clean markdown code blocks if present
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()

            data = json.loads(response)
            clusters = [
                Cluster(
                    id=c["id"],
                    name=c["name"],
                    description=c["description"],
                    key_themes=c["key_themes"],
                )
                for c in data["clusters"]
            ]

            selected_num = data.get("num_clusters", len(clusters))
            print(f"✅ Generated {len(clusters)} clusters")
            if num_clusters is None:
                print(f"  (LLM selected: {selected_num})")

            for cluster in clusters:
                print(f"  - {cluster.name}: {cluster.description}")

            return clusters

        except json.JSONDecodeError as exc:
            print(f"❌ Failed to parse LLM response as JSON: {exc}")
            print(f"Response: {response[:500]}...")
            raise

    def _load_embedding_model(self):
        """Load the embedding model (lazy loading)."""
        if self.embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer

                print("  Loading embedding model (all-MiniLM-L6-v2)...")
                self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            except ImportError:
                print("\n❌ sentence-transformers not installed.")
                print("Install it with: pip install sentence-transformers")
                sys.exit(1)
        return self.embedding_model

    def assign_conversations_with_embeddings(
        self,
        conversations: List[Dict[str, Any]],
        clusters: List[Cluster],
        verbose: bool = False,
    ) -> List[Assignment]:
        """Phase 2: Assign conversations to clusters using embeddings.

        Args:
            conversations: List of conversation dictionaries
            clusters: List of available clusters
            verbose: Whether to show detailed progress

        Returns:
            List of Assignment objects
        """
        from sklearn.metrics.pairwise import cosine_similarity

        print(
            f"\n🔗 Phase 2: Assigning {len(conversations)} conversations using embeddings..."
        )

        # Load embedding model
        model = self._load_embedding_model()

        # Create cluster embeddings from all cluster information
        cluster_texts = []
        for cluster in clusters:
            cluster_text = (
                f"{cluster.name}. {cluster.description}. "
                f"Key themes: {', '.join(cluster.key_themes)}"
            )
            cluster_texts.append(cluster_text)

        print(f"  Embedding {len(clusters)} clusters...")
        cluster_embeddings = model.encode(cluster_texts, show_progress_bar=False)

        # Assign each conversation
        print(f"  Assigning conversations...")
        assignments = []
        batch_size = 100  # Process in batches for progress reporting

        for i in range(0, len(conversations), batch_size):
            batch = conversations[i : i + batch_size]

            if verbose:
                print(
                    f"    Processing {i + 1}-{min(i + batch_size, len(conversations))}..."
                )

            # Create conversation embeddings from keywords
            conv_texts = []
            for conv in batch:
                # Use top 15 keywords for richer representation
                keywords = [kw["term"] for kw in conv["keywords"][:15]]
                conv_texts.append(" ".join(keywords))

            # Get batch embeddings
            conv_embeddings = model.encode(conv_texts, show_progress_bar=False)

            # Calculate similarities and create assignments
            similarities = cosine_similarity(conv_embeddings, cluster_embeddings)

            for j, conv in enumerate(batch):
                # Find best matching cluster
                best_cluster_idx = int(np.argmax(similarities[j]))
                confidence = float(similarities[j][best_cluster_idx])

                assignments.append(
                    Assignment(
                        conversation_id=conv["id"],
                        orig_id=conv["orig_id"],
                        cluster_id=clusters[best_cluster_idx].id,
                        confidence=confidence,
                        top_keywords=[kw["term"] for kw in conv["keywords"][:3]],
                    )
                )

        print(f"✅ Assigned all {len(assignments)} conversations")

        # Print statistics
        confidences = [a.confidence for a in assignments]
        print(f"\n  Confidence statistics:")
        print(f"    Average: {np.mean(confidences):.4f}")
        print(f"    Min: {np.min(confidences):.4f}")
        print(f"    Max: {np.max(confidences):.4f}")

        # Show cluster distribution
        cluster_counts = {}
        for assignment in assignments:
            cluster_counts[assignment.cluster_id] = (
                cluster_counts.get(assignment.cluster_id, 0) + 1
            )

        print(f"\n  Cluster distribution:")
        for cluster in clusters:
            count = cluster_counts.get(cluster.id, 0)
            pct = 100 * count / len(assignments) if assignments else 0
            print(f"    {cluster.name}: {count} conversations ({pct:.1f}%)")

        return assignments


def load_input_data(input_path: Path) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load conversation data from JSON file.

    Args:
        input_path: Path to input JSON file

    Returns:
        Tuple of (conversations list, metadata dict)
    """
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "conversations" not in data:
            raise ValueError("Input file must contain 'conversations' field")

        conversations = data["conversations"]
        metadata = data.get("metadata", {})

        print(f"📂 Loaded {len(conversations)} conversations from {input_path}")
        return conversations, metadata

    except Exception as exc:
        print(f"❌ Failed to load input file: {exc}")
        raise


def save_output(
    output_path: Path,
    clusters: List[Cluster],
    assignments: List[Assignment],
    input_metadata: Dict[str, Any],
    model_name: str,
    cluster_range: Optional[tuple[int, int]] = None,
) -> None:
    """Save clustering results to JSON file.

    Args:
        output_path: Path to output JSON file
        clusters: List of generated clusters
        assignments: List of conversation assignments
        input_metadata: Metadata from input file
        model_name: Name of the clustering model used
        cluster_range: Optional (min, max) cluster range if applicable
    """
    # Calculate statistics
    cluster_sizes = {}
    confidence_scores = []

    for assign in assignments:
        cluster_sizes[assign.cluster_id] = cluster_sizes.get(assign.cluster_id, 0) + 1
        confidence_scores.append(assign.confidence)

    avg_confidence = (
        sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
    )

    metadata: Dict[str, Any] = {
        "total_conversations": len(assignments),
        "num_clusters": len(clusters),
        "clustering_model": model_name,
        "assignment_method": "embeddings (all-MiniLM-L6-v2)",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "average_confidence": round(avg_confidence, 4),
        "input_metadata": input_metadata,
    }

    if cluster_range is not None:
        metadata["target_cluster_range"] = {
            "min": cluster_range[0],
            "max": cluster_range[1],
        }

    output_data = {
        "clusters": [
            {
                "id": c.id,
                "name": c.name,
                "description": c.description,
                "key_themes": c.key_themes,
                "size": cluster_sizes.get(c.id, 0),
            }
            for c in clusters
        ],
        "assignments": [
            {
                "conversation_id": a.conversation_id,
                "orig_id": a.orig_id,
                "cluster_id": a.cluster_id,
                "confidence": round(a.confidence, 4),
                "top_keywords": a.top_keywords,
            }
            for a in assignments
        ],
        "metadata": metadata,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n💾 Results saved to {output_path}")
    print(f"\n📈 Clustering Statistics:")
    print(f"  Total conversations: {len(assignments)}")
    print(f"  Number of clusters: {len(clusters)}")
    if cluster_range is not None:
        print(f"  Requested cluster range: {cluster_range[0]}-{cluster_range[1]}")
    print(f"  Average confidence: {avg_confidence:.4f}")
    print(f"\n  Cluster sizes:")
    for cluster in clusters:
        size = cluster_sizes.get(cluster.id, 0)
        print(f"    {cluster.name}: {size} conversations")


def main(argv: Optional[List[str]] = None) -> None:
    """Main entry point for the clustering script."""
    parser = argparse.ArgumentParser(
        description="Hybrid conversation clustering: LLM for clusters, embeddings for assignment."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default="test_output.json",
        help="Path to input JSON file (default: test_output.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="clustered_output.json",
        help="Path to output JSON file (default: clustered_output.json)",
    )
    parser.add_argument(
        "--num-clusters",
        type=int,
        default=None,
        help="Fixed number of clusters. If omitted, LLM chooses within range.",
    )
    parser.add_argument(
        "--min-clusters",
        type=int,
        default=3,
        help="Minimum clusters for LLM to consider (default: 3). Ignored if --num-clusters set.",
    )
    parser.add_argument(
        "--max-clusters",
        type=int,
        default=5,
        help="Maximum clusters for LLM to consider (default: 5). Ignored if --num-clusters set.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "qwen", "groq", "gemini"],
        default="openai",
        help="LLM provider to use (default: openai)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to use (default: provider-specific default)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed progress",
    )

    args = parser.parse_args(argv)

    # Validation
    if args.num_clusters is not None and args.num_clusters <= 0:
        print("❌ Error: --num-clusters must be positive.")
        sys.exit(1)
    if args.num_clusters is None:
        if args.min_clusters <= 0 or args.max_clusters <= 0:
            print("❌ Error: --min-clusters and --max-clusters must be positive.")
            sys.exit(1)
        if args.min_clusters > args.max_clusters:
            print("❌ Error: --min-clusters cannot exceed --max-clusters.")
            sys.exit(1)

    print("🚀 Starting hybrid conversation clustering")
    print(f"  Provider: {args.provider}")
    if args.num_clusters is not None:
        print(f"  Target clusters: {args.num_clusters}")
    else:
        print(f"  Target cluster range: {args.min_clusters}-{args.max_clusters}")

    # Load input data
    conversations, input_metadata = load_input_data(args.input)

    # Initialize LLM client
    try:
        llm_client = create_llm_client(
            provider=args.provider,
            model_name=args.model,
        )
        print(f"  Model: {llm_client.model_name}")
    except (ValueError, ImportError) as exc:
        print(f"❌ Error: {exc}")
        sys.exit(1)

    # Initialize clustering client
    client = HybridClusteringClient(llm_client=llm_client)

    # Phase 1: Generate clusters using LLM
    clusters = client.generate_clusters(
        conversations=conversations,
        num_clusters=args.num_clusters,
        min_clusters=args.min_clusters,
        max_clusters=args.max_clusters,
        verbose=args.verbose,
    )

    # Phase 2: Assign conversations using embeddings
    assignments = client.assign_conversations_with_embeddings(
        conversations=conversations,
        clusters=clusters,
        verbose=args.verbose,
    )

    # Save results
    cluster_range = (
        (args.min_clusters, args.max_clusters) if args.num_clusters is None else None
    )
    save_output(
        output_path=args.output,
        clusters=clusters,
        assignments=assignments,
        input_metadata=input_metadata,
        model_name=f"{args.provider}/{llm_client.model_name}",
        cluster_range=cluster_range,
    )

    print(f"\n✨ Clustering complete! LLM API calls: {client.call_count}")


if __name__ == "__main__":
    main()
