"""LLM-based conversation clustering using various LLM providers (OpenAI, Qwen, Groq, Gemini)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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


class LLMClusteringClient:
    """Client for LLM-based clustering (supports OpenAI, Qwen, Groq, Gemini)."""

    def __init__(self, llm_client: BaseLLMClient):
        """Initialize the LLM clustering client.

        Args:
            llm_client: An instance of BaseLLMClient (OpenAI, Qwen, Groq, or Gemini)
        """
        self.llm_client = llm_client
        self.call_count = 0

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

        Raises:
            Exception: If API call fails
        """
        self.call_count += 1
        response = self.llm_client.call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if response is None:
            return None
        if not isinstance(response, str):
            raise TypeError(
                f"LLM client returned unexpected type {type(response)!r}; "
                "expected string response."
            )
        return response

    def generate_clusters(
        self,
        conversations: List[Dict[str, Any]],
        num_clusters: int = 2,
        verbose: bool = False,
    ) -> List[Cluster]:
        """Phase 1: Generate topic clusters based on conversation keywords.

        Args:
            conversations: List of conversation dictionaries with keywords
            num_clusters: Number of clusters to generate
            verbose: Whether to show detailed progress

        Returns:
            List of Cluster objects
        """
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

        user_prompt = f"""Analyze these {len(conversations)} conversations and their extracted keywords.

Create {num_clusters} major topic clusters that best categorize these conversations.

Conversations:
{conversations_text}

Requirements:
- Each cluster should represent a distinct topic/domain
- Cluster names should be clear and descriptive (2-5 words)
- Provide a brief description for each cluster
- Identify 3-5 key themes/terms for each cluster

Return ONLY valid JSON in this exact format:
{{
  "clusters": [
    {{
      "id": "cluster_1",
      "name": "Descriptive Name",
      "description": "One sentence description",
      "key_themes": ["theme1", "theme2", "theme3"]
    }}
  ]
}}"""

        if verbose:
            print(f"  Sending {len(conversations)} conversations to LLM...")

        response = self._call_llm(system_prompt, user_prompt, max_tokens=2000)

        if response is None:
            raise ValueError(
                "LLM returned no content while generating clusters. "
                "Check provider logs for details."
            )

        # Parse JSON response
        try:
            # Clean response if it contains markdown code blocks
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

            print(f"✅ Generated {len(clusters)} clusters")
            for cluster in clusters:
                print(f"  - {cluster.name}: {cluster.description}")

            return clusters

        except json.JSONDecodeError as exc:
            print(f"❌ Failed to parse LLM response as JSON: {exc}")
            print(f"Response: {response[:500]}...")
            raise

    def assign_conversations(
        self,
        conversations: List[Dict[str, Any]],
        clusters: List[Cluster],
        batch_size: int = 50,
        verbose: bool = False,
    ) -> List[Assignment]:
        """Phase 2: Assign conversations to clusters.

        Args:
            conversations: List of conversation dictionaries
            clusters: List of available clusters
            batch_size: Number of conversations to process per batch
            verbose: Whether to show detailed progress

        Returns:
            List of Assignment objects
        """
        print(
            f"\n🔗 Phase 2: Assigning {len(conversations)} conversations to clusters..."
        )

        all_assignments = []

        # Process in batches
        num_batches = (len(conversations) + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(conversations))
            batch = conversations[start_idx:end_idx]

            print(
                f"  Processing batch {batch_idx + 1}/{num_batches} ({len(batch)} conversations)..."
            )

            # Format clusters for the prompt
            clusters_json = json.dumps(
                [
                    {
                        "id": c.id,
                        "name": c.name,
                        "description": c.description,
                        "key_themes": c.key_themes,
                    }
                    for c in clusters
                ],
                indent=2,
                ensure_ascii=False,
            )

            # Format conversations for the prompt
            conversation_summaries = []
            for conv in batch:
                keywords_str = ", ".join(kw["term"] for kw in conv["keywords"][:5])
                conversation_summaries.append(
                    f'{{"id": {conv["id"]}, "keywords": "{keywords_str}"}}'
                )

            conversations_text = "\n".join(conversation_summaries)

            system_prompt = (
                "You are an expert at assigning conversations to relevant topic clusters. "
                "Return ONLY valid JSON without any markdown formatting or code blocks."
            )

            user_prompt = f"""Assign each conversation to the most appropriate cluster.

Available Clusters:
{clusters_json}

Conversations to Assign:
{conversations_text}

Requirements:
- Assign each conversation to exactly one cluster
- Provide confidence score (0.0 to 1.0)
- Base assignment on keyword relevance to cluster themes

Return ONLY valid JSON in this exact format:
{{
  "assignments": [
    {{
      "conversation_id": 0,
      "cluster_id": "cluster_1",
      "confidence": 0.92
    }}
  ]
}}"""

            response = self._call_llm(system_prompt, user_prompt, max_tokens=3000)

            if response is None:
                raise ValueError(
                    "LLM returned no content while assigning conversations. "
                    "Check provider logs for details."
                )

            # Parse JSON response
            try:
                # Clean response if it contains markdown code blocks
                if "```json" in response:
                    response = response.split("```json")[1].split("```")[0].strip()
                elif "```" in response:
                    response = response.split("```")[1].split("```")[0].strip()

                data = json.loads(response)

                for assign in data["assignments"]:
                    conv_id = assign["conversation_id"]
                    # Find the original conversation
                    conv = next(c for c in batch if c["id"] == conv_id)
                    top_keywords = [kw["term"] for kw in conv["keywords"][:3]]

                    all_assignments.append(
                        Assignment(
                            conversation_id=conv_id,
                            orig_id=conv["orig_id"],
                            cluster_id=assign["cluster_id"],
                            confidence=assign["confidence"],
                            top_keywords=top_keywords,
                        )
                    )

            except json.JSONDecodeError as exc:
                print(
                    f"❌ Failed to parse LLM response for batch {batch_idx + 1}: {exc}"
                )
                print(f"Response: {response[:500]}...")
                raise

        print(f"✅ Assigned all {len(all_assignments)} conversations")
        return all_assignments


def load_input_data(input_path: Path) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load conversation data from JSON file.

    Args:
        input_path: Path to input JSON file

    Returns:
        Tuple of (conversations list, metadata dict)

    Raises:
        ValueError: If input file format is invalid
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
) -> None:
    """Save clustering results to JSON file.

    Args:
        output_path: Path to output JSON file
        clusters: List of generated clusters
        assignments: List of conversation assignments
        input_metadata: Metadata from input file
        model_name: Name of the clustering model used
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
        "metadata": {
            "total_conversations": len(assignments),
            "num_clusters": len(clusters),
            "clustering_model": model_name,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "average_confidence": round(avg_confidence, 4),
            "input_metadata": input_metadata,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n💾 Results saved to {output_path}")
    print(f"\n📈 Clustering Statistics:")
    print(f"  Total conversations: {len(assignments)}")
    print(f"  Number of clusters: {len(clusters)}")
    print(f"  Average confidence: {avg_confidence:.4f}")
    print(f"\n  Cluster sizes:")
    for cluster in clusters:
        size = cluster_sizes.get(cluster.id, 0)
        print(f"    {cluster.name}: {size} conversations")


def main(argv: Optional[List[str]] = None) -> None:
    """Main entry point for the clustering script."""
    parser = argparse.ArgumentParser(
        description="Cluster conversations using LLM-based topic analysis."
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
        default=4,
        help="Number of clusters to create (default: 4)",
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
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for Phase 2 (default: 50)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed progress",
    )

    args = parser.parse_args(argv)

    print("🚀 Starting LLM-based conversation clustering")
    print(f"  Provider: {args.provider}")
    print(f"  Target clusters: {args.num_clusters}")

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
    client = LLMClusteringClient(llm_client=llm_client)

    # Phase 1: Generate clusters
    clusters = client.generate_clusters(
        conversations=conversations,
        num_clusters=args.num_clusters,
        verbose=args.verbose,
    )

    # Phase 2: Assign conversations
    assignments = client.assign_conversations(
        conversations=conversations,
        clusters=clusters,
        batch_size=args.batch_size,
        verbose=args.verbose,
    )

    # Save results
    save_output(
        output_path=args.output,
        clusters=clusters,
        assignments=assignments,
        input_metadata=input_metadata,
        model_name=f"{args.provider}/{llm_client.model_name}",
    )

    print(f"\n✨ Clustering complete! Total API calls: {client.call_count}")


if __name__ == "__main__":
    main()
