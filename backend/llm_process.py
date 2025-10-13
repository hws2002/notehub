import json
import re
from typing import Dict, List, Tuple, Set
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

from llm import llm_client

# --- Configuration ---
INPUT_PATH = "input_data/conversations.json"
OUTPUT_PATH = "graph_data/graph_data_llm.json"

# Thresholds for different processing stages
SEMANTIC_THRESHOLD = 0.6  # High threshold for automatic semantic linking
LLM_REVIEW_THRESHOLD = 0.4  # Lower threshold for LLM review
KEYWORD_OVERLAP_THRESHOLD = 3  # Minimum shared keywords for consideration

# Token optimization settings
MAX_TITLE_LENGTH = 100  # Truncate very long titles
BATCH_SIZE = 50  # Process LLM calls in batches


class OptimizedGraphProcessor:
    def __init__(self):
        self.llm_client = llm_client
        self.semantic_model = None
        self.conversations = []
        self.nodes = []

    def load_and_preprocess_conversations(self):
        """Step 1: Load conversations and extract minimal necessary data"""
        print("🔄 Loading and preprocessing conversations...")

        try:
            with open(INPUT_PATH, "r", encoding="utf-8") as f:
                raw_conversations = json.load(f)
        except FileNotFoundError:
            print(f"❌ ERROR: Input file not found at {INPUT_PATH}")
            return False

        for i, conv in enumerate(raw_conversations):
            # Extract title (primary signal)
            title = conv.get("title", f"Conversation {i+1}")
            if len(title) > MAX_TITLE_LENGTH:
                title = title[:MAX_TITLE_LENGTH] + "..."

            # Extract minimal content for context (only if needed)
            content_preview = self._extract_content_preview(conv.get("mapping", {}))

            # Create lightweight conversation object
            processed_conv = {
                "id": f"conv_{i}",
                "title": title,
                "content_preview": content_preview[:200],  # Very short preview
                "keywords": self._extract_keywords(title + " " + content_preview[:500]),
            }

            self.conversations.append(processed_conv)

            # Create node (no LLM needed)
            self.nodes.append(
                {"id": f"conv_{i}", "label": title, "type": "conversation"}
            )

        print(f"✅ Processed {len(self.conversations)} conversations")
        return True

    def _extract_content_preview(self, mapping: dict) -> str:
        """Extract a brief content preview without processing entire conversation"""
        if not mapping:
            return ""

        # Get first few meaningful messages only
        messages = []
        count = 0
        for msg_data in mapping.values():
            if count >= 3:  # Only first 3 messages for context
                break

            message_obj = msg_data.get("message", {})
            content_obj = message_obj.get("content") if message_obj else None
            parts = content_obj.get("parts", [""]) if content_obj else [""]

            # Safely extract text from the first part, handling dicts or strings
            message_text = ""
            if parts and len(parts) > 0:
                first_part = parts[0]
                if isinstance(first_part, str):
                    message_text = first_part
                elif isinstance(first_part, dict) and "text" in first_part:
                    message_text = first_part["text"]

            if message_text.strip():
                messages.append(message_text[:100])  # Truncate each message
                count += 1

        return " ".join(messages)

    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract meaningful keywords from text"""
        # Simple but effective keyword extraction
        words = re.findall(r"\b\w{4,}\b", text.lower())

        # Filter common words
        stop_words = {
            "with",
            "that",
            "this",
            "have",
            "will",
            "from",
            "they",
            "been",
            "were",
            "said",
            "each",
            "which",
            "them",
            "than",
            "many",
            "some",
            "time",
            "very",
            "when",
            "much",
            "where",
            "your",
            "well",
            "such",
        }

        return set(word for word in words if word not in stop_words)

    def create_semantic_links(self) -> List[Tuple[str, str, float]]:
        """Step 2: Use semantic similarity for high-confidence links"""
        print("🔄 Creating semantic links...")

        if self.semantic_model is None:
            self.semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Use titles + brief content for embedding
        texts = [
            f"{conv['title']} {conv['content_preview']}" for conv in self.conversations
        ]
        embeddings = self.semantic_model.encode(texts, show_progress_bar=True)

        similarity_matrix = cosine_similarity(embeddings)

        semantic_links = []
        for i in range(len(similarity_matrix)):
            for j in range(i + 1, len(similarity_matrix)):
                similarity = similarity_matrix[i][j]
                if similarity > SEMANTIC_THRESHOLD:
                    semantic_links.append(
                        (
                            self.conversations[i]["id"],
                            self.conversations[j]["id"],
                            float(similarity),
                        )
                    )

        print(f"✅ Found {len(semantic_links)} high-confidence semantic links")
        return semantic_links

    def create_keyword_links(self) -> List[Tuple[str, str, int]]:
        """Step 3: Find keyword-based connections for LLM review"""
        print("🔄 Finding keyword-based candidate links...")

        keyword_candidates = []

        for i in range(len(self.conversations)):
            for j in range(i + 1, len(self.conversations)):
                conv1 = self.conversations[i]
                conv2 = self.conversations[j]

                # Count shared keywords
                shared_keywords = conv1["keywords"] & conv2["keywords"]

                if len(shared_keywords) >= KEYWORD_OVERLAP_THRESHOLD:
                    keyword_candidates.append(
                        (conv1["id"], conv2["id"], len(shared_keywords))
                    )

        print(f"✅ Found {len(keyword_candidates)} keyword-based candidates")
        return keyword_candidates

    def llm_review_candidates(
        self, candidates: List[Tuple[str, str, int]]
    ) -> List[Tuple[str, str, float]]:
        """Step 4: Use LLM only for ambiguous cases - BATCH PROCESSING"""
        print("🔄 LLM reviewing candidate links...")

        if not candidates:
            return []

        llm_links = []
        id_to_conv = {conv["id"]: conv for conv in self.conversations}

        # Process in batches to reduce API calls
        for batch_start in range(0, len(candidates), BATCH_SIZE):
            batch = candidates[batch_start : batch_start + BATCH_SIZE]

            # Create a single prompt for multiple comparisons
            batch_prompt = self._create_batch_prompt(batch, id_to_conv)

            response = self.llm_client.request_llm_output(
                system_prompt="You are an AI that determines connection strengths between conversation topics. Return a JSON array of scores.",
                user_prompt=batch_prompt,
                temperature=0.0,
                max_tokens=500,  # Much smaller since we expect just numbers
            )

            if response:
                scores = self._parse_batch_response(response, batch)
                llm_links.extend(scores)

            print(
                f"    Processed batch {batch_start//BATCH_SIZE + 1}/{(len(candidates)-1)//BATCH_SIZE + 1}"
            )

        print(
            f"✅ LLM reviewed {len(candidates)} candidates, approved {len(llm_links)} links"
        )
        return llm_links

    def _create_batch_prompt(
        self, batch: List[Tuple[str, str, int]], id_to_conv: Dict
    ) -> str:
        """Create a single prompt for multiple conversation comparisons"""
        prompt = """Rate the connection strength between these conversation pairs (0.0-1.0).
Base your judgment primarily on titles. Return ONLY a JSON array of numbers, nothing else.

Pairs to rate:
"""

        for i, (id1, id2, _) in enumerate(batch):
            conv1 = id_to_conv[id1]
            conv2 = id_to_conv[id2]
            prompt += f"{i+1}. \"{conv1['title']}\" vs \"{conv2['title']}\"\n"

        prompt += "\nReturn format: [0.5, 0.8, 0.2, ...]"
        return prompt

    def _parse_batch_response(
        self, response: str, batch: List[Tuple[str, str, int]]
    ) -> List[Tuple[str, str, float]]:
        """Parse LLM batch response into individual scores"""
        try:
            # Extract JSON array from response
            json_match = re.search(r"\[[\d\.,\s]+\]", response)
            if json_match:
                scores = json.loads(json_match.group())

                result = []
                for i, score in enumerate(
                    scores[: len(batch)]
                ):  # Ensure we don't exceed batch size
                    if score > LLM_REVIEW_THRESHOLD:
                        id1, id2, _ = batch[i]
                        result.append((id1, id2, float(score)))

                return result
        except (json.JSONDecodeError, IndexError) as e:
            print(f"⚠️ Warning: Could not parse batch response: {e}")

        return []

    def generate_graph(self):
        """Main pipeline: combines all steps efficiently"""
        print("🚀 Starting optimized graph generation...")

        # Step 1: Load and preprocess (no LLM)
        if not self.load_and_preprocess_conversations():
            return

        # Step 2: High-confidence semantic links (no LLM)
        semantic_links = self.create_semantic_links()

        # Step 3: Find keyword candidates (no LLM)
        keyword_candidates = self.create_keyword_links()

        # Filter out candidates that already have semantic links
        semantic_pairs = {(link[0], link[1]) for link in semantic_links}
        filtered_candidates = [
            candidate
            for candidate in keyword_candidates
            if (candidate[0], candidate[1]) not in semantic_pairs
        ]

        print(f"🔍 Filtered to {len(filtered_candidates)} candidates for LLM review")

        # Step 4: LLM review only for ambiguous cases
        llm_links = self.llm_review_candidates(filtered_candidates)

        # Step 5: Combine all links
        final_links = []

        # Add semantic links
        for source, target, strength in semantic_links:
            final_links.append(
                {
                    "source": source,
                    "target": target,
                    "strength": strength,
                    "type": "semantic",
                }
            )

        # Add LLM-approved links
        for source, target, strength in llm_links:
            final_links.append(
                {
                    "source": source,
                    "target": target,
                    "strength": strength,
                    "type": "llm_approved",
                }
            )

        # Create final graph
        graph_data = {"nodes": self.nodes, "links": final_links}

        # Save to file
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False)

        print(f"\n🎉 Graph generation complete!")
        print(f"   Nodes: {len(self.nodes)}")
        print(
            f"   Links: {len(final_links)} ({len(semantic_links)} semantic + {len(llm_links)} LLM-approved)"
        )
        print(f"   Output: {OUTPUT_PATH}")
        print(f"   Total LLM calls: ~{len(filtered_candidates) // BATCH_SIZE + 1}")


def main():
    processor = OptimizedGraphProcessor()
    processor.generate_graph()


if __name__ == "__main__":
    main()
