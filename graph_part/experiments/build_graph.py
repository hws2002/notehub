"""CLI entry point for building conversation similarity graphs."""

from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import math
import re
import string
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import yaml
from pydantic import ValidationError

from io_schemas import (
    ChatHistory,
    ClusterParams,
    ClusterSummary,
    Conversation,
    Counts,
    Edge,
    GraphParams,
    Keyword,
    KeywordParams,
    Message,
    Metadata,
    Node,
    OutputGraph,
    Params,
    PreprocessParams,
)

np.random.seed(42)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - handled by fallback
    SentenceTransformer = None  # type: ignore

try:
    from keybert import KeyBERT
except ImportError as exc:  # pragma: no cover - library required
    raise SystemExit(
        "KeyBERT is required. Install dependencies via requirements.txt"
    ) from exc

try:
    import hdbscan
except ImportError as exc:  # pragma: no cover - library required
    raise SystemExit(
        "hdbscan is required. Install dependencies via requirements.txt"
    ) from exc

from sklearn.feature_extraction.text import TfidfVectorizer
from stopwordsiso import stopwords as stopwords_iso

MAX_CHARS_PER_CHUNK = 512
CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
URL_RE = re.compile(r"https?://\S+|www\.\S+")
WHITESPACE_RE = re.compile(r"\s+")
SENTENCE_END_RE = re.compile(r"(?<=[\.\!\?。？！])\s+")
TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)
PUNCT_TRANSLATION = str.maketrans("", "", string.punctuation)
FILLER_TOKENS = {
    "ㅎㅎ",
    "ㅋㅋ",
    "lol",
    "haha",
    "uh",
    "um",
    "mmm",
    "음",
    "ㅎㅎㅎ",
    "ㅋㅋㅋ",
}


@dataclass
class KeywordConfig:
    top_n: int
    max_ngram: int
    dedup_thresh: float


@dataclass
class GraphConfig:
    sim_top_k: Optional[int]
    sim_threshold: Optional[float]


@dataclass
class ClusterConfig:
    min_cluster_size: int
    min_samples: int
    metric: str


@dataclass
class PreprocessConfig:
    lower: bool
    strip_urls: bool
    strip_code: bool
    strip_punct: bool
    stopwords_langs: List[str]


@dataclass
class Config:
    embedding_model: str
    keyword: KeywordConfig
    graph: GraphConfig
    cluster: ClusterConfig
    preprocess: PreprocessConfig


@dataclass
class EmbeddingStats:
    elapsed: float
    documents: int
    dimension: int
    norm_mean: Optional[float]
    norm_std: Optional[float]
    similarity_mean: Optional[float]
    similarity_std: Optional[float]
    similarity_min: Optional[float]
    similarity_max: Optional[float]


@dataclass
class KeywordStats:
    elapsed: float
    documents_with_keywords: int
    total_documents: int
    avg_keywords: Optional[float]
    median_keywords: Optional[float]
    unique_keyword_ratio: Optional[float]
    avg_keyword_score: Optional[float]
    max_keyword_score: Optional[float]


@dataclass
class PipelineMetrics:
    embedding: EmbeddingStats
    keyword: KeywordStats


@dataclass
class ConversationSummary:
    """Lightweight conversation summary for LLM-based clustering."""

    id: int
    orig_id: str
    keywords: List[Keyword]
    timestamp: Optional[str]
    num_messages: int


@dataclass
class IntermediateResult:
    """Intermediate result containing only data needed for LLM clustering."""

    conversations: List[ConversationSummary]
    metadata: Dict[str, Any]


class DummySentenceTransformer:
    """Deterministic embedding fallback when the real model cannot be loaded."""

    def __init__(self, model_name: str, dimension: int = 384) -> None:
        self.model_name = model_name
        self._dimension = dimension

    def encode(
        self,
        sentences: Sequence[str],
        *,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = False,
    ) -> np.ndarray:
        vectors: List[np.ndarray] = []
        for sentence in sentences:
            if not sentence:
                vec = np.zeros(self._dimension, dtype=np.float32)
            else:
                digest = hashlib.sha1(sentence.encode("utf-8")).hexdigest()
                seed = int(digest[:16], 16)
                rng = np.random.default_rng(seed)
                vec = rng.standard_normal(self._dimension).astype(np.float32)
            if normalize_embeddings:
                norm = float(np.linalg.norm(vec))
                if norm > 0:
                    vec = vec / norm
            vectors.append(vec)
        stacked = np.vstack(vectors)
        return stacked if convert_to_numpy else stacked.tolist()

    def get_sentence_embedding_dimension(self) -> int:
        return self._dimension


def load_embedding_model(model_name: str):
    """Load a sentence transformer model with a deterministic fallback."""
    if SentenceTransformer is None:
        warnings.warn(
            "sentence-transformers not available; using deterministic fallback embeddings."
        )
        return DummySentenceTransformer(model_name)
    try:
        return SentenceTransformer(model_name)
    except Exception as exc:  # pragma: no cover - network/cache failures
        warnings.warn(
            f"Failed to load {model_name}: {exc}. Using deterministic fallback embeddings."
        )
        return DummySentenceTransformer(model_name)


def load_config(path: Path) -> Config:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    keyword_cfg = KeywordConfig(**data["keyword"])
    graph_cfg = GraphConfig(**data["graph"])
    cluster_cfg = ClusterConfig(**data["cluster"])
    preprocess_cfg = PreprocessConfig(**data["preprocess"])
    return Config(
        embedding_model=data["embedding_model"],
        keyword=keyword_cfg,
        graph=graph_cfg,
        cluster=cluster_cfg,
        preprocess=preprocess_cfg,
    )


def _is_message_like(item: Dict[str, Any]) -> bool:
    required = {"id", "role", "content"}
    return required.issubset(item.keys())


def _group_chatgpt_export(payload: Iterable[Dict[str, Any]]) -> List[Conversation]:
    """Group ChatGPT export data into Conversation objects."""
    conversations: List[Conversation] = []

    for conv_idx, conversation in enumerate(payload):
        mapping = conversation.get("mapping")
        if not isinstance(mapping, dict):
            continue

        title = conversation.get("title", f"Conversation {conv_idx}")
        create_time = conversation.get("create_time")
        update_time = conversation.get("update_time")
        messages: List[Message] = []

        def iter_nodes() -> Iterable[Dict[str, Any]]:
            visited: set[str] = set()
            roots = [node for node in mapping.values() if node.get("parent") is None]

            def sort_key(node: Dict[str, Any]) -> str:
                message = node.get("message")
                if isinstance(message, dict):
                    return str(message.get("id", ""))
                return ""

            roots.sort(key=sort_key)
            queue: deque[Dict[str, Any]] = deque(roots)

            while queue:
                node = queue.popleft()
                node_id = node.get("id")
                if node_id in visited:
                    continue
                visited.add(node_id)
                yield node

                for child_id in node.get("children") or []:
                    child = mapping.get(child_id)
                    if child:
                        queue.append(child)

            for node in mapping.values():
                node_id = node.get("id")
                if node_id not in visited:
                    yield node

        for node in iter_nodes():
            message = node.get("message") or {}
            message_id = message.get("id")
            author = message.get("author") or {}
            role = author.get("role")
            content_obj = message.get("content") or {}

            content: str = ""
            if isinstance(content_obj, dict):
                parts = content_obj.get("parts")
                if isinstance(parts, list):
                    content = "\n".join(str(part) for part in parts)
                elif isinstance(content_obj.get("text"), str):
                    content = content_obj["text"]
            elif isinstance(content_obj, str):
                content = content_obj

            timestamp: Optional[str] = message.get("timestamp")
            if isinstance(timestamp, (int, float)):
                timestamp = None

            if not message_id or role is None or content is None:
                continue

            messages.append(
                Message(
                    id=str(message_id),
                    role=str(role),
                    content=str(content),
                    timestamp=timestamp,
                )
            )

        if messages:
            conv_id = (
                messages[0].id.rsplit("_", 1)[0]
                if "_" in messages[0].id
                else f"conv_{conv_idx}"
            )
            conversations.append(
                Conversation(
                    id=conv_id,
                    title=title,
                    messages=messages,
                    create_time=create_time,
                    update_time=update_time,
                )
            )

    return conversations


def _group_messages_by_conversation(messages: List[Message]) -> List[Conversation]:
    """Group simple message list by conversation ID prefix."""
    from collections import defaultdict

    grouped: Dict[str, List[Message]] = defaultdict(list)

    for msg in messages:
        conv_id = msg.id.rsplit("_", 1)[0] if "_" in msg.id else "conv_0"
        grouped[conv_id].append(msg)

    conversations: List[Conversation] = []
    for conv_id, conv_messages in grouped.items():
        conv_messages.sort(key=lambda m: m.timestamp or "")
        conversations.append(
            Conversation(
                id=conv_id,
                title=f"Conversation {conv_id}",
                messages=conv_messages,
            )
        )

    return conversations


def load_messages(path: Path) -> ChatHistory:
    """
    Load messages and group them into conversations.

    Supports ChatGPT export payloads and flat message lists.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(payload, list):
        if isinstance(payload, dict):
            conversations = _group_chatgpt_export([payload])
            if not conversations:
                raise ValueError(
                    "Input JSON must be a list of message objects or a ChatGPT export mapping."
                )
            return ChatHistory.from_conversations(conversations)
        raise ValueError("Input JSON must be a list of message objects.")

    if not payload:
        return ChatHistory.from_conversations([])

    if payload and all(
        isinstance(item, dict) and _is_message_like(item) for item in payload
    ):
        messages = [Message(**item) for item in payload]
        conversations = _group_messages_by_conversation(messages)
        return ChatHistory.from_conversations(conversations)

    try:
        conversations = _group_chatgpt_export(payload)
        if not conversations:
            raise ValueError("Unable to interpret input JSON as chat messages.")
        return ChatHistory.from_conversations(conversations)
    except Exception as exc:
        raise ValueError(
            "Input must be either a list of messages or ChatGPT export format."
        ) from exc


def strip_code_blocks(text: str) -> str:
    return CODE_BLOCK_RE.sub(" ", text)


def strip_urls(text: str) -> str:
    return URL_RE.sub(" ", text)


def preprocess_text(text: str, cfg: PreprocessConfig) -> str:
    cleaned = text
    if cfg.strip_code:
        cleaned = strip_code_blocks(cleaned)
    if cfg.strip_urls:
        cleaned = strip_urls(cleaned)
    if cfg.lower:
        cleaned = cleaned.lower()
    if cfg.strip_punct:
        cleaned = cleaned.translate(PUNCT_TRANSLATION)
    cleaned = WHITESPACE_RE.sub(" ", cleaned)
    return cleaned.strip()


def chunk_text(text: str, max_chars: int = MAX_CHARS_PER_CHUNK) -> List[str]:
    if len(text) <= max_chars:
        return [text] if text else []
    parts = SENTENCE_END_RE.split(text)
    parts = [part.strip() for part in parts if part.strip()]
    if not parts:
        return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]
    chunks: List[str] = []
    current = ""
    for part in parts:
        candidate = part if not current else f"{current} {part}"
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current.strip())
        if len(part) <= max_chars:
            current = part
        else:
            for start in range(0, len(part), max_chars):
                segment = part[start : start + max_chars].strip()
                if segment:
                    chunks.append(segment)
            current = ""
    if current:
        chunks.append(current.strip())
    return chunks


def get_embedding_dimension(model) -> int:
    if hasattr(model, "get_sentence_embedding_dimension"):
        dim = model.get_sentence_embedding_dimension()
        if isinstance(dim, int):
            return dim
    sample = model.encode(["dimension_probe"], convert_to_numpy=True)
    if isinstance(sample, list):
        sample = np.array(sample)
    return int(sample.shape[1])


def mean_pool_embeddings(model, texts: Sequence[str]) -> np.ndarray:
    embeddings = model.encode(list(texts), convert_to_numpy=True)
    if isinstance(embeddings, list):
        embeddings = np.array(embeddings)
    return embeddings.mean(axis=0)


def generate_embeddings(cleaned_texts: Sequence[str], model) -> np.ndarray:
    dimension = get_embedding_dimension(model)
    vectors: List[np.ndarray] = []
    for text in cleaned_texts:
        chunks = chunk_text(text) if text else []
        if chunks:
            vec = mean_pool_embeddings(model, chunks)
        else:
            vec = np.zeros(dimension, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        vectors.append(vec.astype(np.float32))
    return np.vstack(vectors) if vectors else np.zeros((0, dimension), dtype=np.float32)


def build_stopwords(langs: Iterable[str]) -> List[str]:
    stoplist = set()
    for lang in langs:
        try:
            stoplist.update(stopwords_iso(lang))
        except KeyError:
            warnings.warn(f"No stopwords found for language '{lang}'.")
    stoplist.update(FILLER_TOKENS)
    cleaned = {word.strip().lower() for word in stoplist if word and word.strip()}
    return sorted(cleaned)


def keyword_token_set(term: str) -> set:
    tokens = TOKEN_RE.findall(term.lower())
    if not tokens:
        tokens = list(term.lower())
    return set(tokens)


def jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union else 0.0


def deduplicate_keywords(
    candidates: Sequence[Tuple[str, float]],
    *,
    limit: int,
    threshold: float,
) -> List[Keyword]:
    selected: List[Keyword] = []
    seen: List[set] = []
    for term, score in candidates:
        token_set = keyword_token_set(term)
        if any(jaccard(token_set, prev) >= threshold for prev in seen):
            continue
        seen.append(token_set)
        selected.append(Keyword(term=term, score=float(score)))
        if len(selected) >= limit:
            break
    return selected


def extract_keywords(
    texts: Sequence[str],
    config: KeywordConfig,
    stoplist: Optional[List[str]],
    keybert_model: KeyBERT,
    doc_embeddings: Optional[np.ndarray] = None,
) -> List[List[Keyword]]:
    results: List[List[Keyword]] = []
    stop_words = stoplist if stoplist else None
    doc_embeddings_arr = doc_embeddings
    if doc_embeddings_arr is not None:
        if not isinstance(doc_embeddings_arr, np.ndarray) or doc_embeddings_arr.ndim < 1:
            warnings.warn(
                "Document embeddings must be a numpy array; ignoring provided embeddings."
            )
            doc_embeddings_arr = None
        elif len(doc_embeddings_arr) != len(texts):
            warnings.warn(
                "Document embeddings count does not match number of texts; ignoring provided embeddings."
            )
            doc_embeddings_arr = None

    for idx, text in enumerate(texts):
        if not text:
            results.append([])
            continue
        doc_embedding: Optional[np.ndarray] = None
        if doc_embeddings_arr is not None:
            doc_embedding = doc_embeddings_arr[idx]
            if isinstance(doc_embedding, np.ndarray):
                if doc_embedding.ndim == 1:
                    doc_embedding = doc_embedding.reshape(1, -1)
                elif doc_embedding.ndim != 2:
                    warnings.warn(
                        "Unexpected document embedding shape; falling back to KeyBERT's internal embeddings."
                    )
                    doc_embedding = None
            else:
                doc_embedding = None
        try:
            raw_keywords = keybert_model.extract_keywords(
                text,
                top_n=config.top_n * 4,
                keyphrase_ngram_range=(1, config.max_ngram),
                stop_words=stop_words,
                doc_embeddings=doc_embedding,
            )
        except Exception as exc:  # pragma: no cover - defensive
            warnings.warn(f"Keyword extraction failed: {exc}")
            raw_keywords = []
        deduped = deduplicate_keywords(
            raw_keywords, limit=config.top_n, threshold=config.dedup_thresh
        )
        results.append(deduped)
    return results


def cluster_embeddings(embeddings: np.ndarray, config: ClusterConfig) -> np.ndarray:
    if embeddings.size == 0:
        return np.array([], dtype=int)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=config.min_cluster_size,
        min_samples=config.min_samples,
        metric=config.metric,
        allow_single_cluster=True,
    )
    labels = clusterer.fit_predict(embeddings)
    if np.all(labels == -1) and len(labels) > 0:
        warnings.warn(
            "HDBSCAN produced only outliers; assigning fallback single cluster."
        )
        labels = np.zeros_like(labels)
    return labels


def build_similarity_edges(embeddings: np.ndarray, config: GraphConfig) -> List[Edge]:
    n_nodes = embeddings.shape[0]
    if n_nodes < 2:
        return []
    similarity = embeddings @ embeddings.T
    np.fill_diagonal(similarity, -math.inf)
    edges: Dict[Tuple[int, int], Edge] = {}
    if config.sim_top_k:
        k = min(config.sim_top_k, n_nodes - 1)
        if k > 0:
            for i in range(n_nodes):
                row = similarity[i]
                top_indices = np.argsort(-row)[:k]
                for j in top_indices:
                    if j == i:
                        continue
                    source, target = (i, j) if i < j else (j, i)
                    weight = float(similarity[source, target])
                    edges[(source, target)] = Edge(
                        source=source, target=target, weight=weight, type="similarity"
                    )
    elif config.sim_threshold is not None:
        threshold = float(config.sim_threshold)
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                weight = float(similarity[i, j])
                if weight >= threshold:
                    edges[(i, j)] = Edge(
                        source=i, target=j, weight=weight, type="similarity"
                    )
    else:
        raise ValueError("Graph configuration must specify sim_top_k or sim_threshold.")
    return list(edges.values())


def summarize_clusters(
    texts: Sequence[str],
    labels: Sequence[int],
    stoplist: Optional[List[str]],
    top_terms: int,
) -> Dict[str, ClusterSummary]:
    clusters: Dict[int, List[str]] = {}
    for text, label in zip(texts, labels):
        if label == -1:
            continue
        clusters.setdefault(label, []).append(text)
    summaries: Dict[str, ClusterSummary] = {}
    if not clusters:
        return summaries
    stop_words = stoplist if stoplist else None
    for label, cluster_texts in clusters.items():
        joined = [text for text in cluster_texts if text]
        if not joined:
            continue
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words=stop_words)
        matrix = vectorizer.fit_transform(joined)
        scores = np.asarray(matrix.sum(axis=0)).ravel()
        terms = vectorizer.get_feature_names_out()
        order = np.argsort(-scores)[:top_terms]
        top = [(terms[idx], float(scores[idx])) for idx in order if scores[idx] > 0]
        summaries[str(label)] = ClusterSummary(size=len(cluster_texts), top_terms=top)
    return summaries


def compute_metadata(
    config: Config,
    labels: Sequence[int],
    edges: Sequence[Edge],
    stoplist: Optional[List[str]],
    texts_for_summary: Sequence[str],
) -> Metadata:
    clusters_present = {label for label in labels if label != -1}
    counts = Counts(
        nodes=len(labels),
        edges=len(edges),
        clusters=len(clusters_present),
        outliers=sum(1 for label in labels if label == -1),
    )
    params = Params(
        embedding_model=config.embedding_model,
        embedding_model_digest=hashlib.sha1(
            config.embedding_model.encode("utf-8")
        ).hexdigest()[:8],
        keyword=KeywordParams(**config.keyword.__dict__),
        cluster=ClusterParams(**config.cluster.__dict__),
        graph=GraphParams(**config.graph.__dict__),
        preprocess=PreprocessParams(**config.preprocess.__dict__),
    )
    summaries = summarize_clusters(
        texts=texts_for_summary,
        labels=labels,
        stoplist=stoplist,
        top_terms=config.keyword.top_n,
    )
    return Metadata(clusters=summaries, params=params, counts=counts)


def run_pipeline(
    chat_history: ChatHistory, config: Config, *, collect_metrics: bool = False
) -> Tuple[OutputGraph, Optional[PipelineMetrics]]:
    """
    Build the similarity graph using conversation-level nodes.
    """
    model = load_embedding_model(config.embedding_model)
    stoplist = build_stopwords(config.preprocess.stopwords_langs)

    conversations = chat_history.conversations
    if not conversations:
        conversations = [
            Conversation(id=msg.id, messages=[msg]) for msg in chat_history.messages
        ]

    merged_texts = [conv.get_merged_content() for conv in conversations]
    cleaned_texts = [preprocess_text(text, config.preprocess) for text in merged_texts]
    metrics: Optional[PipelineMetrics] = None

    embed_start = time.perf_counter()
    embeddings = generate_embeddings(cleaned_texts, model)
    embed_elapsed = time.perf_counter() - embed_start

    embed_stats: Optional[EmbeddingStats] = None
    if collect_metrics:
        doc_count = len(cleaned_texts)
        dimension = int(embeddings.shape[1]) if embeddings.size else 0
        norms = np.linalg.norm(embeddings, axis=1) if embeddings.size else np.array([])
        norm_mean = float(norms.mean()) if norms.size else None
        norm_std = float(norms.std(ddof=0)) if norms.size else None
        sim_mean = sim_std = sim_min = sim_max = None
        if embeddings.shape[0] >= 2:
            similarity_matrix = embeddings @ embeddings.T
            triu = np.triu_indices(embeddings.shape[0], k=1)
            similarities = similarity_matrix[triu]
            if similarities.size:
                sim_mean = float(similarities.mean())
                sim_std = float(similarities.std(ddof=0))
                sim_min = float(similarities.min())
                sim_max = float(similarities.max())
        embed_stats = EmbeddingStats(
            elapsed=embed_elapsed,
            documents=doc_count,
            dimension=dimension,
            norm_mean=norm_mean,
            norm_std=norm_std,
            similarity_mean=sim_mean,
            similarity_std=sim_std,
            similarity_min=sim_min,
            similarity_max=sim_max,
        )

    keybert_model = KeyBERT(model=model)

    keyword_start = time.perf_counter()
    keywords = extract_keywords(
        cleaned_texts,
        config.keyword,
        stoplist,
        keybert_model,
        doc_embeddings=embeddings,
    )
    keyword_elapsed = time.perf_counter() - keyword_start

    keyword_stats: Optional[KeywordStats] = None
    if collect_metrics:
        doc_keyword_counts = [len(items) for items in keywords]
        total_docs = len(doc_keyword_counts)
        documents_with_keywords = sum(1 for count in doc_keyword_counts if count > 0)
        avg_keywords = (
            float(np.mean(doc_keyword_counts)) if doc_keyword_counts else None
        )
        median_keywords = (
            float(np.median(doc_keyword_counts)) if doc_keyword_counts else None
        )
        all_keywords = [kw for items in keywords for kw in items]
        unique_terms = {kw.term for kw in all_keywords}
        total_keywords = len(all_keywords)
        unique_ratio = len(unique_terms) / total_keywords if total_keywords else None
        scores = [kw.score for kw in all_keywords]
        avg_score = float(np.mean(scores)) if scores else None
        max_score = float(np.max(scores)) if scores else None
        keyword_stats = KeywordStats(
            elapsed=keyword_elapsed,
            documents_with_keywords=documents_with_keywords,
            total_documents=total_docs,
            avg_keywords=avg_keywords,
            median_keywords=median_keywords,
            unique_keyword_ratio=unique_ratio,
            avg_keyword_score=avg_score,
            max_keyword_score=max_score,
        )

    labels = cluster_embeddings(embeddings, config.cluster)
    edges = build_similarity_edges(embeddings, config.graph)
    metadata = compute_metadata(config, labels, edges, stoplist, cleaned_texts)

    nodes: List[Node] = []
    for idx, (conversation, label, keyword_list, merged_text) in enumerate(
        zip(conversations, labels, keywords, merged_texts)
    ):
        nodes.append(
            Node(
                id=idx,
                orig_id=conversation.id,
                role="conversation",
                text=merged_text,
                timestamp=conversation.get_earliest_timestamp(),
                cluster=int(label),
                keywords=keyword_list,
                num_messages=len(conversation.messages),
                message_ids=[msg.id for msg in conversation.messages],
            )
        )

    if embed_stats and keyword_stats:
        metrics = PipelineMetrics(embedding=embed_stats, keyword=keyword_stats)

    return OutputGraph(nodes=nodes, edges=edges, metadata=metadata), metrics


def run_pipeline_until_keywords(
    chat_history: ChatHistory, config: Config
) -> Tuple[List[ConversationSummary], Dict[str, Any]]:
    """
    Execute pipeline up to keyword extraction (before clustering).
    Returns minimal data needed for LLM-based clustering.
    """
    model = load_embedding_model(config.embedding_model)
    stoplist = build_stopwords(config.preprocess.stopwords_langs)

    conversations = chat_history.conversations
    if not conversations:
        conversations = [
            Conversation(id=msg.id, messages=[msg]) for msg in chat_history.messages
        ]

    merged_texts = [conv.get_merged_content() for conv in conversations]
    cleaned_texts = [preprocess_text(text, config.preprocess) for text in merged_texts]

    # 1. 임베딩 생성 (청킹 + 평균 풀링) ✅
    embeddings = generate_embeddings(cleaned_texts, model)

    # 2. KeyBERT를 사용하되, 사전 계산된 문서 임베딩을 전달하여 재청킹/트렁케이션 방지
    keybert_model = KeyBERT(model=model)
    keywords = extract_keywords(
        cleaned_texts,
        config.keyword,
        stoplist,
        keybert_model,
        doc_embeddings=embeddings,
    )

    # Build conversation summaries
    summaries: List[ConversationSummary] = []
    for idx, (conversation, keyword_list) in enumerate(zip(conversations, keywords)):
        summaries.append(
            ConversationSummary(
                id=idx,
                orig_id=conversation.id,
                keywords=keyword_list,
                timestamp=conversation.get_earliest_timestamp(),
                num_messages=len(conversation.messages),
            )
        )

    # Prepare metadata
    metadata = {
        "total_conversations": len(summaries),
        "embedding_model": config.embedding_model,
        "keyword_params": {
            "top_n": config.keyword.top_n,
            "max_ngram": config.keyword.max_ngram,
            "dedup_thresh": config.keyword.dedup_thresh,
        },
        "preprocess_params": {
            "lower": config.preprocess.lower,
            "strip_urls": config.preprocess.strip_urls,
            "strip_code": config.preprocess.strip_code,
            "strip_punct": config.preprocess.strip_punct,
            "stopwords_langs": config.preprocess.stopwords_langs,
        },
    }

    return summaries, metadata


def _format_float(value: Optional[float], *, precision: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float) and math.isnan(value):
        return "N/A"
    return f"{value:.{precision}f}"


def _print_pipeline_metrics(
    input_path: Path, config: Config, metrics: PipelineMetrics
) -> None:
    embedding = metrics.embedding
    keyword = metrics.keyword

    print("=== Embedding Step ===")
    print(f"Input file: {input_path.resolve()}")
    print(f"Embedding model: {config.embedding_model}")
    print(f"Documents: {embedding.documents}")
    print(f"Elapsed (s): {embedding.elapsed:.4f}")
    print(f"Dimension: {embedding.dimension}")
    norm_mean = _format_float(embedding.norm_mean)
    norm_std = _format_float(embedding.norm_std)
    print(f"Norm mean/std: {norm_mean} / {norm_std}")
    sim_mean = _format_float(embedding.similarity_mean)
    sim_std = _format_float(embedding.similarity_std)
    sim_min = _format_float(embedding.similarity_min)
    sim_max = _format_float(embedding.similarity_max)
    print(
        f"Similarity mean/std/min/max: {sim_mean} / {sim_std} / {sim_min} / {sim_max}"
    )
    print()
    print("=== Keyword Step ===")
    print(f"Keyword model: {config.embedding_model}")
    print(f"Elapsed (s): {keyword.elapsed:.4f}")
    print(
        f"Documents with keywords: {keyword.documents_with_keywords} / {keyword.total_documents}"
    )
    avg_keywords = _format_float(keyword.avg_keywords, precision=2)
    median_keywords = _format_float(keyword.median_keywords, precision=2)
    print(f"Avg/Median keywords per doc: {avg_keywords} / {median_keywords}")
    unique_ratio = _format_float(keyword.unique_keyword_ratio)
    print(f"Unique keyword ratio: {unique_ratio}")
    avg_score = _format_float(keyword.avg_keyword_score)
    max_score = _format_float(keyword.max_keyword_score)
    print(f"Avg/Max keyword score: {avg_score} / {max_score}")


def build_graph(
    input_path: Path,
    output_path: Path,
    config_path: Path,
    *,
    collect_metrics: bool = False,
) -> Union[OutputGraph, Tuple[OutputGraph, Optional[PipelineMetrics]]]:
    config = load_config(config_path)
    messages = load_messages(input_path)
    try:
        messages.messages  # access to trigger validation
    except ValidationError as exc:  # pragma: no cover - defensive
        raise SystemExit(f"Invalid input data: {exc}") from exc
    output_graph, metrics = run_pipeline(
        messages, config, collect_metrics=collect_metrics
    )
    if hasattr(output_graph, "model_dump"):
        output_data = output_graph.model_dump(by_alias=True)  # type: ignore[call-arg]
    else:
        output_data = output_graph.dict(by_alias=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if collect_metrics and metrics:
        _print_pipeline_metrics(input_path, config, metrics)
        return output_graph, metrics
    return output_graph


def build_intermediate_results(
    input_path: Path,
    output_path: Path,
    config_path: Path,
) -> IntermediateResult:
    """
    Build lightweight intermediate results for LLM-based clustering.
    Only extracts keywords, no embeddings or clustering.
    """
    config = load_config(config_path)
    messages = load_messages(input_path)
    try:
        messages.messages  # access to trigger validation
    except ValidationError as exc:  # pragma: no cover - defensive
        raise SystemExit(f"Invalid input data: {exc}") from exc

    summaries, metadata = run_pipeline_until_keywords(messages, config)

    # Create IntermediateResult
    intermediate_result = IntermediateResult(conversations=summaries, metadata=metadata)

    # Convert to dictionary for JSON serialization
    output_data = {
        "conversations": [
            {
                "id": conv.id,
                "orig_id": conv.orig_id,
                "keywords": [
                    {"term": kw.term, "score": kw.score} for kw in conv.keywords
                ],
                "timestamp": conv.timestamp,
                "num_messages": conv.num_messages,
            }
            for conv in intermediate_result.conversations
        ],
        "metadata": intermediate_result.metadata,
    }

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return intermediate_result


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build a similarity graph from a ChatGPT chat history."
    )
    parser.add_argument(
        "--in", dest="input_path", required=True, help="Path to chat history JSON file."
    )
    parser.add_argument(
        "--out", dest="output_path", required=True, help="Path to write graph JSON."
    )
    parser.add_argument(
        "--cfg", dest="config_path", required=True, help="Path to YAML config."
    )
    parser.add_argument(
        "--mode",
        choices=["full", "keywords_only"],
        default="full",
        help=(
            "Processing mode: 'full' builds complete graph with clustering, "
            "'keywords_only' creates lightweight intermediate output for LLM-based clustering."
        ),
    )
    args = parser.parse_args(argv)

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    config_path = Path(args.config_path)

    if args.mode == "keywords_only":
        # Build intermediate results (keywords only)
        intermediate_result = build_intermediate_results(
            input_path, output_path, config_path
        )
        print(
            f"Intermediate results built: {len(intermediate_result.conversations)} conversations extracted. "
            f"Output saved to {output_path.resolve()}."
        )
    else:
        # Build full graph with clustering
        result = build_graph(input_path, output_path, config_path, collect_metrics=True)
        output_graph = result[0] if isinstance(result, tuple) else result
        counts = output_graph.metadata.counts
        clusters = ", ".join(
            f"{cluster_id} (size={summary.size})"
            for cluster_id, summary in output_graph.metadata.clusters.items()
        )
        cluster_info = clusters if clusters else "no clusters"
        print(
            f"Graph built: {counts.nodes} nodes, {counts.edges} edges, "
            f"{counts.clusters} clusters, {counts.outliers} outliers. Clusters: {cluster_info}."
        )


if __name__ == "__main__":  # pragma: no cover
    main()
