from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from data_agent_baseline.agents.understanding.tools.candidate_store import (
    CandidateStore,
    FieldProfile,
    normalize_identifier,
)
from data_agent_baseline.agents.embedding_model import (
    DEFAULT_QUERY_INSTRUCTION,
    get_embedding_model,
)
from data_agent_baseline.benchmark.schema import PublicTask


LOGGER = logging.getLogger(__name__)

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "how",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
}


@dataclass(frozen=True, slots=True)
class HybridRetrievalConfig:
    bm25_top_k: int = 30
    embedding_top_k: int = 30
    final_top_k: int = 12
    max_context_chars: int = 8000
    max_chunk_chars: int = 900
    min_score_threshold: float = 0.08
    sample_value_limit: int = 10
    rrf_k: int = 60
    rrf_bm25_weight: float = 0.4
    rrf_embedding_weight: float = 0.6
    query_instruction_for_retrieval: str = DEFAULT_QUERY_INSTRUCTION


@dataclass(frozen=True, slots=True)
class RetrievalDocument:
    id: str
    source: str
    source_type: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RetrievedContextItem:
    id: str
    source: str
    source_type: str
    text: str
    metadata: dict[str, Any]
    bm25_score: float
    embedding_score: float
    rrf_score: float
    ranks: dict[str, int | None]
    final_score: float
    matched_terms: list[str]


@dataclass(frozen=True, slots=True)
class HybridRetrievalResult:
    items: list[RetrievedContextItem]


class RetrievalCorpusBuilder:
    """Build a task-local retrieval corpus from table profiles and samples."""

    def __init__(self, config: HybridRetrievalConfig | None = None) -> None:
        self.config = config or HybridRetrievalConfig()

    def build(
        self,
        *,
        candidate_store: CandidateStore | None,
    ) -> list[RetrievalDocument]:
        documents: list[RetrievalDocument] = []
        documents.extend(self._candidate_store_documents(candidate_store))
        return self._dedupe_documents(documents)

    def _candidate_store_documents(
        self,
        candidate_store: CandidateStore | None,
    ) -> list[RetrievalDocument]:
        if candidate_store is None:
            return []
        documents: list[RetrievalDocument] = []
        for table_schema in candidate_store.tables:
            fields = list(table_schema.columns)
            documents.append(
                RetrievalDocument(
                    id=f"table_profile:{table_schema.path}:{table_schema.table}",
                    source=table_schema.path,
                    source_type="table_profile",
                    text=_truncate(
                        f"Table profile {table_schema.table}. "
                        f"Kind: {table_schema.kind}. File: {table_schema.path}. "
                        f"Record count: {table_schema.row_count}. "
                        f"Columns: {', '.join(fields)}.",
                        self.config.max_chunk_chars,
                    ),
                    metadata={
                        "file": table_schema.path,
                        "table": table_schema.table,
                        "columns": fields,
                        "record_count": table_schema.row_count,
                        "kind": table_schema.kind,
                        "original_path": table_schema.path,
                    },
                )
            )
        for field_profile in candidate_store.fields:
            documents.append(self._column_profile_document(field_profile))
            sample_doc = self._sample_value_document(field_profile)
            if sample_doc is not None:
                documents.append(sample_doc)
        return documents

    def _column_profile_document(self, profile: FieldProfile) -> RetrievalDocument:
        values = profile.sample_values[:8]
        text = (
            f"Column profile {profile.field}. "
            f"Table: {profile.table or ''}. Column: {profile.name}. File: {profile.path}. "
            f"Source kind: {profile.kind}. Data type: {profile.data_type}. "
            f"Non-null count: {profile.non_null_count}. "
            f"Null count: {profile.null_count}. "
            f"Min: {profile.min_value}. Max: {profile.max_value}. "
            f"Sample values: {', '.join(values)}."
        )
        return RetrievalDocument(
            id=f"column_profile:{profile.field}:{profile.path}",
            source=profile.path,
            source_type="column_profile",
            text=_truncate(text, self.config.max_chunk_chars),
            metadata={
                "file": profile.path,
                "table": profile.table,
                "column": profile.name,
                "field": profile.field,
                "data_type": profile.data_type,
                "kind": profile.kind,
                "is_id_like": profile.is_id_like,
                "sample_values": list(profile.sample_values[: self.config.sample_value_limit]),
                "original_path": profile.path,
            },
        )

    def _sample_value_document(self, profile: FieldProfile) -> RetrievalDocument | None:
        values = profile.sample_values[: self.config.sample_value_limit]
        if not values:
            return None
        text = (
            f"Sample values for {profile.field}. "
            f"Table: {profile.table or ''}. Column: {profile.name}. "
            f"Values: {', '.join(values)}."
        )
        return RetrievalDocument(
            id=f"sample_value:{profile.field}:{profile.path}",
            source=profile.path,
            source_type="sample_value",
            text=_truncate(text, self.config.max_chunk_chars),
            metadata={
                "file": profile.path,
                "table": profile.table,
                "column": profile.name,
                "field": profile.field,
                "values": list(values),
                "data_type": profile.data_type,
                "original_path": profile.path,
            },
        )

    def _dedupe_documents(self, documents: list[RetrievalDocument]) -> list[RetrievalDocument]:
        seen_ids: set[str] = set()
        deduped: list[RetrievalDocument] = []
        for document in documents:
            if document.id in seen_ids:
                continue
            seen_ids.add(document.id)
            deduped.append(document)
        return deduped


class BM25Retriever:
    """Thin wrapper around rank_bm25.BM25Okapi."""

    def __init__(self, *, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.documents: list[RetrievalDocument] = []
        self.corpus_tokens: list[list[str]] = []
        self._ranker: Any | None = None

    def fit(self, documents: list[RetrievalDocument]) -> None:
        from rank_bm25 import BM25Okapi

        self.documents = list(documents)
        self.corpus_tokens = [_tokens(document.text) for document in documents]
        self._ranker = BM25Okapi(self.corpus_tokens, k1=self.k1, b=self.b)

    def search(
        self,
        query_terms: list[str],
        *,
        top_k: int,
    ) -> list[tuple[RetrievalDocument, float, list[str]]]:
        normalized_terms = _dedupe_preserve_order(
            token
            for term in query_terms
            for token in _tokens(term)
            if token not in _STOPWORDS
        )
        if not normalized_terms:
            return []
        scores = self._ranker.get_scores(normalized_terms)
        scored: list[tuple[RetrievalDocument, float, list[str]]] = []
        for index, (document, score) in enumerate(zip(self.documents, scores, strict=False)):
            numeric_score = float(score)
            if numeric_score <= 0:
                continue
            token_set = set(self.corpus_tokens[index])
            matched_terms = [term for term in normalized_terms if term in token_set]
            scored.append((document, numeric_score, matched_terms))
        scored.sort(key=lambda item: (-item[1], item[0].id))
        return scored[:top_k]


class EmbeddingRetriever:
    """Lazy embedding retriever with graceful fallback."""

    def __init__(self, config: HybridRetrievalConfig) -> None:
        self.config = config
        self._model: Any | None = None
        self._available: bool | None = None
        self._embedding_cache: dict[str, np.ndarray] = {}

    @property
    def available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            self._model = self._load_embedding_model()
            self._available = True
        except Exception as exc:
            LOGGER.warning("Embedding retriever unavailable; using lexical fallback: %s", exc)
            self._model = None
            self._available = False
        return self._available

    def search(
        self,
        query: str,
        documents: list[RetrievalDocument],
        *,
        top_k: int,
        cache_key: str,
    ) -> list[tuple[RetrievalDocument, float]]:
        if not documents or not self.available or self._model is None:
            return []
        try:
            corpus_embeddings = self._embedding_cache.get(cache_key)
            if corpus_embeddings is None:
                corpus_embeddings = self._normalize_matrix(
                    self._encode_corpus([document.text for document in documents])
                )
                self._embedding_cache[cache_key] = corpus_embeddings
            query_embedding = self._normalize_matrix(self._encode_queries([query]))[0]
            scores = corpus_embeddings @ query_embedding
            ranked_indexes = np.argsort(-scores)[:top_k]
            return [
                (documents[index], round(float(scores[index]), 4))
                for index in ranked_indexes
                if float(scores[index]) > 0
            ]
        except Exception as exc:
            LOGGER.warning("Embedding search failed; continuing with lexical scores: %s", exc)
            self._available = False
            return []

    def _load_embedding_model(self) -> Any:
        return get_embedding_model(
            query_instruction_for_retrieval=self.config.query_instruction_for_retrieval,
        )

    def _encode_queries(self, queries: list[str]) -> np.ndarray:
        if hasattr(self._model, "encode_queries"):
            return np.asarray(self._model.encode_queries(queries))
        return np.asarray(self._model.encode(queries))

    def _encode_corpus(self, corpus: list[str]) -> np.ndarray:
        if hasattr(self._model, "encode_corpus"):
            return np.asarray(self._model.encode_corpus(corpus))
        return np.asarray(self._model.encode(corpus))

    def _normalize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms


class HybridRetriever:
    """Task-local BM25 + embedding + deterministic rerank retrieval."""

    def __init__(
        self,
        *,
        config: HybridRetrievalConfig | None = None,
        corpus_builder: RetrievalCorpusBuilder | None = None,
        bm25_retriever: BM25Retriever | None = None,
        embedding_retriever: EmbeddingRetriever | None = None,
    ) -> None:
        self.config = config or HybridRetrievalConfig()
        self.corpus_builder = corpus_builder or RetrievalCorpusBuilder(self.config)
        self.bm25_retriever = bm25_retriever or BM25Retriever()
        self.embedding_retriever = embedding_retriever or EmbeddingRetriever(self.config)
        self._corpus_cache: dict[str, list[RetrievalDocument]] = {}

    def retrieve(
        self,
        *,
        task: PublicTask | None,
        question: str,
        candidate_store: CandidateStore | None,
        source_type_filter: set[str] | None = None,
        final_top_k: int | None = None,
    ) -> HybridRetrievalResult:
        corpus_key = self._corpus_key(task, candidate_store)
        base_documents = self._corpus_cache.get(corpus_key)
        if base_documents is None:
            base_documents = self.corpus_builder.build(
                candidate_store=candidate_store,
            )
            self._corpus_cache[corpus_key] = base_documents
        documents = (
            [
                document for document in base_documents if document.source_type in source_type_filter
            ]
            if source_type_filter is not None
            else list(base_documents)
        )

        query_terms = self._query_terms(question)

        self.bm25_retriever.fit(documents)
        bm25_hits = self.bm25_retriever.search(
            query_terms,
            top_k=self.config.bm25_top_k,
        )

        embedding_hits = self.embedding_retriever.search(
            question,
            documents,
            top_k=self.config.embedding_top_k,
            cache_key=self._documents_cache_key(corpus_key, documents, source_type_filter),
        )

        items = self._merge_and_rerank(
            bm25_hits=bm25_hits,
            embedding_hits=embedding_hits,
            final_top_k=final_top_k or self.config.final_top_k,
        )
        return HybridRetrievalResult(
            items=items,
        )

    def _merge_and_rerank(
        self,
        *,
        bm25_hits: list[tuple[RetrievalDocument, float, list[str]]],
        embedding_hits: list[tuple[RetrievalDocument, float]],
        final_top_k: int,
    ) -> list[RetrievedContextItem]:
        docs_by_id: dict[str, RetrievalDocument] = {}
        bm25_by_id: dict[str, float] = {}
        bm25_terms_by_id: dict[str, list[str]] = {}
        embedding_by_id: dict[str, float] = {}
        bm25_rank_by_id: dict[str, int] = {}
        embedding_rank_by_id: dict[str, int] = {}
        for rank, (document, score, matched_terms) in enumerate(bm25_hits, start=1):
            docs_by_id[document.id] = document
            bm25_by_id[document.id] = score
            bm25_terms_by_id[document.id] = matched_terms
            bm25_rank_by_id[document.id] = rank
        for rank, (document, score) in enumerate(embedding_hits, start=1):
            docs_by_id[document.id] = document
            embedding_by_id[document.id] = score
            embedding_rank_by_id[document.id] = rank

        rrf_denominator = self._rrf_max_score(
            bm25_active=bool(bm25_rank_by_id),
            embedding_active=bool(embedding_rank_by_id),
        )

        preliminary: list[RetrievedContextItem] = []
        duplicate_counts = Counter(_normalize_for_duplicate(doc.text) for doc in docs_by_id.values())
        for document_id, document in docs_by_id.items():
            bm25_score = bm25_by_id.get(document_id, 0.0)
            embedding_score = embedding_by_id.get(document_id, 0.0)
            matched_terms = bm25_terms_by_id.get(document_id, [])
            ranks = {
                "bm25": bm25_rank_by_id.get(document_id),
                "embedding": embedding_rank_by_id.get(document_id),
            }
            rrf_score = self._rrf_score(ranks, rrf_denominator)
            length_penalty = self._length_penalty(document)
            duplicate_penalty = 0.04 if duplicate_counts[_normalize_for_duplicate(document.text)] > 1 else 0.0
            final_score = rrf_score - length_penalty - duplicate_penalty
            if final_score < self.config.min_score_threshold:
                continue
            preliminary.append(
                RetrievedContextItem(
                    id=document.id,
                    source=document.source,
                    source_type=document.source_type,
                    text=document.text,
                    metadata=dict(document.metadata),
                    bm25_score=round(bm25_score, 4),
                    embedding_score=round(embedding_score, 4),
                    rrf_score=round(rrf_score, 4),
                    ranks=dict(ranks),
                    final_score=round(final_score, 4),
                    matched_terms=matched_terms,
                )
            )

        preliminary.sort(key=lambda item: (-item.final_score, item.source_type, item.id))
        return self._select_diverse(preliminary, final_top_k)

    def _rrf_max_score(
        self,
        *,
        bm25_active: bool,
        embedding_active: bool,
    ) -> float:
        active_weights = []
        if bm25_active:
            active_weights.append(self.config.rrf_bm25_weight)
        if embedding_active:
            active_weights.append(self.config.rrf_embedding_weight)
        return sum(
            self._rrf_contribution(rank=1, weight=weight)
            for weight in active_weights
        ) or 1.0

    def _rrf_score(self, ranks: dict[str, int | None], denominator: float) -> float:
        raw_score = (
            self._rrf_contribution(
                rank=ranks.get("bm25"),
                weight=self.config.rrf_bm25_weight,
            )
            + self._rrf_contribution(
                rank=ranks.get("embedding"),
                weight=self.config.rrf_embedding_weight,
            )
        )
        return raw_score / denominator if denominator else 0.0

    def _rrf_contribution(self, *, rank: int | None, weight: float) -> float:
        if rank is None or rank <= 0 or weight <= 0:
            return 0.0
        return weight / (self.config.rrf_k + rank)

    def _query_terms(self, query: str) -> list[str]:
        return _dedupe_preserve_order(
            token
            for token in _tokens(query)
            if len(token) > 1 and token not in _STOPWORDS
        )

    def _length_penalty(self, document: RetrievalDocument) -> float:
        overage = max(0, len(document.text) - self.config.max_chunk_chars)
        if overage == 0:
            return 0.0
        return min(0.12, overage / max(1, self.config.max_chunk_chars) * 0.1)

    def _select_diverse(
        self,
        items: list[RetrievedContextItem],
        final_top_k: int,
    ) -> list[RetrievedContextItem]:
        selected: list[RetrievedContextItem] = []
        seen_text: set[str] = set()
        for item in items:
            normalized_text = _normalize_for_duplicate(item.text)
            if normalized_text in seen_text:
                continue
            selected.append(item)
            seen_text.add(normalized_text)
            if len(selected) >= final_top_k:
                break
        return selected

    def _corpus_key(
        self,
        task: PublicTask | None,
        candidate_store: CandidateStore | None,
    ) -> str:
        payload = {
            "task_id": task.task_id if task else None,
            "field_count": len(candidate_store.fields) if candidate_store else 0,
        }
        return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()

    def _documents_cache_key(
        self,
        corpus_key: str,
        documents: list[RetrievalDocument],
        source_type_filter: set[str] | None,
    ) -> str:
        filter_key = ",".join(sorted(source_type_filter)) if source_type_filter else "all"
        ids_hash = hashlib.sha1("|".join(document.id for document in documents).encode()).hexdigest()
        return f"{corpus_key}:{filter_key}:{ids_hash}"


def _tokens(text: Any) -> list[str]:
    return [normalize_identifier(match.group(0)) for match in _TOKEN_PATTERN.finditer(str(text)) if match.group(0)]


def _dedupe_preserve_order(values: Any) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _truncate(text: Any, max_chars: int) -> str:
    value = str(text)
    if len(value) <= max_chars:
        return value
    return value[: max(0, max_chars - 20)].rstrip() + " ...[truncated]"


def _normalize_for_duplicate(text: str) -> str:
    return " ".join(_tokens(text))[:500]
