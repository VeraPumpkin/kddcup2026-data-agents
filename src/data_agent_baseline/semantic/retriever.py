from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from data_agent_baseline.semantic.chunker import ChunkingConfig, KnowledgeChunk, chunk_markdown
from data_agent_baseline.semantic.serializer import read_json, write_json

DEFAULT_BGE_MODEL_NAME = "BAAI/bge-small-en-v1.5"


@dataclass(frozen=True, slots=True)
class EmbeddingRetrieverConfig:
    model_name_or_path: str = DEFAULT_BGE_MODEL_NAME
    device: str = "cpu"
    query_instruction: str = "Represent this sentence for searching relevant passages:"
    chunk_target_chars: int = 900
    chunk_overlap_chars: int = 120


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    heading_path: list[str]
    start_line: int
    end_line: int


@dataclass(frozen=True, slots=True)
class KnowledgeIndex:
    knowledge_path: str
    model_name_or_path: str
    query_instruction: str
    chunks: list[KnowledgeChunk]
    normalized_embeddings: np.ndarray
    cache_dir: str
    debug: dict[str, Any]


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0.0:
        return vector
    return vector / norm


def _load_sentence_transformer(model_name_or_path: str, *, device: str):
    # Formal submission must switch this to a local model path or local_files_only,
    # because the evaluation environment blocks external network access.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name_or_path, device=device)


def _encode_chunks(model, chunks: list[KnowledgeChunk]) -> np.ndarray:
    if not chunks:
        return np.zeros((0, 1), dtype=np.float32)
    embeddings = model.encode(
        [chunk.text for chunk in chunks],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return np.asarray(embeddings, dtype=np.float32)


def _encode_query(model, query: str, *, query_instruction: str) -> np.ndarray:
    prefixed_query = f"{query_instruction} {query}".strip()
    embedding = model.encode(
        prefixed_query,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return _normalize_vector(np.asarray(embedding, dtype=np.float32))


def _index_cache_paths(cache_dir: Path) -> tuple[Path, Path]:
    return cache_dir / "knowledge_index.json", cache_dir / "knowledge_embeddings.npy"


def _build_manifest(
    knowledge_md_path: Path,
    config: EmbeddingRetrieverConfig,
    knowledge_text: str,
) -> dict[str, object]:
    return {
        "knowledge_path": str(knowledge_md_path),
        "knowledge_sha256": _hash_text(knowledge_text),
        "model_name_or_path": config.model_name_or_path,
        "device": config.device,
        "query_instruction": config.query_instruction,
        "chunk_target_chars": config.chunk_target_chars,
        "chunk_overlap_chars": config.chunk_overlap_chars,
    }


def build_knowledge_index(
    knowledge_md_path: Path,
    cache_dir: Path,
    model_name_or_path: str,
    *,
    device: str = "cpu",
    chunk_target_chars: int = 900,
    chunk_overlap_chars: int = 120,
    query_instruction: str = "Represent this sentence for searching relevant passages:",
) -> KnowledgeIndex:
    config = EmbeddingRetrieverConfig(
        model_name_or_path=model_name_or_path,
        device=device,
        query_instruction=query_instruction,
        chunk_target_chars=chunk_target_chars,
        chunk_overlap_chars=chunk_overlap_chars,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata_path, embeddings_path = _index_cache_paths(cache_dir)

    knowledge_text = knowledge_md_path.read_text(encoding="utf-8", errors="replace")
    manifest = _build_manifest(knowledge_md_path, config, knowledge_text)
    if metadata_path.exists() and embeddings_path.exists():
        metadata = read_json(metadata_path)
        if metadata.get("manifest") == manifest:
            embeddings = np.load(embeddings_path)
            chunks = [
                KnowledgeChunk(
                    chunk_id=str(item["chunk_id"]),
                    text=str(item["text"]),
                    heading_path=[str(value) for value in item.get("heading_path", [])],
                    start_line=int(item["start_line"]),
                    end_line=int(item["end_line"]),
                    token_count_estimate=int(item["token_count_estimate"]),
                )
                for item in metadata.get("chunks", [])
            ]
            return KnowledgeIndex(
                knowledge_path=str(knowledge_md_path),
                model_name_or_path=model_name_or_path,
                query_instruction=query_instruction,
                chunks=chunks,
                normalized_embeddings=np.asarray(embeddings, dtype=np.float32),
                cache_dir=str(cache_dir),
                debug={
                    "cache_status": "hit",
                    "manifest": manifest,
                    "chunk_count": len(chunks),
                },
            )

    chunks = chunk_markdown(
        knowledge_md_path,
        config=ChunkingConfig(
            target_chars=chunk_target_chars,
            overlap_chars=chunk_overlap_chars,
        ),
    )
    model = _load_sentence_transformer(model_name_or_path, device=device)
    embeddings = _encode_chunks(model, chunks)
    embeddings = _normalize_rows(embeddings) if len(chunks) else embeddings

    metadata = {
        "manifest": manifest,
        "chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "heading_path": chunk.heading_path,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "token_count_estimate": chunk.token_count_estimate,
            }
            for chunk in chunks
        ],
    }
    write_json(metadata_path, metadata)
    np.save(embeddings_path, embeddings)
    return KnowledgeIndex(
        knowledge_path=str(knowledge_md_path),
        model_name_or_path=model_name_or_path,
        query_instruction=query_instruction,
        chunks=chunks,
        normalized_embeddings=embeddings,
        cache_dir=str(cache_dir),
        debug={
            "cache_status": "miss",
            "manifest": manifest,
            "chunk_count": len(chunks),
        },
    )


def retrieve_knowledge(
    query: str,
    knowledge_index: KnowledgeIndex,
    *,
    top_k: int = 5,
) -> list[RetrievedChunk]:
    if not knowledge_index.chunks:
        return []
    model = _load_sentence_transformer(
        knowledge_index.model_name_or_path,
        device="cpu",
    )
    query_embedding = _encode_query(
        model,
        query,
        query_instruction=knowledge_index.query_instruction,
    )
    scores = np.dot(knowledge_index.normalized_embeddings, query_embedding)
    top_indices = np.argsort(-scores)[: max(1, top_k)]
    retrieved: list[RetrievedChunk] = []
    for index in top_indices:
        chunk = knowledge_index.chunks[int(index)]
        retrieved.append(
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                score=float(scores[int(index)]),
                heading_path=list(chunk.heading_path),
                start_line=chunk.start_line,
                end_line=chunk.end_line,
            )
        )
    return retrieved
