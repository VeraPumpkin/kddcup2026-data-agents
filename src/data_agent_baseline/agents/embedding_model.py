from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_LOCAL_EMBEDDING_MODEL = "embedding_model"
DEFAULT_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages:"
DEFAULT_BATCH_SIZE = 256
DEFAULT_MAX_LENGTH = 512
CPU_DEVICE = "cpu"

_MODEL_CACHE: dict[str, Any] = {}
_MODEL_CACHE_LOCK = threading.Lock()


class SentenceTransformerEmbeddingModel:
    def __init__(
        self,
        *,
        model_path: Path,
        query_instruction_for_retrieval: str,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_length: int = DEFAULT_MAX_LENGTH,
    ) -> None:
        self.model_path = model_path
        self.query_instruction_for_retrieval = query_instruction_for_retrieval
        self.batch_size = batch_size
        self.max_length = max_length
        self.model = _load_sentence_transformer(model_path, max_length=self.max_length)

    def encode_queries(self, queries: list[str] | str) -> np.ndarray:
        return self._encode(queries, add_query_instruction=True)

    def encode_corpus(self, corpus: list[str] | str) -> np.ndarray:
        return self._encode(corpus, add_query_instruction=False)

    def encode(self, sentences: list[str] | str) -> np.ndarray:
        return self._encode(sentences, add_query_instruction=False)

    def _encode(self, texts: list[str] | str, *, add_query_instruction: bool) -> np.ndarray:
        input_was_string = isinstance(texts, str)
        normalized_texts = [texts] if input_was_string else [str(text) for text in texts]
        if not normalized_texts:
            return np.empty((0, 0), dtype=np.float32)
        if add_query_instruction and self.query_instruction_for_retrieval:
            normalized_texts = [
                f"{self.query_instruction_for_retrieval}{text}" for text in normalized_texts
            ]

        encoded = self.model.encode(
            normalized_texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        encoded = np.asarray(encoded, dtype=np.float32)
        return encoded[0] if input_was_string else encoded


def get_embedding_model(
    *,
    query_instruction_for_retrieval: str = DEFAULT_QUERY_INSTRUCTION,
) -> Any:
    cache_key = query_instruction_for_retrieval
    with _MODEL_CACHE_LOCK:
        cached = _MODEL_CACHE.get(cache_key)
        if cached is not None:
            return cached
        model = _instantiate_embedding_model(
            query_instruction_for_retrieval=query_instruction_for_retrieval,
        )
        _MODEL_CACHE[cache_key] = model
        return model


def embedding_model_cache_size() -> int:
    return len(_MODEL_CACHE)


def _instantiate_embedding_model(
    *,
    query_instruction_for_retrieval: str,
) -> Any:
    model_path = Path(DEFAULT_LOCAL_EMBEDDING_MODEL).expanduser()
    if not model_path.is_dir():
        raise FileNotFoundError(
            f"Embedding model directory does not exist: {DEFAULT_LOCAL_EMBEDDING_MODEL}."
        )

    return SentenceTransformerEmbeddingModel(
        model_path=model_path,
        query_instruction_for_retrieval=query_instruction_for_retrieval,
    )


def _load_sentence_transformer(model_path: Path, *, max_length: int) -> SentenceTransformer:
    model = SentenceTransformer(
        str(model_path),
        device=CPU_DEVICE,
        local_files_only=True,
        trust_remote_code=False,
    )
    model.max_seq_length = max_length
    return model
