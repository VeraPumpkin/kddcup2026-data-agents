from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

DEFAULT_LOCAL_EMBEDDING_MODEL = "embedding_model"
DEFAULT_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages:"
DEFAULT_BATCH_SIZE = 256
DEFAULT_MAX_LENGTH = 512
CPU_DEVICE = "cpu"

_MODEL_CACHE: dict[str, Any] = {}
_MODEL_CACHE_LOCK = threading.Lock()


class TransformersEmbeddingModel:
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
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self.model = AutoModel.from_pretrained(str(model_path))
        self.model.to(CPU_DEVICE)
        self.model.float()
        self.model.eval()

    def encode_queries(self, queries: list[str] | str) -> np.ndarray:
        return self._encode(queries, add_query_instruction=True)

    def encode_corpus(self, corpus: list[str] | str) -> np.ndarray:
        return self._encode(corpus, add_query_instruction=False)

    def encode(self, sentences: list[str] | str) -> np.ndarray:
        return self._encode(sentences, add_query_instruction=False)

    def _encode(self, texts: list[str] | str, *, add_query_instruction: bool) -> np.ndarray:
        input_was_string = isinstance(texts, str)
        normalized_texts = [texts] if input_was_string else [str(text) for text in texts]
        if add_query_instruction and self.query_instruction_for_retrieval:
            normalized_texts = [
                f"{self.query_instruction_for_retrieval}{text}" for text in normalized_texts
            ]

        embeddings: list[np.ndarray] = []
        for start in range(0, len(normalized_texts), self.batch_size):
            batch_texts = normalized_texts[start : start + self.batch_size]
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {key: value.to(CPU_DEVICE) for key, value in inputs.items()}
            with torch.inference_mode():
                outputs = self.model(**inputs, return_dict=True)
                batch_embeddings = outputs.last_hidden_state[:, 0]
                batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
            embeddings.append(batch_embeddings.detach().cpu().numpy())

        encoded = (
            np.concatenate(embeddings, axis=0).astype(np.float32, copy=False)
            if embeddings
            else np.empty((0, 0), dtype=np.float32)
        )
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

    return TransformersEmbeddingModel(
        model_path=model_path,
        query_instruction_for_retrieval=query_instruction_for_retrieval,
    )
