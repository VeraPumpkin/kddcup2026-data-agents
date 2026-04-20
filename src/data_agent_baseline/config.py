from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from data_agent_baseline.semantic.retriever import DEFAULT_BGE_MODEL_NAME

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _default_dataset_root() -> Path:
    return PROJECT_ROOT / "data" / "public" / "input"


def _default_run_output_dir() -> Path:
    return PROJECT_ROOT / "artifacts" / "runs"


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    root_path: Path = field(default_factory=_default_dataset_root)


@dataclass(frozen=True, slots=True)
class AgentConfig:
    model: str = "gpt-4.1-mini"
    api_base: str = "https://api.openai.com/v1"
    api_key: str = ""
    max_steps: int = 16
    temperature: float = 0.0


@dataclass(frozen=True, slots=True)
class RunConfig:
    output_dir: Path = field(default_factory=_default_run_output_dir)
    run_id: str | None = None
    max_workers: int = 4
    task_timeout_seconds: int = 600


@dataclass(frozen=True, slots=True)
class EmbeddingConfig:
    model_name_or_path: str = DEFAULT_BGE_MODEL_NAME
    device: str = "cpu"
    chunk_target_chars: int = 900
    chunk_overlap_chars: int = 120
    retrieval_top_k: int = 5
    query_instruction: str = "Represent this sentence for searching relevant passages:"


@dataclass(frozen=True, slots=True)
class AppConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    run: RunConfig = field(default_factory=RunConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)


def _path_value(raw_value: str | None, default_value: Path) -> Path:
    if not raw_value:
        return default_value
    candidate = Path(raw_value)
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def load_app_config(config_path: Path) -> AppConfig:
    payload = yaml.safe_load(config_path.read_text()) or {}
    dataset_defaults = DatasetConfig()
    agent_defaults = AgentConfig()
    run_defaults = RunConfig()
    embedding_defaults = EmbeddingConfig()

    dataset_payload = payload.get("dataset", {})
    agent_payload = payload.get("agent", {})
    run_payload = payload.get("run", {})
    embedding_payload = payload.get("embedding", {})

    dataset_config = DatasetConfig(
        root_path=_path_value(dataset_payload.get("root_path"), dataset_defaults.root_path),
    )
    agent_config = AgentConfig(
        model=str(agent_payload.get("model", agent_defaults.model)),
        api_base=str(agent_payload.get("api_base", agent_defaults.api_base)),
        api_key=str(agent_payload.get("api_key", agent_defaults.api_key)),
        max_steps=int(agent_payload.get("max_steps", agent_defaults.max_steps)),
        temperature=float(agent_payload.get("temperature", agent_defaults.temperature)),
    )
    raw_run_id = run_payload.get("run_id")
    run_id = run_defaults.run_id
    if raw_run_id is not None:
        normalized_run_id = str(raw_run_id).strip()
        run_id = normalized_run_id or None

    run_config = RunConfig(
        output_dir=_path_value(run_payload.get("output_dir"), run_defaults.output_dir),
        run_id=run_id,
        max_workers=int(run_payload.get("max_workers", run_defaults.max_workers)),
        task_timeout_seconds=int(run_payload.get("task_timeout_seconds", run_defaults.task_timeout_seconds)),
    )
    embedding_config = EmbeddingConfig(
        model_name_or_path=str(
            embedding_payload.get("model_name_or_path", embedding_defaults.model_name_or_path)
        ),
        device=str(embedding_payload.get("device", embedding_defaults.device)),
        chunk_target_chars=int(
            embedding_payload.get("chunk_target_chars", embedding_defaults.chunk_target_chars)
        ),
        chunk_overlap_chars=int(
            embedding_payload.get("chunk_overlap_chars", embedding_defaults.chunk_overlap_chars)
        ),
        retrieval_top_k=int(
            embedding_payload.get("retrieval_top_k", embedding_defaults.retrieval_top_k)
        ),
        query_instruction=str(
            embedding_payload.get("query_instruction", embedding_defaults.query_instruction)
        ),
    )
    return AppConfig(
        dataset=dataset_config,
        agent=agent_config,
        run=run_config,
        embedding=embedding_config,
    )
