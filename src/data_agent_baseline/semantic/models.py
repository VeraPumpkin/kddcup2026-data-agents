from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class SourceSpan:
    path: str
    start_line: int | None = None
    end_line: int | None = None
    excerpt: str | None = None


@dataclass(frozen=True, slots=True)
class ValueMapping:
    source_text: str
    normalized_value: str
    target_column_candidates: list[str] = field(default_factory=list)
    target_table_candidates: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ConceptEntry:
    canonical_name: str
    aliases: list[str] = field(default_factory=list)
    definition: str | None = None
    source_span: SourceSpan | None = None
    constraints: list[str] = field(default_factory=list)
    value_mappings: list[ValueMapping] = field(default_factory=list)
    time_scope: str | None = None
    unit: str | None = None
    confidence: float = 0.0


@dataclass(frozen=True, slots=True)
class SchemaFieldProfile:
    source_file: str
    table_name: str
    field_name: str
    dtype: str
    sample_values: list[str] = field(default_factory=list)
    min_value: float | int | str | None = None
    max_value: float | int | str | None = None
    unique_ratio: float = 0.0
    null_ratio: float = 0.0
    semantic_tags: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class JoinCandidate:
    left_table: str
    right_table: str
    left_field: str
    right_field: str
    score: float
    reason: str


@dataclass(frozen=True, slots=True)
class SemanticLink:
    concept_name: str
    matched_table: str
    matched_field: str
    matched_values: list[str] = field(default_factory=list)
    link_type: str = "candidate"
    slot_name: str | None = None
    slot_type: str | None = None
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class SemanticLayer:
    task_id: str
    question: str
    knowledge_present: bool
    concepts: list[ConceptEntry] = field(default_factory=list)
    schema_profiles: list[SchemaFieldProfile] = field(default_factory=list)
    join_candidates: list[JoinCandidate] = field(default_factory=list)
    semantic_links: list[SemanticLink] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    build_trace: list[dict[str, object]] = field(default_factory=list)
    retrieved_knowledge_chunks: list[dict[str, object]] = field(default_factory=list)
