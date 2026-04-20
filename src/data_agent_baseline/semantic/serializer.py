from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from data_agent_baseline.semantic.models import (
    ConceptEntry,
    JoinCandidate,
    SchemaFieldProfile,
    SemanticLayer,
    SemanticLink,
    SourceSpan,
    ValueMapping,
)


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    return value


def semantic_layer_to_dict(layer: SemanticLayer) -> dict[str, Any]:
    return _to_jsonable(layer)


def _source_span_from_dict(payload: dict[str, Any] | None) -> SourceSpan | None:
    if payload is None:
        return None
    return SourceSpan(
        path=str(payload["path"]),
        start_line=payload.get("start_line"),
        end_line=payload.get("end_line"),
        excerpt=payload.get("excerpt"),
    )


def _value_mapping_from_dict(payload: dict[str, Any]) -> ValueMapping:
    return ValueMapping(
        source_text=str(payload["source_text"]),
        normalized_value=str(payload["normalized_value"]),
        target_column_candidates=[str(item) for item in payload.get("target_column_candidates", [])],
        target_table_candidates=[str(item) for item in payload.get("target_table_candidates", [])],
        evidence=[str(item) for item in payload.get("evidence", [])],
    )


def semantic_layer_from_dict(payload: dict[str, Any]) -> SemanticLayer:
    return SemanticLayer(
        task_id=str(payload["task_id"]),
        question=str(payload["question"]),
        knowledge_present=bool(payload["knowledge_present"]),
        concepts=[
            ConceptEntry(
                canonical_name=str(item["canonical_name"]),
                aliases=[str(alias) for alias in item.get("aliases", [])],
                definition=item.get("definition"),
                source_span=_source_span_from_dict(item.get("source_span")),
                constraints=[str(constraint) for constraint in item.get("constraints", [])],
                value_mappings=[
                    _value_mapping_from_dict(mapping) for mapping in item.get("value_mappings", [])
                ],
                time_scope=item.get("time_scope"),
                unit=item.get("unit"),
                confidence=float(item.get("confidence", 0.0)),
            )
            for item in payload.get("concepts", [])
        ],
        schema_profiles=[
            SchemaFieldProfile(
                source_file=str(item["source_file"]),
                table_name=str(item["table_name"]),
                field_name=str(item["field_name"]),
                dtype=str(item["dtype"]),
                sample_values=[str(value) for value in item.get("sample_values", [])],
                min_value=item.get("min_value"),
                max_value=item.get("max_value"),
                unique_ratio=float(item.get("unique_ratio", 0.0)),
                null_ratio=float(item.get("null_ratio", 0.0)),
                semantic_tags=[str(tag) for tag in item.get("semantic_tags", [])],
            )
            for item in payload.get("schema_profiles", [])
        ],
        join_candidates=[
            JoinCandidate(
                left_table=str(item["left_table"]),
                right_table=str(item["right_table"]),
                left_field=str(item["left_field"]),
                right_field=str(item["right_field"]),
                score=float(item["score"]),
                reason=str(item["reason"]),
            )
            for item in payload.get("join_candidates", [])
        ],
        semantic_links=[
            SemanticLink(
                concept_name=str(item["concept_name"]),
                matched_table=str(item["matched_table"]),
                matched_field=str(item["matched_field"]),
                matched_values=[str(value) for value in item.get("matched_values", [])],
                link_type=str(item.get("link_type", "candidate")),
                slot_name=item.get("slot_name"),
                slot_type=item.get("slot_type"),
                confidence=float(item.get("confidence", 0.0)),
                evidence=[str(value) for value in item.get("evidence", [])],
            )
            for item in payload.get("semantic_links", [])
        ],
        warnings=[str(item) for item in payload.get("warnings", [])],
        build_trace=list(payload.get("build_trace", [])),
        retrieved_knowledge_chunks=list(payload.get("retrieved_knowledge_chunks", [])),
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
