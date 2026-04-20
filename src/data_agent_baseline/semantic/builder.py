from __future__ import annotations

import hashlib
from pathlib import Path
from time import perf_counter

from data_agent_baseline.semantic.knowledge_parser import parse_knowledge_markdown
from data_agent_baseline.semantic.matcher import match_concepts_to_schema
from data_agent_baseline.semantic.models import SemanticLayer
from data_agent_baseline.semantic.retriever import (
    DEFAULT_BGE_MODEL_NAME,
    build_knowledge_index,
    retrieve_knowledge,
)
from data_agent_baseline.semantic.schema_profiler import (
    SUPPORTED_STRUCTURED_EXTENSIONS,
    profile_structured_sources,
)
from data_agent_baseline.semantic.serializer import (
    read_json,
    semantic_layer_from_dict,
    semantic_layer_to_dict,
    write_json,
)


def _hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _discover_knowledge_path(task_dir: Path) -> Path | None:
    candidates = [
        task_dir / "knowledge.md",
        task_dir / "context" / "knowledge.md",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _discover_structured_paths(context_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted(context_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in SUPPORTED_STRUCTURED_EXTENSIONS
    ]


def _input_manifest(
    task_dir: Path,
    question: str,
    embedding_config: dict[str, object] | None = None,
) -> dict[str, object]:
    context_dir = task_dir / "context"
    knowledge_path = _discover_knowledge_path(task_dir)
    source_paths = _discover_structured_paths(context_dir)
    manifest_sources: list[dict[str, object]] = []
    for path in source_paths:
        manifest_sources.append(
            {
                "relative_path": path.relative_to(task_dir).as_posix(),
                "sha256": _hash_file(path),
                "size": path.stat().st_size,
            }
        )
    knowledge_manifest = None
    if knowledge_path is not None:
        knowledge_manifest = {
            "relative_path": knowledge_path.relative_to(task_dir).as_posix(),
            "sha256": _hash_file(knowledge_path),
            "size": knowledge_path.stat().st_size,
        }
    manifest = {
        "task_dir": str(task_dir),
        "question": question,
        "knowledge": knowledge_manifest,
        "sources": manifest_sources,
        "embedding": dict(embedding_config or {}),
    }
    return manifest


def _cache_paths(task_dir: Path, output_dir: Path | None) -> tuple[Path, Path, Path]:
    cache_dir = output_dir or (task_dir / ".semantic_cache")
    return (
        cache_dir / "semantic_layer.json",
        cache_dir / "semantic_layer_debug.json",
        cache_dir / "semantic_layer_manifest.json",
    )


def _load_cached_layer(
    task_dir: Path,
    question: str,
    output_dir: Path | None,
    embedding_config: dict[str, object] | None = None,
) -> SemanticLayer | None:
    layer_path, _debug_path, manifest_path = _cache_paths(task_dir, output_dir)
    if not layer_path.exists() or not manifest_path.exists():
        return None
    manifest = _input_manifest(task_dir, question, embedding_config)
    cached_manifest = read_json(manifest_path)
    if cached_manifest != manifest:
        return None
    payload = read_json(layer_path)
    layer = semantic_layer_from_dict(payload)
    updated_trace = list(layer.build_trace)
    updated_trace.append({"stage": "cache", "status": "hit"})
    return SemanticLayer(
        task_id=layer.task_id,
        question=layer.question,
        knowledge_present=layer.knowledge_present,
        concepts=layer.concepts,
        schema_profiles=layer.schema_profiles,
        join_candidates=layer.join_candidates,
        semantic_links=layer.semantic_links,
        warnings=layer.warnings,
        build_trace=updated_trace,
        retrieved_knowledge_chunks=layer.retrieved_knowledge_chunks,
    )


def _write_layer_artifacts(
    task_dir: Path,
    question: str,
    layer: SemanticLayer,
    debug_payload: dict[str, object],
    output_dir: Path | None,
    embedding_config: dict[str, object] | None = None,
) -> None:
    layer_path, debug_path, manifest_path = _cache_paths(task_dir, output_dir)
    write_json(layer_path, semantic_layer_to_dict(layer))
    write_json(debug_path, debug_payload)
    write_json(manifest_path, _input_manifest(task_dir, question, embedding_config))


def build_semantic_layer_for_task(
    task_dir: Path,
    question: str,
    *,
    output_dir: Path | None = None,
    embedding_config: dict[str, object] | None = None,
) -> SemanticLayer:
    effective_embedding_config = dict(embedding_config or {})
    cached = _load_cached_layer(task_dir, question, output_dir, effective_embedding_config)
    if cached is not None:
        debug_payload = {
            "warnings": cached.warnings,
            "build_trace": cached.build_trace,
            "cache": {"status": "hit"},
        }
        _write_layer_artifacts(
            task_dir,
            question,
            cached,
            debug_payload,
            output_dir,
            effective_embedding_config,
        )
        return cached

    started_at = perf_counter()
    context_dir = task_dir / "context"
    warnings: list[str] = []
    build_trace: list[dict[str, object]] = [{"stage": "cache", "status": "miss"}]

    knowledge_path = _discover_knowledge_path(task_dir)
    knowledge_present = knowledge_path is not None
    concepts = []
    knowledge_debug: dict[str, object] = {"knowledge_path": None, "concept_count": 0}
    retrieved_chunk_payloads: list[dict[str, object]] = []
    retrieved_chunks = []
    if knowledge_path is not None:
        knowledge_index = build_knowledge_index(
            knowledge_path,
            (output_dir or task_dir / ".semantic_cache") / "knowledge_index",
            str(effective_embedding_config.get("model_name_or_path", DEFAULT_BGE_MODEL_NAME)),
            device=str(effective_embedding_config.get("device", "cpu")),
            chunk_target_chars=int(effective_embedding_config.get("chunk_target_chars", 900)),
            chunk_overlap_chars=int(effective_embedding_config.get("chunk_overlap_chars", 120)),
            query_instruction=str(
                effective_embedding_config.get(
                    "query_instruction", "Represent this sentence for searching relevant passages:"
                )
            ),
        )
        retrieved_chunks = retrieve_knowledge(
            question,
            knowledge_index,
            top_k=int(effective_embedding_config.get("retrieval_top_k", 5)),
        )
        knowledge_result = parse_knowledge_markdown(
            knowledge_path,
            retrieved_chunks=retrieved_chunks,
        )
        concepts = knowledge_result.concepts
        warnings.extend(knowledge_result.warnings)
        retrieved_chunk_payloads = [
            {
                "chunk_id": chunk.chunk_id,
                "score": round(chunk.score, 6),
                "heading_path": chunk.heading_path,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "text": chunk.text,
            }
            for chunk in knowledge_result.retrieved_chunks
        ]
        knowledge_debug = {
            **knowledge_result.debug,
            "knowledge_index": knowledge_index.debug,
        }
        build_trace.append(
            {
                "stage": "knowledge",
                "concept_count": len(concepts),
                "retrieved_chunk_count": len(retrieved_chunk_payloads),
            }
        )
    else:
        warnings.append("knowledge.md not found; semantic layer built from schema/profile only.")
        build_trace.append({"stage": "knowledge", "status": "missing"})

    schema_result = profile_structured_sources(context_dir)
    warnings.extend(schema_result.warnings)
    build_trace.append(
        {
            "stage": "schema_profile",
            "field_count": len(schema_result.schema_profiles),
            "join_candidate_count": len(schema_result.join_candidates),
        }
    )

    matching_result = match_concepts_to_schema(
        question=question,
        concepts=concepts,
        schema_profiles=schema_result.schema_profiles,
        join_candidates=schema_result.join_candidates,
        retrieved_chunks=retrieved_chunks,
    )
    warnings.extend(matching_result.warnings)
    build_trace.append(
        {
            "stage": "matcher",
            "semantic_link_count": len(matching_result.semantic_links),
        }
    )
    build_trace.append(
        {
            "stage": "complete",
            "elapsed_seconds": round(perf_counter() - started_at, 4),
        }
    )

    layer = SemanticLayer(
        task_id=task_dir.name,
        question=question,
        knowledge_present=knowledge_present,
        concepts=matching_result.concepts,
        schema_profiles=schema_result.schema_profiles,
        join_candidates=schema_result.join_candidates,
        semantic_links=matching_result.semantic_links,
        warnings=warnings,
        build_trace=build_trace,
        retrieved_knowledge_chunks=retrieved_chunk_payloads,
    )
    debug_payload = {
        "knowledge_extraction": knowledge_debug,
        "schema_profiling": schema_result.debug,
        "matching": matching_result.debug,
        "warnings": warnings,
        "build_trace": build_trace,
        "cache": {"status": "miss"},
    }
    _write_layer_artifacts(
        task_dir,
        question,
        layer,
        debug_payload,
        output_dir,
        effective_embedding_config,
    )
    return layer


def render_semantic_layer_summary(layer: SemanticLayer, *, max_chars: int = 1800) -> str:
    sections: list[str] = []

    if layer.retrieved_knowledge_chunks:
        retrieval_lines = [
            f"- {' / '.join(chunk.get('heading_path', [])) or '(root)'} "
            f"[score={float(chunk.get('score', 0.0)):.2f}]"
            for chunk in layer.retrieved_knowledge_chunks[:4]
        ]
        sections.append("Retrieved knowledge evidence:\n" + "\n".join(retrieval_lines))

    if layer.concepts:
        concept_lines = []
        for concept in sorted(layer.concepts, key=lambda item: (-item.confidence, item.canonical_name))[:5]:
            line = f"- {concept.canonical_name}"
            if concept.definition:
                line += f": {concept.definition}"
            details: list[str] = []
            if concept.value_mappings:
                rendered = ", ".join(
                    f"{mapping.source_text}->{mapping.normalized_value}"
                    for mapping in concept.value_mappings[:3]
                )
                details.append(f"mappings={rendered}")
            if concept.time_scope:
                details.append(f"time_scope={concept.time_scope}")
            if concept.unit:
                details.append(f"unit={concept.unit}")
            if details:
                line += f" ({'; '.join(details)})"
            concept_lines.append(line)
        sections.append("Key concepts:\n" + "\n".join(concept_lines))

    if layer.semantic_links:
        link_lines = []
        for link in sorted(layer.semantic_links, key=lambda item: (-item.confidence, item.concept_name))[:6]:
            rendered = f"- {link.concept_name} -> {link.matched_table}.{link.matched_field}"
            if link.matched_values:
                rendered += f" = {', '.join(link.matched_values)}"
            rendered += f" [{link.link_type}, conf={link.confidence:.2f}]"
            link_lines.append(rendered)
        sections.append("High-confidence semantic links:\n" + "\n".join(link_lines))

    if layer.join_candidates:
        join_lines = [
            f"- {join.left_table}.{join.left_field} <-> {join.right_table}.{join.right_field} "
            f"[score={join.score:.2f}]"
            for join in layer.join_candidates[:4]
        ]
        sections.append("Likely joins:\n" + "\n".join(join_lines))

    if layer.warnings:
        warning_lines = [f"- {warning}" for warning in layer.warnings[:4]]
        sections.append("Warnings:\n" + "\n".join(warning_lines))

    summary = "\n\n".join(section for section in sections if section).strip()
    if len(summary) <= max_chars:
        return summary
    return summary[: max_chars - 3].rstrip() + "..."
