from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from data_agent_baseline.semantic.retriever import RetrievedChunk
from data_agent_baseline.semantic.models import ConceptEntry, SourceSpan, ValueMapping

HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(?P<title>.+?)\s*$")
DEFINITION_PATTERNS = [
    re.compile(r"^(?P<term>[A-Za-z][A-Za-z0-9 _/\-()]+?)\s*:\s*(?P<definition>.+)$"),
    re.compile(r"^(?P<term>[A-Za-z][A-Za-z0-9 _/\-()]+?)\s+means\s+(?P<definition>.+)$", re.I),
    re.compile(r"^(?P<term>[A-Za-z][A-Za-z0-9 _/\-()]+?)\s+refers to\s+(?P<definition>.+)$", re.I),
    re.compile(r"^(?P<term>[A-Za-z][A-Za-z0-9 _/\-()]+?)\s*=\s*(?P<definition>.+)$"),
]
NUMERIC_LEADING_MAPPING_PATTERN = re.compile(
    r"^(?P<code>[+-]?\d+(?:\.\d+)?)\s*(?:=|->|indicates|means|maps to)\s*(?P<label>.+)$",
    re.I,
)
NUMERIC_TRAILING_MAPPING_PATTERN = re.compile(
    r"^(?P<label>.+?)\s*(?:=|->|maps to|encoded as)\s*(?P<code>[+-]?\d+(?:\.\d+)?)$",
    re.I,
)
ALIASES_PATTERN = re.compile(r"\b(?:aka|alias(?:es)?|also called)\b\s*(?::|-)?\s*(?P<aliases>.+)$", re.I)
UNIT_PATTERN = re.compile(r"\bunit(?:s)?\b\s*(?::|=)?\s*(?P<unit>[A-Za-z%/0-9 _-]+)", re.I)
TIME_SCOPE_PATTERN = re.compile(
    r"\b(?:time scope|measured within|within|during|over)\b\s*(?::|=)?\s*(?P<scope>.+)$",
    re.I,
)
CONSTRAINT_PREFIXES = ("must ", "should ", "only ", "exclude ", "drop ", "filter ", "keep ")
COLUMN_HINT_PATTERN = re.compile(r"\bin\s+([A-Za-z][A-Za-z0-9_]*)\s+column\b", re.I)
TABLE_HINT_PATTERN = re.compile(r"\bin\s+([A-Za-z][A-Za-z0-9_]*)\s+table\b", re.I)


@dataclass(frozen=True, slots=True)
class KnowledgeParseResult:
    concepts: list[ConceptEntry]
    retrieved_chunks: list[RetrievedChunk]
    debug: dict[str, object]
    warnings: list[str]


def _split_lines_with_numbers(text: str) -> list[tuple[int, str]]:
    return [(index, line.rstrip()) for index, line in enumerate(text.splitlines(), start=1)]


def _clean_term(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text.strip(" -*\t"))
    return cleaned.rstrip(".")


def _tokenize(text: str) -> list[str]:
    return [token for token in re.findall(r"[A-Za-z0-9_]+", text.lower()) if len(token) > 1]


def _extract_aliases(text: str) -> list[str]:
    match = ALIASES_PATTERN.search(text)
    if match is None:
        return []
    aliases_text = match.group("aliases")
    parts = re.split(r",|/|\bor\b", aliases_text)
    return [_clean_term(part) for part in parts if _clean_term(part)]


def _extract_unit(text: str) -> str | None:
    match = UNIT_PATTERN.search(text)
    return _clean_term(match.group("unit")) if match else None


def _extract_time_scope(text: str) -> str | None:
    match = TIME_SCOPE_PATTERN.search(text)
    return _clean_term(match.group("scope")) if match else None


def _mapping_hint_candidates(text: str) -> tuple[list[str], list[str]]:
    columns = re.findall(r"\b[A-Za-z][A-Za-z0-9_]*\b", text)
    target_columns = [item for item in columns if "_" in item or item.lower().endswith("id")]
    target_columns.extend(COLUMN_HINT_PATTERN.findall(text))
    target_tables = TABLE_HINT_PATTERN.findall(text)
    return sorted(set(target_columns)), sorted(set(target_tables))


def _extract_value_mapping(
    normalized_line: str,
    canonical_name: str | None,
    knowledge_path: Path,
    line_number: int,
) -> ValueMapping | None:
    numeric_leading = NUMERIC_LEADING_MAPPING_PATTERN.match(normalized_line)
    if numeric_leading:
        normalized_value = _clean_term(numeric_leading.group("code"))
        source_text = _clean_term(numeric_leading.group("label"))
        if canonical_name and canonical_name.lower() in source_text.lower():
            source_text = canonical_name
        target_columns, target_tables = _mapping_hint_candidates(normalized_line)
        return ValueMapping(
            source_text=source_text,
            normalized_value=normalized_value,
            target_column_candidates=target_columns,
            target_table_candidates=target_tables,
            evidence=[f"{knowledge_path.name}:L{line_number}"],
        )

    numeric_trailing = NUMERIC_TRAILING_MAPPING_PATTERN.match(normalized_line)
    if numeric_trailing:
        source_text = _clean_term(numeric_trailing.group("label"))
        normalized_value = _clean_term(numeric_trailing.group("code"))
        if canonical_name and source_text.lower() == canonical_name.lower():
            target_columns, target_tables = _mapping_hint_candidates(normalized_line)
            return ValueMapping(
                source_text=source_text,
                normalized_value=normalized_value,
                target_column_candidates=target_columns,
                target_table_candidates=target_tables,
                evidence=[f"{knowledge_path.name}:L{line_number}"],
            )
    return None


def parse_knowledge_markdown(
    knowledge_path: Path,
    *,
    retrieved_chunks: list[RetrievedChunk] | None = None,
) -> KnowledgeParseResult:
    effective_chunks = list(retrieved_chunks or [])
    if effective_chunks:
        text = "\n\n".join(chunk.text for chunk in effective_chunks)
    else:
        text = knowledge_path.read_text(encoding="utf-8", errors="replace")
    numbered_lines = _split_lines_with_numbers(text)
    concepts: list[ConceptEntry] = []
    warnings: list[str] = []
    debug_sections: list[dict[str, object]] = []

    current_heading: tuple[str, int] | None = None
    block_lines: list[tuple[int, str]] = []

    def flush_block() -> None:
        nonlocal block_lines
        if current_heading is None and not block_lines:
            return

        source_start = current_heading[1] if current_heading is not None else None
        heading_title = current_heading[0] if current_heading is not None else None
        content_lines = [line for _, line in block_lines if line.strip()]
        content_text = "\n".join(content_lines).strip()
        if not heading_title and not content_text:
            block_lines = []
            return

        canonical_name = heading_title
        definition = None
        aliases: list[str] = []
        constraints: list[str] = []
        value_mappings: list[ValueMapping] = []
        unit = None
        time_scope = None
        evidence_lines: list[str] = []

        for line_number, raw_line in block_lines:
            stripped = raw_line.strip()
            if not stripped:
                continue
            normalized = re.sub(r"^(?:[-*]|\d+\.)\s*", "", stripped).strip()
            if definition is None:
                for pattern in DEFINITION_PATTERNS:
                    match = pattern.match(normalized)
                    if match:
                        candidate_term = _clean_term(match.group("term"))
                        candidate_definition = _clean_term(match.group("definition"))
                        if canonical_name is None:
                            canonical_name = candidate_term
                        elif candidate_term.lower() == canonical_name.lower():
                            definition = candidate_definition
                        else:
                            aliases.append(candidate_term)
                            definition = candidate_definition
                        evidence_lines.append(f"L{line_number}: definition pattern")
                        break

            aliases.extend(_extract_aliases(normalized))

            extracted_unit = _extract_unit(normalized)
            if extracted_unit and unit is None:
                unit = extracted_unit
                evidence_lines.append(f"L{line_number}: unit")

            extracted_time_scope = _extract_time_scope(normalized)
            if extracted_time_scope and time_scope is None:
                time_scope = extracted_time_scope
                evidence_lines.append(f"L{line_number}: time scope")

            lowered = normalized.lower()
            if lowered.startswith(CONSTRAINT_PREFIXES):
                constraints.append(normalized)
                evidence_lines.append(f"L{line_number}: constraint")

            mapping = _extract_value_mapping(normalized, canonical_name, knowledge_path, line_number)
            if mapping is not None:
                value_mappings.append(mapping)
                evidence_lines.append(f"L{line_number}: value mapping")

        if definition is None and content_lines:
            definition = _clean_term(content_lines[0])

        candidate_name = _clean_term(canonical_name or "")
        if not candidate_name:
            block_lines = []
            return

        aliases = sorted({alias for alias in aliases if alias and alias.lower() != candidate_name.lower()})
        confidence = 0.45
        if definition:
            confidence += 0.2
        if aliases:
            confidence += 0.1
        if value_mappings:
            confidence += 0.15
        if unit or time_scope:
            confidence += 0.05

        excerpt_parts = [candidate_name]
        if definition:
            excerpt_parts.append(definition)
        excerpt = " | ".join(excerpt_parts)[:240]
        end_line = block_lines[-1][0] if block_lines else source_start
        concepts.append(
            ConceptEntry(
                canonical_name=candidate_name,
                aliases=aliases,
                definition=definition,
                source_span=SourceSpan(
                    path=knowledge_path.name,
                    start_line=source_start,
                    end_line=end_line,
                    excerpt=excerpt,
                ),
                constraints=constraints,
                value_mappings=value_mappings,
                time_scope=time_scope,
                unit=unit,
                confidence=min(confidence, 0.95),
            )
        )
        debug_sections.append(
            {
                "heading": heading_title,
                "start_line": source_start,
                "line_count": len(block_lines),
                "aliases": aliases,
                "mapping_count": len(value_mappings),
                "evidence": evidence_lines,
            }
        )
        block_lines = []

    for line_number, line in numbered_lines:
        heading_match = HEADING_PATTERN.match(line)
        if heading_match:
            flush_block()
            current_heading = (_clean_term(heading_match.group("title")), line_number)
            block_lines = []
            continue
        block_lines.append((line_number, line))
    flush_block()

    if not concepts:
        warnings.append(f"No structured concepts extracted from {knowledge_path.name}.")

    debug = {
        "knowledge_path": str(knowledge_path),
        "concept_count": len(concepts),
        "retrieved_chunk_count": len(effective_chunks),
        "retrieved_chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "score": round(chunk.score, 6),
                "heading_path": chunk.heading_path,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
            }
            for chunk in effective_chunks
        ],
        "sections": debug_sections,
        "tokens_seen": sorted({token for token in _tokenize(text)})[:200],
    }
    return KnowledgeParseResult(
        concepts=concepts,
        retrieved_chunks=effective_chunks,
        debug=debug,
        warnings=warnings,
    )
