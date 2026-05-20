from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from data_agent_baseline.tools.structured_context import StructuredContextStore


MAX_QUESTION_PHRASES = 80
MAX_ANCHORS = 80
MAX_ROW_VALUES = 12
_QUOTED_PHRASE_RE = re.compile(r'"([^"]+)"|“([^”]+)”|‘([^’]+)’')
_TITLE_SPAN_RE = re.compile(
    r"\b[A-Z][A-Za-z0-9]*(?:['’]s)?(?:\s+[A-Z][A-Za-z0-9]*(?:['’]s)?)+"
)
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_STOPWORDS = {
    "a",
    "about",
    "among",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "does",
    "for",
    "from",
    "have",
    "how",
    "in",
    "include",
    "is",
    "it",
    "many",
    "more",
    "of",
    "on",
    "or",
    "status",
    "than",
    "that",
    "the",
    "their",
    "them",
    "to",
    "was",
    "were",
    "what",
    "which",
    "who",
    "whose",
    "with",
    "write",
}
_KEY_FIELD_NAMES = {"id", "key", "code", "uuid", "guid"}
_KEY_FIELD_SUFFIXES = ("_id", "_key", "_code", "_uuid", "_guid")
_KEY_FIELD_PREFIXES = ("link_to_", "ref_", "fk_")


@dataclass(frozen=True, slots=True)
class DocAnchorCandidate:
    anchor_value: str
    anchor_type: str
    source_field: str
    source_table: str
    matched_field: str
    matched_value: str
    matched_phrase: str
    row_values: dict[str, Any]
    confidence: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchor_value": self.anchor_value,
            "anchor_type": self.anchor_type,
            "source_field": self.source_field,
            "source_table": self.source_table,
            "matched_field": self.matched_field,
            "matched_value": self.matched_value,
            "matched_phrase": self.matched_phrase,
            "row_values": dict(self.row_values),
            "confidence": self.confidence,
        }


@dataclass(frozen=True, slots=True)
class DocAnchorPrepassOutput:
    question_phrases: list[str]
    anchors: list[DocAnchorCandidate]
    anchor_context: str
    diagnostics: list[str]

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "question_phrases": list(self.question_phrases),
            "anchors": [anchor.to_dict() for anchor in self.anchors],
            "diagnostics": list(self.diagnostics),
        }


def extract_question_phrases(question: str) -> list[str]:
    """Extract original wording spans that may identify records or entities."""
    question_text = str(question or "")
    phrases: list[str] = []

    for match in _QUOTED_PHRASE_RE.finditer(question_text):
        _append_phrase(phrases, next(group for group in match.groups() if group))

    for match in _TITLE_SPAN_RE.finditer(question_text):
        phrase = re.sub(r"['’]s\b", "", match.group(0)).strip()
        _append_phrase(phrases, phrase)

    tokens = _TOKEN_RE.findall(question_text)
    content_tokens = [token for token in tokens if token.lower() not in _STOPWORDS]
    for size in range(min(6, len(content_tokens)), 0, -1):
        for start in range(0, len(content_tokens) - size + 1):
            phrase = " ".join(content_tokens[start : start + size])
            _append_phrase(phrases, phrase)
            if len(phrases) >= MAX_QUESTION_PHRASES:
                return phrases

    return phrases[:MAX_QUESTION_PHRASES]


def build_doc_anchor_prepass(
    *,
    question: str,
    structured_store: StructuredContextStore,
    limit_per_phrase: int = 20,
) -> DocAnchorPrepassOutput:
    phrases = extract_question_phrases(question)
    anchors: list[DocAnchorCandidate] = []
    diagnostics: list[str] = []
    seen: set[tuple[str, str, str, str]] = set()

    for phrase in phrases:
        if not phrase.strip():
            continue
        for field in structured_store.fields():
            if field.kind == "doc_annotation":
                continue
            rows = structured_store.match_rows(
                field.field,
                phrase,
                limit=limit_per_phrase,
            )
            if not rows:
                continue
            for row in rows:
                matched_value = _string_or_empty(row.get(field.name))
                for anchor_field, anchor_value in _anchor_values_from_row(
                    row,
                    table=field.table,
                ):
                    source_field = f"{field.table}.{anchor_field}"
                    key = (anchor_value, source_field, field.field, phrase)
                    if key in seen:
                        continue
                    seen.add(key)
                    anchors.append(
                        DocAnchorCandidate(
                            anchor_value=anchor_value,
                            anchor_type=_anchor_type(anchor_field),
                            source_field=source_field,
                            source_table=field.table,
                            matched_field=field.field,
                            matched_value=matched_value,
                            matched_phrase=phrase,
                            row_values=_compact_row_values(
                                row,
                                matched_column=field.name,
                            ),
                            confidence="exact_value_match",
                        )
                    )
                    if len(anchors) >= MAX_ANCHORS:
                        return _output(phrases, anchors, diagnostics)

    return _output(phrases, anchors, diagnostics)


def _output(
    phrases: list[str],
    anchors: list[DocAnchorCandidate],
    diagnostics: list[str],
) -> DocAnchorPrepassOutput:
    payload = {
        "question_phrases": phrases,
        "anchors": [anchor.to_dict() for anchor in anchors],
        "diagnostics": diagnostics,
    }
    return DocAnchorPrepassOutput(
        question_phrases=phrases,
        anchors=anchors,
        anchor_context=json.dumps(payload, ensure_ascii=False, indent=2),
        diagnostics=diagnostics,
    )


def _anchor_values_from_row(
    row: dict[str, Any],
    *,
    table: str,
) -> list[tuple[str, str]]:
    values: list[tuple[str, str]] = []
    for column, value in row.items():
        if not _is_anchor_field(column, table=table):
            continue
        text = _string_or_empty(value)
        if text:
            values.append((column, text))
    return values


def _compact_row_values(
    row: dict[str, Any],
    *,
    matched_column: str,
) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for column, value in row.items():
        if column == matched_column or _is_anchor_field(column, table=""):
            compact[column] = value
        if len(compact) >= MAX_ROW_VALUES:
            break
    return compact


def _is_anchor_field(column: str, *, table: str) -> bool:
    normalized = _normalize_identifier(column)
    normalized_table = _normalize_identifier(table)
    if normalized in _KEY_FIELD_NAMES:
        return True
    if normalized in {f"{normalized_table}_id", f"{normalized_table}_key", f"{normalized_table}_code"}:
        return True
    if normalized.endswith(_KEY_FIELD_SUFFIXES):
        return True
    return normalized.startswith(_KEY_FIELD_PREFIXES)


def _anchor_type(column: str) -> str:
    normalized = _normalize_identifier(column)
    for prefix in _KEY_FIELD_PREFIXES:
        if normalized.startswith(prefix) and len(normalized) > len(prefix):
            return normalized[len(prefix) :]
    return normalized


def _append_phrase(phrases: list[str], value: str) -> None:
    phrase = re.sub(r"\s+", " ", str(value or "")).strip(" ,.;:!?")
    if len(phrase) < 2:
        return
    if phrase.lower() in _STOPWORDS:
        return
    if phrase not in phrases:
        phrases.append(phrase)


def _normalize_identifier(value: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", str(value).lower())).strip("_")


def _string_or_empty(value: Any) -> str:
    text = str(value or "").strip()
    return text
