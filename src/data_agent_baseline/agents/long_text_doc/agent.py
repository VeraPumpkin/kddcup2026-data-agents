from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data_agent_baseline.agents.long_text_doc.prompt import build_long_text_doc_fact_messages
from data_agent_baseline.agents.model import OpenAIModelAdapter
from data_agent_baseline.agents.react import parse_model_step
from data_agent_baseline.agents.runtime import StepRecord
from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.tools.structured_context import StructuredContextStore


LOGGER = logging.getLogger(__name__)

SOURCE_AGENT = "LongTextDocFactAgent"
DEFAULT_DOC_GLOB = "doc/*.md"
EXTRACT_BATCH_ACTION = "extract_document_batch_facts"
DEFAULT_DOC_BATCH_CHAR_BUDGET = 48_000

_BLANK_LINE_PATTERN = re.compile(r"(?:\r?\n)[ \t]*(?:\r?\n)+")
_ALLOWED_VALUE_TYPES = {"string", "number", "date", "boolean", "enum", "unknown"}
_ALLOWED_STATUSES = {"current", "previous", "corrected", "negated", "unknown"}
_FORBIDDEN_BATCH_OUTPUT_KEYS = {"paragraph_text", "paragraph_index", "file_path"}


@dataclass(frozen=True, slots=True)
class MarkdownParagraph:
    file_path: str
    paragraph_id: str
    paragraph_text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_path": self.file_path,
            "paragraph_id": self.paragraph_id,
        }

    def prompt_dict(self) -> dict[str, Any]:
        return {
            "file_path": self.file_path,
            "paragraph_id": self.paragraph_id,
            "paragraph_text": self.paragraph_text,
        }

    def duckdb_row(self) -> dict[str, Any]:
        return {
            "paragraph_id": self.paragraph_id,
        }


@dataclass(frozen=True, slots=True)
class MarkdownDocument:
    file_path: str
    absolute_path: Path
    full_text: str
    paragraphs: list[MarkdownParagraph]


@dataclass(frozen=True, slots=True)
class DocAttributeFact:
    file_path: str
    paragraph_id: str
    record_anchor_name: str | None
    record_anchor_type: str | None
    entity_name_raw: str
    entity_value_raw: str
    value_type: str
    unit: str | None
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_path": self.file_path,
            "paragraph_id": self.paragraph_id,
            "record_anchor_name": self.record_anchor_name,
            "record_anchor_type": self.record_anchor_type,
            "entity_name_raw": self.entity_name_raw,
            "entity_value_raw": self.entity_value_raw,
            "value_type": self.value_type,
            "unit": self.unit,
            "status": self.status,
        }

    def duckdb_row(self) -> dict[str, Any]:
        return {
            "paragraph_id": self.paragraph_id,
            "record_anchor_name": self.record_anchor_name,
            "record_anchor_type": self.record_anchor_type,
            "entity_name_raw": self.entity_name_raw,
            "entity_value_raw": self.entity_value_raw,
            "value_type": self.value_type,
            "unit": self.unit,
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class DocRelationFact:
    file_path: str
    paragraph_id: str
    subject_name: str
    relation_type: str
    object_name_or_value: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_path": self.file_path,
            "paragraph_id": self.paragraph_id,
            "subject_name": self.subject_name,
            "relation_type": self.relation_type,
            "object_name_or_value": self.object_name_or_value,
        }

    def duckdb_row(self) -> dict[str, Any]:
        return {
            "paragraph_id": self.paragraph_id,
            "subject_name": self.subject_name,
            "relation_type": self.relation_type,
            "object_name_or_value": self.object_name_or_value,
        }


@dataclass(frozen=True, slots=True)
class LongTextDocFactOutput:
    paragraphs: list[MarkdownParagraph]
    facts: list[DocAttributeFact]
    relations: list[DocRelationFact]

    def to_dict(self) -> dict[str, Any]:
        return {
            "paragraphs": [item.to_dict() for item in self.paragraphs],
            "facts": [item.to_dict() for item in self.facts],
            "relations": [item.to_dict() for item in self.relations],
        }


@dataclass(frozen=True, slots=True)
class _ParagraphBatch:
    batch_id: str
    paragraphs: list[MarkdownParagraph]


class LongTextDocFactAgent:
    """Extract paragraph-level structured facts from context/doc/*.md files."""

    def __init__(
        self,
        *,
        model: OpenAIModelAdapter,
    ) -> None:
        self.model = model
        self.doc_batch_char_budget = DEFAULT_DOC_BATCH_CHAR_BUDGET
        self.last_steps: list[StepRecord] = []
        self.last_output: LongTextDocFactOutput | None = None

    def run(
        self,
        task: PublicTask,
        *,
        structured_store: StructuredContextStore,
    ) -> LongTextDocFactOutput:
        LOGGER.warning("LongTextDocFactAgent start task_id=%s", task.task_id)
        self.last_steps = []
        documents = self._load_documents(task)
        all_paragraphs = [paragraph for document in documents for paragraph in document.paragraphs]
        facts: list[DocAttributeFact] = []
        relations: list[DocRelationFact] = []
        failed_batch_count = 0
        json_parse_failure_count = 0
        batches, oversized_paragraph_count = self._build_paragraph_batches(all_paragraphs)
        for batch in batches:
            payload, failure_status = self._extract_batch(batch)
            if payload is None:
                failed_batch_count += 1
                if failure_status == "json_parse_error":
                    json_parse_failure_count += 1
                continue
            batch_facts, batch_relations = self._flatten_payload(
                batch.paragraphs,
                payload,
            )
            facts.extend(batch_facts)
            relations.extend(batch_relations)

        if all_paragraphs:
            structured_store.register_doc_fact_tables(
                paragraphs=[item.duckdb_row() for item in all_paragraphs],
                facts=[item.duckdb_row() for item in facts],
                relations=[item.duckdb_row() for item in relations],
            )
        output = LongTextDocFactOutput(
            paragraphs=all_paragraphs,
            facts=facts,
            relations=relations,
        )
        self.last_output = output
        LOGGER.warning(
            "LongTextDocFactAgent finish task_id=%s",
            task.task_id,
        )
        return output

    def _load_documents(self, task: PublicTask) -> list[MarkdownDocument]:
        paths = [
            path
            for path in sorted(task.context_dir.glob(DEFAULT_DOC_GLOB))
            if path.is_file() and path.suffix.lower() == ".md" and self._is_doc_markdown(task, path)
        ]
        documents: list[MarkdownDocument] = []
        for path in paths:
            relative_path = path.relative_to(task.context_dir).as_posix()
            text = path.read_text(errors="replace")
            documents.append(
                MarkdownDocument(
                    file_path=relative_path,
                    absolute_path=path,
                    full_text=text,
                    paragraphs=self._paragraphs_for_file(
                        file_path=relative_path,
                        text=text,
                    ),
                )
            )
        return documents

    def _paragraphs_for_file(
        self,
        *,
        file_path: str,
        text: str,
    ) -> list[MarkdownParagraph]:
        paragraphs: list[MarkdownParagraph] = []
        document_stem = Path(file_path).stem
        for paragraph_text in _split_paragraphs(text):
            paragraph_index = len(paragraphs)
            paragraphs.append(
                MarkdownParagraph(
                    file_path=file_path,
                    paragraph_id=f"{document_stem}::p{paragraph_index}",
                    paragraph_text=paragraph_text,
                )
            )
        return paragraphs

    def _build_paragraph_batches(
        self,
        paragraphs: list[MarkdownParagraph],
    ) -> tuple[list[_ParagraphBatch], int]:
        batches: list[_ParagraphBatch] = []
        current: list[MarkdownParagraph] = []
        oversized_paragraph_count = 0

        for paragraph in paragraphs:
            batch_id = self._batch_id(len(batches))
            candidate = [*current, paragraph]
            candidate_chars = self._batch_user_payload_char_count(batch_id, candidate)
            if current and candidate_chars > self.doc_batch_char_budget:
                batches.append(_ParagraphBatch(batch_id=batch_id, paragraphs=current))
                current = [paragraph]
                single_chars = self._batch_user_payload_char_count(
                    self._batch_id(len(batches)),
                    current,
                )
                if single_chars > self.doc_batch_char_budget:
                    oversized_paragraph_count += 1
                continue
            if not current and candidate_chars > self.doc_batch_char_budget:
                oversized_paragraph_count += 1
            current = candidate

        if current:
            batches.append(_ParagraphBatch(batch_id=self._batch_id(len(batches)), paragraphs=current))
        return batches, oversized_paragraph_count

    def _extract_batch(
        self,
        batch: _ParagraphBatch,
    ) -> tuple[dict[str, Any] | None, str | None]:
        messages = build_long_text_doc_fact_messages(
            batch_id=batch.batch_id,
            paragraphs=[paragraph.prompt_dict() for paragraph in batch.paragraphs],
        )
        step_index = len(self.last_steps) + 1
        raw_response = ""
        try:
            raw_response = self.model.complete(messages)
        except Exception as exc:
            self._record_step(
                step_index=step_index,
                thought="",
                action="__error__",
                action_input={},
                raw_response=raw_response,
                observation={"ok": False, "error": str(exc), "batch_id": batch.batch_id},
                ok=False,
                status="model_error",
            )
            return None, "model_error"

        try:
            model_step = parse_model_step(raw_response)
        except Exception as exc:
            self._record_step(
                step_index=step_index,
                thought="",
                action="__error__",
                action_input={},
                raw_response=raw_response,
                observation={"ok": False, "error": str(exc), "batch_id": batch.batch_id},
                ok=False,
                status="json_parse_error",
            )
            return None, "json_parse_error"

        try:
            payload = self._parse_batch_payload(model_step.action, model_step.action_input, batch)
        except Exception as exc:
            self._record_step(
                step_index=step_index,
                thought=model_step.thought,
                action=model_step.action,
                action_input=model_step.action_input,
                raw_response=model_step.raw_response,
                observation={"ok": False, "error": str(exc), "batch_id": batch.batch_id},
                ok=False,
                status="schema_validation_error",
            )
            return None, "schema_validation_error"

        self._record_step(
            step_index=step_index,
            thought=model_step.thought,
            action=model_step.action,
            action_input=model_step.action_input,
            raw_response=model_step.raw_response,
            observation={
                "ok": True,
                "tool": model_step.action,
                "batch_id": batch.batch_id,
                "paragraph_count": len(payload.get("paragraphs", [])),
            },
            ok=True,
            status="parsed",
        )
        return payload, None

    def _flatten_payload(
        self,
        paragraphs: list[MarkdownParagraph],
        payload: dict[str, Any],
    ) -> tuple[list[DocAttributeFact], list[DocRelationFact]]:
        paragraph_by_id = {paragraph.paragraph_id: paragraph for paragraph in paragraphs}
        facts: list[DocAttributeFact] = []
        relations: list[DocRelationFact] = []

        for paragraph_payload in payload.get("paragraphs", []):
            if not isinstance(paragraph_payload, dict):
                continue
            paragraph = paragraph_by_id.get(str(paragraph_payload.get("paragraph_id") or ""))
            if paragraph is None:
                continue
            records = paragraph_payload.get("records", [])
            if not isinstance(records, list):
                continue
            for record in records:
                if not isinstance(record, dict):
                    continue
                anchor = record.get("record_anchor")
                anchor_payload = anchor if isinstance(anchor, dict) else {}
                anchor_name = _optional_str(anchor_payload.get("name"))
                anchor_type = _optional_str(anchor_payload.get("type")) or "unknown"

                attributes = record.get("attributes", [])
                if isinstance(attributes, list):
                    for attribute in attributes:
                        if not isinstance(attribute, dict):
                            continue
                        facts.append(
                            DocAttributeFact(
                                file_path=paragraph.file_path,
                                paragraph_id=paragraph.paragraph_id,
                                record_anchor_name=anchor_name,
                                record_anchor_type=anchor_type,
                                entity_name_raw=str(attribute.get("entity_name_raw") or ""),
                                entity_value_raw=str(attribute.get("entity_value_raw") or ""),
                                value_type=_allowed_text(
                                    attribute.get("value_type"),
                                    _ALLOWED_VALUE_TYPES,
                                    default="unknown",
                                ),
                                unit=_optional_str(attribute.get("unit")),
                                status=_allowed_text(
                                    attribute.get("status"),
                                    _ALLOWED_STATUSES,
                                    default="unknown",
                                ),
                            )
                        )

                relation_items = record.get("relations", [])
                if isinstance(relation_items, list):
                    for relation in relation_items:
                        if not isinstance(relation, dict):
                            continue
                        relations.append(
                            DocRelationFact(
                                file_path=paragraph.file_path,
                                paragraph_id=paragraph.paragraph_id,
                                subject_name=str(relation.get("subject_name") or ""),
                                relation_type=str(relation.get("relation_type") or ""),
                                object_name_or_value=str(relation.get("object_name_or_value") or ""),
                            )
                        )
        return facts, relations

    def _parse_batch_payload(
        self,
        action: str,
        action_input: dict[str, Any],
        batch: _ParagraphBatch,
    ) -> dict[str, Any]:
        if action != EXTRACT_BATCH_ACTION:
            raise ValueError(f"Long text doc action must be {EXTRACT_BATCH_ACTION}.")
        payload = dict(action_input)
        self._validate_payload_shape(
            payload,
            batch_id=batch.batch_id,
            allowed_paragraph_ids={paragraph.paragraph_id for paragraph in batch.paragraphs},
        )
        return payload

    def _validate_payload_shape(
        self,
        payload: dict[str, Any],
        *,
        batch_id: str,
        allowed_paragraph_ids: set[str],
    ) -> None:
        if any(key in payload for key in _FORBIDDEN_BATCH_OUTPUT_KEYS):
            raise ValueError("Long text doc batch payload must not include source paragraph fields.")
        if str(payload.get("batch_id") or "") != batch_id:
            raise ValueError("Long text doc extraction payload batch_id does not match request.")
        paragraphs = payload.get("paragraphs")
        if not isinstance(paragraphs, list):
            raise ValueError("Long text doc extraction payload must contain paragraphs list.")
        for index, paragraph in enumerate(paragraphs):
            if not isinstance(paragraph, dict):
                raise ValueError(f"paragraphs[{index}] must be an object.")
            if any(key in paragraph for key in _FORBIDDEN_BATCH_OUTPUT_KEYS):
                raise ValueError(
                    f"paragraphs[{index}] must not include source paragraph fields."
                )
            paragraph_id = str(paragraph.get("paragraph_id") or "").strip()
            if not paragraph_id:
                raise ValueError(f"paragraphs[{index}].paragraph_id must be a non-empty string.")
            if paragraph_id not in allowed_paragraph_ids:
                raise ValueError(f"paragraphs[{index}].paragraph_id is not in the current batch.")
            if not isinstance(paragraph.get("records"), list):
                raise ValueError(f"paragraphs[{index}].records must be a list.")
            for record_index, record in enumerate(paragraph["records"]):
                if not isinstance(record, dict):
                    raise ValueError(
                        f"paragraphs[{index}].records[{record_index}] must be an object."
                    )
                anchor = record.get("record_anchor")
                if anchor is not None and not isinstance(anchor, dict):
                    raise ValueError(
                        f"paragraphs[{index}].records[{record_index}].record_anchor must be an object or null."
                    )
                attributes = record.get("attributes", [])
                if not isinstance(attributes, list):
                    raise ValueError(
                        f"paragraphs[{index}].records[{record_index}].attributes must be a list."
                    )
                relations = record.get("relations", [])
                if not isinstance(relations, list):
                    raise ValueError(
                        f"paragraphs[{index}].records[{record_index}].relations must be a list."
                    )
                for attribute_index, attribute in enumerate(attributes):
                    if not isinstance(attribute, dict):
                        raise ValueError(
                            f"paragraphs[{index}].records[{record_index}].attributes[{attribute_index}] must be an object."
                        )
                for relation_index, relation in enumerate(relations):
                    if not isinstance(relation, dict):
                        raise ValueError(
                            f"paragraphs[{index}].records[{record_index}].relations[{relation_index}] must be an object."
                        )

    def _record_step(
        self,
        *,
        step_index: int,
        thought: str,
        action: str,
        action_input: dict[str, Any],
        raw_response: Any,
        observation: dict[str, Any],
        ok: bool,
        status: str,
    ) -> None:
        self.last_steps.append(
            StepRecord(
                step_index=step_index,
                thought=thought,
                action=action,
                action_input=action_input,
                raw_response=raw_response,
                observation=observation,
                ok=ok,
                agent_state={
                    "agent_name": SOURCE_AGENT,
                    "status": status,
                },
            )
        )

    def _is_doc_markdown(self, task: PublicTask, path: Path) -> bool:
        try:
            relative = path.relative_to(task.context_dir).as_posix()
        except ValueError:
            return False
        return relative.startswith("doc/") and path.suffix.lower() == ".md"

    def _batch_user_payload_char_count(
        self,
        batch_id: str,
        paragraphs: list[MarkdownParagraph],
    ) -> int:
        messages = build_long_text_doc_fact_messages(
            batch_id=batch_id,
            paragraphs=[paragraph.prompt_dict() for paragraph in paragraphs],
        )
        return len(messages[1].content)

    def _batch_id(self, index: int) -> str:
        return f"doc_batch_{index}"

def _split_paragraphs(text: str) -> list[str]:
    return [paragraph.strip() for paragraph in _BLANK_LINE_PATTERN.split(text) if paragraph.strip()]


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _allowed_text(value: Any, allowed: set[str], *, default: str) -> str:
    text = str(value or "").strip().lower()
    return text if text in allowed else default
