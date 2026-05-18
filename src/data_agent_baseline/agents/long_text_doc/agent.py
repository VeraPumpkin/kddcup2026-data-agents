from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from data_agent_baseline.agents.long_text_doc.prompt import (
    TARGETED_DOC_EVIDENCE_ACTION,
    build_long_text_doc_fact_messages,
    build_long_text_doc_repair_message,
)
from data_agent_baseline.agents.model import ModelMessage, OpenAIModelAdapter
from data_agent_baseline.agents.react import parse_model_step
from data_agent_baseline.agents.runtime import StepRecord
from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.tools.structured_context import StructuredContextStore


LOGGER = logging.getLogger(__name__)

SOURCE_AGENT = "LongTextDocFactAgent"
DEFAULT_DOC_GLOB = "doc/*.md"
MAX_EVIDENCE_CONTEXT_CHARS = 20_000

_BLANK_LINE_PATTERN = re.compile(r"(?:\r?\n)[ \t]*(?:\r?\n)+")
_ALLOWED_VALUE_TYPES = {"string", "number", "date", "boolean", "enum", "unknown"}
_ALLOWED_STATUSES = {"current", "previous", "corrected", "negated", "unknown"}


@dataclass(frozen=True, slots=True)
class MarkdownParagraph:
    file_path: str
    paragraph_id: str
    paragraph_index: int
    paragraph_text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_path": self.file_path,
            "paragraph_id": self.paragraph_id,
            "paragraph_index": self.paragraph_index,
        }

    def prompt_dict(self) -> dict[str, Any]:
        return {
            "file_path": self.file_path,
            "paragraph_id": self.paragraph_id,
            "paragraph_index": self.paragraph_index,
            "paragraph_text": self.paragraph_text,
        }

    def duckdb_row(self) -> dict[str, Any]:
        return {
            "paragraph_id": self.paragraph_id,
            "file_path": self.file_path,
            "paragraph_index": self.paragraph_index,
            "paragraph_text": self.paragraph_text,
        }


@dataclass(frozen=True, slots=True)
class MarkdownDocument:
    file_path: str
    absolute_path: Path
    full_text: str
    paragraphs: list[MarkdownParagraph]

    def prompt_dict(self) -> dict[str, Any]:
        return {
            "file_path": self.file_path,
            "paragraphs": [paragraph.prompt_dict() for paragraph in self.paragraphs],
        }


@dataclass(frozen=True, slots=True)
class DocEvidence:
    evidence_id: str
    paragraph_id: str
    file_path: str
    evidence_text: str
    evidence_role: str
    record_anchor_name: str | None
    record_anchor_type: str | None
    target_name: str
    target_value: str
    value_type: str
    unit: str | None
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "paragraph_id": self.paragraph_id,
            "file_path": self.file_path,
            "evidence_text": self.evidence_text,
            "evidence_role": self.evidence_role,
            "record_anchor_name": self.record_anchor_name,
            "record_anchor_type": self.record_anchor_type,
            "target_name": self.target_name,
            "target_value": self.target_value,
            "value_type": self.value_type,
            "unit": self.unit,
            "status": self.status,
        }

    def duckdb_row(self) -> dict[str, Any]:
        return self.to_dict()


@dataclass(frozen=True, slots=True)
class DocAttributeFact:
    evidence_id: str
    paragraph_id: str
    record_anchor_name: str | None
    record_anchor_type: str | None
    entity_name_raw: str
    entity_value_raw: str
    value_type: str
    unit: str | None
    status: str
    evidence_text: str
    evidence_role: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "paragraph_id": self.paragraph_id,
            "record_anchor_name": self.record_anchor_name,
            "record_anchor_type": self.record_anchor_type,
            "entity_name_raw": self.entity_name_raw,
            "entity_value_raw": self.entity_value_raw,
            "value_type": self.value_type,
            "unit": self.unit,
            "status": self.status,
            "evidence_text": self.evidence_text,
            "evidence_role": self.evidence_role,
        }

    def duckdb_row(self) -> dict[str, Any]:
        return self.to_dict()


@dataclass(frozen=True, slots=True)
class LongTextDocFactOutput:
    paragraphs: list[MarkdownParagraph]
    evidence: list[DocEvidence]
    facts: list[DocAttributeFact]
    evidence_context: str
    diagnostics: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "paragraphs": [item.to_dict() for item in self.paragraphs],
            "evidence": [item.to_dict() for item in self.evidence],
            "facts": [item.to_dict() for item in self.facts],
            "evidence_context": self.evidence_context,
            "diagnostics": list(self.diagnostics),
        }


class LongTextDocFactAgent:
    """Extract question-targeted evidence from context/doc/*.md files."""

    def __init__(
        self,
        *,
        model: OpenAIModelAdapter,
    ) -> None:
        self.model = model
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
        evidence: list[DocEvidence] = []
        diagnostics: list[str] = []

        if all_paragraphs:
            evidence, diagnostics = self._extract_targeted_evidence(
                task=task,
                structured_store=structured_store,
                documents=documents,
                paragraphs=all_paragraphs,
            )

        facts = self._facts_from_evidence(evidence)
        evidence_context = self._render_evidence_context(evidence, diagnostics)
        structured_store.register_doc_evidence_tables(
            paragraphs=[item.duckdb_row() for item in all_paragraphs],
            evidence=[item.duckdb_row() for item in evidence],
            facts=[item.duckdb_row() for item in facts],
        )
        output = LongTextDocFactOutput(
            paragraphs=all_paragraphs,
            evidence=evidence,
            facts=facts,
            evidence_context=evidence_context,
            diagnostics=diagnostics,
        )
        self.last_output = output
        LOGGER.warning("LongTextDocFactAgent finish task_id=%s", task.task_id)
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
                    paragraph_index=paragraph_index,
                    paragraph_text=paragraph_text,
                )
            )
        return paragraphs

    def _extract_targeted_evidence(
        self,
        *,
        task: PublicTask,
        structured_store: StructuredContextStore,
        documents: list[MarkdownDocument],
        paragraphs: list[MarkdownParagraph],
    ) -> tuple[list[DocEvidence], list[str]]:
        messages = build_long_text_doc_fact_messages(
            task_id=task.task_id,
            question=task.question,
            knowledge_context=self._read_knowledge_context(task),
            schema_summary=structured_store.inspect_schema(),
            documents=[document.prompt_dict() for document in documents],
        )
        paragraph_by_id = {paragraph.paragraph_id: paragraph for paragraph in paragraphs}
        raw_response = ""
        last_error = ""
        for attempt in range(1, 3):
            step_index = len(self.last_steps) + 1
            try:
                raw_response = self.model.complete(messages)
                model_step = parse_model_step(raw_response)
                evidence, diagnostics = self._parse_targeted_payload(
                    model_step.action,
                    model_step.action_input,
                    paragraph_by_id=paragraph_by_id,
                )
            except Exception as exc:
                last_error = str(exc)
                status = self._failure_status(exc)
                self._record_step(
                    step_index=step_index,
                    thought="",
                    action="__error__",
                    action_input={"attempt": attempt},
                    raw_response=raw_response,
                    observation={
                        "ok": False,
                        "error": last_error,
                        "attempt": attempt,
                    },
                    ok=False,
                    status=status,
                )
                if attempt == 1:
                    if raw_response:
                        messages.append(ModelMessage(role="assistant", content=raw_response))
                    messages.append(build_long_text_doc_repair_message(last_error))
                    continue
                return [], [f"targeted_doc_evidence_failed: {last_error}"]

            self._record_step(
                step_index=step_index,
                thought=model_step.thought,
                action=model_step.action,
                action_input=model_step.action_input,
                raw_response=model_step.raw_response,
                observation={
                    "ok": True,
                    "tool": model_step.action,
                    "attempt": attempt,
                    "paragraph_count": len(paragraphs),
                    "evidence_count": len(evidence),
                    "diagnostics": diagnostics,
                },
                ok=True,
                status="parsed",
            )
            return evidence, diagnostics

        return [], [f"targeted_doc_evidence_failed: {last_error}"]

    def _parse_targeted_payload(
        self,
        action: str,
        action_input: dict[str, Any],
        *,
        paragraph_by_id: dict[str, MarkdownParagraph],
    ) -> tuple[list[DocEvidence], list[str]]:
        if action != TARGETED_DOC_EVIDENCE_ACTION:
            raise ValueError(f"Long text doc action must be {TARGETED_DOC_EVIDENCE_ACTION}.")
        payload = dict(action_input)
        evidence_items = payload.get("evidence")
        if not isinstance(evidence_items, list):
            raise ValueError("targeted document evidence payload must contain evidence list.")

        evidence: list[DocEvidence] = []
        diagnostics = [
            str(item)
            for item in payload.get("diagnostics", [])
            if isinstance(item, str) and item.strip()
        ]
        used_ids: set[str] = set()
        for index, item in enumerate(evidence_items):
            if not isinstance(item, dict):
                diagnostics.append(f"rejected evidence[{index}]: item is not an object")
                continue
            paragraph_id = str(item.get("paragraph_id") or "").strip()
            paragraph = paragraph_by_id.get(paragraph_id)
            if paragraph is None:
                diagnostics.append(f"rejected evidence[{index}]: unknown paragraph_id {paragraph_id}")
                continue
            evidence_text = str(item.get("evidence_text") or "").strip()
            if not evidence_text:
                diagnostics.append(f"rejected evidence[{index}]: empty evidence_text")
                continue
            if evidence_text not in paragraph.paragraph_text:
                diagnostics.append(
                    f"rejected evidence[{index}]: evidence_text is not an exact paragraph substring"
                )
                continue
            target_name = str(item.get("target_name") or "").strip()
            target_value = str(item.get("target_value") or "").strip()
            if not target_name or not target_value:
                diagnostics.append(f"rejected evidence[{index}]: missing target_name or target_value")
                continue

            evidence_id = str(item.get("evidence_id") or "").strip()
            if not evidence_id or evidence_id in used_ids:
                evidence_id = self._evidence_id(paragraph_id, index, used_ids)
            used_ids.add(evidence_id)

            evidence.append(
                DocEvidence(
                    evidence_id=evidence_id,
                    paragraph_id=paragraph.paragraph_id,
                    file_path=paragraph.file_path,
                    evidence_text=evidence_text,
                    evidence_role=str(item.get("evidence_role") or "supporting_fact").strip(),
                    record_anchor_name=_optional_str(item.get("record_anchor_name")),
                    record_anchor_type=_optional_str(item.get("record_anchor_type")),
                    target_name=target_name,
                    target_value=target_value,
                    value_type=_allowed_text(
                        item.get("value_type"),
                        _ALLOWED_VALUE_TYPES,
                        default="unknown",
                    ),
                    unit=_optional_str(item.get("unit")),
                    status=_allowed_text(
                        item.get("status"),
                        _ALLOWED_STATUSES,
                        default="unknown",
                    ),
                )
            )
        return evidence, diagnostics

    def _facts_from_evidence(self, evidence: list[DocEvidence]) -> list[DocAttributeFact]:
        facts: list[DocAttributeFact] = []
        for item in evidence:
            facts.append(
                DocAttributeFact(
                    evidence_id=item.evidence_id,
                    paragraph_id=item.paragraph_id,
                    record_anchor_name=item.record_anchor_name,
                    record_anchor_type=item.record_anchor_type,
                    entity_name_raw=item.target_name,
                    entity_value_raw=item.target_value,
                    value_type=item.value_type,
                    unit=item.unit,
                    status=item.status,
                    evidence_text=item.evidence_text,
                    evidence_role=item.evidence_role,
                )
            )
        return facts

    def _render_evidence_context(
        self,
        evidence: list[DocEvidence],
        diagnostics: list[str],
    ) -> str:
        lines = []
        if not evidence:
            lines.append("No targeted document evidence extracted.")
        for item in evidence:
            anchor = item.record_anchor_name or ""
            target = item.target_name or ""
            value = item.target_value or ""
            evidence_text = _single_line(item.evidence_text)
            lines.append(
                f"- {item.evidence_id} | role={item.evidence_role} | "
                f"paragraph={item.paragraph_id} | anchor={anchor} | "
                f"target={target} | value={value} | status={item.status} | "
                f"evidence={evidence_text}"
            )
        for diagnostic in diagnostics:
            lines.append(f"- diagnostic: {_single_line(diagnostic)}")
        context = "\n".join(lines)
        if len(context) <= MAX_EVIDENCE_CONTEXT_CHARS:
            return context
        return context[: MAX_EVIDENCE_CONTEXT_CHARS - 20].rstrip() + "\n...[truncated]"

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

    def _read_knowledge_context(self, task: PublicTask) -> str:
        knowledge_path = task.context_dir / "knowledge.md"
        if not knowledge_path.is_file():
            return ""
        return knowledge_path.read_text(errors="replace")

    def _evidence_id(
        self,
        paragraph_id: str,
        index: int,
        used_ids: set[str],
    ) -> str:
        base = f"doc_evidence::{paragraph_id}::{index}"
        candidate = base
        suffix = 1
        while candidate in used_ids:
            candidate = f"{base}_{suffix}"
            suffix += 1
        return candidate

    def _failure_status(self, exc: Exception) -> str:
        message = str(exc)
        if "JSON" in message or "Expecting" in message:
            return "json_parse_error"
        return "schema_validation_error"


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


def _single_line(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()
