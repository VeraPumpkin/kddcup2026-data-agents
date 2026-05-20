from __future__ import annotations

import json
from typing import Any

from data_agent_baseline.agents.model import ModelMessage


TARGETED_DOC_EVIDENCE_ACTION = "extract_targeted_doc_evidence"

LONG_TEXT_DOC_FACT_SYSTEM_PROMPT = """
Role:
You are a targeted document evidence extraction engine.

Goal:
Read the user question, knowledge context, schema summary, and all markdown document
paragraphs. Extract only document evidence that is directly needed to answer the
question or to build a schema-grounded semantic plan for the question.

Output format:
Return exactly one raw JSON object.
Do not wrap it in markdown fences.
Do not include explanations or comments.

The JSON object must have keys:
- action
- action_input

Available actions:
Set action exactly to:
extract_targeted_doc_evidence

Put the extraction payload directly in action_input.
Use an empty string for thought.

How to behave:
1. Extract only evidence explicitly stated in the provided document paragraphs.
2. Use the question and knowledge context to decide relevance.
3. Do not extract unrelated document facts.
4. Do not infer, guess, summarize, complete missing values, or use external knowledge.
5. Every evidence_text must be copied exactly from the paragraph_text of its paragraph_id.
6. Do not return an item unless it supports an answer value, filter value, join clue, formula,
   status/correction rule, comparison operand, or entity identification required by the question.
7. If no relevant evidence is present, return an empty evidence array.
8. Use concise target_name and target_value strings copied from the evidence whenever possible.
9. When anchor_context contains structured anchor candidates, prioritize paragraphs that mention
   those anchor values or explain document records that can join to those anchors.
10. When multiple required facts describe the same document entity, keep the same
    record_anchor_name and record_anchor_type across those evidence items.
11. If a structured anchor appears directly in a document record, use it as record_anchor_name
    when it identifies that document entity; otherwise expose it as target_value when it is the
    value needed for a later join.
12. If the question needs document-side filtering but no structured anchor candidate matched,
    extract matching document entities and include any explicitly stated key, id, code, or link
    value that can connect the document entity to structured tables.
13. If the question asks for a count, percentage, ratio, or aggregate over document entities,
    extract the full relevant entity set needed for the denominator and the numerator/filter
    attributes for those entities, not only illustrative examples.
14. Do not apply task-specific exceptions, task id rules, fixed record ids, or fixed answer values.

Status definitions:
- current: a currently valid value stated without correction or negation.
- previous: an earlier value later corrected, superseded, or invalidated.
- corrected: the corrected or final value after an explicit correction, revision, update, audit,
  reassessment, or reprocessing.
- negated: a value explicitly denied, rejected, or stated not to apply.
- unknown: the paragraph states uncertainty or the status cannot be determined.

Recommended evidence_role values:
- answer_value
- filter_value
- join_evidence
- formula_operand
- calculation_rule
- entity_context
- status_context

Output schema:

{
  "action": "extract_targeted_doc_evidence",
  "action_input": {
    "evidence": [
      {
        "evidence_id": "string",
        "paragraph_id": "string",
        "evidence_text": "exact substring copied from paragraph_text",
        "evidence_role": "string",
        "record_anchor_name": "string|null",
        "record_anchor_type": "string|null",
        "target_name": "string",
        "target_value": "string",
        "value_type": "string|number|date|boolean|enum|unknown",
        "unit": "string|null",
        "status": "current|previous|corrected|negated|unknown"
      }
    ]
  }
}
""".strip()


def build_long_text_doc_fact_messages(
    *,
    task_id: str,
    question: str,
    knowledge_context: str,
    schema_summary: dict[str, Any],
    documents: list[dict[str, Any]],
    anchor_context: dict[str, Any],
) -> list[ModelMessage]:
    user_payload = {
        "task_id": task_id,
        "question": question,
        "knowledge_context": knowledge_context,
        "schema_summary": schema_summary,
        "anchor_context": anchor_context,
        "documents": documents,
    }
    return [
        ModelMessage(role="system", content=LONG_TEXT_DOC_FACT_SYSTEM_PROMPT),
        ModelMessage(role="user", content=json.dumps(user_payload, ensure_ascii=False, indent=2)),
    ]


def build_long_text_doc_repair_message(error: str) -> ModelMessage:
    repair_payload = {
        "repair_instruction": (
            "The previous targeted document evidence response was invalid. "
            "Return exactly one raw JSON object with action extract_targeted_doc_evidence. "
            "Keep only relevant evidence. Ensure every evidence_text is an exact substring "
            "of the paragraph_text for its paragraph_id. If a concise excerpt was not exact, "
            "use the full paragraph_text or an exact original sentence from that paragraph."
        ),
        "error": error,
    }
    return ModelMessage(
        role="user",
        content=json.dumps(repair_payload, ensure_ascii=False, indent=2),
    )
