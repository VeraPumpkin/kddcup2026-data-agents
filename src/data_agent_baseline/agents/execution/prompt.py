from __future__ import annotations

import json
from typing import Any

from data_agent_baseline.agents.model import ModelMessage
from data_agent_baseline.tools.structured_context import StructuredContextStore


SEMANTIC_PLAN_EXECUTION_SYSTEM_PROMPT = """
Role:
You are the ExecutionAgent in a multi-agent data system.

Goal:
You receive a structured semantic_plan JSON object from QuestionUnderstandAgent. Use it with the registered DuckDB schema to construct and execute the necessary SQL or Python operations, then submit the final prediction table with the answer tool.

Interaction rules:
1. Use exactly one tool action per turn.
2. Use the answer tool for final output.
3. Do not redo semantic schema linking or invent missing intent.
4. Do not treat semantic_plan as natural language, executable SQL, or Python.
5. Use DuckDB schema and table inspection only to confirm queryable tables/columns and generate executable SQL; do not use them to reinterpret business intent or replace semantic_plan source fields.
6. If details are insufficient, inspect only relevant context assets or DuckDB tables.

Output format:
1. Return exactly one fenced JSON object with keys thought, action, and action_input.

Remember:
- Keep thought concise, visible, and grounded in observed data.
""".strip()


EXECUTION_INSTRUCTIONS = [
    "Use structured semantic_plan answer_columns, filters, joins, and group_by as the only source of business intent, then construct the executable query or Python logic.",
    "Use registered DuckDB tables for CSV/JSON context data, and inspect them only to confirm available tables, columns, types, and values needed to execute the semantic_plan.",
    "Use doc_paragraphs, doc_evidence, and doc_facts for markdown evidence and facts.",
    "Use doc_evidence.target_value, doc_evidence.evidence_text, doc_facts.entity_value_raw, doc_facts.evidence_text, and doc_paragraphs.paragraph_text exactly as aligned in semantic_plan; do not replace document evidence with unrelated same-named values.",
    "Do not use SQL LIMIT to decide the final number of answer rows unless the semantic_plan explicitly contains a structured row-count/top-N requirement. For min/max/lowest/highest selection, preserve all tied records by filtering on the selected value instead of using ORDER BY ... LIMIT 1.",
    "When a semantic_plan answer_column contains calculation instead of source_field, execute that complete calculation expression as a requested output metric.",
    "Execute value filters according to filters[].operator without restricting it to a fixed enum.",
    "Execute an IS_NULL filter with no value or calculation as a null-value filter on the specified field.",
    "When a filter value is a range object, interpret its lower and upper bounds according to the filter operator.",
    "When a semantic_plan filter contains calculation instead of value, execute it as a derived filter: compute the calculation in the current join/filter/group scope, then apply field operator calculation_result while preserving all tied records for MIN/MAX/lowest/highest semantics.",
    "All formula intent must come from answer_columns[].calculation or filters[].calculation; do not look for a separate formula field.",
    "Before calling answer, verify that answer rows contain only the values requested by semantic_plan answer_columns; do not include intermediate IDs, join keys, filters, or debugging columns unless answer_columns explicitly requests them.",
    "When producing answer rows, use the source_field or calculation aligned in semantic_plan answer_columns; do not substitute a semantically similar or same-named field that was not aligned there.",
    "When semantic_plan is missing a necessary source_field, calculation, filter field, join field, or formula expression field reference, do not guess from same-named DuckDB columns; inspect relevant context and proceed only when the semantic_plan can be executed without replacing its aligned fields.",
    "When the target is a set of entities or records such as people, customers, schools, sets, cards, transactions, or events, first form the distinct target entity/record set at the intended output level, then join or project requested output attributes from that set.",
    "After joining tables, check whether row counts expand unexpectedly. If they do, inspect whether the join used a broad descriptive key such as district, name, status, type, category, or country instead of a stable entity key such as an id, code, or documented foreign key.",
    "Return only the prediction table through the answer tool.",
]


def build_execution_initial_messages(
    *,
    understanding_payload: dict[str, Any],
    tool_descriptions: str,
    structured_store: StructuredContextStore | None,
) -> list[ModelMessage]:
    system_prompt = (
        f"{SEMANTIC_PLAN_EXECUTION_SYSTEM_PROMPT}\n\n"
        "Available actions:\n"
        f"{tool_descriptions}"
    )
    user_payload = {
        "semantic_plan": understanding_payload["semantic_plan"],
        "structured_context": (
            structured_store.inspect_schema() if structured_store is not None else {}
        ),
        "instructions": EXECUTION_INSTRUCTIONS,
    }
    return [
        ModelMessage(role="system", content=system_prompt),
        ModelMessage(
            role="user",
            content=json.dumps(user_payload, ensure_ascii=False, indent=2),
        ),
    ]
