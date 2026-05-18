from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from data_agent_baseline.agents.shared.utils import _json_compatible
from data_agent_baseline.agents.understanding.tools.candidate_store import (
    CandidateStore,
    FieldProfile,
    coerce_scalar,
)
from data_agent_baseline.agents.understanding.tools.hybrid_retrieval import (
    HybridRetriever,
    HybridRetrievalConfig,
    RetrievedContextItem,
)
from data_agent_baseline.agents.understanding.tools.join_search import RuleBasedJoinSearcher
from data_agent_baseline.agents.understanding.tools.normalization import (
    normalize_identifier,
    parse_colon_duration,
)
from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.tools.registry import ToolSpec

DEFAULT_SCHEMA_SEARCH_TOP_K = 10
VALUE_RESOLVER_DISTINCT_VALUE_LIMIT = 5000
DEFAULT_JOIN_SEARCH_TOP_K = 80
MAX_JOIN_PATH_DEPTH = 3
JOIN_PATH_TOP_K = 3
SUPPORTED_JOIN_TYPES = {"foreign_key", "shared_key", "value_overlap", "row_order"}
JOIN_TYPE_BASE_SCORES = {
    "foreign_key": 0.95,
    "shared_key": 0.72,
    "value_overlap": 0.5,
    "row_order": 0.2,
}
EDGE_QUALITY_RANK = {"weak": 1, "medium": 2, "strong": 3}
RISK_RANK = {"low": 0, "medium": 1, "high": 2}
MULTIPLICITY_RANK = {
    "one_to_one": 0,
    "many_to_one": 1,
    "one_to_many": 1,
    "many_to_many": 2,
    "unknown": 3,
}
FIELD_REFERENCE_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*\b")
SQL_LIKE_CALCULATION_RE = re.compile(
    r"\b(SELECT|FROM|WHERE|JOIN|CASE|WHEN|THEN|ELSE|END|IN|EXISTS|OVER|PARTITION|"
    r"GROUP\s+BY|ORDER\s+BY|HAVING|UNION)\b",
    re.IGNORECASE,
)
QUESTION_PHRASE_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


@dataclass(frozen=True, slots=True)
class UnderstandingToolResult:
    content: dict[str, Any]
    terminal: bool = False
    payload: dict[str, Any] | None = None


class UnderstandingToolRegistry:
    """Stateful schema-understanding tools exposed to QuestionUnderstandAgent."""

    def __init__(
        self,
        *,
        task: PublicTask,
        candidate_store: CandidateStore,
        knowledge_context: str = "",
        doc_evidence_context: str = "",
    ) -> None:
        self.task = task
        self.original_question = str(task.question or "")
        self.knowledge_context = str(knowledge_context or "")
        self.doc_evidence_context = str(doc_evidence_context or "")
        self.candidate_store = candidate_store
        self.database_name = task.context_dir.name or task.task_id
        self.retrieval_config = HybridRetrievalConfig(
            final_top_k=DEFAULT_SCHEMA_SEARCH_TOP_K,
            bm25_top_k=50,
            embedding_top_k=50,
        )
        self.hybrid_retriever = HybridRetriever(config=self.retrieval_config)
        self.join_searcher = RuleBasedJoinSearcher(top_k=DEFAULT_JOIN_SEARCH_TOP_K)
        self._join_candidates = None
        self._returned_join_edge_types: dict[tuple[str, str], set[str]] = {}
        self._returned_join_edges: dict[tuple[str, str], list[dict[str, Any]]] = {}
        self._returned_join_paths: list[dict[str, Any]] = []
        self._evidence_by_id: dict[str, dict[str, Any]] = {}
        self.schema_searches: list[dict[str, Any]] = []
        self.value_resolutions: list[dict[str, Any]] = []
        self.table_neighbor_calls: list[dict[str, Any]] = []
        self.specs = self._build_specs()

    def describe_for_prompt(self) -> str:
        lines = []
        for name in sorted(self.specs):
            spec = self.specs[name]
            lines.append(f"- {spec.name}: {spec.description}")
            lines.append(f"  input_schema: {spec.input_schema}")
        return "\n".join(lines)

    def _build_specs(self) -> dict[str, ToolSpec]:
        answer_column_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "meaning": {"type": "string"},
                "source_field": {"type": "string"},
                "calculation": {
                    "type": "string",
                    "description": "Direct calculation expression using exact table.column fields.",
                },
            },
            "required": ["name", "meaning"],
            "additionalProperties": False,
        }
        filter_schema = {
            "type": "object",
            "properties": {
                "field": {"type": "string"},
                "operator": {"type": "string"},
                "value": {"description": "Grounded filter value; may be scalar, array, or null."},
                "calculation": {
                    "description": "Derived calculation expression using exact table.column fields.",
                },
            },
            "required": ["field", "operator"],
            "additionalProperties": False,
        }
        join_schema = {
            "type": "object",
            "properties": {
                "left": {"type": "string"},
                "operator": {"type": "string"},
                "right": {"type": "string"},
                "join_type": {"type": "string", "enum": sorted(SUPPORTED_JOIN_TYPES)},
            },
            "required": ["left", "operator", "right"],
            "additionalProperties": False,
        }
        return {
            "semantic_schema_search": ToolSpec(
                name="semantic_schema_search",
                description=(
                    "Search the database schema for schema concept phrases. "
                    "Initial queries must be exact phrases extracted from the original user "
                    "question. Follow-up operand queries may be exact phrases copied from the "
                    "provided knowledge context evidence or targeted document evidence. Do not "
                    "rewrite, expand, normalize, translate, generalize, invent query text, or "
                    "use schema-derived table names or field names."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "queries": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["queries"],
                    "additionalProperties": False,
                },
            ),
            "get_table_neighbors": ToolSpec(
                name="get_table_neighbors",
                description="Return verifiable join paths for tables that need to be connected.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "tables": {"type": "array", "items": {"type": "string"}},
                        "required_columns": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["tables"],
                    "additionalProperties": False,
                },
            ),
            "value_resolver": ToolSpec(
                name="value_resolver",
                description=(
                    "Validate and normalize a raw filter value for one selected database "
                    "schema field. selected_column must be a schema field in table.column "
                    "form. This tool does not find schema fields or join paths."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "selected_column": {
                            "type": "string",
                            "description": "Database schema field in table.column form.",
                        },
                        "value": {},
                    },
                    "required": ["selected_column", "value"],
                    "additionalProperties": False,
                },
            ),
            "finalize_understanding": ToolSpec(
                name="finalize_understanding",
                description="Submit the final semantic plan.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "semantic_plan": {
                            "type": "object",
                            "properties": {
                                "answer_columns": {
                                    "type": "array",
                                    "items": answer_column_schema,
                                },
                                "output_grain": {"type": "string"},
                                "filters": {
                                    "type": "array",
                                    "items": filter_schema,
                                    "minItems": 1,
                                },
                                "joins": {
                                    "type": "array",
                                    "items": join_schema,
                                },
                                "group_by": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": [
                                "answer_columns",
                                "output_grain",
                                "filters",
                                "joins",
                                "group_by",
                            ],
                            "additionalProperties": False,
                        },
                    },
                    "required": ["semantic_plan"],
                    "additionalProperties": False,
                },
            ),
        }

    def execute(self, *, action: str, action_input: dict[str, Any]) -> UnderstandingToolResult:
        if action == "semantic_schema_search":
            return self._semantic_schema_search(action_input)
        if action == "get_table_neighbors":
            return self._get_table_neighbors(action_input)
        if action == "value_resolver":
            return self._value_resolver(action_input)
        if action == "finalize_understanding":
            return self._finalize_understanding(action_input)
        raise KeyError(f"Unknown understanding tool: {action}")

    def _semantic_schema_search(self, action_input: dict[str, Any]) -> UnderstandingToolResult:
        extra_keys = sorted(set(action_input) - {"queries"})
        if extra_keys:
            raise ValueError(
                "semantic_schema_search action_input contains unsupported keys: "
                + ", ".join(extra_keys)
            )
        queries = self._schema_search_queries(action_input.get("queries"))
        query_sources = {}
        for query in queries:
            query_sources[query] = self._validate_schema_search_query(query)

        content = {}
        for query in queries:
            result = self._run_schema_search(
                query,
                top_k=DEFAULT_SCHEMA_SEARCH_TOP_K,
                query_source=query_sources[query],
            )
            content[query] = {"candidates": result["candidates"]}
        return UnderstandingToolResult(content=content)

    def _schema_search_queries(self, value: Any) -> list[str]:
        if not isinstance(value, list) or not value:
            raise ValueError("queries must be a non-empty array.")
        queries: list[str] = []
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("queries must contain only non-empty strings.")
            query = item.strip()
            if query not in queries:
                queries.append(query)
        return queries

    def _validate_schema_search_query(self, query: str) -> str:
        question_tokens = self._question_phrase_tokens(self.original_question)
        knowledge_context_tokens = self._question_phrase_tokens(self.knowledge_context)
        doc_evidence_context_tokens = self._question_phrase_tokens(self.doc_evidence_context)
        query_tokens = self._question_phrase_tokens(query)
        if not query_tokens:
            raise ValueError(
                "semantic_schema_search.queries item must include a phrase from the original "
                "question, provided knowledge context evidence, or targeted document evidence."
            )
        if self._contains_question_phrase(question_tokens, query_tokens):
            return "question"
        if self._contains_question_phrase(knowledge_context_tokens, query_tokens):
            return "knowledge_context"
        if self._contains_question_phrase(doc_evidence_context_tokens, query_tokens):
            return "doc_evidence_context"
        raise ValueError(
            "semantic_schema_search.queries item must be a phrase from the original question, "
            "provided knowledge context evidence, or targeted document evidence. Do not use "
            "rewritten, expanded, normalized, translated, generalized, schema-derived, or "
            "invented query text."
        )

    def _contains_question_phrase(
        self,
        question_tokens: list[str],
        query_tokens: list[str],
    ) -> bool:
        if not question_tokens or len(query_tokens) > len(question_tokens):
            return False
        phrase_length = len(query_tokens)
        for start in range(0, len(question_tokens) - phrase_length + 1):
            if question_tokens[start : start + phrase_length] == query_tokens:
                return True
        return False

    def _question_phrase_tokens(self, text: str) -> list[str]:
        return [match.group(0).lower() for match in QUESTION_PHRASE_TOKEN_RE.finditer(text)]

    def _get_table_neighbors(self, action_input: dict[str, Any]) -> UnderstandingToolResult:
        extra_keys = sorted(set(action_input) - {"tables", "required_columns"})
        if extra_keys:
            raise ValueError(
                "get_table_neighbors action_input contains unsupported keys: "
                + ", ".join(extra_keys)
            )
        requested_tables = self._string_list(action_input.get("tables"))
        resolved_tables = []
        for table in requested_tables:
            resolved = self._resolve_table(table)
            if resolved is None:
                raise ValueError(f"Unknown table: {table}.")
            if resolved not in resolved_tables:
                resolved_tables.append(resolved)
        if len(resolved_tables) < 2:
            raise ValueError("tables must contain at least two known tables.")

        required_columns = self._required_join_columns(action_input.get("required_columns"))
        paths = self._join_paths_for_tables(resolved_tables, required_columns=required_columns)
        call = {
            "tables": resolved_tables,
            "required_columns": required_columns,
            "returned_path_count": len(paths),
        }
        self.table_neighbor_calls.append(call)
        return UnderstandingToolResult(
            content={
                "tables": resolved_tables,
                "required_columns": required_columns,
                "paths": paths,
            }
        )

    def _required_join_columns(self, value: Any) -> list[str]:
        if value is None:
            return []
        columns = self._string_list(value)
        valid_fields = {
            profile.field
            for profile in self.candidate_store.fields
            if isinstance(profile.field, str) and profile.field
        }
        for column in columns:
            self._validate_schema_field(column, valid_fields, "get_table_neighbors.required_columns")
        return columns

    def _value_resolver(self, action_input: dict[str, Any]) -> UnderstandingToolResult:
        extra_keys = sorted(set(action_input) - {"selected_column", "value"})
        if extra_keys:
            raise ValueError(
                "value_resolver action_input contains unsupported keys: "
                + ", ".join(extra_keys)
            )
        selected_column = self._required_string(
            action_input.get("selected_column"),
            "value_resolver.selected_column",
        )
        if "value" not in action_input:
            raise ValueError("value_resolver.value is required.")
        profile = self.candidate_store.field_profile(selected_column)
        if profile is None:
            raise ValueError(f"Unknown selected_column: {selected_column}.")

        resolved = self._resolve_value_for_profile(
            profile=profile,
            raw_value=action_input.get("value"),
        )
        self.value_resolutions.append(resolved)
        return UnderstandingToolResult(content=resolved)

    def _finalize_understanding(self, action_input: dict[str, Any]) -> UnderstandingToolResult:
        semantic_plan = self._validate_semantic_plan(action_input.get("semantic_plan"))
        payload = {
            "semantic_plan": semantic_plan,
        }
        return UnderstandingToolResult(
            content={
                "status": "submitted",
                "semantic_plan": semantic_plan,
            },
            terminal=True,
            payload=payload,
        )

    def _validate_semantic_plan(self, value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError("semantic_plan must be a JSON object.")

        plan = dict(value)
        required_keys = {
            "answer_columns",
            "output_grain",
            "filters",
            "joins",
            "group_by",
        }
        missing_keys = sorted(required_keys - set(plan))
        if missing_keys:
            raise ValueError(f"semantic_plan is missing required keys: {', '.join(missing_keys)}.")
        extra_keys = sorted(set(plan) - required_keys)
        if extra_keys:
            raise ValueError(f"semantic_plan contains unsupported keys: {', '.join(extra_keys)}.")

        valid_fields = {
            profile.field
            for profile in self.candidate_store.fields
            if isinstance(profile.field, str) and profile.field
        }
        valid_column_names = {
            profile.name
            for profile in self.candidate_store.fields
            if isinstance(profile.name, str) and profile.name
        }
        required_plan_fields: set[str] = set()

        answer_columns = self._require_list(plan["answer_columns"], "answer_columns")
        if not answer_columns:
            raise ValueError("semantic_plan.answer_columns must be a non-empty array.")
        filters = self._require_list(plan["filters"], "filters")
        if not filters:
            raise ValueError("semantic_plan.filters must be a non-empty array.")
        joins = self._require_list(plan["joins"], "joins")
        group_by = self._require_list(plan["group_by"], "group_by")

        if not isinstance(plan["output_grain"], str):
            raise ValueError("semantic_plan.output_grain must be a string.")

        for index, answer_column in enumerate(answer_columns, start=1):
            if not isinstance(answer_column, dict):
                raise ValueError(f"semantic_plan.answer_columns[{index}] must be an object.")
            self._reject_unsupported_keys(
                answer_column,
                {"name", "meaning", "source_field", "calculation"},
                f"answer_columns[{index}]",
            )
            self._required_string(answer_column.get("name"), f"answer_columns[{index}].name")
            if not isinstance(answer_column.get("meaning"), str):
                raise ValueError(f"semantic_plan.answer_columns[{index}].meaning must be a string.")
            has_source_field = "source_field" in answer_column and answer_column.get("source_field") is not None
            has_calculation = "calculation" in answer_column and answer_column.get("calculation") is not None
            if has_source_field == has_calculation:
                raise ValueError(
                    f"semantic_plan.answer_columns[{index}] must include exactly one of source_field or calculation."
                )
            if has_source_field:
                source_field = self._required_string(
                    answer_column.get("source_field"),
                    f"answer_columns[{index}].source_field",
                )
                self._validate_schema_field(
                    source_field,
                    valid_fields,
                    f"answer_columns[{index}].source_field",
                )
                required_plan_fields.add(source_field)
            if has_calculation:
                calculation = self._required_string(
                    answer_column.get("calculation"),
                    f"answer_columns[{index}].calculation",
                )
                self._validate_direct_calculation_expression(
                    calculation,
                    valid_fields,
                    valid_column_names,
                    f"answer_columns[{index}]",
                )
                required_plan_fields.update(FIELD_REFERENCE_RE.findall(calculation))

        for index, filter_item in enumerate(filters, start=1):
            if not isinstance(filter_item, dict):
                raise ValueError(f"semantic_plan.filters[{index}] must be an object.")
            self._reject_unsupported_keys(
                filter_item,
                {"field", "operator", "value", "calculation"},
                f"filters[{index}]",
            )
            field = self._required_string(filter_item.get("field"), f"filters[{index}].field")
            operator = self._required_string(
                filter_item.get("operator"),
                f"filters[{index}].operator",
            )
            normalized_operator = operator.strip().upper()
            has_value = "value" in filter_item
            has_calculation = "calculation" in filter_item
            if normalized_operator in {"IS_NOT_NULL", "IS NOT NULL"}:
                raise ValueError(
                    f"semantic_plan.filters[{index}].operator IS_NOT_NULL is unsupported. "
                    "For extrema selection, use calculation."
                )
            is_null_filter = normalized_operator == "IS_NULL"
            if is_null_filter:
                if has_value or has_calculation:
                    raise ValueError(
                        f"semantic_plan.filters[{index}] with IS_NULL must not include value "
                        "or calculation."
                    )
            elif has_value and has_calculation:
                raise ValueError(
                    f"semantic_plan.filters[{index}] must include exactly one of value or calculation."
                )
            elif not has_value and not has_calculation:
                raise ValueError(
                    f"semantic_plan.filters[{index}] must include value or calculation; "
                    "IS_NOT_NULL is unsupported. For extrema selection, use calculation."
                )
            self._validate_schema_field(field, valid_fields, f"filters[{index}].field")
            required_plan_fields.add(field)
            if has_value and not is_null_filter:
                self._validate_filter_value_resolution(
                    filter_item=filter_item,
                    field=field,
                    operator=operator,
                    path=f"filters[{index}]",
                )
            if has_calculation:
                self._validate_direct_calculation_expression(
                    filter_item.get("calculation"),
                    valid_fields,
                    valid_column_names,
                    f"filters[{index}]",
                )
                required_plan_fields.update(
                    FIELD_REFERENCE_RE.findall(str(filter_item.get("calculation")))
                )

        for index, join in enumerate(joins, start=1):
            if not isinstance(join, dict):
                raise ValueError(f"semantic_plan.joins[{index}] must be an object.")
            self._reject_unsupported_keys(
                join,
                {"left", "operator", "right", "join_type"},
                f"joins[{index}]",
            )
            left = self._required_string(join.get("left"), f"joins[{index}].left")
            operator = self._required_string(join.get("operator"), f"joins[{index}].operator")
            right = self._required_string(join.get("right"), f"joins[{index}].right")
            join_type = join.get("join_type")
            if join_type is not None and not isinstance(join_type, str):
                raise ValueError(f"semantic_plan.joins[{index}].join_type must be a string.")
            self._validate_schema_field(left, valid_fields, f"joins[{index}].left")
            self._validate_schema_field(right, valid_fields, f"joins[{index}].right")
            self._validate_join_evidence(
                left=left,
                right=right,
                operator=operator,
                join_type=join_type,
                path=f"joins[{index}]",
            )

        for index, field in enumerate(group_by, start=1):
            field_name = self._required_string(field, f"group_by[{index}]")
            self._validate_schema_field(field_name, valid_fields, f"group_by[{index}]")
            required_plan_fields.add(field_name)

        self._validate_join_structure(joins, required_plan_fields)
        return _json_compatible(plan)

    def _reject_unsupported_keys(
        self,
        value: dict[str, Any],
        allowed_keys: set[str],
        path: str,
    ) -> None:
        extra_keys = sorted(set(value) - allowed_keys)
        if extra_keys:
            raise ValueError(
                f"semantic_plan.{path} contains unsupported keys: {', '.join(extra_keys)}."
            )

    def _require_list(self, value: Any, path: str) -> list[Any]:
        if not isinstance(value, list):
            raise ValueError(f"semantic_plan.{path} must be an array.")
        return value

    def _required_string(self, value: Any, path: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"semantic_plan.{path} must be a non-empty string.")
        return value.strip()

    def _validate_calculation_expression(
        self,
        expression: str,
        valid_fields: set[str],
        valid_column_names: set[str],
        path: str,
    ) -> None:
        sql_like_match = SQL_LIKE_CALCULATION_RE.search(expression)
        if sql_like_match:
            token = sql_like_match.group(1).upper()
            raise ValueError(
                f"semantic_plan.{path} contains SQL query syntax `{token}`. "
                "Calculation expressions must be semantic formulas over exact table.column "
                "fields; put row selection in filters and relationships in joins."
            )

        for field in FIELD_REFERENCE_RE.findall(expression):
            self._validate_schema_field(field, valid_fields, path)

        for column_name in valid_column_names:
            pattern = re.compile(rf"(?<![\w.]){re.escape(column_name)}(?![\w.])")
            if pattern.search(expression):
                raise ValueError(
                    f"semantic_plan.{path} uses bare column name {column_name}; "
                    "use exact table.column field references in calculation expressions."
                )

    def _validate_direct_calculation_expression(
        self,
        value: Any,
        valid_fields: set[str],
        valid_column_names: set[str],
        path: str,
    ) -> None:
        expression = self._required_string(value, f"{path}.calculation")
        self._validate_calculation_expression(
            expression,
            valid_fields,
            valid_column_names,
            f"{path}.calculation",
        )
        if not FIELD_REFERENCE_RE.findall(expression):
            raise ValueError(
                f"semantic_plan.{path}.calculation must reference at least one "
                "schema field in table.column form."
            )

    def _validate_schema_field(self, field: str, valid_fields: set[str], path: str) -> None:
        if field not in valid_fields:
            raise ValueError(f"semantic_plan.{path} references unknown schema field: {field}.")

    def _validate_filter_value_resolution(
        self,
        *,
        filter_item: dict[str, Any],
        field: str,
        operator: str,
        path: str,
    ) -> None:
        value = filter_item.get("value")
        if value is None and operator.strip().upper() in {"IS NULL", "IS NOT NULL"}:
            return
        if value is None:
            return
        if any(
            self._value_resolution_matches(
                resolution,
                field=field,
                value=value,
            )
            for resolution in self.value_resolutions
            if resolution.get("status") == "resolved"
        ):
            return
        raise ValueError(
            f"semantic_plan.{path} must use a value returned by value_resolver."
        )

    def _value_resolution_matches(
        self,
        resolution: dict[str, Any],
        *,
        field: str,
        value: Any,
    ) -> bool:
        resolved_field = self._string_or_none(resolution.get("selected_column"))
        if resolved_field != field:
            return False
        final_value = _json_compatible(value)
        return final_value == _json_compatible(resolution.get("resolved_value"))

    def _validate_join_evidence(
        self,
        *,
        left: str,
        right: str,
        operator: str,
        join_type: Any,
        path: str,
    ) -> None:
        if operator.strip() != "=":
            raise ValueError(f"semantic_plan.{path}.operator must be `=` for join_search evidence.")
        normalized_join_type = join_type.strip() if isinstance(join_type, str) else None
        if normalized_join_type is not None and normalized_join_type not in SUPPORTED_JOIN_TYPES:
            allowed = ", ".join(sorted(SUPPORTED_JOIN_TYPES))
            raise ValueError(
                f"semantic_plan.{path}.join_type must be one of: {allowed}."
            )
        edge_key = self._join_edge_key(left, right)
        supported_types = self._returned_join_edge_types.get(edge_key)
        if not supported_types:
            raise ValueError(
                f"semantic_plan.{path} join `{left} = {right}` is not in returned "
                "get_table_neighbors paths."
            )
        if normalized_join_type is not None and normalized_join_type not in supported_types:
            allowed = ", ".join(sorted(supported_types))
            raise ValueError(
                f"semantic_plan.{path}.join_type `{normalized_join_type}` does not match "
                f"returned join evidence type: {allowed}."
            )
        returned_steps = self._returned_join_edges.get(edge_key, [])
        if self._uses_unmarked_weak_join(returned_steps, normalized_join_type):
            raise ValueError(
                f"semantic_plan.{path} join `{left} = {right}` uses row_order or "
                "weak_value_overlap without returned warning or requires_verification marker."
            )

    def _uses_unmarked_weak_join(
        self,
        returned_steps: list[dict[str, Any]],
        join_type: str | None,
    ) -> bool:
        for step in returned_steps:
            step_join_type = self._string_or_none(step.get("join_type"))
            if join_type is not None and step_join_type != join_type:
                continue
            weak_step = (
                step_join_type == "row_order"
                or step.get("overlap_tier") == "weak_value_overlap"
            )
            if not weak_step:
                continue
            has_warning = isinstance(step.get("warning"), str) and bool(str(step["warning"]).strip())
            if not has_warning and step.get("requires_verification") is not True:
                return True
        return False

    def _validate_join_structure(
        self,
        joins: list[Any],
        required_plan_fields: set[str],
    ) -> None:
        required_tables = self._tables_for_fields(required_plan_fields)
        if len(required_tables) <= 1:
            return
        if not joins:
            raise ValueError(
                "semantic_plan.joins must connect all required tables: "
                + ", ".join(sorted(required_tables))
            )
        selected_edge_keys: set[tuple[str, str]] = set()
        selected_tables: set[str] = set()
        graph: dict[str, set[str]] = {}
        for join in joins:
            if not isinstance(join, dict):
                continue
            left = self._string_or_none(join.get("left"))
            right = self._string_or_none(join.get("right"))
            if left is None or right is None:
                continue
            left_table = self._field_table(left)
            right_table = self._field_table(right)
            if left_table is None or right_table is None:
                continue
            selected_edge_keys.add(self._join_edge_key(left, right))
            selected_tables.update({left_table, right_table})
            graph.setdefault(left_table, set()).add(right_table)
            graph.setdefault(right_table, set()).add(left_table)
        missing_tables = sorted(required_tables - selected_tables)
        if missing_tables:
            raise ValueError(
                "semantic_plan.joins selected path lacks required table coverage: "
                + ", ".join(missing_tables)
            )
        start = next(iter(required_tables))
        visited = self._connected_tables(graph, start)
        disconnected = sorted(required_tables - visited)
        if disconnected:
            raise ValueError(
                "semantic_plan.joins do not connect required tables: "
                + ", ".join(disconnected)
            )
        self._validate_selected_join_paths_match_returned_paths(joins)
        self._validate_selected_avoid_paths(selected_edge_keys, required_tables)

    def _validate_selected_join_paths_match_returned_paths(self, joins: list[Any]) -> None:
        selected_joins = [join for join in joins if isinstance(join, dict)]
        if not selected_joins:
            return

        paths_by_pair: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for path in self._returned_join_paths:
            pair_key = self._join_path_pair_key(path)
            if pair_key is None:
                continue
            if not self._join_path_signature(path):
                continue
            paths_by_pair.setdefault(pair_key, []).append(path)

        for pair_key, paths in paths_by_pair.items():
            path_signatures = [self._join_path_signature(path) for path in paths]
            selected_for_pair = [
                join
                for join in selected_joins
                if self._join_matches_any_path(join, path_signatures)
            ]
            if not selected_for_pair:
                continue
            if any(
                self._selected_joins_are_covered_by_path(selected_for_pair, path_signature)
                for path_signature in path_signatures
            ):
                continue
            table_pair = "-".join(pair_key)
            selected_text = "; ".join(
                self._format_join_step(join) for join in selected_for_pair
            )
            path_options = "; ".join(
                self._format_join_path_option(index, path)
                for index, path in enumerate(paths, start=1)
            )
            raise ValueError(
                "semantic_plan.joins combines multiple alternative get_table_neighbors "
                f"paths for table pair {table_pair}. Candidate paths returned for the same "
                "source/target table pair are alternatives; choose one returned path. "
                f"Selected joins: {selected_text}. Returned path options: {path_options}."
            )

    def _join_matches_any_path(
        self,
        join: dict[str, Any],
        path_signatures: list[set[tuple[tuple[str, str], str | None]]],
    ) -> bool:
        signature = self._join_edge_signature(join)
        if signature is None:
            return False
        return any(
            self._signature_matches_returned(signature, returned_signature)
            for path_signature in path_signatures
            for returned_signature in path_signature
        )

    def _selected_joins_are_covered_by_path(
        self,
        joins: list[dict[str, Any]],
        path_signature: set[tuple[tuple[str, str], str | None]],
    ) -> bool:
        for join in joins:
            signature = self._join_edge_signature(join)
            if signature is None:
                return False
            if not any(
                self._signature_matches_returned(signature, returned_signature)
                for returned_signature in path_signature
            ):
                return False
        return True

    def _signature_matches_returned(
        self,
        selected: tuple[tuple[str, str], str | None],
        returned: tuple[tuple[str, str], str | None],
    ) -> bool:
        selected_edge, selected_type = selected
        returned_edge, returned_type = returned
        return selected_edge == returned_edge and (
            selected_type is None or selected_type == returned_type
        )

    def _join_edge_signature(
        self,
        join_or_step: dict[str, Any],
    ) -> tuple[tuple[str, str], str | None] | None:
        left = self._string_or_none(join_or_step.get("left"))
        right = self._string_or_none(join_or_step.get("right"))
        if left is None or right is None:
            return None
        return self._join_edge_key(left, right), self._string_or_none(join_or_step.get("join_type"))

    def _join_path_signature(
        self,
        path: dict[str, Any],
    ) -> set[tuple[tuple[str, str], str | None]]:
        steps = path.get("steps")
        if not isinstance(steps, list):
            return set()
        signatures: set[tuple[tuple[str, str], str | None]] = set()
        for step in steps:
            if not isinstance(step, dict):
                continue
            signature = self._join_edge_signature(step)
            if signature is not None:
                signatures.add(signature)
        return signatures

    def _join_path_pair_key(self, path: dict[str, Any]) -> tuple[str, str] | None:
        from_table = self._string_or_none(path.get("from_table"))
        to_table = self._string_or_none(path.get("to_table"))
        if from_table is None or to_table is None:
            return None
        return tuple(sorted([from_table, to_table]))

    def _format_join_path_option(self, index: int, path: dict[str, Any]) -> str:
        steps = path.get("steps")
        if not isinstance(steps, list):
            return f"path {index}: <empty>"
        step_text = " -> ".join(
            self._format_join_step(step)
            for step in steps
            if isinstance(step, dict)
        )
        return f"path {index}: {step_text or '<empty>'}"

    def _format_join_step(self, join_or_step: dict[str, Any]) -> str:
        left = self._string_or_none(join_or_step.get("left")) or "?"
        right = self._string_or_none(join_or_step.get("right")) or "?"
        join_type = self._string_or_none(join_or_step.get("join_type")) or "unspecified"
        return f"{left} = {right} ({join_type})"

    def _validate_selected_avoid_paths(
        self,
        selected_edge_keys: set[tuple[str, str]],
        required_tables: set[str],
    ) -> None:
        for path in self._returned_join_paths:
            path_edge_keys = self._path_edge_keys(path)
            if not path_edge_keys or path_edge_keys != selected_edge_keys:
                continue
            summary = path.get("summary")
            if not isinstance(summary, dict):
                continue
            if summary.get("structural_recommendation") != "avoid_if_alternative":
                continue
            if self._has_safer_equivalent_path(path, required_tables):
                raise ValueError(
                    "semantic_plan.joins selected path has high fanout risk and a safer "
                    "candidate path with the same required table coverage was returned. "
                    "Retry by selecting a lower-risk returned join path, or verify whether "
                    "this broad-field join is intended."
                )

    def _has_safer_equivalent_path(
        self,
        selected_path: dict[str, Any],
        required_tables: set[str],
    ) -> bool:
        selected_summary = selected_path.get("summary")
        if not isinstance(selected_summary, dict):
            return False
        selected_risk = RISK_RANK.get(str(selected_summary.get("structural_risk_level")), 3)
        selected_required_coverage = set(selected_summary.get("required_tables_covered") or [])
        selected_entities = self._path_entity_aliases(selected_path)
        for candidate in self._returned_join_paths:
            if candidate is selected_path:
                continue
            if candidate.get("from_table") != selected_path.get("from_table"):
                continue
            if candidate.get("to_table") != selected_path.get("to_table"):
                continue
            summary = candidate.get("summary")
            if not isinstance(summary, dict):
                continue
            if set(summary.get("required_tables_covered") or []) != selected_required_coverage:
                continue
            if required_tables and set(summary.get("required_tables_covered") or []) != required_tables:
                continue
            candidate_risk = RISK_RANK.get(str(summary.get("structural_risk_level")), 3)
            if candidate_risk >= selected_risk:
                continue
            candidate_entities = self._path_entity_aliases(candidate)
            if selected_entities and candidate_entities and not (selected_entities & candidate_entities):
                continue
            return True
        return False

    def _path_edge_keys(self, path: dict[str, Any]) -> set[tuple[str, str]]:
        steps = path.get("steps")
        if not isinstance(steps, list):
            return set()
        keys: set[tuple[str, str]] = set()
        for step in steps:
            if not isinstance(step, dict):
                continue
            left = self._string_or_none(step.get("left"))
            right = self._string_or_none(step.get("right"))
            if left is not None and right is not None:
                keys.add(self._join_edge_key(left, right))
        return keys

    def _path_entity_aliases(self, path: dict[str, Any]) -> set[str]:
        aliases: set[str] = set()
        steps = path.get("steps")
        if not isinstance(steps, list):
            return aliases
        for step in steps:
            if not isinstance(step, dict):
                continue
            for field in [step.get("left"), step.get("right")]:
                text = self._string_or_none(field)
                if text is None:
                    continue
                profile = self.candidate_store.field_profile(text)
                if profile is not None:
                    aliases.update(self._field_entity_aliases(profile))
                aliases.update(self._field_reference_aliases(text))
        return aliases

    def _field_reference_aliases(self, field: str) -> set[str]:
        if "." not in field:
            return set()
        table, column = field.split(".", 1)
        aliases = set(normalize_identifier(table).split("_"))
        column_tokens = set(normalize_identifier(column).split("_"))
        aliases.update(column_tokens - {"id", "key", "code"})
        return {alias for alias in aliases if alias}

    def _field_entity_aliases(self, profile: FieldProfile) -> set[str]:
        aliases: set[str] = set()
        if profile.table:
            aliases.update(normalize_identifier(profile.table).split("_"))
        name_tokens = set(normalize_identifier(profile.name).split("_"))
        aliases.update(name_tokens - {"id", "key", "code"})
        return {alias for alias in aliases if alias}

    def _connected_tables(self, graph: dict[str, set[str]], start: str) -> set[str]:
        visited: set[str] = set()
        frontier = [start]
        while frontier:
            table = frontier.pop()
            if table in visited:
                continue
            visited.add(table)
            frontier.extend(sorted(graph.get(table, set()) - visited))
        return visited

    def _run_schema_search(
        self,
        query: str,
        *,
        top_k: int,
        query_source: str,
    ) -> dict[str, Any]:
        result = self.hybrid_retriever.retrieve(
            task=self.task,
            question=query,
            candidate_store=self.candidate_store,
            final_top_k=top_k,
        )
        candidates = [self._candidate_from_item(item) for item in list(result.items)[:top_k]]
        for candidate in candidates:
            self._remember_evidence(candidate["id"], candidate)
        payload = {
            "query": query,
            "top_k": top_k,
            "database": self.database_name,
            "candidates": candidates,
        }
        self.schema_searches.append(
            {
                "query": query,
                "query_source": query_source,
                "top_k": top_k,
                "returned_ids": [item["id"] for item in candidates],
            }
        )
        return payload

    def _candidate_from_item(self, item: RetrievedContextItem) -> dict[str, Any]:
        metadata = dict(item.metadata)
        payload: dict[str, Any] = {
            "id": item.id,
            "kind": item.source_type,
            "database": self.database_name,
            "source": item.source,
            "text": item.text,
            "matched_terms": list(item.matched_terms),
            "metadata": metadata,
        }
        field = metadata.get("field")
        table = metadata.get("table")
        column = metadata.get("column")
        if isinstance(field, str):
            payload["field"] = field
        if isinstance(table, str):
            payload["table"] = table
        if isinstance(column, str):
            payload["column"] = column
        if item.source_type == "table_profile":
            payload["table"] = metadata.get("table")
            payload["columns"] = metadata.get("columns", [])
        return payload

    def _resolve_value_for_profile(
        self,
        *,
        profile: FieldProfile,
        raw_value: Any,
    ) -> dict[str, Any]:
        structured_store = self.candidate_store.structured_store
        exact_matches: list[str] = []
        if structured_store is not None:
            exact_matches = list(
                structured_store.match_value(
                    profile.field,
                    raw_value,
                    limit=20,
                ).get("matches", [])
            )
        if exact_matches:
            matched_value = coerce_scalar(exact_matches[0])
            return self._resolved_value_payload(
                profile=profile,
                raw_value=raw_value,
                status="resolved",
                match_type="exact_table_value",
                resolved_value=matched_value,
                matches=exact_matches,
                evidence=[
                    f"exact normalized value match `{raw_value}` in table data for `{profile.field}`"
                ],
            )

        duration_payload = self._resolve_colon_duration_value(
            profile=profile,
            raw_value=raw_value,
        )
        if duration_payload is not None:
            return duration_payload

        range_payload = self._resolve_range_literal_value(
            profile=profile,
            raw_value=raw_value,
        )
        if range_payload is not None:
            return range_payload

        match = self.candidate_store.match_filter_value(
            raw_value=str(raw_value),
            operator="=",
            profile=profile,
            use_sample_values=structured_store is None,
        )
        if match is not None:
            resolved_value = self._resolved_value_from_match(match.candidate_value)
            return self._resolved_value_payload(
                profile=profile,
                raw_value=raw_value,
                status="resolved",
                match_type=match.match_type,
                resolved_value=resolved_value,
                matches=self._matches_from_candidate_value(match.candidate_value),
                evidence=list(match.evidence),
            )

        numeric_payload = self._resolve_numeric_literal_value(
            profile=profile,
            raw_value=raw_value,
        )
        if numeric_payload is not None:
            return numeric_payload

        return self._resolved_value_payload(
            profile=profile,
            raw_value=raw_value,
            status="unresolved",
            match_type="no_value_match",
            resolved_value=None,
            matches=[],
            evidence=[
                f"value `{raw_value}` was not found or normalized for `{profile.field}`"
            ],
        )

    def _resolve_colon_duration_value(
        self,
        *,
        profile: FieldProfile,
        raw_value: Any,
    ) -> dict[str, Any] | None:
        raw_duration = parse_colon_duration(raw_value)
        if raw_duration is None:
            return None

        candidate_values = self._value_candidates_for_profile(profile)
        matched_values: list[str] = []
        prefix_counts: dict[str, int] = {}
        for candidate in candidate_values:
            candidate_duration = parse_colon_duration(candidate)
            if candidate_duration is None:
                continue
            if candidate_duration.second_bucket != raw_duration.second_bucket:
                continue
            candidate_text = str(candidate)
            matched_values.append(candidate_text)
            prefix_counts[candidate_duration.second_prefix] = (
                prefix_counts.get(candidate_duration.second_prefix, 0) + 1
            )

        if not matched_values or not prefix_counts:
            return None

        prefix = sorted(prefix_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        normalized_value = f"{prefix}%"
        return self._resolved_value_payload(
            profile=profile,
            raw_value=raw_value,
            status="resolved",
            match_type="duration_prefix_match",
            resolved_value=normalized_value,
            matches=matched_values[:10],
            evidence=[
                f"duration value `{raw_value}` maps to second bucket "
                f"{raw_duration.second_bucket}",
                f"`{profile.field}` has values with duration prefix `{prefix}`",
            ],
        )

    def _value_candidates_for_profile(self, profile: FieldProfile) -> list[str]:
        structured_store = self.candidate_store.structured_store
        if structured_store is not None:
            return structured_store.distinct_values(
                profile.field,
                limit=VALUE_RESOLVER_DISTINCT_VALUE_LIMIT,
            )
        return list(profile.sample_values)

    def _resolve_range_literal_value(
        self,
        *,
        profile: FieldProfile,
        raw_value: Any,
    ) -> dict[str, Any] | None:
        if not isinstance(raw_value, dict):
            return None
        if set(raw_value) != {"lower", "upper"}:
            return None
        lower = raw_value.get("lower")
        upper = raw_value.get("upper")
        if profile.data_type in {"number", "integer"}:
            lower_number = self._parse_literal_number(lower)
            upper_number = self._parse_literal_number(upper)
            if lower_number is None or upper_number is None:
                return None
            if float(lower_number) > float(upper_number):
                lower_number, upper_number = upper_number, lower_number
            resolved_value = {"lower": lower_number, "upper": upper_number}
            return self._resolved_value_payload(
                profile=profile,
                raw_value=raw_value,
                status="resolved",
                match_type="range_literal",
                resolved_value=resolved_value,
                matches=[],
                evidence=[
                    f"range literal is compatible with numeric field `{profile.field}`"
                ],
            )
        if profile.data_type == "date":
            lower_text = self._string_or_none(lower)
            upper_text = self._string_or_none(upper)
            if lower_text is None or upper_text is None:
                return None
            resolved_value = {"lower": lower_text, "upper": upper_text}
            return self._resolved_value_payload(
                profile=profile,
                raw_value=raw_value,
                status="resolved",
                match_type="range_literal",
                resolved_value=resolved_value,
                matches=[],
                evidence=[
                    f"range literal is compatible with date-like field `{profile.field}`"
                ],
            )
        return None

    def _resolve_numeric_literal_value(
        self,
        *,
        profile: FieldProfile,
        raw_value: Any,
    ) -> dict[str, Any] | None:
        if profile.data_type not in {"number", "integer"}:
            return None
        number = self._parse_literal_number(raw_value)
        if number is None:
            return None
        return self._resolved_value_payload(
            profile=profile,
            raw_value=raw_value,
            status="resolved",
            match_type="numeric_literal",
            resolved_value=number,
            matches=[],
            evidence=[
                f"numeric literal is compatible with numeric field `{profile.field}`"
            ],
        )

    def _parse_literal_number(self, value: Any) -> int | float | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return value
        if not isinstance(value, str):
            return None
        text = value.strip().replace(",", "")
        if text.endswith("%"):
            text = text[:-1].strip()
        if re.fullmatch(r"[-+]?\d+", text):
            return int(text)
        if re.fullmatch(r"[-+]?\d+\.\d+", text):
            return float(text)
        return None

    def _resolved_value_from_match(self, candidate_value: Any) -> Any:
        if isinstance(candidate_value, dict):
            candidate_operator = self._string_or_none(candidate_value.get("operator"))
            if candidate_operator == "between":
                return {
                    "lower": candidate_value.get("lower"),
                    "upper": candidate_value.get("upper"),
                }
            if candidate_operator == "=" and "value" in candidate_value:
                return candidate_value.get("value")
        return candidate_value

    def _matches_from_candidate_value(self, candidate_value: Any) -> list[Any]:
        if candidate_value is None:
            return []
        if isinstance(candidate_value, dict):
            lower = candidate_value.get("lower")
            upper = candidate_value.get("upper")
            if lower is not None and upper is not None:
                return [lower, upper]
            value = candidate_value.get("value")
            return [] if value is None else [value]
        return [candidate_value]

    def _resolved_value_payload(
        self,
        *,
        profile: FieldProfile,
        raw_value: Any,
        status: str,
        match_type: str,
        resolved_value: Any,
        matches: list[Any],
        evidence: list[str],
    ) -> dict[str, Any]:
        return _json_compatible(
            {
                "selected_column": profile.field,
                "raw_value": raw_value,
                "status": status,
                "match_type": match_type,
                "resolved_value": resolved_value,
                "matches": matches,
                "evidence": evidence,
            }
        )

    def _join_candidates_list(self):
        if self._join_candidates is None:
            self._join_candidates = self.join_searcher.search(self.candidate_store)
        return self._join_candidates

    def _join_paths_for_tables(
        self,
        tables: list[str],
        *,
        required_columns: list[str],
    ) -> list[dict[str, Any]]:
        paths: list[dict[str, Any]] = []
        required_tables = self._tables_for_fields(required_columns)
        for left_index, left_table in enumerate(tables):
            for right_table in tables[left_index + 1 :]:
                candidate_paths = self._join_paths_between(
                    left_table,
                    right_table,
                    required_tables=required_tables,
                )
                if candidate_paths:
                    for path_payload in candidate_paths:
                        self._remember_returned_join_path(path_payload)
                        paths.append(path_payload)
                    continue
                paths.append(
                    {
                        "from_table": left_table,
                        "to_table": right_table,
                        "summary": self._empty_join_path_summary(required_tables),
                        "steps": [],
                        "reason": "no_join_path_found",
                    }
                )
        return paths

    def _join_paths_between(
        self,
        source_table: str,
        target_table: str,
        *,
        required_tables: set[str],
    ) -> list[dict[str, Any]]:
        graph = self._join_graph()
        candidate_paths: list[list[dict[str, Any]]] = []
        frontier: list[tuple[str, list[dict[str, Any]], set[str]]] = [
            (source_table, [], {source_table})
        ]
        while frontier:
            current_table, current_path, visited = frontier.pop(0)
            if current_table == target_table:
                if current_path:
                    candidate_paths.append(current_path)
                continue
            if len(current_path) >= MAX_JOIN_PATH_DEPTH:
                continue
            for edge in graph.get(current_table, []):
                neighbor_table = edge["neighbor_table"]
                if neighbor_table in visited:
                    continue
                frontier.append(
                    (
                        neighbor_table,
                        [*current_path, edge],
                        {*visited, neighbor_table},
                    )
                )
        ranked_paths = sorted(
            candidate_paths,
            key=lambda path: self._join_path_sort_key(path, required_tables),
        )
        ranked_path_payloads = [
            self._join_path_payload(
                source_table,
                target_table,
                path,
                required_tables=required_tables,
            )
            for path in ranked_paths
        ]
        return self._select_join_paths_for_llm(ranked_path_payloads)

    def _select_join_paths_for_llm(
        self,
        ranked_path_payloads: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not ranked_path_payloads:
            return []

        indexed_paths = list(enumerate(ranked_path_payloads))
        primary_paths = [
            (index, path)
            for index, path in indexed_paths
            if not self._is_fallback_join_path(path)
        ]
        if not primary_paths:
            return ranked_path_payloads[:JOIN_PATH_TOP_K]

        selected_indices = {
            index
            for index, _ in primary_paths[:JOIN_PATH_TOP_K]
        }
        if len(selected_indices) >= JOIN_PATH_TOP_K:
            return [
                path
                for index, path in indexed_paths
                if index in selected_indices
            ]

        selected_primary_coverage = max(
            self._join_path_required_coverage(path)
            for _, path in primary_paths[:JOIN_PATH_TOP_K]
        )
        for index, path in indexed_paths:
            if index in selected_indices:
                continue
            if not self._is_fallback_join_path(path):
                continue
            if self._join_path_required_coverage(path) <= selected_primary_coverage:
                continue
            selected_indices.add(index)
            if len(selected_indices) >= JOIN_PATH_TOP_K:
                break

        return [
            path
            for index, path in indexed_paths
            if index in selected_indices
        ]

    def _is_fallback_join_path(self, path_payload: dict[str, Any]) -> bool:
        summary = path_payload.get("summary")
        summary = summary if isinstance(summary, dict) else {}
        if self._positive_summary_count(summary, "row_order_count"):
            return True
        if self._positive_summary_count(summary, "weak_value_overlap_count"):
            return True
        if summary.get("has_broad_name_overlap") is True:
            return True
        steps = path_payload.get("steps")
        if not isinstance(steps, list):
            return False
        return any(
            isinstance(step, dict) and step.get("requires_verification") is True
            for step in steps
        )

    def _positive_summary_count(self, summary: dict[str, Any], key: str) -> bool:
        value = summary.get(key)
        return isinstance(value, (int, float)) and value > 0

    def _join_path_required_coverage(self, path_payload: dict[str, Any]) -> float:
        summary = path_payload.get("summary")
        summary = summary if isinstance(summary, dict) else {}
        value = summary.get("required_coverage")
        if isinstance(value, (int, float)):
            return float(value)
        return 0.0

    def _join_path_sort_key(
        self,
        path: list[dict[str, Any]],
        required_tables: set[str],
    ) -> tuple[int, float, float, int, int, int, int, int, str]:
        summary = self._join_path_summary(path, required_tables)
        serialized = "|".join(
            f"{edge['candidate'].get('left')}={edge['candidate'].get('right')}"
            for edge in path
        )
        return (
            -int(summary["min_edge_quality_rank"]),
            -int(summary["mean_edge_quality_rank"]),
            -float(summary["required_coverage"]),
            RISK_RANK.get(str(summary["max_fanout_risk"]), 3),
            RISK_RANK.get(str(summary["max_many_to_many_risk"]), 3),
            int(summary["row_order_count"]),
            int(summary["weak_value_overlap_count"]),
            len(path),
            serialized,
        )

    def _join_edge_score(self, edge: dict[str, Any]) -> float:
        candidate_payload = edge["candidate"]
        raw_score = edge.get("score")
        if isinstance(raw_score, (int, float)):
            return float(raw_score)
        join_type = candidate_payload.get("join_type")
        return JOIN_TYPE_BASE_SCORES.get(str(join_type), 0.0)

    def _join_edge_quality(self, candidate_payload: dict[str, Any]) -> str:
        join_type = str(candidate_payload.get("join_type") or "")
        metadata = candidate_payload.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        if join_type == "foreign_key":
            return "strong"
        if join_type == "shared_key":
            return "medium"
        if join_type == "row_order":
            return "weak"
        overlap_tier = metadata.get("overlap_tier")
        if overlap_tier == "key_overlap":
            return "strong"
        if overlap_tier == "dimension_name_overlap":
            return "medium"
        return "weak"

    def _join_path_payload(
        self,
        source_table: str,
        target_table: str,
        path: list[dict[str, Any]],
        *,
        required_tables: set[str],
    ) -> dict[str, Any]:
        steps = [self._join_step_payload(edge["candidate"]) for edge in path]
        summary = self._join_path_summary(path, required_tables, steps=steps)
        summary.pop("min_edge_quality_rank", None)
        summary.pop("mean_edge_quality_rank", None)
        return {
            "from_table": source_table,
            "to_table": target_table,
            "summary": summary,
            "steps": steps,
        }

    def _join_path_summary(
        self,
        path: list[dict[str, Any]],
        required_tables: set[str],
        *,
        steps: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        steps = steps if steps is not None else [
            self._join_step_payload(edge["candidate"]) for edge in path
        ]
        edge_qualities = [self._join_edge_quality(edge["candidate"]) for edge in path]
        quality_ranks = [EDGE_QUALITY_RANK.get(quality, 0) for quality in edge_qualities]
        min_rank = min(quality_ranks) if quality_ranks else 0
        mean_rank = (sum(quality_ranks) / len(quality_ranks)) if quality_ranks else 0.0
        tables_covered = sorted(
            {
                table
                for step in steps
                for table in [step.get("left_table"), step.get("right_table")]
                if isinstance(table, str) and table
            }
        )
        required_tables_covered = sorted(required_tables & set(tables_covered))
        missing_required_tables = sorted(required_tables - set(tables_covered))
        required_coverage = (
            len(required_tables_covered) / len(required_tables)
            if required_tables
            else 1.0
        )
        weakest_index = quality_ranks.index(min_rank) if quality_ranks else -1
        weakest_edge = ""
        if weakest_index >= 0:
            weakest_step = steps[weakest_index]
            weakest_edge = f"{weakest_step.get('left')} = {weakest_step.get('right')}"
        fanout_risks = [str(step.get("fanout_risk") or "low") for step in steps]
        many_to_many_risks = [str(step.get("many_to_many_risk") or "low") for step in steps]
        multiplicities = [str(step.get("multiplicity") or "unknown") for step in steps]
        row_order_count = sum(1 for step in steps if step.get("join_type") == "row_order")
        weak_value_overlap_count = sum(
            1 for step in steps if step.get("overlap_tier") == "weak_value_overlap"
        )
        has_broad_name_overlap = any(
            step.get("overlap_tier") == "broad_name_overlap" for step in steps
        )
        structural_risk_level = self._path_structural_risk(
            fanout_risks,
            many_to_many_risks,
            row_order_count,
            weak_value_overlap_count,
            has_broad_name_overlap,
        )
        return {
            "path_length": len(steps),
            "tables_covered": tables_covered,
            "required_tables_covered": required_tables_covered,
            "missing_required_tables": missing_required_tables,
            "required_coverage": round(required_coverage, 3),
            "weakest_edge_quality": edge_qualities[weakest_index] if weakest_index >= 0 else "weak",
            "weakest_edge": weakest_edge,
            "mean_edge_quality": self._quality_label(mean_rank),
            "min_edge_quality_rank": min_rank,
            "mean_edge_quality_rank": round(mean_rank, 3),
            "max_fanout_risk": self._max_ranked_value(fanout_risks, RISK_RANK, default="low"),
            "worst_multiplicity": self._max_ranked_value(
                multiplicities,
                MULTIPLICITY_RANK,
                default="unknown",
            ),
            "max_many_to_many_risk": self._max_ranked_value(
                many_to_many_risks,
                RISK_RANK,
                default="low",
            ),
            "min_containment": self._min_numeric_step_value(steps, "containment"),
            "min_jaccard": self._min_numeric_step_value(steps, "jaccard"),
            "row_order_count": row_order_count,
            "weak_value_overlap_count": weak_value_overlap_count,
            "has_broad_name_overlap": has_broad_name_overlap,
            "structural_risk_level": structural_risk_level,
            "structural_recommendation": self._structural_recommendation(structural_risk_level),
            "reason": self._join_path_reason(
                structural_risk_level,
                missing_required_tables,
                row_order_count,
                weak_value_overlap_count,
                has_broad_name_overlap,
            ),
        }

    def _empty_join_path_summary(self, required_tables: set[str]) -> dict[str, Any]:
        return {
            "path_length": 0,
            "tables_covered": [],
            "required_tables_covered": [],
            "missing_required_tables": sorted(required_tables),
            "required_coverage": 0.0 if required_tables else 1.0,
            "weakest_edge_quality": "weak",
            "weakest_edge": "",
            "mean_edge_quality": "weak",
            "max_fanout_risk": "low",
            "worst_multiplicity": "unknown",
            "max_many_to_many_risk": "low",
            "min_containment": None,
            "min_jaccard": None,
            "row_order_count": 0,
            "weak_value_overlap_count": 0,
            "has_broad_name_overlap": False,
            "structural_risk_level": "high" if required_tables else "medium",
            "structural_recommendation": "avoid_if_alternative",
            "reason": "no join path found",
        }

    def _quality_label(self, rank: float) -> str:
        if rank >= 2.5:
            return "strong"
        if rank >= 1.5:
            return "medium"
        return "weak"

    def _max_ranked_value(
        self,
        values: list[str],
        ranks: dict[str, int],
        *,
        default: str,
    ) -> str:
        if not values:
            return default
        return max(values, key=lambda value: ranks.get(value, ranks.get(default, 0)))

    def _min_numeric_step_value(self, steps: list[dict[str, Any]], key: str) -> float | None:
        values = [step.get(key) for step in steps if isinstance(step.get(key), (int, float))]
        if not values:
            return None
        return round(float(min(values)), 3)

    def _path_structural_risk(
        self,
        fanout_risks: list[str],
        many_to_many_risks: list[str],
        row_order_count: int,
        weak_value_overlap_count: int,
        has_broad_name_overlap: bool,
    ) -> str:
        if "high" in fanout_risks or "high" in many_to_many_risks:
            return "high"
        if row_order_count or weak_value_overlap_count or has_broad_name_overlap:
            return "medium"
        if "medium" in fanout_risks or "medium" in many_to_many_risks:
            return "medium"
        return "low"

    def _structural_recommendation(self, risk_level: str) -> str:
        if risk_level == "low":
            return "safe"
        if risk_level == "medium":
            return "verify"
        return "avoid_if_alternative"

    def _join_path_reason(
        self,
        risk_level: str,
        missing_required_tables: list[str],
        row_order_count: int,
        weak_value_overlap_count: int,
        has_broad_name_overlap: bool,
    ) -> str:
        if missing_required_tables:
            return "path does not cover all required tables"
        reasons = []
        if row_order_count:
            reasons.append("contains row-order join")
        if weak_value_overlap_count:
            reasons.append("contains weak value-overlap join")
        if has_broad_name_overlap:
            reasons.append("contains broad name-overlap join")
        if reasons:
            return "; ".join(reasons)
        if risk_level == "high":
            return "path has high structural fanout risk"
        if risk_level == "medium":
            return "path should be verified against question semantics"
        return "path has low structural risk"

    def _join_graph(self) -> dict[str, list[dict[str, Any]]]:
        graph: dict[str, list[dict[str, Any]]] = {}
        for candidate in self._join_candidates_list():
            payload = self._join_candidate_payload(candidate)
            tables = sorted(self._join_tables(candidate.left, candidate.right))
            if len(tables) != 2:
                continue
            graph.setdefault(tables[0], []).append(
                {"neighbor_table": tables[1], "candidate": payload, "score": candidate.score}
            )
            graph.setdefault(tables[1], []).append(
                {"neighbor_table": tables[0], "candidate": payload, "score": candidate.score}
            )
        return graph

    def _join_step_payload(self, candidate_payload: dict[str, Any]) -> dict[str, Any]:
        left = str(candidate_payload["left"])
        right = str(candidate_payload["right"])
        metadata = candidate_payload.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        payload = {
            "left_table": self._field_table(left),
            "left_column": self._field_column(left),
            "left": left,
            "operator": "=",
            "right_table": self._field_table(right),
            "right_column": self._field_column(right),
            "right": right,
            "join_type": candidate_payload.get("join_type"),
            "evidence": list(candidate_payload.get("evidence") or []),
            "structural_quality": self._join_edge_quality(candidate_payload),
            "multiplicity": metadata.get("multiplicity", "unknown"),
            "many_to_many_risk": metadata.get("many_to_many_risk", "low"),
            "fanout_risk": metadata.get("fanout_risk", "low"),
        }
        for key in [
            "overlap_tier",
            "left_match_multiplier",
            "right_match_multiplier",
            "overlap_count",
            "left_coverage",
            "right_coverage",
            "containment",
            "jaccard",
            "left_unique_ratio",
            "right_unique_ratio",
        ]:
            if key in metadata:
                payload[key] = metadata[key]
        sample_values = metadata.get("sample_values")
        if isinstance(sample_values, list) and sample_values:
            payload["sample_values"] = [
                str(value)
                for value in sample_values[:5]
                if str(value).strip()
            ]
        normalization_examples = metadata.get("normalization_examples")
        if isinstance(normalization_examples, list) and normalization_examples:
            payload["normalization_examples"] = normalization_examples[:3]
        warning = metadata.get("warning")
        if isinstance(warning, str) and warning.strip():
            payload["warning"] = warning.strip()
        if metadata.get("requires_verification") is True:
            payload["requires_verification"] = True
        return payload

    def _join_candidate_payload(self, candidate) -> dict[str, Any]:
        evidence_id = f"join:{candidate.left}:{candidate.right}"
        payload = {
            "id": evidence_id,
            "kind": "join",
            "database": self.database_name,
            "left": candidate.left,
            "right": candidate.right,
            "evidence": list(candidate.evidence),
            "metadata": dict(candidate.metadata),
            "join_type": candidate.metadata.get("join_type"),
            "tables": sorted(self._join_tables(candidate.left, candidate.right)),
        }
        self._remember_evidence(evidence_id, payload)
        return payload

    def _remember_returned_join_step(self, step: dict[str, Any]) -> None:
        left = self._string_or_none(step.get("left"))
        right = self._string_or_none(step.get("right"))
        if left is None or right is None:
            return
        join_type = self._string_or_none(step.get("join_type"))
        edge_key = self._join_edge_key(left, right)
        self._returned_join_edge_types.setdefault(edge_key, set())
        if join_type is not None:
            self._returned_join_edge_types[edge_key].add(join_type)
        self._returned_join_edges.setdefault(edge_key, []).append(_json_compatible(step))

    def _remember_returned_join_path(self, path_payload: dict[str, Any]) -> None:
        steps = path_payload.get("steps")
        if not isinstance(steps, list):
            return
        for step in steps:
            if isinstance(step, dict):
                self._remember_returned_join_step(step)
        self._returned_join_paths.append(_json_compatible(path_payload))

    def _join_edge_key(self, left: str, right: str) -> tuple[str, str]:
        return tuple(sorted([left, right]))

    def _join_tables(self, left: str, right: str) -> set[str]:
        tables = set()
        for field in [left, right]:
            if "." in field:
                tables.add(field.split(".", 1)[0])
        return tables

    def _tables_for_fields(self, fields: list[str] | set[str]) -> set[str]:
        return {
            field.split(".", 1)[0]
            for field in fields
            if isinstance(field, str) and "." in field
        }

    def _field_table(self, field: str) -> str | None:
        return field.split(".", 1)[0] if "." in field else None

    def _field_column(self, field: str) -> str:
        return field.split(".", 1)[1] if "." in field else field

    def _dedupe_strings(self, values: list[str]) -> list[str]:
        result: list[str] = []
        for value in values:
            if value and value not in result:
                result.append(value)
        return result

    def _resolve_table(self, value: Any) -> str | None:
        text = self._string_or_none(value)
        if text is None:
            return None
        evidence = self._evidence_by_id.get(text)
        if evidence is not None:
            metadata = evidence.get("metadata")
            metadata_table = (
                metadata.get("table") if isinstance(metadata, dict) else None
            )
            text = self._string_or_none(evidence.get("table") or metadata_table)
            if text is None:
                return None
        normalized = normalize_identifier(text)
        for table_schema in self.candidate_store.tables:
            if normalize_identifier(table_schema.table) == normalized:
                return table_schema.table
        return None

    def _resolve_field_profile(self, value: Any) -> FieldProfile | None:
        text = self._string_or_none(value)
        if text is None:
            return None
        evidence = self._evidence_by_id.get(text)
        if evidence is not None:
            metadata = evidence.get("metadata")
            if isinstance(metadata, dict):
                text = self._string_or_none(metadata.get("field")) or text
            text = self._string_or_none(evidence.get("field")) or text
        return self.candidate_store.field_profile(text)

    def _remember_evidence(self, evidence_id: str, payload: dict[str, Any]) -> None:
        self._evidence_by_id[evidence_id] = _json_compatible(payload)

    def _string_list(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        result: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip() and item.strip() not in result:
                result.append(item.strip())
        return result

    def _string_or_none(self, value: Any) -> str | None:
        if isinstance(value, str) and value.strip():
            return value.strip()
        return None
