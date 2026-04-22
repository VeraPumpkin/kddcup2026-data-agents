from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from data_agent_baseline.benchmark.schema import AnswerTable, PublicTask
from data_agent_baseline.tools.filesystem import (
    list_context_tree,
)
from data_agent_baseline.tools.python_exec import execute_python_code
from data_agent_baseline.tools.structured_context import StructuredContextStore

EXECUTE_PYTHON_TIMEOUT_SECONDS = 30


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolExecutionResult:
    ok: bool
    content: dict[str, Any]
    is_terminal: bool = False
    answer: AnswerTable | None = None


ToolHandler = Callable[
    [PublicTask, dict[str, Any], StructuredContextStore | None],
    ToolExecutionResult,
]


def _list_context(
    task: PublicTask,
    action_input: dict[str, Any],
    structured_store: StructuredContextStore | None = None,
) -> ToolExecutionResult:
    max_depth = int(action_input.get("max_depth", 4))
    return ToolExecutionResult(ok=True, content=list_context_tree(task, max_depth=max_depth))


def _execute_duckdb_sql(
    task: PublicTask,
    action_input: dict[str, Any],
    structured_store: StructuredContextStore | None = None,
) -> ToolExecutionResult:
    owns_store = structured_store is None
    store = structured_store or StructuredContextStore.from_task(task)
    sql = str(action_input["sql"])
    limit = int(action_input.get("limit", 200))
    try:
        return ToolExecutionResult(ok=True, content=store.query(sql, limit=limit))
    finally:
        if owns_store:
            store.close()


def _execute_python(
    task: PublicTask,
    action_input: dict[str, Any],
    structured_store: StructuredContextStore | None = None,
) -> ToolExecutionResult:
    code = str(action_input["code"])
    content = execute_python_code(
        context_root=task.context_dir,
        code=code,
        timeout_seconds=EXECUTE_PYTHON_TIMEOUT_SECONDS,
    )
    return ToolExecutionResult(ok=bool(content.get("success")), content=content)


def _answer(
    _: PublicTask,
    action_input: dict[str, Any],
    structured_store: StructuredContextStore | None = None,
) -> ToolExecutionResult:
    columns = action_input.get("columns")
    rows = action_input.get("rows")
    if not isinstance(columns, list) or not columns or not all(isinstance(item, str) for item in columns):
        raise ValueError("answer.columns must be a non-empty list of strings.")
    if not isinstance(rows, list):
        raise ValueError("answer.rows must be a list.")

    normalized_rows: list[list[Any]] = []
    for row in rows:
        if not isinstance(row, list):
            raise ValueError("Each answer row must be a list.")
        if len(row) != len(columns):
            raise ValueError("Each answer row must match the number of columns.")
        normalized_rows.append(list(row))

    answer = AnswerTable(columns=list(columns), rows=normalized_rows)
    return ToolExecutionResult(
        ok=True,
        content={
            "status": "submitted",
            "column_count": len(columns),
            "row_count": len(normalized_rows),
        },
        is_terminal=True,
        answer=answer,
    )


@dataclass(slots=True)
class ToolRegistry:
    specs: dict[str, ToolSpec]
    handlers: dict[str, ToolHandler]

    def describe_for_prompt(self) -> str:
        lines = []
        for name in sorted(self.specs):
            spec = self.specs[name]
            lines.append(f"- {spec.name}: {spec.description}")
            lines.append(f"  input_schema: {spec.input_schema}")
        return "\n".join(lines)

    def execute(
        self,
        task: PublicTask,
        action: str,
        action_input: dict[str, Any],
        *,
        structured_store: StructuredContextStore | None = None,
    ) -> ToolExecutionResult:
        if action not in self.handlers:
            raise KeyError(f"Unknown tool: {action}")
        return self.handlers[action](task, action_input, structured_store)


def create_default_tool_registry() -> ToolRegistry:
    specs = {
        "answer": ToolSpec(
            name="answer",
            description="Submit the final answer table. This is the only valid terminating action.",
            input_schema={
                "type": "object",
                "properties": {
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                    },
                    "rows": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {},
                        },
                    },
                },
                "required": ["columns", "rows"],
                "additionalProperties": False,
            },
        ),
        "execute_duckdb_sql": ToolSpec(
            name="execute_duckdb_sql",
            description=(
                "Run read-only SQL against the per-task DuckDB store containing registered "
                "CSV tables, JSON records tables, SQLite/DB tables, and doc annotation tables."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "Read-only SQL statement starting with SELECT, WITH, PRAGMA, DESCRIBE, or SHOW.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of rows to return.",
                        "default": 200,
                        "minimum": 1,
                    },
                },
                "required": ["sql"],
                "additionalProperties": False,
            },
        ),
        "execute_python": ToolSpec(
            name="execute_python",
            description=(
                "Execute arbitrary Python code with the task context directory as the "
                "working directory. The tool returns the code's captured stdout as `output`. "
                f"The execution timeout is fixed at {EXECUTE_PYTHON_TIMEOUT_SECONDS} seconds."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute inside a copy of the task context.",
                    },
                },
                "required": ["code"],
                "additionalProperties": False,
            },
        ),
    }
    handlers = {
        "answer": _answer,
        "execute_duckdb_sql": _execute_duckdb_sql,
        "execute_python": _execute_python,
    }
    return ToolRegistry(specs=specs, handlers=handlers)
