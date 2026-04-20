from __future__ import annotations

import json

from data_agent_baseline.benchmark.schema import PublicTask


REACT_SYSTEM_PROMPT = """
You are a ReAct-style data agent.

You are solving a task from a public dataset. You may only inspect files inside the task's `context/` directory through the provided tools.

Rules:
1. If a semantic layer summary is provided, treat it as the primary source for semantics, mappings, coding rules, and join hints.
2. Base your answer only on information you can observe through the provided tools.
3. The task is complete only when you call the `answer` tool.
4. The `answer` tool must receive a table with `columns` and `rows`.
5. Always return exactly one JSON object with keys `thought`, `action`, and `action_input`.
6. Always wrap that JSON object in exactly one fenced code block that starts with ```json and ends with ```.
7. Do not output any text before or after the fenced JSON block.
8. When joining or merging multiple tables/files, strictly perform an INNER JOIN to ensure data consistency, and drop any resulting rows that contain null, NaN, or empty values in the requested output columns.
9. When outputting the final answer, copy all IDs EXACTLY as they appear in the original data. Do NOT truncate, drop trailing digits, or modify the IDs in any way.
10. Use this priority order when a semantic layer is present: semantic layer first, executor tools second, raw file reads last.
11. If the semantic layer already gives enough concept mappings, value mappings, join hints, or executor-ready signals, go directly to `execute_context_sql` or `execute_python`.
12. Do not use `read_doc`, `read_json`, or `read_csv` as a default first step when a semantic layer is present.
13. Only use raw file read tools as fallback for `verify_mapping`, `extract_exact_evidence`, `semantic_gap`, or `read_uncovered_field`.
14. If raw file reading is necessary, make it narrow and local. Provide a `scope` such as `targeted`, `section_only`, or `local_preview`.
15. Do not reread raw files to rediscover definitions, coding rules, or knowledge passages already covered by the semantic layer.
16. Do not use `list_context` as a default first step when a semantic layer is already present and executor-ready. The semantic layer and preprocessing have already explored the environment.
17. Do not read `knowledge.md` again just to rediscover definitions or severity rules already covered by the semantic layer.
18. Do not read JSON or CSV previews just to understand table structure or join paths when semantic-layer join hints and schema profiles are already sufficient for execution.

Keep reasoning concise and grounded in the observed data.
""".strip()

RESPONSE_EXAMPLES = """
Example response when you need to inspect the context:
```json
{"thought":"I should inspect the available files first.","action":"list_context","action_input":{"max_depth":4}}
```

Example response when you have the final answer:
```json
{"thought":"I have the final result table.","action":"answer","action_input":{"columns":["average_long_shots"],"rows":[["63.5"]]}}
```
""".strip()


def build_system_prompt(tool_descriptions: str, system_prompt: str | None = None) -> str:
    base_prompt = system_prompt or REACT_SYSTEM_PROMPT
    return (
        f"{base_prompt}\n\n"
        "Available tools:\n"
        f"{tool_descriptions}\n\n"
        f"{RESPONSE_EXAMPLES}\n\n"
        "You must always return a single ```json fenced block containing one JSON object "
        "with keys `thought`, `action`, and `action_input`, and no extra text."
    )


def build_task_prompt(task: PublicTask, semantic_summary: str | None = None) -> str:
    prompt = (
        f"Question: {task.question}\n"
        "All tool file paths are relative to the task context directory. "
        "When you have the final table, call the `answer` tool."
    )
    if semantic_summary:
        prompt += (
            "\n\n"
            "Semantic layer summary built from local task context:\n"
            f"{semantic_summary}\n\n"
            "Use this summary as your main planning context. "
            "If the semantic layer already covers the semantics you need, move directly to executor tools instead of previewing raw files."
        )
    return prompt


def build_observation_prompt(observation: dict[str, object]) -> str:
    rendered = json.dumps(observation, ensure_ascii=False, indent=2)
    return f"Observation:\n{rendered}"
