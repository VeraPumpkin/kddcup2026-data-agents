from __future__ import annotations

import json
import re
from dataclasses import dataclass

from data_agent_baseline.agents.model import ModelAdapter, ModelMessage, ModelStep
from data_agent_baseline.agents.prompt import (
    REACT_SYSTEM_PROMPT,
    build_observation_prompt,
    build_system_prompt,
    build_task_prompt,
)
from data_agent_baseline.agents.runtime import AgentRunResult, AgentRuntimeState, StepRecord
from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.tools.registry import ToolRegistry


@dataclass(frozen=True, slots=True)
class ReActAgentConfig:
    max_steps: int = 16


def _strip_json_fence(raw_response: str) -> str:
    text = raw_response.strip()
    fence_match = re.search(r"```json\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fence_match is not None:
        return fence_match.group(1).strip()
    generic_fence_match = re.search(r"```\s*(.*?)\s*```", text, flags=re.DOTALL)
    if generic_fence_match is not None:
        return generic_fence_match.group(1).strip()
    return text


def _load_single_json_object(text: str) -> dict[str, object]:
    payload, end = json.JSONDecoder().raw_decode(text)
    remainder = text[end:].strip()
    if remainder:
        cleaned_remainder = re.sub(r"(?:\\[nrt])+", "", remainder).strip()
        if cleaned_remainder:
            raise ValueError("Model response must contain only one JSON object.")
    if not isinstance(payload, dict):
        raise ValueError("Model response must be a JSON object.")
    return payload


def parse_model_step(raw_response: str) -> ModelStep:
    normalized = _strip_json_fence(raw_response)
    payload = _load_single_json_object(normalized)

    thought = payload.get("thought", "")
    action = payload.get("action")
    action_input = payload.get("action_input", {})
    if not isinstance(thought, str):
        raise ValueError("thought must be a string.")
    if not isinstance(action, str) or not action:
        raise ValueError("action must be a non-empty string.")
    if not isinstance(action_input, dict):
        raise ValueError("action_input must be a JSON object.")

    return ModelStep(
        thought=thought,
        action=action,
        action_input=action_input,
        raw_response=raw_response,
    )


class ReActAgent:
    def __init__(
        self,
        *,
        model: ModelAdapter,
        tools: ToolRegistry,
        config: ReActAgentConfig | None = None,
        system_prompt: str | None = None,
        semantic_summary: str | None = None,
        semantic_context: dict[str, object] | None = None,
    ) -> None:
        self.model = model
        self.tools = tools
        self.config = config or ReActAgentConfig()
        self.system_prompt = system_prompt or REACT_SYSTEM_PROMPT
        self.semantic_summary = semantic_summary
        self.semantic_context = dict(semantic_context or {})

    def _semantic_layer_present(self) -> bool:
        return bool(self.semantic_summary and self.semantic_summary.strip())

    def _assess_semantic_coverage(self, question: str) -> dict[str, object]:
        concept_count = int(self.semantic_context.get("concept_count", 0))
        semantic_link_count = int(self.semantic_context.get("semantic_link_count", 0))
        value_mapping_count = int(self.semantic_context.get("value_mapping_count", 0))
        join_candidate_count = int(self.semantic_context.get("join_candidate_count", 0))
        retrieved_chunk_count = int(self.semantic_context.get("retrieved_chunk_count", 0))
        schema_profile_count = int(self.semantic_context.get("schema_profile_count", 0))

        has_definitions = concept_count > 0 or retrieved_chunk_count > 0
        has_field_mapping = semantic_link_count > 0
        has_value_mapping = value_mapping_count > 0
        has_join_candidate = join_candidate_count > 0
        has_executor_readiness = schema_profile_count > 0 and (
            has_field_mapping or has_value_mapping or has_join_candidate
        )
        return {
            "question": question,
            "has_definitions": has_definitions,
            "has_field_mapping": has_field_mapping,
            "has_value_mapping": has_value_mapping,
            "has_join_candidate": has_join_candidate,
            "has_executor_readiness": has_executor_readiness,
            "concept_count": concept_count,
            "semantic_link_count": semantic_link_count,
            "value_mapping_count": value_mapping_count,
            "join_candidate_count": join_candidate_count,
            "retrieved_chunk_count": retrieved_chunk_count,
            "schema_profile_count": schema_profile_count,
        }

    def _build_raw_read_guidance(
        self,
        tool_name: str,
        *,
        coverage_state: dict[str, object],
        denial_reason: str,
    ) -> dict[str, object]:
        replacement_tools = (
            ["execute_context_sql", "execute_python", "inspect_sqlite_schema"]
            if bool(coverage_state.get("has_executor_readiness"))
            else ["inspect_sqlite_schema", "execute_python", "execute_context_sql"]
        )
        supplied = [
            key
            for key, present in {
                "definitions": coverage_state.get("has_definitions"),
                "field_mappings": coverage_state.get("has_field_mapping"),
                "value_mappings": coverage_state.get("has_value_mapping"),
                "join_candidates": coverage_state.get("has_join_candidate"),
                "executor_readiness": coverage_state.get("has_executor_readiness"),
            }.items()
            if bool(present)
        ]
        return {
            "ok": False,
            "tool": tool_name,
            "content": {
                "error_type": "raw_read_restricted",
                "denial_reason": denial_reason,
                "semantic_layer_present": True,
                "semantic_coverage": coverage_state,
                "semantic_layer_supplies": supplied,
                "message": (
                    f"{tool_name} is restricted because the semantic layer already covers enough context "
                    "to plan or execute without broad raw-file previewing."
                ),
                "allowed_reasons": [
                    "verify_mapping",
                    "extract_exact_evidence",
                    "semantic_gap",
                    "read_uncovered_field",
                ],
                "allowed_scopes": ["targeted", "section_only", "local_preview"],
                "suggested_next_actions": [
                    "Use semantic-layer mappings and retrieved knowledge evidence first.",
                    f"Prefer {', '.join(replacement_tools)} before raw-file preview tools.",
                    f"If raw reading is still necessary, retry {tool_name} with both a valid `reason` and a narrow `scope`.",
                ],
            },
        }

    def _build_preprocessing_guidance(
        self,
        tool_name: str,
        *,
        coverage_state: dict[str, object],
        denial_reason: str,
    ) -> dict[str, object]:
        return {
            "ok": False,
            "tool": tool_name,
            "content": {
                "error_type": "preprocessing_already_sufficient",
                "denial_reason": denial_reason,
                "semantic_layer_present": True,
                "semantic_coverage": coverage_state,
                "message": (
                    f"{tool_name} is restricted because preprocessing and the semantic layer already provide enough "
                    "environment context to begin planning or execution."
                ),
                "suggested_next_actions": [
                    "Plan directly from the semantic layer.",
                    "Move to execute_context_sql or execute_python if executor readiness is available.",
                    "Use inspect_sqlite_schema only if a specific schema detail is still missing.",
                ],
            },
        }

    def _should_allow_preprocessing_read(
        self,
        *,
        tool_name: str,
        action_input: dict[str, object],
        question: str,
        state: AgentRuntimeState,
    ) -> tuple[bool, str | None]:
        if tool_name != "list_context":
            return True, None
        if not state.semantic_layer_present:
            return True, None
        coverage_state = state.semantic_coverage or self._assess_semantic_coverage(question)
        reason = action_input.get("reason")
        if bool(coverage_state.get("has_executor_readiness")) and reason != "locate_uncovered_asset":
            return False, "semantic_preprocessing_already_sufficient"
        return True, str(reason) if isinstance(reason, str) else None

    def _should_allow_raw_read(
        self,
        *,
        tool_name: str,
        action_input: dict[str, object],
        question: str,
        state: AgentRuntimeState,
    ) -> tuple[bool, str | None]:
        if tool_name not in {"read_csv", "read_json", "read_doc"}:
            return True, None
        if not state.semantic_layer_present:
            return True, None

        allowed_reasons = {
            "verify_mapping",
            "extract_exact_evidence",
            "semantic_gap",
            "read_uncovered_field",
        }
        allowed_scopes = {"targeted", "section_only", "local_preview"}
        coverage_state = state.semantic_coverage or self._assess_semantic_coverage(question)
        reason = action_input.get("reason")
        scope = action_input.get("scope")
        if not isinstance(reason, str) or reason not in allowed_reasons:
            return False, "missing_or_invalid_reason"
        if not isinstance(scope, str) or scope not in allowed_scopes:
            return False, "missing_or_invalid_scope"
        path = str(action_input.get("path", ""))
        if (
            bool(coverage_state.get("has_definitions"))
            and tool_name == "read_doc"
            and path.endswith("knowledge.md")
            and reason == "extract_exact_evidence"
        ):
            return False, "semantic_definitions_already_available"
        if (
            bool(coverage_state.get("has_executor_readiness"))
            and tool_name in {"read_json", "read_csv"}
            and reason == "extract_exact_evidence"
            and not bool(coverage_state.get("has_field_mapping"))
            and bool(coverage_state.get("has_join_candidate"))
        ):
            return False, "prefer_executor_over_structure_preview"
        if bool(coverage_state.get("has_executor_readiness")) and reason == "semantic_gap":
            return False, "semantic_coverage_already_sufficient"
        return True, reason

    def _build_messages(self, task: PublicTask, state: AgentRuntimeState) -> list[ModelMessage]:
        system_content = build_system_prompt(
            self.tools.describe_for_prompt(),
            system_prompt=self.system_prompt,
        )
        messages = [ModelMessage(role="system", content=system_content)]
        messages.append(
            ModelMessage(
                role="user",
                content=build_task_prompt(task, semantic_summary=self.semantic_summary),
            )
        )
        for step in state.steps:
            messages.append(ModelMessage(role="assistant", content=step.raw_response))
            messages.append(
                ModelMessage(role="user", content=build_observation_prompt(step.observation))
            )
        return messages

    def run(self, task: PublicTask) -> AgentRunResult:
        state = AgentRuntimeState(
            semantic_layer_present=self._semantic_layer_present(),
            semantic_coverage=self._assess_semantic_coverage(task.question),
        )
        for step_index in range(1, self.config.max_steps + 1):
            raw_response = self.model.complete(self._build_messages(task, state))
            try:
                model_step = parse_model_step(raw_response)
                allow_preprocessing_read, preprocessing_reason = self._should_allow_preprocessing_read(
                    tool_name=model_step.action,
                    action_input=model_step.action_input,
                    question=task.question,
                    state=state,
                )
                if not allow_preprocessing_read:
                    observation = self._build_preprocessing_guidance(
                        model_step.action,
                        coverage_state=state.semantic_coverage,
                        denial_reason=str(preprocessing_reason),
                    )
                    state.steps.append(
                        StepRecord(
                            step_index=step_index,
                            thought=model_step.thought,
                            action=model_step.action,
                            action_input=model_step.action_input,
                            raw_response=raw_response,
                            observation=observation,
                            ok=False,
                        )
                    )
                    continue
                allow_raw_read, raw_read_reason = self._should_allow_raw_read(
                    tool_name=model_step.action,
                    action_input=model_step.action_input,
                    question=task.question,
                    state=state,
                )
                if not allow_raw_read:
                    state.raw_read_denials += 1
                    tool_result = None
                    observation = self._build_raw_read_guidance(
                        model_step.action,
                        coverage_state=state.semantic_coverage,
                        denial_reason=str(raw_read_reason),
                    )
                    ok = False
                else:
                    tool_result = self.tools.execute(task, model_step.action, model_step.action_input)
                    observation = {
                        "ok": tool_result.ok,
                        "tool": model_step.action,
                        "content": tool_result.content,
                    }
                    if model_step.action in {"read_csv", "read_json", "read_doc"}:
                        observation["raw_read_reason"] = raw_read_reason
                        observation["raw_read_scope"] = model_step.action_input.get("scope")
                    if state.semantic_layer_present:
                        observation["semantic_layer_present"] = True
                        observation["semantic_coverage"] = state.semantic_coverage
                    ok = tool_result.ok
                step_record = StepRecord(
                    step_index=step_index,
                    thought=model_step.thought,
                    action=model_step.action,
                    action_input=model_step.action_input,
                    raw_response=raw_response,
                    observation=observation,
                    ok=ok,
                )
                state.steps.append(step_record)
                if tool_result is not None and tool_result.is_terminal:
                    state.answer = tool_result.answer
                    break
            except Exception as exc:
                observation = {
                    "ok": False,
                    "error": str(exc),
                }
                state.steps.append(
                    StepRecord(
                        step_index=step_index,
                        thought="",
                        action="__error__",
                        action_input={},
                        raw_response=raw_response,
                        observation=observation,
                        ok=False,
                    )
                )

        if state.answer is None and state.failure_reason is None:
            state.failure_reason = "Agent did not submit an answer within max_steps."

        return AgentRunResult(
            task_id=task.task_id,
            answer=state.answer,
            steps=list(state.steps),
            failure_reason=state.failure_reason,
        )
