from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from data_agent_baseline.agents.model import ModelMessage, OpenAIModelAdapter
from data_agent_baseline.agents.execution.prompt import build_execution_initial_messages
from data_agent_baseline.agents.react import parse_model_step
from data_agent_baseline.agents.runtime import AgentRunResult, AgentRuntimeState, StepRecord
from data_agent_baseline.agents.shared import QuestionUnderstandingOutput
from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.tools.registry import ToolRegistry
from data_agent_baseline.tools.structured_context import StructuredContextStore


MISSING_UNDERSTANDING_OUTPUT_FAILURE_REASON = (
    "ExecutionAgent requires semantic_plan output from QuestionUnderstandAgent."
)
MAX_STEPS_FAILURE_REASON = "Agent did not submit an answer within max_steps."
MODEL_ACTION_REPAIR_INSTRUCTION = (
    "Return exactly one fenced JSON object with keys thought, action, and action_input. "
    "Use `answer` only when the final table is ready."
)
TOOL_FAILURE_REPAIR_INSTRUCTION = (
    "Choose a tool that matches the file type and continue toward the final answer."
)


@dataclass(frozen=True, slots=True)
class ExecutionAgentConfig:
    max_steps: int = 16


class ExecutionAgent:
    """Execution-focused agent that consumes structured execution specifications."""

    def __init__(
        self,
        *,
        model: OpenAIModelAdapter,
        tools: ToolRegistry,
        max_steps: int = 16,
    ) -> None:
        self.model = model
        self.tools = tools
        self.config = ExecutionAgentConfig(max_steps=max_steps)

    def _build_messages(
        self,
        initial_messages: list[ModelMessage],
        state: AgentRuntimeState,
    ) -> list[ModelMessage]:
        messages = list(initial_messages)
        for step in state.steps:
            messages.append(ModelMessage(role="assistant", content=str(step.raw_response)))
            messages.append(
                ModelMessage(
                    role="user",
                    content=json.dumps(step.observation, ensure_ascii=False),
                )
            )
        return messages

    def _agent_state(self, status: str) -> dict[str, Any]:
        return {
            "agent_name": "ExecutionAgent",
            "status": status,
        }

    def run(
        self,
        task: PublicTask,
        question_understanding: QuestionUnderstandingOutput | dict[str, Any] | None,
        max_steps: int | None = None,
        structured_store: StructuredContextStore | None = None,
    ) -> AgentRunResult:
        state = AgentRuntimeState()
        effective_max_steps = self.config.max_steps if max_steps is None else max_steps
        understanding_payload = self._understanding_payload(question_understanding)
        if understanding_payload is None:
            return AgentRunResult(
                task_id=task.task_id,
                answer=None,
                steps=[],
                failure_reason=MISSING_UNDERSTANDING_OUTPUT_FAILURE_REASON,
                remaining_steps=effective_max_steps,
            )
        initial_messages = self._initial_messages(
            understanding_payload,
            structured_store=structured_store,
        )
        if effective_max_steps <= 0:
            return AgentRunResult(
                task_id=task.task_id,
                answer=None,
                steps=[],
                failure_reason=MAX_STEPS_FAILURE_REASON,
                remaining_steps=0,
            )

        for step_index in range(1, effective_max_steps + 1):
            raw_response = ""
            try:
                raw_response = self.model.complete(
                    self._build_messages(initial_messages, state)
                )
                model_step = parse_model_step(raw_response)
            except Exception as exc:
                observation = {
                    "ok": False,
                    "tool": "__model__",
                    "error": str(exc),
                    "repair_instruction": MODEL_ACTION_REPAIR_INSTRUCTION,
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
                        agent_state=self._agent_state("parse_error"),
                    )
                )
                continue

            action = model_step.action
            action_input = model_step.action_input
            tool_result = None
            try:
                tool_result = self.tools.execute(
                    task,
                    action,
                    action_input,
                    structured_store=structured_store,
                )
                observation = {
                    "ok": tool_result.ok,
                    "tool": action,
                    "content": tool_result.content,
                }
                ok = tool_result.ok
            except Exception as exc:
                observation = {
                    "ok": False,
                    "tool": action,
                    "error": str(exc),
                    "repair_instruction": TOOL_FAILURE_REPAIR_INSTRUCTION,
                }
                ok = False

            status = (
                "terminal"
                if ok and tool_result is not None and tool_result.is_terminal
                else "tool_result"
            )
            step_record = StepRecord(
                step_index=step_index,
                thought=model_step.thought,
                action=action,
                action_input=action_input,
                raw_response=model_step.raw_response,
                observation=observation,
                ok=ok,
                agent_state=self._agent_state(status),
            )
            state.steps.append(step_record)
            if ok and tool_result is not None and tool_result.is_terminal:
                state.answer = tool_result.answer
                break

        if state.answer is None and state.failure_reason is None:
            state.failure_reason = MAX_STEPS_FAILURE_REASON

        return AgentRunResult(
            task_id=task.task_id,
            answer=state.answer,
            steps=list(state.steps),
            failure_reason=state.failure_reason,
            remaining_steps=max(0, effective_max_steps - len(state.steps)),
        )

    def _understanding_payload(
        self,
        question_understanding: QuestionUnderstandingOutput | dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if question_understanding is None:
            return None
        if isinstance(question_understanding, QuestionUnderstandingOutput):
            payload = question_understanding.to_dict()
        elif isinstance(question_understanding, dict):
            payload = dict(question_understanding)
        else:
            return None
        semantic_plan = payload.get("semantic_plan")
        if not isinstance(semantic_plan, dict):
            return None
        return payload

    def _initial_messages(
        self,
        understanding_payload: dict[str, Any],
        *,
        structured_store: StructuredContextStore | None,
    ) -> list[ModelMessage]:
        return build_execution_initial_messages(
            understanding_payload=understanding_payload,
            tool_descriptions=self.tools.describe_for_prompt(),
            structured_store=structured_store,
        )
