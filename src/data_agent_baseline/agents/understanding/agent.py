from __future__ import annotations

import json
from typing import Any

from data_agent_baseline.agents.model import ModelMessage, OpenAIModelAdapter
from data_agent_baseline.agents.react import parse_model_step
from data_agent_baseline.agents.runtime import StepRecord
from data_agent_baseline.agents.shared import QuestionUnderstandingOutput
from data_agent_baseline.agents.understanding.prompt import build_question_understanding_messages
from data_agent_baseline.agents.understanding.tools import (
    CandidateStore,
    CandidateStoreBuilder,
    UnderstandingToolRegistry,
)
from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.tools.structured_context import StructuredContextStore


class QuestionUnderstandingLoopError(RuntimeError):
    def __init__(self, message: str, steps: list[StepRecord]) -> None:
        super().__init__(message)
        self.steps = list(steps)


class QuestionUnderstandAgent:
    """Understand a natural-language question as a schema-grounded semantic plan."""

    def __init__(
        self,
        *,
        model: OpenAIModelAdapter,
    ) -> None:
        self.model = model
        self.candidate_store_builder = CandidateStoreBuilder()
        self.last_candidate_store: CandidateStore | None = None
        self.last_understanding_steps: list[StepRecord] = []
        self.last_structured_store: StructuredContextStore | None = None

    def run(
        self,
        task: PublicTask,
        *,
        structured_store: StructuredContextStore,
        max_steps: int,
        doc_evidence_context: str = "",
    ) -> QuestionUnderstandingOutput:
        self.last_understanding_steps = []
        if max_steps <= 0:
            raise QuestionUnderstandingLoopError(
                "QuestionUnderstandAgent did not receive any remaining steps.",
                [],
            )

        self.last_structured_store = structured_store
        candidate_store = self.candidate_store_builder.build(
            structured_store=structured_store,
        )
        self.last_candidate_store = candidate_store
        knowledge_context = self._read_knowledge_context(task)

        tool_registry = UnderstandingToolRegistry(
            task=task,
            candidate_store=candidate_store,
            knowledge_context=knowledge_context,
            doc_evidence_context=doc_evidence_context,
        )
        decision, steps = self._run_loop(
            task=task,
            tool_registry=tool_registry,
            knowledge_context=knowledge_context,
            doc_evidence_context=doc_evidence_context,
            max_steps=max_steps,
        )
        self.last_understanding_steps = steps

        semantic_plan = decision.get("semantic_plan")
        output = QuestionUnderstandingOutput(
            semantic_plan=dict(semantic_plan) if isinstance(semantic_plan, dict) else {},
            remaining_steps=max_steps - len(steps),
        )
        return output

    def _run_loop(
        self,
        *,
        task: PublicTask,
        tool_registry: UnderstandingToolRegistry,
        knowledge_context: str,
        doc_evidence_context: str,
        max_steps: int,
    ) -> tuple[dict[str, Any], list[StepRecord]]:
        messages = build_question_understanding_messages(
            question=task.question,
            tool_descriptions=tool_registry.describe_for_prompt(),
            knowledge_context=knowledge_context,
            doc_evidence_context=doc_evidence_context,
        )
        steps: list[StepRecord] = []
        for step_index in range(1, max_steps + 1):
            raw_response = ""
            try:
                raw_response = self.model.complete(messages)
                model_step = parse_model_step(raw_response)
            except Exception as exc:
                observation = {
                    "ok": False,
                    "tool": "__model__",
                    "error": str(exc),
                    "repair_instruction": (
                        "Return exactly one fenced JSON object with keys thought, action, "
                        "and action_input. Choose an available tool, or finish with "
                        "finalize_understanding when the semantic plan is ready."
                    ),
                }
                steps.append(
                    StepRecord(
                        step_index=step_index,
                        thought="",
                        action="__error__",
                        action_input={},
                        raw_response=raw_response,
                        observation=observation,
                        ok=False,
                        agent_state=self._agent_state("model_error"),
                    )
                )
                if raw_response:
                    messages.append(ModelMessage(role="assistant", content=raw_response))
                messages.append(
                    ModelMessage(
                        role="user",
                        content=json.dumps(observation, ensure_ascii=False),
                    )
                )
                continue

            action = model_step.action
            action_input = model_step.action_input
            try:
                tool_result = tool_registry.execute(
                    action=action,
                    action_input=action_input,
                )
                observation = {"ok": True, "tool": action, "content": tool_result.content}
                ok = True
            except Exception as exc:
                tool_result = None
                observation = {
                    "ok": False,
                    "tool": action,
                    "error": str(exc),
                    "repair_instruction": (
                        "Choose one available action and pass action_input matching its schema."
                    ),
                }
                ok = False

            steps.append(
                StepRecord(
                    step_index=step_index,
                    thought=model_step.thought,
                    action=action,
                    action_input=action_input,
                    raw_response=model_step.raw_response,
                    observation=observation,
                    ok=ok,
                    agent_state=self._agent_state("running"),
                )
            )
            if ok and tool_result is not None and tool_result.terminal:
                return tool_result.payload or {}, steps

            messages.append(ModelMessage(role="assistant", content=model_step.raw_response))
            messages.append(
                ModelMessage(
                    role="user",
                    content=json.dumps(observation, ensure_ascii=False),
                )
            )

        raise QuestionUnderstandingLoopError(
            "QuestionUnderstandAgent did not call finalize_understanding within max_steps.",
            steps,
        )

    def _agent_state(self, status: str) -> dict[str, Any]:
        return {
            "agent_name": "QuestionUnderstandAgent",
            "status": status,
        }

    def _read_knowledge_context(self, task: PublicTask) -> str:
        knowledge_path = task.context_dir / "knowledge.md"
        if not knowledge_path.is_file():
            return ""
        return knowledge_path.read_text(errors="replace")
