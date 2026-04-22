from __future__ import annotations

from dataclasses import replace

from data_agent_baseline.agents.execution import ExecutionAgent
from data_agent_baseline.agents.long_text_doc import LongTextDocFactAgent
from data_agent_baseline.agents.model import OpenAIModelAdapter
from data_agent_baseline.agents.runtime import AgentRunResult, StepRecord
from data_agent_baseline.agents.understanding import (
    QuestionUnderstandAgent,
    QuestionUnderstandingLoopError,
)
from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.tools.registry import ToolRegistry
from data_agent_baseline.tools.structured_context import StructuredContextStore

MAX_STEPS_FAILURE_REASON = "Agent did not submit an answer within max_steps."


class MultiAgentOrchestrator:
    """Question understanding and execution pipeline."""

    def __init__(
        self,
        *,
        model: OpenAIModelAdapter,
        tools: ToolRegistry,
        max_steps: int = 16,
    ) -> None:
        self.model = model
        self.tools = tools
        self.max_steps = max_steps
        self.long_text_doc_agent = LongTextDocFactAgent(model=model)
        self.question_understand_agent = QuestionUnderstandAgent(model=model)
        self.execution_agent = ExecutionAgent(
            model=model,
            tools=tools,
            max_steps=max_steps,
        )

    def run(self, task: PublicTask) -> AgentRunResult:
        structured_store: StructuredContextStore | None = None
        long_text_doc_steps: list[StepRecord] = []
        understanding_steps: list[StepRecord] = []
        self.long_text_doc_agent.last_steps = []
        self.question_understand_agent.last_understanding_steps = []
        if self.max_steps <= 0:
            return AgentRunResult(
                task_id=task.task_id,
                answer=None,
                steps=[],
                failure_reason=MAX_STEPS_FAILURE_REASON,
            )

        try:
            try:
                structured_store = StructuredContextStore.from_task(task)
                if self._has_doc_context(task):
                    self.long_text_doc_agent.run(
                        task,
                        structured_store=structured_store,
                    )
                    long_text_doc_steps = list(self.long_text_doc_agent.last_steps)
                understanding_output = self.question_understand_agent.run(
                    task,
                    structured_store=structured_store,
                    max_steps=self.max_steps,
                )
                understanding_steps = list(
                    self.question_understand_agent.last_understanding_steps
                )
                candidate_store = self.question_understand_agent.last_candidate_store
                if candidate_store is None:
                    raise RuntimeError("Question understanding did not build a candidate store.")

            except QuestionUnderstandingLoopError as exc:
                understanding_steps = list(exc.steps)
                return AgentRunResult(
                    task_id=task.task_id,
                    answer=None,
                    steps=self._top_level_steps(
                        ("LongTextDocFactAgent", long_text_doc_steps),
                        ("QuestionUnderstandAgent", understanding_steps),
                    ),
                    failure_reason=f"Pipeline failed before execution: {exc}",
                )
            except Exception as exc:
                return AgentRunResult(
                    task_id=task.task_id,
                    answer=None,
                    steps=self._top_level_steps(
                        ("LongTextDocFactAgent", list(self.long_text_doc_agent.last_steps)),
                        (
                            "QuestionUnderstandAgent",
                            list(self.question_understand_agent.last_understanding_steps),
                        ),
                    ),
                    failure_reason=f"Pipeline failed before execution: {exc}",
                )

            run_result = self.execution_agent.run(
                task,
                understanding_output,
                max_steps=understanding_output.remaining_steps,
                structured_store=structured_store,
            )
            return AgentRunResult(
                task_id=run_result.task_id,
                answer=run_result.answer,
                steps=self._top_level_steps(
                    ("LongTextDocFactAgent", list(self.long_text_doc_agent.last_steps)),
                    (
                        "QuestionUnderstandAgent",
                        list(self.question_understand_agent.last_understanding_steps),
                    ),
                    ("ExecutionAgent", list(run_result.steps)),
                ),
                failure_reason=run_result.failure_reason,
                remaining_steps=run_result.remaining_steps,
            )
        finally:
            self._close_structured_store(structured_store)

    def _close_structured_store(self, structured_store: StructuredContextStore | None) -> None:
        if structured_store is None:
            return
        structured_store.close()
        if self.question_understand_agent.last_structured_store is structured_store:
            self.question_understand_agent.last_structured_store = None
        candidate_store = self.question_understand_agent.last_candidate_store
        if getattr(candidate_store, "structured_store", None) is structured_store:
            candidate_store.structured_store = None

    def _has_doc_context(self, task: PublicTask) -> bool:
        return (task.context_dir / "doc").is_dir()

    def _top_level_steps(
        self,
        *step_groups: tuple[str, list[StepRecord]],
    ) -> list[StepRecord]:
        steps: list[StepRecord] = []
        next_step_index = 1
        for agent_name, group_steps in step_groups:
            for step in group_steps:
                steps.append(
                    replace(
                        step,
                        step_index=next_step_index,
                        agent_state={"agent_name": agent_name},
                    )
                )
                next_step_index += 1
        return steps
