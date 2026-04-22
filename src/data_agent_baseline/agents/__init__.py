from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "AgentRunResult": "data_agent_baseline.agents.runtime",
    "AgentRuntimeState": "data_agent_baseline.agents.runtime",
    "ExecutionAgent": "data_agent_baseline.agents.execution",
    "LongTextDocFactAgent": "data_agent_baseline.agents.long_text_doc",
    "LongTextDocFactOutput": "data_agent_baseline.agents.long_text_doc",
    "ModelMessage": "data_agent_baseline.agents.model",
    "MultiAgentOrchestrator": "data_agent_baseline.agents.orchestrator",
    "OpenAIModelAdapter": "data_agent_baseline.agents.model",
    "QuestionUnderstandAgent": "data_agent_baseline.agents.understanding",
    "QuestionUnderstandingOutput": "data_agent_baseline.agents.shared",
    "REACT_SYSTEM_PROMPT": "data_agent_baseline.agents.prompt",
    "ReActAgent": "data_agent_baseline.agents.react",
    "ReActAgentConfig": "data_agent_baseline.agents.react",
    "StepRecord": "data_agent_baseline.agents.runtime",
    "build_observation_prompt": "data_agent_baseline.agents.prompt",
    "build_system_prompt": "data_agent_baseline.agents.prompt",
    "build_task_prompt": "data_agent_baseline.agents.prompt",
    "parse_model_step": "data_agent_baseline.agents.react",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
