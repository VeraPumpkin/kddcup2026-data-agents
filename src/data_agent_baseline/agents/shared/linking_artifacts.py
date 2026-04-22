from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from data_agent_baseline.agents.shared.utils import _json_compatible


@dataclass(frozen=True, slots=True)
class QuestionUnderstandingOutput:
    semantic_plan: dict[str, Any]
    remaining_steps: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "semantic_plan": _json_compatible(self.semantic_plan),
            "remaining_steps": self.remaining_steps,
        }
