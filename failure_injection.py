"""P1.3 的测试专用故障注入层；默认关闭，不改变生产工具行为。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


FAILURE_TYPES = {
    "planner_failure", "execution_failure", "retrieval_failure",
    "source_validation_failure", "evidence_gap", "generation_grounding_failure",
    "semantic_judge_failure", "timeout", "empty_result", "invalid_input",
    "unknown_tool", "partial_tool_failure", "no_relation", "memory_failure",
}


@dataclass
class FailureInjectionConfig:
    """按目标组件注入故障，如 ``{"web_search": "timeout"}``。"""

    failures: dict[str, str | dict[str, Any]] = field(default_factory=dict)

    def for_target(self, target: str) -> dict[str, Any] | None:
        raw = self.failures.get(target)
        if raw is None:
            return None
        if isinstance(raw, str):
            raw = {"failure_type": raw}
        failure_type = raw.get("failure_type", "execution_failure")
        if failure_type not in FAILURE_TYPES:
            failure_type = "execution_failure"
        return {
            "failure_type": failure_type,
            "message": raw.get("message") or f"测试注入：{target} {failure_type}",
            "recoverable": bool(raw.get("recoverable", True)),
            "details": dict(raw.get("details") or {}),
        }


def injected_failure(config: FailureInjectionConfig | dict[str, Any] | None, target: str) -> dict[str, Any] | None:
    """兼容 dataclass 与 dict，方便现有测试通过 ConversationState 注入。"""
    if config is None:
        return None
    if isinstance(config, dict):
        config = FailureInjectionConfig(config)
    return config.for_target(target)
