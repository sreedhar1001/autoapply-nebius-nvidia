"""Centralized ModelRouter for dynamic Three-Tier NVIDIA Nemotron routing on Nebius Token Factory."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from src.llm.nebius_client import (
    MODEL_TIERS,
    InferenceTelemetryEvent,
    NebiusNemotronClient,
    TaskType,
    TierConfig,
)


class ModelRouter(NebiusNemotronClient):
    """Centralized model router selecting NVIDIA Nemotron tier based on task complexity.

    Tiers:
      - Tier 1 (fast):       NEBIUS_MODEL_NANO  (temperature=0.1, max_tokens=512)
      - Tier 2 (structured): NEBIUS_MODEL_SUPER (temperature=0.2, max_tokens=2048)
      - Tier 3 (reasoning):  NEBIUS_MODEL_ULTRA (temperature=0.6, max_tokens=4096)
    """

    def route(
        self,
        task_type: TaskType,
        messages: List[Dict[str, str]],
        response_format: Optional[Dict[str, Any]] = None,
        step_name: str = "Agent Step",
        fallback_generator: Optional[Callable[[], str]] = None,
    ) -> str:
        """Alias for route_completion for convenient invocation."""
        return self.route_completion(
            task_type=task_type,
            messages=messages,
            response_format=response_format,
            step_name=step_name,
            fallback_generator=fallback_generator,
        )


__all__ = [
    "MODEL_TIERS",
    "InferenceTelemetryEvent",
    "ModelRouter",
    "NebiusNemotronClient",
    "TaskType",
    "TierConfig",
]
