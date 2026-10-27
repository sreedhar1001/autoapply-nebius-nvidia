"""Nebius Token Factory Unified Client & Three-Tier NVIDIA Nemotron Router.

Routes all LLM inference exclusively through Nebius Token Factory
(https://api.studio.nebius.ai/v1/) using open-source NVIDIA Nemotron models:
  - Tier 1 (fast):       nvidia/Llama-3_1-Nemotron-Nano-8B-v1
  - Tier 2 (structured): nvidia/Llama-3_3-Nemotron-Super-49B-v1
  - Tier 3 (reasoning):  nvidia/Llama-3_1-Nemotron-Ultra-253B-v1
"""

from __future__ import annotations

import dataclasses
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import time
from typing import Any, Callable, Dict, List, Literal, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

TaskType = Literal["fast", "structured", "reasoning"]

DEFAULT_NEBIUS_BASE_URL = "https://api.studio.nebius.ai/v1/"
DEFAULT_MODEL_NANO = "nvidia/Llama-3_1-Nemotron-Nano-8B-v1"
DEFAULT_MODEL_SUPER = "nvidia/Llama-3_3-Nemotron-Super-49B-v1"
DEFAULT_MODEL_ULTRA = "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1"


@dataclasses.dataclass(frozen=True)
class TierConfig:
    """Configuration for a specific NVIDIA Nemotron model tier on Nebius Token Factory."""

    task_type: TaskType
    tier_label: str
    short_name: str
    env_var: str
    default_model_id: str
    temperature: float
    max_tokens: int
    relative_credit_weight: float
    description: str

    @property
    def model_id(self) -> str:
        """Resolve the active Nemotron model ID from environment or default."""
        val = os.getenv(self.env_var, "").strip()
        return val if val else self.default_model_id


MODEL_TIERS: Dict[TaskType, TierConfig] = {
    "fast": TierConfig(
        task_type="fast",
        tier_label="Tier 1 — Fast / Low-Latency (Nano)",
        short_name="Nano",
        env_var="NEBIUS_MODEL_NANO",
        default_model_id=DEFAULT_MODEL_NANO,
        temperature=0.1,
        max_tokens=512,
        relative_credit_weight=1.0,
        description="Intent routing, keyword extraction, quick skill tag matching, guardrails",
    ),
    "structured": TierConfig(
        task_type="structured",
        tier_label="Tier 2 — Balanced / Extraction (Super)",
        short_name="Super",
        env_var="NEBIUS_MODEL_SUPER",
        default_model_id=DEFAULT_MODEL_SUPER,
        temperature=0.2,
        max_tokens=2048,
        relative_credit_weight=4.5,
        description="Structured JSON resume parsing, Adzuna job scoring, RAG & Tavily synthesis",
    ),
    "reasoning": TierConfig(
        task_type="reasoning",
        tier_label="Tier 3 — Deep Reasoning & Synthesis (Ultra)",
        short_name="Ultra",
        env_var="NEBIUS_MODEL_ULTRA",
        default_model_id=DEFAULT_MODEL_ULTRA,
        temperature=0.6,
        max_tokens=4096,
        relative_credit_weight=18.0,
        description="Tailored cover letters, resume bullet rewrites, gap analysis, interview strategy",
    ),
}


@dataclasses.dataclass
class InferenceTelemetryEvent:
    """Single inference call telemetry record for the Nebius Token Factory monitor."""

    timestamp: str
    step_name: str
    task_type: TaskType
    model_tier: str
    tier_short: str
    model_id: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    tokens_per_sec: float
    cached: bool
    execution_mode: str  # "nebius_live" | "cached_replay" | "demo_fallback"
    request_trace: Dict[str, Any]
    response_preview: str

    @property
    def token_usage(self) -> Dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "step_name": self.step_name,
            "task_type": self.task_type,
            "model_tier": self.model_tier,
            "tier_short": self.tier_short,
            "model_id": self.model_id,
            "latency_ms": round(self.latency_ms, 1),
            "token_usage": self.token_usage,
            "tokens_per_sec": round(self.tokens_per_sec, 1),
            "cached": self.cached,
            "execution_mode": self.execution_mode,
            "request_trace": self.request_trace,
            "response_preview": self.response_preview,
        }


class NebiusNemotronClient:
    """Unified OpenAI-compatible client for NVIDIA Nemotron models on Nebius Token Factory."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_calls_per_session: Optional[int] = None,
        enable_cache: Optional[bool] = None,
    ) -> None:
        raw_url = (
            base_url
            or os.getenv("NEBIUS_BASE_URL", DEFAULT_NEBIUS_BASE_URL)
            or DEFAULT_NEBIUS_BASE_URL
        ).strip()
        # Clean markdown link syntax if copied from a template file
        if raw_url.startswith("[") and "](" in raw_url:
            match = re.search(r"\]\((https?://[^\)]+)\)", raw_url)
            if match:
                raw_url = match.group(1)

        self.base_url: str = raw_url
        self.api_key: str = (
            api_key if api_key is not None else os.getenv("NEBIUS_API_KEY", "")
        ).strip()

        self.max_calls_per_session: int = (
            max_calls_per_session
            if max_calls_per_session is not None
            else int(os.getenv("MAX_CALLS_PER_SESSION", "40"))
        )
        self.enable_cache: bool = (
            enable_cache
            if enable_cache is not None
            else os.getenv("ENABLE_RESPONSE_CACHE", "true").lower() in ("1", "true", "yes")
        )

        self._telemetry_log: List[InferenceTelemetryEvent] = []
        self._response_cache: Dict[str, Dict[str, Any]] = {}
        self._session_call_count: int = 0

        self._client: Optional[OpenAI] = None
        if self.has_valid_api_key:
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
            )

    @property
    def has_valid_api_key(self) -> bool:
        """Check whether a non-placeholder Nebius API key is configured."""
        placeholders = {
            "",
            "your_nebius_token_factory_api_key",
            "your-api-key",
            "sk-placeholder",
        }
        return self.api_key not in placeholders and len(self.api_key) > 8

    def configure(self, api_key: str, base_url: Optional[str] = None) -> None:
        """Dynamically update Nebius API credentials at runtime (e.g. from Streamlit UI)."""
        self.api_key = api_key.strip()
        if base_url:
            self.base_url = base_url.strip()
        if self.has_valid_api_key:
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        else:
            self._client = None

    def get_tier_config(self, task_type: TaskType) -> TierConfig:
        """Return the Nemotron tier configuration for the requested task complexity."""
        if task_type not in MODEL_TIERS:
            raise ValueError(
                f"Invalid task_type '{task_type}'. Must be one of: {list(MODEL_TIERS.keys())}"
            )
        return MODEL_TIERS[task_type]

    @staticmethod
    def _compute_cache_key(
        task_type: TaskType,
        model_id: str,
        messages: List[Dict[str, str]],
        response_format: Optional[Dict[str, Any]],
    ) -> str:
        serialized = json.dumps(
            {
                "task_type": task_type,
                "model_id": model_id,
                "messages": messages,
                "response_format": response_format,
            },
            sort_keys=True,
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @staticmethod
    def _strip_reasoning_tags(text: str) -> str:
        """Strip <think>...</think> blocks if emitted by reasoning Nemotron models."""
        if "<think>" in text and "</think>" in text:
            cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            if cleaned:
                return cleaned
        return text.strip()

    def route_completion(
        self,
        task_type: TaskType,
        messages: List[Dict[str, str]],
        response_format: Optional[Dict[str, Any]] = None,
        step_name: str = "Agent Step",
        fallback_generator: Optional[Callable[[], str]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Route a completion request to the appropriate NVIDIA Nemotron tier on Nebius.

        Args:
            task_type: "fast" (Nano), "structured" (Super), or "reasoning" (Ultra).
            messages: OpenAI-compatible chat message list.
            response_format: Optional response format (e.g., {"type": "json_object"}).
            step_name: Human-readable pipeline step name for telemetry badges.
            fallback_generator: Deterministic fallback callable for offline/rate-limited protection.
            temperature: Optional override for tier default temperature.
            max_tokens: Optional override for tier default max_tokens.

        Returns:
            Model completion content string.
        """
        tier = self.get_tier_config(task_type)
        active_model = tier.model_id
        active_temp = temperature if temperature is not None else tier.temperature
        active_max_tokens = max_tokens if max_tokens is not None else tier.max_tokens

        request_payload: Dict[str, Any] = {
            "provider": "Nebius Token Factory",
            "base_url": self.base_url,
            "model": active_model,
            "tier": tier.tier_label,
            "task_type": task_type,
            "temperature": active_temp,
            "max_tokens": active_max_tokens,
            "messages": messages,
        }
        if response_format:
            request_payload["response_format"] = response_format

        cache_key = self._compute_cache_key(task_type, active_model, messages, response_format)
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # 1. Check response cache first (Demo Protection)
        if self.enable_cache and cache_key in self._response_cache:
            cached_entry = self._response_cache[cache_key]
            event = InferenceTelemetryEvent(
                timestamp=now_iso,
                step_name=step_name,
                task_type=task_type,
                model_tier=tier.tier_label,
                tier_short=tier.short_name,
                model_id=active_model,
                latency_ms=4.2,
                prompt_tokens=cached_entry["prompt_tokens"],
                completion_tokens=cached_entry["completion_tokens"],
                total_tokens=cached_entry["total_tokens"],
                tokens_per_sec=cached_entry["tokens_per_sec"],
                cached=True,
                execution_mode="cached_replay",
                request_trace=request_payload,
                response_preview=cached_entry["content"][:500],
            )
            self._telemetry_log.append(event)
            return cached_entry["content"]

        # 2. Check session rate limit & live API availability
        within_rate_limit = self._session_call_count < self.max_calls_per_session
        if self._client is not None and within_rate_limit:
            start_perf = time.perf_counter()
            try:
                api_kwargs: Dict[str, Any] = {
                    "model": active_model,
                    "messages": messages,
                    "temperature": active_temp,
                    "max_tokens": active_max_tokens,
                }
                if response_format is not None:
                    api_kwargs["response_format"] = response_format

                completion = self._client.chat.completions.create(**api_kwargs)
                elapsed_ms = (time.perf_counter() - start_perf) * 1000.0
                self._session_call_count += 1

                raw_content = completion.choices[0].message.content or ""
                content = self._strip_reasoning_tags(raw_content)

                usage = getattr(completion, "usage", None)
                prompt_toks = getattr(usage, "prompt_tokens", 0) or max(
                    1, sum(len(m.get("content", "")) for m in messages) // 4
                )
                comp_toks = getattr(usage, "completion_tokens", 0) or max(1, len(content) // 4)
                total_toks = getattr(usage, "total_tokens", 0) or (prompt_toks + comp_toks)
                tps = comp_toks / max(elapsed_ms / 1000.0, 0.001)

                if self.enable_cache:
                    self._response_cache[cache_key] = {
                        "content": content,
                        "prompt_tokens": prompt_toks,
                        "completion_tokens": comp_toks,
                        "total_tokens": total_toks,
                        "tokens_per_sec": tps,
                    }

                event = InferenceTelemetryEvent(
                    timestamp=now_iso,
                    step_name=step_name,
                    task_type=task_type,
                    model_tier=tier.tier_label,
                    tier_short=tier.short_name,
                    model_id=active_model,
                    latency_ms=elapsed_ms,
                    prompt_tokens=prompt_toks,
                    completion_tokens=comp_toks,
                    total_tokens=total_toks,
                    tokens_per_sec=tps,
                    cached=False,
                    execution_mode="nebius_live",
                    request_trace=request_payload,
                    response_preview=content[:500],
                )
                self._telemetry_log.append(event)
                return content
            except Exception as exc:
                # If response_format={"type": "json_object"} was rejected by a specific endpoint,
                # retry once without response_format before falling back
                if response_format is not None:
                    try:
                        retry_start = time.perf_counter()
                        completion = self._client.chat.completions.create(
                            model=active_model,
                            messages=messages,
                            temperature=active_temp,
                            max_tokens=active_max_tokens,
                        )
                        elapsed_ms = (time.perf_counter() - retry_start) * 1000.0
                        self._session_call_count += 1
                        raw_content = completion.choices[0].message.content or ""
                        content = self._strip_reasoning_tags(raw_content)

                        usage = getattr(completion, "usage", None)
                        prompt_toks = getattr(usage, "prompt_tokens", 0) or max(
                            1, sum(len(m.get("content", "")) for m in messages) // 4
                        )
                        comp_toks = getattr(usage, "completion_tokens", 0) or max(
                            1, len(content) // 4
                        )
                        total_toks = prompt_toks + comp_toks
                        tps = comp_toks / max(elapsed_ms / 1000.0, 0.001)

                        if self.enable_cache:
                            self._response_cache[cache_key] = {
                                "content": content,
                                "prompt_tokens": prompt_toks,
                                "completion_tokens": comp_toks,
                                "total_tokens": total_toks,
                                "tokens_per_sec": tps,
                            }

                        event = InferenceTelemetryEvent(
                            timestamp=now_iso,
                            step_name=step_name,
                            task_type=task_type,
                            model_tier=tier.tier_label,
                            tier_short=tier.short_name,
                            model_id=active_model,
                            latency_ms=elapsed_ms,
                            prompt_tokens=prompt_toks,
                            completion_tokens=comp_toks,
                            total_tokens=total_toks,
                            tokens_per_sec=tps,
                            cached=False,
                            execution_mode="nebius_live",
                            request_trace=request_payload,
                            response_preview=content[:500],
                        )
                        self._telemetry_log.append(event)
                        return content
                    except Exception:
                        pass
                request_payload["fallback_reason"] = f"Nebius API fallback activated: {type(exc).__name__}"
        else:
            reason = (
                "Session rate limit reached"
                if not within_rate_limit
                else "NEBIUS_API_KEY not configured (Demo Protection Mode)"
            )
            request_payload["fallback_reason"] = reason

        # 3. Demo Protection Fallback Execution
        start_sim = time.perf_counter()
        if fallback_generator is not None:
            content = fallback_generator()
        else:
            content = json.dumps({"status": "ok", "model": active_model, "tier": tier.short_name})

        # Simulate realistic tier-proportional latency so UI telemetry demonstrates Nano vs Super vs Ultra characteristics
        simulated_latencies = {"fast": 185.0, "structured": 640.0, "reasoning": 1420.0}
        actual_sim_ms = (time.perf_counter() - start_sim) * 1000.0
        latency_ms = max(actual_sim_ms, simulated_latencies.get(task_type, 400.0))

        prompt_toks = max(24, sum(len(m.get("content", "")) for m in messages) // 4)
        comp_toks = max(16, len(content) // 4)
        total_toks = prompt_toks + comp_toks
        tps = comp_toks / (latency_ms / 1000.0)

        if self.enable_cache:
            self._response_cache[cache_key] = {
                "content": content,
                "prompt_tokens": prompt_toks,
                "completion_tokens": comp_toks,
                "total_tokens": total_toks,
                "tokens_per_sec": tps,
            }

        event = InferenceTelemetryEvent(
            timestamp=now_iso,
            step_name=step_name,
            task_type=task_type,
            model_tier=tier.tier_label,
            tier_short=tier.short_name,
            model_id=active_model,
            latency_ms=latency_ms,
            prompt_tokens=prompt_toks,
            completion_tokens=comp_toks,
            total_tokens=total_toks,
            tokens_per_sec=tps,
            cached=False,
            execution_mode="demo_fallback",
            request_trace=request_payload,
            response_preview=content[:500],
        )
        self._telemetry_log.append(event)
        return content

    @property
    def telemetry_events(self) -> List[InferenceTelemetryEvent]:
        """Return all logged inference telemetry events in chronological order."""
        return list(self._telemetry_log)

    def clear_telemetry(self) -> None:
        """Clear logged telemetry events for a fresh run."""
        self._telemetry_log.clear()

    def get_telemetry_summary(self) -> Dict[str, Any]:
        """Compute live Nebius Token Factory multi-tier routing telemetry & cost savings."""
        tier_counts = {"Nano": 0, "Super": 0, "Ultra": 0}
        tier_tokens = {"Nano": 0, "Super": 0, "Ultra": 0}
        tier_latencies: Dict[str, List[float]] = {"Nano": [], "Super": [], "Ultra": []}

        actual_weighted_cost = 0.0
        all_ultra_weighted_cost = 0.0
        ultra_weight = MODEL_TIERS["reasoning"].relative_credit_weight

        for ev in self._telemetry_log:
            short = ev.tier_short
            tier_counts[short] = tier_counts.get(short, 0) + 1
            tier_tokens[short] = tier_tokens.get(short, 0) + ev.total_tokens
            tier_latencies.setdefault(short, []).append(ev.latency_ms)

            weight = MODEL_TIERS[ev.task_type].relative_credit_weight
            actual_weighted_cost += ev.total_tokens * weight
            all_ultra_weighted_cost += ev.total_tokens * ultra_weight

        if all_ultra_weighted_cost > 0:
            savings_pct = max(
                0.0,
                (1.0 - (actual_weighted_cost / all_ultra_weighted_cost)) * 100.0,
            )
        else:
            savings_pct = 0.0

        avg_latency_by_tier = {
            k: round(sum(v) / len(v), 1) if v else 0.0 for k, v in tier_latencies.items()
        }

        last_event = self._telemetry_log[-1].to_dict() if self._telemetry_log else None

        return {
            "total_calls": len(self._telemetry_log),
            "session_live_calls": self._session_call_count,
            "max_calls_per_session": self.max_calls_per_session,
            "tier_counts": tier_counts,
            "tier_tokens": tier_tokens,
            "avg_latency_ms_by_tier": avg_latency_by_tier,
            "total_tokens": sum(tier_tokens.values()),
            "credit_savings_pct": round(savings_pct, 1),
            "is_live_api": self.has_valid_api_key,
            "base_url": self.base_url,
            "active_models": {
                "Nano": MODEL_TIERS["fast"].model_id,
                "Super": MODEL_TIERS["structured"].model_id,
                "Ultra": MODEL_TIERS["reasoning"].model_id,
            },
            "last_request_trace": last_event,
            "events": [e.to_dict() for e in self._telemetry_log],
        }
