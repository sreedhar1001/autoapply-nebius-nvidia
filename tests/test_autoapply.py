"""Automated test suite for AutoApply AI (Nebius x NVIDIA Edition)."""

from __future__ import annotations

from pathlib import Path

from src.agents.ats_checker import run_ats_audit
from src.agents.workflow import AutoApplyWorkflow
from src.llm.nebius_client import MODEL_TIERS, NebiusNemotronClient
from src.llm.nebius_router import ModelRouter
from src.tools.resume_loader import SAMPLE_RESUMES, extract_text_from_upload


def test_three_tier_model_routing_configuration() -> None:
    """Verify Nano, Super, and Ultra tiers map to NVIDIA Nemotron models with exact configs."""
    router = ModelRouter(api_key="", enable_cache=True)

    fast_cfg = router.get_tier_config("fast")
    assert "Nemotron" in fast_cfg.model_id
    assert fast_cfg.temperature == 0.1
    assert fast_cfg.max_tokens == 512

    struct_cfg = router.get_tier_config("structured")
    assert "Nemotron" in struct_cfg.model_id
    assert struct_cfg.temperature == 0.2
    assert struct_cfg.max_tokens == 2048

    reason_cfg = router.get_tier_config("reasoning")
    assert "Nemotron" in reason_cfg.model_id
    assert reason_cfg.temperature == 0.6
    assert reason_cfg.max_tokens == 4096


def test_telemetry_and_caching_on_router() -> None:
    """Verify telemetry logging, cache replay, and credit savings computation."""
    client = NebiusNemotronClient(api_key="", enable_cache=True)
    messages = [{"role": "user", "content": "Extract fast tags from Python ML resume"}]

    res1 = client.route_completion(
        task_type="fast",
        messages=messages,
        step_name="Unit Test Fast Step",
        fallback_generator=lambda: '{"tags": ["Python", "Nemotron"]}',
    )
    assert "Nemotron" in res1
    assert len(client.telemetry_events) == 1
    assert client.telemetry_events[0].cached is False
    assert client.telemetry_events[0].tier_short == "Nano"

    # Second identical call should hit the deterministic response cache
    res2 = client.route_completion(
        task_type="fast",
        messages=messages,
        step_name="Unit Test Fast Step Cached",
        fallback_generator=lambda: '{"tags": ["Python", "Nemotron"]}',
    )
    assert res2 == res1
    assert len(client.telemetry_events) == 2
    assert client.telemetry_events[1].cached is True

    summary = client.get_telemetry_summary()
    assert summary["total_calls"] == 2
    assert summary["tier_counts"]["Nano"] == 2
    assert summary["credit_savings_pct"] > 80.0
    assert summary["last_request_trace"] is not None


def test_full_four_node_multi_agent_workflow() -> None:
    """Verify the 4-node AutoApplyWorkflow executes Nano, Super, and Ultra steps end-to-end."""
    client = NebiusNemotronClient(api_key="", enable_cache=False)
    workflow = AutoApplyWorkflow(client=client)

    sample_resume = list(SAMPLE_RESUMES.values())[0]
    completed_steps = []

    state = workflow.run_full_pipeline(
        raw_resume_text=sample_resume,
        target_role_override="Senior AI Inference Engineer",
        target_location="San Francisco, CA",
        on_step_complete=lambda idx, label, st: completed_steps.append((idx, label)),
    )

    assert len(completed_steps) == 4
    assert state.candidate_profile.get("name")
    assert len(state.discovered_jobs) >= 1
    assert state.active_company_intel.get("company")
    assert 0 <= state.match_evaluation.get("overall_match_score", 0) <= 100
    assert len(state.tailoring_package.get("tailored_cover_letter", "")) > 100
    assert len(state.tailoring_package.get("resume_bullet_rewrites", [])) >= 2
    assert state.ats_report.get("overall_ats_score_after", 0) >= state.ats_report.get(
        "overall_ats_score_before", 0
    )
    assert len(state.ats_report.get("keyword_matrix", [])) >= 5

    summary = client.get_telemetry_summary()
    assert summary["tier_counts"]["Nano"] >= 3
    assert summary["tier_counts"]["Super"] >= 3
    assert summary["tier_counts"]["Ultra"] >= 1
    assert summary["credit_savings_pct"] > 45.0


def test_job_description_ats_checker() -> None:
    """Verify the Job Description ATS checker extracts keywords, hygiene metrics, and remediation."""
    client = NebiusNemotronClient(api_key="", enable_cache=False)
    sample_resume = list(SAMPLE_RESUMES.values())[0]
    custom_jd = (
        "Seeking a Senior AI Engineer experienced in Python, NVIDIA Nemotron, OpenAI SDK, "
        "FastAPI, Kubernetes, RAG, Multi-Agent systems, and Telemetry."
    )
    report = run_ats_audit(
        client=client,
        raw_resume_text=sample_resume,
        job_title="Senior AI Engineer",
        company_name="Nebius AI Studio",
        job_description=custom_jd,
    )
    assert 0 < report["overall_ats_score_before"] <= 100
    assert report["overall_ats_score_after"] >= report["overall_ats_score_before"]
    assert report["hygiene"]["formatting_score"] > 60
    assert any(row["keyword"] == "Python" for row in report["keyword_matrix"])
    assert len(report["top_remediation_steps"]) >= 2


def test_resume_text_loader() -> None:
    """Verify plain-text/Markdown upload extraction."""
    raw = b"JANE DOE\nSenior AI Engineer\nSkills: Python, NVIDIA Nemotron"
    extracted = extract_text_from_upload(raw, "resume.txt")
    assert "JANE DOE" in extracted
    assert "NVIDIA Nemotron" in extracted


def test_hackathon_compliance_rules() -> None:
    """Verify LICENSE, .env.example, and zero proprietary model strings in active code."""
    root = Path(__file__).resolve().parent.parent
    assert (root / "LICENSE").exists()
    assert (root / ".env.example").exists()

    env_text = (root / ".env.example").read_text(encoding="utf-8")
    for required_var in [
        "NEBIUS_API_KEY",
        "NEBIUS_BASE_URL",
        "NEBIUS_MODEL_NANO",
        "NEBIUS_MODEL_SUPER",
        "NEBIUS_MODEL_ULTRA",
        "ADZUNA_APP_ID",
        "ADZUNA_API_KEY",
        "TAVILY_API_KEY",
    ]:
        assert required_var in env_text

    # Scan application source files for forbidden proprietary model IDs
    forbidden_prefixes = ("gpt-4o", "gpt-3.5-turbo", "claude-", "gemini-")
    code_files = list((root / "src").rglob("*.py")) + [root / "app.py"]
    for file_path in code_files:
        content = file_path.read_text(encoding="utf-8").lower()
        for forbidden in forbidden_prefixes:
            assert forbidden not in content, f"Forbidden model string '{forbidden}' found in {file_path}"
