"""AutoApply AI (Nebius x NVIDIA Edition) — Streamlit Multi-Agent Career Orchestration Platform.

Powered 100% by open-source NVIDIA Nemotron models (Nano, Super, Ultra) on Nebius Token Factory
with live job discovery (Adzuna) and company engineering intelligence (Tavily).
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

import streamlit as st

from src.agents.ats_checker import run_ats_audit
from src.agents.workflow import AutoApplyWorkflow, WorkflowState
from src.llm.nebius_client import MODEL_TIERS, NebiusNemotronClient
from src.tools.resume_loader import SAMPLE_RESUMES, extract_text_from_upload

st.set_page_config(
    page_title="AutoApply AI — Nebius x NVIDIA Nemotron Edition",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Custom CSS for NVIDIA Green + Nebius Studio Dark Theme
# -----------------------------------------------------------------------------
st.markdown(
    """
<style>
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2.5rem;
    }
    .hero-banner {
        background: linear-gradient(135deg, #0b132b 0%, #111b2e 55%, #1a2c12 100%);
        border: 1px solid #76B900;
        border-radius: 14px;
        padding: 1.4rem 1.8rem;
        margin-bottom: 1.2rem;
    }
    .hero-title {
        font-size: 1.9rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0 0 0.35rem 0;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        font-size: 0.98rem;
        color: #cbd5e1;
        margin: 0;
    }
    .badge-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 0.85rem;
    }
    .pill-badge {
        display: inline-block;
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .pill-nvidia {
        background: rgba(118, 185, 0, 0.18);
        color: #9be61d;
        border: 1px solid #76B900;
    }
    .pill-nebius {
        background: rgba(56, 189, 248, 0.16);
        color: #7dd3fc;
        border: 1px solid #38bdf8;
    }
    .pill-tavily {
        background: rgba(168, 85, 247, 0.16);
        color: #d8b4fe;
        border: 1px solid #a855f7;
    }
    .tier-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.6rem;
    }
    .kw-match {
        display: inline-block;
        background: rgba(34, 197, 94, 0.15);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.45);
        border-radius: 6px;
        padding: 0.2rem 0.55rem;
        margin: 0.2rem;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .kw-gap {
        display: inline-block;
        background: rgba(249, 115, 22, 0.16);
        color: #fb923c;
        border: 1px solid rgba(249, 115, 22, 0.45);
        border-radius: 6px;
        padding: 0.2rem 0.55rem;
        margin: 0.2rem;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .rewrite-box-before {
        background: rgba(239, 68, 68, 0.08);
        border-left: 4px solid #ef4444;
        padding: 0.75rem 1rem;
        border-radius: 6px;
        margin-bottom: 0.4rem;
    }
    .rewrite-box-after {
        background: rgba(118, 185, 0, 0.12);
        border-left: 4px solid #76B900;
        padding: 0.75rem 1rem;
        border-radius: 6px;
        margin-bottom: 0.4rem;
    }
    .telemetry-table {
        width: 100%;
        border-collapse: collapse;
        margin: 0.8rem 0 1.2rem 0;
        font-size: 0.86rem;
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 8px;
        overflow: hidden;
    }
    .telemetry-table th {
        background: #1e293b;
        color: #f8fafc;
        text-align: left;
        padding: 0.65rem 0.85rem;
        font-weight: 700;
        border-bottom: 1px solid #334155;
    }
    .telemetry-table td {
        padding: 0.6rem 0.85rem;
        border-bottom: 1px solid #1e293b;
        color: #e2e8f0;
    }
    .telemetry-table tr:hover td {
        background: rgba(56, 189, 248, 0.06);
    }
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
def _init_session_state() -> None:
    if "nebius_client" not in st.session_state:
        st.session_state.nebius_client = NebiusNemotronClient()
    if "workflow_state" not in st.session_state:
        st.session_state.workflow_state = None
    if "adzuna_app_id" not in st.session_state:
        st.session_state.adzuna_app_id = os.getenv("ADZUNA_APP_ID", "")
    if "adzuna_api_key" not in st.session_state:
        st.session_state.adzuna_api_key = os.getenv("ADZUNA_API_KEY", "")
    if "tavily_api_key" not in st.session_state:
        st.session_state.tavily_api_key = os.getenv("TAVILY_API_KEY", "")


_init_session_state()
client: NebiusNemotronClient = st.session_state.nebius_client


# -----------------------------------------------------------------------------
# Sidebar: Nebius x NVIDIA Live Inference Monitor & API Configuration
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚡ Nebius x NVIDIA Live Inference Monitor")
    telemetry = client.get_telemetry_summary()

    if telemetry["is_live_api"]:
        st.success("🟢 **Nebius Token Factory: LIVE API**")
    else:
        st.info(
            "🛡️ **Demo Protection Mode Active**\n\n"
            "Add `NEBIUS_API_KEY` below or in `.env` for live cloud calls. "
            "Deterministic Nemotron routing & caching are active."
        )

    col_n, col_s, col_u = st.columns(3)
    col_n.metric("⚡ Nano", telemetry["tier_counts"].get("Nano", 0))
    col_s.metric("🧩 Super", telemetry["tier_counts"].get("Super", 0))
    col_u.metric("🧠 Ultra", telemetry["tier_counts"].get("Ultra", 0))

    col_sav, col_tok = st.columns(2)
    col_sav.metric(
        "💰 Credit Savings",
        f"{telemetry['credit_savings_pct']}%",
        help="Credit savings achieved by routing fast/structured tasks to Nemotron Nano & Super instead of running 100% of tokens on Nemotron Ultra.",
    )
    col_tok.metric("🔢 Total Tokens", f"{telemetry['total_tokens']:,}")

    st.caption(
        f"Session Rate Guard: **{telemetry['session_live_calls']} / {telemetry['max_calls_per_session']}** live calls used"
    )

    with st.expander("📊 Active Three-Tier Model Routing Map", expanded=False):
        for t_key, t_cfg in MODEL_TIERS.items():
            avg_lat = telemetry["avg_latency_ms_by_tier"].get(t_cfg.short_name, 0.0)
            st.markdown(
                f"**{t_cfg.short_name}** (`{t_key}`)\n"
                f"- Model: `{t_cfg.model_id}`\n"
                f"- Temp: `{t_cfg.temperature}` | Max Tokens: `{t_cfg.max_tokens}`\n"
                f"- Avg Latency: `{avg_lat} ms`"
            )

    with st.expander("🔍 Live JSON Trace (Last Nebius Request)", expanded=True):
        if telemetry["last_request_trace"]:
            st.json(telemetry["last_request_trace"])
        else:
            st.caption("Run the 4-Agent pipeline to inspect live Nebius Token Factory JSON traces.")

    st.divider()
    with st.expander("🔑 API Credentials & Demo Controls", expanded=False):
        input_nebius_key = st.text_input(
            "NEBIUS_API_KEY",
            value=client.api_key if client.has_valid_api_key else "",
            type="password",
            placeholder="Enter Nebius Token Factory key...",
        )
        input_tavily_key = st.text_input(
            "TAVILY_API_KEY (Bonus Track)",
            value=st.session_state.tavily_api_key
            if st.session_state.tavily_api_key != "your_tavily_api_key"
            else "",
            type="password",
            placeholder="tvly-...",
        )
        input_adzuna_id = st.text_input(
            "ADZUNA_APP_ID",
            value=st.session_state.adzuna_app_id
            if st.session_state.adzuna_app_id != "your_adzuna_app_id"
            else "",
        )
        input_adzuna_key = st.text_input(
            "ADZUNA_API_KEY",
            value=st.session_state.adzuna_api_key
            if st.session_state.adzuna_api_key != "your_adzuna_api_key"
            else "",
            type="password",
        )
        if st.button("Save Credentials", use_container_width=True):
            client.configure(api_key=input_nebius_key)
            st.session_state.tavily_api_key = input_tavily_key.strip()
            st.session_state.adzuna_app_id = input_adzuna_id.strip()
            st.session_state.adzuna_api_key = input_adzuna_key.strip()
            st.toast("Updated Nebius, Tavily, and Adzuna credentials!", icon="✅")
            st.rerun()

        if st.button("Reset Telemetry Log", use_container_width=True):
            client.clear_telemetry()
            st.rerun()


# -----------------------------------------------------------------------------
# Hero Header
# -----------------------------------------------------------------------------
st.markdown(
    """
<div class="hero-banner">
    <div class="hero-title">⚡ AutoApply AI — Nebius x NVIDIA Nemotron Edition</div>
    <p class="hero-subtitle">
        Autonomous 4-Agent Career Orchestration Platform powered 100% by open-source
        <b>NVIDIA Nemotron (Nano 8B • Super 49B • Ultra 253B)</b> on <b>Nebius Token Factory</b>
        with live company engineering intelligence via <b>Tavily Search</b>.
    </p>
    <div class="badge-row">
        <span class="pill-badge pill-nvidia">🟢 Tier 1: Nemotron Nano 8B (Fast Triage & Query Planning)</span>
        <span class="pill-badge pill-nebius">🔵 Tier 2: Nemotron Super 49B (JSON Resume Parser & Match Scorer)</span>
        <span class="pill-badge pill-nvidia">🟢 Tier 3: Nemotron Ultra 253B (Grounded Cover Letter & Bullet Synthesis)</span>
        <span class="pill-badge pill-tavily">🟣 Tavily Live Company Engineering Intel + Adzuna Jobs</span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Top Input Panel: Candidate Resume & Target Job Settings
# -----------------------------------------------------------------------------
with st.container(border=True):
    st.subheader("1️⃣ Candidate Resume & Target Role Configuration")
    col_left, col_right = st.columns([3, 2])

    with col_left:
        input_mode = st.radio(
            "Resume Source",
            options=[
                "Curated Demo Persona (1-Click Test)",
                "Upload Resume File (PDF / TXT / MD)",
                "Paste Raw Resume Text",
            ],
            horizontal=True,
        )

        raw_resume_text = ""
        if input_mode == "Curated Demo Persona (1-Click Test)":
            persona_choice = st.selectbox(
                "Select Candidate Persona",
                options=list(SAMPLE_RESUMES.keys()),
            )
            raw_resume_text = st.text_area(
                "Resume Content (Editable)",
                value=SAMPLE_RESUMES[persona_choice],
                height=210,
            )
        elif input_mode == "Upload Resume File (PDF / TXT / MD)":
            uploaded = st.file_uploader(
                "Upload PDF, TXT, or Markdown Resume",
                type=["pdf", "txt", "md"],
            )
            if uploaded is not None:
                try:
                    raw_resume_text = extract_text_from_upload(uploaded.read(), uploaded.name)
                    st.success(f"Extracted {len(raw_resume_text):,} characters from `{uploaded.name}`.")
                    with st.expander("Preview Extracted Resume Text"):
                        st.text(raw_resume_text[:2000])
                except Exception as exc:
                    st.error(f"Could not parse uploaded file: {exc}")
        else:
            raw_resume_text = st.text_area(
                "Paste Your Full Resume Text",
                value=list(SAMPLE_RESUMES.values())[0],
                height=210,
            )

    with col_right:
        target_role_input = st.text_input(
            "Target Role Override (Optional — auto-detected by Nemotron Super if blank)",
            placeholder="e.g., Senior AI Inference & Agentic Systems Engineer",
        )
        target_location_input = st.text_input(
            "Target Location",
            value="San Francisco, CA / Remote",
        )
        with st.expander("Optional: Paste Specific Job Description & Company"):
            custom_company_input = st.text_input(
                "Hiring Company Name",
                placeholder="e.g., Nebius AI Studio",
            )
            custom_jd_input = st.text_area(
                "Custom Job Description",
                placeholder="Paste a specific job description here to evaluate against, or leave blank to discover jobs automatically...",
                height=100,
            )

        run_clicked = st.button(
            "🚀 Run 4-Agent Nemotron Pipeline on Nebius",
            type="primary",
            use_container_width=True,
        )

# -----------------------------------------------------------------------------
# Execute 4-Node Multi-Agent Pipeline on Button Click
# -----------------------------------------------------------------------------
if run_clicked:
    if not raw_resume_text.strip():
        st.warning("Please provide or select a candidate resume before launching the pipeline.")
    else:
        workflow = AutoApplyWorkflow(
            client=client,
            adzuna_app_id=st.session_state.adzuna_app_id,
            adzuna_api_key=st.session_state.adzuna_api_key,
            tavily_api_key=st.session_state.tavily_api_key,
        )

        progress_bar = st.progress(0, text="Initializing Nebius Token Factory Multi-Tier Router...")
        status_box = st.status(
            "Orchestrating 4-Node NVIDIA Nemotron Agent Workflow...", expanded=True
        )

        def _update_step(step_num: int, step_label: str, current_state: WorkflowState) -> None:
            pct = int((step_num / 4) * 100)
            progress_bar.progress(pct, text=f"Completed {step_label}")
            latest_ev = client.telemetry_events[-1] if client.telemetry_events else None
            if latest_ev:
                status_box.write(
                    f"✅ **{step_label}** → Routed to `{latest_ev.model_id}` "
                    f"(`{latest_ev.latency_ms:.0f} ms`, `{latest_ev.total_tokens} tokens`, `{latest_ev.tokens_per_sec:.1f} tok/s`)"
                )

        final_state = workflow.run_full_pipeline(
            raw_resume_text=raw_resume_text,
            target_role_override=target_role_input,
            target_location=target_location_input,
            custom_job_description=custom_jd_input,
            custom_company_name=custom_company_input,
            selected_job_index=0,
            on_step_complete=_update_step,
        )
        status_box.update(
            label="✅ 4-Node NVIDIA Nemotron Career Pipeline Complete!",
            state="complete",
            expanded=False,
        )
        st.session_state.workflow_state = final_state
        st.rerun()


# -----------------------------------------------------------------------------
# Render Results if Workflow State Exists
# -----------------------------------------------------------------------------
state: WorkflowState | None = st.session_state.workflow_state

if state is None:
    st.info(
        "👆 Click **🚀 Run 4-Agent Nemotron Pipeline on Nebius** above to execute the "
        "Parser (`Super`), Discovery & Tavily Intel (`Nano`), Match Evaluator (`Super`), "
        "and Tailoring & Cover Letter (`Ultra`) agents."
    )
else:
    profile: Dict[str, Any] = state.candidate_profile
    active_job: Dict[str, Any] = state.active_job
    intel: Dict[str, Any] = state.active_company_intel
    evaluation: Dict[str, Any] = state.match_evaluation
    tailoring: Dict[str, Any] = state.tailoring_package

    # Top Quick-Switch Bar for Discovered Jobs
    if len(state.discovered_jobs) > 1:
        with st.container(border=True):
            col_sel, col_btn = st.columns([4, 1])
            job_labels = [
                f"{idx + 1}. {j.get('title')} @ {j.get('company')} ({j.get('location')})"
                for idx, j in enumerate(state.discovered_jobs)
            ]
            with col_sel:
                chosen_idx = st.selectbox(
                    "🎯 Active Target Job Role (Switch to re-run Match Evaluator & Ultra Tailoring for another discovered role)",
                    options=list(range(len(job_labels))),
                    format_func=lambda i: job_labels[i],
                    index=state.selected_job_index,
                )
            with col_btn:
                st.write("")
                if st.button("Re-Tailor for Selected Job", use_container_width=True):
                    state.selected_job_index = chosen_idx
                    state.active_job = state.discovered_jobs[chosen_idx]
                    comp = state.active_job.get("company", "Nebius AI Studio")
                    state.active_company_intel = state.company_intel_map.get(comp, {})
                    wf = AutoApplyWorkflow(
                        client=client,
                        adzuna_app_id=st.session_state.adzuna_app_id,
                        adzuna_api_key=st.session_state.adzuna_api_key,
                        tavily_api_key=st.session_state.tavily_api_key,
                    )
                    wf.run_match_evaluator_agent(state)
                    wf.run_tailoring_agent(state)
                    st.session_state.workflow_state = state
                    st.rerun()

    tab_overview, tab_ats, tab_intel, tab_profile, tab_interview, tab_telemetry = st.tabs(
        [
            "🎯 Match Score, Rewrites & Cover Letter",
            "📋 ATS Checker & JD Scanner",
            "🌐 Adzuna Jobs & Tavily Company Intel",
            "👤 Parsed Candidate JSON (Nemotron Super)",
            "🎤 Interview & Gap Strategy (Nemotron Ultra)",
            "⚡ Nebius Token Factory Telemetry Proof",
        ]
    )

    # =========================================================================
    # TAB 1: Match Score, Keyword Gaps, Resume Bullet Rewrites & Cover Letter
    # =========================================================================
    with tab_overview:
        score = int(evaluation.get("overall_match_score", 85))
        dims = evaluation.get("dimension_scores", {})

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "🎯 Semantic Match Score",
            f"{score}%",
            delta="Evaluated by Nemotron Super",
        )
        m2.metric(
            "🛠️ Technical Alignment",
            f"{dims.get('technical_skills_alignment', score)}%",
        )
        m3.metric(
            "📈 Experience Depth",
            f"{dims.get('experience_and_impact_depth', score)}%",
        )
        m4.metric(
            "🏢 Company Stack Fit",
            f"{dims.get('company_architecture_fit', score)}%",
        )

        col_kw1, col_kw2 = st.columns(2)
        with col_kw1:
            with st.container(border=True):
                st.markdown("#### ✅ Matched Keywords & Core Strengths (`Nemotron Super`)")
                matched_kws = evaluation.get("matched_keywords", [])
                if matched_kws:
                    pills_html = "".join(
                        f'<span class="kw-match">✓ {kw}</span>' for kw in matched_kws
                    )
                    st.markdown(pills_html, unsafe_allow_html=True)
                for s_pt in evaluation.get("strengths_summary", []):
                    st.markdown(f"- {s_pt}")

        with col_kw2:
            with st.container(border=True):
                st.markdown("#### ⚠️ Missing Keywords & Gap Analysis (`Nemotron Super`)")
                missing_kws = evaluation.get("missing_keywords", [])
                if missing_kws:
                    gap_html = "".join(
                        f'<span class="kw-gap">+ {kw}</span>' for kw in missing_kws
                    )
                    st.markdown(gap_html, unsafe_allow_html=True)
                for g_pt in evaluation.get("gap_mitigation_advice", []):
                    st.markdown(f"- {g_pt}")

        st.divider()
        col_rewrites, col_letter = st.columns([1.15, 1])

        with col_rewrites:
            st.markdown(
                "### ✍️ Project-Grounded Resume Bullet Rewrites (`Nemotron Ultra 253B`)"
            )
            st.caption(
                "Strictly grounded in the candidate's actual projects and experience—zero fabricated roles."
            )
            for idx, item in enumerate(tailoring.get("resume_bullet_rewrites", []), start=1):
                with st.container(border=True):
                    st.markdown(
                        f"**#{idx} — Grounded in:** `{item.get('source_project_or_role', 'Project')}`"
                    )
                    st.markdown(
                        f'<div class="rewrite-box-before"><b>Before:</b> {item.get("before_bullet", "")}</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f'<div class="rewrite-box-after"><b>After (Nemotron Ultra):</b> {item.get("after_bullet", "")}</div>',
                        unsafe_allow_html=True,
                    )
                    injected = ", ".join(item.get("keywords_injected", []))
                    st.caption(
                        f"🎯 **Injected ATS Keywords:** `{injected}` | 📌 *{item.get('grounding_note', '')}*"
                    )

        with col_letter:
            st.markdown(
                f"### ✉️ Hyper-Personalized Cover Letter for {active_job.get('company')} (`Nemotron Ultra 253B`)"
            )
            st.caption(
                "Synthesizes candidate project metrics with live company engineering context from Tavily."
            )
            cover_letter_text = tailoring.get("tailored_cover_letter", "")
            st.text_area(
                "Tailored Cover Letter (Copy or Download below)",
                value=cover_letter_text,
                height=420,
            )

            dl_col1, dl_col2 = st.columns(2)
            markdown_bundle = (
                f"# Tailored Application Package — {profile.get('name')}\n"
                f"**Target Role:** {active_job.get('title')} @ {active_job.get('company')}\n"
                f"**Semantic Match Score:** {score}%\n\n"
                f"## Cover Letter (Generated by NVIDIA Nemotron Ultra on Nebius Token Factory)\n\n"
                f"{cover_letter_text}\n\n"
                f"## Grounded Resume Bullet Rewrites\n"
                + "\n".join(
                    f"- **{b.get('source_project_or_role')}**: {b.get('after_bullet')}"
                    for b in tailoring.get("resume_bullet_rewrites", [])
                )
            )
            dl_col1.download_button(
                "📥 Download Application (.md)",
                data=markdown_bundle,
                file_name=f"AutoApply_{active_job.get('company', 'Application').replace(' ', '_')}.md",
                mime="text/markdown",
                use_container_width=True,
            )
            dl_col2.download_button(
                "📦 Export Full State (.json)",
                data=json.dumps(state.to_dict(), indent=2),
                file_name="autoapply_nemotron_state.json",
                mime="application/json",
                use_container_width=True,
            )

    # =========================================================================
    # TAB 2: ATS Checker & Job Description Scanner
    # =========================================================================
    with tab_ats:
        st.markdown(
            "### 📋 Job-Description-Grounded ATS Checker (`Nemotron Nano` + `Super`)"
        )
        st.caption(
            "Audits your resume against the exact Job Description for keyword density, "
            "ATS parser compatibility, section headings, quantified metrics, and knockout risk."
        )

        with st.expander(
            "🔎 Scan Against Any Custom Job Description (Edit & Re-Scan Live)",
            expanded=False,
        ):
            jd_col1, jd_col2 = st.columns(2)
            scan_job_title = jd_col1.text_input(
                "Target Job Title for ATS Scan",
                value=active_job.get("title", "Senior AI Engineer"),
                key="ats_scan_title",
            )
            scan_company_name = jd_col2.text_input(
                "Hiring Company",
                value=active_job.get("company", "Nebius AI Studio"),
                key="ats_scan_company",
            )
            scan_jd_text = st.text_area(
                "Job Description Text (Paste any full JD to recompute keyword match & ATS score)",
                value=active_job.get("description", ""),
                height=140,
                key="ats_scan_jd",
            )
            if st.button(
                "🔍 Run Deep ATS Check on This Job Description",
                type="primary",
                use_container_width=True,
            ):
                state.ats_report = run_ats_audit(
                    client=client,
                    raw_resume_text=state.raw_resume_text,
                    job_title=scan_job_title,
                    company_name=scan_company_name,
                    job_description=scan_jd_text,
                    explicit_keywords=[],
                    tailored_bullets=tailoring.get("resume_bullet_rewrites", []),
                )
                st.session_state.workflow_state = state
                st.toast("Completed Nemotron ATS Job Description Scan!", icon="📋")
                st.rerun()

        ats_rep: Dict[str, Any] = state.ats_report
        if not ats_rep:
            ats_rep = run_ats_audit(
                client=client,
                raw_resume_text=state.raw_resume_text,
                job_title=active_job.get("title", "Senior AI Engineer"),
                company_name=active_job.get("company", "Nebius AI Studio"),
                job_description=active_job.get("description", ""),
                explicit_keywords=active_job.get("required_keywords", []),
                tailored_bullets=tailoring.get("resume_bullet_rewrites", []),
            )
            state.ats_report = ats_rep

        hygiene = ats_rep.get("hygiene", {})
        nano_triage = ats_rep.get("nano_triage", {})

        a1, a2, a3, a4, a5 = st.columns(5)
        a1.metric(
            "📄 Original ATS Score",
            f"{ats_rep.get('overall_ats_score_before', 72)}%",
            help="Baseline ATS score of your raw uploaded resume against the Job Description.",
        )
        a2.metric(
            "🚀 Tailored ATS Score",
            f"{ats_rep.get('overall_ats_score_after', 90)}%",
            delta=f"+{ats_rep.get('score_delta', 18)}% after Ultra rewrites",
        )
        a3.metric(
            "🔑 JD Keyword Match",
            f"{ats_rep.get('keyword_match_rate_after', 88)}%",
            delta=f"From {ats_rep.get('keyword_match_rate_before', 65)}% raw",
        )
        a4.metric(
            "📐 Formatting Hygiene",
            f"{hygiene.get('formatting_score', 90)}%",
        )
        a5.metric(
            "📊 Quantified Impact",
            f"{hygiene.get('quantified_impact_score', 85)}%",
            delta=f"{hygiene.get('metric_count', 4)} metrics found",
        )

        verdict_col1, verdict_col2 = st.columns([1.15, 1])
        with verdict_col1:
            with st.container(border=True):
                st.markdown(
                    f"#### 🏆 ATS Shortlist Verdict: `{ats_rep.get('ats_pass_tier', 'High Pass')}`"
                )
                st.write(
                    f"**Recruiter Knockout Risk (`Nemotron Super`):** {ats_rep.get('recruiter_knockout_risk', '')}"
                )
                st.caption(
                    f"⚡ **Nano Parser Check:** {nano_triage.get('ats_parser_compatibility', '')} | "
                    f"**Role Title Alignment:** `{nano_triage.get('role_title_alignment', 'Strong')}`"
                )
                st.markdown("**Top 3 ATS Remediation Actions:**")
                for step_txt in ats_rep.get("top_remediation_steps", []):
                    st.markdown(f"- {step_txt}")

        with verdict_col2:
            with st.container(border=True):
                st.markdown("#### 🧱 Structural & Formatting Hygiene Checklist")
                for chk in hygiene.get("structural_checks", []):
                    icon = "✅" if chk.get("status") == "PASS" else "⚠️"
                    st.markdown(
                        f"{icon} **{chk.get('check')}** — *{chk.get('detail')}*"
                    )
                st.markdown("**Copy-Ready ATS Skills Block (`Nemotron Super`):**")
                st.code(ats_rep.get("optimized_ats_skills_block", ""), language="text")

        st.markdown(
            f"#### 🔍 Job Description Keyword Frequency & Coverage Matrix (`{ats_rep.get('job_title')}` @ `{ats_rep.get('company_name')}`)"
        )
        kw_rows = []
        for row in ats_rep.get("keyword_matrix", []):
            cnt = row.get("count_in_resume", 0)
            covered = row.get("covered_after_tailoring", False)
            if cnt > 0:
                badge = f"<span class='pill-badge pill-nvidia'>✓ Matched ({cnt}x)</span>"
            elif covered:
                badge = "<span class='pill-badge pill-nebius'>✨ Injected by Ultra</span>"
            else:
                badge = "<span class='kw-gap'>⚠️ Missing — Add to Skills</span>"

            kw_rows.append(
                f"<tr>"
                f"<td><b>{row.get('keyword')}</b></td>"
                f"<td>{row.get('category')}</td>"
                f"<td><code>{row.get('priority')}</code></td>"
                f"<td>{cnt}x</td>"
                f"<td>{badge}</td>"
                f"</tr>"
            )

        if kw_rows:
            kw_table_html = (
                "<table class='telemetry-table'>"
                "<thead><tr>"
                "<th>JD Keyword / Skill</th>"
                "<th>ATS Category</th>"
                "<th>Priority</th>"
                "<th>Raw Resume Frequency</th>"
                "<th>ATS Coverage Status</th>"
                "</tr></thead>"
                f"<tbody>{''.join(kw_rows)}</tbody>"
                "</table>"
            )
            st.markdown(kw_table_html, unsafe_allow_html=True)

    # =========================================================================
    # TAB 3: Adzuna Live Job Discovery & Tavily Company Intelligence
    # =========================================================================
    with tab_intel:
        st.markdown("### 🌐 Live Job Discovery (`Adzuna`) & Company Engineering Intel (`Tavily`)")
        st.caption(
            f"**Nemotron Nano Search Query Strategy:** `{state.search_strategy.get('adzuna_search_query')}` | "
            f"**Priority Keywords:** `{', '.join(state.search_strategy.get('priority_keywords', []))}`"
        )

        col_jobs, col_tavily = st.columns([1, 1.1])
        with col_jobs:
            st.markdown("#### 💼 Discovered Target Roles")
            for idx, job in enumerate(state.discovered_jobs):
                is_active = idx == state.selected_job_index
                badge = "🟢 ACTIVE TARGET" if is_active else f"Role #{idx + 1}"
                with st.container(border=True):
                    st.markdown(
                        f"**{job.get('title')}** — `{job.get('company')}` ({badge})"
                    )
                    st.caption(
                        f"📍 {job.get('location')} | 💰 {job.get('salary_range')} | Source: `{job.get('source')}`"
                    )
                    st.write(job.get("description", ""))
                    if job.get("url"):
                        st.markdown(f"[🔗 View Job Posting]({job.get('url')})")

        with col_tavily:
            st.markdown(
                f"#### 🔎 Tavily Live Company Intelligence: `{active_job.get('company')}`"
            )
            with st.container(border=True):
                src_badge = (
                    "🟢 Live Tavily API"
                    if intel.get("source") == "tavily_live"
                    else "🛡️ Curated Intelligence (Add TAVILY_API_KEY for live web search)"
                )
                st.caption(
                    f"**Source:** {src_badge} | **Query:** `{intel.get('query_used', '')}`"
                )
                st.markdown(f"**Executive Summary:**\n{intel.get('summary', '')}")

                nano_digest = intel.get("nano_digest", {})
                if nano_digest:
                    st.info(
                        f"⚡ **Nemotron Nano Fast Triage Hook:**\n\n"
                        f"{nano_digest.get('top_talking_point', '')}"
                    )

                st.markdown("**Recent Engineering & Product Highlights:**")
                for hl in intel.get("recent_highlights", []):
                    st.markdown(f"- {hl}")

                if intel.get("articles"):
                    st.markdown("**Grounded Web Sources / Engineering Posts:**")
                    for art in intel.get("articles", []):
                        st.markdown(
                            f"- **[{art.get('title')}]({art.get('url')})** — *{art.get('snippet')}*"
                        )

    # =========================================================================
    # TAB 3: Structured Candidate Profile (Nemotron Super)
    # =========================================================================
    with tab_profile:
        st.markdown("### 👤 Structured Candidate Profile (`Nemotron Super 49B`)")
        p_col1, p_col2 = st.columns([1.2, 1])

        with p_col1:
            with st.container(border=True):
                st.markdown(f"#### {profile.get('name')} — *{profile.get('headline')}*")
                st.caption(
                    f"📍 {profile.get('location')} | ⏳ {profile.get('years_experience')} Years Experience | "
                    f"⚡ Nano Guardrail Check: `Valid Resume = {state.intent_check.get('is_valid_resume', True)}`"
                )
                st.write(profile.get("summary", ""))

                skills_obj = profile.get("skills", {})
                st.markdown("**Languages:** " + ", ".join(f"`{s}`" for s in skills_obj.get("languages", [])))
                st.markdown(
                    "**AI / ML Frameworks:** "
                    + ", ".join(f"`{s}`" for s in skills_obj.get("ai_ml_frameworks", []))
                )
                st.markdown(
                    "**Cloud & Infrastructure:** "
                    + ", ".join(f"`{s}`" for s in skills_obj.get("cloud_and_infra", []))
                )

            st.markdown("#### 🚀 Extracted Candidate Projects (Used for Grounding by Ultra)")
            for proj in profile.get("projects", []):
                with st.container(border=True):
                    st.markdown(
                        f"**{proj.get('name')}** — Tech Stack: {', '.join(f'`{t}`' for t in proj.get('tech_stack', []))}"
                    )
                    st.write(proj.get("impact", ""))

        with p_col2:
            st.markdown("#### 🧾 Strict JSON Schema Output (`Nemotron Super`)")
            st.json(profile)

    # =========================================================================
    # TAB 4: Technical Interview & Gap Mitigation Strategy (Nemotron Ultra)
    # =========================================================================
    with tab_interview:
        st.markdown(
            f"### 🎤 Technical Interview & Gap Mitigation Strategy for `{active_job.get('company')}` (`Nemotron Ultra 253B`)"
        )
        for idx, item in enumerate(tailoring.get("interview_prep_strategy", []), start=1):
            with st.container(border=True):
                st.markdown(f"#### Pillar #{idx}: {item.get('topic')}")
                st.markdown(f"**Why It Matters for This Role:** {item.get('why_it_matters')}")
                st.success(f"🎯 **Actionable Prep Strategy:** {item.get('prep_action')}")

    # =========================================================================
    # TAB 5: Nebius Token Factory Telemetry & Proof Inspector
    # =========================================================================
    with tab_telemetry:
        st.markdown("### ⚡ Nebius Token Factory Multi-Tier Routing Telemetry")
        st.write(
            "Visual verification of dynamic task routing across **NVIDIA Nemotron Nano (8B)**, "
            "**Super (49B)**, and **Ultra (253B)** via Nebius Token Factory (`https://api.studio.nebius.ai/v1/`)."
        )

        tele_summary = client.get_telemetry_summary()
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Routed Calls", tele_summary["total_calls"])
        c2.metric("Tier 1: Nano Calls", tele_summary["tier_counts"].get("Nano", 0))
        c3.metric("Tier 2: Super Calls", tele_summary["tier_counts"].get("Super", 0))
        c4.metric("Tier 3: Ultra Calls", tele_summary["tier_counts"].get("Ultra", 0))
        c5.metric(
            "Credit Cost Savings",
            f"{tele_summary['credit_savings_pct']}%",
            delta="vs. 100% Ultra baseline",
        )

        if tele_summary["events"]:
            rows_html = []
            for ev in tele_summary["events"]:
                tier_badge_cls = (
                    "pill-nvidia"
                    if ev["tier_short"] in ("Nano", "Ultra")
                    else "pill-nebius"
                )
                rows_html.append(
                    f"<tr>"
                    f"<td><code>{ev['timestamp']}</code></td>"
                    f"<td><b>{ev['step_name']}</b></td>"
                    f"<td><span class='pill-badge {tier_badge_cls}'>{ev['tier_short']}</span></td>"
                    f"<td><code>{ev['model_id']}</code></td>"
                    f"<td>{ev['latency_ms']} ms</td>"
                    f"<td>{ev['token_usage']['prompt_tokens']}</td>"
                    f"<td>{ev['token_usage']['completion_tokens']}</td>"
                    f"<td><b>{ev['token_usage']['total_tokens']}</b></td>"
                    f"<td>{ev['tokens_per_sec']} tok/s</td>"
                    f"<td><code>{ev['execution_mode']}</code></td>"
                    f"</tr>"
                )
            table_html = (
                "<table class='telemetry-table'>"
                "<thead><tr>"
                "<th>Timestamp (UTC)</th>"
                "<th>Pipeline Step</th>"
                "<th>Tier</th>"
                "<th>NVIDIA Nemotron Model ID</th>"
                "<th>Latency</th>"
                "<th>Prompt Tok</th>"
                "<th>Comp Tok</th>"
                "<th>Total Tok</th>"
                "<th>Throughput</th>"
                "<th>Mode</th>"
                "</tr></thead>"
                f"<tbody>{''.join(rows_html)}</tbody>"
                "</table>"
            )
            st.markdown(table_html, unsafe_allow_html=True)

            st.markdown("#### 🔍 Full Per-Step Request & Response JSON Traces")
            for idx, ev in enumerate(tele_summary["events"], start=1):
                with st.expander(
                    f"Call #{idx}: {ev['step_name']} — {ev['model_id']} ({ev['latency_ms']} ms)"
                ):
                    st.json(ev)
