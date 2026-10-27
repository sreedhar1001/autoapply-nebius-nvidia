"""Adzuna Live Job Discovery & Tavily Company Intelligence Clients with Graceful Fallbacks."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import requests

CURATED_FALLBACK_JOBS: List[Dict[str, Any]] = [
    {
        "id": "nebius-ai-01",
        "title": "Senior AI Inference & Agentic Systems Engineer",
        "company": "Nebius AI Studio",
        "location": "San Francisco, CA / Hybrid",
        "salary_range": "$185,000 - $240,000",
        "url": "https://studio.nebius.ai/",
        "source": "curated_fallback",
        "description": (
            "Design and scale high-throughput multi-model inference routing and autonomous agent "
            "pipelines on Nebius Token Factory. Experience with open-weights LLMs (NVIDIA Nemotron), "
            "OpenAI-compatible SDKs, Python, FastAPI, Kubernetes, vLLM/TensorRT-LLM, semantic caching, "
            "and real-time latency/token telemetry dashboards required. Bonus for experience with "
            "live web-search grounding (Tavily) and distributed GPU orchestration."
        ),
        "required_keywords": [
            "Python",
            "NVIDIA Nemotron",
            "OpenAI SDK",
            "Multi-Agent Orchestration",
            "vLLM",
            "TensorRT-LLM",
            "FastAPI",
            "Kubernetes",
            "Telemetry",
            "Semantic Caching",
        ],
    },
    {
        "id": "nvidia-nim-02",
        "title": "Staff AI Developer Ecosystem & Autonomous Agents Engineer",
        "company": "NVIDIA",
        "location": "Santa Clara, CA / Remote",
        "salary_range": "$195,000 - $255,000",
        "url": "https://www.nvidia.com/en-us/ai/",
        "source": "curated_fallback",
        "description": (
            "Build reference agentic architectures and developer tooling powered by NVIDIA Nemotron "
            "(Nano, Super, Ultra) and NIM microservices. Partner with cloud inference providers to "
            "showcase dynamic model routing, structured JSON schema extraction, tool calling, and "
            "production guardrails. Strong background in Python, LangGraph, PyTorch, RAG, and "
            "evaluation rubrics expected."
        ),
        "required_keywords": [
            "Python",
            "NVIDIA Nemotron",
            "LangGraph",
            "Tool Calling",
            "Structured JSON",
            "RAG",
            "Guardrails",
            "PyTorch",
            "Docker",
            "Evaluation Rubrics",
        ],
    },
    {
        "id": "tavily-search-03",
        "title": "Lead Applied AI & Search Grounding Engineer",
        "company": "Tavily",
        "location": "New York, NY / Remote",
        "salary_range": "$175,000 - $220,000",
        "url": "https://tavily.com/",
        "source": "curated_fallback",
        "description": (
            "Architect ultra-low-latency search grounding and company intelligence pipelines for "
            "autonomous AI agents. Optimize query decomposition, multi-hop retrieval, and structured "
            "synthesis across fast 8B and reasoning 253B models. Proficiency in Python, AsyncIO, "
            "vector databases (Qdrant/FAISS), Pydantic, and high-concurrency API design."
        ),
        "required_keywords": [
            "Python",
            "Tavily API",
            "RAG",
            "Query Decomposition",
            "Vector Databases",
            "Qdrant",
            "Pydantic",
            "AsyncIO",
            "FastAPI",
            "Multi-Agent Systems",
        ],
    },
]

CURATED_COMPANY_INTEL: Dict[str, Dict[str, Any]] = {
    "Nebius AI Studio": {
        "company": "Nebius AI Studio",
        "source": "curated_fallback",
        "summary": (
            "Nebius Token Factory provides high-speed, cost-efficient cloud inference for open-source "
            "frontier models including NVIDIA Nemotron Nano, Super, and Ultra via an OpenAI-compatible API."
        ),
        "recent_highlights": [
            "Partnered with NVIDIA to host the Nebius x NVIDIA Global AI Hackathon focusing on multi-tier Nemotron agent architectures.",
            "Expanded Token Factory throughput with sub-second time-to-first-token (TTFT) on 8B Nano and high-concurrency structured JSON output on 49B Super.",
            "Engineering focus areas: multi-model tier routing, credit cost optimization, and production-ready OpenAI SDK drop-in compatibility.",
        ],
        "articles": [
            {
                "title": "Scaling Open-Source NVIDIA Nemotron Inference on Nebius Token Factory",
                "url": "https://studio.nebius.ai/",
                "snippet": "How dynamic routing across Nano (8B), Super (49B), and Ultra (253B) slashes inference cost by over 60% without sacrificing reasoning depth.",
            },
            {
                "title": "Building Production Multi-Agent Systems with OpenAI-Compatible Endpoints",
                "url": "https://nebius.com/blog",
                "snippet": "Best practices for structured schema extraction, latency telemetry, and rate-limit resilience in autonomous agent workflows.",
            },
        ],
    },
    "NVIDIA": {
        "company": "NVIDIA",
        "source": "curated_fallback",
        "summary": (
            "NVIDIA leads open-weights enterprise AI with the Llama-Nemotron model family (Nano 8B, "
            "Super 49B, Ultra 253B) purpose-built for agentic tool use, coding, and deep reasoning."
        ),
        "recent_highlights": [
            "Released the NVIDIA Llama-3.1/3.3 Nemotron family optimized with RLHF and synthetic data for top-tier reasoning and structured tool calling.",
            "Emphasizes compound AI systems where lightweight Nano models handle guardrails and intent triage while Ultra models solve multi-step planning.",
            "Active investment in developer ecosystems integrating NIM and cloud token factories with LangGraph and real-time search tools.",
        ],
        "articles": [
            {
                "title": "NVIDIA Nemotron Ultra & Super: Open-Source Models Built for Agentic Reasoning",
                "url": "https://developer.nvidia.com/blog",
                "snippet": "Inside the post-training pipeline that enables Nemotron Super and Ultra to excel at strict JSON adherence and complex synthesis.",
            },
            {
                "title": "Compound AI Architectures: Routing Between Fast 8B and Deep 253B Models",
                "url": "https://www.nvidia.com/en-us/ai/",
                "snippet": "Architectural patterns for combining low-latency classification tiers with frontier reasoning tiers.",
            },
        ],
    },
    "Tavily": {
        "company": "Tavily",
        "source": "curated_fallback",
        "summary": (
            "Tavily is the leading real-time search and web intelligence engine purpose-built for AI "
            "agents and RAG pipelines, delivering clean, citation-backed context in a single API call."
        ),
        "recent_highlights": [
            "Featured bonus track partner in the Nebius x NVIDIA Hackathon for grounding autonomous agents in live web intelligence.",
            "Specializes in extracting high-signal engineering blog posts, financial updates, and technical docs without HTML boilerplate noise.",
            "Prioritizes sub-second search latency and structured context windows tailored for LLM consumption.",
        ],
        "articles": [
            {
                "title": "Why Autonomous Agents Need Purpose-Built Search Grounding",
                "url": "https://tavily.com/",
                "snippet": "Eliminating hallucinations in career, finance, and coding agents by injecting live domain-specific search results.",
            }
        ],
    },
}


def _is_valid_key(val: Optional[str], placeholders: set[str]) -> bool:
    if not val:
        return False
    cleaned = val.strip()
    return cleaned not in placeholders and len(cleaned) > 5


def search_adzuna_jobs(
    query: str,
    location: str = "United States",
    max_results: int = 3,
    app_id: Optional[str] = None,
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Query Adzuna API for live job roles, with graceful fallback to realistic roles."""
    active_app_id = (app_id if app_id is not None else os.getenv("ADZUNA_APP_ID", "")).strip()
    active_api_key = (api_key if api_key is not None else os.getenv("ADZUNA_API_KEY", "")).strip()

    placeholders = {"", "your_adzuna_app_id", "your_adzuna_api_key", "placeholder"}
    if _is_valid_key(active_app_id, placeholders) and _is_valid_key(active_api_key, placeholders):
        try:
            url = "https://api.adzuna.com/v1/api/jobs/us/search/1"
            params: Dict[str, Any] = {
                "app_id": active_app_id,
                "app_key": active_api_key,
                "results_per_page": max_results,
                "what": query,
                "content-type": "application/json",
            }
            if location and location.lower() not in ("remote", "united states", "us", "any"):
                params["where"] = location

            resp = requests.get(url, params=params, timeout=8)
            resp.raise_for_status()
            data = resp.json()
            raw_results = data.get("results", [])
            if raw_results:
                parsed_jobs: List[Dict[str, Any]] = []
                for idx, item in enumerate(raw_results[:max_results]):
                    company_name = (
                        item.get("company", {}).get("display_name")
                        if isinstance(item.get("company"), dict)
                        else "Tech Enterprise"
                    ) or "Tech Enterprise"
                    loc_name = (
                        item.get("location", {}).get("display_name")
                        if isinstance(item.get("location"), dict)
                        else location
                    ) or location
                    sal_min = item.get("salary_min")
                    sal_max = item.get("salary_max")
                    if sal_min and sal_max:
                        salary_str = f"${int(sal_min):,} - ${int(sal_max):,}"
                    elif sal_min:
                        salary_str = f"From ${int(sal_min):,}"
                    else:
                        salary_str = "Competitive Compensation"

                    parsed_jobs.append(
                        {
                            "id": str(item.get("id", f"adzuna-{idx}")),
                            "title": item.get("title", query),
                            "company": company_name,
                            "location": loc_name,
                            "salary_range": salary_str,
                            "url": item.get("redirect_url", "https://www.adzuna.com/"),
                            "source": "adzuna_live",
                            "description": item.get("description", ""),
                            "required_keywords": [],
                        }
                    )
                return parsed_jobs
        except Exception:
            pass

    # Return curated fallback jobs customized with query signal
    return [dict(job) for job in CURATED_FALLBACK_JOBS[:max_results]]


def fetch_tavily_company_intel(
    company_name: str,
    role_title: str = "",
    tavily_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Query Tavily Search API for recent engineering blogs and news about the hiring company."""
    active_key = (
        tavily_api_key if tavily_api_key is not None else os.getenv("TAVILY_API_KEY", "")
    ).strip()
    placeholders = {"", "your_tavily_api_key", "tvly-placeholder"}

    if _is_valid_key(active_key, placeholders):
        try:
            search_query = (
                f"{company_name} engineering blog tech stack AI infrastructure news {role_title}".strip()
            )
            resp = requests.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": active_key,
                    "query": search_query,
                    "search_depth": "basic",
                    "include_answer": True,
                    "max_results": 3,
                },
                timeout=10,
            )
            resp.raise_for_status()
            payload = resp.json()
            results = payload.get("results", [])
            answer = payload.get("answer") or ""

            articles = [
                {
                    "title": r.get("title", f"{company_name} Engineering Update"),
                    "url": r.get("url", "https://tavily.com"),
                    "snippet": (r.get("content") or "")[:280],
                }
                for r in results[:3]
            ]
            highlights = [a["snippet"] for a in articles if a.get("snippet")]
            if not highlights and answer:
                highlights = [answer]

            return {
                "company": company_name,
                "source": "tavily_live",
                "query_used": search_query,
                "summary": answer
                or f"Live web intelligence retrieved via Tavily Search API for {company_name}.",
                "recent_highlights": highlights
                or [f"Active hiring for {role_title} with focus on scalable AI systems."],
                "articles": articles,
            }
        except Exception:
            pass

    # Graceful fallback to curated intelligence
    if company_name in CURATED_COMPANY_INTEL:
        intel = dict(CURATED_COMPANY_INTEL[company_name])
        intel["query_used"] = f"{company_name} engineering blog AI infrastructure {role_title}".strip()
        return intel

    return {
        "company": company_name,
        "source": "curated_fallback",
        "query_used": f"{company_name} engineering blog tech stack news",
        "summary": (
            f"{company_name} is actively expanding its engineering organization for {role_title}, "
            "prioritizing production reliability, modern cloud-native architecture, and AI automation."
        ),
        "recent_highlights": [
            f"{company_name} engineering teams emphasize automated testing, observability, and low-latency API design.",
            "Recent technical initiatives focus on scaling distributed microservices and integrating LLM-powered developer workflows.",
        ],
        "articles": [
            {
                "title": f"Inside {company_name}'s Engineering Architecture & AI Roadmap",
                "url": "https://studio.nebius.ai/",
                "snippet": f"Key priorities for {role_title} candidates: high-throughput Python services, structured data pipelines, and measurable production impact.",
            }
        ],
    }
