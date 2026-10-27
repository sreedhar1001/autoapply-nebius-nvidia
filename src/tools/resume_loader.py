"""Resume file parser (PDF / TXT / Markdown) and curated sample candidate personas."""

from __future__ import annotations

import io
from typing import Dict

from pypdf import PdfReader

SAMPLE_RESUMES: Dict[str, str] = {
    "AI / ML Inference Systems Engineer (Alex Rivera)": """ALEX RIVERA
San Francisco, CA | alex.rivera.ai@email.com | github.com/alexrivera-ml

SUMMARY
Senior Machine Learning & Distributed Inference Engineer with 5+ years of experience designing high-throughput LLM serving pipelines, multi-agent orchestration systems, and GPU cluster scheduling microservices. Proven track record reducing p99 token latency by 48% and scaling production RAG systems to 2M+ daily queries.

TECHNICAL SKILLS
Languages: Python, C++, Rust, SQL, Bash, TypeScript
ML & AI Systems: PyTorch, vLLM, TensorRT-LLM, NVIDIA Triton, LangGraph, OpenAI SDK, HuggingFace Transformers, FAISS, Qdrant
Cloud & Infrastructure: Kubernetes, Docker, Ray, AWS (EKS, SageMaker), Terraform, Prometheus, Grafana, Redis, FastAPI

EXPERIENCE
Senior ML Infrastructure Engineer | NeuralScale Cloud (2023 - Present)
- Architected a dynamic multi-tier LLM model router dispatching queries across 8B, 49B, and 253B parameter open-weights models, cutting GPU inference spend by 61% while maintaining 99.4% accuracy on complex reasoning benchmarks.
- Built real-time telemetry and token-throughput observability pipelines using FastAPI, Prometheus, and OpenTelemetry across 128 H100 GPUs.
- Implemented automated semantic caching and token-bucket rate limiting for external enterprise API gateways.

Machine Learning Engineer | DataPulse AI (2021 - 2023)
- Developed an end-to-end hybrid retrieval-augmented generation (RAG) platform combining dense vector search (Qdrant) with BM25 reranking for technical documentation search.
- Reduced document ingestion and structured JSON extraction latency by 3.2x using asynchronous Python workers and batched inference calls.

PROJECTS
1. SpeculativeRoute-OSS
   Tech Stack: Python, OpenAI SDK, FastAPI, Redis, Pytest
   Outcome: Open-source latency-aware LLM router that dynamically routes classification tasks to 8B models and synthesis tasks to frontier reasoning models with live cost telemetry.

2. AgenticCodeVerifier
   Tech Stack: Python, LangGraph, Docker, Pytest, AST Analysis
   Outcome: Autonomous multi-agent coding sandbox that synthesizes unit tests, executes isolated test suites, and self-heals traceback errors in under 4 seconds.

3. StreamRAG Engine
   Tech Stack: Python, Qdrant, Tavily API, Streamlit, AsyncIO
   Outcome: Live web-grounded research assistant combining real-time search APIs with structured citation verification to eliminate hallucinated references.

EDUCATION
B.S. in Computer Science, University of California, Berkeley
""",
    "Full-Stack AI Product & Agent Engineer (Priya Nair)": """PRIYA NAIR
Austin, TX / Remote | priya.nair.dev@email.com | github.com/priyanair-dev

SUMMARY
Full-Stack AI Engineer & Agentic Workflow Architect with 4 years of experience building customer-facing AI copilot applications, structured extraction pipelines, and real-time observability dashboards. Passionate about reliable tool-calling agents, sub-second UX, and production cost optimization.

TECHNICAL SKILLS
Languages: Python, TypeScript, JavaScript, SQL, Go
AI & Agent Frameworks: OpenAI SDK, LangChain, LangGraph, Pydantic, Tavily Search API, Structured Outputs, Prompt Engineering
Frontend & Backend: React, Next.js, Streamlit, FastAPI, Node.js, PostgreSQL, Redis, Docker, GitHub Actions

EXPERIENCE
AI Product Engineer | WorkflowLabs Inc. (2023 - Present)
- Built a multi-agent customer intelligence platform that automates company research, CRM schema extraction, and personalized outreach drafting for 15,000+ active users.
- Integrated live web search grounding via Tavily API and structured JSON validation with Pydantic, increasing factual accuracy of generated briefs from 74% to 97%.
- Designed a live token-usage and latency telemetry dashboard enabling enterprise admins to audit per-step agent costs.

Software Engineer (Backend & Data) | CloudMetric Systems (2022 - 2023)
- Engineered high-throughput REST APIs in FastAPI and PostgreSQL handling 800 req/sec with automated retry and circuit-breaker resilience.
- Automated PDF document parsing and invoice schema extraction workflows with 98.5% field-level precision.

PROJECTS
1. CareerPilot Multi-Agent Suite
   Tech Stack: Python, Streamlit, OpenAI SDK, Tavily API, PyPDF
   Outcome: Autonomous job discovery and resume tailoring pipeline with live company engineering blog analysis and ATS keyword gap scoring.

2. SchemaGuard JSON Validator
   Tech Stack: Python, Pydantic, FastAPI, Pytest
   Outcome: Deterministic JSON schema enforcement middleware that repairs malformed LLM tool calls with zero downstream crashes.

EDUCATION
M.S. in Software Engineering, University of Texas at Austin
""",
    "Staff Distributed Systems & Platform Engineer (Marcus Vance)": """MARCUS VANCE
Seattle, WA | marcus.vance.infra@email.com | github.com/mvance-sys

SUMMARY
Staff Distributed Systems & AI Platform Engineer with 7+ years of experience scaling cloud-native microservices, GPU inference gateways, and high-availability data pipelines. Deep expertise in Kubernetes operators, observability, and high-throughput API gateways.

TECHNICAL SKILLS
Languages: Go, Python, Rust, C++, SQL
Distributed Systems & Cloud: Kubernetes, Envoy, gRPC, Kafka, Ray, AWS, GCP, Terraform, Helm, Cilium
AI Platform & Observability: OpenAI-Compatible Inference Gateways, Prometheus, Grafana, OpenTelemetry, Redis, Vector Databases

EXPERIENCE
Staff Platform Engineer | Hyperion Cloud Technologies (2022 - Present)
- Led the architecture of a multi-region AI inference gateway serving 50M+ daily requests with automatic failover, tenant rate-limiting, and real-time cost attribution.
- Reduced cluster tail latency (p99) by 39% by implementing intelligent request routing based on prompt complexity and token budget estimates.

Senior Distributed Systems Engineer | ScaleGrid Corp (2019 - 2022)
- Designed event-driven ingestion pipelines using Kafka, Go, and Python processing 1.5 TB/day of telemetry metrics.
- Authored comprehensive chaos-testing suites and automated CI/CD deployment guardrails.

PROJECTS
1. TokenGate Proxy
   Tech Stack: Go, Python, Redis, OpenTelemetry
   Outcome: High-speed OpenAI-compatible proxy with session rate limiting, SHA-256 semantic response caching, and live token/sec metrics.

2. KubeGPU Autoscale Controller
   Tech Stack: Go, Kubernetes Operator SDK, Prometheus
   Outcome: Custom Kubernetes controller scaling inference pods based on queue depth and token generation throughput.

EDUCATION
B.S. in Computer Engineering, University of Washington
""",
}


def extract_text_from_upload(file_bytes: bytes, filename: str) -> str:
    """Extract clean plain text from an uploaded PDF, TXT, or Markdown resume file."""
    lower_name = filename.lower()
    if lower_name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(file_bytes))
        pages = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                pages.append(page_text.strip())
        extracted = "\n\n".join(pages).strip()
        if not extracted:
            raise ValueError("Uploaded PDF contained no extractable text.")
        return extracted

    # Default to UTF-8 text decoding for .txt, .md, etc.
    return file_bytes.decode("utf-8", errors="replace").strip()
