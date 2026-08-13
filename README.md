# IBLOG Adaptive Learning Tutor — Optimized Project Directory

The adaptive learning tutor for the IBLOG e-learning platform: a B2B multi-tenant,
domain-agnostic AI assistant that tutors every tenant's users from that tenant's own
course materials (RAG + a fine-tuned LLM). Multilingual by design — **French primary,
Darija secondary but mandatory, English in scope**. MVP features: personalized
explanations, quiz generation from tenant-provided course content, and a
conversational assistant (text + audio), all multilingual.

This directory is the working repo — treat it as authoritative. It contains the full
codebase (`app/`, `tests/`), the data pipeline (`data/`, `raw/`), raw generation output
(`full_run_export/`, `incident_export/`), and all knowledge documents capturing debugging
history and architectural decisions.

**Tenant model.** Each tenant uploads its own course materials and brings its own
language mix; onboarding means adding documents and a per-tenant LoRA adapter, not code
changes. **Tenant #1** (Moroccan safety/security regulations, "sécurité et sûreté") is the
current live instance, served by two merged Atlas-Chat-9B fine-tunes (Q4_K_M GGUF via
Ollama) selected per turn by resolved response language: `IBLOG_TUTOR:latest` (Arabic-
script Darija) and `iblog-tutor-fr:latest` (French). The platform direction is a
**neutral multilingual base model with per-tenant, hot-swappable LoRA adapters**
(currently frozen-merged, not hot-swappable) before onboarding tenant #2 — see
`docs/architecture/serving.md` for what's deployed vs. that target.
