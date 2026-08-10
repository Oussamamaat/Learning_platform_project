# Product Requirements Document (PRD)
## Project: Assistant IA - MVP (IBLOG Services)
**Author:** Oussama Maataoui  
**Target Delivery Date:** August 30, 2026  
**Architecture:** API-First Microservice 

> **⚠️ STRICT COMPLIANCE DIRECTIVE:**
> This PRD governs the technical execution and delivery schedule of the MVP. However, all feature scopes, business logic, and regulatory constraints must strictly respect the official functional specifications. **Refer to the `cahier_des_charges` document located in this root folder for all baseline business rules.** If a conflict arises between this technical PRD and the functional specifications regarding scope, the functional specifications take precedence unless explicitly negotiated with the CEO.

---

## 1. Project Overview
The objective is to deliver a functional, demonstrable MVP of the "Assistant IA" module (Section 3.1.7 of the specifications). This system acts as a personalized AI tutor for a B2B multi-tenant, domain-agnostic e-learning platform: each tenant's users are tutored from that tenant's own course materials (RAG + a fine-tuned LLM). Multilingual — **French primary, Darija secondary but mandatory, English in scope**. The system is built as an independent microservice designed to integrate seamlessly into the main e-learning platform. **Tenant #1** is the Moroccan safety and security regulations ("sécurité et sûreté") domain, whose language mix is Darija (Arabic script) + French.

## 2. Core Scope (MVP Priorities)
1.  **Conversational Assistant:** Text and Audio support, multilingual (French primary, Darija, English), grounded in the tenant's course materials.
2.  **Personalized Explanations:** Pedagogical, step-by-step scaffolding of complex concepts rather than direct answer dumping.
3.  **Quiz Generation:** Dynamic creation of assessment questions based strictly on the retrieved course material.

*Note: Automated summaries and difficulty analysis are deferred to the post-MVP roadmap. Multilingual support is in MVP scope — French primary, Darija mandatory for tenant #1, English in scope.*

## 3. System Architecture & Tech Stack
*   **Backend Framework:** FastAPI (Python) for asynchronous routing and API endpoint generation.
*   **Database (RAG):** PostgreSQL with the `pgvector` extension for semantic search and document retrieval.
*   **AI Engine (LLM):** Neutral multilingual base model with per-tenant LoRA adapters. Tenant #1 currently runs a Darija/French-tuned instance (`atlas-darija-tutor`, Atlas-Chat-9B LoRA) locally.
*   **Fine-Tuning Framework:** `Unsloth` utilizing 4-bit Quantization (QLoRA) to adapt the model's pedagogical behavior within hardware constraints (ASUS ROG Strix G16 / 8GB VRAM).
*   **Audio Processing:** Multilingual Speech-to-Text (STT); a community fine-tuned Whisper checkpoint for Darija STT covers tenant #1.
*   **DevOps / Delivery:** Docker & Docker Compose for isolated, multi-container deployment.

---

## 4. Weekly Execution Plan

### Week 1: Foundation, Data Architecture & API Contract
**Objective:** Establish the data layer, define communication protocols with the frontend team, and prepare the text processing pipeline.
*   **Task 1.1:** Define and document the REST API Contract (Swagger/OpenAPI). Specify exact JSON request/response schemas for `/chat`, `/audio`, and `/quiz` endpoints.
*   **Task 1.2:** Configure and spin up the local `pgvector` database via Docker.
*   **Task 1.3:** Develop the Python ingestion script using chunking logic (e.g., LangChain `RecursiveCharacterTextSplitter`) to clean, split, and embed regulatory safety texts into the vector database.
*   **Task 1.4:** Generate synthetic pedagogical data (Q&A pairs) to serve as a placeholder for LoRA fine-tuning while awaiting official course materials.
### Week 2: Core Backend & Retrieval Augmented Generation (RAG)
**Objective:** Build the N-tier logic connecting incoming requests to the semantic search database.
*   **Task 2.1:** Scaffold the FastAPI backend application architecture (routers, controllers, Pydantic models).
*   **Task 2.2:** Implement the embedding generation pipeline to convert user queries into vectors.
*   **Task 2.3:** Develop the semantic search algorithm to retrieve the top-K most relevant document chunks from `pgvector`.
*   **Task 2.4:** Build the prompt-injection logic that combines the retrieved context with the user's query before sending it to the LLM layer.

### Week 3: AI Baseline & Fine-Tuning Preparation
**Objective:** Deploy the base AI model and configure the memory-efficient training environment.
*   **Task 3.1:** Pull and configure a pre-trained open-source LLM checkpoint (neutral multilingual base for the platform; tenant #1's Darija-tuned instance via `atlas-darija-tutor`).
*   **Task 3.2:** Set up the isolated `Unsloth` training environment to enable QLoRA (4-bit quantization).
*   **Task 3.3:** Format the synthetic Q&A data from Week 1 into the strict JSONL format required for instruction tuning.
*   **Task 3.4:** Execute a short, low-epoch test training loop to validate VRAM consumption and prevent hardware crashes.

### Week 4: LoRA Training & Pedagogical Prompt Engineering
**Objective:** Teach the model *how* to act like a tutor and format outputs correctly, across the multilingual scope (French primary, Darija, English).
*   **Task 4.1:** Execute the full QLoRA fine-tuning run on the pedagogical dataset. Evaluate loss metrics and adjust hyperparameters (Rank, Alpha, Batch Size) as needed.
*   **Task 4.2:** Merge the trained LoRA adapter weights with the base model.
*   **Task 4.3:** Engineer and test strict system prompt chains instructing the model to generate structured JSON output for the Quiz Generation feature.
*   **Task 4.4:** Conduct baseline hallucination testing: ensure the model relies on RAG context rather than making up Moroccan safety laws.

### Week 5: Team Integration & API Wiring
**Objective:** Connect the isolated microservice to the company's main web platform.
*   **Task 5.1:** Collaborate directly with the frontend team (General Directorate/Platform Devs) to wire the React/Next.js client to the FastAPI endpoints based on the Week 1 Swagger contract.
*   **Task 5.2:** Resolve Cross-Origin Resource Sharing (CORS) policies and network routing issues.
*   **Task 5.3:** Validate state management on the client side (e.g., ensuring chat history arrays are correctly passed back to the API).

### Week 6: Audio Pipeline & End-to-End Testing
**Objective:** Implement multilingual audio transcription (Darija STT for tenant #1) and finalize system stability.
*   **Task 6.1:** Integrate a community-trained Darija Whisper model (STT) into a dedicated FastAPI asynchronous endpoint (part of the multilingual audio pipeline).
*   **Task 6.2:** Develop logic to receive audio blobs from the frontend, convert them to standard formats (e.g., WAV), transcribe them, and pass the text to the RAG pipeline.
*   **Task 6.3:** Run end-to-end load and latency tests. Optimize inference speeds and vector retrieval times.
*   **Task 6.4:** (Optional/Fallback) Integrate a basic Arabic Text-to-Speech (TTS) response layer if frontend requirements demand audio output.

### Week 7: Containerization & DevOps Delivery
**Objective:** Package the entire microservice architecture for production-ready deployment.
*   **Task 7.1:** Write optimized, multi-stage `Dockerfile` configurations for the FastAPI application and the LLM inference service to minimize image bloat.
*   **Task 7.2:** Construct the master `docker-compose.yml` file to orchestrate the internal networking between the backend, the `pgvector` database, and the model service.
*   **Task 7.3:** Finalize technical documentation (Markdown) detailing how to boot the containers, update environment variables, and ingest new training documents.
*   **Task 7.4:** Execute the final MVP Demonstration with the CEO.