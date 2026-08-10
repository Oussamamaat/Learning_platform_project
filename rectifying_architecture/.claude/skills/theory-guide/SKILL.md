---
name: theory-guide
description: Consults industry best practices on LLM fine-tuning and adaptive platforms
---

# Theory Guide 

This guide compiles production-grade systems design patterns and model-tuning strategies relevant for multi-tenant, bilingual, retrieval-augmented tutoring and interactive e-learning applications. It integrates recommendations from Suhas Pai's *Designing Large Language Model Applications* and Sampriti Mitra's *Systems Design in the LLM Era*.

---

## 1. Multi-Tenant RAG & Data Ingestion Pipeline

### Data Isolation & Query-Time Filtering
In a B2B multi-tenant SaaS context, strict data boundary enforcement is paramount.
* **Tenant-Level Filtering**: When querying vector databases for RAG context, do not rely on the LLM to isolate tenant boundaries. Every RAG query must explicitly append a filter constraint (e.g., `WHERE tenant_id = :tenant_id` or `WHERE user_id = :user_id`) at the database layer [315]. This metadata-level isolation prevents cross-tenant data leakage and ensures employees only retrieve context from their specific company's private catalog [315, 474].
* **Relational and Vector Polyglot Storage**: Relational databases like PostgreSQL with extensions such as `pgvector` are highly effective [389]. They provide robust row-level relational isolation (ACID constraints) alongside high-dimensional index searching (e.g., HNSW) under a unified engine, simplifying transactional multi-tenancy [389].

### PII Redaction & Data Cleaning Middleware
To protect enterprise data privacy and ensure compliance (PCI, PHI, etc.), raw text must be scrubbed before passing into public model APIs or vector indexes.
* **Ingestion Isolation**: Connectors must be strictly configured to ingest only data marked with the correct clearance levels (e.g., "PUBLICLY AVAILABLE" or tenant-approved) [474].
* **Scrubbing Middleware**: Raw text chunks must pass through a sanitization filter prior to vectorization or disk storage [271, 315, 458, 475].
* **Client-Side/Pre-Inference Scrubbing**: Implement client-side or gateway-level data cleaning middleware that utilizes regular expressions and entropy detection to scan for sensitive identifiers like credit card numbers, Social Security Numbers (SSNs), emails, API keys, or database credentials [271, 362, 475]. Discovered PII should be redacted and replaced with descriptive tokens (e.g., `<SECRET_REMOVED>` or `<PII_MASKED>`) [362, 475]. This is vital because once PII is embedded in a vector, it is extremely difficult to selectively purge it without re-indexing the entire database corpus [458].

### Document Representation & Smart Chunking
Ingestion quality directly governs generation quality.
* **De-noising and Normalization**: Strip irrelevant boilerplate (HTML tags, navigation footers, or headers) to prevent semantic dilution [270, 271]. Standardize Unicode characters and formatting schemas so the vector index remains mathematically consistent [271].
* **Smart Chunking**: Chunks must be created using natural syntactic boundaries (using abstract syntax trees or logical header parsers rather than random token length cuts) [339, 479].
* **Continuity Context**: Every chunk's metadata must preserve context from its parent document (e.g., parent section titles, document name, or chapter metadata) [472]. During the generation step, these structures are merged with the query, providing the LLM with structured boundaries rather than disconnected text fragments [472].

---

## 2. Serving Path Architecture & Caching Tiers

To balance sub-second latency targets with the stochastic, slow nature of generative models, the serving architecture must segregate operations into real-time, asynchronous, and cached paths.

### Interactive Tutoring & Real-Time Streams
Interactive tutoring requires immediate, conversational feedback.
* **Response Streaming**: A standard blocking HTTP request forces the user to wait several seconds for a full answer [294, 296]. To improve the perceived latency, implement token-by-token response streaming [296]. 
* **Server-Sent Events (SSE)**: Utilize Server-Sent Events (SSE) over standard HTTP/2, which provides a unidirectional (server-to-client), firewall-friendly, persistent connection [297]. The server sets the HTTP header `content-type: text/event-stream` and returns a generator that streams tokens to the client as they are generated, bringing perceived time-to-first-token (TTFT) down to milliseconds [296, 297].
* **Semantic Caching**: To skip expensive model inference for common conversational paths (e.g., standard platform onboarding questions), use a semantic cache [299, 344]. Convert incoming queries to vectors and execute a similarity search in a low-latency cache [299]. If similarity scores exceed a set threshold (e.g., 0.95 or 0.85 depending on tolerance), serve the cached response instantly, bypassing the LLM [299, 445].

### Proactive Quiz Caching ("The Warm Path")
Dynamic quiz generation at runtime imposes heavy GPU processing delays. The **proactive curation (warm path)** pattern addresses this constraint:
1. **Event-Driven Trigger**: When an employee completes a learning module, the application publishes a lightweight message (e.g., `{"userId": "user_123"}`) to a low-priority refill SQS queue [370, 371].
2. **Buffer Check**: A background worker polls the queue and checks the user's active precomputed list buffer in Redis [370]. If the pre-cached playlist is running low (e.g., fewer than 5–10 lessons/quizzes remain), the refill routine initiates [370].
3. **Asynchronous RAG Execution**: The background worker fetches the student's learning history, progress metadata, and a pool of candidate questions or relevant RAG documents [371]. 
4. **Offline Generation**: The worker calls the LLM orchestrator asynchronously [371]. The model processes the document context and sequences a fresh batch of tailored quiz items or precomputed lessons [371].
5. **Redis Buffer Refill**: The resulting structured quiz payload is written to a Redis list using atomic pushing operations [372, 388].
6. **Instant Delivery**: When the student finally requests a quiz, the lesson curator service executes an atomic `LPOP` operation on the Redis list cache [369, 372, 486]. This serves the personalized quiz instantly (under 50 milliseconds) with zero dynamic GPU execution overhead at the moment of request [369, 388].

### Synchronous Fallback ("The Cold Path")
If a new or returning user requests a lesson/quiz before the background worker has refilled their Redis list buffer, the `LPOP` operation returns `nil` (a cache miss) [373].
* **Bypass LLM Orchestration**: To protect the user from latency bottlenecks, immediately bypass the slow LLM orchestrator [374].
* **Rule-Based Database Query**: Execute a fast, deterministic database query directly against the question bank, filtering out recently seen questions using the student’s history logs and sorting by their active skill level [374].
* **Trigger Background Worker**: Serve this baseline quiz/lesson to the user synchronously to maintain responsiveness, and simultaneously trigger the asynchronous background worker to refill the cache for subsequent turns [375].

### Multi-Level Caching Topology
To protect backend databases and optimize cost of goods sold (COGS), implement a tiered caching strategy based on the 80/20 query frequency rule [298, 428]:
* **L1 Cache (Top 5% of queries)**: Store exact matches in JVM local memory for sub-millisecond retrieval [429].
* **L2 Cache (Top 20% of queries)**: Store exact matches in a distributed Redis cluster [429]. Hash the normalized prompt string (applying case folding and whitespace trimming to prevent minor key misses) to locate the exact-match JSON output [298, 347].
* **Warm Tier (The Long Tail)**: Store pre-processed database queries or vector embeddings in a semantic cache, resolving diverse phrasings of identical user intents [299, 430].

---

## 3. Model Tuning & Training Optimization

### Task Adaptation vs. Knowledge Memorization
* **RAG for Facts**: Do not fine-tune models to memorize static facts, regulations, or private tenant directories [69, 97]. Memorization of rare concepts is probabilistically inefficient and prone to hallucinations [148, 214]. RAG is thousands of times cheaper and allows immediate factual updates via document synchronization without re-training [146, 147, 261].
* **Fine-Tuning for Behavior**: Use Supervised Fine-Tuning (SFT) specifically to teach the model **behavioral patterns**—such as active Socratic dialogue, code-switching mechanics (shifting from French technical terms to Moroccan Arabic conversational structures), and strict compliance with custom JSON output formats [23, 69, 80, 260].

### Dataset Composition & Catastrophic Forgetting
When a model is trained exclusively on highly narrow domain-specific SFT tasks, it experiences **catastrophic forgetting**, causing general conversation and reasoning capabilities to decay [94].
* **Dataset Balancing (Data Mixtures)**: Ensure your training data is structured as an optimal data mixture [36]. Mix task-specific training data with general-purpose instruction datasets [36, 79]. For instance, incorporate general instruction-tuning samples (like FLAN or Alpaca) as an "anchor dataset" (representing 10% or more of the overall token mix) to keep the base conversational reasoning skills stable [36, 79, 82].
* **Noise Embeddings**: To prevent the model from overfitting to the rigid formatting, wording, or sequence lengths of your training files, inject small noise embeddings into the input layers during training [73, 74]. This technique reduces stylistic overfitting and encourages the generation of more detailed, information-dense explanations [73].

### Parameter-Efficient Fine-Tuning (PEFT) & LoRA Configuration
For resource-constrained hardware configurations (e.g., standard consumer GPUs with limited VRAM), full fine-tuning of all model weights is mathematically infeasible [74, 99]. Use Parameter-Efficient Fine-Tuning (PEFT) via Low-Rank Adaptation (LoRA) [75, 76, 99].
* **Rank ($r$) and Alpha ($\alpha$)**: For stable behavioral adaptation, set LoRA attention dimension $r$ and scaling factor $\alpha$ to reasonable defaults (e.g., $r = 64$ and $\alpha = 8$ or $r=16, \alpha=16$ depending on training depth) [76].
* **Dropout**: Incorporate `lora_dropout = 0.1` to reduce overfitting [76].
* **Target Modules**: Apply LoRA transformations to the projection matrices of the model's self-attention and MLP layers [76].
* **Reduced Precision (QLoRA)**: Load the base model in 4-bit or 8-bit precision formats using quantization libraries (like `bitsandbytes` and FP4 formats) to minimize active GPU memory overhead [77]. Use paged optimizers (such as `Paged AdamW`) to manage gradient checkpoint memory spikes on consumer hardware.

---

## 4. LLM Orchestration, Reliability & Tool Usage

### Centralized LLM Gateway (GenAI Service) Pattern
To prevent high architectural coupling, do not call model providers directly from your core product microservices [287, 288]. Instead, implement a centralized **LLM Gateway** [288].
* **Unified Abstraction**: Expose a single internal API endpoint that handles prompt assembly, authentication, routing, and provider-specific payload translation [288, 289, 412].
* **Prompt Template Repository**: Store prompt templates in a single, cached location (or inside code variables to bypass DB fetching latencies) to prevent constructing system prompts from scratch on every transaction [338, 412, 470].

### Multi-Model Tiered Routing
Implement a rule-based routing engine inside the LLM Gateway to dispatch incoming requests to the smallest, cheapest model that is capable of resolving the subtask [131, 183, 302].
* **Small / Local SLMs**: Route low-complexity tasks—such as intent classification, spelling checks, entity extraction, or PII scrubbing—to small, fast, locally-hosted or serverless open-weights models (e.g., 3B to 8B parameter models) [151, 378, 469].
* **Large Reasoning Models**: Route complex conversational tutoring, context synthesis, or multi-turn logical explanations to larger, highly capable reasoning engines [378].
* **Cost Limits**: Enforce strict input token caps at the gateway to prevent runaway API billing spikes or model denial-of-service (MDoS) [303, 316].

### Circuit Breakers, Retries, and Fallbacks
Since external model APIs are inherently slow and prone to transient network failures or rate limits (HTTP 429), protect your system with robust fallback patterns [285, 379]:
* **Capped Retries**: For interactive flows, use capped exponential backoff with jitter (e.g., retry up to 3 times, capping maximum wait times at 1 second) [352, 353]. For asynchronous background tasks (like quiz pre-generation), use aggressive exponential backoff (e.g., waiting minutes) since no active user is waiting on the thread [293, 353].
* **The Circuit Breaker Pattern**: Wrap each model provider in a circuit breaker [381]. If a provider exceeds a failure rate threshold (e.g., 5 consecutive errors or a 20% error rate within 60 seconds), trip the circuit to the "Open" state [382].
* **Rerouting & Tiered Fallbacks**: When a circuit trips or a primary model times out, immediately route subsequent prompts to a pre-defined backup model (e.g., falling back from a premium model to a faster local instance) [290, 291, 383, 384]. After a cooling period (e.g., 30 seconds), allow a single probe request to pass through (Half-Open state); close the circuit and resume normal traffic if it succeeds [292, 382, 383].

### Function Calling for Deterministic Verification
Large language models are probabilistic and often struggle with precise logical checks, mathematical grading, or binary verification [213, 398].
* **Deterministic Tools**: Instead of prompting the LLM to grade a student's answer directly, utilize **Function Calling (Tool Use)** [44, 398].
* **The Workflow**: The orchestrator prompts the LLM to identify the student's answer and invoke a pre-defined verification function (e.g., `verify_math_solution(student_answer, correct_formula)`) [398]. The application gateway intercepts the tool call, executes the logic deterministically inside a secure sandbox (such as a local Python runtime or math evaluator), and returns a clean, binary `True/False` state back to the LLM [398, 399]. The LLM then uses this verified ground truth to formulate its conversational, Socratic feedback [399]. This maintains 100% grading accuracy while retaining the friendly persona of the tutor [399].

---

## 5. Testing, Evaluation & Observability

### Golden Datasets & Quantitative Evaluations
Traditional pass/fail assertions do not work on probabilistic AI systems [268, 309, 322]. Implement **Quantitative Evaluation Testing** to measure performance as objectively as possible [229, 280, 322]:
* **The Golden Dataset**: Curate a permanent suite of 50–100 representative multi-turn conversations containing realistic user queries, Edge-case inputs (e.g., adversarial prompt injections), and ideal responses [228, 309, 482].
* **LLM-as-a-Judge**: In your CI/CD pipeline, run every new prompt template or model merge against the Golden Dataset [228, 309]. Utilize a highly capable, independent model to rate the generated responses against the ideal outputs based on clear rubric dimensions (grounding, alignment, and formatting) [308, 485].
* **Compiler-as-a-Judge / Tool-as-a-Judge**: For structured programming or schema tasks, feed outputs directly into validation utilities (e.g., parsing the JSON output against a strict schema validator) [360]. If the code or JSON fails to parse, it receives an automatic zero score [360]. This provides rapid, deterministic feedback without incurring the cost of LLM-based evaluation [360].

### Operational Observability Signals
To monitor system health and catch quality regressions in production, establish alerts around these key metrics [282, 401, 475]:
1. **P99 Lesson Serving Latency**: The total time from the initial user request to the delivery of the lesson or quiz [401]. Spikes indicate cache misses or provider congestion [379, 401].
2. **Cache Hit Rate**: Monitor L1/L2 and warm cache hits [401]. A sharp drop in the cache hit rate indicates a shift in query patterns, forcing more users onto the slow, expensive synchronous fallback path [401].
3. **Time to First Token (TTFT)**: Tracks how long the user waits before the first word of streamed responses appears on the client UI [282]. High TTFT severely impacts user engagement [319].
4. **Daily Cost Ceilings**: Monitor token usage metrics per tenant and per user [476]. Alert the team if the daily model billing exceeds baseline quotas [476].
