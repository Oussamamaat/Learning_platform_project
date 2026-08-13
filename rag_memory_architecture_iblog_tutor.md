# Advanced RAG Pipeline Memory Architecture for IBLOG_TUTOR

Tackling **memory** in an Advanced RAG pipeline—especially for a French Socratic tutor (`IBLOG_TUTOR`)—is a two-fold challenge:

1. **Context Window / Token Memory** (managing short-term conversation context sent to Gemma-2-9B).
2. **State / User Memory** (storing long-term student progress, past mistakes, and learned concepts across sessions).

For a Socratic agent, memory isn't just about dumping prior turns into the prompt—it must preserve **pedagogical continuity** without swamping your prompt budget or diluting Gemma-2-9B's focus.

---

## 1. Short-Term Memory: Conversation & Prompt Window Management

When a user chats continuously, appending raw transcript turns degrades generation speed, inflates GPU KV-cache usage, and eventually hits the context ceiling.

### Strategy A: Sliding Window + Buffer
Keep the last **N turns** (e.g., last 3–4 exchanges) in full verbatim detail to preserve immediate conversational fluidness.

### Strategy B: Socratic Context Summarization
Every time the context exceeds a threshold (e.g., > 6 dialogue turns), run a background lightweight job (or use Gemma) to produce a **Socratic State Summary**.

Instead of a generic summary, extract specific learning signals:
* **Current Objective:** What problem is the student trying to solve right now?
* **Student's Mental Model / Misconceptions:** What did they misinterpret 2 turns ago?
* **Guided Hints Given:** What clues have already been provided? (Prevents repeating the same hints).

```text
[System Prompt + Socratic Persona]
[Persistent Student Profile & Long-Term Memory]
[Socratic State Summary of Current Session]
[Retrieved Documents via RAG]
[Recent 2-3 Dialogue Turns]
[User Input]
```

---

## 2. Long-Term Memory: Cross-Session Student Profiling

A Socratic tutor shines when it remembers what the student struggled with yesterday. Standard vector search over past chat logs is usually too noisy.

### Hybrid Long-Term Storage Model

| Memory Type | What It Stores | Technology / Structure | Socratic Purpose |
| :--- | :--- | :--- | :--- |
| **Episodic Memory** | Raw past interactions & Q&A sessions | Vector Database (e.g., Qdrant / Chroma) | Allows the RAG engine to query past chat history for context. |
| **Semantic / State Memory** | Key facts, mastered concepts, persistent weak spots | Key-Value / Relational (e.g., Redis / Postgres JSON) | Directly injected into the prompt as a lightweight JSON profile. |

### Example Student Profile Payload
```json
{
  "student_id": "usr_84920",
  "mastered_concepts": ["Docker networks", "Container lifecycle"],
  "struggling_concepts": ["Volume persistence vs Bind mounts"],
  "preferred_socratic_style": "analogies_first"
}
```

---

## 3. Advanced RAG Memory Architecture for `IBLOG_TUTOR`

To keep RAG retrieval precise without bloating Gemma-2-9B's prompt, structure your retrieval & memory pipeline in **3 distinct steps**:

```
                  ┌────────────────────────┐
                  │       User Input       │
                  └───────────┬────────────┘
                              │
               ┌──────────────┴──────────────┐
               ▼                             ▼
   ┌──────────────────────┐     ┌──────────────────────┐
   │ Contextual Query     │     │ Memory Retrieval     │
   │ Rewriting (HyDE /    │     │ (Fetch Student State │
   │ Condensed Query)     │     │ & Relevant History)  │
   └───────────┬──────────┘     └───────────┬──────────┘
               │                             │
               ▼                             │
   ┌──────────────────────┐                  │
   │ Knowledge Retrieval  │                  │
   │ (Vector DB + Rerank) │                  │
   └───────────┬──────────┘                  │
               │                             │
               └──────────────┬──────────────┘
                              ▼
                ┌───────────────────────────┐
                │  Construct Final Context  │
                │  & Prompt for Gemma-2-9B  │
                └───────────────────────────┘
```

### Key RAG Techniques to Implement:
1. **Contextual Query Condensing:** Don't feed raw user input (like *"Why does that happen?"*) to your vector database. Use past short-term memory to rephrase it: *"Why does Docker container data disappear on restart without volumes?"*
2. **Reranking & Filtering:** Filter retrieved chunks using cross-encoders (e.g., `bge-reranker-large`) so only top 2–3 relevant chunks enter the prompt.

---

## Summary Recommendations for Implementation

1. **Short-Term:** Implement a **Sliding Window + Socratic Summarizer** (store full turns for the immediate exchange, summarize older turns into a structured JSON state).
2. **Long-Term:** Store persistent concept mastery / weaknesses in **Redis/Postgres**, fetching this small payload on session start.
3. **Retrieval Memory Integration:** Use **Condensed Query Generation** to combine short-term chat context with the new query before searching your RAG vector index.
