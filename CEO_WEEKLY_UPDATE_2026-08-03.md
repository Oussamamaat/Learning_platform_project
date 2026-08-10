# LLM Architecture Progress — Week Ending 2026-08-03

**Audience:** CEO
**Reading time:** ~10 minutes
**Prepared by:** Engineering

---

## 1. Data Preparation

We ingested tenant #1's 37-document corpus — covering three of its target regulatory domains, industrial safety, "sécurité," and blockchain-adjacent compliance — chunked at 400 characters with 50-character overlap and embedded with a multilingual sentence model into a pgvector store. Retrieval returns the top 5 matching chunks per query. Tenant #1's language mix is Darija (Arabic script, mandatory) alongside French; the platform itself is multilingual — French primary, English in scope.

One point needs to be stated plainly: **this corpus is a mock corpus, not the client's real regulatory documents.** It exists to validate that the retrieval pipeline works end-to-end — that a question goes in, the right document comes back, and the model grounds its answer in that document — while we wait on the official course materials. Every dataset-sizing decision downstream of this (row counts, domain balance, when to stop generating) is scoped to this placeholder corpus and will need to be revisited once real documents arrive. That revisit is not a failure of this week's work; it's a dependency we're flagging now so it isn't a surprise later.

## 2. Model Fine-Tuning

We fine-tuned Atlas-Chat-9B — a Gemma-2-based model already pretrained on roughly 450,000 Darija instructions, the language of tenant #1's learners — using LoRA via Unsloth, rather than training a model from scratch or teaching it a new language.

The efficiency case is direct: the adapter touches 54 million parameters, **0.585% of the model's 9.24 billion**, targeting only the attention and feed-forward projection layers — the embedding and output layers, which alone account for 17x the adapter's total size, are left untouched because we are not teaching new vocabulary, only reshaping behavior the model already has. Training ran on Kaggle's free dual-T4 tier in 4-bit precision with an 8-bit optimizer, keeping the entire fine-tune inside a no-cost compute budget. The resulting adapter was merged into a standalone deployable model on a single consumer laptop GPU in under 40 minutes.

The framing that makes this coherent: our training set is roughly 3,000 rows, under 1% the size of the data that gave Atlas its Darija fluency in the first place. A set this size cannot teach a language. What it can do — and what we designed it to do — is reshape how a model that already speaks tenant #1's language (Darija) behaves: when it cites a law, when it refuses, when it teaches Socratically instead of just answering.

## 3. Model Evaluation & Diagnostics

We evaluated tenant #1's fine-tuned model — internally named **`IBLOG_TUTOR`** — by running it against a held-out set of 307 questions never seen during training, then scoring its answers against the same automated checks used to grade the training data itself: does it answer in Arabic-script Darija, does it cite the law it's supposed to cite, does it correctly blend in French technical terms.

The headline result is positive and, on our single most important safety check, better than the untuned base model. In a controlled test, the base model was shown a question with no supporting legal text and correctly said "no such law is specified in this text." Our first fine-tuned attempt, shown the same question, confidently fabricated a law number. That is the failure mode we most need to avoid in a compliance product, and this week's model — `IBLOG_TUTOR` — passes that test cleanly: it declines when there's nothing to cite, and it names the correct article when there is.

One real gap remains, and we are not glossing over it. On the code-switching check — correctly weaving in French regulatory terms rather than translating them, which is how these documents are actually written — the model scores **60.8%**, against a **93.8%** baseline measured on the training data that was supposed to teach this exact behavior. That is a genuine 33-point gap, and while the model still clears our minimum acceptance bar, it clears it by well under one percentage point — closer to noise than to margin.

The open question is *where* that gap came from: whether it was already present in the trained adapter, or whether it was introduced during this week's merge step that converted the adapter into a standalone deployable model. Those have different fixes — one means retraining, the other means changing how we package the model — so we do not want to guess. The diagnostic next step is already scoped: re-run the identical 307-question evaluation against the pre-merge adapter. That isolates the merge step as a variable and tells us definitively which side of that line the problem sits on. It takes roughly 70 minutes and has not yet been run; it's the top item in next steps below.

## 4. RAG Pipeline Deployment

There is a documented gap between our original architecture plan and what is actually running today, and it's worth being explicit about rather than letting the documentation quietly go stale.

The original plan, still written into our architecture docs, is **vLLM with multi-LoRA serving**: one frozen base model with multiple lightweight adapters hot-swapped in per capability — a tutor adapter, a diagram-generation adapter, and so on, all sharing one set of base weights in memory. What's actually deployed for this week's milestone is the **opposite of that**: a single fully-merged model in Ollama. Merging bakes the adapter permanently into the weights, which is simpler to ship and faster to get to a demo, but it forecloses hot-swapping — you cannot pull a merged adapter back out.

This was a deliberate trade for MVP speed, not an oversight, and the recommended path is to keep it that way through the MVP milestone: ship the merged model now, and treat multi-LoRA as a post-MVP migration once we have more than one capability that actually needs to share a base model. Reversing that decision later costs an integration effort we have not started; making it now costs nothing we weren't already going to spend.

**Platform direction (decided):** a neutral multilingual base model with one per-tenant LoRA adapter — a tenant's language and domain become an adapter, not a baked-in base — to be in place before tenant #2 onboarding. The fine-tune in this update is tenant #1's adapter on that path.

The second open piece is conversational state. As built today, the serving path has no conversation-history mechanism at all — every question is answered independently, with no memory of what came before in the same session. That is a real product gap, because roughly half of the Socratic-teaching training data was specifically written as multi-turn dialogue, meaning we trained a multi-turn behavior into a serving path that cannot express multiple turns. This needs a decision this week, since it sits on the critical path for the "conversational assistant" MVP feature to work as designed. The three options are: add server-side session history (the architecturally correct fix), ship stateless for MVP and document the limitation, or push history-replay to the client. Recommendation is the first option, scoped as a Redis-backed session store keyed by conversation ID — it's a bounded, well-understood piece of work.

## 5. System Prompt Safety

This is the least visible decision this week and arguably the most safety-critical one, so it earns its own section rather than a footnote.

Gemma-2, the model family underneath Atlas-Chat, has no native "system" role in its chat template — every conversation turn is structured as user-then-model, with no separate channel for instructions. Our training data assumes a system prompt exists (it carries the retrieved legal context, the persona, the behavioral rules). We resolved this with a custom template that merges the system instructions into the first user turn, and the critical decision is this: **that exact template is asserted byte-identical between training and production, checked automatically at build time, not just documented.**

The reason this matters is that a silent mismatch here is invisible in normal testing and catastrophic in production. If the production prompt format drifts even slightly from what the model was trained on — a different phrase order, an extra newline, a missing delimiter — the model doesn't error out. It just quietly performs worse, in ways that look like random quality variance rather than a configuration bug, and there is no obvious signal pointing back to the cause. Automating the parity check at build time means that class of failure gets caught before deployment instead of being debugged after a client notices degraded answers. When we added language routing this week — French questions now answered in French, alongside the Arabic-script Darija default — we deliberately built it as a *separate* template rather than modifying the trained one, specifically to keep this invariant intact.

## 6. Next Steps, Open Questions, and Where We Need Your Call

Most of the remaining work is now decision-bound, not build-bound. We've compiled a full decision register (`resurrection.md`) covering every architectural choice made so far, but four items are genuinely time-sensitive and benefit from your input this week:

1. **Run the merge-vs-adapter control eval (~70 min, no cost).** Directly resolves the code-switching gap from Section 3. This should happen before we treat that gap as "the model's problem" or "the packaging's problem" — right now we don't know which.
2. **Is the 37-document corpus (Section 1) still a placeholder, or is it now the real corpus we should build against?** If real client materials are still coming, we should pause further dataset scaling rather than generate more rows against documents that are about to be replaced.
3. **Conversation history (Section 4).** The Socratic-tutoring MVP feature was trained to be multi-turn but currently cannot be served that way. This needs a build decision this week to stay on schedule.
4. **Licensing: is `IBLOG_TUTOR` staying internal, or is it being hosted for clients as a SaaS product?** This matters because Gemma's terms of use classify anything trained on Atlas-generated data as a "model derivative," and external hosting activates pass-through obligations — attribution, use-policy compliance — that internal-only use does not. If external hosting is the plan, as discussed, this needs a legal read before the next client-facing demo, not after.

Two additional quality gaps are worth your awareness but are not blocking: we've never run a native Darija speaker's judgment against the model's output (every quality number so far is machine-scored), and we've never measured whether our retrieval step actually returns the correct document — today, a wrong-document retrieval and a model hallucination would look identical in our metrics. Neither blocks this week's demo; both should be closed before a wider rollout.

---

## Recommended Demos

**Demo 1 — Base model vs. `IBLOG_TUTOR`, same prompt, side by side.**
This is the clearest way to make the fine-tuning investment visible. Ask both models the identical question and let the difference speak for itself — specifically, a question the base model gets wrong in a way that matters (declines to cite, or over-cites). Suggested prompt for this demo:

> *"شنو كايقول القانون على السلامة فالمصانع؟"* (What does the law say about factory safety?) — asked with **no supporting document in context.**

The base model tends to answer anyway, sometimes inventing a plausible-sounding article number. `IBLOG_TUTOR` should decline and say the information isn't available in what it was given — the exact behavior documented in Section 3, and the single strongest, most demonstrable result of this week's work.

**Demo 2 — Questions that show what's working right now.**

These are chosen to land in the model's proven strengths — grounded citation and Socratic teaching in tenant #1's language (Darija), and French-language handling — and to avoid the two known rough edges (code-switching density, and a refusal that can misstate its own scope). Use these with a real supporting document loaded into context via the RAG pipeline, not as bare questions.

*Arabic script (Darija), with document context loaded:*
- *"فهمني علاش خاصنا نلبسو معدات الحماية فهاد الورشة."* — "Explain to me why we need to wear protective equipment in this workshop." (Tests Socratic teaching style — expect a guiding question back, not a flat answer.)
- *"شنو كايقول القانون بالضبط على هاد النقطة؟"* — "What exactly does the law say on this point?" (Tests citation grounding — expect a specific article number, quoted from the loaded document.)
- *"عطيني كيز ديال 3 اسئلة على هاد الموضوع."* — "Give me a 3-question quiz on this topic." (Tests structured quiz generation — expect clean, valid structured output.)

*French, with document context loaded:*
- *"Quelles sont les obligations de l'employeur en matière de sécurité industrielle selon ce document ?"* (Tests French-language routing, added this week, and citation grounding together.)
- *"Explique-moi cette clause comme si je découvrais le sujet."* (Tests Socratic/explanatory tone in French.)

One thing to avoid live: don't probe with a question clearly outside the loaded document's domain (for example, a blockchain question while a workplace-safety document is loaded) — we have an open, known issue where the refusal in that case can misstate *why* it's declining. It's a safe failure (it declines rather than fabricating) but it reads awkwardly in front of an audience, and it's already on the list to fix.
