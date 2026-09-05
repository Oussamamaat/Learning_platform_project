# ADR 0006: Darija TTS Survey (Research Only, No Training This Session)

**Status:** Survey complete; no candidate qualifies for immediate use; fine-tune plan identified
pending eval criteria before any GPU spend
**Date:** 2026-09-04
**Depends on:** `app/services/tts.py` (already-settled licensing constraints),
`docs/architecture/voice-assistant.md`, `POST_LEASE_MVP_SPRINT_PLAN.md` item 3

## Problem

Piper's `ar_JO-kareem-medium` (Jordanian Arabic, the closest off-the-shelf voice to Moroccan
Darija at MVP time) was rejected on live listening on the 2026-09-04 lease: "sounds so generic and
really bad, the Jordanian accent doesn't suit the Moroccan accent which is more fluid." No
replacement was selected. Two higher-quality alternatives, XTTS-v2 and MMS-TTS, were already
rejected earlier on licensing (Coqui CPML and CC-BY-NC, both non-commercial-only — unusable in a
B2B product) — that constraint is settled, not re-litigated here.

## Investigation method

Delegated a web survey (subagent, no code changes) covering: Piper checkpoints fine-tuned on
Moroccan Darija; other commercially-licensed self-hostable engines with any Arabic-dialect voice;
the AtlasIA/Moroccan-NLP community's output on HuggingFace, including whether
`atlasia/DODa-audio-dataset` (already cached locally, named in `voice-assistant.md` as the intended
fine-tune corpus) is actually TTS-suitable data or something else; and any other Darija/Maghrebi
TTS models. For each candidate: license (commercial-use permitted?), offline self-hostability,
quality signal, and immediate-use vs. needs-fine-tuning.

## Options considered

Every candidate found is one of: (a) usable immediately, no training; (b) needs fine-tuning; or
(c) disqualified outright (non-commercial license, no license declared, or no working model
artifact). The text-only-Darija-for-MVP option was evaluated as a genuine alternative throughout,
not a fallback of last resort.

## Evidence

**`atlasia/DODa-audio-dataset` is genuinely TTS-suitable**, not an ASR/translation artifact wearing
a TTS label: 12,743 sentence-level recordings, 9h46m, 7 named speakers (4F/3M), 16kHz, MIT-licensed,
Wit.ai-transcribed with manual correction. Its own dataset card lists TTS synthesis as an explicit
intended use. It is multi-speaker sentence-level data (not single-speaker studio recording), which
fits a modern multi-speaker/zero-shot base better than Piper's older single-speaker-oriented
architecture.

**Bucket (a) — usable immediately, no training: empty.** Every Darija-labeled TTS artifact found
either inherits a non-commercial license despite marketing as open, has no license declared at all,
or has a self-reported quality benchmark too poor to ship. The closest, Habibi-TTS's MAR checkpoint
(Apache-2.0 weights, MIT code, F5-TTS architecture), reports WER-S 41.5% and UTMOS 1.98/5 in its own
paper — not a shippable voice, a fine-tuning starting point.

**Bucket (b) — needs fine-tuning, ranked:**

| rank | candidate | base license | data | effort |
|---|---|---|---|---|
| 1 | OuteTTS-1.0 (0.6B–1B) | Apache-2.0 | `KandirResearch/DarijaTTS-clean` (21k samples, already OuteTTS-formatted) + `atlasia/DODa-audio-dataset` | Light — small model, existing recipe to adapt, single GPU, likely days |
| 2 | Habibi-TTS MAR (F5-TTS) warm-start | Apache-2.0 weights / MIT code | Continued fine-tune on DODa + tenant-domain audio | Medium — larger flow-matching model, heavier than #1, better starting prior |
| 3 | Qwen3-TTS + LoRA | Apache-2.0 base | DODa/DarijaTTS-clean, redo `loubna1101`'s unlicensed proof-of-concept properly | Medium — mechanically proven, needs a real license + eval attached |
| 4 | Piper trained from scratch on DODa | MIT | DODa-audio-dataset | Feasible but a worse architecture fit than 1–3 for 7-speaker data; lowest priority |

**Disqualified, named so the choice isn't silently re-litigated:** `medmac01/xtt2_darija_v0.1`
(built on XTTS-v2, inherits CPML). `KandirResearch/DarijaTTS-v0.1-500M` (self-labeled Apache-2.0,
but its base `OuteAI/OuteTTS-0.2-500M` is CC-BY-NC-4.0 — the downstream label appears to be a
self-reported error; treat as NC-encumbered). `NadaLb23/Tacotron2-Darija` and
`HAMMALE/speecht5-darija` (no license, no complete/attached model artifact).

## Decision

1. **Ship no TTS engine change this session.** No candidate clears the immediate-use bar.
2. **Darija ships text-only for the MVP; French keeps its working voice.** This is not a
   training-avoidance fallback — it is a real, evaluated option: `fr_FR-siwis-medium` is a native
   French voice that was never rejected, so asymmetric modality support (voice in one language,
   text in the other) already has precedent in this platform, and shipping it costs nothing beyond
   what already exists (`TtsEngine`'s `"none"` default fails loudly and predictably per-language if
   wired that way — a decision for implementation, not this ADR).
3. **The ready-to-execute fine-tune plan, for when GPU access resumes**: OuteTTS-1.0 (0.6B or 1B)
   fine-tuned on `KandirResearch/DarijaTTS-clean` plus `atlasia/DODa-audio-dataset`. Eval criteria
   to define **before** that run starts (not built this session, named so the fast-follow has a
   concrete next step): a held-out subset of DODa for objective WER/UTMOS-style scoring, plus a
   live-listening acceptance test mirroring how Piper's rejection was actually decided (the
   deciding factor there was live listening, not the synthesis-speed numbers `benchmark_results/
   phase2_2026-09-04_eval_tts_results.json` recorded).

## Rationale

Piper was chosen originally specifically because it was MIT — the same commercial-use bar every
candidate here was screened against, so this survey doesn't loosen a constraint the project already
committed to. Ranking OuteTTS-1.0 first is a license and effort argument, not a raw-quality
argument: it is the only base in a genuinely clean license lineage with an already-proven training
recipe (KandirResearch's own attempt, minus its base-model license problem) and a small enough model
to fine-tune quickly once GPU access resumes. Not training this session is consistent with the
§3 finetune-assessment decision (ADR to follow) — launching against undefined eval criteria is the
failure mode this project has already paid for once, and TTS quality is exactly as easy to
mis-judge from a synthesis-speed number as tutoring quality is from a WER number (see ADR 0009).

## Constraints acknowledged

The survey itself was free and local (web research, no GPU). The fine-tune plan is explicitly
gated on GPU access resuming — nothing here commits lease time; Phase B executes it only if items
1–2's evidence and this ADR's eval criteria are both in place first, per the sprint's own
sequencing decision.
