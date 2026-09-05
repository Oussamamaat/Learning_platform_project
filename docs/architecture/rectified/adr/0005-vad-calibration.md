# ADR 0005: VAD Calibration

**Status:** Phase A complete (config knob, instrumentation, offline calibration, tests); root-cause
diagnosis of the live mic failure deferred to a live-mic Phase B run
**Date:** 2026-09-04
**Depends on:** `benchmark_results/README.md`, `POST_LEASE_MVP_SPRINT_PLAN.md` item 2

## Problem

On the 2026-09-04 Akash lease, the live voice interface did not reliably detect speech
(`speech_start` never fired) and barge-in (talking over the assistant mid-response) never
interrupted playback — both trace to the same code path,
`app.services.vad.EnergyEndpointer.push()`, called identically from `app/routers/voice.py:293`
(barge-in) and `:303` (endpointing). A prior planning pass carried forward a hypothesis that the
fixed RMS threshold (`500.0`, non-adaptive) was miscalibrated — "too high" — for a real
browser/mic setup, and that it needed lowering, plus that it had no config override so testing a
fix on the lease would have cost a pod restart or image rebuild.

## Investigation method

Read `app/services/vad.py` in full: `EnergyEndpointer` is a dependency-free RMS-threshold VAD,
`threshold=500.0` (RMS of raw int16 samples, not dB), `hangover_ms=400`, `min_speech_ms=200`,
operating on 16kHz mono 16-bit PCM in 20ms/640-byte frames. `tests/test_vad.py` (7 tests) uses only
synthetic square waves at amplitude 0 and 10000 — nothing had ever tested the default against real
speech.

Wrote and ran an offline sweep (`scripts/calibrate_vad.py`) driving the REAL `EnergyEndpointer`
against the repo's 30 real, already-committed 16kHz mono speech recordings
(`tests/data/voice_eval/*.wav` — 16 Darija, 8 French, 6 code-switched), at thresholds 500.0 (current
default), 300.0, 200.0, 100.0, 60.0. No mic, no GPU, no model — pure signal processing over
committed fixtures.

Traced the frontend capture path (`frontend/src/hooks/useVoiceSession.ts`,
`frontend/src/audio/pcm-worklet.js`) to identify what could plausibly differ between this offline
corpus and a live browser session.

## Options considered

- **Lower the default threshold** (the carried-over hypothesis) — tested directly by the sweep,
  not assumed.
- **Guess and ship a lower default without evidence** — rejected once the sweep contradicted the
  premise; shipping an unjustified change here trades a real (if unconfirmed) upstream cause for a
  cosmetic one that degrades false-trigger resistance for no measured benefit.
- **Defer all VAD work to the lease** — rejected: the config knob, instrumentation, and offline
  calibration are correct and useful regardless of what a live run finds, and they're what turn a
  future lease run into a diagnosable trace instead of a repeat of "it didn't work."

## Evidence

Frame-RMS distribution across all 6291 frames in the 30-file corpus: p10 = 23, p25 = 145,
**p50 = 964**, p75 = 2344, p90 = 4728. 61.4% of all frames exceed 500.

| threshold | files detected (of 30) | mean segments/file | mean time-to-detect |
|---|---|---|---|
| **500.0 (current default)** | **30/30** | 1.20 | 834 ms |
| 300.0 | 30/30 | 1.17 | 785 ms |
| 200.0 | 30/30 | 1.13 | 759 ms |
| 100.0 | 30/30 | 1.10 | 715 ms |
| 60.0 | 30/30 | 1.03 | 675 ms |

**The current threshold is not the confirmed cause.** Every tested threshold down to 60.0 detects
speech in all 30 files; lowering it buys at most ~160ms of earlier detection on this corpus and
costs false-trigger resistance (p25 of real speech frames is 145 — near several of the lower
candidate thresholds).

**Where the real cause more plausibly lives**, traced but not yet confirmed without a live run:
`useVoiceSession.ts` requests `getUserMedia({echoCancellation: true, noiseSuppression: true,
autoGainControl: true})` — all three browser-side audio processors run before a sample ever reaches
the server, and `noiseSuppression`/`autoGainControl` specifically can suppress or renormalize
low-energy speech in a way a fixed server-side RMS floor cannot see or compensate for.
`new AudioContext({sampleRate: 16000})` is a *request*, not a guarantee — if the browser/OS declines
it, `pcm-worklet.js`'s 320-sample frames are no longer 20ms, silently changing what `hangover_ms`/
`min_speech_ms` mean. Nothing in `voice.py` validates inbound frame size against `FRAME_BYTES`.

## Decision

1. **Do not lower the default threshold.** Unjustified by the only real-speech evidence available.
2. **Ship, regardless of root cause (all landed this session):**
   - `vad_threshold`, `vad_hangover_ms`, `vad_min_speech_ms` added to `app/config.py`'s `Settings`,
     following the `retrieval_backend` comment-block convention; `app/routers/voice.py`'s
     `EnergyEndpointer()` construction now reads them explicitly instead of relying on bare
     constructor defaults. Confirmed env-var override works
     (`VAD_THRESHOLD=123.0` → `Settings().vad_threshold == 123.0`) via pydantic-settings, no new
     plumbing. This alone fixes the actual blocker from the prior lease — testing a threshold
     change no longer requires a pod restart or image rebuild, only an env var + process restart.
   - `vad_debug_log` (off by default) plus a `_debug_push` helper in `voice.py` that logs, per
     inbound frame: byte length, computed RMS, session state, and any endpointer event — gated
     behind the new setting so it costs nothing when off. This is what makes a future live-mic run
     diagnostic (an RMS trace to compare against the offline distribution above) rather than another
     unexplained "it didn't work."
   - `scripts/calibrate_vad.py` committed so the sweep table above is a reproducible one-command
     artifact, not a one-off console run.
   - Two new tests in `tests/test_vad.py` covering the settings-to-constructor wiring.
3. **Root-cause diagnosis of the live failure is deferred to Phase B** (a live mic session against
   a redeployed lease, per the sprint plan) — capture the RMS trace via `vad_debug_log`, compare
   against the offline distribution, and confirm or refute the browser-AGC/sample-rate/frame-size
   hypotheses above with real data before changing anything else.

## Rationale

The standing rule (confirm before fixing) applies here exactly as it did for item 1: the carried-
over hypothesis was specific and testable, the test was cheap and free to run, and it came back
negative. Shipping a threshold change anyway would have been guessing dressed up as a fix. What
*is* correct to ship regardless of the live root cause — configurability and instrumentation — was
shipped, because both are justified independently of which hypothesis turns out to be right, and
both are exactly what the module's own docstring already asked for ("tune per deployment") and what
the prior lease run was blocked on (no override without an image rebuild).

## Constraints acknowledged

No GPU needed for any of this — pure signal processing and configuration plumbing, fully doable
locally. What genuinely needs live infrastructure is the live-mic RMS trace itself: a browser's
actual AGC/noise-suppression behavior and actual delivered sample rate cannot be observed from a
committed `.wav` corpus, only from a real `getUserMedia` session reaching a real server. That is
scoped narrowly to Phase B, with the instrumentation already in place so the run only needs to
happen once.
