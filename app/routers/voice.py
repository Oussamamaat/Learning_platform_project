"""
Voice Session (open-mic conversational voice)
──────────────────────────────────────────────
WS /api/v1/voice/session -- see docs/architecture/voice-assistant.md for
the full blueprint this implements (latency budget, deployment comparison,
vendor bake-off status).

Cannot be exercised end-to-end on this machine: settings.stt_engine and
settings.tts_engine both default to "none" (no vendor selected -- Phase 0
bake-off not run, needs a rented GPU this laptop's 8GB card does not have
alongside the resident tutor model). The state machine, turn-resolution
reuse (app.services.turn), and streaming-generation wiring below are
real and covered by tests/test_voice_session.py against fake STT/TTS
engines; they have NOT been exercised against a live STT/TTS vendor or a
live microphone.

Transport is raw 16-bit mono PCM @16kHz binary WebSocket frames, not the
Opus the blueprint recommends -- a deliberate MVP simplification (no codec
dependency) rather than a rejection of that recommendation; Opus is a
bandwidth optimization, not a correctness requirement, over same-machine/
dev-network WebSocket traffic. Revisit before a real network deployment.

Endpointing is app.services.vad.EnergyEndpointer, a dependency-free
RMS-threshold VAD -- a real, working default, coarser than the Silero VAD
the blueprint calls for (onnxruntime is not yet a project dependency).

Sync services (STT, TTS, turn resolution, Ollama via urllib) are all
blocking -- every call into them is dispatched through asyncio.to_thread
so one voice session can never stall the event loop for every other
connection this process serves (chat, quiz, ingest all share it).
"""
import asyncio
import json
import logging
import re
import threading
import uuid
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.config import get_tenant_id, get_user_id
from app.errors import AppError
from app.services.citations import extract_citations
from app.services.llm import stream_llm_response
from app.services.stt import get_stt_engine, SttUnavailableError, TranscriptChunk
from app.services.tts import get_tts_engine, TtsUnavailableError
from app.services.turn import resolve_turn, refusal_text, load_prior_turns, persist_turn
from app.services.vad import EnergyEndpointer, FRAME_BYTES, SAMPLE_RATE

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/voice", tags=["voice"])

# Sentence-boundary split for per-sentence TTS streaming -- same pattern as
# app.services.history's own _SENTENCE_SPLIT, kept as a separate constant
# here rather than importing that one: it exists there for an unrelated
# concern (extracting already-asked Socratic questions from stored turns),
# and coupling this module to history.py for a regex would be a coupling
# with no shared reason to change together.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?؟])\s+")


class _State:
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"


async def _send_json(websocket: WebSocket, payload: dict) -> None:
    await websocket.send_text(json.dumps(payload))


def _split_ready_sentences(buffer: str) -> tuple[list[str], str]:
    """Split `buffer` into complete sentences plus a trailing remainder
    that may still grow (no terminal punctuation yet). Complete sentences
    are ready to hand to TTS immediately; the remainder is fed back into
    the next call once more text has streamed in."""
    parts = _SENTENCE_SPLIT.split(buffer)
    if len(parts) <= 1:
        return [], buffer
    return parts[:-1], parts[-1]


def _answer_worker(
    *,
    turn,
    prior_turns: list[dict],
    cancel_flag: threading.Event,
    loop: asyncio.AbstractEventLoop,
    out_queue: "asyncio.Queue[tuple[str, object]]",
    tts_engine,
) -> str:
    """Runs on a worker thread (see module docstring): drains
    stream_llm_response, splits it into sentences as they complete, and
    synthesizes+enqueues audio per sentence so the client hears sentence 1
    while the model is still generating sentence 2 -- see
    docs/architecture/voice-assistant.md Part 4 point 1, the single
    biggest latency lever in the blueprint.

    Cooperative cancellation only: setting `cancel_flag` stops this
    function from sending any FURTHER text/audio and returns immediately
    after the in-flight chunk, but does not abort the underlying HTTP
    request already sent to Ollama -- urllib gives no clean way to cancel
    a socket read from another thread. Ollama will finish generating that
    turn server-side regardless; this only stops it from reaching the
    user. Acceptable for MVP barge-in (the user-visible effect -- instant
    silence -- is correct); a real cancel-in-flight would need Ollama's
    streaming response read on a socket this thread can shutdown()
    directly, tracked as a follow-up, not required for MVP.

    Returns the text actually enqueued for the client (used by the caller
    to persist only what was truly delivered, not the full generation --
    see app.services.turn.persist_turn's docstring on why barge-in must
    never let the model believe it said something the user never heard).
    """

    def emit(kind: str, item: object) -> None:
        loop.call_soon_threadsafe(out_queue.put_nowait, (kind, item))

    def speak(sentence: str, delivered: list[str]) -> None:
        sentence = sentence.strip()
        if not sentence:
            return
        delivered.append(sentence)
        try:
            audio = tts_engine.synthesize(sentence, language=turn.response_lang)
            emit("bytes", audio)
        except TtsUnavailableError as e:
            emit("text", json.dumps({"type": "error", "code": "tts_unavailable", "detail": str(e)}))

    delivered: list[str] = []
    buffer = ""
    try:
        for delta in stream_llm_response(
            query=turn.message, context=turn.context, domain=turn.domain,
            language=turn.response_lang, history=prior_turns,
        ):
            if cancel_flag.is_set():
                break
            buffer += delta
            emit("text", json.dumps({"type": "answer.delta", "text": delta}))
            ready, buffer = _split_ready_sentences(buffer)
            for sentence in ready:
                if cancel_flag.is_set():
                    break
                speak(sentence, delivered)
        if not cancel_flag.is_set():
            speak(buffer, delivered)
    except AppError as e:
        emit("text", json.dumps({"type": "error", "code": e.code, "detail": e.message}))
    finally:
        emit("text", json.dumps({"type": "audio.end"}))

    return " ".join(delivered)


@router.websocket("/session")
async def voice_session(
    websocket: WebSocket,
    tenant_id: Optional[str] = None,
    session_id: Optional[str] = None,
    domain: Optional[str] = None,
) -> None:
    await websocket.accept()

    tenant_id = get_tenant_id(tenant_id)
    user_id = get_user_id()
    session_id = session_id or uuid.uuid4().hex

    try:
        stt_engine = get_stt_engine()
        tts_engine = get_tts_engine()
    except (SttUnavailableError, TtsUnavailableError) as e:
        await _send_json(websocket, {"type": "error", "code": "voice_unavailable", "detail": str(e)})
        await websocket.close()
        return

    endpointer = EnergyEndpointer()
    state = _State.LISTENING
    # Set after the first successful (non-refusal) turn, then passed back
    # in as resolve_turn's explicit_language on every later turn in this
    # session -- pins the session to one model instead of paying the
    # ~30s VRAM swap a mid-session French<->Darija flip costs on a card
    # that cannot hold both tutors resident (docs/architecture/
    # cloud-scaling-plan.md Part 5 / Part 4 optimization 3). A real
    # in-message "switch to French" instruction still overrides this --
    # resolve_language's own precedence (explicit < in-message < sticky
    # override < script) is unchanged; this only supplies the "sticky"
    # layer's starting value from THIS session's own history instead of
    # leaving it unset on every turn.
    pinned_language: Optional[str] = None

    loop = asyncio.get_running_loop()
    out_queue: "asyncio.Queue[tuple[str, object]]" = asyncio.Queue()
    cancel_flag = threading.Event()
    worker_task: Optional[asyncio.Task] = None

    async def drain_outbound() -> None:
        while True:
            kind, item = await out_queue.get()
            if kind == "close":
                return
            if kind == "text":
                await websocket.send_text(item)
            else:
                await websocket.send_bytes(item)

    drain_task = asyncio.create_task(drain_outbound())

    async def begin_answer(turn) -> None:
        nonlocal state, worker_task
        state = _State.THINKING
        prior_turns = await asyncio.to_thread(load_prior_turns, turn)

        if turn.is_refusal:
            text = refusal_text(turn)
            await _send_json(websocket, {"type": "answer.delta", "text": text})
            await _send_json(websocket, {"type": "audio.start", "sample_rate": tts_engine.sample_rate})
            audio = None
            try:
                audio = await asyncio.to_thread(tts_engine.synthesize, text, language=turn.response_lang)
            except TtsUnavailableError as e:
                await _send_json(websocket, {"type": "error", "code": "tts_unavailable", "detail": str(e)})
            if audio is not None:
                await websocket.send_bytes(audio)
            await _send_json(websocket, {"type": "audio.end"})
            await asyncio.to_thread(persist_turn, turn, assistant_content=text)
            state = _State.LISTENING
            return

        # extract_citations returns {(head, number): {"canonical", "arabizi"}}
        # -- a dict with TUPLE keys, which json.dumps cannot serialize (and
        # which the frontend's `citations: string[]` contract does not want
        # anyway). Send the canonical reference labels as a flat list. This
        # is the common path for tenant #1 (regulatory context is full of
        # article/law numbers), so serializing the raw dict here crashed
        # every grounded answer -- see tests/test_voice_session.py.
        citation_labels = [entry["canonical"] for entry in extract_citations(turn.context).values()]
        await _send_json(websocket, {"type": "citations", "sources": citation_labels})
        await _send_json(websocket, {"type": "audio.start", "sample_rate": tts_engine.sample_rate})
        state = _State.SPEAKING
        cancel_flag.clear()

        def run_and_persist() -> None:
            nonlocal state, pinned_language
            spoken = _answer_worker(
                turn=turn, prior_turns=prior_turns, cancel_flag=cancel_flag,
                loop=loop, out_queue=out_queue, tts_engine=tts_engine,
            )
            persist_turn(turn, assistant_content=spoken or refusal_text(turn))
            pinned_language = turn.response_lang
            state = _State.LISTENING

        worker_task = asyncio.create_task(asyncio.to_thread(run_and_persist))

    async def handle_utterance(audio_bytes: bytes) -> None:
        nonlocal worker_task
        if not audio_bytes:
            return
        try:
            transcript: TranscriptChunk = await asyncio.to_thread(
                stt_engine.transcribe, audio_bytes, sample_rate=SAMPLE_RATE,
                language_hint=pinned_language,
            )
        except SttUnavailableError as e:
            await _send_json(websocket, {"type": "error", "code": "stt_unavailable", "detail": str(e)})
            return
        if not transcript.text.strip():
            return
        await _send_json(websocket, {"type": "transcript.final", "text": transcript.text})
        turn = await asyncio.to_thread(
            resolve_turn, transcript.text, tenant_id=tenant_id, user_id=user_id,
            session_id=session_id, requested_domain=domain, explicit_language=pinned_language,
        )
        await begin_answer(turn)

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break

            if "bytes" in message and message["bytes"] is not None:
                frame = message["bytes"]
                if state == _State.SPEAKING:
                    # Barge-in: any speech energy while the assistant is
                    # talking cancels the in-flight answer (see
                    # _answer_worker's docstring for what "cancels" means)
                    # and tells the client to flush its playback queue --
                    # sample-accurate flush is the client's job (a WebAudio
                    # scheduled buffer queue, not <audio>; see
                    # frontend/src/hooks/useVoiceSession.ts).
                    event = endpointer.push(frame)
                    if event == "speech_start":
                        cancel_flag.set()
                        if worker_task is not None:
                            await worker_task
                        await _send_json(websocket, {"type": "barge_in"})
                        state = _State.LISTENING
                    continue
                if state != _State.LISTENING:
                    continue
                event = endpointer.push(frame)
                if event == "speech_start":
                    await _send_json(websocket, {"type": "transcript.partial", "text": ""})
                elif event == "speech_end":
                    audio_bytes = endpointer.take_utterance()
                    await handle_utterance(audio_bytes)
            elif "text" in message and message["text"] is not None:
                try:
                    control = json.loads(message["text"])
                except json.JSONDecodeError:
                    continue
                if control.get("type") == "end":
                    break
    except WebSocketDisconnect:
        pass
    finally:
        cancel_flag.set()
        if worker_task is not None:
            try:
                await worker_task
            except Exception:
                logger.exception("voice session %s: answer worker raised during teardown", session_id)
        await out_queue.put(("close", None))
        await drain_task
        try:
            await websocket.close()
        except Exception:
            pass
