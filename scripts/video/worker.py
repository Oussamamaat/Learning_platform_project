"""
Explanatory-video worker -- the missing half of the video_jobs contract.

docs/architecture/video-generation-interface.md has specified this since
2026-08-18: we write 'pending' rows, "the video worker" claims them and
reports back. Nothing ever claimed one. This is that worker.

    GET   /api/v1/video/jobs?status=pending     claim
    PATCH /api/v1/video/jobs/{id} status=processing
        Phase 1  scenes + video prompts   (partner's Ollama planner)
        Phase 2  narration audio          (OUR app.services.tts)
        Phase 3  clips + assembly         (partner's renderer + ffmpeg)
    PATCH /api/v1/video/jobs/{id} status=ready|error

Runs in .gguf_venv. One job at a time, same single-worker shape as
app/services/ingest_queue.py -- and for a sharper reason here: Phase 1
holds an Ollama model and Phase 2 holds a 5.6GB XTTS checkpoint, which on
an 8GB card must not co-reside. Phases run sequentially, so they don't.

WHY PHASE 2 IS OURS AND NOT THEIRS
The partner repo splits into three mutually incompatible venvs (.venv for
Ollama, venv_tts for Chatterbox, venv_darija for XTTS) because Ollama and
Chatterbox deadlocked in one process. We don't inherit that: their Phase 1
is a thin `import ollama` HTTP client, and for Phase 2 this repo already
runs the SAME checkpoint their narration_tts.py downloads
(medmac01/darija_xtt_2.0) through app/services/tts.py's XttsDarijaEngine --
already out-of-process in its own venv, already warmed at app startup,
already carrying fallback and idle VRAM release, and multilingual enough
to serve fr and ar-MA from one engine (see that class's docstring). So the
whole worker is one venv, and their two TTS venvs never need to exist here.

Usage (from repo root):
    .gguf_venv/Scripts/python.exe scripts/video/worker.py            # poll forever
    .gguf_venv/Scripts/python.exe scripts/video/worker.py --once     # one drain pass
    .gguf_venv/Scripts/python.exe scripts/video/worker.py --once --mock
"""
import argparse
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request
import wave
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Callable, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from app.config import get_settings  # noqa: E402

logger = logging.getLogger("video_worker")

# The partner pipeline is scene-mode only. Their orchestrator's every phase
# loop reads `if job.get("mode") != "scene": continue`, and
# src/avatar_generator.py is an empty file -- so an avatar job has nothing
# to run. Refused with this message rather than silently rendering scene
# footage under an avatar label.
AVATAR_UNAVAILABLE = (
    "Avatar mode is not implemented in the video pipeline yet "
    "(avatar_generator.py is still an empty stub upstream, and every phase "
    "of the partner orchestrator skips non-scene jobs). Generate this video "
    "in scene mode, or wait for avatar support to land."
)


# ---------------------------------------------------------------------------
# Partner pipeline import
# ---------------------------------------------------------------------------

def load_partner_pipeline(pipeline_dir: str):
    """Import the partner modules from a local clone of HajarAmamou/PFA.

    Their modules import each other FLAT (`from frame_extractor import
    FFMPEG_BINARY`, `from planner import generate_scenes`), so both src/
    and src/scene_planner/ go on sys.path rather than being imported as a
    package. Their repo is never edited from here -- a git pull in that
    clone is how their in-flight work (wan_client, avatar_generator)
    arrives.
    """
    root = Path(pipeline_dir).expanduser().resolve()
    src = root / "src"
    scene_planner = src / "scene_planner"
    if not scene_planner.is_dir():
        raise FileNotFoundError(
            f"settings.video_pipeline_dir points at {root}, which does not look like a "
            "clone of the video pipeline (expected src/scene_planner/ inside it). "
            "Clone https://github.com/HajarAmamou/PFA and set video_pipeline_dir to it."
        )
    for path in (str(scene_planner), str(src)):
        if path not in sys.path:
            sys.path.insert(0, path)

    settings = get_settings()
    # Their planner/prompt_builder call `ollama.chat` with no explicit
    # host, which defaults to localhost:11434. Set before the import below,
    # because the ollama package reads OLLAMA_HOST when it builds its
    # default client -- after the import it would be too late. This is how
    # their code follows OUR configured Ollama without being edited.
    os.environ.setdefault("OLLAMA_HOST", settings.ollama_base_url)
    # Same trick for the partner's ffmpeg lookup (frame_extractor.py and
    # video_editor.py both read FFMPEG_BINARY at import time). Only set
    # when this deployment configured one -- otherwise their "ffmpeg"
    # default is correct and already on PATH.
    if settings.tts_xtts_ffmpeg_bin:
        os.environ.setdefault(
            "FFMPEG_BINARY", str(Path(settings.tts_xtts_ffmpeg_bin) / "ffmpeg")
        )

    import job_adapter
    import planner
    import prompt_builder
    import frame_extractor
    import video_editor
    import wan_client

    # Both modules hardcode MODEL = "qwen2.5:7b-instruct". Point them at
    # whatever this deployment actually has pulled, without editing theirs.
    planner.MODEL = settings.video_planner_model
    prompt_builder.MODEL = settings.video_planner_model

    return {
        "job_adapter": job_adapter,
        "planner": planner,
        "prompt_builder": prompt_builder,
        "frame_extractor": frame_extractor,
        "video_editor": video_editor,
        "wan_client": wan_client,
    }


def _call_with_timeout(fn, *args, timeout: float, **kwargs):
    """Run fn with a deadline, in an executor created for THIS call only.

    The partner's orchestrator records why the executor is not shared: a
    local model can loop silently with no error, and a reused worker thread
    left blocked by one such call then blocked every later call too. A
    Python thread can't be killed, so an abandoned call keeps running in the
    background -- but it no longer holds up the queue.
    """
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        return executor.submit(fn, *args, **kwargs).result(timeout=timeout)
    finally:
        executor.shutdown(wait=False)


# ---------------------------------------------------------------------------
# API client (the contract, over HTTP -- no shared code with the app)
# ---------------------------------------------------------------------------

class VideoApi:
    """The four endpoints of docs/architecture/video-generation-interface.md.

    Deliberately HTTP rather than importing the router: the worker is a
    separate process by design (it holds GPU models the API process must
    not), and going through the same endpoints the partner's own worker
    would use keeps this honest about what the contract actually supports.
    """

    def __init__(self, base_url: str, role: str = "admin", timeout: float = 30.0):
        self.base = base_url.rstrip("/") + "/api/v1/video"
        self.role = role
        self.timeout = timeout

    def _request(self, path: str, method: str = "GET", payload: Optional[dict] = None):
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"X-User-Role": self.role}
        if data is not None:
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(
            f"{self.base}{path}", data=data, headers=headers, method=method
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def claim_pending(self) -> list:
        return self._request("/jobs?status=pending")

    def mark_processing(self, job_id: str) -> None:
        self._request(f"/jobs/{job_id}", method="PATCH", payload={"status": "processing"})

    def mark_ready(self, job_id: str, video_url: str) -> None:
        self._request(
            f"/jobs/{job_id}", method="PATCH",
            payload={"status": "ready", "video_url": video_url},
        )

    def mark_error(self, job_id: str, message: str) -> None:
        self._request(
            f"/jobs/{job_id}", method="PATCH",
            payload={"status": "error", "error_message": message[:2000]},
        )


# ---------------------------------------------------------------------------
# Phase 2 -- narration through our own TTS
# ---------------------------------------------------------------------------

# app.services.tts speaks "fr" | "darija"; video_jobs speaks the Language
# enum's "fr" | "en" | "ar-MA".
_TTS_LANGUAGE = {"fr": "fr", "ar-MA": "darija", "en": "fr"}


def write_pcm_as_wav(pcm: bytes, sample_rate: int, output_path: str) -> str:
    """Wrap raw 16-bit mono PCM in a WAV container.

    app.services.tts returns bare PCM because its caller (the voice
    WebSocket) streams it straight to a browser that was told the rate out
    of band. ffmpeg gets no such briefing, so the header has to exist
    before video_editor muxes this against a clip.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return str(out)


def run_phase_narration(scenes: list, language: str, job_dir: Path) -> None:
    """Synthesize one WAV per scene, in place (sets scene['audio_path'])."""
    from app.services.tts import get_tts_engine

    engine = get_tts_engine()
    tts_language = _TTS_LANGUAGE.get(language, "fr")
    logger.info("Phase 2: narration via %s (%s)", engine.name, tts_language)

    for scene in scenes:
        narration = (scene.get("narration") or "").strip()
        if not narration:
            raise ValueError(
                f"Scene {scene.get('scene_id')} has no narration text to speak."
            )
        pcm = engine.synthesize(narration, language=tts_language)
        scene["audio_path"] = write_pcm_as_wav(
            pcm, engine.sample_rate, str(job_dir / f"scene_{scene.get('scene_id')}_audio.wav")
        )


# ---------------------------------------------------------------------------
# Phases 1 and 3 -- the partner's code, driven from here
# ---------------------------------------------------------------------------

def run_phase_prompts(modules: dict, adapted: dict, timeout: float) -> list:
    """Split the explanation into scenes and build each scene's video prompt.

    Scene 1 gets an initial (text->video) prompt; every later scene gets a
    continuation prompt, which is what lets Phase 3 seed it with the
    previous clip's last frame for visual continuity.
    """
    planner = modules["planner"]
    prompt_builder = modules["prompt_builder"]

    logger.info("Phase 1: planning scenes (%s)", get_settings().video_planner_model)
    plan = _call_with_timeout(planner.generate_scenes, adapted, timeout=timeout)
    plan = planner.validate_scenes(plan, expected_duration=adapted["duration"])
    scenes = plan["scenes"]

    for index, scene in enumerate(scenes):
        is_continuation = index > 0
        builder = (
            prompt_builder.build_continuation_prompt
            if is_continuation
            else prompt_builder.build_video_prompt
        )
        scene["video_prompt"] = _call_with_timeout(builder, scene, timeout=timeout)
        scene["is_continuation"] = is_continuation

    logger.info("Phase 1: %d scene(s) planned", len(scenes))
    return scenes


def resolve_renderer(modules: dict, force_mock: bool = False) -> Callable:
    """Pick the real Wan2.2 renderer or the partner's mock.

    The mock is not a stub: it shells ffmpeg to produce a real, playable
    clip of the right duration and dimensions, so frame extraction, audio
    muxing and concatenation are all genuinely exercised -- only the
    footage is a solid colour. That makes the no-GPU path a true end-to-end
    test rather than a pretend one.
    """
    settings = get_settings()
    if force_mock or not settings.comfyui_url:
        logger.info("Phase 3: mock renderer (settings.comfyui_url is not set)")
        return modules["wan_client"].generate_video_mock

    from scripts.video.wan_comfyui import make_renderer

    logger.info("Phase 3: ComfyUI renderer at %s", settings.comfyui_url)
    return make_renderer(settings.comfyui_url, settings.comfyui_workflow_path)


def run_phase_render(modules: dict, scenes: list, job_dir: Path, render: Callable) -> str:
    """Generate each scene's clip, then mux narration and concatenate."""
    frame_extractor = modules["frame_extractor"]
    video_editor = modules["video_editor"]

    clip_paths, audio_paths = [], []
    reference_image = None

    logger.info("Phase 3: rendering %d clip(s)", len(scenes))
    for scene in scenes:
        scene_id = scene.get("scene_id")
        clip_path = render(
            prompt=scene["video_prompt"],
            reference_image=reference_image,
            duration=scene["duration"],
            output_path=str(job_dir / f"scene_{scene_id}_clip.mp4"),
        )
        clip_paths.append(clip_path)
        audio_paths.append(scene["audio_path"])
        # Seeds the NEXT scene, so consecutive clips read as one continuous
        # video instead of unrelated shots.
        reference_image = frame_extractor.extract_last_frame(
            clip_path, output_path=str(job_dir / f"scene_{scene_id}_last_frame.png")
        )

    return video_editor.assemble_video(
        clip_paths=clip_paths,
        audio_paths=audio_paths,
        output_path=str(job_dir / "video_finale.mp4"),
    )


# ---------------------------------------------------------------------------
# One job
# ---------------------------------------------------------------------------

def process_job(job: dict, modules: dict, api: VideoApi, force_mock: bool = False) -> str:
    """Run one job end to end and return the relative URL of the video.

    Raises on failure; the caller turns that into status='error' so the row
    never sits at 'processing' forever with nothing watching it.
    """
    settings = get_settings()
    job_id = job["id"]

    if job.get("mode", "scene") == "avatar":
        raise NotImplementedError(AVATAR_UNAVAILABLE)

    job_dir = Path(settings.video_output_dir) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Their adapter turns a video_jobs row into the normalized dict their
    # planner expects -- the row already carries exactly the field
    # signature it detects as 'avatar' shape, which is the job envelope,
    # not the render mode.
    adapted = modules["job_adapter"].adapt_avatar_job(job)

    scenes = run_phase_prompts(modules, adapted, timeout=settings.video_planner_timeout_seconds)
    run_phase_narration(scenes, job["language"], job_dir)
    final_path = run_phase_render(modules, scenes, job_dir, resolve_renderer(modules, force_mock))

    # Checkpoint alongside the video: the scene plan, prompts and audio
    # paths that produced it. Costs nothing and is the difference between
    # "the video looks wrong" and knowing which scene prompt caused it.
    (job_dir / "pipeline.json").write_text(
        json.dumps({"job": job, "scenes": scenes, "final": final_path}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Job %s rendered -> %s", job_id, final_path)
    return f"/media/{job_id}/video_finale.mp4"


def drain_once(api: VideoApi, modules: dict, force_mock: bool = False) -> int:
    """Process every currently-pending job. Returns how many were handled."""
    try:
        pending = api.claim_pending()
    except urllib.error.URLError as exc:
        logger.warning("Cannot reach the API: %s", exc)
        return 0

    for job in pending:
        job_id = job["id"]
        logger.info("Claiming job %s (%s, %s)", job_id, job.get("mode"), job.get("language"))
        try:
            api.mark_processing(job_id)
        except urllib.error.URLError as exc:
            logger.warning("Could not claim %s: %s", job_id, exc)
            continue

        try:
            api.mark_ready(job_id, process_job(job, modules, api, force_mock))
        except FutureTimeoutError:
            api.mark_error(
                job_id,
                f"Scene planning exceeded "
                f"{get_settings().video_planner_timeout_seconds:.0f}s and was abandoned.",
            )
        except Exception as exc:
            logger.exception("Job %s failed", job_id)
            api.mark_error(job_id, f"{type(exc).__name__}: {exc}")

    return len(pending)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--once", action="store_true", help="One drain pass, then exit.")
    parser.add_argument(
        "--mock", action="store_true",
        help="Force the mock renderer even if settings.comfyui_url is set.",
    )
    parser.add_argument("--api-base", default="http://127.0.0.1:8000")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    settings = get_settings()
    if not settings.video_pipeline_dir:
        logger.error(
            "settings.video_pipeline_dir is empty -- clone "
            "https://github.com/HajarAmamou/PFA and set it (e.g. VIDEO_PIPELINE_DIR=... in .env). "
            "The API keeps accepting jobs; they stay 'pending' until a worker can run."
        )
        return 2

    try:
        modules = load_partner_pipeline(settings.video_pipeline_dir)
    except (FileNotFoundError, ImportError) as exc:
        logger.error("%s", exc)
        return 2

    api = VideoApi(args.api_base)
    if args.once:
        drain_once(api, modules, args.mock)
        return 0

    logger.info("Polling %s every %.0fs", args.api_base, settings.video_worker_poll_seconds)
    while True:
        drain_once(api, modules, args.mock)
        time.sleep(settings.video_worker_poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
