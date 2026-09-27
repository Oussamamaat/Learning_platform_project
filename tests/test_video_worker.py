"""
Tests for scripts/video/worker.py -- the process that claims pending
video_jobs rows and drives the partner pipeline.

The partner modules (planner, prompt_builder, frame_extractor,
video_editor, wan_client, job_adapter) live in a separate clone pointed at
by settings.video_pipeline_dir and are NOT importable here, so they are
stubbed. That is the point of these tests: they cover the worker's own
dispatch, error reporting and audio handling, not Ollama's scene planning
or Wan2.2's rendering, neither of which belongs in a unit test.

The one thing not stubbed is the PCM->WAV wrap, which is exercised for
real -- it is the actual seam between our TTS (raw PCM, app/services/tts.py)
and ffmpeg (needs a container), and a silently wrong header there produces
a video whose audio plays at the wrong speed rather than an exception.
"""
import sys
import types
import wave
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.video import worker as video_worker  # noqa: E402


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

class FakeApi:
    """Records what the worker reported, in order."""

    def __init__(self, pending):
        self.pending = pending
        self.calls = []

    def claim_pending(self):
        return self.pending

    def mark_processing(self, job_id):
        self.calls.append(("processing", job_id, None))

    def mark_ready(self, job_id, video_url):
        self.calls.append(("ready", job_id, video_url))

    def mark_error(self, job_id, message):
        self.calls.append(("error", job_id, message))

    def statuses(self):
        return [c[0] for c in self.calls]


class FakeEngine:
    name = "fake_xtts"
    sample_rate = 24000

    def __init__(self):
        self.spoken = []

    def synthesize(self, text, *, language):
        self.spoken.append((text, language))
        # 0.1s of silence: 16-bit mono at this rate.
        return b"\x00\x00" * (self.sample_rate // 10)


def _job(**overrides):
    job = {
        "id": "11111111-2222-3333-4444-555555555555",
        "tenant_id": "company_abc",
        "session_id": None,
        "input_text": "Le port du gilet réfléchissant est obligatoire pendant la ronde.",
        "title": "Ronde de nuit",
        "language": "fr",
        "mode": "scene",
        "status": "pending",
        "video_url": None,
        "error_message": None,
        "created_at": "2026-09-21T09:00:00Z",
    }
    job.update(overrides)
    return job


def _scene(scene_id=1):
    return {
        "scene_id": scene_id,
        "duration": 5,
        "narration": f"Narration de la scène {scene_id}.",
        "objective": "Montrer l'équipement obligatoire",
    }


def _fake_modules(scenes=None, tmp_path=None):
    """Stand-ins for the partner clone's modules, recording their calls."""
    scenes = scenes if scenes is not None else [_scene(1), _scene(2)]
    calls = {"rendered": [], "frames": [], "assembled": []}

    job_adapter = types.SimpleNamespace(
        adapt_avatar_job=lambda job: {
            "mode": "avatar",
            "title": job.get("title", ""),
            "content": job["input_text"],
            "language": job["language"],
            "tts_language": job["language"],
            "duration": 30,
        }
    )
    planner = types.SimpleNamespace(
        MODEL="stub",
        generate_scenes=lambda adapted: {"scenes": [dict(s) for s in scenes]},
        validate_scenes=lambda plan, expected_duration: plan,
    )
    prompt_builder = types.SimpleNamespace(
        MODEL="stub",
        build_video_prompt=lambda scene: f"initial prompt {scene['scene_id']}",
        build_continuation_prompt=lambda scene: f"continuation prompt {scene['scene_id']}",
    )

    def _render(prompt, reference_image=None, duration=5, output_path="out.mp4"):
        calls["rendered"].append((prompt, reference_image, duration))
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"fake clip")
        return output_path

    def _extract(clip_path, output_path=None):
        calls["frames"].append(clip_path)
        Path(output_path).write_bytes(b"fake frame")
        return output_path

    def _assemble(clip_paths, audio_paths, output_path):
        calls["assembled"].append((list(clip_paths), list(audio_paths)))
        Path(output_path).write_bytes(b"fake video")
        return output_path

    return {
        "job_adapter": job_adapter,
        "planner": planner,
        "prompt_builder": prompt_builder,
        "frame_extractor": types.SimpleNamespace(extract_last_frame=_extract),
        "video_editor": types.SimpleNamespace(assemble_video=_assemble),
        "wan_client": types.SimpleNamespace(generate_video_mock=_render, generate_video=_render),
    }, calls


@pytest.fixture
def worker_settings(monkeypatch, tmp_path):
    """A real Settings instance with video output pointed at tmp_path --
    mutating the instance, not the pydantic class (see
    tests/test_ingest_router.py's fixture for why the class-level
    monkeypatch is silently ignored)."""
    from app.config import get_settings

    settings = get_settings().model_copy()
    settings.video_output_dir = str(tmp_path / "videos")
    settings.comfyui_url = ""
    monkeypatch.setattr(video_worker, "get_settings", lambda: settings)
    return settings


@pytest.fixture
def fake_tts(monkeypatch):
    engine = FakeEngine()
    fake_module = types.ModuleType("app.services.tts")
    fake_module.get_tts_engine = lambda: engine
    monkeypatch.setitem(sys.modules, "app.services.tts", fake_module)
    return engine


# ---------------------------------------------------------------------------
# Avatar mode: refused, with a reason
# ---------------------------------------------------------------------------

def test_avatar_job_is_refused_not_silently_rendered(worker_settings, fake_tts):
    """Avatar mode has no implementation upstream (avatar_generator.py is an
    empty file). It must fail loudly rather than quietly produce scene
    footage under an avatar label."""
    modules, calls = _fake_modules()
    api = FakeApi([_job(mode="avatar")])

    video_worker.drain_once(api, modules)

    assert api.statuses() == ["processing", "error"]
    message = api.calls[-1][2]
    assert "avatar" in message.lower()
    assert "not implemented" in message.lower()
    # And nothing was rendered on the way to that refusal.
    assert calls["rendered"] == []
    assert fake_tts.spoken == []


# ---------------------------------------------------------------------------
# Scene mode: the happy path
# ---------------------------------------------------------------------------

def test_scene_job_runs_all_three_phases(worker_settings, fake_tts):
    modules, calls = _fake_modules()
    api = FakeApi([_job()])

    video_worker.drain_once(api, modules)

    assert api.statuses() == ["processing", "ready"]
    video_url = api.calls[-1][2]
    assert video_url == "/media/11111111-2222-3333-4444-555555555555/video_finale.mp4"

    # Phase 2 spoke every scene's narration, in our TTS's vocabulary.
    assert [t for t, _ in fake_tts.spoken] == [
        "Narration de la scène 1.",
        "Narration de la scène 2.",
    ]
    assert {lang for _, lang in fake_tts.spoken} == {"fr"}

    # Phase 3 rendered both scenes, and seeded the second with the first's
    # last frame -- that continuity is the whole reason frames are extracted.
    assert len(calls["rendered"]) == 2
    assert calls["rendered"][0][1] is None
    assert calls["rendered"][1][1] is not None
    assert calls["rendered"][0][0].startswith("initial prompt")
    assert calls["rendered"][1][0].startswith("continuation prompt")

    clips, audios = calls["assembled"][0]
    assert len(clips) == len(audios) == 2


def test_darija_job_uses_the_darija_voice(worker_settings, fake_tts):
    """video_jobs speaks 'ar-MA'; app.services.tts speaks 'darija'. A
    mismatch here is silent -- the video would just narrate in the wrong
    language."""
    modules, _ = _fake_modules()
    video_worker.drain_once(FakeApi([_job(language="ar-MA")]), modules)

    assert {lang for _, lang in fake_tts.spoken} == {"darija"}


def test_checkpoint_written_next_to_the_video(worker_settings, fake_tts):
    """The scene plan and prompts that produced a video, kept beside it --
    the difference between 'the video looks wrong' and knowing which scene
    prompt caused it."""
    modules, _ = _fake_modules()
    video_worker.drain_once(FakeApi([_job()]), modules)

    checkpoint = (
        Path(worker_settings.video_output_dir)
        / "11111111-2222-3333-4444-555555555555"
        / "pipeline.json"
    )
    assert checkpoint.exists()
    assert "initial prompt 1" in checkpoint.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Failure reporting -- a job must never be left at 'processing'
# ---------------------------------------------------------------------------

def test_phase_one_failure_is_reported_as_error(worker_settings, fake_tts):
    modules, _ = _fake_modules()

    def _boom(adapted):
        raise RuntimeError("ollama refused to answer")

    modules["planner"].generate_scenes = _boom
    api = FakeApi([_job()])

    video_worker.drain_once(api, modules)

    assert api.statuses() == ["processing", "error"]
    assert "ollama refused to answer" in api.calls[-1][2]


def test_narration_failure_is_reported_as_error(worker_settings, fake_tts):
    """A scene with no narration text would otherwise reach TTS as an empty
    string and produce a silent clip."""
    modules, _ = _fake_modules(scenes=[{"scene_id": 1, "duration": 5, "narration": "  "}])
    api = FakeApi([_job()])

    video_worker.drain_once(api, modules)

    assert api.statuses() == ["processing", "error"]
    assert "narration" in api.calls[-1][2].lower()


def test_one_failing_job_does_not_stop_the_next(worker_settings, fake_tts):
    modules, _ = _fake_modules()
    api = FakeApi([_job(id="bad", mode="avatar"), _job(id="good")])

    video_worker.drain_once(api, modules)

    assert api.statuses() == ["processing", "error", "processing", "ready"]


# ---------------------------------------------------------------------------
# PCM -> WAV, exercised for real
# ---------------------------------------------------------------------------

def test_pcm_is_wrapped_with_a_correct_wav_header(tmp_path):
    pcm = b"\x01\x02" * 1200
    out = video_worker.write_pcm_as_wav(pcm, 24000, str(tmp_path / "a" / "scene.wav"))

    with wave.open(out, "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 24000
        assert wav.getnframes() == 1200
        assert wav.readframes(1200) == pcm


# ---------------------------------------------------------------------------
# Renderer selection
# ---------------------------------------------------------------------------

def test_mock_renderer_used_when_no_comfyui_configured(worker_settings):
    modules, _ = _fake_modules()
    worker_settings.comfyui_url = ""
    assert video_worker.resolve_renderer(modules) is modules["wan_client"].generate_video_mock


def test_mock_flag_wins_over_a_configured_comfyui(worker_settings):
    modules, _ = _fake_modules()
    worker_settings.comfyui_url = "http://example.invalid:8188"
    assert (
        video_worker.resolve_renderer(modules, force_mock=True)
        is modules["wan_client"].generate_video_mock
    )
