import { useState } from "react";
import {
  AlertCircle,
  CheckCircle2,
  Clapperboard,
  Film,
  Loader2,
  UserSquare2,
  Video,
} from "lucide-react";
import { useApp } from "../../context/AppContext";
import { API_BASE } from "../../services/api";
import type { Language, VideoJob, VideoMode } from "../../types/api";

// Course-authoring surface: Admins and Tenants turn a written explanation
// into a narrated video for a course. Never rendered for employees
// (Workspace gates it on canAuthorCourses) and refused server-side for
// them too (app/routers/video.py's require_role).
//
// Avatar mode is OFFERED BUT DISABLED, with the reason on screen. Hiding
// it would read as a feature nobody thought of; showing it disabled says
// "this exists and is coming", which is the true state -- the pipeline's
// avatar_generator.py is still an empty stub upstream.

const LANGUAGES: { value: Language; label: string }[] = [
  { value: "fr", label: "Français" },
  { value: "ar-MA", label: "Darija" },
];

const MAX_TEXT = 8000; // app.models.schemas.VideoGenerateRequest.text

function StatusChip({ status }: { status: VideoJob["status"] }) {
  const map = {
    pending: { Icon: Loader2, spin: true, label: "Queued", cls: "text-ink-faint" },
    processing: { Icon: Loader2, spin: true, label: "Rendering", cls: "text-brand" },
    ready: { Icon: CheckCircle2, spin: false, label: "Ready", cls: "text-emerald-600" },
    error: { Icon: AlertCircle, spin: false, label: "Failed", cls: "text-red-600" },
  }[status];
  return (
    <span className={`flex items-center gap-1.5 text-[11.5px] font-semibold ${map.cls}`}>
      <map.Icon className={`h-3.5 w-3.5 ${map.spin ? "animate-spin" : ""}`} />
      {map.label}
    </span>
  );
}

function JobCard({ job }: { job: VideoJob }) {
  return (
    <li className="rounded-xl border border-edge bg-surface p-3.5">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="truncate text-[13px] font-semibold text-ink">
            {job.title?.trim() || "Untitled video"}
          </p>
          <p className="mt-0.5 flex items-center gap-1.5 text-[11px] text-ink-faint">
            <span>{job.language === "ar-MA" ? "Darija" : "Français"}</span>
            <span aria-hidden>·</span>
            <span className="capitalize">{job.mode}</span>
            <span aria-hidden>·</span>
            <span>{new Date(job.created_at).toLocaleString()}</span>
          </p>
        </div>
        <StatusChip status={job.status} />
      </div>

      {job.status === "ready" && job.video_url && (
        <video
          // video_url is relative to the API base (/media/... , served by
          // app/main.py's StaticFiles mount), not to the Vite dev server.
          src={`${API_BASE}${job.video_url}`}
          controls
          preload="metadata"
          className="mt-3 w-full rounded-lg border border-edge bg-black"
        />
      )}

      {job.status === "error" && job.error_message && (
        <p className="mt-2.5 rounded-lg bg-red-50 px-3 py-2 text-[11.5px] leading-relaxed text-red-700">
          {job.error_message}
        </p>
      )}

      {(job.status === "pending" || job.status === "processing") && (
        <p className="mt-2 text-[11px] leading-relaxed text-ink-faint">
          Scene planning, narration and rendering run in sequence — this takes minutes.
        </p>
      )}
    </li>
  );
}

export default function VideoStudio() {
  const { videoJobs, submitVideoJob, videoSubmitting } = useApp();
  const [title, setTitle] = useState("");
  const [text, setText] = useState("");
  const [language, setLanguage] = useState<Language>("fr");
  const [mode, setMode] = useState<VideoMode>("scene");

  const canSubmit = text.trim().length > 0 && text.length <= MAX_TEXT && !videoSubmitting;

  const handleSubmit = async () => {
    if (!canSubmit) return;
    await submitVideoJob({
      text: text.trim(),
      title: title.trim() || undefined,
      language,
      mode,
    });
    setText("");
    setTitle("");
  };

  return (
    <div className="flex h-full min-h-0 flex-1 overflow-y-auto">
      <div className="mx-auto w-full max-w-3xl px-6 py-6">
        <header className="mb-5 flex items-center gap-2.5">
          <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-brand-soft">
            <Clapperboard className="h-4.5 w-4.5 text-brand-deep" />
          </div>
          <div>
            <h2 className="text-[15px] font-semibold text-ink">Video Studio</h2>
            <p className="text-[11px] text-ink-faint">
              Turn a written explanation into a narrated course video
            </p>
          </div>
        </header>

        <section className="rounded-2xl border border-edge bg-surface-2 p-5">
          <label className="block">
            <span className="mb-1.5 block text-[11px] font-semibold uppercase tracking-wide text-ink-faint">
              Title <span className="font-normal normal-case tracking-normal">(optional)</span>
            </span>
            <input
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              maxLength={300}
              placeholder="Ronde de nuit : équipement obligatoire"
              className="w-full rounded-xl border border-edge bg-surface px-3.5 py-2.5 text-[13.5px] text-ink placeholder:text-ink-faint outline-none transition-colors duration-150 ease-spring focus:border-brand"
            />
          </label>

          <label className="mt-4 block">
            <span className="mb-1.5 block text-[11px] font-semibold uppercase tracking-wide text-ink-faint">
              Explanation
            </span>
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              rows={7}
              placeholder="Paste the finished explanation the video should narrate — not a question. The pipeline splits it into scenes and narrates each one."
              className="w-full resize-none rounded-xl border border-edge bg-surface px-3.5 py-2.5 text-[13.5px] leading-relaxed text-ink placeholder:text-ink-faint outline-none transition-colors duration-150 ease-spring focus:border-brand"
            />
            <span
              className={`mt-1 block text-right text-[10.5px] ${
                text.length > MAX_TEXT ? "font-semibold text-red-600" : "text-ink-faint"
              }`}
            >
              {text.length.toLocaleString()} / {MAX_TEXT.toLocaleString()}
            </span>
          </label>

          <div className="mt-3 grid gap-4 sm:grid-cols-2">
            <div>
              <span className="mb-1.5 block text-[11px] font-semibold uppercase tracking-wide text-ink-faint">
                Narration language
              </span>
              <div className="flex gap-2">
                {LANGUAGES.map((l) => (
                  <button
                    key={l.value}
                    type="button"
                    onClick={() => setLanguage(l.value)}
                    aria-pressed={language === l.value}
                    className={`press flex-1 rounded-xl border px-3 py-2 text-[12.5px] font-semibold ${
                      language === l.value
                        ? "border-brand bg-brand-soft text-brand-deep"
                        : "border-edge bg-surface text-ink-dim hover:border-brand"
                    }`}
                  >
                    {l.label}
                  </button>
                ))}
              </div>
            </div>

            <div>
              <span className="mb-1.5 block text-[11px] font-semibold uppercase tracking-wide text-ink-faint">
                Mode
              </span>
              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={() => setMode("scene")}
                  aria-pressed={mode === "scene"}
                  className={`press flex flex-1 items-center justify-center gap-1.5 rounded-xl border px-3 py-2 text-[12.5px] font-semibold ${
                    mode === "scene"
                      ? "border-brand bg-brand-soft text-brand-deep"
                      : "border-edge bg-surface text-ink-dim hover:border-brand"
                  }`}
                >
                  <Film className="h-3.5 w-3.5" />
                  Scenes
                </button>
                <button
                  type="button"
                  disabled
                  title="Avatar mode is not implemented in the video pipeline yet — avatar_generator.py is still an empty stub upstream."
                  className="flex flex-1 cursor-not-allowed items-center justify-center gap-1.5 rounded-xl border border-edge bg-surface-3 px-3 py-2 text-[12.5px] font-semibold text-ink-faint opacity-70"
                >
                  <UserSquare2 className="h-3.5 w-3.5" />
                  Avatar
                </button>
              </div>
              <p className="mt-1.5 text-[10.5px] leading-snug text-ink-faint">
                Avatar mode is not built yet in the generation pipeline.
              </p>
            </div>
          </div>

          <button
            type="button"
            onClick={() => void handleSubmit()}
            disabled={!canSubmit}
            className="press mt-5 flex w-full items-center justify-center gap-2 rounded-xl bg-brand px-4 py-2.5 text-[13.5px] font-semibold text-white disabled:cursor-not-allowed disabled:opacity-50"
          >
            {videoSubmitting ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Video className="h-4 w-4" />
            )}
            Generate video
          </button>
        </section>

        <section className="mt-6">
          <h3 className="px-1 pb-2 text-[10.5px] font-semibold uppercase tracking-[0.14em] text-ink-faint">
            Videos
          </h3>
          {videoJobs.length === 0 ? (
            <p className="rounded-xl border border-dashed border-edge px-4 py-8 text-center text-[12.5px] text-ink-faint">
              No videos yet. Generated videos appear here as they finish.
            </p>
          ) : (
            <ul className="space-y-2.5">
              {videoJobs.map((job) => (
                <JobCard key={job.id} job={job} />
              ))}
            </ul>
          )}
        </section>
      </div>
    </div>
  );
}
