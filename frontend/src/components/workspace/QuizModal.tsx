import { useState } from "react";
import { FileQuestion, Loader2, Minus, Plus, X } from "lucide-react";
import { useApp } from "../../context/AppContext";

// Language is auto-detected server-side from `topic` (app.services.routing,
// same script-detection + instruction-override logic chat uses) -- no
// selector here since 2026-08-12.
export default function QuizModal() {
  const { quizModalOpen, setQuizModalOpen, generateQuizInSession } = useApp();
  const [topic, setTopic] = useState("");
  const [numQuestions, setNumQuestions] = useState(5);
  const [loading, setLoading] = useState(false);

  if (!quizModalOpen) return null;

  const canSubmit = topic.trim().length > 0 && numQuestions >= 1 && numQuestions <= 20 && !loading;

  const handleSubmit = async () => {
    if (!canSubmit) return;
    setLoading(true);
    try {
      await generateQuizInSession({ topic: topic.trim(), numQuestions });
      setTopic("");
      setNumQuestions(5);
      setQuizModalOpen(false);
    } finally {
      setLoading(false);
    }
  };

  const step = (delta: number) => {
    setNumQuestions((prev) => Math.min(20, Math.max(1, prev + delta)));
  };

  return (
    <div
      className="fixed inset-0 z-40 flex items-center justify-center bg-black/35 p-4 backdrop-blur-sm animate-fade-in"
      onClick={() => !loading && setQuizModalOpen(false)}
      role="dialog"
      aria-modal="true"
      aria-label="Generate quiz"
    >
      <div
        className="glass-panel w-full max-w-md rounded-2xl p-5 shadow-[0_2px_6px_rgba(60,64,67,0.18),0_10px_28px_rgba(60,64,67,0.16)] animate-materialize"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-4 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-brand-soft">
              <FileQuestion className="h-4.5 w-4.5 text-brand-deep" />
            </div>
            <div>
              <h2 className="text-[15px] font-semibold text-ink">Generate Quiz</h2>
              <p className="text-[11px] text-ink-faint">
                Grounded in retrieved tenant documents
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={() => !loading && setQuizModalOpen(false)}
            className="press-icon rounded-lg p-1.5 text-ink-faint hover:bg-surface-3 hover:text-ink"
            aria-label="Close quiz dialog"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <label className="block">
          <span className="mb-1.5 block text-[11px] font-semibold uppercase tracking-wide text-ink-faint">
            Topic
          </span>
          <textarea
            value={topic}
            onChange={(e) => setTopic(e.target.value)}
            placeholder="e.g. LOTO lockout/tagout procedure, Article 16 of the Labour Code… (language detected automatically)"
            rows={3}
            className="w-full resize-none rounded-xl border border-edge bg-surface px-3.5 py-2.5 text-[13.5px] text-ink placeholder:text-ink-faint outline-none transition-colors duration-150 ease-spring focus:border-brand"
          />
        </label>

        <div className="mt-4">
          <span className="mb-1.5 block text-[11px] font-semibold uppercase tracking-wide text-ink-faint">
            Number of questions
          </span>
          <div className="flex items-center gap-2.5">
            <button
              type="button"
              onClick={() => step(-1)}
              disabled={numQuestions <= 1}
              className="press-icon flex h-7 w-7 items-center justify-center rounded-lg border border-edge bg-surface text-ink-dim hover:border-brand hover:text-ink disabled:opacity-40"
              aria-label="Fewer questions"
            >
              <Minus className="h-3.5 w-3.5" />
            </button>
            <span className="w-8 text-center font-mono text-[14px] font-semibold text-ink">
              {numQuestions}
            </span>
            <button
              type="button"
              onClick={() => step(1)}
              disabled={numQuestions >= 20}
              className="press-icon flex h-7 w-7 items-center justify-center rounded-lg border border-edge bg-surface text-ink-dim hover:border-brand hover:text-ink disabled:opacity-40"
              aria-label="More questions"
            >
              <Plus className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>

        <button
          type="button"
          onClick={() => void handleSubmit()}
          disabled={!canSubmit}
          className="press mt-5 flex w-full items-center justify-center gap-2 rounded-xl bg-brand px-4 py-2.5 text-[13.5px] font-semibold text-white hover:bg-brand-dark disabled:cursor-not-allowed disabled:bg-brand/40"
        >
          {loading ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin" />
              Generating…
            </>
          ) : (
            "Generate quiz"
          )}
        </button>
      </div>
    </div>
  );
}
