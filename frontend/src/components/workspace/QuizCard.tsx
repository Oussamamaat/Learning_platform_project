import { useMemo, useState } from "react";
import { CheckCircle2, CircleX, HelpCircle } from "lucide-react";
import type { QuizResponse } from "../../types/api";

const LETTERS = ["A", "B", "C", "D", "E", "F"];

export default function QuizCard({ quiz }: { quiz: QuizResponse }) {
  const [answers, setAnswers] = useState<Record<number, number>>({});

  const score = useMemo(
    () => quiz.questions.filter((q, i) => answers[i] === q.correct_index).length,
    [answers, quiz.questions],
  );
  const answeredCount = Object.keys(answers).length;
  const done = answeredCount === quiz.questions.length;
  // This quiz's own resolved language (app.services.routing), not the
  // session-global activeLanguage -- more accurate now that each quiz
  // generation is auto-detected independently from its own topic text.
  const dir = quiz.language === "darija" ? "rtl" : "ltr";

  return (
    <div dir={dir} className="space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="text-[13px] font-semibold text-ink">Quiz — {quiz.topic}</p>
        <span
          className={`rounded-full border px-2.5 py-0.5 text-[10.5px] font-semibold ${
            done
              ? "border-success/30 bg-success-soft text-success-deep"
              : "border-edge bg-surface-2 text-ink-faint"
          }`}
        >
          {done ? `Score: ${score}/${quiz.questions.length}` : `Answered ${answeredCount}/${quiz.questions.length}`}
        </span>
      </div>

      {quiz.total_questions < quiz.requested_questions && (
        <p className="text-[11.5px] text-ink-faint">
          {quiz.total_questions} of {quiz.requested_questions} requested — limited source
          material for this topic.
        </p>
      )}

      {quiz.questions.map((q, qi) => {
        const chosen = answers[qi];
        const answered = chosen !== undefined;
        return (
          <div key={qi} className="rounded-xl border border-edge bg-surface-2/60 p-3.5">
            <p className="text-[13px] font-semibold leading-snug text-ink">
              <span className="mr-1.5 text-ink-faint">{qi + 1}.</span>
              {q.question}
            </p>
            <div className="mt-2.5 grid gap-1.5">
              {q.options.map((option, oi) => {
                const isCorrect = oi === q.correct_index;
                const isChosen = chosen === oi;
                let style = "border-edge bg-surface text-ink-dim hover:border-brand hover:text-ink";
                if (answered) {
                  if (isCorrect) {
                    style = "border-success/40 bg-success-soft text-success-deep";
                  } else if (isChosen) {
                    style = "border-danger/40 bg-danger-soft text-danger-deep";
                  } else {
                    style = "border-edge bg-surface text-ink-faint opacity-70";
                  }
                }
                return (
                  <button
                    key={oi}
                    type="button"
                    disabled={answered}
                    onClick={() => setAnswers((prev) => ({ ...prev, [qi]: oi }))}
                    className={`flex items-start gap-2 rounded-lg border px-3 py-2 text-left text-[12.5px] leading-snug transition-colors ${style}`}
                  >
                    <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-md border border-current/30 font-mono text-[10px] font-semibold">
                      {answered && isCorrect ? (
                        <CheckCircle2 className="h-3.5 w-3.5" />
                      ) : answered && isChosen ? (
                        <CircleX className="h-3.5 w-3.5" />
                      ) : (
                        LETTERS[oi] ?? oi + 1
                      )}
                    </span>
                    <span>{option}</span>
                  </button>
                );
              })}
            </div>
            {answered && (
              <div
                className={`mt-2.5 flex items-start gap-2 rounded-lg border px-3 py-2 text-[12px] leading-relaxed animate-fade-in ${
                  chosen === q.correct_index
                    ? "border-success/25 bg-success-soft text-success-deep"
                    : "border-warn/30 bg-warn-soft text-ink-dim"
                }`}
              >
                {chosen === q.correct_index ? (
                  <CheckCircle2 className="mt-0.5 h-3.5 w-3.5 shrink-0 text-success" />
                ) : (
                  <HelpCircle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-warn" />
                )}
                <p>
                  <span className="font-semibold">
                    {chosen === q.correct_index ? "Correct. " : "Not quite. "}
                  </span>
                  {q.explanation || "No explanation was returned for this question."}
                </p>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
