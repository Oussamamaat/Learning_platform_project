import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { CircleAlert, FileText, Languages, User } from "lucide-react";
import type { ChatMessage } from "../../hooks/useChatSessions";
import { useApp } from "../../context/AppContext";
import QuizCard from "./QuizCard";

// Gemini-style 4-point sparkle, drawn as flat geometry (no gradients).
export function SparkleMark({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" className={className} aria-hidden="true" fill="currentColor">
      <path d="M12 2.5c.55 4.4 3.55 7.4 8 8-4.45.6-7.45 3.6-8 8-.55-4.4-3.55-7.4-8-8 4.45-.6 7.45-3.6 8-8Z" />
      <path d="M19.5 14.5c.22 1.76 1.42 2.96 3.2 3.2-1.78.24-2.98 1.44-3.2 3.2-.22-1.76-1.42-2.96-3.2-3.2 1.78-.24 2.98-1.44 3.2-3.2Z" opacity="0.7" />
      <path d="M4.5 14.5c.22 1.76 1.42 2.96 3.2 3.2-1.78.24-2.98 1.44-3.2 3.2-.22-1.76-1.42-2.96-3.2-3.2 1.78-.24 2.98-1.44 3.2-3.2Z" opacity="0.45" />
    </svg>
  );
}

function TypingDots() {
  return (
    <div className="flex items-center gap-1 py-1" aria-label="Thinking">
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className="h-1.5 w-1.5 rounded-full bg-ink-faint animate-typing"
          style={{ animationDelay: `${i * 0.18}s` }}
        />
      ))}
    </div>
  );
}

function CrossLanguageNote() {
  return (
    <div
      className="mt-2.5 flex items-center gap-1.5 rounded-lg border border-edge bg-surface-2 px-2.5 py-1.5 text-[11px] text-ink-dim"
      title="Not enough source material in the requested language was found, so this answer draws on sources written in a different script."
    >
      <Languages className="h-3 w-3 shrink-0 text-ink-faint" />
      <span>Grounded in a source written in a different language</span>
    </div>
  );
}

function SourcesRow({ sources }: { sources: string[] }) {
  if (sources.length === 0) return null;
  return (
    <div className="mt-2.5 flex flex-wrap gap-1.5 border-t border-edge pt-2.5">
      {sources.map((source, i) => (
        <span
          key={`${source}-${i}`}
          title={source}
          className="inline-flex max-w-full items-center gap-1.5 rounded-full border border-edge bg-surface-2 px-2.5 py-1 text-[10.5px] text-ink-dim"
        >
          <FileText className="h-3 w-3 shrink-0 text-brand" />
          <span className="max-w-52 truncate">{source}</span>
        </span>
      ))}
    </div>
  );
}

function AssistantBody({ message }: { message: ChatMessage }) {
  if (message.quiz) {
    return <QuizCard quiz={message.quiz} />;
  }

  if (message.pending) {
    return <TypingDots />;
  }

  if (message.error) {
    return (
      <div className="flex items-start gap-2 rounded-xl border border-danger/30 bg-danger-soft px-3.5 py-3">
        <CircleAlert className="mt-0.5 h-4 w-4 shrink-0 text-danger" />
        <p className="text-[13px] leading-relaxed text-danger-deep">{message.content}</p>
      </div>
    );
  }

  return <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>;
}

export default function MessageBubble({ message }: { message: ChatMessage }) {
  const { activeLanguage } = useApp();

  if (message.role === "user") {
    return (
      <div className="flex justify-end animate-fade-in">
        <div className="flex max-w-[82%] items-end gap-2.5">
          <div className="rounded-2xl rounded-br-md bg-brand-soft px-4 py-2.5 text-[13.5px] leading-relaxed text-ink">
            {message.content}
          </div>
          <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full border border-edge bg-surface-2">
            <User className="h-3.5 w-3.5 text-ink-faint" />
          </div>
        </div>
      </div>
    );
  }

  const dir = activeLanguage === "ar-MA" ? "rtl" : "ltr";

  return (
    <div className="flex justify-start animate-fade-in">
      <div className="flex max-w-[88%] items-start gap-2.5">
        <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full border border-edge bg-surface text-brand">
          <SparkleMark className="h-4 w-4" />
        </div>
        <div className="min-w-0 rounded-2xl rounded-tl-md border border-edge bg-surface px-4 py-3">
          <div dir={dir} className="min-w-0">
            <AssistantBody message={message} />
          </div>
          {message.sources && message.sources.length > 0 && (
            <SourcesRow sources={message.sources} />
          )}
          {message.crossLanguage && <CrossLanguageNote />}
        </div>
      </div>
    </div>
  );
}
