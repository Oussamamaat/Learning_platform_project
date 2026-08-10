import { useEffect, useRef } from "react";
import { useApp } from "../../context/AppContext";
import MessageBubble, { SparkleMark } from "./MessageBubble";

function EmptyState() {
  const { activeDomain, activeLanguage } = useApp();
  return (
    <div className="flex h-full flex-col items-center justify-center px-6 text-center">
      <SparkleMark className="mb-5 h-8 w-8 text-brand" />
      <h2 className="text-xl font-semibold tracking-tight text-ink">Ask the Atlas Tutor</h2>
      <p className="mt-2 max-w-md text-[13.5px] leading-relaxed text-ink-dim">
        Watch the pipeline live: RAG retrieval against tenant docs, the fine-tuned
        model, citation injection, and quiz grounding — click by click.
      </p>
      <div className="mt-5 flex flex-wrap items-center justify-center gap-2 text-[11.5px]">
        <span className="rounded-full border border-edge bg-surface px-3 py-1 text-ink-faint">
          Domain: <span className="font-semibold text-ink">{activeDomain}</span>
        </span>
        <span className="rounded-full border border-edge bg-surface px-3 py-1 text-ink-faint">
          Language: <span className="font-semibold text-ink">{activeLanguage}</span>
        </span>
        <span className="rounded-full border border-edge bg-surface px-3 py-1 text-ink-faint">
          Model: <span className="font-mono font-medium text-ink">IBLOG_TUTOR:latest</span>
        </span>
      </div>
    </div>
  );
}

export default function ChatStream() {
  const { activeSession } = useApp();
  const bottomRef = useRef<HTMLDivElement>(null);
  const messages = activeSession?.messages ?? [];
  const lastPending = messages.at(-1)?.pending ?? false;

  useEffect(() => {
    const el = bottomRef.current;
    if (el) {
      el.scrollIntoView({ behavior: "smooth", block: "end" });
    }
  }, [messages.length, lastPending]);

  if (messages.length === 0) {
    return (
      <div className="chat-scroll flex-1 overflow-y-auto">
        <EmptyState />
      </div>
    );
  }

  return (
    <div className="chat-scroll flex-1 overflow-y-auto">
      <div className="mx-auto flex w-full max-w-3xl flex-col gap-5 px-4 py-6 sm:px-6">
        {messages.map((message) => (
          <MessageBubble key={message.id} message={message} />
        ))}
        <div ref={bottomRef} className="h-px" />
      </div>
    </div>
  );
}
