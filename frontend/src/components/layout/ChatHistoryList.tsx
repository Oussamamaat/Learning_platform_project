import { MessageSquare, Trash2 } from "lucide-react";
import { useApp } from "../../context/AppContext";
import type { ChatSession } from "../../hooks/useChatSessions";

function formatTime(ts: number): string {
  const d = new Date(ts);
  const today = new Date();
  const sameDay = d.toDateString() === today.toDateString();
  if (sameDay) {
    return d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
  }
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

export default function ChatHistoryList() {
  const { sessions, activeSessionId, selectSession, deleteSession } = useApp();

  if (sessions.length === 0) {
    return (
      <div className="px-5 py-6 text-center">
        <MessageSquare className="mx-auto mb-2 h-4 w-4 text-ink-faint" />
        <p className="text-[12px] text-ink-faint">
          No conversations yet — start a new chat to begin testing the pipeline.
        </p>
      </div>
    );
  }

  return (
    <nav aria-label="Chat history" className="mt-4 flex-1 space-y-1.5 overflow-y-auto px-3 pb-2">
      <p className="px-1 pb-1.5 text-[10.5px] font-semibold uppercase tracking-[0.14em] text-ink-faint">
        Recent chats
      </p>
      {sessions.map((session: ChatSession) => {
        const active = session.id === activeSessionId;
        return (
          <div
            key={session.id}
            className={`group flex cursor-pointer items-center gap-2 rounded-lg border px-2.5 py-2 transition-colors ${
              active
                ? "border-brand bg-brand-soft text-brand-deep"
                : "border-edge bg-surface text-ink-dim hover:border-ink-faint/60 hover:text-ink"
            }`}
            onClick={() => selectSession(session.id)}
          >
            <MessageSquare
              className={`h-3.5 w-3.5 shrink-0 ${active ? "text-brand" : "text-ink-faint"}`}
            />
            <div className="min-w-0 flex-1">
              <p className="truncate text-[12.5px] font-medium leading-snug">{session.title}</p>
              <p className={`text-[10.5px] ${active ? "text-brand" : "text-ink-faint"}`}>
                {formatTime(session.updatedAt)}
              </p>
            </div>
            <button
              type="button"
              aria-label={`Delete session ${session.title}`}
              onClick={(e) => {
                e.stopPropagation();
                deleteSession(session.id);
              }}
              className="shrink-0 rounded-md p-1 text-ink-faint opacity-0 transition-opacity hover:bg-danger-soft hover:text-danger group-hover:opacity-100"
            >
              <Trash2 className="h-3.5 w-3.5" />
            </button>
          </div>
        );
      })}
    </nav>
  );
}
