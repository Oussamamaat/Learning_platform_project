import { Bot } from "lucide-react";
import { useApp } from "../../context/AppContext";
import LanguageSelector from "./LanguageSelector";
import DomainSelector from "./DomainSelector";

export default function TopBar() {
  const { activeSession } = useApp();

  return (
    <header className="flex shrink-0 items-center justify-between gap-4 border-b border-edge bg-surface px-5 py-3">
      <div className="min-w-0">
        <h2 className="truncate text-[14.5px] font-semibold text-ink">
          {activeSession ? activeSession.title : "New chat"}
        </h2>
        <p className="hidden text-[11px] text-ink-faint sm:block">
          RAG retrieval → LLM → citation injection / quiz grounding
        </p>
      </div>
      <div className="flex shrink-0 items-center gap-2">
        <div className="hidden items-center gap-1.5 rounded-full border border-edge bg-surface px-3 py-1.5 md:flex">
          <Bot className="h-3.5 w-3.5 text-brand" />
          <span className="font-mono text-[11px] font-medium text-ink-dim">IBLOG_TUTOR:latest</span>
        </div>
        <LanguageSelector />
        <DomainSelector />
      </div>
    </header>
  );
}
