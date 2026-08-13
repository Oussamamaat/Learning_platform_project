import { Bot, Languages, Loader2 } from "lucide-react";
import { useApp } from "../../context/AppContext";

const DOMAIN_LABEL: Record<string, string> = {
  industrial: "Industrial",
  securite: "Sécurité",
  blockchain: "Blockchain",
};

// Read-only since 2026-08-11 (Automatic Domain Routing) -- domain and
// response language are resolved server-side per turn
// (app.services.routing), not picked by the user beforehand. These badges
// show what was actually decided, updated from each response.
export default function TopBar() {
  const { activeSession, activeDomain, activeLanguage, isModelSwapping, lastDomainSource } =
    useApp();

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
        <div className="flex items-center gap-1.5 rounded-full border border-edge bg-surface py-1 pl-2.5 pr-3 text-[12.5px] font-semibold text-ink">
          {isModelSwapping ? (
            <Loader2 className="h-3.5 w-3.5 animate-spin text-brand" />
          ) : (
            <Languages className="h-3.5 w-3.5 text-ink-faint" />
          )}
          {activeLanguage === "ar-MA" ? "Darija" : "Français"}
        </div>
        <div className="flex items-center gap-1.5 rounded-full border border-edge bg-surface py-1 pl-2.5 pr-3 text-[12.5px] font-semibold text-ink">
          {DOMAIN_LABEL[activeDomain] ?? activeDomain}
          {lastDomainSource === "retrieval" && (
            <span className="rounded-full bg-brand-soft px-1.5 py-0.5 text-[9.5px] font-semibold uppercase tracking-wide text-brand-deep">
              auto
            </span>
          )}
        </div>
      </div>
    </header>
  );
}
