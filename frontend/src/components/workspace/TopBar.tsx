import { Bot, Building2, Languages, Loader2, Users } from "lucide-react";
import { useApp } from "../../context/AppContext";
import type { ViewMode } from "../../context/AppContext";

const DOMAIN_LABEL: Record<string, string> = {
  industrial: "Industrial",
  securite: "Sécurité",
  blockchain: "Blockchain",
};

// A demo affordance, not a security boundary -- there is no auth in this
// codebase (app/config.py's get_tenant_id/get_user_id both ignore client
// input by design, ADR 0001). Switches between the tenant view (Sources
// panel visible, upload/toggle/delete) and the employee view (chat only,
// read-only "grounded in N sources" indicator) purely client-side.
function ViewModeToggle() {
  const { viewMode, setViewMode } = useApp();
  const options: { mode: ViewMode; label: string; Icon: typeof Building2 }[] = [
    { mode: "tenant", label: "Tenant", Icon: Building2 },
    { mode: "employee", label: "Employee", Icon: Users },
  ];
  return (
    <div className="flex items-center gap-0.5 rounded-full border border-edge bg-surface-2 p-0.5">
      {options.map(({ mode, label, Icon }) => (
        <button
          key={mode}
          type="button"
          onClick={() => setViewMode(mode)}
          aria-pressed={viewMode === mode}
          className={`press flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[11.5px] font-semibold ${
            viewMode === mode
              ? "bg-brand text-white"
              : "text-ink-faint hover:text-ink"
          }`}
        >
          <Icon className="h-3 w-3" />
          {label}
        </button>
      ))}
    </div>
  );
}

// Read-only since 2026-08-11 (Automatic Domain Routing) -- domain and
// response language are resolved server-side per turn
// (app.services.routing), not picked by the user beforehand. These badges
// show what was actually decided, updated from each response.
export default function TopBar() {
  const {
    activeSession,
    activeDomain,
    activeLanguage,
    isModelSwapping,
    lastDomainSource,
    viewMode,
    sources,
  } = useApp();
  const readyCount = sources.filter((s) => s.status === "ready" && s.enabled).length;

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
        <ViewModeToggle />
        {viewMode === "employee" && (
          <div className="hidden items-center gap-1.5 rounded-full border border-edge bg-surface py-1 pl-2.5 pr-3 text-[12.5px] font-semibold text-ink md:flex">
            Global knowledge{readyCount > 0 ? ` + ${readyCount} custom source${readyCount === 1 ? "" : "s"}` : ""}
          </div>
        )}
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
