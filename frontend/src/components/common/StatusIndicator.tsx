import { useApp } from "../../context/AppContext";

const DOT_COLOR: Record<string, string> = {
  online: "bg-success",
  offline: "bg-danger",
  checking: "bg-warn animate-pulse",
};

const LABEL: Record<string, string> = {
  online: "Pipeline online",
  offline: "Pipeline offline",
  checking: "Checking…",
};

export default function StatusIndicator({ compact = false }: { compact?: boolean }) {
  const { health, modelName } = useApp();

  return (
    <div
      className="flex min-w-0 items-center gap-2 rounded-full border border-edge bg-surface px-3 py-1.5"
      title={health.lastChecked ? `Last checked ${new Date(health.lastChecked).toLocaleTimeString()}` : "Not checked yet"}
    >
      <span className={`h-2 w-2 shrink-0 rounded-full ${DOT_COLOR[health.status]}`} />
      <span className="min-w-0 truncate font-mono text-[11px] font-medium text-ink-dim">
        {modelName}
      </span>
      {!compact && (
        <span className="shrink-0 text-[11px] text-ink-faint">· {LABEL[health.status]}</span>
      )}
    </div>
  );
}
