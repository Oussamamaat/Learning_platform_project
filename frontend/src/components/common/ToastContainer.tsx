import { CheckCircle2, CircleAlert, Info, X } from "lucide-react";
import { useToast } from "../../hooks/useToast";
import type { ToastKind } from "../../hooks/useToast";

const ICONS: Record<ToastKind, typeof Info> = {
  info: Info,
  success: CheckCircle2,
  error: CircleAlert,
};

const STYLES: Record<ToastKind, string> = {
  info: "text-brand",
  success: "text-success",
  error: "text-danger",
};

export default function ToastContainer() {
  const { toasts, dismiss } = useToast();

  if (toasts.length === 0) return null;

  return (
    <div className="pointer-events-none fixed left-1/2 top-4 z-50 flex w-full max-w-md -translate-x-1/2 flex-col items-center gap-2 px-4">
      {toasts.map((toast) => {
        const Icon = ICONS[toast.kind];
        return (
          <div
            key={toast.id}
            role="status"
            className={`glass-panel pointer-events-auto flex w-full items-start gap-2.5 rounded-xl px-3.5 py-2.5 shadow-[0_1px_3px_rgba(60,64,67,0.2),0_4px_10px_rgba(60,64,67,0.12)] animate-materialize ${STYLES[toast.kind]}`}
          >
            <Icon className="mt-0.5 h-4 w-4 shrink-0" />
            <p className="min-w-0 flex-1 text-[13px] leading-snug text-ink">{toast.message}</p>
            <button
              type="button"
              onClick={() => dismiss(toast.id)}
              className="press-icon shrink-0 rounded-md p-0.5 text-ink-faint hover:bg-surface-3 hover:text-ink"
              aria-label="Dismiss notification"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          </div>
        );
      })}
    </div>
  );
}
