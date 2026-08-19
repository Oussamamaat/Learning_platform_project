import { FileQuestion, Video } from "lucide-react";
import { useApp } from "../../context/AppContext";

export default function QuickActions() {
  const { setQuizModalOpen, toastInfo } = useApp();

  return (
    <div className="px-4 pt-3">
      <p className="px-1 pb-2 text-[10.5px] font-semibold uppercase tracking-[0.14em] text-ink-faint">
        Quick actions
      </p>
      <div className="grid grid-cols-2 gap-2">
        <button
          type="button"
          onClick={() => setQuizModalOpen(true)}
          className="press group flex flex-col items-start gap-1.5 rounded-xl border border-edge bg-surface px-3 py-2.5 text-left hover:border-brand hover:bg-brand-soft"
        >
          <FileQuestion className="h-4 w-4 text-brand transition-transform group-hover:scale-110" />
          <span className="text-[12.5px] font-semibold text-ink">Generate Quiz</span>
          <span className="text-[10.5px] leading-snug text-ink-faint">
            Grounded questions from tenant docs
          </span>
        </button>
        <button
          type="button"
          onClick={() => toastInfo("Video generation is coming soon.")}
          className="press group flex flex-col items-start gap-1.5 rounded-xl border border-edge bg-surface px-3 py-2.5 text-left hover:bg-surface-3"
        >
          <Video className="h-4 w-4 text-ink-faint transition-transform group-hover:scale-110" />
          <span className="text-[12.5px] font-semibold text-ink-dim">Generate Video</span>
          <span className="text-[10.5px] leading-snug text-ink-faint">Coming soon</span>
        </button>
      </div>
    </div>
  );
}
