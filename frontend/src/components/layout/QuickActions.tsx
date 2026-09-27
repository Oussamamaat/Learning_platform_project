import { FileQuestion, Video } from "lucide-react";
import { useApp } from "../../context/AppContext";

export default function QuickActions() {
  const { setQuizModalOpen, canAuthorCourses, setWorkspaceTab } = useApp();

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
        {/* Course authoring, so Admin/Tenant only -- an employee gets the
            slot rendered as unavailable rather than removed, so the two
            actions keep their grid alignment and the restriction is
            visible rather than silent. The backend refuses it for them
            regardless (app/routers/video.py's require_role). */}
        <button
          type="button"
          disabled={!canAuthorCourses}
          onClick={() => setWorkspaceTab("video")}
          title={
            canAuthorCourses
              ? undefined
              : "Video generation is part of course creation — available to Admins and Tenants."
          }
          className={
            canAuthorCourses
              ? "press group flex flex-col items-start gap-1.5 rounded-xl border border-edge bg-surface px-3 py-2.5 text-left hover:border-brand hover:bg-brand-soft"
              : "flex cursor-not-allowed flex-col items-start gap-1.5 rounded-xl border border-edge bg-surface-3 px-3 py-2.5 text-left opacity-70"
          }
        >
          <Video
            className={
              canAuthorCourses
                ? "h-4 w-4 text-brand transition-transform group-hover:scale-110"
                : "h-4 w-4 text-ink-faint"
            }
          />
          <span
            className={
              canAuthorCourses
                ? "text-[12.5px] font-semibold text-ink"
                : "text-[12.5px] font-semibold text-ink-dim"
            }
          >
            Generate Video
          </span>
          <span className="text-[10.5px] leading-snug text-ink-faint">
            {canAuthorCourses ? "Narrated video from an explanation" : "Admins and Tenants only"}
          </span>
        </button>
      </div>
    </div>
  );
}
