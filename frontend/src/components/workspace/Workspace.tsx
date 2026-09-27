import { Clapperboard, MessagesSquare } from "lucide-react";
import TopBar from "./TopBar";
import ChatStream from "./ChatStream";
import InputArea from "./InputArea";
import VideoStudio from "../video/VideoStudio";
import { useApp } from "../../context/AppContext";
import type { WorkspaceTab } from "../../context/AppContext";

// The tenant/admin management surface is a tab strip rather than a second
// route: this app has no router, and video generation belongs beside the
// chat it produces content for, not in a separate destination. Employees
// never see the strip -- they get chat only, and the backend refuses
// video generation for them regardless (app/services/roles.py).
function WorkspaceTabs() {
  const { workspaceTab, setWorkspaceTab } = useApp();
  const tabs: { tab: WorkspaceTab; label: string; Icon: typeof MessagesSquare }[] = [
    { tab: "chat", label: "Chat", Icon: MessagesSquare },
    { tab: "video", label: "Video Studio", Icon: Clapperboard },
  ];
  return (
    <div className="flex shrink-0 items-center gap-1 border-b border-edge bg-surface px-5">
      {tabs.map(({ tab, label, Icon }) => (
        <button
          key={tab}
          type="button"
          onClick={() => setWorkspaceTab(tab)}
          aria-current={workspaceTab === tab ? "page" : undefined}
          className={`press flex items-center gap-1.5 border-b-2 px-3 py-2.5 text-[12.5px] font-semibold transition-colors ${
            workspaceTab === tab
              ? "border-brand text-brand-deep"
              : "border-transparent text-ink-faint hover:text-ink"
          }`}
        >
          <Icon className="h-3.5 w-3.5" />
          {label}
        </button>
      ))}
    </div>
  );
}

export default function Workspace() {
  const { canAuthorCourses, workspaceTab } = useApp();
  const showVideo = canAuthorCourses && workspaceTab === "video";

  return (
    <main className="flex h-full min-w-0 flex-1 flex-col">
      <TopBar />
      {canAuthorCourses && <WorkspaceTabs />}
      {showVideo ? (
        <VideoStudio />
      ) : (
        <>
          <ChatStream />
          <InputArea />
        </>
      )}
    </main>
  );
}
