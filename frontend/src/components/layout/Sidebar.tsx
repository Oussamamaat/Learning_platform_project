import SidebarHeader from "./SidebarHeader";
import NewChatButton from "./NewChatButton";
import QuickActions from "./QuickActions";
import ChatHistoryList from "./ChatHistoryList";
import UserFooter from "./UserFooter";

export default function Sidebar() {
  return (
    <aside className="flex h-full w-72 shrink-0 flex-col border-r border-edge bg-surface-2">
      <SidebarHeader />
      <div className="px-4">
        <NewChatButton />
        <QuickActions />
      </div>
      <ChatHistoryList />
      <UserFooter />
    </aside>
  );
}
