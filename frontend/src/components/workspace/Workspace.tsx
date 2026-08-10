import TopBar from "./TopBar";
import ChatStream from "./ChatStream";
import InputArea from "./InputArea";

export default function Workspace() {
  return (
    <main className="flex h-full min-w-0 flex-1 flex-col">
      <TopBar />
      <ChatStream />
      <InputArea />
    </main>
  );
}
