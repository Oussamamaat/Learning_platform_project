import Sidebar from "./components/layout/Sidebar";
import Workspace from "./components/workspace/Workspace";
import QuizModal from "./components/workspace/QuizModal";
import ToastContainer from "./components/common/ToastContainer";
import SourcesPanel from "./components/sources/SourcesPanel";
import { useApp } from "./context/AppContext";

function App() {
  const { viewMode, quizModalOpen } = useApp();
  return (
    <div className="flex h-full overflow-hidden bg-surface text-ink">
      {/* Dim-to-focus: pushes the background back/down instead of just
          dimming it, so the quiz dialog reads as a task pulled in front of
          the workspace rather than a layer stacked on top. Purely
          declarative on the modal's existing open state -- no new
          behavior. apple-design skill §12. */}
      <div
        className={`flex h-full min-w-0 flex-1 transition-all duration-300 ease-spring ${
          quizModalOpen ? "scale-[0.985] opacity-90 blur-[1px]" : ""
        }`}
      >
        <Sidebar />
        <Workspace />
        {viewMode === "tenant" && <SourcesPanel />}
      </div>
      <QuizModal />
      <ToastContainer />
    </div>
  );
}

export default App;
