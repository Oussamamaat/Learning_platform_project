import Sidebar from "./components/layout/Sidebar";
import Workspace from "./components/workspace/Workspace";
import QuizModal from "./components/workspace/QuizModal";
import ToastContainer from "./components/common/ToastContainer";

function App() {
  return (
    <div className="flex h-full overflow-hidden bg-surface text-ink">
      <Sidebar />
      <Workspace />
      <QuizModal />
      <ToastContainer />
    </div>
  );
}

export default App;
