import { Plus } from "lucide-react";
import { useApp } from "../../context/AppContext";

export default function NewChatButton() {
  const { newSession, setActiveDomain } = useApp();

  const handleClick = () => {
    setActiveDomain("industrial");
    newSession();
  };

  return (
    <button
      type="button"
      onClick={handleClick}
      className="press flex w-full items-center gap-2.5 rounded-xl bg-brand px-3.5 py-2.5 text-[13.5px] font-semibold text-white hover:bg-brand-dark"
    >
      <Plus className="h-4 w-4" />
      New chat
    </button>
  );
}
