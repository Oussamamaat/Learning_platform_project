import { FolderCog } from "lucide-react";
import { useApp } from "../../context/AppContext";
import type { Domain } from "../../types/api";

const OPTIONS: { value: Domain; label: string }[] = [
  { value: "industrial", label: "Industrial" },
  { value: "securite", label: "Sécurité" },
  { value: "blockchain", label: "Blockchain" },
];

export default function DomainSelector() {
  const { activeDomain, setActiveDomain } = useApp();

  return (
    <label className="flex items-center gap-1.5 rounded-full border border-edge bg-surface py-1 pl-2.5 pr-1 transition-colors hover:border-ink-faint/50">
      <FolderCog className="h-3.5 w-3.5 text-ink-faint" />
      <span className="sr-only">Tenant domain</span>
      <select
        value={activeDomain}
        onChange={(e) => setActiveDomain(e.target.value as Domain)}
        className="cursor-pointer appearance-none bg-transparent pr-3 text-[12.5px] font-semibold text-ink outline-none"
      >
        {OPTIONS.map((opt) => (
          <option key={opt.value} value={opt.value} className="bg-surface text-ink">
            {opt.label}
          </option>
        ))}
      </select>
    </label>
  );
}
