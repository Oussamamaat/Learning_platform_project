import { Languages } from "lucide-react";
import { useApp } from "../../context/AppContext";
import type { Language } from "../../types/api";

const OPTIONS: { value: Language; label: string; sub: string }[] = [
  { value: "fr", label: "Français", sub: "FR" },
  { value: "ar-MA", label: "Darija", sub: "AR-MA" },
];

export default function LanguageSelector() {
  const { activeLanguage, setActiveLanguage } = useApp();

  return (
    <label className="flex items-center gap-1.5 rounded-full border border-edge bg-surface py-1 pl-2.5 pr-1 transition-colors hover:border-ink-faint/50">
      <Languages className="h-3.5 w-3.5 text-ink-faint" />
      <span className="sr-only">Response language</span>
      <select
        value={activeLanguage}
        onChange={(e) => setActiveLanguage(e.target.value as Language)}
        className="cursor-pointer appearance-none bg-transparent pr-3 text-[12.5px] font-semibold text-ink outline-none"
      >
        {OPTIONS.map((opt) => (
          <option key={opt.value} value={opt.value} className="bg-surface text-ink">
            {opt.label} ({opt.sub})
          </option>
        ))}
      </select>
    </label>
  );
}
