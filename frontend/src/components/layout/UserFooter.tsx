import { Building2 } from "lucide-react";
import { useApp } from "../../context/AppContext";
import type { Domain } from "../../types/api";

const TENANT_BADGE: Record<Domain, string> = {
  industrial: "Industrial · company_abc",
  securite: "Sécurité · company_abc",
  blockchain: "Blockchain · company_abc",
};

export default function UserFooter() {
  const { activeDomain } = useApp();

  return (
    <div className="flex items-center gap-3 border-t border-edge bg-surface-2 px-4 py-3">
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-edge bg-surface text-[11px] font-semibold text-ink-dim">
        OM
      </div>
      <div className="min-w-0 flex-1">
        <p className="truncate text-[12.5px] font-semibold text-ink">Oussama Maataoui</p>
        <p className="flex items-center gap-1 truncate text-[10.5px] text-ink-faint">
          <Building2 className="h-3 w-3 shrink-0" />
          {TENANT_BADGE[activeDomain]}
        </p>
      </div>
    </div>
  );
}
