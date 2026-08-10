import StatusIndicator from "../common/StatusIndicator";

export default function SidebarHeader() {
  return (
    <div className="flex items-center justify-between gap-3 border-b border-edge px-4 py-4">
      <div className="flex min-w-0 items-center gap-2.5">
        <div className="min-w-0">
          <h1 className="text-[15px] font-semibold leading-tight tracking-tight text-ink">
            Atlas Tutor
          </h1>
          <p className="truncate text-[11px] text-ink-faint">Pipeline testing workspace</p>
        </div>
      </div>
      <StatusIndicator compact />
    </div>
  );
}