import { useState } from "react";
import {
  AlertCircle,
  File,
  FileSpreadsheet,
  FileText,
  Image as ImageIcon,
  Loader2,
  Presentation,
  Trash2,
} from "lucide-react";
import type { SourceFile } from "../../types/api";

function iconForFilename(filename: string) {
  const ext = filename.split(".").pop()?.toLowerCase() ?? "";
  if (["pdf", "txt", "md"].includes(ext)) return FileText;
  if (["docx"].includes(ext)) return FileText;
  if (["pptx"].includes(ext)) return Presentation;
  if (["xlsx", "csv"].includes(ext)) return FileSpreadsheet;
  if (["png", "jpg", "jpeg", "tiff", "tif"].includes(ext)) return ImageIcon;
  return File;
}

function StatusPill({ source }: { source: SourceFile }) {
  switch (source.status) {
    case "pending":
      return (
        <span className="inline-flex items-center gap-1 rounded-full bg-surface-3 px-2 py-0.5 text-[10px] text-ink-faint">
          <Loader2 className="h-2.5 w-2.5 animate-spin" /> Uploading
        </span>
      );
    case "processing":
      return (
        <span className="inline-flex items-center gap-1 rounded-full bg-warn-soft px-2 py-0.5 text-[10px] text-warn">
          <Loader2 className="h-2.5 w-2.5 animate-spin" /> Processing
        </span>
      );
    case "ready":
      return (
        <span className="inline-flex items-center gap-1 rounded-full bg-success-soft px-2 py-0.5 text-[10px] text-success-deep">
          {source.chunk_count} chunk{source.chunk_count === 1 ? "" : "s"}
        </span>
      );
    case "partial": {
      const skipped = source.unprocessed_pages?.length ?? 0;
      return (
        <span className="inline-flex items-center gap-1 rounded-full bg-warn-soft px-2 py-0.5 text-[10px] text-warn">
          <AlertCircle className="h-2.5 w-2.5" />
          {source.chunk_count} chunk{source.chunk_count === 1 ? "" : "s"}
          {skipped > 0 ? ` · ${skipped} page${skipped === 1 ? "" : "s"} skipped` : ""}
        </span>
      );
    }
    case "error":
      return (
        <span className="inline-flex items-center gap-1 rounded-full bg-danger-soft px-2 py-0.5 text-[10px] text-danger-deep">
          <AlertCircle className="h-2.5 w-2.5" /> Error
        </span>
      );
    default:
      return null;
  }
}

interface SourceItemProps {
  source: SourceFile;
  onToggle: (id: string, enabled: boolean) => void;
  onDelete: (id: string) => void;
}

export default function SourceItem({ source, onToggle, onDelete }: SourceItemProps) {
  const Icon = iconForFilename(source.filename);
  const isBusy = source.status === "pending" || source.status === "processing";
  const [errorExpanded, setErrorExpanded] = useState(false);

  return (
    <div className="rounded-lg border border-edge bg-surface px-2.5 py-2">
      <div className="flex items-center gap-2">
        <Icon className="h-3.5 w-3.5 shrink-0 text-ink-faint" />
        <div className="min-w-0 flex-1">
          <p
            title={source.filename}
            className={`truncate text-[12.5px] font-medium leading-snug ${
              !source.enabled ? "text-ink-faint line-through" : "text-ink"
            }`}
          >
            {source.filename}
          </p>
          <div className="mt-0.5">
            <StatusPill source={source} />
          </div>
        </div>
        {!isBusy && (
          <input
            type="checkbox"
            checked={source.enabled}
            onChange={(e) => onToggle(source.id, e.target.checked)}
            aria-label={`Enable ${source.filename} in chat context`}
            className="h-3.5 w-3.5 shrink-0 accent-brand"
          />
        )}
        <button
          type="button"
          onClick={() => onDelete(source.id)}
          aria-label={`Delete ${source.filename}`}
          className="press-icon shrink-0 rounded-md p-1 text-ink-faint hover:bg-danger-soft hover:text-danger"
        >
          <Trash2 className="h-3.5 w-3.5" />
        </button>
      </div>
      {source.status === "error" && source.error_message && (
        <button
          type="button"
          onClick={() => setErrorExpanded((v) => !v)}
          className="mt-1.5 w-full text-left text-[10.5px] text-danger-deep"
        >
          {errorExpanded ? source.error_message : "Show error details"}
        </button>
      )}
      {source.status === "partial" && (source.unprocessed_pages?.length ?? 0) > 0 && (
        <button
          type="button"
          onClick={() => setErrorExpanded((v) => !v)}
          className="mt-1.5 w-full text-left text-[10.5px] text-warn"
        >
          {errorExpanded
            ? `Skipped: ${source.unprocessed_pages!
                .map((p) => `page ${p.page}${p.detail ? ` (${p.detail})` : ` (${p.reason})`}`)
                .join("; ")}. The rest of the document is available.`
            : "Show skipped pages"}
        </button>
      )}
    </div>
  );
}
