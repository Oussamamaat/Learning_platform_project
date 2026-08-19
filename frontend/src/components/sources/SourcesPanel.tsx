import { FileStack } from "lucide-react";
import { useApp } from "../../context/AppContext";
import UploadDropZone from "./UploadDropZone";
import SourceItem from "./SourceItem";

export default function SourcesPanel() {
  const { sources, uploadFiles, toggleSource, removeSource, degraded } = useApp();
  // "partial" is included: its chunks are already retrievable, same as
  // "ready" -- see app.services.sources.active_source_ids.
  const readyCount = sources.filter((s) => s.status === "ready" || s.status === "partial").length;
  const totalChunks = sources.reduce((sum, s) => sum + s.chunk_count, 0);

  return (
    <aside className="flex h-full w-80 shrink-0 flex-col border-l border-edge bg-surface-2">
      <div className="flex items-center gap-2 border-b border-edge px-4 py-3.5">
        <FileStack className="h-4 w-4 text-brand" />
        <div className="min-w-0">
          <p className="text-[13px] font-semibold leading-tight text-ink">Sources</p>
          <p className="text-[10.5px] leading-tight text-ink-faint">
            {readyCount} ready · {totalChunks} chunk{totalChunks === 1 ? "" : "s"}
          </p>
        </div>
      </div>

      {degraded && (
        <div className="mx-4 mt-3 rounded-lg border border-warn/30 bg-warn-soft px-2.5 py-2 text-[11px] text-warn">
          Answering from the built-in corpus only — uploaded sources are
          temporarily unavailable.
        </div>
      )}

      <div className="px-4 pt-3.5">
        <UploadDropZone onFiles={(files) => void uploadFiles(files)} />
      </div>

      {sources.length === 0 ? (
        <div className="px-5 py-6 text-center">
          <p className="text-[12px] text-ink-faint">
            No documents uploaded yet — drop a file above to add it to this
            tenant's knowledge base.
          </p>
        </div>
      ) : (
        <nav
          aria-label="Uploaded sources"
          className="mt-3.5 flex-1 space-y-1.5 overflow-y-auto px-4 pb-4"
        >
          <p className="px-1 pb-1.5 text-[10.5px] font-semibold uppercase tracking-[0.14em] text-ink-faint">
            Uploaded documents
          </p>
          {sources.map((source) => (
            <SourceItem
              key={source.id}
              source={source}
              onToggle={(id, enabled) => void toggleSource(id, enabled)}
              onDelete={(id) => void removeSource(id)}
            />
          ))}
        </nav>
      )}
    </aside>
  );
}
