import { useCallback, useEffect, useState } from "react";
import { deleteSource, listSources, setSourceEnabled, uploadSources } from "../services/api";
import type { SourceFile } from "../types/api";

const POLL_INTERVAL_MS = 2500;

export function useSources() {
  const [sources, setSources] = useState<SourceFile[]>([]);
  const [degraded, setDegraded] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const { sources: fetched } = await listSources();
      setSources(fetched);
    } catch {
      // Fail quiet -- the panel just keeps showing its last-known state;
      // uploadFiles/toggle/delete surface their own errors via toasts at
      // the call site (AppContext), this hook doesn't own toast display.
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  // Self-stopping poll: only runs while at least one source is still
  // uploading/processing, so an idle Sources panel costs nothing.
  useEffect(() => {
    const busy = sources.some((s) => s.status === "pending" || s.status === "processing");
    if (!busy) return;
    const timer = window.setInterval(() => void refresh(), POLL_INTERVAL_MS);
    return () => window.clearInterval(timer);
  }, [sources, refresh]);

  const uploadFiles = useCallback(
    async (files: File[]) => {
      // Optimistic placeholders so drag-drop feels instant; refresh()
      // reconciles them against the server's real rows (and real ids) once
      // the upload response lands.
      const placeholders: SourceFile[] = files.map((f) => ({
        id: `pending-${f.name}-${Date.now()}-${Math.random().toString(36).slice(2)}`,
        filename: f.name,
        status: "pending",
        enabled: true,
        chunk_count: 0,
        size_bytes: f.size,
        created_at: new Date().toISOString(),
      }));
      setSources((prev) => [...placeholders, ...prev]);
      try {
        await uploadSources(files);
      } finally {
        await refresh();
      }
    },
    [refresh],
  );

  const toggleSource = useCallback(async (id: string, enabled: boolean) => {
    setSources((prev) => prev.map((s) => (s.id === id ? { ...s, enabled } : s)));
    await setSourceEnabled(id, enabled);
  }, []);

  const removeSource = useCallback(async (id: string) => {
    setSources((prev) => prev.filter((s) => s.id !== id));
    await deleteSource(id);
  }, []);

  // "partial" is retrievable too -- its successfully-parsed pages are
  // already chunked server-side (app.services.sources.active_source_ids
  // treats ready/partial identically). Excluding it here would silently
  // narrow it back out even though the backend now allows it, since this
  // list is sent as a NARROWING filter, never a widening one.
  const activeSourceIds = sources
    .filter((s) => (s.status === "ready" || s.status === "partial") && s.enabled)
    .map((s) => s.id);

  return {
    sources,
    activeSourceIds,
    uploadFiles,
    toggleSource,
    removeSource,
    refresh,
    degraded,
    setDegraded,
  };
}
