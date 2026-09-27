import { useCallback, useEffect, useState } from "react";
import { generateVideo, listVideoJobs } from "../services/api";
import type { Role, VideoGenerateRequest, VideoJob } from "../types/api";

// Slower than useSources' 2.5s: a video job is minutes of scene planning,
// narration and rendering, not seconds of chunking, so polling faster only
// adds requests without making the result arrive sooner.
const POLL_INTERVAL_MS = 6000;

export function useVideoJobs(role: Role, enabled: boolean) {
  const [jobs, setJobs] = useState<VideoJob[]>([]);
  const [submitting, setSubmitting] = useState(false);

  const refresh = useCallback(async () => {
    if (!enabled) return;
    try {
      setJobs(await listVideoJobs(role));
    } catch {
      // Fail quiet, same contract as useSources.refresh -- the panel keeps
      // showing its last-known state and submit() surfaces its own errors
      // through the caller's toast.
    }
  }, [enabled, role]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  // Self-stopping poll: only runs while something is actually in flight,
  // so an idle Studio tab costs nothing.
  useEffect(() => {
    if (!enabled) return;
    const busy = jobs.some((j) => j.status === "pending" || j.status === "processing");
    if (!busy) return;
    const timer = window.setInterval(() => void refresh(), POLL_INTERVAL_MS);
    return () => window.clearInterval(timer);
  }, [enabled, jobs, refresh]);

  const submit = useCallback(
    async (req: VideoGenerateRequest) => {
      setSubmitting(true);
      try {
        // No optimistic placeholder here, unlike useSources.uploadFiles:
        // POST /generate only inserts a row and returns, so the real job
        // (with its real id) lands fast enough that a placeholder would
        // just flicker.
        const job = await generateVideo(req, role);
        setJobs((prev) => [job, ...prev]);
        return job;
      } finally {
        setSubmitting(false);
      }
    },
    [role],
  );

  return { jobs, submit, submitting, refresh };
}
