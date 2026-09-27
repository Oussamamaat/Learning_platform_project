import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import type { ReactNode } from "react";
import type { Domain, DomainSource, Language, ResponseLang, VideoMode } from "../types/api";
import { generateQuiz, pingHealth, sendChatMessage } from "../services/api";
import {
  newMessageId,
  useChatSessions,
} from "../hooks/useChatSessions";
import { useSources } from "../hooks/useSources";
import { useVideoJobs } from "../hooks/useVideoJobs";
import { useToast } from "../hooks/useToast";
import type { ToastItem } from "../hooks/useToast";

// Doubles as the caller's role, sent to the backend as X-User-Role.
// "admin" was added 2026-09-21 with video generation: course-authoring
// features are Admin/Tenant-only, and an employee is a learner who
// creates nothing (app/services/roles.py's Role).
export type ViewMode = "admin" | "tenant" | "employee";
const VIEW_MODE_STORAGE_KEY = "atlas_tutor.view.v1";
// Which panel the tenant/admin workspace is showing. Employees never see
// the switcher -- there is only chat for them.
export type WorkspaceTab = "chat" | "video";

export const MODEL_NAME = "IBLOG_TUTOR:latest";

export type HealthStatus = "online" | "offline" | "checking";

interface HealthState {
  status: HealthStatus;
  lastChecked: number | null;
}

export interface GenerateQuizPayload {
  topic: string;
  numQuestions: number;
}

export interface GenerateVideoPayload {
  text: string;
  title?: string;
  language: Language;
  mode: VideoMode;
}

// A language switch under serial model loading (one model resident in
// VRAM at a time) is a ~30s infrastructure event -- a cold model load
// dominates at ~25-30s, vs 5-9s once warm (docs/architecture/rectified,
// analyze_04). Matches that measured ceiling so the loading state clears
// itself even if no message is sent after the switch.
const MODEL_SWAP_TIMEOUT_MS = 30000;

// Client-side mirror of app.services.llm.detect_query_language's core
// script check (Arabic-range count vs. Latin-letter count) -- just enough
// to guess, BEFORE the server round-trip, whether this message is likely
// to need the OTHER resident model (ary/fr are served by different models
// under serial loading, so a language flip is a real ~30s swap, not a
// relabel). The server's own resolution (app.services.routing, including
// the Arabizi tiebreaker and any in-message instruction) is authoritative;
// this only starts the loading indicator a beat earlier than waiting for
// the response would.
function looksLikeArabicScript(text: string): boolean {
  let arabic = 0;
  let latin = 0;
  for (const ch of text) {
    const code = ch.codePointAt(0) ?? 0;
    if (code >= 0x0600 && code <= 0x06ff) arabic++;
    else if (/[a-zA-Z]/.test(ch)) latin++;
  }
  return arabic > latin;
}

function responseLangToLanguage(lang: ResponseLang): Language {
  return lang === "darija" ? "ar-MA" : "fr";
}

interface AppContextValue {
  // Display state only, since 2026-08-11 (Automatic Domain Routing) --
  // there is no user-facing selector for either anymore. Both are seeded
  // with a reasonable pre-first-message default and then kept in sync
  // with each response's resolved domain/language (app.services.routing).
  activeDomain: Domain;
  setActiveDomain: (domain: Domain) => void;
  activeLanguage: Language;
  lastDomainSource: DomainSource | null;
  isModelSwapping: boolean;
  modelName: string;
  health: HealthState;
  refreshHealth: () => Promise<void>;
  sessions: ReturnType<typeof useChatSessions>["sessions"];
  activeSession: ReturnType<typeof useChatSessions>["activeSession"];
  activeSessionId: string | null;
  newSession: () => string;
  deleteSession: (id: string) => void;
  selectSession: (id: string) => void;
  sendMessage: (text: string) => Promise<void>;
  generateQuizInSession: (payload: GenerateQuizPayload) => Promise<void>;
  isSending: boolean;
  quizModalOpen: boolean;
  setQuizModalOpen: (open: boolean) => void;
  toastError: (message: string) => void;
  toastSuccess: (message: string) => void;
  toastInfo: (message: string) => void;
  toasts: ToastItem[];
  dismissToast: (id: number) => void;
  // Tenant document uploads (Sources panel) + the tenant/employee view
  // toggle -- purely a client-side presentation split (no auth in this
  // codebase), see TopBar.tsx.
  viewMode: ViewMode;
  setViewMode: (mode: ViewMode) => void;
  // Course authoring (video generation) is Admin/Tenant only -- mirrors
  // app.services.roles.COURSE_AUTHOR_ROLES. The backend enforces it too;
  // this just keeps the UI from offering what would be refused.
  canAuthorCourses: boolean;
  workspaceTab: WorkspaceTab;
  setWorkspaceTab: (tab: WorkspaceTab) => void;
  videoJobs: ReturnType<typeof useVideoJobs>["jobs"];
  submitVideoJob: (payload: GenerateVideoPayload) => Promise<void>;
  videoSubmitting: boolean;
  sources: ReturnType<typeof useSources>["sources"];
  uploadFiles: ReturnType<typeof useSources>["uploadFiles"];
  toggleSource: ReturnType<typeof useSources>["toggleSource"];
  removeSource: ReturnType<typeof useSources>["removeSource"];
  degraded: boolean;
}

const AppContext = createContext<AppContextValue | null>(null);

export function AppProvider({ children }: { children: ReactNode }) {
  const [activeDomain, setActiveDomain] = useState<Domain>("industrial");
  const [activeLanguage, setActiveLanguage] = useState<Language>("fr");
  const [lastDomainSource, setLastDomainSource] = useState<DomainSource | null>(null);
  const [isModelSwapping, setIsModelSwapping] = useState(false);
  const [health, setHealth] = useState<HealthState>({ status: "checking", lastChecked: null });
  const [quizModalOpen, setQuizModalOpen] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const [viewMode, setViewModeState] = useState<ViewMode>(() => {
    try {
      const stored = window.localStorage.getItem(VIEW_MODE_STORAGE_KEY);
      if (stored === "employee" || stored === "admin" || stored === "tenant") return stored;
      return "tenant";
    } catch {
      return "tenant";
    }
  });
  const [workspaceTab, setWorkspaceTab] = useState<WorkspaceTab>("chat");

  const setViewMode = useCallback((mode: ViewMode) => {
    setViewModeState(mode);
    // Switching to employee must not leave the workspace parked on a tab
    // that role cannot see (and whose backend calls it would be refused
    // for) -- send it back to chat.
    if (mode === "employee") setWorkspaceTab("chat");
    try {
      window.localStorage.setItem(VIEW_MODE_STORAGE_KEY, mode);
    } catch {
      // Storage unavailable -- the toggle still works for this session.
    }
  }, []);

  const { toasts, dismiss: dismissToast, toastError, toastSuccess, push: toastInfo } = useToast();

  const chat = useChatSessions();
  const { activeSessionId, ensureActiveSession, addMessage, updateMessage } = chat;

  const src = useSources();
  const { activeSourceIds, setDegraded } = src;

  const canAuthorCourses = viewMode !== "employee";
  const video = useVideoJobs(viewMode, canAuthorCourses);

  const swapTimerRef = useRef<number | null>(null);

  useEffect(() => {
    return () => {
      if (swapTimerRef.current) window.clearTimeout(swapTimerRef.current);
    };
  }, []);

  const sendingRef = useRef(false);

  const refreshHealth = useCallback(async () => {
    setHealth((prev) => ({ ...prev, status: "checking" }));
    const ok = await pingHealth();
    setHealth({ status: ok ? "online" : "offline", lastChecked: Date.now() });
  }, []);

  useEffect(() => {
    void refreshHealth();
    const timer = window.setInterval(() => void refreshHealth(), 20000);
    return () => window.clearInterval(timer);
  }, [refreshHealth]);

  const sendMessage = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      if (!trimmed || sendingRef.current) return;
      sendingRef.current = true;
      setIsSending(true);

      const sessionId = ensureActiveSession();
      const userId = newMessageId();
      const assistantId = newMessageId();
      const now = Date.now();

      addMessage(sessionId, { id: userId, role: "user", content: trimmed, createdAt: now });
      addMessage(sessionId, {
        id: assistantId,
        role: "assistant",
        content: "",
        createdAt: now + 1,
        pending: true,
      });

      // Optimistic swap indicator: if this message's script implies a
      // different resident model than the last resolved response, start
      // the loading state now instead of waiting for the round trip --
      // ary/fr are served by different models under serial loading, so a
      // language flip is a real ~30s infrastructure event.
      const likelyNextLanguage: Language = looksLikeArabicScript(trimmed) ? "ar-MA" : "fr";
      if (likelyNextLanguage !== activeLanguage) {
        setIsModelSwapping(true);
        if (swapTimerRef.current) window.clearTimeout(swapTimerRef.current);
        swapTimerRef.current = window.setTimeout(() => {
          setIsModelSwapping(false);
          swapTimerRef.current = null;
        }, MODEL_SWAP_TIMEOUT_MS);
      }

      try {
        // domain/language omitted -- the server resolves both
        // automatically (app.services.routing); no user-facing selector
        // sets them anymore. ChatResponse.domain/domain_source/language
        // report back what was actually decided. active_source_ids is a
        // narrowing hint only (app.services.sources.active_source_ids
        // intersects it against server-side ready+enabled state) -- an
        // empty array here means every currently-enabled upload is
        // eligible, not "none", which is why it's omitted rather than
        // sent as [] when there are no uploads yet.
        const reply = await sendChatMessage({
          message: trimmed,
          session_id: sessionId,
          active_source_ids: activeSourceIds.length > 0 ? activeSourceIds : undefined,
        });
        updateMessage(sessionId, assistantId, {
          content: reply.response,
          sources: reply.sources,
          crossLanguage: reply.cross_language,
          priorQuestions: reply.prior_questions,
          diagram: reply.diagram,
          answeredFromWeb: reply.answered_from_web,
          externalSources: reply.external_sources,
          pending: false,
        });
        setActiveDomain(reply.domain);
        setActiveLanguage(responseLangToLanguage(reply.language));
        setLastDomainSource(reply.domain_source);
        setDegraded(reply.degraded);
        // A successful reply proves the resident model is already warm --
        // no need to keep the swap-loading state around for its full
        // timeout once real evidence says the swap (if any) is done.
        if (swapTimerRef.current) {
          window.clearTimeout(swapTimerRef.current);
          swapTimerRef.current = null;
        }
        setIsModelSwapping(false);
      } catch (err) {
        const message = err instanceof Error ? err.message : "Unexpected error";
        toastError(message);
        updateMessage(sessionId, assistantId, {
          content: `Request failed: ${message}`,
          pending: false,
          error: true,
        });
      } finally {
        sendingRef.current = false;
        setIsSending(false);
      }
    },
    [
      activeLanguage,
      activeSourceIds,
      addMessage,
      ensureActiveSession,
      setDegraded,
      toastError,
      updateMessage,
    ],
  );

  const generateQuizInSession = useCallback(
    async (payload: GenerateQuizPayload) => {
      const sessionId = ensureActiveSession();
      const messageId = newMessageId();
      const now = Date.now();

      try {
        // language omitted -- auto-detected server-side from `topic`
        // (app.services.routing), same as chat. quiz.domain/domain_source/
        // language report what was actually decided.
        const quiz = await generateQuiz({
          topic: payload.topic,
          num_questions: payload.numQuestions,
        });
        setActiveDomain(quiz.domain);
        setActiveLanguage(responseLangToLanguage(quiz.language));
        setLastDomainSource(quiz.domain_source);

        if (quiz.questions.length > 0) {
          addMessage(sessionId, {
            id: messageId,
            role: "assistant",
            content: "",
            createdAt: now,
            quiz,
            sources: quiz.sources,
          });
          toastSuccess(
            quiz.total_questions < quiz.requested_questions
              ? `Quiz generated — ${quiz.total_questions}/${quiz.requested_questions} requested (limited source material)`
              : `Quiz generated — ${quiz.total_questions} grounded question(s)`,
          );
        } else {
          addMessage(sessionId, {
            id: messageId,
            role: "assistant",
            content: quiz.message ?? "No grounded questions could be generated for this topic.",
            createdAt: now,
            sources: quiz.sources,
          });
          toastInfo(
            quiz.message
              ? "No grounded quiz content available"
              : "Quiz returned without questions",
          );
        }
      } catch (err) {
        const message = err instanceof Error ? err.message : "Unexpected error";
        toastError(message);
        addMessage(sessionId, {
          id: messageId,
          role: "assistant",
          content: `Quiz generation failed: ${message}`,
          createdAt: now,
          error: true,
        });
      }
    },
    [addMessage, ensureActiveSession, toastError, toastInfo, toastSuccess],
  );

  const submitVideoJob = useCallback(
    async (payload: GenerateVideoPayload) => {
      try {
        await video.submit({
          text: payload.text,
          title: payload.title,
          language: payload.language,
          mode: payload.mode,
        });
        toastSuccess(
          "Video queued — the worker picks it up on its next poll. This takes minutes, not seconds.",
        );
      } catch (err) {
        const message = err instanceof Error ? err.message : "Unexpected error";
        // ROLE_FORBIDDEN lands here if the view was switched to employee
        // between render and submit -- the message from the backend
        // already names the role and what it may do, so pass it through.
        toastError(message);
      }
    },
    [toastError, toastSuccess, video],
  );

  const value = useMemo<AppContextValue>(
    () => ({
      activeDomain,
      setActiveDomain,
      activeLanguage,
      lastDomainSource,
      isModelSwapping,
      modelName: MODEL_NAME,
      health,
      refreshHealth,
      sessions: chat.sessions,
      activeSession: chat.activeSession,
      activeSessionId,
      newSession: chat.newSession,
      deleteSession: chat.deleteSession,
      selectSession: chat.selectSession,
      sendMessage,
      generateQuizInSession,
      isSending,
      quizModalOpen,
      setQuizModalOpen,
      toastError,
      toastSuccess,
      toastInfo,
      toasts,
      dismissToast,
      viewMode,
      setViewMode,
      canAuthorCourses,
      workspaceTab,
      setWorkspaceTab,
      videoJobs: video.jobs,
      submitVideoJob,
      videoSubmitting: video.submitting,
      sources: src.sources,
      uploadFiles: src.uploadFiles,
      toggleSource: src.toggleSource,
      removeSource: src.removeSource,
      degraded: src.degraded,
    }),
    [
      activeDomain,
      activeLanguage,
      lastDomainSource,
      isModelSwapping,
      health,
      refreshHealth,
      chat.sessions,
      chat.activeSession,
      chat.newSession,
      chat.deleteSession,
      chat.selectSession,
      activeSessionId,
      sendMessage,
      generateQuizInSession,
      isSending,
      quizModalOpen,
      toastError,
      toastSuccess,
      toastInfo,
      toasts,
      dismissToast,
      viewMode,
      setViewMode,
      canAuthorCourses,
      workspaceTab,
      video.jobs,
      submitVideoJob,
      video.submitting,
      src.sources,
      src.uploadFiles,
      src.toggleSource,
      src.removeSource,
      src.degraded,
    ],
  );

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

export function useApp(): AppContextValue {
  const ctx = useContext(AppContext);
  if (!ctx) {
    throw new Error("useApp must be used within an AppProvider");
  }
  return ctx;
}
