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
import type { Domain, DomainSource, Language, ResponseLang } from "../types/api";
import { generateQuiz, pingHealth, sendChatMessage } from "../services/api";
import {
  newMessageId,
  useChatSessions,
} from "../hooks/useChatSessions";
import { useToast } from "../hooks/useToast";
import type { ToastItem } from "../hooks/useToast";

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

  const { toasts, dismiss: dismissToast, toastError, toastSuccess, push: toastInfo } = useToast();

  const chat = useChatSessions();
  const { activeSessionId, ensureActiveSession, addMessage, updateMessage } = chat;

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
        // report back what was actually decided.
        const reply = await sendChatMessage({ message: trimmed, session_id: sessionId });
        updateMessage(sessionId, assistantId, {
          content: reply.response,
          sources: reply.sources,
          crossLanguage: reply.cross_language,
          priorQuestions: reply.prior_questions,
          pending: false,
        });
        setActiveDomain(reply.domain);
        setActiveLanguage(responseLangToLanguage(reply.language));
        setLastDomainSource(reply.domain_source);
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
      addMessage,
      ensureActiveSession,
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
