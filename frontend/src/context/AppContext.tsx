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
import type { Domain, Language } from "../types/api";
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
  language: Language;
}

interface AppContextValue {
  activeDomain: Domain;
  setActiveDomain: (domain: Domain) => void;
  activeLanguage: Language;
  setActiveLanguage: (language: Language) => void;
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
  const [health, setHealth] = useState<HealthState>({ status: "checking", lastChecked: null });
  const [quizModalOpen, setQuizModalOpen] = useState(false);
  const [isSending, setIsSending] = useState(false);

  const { toasts, dismiss: dismissToast, toastError, toastSuccess, push: toastInfo } = useToast();

  const chat = useChatSessions();
  const { activeSessionId, ensureActiveSession, addMessage, updateMessage } = chat;

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

      try {
        const reply = await sendChatMessage({
          message: trimmed,
          session_id: sessionId,
          domain: activeDomain,
          language: activeLanguage,
        });
        updateMessage(sessionId, assistantId, {
          content: reply.response,
          sources: reply.sources,
          pending: false,
        });
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
      activeDomain,
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
        const quiz = await generateQuiz({
          topic: payload.topic,
          num_questions: payload.numQuestions,
          language: payload.language,
        });

        if (quiz.questions.length > 0) {
          addMessage(sessionId, {
            id: messageId,
            role: "assistant",
            content: "",
            createdAt: now,
            quiz,
            sources: quiz.sources,
          });
          toastSuccess(`Quiz generated — ${quiz.total_questions} grounded question(s)`);
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
      setActiveLanguage,
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
