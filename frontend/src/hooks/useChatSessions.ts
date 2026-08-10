import { useCallback, useEffect, useState } from "react";
import type { QuizResponse } from "../types/api";

export type MessageRole = "user" | "assistant";

export interface ChatMessage {
  id: string;
  role: MessageRole;
  content: string;
  createdAt: number;
  sources?: string[];
  quiz?: QuizResponse;
  pending?: boolean;
  error?: boolean;
}

export interface ChatSession {
  id: string;
  title: string;
  messages: ChatMessage[];
  createdAt: number;
  updatedAt: number;
}

const STORAGE_KEY = "atlas_tutor.sessions.v1";
const TITLE_MAX_LEN = 44;

export function newMessageId(): string {
  return typeof crypto !== "undefined" && crypto.randomUUID
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function titleFromText(text: string): string {
  const flat = text.replace(/\s+/g, " ").trim();
  return flat.length > TITLE_MAX_LEN ? `${flat.slice(0, TITLE_MAX_LEN)}…` : flat || "New chat";
}

function getDemoSessions(): ChatSession[] {
  const now = Date.now();
  return [
    {
      id: "demo-fr",
      title: "French Tutor (Fine-tuned Model)",
      messages: [
        {
          id: "msg-fr-1",
          role: "user",
          content: "Pourquoi est-ce que le port des equipements de protection individuelle est obligatoire sur les postes a risque ?",
          createdAt: now - 300000,
        },
        {
          id: "msg-fr-2",
          role: "assistant",
          content: "Le port des equipements de protection individuelle (EPI) est obligatoire sur les postes a risque pour proteger votre sante et votre securite au travail. Conformement a l'article 281 du Code du travail marocain, les employeurs ont la responsabilite de fournir les EPI appropries.\n\nMais avant de continuer, j'aimerais comprendre votre situation. Quels types de risques rencontrez-vous dans votre environnement de travail ? Sont-ce des risques mecaniques, chimiques, ou autres ?",
          createdAt: now - 290000,
        },
        {
          id: "msg-fr-3",
          role: "user",
          content: "D'accord, mais comment je peux savoir quels equipements porter pour mon poste specifique ?",
          createdAt: now - 200000,
        },
        {
          id: "msg-fr-4",
          role: "assistant",
          content: "Excellente question ! Le choix des equipements depend directement du type de risques presents a votre poste.\n\nPour repondre efficacement, pouvez-vous me decrire :\n1. Quel est votre domaine d'activite (usine, construction, chimie, etc.) ?\n2. Quels types de dangers sont presents ? (bruit, chaleur, produits chimiques, machines en mouvement)\n3. Avez-vous deja recu une analyse des risques de votre poste ?\n\nLes EPI typiques incluent les casques, les gants, les chaussures de securite, les lunettes, et les masques respiratoires selon les besoins.",
          createdAt: now - 190000,
        },
        {
          id: "msg-fr-5",
          role: "user",
          content: "Et si je travaille dans un endroit avec beaucoup de risques chimiques ? Quels equipements supplementaires dois-je utiliser ?",
          createdAt: now - 100000,
        },
        {
          id: "msg-fr-6",
          role: "assistant",
          content: "Tres bonne question ! Les risques chimiques necessitent une protection renforcee et specifique.\n\nPour une exposition chimique, vous devez utiliser :\n- **Masque respiratoire approprie** (cartouche adaptee au type de produit chimique)\n- **Gants de protection chimique** (nitrile, neoprene selon le produit)\n- **Vetements de protection** (combinaison chimique si exposition directe)\n- **Lunettes de protection** etanche\n- **Chaussures de securite** resistantes aux produits chimiques\n\nMaintenant, une question importante pour evaluer votre comprehension : savez-vous comment verifier que votre masque est bien adapte et correctement porte ? C'est une etape cruciale pour votre protection.",
          createdAt: now - 90000,
        },
      ],
      createdAt: now - 300000,
      updatedAt: now - 90000,
    },
    {
      id: "demo-darija",
      title: "Darija Tutor (Base Model)",
      messages: [
        {
          id: "msg-dar-1",
          role: "user",
          content: "شنو هي فايدة لبس المعدات الديال السلامة فالشغل ?",
          createdAt: now - 300000,
        },
        {
          id: "msg-dar-2",
          role: "assistant",
          content: "المعدات ديال السلامة (EPI) كتحمي جسدك من الاصابات والامراض فالشغل. المادة 281 من قانون الشغل المغربي قالت انه واجب نلبس المعدات هاد فالشغل الخطير.\n\nالمعدات تشمل : الخوذة، القفازات، الحذاء الامن، والنظارات. شنو هي انواع المخاطر اللي عندك فالشغل ديالك ?",
          createdAt: now - 290000,
        },
        {
          id: "msg-dar-3",
          role: "user",
          content: "واخا فهمت. كيفاش نختار المعدات الصحيحة حسب الشغل ديالي وحسب نوع المخاطر ?",
          createdAt: now - 200000,
        },
        {
          id: "msg-dar-4",
          role: "assistant",
          content: "المعدات تختار حسب المخاطر اللي عندك. إذا كنت تشتغل قرب الآلات، لازم تلبس خوذة وقفازات ومجامة. إذا كنت تشتغل مع المواد الكيمياية، لازم تلبس قناع حماية وملابس حماية خاصة.\n\nكل صنف من الشغل عندو مخاطر مختلفة. قول لي شنو هو نوع الشغل ديالك باش نقدر ننصحك بالمعدات الصحيحة.",
          createdAt: now - 190000,
        },
        {
          id: "msg-dar-5",
          role: "user",
          content: "واش كاين معدات خاصة اذا كنت نشتغل قرب ديال المواد الخطيرة او الكيميائية ?",
          createdAt: now - 100000,
        },
        {
          id: "msg-dar-6",
          role: "assistant",
          content: "آه كاين معدات خاصة جداً للمواد الخطيرة. لازم تلبس :\n- القناع الخاص بالمواد الكيمياية (ماسك مع الفلاتر)\n- القفازات الخاصة (نيتريل او مطاط)\n- الملابس الحماية (جلاسة)\n- النظارات الحماية\n- الحذاء الامن\n\nهاد المعدات كتحميك من التسمم والحروق والاستنشاق للغازات الخطيرة. المهم تتأكد من قبل ما تشتغل.",
          createdAt: now - 90000,
        },
      ],
      createdAt: now - 300000,
      updatedAt: now - 90000,
    },
  ];
}

function loadSessions(): ChatSession[] {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return getDemoSessions();
    const parsed = JSON.parse(raw) as ChatSession[];
    return Array.isArray(parsed) ? parsed : getDemoSessions();
  } catch {
    return getDemoSessions();
  }
}

function makeSession(): ChatSession {
  const now = Date.now();
  return { id: newMessageId(), title: "New chat", messages: [], createdAt: now, updatedAt: now };
}

export function useChatSessions() {
  const [sessions, setSessions] = useState<ChatSession[]>(loadSessions);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);

  useEffect(() => {
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions));
    } catch {
      // Storage unavailable (private mode / quota) — sessions stay in memory.
    }
  }, [sessions]);

  const activeSession =
    sessions.find((s) => s.id === activeSessionId) ?? null;

  const newSession = useCallback((): string => {
    const session = makeSession();
    setSessions((prev) => [session, ...prev]);
    setActiveSessionId(session.id);
    return session.id;
  }, []);

  const selectSession = useCallback((id: string) => {
    setActiveSessionId(id);
  }, []);

  const deleteSession = useCallback((id: string) => {
    setSessions((prev) => {
      const next = prev.filter((s) => s.id !== id);
      return next;
    });
    setActiveSessionId((current) => (current === id ? null : current));
  }, []);

  const ensureActiveSession = useCallback((): string => {
    if (activeSessionId) return activeSessionId;
    return newSession();
  }, [activeSessionId, newSession]);

  const addMessage = useCallback(
    (sessionId: string, message: ChatMessage) => {
      setSessions((prev) =>
        prev.map((s) => {
          if (s.id !== sessionId) return s;
          const title =
            message.role === "user" && s.messages.length === 0 && message.content
              ? titleFromText(message.content)
              : s.title;
          return {
            ...s,
            title,
            messages: [...s.messages, message],
            updatedAt: message.createdAt,
          };
        }),
      );
    },
    [],
  );

  const updateMessage = useCallback(
    (sessionId: string, messageId: string, patch: Partial<ChatMessage>) => {
      setSessions((prev) =>
        prev.map((s) => {
          if (s.id !== sessionId) return s;
          return {
            ...s,
            messages: s.messages.map((m) =>
              m.id === messageId ? { ...m, ...patch } : m,
            ),
            updatedAt: Date.now(),
          };
        }),
      );
    },
    [],
  );

  return {
    sessions,
    activeSession,
    activeSessionId,
    newSession,
    selectSession,
    deleteSession,
    ensureActiveSession,
    addMessage,
    updateMessage,
  };
}
