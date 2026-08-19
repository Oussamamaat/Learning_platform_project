import { useRef, useState } from "react";
import { Loader2, Paperclip, Send } from "lucide-react";
import { useApp } from "../../context/AppContext";
import type { Domain, Language } from "../../types/api";

const QUICK_PROMPTS: Record<Domain, Record<Language, string[]>> = {
  industrial: {
    fr: [
      "Explique la procédure de consignation LOTO",
      "C'est quoi un espace confiné ?",
      "Comment fonctionne un permis de travail ?",
    ],
    "ar-MA": [
      "شرح ليا إجراء LOTO ديال العزلة",
      "شنو هو الفضاء المحصور؟",
      "كيفاش كيخدم تصريح الخدمة؟",
    ],
    en: [
      "Explain the LOTO lockout/tagout procedure",
      "What is a confined space?",
      "How does a work permit work?",
    ],
  },
  securite: {
    fr: [
      "Quelles sont les procédures d'évacuation ?",
      "C'est quoi une évaluation des risques incendie ?",
      "Qui est responsable de la sécurité sur site ?",
    ],
    "ar-MA": [
      "شنو هي إجراءات الإخلاء؟",
      "شنو هو التقييم ديال خطر الحريق؟",
      "شكون المسؤول على السلامة فالموقع؟",
    ],
    en: [
      "What are the evacuation procedures?",
      "What is a fire risk assessment?",
      "Who is responsible for on-site safety?",
    ],
  },
  blockchain: {
    fr: [
      "C'est quoi un smart contract ?",
      "Explique le consensus proof-of-work",
      "Comment sécuriser une transaction ?",
    ],
    "ar-MA": [
      "شنو هو العقد الذكي؟",
      "اشرح لي آلية إجماع إثبات العمل",
      "كيفاش نأمنو عملية؟",
    ],
    en: [
      "What is a smart contract?",
      "Explain proof-of-work consensus",
      "How do you secure a transaction?",
    ],
  },
};

const UPLOAD_ACCEPT = ".txt,.md,.pdf,.docx,.pptx,.xlsx,.csv,.png,.jpg,.jpeg,.tiff,.tif";

export default function InputArea() {
  const { sendMessage, isSending, isModelSwapping, activeDomain, activeLanguage, viewMode, uploadFiles } =
    useApp();
  const [text, setText] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const canSend = text.trim().length > 0 && !isSending && !isModelSwapping;

  const handleSend = () => {
    if (!canSend) return;
    void sendMessage(text);
    setText("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setText(e.target.value);
    const el = e.target;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  };

  return (
    <div className="shrink-0 px-4 pb-4 sm:px-6">
      <div className="mx-auto w-full max-w-3xl">
        <div className="mb-2 flex gap-1.5 overflow-x-auto pb-0.5">
          {QUICK_PROMPTS[activeDomain][activeLanguage].map((prompt) => (
            <button
              key={prompt}
              type="button"
              disabled={isSending || isModelSwapping}
              onClick={() => void sendMessage(prompt)}
              className="press shrink-0 rounded-full border border-edge bg-surface px-3 py-1.5 text-[11.5px] text-ink-dim hover:border-brand hover:bg-brand-soft hover:text-brand-deep disabled:opacity-50"
            >
              {prompt}
            </button>
          ))}
        </div>

        <div className="flex items-end gap-2 rounded-2xl border border-edge bg-surface px-2.5 py-2 transition-colors duration-150 ease-spring focus-within:border-brand">
          {viewMode === "tenant" && (
            <>
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="press-icon flex h-9 w-9 shrink-0 items-center justify-center rounded-xl text-ink-faint hover:bg-surface-3 hover:text-ink"
                aria-label="Upload a document to this tenant's knowledge base"
              >
                <Paperclip className="h-4.5 w-4.5" />
              </button>
              <input
                ref={fileInputRef}
                type="file"
                multiple
                accept={UPLOAD_ACCEPT}
                className="hidden"
                onChange={(e) => {
                  const files = Array.from(e.target.files ?? []);
                  if (files.length) void uploadFiles(files);
                  e.target.value = "";
                }}
              />
            </>
          )}
          <textarea
            ref={textareaRef}
            value={text}
            onChange={handleInput}
            onKeyDown={handleKeyDown}
            rows={1}
            disabled={isModelSwapping}
            placeholder={
              isModelSwapping
                ? "Switching tutor language — the model is loading…"
                : "Ask the tutor about the tenant's course material…"
            }
            className="max-h-[200px] min-h-[36px] flex-1 resize-none bg-transparent px-1.5 py-1.5 text-[13.5px] leading-relaxed text-ink placeholder:text-ink-faint outline-none disabled:cursor-not-allowed"
          />
          <button
            type="button"
            onClick={handleSend}
            disabled={!canSend}
            className="press-icon flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-brand text-white hover:bg-brand-dark disabled:cursor-not-allowed disabled:bg-brand/40"
            aria-label="Send message"
          >
            {isSending || isModelSwapping ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Send className="h-4 w-4" />
            )}
          </button>
        </div>

        <p className="mt-1.5 text-center text-[10.5px] text-ink-faint">
          {isModelSwapping
            ? "Loading the tutor for the newly selected language — this can take up to 30 seconds."
            : "Atlas Tutor may make mistakes — verify grounded citations against the sources."}
        </p>
      </div>
    </div>
  );
}
