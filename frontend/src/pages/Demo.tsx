import { useState, useEffect } from 'react';
import { ChevronRight, Loader } from 'lucide-react';

interface DemoResponse {
  turn: number;
  language: string;
  model: string;
  user_question: string;
  assistant_response: string;
  context: string;
}

export default function Demo() {
  const [turn, setTurn] = useState(1);
  const [french, setFrench] = useState<DemoResponse | null>(null);
  const [darija, setDarija] = useState<DemoResponse | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    loadConversation(turn);
  }, [turn]);

  const loadConversation = async (turnNum: number) => {
    setLoading(true);
    try {
      // Fetch French conversation
      const frRes = await fetch('http://localhost:8000/api/v1/demo/conversation', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ language: 'fr', turn: turnNum }),
      });
      if (frRes.ok) {
        setFrench(await frRes.json());
      }

      // Fetch Darija conversation
      const daRes = await fetch('http://localhost:8000/api/v1/demo/conversation', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ language: 'darija', turn: turnNum }),
      });
      if (daRes.ok) {
        setDarija(await daRes.json());
      }
    } catch (err) {
      console.error('Failed to load demo:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 to-slate-800 p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-12">
          <h1 className="text-4xl font-bold text-white mb-2">Fine-Tuned Model Demo</h1>
          <p className="text-slate-400">Multi-turn conversations showcasing adaptive learning in French & Darija</p>
        </div>

        {/* Turn selector */}
        <div className="flex gap-4 mb-8">
          {[1, 2, 3].map((t) => (
            <button
              key={t}
              onClick={() => setTurn(t)}
              className={`px-6 py-3 rounded-lg font-semibold transition ${
                turn === t
                  ? 'bg-blue-600 text-white'
                  : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
              }`}
            >
              Turn {t}
            </button>
          ))}
        </div>

        {loading && (
          <div className="flex justify-center py-12">
            <Loader className="w-8 h-8 text-blue-400 animate-spin" />
          </div>
        )}

        {!loading && (french || darija) && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* French conversation */}
            {french && (
              <div className="bg-slate-700/50 backdrop-blur-sm rounded-xl p-8 border border-slate-600">
                <h2 className="text-2xl font-bold text-white mb-2">Français 🇫🇷</h2>
                <p className="text-sm text-slate-400 mb-6">{french.model}</p>

                {/* Context */}
                <div className="bg-slate-800 rounded-lg p-4 mb-6 border border-slate-600">
                  <p className="text-xs text-slate-400 uppercase tracking-wide mb-2">Context</p>
                  <p className="text-slate-300 text-sm leading-relaxed">{french.context}</p>
                </div>

                {/* User question */}
                <div className="mb-4">
                  <div className="flex items-start gap-3">
                    <div className="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center text-white text-xs font-bold flex-shrink-0">
                      U
                    </div>
                    <div>
                      <p className="text-xs text-slate-400 mb-1">You</p>
                      <p className="text-slate-200">{french.user_question}</p>
                    </div>
                  </div>
                </div>

                {/* Assistant response */}
                <div className="mb-4">
                  <div className="flex items-start gap-3">
                    <div className="w-8 h-8 rounded-full bg-emerald-600 flex items-center justify-center text-white text-xs font-bold flex-shrink-0">
                      AI
                    </div>
                    <div>
                      <p className="text-xs text-slate-400 mb-1">Assistant</p>
                      <p className="text-slate-200 leading-relaxed">{french.assistant_response}</p>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Darija conversation */}
            {darija && (
              <div className="bg-slate-700/50 backdrop-blur-sm rounded-xl p-8 border border-slate-600">
                <h2 className="text-2xl font-bold text-white mb-2">Darija 🇲🇦</h2>
                <p className="text-sm text-slate-400 mb-6">{darija.model}</p>

                {/* Context */}
                <div className="bg-slate-800 rounded-lg p-4 mb-6 border border-slate-600">
                  <p className="text-xs text-slate-400 uppercase tracking-wide mb-2">السياق</p>
                  <p className="text-slate-300 text-sm leading-relaxed">{darija.context}</p>
                </div>

                {/* User question */}
                <div className="mb-4">
                  <div className="flex items-start gap-3">
                    <div className="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center text-white text-xs font-bold flex-shrink-0">
                      U
                    </div>
                    <div>
                      <p className="text-xs text-slate-400 mb-1">أنت</p>
                      <p className="text-slate-200">{darija.user_question}</p>
                    </div>
                  </div>
                </div>

                {/* Assistant response */}
                <div className="mb-4">
                  <div className="flex items-start gap-3">
                    <div className="w-8 h-8 rounded-full bg-emerald-600 flex items-center justify-center text-white text-xs font-bold flex-shrink-0">
                      AI
                    </div>
                    <div>
                      <p className="text-xs text-slate-400 mb-1">المساعد</p>
                      <p className="text-slate-200 leading-relaxed text-right">{darija.assistant_response}</p>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Key features callout */}
        <div className="mt-12 bg-blue-600/20 border border-blue-500/30 rounded-xl p-8">
          <h3 className="text-lg font-bold text-blue-300 mb-4">✨ What You're Seeing</h3>
          <ul className="space-y-2 text-slate-300">
            <li className="flex items-center gap-2">
              <ChevronRight className="w-4 h-4 text-blue-400" />
              Multi-turn conversation showing adaptive learning progression
            </li>
            <li className="flex items-center gap-2">
              <ChevronRight className="w-4 h-4 text-blue-400" />
              Fine-tuned French model (iblog-tutor-fr) vs base Darija model (IBLOG_TUTOR)
            </li>
            <li className="flex items-center gap-2">
              <ChevronRight className="w-4 h-4 text-blue-400" />
              Socratic teaching method with grounded citations
            </li>
            <li className="flex items-center gap-2">
              <ChevronRight className="w-4 h-4 text-blue-400" />
              Bilingual domain-specific enterprise tutoring
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
}
