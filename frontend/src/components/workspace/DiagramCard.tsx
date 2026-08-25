import { useEffect, useId, useRef, useState } from "react";
import { ClipboardCopy, Sparkles } from "lucide-react";
import type { DiagramPayload } from "../../types/api";
import CandlestickChart from "./CandlestickChart";

// Lazy-loaded and cached at module scope -- mermaid is a large bundle and
// must not enter the main chunk for the large majority of sessions that
// never trigger a diagram (app.services.diagrams.detect_diagram_intent
// only fires on an explicit ask). dompurify is bundled alongside it for
// the same reason, even though it's far smaller on its own: it is only
// ever needed at the exact point a mermaid SVG string is sanitized, so
// there is no reason for it to load any earlier than mermaid does.
let mermaidModulePromise: Promise<[typeof import("mermaid"), typeof import("dompurify")]> | null = null;

function loadMermaid() {
  if (!mermaidModulePromise) {
    mermaidModulePromise = Promise.all([import("mermaid"), import("dompurify")]).then(
      ([mermaidMod, dompurifyMod]) => {
        // Literal hex values from src/index.css's @theme tokens, not
        // var(--color-...) references -- this app has no dark-mode
        // variant (confirmed: no prefers-color-scheme / data-theme block
        // in index.css), so there is only one palette to match. "base" is
        // the one mermaid theme meant to be recolored via themeVariables.
        mermaidMod.default.initialize({
          startOnLoad: false,
          securityLevel: "strict",
          // Mermaid's default label rendering wraps text in a <foreignObject>
          // containing HTML (<div>/<span>/<p>). DOMPurify's SVG-only profile
          // used below (deliberately not the "html" profile, to keep this
          // sanitizer's allowed surface as small as possible for
          // model-originated content) strips that HTML wholesale, which
          // leaves every node visually empty -- confirmed live with
          // Playwright during development. htmlLabels: false makes mermaid
          // emit labels as plain SVG <text>, so there is no HTML for
          // DOMPurify's SVG profile to strip, and no reason to widen it.
          htmlLabels: false,
          theme: "base",
          themeVariables: {
            primaryColor: "#e8f0fe", // --color-brand-soft
            primaryTextColor: "#1f1f1f", // --color-ink
            primaryBorderColor: "#1a73e8", // --color-brand
            lineColor: "#5f6368", // --color-ink-faint
            secondaryColor: "#f8f9fa", // --color-surface-2
            tertiaryColor: "#f1f3f4", // --color-surface-3
            background: "#ffffff", // --color-surface
            mainBkg: "#e8f0fe",
            nodeBorder: "#1a73e8",
            clusterBkg: "#f8f9fa",
            edgeLabelBackground: "#ffffff",
            fontFamily: "'Instrument Sans', ui-sans-serif, system-ui, sans-serif",
          },
        });
        return [mermaidMod, dompurifyMod] as const;
      },
    );
  }
  return mermaidModulePromise;
}

function MermaidDiagram({ source }: { source: string }) {
  // useId()'s value contains ':' characters, invalid inside an SVG element
  // id -- mermaid.render's first argument becomes exactly that.
  const renderId = useId().replace(/:/g, "_");
  const [html, setHtml] = useState<string | null>(null);
  const [error, setError] = useState(false);
  const cancelledRef = useRef(false);

  useEffect(() => {
    cancelledRef.current = false;
    setError(false);
    setHtml(null);
    loadMermaid()
      .then(async ([mermaidMod, dompurifyMod]) => {
        const { svg } = await mermaidMod.default.render(`diagram-${renderId}`, source);
        if (cancelledRef.current) return;
        // Belt-and-braces over mermaid's own securityLevel: "strict" --
        // every label in this SVG ultimately originates from a model this
        // platform has measured as prompt-injectable (see
        // docs/architecture/diagram-generation.md), so the rendered
        // markup is sanitized before it ever reaches
        // dangerouslySetInnerHTML, the one place on the diagram render
        // path that isn't safe by construction the way CandlestickChart's
        // plain React JSX is.
        setHtml(dompurifyMod.default.sanitize(svg, { USE_PROFILES: { svg: true, svgFilters: true } }));
      })
      .catch(() => {
        if (!cancelledRef.current) setError(true);
      });
    return () => {
      cancelledRef.current = true;
    };
  }, [source, renderId]);

  if (error) {
    return (
      <details className="rounded-lg border border-edge bg-surface-2 px-3 py-2 text-[11.5px] text-ink-dim">
        <summary className="cursor-pointer select-none text-ink-faint">
          Le diagramme n'a pas pu s'afficher — voir le code source
        </summary>
        <pre className="mt-2 overflow-x-auto whitespace-pre-wrap text-[10.5px]">{source}</pre>
      </details>
    );
  }

  if (!html) {
    return <div className="h-24 animate-pulse rounded-lg bg-surface-2" aria-hidden="true" />;
  }

  return (
    <div
      className="overflow-x-auto rounded-lg border border-edge bg-surface p-3 [&_svg]:mx-auto [&_svg]:max-w-none"
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}

async function copyMermaidSource(text: string) {
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    // Clipboard API unavailable (insecure context, missing permission) --
    // silently no-op, same as this app's other best-effort browser calls.
  }
}

export default function DiagramCard({ diagram }: { diagram: DiagramPayload }) {
  return (
    <div className="mt-1 space-y-2">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="flex items-center gap-1.5 text-[12.5px] font-semibold text-ink">
          <Sparkles className="h-3.5 w-3.5 text-brand" />
          {diagram.title}
        </p>
        {!diagram.grounded && (
          <span
            className="rounded-full border border-warn/30 bg-warn-soft px-2 py-0.5 text-[10px] font-semibold text-ink-dim"
            title="Ce diagramme illustre le sujet demande mais ne provient pas de vos documents televerses."
          >
            Illustratif — pas issu de vos documents
          </span>
        )}
      </div>

      {diagram.kind === "candlestick" ? (
        <CandlestickChart candles={diagram.spec.candles ?? []} />
      ) : (
        diagram.mermaid && <MermaidDiagram source={diagram.mermaid} />
      )}

      {diagram.mermaid && (
        <button
          type="button"
          onClick={() => copyMermaidSource(diagram.mermaid ?? "")}
          className="press inline-flex items-center gap-1.5 rounded-full border border-edge bg-surface-2 px-2.5 py-1 text-[10.5px] text-ink-dim hover:border-brand hover:text-ink"
        >
          <ClipboardCopy className="h-3 w-3" />
          Copier le code Mermaid
        </button>
      )}
    </div>
  );
}
