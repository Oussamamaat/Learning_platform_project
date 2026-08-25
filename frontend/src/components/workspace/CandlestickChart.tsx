import { useMemo } from "react";
import type { Candle } from "../../types/api";

// Hand-written SVG rather than a charting library: no library in this
// stack (Recharts included) ships an OHLC/candlestick chart type, so a
// custom shape would be needed either way -- writing it directly means no
// second charting dependency alongside mermaid, and Tailwind classes work
// here (a build-time-scanned className), unlike a server-rendered SVG
// string. React escapes every text child by construction, so a label
// straight from an LLM (measured injectable -- see
// docs/architecture/diagram-generation.md) can never break out of markup
// here; nothing on this path uses dangerouslySetInnerHTML.
const WIDTH = 640;
const HEIGHT = 220;
const PADDING = { top: 16, right: 16, bottom: 28, left: 44 };

function formatPrice(n: number): string {
  return Number.isInteger(n) ? String(n) : n.toFixed(2);
}

export default function CandlestickChart({ candles }: { candles: Candle[] }) {
  const { plotted, minLow, maxHigh } = useMemo(() => {
    const lows = candles.map((c) => c.low);
    const highs = candles.map((c) => c.high);
    const minLow = Math.min(...lows);
    const maxHigh = Math.max(...highs);
    const span = maxHigh - minLow || 1;
    const plotW = WIDTH - PADDING.left - PADDING.right;
    const plotH = HEIGHT - PADDING.top - PADDING.bottom;
    const slot = plotW / candles.length;
    const y = (price: number) => PADDING.top + plotH * (1 - (price - minLow) / span);
    const plotted = candles.map((c, i) => {
      const cx = PADDING.left + slot * (i + 0.5);
      const bullish = c.close >= c.open;
      const bodyTop = y(Math.max(c.open, c.close));
      const bodyBottom = y(Math.min(c.open, c.close));
      return {
        candle: c,
        cx,
        bodyWidth: Math.max(slot * 0.55, 3),
        wickTop: y(c.high),
        wickBottom: y(c.low),
        bodyTop,
        bodyBottom: Math.max(bodyBottom, bodyTop + 1.5), // visible even for close==open
        bullish,
      };
    });
    return { plotted, minLow, maxHigh };
  }, [candles]);

  return (
    <div className="overflow-x-auto">
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        width="100%"
        role="img"
        aria-label={`Graphique en chandeliers japonais, ${candles.length} bougies`}
        className="min-w-[420px]"
      >
        <line
          x1={PADDING.left} y1={PADDING.top}
          x2={PADDING.left} y2={HEIGHT - PADDING.bottom}
          className="stroke-edge" strokeWidth={1}
        />
        <line
          x1={PADDING.left} y1={HEIGHT - PADDING.bottom}
          x2={WIDTH - PADDING.right} y2={HEIGHT - PADDING.bottom}
          className="stroke-edge" strokeWidth={1}
        />
        <text x={4} y={PADDING.top + 4} className="fill-ink-faint text-[9px]">
          {formatPrice(maxHigh)}
        </text>
        <text x={4} y={HEIGHT - PADDING.bottom} className="fill-ink-faint text-[9px]">
          {formatPrice(minLow)}
        </text>

        {plotted.map((p, i) => (
          <g key={i} className="transition-opacity hover:opacity-70">
            <title>
              {p.candle.label || `#${i + 1}`}: O {formatPrice(p.candle.open)} H{" "}
              {formatPrice(p.candle.high)} L {formatPrice(p.candle.low)} C{" "}
              {formatPrice(p.candle.close)}
            </title>
            <line
              x1={p.cx} y1={p.wickTop} x2={p.cx} y2={p.wickBottom}
              className={p.bullish ? "stroke-success" : "stroke-danger"}
              strokeWidth={1.5}
            />
            <rect
              x={p.cx - p.bodyWidth / 2}
              y={p.bodyTop}
              width={p.bodyWidth}
              height={p.bodyBottom - p.bodyTop}
              className={p.bullish ? "fill-success" : "fill-danger"}
              rx={1}
            />
            {candles.length <= 16 && (
              <text
                x={p.cx}
                y={HEIGHT - PADDING.bottom + 12}
                textAnchor="middle"
                className="fill-ink-faint text-[8.5px]"
              >
                {p.candle.label.length > 6 ? `${p.candle.label.slice(0, 5)}…` : p.candle.label}
              </text>
            )}
          </g>
        ))}
      </svg>
    </div>
  );
}
