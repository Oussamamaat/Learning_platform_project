// Real-Mermaid-parser CI gate for generated diagrams.
//
// Standing rule from the archived feasibility study
// (_archive/docs_superseded/STRATEGY_V2.md:251): "Every mermaid block must
// parse. Real parser, not a regex." Node is available in this environment
// (see docs/architecture/diagram-generation.md), so tests/test_diagram_
// mermaid_parses.py invokes this script as a subprocess to check every
// fixture app.services.diagram_render produces against the actual mermaid
// parser -- not a hand-rolled syntax approximation.
//
// mermaid needs a real `window` (its bundled DOMPurify is constructed
// against one) -- plain Node fails with "DOMPurify.addHook is not a
// function" without it, hence jsdom here. This script is dev/CI tooling
// only, never shipped to the browser bundle (see frontend/package.json's
// devDependencies).
//
// Usage: node scripts/verify_mermaid.mjs <path-to-fixtures.json>
//   fixtures.json: [{ "kind": string, "source": string }, ...]
// Prints one JSON object per line to stdout: { kind, ok, error? }.

import { readFileSync } from "node:fs";
import { JSDOM } from "jsdom";

const dom = new JSDOM("<!DOCTYPE html><html><body></body></html>");
global.window = dom.window;
global.document = dom.window.document;
global.SVGElement = dom.window.SVGElement;
Object.defineProperty(global, "navigator", { value: dom.window.navigator, configurable: true });

const mermaid = (await import("mermaid")).default;
mermaid.initialize({ startOnLoad: false, securityLevel: "strict" });

const fixturesPath = process.argv[2];
if (!fixturesPath) {
  console.error("usage: node verify_mermaid.mjs <fixtures.json>");
  process.exit(2);
}
const fixtures = JSON.parse(readFileSync(fixturesPath, "utf-8"));

let allOk = true;
for (const { kind, source } of fixtures) {
  try {
    await mermaid.parse(source);
    console.log(JSON.stringify({ kind, ok: true }));
  } catch (e) {
    allOk = false;
    console.log(JSON.stringify({ kind, ok: false, error: String(e && e.message ? e.message : e) }));
  }
}
process.exit(allOk ? 0 : 1);
