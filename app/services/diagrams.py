"""
Diagram Generation Service
────────────────────────────
Turns a chat message that asks for a diagram into a DiagramPayload, via the
same fine-tuned model chat serving uses. Structurally mirrors
app/services/quiz.py (prompt + JSON `format` schema + Ollama call + parse),
but with two extra tiers quiz doesn't need:

1. HEAL (salvage, never discard) -- a single dangling edge or an over-limit
   node count must not cost a whole ~90s Ollama round trip. Repairs are
   recorded in DiagramPayload.repairs, never applied silently (same
   provenance convention as ChatResponse.domain_source / .degraded and
   grounding.py's `_reject_reason`). Healing never INVENTS content --
   dropping a dangling edge is lossless-safe, synthesizing a phantom node
   to satisfy it would not be.
2. GATE (reject) -- language (structural labels must be Latin-script French,
   never Arabic/CJK) and grounding (no reference absent from the retrieved
   context, reusing generate_training_data.row_has_ungrounded_reference,
   the same detector grounding.py wraps for quiz).

Why the model emits JSON, not Mermaid directly: this is exactly the
structure/truth split quiz.py's own docstring describes -- constrained
decoding (Ollama's `format`) guarantees SHAPE, never TRUTHFULNESS.
_archive/docs_superseded/STRATEGY_V2.md:124-142 measured the fine-tune
writing Mermaid directly and found two recurring SYNTAX defects (unquoted
labels, malformed edge-label syntax) despite reliable semantic content
(6/6). Both defects vanish structurally once the model only ever emits a
validated JSON spec and app.services.diagram_render owns every character
of Mermaid syntax -- no retraining needed, unlike the archived plan's
proposed `iblog-diagram` adapter.

Why intent detection is a deterministic keyword router, not a model
judgement: docs/architecture/rectified/analyze_01.md §0 names this repo's
root defect class as "deterministic work offloaded to probabilistic
weights", and app/services/quiz.py:96-100 is the recorded evidence a 9B
adapter does not reliably follow instructions/counts it wasn't trained on.
Whether THIS message wants a diagram, and of which kind, is answered by
regex before any model is involved.
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel, Field, ValidationError

from app.config import get_settings
from app.models.schemas import Candle, DiagramPayload
from app.services import diagram_render
from app.services.generate_training_data import (
    has_arabic_script,
    row_has_ungrounded_reference,
)
from app.services.llm import _call_ollama_generate, build_diagram_prompt, diagram_kind_hint

logger = logging.getLogger(__name__)

__all__ = ["DiagramIntent", "detect_diagram_intent", "generate_diagram"]


# ── Intent detection ─────────────────────────────────────────────────────


@dataclass
class DiagramIntent:
    """kind: one of DiagramKind's values. source: "keyword" (Tier 1,
    deterministic) or "semantic" (Tier 2, not yet implemented -- see
    _semantic_fallback). A dataclass rather than Optional[str] from the
    start so this mirrors app.services.routing.LanguageResolution's own
    provenance-carrying shape, and so Tier 2 is a non-breaking addition."""

    kind: str
    source: str
    confidence: float


# Kind-specific keyword sets, checked before the generic diagram-noun/
# draw-verb fallback below -- "bougie" alone should route straight to
# candlestick without also needing "dessine" or "schéma" in the message.
# Checked in this order (dict iteration order), so a message matching more
# than one kind's keywords takes the first (most it encounters).
_KIND_KEYWORDS: dict[str, list[str]] = {
    "candlestick": [
        "bougie", "bougies", "chandelier", "chandeliers", "candlestick",
        "candlesticks", "ohlc", "chandeliers japonais",
    ],
    "pie": ["camembert", "pie chart", "répartition en", "repartition en", "secteurs"],
    "sequence": [
        "diagramme de séquence", "diagramme de sequence", "sequence diagram",
        "échange de messages", "echange de messages",
    ],
    "mindmap": ["carte mentale", "mind map", "mindmap"],
    "xy": [
        "histogramme", "courbe de", "graphique en barres", "bar chart",
        "line chart", "évolution de", "evolution de",
    ],
    "flowchart": ["organigramme", "flowchart", "logigramme", "étapes du processus", "etapes du processus"],
}

# Generic diagram nouns / draw verbs -- any of these without a more specific
# kind keyword above falls back to "flowchart" at lower confidence. Kept
# deliberately narrow: a bare topic noun ("le processus de consignation")
# must NEVER trigger on its own, only an explicit ask to draw/show something.
_DIAGRAM_NOUNS_FR = ["schéma", "schema", "diagramme", "graphique", "diagramme de", "visuel"]
_DRAW_VERBS_FR = [
    "dessine", "trace", "représente", "represente", "visualise",
    "schématise", "schematise", "génère un schéma", "genere un schema",
    "montre-moi", "montre moi",
]
_DIAGRAM_MARKERS_AR = ["رسم", "مخطط", "شكل بياني"]


def detect_diagram_intent(message: str) -> Optional[DiagramIntent]:
    """Tier 1 (deterministic keywords), falling through to Tier 2 (semantic,
    stubbed) only when Tier 1 finds nothing. Returns None for an ordinary
    chat message -- the caller (app/routers/chat.py) then proceeds exactly
    as it does today, no diagram involved."""
    lowered = (message or "").lower()

    for kind, keywords in _KIND_KEYWORDS.items():
        if any(kw in lowered for kw in keywords):
            return DiagramIntent(kind=kind, source="keyword", confidence=1.0)

    if (
        any(n in lowered for n in _DIAGRAM_NOUNS_FR)
        or any(v in lowered for v in _DRAW_VERBS_FR)
        or any(m in message for m in _DIAGRAM_MARKERS_AR)
    ):
        return DiagramIntent(kind="flowchart", source="keyword", confidence=0.6)

    return _semantic_fallback(message)


def _semantic_fallback(message: str) -> Optional[DiagramIntent]:
    """TODO(diagram-intent-tier-2): catch implicit diagram requests that
    carry no Tier-1 keyword ("peux-tu me montrer visuellement ?"). Must
    only ever run when Tier 1 returns nothing -- Tier 1's false-positive
    guard (a bare topic noun must never trigger) is load-bearing and must
    not be diluted by a fuzzier tier voting alongside it.

    The intended cheap implementation: cosine similarity between the
    message's embedding and a handful of French exemplar phrases, using
    the embedding model already resident at process startup
    (app/main.py's _preload_embedding_model -> app.services.ingestion.
    load_embedding_model) -- a few lines, no new infrastructure. NOT a
    zero-shot LLM classifier, which would add a GPU round trip to every
    chat turn, including the large majority that want no diagram at all.
    """
    return None


# ── Per-kind LLM-facing specs (parsed from the model's JSON) ────────────


class FlowNode(BaseModel):
    id: str
    label: str


class FlowEdge(BaseModel):
    source: str
    target: str
    label: Optional[str] = None


class FlowchartSpec(BaseModel):
    title: str
    caption: str
    direction: str = "TD"
    nodes: list[FlowNode]
    edges: list[FlowEdge] = Field(default_factory=list)


class SeqMessage(BaseModel):
    from_: str = Field(alias="from")
    to: str
    text: str

    model_config = {"populate_by_name": True}


class SequenceSpec(BaseModel):
    title: str
    caption: str
    participants: list[str]
    messages: list[SeqMessage] = Field(default_factory=list)


class MindBranch(BaseModel):
    label: str
    children: list[str] = Field(default_factory=list)


class MindmapSpec(BaseModel):
    title: str
    caption: str
    root: str
    branches: list[MindBranch] = Field(default_factory=list)


class PieSlice(BaseModel):
    label: str
    value: float


class PieSpec(BaseModel):
    title: str
    caption: str
    slices: list[PieSlice]


class XySpec(BaseModel):
    title: str
    caption: str
    x_labels: list[str]
    y_axis_label: str = ""
    chart_type: str = "bar"
    values: list[float]


class CandleItem(BaseModel):
    label: str = ""
    open: float
    high: float
    low: float
    close: float


class CandlestickSpec(BaseModel):
    title: str
    caption: str
    candles: list[CandleItem]


_SPEC_MODELS: dict[str, type[BaseModel]] = {
    "flowchart": FlowchartSpec,
    "sequence": SequenceSpec,
    "mindmap": MindmapSpec,
    "pie": PieSpec,
    "xy": XySpec,
    "candlestick": CandlestickSpec,
}


# ── Per-kind JSON Schemas for Ollama's `format` ──────────────────────────
#
# Kept FLAT and SHALLOW deliberately -- no nested $ref/$defs (unlike
# pydantic's own .model_json_schema(), which would emit them for nested
# models). Deep nesting degrades a 9B model even when the grammar
# guarantees parseability, and quiz.py's own hand-rolled _quiz_format_schema
# is the proven-working precedent for this style. Duplicating field shapes
# against the Pydantic models above (rather than generating one from the
# other) is the same deliberate trade-off quiz.py's _QUESTION_SCHEMA
# docstring makes for the same reason: serving's schema needs a different
# shape than a generated one would produce. Keep both in sync by hand.

_FLOWCHART_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "minLength": 3, "maxLength": 120},
        "caption": {"type": "string", "minLength": 10, "maxLength": 400},
        "direction": {"type": "string", "enum": ["TD", "LR"]},
        "nodes": {
            "type": "array",
            "minItems": 1,
            "maxItems": 20,
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "minLength": 1, "maxLength": 20},
                    "label": {"type": "string", "minLength": 1, "maxLength": 120},
                },
                "required": ["id", "label"],
            },
        },
        "edges": {
            "type": "array",
            "maxItems": 30,
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "minLength": 1, "maxLength": 20},
                    "target": {"type": "string", "minLength": 1, "maxLength": 20},
                    "label": {"type": "string", "maxLength": 60},
                },
                "required": ["source", "target"],
            },
        },
    },
    "required": ["title", "caption", "nodes", "edges"],
}

_SEQUENCE_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "minLength": 3, "maxLength": 120},
        "caption": {"type": "string", "minLength": 10, "maxLength": 400},
        "participants": {
            "type": "array",
            "minItems": 2,
            "maxItems": 8,
            "items": {"type": "string", "minLength": 1, "maxLength": 40},
        },
        "messages": {
            "type": "array",
            "minItems": 1,
            "maxItems": 20,
            "items": {
                "type": "object",
                "properties": {
                    "from": {"type": "string", "minLength": 1, "maxLength": 40},
                    "to": {"type": "string", "minLength": 1, "maxLength": 40},
                    "text": {"type": "string", "minLength": 1, "maxLength": 160},
                },
                "required": ["from", "to", "text"],
            },
        },
    },
    "required": ["title", "caption", "participants", "messages"],
}

_MINDMAP_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "minLength": 3, "maxLength": 120},
        "caption": {"type": "string", "minLength": 10, "maxLength": 400},
        "root": {"type": "string", "minLength": 1, "maxLength": 60},
        "branches": {
            "type": "array",
            "minItems": 1,
            "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "minLength": 1, "maxLength": 60},
                    "children": {
                        "type": "array",
                        "maxItems": 6,
                        "items": {"type": "string", "minLength": 1, "maxLength": 60},
                    },
                },
                "required": ["label", "children"],
            },
        },
    },
    "required": ["title", "caption", "root", "branches"],
}

_PIE_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "minLength": 3, "maxLength": 120},
        "caption": {"type": "string", "minLength": 10, "maxLength": 400},
        "slices": {
            "type": "array",
            "minItems": 2,
            "maxItems": 10,
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "minLength": 1, "maxLength": 60},
                    "value": {"type": "number", "minimum": 0},
                },
                "required": ["label", "value"],
            },
        },
    },
    "required": ["title", "caption", "slices"],
}

_XY_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "minLength": 3, "maxLength": 120},
        "caption": {"type": "string", "minLength": 10, "maxLength": 400},
        "x_labels": {
            "type": "array",
            "minItems": 2,
            "maxItems": 15,
            "items": {"type": "string", "minLength": 1, "maxLength": 40},
        },
        "y_axis_label": {"type": "string", "maxLength": 60},
        "chart_type": {"type": "string", "enum": ["bar", "line"]},
        "values": {
            "type": "array",
            "minItems": 2,
            "maxItems": 15,
            "items": {"type": "number"},
        },
    },
    "required": ["title", "caption", "x_labels", "values"],
}

_CANDLESTICK_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "minLength": 3, "maxLength": 120},
        "caption": {"type": "string", "minLength": 10, "maxLength": 400},
        "candles": {
            "type": "array",
            "minItems": 2,
            "maxItems": 20,
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "maxLength": 30},
                    "open": {"type": "number"},
                    "high": {"type": "number"},
                    "low": {"type": "number"},
                    "close": {"type": "number"},
                },
                "required": ["label", "open", "high", "low", "close"],
            },
        },
    },
    "required": ["title", "caption", "candles"],
}

_SCHEMAS: dict[str, dict] = {
    "flowchart": _FLOWCHART_SCHEMA,
    "sequence": _SEQUENCE_SCHEMA,
    "mindmap": _MINDMAP_SCHEMA,
    "pie": _PIE_SCHEMA,
    "xy": _XY_SCHEMA,
    "candlestick": _CANDLESTICK_SCHEMA,
}


# ── Tier A: heal (salvage, never discard) ────────────────────────────────
#
# Each function returns (healed_spec_or_None, repairs). None means the
# salvage floor wasn't met -- too little survived to be a real diagram, so
# the caller retries generation rather than rendering a husk. Every repair
# taken is a REMOVAL (a dangling edge, an empty label, an item past the
# node-count ceiling), never an invention -- there is deliberately no
# "synthesize a phantom node" path anywhere in this module.


def _heal_flowchart(spec: FlowchartSpec, max_nodes: int):
    repairs: list[str] = []
    seen_ids: set[str] = set()
    nodes: list[FlowNode] = []
    for n in spec.nodes:
        label = (n.label or "").strip()
        if not label:
            repairs.append(f"dropped node '{n.id}' with an empty label")
            continue
        if n.id in seen_ids:
            repairs.append(f"dropped duplicate node id '{n.id}'")
            continue
        seen_ids.add(n.id)
        nodes.append(n)
    if len(nodes) > max_nodes:
        repairs.append(f"truncated nodes from {len(nodes)} to {max_nodes}")
        nodes = nodes[:max_nodes]
    valid_ids = {n.id for n in nodes}

    edges: list[FlowEdge] = []
    for e in spec.edges:
        if e.source == e.target:
            repairs.append(f"dropped self-loop edge on '{e.source}'")
            continue
        if e.source not in valid_ids or e.target not in valid_ids:
            repairs.append(f"dropped dangling edge {e.source!r} -> {e.target!r}")
            continue
        edges.append(e)

    if len(nodes) < 2 or len(edges) < 1:
        return None, repairs
    return spec.model_copy(update={"nodes": nodes, "edges": edges}), repairs


def _heal_sequence(spec: SequenceSpec, max_nodes: int):
    repairs: list[str] = []
    seen: set[str] = set()
    participants: list[str] = []
    for p in spec.participants:
        name = (p or "").strip()
        if not name:
            continue
        if name in seen:
            repairs.append(f"dropped duplicate participant '{name}'")
            continue
        seen.add(name)
        participants.append(name)
    if len(participants) > max_nodes:
        repairs.append(f"truncated participants from {len(participants)} to {max_nodes}")
        participants = participants[:max_nodes]
    valid = set(participants)

    max_messages = max_nodes * 3
    messages: list[SeqMessage] = []
    for m in spec.messages:
        text = (m.text or "").strip()
        if not text:
            continue
        if m.from_ not in valid or m.to not in valid:
            repairs.append(f"dropped message referencing an unknown participant ({m.from_} -> {m.to})")
            continue
        messages.append(m)
    if len(messages) > max_messages:
        repairs.append(f"truncated messages to {max_messages}")
        messages = messages[:max_messages]

    if len(participants) < 2 or len(messages) < 1:
        return None, repairs
    return spec.model_copy(update={"participants": participants, "messages": messages}), repairs


def _heal_mindmap(spec: MindmapSpec, max_nodes: int):
    repairs: list[str] = []
    root = (spec.root or "").strip()
    if not root:
        repairs.append("root label was empty -- cannot salvage a mindmap with no root")
        return None, repairs

    branches: list[MindBranch] = []
    total = 1  # the root itself counts toward max_nodes
    for b in spec.branches:
        label = (b.label or "").strip()
        if not label:
            repairs.append("dropped a branch with an empty label")
            continue
        if total >= max_nodes:
            repairs.append(f"truncated branches at max_nodes={max_nodes}")
            break
        total += 1
        children: list[str] = []
        for c in b.children:
            child = (c or "").strip()
            if not child:
                continue
            if total >= max_nodes:
                repairs.append(f"truncated mindmap children at max_nodes={max_nodes}")
                break
            total += 1
            children.append(child)
        branches.append(MindBranch(label=label, children=children))

    if not branches:
        return None, repairs
    return spec.model_copy(update={"root": root, "branches": branches}), repairs


def _heal_pie(spec: PieSpec, max_nodes: int):
    repairs: list[str] = []
    slices: list[PieSlice] = []
    for s in spec.slices:
        label = (s.label or "").strip()
        if not label or s.value is None or s.value < 0:
            repairs.append(f"dropped invalid pie slice '{s.label}'")
            continue
        slices.append(s)
    if len(slices) > max_nodes:
        repairs.append(f"truncated slices from {len(slices)} to {max_nodes}")
        slices = slices[:max_nodes]
    if len(slices) < 2:
        return None, repairs
    return spec.model_copy(update={"slices": slices}), repairs


def _heal_xy(spec: XySpec, max_nodes: int):
    repairs: list[str] = []
    x_labels = list(spec.x_labels)
    values = list(spec.values)
    if len(x_labels) != len(values):
        repairs.append(
            f"truncated mismatched x_labels/values lengths ({len(x_labels)} vs {len(values)})"
        )
        n = min(len(x_labels), len(values))
        x_labels, values = x_labels[:n], values[:n]

    cleaned_labels: list[str] = []
    cleaned_values: list[float] = []
    for x, v in zip(x_labels, values):
        if not (x or "").strip():
            repairs.append("dropped a point with an empty x-axis label")
            continue
        cleaned_labels.append(x)
        cleaned_values.append(v)

    if len(cleaned_labels) > max_nodes:
        repairs.append(f"truncated points from {len(cleaned_labels)} to {max_nodes}")
        cleaned_labels = cleaned_labels[:max_nodes]
        cleaned_values = cleaned_values[:max_nodes]

    if len(cleaned_labels) < 2:
        return None, repairs
    return spec.model_copy(update={"x_labels": cleaned_labels, "values": cleaned_values}), repairs


def _heal_candlestick(spec: CandlestickSpec, max_nodes: int):
    repairs: list[str] = []
    candles: list[CandleItem] = []
    for c in spec.candles:
        lo, hi, op, cl = c.low, c.high, c.open, c.close
        if not (lo <= min(op, cl) <= max(op, cl) <= hi):
            repairs.append(f"dropped OHLC-incoherent candle '{c.label}'")
            continue
        candles.append(c)
    if len(candles) > max_nodes:
        repairs.append(f"truncated candles from {len(candles)} to {max_nodes}")
        candles = candles[:max_nodes]
    if len(candles) < 2:
        return None, repairs
    return spec.model_copy(update={"candles": candles}), repairs


_HEAL = {
    "flowchart": _heal_flowchart,
    "sequence": _heal_sequence,
    "mindmap": _heal_mindmap,
    "pie": _heal_pie,
    "xy": _heal_xy,
    "candlestick": _heal_candlestick,
}


# ── Tier B: gate (reject -- only what can't be salvaged without inventing) ──

# Mirrors generate_training_data.py:2489's _CJK exactly (same ranges, same
# reasoning: these characters are never legitimate in Darija, French, or
# Arabic output). Not imported directly -- that name is module-private
# there, so this is a deliberate, documented duplicate, the same choice
# quiz.py's _QUESTION_SCHEMA makes against QUIZ_CONTENT_SCHEMA.
_CJK = re.compile(r"[　-鿿가-힯]")


def _structural_labels(kind: str, spec) -> list[str]:
    """Every diagram-internal label that must be in `settings.
    diagram_label_language` -- everything EXCEPT `caption`, which follows
    the turn's own response language instead (see DiagramPayload.caption's
    docstring in app/models/schemas.py)."""
    if kind == "flowchart":
        labels = [spec.title] + [n.label for n in spec.nodes]
        labels += [e.label for e in spec.edges if e.label]
        return labels
    if kind == "sequence":
        return [spec.title] + list(spec.participants) + [m.text for m in spec.messages]
    if kind == "mindmap":
        labels = [spec.title, spec.root]
        for b in spec.branches:
            labels.append(b.label)
            labels.extend(b.children)
        return labels
    if kind == "pie":
        return [spec.title] + [s.label for s in spec.slices]
    if kind == "xy":
        labels = [spec.title]
        if spec.y_axis_label:
            labels.append(spec.y_axis_label)
        labels.extend(spec.x_labels)
        return labels
    if kind == "candlestick":
        return [spec.title] + [c.label for c in spec.candles if c.label]
    return []


def _labels_pass_language_gate(labels: list[str], target_language: str) -> bool:
    for text in labels:
        if not text:
            continue
        if _CJK.search(text):
            return False
        if target_language == "fr" and has_arabic_script(text):
            return False
    return True


def _ungrounded_labels(labels: list[str], context: str) -> list[str]:
    """Reuses the exact detector grounding.py wraps for quiz questions --
    same standard, same reasoning (docstring at the top of grounding.py):
    serving and the training-data pipeline stay held to one definition of
    "fabricated reference" rather than two independently-drifting ones."""
    row = {
        "component": "diagram_generation",
        "messages": [{"role": "assistant", "content": " ".join(labels)}],
    }
    return row_has_ungrounded_reference(row, context)


# ── Orchestration ─────────────────────────────────────────────────────────


def _build_user_turn(
    kind: str, message: str, language: str, rejection_reason: Optional[str]
) -> str:
    turn = (message or "").strip()
    hint = diagram_kind_hint(kind, language)
    if hint:
        turn = f"{turn}\n\n{hint}"
    if rejection_reason:
        # A correction note, not a natural user question -- kept in English
        # for the non-French path, matching build_diagram_prompt's own
        # English-meta-instruction register for that branch (see llm.py).
        retry_note = (
            f"\n\nTa reponse precedente etait invalide ({rejection_reason}). "
            "Corrige-la et respecte strictement le schema JSON demande."
            if language == "fr"
            else (
                f"\n\nThe previous JSON was invalid ({rejection_reason}). "
                "Fix it and strictly follow the required JSON schema."
            )
        )
        turn += retry_note
    return turn


def generate_diagram(
    *,
    message: str,
    intent: DiagramIntent,
    domain: str,
    context: str,
    language: str,
    sources: list[str],
    caller_candles: Optional[list[Candle]] = None,
) -> Optional[DiagramPayload]:
    """Generate, heal, and gate a diagram, retrying once on any failure.
    Returns None (never raises for a content failure) when nothing
    salvageable came back after the retry -- the caller
    (app/routers/chat.py) then falls back to a normal chat turn with no
    diagram, exactly the fallback _archive/docs_superseded/STRATEGY_V2.md's
    archived design specified, reached deterministically.

    Raises AppError (OllamaConnectionError, GenerationError) exactly as
    generate_llm_response and generate_quiz_questions do -- a connectivity
    failure is not a content failure and must not be swallowed into a
    silent None; the caller is expected to catch AppError the same way
    chat.py and quiz.py already do.
    """
    settings = get_settings()
    if not settings.diagrams_enabled:
        return None

    kind = intent.kind
    schema = _SCHEMAS[kind]
    spec_model = _SPEC_MODELS[kind]
    heal = _HEAL[kind]
    model = settings.ollama_model_fr if language == "fr" else settings.ollama_model

    rejection_reason: Optional[str] = None
    for attempt in range(2):
        system = build_diagram_prompt(kind, domain, context, language)
        user_prompt = _build_user_turn(kind, message, language, rejection_reason)
        raw = _call_ollama_generate(model, user_prompt, system, format_schema=schema)

        try:
            payload_obj = json.loads(raw)
            spec = spec_model.model_validate(payload_obj)
        except (json.JSONDecodeError, ValidationError) as e:
            rejection_reason = "invalid JSON/schema"
            logger.warning(
                "diagram: attempt %d invalid output for kind=%s: %s", attempt, kind, e
            )
            continue

        if kind == "candlestick" and caller_candles:
            spec = spec.model_copy(
                update={
                    "candles": [
                        CandleItem(label=c.label, open=c.open, high=c.high, low=c.low, close=c.close)
                        for c in caller_candles
                    ]
                }
            )

        healed, repairs = heal(spec, settings.diagram_max_nodes)
        if healed is None:
            rejection_reason = "salvage floor not met: " + "; ".join(repairs[:3])
            logger.info(
                "diagram: attempt %d rejected kind=%s reason=%s", attempt, kind, rejection_reason
            )
            continue

        labels = _structural_labels(kind, healed)
        if not _labels_pass_language_gate(labels, settings.diagram_label_language):
            rejection_reason = "structural labels not in the target language"
            logger.info("diagram: attempt %d language gate failed kind=%s", attempt, kind)
            continue

        grounded = bool(context.strip())
        if grounded:
            offenders = _ungrounded_labels(labels, context)
            if offenders:
                rejection_reason = "ungrounded reference: " + ", ".join(offenders[:3])
                logger.info(
                    "diagram: attempt %d ungrounded kind=%s offenders=%s", attempt, kind, offenders
                )
                continue

        if kind == "candlestick":
            mermaid = None
            spec_dict = {"candles": [c.model_dump() for c in healed.candles]}
        else:
            mermaid = diagram_render.render(kind, healed)
            spec_dict = {}

        if repairs:
            logger.info("diagram: kind=%s healed with repairs=%s", kind, repairs)

        return DiagramPayload(
            kind=kind,
            kind_source=intent.source,
            title=healed.title,
            caption=healed.caption,
            mermaid=mermaid,
            spec=spec_dict,
            grounded=grounded,
            repairs=repairs,
            sources=sources if grounded else [],
        )

    logger.warning(
        "diagram generation gave up after retry: kind=%s reason=%s", kind, rejection_reason
    )
    return None
