"""
Mermaid Diagram Renderer
─────────────────────────
Deterministic spec -> Mermaid string. No LLM call, no DB, no settings read --
pure functions, so this module is testable without mocks (tests/test_routing.py's
style, not tests/test_quiz.py's).

Why this exists at all rather than letting the model write Mermaid directly:
_archive/docs_superseded/STRATEGY_V2.md:124-142 measured the fine-tune's
Mermaid emission and found it reliable in substance (6/6) but syntactically
broken in two specific, recurring ways -- unquoted Arabic/French labels
(`A[Coût élevé]` instead of `A["Coût élevé"]`) and malformed edge labels
(`D -- [x] --> E` instead of `D -->|"x"| E`). Both failure modes are
syntax, not semantics, so this module retires them structurally: the model
only ever emits a JSON spec (validated in app/services/diagrams.py), and
this module is the single place that knows how to write valid Mermaid.

Callers pass an ALREADY-HEALED spec (app.services.diagrams's heal tier has
already dropped dangling edges, deduped ids, and enforced the salvage
floor). The id-reindexing here is a second, independent safety net -- even
a spec this module didn't validate can't inject malformed Mermaid syntax
via a hostile id string; it just can't reference a node that doesn't
exist (dangling references are silently skipped, defensively, even though
the heal tier upstream should have already removed them).
"""

import re

# Control characters (excluding \t\n\r, handled separately below) -- stripped
# so a hostile or garbled label can't break out of a Mermaid line via an
# embedded control byte.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def mermaid_label(text: str) -> str:
    """Escape text for safe embedding inside a QUOTED Mermaid label.

    Mermaid reads a bare '#' as the start of an HTML entity reference and a
    bare '"' as the end of the label -- both routinely appear in French
    text ("Coût > 50 %", "l'employé", "Sécurité (niveau 2)"). This function
    only escapes; it deliberately does NOT add the surrounding quotes --
    every call site wraps the result in `"..."` itself, because the quoting
    syntax differs slightly per Mermaid construct (node shape, edge label,
    pie slice, chart title).
    """
    text = _CONTROL_CHARS.sub("", text or "").strip()
    # '&' must be escaped first, or the '&' introduced by the '"'/'#'
    # replacements below would itself get escaped on a second pass.
    text = text.replace("&", "&amp;")
    text = text.replace('"', "&quot;")
    text = text.replace("#", "&#35;")
    text = text.replace("\r\n", "<br/>").replace("\n", "<br/>").replace("\r", "<br/>")
    return text


def _format_number(n: float) -> str:
    """Render a number the way a human would type it in Mermaid source --
    '42' not '42.0', but '3.14' kept as-is."""
    try:
        f = float(n)
    except (TypeError, ValueError):
        return "0"
    if f.is_integer():
        return str(int(f))
    return f"{f:.2f}".rstrip("0").rstrip(".")


def _index_ids(ids: list[str]) -> dict[str, str]:
    """Map arbitrary/untrusted model-supplied ids to safe sequential Mermaid
    ids (n0, n1, ...). First occurrence of a given original id wins; later
    duplicates map to the same safe id. Mermaid ids may not contain spaces
    or most punctuation, so re-issuing them entirely -- rather than trying
    to sanitize the original string -- is the simplest way to guarantee
    every emitted id is valid regardless of what the model wrote.
    """
    mapping: dict[str, str] = {}
    for original in ids:
        key = str(original)
        if key not in mapping:
            mapping[key] = f"n{len(mapping)}"
    return mapping


def render_flowchart(spec) -> str:
    """spec: app.services.diagrams.FlowchartSpec (title, caption, direction,
    nodes: list[FlowNode(id, label)], edges: list[FlowEdge(source, target, label)])."""
    direction = spec.direction if spec.direction in ("TD", "LR") else "TD"
    id_map = _index_ids([n.id for n in spec.nodes])

    lines = [f"flowchart {direction}"]
    for n in spec.nodes:
        lines.append(f'    {id_map[n.id]}["{mermaid_label(n.label)}"]')
    for e in spec.edges:
        src = id_map.get(e.source)
        tgt = id_map.get(e.target)
        if src is None or tgt is None:
            # Defensive only -- the heal tier in diagrams.py should already
            # have dropped any edge referencing a missing node.
            continue
        if e.label:
            lines.append(f'    {src} -->|"{mermaid_label(e.label)}"| {tgt}')
        else:
            lines.append(f"    {src} --> {tgt}")
    return "\n".join(lines)


def _sequence_label(text: str) -> str:
    """Escaping for a QUOTED sequenceDiagram participant label -- NOT
    mermaid_label. sequenceDiagram's participant/message grammar is not the
    HTML-entity-sanitized text flowchart/pie/xy/mindmap labels go through
    (mermaid.parse() rejects "&amp;"/"&quot;" here, confirmed live against
    the real parser -- see tests/test_diagram_mermaid_parses.py). '&' and
    '#' need no escaping and must be left raw. A literal '"' cannot be
    embedded in a quoted participant label at all in this grammar -- there
    is no working escape sequence for it -- so it is stripped rather than
    escaped, and newlines collapse to spaces since even a quoted
    declaration is one line.
    """
    text = _CONTROL_CHARS.sub("", text or "").strip()
    text = text.replace('"', "").replace("\n", " ").replace("\r", " ")
    return text


def _sequence_message_text(text: str) -> str:
    """Sanitizing for sequenceDiagram message text, which runs RAW to end
    of line -- no surrounding quotes, no HTML-entity escaping (confirmed
    live: escaping actively breaks it here, unlike every other kind this
    module renders). Only control characters, newlines, and ':' (the
    message-text separator) need handling."""
    text = _CONTROL_CHARS.sub("", text or "").strip()
    text = text.replace("\n", " ").replace("\r", " ").replace(":", " -")
    return text


def render_sequence(spec) -> str:
    """spec: app.services.diagrams.SequenceSpec (title, caption,
    participants: list[str], messages: list[SeqMessage(from_, to, text)]).

    Uses _sequence_label/_sequence_message_text, NOT mermaid_label --
    confirmed live against the real mermaid parser (tests/test_diagram_
    mermaid_parses.py caught this during development) that sequenceDiagram's
    grammar is not the HTML-entity-sanitized text flowchart/pie/xy/mindmap
    labels go through: `&amp;`/`&quot;` are parse errors here, where they
    are required elsewhere. See those functions' docstrings.
    """
    id_map = _index_ids(spec.participants)

    lines = ["sequenceDiagram"]
    for p in spec.participants:
        lines.append(f'    participant {id_map[p]} as "{_sequence_label(p)}"')
    for m in spec.messages:
        src = id_map.get(m.from_)
        tgt = id_map.get(m.to)
        if src is None or tgt is None:
            continue
        lines.append(f"    {src}->>{tgt}: {_sequence_message_text(m.text)}")
    return "\n".join(lines)


def render_mindmap(spec) -> str:
    """spec: app.services.diagrams.MindmapSpec (title, caption, root: str,
    branches: list[MindBranch(label, children: list[str])]).

    Mermaid mindmap syntax is whitespace/indentation-sensitive rather than
    edge-based, and each node needs its own unique leading id token before
    the shape delimiter -- unlike flowchart/sequence there is nothing to
    cross-reference, so a simple running counter is enough (no id_map
    needed; nothing points back at these ids).
    """
    counter = 0

    def next_id() -> str:
        nonlocal counter
        counter += 1
        return f"m{counter}"

    lines = ["mindmap"]
    lines.append(f'  {next_id()}("{mermaid_label(spec.root)}")')
    for b in spec.branches:
        lines.append(f'    {next_id()}("{mermaid_label(b.label)}")')
        for c in b.children:
            lines.append(f'      {next_id()}("{mermaid_label(c)}")')
    return "\n".join(lines)


def render_pie(spec) -> str:
    """spec: app.services.diagrams.PieSpec (title, caption,
    slices: list[PieSlice(label, value)]).

    Mermaid's `pie title <text>` directive reads to end-of-line as raw text
    -- it does not support (and would literally display) surrounding
    quotes, unlike a slice label. So the title is escaped but never quoted;
    each slice label is escaped AND quoted, per Mermaid's own pie syntax.
    """
    lines = [f"pie title {mermaid_label(spec.title)}"]
    for s in spec.slices:
        lines.append(f'    "{mermaid_label(s.label)}" : {_format_number(s.value)}')
    return "\n".join(lines)


def render_xy(spec) -> str:
    """spec: app.services.diagrams.XySpec (title, caption,
    x_labels: list[str], y_axis_label: str, chart_type: "bar"|"line",
    values: list[float]).

    Assumes len(x_labels) == len(values) -- the heal tier in diagrams.py
    truncates both lists to the shorter length before this is ever called.
    """
    lines = ["xychart-beta"]
    lines.append(f'    title "{mermaid_label(spec.title)}"')
    x_labels = ", ".join(f'"{mermaid_label(x)}"' for x in spec.x_labels)
    lines.append(f"    x-axis [{x_labels}]")
    if spec.y_axis_label:
        lines.append(f'    y-axis "{mermaid_label(spec.y_axis_label)}"')
    chart_type = spec.chart_type if spec.chart_type in ("bar", "line") else "bar"
    values = ", ".join(_format_number(v) for v in spec.values)
    lines.append(f"    {chart_type} [{values}]")
    return "\n".join(lines)


RENDERERS = {
    "flowchart": render_flowchart,
    "sequence": render_sequence,
    "mindmap": render_mindmap,
    "pie": render_pie,
    "xy": render_xy,
    # "candlestick" is deliberately absent: Mermaid has no OHLC diagram
    # type, so candlesticks render client-side in React from the raw JSON
    # spec (app.models.schemas.DiagramPayload.spec) -- see the plan doc.
}


def render(kind: str, spec) -> str:
    """Dispatch to the renderer for `kind`. Raises KeyError for
    "candlestick" or any unknown kind -- callers must check RENDERERS (or
    kind != "candlestick") before calling this."""
    return RENDERERS[kind](spec)
