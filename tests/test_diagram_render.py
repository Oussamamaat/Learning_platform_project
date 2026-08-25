"""
Tests for app/services/diagram_render.py -- pure, unmocked (tests/test_routing.py's
style). Covers French-label escaping (the fix for the two failure modes
_archive/docs_superseded/STRATEGY_V2.md:124-142 measured against the raw
fine-tune: unquoted labels and malformed edge-label syntax), id reindexing,
and per-kind Mermaid syntax shape. Written alongside the diagram-generation
feature (2026-08-23) -- see the plan doc for the full architecture.
"""

from types import SimpleNamespace

from app.services.diagram_render import (
    mermaid_label,
    _sequence_label,
    _sequence_message_text,
    render_flowchart,
    render_sequence,
    render_mindmap,
    render_pie,
    render_xy,
    render,
)


# --- mermaid_label escaping -----------------------------------------------


def test_label_escapes_quotes():
    assert mermaid_label('Sécurité (niveau 2) : "urgent"') == 'Sécurité (niveau 2) : &quot;urgent&quot;'


def test_label_escapes_hash():
    assert mermaid_label("Coût > 50 % (#1 priorité)") == "Coût > 50 % (&#35;1 priorité)"


def test_label_preserves_apostrophe():
    # Apostrophes are not a Mermaid syntax hazard -- only '"' and '#' are.
    assert mermaid_label("l'employé") == "l'employé"


def test_label_converts_newlines_to_br():
    assert mermaid_label("ligne 1\nligne 2") == "ligne 1<br/>ligne 2"


def test_label_strips_control_chars():
    assert mermaid_label("texte\x00\x07propre") == "textepropre"


def test_label_escapes_ampersand_before_quote_replacement():
    # '&' must be escaped first or the '&' introduced by '"' -> '&quot;'
    # would itself get mangled on a second pass.
    assert mermaid_label('A & "B"') == "A &amp; &quot;B&quot;"


def test_label_handles_empty_and_none():
    assert mermaid_label("") == ""
    assert mermaid_label(None) == ""


# --- flowchart --------------------------------------------------------------


def _flow_node(id, label):
    return SimpleNamespace(id=id, label=label)


def _flow_edge(source, target, label=None):
    return SimpleNamespace(source=source, target=target, label=label)


def test_flowchart_basic_shape():
    spec = SimpleNamespace(
        direction="TD",
        nodes=[_flow_node("a", "Identifier le risque"), _flow_node("b", "Évaluer")],
        edges=[_flow_edge("a", "b")],
    )
    out = render_flowchart(spec)
    assert out.splitlines()[0] == "flowchart TD"
    assert 'n0["Identifier le risque"]' in out
    assert 'n1["Évaluer"]' in out
    assert "n0 --> n1" in out


def test_flowchart_edge_label_is_quoted_pipe_syntax():
    spec = SimpleNamespace(
        direction="LR",
        nodes=[_flow_node("a", "D"), _flow_node("b", "E")],
        edges=[_flow_edge("a", "b", "آلات متحركة")],
    )
    out = render_flowchart(spec)
    # Retires archived failure mode #2 (STRATEGY_V2.md:133-135): the model
    # emitted `D -- [آلات متحركة] --> E`; correct Mermaid is the piped form.
    assert 'n0 -->|"آلات متحركة"| n1' in out


def test_flowchart_reindexes_hostile_ids():
    spec = SimpleNamespace(
        direction="TD",
        nodes=[_flow_node('weird id "with quotes"', "Label")],
        edges=[],
    )
    out = render_flowchart(spec)
    assert "weird id" not in out
    assert "n0" in out


def test_flowchart_drops_dangling_edge_defensively():
    spec = SimpleNamespace(
        direction="TD",
        nodes=[_flow_node("a", "A")],
        edges=[_flow_edge("a", "does-not-exist")],
    )
    out = render_flowchart(spec)
    assert "-->" not in out


def test_flowchart_invalid_direction_falls_back_to_td():
    spec = SimpleNamespace(direction="XX", nodes=[_flow_node("a", "A")], edges=[])
    assert render_flowchart(spec).splitlines()[0] == "flowchart TD"


# --- sequence -----------------------------------------------------------


def _seq_msg(from_, to, text):
    return SimpleNamespace(from_=from_, to=to, text=text)


def test_sequence_basic_shape():
    spec = SimpleNamespace(
        participants=["Alice", "Bob"],
        messages=[_seq_msg("Alice", "Bob", "Bonjour")],
    )
    out = render_sequence(spec)
    assert out.splitlines()[0] == "sequenceDiagram"
    # Always quoted -- sequenceDiagram participant labels need quoting for
    # anything but a bare identifier, and the real-parser gate (tests/
    # test_diagram_mermaid_parses.py) is what caught the original unquoted
    # form silently breaking on '&' in real content.
    assert 'participant n0 as "Alice"' in out
    assert 'participant n1 as "Bob"' in out
    assert "n0->>n1: Bonjour" in out


def test_sequence_message_colon_is_stripped():
    spec = SimpleNamespace(
        participants=["A", "B"],
        messages=[_seq_msg("A", "B", "Ratio: 50%")],
    )
    out = render_sequence(spec)
    # ':' is the mermaid message separator -- a stray colon in the text
    # must not survive into the message body.
    line = [l for l in out.splitlines() if l.strip().startswith("n0->>n1")][0]
    assert line.count(":") == 1


def test_sequence_drops_message_with_unknown_participant():
    spec = SimpleNamespace(participants=["A"], messages=[_seq_msg("A", "Ghost", "hi")])
    out = render_sequence(spec)
    assert "->>" not in out


def test_sequence_label_leaves_ampersand_and_hash_raw():
    # Discovered live by the real-parser gate (tests/test_diagram_mermaid_
    # parses.py): sequenceDiagram's grammar is NOT the HTML-entity-
    # sanitized text flowchart/pie/xy/mindmap labels go through --
    # mermaid.parse() rejects "&amp;"/"&quot;" here, where mermaid_label's
    # escaping would produce exactly that and break the parse.
    assert _sequence_label("Securite & Prevention #1") == "Securite & Prevention #1"


def test_sequence_label_strips_embedded_quote():
    # No working escape sequence for '"' exists inside a quoted
    # sequenceDiagram participant label -- stripped, not escaped.
    assert '"' not in _sequence_label('Le "Chef" de poste')


def test_sequence_message_text_leaves_ampersand_and_hash_raw():
    assert _sequence_message_text("Ratio 50% & plus, item #1") == "Ratio 50% & plus, item #1"


def test_sequence_message_text_strips_colon():
    assert ":" not in _sequence_message_text("Ratio: 50%")


# --- mindmap --------------------------------------------------------------


def _branch(label, children):
    return SimpleNamespace(label=label, children=children)


def test_mindmap_basic_shape_and_unique_ids():
    spec = SimpleNamespace(
        root="Sécurité",
        branches=[_branch("EPI", ["Casque", "Gants"])],
    )
    out = render_mindmap(spec)
    lines = out.splitlines()
    assert lines[0] == "mindmap"
    ids = [l.split("(")[0].strip() for l in lines[1:]]
    assert len(ids) == len(set(ids))  # every node has a unique id token


def test_mindmap_indentation_nests_children_deeper_than_branches():
    spec = SimpleNamespace(root="R", branches=[_branch("B", ["C"])])
    lines = render_mindmap(spec).splitlines()
    root_indent = len(lines[1]) - len(lines[1].lstrip(" "))
    branch_indent = len(lines[2]) - len(lines[2].lstrip(" "))
    child_indent = len(lines[3]) - len(lines[3].lstrip(" "))
    assert root_indent < branch_indent < child_indent


# --- pie --------------------------------------------------------------------


def _slice(label, value):
    return SimpleNamespace(label=label, value=value)


def test_pie_basic_shape():
    spec = SimpleNamespace(title="Répartition", slices=[_slice("A", 60), _slice("B", 40)])
    out = render_pie(spec)
    assert out.splitlines()[0] == "pie title Répartition"
    assert '"A" : 60' in out
    assert '"B" : 40' in out


def test_pie_title_is_escaped_but_not_quoted():
    spec = SimpleNamespace(title='Titre "spécial" #1', slices=[_slice("A", 1)])
    out = render_pie(spec)
    first_line = out.splitlines()[0]
    assert first_line == 'pie title Titre &quot;spécial&quot; &#35;1'


def test_pie_value_formatting_drops_trailing_zero():
    spec = SimpleNamespace(title="T", slices=[_slice("A", 33.333)])
    out = render_pie(spec)
    assert '"A" : 33.33' in out


# --- xy -----------------------------------------------------------------


def test_xy_basic_shape():
    spec = SimpleNamespace(
        title="Ventes",
        x_labels=["Jan", "Fév"],
        y_axis_label="Unités",
        chart_type="bar",
        values=[10, 20],
    )
    out = render_xy(spec)
    assert out.splitlines()[0] == "xychart-beta"
    assert 'title "Ventes"' in out
    assert 'x-axis ["Jan", "Fév"]' in out
    assert 'y-axis "Unités"' in out
    assert "bar [10, 20]" in out


def test_xy_invalid_chart_type_falls_back_to_bar():
    spec = SimpleNamespace(
        title="T", x_labels=["A"], y_axis_label="", chart_type="scatter", values=[1]
    )
    out = render_xy(spec)
    assert "bar [1]" in out


# --- dispatch -----------------------------------------------------------


def test_render_dispatches_by_kind():
    spec = SimpleNamespace(title="T", slices=[_slice("A", 1)])
    assert render("pie", spec).startswith("pie title")


def test_render_raises_for_candlestick():
    import pytest

    with pytest.raises(KeyError):
        render("candlestick", SimpleNamespace())
