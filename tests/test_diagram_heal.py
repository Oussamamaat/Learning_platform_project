"""
Tests for app/services/diagrams.py's Tier A (heal/salvage) functions --
pure, unmocked, tests/test_routing.py's style. Written alongside the
diagram-generation feature (2026-08-23).

Core invariant under test throughout: healing only ever REMOVES (a
dangling edge, an empty label, an item past the node-count ceiling) and
records what it removed in `repairs` -- it never invents a phantom node or
any other content to satisfy a reference. See the plan doc / diagrams.py's
module docstring for why.
"""

from app.services.diagrams import (
    FlowchartSpec, FlowNode, FlowEdge,
    SequenceSpec, SeqMessage,
    MindmapSpec, MindBranch,
    PieSpec, PieSlice,
    XySpec,
    CandlestickSpec, CandleItem,
    _heal_flowchart, _heal_sequence, _heal_mindmap, _heal_pie, _heal_xy, _heal_candlestick,
)


# --- flowchart ------------------------------------------------------------


def test_flowchart_heal_drops_dangling_edge_and_records_repair():
    spec = FlowchartSpec(
        title="T", caption="Une legende suffisamment longue.",
        nodes=[FlowNode(id="a", label="A"), FlowNode(id="b", label="B")],
        edges=[FlowEdge(source="a", target="b"), FlowEdge(source="a", target="ghost")],
    )
    healed, repairs = _heal_flowchart(spec, max_nodes=14)
    assert healed is not None
    assert len(healed.edges) == 1
    assert any("dangling" in r for r in repairs)
    # No phantom node was synthesised to satisfy the dangling reference.
    assert {n.id for n in healed.nodes} == {"a", "b"}


def test_flowchart_heal_drops_self_loop():
    spec = FlowchartSpec(
        title="T", caption="Une legende suffisamment longue.",
        nodes=[FlowNode(id="a", label="A"), FlowNode(id="b", label="B")],
        edges=[FlowEdge(source="a", target="a"), FlowEdge(source="a", target="b")],
    )
    healed, repairs = _heal_flowchart(spec, max_nodes=14)
    assert healed is not None
    assert len(healed.edges) == 1
    assert any("self-loop" in r for r in repairs)


def test_flowchart_heal_dedupes_ids_keeping_first():
    spec = FlowchartSpec(
        title="T", caption="Une legende suffisamment longue.",
        nodes=[FlowNode(id="a", label="First"), FlowNode(id="a", label="Second"), FlowNode(id="b", label="B")],
        edges=[FlowEdge(source="a", target="b")],
    )
    healed, repairs = _heal_flowchart(spec, max_nodes=14)
    assert healed is not None
    assert len(healed.nodes) == 2
    assert healed.nodes[0].label == "First"
    assert any("duplicate" in r for r in repairs)


def test_flowchart_heal_drops_empty_label_node():
    spec = FlowchartSpec(
        title="T", caption="Une legende suffisamment longue.",
        nodes=[FlowNode(id="a", label="   "), FlowNode(id="b", label="B"), FlowNode(id="c", label="C")],
        edges=[FlowEdge(source="b", target="c")],
    )
    healed, repairs = _heal_flowchart(spec, max_nodes=14)
    assert healed is not None
    assert "a" not in {n.id for n in healed.nodes}
    assert any("empty label" in r for r in repairs)


def test_flowchart_heal_truncates_to_max_nodes():
    nodes = [FlowNode(id=str(i), label=f"N{i}") for i in range(20)]
    edges = [FlowEdge(source=str(i), target=str(i + 1)) for i in range(19)]
    spec = FlowchartSpec(title="T", caption="Legende suffisamment longue.", nodes=nodes, edges=edges)
    healed, repairs = _heal_flowchart(spec, max_nodes=5)
    assert healed is not None
    assert len(healed.nodes) == 5
    assert all(e.source in {"0", "1", "2", "3", "4"} for e in healed.edges)
    assert any("truncated" in r for r in repairs)


def test_flowchart_heal_rejects_below_salvage_floor():
    # A single node with no edges is not a salvageable flowchart.
    spec = FlowchartSpec(
        title="T", caption="Une legende suffisamment longue.",
        nodes=[FlowNode(id="a", label="A")],
        edges=[],
    )
    healed, repairs = _heal_flowchart(spec, max_nodes=14)
    assert healed is None


# --- sequence -------------------------------------------------------------


def test_sequence_heal_drops_message_with_unknown_participant():
    spec = SequenceSpec(
        title="T", caption="Une legende suffisamment longue.",
        participants=["Alice", "Bob"],
        messages=[SeqMessage(**{"from": "Alice", "to": "Bob", "text": "Bonjour"}),
                  SeqMessage(**{"from": "Alice", "to": "Ghost", "text": "hi"})],
    )
    healed, repairs = _heal_sequence(spec, max_nodes=14)
    assert healed is not None
    assert len(healed.messages) == 1
    assert any("unknown participant" in r for r in repairs)


def test_sequence_heal_rejects_below_salvage_floor():
    spec = SequenceSpec(
        title="T", caption="Une legende suffisamment longue.",
        participants=["Alice"], messages=[],
    )
    healed, repairs = _heal_sequence(spec, max_nodes=14)
    assert healed is None


# --- mindmap ----------------------------------------------------------------


def test_mindmap_heal_rejects_empty_root():
    spec = MindmapSpec(
        title="T", caption="Une legende suffisamment longue.",
        root="   ", branches=[MindBranch(label="B", children=["C"])],
    )
    healed, repairs = _heal_mindmap(spec, max_nodes=14)
    assert healed is None
    assert any("root" in r for r in repairs)


def test_mindmap_heal_drops_empty_branch_label():
    spec = MindmapSpec(
        title="T", caption="Une legende suffisamment longue.",
        root="Root",
        branches=[MindBranch(label="", children=[]), MindBranch(label="Valid", children=["Child"])],
    )
    healed, repairs = _heal_mindmap(spec, max_nodes=14)
    assert healed is not None
    assert len(healed.branches) == 1
    assert healed.branches[0].label == "Valid"


# --- pie ----------------------------------------------------------------


def test_pie_heal_drops_negative_value_slice():
    spec = PieSpec(
        title="T", caption="Une legende suffisamment longue.",
        slices=[PieSlice(label="A", value=-5), PieSlice(label="B", value=50), PieSlice(label="C", value=50)],
    )
    healed, repairs = _heal_pie(spec, max_nodes=14)
    assert healed is not None
    assert len(healed.slices) == 2
    assert any("invalid pie slice" in r for r in repairs)


def test_pie_heal_rejects_below_salvage_floor():
    spec = PieSpec(title="T", caption="Une legende suffisamment longue.", slices=[PieSlice(label="A", value=100)])
    healed, repairs = _heal_pie(spec, max_nodes=14)
    assert healed is None


# --- xy -------------------------------------------------------------------


def test_xy_heal_truncates_mismatched_lengths():
    spec = XySpec(
        title="T", caption="Une legende suffisamment longue.",
        x_labels=["Jan", "Fev", "Mar"], values=[10, 20],
    )
    healed, repairs = _heal_xy(spec, max_nodes=14)
    assert healed is not None
    assert len(healed.x_labels) == len(healed.values) == 2
    assert any("mismatched" in r for r in repairs)


def test_xy_heal_rejects_below_salvage_floor():
    spec = XySpec(title="T", caption="Une legende suffisamment longue.", x_labels=["Jan"], values=[10])
    healed, repairs = _heal_xy(spec, max_nodes=14)
    assert healed is None


# --- candlestick ------------------------------------------------------------


def test_candlestick_heal_drops_ohlc_incoherent_candle():
    spec = CandlestickSpec(
        title="T", caption="Une legende suffisamment longue.",
        candles=[
            CandleItem(label="Bon", open=10, high=12, low=8, close=11),
            # high < open -- not a coherent OHLC bar.
            CandleItem(label="Mauvais", open=10, high=9, low=8, close=11),
            CandleItem(label="Bon2", open=11, high=13, low=9, close=12),
        ],
    )
    healed, repairs = _heal_candlestick(spec, max_nodes=14)
    assert healed is not None
    assert len(healed.candles) == 2
    assert any("OHLC-incoherent" in r for r in repairs)


def test_candlestick_heal_accepts_close_equals_open():
    spec = CandlestickSpec(
        title="T", caption="Une legende suffisamment longue.",
        candles=[
            CandleItem(label="A", open=10, high=11, low=9, close=10),
            CandleItem(label="B", open=10, high=11, low=9, close=10),
        ],
    )
    healed, repairs = _heal_candlestick(spec, max_nodes=14)
    assert healed is not None
    assert len(healed.candles) == 2


def test_candlestick_heal_rejects_below_salvage_floor():
    spec = CandlestickSpec(
        title="T", caption="Une legende suffisamment longue.",
        candles=[CandleItem(label="Only", open=10, high=9, low=8, close=11)],  # incoherent, dropped
    )
    healed, repairs = _heal_candlestick(spec, max_nodes=14)
    assert healed is None
