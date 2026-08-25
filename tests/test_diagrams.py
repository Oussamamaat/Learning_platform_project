"""
Tests for app/services/diagrams.py's intent router and generate_diagram
orchestration. Ollama is faked with the same context-manager stub
tests/test_quiz.py:33-48 uses, patched at the module that actually calls
urllib (app.services.llm, since app.services.diagrams calls
llm._call_ollama_generate rather than rolling its own urllib request --
see llm.py:449). Written alongside the diagram-generation feature
(2026-08-23).
"""

import json
from unittest.mock import patch

import pytest

from app.models.schemas import Candle
from app.services.diagrams import DiagramIntent, detect_diagram_intent, generate_diagram


class _FakeResponse:
    def __init__(self, body):
        self._body = body

    def read(self):
        return json.dumps(self._body).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _fake_ollama(spec: dict):
    return _FakeResponse({"response": json.dumps(spec)})


def _fails_if_called(*args, **kwargs):
    raise AssertionError("Ollama must not be called")


_FLOWCHART_SPEC = {
    "title": "Consignation",
    "caption": "Les etapes de consignation avant intervention.",
    "direction": "TD",
    "nodes": [
        {"id": "a", "label": "Couper l'alimentation"},
        {"id": "b", "label": "Verrouiller"},
    ],
    "edges": [{"source": "a", "target": "b", "label": "puis"}],
}

_CONTEXT = (
    "Article 12 : il faut couper l'alimentation puis verrouiller la machine "
    "avant toute intervention de maintenance."
)


# --- intent detection (Tier 1) -------------------------------------------
# --- False-positive guard: a bare topic noun must never trigger. ---------


@pytest.mark.parametrize("message", [
    "Explique-moi le processus de consignation",
    "Quelle est la procedure de securite pour les EPI ?",
    "C'est quoi un role fonctionnel dans une entreprise ?",
    "Peux-tu m'aider avec la maintenance de la machine ?",
])
def test_no_diagram_keyword_does_not_trigger(message):
    assert detect_diagram_intent(message) is None


@pytest.mark.parametrize("message,expected_kind", [
    ("Dessine-moi un schema des etapes de consignation", "flowchart"),
    ("Montre-moi une bougie en marteau haussier", "candlestick"),
    ("Fais-moi un camembert de la repartition des incidents", "pie"),
    ("Genere une carte mentale des risques", "mindmap"),
    ("Trace un histogramme des accidents par mois", "xy"),
    ("Produis un organigramme du processus", "flowchart"),
])
def test_keyword_triggers_expected_kind(message, expected_kind):
    intent = detect_diagram_intent(message)
    assert intent is not None
    assert intent.kind == expected_kind
    assert intent.source == "keyword"


def test_flowchart_fallback_has_lower_confidence_than_specific_keyword():
    generic = detect_diagram_intent("Dessine-moi un schema")
    specific = detect_diagram_intent("Montre-moi une bougie")
    assert generic.confidence < specific.confidence


# --- generate_diagram: happy path -----------------------------------------


def test_generate_diagram_returns_grounded_payload():
    with patch("app.services.llm.urllib.request.urlopen", return_value=_fake_ollama(_FLOWCHART_SPEC)):
        result = generate_diagram(
            message="Dessine-moi un schema de consignation",
            intent=DiagramIntent(kind="flowchart", source="keyword", confidence=0.6),
            domain="industrial",
            context=_CONTEXT,
            language="fr",
            sources=["doc1.pdf"],
        )
    assert result is not None
    assert result.kind == "flowchart"
    assert result.grounded is True
    assert result.sources == ["doc1.pdf"]
    assert "flowchart TD" in result.mermaid
    assert result.repairs == []


def test_generate_diagram_ungrounded_when_context_empty():
    with patch("app.services.llm.urllib.request.urlopen", return_value=_fake_ollama(_FLOWCHART_SPEC)):
        result = generate_diagram(
            message="Dessine-moi un schema",
            intent=DiagramIntent(kind="flowchart", source="keyword", confidence=0.6),
            domain="industrial",
            context="",
            language="fr",
            sources=["doc1.pdf"],
        )
    # No refusal for an ungrounded diagram -- it's still produced, just flagged.
    assert result is not None
    assert result.grounded is False
    assert result.sources == []


# --- generate_diagram: disabled kill-switch --------------------------------


def test_generate_diagram_returns_none_and_never_calls_ollama_when_disabled():
    from app.config import get_settings

    class _Disabled:
        diagrams_enabled = False

    with patch("app.services.diagrams.get_settings", return_value=_Disabled()), \
         patch("app.services.llm.urllib.request.urlopen", side_effect=_fails_if_called):
        result = generate_diagram(
            message="Dessine-moi un schema",
            intent=DiagramIntent(kind="flowchart", source="keyword", confidence=0.6),
            domain="industrial",
            context=_CONTEXT,
            language="fr",
            sources=[],
        )
    assert result is None


# --- generate_diagram: language gate ---------------------------------------


def test_generate_diagram_rejects_arabic_structural_labels_after_retry():
    spec_with_arabic_labels = {
        "title": "Consignation",
        "caption": "شرح بالدارجة",
        "direction": "TD",
        "nodes": [{"id": "a", "label": "قطع التيار"}, {"id": "b", "label": "قفل"}],
        "edges": [{"source": "a", "target": "b"}],
    }
    with patch(
        "app.services.llm.urllib.request.urlopen",
        return_value=_fake_ollama(spec_with_arabic_labels),
    ):
        result = generate_diagram(
            message="Dessine-moi un schema",
            intent=DiagramIntent(kind="flowchart", source="keyword", confidence=0.6),
            domain="industrial",
            context=_CONTEXT,
            language="fr",
            sources=["doc1.pdf"],
        )
    assert result is None


# --- generate_diagram: grounding gate + retry -----------------------------


def test_generate_diagram_rejects_ungrounded_reference_after_retry():
    spec_with_fabricated_ref = {
        "title": "T",
        "caption": "Legende suffisamment longue pour passer le schema.",
        "direction": "TD",
        "nodes": [
            {"id": "a", "label": "Selon Article 999 il faut faire X"},
            {"id": "b", "label": "Y"},
        ],
        "edges": [{"source": "a", "target": "b"}],
    }
    with patch(
        "app.services.llm.urllib.request.urlopen",
        return_value=_fake_ollama(spec_with_fabricated_ref),
    ):
        result = generate_diagram(
            message="Dessine-moi un schema",
            intent=DiagramIntent(kind="flowchart", source="keyword", confidence=0.6),
            domain="industrial",
            context="Le contexte ne mentionne aucun article precis.",
            language="fr",
            sources=["doc1.pdf"],
        )
    assert result is None


def test_generate_diagram_recovers_on_retry_after_first_attempt_ungrounded():
    bad = {
        "title": "T",
        "caption": "Legende suffisamment longue pour passer le schema.",
        "direction": "TD",
        "nodes": [{"id": "a", "label": "Article 999"}, {"id": "b", "label": "Y"}],
        "edges": [{"source": "a", "target": "b"}],
    }
    good = dict(_FLOWCHART_SPEC)
    with patch(
        "app.services.llm.urllib.request.urlopen",
        side_effect=[_fake_ollama(bad), _fake_ollama(good)],
    ):
        result = generate_diagram(
            message="Dessine-moi un schema",
            intent=DiagramIntent(kind="flowchart", source="keyword", confidence=0.6),
            domain="industrial",
            context=_CONTEXT,
            language="fr",
            sources=["doc1.pdf"],
        )
    assert result is not None
    assert result.grounded is True


# --- generate_diagram: caller-supplied candles bypass model invention -----


def test_generate_diagram_caller_candles_override_model_output():
    model_spec = {
        "title": "Marteau haussier",
        "caption": "Illustration d'un marteau haussier.",
        "candles": [
            {"label": "J1", "open": 10, "high": 12, "low": 8, "close": 11},
            {"label": "J2", "open": 11, "high": 13, "low": 9, "close": 12},
        ],
    }
    caller_candles = [
        Candle(label="Reel1", open=100, high=110, low=95, close=105),
        Candle(label="Reel2", open=105, high=115, low=100, close=112),
    ]
    with patch("app.services.llm.urllib.request.urlopen", return_value=_fake_ollama(model_spec)):
        result = generate_diagram(
            message="Montre-moi une bougie",
            intent=DiagramIntent(kind="candlestick", source="keyword", confidence=1.0),
            domain="industrial",
            context="",
            language="fr",
            sources=[],
            caller_candles=caller_candles,
        )
    assert result is not None
    assert result.mermaid is None
    labels = [c["label"] for c in result.spec["candles"]]
    assert labels == ["Reel1", "Reel2"]
