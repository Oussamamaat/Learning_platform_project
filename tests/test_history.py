"""
Tests for app/services/history.py.

Two layers:
  - _build_alternating_window is a pure function -- tested directly with
    fabricated rows, no DB.
  - load_window/append_exchange/pin_context/get_pinned are tested against
    an in-memory SQLite engine (monkeypatching history._get_engine), since
    ChatSession/ChatMessage now use portable JSON, not Postgres-only JSONB.
    This is real integration coverage of the (domain, language, segment_id)
    filtering, not a re-mock of SQLAlchemy's query builder. No live
    Postgres required, consistent with the rest of this suite.
"""
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine

from app.models.database import Base, ChatMessage, ChatSession
from app.services import history


def _row(role, content):
    return SimpleNamespace(role=role, content=content)


# --- _build_alternating_window (pure function) ------------------------------

def test_well_formed_window_passes_through():
    rows = [_row("user", "a"), _row("assistant", "b"), _row("user", "c"), _row("assistant", "d")]
    out = history._build_alternating_window(rows, max_messages=4, max_chars=10_000)
    assert [r["role"] for r in out] == ["user", "assistant", "user", "assistant"]


def test_trailing_dangling_user_is_dropped():
    rows = [_row("user", "a"), _row("assistant", "b"), _row("user", "c")]
    out = history._build_alternating_window(rows, max_messages=4, max_chars=10_000)
    assert out == [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]


def test_leading_dangling_assistant_is_dropped():
    rows = [_row("assistant", "stray"), _row("user", "a"), _row("assistant", "b")]
    out = history._build_alternating_window(rows, max_messages=4, max_chars=10_000)
    assert out == [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]


def test_eviction_keeps_most_recent_starting_on_user():
    rows = [
        _row("user", "u1"), _row("assistant", "a1"),
        _row("user", "u2"), _row("assistant", "a2"),
        _row("user", "u3"), _row("assistant", "a3"),
    ]
    out = history._build_alternating_window(rows, max_messages=4, max_chars=10_000)
    assert [r["content"] for r in out] == ["u2", "a2", "u3", "a3"]


def test_char_budget_evicts_oldest_pair_first():
    rows = [
        _row("user", "x" * 50), _row("assistant", "y" * 50),
        _row("user", "z" * 50), _row("assistant", "w" * 50),
    ]
    out = history._build_alternating_window(rows, max_messages=10, max_chars=150)
    assert [r["content"][0] for r in out] == ["z", "w"]


def test_empty_input_returns_empty_window():
    assert history._build_alternating_window([], max_messages=4, max_chars=1000) == []


# --- integration against in-memory SQLite -----------------------------------

@pytest.fixture
def sqlite_engine(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine, tables=[ChatSession.__table__, ChatMessage.__table__])
    monkeypatch.setattr(history, "_engine", engine)
    monkeypatch.setattr(history, "_get_engine", lambda: engine)
    return engine


def test_append_then_load_round_trips(sqlite_engine):
    history.append_exchange(
        "sess-1", tenant_id="t1", user_id="u1", domain="industrial", language="fr",
        segment_id=1, user_content="Bonjour", assistant_content="Salut !",
    )
    window = history.load_window("sess-1", domain="industrial", language="fr", segment_id=1)
    assert window == [
        {"role": "user", "content": "Bonjour"},
        {"role": "assistant", "content": "Salut !"},
    ]


def test_domain_language_filtering_excludes_other_context(sqlite_engine):
    history.append_exchange(
        "sess-1", tenant_id="t1", user_id="u1", domain="industrial", language="darija",
        segment_id=1, user_content="شنو", assistant_content="جواب",
    )
    # Same session, different domain and language -- must not see the darija turn.
    window = history.load_window("sess-1", domain="securite", language="fr", segment_id=1)
    assert window == []

    # And the original filter still finds it.
    window = history.load_window("sess-1", domain="industrial", language="darija", segment_id=1)
    assert len(window) == 2


def test_segment_filtering_excludes_closed_out_segment(sqlite_engine):
    history.append_exchange(
        "sess-1", tenant_id="t1", user_id="u1", domain="industrial", language="fr",
        segment_id=1, user_content="Q1", assistant_content="A1",
    )
    # New segment (topic switch) -- nothing there yet.
    window = history.load_window("sess-1", domain="industrial", language="fr", segment_id=2)
    assert window == []
    # Old segment still intact.
    window = history.load_window("sess-1", domain="industrial", language="fr", segment_id=1)
    assert len(window) == 2


def test_multiple_exchanges_evicted_to_window_size(sqlite_engine):
    for i in range(3):
        history.append_exchange(
            "sess-1", tenant_id="t1", user_id="u1", domain="industrial", language="fr",
            segment_id=1, user_content=f"Q{i}", assistant_content=f"A{i}",
        )
    window = history.load_window(
        "sess-1", domain="industrial", language="fr", segment_id=1, max_messages=4
    )
    assert [m["content"] for m in window] == ["Q1", "A1", "Q2", "A2"]


def test_pin_and_get_pinned_round_trip(sqlite_engine):
    history.pin_context(
        "sess-1", tenant_id="t1", user_id="u1", domain="industrial", language="fr",
        segment_id=1, context="Article 283...", sources=["1.1.md"], fingerprint="abc123",
    )
    pinned = history.get_pinned("sess-1")
    assert pinned["context"] == "Article 283..."
    assert pinned["sources"] == ["1.1.md"]
    assert pinned["fingerprint"] == "abc123"


def test_get_pinned_returns_none_for_unknown_session(sqlite_engine):
    assert history.get_pinned("nonexistent") is None


# --- extract_dropped_questions (Stage 1f: Socratic state, no background job) --

def test_dropped_questions_extracted_from_turns_outside_window(sqlite_engine):
    # 3 exchanges; window keeps the most recent 2 (MAX_WINDOW_MESSAGES=4 ->
    # 2 assistant turns kept), so exchange 1's question should be "dropped".
    history.append_exchange(
        "sess-1", tenant_id="t1", user_id="u1", domain="industrial", language="fr",
        segment_id=1, user_content="Q1",
        assistant_content="Voici le principe. Sais-tu pourquoi c'est important ?",
    )
    history.append_exchange(
        "sess-1", tenant_id="t1", user_id="u1", domain="industrial", language="fr",
        segment_id=1, user_content="Q2", assistant_content="Bonne question. Et ensuite ?",
    )
    history.append_exchange(
        "sess-1", tenant_id="t1", user_id="u1", domain="industrial", language="fr",
        segment_id=1, user_content="Q3", assistant_content="Exact, sans point d'interrogation ici.",
    )
    dropped = history.extract_dropped_questions(
        "sess-1", domain="industrial", language="fr", segment_id=1, kept_messages=4
    )
    assert dropped == ["Sais-tu pourquoi c'est important ?"]


def test_dropped_questions_handles_arabic_question_mark(sqlite_engine):
    history.append_exchange(
        "sess-1", tenant_id="t1", user_id="u1", domain="industrial", language="darija",
        segment_id=1, user_content="Q1", assistant_content="واش فهمتي؟ هاد شي مهم.",
    )
    history.append_exchange(
        "sess-1", tenant_id="t1", user_id="u1", domain="industrial", language="darija",
        segment_id=1, user_content="Q2", assistant_content="زوين.",
    )
    dropped = history.extract_dropped_questions(
        "sess-1", domain="industrial", language="darija", segment_id=1, kept_messages=2
    )
    assert dropped == ["واش فهمتي؟"]


def test_dropped_questions_returns_empty_when_nothing_falls_out_of_window(sqlite_engine):
    history.append_exchange(
        "sess-1", tenant_id="t1", user_id="u1", domain="industrial", language="fr",
        segment_id=1, user_content="Q1", assistant_content="Reponse. Une question ?",
    )
    dropped = history.extract_dropped_questions(
        "sess-1", domain="industrial", language="fr", segment_id=1, kept_messages=4
    )
    assert dropped == []


def test_dropped_questions_fails_open_on_db_error(monkeypatch):
    def _raise():
        raise RuntimeError("connection refused")
    monkeypatch.setattr(history, "_get_engine", _raise)
    assert history.extract_dropped_questions(
        "sess-x", domain="industrial", language="fr", segment_id=1
    ) == []


# --- fail-open ----------------------------------------------------------

def test_load_window_fails_open_on_db_error(monkeypatch):
    def _raise():
        raise RuntimeError("connection refused")
    monkeypatch.setattr(history, "_get_engine", _raise)
    assert history.load_window("sess-x", domain="industrial", language="fr", segment_id=1) == []


def test_append_exchange_fails_open_on_db_error(monkeypatch):
    def _raise():
        raise RuntimeError("connection refused")
    monkeypatch.setattr(history, "_get_engine", _raise)
    # Must not raise.
    history.append_exchange(
        "sess-x", tenant_id="t1", user_id="u1", domain="industrial", language="fr",
        segment_id=1, user_content="q", assistant_content="a",
    )


def test_get_pinned_fails_open_on_db_error(monkeypatch):
    def _raise():
        raise RuntimeError("connection refused")
    monkeypatch.setattr(history, "_get_engine", _raise)
    assert history.get_pinned("sess-x") is None
