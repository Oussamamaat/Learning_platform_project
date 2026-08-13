from app.services.learner_state import get_learner_state


def test_returns_empty_dict_today():
    """Read-only seam: no learner-state table exists yet (ADR 0003 owns
    it), so this must return {} for any input -- callers treat {} as
    'nothing known', never as an error."""
    assert get_learner_state("u1", "t1", "industrial") == {}
    assert get_learner_state("", "", "") == {}
