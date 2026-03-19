"""Tests for C16: Persona Rotation in debate rounds."""

from epistemic_agents.panel import DEBATE_PERSONAS, _get_persona_for_round


def test_persona_count():
    """At least 7 personas to cover a full 7-provider panel."""
    assert len(DEBATE_PERSONAS) >= 7


def test_persona_rotation_no_repeat_same_round():
    """Different providers in the same round get different personas."""
    round_num = 2
    personas = [_get_persona_for_round(i, round_num) for i in range(7)]
    assert len(set(personas)) == 7, "All providers should get distinct personas in a round"


def test_persona_rotation_shifts_across_rounds():
    """Same provider gets a different persona in different rounds."""
    provider_idx = 0
    p_round2 = _get_persona_for_round(provider_idx, 2)
    p_round3 = _get_persona_for_round(provider_idx, 3)
    assert p_round2 != p_round3, "Same provider should get different persona across rounds"


def test_persona_rotation_wraps():
    """Rotation works for provider indices beyond persona count."""
    n = len(DEBATE_PERSONAS)
    p1 = _get_persona_for_round(0, 2)
    p2 = _get_persona_for_round(n, 2)
    assert p1 == p2, "Should wrap around"


def test_all_personas_are_strings():
    for p in DEBATE_PERSONAS:
        assert isinstance(p, str)
        assert "LENS:" in p
