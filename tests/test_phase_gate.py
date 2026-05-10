"""The phase gate: a phase declared done in docs/phase_status.json has no placeholder left, has its models registered, and has a
passing fast + real-data record in docs/results/test_log.md."""
import pytest

import phase_gate as g
from models import REGISTRY

PLACEHOLDERS = {"test_shap.py": 8, "test_simulator.py": 10}      # conformal (7) was replaced by real tests


def test_no_placeholder_remains_for_a_done_phase():
    assert g.blocked_placeholders(g.done_phases()) == []


def test_models_a_done_phase_must_deliver_are_registered():
    for phase in g.done_phases():
        for name in g.REQUIRED_MODELS.get(phase, []):
            assert name in REGISTRY, f"phase {phase} is done but model '{name}' is not registered"


def test_every_done_phase_from_5_has_a_passing_test_record():
    log = g.LOG.read_text(encoding="utf-8") if g.LOG.exists() else ""
    need = {p for p in g.done_phases() if p >= g.FIRST_GATED_PHASE}
    assert need <= g.phases_with_passing_record(log), f"no passing fast + real-data record for phases {sorted(need - g.phases_with_passing_record(log))}"


def test_placeholders_carry_their_phase_tag():
    """the gate only works if every placeholder is tagged"""
    for name, phase in PLACEHOLDERS.items():
        assert f"PENDING-PHASE: {phase}" in (g.TESTS / name).read_text(encoding="utf-8"), name


def test_the_gate_blocks_when_it_should(tmp_path):
    """gate logic on synthetic inputs: fires for a done phase, stays quiet otherwise"""
    (tmp_path / "test_x.py").write_text("# PENDING-PHASE: 8\nimport pytest\n", encoding="utf-8")
    (tmp_path / "test_y.py").write_text("def test_ok(): pass\n", encoding="utf-8")
    assert g.blocked_placeholders({5, 8}, tmp_path) == ["test_x.py is still a placeholder for phase 8"]
    assert g.blocked_placeholders({5, 6}, tmp_path) == []


@pytest.mark.parametrize("row,expected", [
    ("| d | Phase 5 | abc | clean | fast: 146 passed, 0 failed, 4 skipped | realdata: 7 passed, 0 failed, 3 skipped |", {5}),
    ("| d | Phase 5 | abc | clean | fast: 146 passed, 1 failed, 4 skipped | realdata: 7 passed, 0 failed, 3 skipped |", set()),
    ("| d | Phase 5 | abc | clean | fast: 146 passed, 0 failed, 4 skipped | realdata: 0 passed, 0 failed, 10 skipped |", set()),
    ("| d | Phase 6 | abc | clean | fast: 200 passed, 0 failed, 3 skipped | realdata: 9 passed, 2 failed, 3 skipped |", set()),
])
def test_record_parsing(row, expected):
    assert g.phases_with_passing_record(row) == expected
