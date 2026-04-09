"""Phase gate (design section 15): what must hold before a phase is declared done in docs/phase_status.json."""
import json
import re

from config import ROOT

STATUS = ROOT / "docs" / "phase_status.json"
LOG = ROOT / "docs" / "results" / "test_log.md"
TESTS = ROOT / "tests"
FIRST_GATED_PHASE = 5                          # tests are a completion criterion from Phase 5 on
REQUIRED_MODELS = {6: ["lr", "rf", "xgb"]}     # models a phase must have registered in models.REGISTRY
TAG = re.compile(r"PENDING-PHASE:\s*(\d+)")
ROW = re.compile(r"\|\s*Phase (\d+)\s*\|.*?fast: (\d+) passed, (\d+) failed.*?realdata: (\d+) passed, (\d+) failed")


def done_phases():
    return set(json.loads(STATUS.read_text())["done"])


def blocked_placeholders(done, tests_dir=TESTS):
    """placeholder test files tagged `PENDING-PHASE: N` where N is declared done"""
    out = []
    for f in sorted(tests_dir.glob("*.py")):
        for m in TAG.finditer(f.read_text(encoding="utf-8")):
            if int(m.group(1)) in done:
                out.append(f"{f.name} is still a placeholder for phase {m.group(1)}")
    return out


def phases_with_passing_record(log_text):
    """phases with a log row that has passing fast and real-data runs (nothing failed, something ran)"""
    ok = set()
    for m in ROW.finditer(log_text):
        phase, fp, ff, rp, rf = map(int, m.groups())
        if fp > 0 and rp > 0 and ff == 0 and rf == 0:
            ok.add(phase)
    return ok
