"""Run the fast suite and the real-data tests and append the result to docs/results/test_log.md.
Run at each phase's commit: uv run python ml/record_tests.py <phase>   (exit code 1 if anything failed)"""
import re
import subprocess
import sys
from datetime import date

from config import ROOT
from phase_gate import LOG


def run(*args):
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *args], capture_output=True, text=True, cwd=ROOT)
    tail = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""
    n = lambda word: int(m.group(1)) if (m := re.search(rf"(\d+) {word}", tail)) else 0
    return n("passed"), n("failed") + n("error"), n("skipped"), p.returncode


def main(phase):
    fast = run()
    real = run("-m", "realdata")
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", ".", ":!docs/results/test_log.md"], capture_output=True, text=True,
                           cwd=ROOT).stdout.strip()
    row = (f"| {date.today()} | Phase {phase} | {head} | {'uncommitted changes' if dirty else 'clean'} | "
           f"fast: {fast[0]} passed, {fast[1]} failed, {fast[2]} skipped | realdata: {real[0]} passed, {real[1]} failed, {real[2]} skipped |")
    if not LOG.exists():
        LOG.parent.mkdir(parents=True, exist_ok=True)
        LOG.write_text("# Test record\n\nFast suite and real-data tests, re-run and recorded at each phase commit "
                       "(`ml/record_tests.py`). A phase may be declared done only with a passing row here.\n\n"
                       "| Date | Phase | Code commit | Tree | Fast suite | Real-data tests |\n|---|---|---|---|---|---|\n", encoding="utf-8")
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(row + "\n")
    print(row)
    sys.exit(0 if fast[1] == real[1] == 0 and fast[0] > 0 and real[0] > 0 else 1)


if __name__ == "__main__":
    main(sys.argv[1])
