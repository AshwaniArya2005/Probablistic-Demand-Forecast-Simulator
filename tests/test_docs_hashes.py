"""Every commit hash cited in the docs must resolve in this repository (a rewritten or amended history would leave dangling references)."""
import re
import subprocess

import pytest

from config import ROOT


def _git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True, cwd=ROOT)


def _cited():
    hashes = {}
    for f in [*ROOT.glob("docs/**/*.md"), *ROOT.glob("*.md")]:
        text = f.read_text(encoding="utf-8")
        for h in set(re.findall(r"`([0-9a-f]{7,12})`", text)) | set(re.findall(r"\| ([0-9a-f]{7}) \|", text)):
            hashes.setdefault(h, []).append(f.name)
    return hashes


def test_cited_commit_hashes_resolve():
    probe = _git("rev-parse", "--is-shallow-repository")
    if probe.returncode != 0 or probe.stdout.strip() != "false":
        pytest.skip("not a full git checkout (shallow clone or no git): cited commits may be absent")
    missing = {h: fs for h, fs in _cited().items() if _git("cat-file", "-e", f"{h}^{{commit}}").returncode != 0}
    assert not missing, f"docs cite commits that do not exist here: {missing}"


def test_the_scan_finds_hashes():
    """the check must not pass vacuously"""
    assert len(_cited()) >= 5
