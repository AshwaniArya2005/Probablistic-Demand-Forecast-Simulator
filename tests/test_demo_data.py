"""The precomputed demo data (ml/build_demo_data.py): built from the committed result files, aggregates only, and in step with the Express service's expected data version."""
import json
import re

import pytest

import build_demo_data as b
from config import ROOT


@pytest.fixture(scope="module")
def snap():
    b.build()
    return json.loads((ROOT / "web" / "snapshot.json").read_text(encoding="utf-8"))


def test_the_data_version_matches_the_express_service(snap):
    js = (ROOT / "server" / "src" / "dataVersion.js").read_text(encoding="utf-8")
    assert re.search(r"EXPECTED_DATA_VERSION: '([^']+)'", js).group(1) == b.DATA_VERSION == snap["data_version"]


def test_only_the_scenarios_that_were_run_are_present_with_b3a_first(snap):
    assert [s["id"] for s in snap["scenarios"]] == ["primary", "L7", "randlead", "nofp", "zerodemand"]
    for s in snap["scenarios"]:
        for key in ("comparison_comparator_anchored", "comparison_quantile_anchored"):
            assert s[key][0]["comparator"].startswith("B3a"), (s["id"], key)             # the fair benchmark leads every comparison
        assert {r["comparator"].split(" ")[0] for r in s["comparison_comparator_anchored"]} == {"B3a", "B2", "Naive"}
        assert s["curves"] and s["cost"] and s["service_gap"]


def test_the_primary_scenario_carries_the_logged_numbers(snap):
    p = snap["scenarios"][0]
    row = p["comparison_comparator_anchored"][0]
    assert row["mean over matched"] == "16.0%" and row["95% interval (items)"] == "[0.7%, 24.9%]" and row["alpha 0.99"] == "46.3%"
    assert p["comparison_quantile_anchored"][0]["mean over matched"] == "12.7%"
    assert len(p["predictions"]) == 3
    assert "-0.0410" in snap["forecast_accuracy"]["against_b3a"]


def test_the_caveats_and_null_tests_are_shipped(snap):
    text = " ".join(snap["caveats"])
    assert "not evidence of forecast skill" in text and "unresolved" in text and "not variation between periods" in text
    assert {r["demand"].split(" ")[0] for r in snap["null_tests"]["test"]} == {"real", "BLOCK", "SWAP"}


def test_the_snapshot_contains_no_m5_rows_or_series_ids(snap):
    text = json.dumps(snap)
    assert not re.search(r"_(CA|TX|WI)_[123]_evaluation", text)              # no series identifiers
    assert not re.search(r"\b20(1[1-6])-\d\d-\d\d\b", text.replace(snap["built"], ""))          # no per-day data (only the build date)


def test_the_parser_reads_a_markdown_table_by_its_heading():
    md = "intro\n\n**Title A**\n\n| a | b |\n|:--|--:|\n| 1 | 2 |\n| 3 | 4 |\n\ntext\n\n**Title B**\n\n| c |\n|---|\n| 5 |\n"
    t = b.parse_tables(md)
    assert [x["title"] for x in t] == ["**Title A**", "**Title B**"]
    assert b.rows_of(t[0]) == [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}]
    with pytest.raises(SystemExit):
        b.find(t, "**Missing")
