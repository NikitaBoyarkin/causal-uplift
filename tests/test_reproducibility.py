"""Tests for the reproduction harness (scripts/reproduce.py).

The comparison tests are pure and fast. The determinism test drives the real
pipeline twice through ``run_reproduction`` (n_boot=10, n_null=10) - about 15 s
in total on a laptop - and asserts the two ``metrics.json`` files are identical
at rtol=1e-9.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys

import pytest

from scripts.reproduce import (
    DEFAULT_REFERENCE,
    compare_metrics,
    normalize_metrics,
    run_reproduction,
)

ROOT = pathlib.Path(__file__).parent.parent
SCRIPT = ROOT / "scripts" / "reproduce.py"


# --- compare_metrics ---------------------------------------------------------


def test_equal_trees_match():
    tree = {"cuped": {"rho": 0.5, "naive": {"ate": 1.0}}, "n": {"test": 6000}, "flag": True}
    result = compare_metrics(tree, json.loads(json.dumps(tree)))
    assert result == {"ok": True, "diffs": []}


def test_float_within_tolerance_matches():
    result = compare_metrics({"a": 1.0}, {"a": 1.0 + 1e-9})
    assert result["ok"] is True
    assert result["diffs"] == []


def test_float_outside_tolerance_reports_dotted_path():
    reference = {"uplift": {"t_learner": {"auuc": 0.5}}}
    candidate = {"uplift": {"t_learner": {"auuc": 0.6}}}
    result = compare_metrics(reference, candidate)
    assert result["ok"] is False
    assert result["diffs"] == [{"path": "uplift.t_learner.auuc", "reference": 0.5, "candidate": 0.6}]


def test_tight_rtol_flags_tiny_float_drift():
    result = compare_metrics({"a": 1.0}, {"a": 1.0 + 1e-9}, rtol=1e-12)
    assert result["ok"] is False
    assert result["diffs"][0]["path"] == "a"


@pytest.mark.parametrize(
    ("reference", "candidate"),
    [
        (1, 2),
        (1, "1"),
        ("a", 1),
        (True, False),
        (True, 1),
        (0, False),
    ],
)
def test_scalar_type_and_value_mismatch_is_a_diff(reference, candidate):
    result = compare_metrics({"a": reference}, {"a": candidate})
    assert result["ok"] is False
    assert result["diffs"] == [{"path": "a", "reference": reference, "candidate": candidate}]


def test_missing_key_on_each_side_is_a_diff():
    left_only = compare_metrics({"a": {"b": 1}}, {"a": {}})
    assert left_only["ok"] is False
    assert left_only["diffs"] == [{"path": "a.b", "reference": 1, "candidate": "<missing>"}]

    right_only = compare_metrics({"a": {}}, {"a": {"b": 1}})
    assert right_only["ok"] is False
    assert right_only["diffs"] == [{"path": "a.b", "reference": "<missing>", "candidate": 1}]


def test_none_is_an_exact_match_value():
    assert compare_metrics({"a": None}, {"a": None})["ok"] is True
    assert compare_metrics({"a": None}, {"a": 0})["ok"] is False
    assert compare_metrics({"a": 0}, {"a": None})["ok"] is False


def test_list_elements_and_length_are_compared():
    element = compare_metrics({"a": [1, 2]}, {"a": [1, 3]})
    assert element["diffs"] == [{"path": "a.1", "reference": 2, "candidate": 3}]

    length = compare_metrics({"a": [1, 2]}, {"a": [1]})
    assert length["ok"] is False
    assert length["diffs"][0]["path"] == "a"


def test_container_type_mismatch_is_a_diff():
    result = compare_metrics({"a": {"b": 1}}, {"a": [1]})
    assert result["ok"] is False
    assert result["diffs"][0]["path"] == "a"


# --- reproduction ------------------------------------------------------------


def test_pipeline_is_deterministic_across_processes(tmp_path):
    """Two fresh generator+pipeline runs must agree at rtol=1e-9.

    ``data.path`` is the one field that legitimately differs (it records the
    scratch directory), so both trees are normalised before the strict compare;
    the CSV content identity is asserted separately through ``data.sha256``.
    """
    absent_reference = tmp_path / "absent" / "metrics.json"
    first = run_reproduction(tmp_path / "run1", n_boot=10, n_null=10, reference=absent_reference)
    second = run_reproduction(tmp_path / "run2", n_boot=10, n_null=10, reference=absent_reference)

    assert first["status"] == "reference-missing", first
    assert second["status"] == "reference-missing", second
    first_metrics = json.loads(pathlib.Path(first["fresh_metrics"]).read_text())
    second_metrics = json.loads(pathlib.Path(second["fresh_metrics"]).read_text())
    assert first_metrics["data"]["sha256"] == second_metrics["data"]["sha256"]
    assert first_metrics["data"]["seed"] == second_metrics["data"]["seed"] == 42

    result = compare_metrics(
        normalize_metrics(first_metrics), normalize_metrics(second_metrics), rtol=1e-9
    )
    assert result["ok"] is True, result["diffs"][:10]
    assert set(first_metrics) >= {"cuped", "uplift", "null", "segments", "n", "data"}


def test_cli_exits_2_when_reference_is_missing(tmp_path):
    missing = tmp_path / "no-such-dir" / "metrics.json"
    workdir = tmp_path / "workdir"
    proc = subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--reference", str(missing),
            "--workdir", str(workdir),
            "--n-boot", "2", "--n-null", "2",
        ],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 2, proc.stderr[-2000:]
    assert "missing" in proc.stdout
    assert not workdir.exists(), "CLI must not run the pipeline when the reference is absent"


def test_default_reference_points_into_reports():
    assert DEFAULT_REFERENCE == ROOT / "reports" / "reference" / "metrics.json"


def test_committed_reference_artifact_exists_and_carries_the_verdict():
    """The reference is the thing reproduction compares against: it must ship.

    The numbers are pinned, not just "some verdict": a reference regenerated by a
    broken rule (wrong break-even, wrong campaign size, wrong cost branch) passes
    a membership check but fails here.
    """
    assert DEFAULT_REFERENCE.is_file(), (
        "missing committed reference; run scripts/reproduce.py --write-reference once"
    )
    ref = json.loads(DEFAULT_REFERENCE.read_text())

    assert ref["data"]["sha256"] == "4f02ecd2cbc18d82c06766dbd1215fa2599f5fa95f94d254e8fe06841bcfd7c5"
    assert ref["data"]["rows"] == 20_000
    assert ref["data"]["seed"] == 42
    assert ref["data"]["generated"] is True
    assert ref["data"]["path"] == "experiment.csv"  # normalised, not a scratch path

    decision = ref["decision"]
    assert decision["decision"] == "ship"
    assert decision["target_segment"] == "new"
    assert decision["n_target"] == 6015
    assert decision["break_even_uplift"] == pytest.approx(0.05)
    assert decision["observed_uplift"] == pytest.approx(0.1108168, abs=1e-7)
    assert decision["ci"] == pytest.approx([0.0893663, 0.1356808], abs=1e-7)
    assert decision["expected_net_per_user"] == pytest.approx(2.432673, abs=1e-6)

    err = decision["cost_of_error"]
    assert err["ship_if_no_effect"]["total"] == pytest.approx(12_030.0, abs=0.01)
    assert err["no_ship_if_effect"]["total"] == pytest.approx(14_632.53, abs=0.01)
    assert err["worst_case_total"] == pytest.approx(14_632.53, abs=0.01)

    # The prereg hash is the anti-rewriting lock: editing the document without
    # refreshing the reference fails here, not silently at reproduction time.
    doc_sha = hashlib.sha256((ROOT / "docs" / "preregistration.md").read_bytes()).hexdigest()
    assert ref["preregistration"] == {"path": "docs/preregistration.md", "sha256": doc_sha}
