"""Tests for the self-bootstrapping dataset path.

`data/*.csv` is gitignored, so a fresh clone has no experiment CSV. These tests
bind the generator API, its byte-for-byte reproducibility, and the CLI's ability
to bootstrap the CSV from a foreign working directory.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys

from data.generate_data import SEED, N, build_dataset, write_dataset

ROOT = pathlib.Path(__file__).parent.parent
CSV_SHA256 = "4f02ecd2cbc18d82c06766dbd1215fa2599f5fa95f94d254e8fe06841bcfd7c5"


def test_write_dataset_is_byte_identical(tmp_path):
    """Default generation must reproduce the shipped CSV exactly, not just closely."""
    path = tmp_path / "nested" / "experiment.csv"
    df = write_dataset(path)
    assert path.parent.is_dir()  # parent dirs are created
    assert len(df) == N == 20_000
    assert hashlib.sha256(path.read_bytes()).hexdigest() == CSV_SHA256


def test_build_dataset_is_deterministic():
    a = build_dataset(seed=SEED, n=5_000)
    b = build_dataset(seed=SEED, n=5_000)
    assert a.equals(b)
    assert list(a.columns) == [
        "user_id",
        "segment",
        "x_pre",
        "treatment",
        "y_cont",
        "y_bin",
        "tau_true",
    ]


def test_cli_bootstraps_dataset_from_scratch(tmp_path):
    """No CSV, foreign cwd: the CLI must generate the data and record provenance."""
    data_path = tmp_path / "fresh" / "exp.csv"
    out_dir = tmp_path / "out"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "run.py"),
            "--data-path",
            str(data_path),
            "--out-dir",
            str(out_dir),
            "--no-plot",
            "--n-boot",
            "10",
            "--n-null",
            "10",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert data_path.exists()

    out = json.loads((out_dir / "metrics.json").read_text())
    assert out["data"]["sha256"] == hashlib.sha256(data_path.read_bytes()).hexdigest()
    assert out["data"]["rows"] == 20_000
    assert out["data"]["seed"] == 42
    assert out["data"]["path"] == str(data_path)
    assert out["n"]["bootstrap"] == 10
    assert not (out_dir / "uplift.png").exists()  # --no-plot honoured
