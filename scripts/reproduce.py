"""Reproduce the causal-uplift results artifact and diff it against a reference.

One command regenerates the dataset, reruns the pipeline into a scratch
directory, and compares the fresh ``metrics.json`` with the committed
reference artifact::

    uv run python scripts/reproduce.py
    uv run python scripts/reproduce.py --workdir /tmp/repro --n-boot 400

The maintainer refreshes the reference once, after reviewing a run::

    uv run python scripts/reproduce.py --write-reference

Exit codes: 0 all sections match, 1 mismatch or pipeline failure, 2 reference
artifact missing (and ``--write-reference`` not passed).

The comparison is generic: every top-level key of the reference is a section,
and every nested key is compared recursively, so a new metric needs no change
here. ``data.path`` is the one volatile field - it records where the CSV
happened to live, so it is normalised to its file name before comparing;
content identity is covered by ``data.sha256``.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
# Running this file directly puts ``scripts/`` on sys.path, not the project root;
# the pipeline modules (``run``, ``src``, ``data``) are imported from there.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_REFERENCE = ROOT / "reports" / "reference" / "metrics.json"

# Rendering order; any section added later is appended alphabetically.
SECTION_ORDER = (
    "data", "cuped", "uplift", "null", "segments", "decision", "preregistration", "n",
)
PIPELINE_TIMEOUT = 1800
MISSING = "<missing>"

TABLE_WIDTH = 16
STATUS_WIDTH = 10
DETAIL_WIDTH = 120


# --- comparison --------------------------------------------------------------


def compare_metrics(
    reference: dict, candidate: dict, rtol: float = 1e-6, atol: float = 1e-9
) -> dict:
    """Recursively compare two metrics trees.

    Numbers are compared with ``math.isclose`` (an int and a float holding the
    same value are equal); booleans, strings and ``None`` are compared exactly;
    dicts and lists are walked element by element, so the list of diffs names
    the deepest disagreeing key as a dotted path.

    Missing keys are reported with the ``"<missing>"`` marker in place of the
    value. Returns ``{"ok": bool, "diffs": [{"path", "reference", "candidate"}]}``.
    """
    diffs: list[dict] = []
    _compare(reference, candidate, "", diffs, rtol, atol)
    return {"ok": not diffs, "diffs": diffs}


def _compare(ref, cand, path: str, diffs: list[dict], rtol: float, atol: float) -> None:
    if isinstance(ref, dict) and isinstance(cand, dict):
        for key in sorted(set(ref) | set(cand), key=str):
            child = f"{path}.{key}" if path else str(key)
            if key not in ref:
                diffs.append(_entry(child, MISSING, cand[key]))
            elif key not in cand:
                diffs.append(_entry(child, ref[key], MISSING))
            else:
                _compare(ref[key], cand[key], child, diffs, rtol, atol)
        return

    if isinstance(ref, list) and isinstance(cand, list):
        if len(ref) != len(cand):
            diffs.append(_entry(path, ref, cand))
            return
        for index, (ref_item, cand_item) in enumerate(zip(ref, cand, strict=True)):
            child = f"{path}.{index}" if path else str(index)
            _compare(ref_item, cand_item, child, diffs, rtol, atol)
        return

    if isinstance(ref, dict | list) or isinstance(cand, dict | list):
        diffs.append(_entry(path, ref, cand))
        return

    if not _scalars_equal(ref, cand, rtol, atol):
        diffs.append(_entry(path, ref, cand))


def _scalars_equal(ref, cand, rtol: float, atol: float) -> bool:
    """True when two leaves agree; bool is checked before int (it subclasses int)."""
    if isinstance(ref, bool) or isinstance(cand, bool):
        return isinstance(ref, bool) and isinstance(cand, bool) and ref == cand
    if _is_number(ref) and _is_number(cand):
        return math.isclose(float(ref), float(cand), rel_tol=rtol, abs_tol=atol)
    if ref is None or cand is None:
        return ref is None and cand is None
    return isinstance(ref, str) and isinstance(cand, str) and ref == cand


def _is_number(value) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _entry(path: str, reference, candidate) -> dict:
    return {"path": path, "reference": reference, "candidate": candidate}


# --- reproduction ------------------------------------------------------------


def run_reproduction(
    workdir: str | pathlib.Path,
    n_boot: int = 400,
    n_null: int = 200,
    reference: str | pathlib.Path | None = None,
) -> dict:
    """Regenerate the dataset, rerun the pipeline, compare against ``reference``.

    Steps: ``write_dataset(workdir/"data"/"experiment.csv")``, then
    ``run.py --data-path ... --out-dir workdir/"reports" --no-plot --n-boot N
    --n-null N`` in a subprocess, then a recursive diff of the fresh
    ``metrics.json`` against the reference artifact.

    ``reference`` is a path; ``None`` means ``reports/reference/metrics.json``.
    The returned record carries ``ok``/``status``, the full diff list, and a
    per-section status map, so callers can render a table without re-parsing.
    """
    workdir = pathlib.Path(workdir).expanduser().resolve()
    reference_path = (
        pathlib.Path(reference).expanduser().resolve() if reference is not None else DEFAULT_REFERENCE
    )
    fresh_path = workdir / "reports" / "metrics.json"
    record = {
        "ok": False,
        "status": "reference-missing",
        "reference": str(reference_path),
        "workdir": str(workdir),
        "fresh_metrics": str(fresh_path),
        "normalized": ["data.path"],
        "sections": {},
        "diffs": [],
    }

    try:
        seed = _write_dataset(workdir / "data" / "experiment.csv")
        proc = _run_pipeline(
            data_path=workdir / "data" / "experiment.csv",
            out_dir=workdir / "reports",
            n_boot=n_boot,
            n_null=n_null,
            data_seed=seed,
        )
    except (ImportError, OSError, ValueError, subprocess.SubprocessError) as exc:
        record.update(status="pipeline-failed", error=_short(f"{type(exc).__name__}: {exc}"))
        return record

    if proc.returncode != 0 or not fresh_path.is_file():
        record.update(
            status="pipeline-failed",
            error=_short(f"run.py rc={proc.returncode}: {proc.stderr.strip()[-400:]}"),
        )
        return record

    try:
        candidate = json.loads(fresh_path.read_text())
        reference_metrics = (
            json.loads(reference_path.read_text()) if reference_path.is_file() else None
        )
    except (json.JSONDecodeError, OSError) as exc:
        record.update(status="pipeline-failed", error=_short(f"unreadable artifact: {exc}"))
        return record
    if reference_metrics is None:
        return record

    normalized_reference = normalize_metrics(reference_metrics)
    normalized_candidate = normalize_metrics(candidate)
    result = compare_metrics(normalized_reference, normalized_candidate)
    record.update(
        ok=result["ok"],
        status="match" if result["ok"] else "mismatch",
        diffs=result["diffs"],
        sections=_sections(normalized_reference, normalized_candidate, result["diffs"]),
    )
    return record


def _write_dataset(path: pathlib.Path) -> int:
    """Import the generator on call: it is part of the frozen data contract.

    Returns the seed, so the pipeline can record it: ``run.py`` sees an existing
    CSV and cannot tell that this harness wrote it.
    """
    from data.generate_data import SEED, write_dataset

    write_dataset(path)
    return SEED


def _run_pipeline(
    data_path, out_dir, n_boot: int, n_null: int, data_seed: int | None = None
) -> subprocess.CompletedProcess:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(ROOT / "run.py"),
        "--data-path", str(data_path),
        "--out-dir", str(out_dir),
        "--no-plot",
        "--n-boot", str(n_boot),
        "--n-null", str(n_null),
    ]
    if data_seed is not None:
        cmd += ["--data-seed", str(data_seed)]
    return subprocess.run(
        cmd, cwd=ROOT, capture_output=True, text=True, timeout=PIPELINE_TIMEOUT, check=False
    )


def normalize_metrics(metrics: dict) -> dict:
    """Copy the tree, replacing ``data.path`` with its file name (see module doc).

    ``data.path`` records where the CSV happened to live, so two runs of the
    same generator never agree on it. Only the file name is kept; content
    identity is covered by ``data.sha256``.
    """
    if not isinstance(metrics, dict):
        return metrics
    out = copy.deepcopy(metrics)
    data = out.get("data")
    if isinstance(data, dict) and isinstance(data.get("path"), str):
        data["path"] = pathlib.PurePath(data["path"]).name
    return out


def _sections(reference: dict, candidate: dict, diffs: list[dict]) -> dict:
    """Per-top-level-key status: match / mismatch / missing."""
    if not isinstance(reference, dict) or not isinstance(candidate, dict):
        return {"metrics": _section(diffs, present=True, key="")}

    keys = [key for key in SECTION_ORDER if key in reference or key in candidate]
    keys += sorted((set(reference) | set(candidate)) - set(keys), key=str)

    sections = {}
    for key in keys:
        present = key in reference and key in candidate
        section = _section(diffs, present=present, key=key)
        if section["status"] == "match" and key == "data" and isinstance(candidate[key], dict):
            section["detail"] = _data_detail(candidate[key])
        sections[key] = section
    return sections


def _section(diffs: list[dict], present: bool, key: str) -> dict:
    scoped = [d for d in diffs if key == "" or d["path"] == key or d["path"].startswith(f"{key}.")]
    if not present:
        return {"status": "missing", "diffs": len(scoped), "detail": "key absent on one side"}
    if scoped:
        first = ", ".join(d["path"] for d in scoped[:3])
        return {"status": "mismatch", "diffs": len(scoped), "detail": _short(f"{len(scoped)} diff: {first}")}
    return {"status": "match", "diffs": 0, "detail": ""}


def _data_detail(data: dict) -> str:
    sha = str(data.get("sha256", ""))[:12]
    return f"sha256={sha} rows={data.get('rows')} seed={data.get('seed')}"


# --- CLI ---------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Regenerate causal-uplift metrics and diff a reference.")
    ap.add_argument("--reference", default=str(DEFAULT_REFERENCE), help="reference metrics.json")
    ap.add_argument("--workdir", default=None, help="scratch dir (default: a fresh temp dir)")
    ap.add_argument("--n-boot", type=int, default=400, help="bootstrap resamples per metric")
    ap.add_argument("--n-null", type=int, default=200, help="random scorers for the null")
    ap.add_argument("--write-reference", action="store_true", help="copy the fresh artifact to --reference")
    args = ap.parse_args(argv)

    reference = pathlib.Path(args.reference).expanduser().resolve()
    if not args.write_reference and not reference.is_file():
        record = {
            "ok": False,
            "status": "reference-missing",
            "reference": str(reference),
            "workdir": args.workdir or "<not created>",
            "sections": {},
        }
        _print_table(record)
        return 2

    workdir = (
        pathlib.Path(args.workdir).expanduser()
        if args.workdir
        else pathlib.Path(tempfile.mkdtemp(prefix="causal-uplift-repro-"))
    )
    record = run_reproduction(workdir, n_boot=args.n_boot, n_null=args.n_null, reference=reference)

    if args.write_reference:
        fresh = pathlib.Path(record["fresh_metrics"])
        if record["status"] == "pipeline-failed" or not fresh.is_file():
            _print_table(record)
            return 1
        reference.parent.mkdir(parents=True, exist_ok=True)
        # Store the normalised tree: data.path records the scratch directory, and
        # an absolute /private/tmp path must not leak into the repository.
        normalized = normalize_metrics(json.loads(fresh.read_text()))
        reference.write_text(json.dumps(normalized, indent=2, allow_nan=False))
        _print_table(record)
        print(f"reference written: {reference}")
        return 0

    _print_table(record)
    return 0 if record["ok"] else 1


def _print_table(record: dict) -> None:
    ref_status = "missing" if record.get("status") == "reference-missing" else "ok"
    rows = [("reference", ref_status, record.get("reference", ""))]
    if record.get("error"):
        rows.append(("pipeline", "failed", record["error"]))
    for name, section in record.get("sections", {}).items():
        rows.append((name, section["status"], _short(section["detail"])))
    rows.append(("workdir", "ok", record.get("workdir", "")))

    print(f"{'section':<{TABLE_WIDTH}}{'status':<{STATUS_WIDTH}}detail")
    for name, status, detail in rows:
        print(f"{name:<{TABLE_WIDTH}}{status:<{STATUS_WIDTH}}{detail}")


def _short(text: str, width: int = DETAIL_WIDTH) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= width else f"{text[: width - 3]}..."


if __name__ == "__main__":
    raise SystemExit(main())
