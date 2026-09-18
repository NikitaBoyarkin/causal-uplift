"""Ensure the experiment dataset exists before tests."""
import pathlib, subprocess, sys

ROOT = pathlib.Path(__file__).parent.parent
DATA = ROOT / "data" / "experiment.csv"


def pytest_configure():
    if not DATA.exists():
        subprocess.run([sys.executable, str(ROOT / "data" / "generate_data.py")], check=True)
