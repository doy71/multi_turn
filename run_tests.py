from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run_cmd(args):
    proc = subprocess.run(args, cwd=ROOT)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main():
    print("Running offline core tests...\n")
    run_cmd([sys.executable, "test_core.py"])

    print("\nRunning dataset-builder tests...\n")
    sys.path.insert(0, str(ROOT))
    from tests.test_dataset_builder import run as run_dataset_tests

    run_dataset_tests()
    print("\nAll test suites passed.")


if __name__ == "__main__":
    main()
