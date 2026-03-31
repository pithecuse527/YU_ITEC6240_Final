#!/usr/bin/env python3
"""Run every training script under ``models/``, then refresh combined metrics + plots.

Usage (from repo root)::

    python models/run_all_models.py

Or from ``models/``::

    python run_all_models.py

Use ``--no-summarize`` to skip ``summarize_performance_metrics.py`` at the end.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent

# Order: unified simple_net* pipelines first, then alternate CatBoost / logistic scripts.
DEFAULT_SCRIPTS = [
    "simple_net.py",
    "simple_net_logistic.py",
    "simple_net_knn.py",
    "simple_net_rf.py",
    "simple_net_catboost.py",
    "simple_net_svm.py",
    "simple_net_xgboost.py",
    "simple_catboost_clf.py",
    "simple_logistic_regression.py",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-summarize",
        action="store_true",
        help="Do not run summarize_performance_metrics.py after all models.",
    )
    parser.add_argument(
        "scripts",
        nargs="*",
        help=f"Basenames under models/ (default: {len(DEFAULT_SCRIPTS)} built-in scripts).",
    )
    args = parser.parse_args(argv)

    names = args.scripts if args.scripts else DEFAULT_SCRIPTS
    failed: list[str] = []

    for name in names:
        path = MODELS_DIR / name
        if not path.is_file():
            print(f"[run_all_models] skip (missing): {path}", file=sys.stderr)
            continue
        print(f"\n{'=' * 60}\nRunning {name}\n{'=' * 60}")
        proc = subprocess.run(
            [sys.executable, str(path)],
            cwd=str(MODELS_DIR),
        )
        if proc.returncode != 0:
            failed.append(name)
            print(f"[run_all_models] FAILED: {name} (exit {proc.returncode})", file=sys.stderr)

    if not args.no_summarize:
        summ = MODELS_DIR / "summarize_performance_metrics.py"
        if summ.is_file():
            print(f"\n{'=' * 60}\nRunning summarize_performance_metrics.py\n{'=' * 60}")
            proc = subprocess.run([sys.executable, str(summ)], cwd=str(MODELS_DIR))
            if proc.returncode != 0:
                failed.append("summarize_performance_metrics.py")
        else:
            print(f"[run_all_models] missing {summ}, skip summarize", file=sys.stderr)

    if failed:
        print(f"\n[run_all_models] Finished with failures: {failed}", file=sys.stderr)
        return 1
    print("\n[run_all_models] All steps completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
