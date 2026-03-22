# %%
"""Aggregate `performance_metrics/*.csv` into one table and plot with Plotly Express."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_MODELS_DIR = Path(__file__).resolve().parent
if str(_MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(_MODELS_DIR))

import pandas as pd
import plotly.express as px

from model_run_utils import METRICS_ROOT

METRIC_COLS = ["accuracy", "auc_roc", "precision", "recall", "f1"]


def load_combined_metrics(metrics_dir: Path | None = None) -> pd.DataFrame:
    root = metrics_dir if metrics_dir is not None else METRICS_ROOT
    paths = sorted(root.glob("*.csv"))
    paths = [p for p in paths if p.name != "combined_performance_metrics.csv"]
    if not paths:
        raise FileNotFoundError(f"No metric CSVs under {root}")

    frames: list[pd.DataFrame] = []
    for p in paths:
        df = pd.read_csv(p, on_bad_lines="skip")
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    required = {"model", *METRIC_COLS}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"Missing columns {missing} after loading {root}")

    out = out.dropna(subset=["model"])
    out = out.drop_duplicates(subset=["model"], keep="last")
    for c in METRIC_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=METRIC_COLS, how="all")
    return out.sort_values("auc_roc", ascending=False, na_position="last").reset_index(drop=True)


def metrics_long(df: pd.DataFrame) -> pd.DataFrame:
    return df.melt(id_vars=["model"], value_vars=METRIC_COLS, var_name="metric", value_name="value")


def figure_grouped_bars(long_df: pd.DataFrame):
    return px.bar(
        long_df,
        x="model",
        y="value",
        color="metric",
        barmode="group",
        title="Model performance metrics (test / OOB holdout)",
        labels={"model": "Model", "value": "Score", "metric": "Metric"},
        category_orders={"metric": METRIC_COLS},
    )


def figure_faceted(long_df: pd.DataFrame):
    return px.bar(
        long_df,
        x="model",
        y="value",
        facet_col="metric",
        facet_col_wrap=3,
        title="Model performance by metric",
        labels={"model": "Model", "value": "Score"},
        category_orders={"metric": METRIC_COLS},
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=None,
        help=f"Folder of per-model CSVs (default: {METRICS_ROOT})",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Write combined table (default: metrics_dir/combined_performance_metrics.csv)",
    )
    parser.add_argument(
        "--out-html",
        type=Path,
        default=None,
        help="Write both figures to one HTML file (default: metrics_dir/performance_metrics_plots.html)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open figures in browser (default: also show when running under Jupyter)",
    )
    args = parser.parse_args(argv)

    root = args.metrics_dir or METRICS_ROOT
    combined = load_combined_metrics(root)
    out_csv = args.out_csv or (root / "combined_performance_metrics.csv")
    out_html = args.out_html or (root / "performance_metrics_plots.html")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    print(combined.to_string(index=False))
    print(f"\nWrote combined table: {out_csv}")
    combined.to_csv(out_csv, index=False)

    long_df = metrics_long(combined)
    fig_group = figure_grouped_bars(long_df)
    fig_group.update_layout(xaxis_tickangle=-35, legend_title_text="Metric")
    fig_group.update_yaxes(range=[0, 1])

    fig_facet = figure_faceted(long_df)
    fig_facet.update_layout(xaxis_tickangle=-35)
    fig_facet.for_each_yaxis(lambda ax: ax.update(range=[0, 1]))

    plotly_cdn = "https://cdn.plot.ly/plotly-2.35.2.min.js"
    html_body = (
        "<!DOCTYPE html>\n<html lang='en'><head><meta charset='utf-8'/>"
        f"<title>Performance metrics</title><script src='{plotly_cdn}'></script></head><body>\n"
        "<h2>Grouped metrics</h2>\n"
        + fig_group.to_html(full_html=False, include_plotlyjs=False)
        + "\n<h2>Faceted by metric</h2>\n"
        + fig_facet.to_html(full_html=False, include_plotlyjs=False)
        + "\n</body></html>"
    )
    out_html.write_text(html_body, encoding="utf-8")
    print(f"Wrote plots: {out_html}")

    show = args.show or ("ipykernel" in sys.modules)
    if show:
        fig_group.show()
        fig_facet.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# %%
