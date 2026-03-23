"""
Fancy Streamlit UI for the same pipeline as ``simple_net_*.py``:
load data → nested-CV Optuna on train pool → stratified OOF CV → optional MDA/SHAP on holdout.

Run from the ``models`` directory::

    streamlit run streamlit_dashboard.py

Or from repo root::

    streamlit run models/streamlit_dashboard.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
import traceback
import uuid
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

# -----------------------------------------------------------------------------
# Imports from sibling package (``models/`` on sys.path)
# -----------------------------------------------------------------------------
_MODELS_DIR = Path(__file__).resolve().parent
if str(_MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(_MODELS_DIR))

import optuna
from model_run_utils import (
    DATA_COMBINED_DEFAULT,
    EXPERIMENT,
    ExperimentConfig,
    METRICS_ROOT,
    build_run_config_json,
    fit_simple_net_on_pool_for_holdout,
    load_combined_xy,
    oof_scalar_metrics,
    pool_cv_oof_predictions,
    roc_oob_figure,
    run_permutation_mda,
    run_shap_proba,
    split_pool_holdout,
    write_performance_metrics_csv,
)
from optuna_hpo import get_simple_net_tune_fn

# Quieter background logs while Optuna runs in the UI
logging.getLogger("optuna_hpo").setLevel(logging.WARNING)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,600;0,9..40,700;1,9..40,400&family=JetBrains+Mono:wght@400;600&display=swap');
        :root {
            --bg-deep: #0b0f19;
            --bg-card: rgba(22, 28, 45, 0.72);
            --border: rgba(129, 140, 248, 0.22);
            --accent: #818cf8;
            --accent-2: #34d399;
            --accent-3: #f472b6;
            --text: #f1f5f9;
            --muted: #94a3b8;
        }
        .stApp {
            background: radial-gradient(1200px 600px at 10% -10%, rgba(99,102,241,0.18), transparent 55%),
                        radial-gradient(900px 500px at 100% 0%, rgba(52,211,153,0.12), transparent 50%),
                        radial-gradient(800px 400px at 50% 100%, rgba(244,114,182,0.08), transparent 45%),
                        linear-gradient(180deg, #0b0f19 0%, #111827 100%) !important;
        }
        html, body, [class*="css"] {
            font-family: 'DM Sans', system-ui, sans-serif;
        }
        code, pre, .stCodeBlock {
            font-family: 'JetBrains Mono', monospace !important;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(175deg, rgba(17,24,39,0.97) 0%, rgba(15,23,42,0.98) 100%) !important;
            border-right: 1px solid var(--border) !important;
        }
        [data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
        .sb-seg {
            margin: 1.15rem 0 0.65rem 0;
            padding: 0.35rem 0 0.4rem 0;
            border-bottom: 1px solid rgba(129,140,248,0.15);
        }
        .sb-seg-title {
            font-size: 0.68rem;
            font-weight: 700;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: #a5b4fc;
            margin: 0;
        }
        .sb-seg-sub {
            font-size: 0.72rem;
            color: var(--muted);
            margin: 0.2rem 0 0 0;
            line-height: 1.35;
        }
        .lab-hero {
            position: relative;
            overflow: hidden;
            background: linear-gradient(125deg, rgba(30,27,75,0.95) 0%, rgba(15,23,42,0.92) 45%, rgba(17,94,89,0.35) 100%);
            color: var(--text);
            padding: 1.85rem 1.75rem 1.6rem 1.75rem;
            border-radius: 20px;
            margin-bottom: 1.5rem;
            border: 1px solid var(--border);
            box-shadow: 0 24px 48px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.06);
        }
        .lab-hero::before {
            content: "";
            position: absolute;
            top: -40%; right: -10%;
            width: 55%; height: 140%;
            background: radial-gradient(circle, rgba(129,140,248,0.25) 0%, transparent 65%);
            pointer-events: none;
        }
        .lab-hero-inner { position: relative; z-index: 1; }
        .lab-hero h1 {
            margin: 0;
            font-weight: 700;
            letter-spacing: -0.035em;
            font-size: clamp(1.45rem, 3vw, 2rem);
            background: linear-gradient(90deg, #fff 0%, #c7d2fe 50%, #99f6e4 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .lab-hero p {
            margin: 0.65rem 0 0 0;
            opacity: 0.9;
            font-size: 0.98rem;
            line-height: 1.55;
            color: #cbd5e1;
            max-width: 52rem;
        }
        .hero-chips {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 1.1rem;
        }
        .hero-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.35rem 0.75rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            font-family: 'JetBrains Mono', monospace;
            background: rgba(15,23,42,0.55);
            border: 1px solid rgba(129,140,248,0.25);
            color: #e2e8f0;
        }
        .hero-chip em { font-style: normal; color: #a5b4fc; }
        .section-wrap {
            display: flex;
            gap: 1rem;
            align-items: flex-start;
            margin: 1.75rem 0 0.85rem 0;
            padding-bottom: 0.5rem;
        }
        .section-rail {
            width: 4px;
            min-height: 2.5rem;
            border-radius: 4px;
            background: linear-gradient(180deg, var(--accent) 0%, var(--accent-2) 100%);
            flex-shrink: 0;
            margin-top: 0.2rem;
        }
        .section-rail.pink { background: linear-gradient(180deg, var(--accent-3) 0%, var(--accent) 100%); }
        .section-rail.teal { background: linear-gradient(180deg, var(--accent-2) 0%, #2dd4bf 100%); }
        .section-badge {
            display: inline-block;
            font-size: 0.65rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: #a5b4fc;
            margin-bottom: 0.25rem;
            font-family: 'JetBrains Mono', monospace;
        }
        .section-h2 {
            margin: 0;
            font-size: 1.2rem;
            font-weight: 700;
            color: #f8fafc;
            letter-spacing: -0.02em;
        }
        .section-p {
            margin: 0.35rem 0 0 0;
            font-size: 0.88rem;
            color: var(--muted);
            line-height: 1.5;
            max-width: 48rem;
        }
        .panel {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1rem 1.15rem;
            margin-bottom: 1rem;
            backdrop-filter: blur(12px);
        }
        .step-pill {
            display: inline-block;
            background: rgba(99, 102, 241, 0.22);
            color: #c7d2fe;
            padding: 0.2rem 0.65rem;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 600;
            margin-right: 0.35rem;
            font-family: 'JetBrains Mono', monospace;
        }
        div[data-testid="stMetricValue"] {
            font-variant-numeric: tabular-nums;
            color: #f8fafc !important;
        }
        div[data-testid="stMetricLabel"] { color: #94a3b8 !important; }
        .stTabs [data-baseweb="tab-list"] {
            gap: 6px;
            background: rgba(15,23,42,0.5);
            padding: 6px;
            border-radius: 14px;
            border: 1px solid var(--border);
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 10px !important;
            padding: 0.55rem 1.1rem !important;
            font-weight: 600 !important;
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, rgba(99,102,241,0.45) 0%, rgba(45,212,191,0.2) 100%) !important;
            border: 1px solid rgba(129,140,248,0.4) !important;
        }
        div[data-testid="stExpander"] details {
            background: rgba(15,23,42,0.4);
            border: 1px solid rgba(129,140,248,0.15);
            border-radius: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _section_header(badge: str, title: str, subtitle: str, rail: str = "default") -> None:
    rail_cls = "section-rail"
    if rail == "pink":
        rail_cls += " pink"
    elif rail == "teal":
        rail_cls += " teal"
    st.markdown(
        f'<div class="section-wrap"><div class="{rail_cls}"></div><div>'
        f'<span class="section-badge">{badge}</span>'
        f'<h2 class="section-h2">{title}</h2>'
        f'<p class="section-p">{subtitle}</p></div></div>',
        unsafe_allow_html=True,
    )


def _sidebar_segment(title: str, subtitle: str = "") -> None:
    sub = f'<p class="sb-seg-sub">{subtitle}</p>' if subtitle else ""
    st.markdown(
        f'<div class="sb-seg"><p class="sb-seg-title">{title}</p>{sub}</div>',
        unsafe_allow_html=True,
    )


def _apply_chart_theme(fig) -> None:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(17, 24, 39, 0.88)",
        plot_bgcolor="rgba(15, 23, 42, 0.55)",
        font=dict(family="DM Sans, sans-serif", color="#e2e8f0", size=12),
        title_font=dict(size=15),
        margin=dict(l=48, r=24, t=56, b=48),
    )


def confusion_fig(y_true, y_pred, title: str) -> Any:
    labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig = px.imshow(
        cm,
        text_auto=True,
        labels=dict(x="Predicted", y="True", color="Count"),
        x=[str(x) for x in labels],
        y=[str(x) for x in labels],
        color_continuous_scale="Viridis",
        title=title,
    )
    fig.update_layout(margin=dict(l=40, r=40, t=50, b=40))
    return fig


def _as_experiment_cfg(obj: Any) -> ExperimentConfig:
    if isinstance(obj, ExperimentConfig):
        return obj
    if isinstance(obj, dict):
        return ExperimentConfig(**obj)
    raise TypeError(f"Cannot coerce cfg from {type(obj)!r}")


def _reload_pool_holdout_for_run(r: dict) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, ExperimentConfig]:
    csv_path = Path(r["csv_path"])
    feature_names, X, y = load_combined_xy(csv_path)
    cfg = _as_experiment_cfg(r["cfg"])
    X_pool, X_holdout, y_pool, y_holdout = split_pool_holdout(X, y, cfg)
    return feature_names, X_pool, X_holdout, y_pool, y_holdout, cfg


def _interpretability_cfg(r: dict) -> ExperimentConfig:
    """MDA/SHAP knobs from sidebar, everything else from the saved run."""
    base = _as_experiment_cfg(r["cfg"])
    return replace(
        base,
        mda_n_repeats=int(st.session_state.get("sb_mda", base.mda_n_repeats)),
        shap_max_background=int(st.session_state.get("sb_sbg", base.shap_max_background)),
        shap_max_explain=int(st.session_state.get("sb_sex", base.shap_max_explain)),
        shap_class_index=int(st.session_state.get("sb_sci", base.shap_class_index)),
    )


def compute_mda_plots(r: dict, icfg: ExperimentConfig) -> tuple[Any, pd.DataFrame]:
    fn, Xp, Xh, yp, yh, cfg = _reload_pool_holdout_for_run(r)
    clf, _X_pi, X_hi = fit_simple_net_on_pool_for_holdout(
        r["model_id"], Xp, Xh, yp, cfg, r["est_kw"]
    )
    mda_df = run_permutation_mda(clf, X_hi, yh, fn, cfg=icfg)
    fig = px.violin(
        mda_df,
        x="feature",
        y="mda",
        box=True,
        points="all",
        title=f"MDA (permutation) — holdout · {MODEL_LABELS[r['model_id']]}",
    )
    fig.update_xaxes(tickangle=-45)
    return fig, mda_df


def compute_shap_plots(r: dict, icfg: ExperimentConfig) -> tuple[Any, pd.DataFrame]:
    fn, Xp, Xh, yp, yh, cfg = _reload_pool_holdout_for_run(r)
    clf, X_pi, X_hi = fit_simple_net_on_pool_for_holdout(
        r["model_id"], Xp, Xh, yp, cfg, r["est_kw"]
    )
    shap_df = run_shap_proba(clf, X_pi, X_hi, fn, cfg=icfg)
    fig = px.violin(
        shap_df,
        x="feature",
        y="shap",
        box=True,
        points=False,
        title=f"SHAP · P(class=1) — {MODEL_LABELS[r['model_id']]}",
    )
    fig.update_xaxes(tickangle=-45)
    return fig, shap_df


def recompute_oof_from_run(r: dict) -> dict:
    """Re-run stratified OOF on the pool using saved est_kw (same split as training)."""
    _, Xp, _, yp, _, cfg = _reload_pool_holdout_for_run(r)
    y_true, y_pred, y_score = pool_cv_oof_predictions(r["model_id"], Xp, yp, cfg, r["est_kw"])
    metrics = oof_scalar_metrics(y_true, y_pred, y_score)
    roc_fig, _ = roc_oob_figure(
        y_true,
        y_score,
        title_prefix=f"ROC — OOF · {MODEL_LABELS[r['model_id']]}",
    )
    cm_fig = confusion_fig(
        y_true,
        y_pred,
        title=f"Confusion matrix (OOF) — {MODEL_LABELS[r['model_id']]}",
    )
    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_score": y_score,
        "metrics": metrics,
        "roc_fig": roc_fig,
        "cm_fig": cm_fig,
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
    }


MODEL_LABELS: dict[str, str] = {
    "mlp": "Neural network (MLP)",
    "knn": "KNN",
    "logistic": "Logistic regression",
    "rf": "Random forest",
    "catboost": "CatBoost",
    "svm": "RBF SVM",
    "xgboost": "XGBoost",
}


def run_single_model(
    model_id: str,
    cfg: ExperimentConfig,
    csv_path: Path,
    *,
    progress_placeholder,
    status,
    do_mda_shap: bool,
) -> dict:
    """Execute full pipeline for one model; return serializable-ish result dict."""
    t0 = time.perf_counter()

    status.update(label="Loading CSV & stratified pool / holdout…", state="running")
    feature_names, X, y = load_combined_xy(csv_path)
    X_pool, X_holdout, y_pool, y_holdout = split_pool_holdout(X, y, cfg)

    n_trials = cfg.optuna_n_trials

    def _cb(study_: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if progress_placeholder is None:
            return
        best = study_.best_value if study_.best_trial is not None else float("nan")
        progress_placeholder.progress(
            min((trial.number + 1) / max(n_trials, 1), 1.0),
            text=f"Optuna trial {trial.number + 1}/{n_trials} · best nested-CV ROC-AUC = {best:.5f}",
        )

    status.update(label=f"Nested-CV Optuna ({MODEL_LABELS[model_id]})…", state="running")
    study, est_kw = get_simple_net_tune_fn(model_id)(
        X_pool,
        y_pool,
        cfg,
        optuna_callbacks=[_cb],
        show_progress_bar=False,
    )
    if progress_placeholder is not None:
        progress_placeholder.empty()

    status.update(label="Stratified OOF CV on train pool (honest imputer per fold)…", state="running")
    y_true, y_pred, y_score = pool_cv_oof_predictions(model_id, X_pool, y_pool, cfg, est_kw)
    metrics = oof_scalar_metrics(y_true, y_pred, y_score)
    roc_fig, _auc = roc_oob_figure(
        y_true,
        y_score,
        title_prefix=f"ROC — OOF · {MODEL_LABELS[model_id]}",
    )

    mda_fig, shap_fig = None, None
    mda_df, shap_df = None, None
    if do_mda_shap:
        status.update(label="Fitting on full pool → MDA & SHAP on holdout…", state="running")
        clf_final, X_pi, X_hi = fit_simple_net_on_pool_for_holdout(
            model_id, X_pool, X_holdout, y_pool, cfg, est_kw
        )
        mda_df = run_permutation_mda(clf_final, X_hi, y_holdout, feature_names, cfg=cfg)
        mda_fig = px.violin(
            mda_df,
            x="feature",
            y="mda",
            box=True,
            points="all",
            title=f"MDA (permutation) — holdout · {MODEL_LABELS[model_id]}",
        )
        mda_fig.update_xaxes(tickangle=-45)
        shap_df = run_shap_proba(clf_final, X_pi, X_hi, feature_names, cfg=cfg)
        shap_fig = px.violin(
            shap_df,
            x="feature",
            y="shap",
            box=True,
            points=False,
            title=f"SHAP · P(class=1) — {MODEL_LABELS[model_id]}",
        )
        shap_fig.update_xaxes(tickangle=-45)

    cm_fig = confusion_fig(
        y_true,
        y_pred,
        title=f"Confusion matrix (OOF) — {MODEL_LABELS[model_id]}",
    )
    run_cfg_json = build_run_config_json(
        cfg=cfg,
        estimator_kw=est_kw,
        optuna_best_value=float(study.best_value),
        extra={"model_id": model_id, "dashboard": "streamlit_dashboard.py"},
    )
    elapsed = time.perf_counter() - t0
    status.update(label=f"Done · {MODEL_LABELS[model_id]} ({elapsed:.0f}s)", state="complete")

    return {
        "model_id": model_id,
        "label": MODEL_LABELS[model_id],
        "csv_path": str(csv_path.resolve()),
        "cfg": cfg,
        "feature_names": feature_names,
        "est_kw": est_kw,
        "study_best_value": float(study.best_value),
        "n_trials_actual": len(study.trials),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_score": y_score,
        "metrics": metrics,
        "roc_fig": roc_fig,
        "cm_fig": cm_fig,
        "mda_fig": mda_fig,
        "shap_fig": shap_fig,
        "mda_df": mda_df,
        "shap_df": shap_df,
        "has_mda": mda_fig is not None,
        "has_shap": shap_fig is not None,
        "run_config_json": run_cfg_json,
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
        "elapsed_sec": elapsed,
        "pool_n": int(len(y_pool)),
        "holdout_n": int(len(y_holdout)),
    }


def build_cfg_from_sidebar() -> ExperimentConfig:
    return replace(
        EXPERIMENT,
        random_state=int(st.session_state.get("sb_rs", EXPERIMENT.random_state)),
        test_size=float(st.session_state.get("sb_ts", EXPERIMENT.test_size)),
        cv_n_splits=int(st.session_state.get("sb_cv", EXPERIMENT.cv_n_splits)),
        cv_shuffle=bool(st.session_state.get("sb_cvsh", EXPERIMENT.cv_shuffle)),
        imputer_strategy=str(st.session_state.get("sb_imp", EXPERIMENT.imputer_strategy)),
        mda_n_repeats=int(st.session_state.get("sb_mda", EXPERIMENT.mda_n_repeats)),
        shap_max_background=int(st.session_state.get("sb_sbg", EXPERIMENT.shap_max_background)),
        shap_max_explain=int(st.session_state.get("sb_sex", EXPERIMENT.shap_max_explain)),
        shap_class_index=int(st.session_state.get("sb_sci", EXPERIMENT.shap_class_index)),
        optuna_n_trials=int(st.session_state.get("sb_ntr", EXPERIMENT.optuna_n_trials)),
        optuna_outer_splits=int(st.session_state.get("sb_out", EXPERIMENT.optuna_outer_splits)),
        optuna_inner_splits=int(st.session_state.get("sb_inn", EXPERIMENT.optuna_inner_splits)),
        optuna_show_progress=bool(st.session_state.get("sb_osp", False)),
    )


def main() -> None:
    st.set_page_config(
        page_title="ML training lab",
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon="◈",
    )
    _inject_css()

    st.session_state.setdefault("results", {})
    n_session_runs = len(st.session_state["results"])

    st.markdown(
        f"""
        <div class="lab-hero"><div class="lab-hero-inner">
        <h1>Cardiac risk · training laboratory</h1>
        <p>Mirror of the <code>simple_net_*.py</code> stack: <strong>nested Optuna</strong> on the train pool only,
        then <strong>stratified OOF</strong> for honest metrics, then <strong>MDA / SHAP</strong> on the holdout
        when you ask for them.</p>
        <div class="hero-chips">
            <span class="hero-chip">Session runs <em>{n_session_runs}</em></span>
            <span class="hero-chip">Nested CV <em>pool-only</em></span>
            <span class="hero-chip">Holdout <em>never</em> in HPO</span>
        </div>
        </div></div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown(
            '<p style="font-size:1.05rem;font-weight:700;color:#e2e8f0;margin:0 0 0.25rem 0;">Control deck</p>'
            '<p style="font-size:0.78rem;color:#94a3b8;margin:0 0 0.5rem 0;line-height:1.4;">Configure, then launch training.</p>',
            unsafe_allow_html=True,
        )
        _sidebar_segment("01 · Data source", "Path or upload; must include a target column.")
        csv_str = st.text_input(
            "CSV path",
            value=str(DATA_COMBINED_DEFAULT),
            help="Table with a `target` column; features are all other columns.",
        )
        csv_path = Path(csv_str).expanduser()
        up = st.file_uploader("Or upload CSV", type=["csv"])
        if up is not None:
            tmp_dir = Path(st.session_state.setdefault("_upload_dir", str(_MODELS_DIR / "_streamlit_uploads")))
            tmp_dir.mkdir(parents=True, exist_ok=True)
            save_p = tmp_dir / up.name
            save_p.write_bytes(up.getbuffer())
            csv_path = save_p
            st.caption(f"Using upload: `{save_p.name}`")

        _sidebar_segment("02 · Splits & preprocessing", "Stratified pool/holdout, imputer, OOF folds.")
        c1, c2 = st.columns(2)
        with c1:
            st.number_input("Random seed", key="sb_rs", min_value=0, max_value=2**31 - 1, value=42, step=1)
        with c2:
            st.slider("Holdout fraction", key="sb_ts", min_value=0.05, max_value=0.5, value=0.2, step=0.05)
        st.number_input("OOF CV folds (pool)", key="sb_cv", min_value=2, max_value=15, value=5)
        st.checkbox("Shuffle CV folds", key="sb_cvsh", value=True)
        st.selectbox(
            "Imputer strategy",
            options=["median", "mean", "most_frequent"],
            key="sb_imp",
            index=0,
        )

        _sidebar_segment("03 · Hyperparameter search", "Nested CV; holdout excluded from tuning.")
        st.number_input("Trials", key="sb_ntr", min_value=1, max_value=500, value=24, step=1)
        c3, c4 = st.columns(2)
        with c3:
            st.number_input("Outer splits", key="sb_out", min_value=2, max_value=10, value=4)
        with c4:
            st.number_input("Inner splits", key="sb_inn", min_value=2, max_value=10, value=3)
        st.checkbox("Optuna tqdm bar (terminal)", key="sb_osp", value=False)

        _sidebar_segment("04 · Interpretability knobs", "Used by Model lab buttons or optional train-time pass.")
        st.caption("Prefer on-demand actions in **Model lab** after training.")
        st.checkbox("Also run MDA + SHAP during training", key="sb_mda_shap", value=False)
        st.number_input("MDA shuffle repeats / feature", key="sb_mda", min_value=5, max_value=200, value=50)
        st.number_input("SHAP max background", key="sb_sbg", min_value=20, max_value=500, value=100)
        st.number_input("SHAP max explain rows", key="sb_sex", min_value=20, max_value=500, value=150)
        st.number_input("SHAP class index", key="sb_sci", min_value=0, max_value=4, value=1)

        _sidebar_segment("05 · Launch", "Pick estimators and start the pipeline.")
        model_ids = list(MODEL_LABELS.keys())
        chosen = st.multiselect(
            "Select one or more",
            options=model_ids,
            default=["logistic"],
            format_func=lambda x: MODEL_LABELS[x],
        )
        run_btn = st.button("Run pipeline", type="primary", use_container_width=True)
        if st.button("Clear cached results", use_container_width=True):
            st.session_state["results"] = {}
            st.rerun()

    cfg_preview = build_cfg_from_sidebar()
    _section_header(
        "Workspace",
        "Configuration snapshot",
        "Exactly what `ExperimentConfig` will see for the next training job — seeds, splits, Optuna budget, and SHAP/MDA limits.",
    )
    with st.expander("View resolved `ExperimentConfig` (JSON)", expanded=False):
        st.json(json.loads(json.dumps(asdict(cfg_preview), default=str)))

    tab_board, tab_explorer, tab_pipeline = st.tabs(
        ["◆  Leaderboard", "◇  Model lab", "◎  Pipeline story"]
    )

    if run_btn and chosen:
        if not csv_path.is_file():
            st.error(f"CSV not found: `{csv_path}`")
        else:
            progress = st.progress(0.0, text="Ready")
            for mid in chosen:
                key = f"{mid}_{uuid.uuid4().hex[:10]}"
                with st.status(f"Training · {MODEL_LABELS[mid]}", expanded=True) as status:
                    try:
                        res = run_single_model(
                            mid,
                            cfg_preview,
                            csv_path,
                            progress_placeholder=progress,
                            status=status,
                            do_mda_shap=st.session_state["sb_mda_shap"],
                        )
                        st.session_state["results"][key] = res
                        st.success(f"Finished {MODEL_LABELS[mid]} in {res['elapsed_sec']:.1f}s")
                    except Exception as e:
                        status.update(label=f"Failed · {MODEL_LABELS[mid]}", state="error")
                        st.error(f"{type(e).__name__}: {e}")
                        with st.expander("Traceback"):
                            st.code(traceback.format_exc())

    results: dict = st.session_state["results"]

    with tab_board:
        _section_header(
            "Segment A · Session ranking",
            "Leaderboard",
            "Out-of-fold metrics on the train pool for every run in this browser session — sortable table plus a quick visual.",
            rail="default",
        )
        if not results:
            st.info("Train at least one model from the sidebar to populate this board.")
        else:
            rows = []
            for k, r in results.items():
                m = r["metrics"]
                rows.append(
                    {
                        "run_key": k,
                        "model": r["label"],
                        "AUC-ROC": m["auc_roc"],
                        "Accuracy": m["accuracy"],
                        "F1 (weighted)": m["f1"],
                        "Precision": m["precision"],
                        "Recall": m["recall"],
                        "Optuna nested CV AUC": r["study_best_value"],
                        "Time (s)": round(r["elapsed_sec"], 1),
                        "MDA": "✓" if (r.get("has_mda") or r.get("mda_fig") is not None) else "—",
                        "SHAP": "✓" if (r.get("has_shap") or r.get("shap_fig") is not None) else "—",
                    }
                )
            df = pd.DataFrame(rows).sort_values("AUC-ROC", ascending=False)
            with st.container(border=True):
                st.dataframe(df, use_container_width=True, hide_index=True)
            _section_header(
                "Segment B · Distribution",
                "AUC at a glance",
                "Grouped bars by model name for this session (multiple runs of the same family appear as separate x positions).",
                rail="teal",
            )
            fig = px.bar(
                df,
                x="model",
                y="AUC-ROC",
                color="model",
                title="OOF AUC-ROC (this session)",
                text_auto=".3f",
                color_discrete_sequence=px.colors.qualitative.Bold,
            )
            fig.update_layout(showlegend=False, yaxis_range=[0, 1])
            _apply_chart_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

    with tab_explorer:
        _section_header(
            "Model laboratory",
            "Per-run inspection",
            "Pick a session run, read the OOF snapshot, fire interpretability jobs, and export artifacts — each block is a separate segment below.",
            rail="pink",
        )
        if not results:
            st.info("No runs yet — start one from the sidebar.")
        else:
            with st.container(border=True):
                pick = st.selectbox(
                    "Session run",
                    options=list(results.keys()),
                    format_func=lambda k: (
                        f"{results[k]['label']} — {k[:20]}…"
                        if len(k) > 24
                        else f"{results[k]['label']} — {k}"
                    ),
                )
            r = results[pick]
            m = r["metrics"]

            _section_header(
                "Segment 1 · Performance snapshot",
                "Out-of-fold headline metrics",
                "These numbers come from stratified CV on the train pool with the tuned estimator — not the holdout.",
                rail="teal",
            )
            with st.container(border=True):
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("AUC-ROC", f"{m['auc_roc']:.4f}")
                c2.metric("Accuracy", f"{m['accuracy']:.4f}")
                c3.metric("F1 (weighted)", f"{m['f1']:.4f}")
                c4.metric("Nested-CV score", f"{r['study_best_value']:.4f}")
                c5.metric("Wall time", f"{r['elapsed_sec']:.0f}s")

            miss_csv = "csv_path" not in r
            st.caption(
                f"Pool **{r['pool_n']}** rows · holdout **{r['holdout_n']}** · "
                f"{r['n_trials_actual']} Optuna trials"
                + (
                    f" · data `{Path(r['csv_path']).name}`"
                    if not miss_csv
                    else " · (re-train to enable on-demand MDA/SHAP — missing `csv_path`)"
                )
            )

            _section_header(
                "Segment 2 · Actions",
                "On-demand compute",
                "Each action reloads the CSV, refits on the **full pool** with saved hyperparameters, then runs the requested step (or re-runs OOF only).",
            )
            icfg = _interpretability_cfg(r)
            b1, b2, b3, b4, b5 = st.columns(5)
            with b1:
                run_mda = st.button(
                    "Run MDA",
                    key=f"act_mda_{pick}",
                    help="Permutation MDA on holdout (slow)",
                    disabled=miss_csv,
                )
            with b2:
                run_shap = st.button(
                    "Run SHAP",
                    key=f"act_shap_{pick}",
                    help="SHAP for P(class=1); pool background",
                    disabled=miss_csv,
                )
            with b3:
                run_both = st.button(
                    "MDA + SHAP",
                    key=f"act_both_{pick}",
                    type="primary",
                    disabled=miss_csv,
                )
            with b4:
                run_oob = st.button(
                    "Re-run OOF CV",
                    key=f"act_oob_{pick}",
                    help="Same est_kw, same split as training",
                    disabled=miss_csv,
                )
            with b5:
                export_csv = st.button("Save metrics CSV", key=f"act_csv_{pick}", help=f"Write under {METRICS_ROOT}")

            if run_mda and not miss_csv:
                with st.spinner("Computing MDA on holdout…"):
                    try:
                        fig, df_m = compute_mda_plots(r, icfg)
                        r = {**r, "mda_fig": fig, "mda_df": df_m, "has_mda": True}
                        st.session_state["results"][pick] = r
                        st.success("MDA complete.")
                    except Exception as e:
                        st.error(f"{type(e).__name__}: {e}")
                st.rerun()

            if run_shap and not miss_csv:
                with st.spinner("Computing SHAP…"):
                    try:
                        fig, df_s = compute_shap_plots(r, icfg)
                        r = {**r, "shap_fig": fig, "shap_df": df_s, "has_shap": True}
                        st.session_state["results"][pick] = r
                        st.success("SHAP complete.")
                    except Exception as e:
                        st.error(f"{type(e).__name__}: {e}")
                st.rerun()

            if run_both and not miss_csv:
                with st.spinner("Single pool refit → MDA + SHAP…"):
                    try:
                        fn, Xp, Xh, yp, yh, cfg = _reload_pool_holdout_for_run(r)
                        clf, X_pi, X_hi = fit_simple_net_on_pool_for_holdout(
                            r["model_id"], Xp, Xh, yp, cfg, r["est_kw"]
                        )
                        mda_df = run_permutation_mda(clf, X_hi, yh, fn, cfg=icfg)
                        shap_df = run_shap_proba(clf, X_pi, X_hi, fn, cfg=icfg)
                        fig_m = px.violin(
                            mda_df,
                            x="feature",
                            y="mda",
                            box=True,
                            points="all",
                            title=f"MDA (permutation) — holdout · {MODEL_LABELS[r['model_id']]}",
                        )
                        fig_m.update_xaxes(tickangle=-45)
                        fig_s = px.violin(
                            shap_df,
                            x="feature",
                            y="shap",
                            box=True,
                            points=False,
                            title=f"SHAP · P(class=1) — {MODEL_LABELS[r['model_id']]}",
                        )
                        fig_s.update_xaxes(tickangle=-45)
                        _apply_chart_theme(fig_m)
                        _apply_chart_theme(fig_s)
                        st.session_state["results"][pick] = {
                            **r,
                            "mda_fig": fig_m,
                            "mda_df": mda_df,
                            "shap_fig": fig_s,
                            "shap_df": shap_df,
                            "has_mda": True,
                            "has_shap": True,
                        }
                        st.success("MDA + SHAP complete.")
                    except Exception as e:
                        st.error(f"{type(e).__name__}: {e}")
                st.rerun()

            if run_oob and not miss_csv:
                with st.spinner("Re-running stratified OOF CV…"):
                    try:
                        updates = recompute_oof_from_run(r)
                        st.session_state["results"][pick] = {**r, **updates}
                        st.success("OOF metrics refreshed.")
                    except Exception as e:
                        st.error(f"{type(e).__name__}: {e}")
                st.rerun()

            if export_csv:
                try:
                    safe = "".join(c if c.isalnum() else "_" for c in pick)[:48]
                    out_p = METRICS_ROOT / f"dashboard_{safe}.csv"
                    METRICS_ROOT.mkdir(parents=True, exist_ok=True)
                    write_performance_metrics_csv(
                        out_p,
                        model_name=f"dashboard_{r['model_id']}_{safe[-8:]}",
                        y_true=r["y_true"],
                        y_pred=r["y_pred"],
                        y_score=r["y_score"],
                        run_config_json=r["run_config_json"],
                    )
                    st.success(f"Wrote `{out_p}`")
                except Exception as e:
                    st.error(f"{type(e).__name__}: {e}")

            r = st.session_state["results"][pick]

            _section_header(
                "Segment 3 · Diagnostics",
                "ROC & confusion (OOF)",
                "Calibration-style ROC uses pooled out-of-fold scores; confusion matrix uses the same OOF predictions.",
                rail="default",
            )
            ec1, ec2 = st.columns(2)
            with ec1:
                _f_roc = r["roc_fig"]
                _apply_chart_theme(_f_roc)
                st.plotly_chart(_f_roc, use_container_width=True)
            with ec2:
                _f_cm = r["cm_fig"]
                _apply_chart_theme(_f_cm)
                st.plotly_chart(_f_cm, use_container_width=True)

            _section_header(
                "Segment 4 · Tuned hyperparameters",
                "Best `estimator_kw` from Optuna",
                "These kwargs are what the scripts persist into `run_config_json` alongside `ExperimentConfig`.",
                rail="teal",
            )
            with st.container(border=True):
                st.json(r["est_kw"])

            with st.expander("Classification report (OOF text)"):
                st.text(r["classification_report"])

            with st.expander("Reproducibility bundle"):
                st.download_button(
                    "Download run_config_json",
                    data=r["run_config_json"],
                    file_name=f"run_config_{r['model_id']}.json",
                    mime="application/json",
                    key=f"dl_cfg_{pick}",
                )
                st.code(r["run_config_json"], language="json")

            _section_header(
                "Segment 5 · Interpretability",
                "MDA & SHAP on the holdout",
                "Violins appear after you run the actions in Segment 2 (or if you enabled train-time MDA+SHAP). Download long-form CSVs for offline plots.",
                rail="pink",
            )
            if r.get("mda_fig") is not None:
                _f_m = r["mda_fig"]
                _apply_chart_theme(_f_m)
                st.plotly_chart(_f_m, use_container_width=True)
                if r.get("mda_df") is not None:
                    st.download_button(
                        "Download MDA (long CSV)",
                        data=r["mda_df"].to_csv(index=False).encode("utf-8"),
                        file_name=f"mda_long_{r['model_id']}.csv",
                        mime="text/csv",
                        key=f"dl_mda_{pick}",
                    )
            if r.get("shap_fig") is not None:
                _f_s = r["shap_fig"]
                _apply_chart_theme(_f_s)
                st.plotly_chart(_f_s, use_container_width=True)
                if r.get("shap_df") is not None:
                    st.download_button(
                        "Download SHAP (long CSV)",
                        data=r["shap_df"].to_csv(index=False).encode("utf-8"),
                        file_name=f"shap_long_{r['model_id']}.csv",
                        mime="text/csv",
                        key=f"dl_shap_{pick}",
                    )
            if r.get("mda_fig") is None and r.get("shap_fig") is None:
                st.info(
                    "Nothing here yet — use **Run MDA**, **Run SHAP**, or **MDA + SHAP** in Segment 2. "
                    "Sidebar controls set MDA repeats and SHAP row caps."
                )

    with tab_pipeline:
        _section_header(
            "Pipeline atlas",
            "End-to-end story",
            "Five stages, matching `simple_net_*.py` and `optuna_hpo.py`. Each stage is isolated so you can explain it to collaborators.",
            rail="teal",
        )
        _section_header(
            "Step 1 · Ingest",
            "Load & validate",
            "Read a CSV with a `target` column; every other column is cast to float features (same as `load_combined_xy`).",
        )
        _section_header(
            "Step 2 · Split",
            "Stratified pool vs holdout",
            "`train_test_split` with `stratify=y` and your sidebar seed. The holdout is reserved for MDA/SHAP only — it never enters Optuna.",
        )
        _section_header(
            "Step 3 · Tune",
            "Nested CV on the pool",
            "Optuna proposes hyperparameters; each trial is scored by mean ROC-AUC over outer×inner stratified folds. Imputers (and scalers for KNN/SVM) fit only on inner-train rows.",
        )
        _section_header(
            "Step 4 · Report",
            "Honest OOF metrics",
            "With the best `estimator_kw`, a stratified K-fold pass over the full pool builds out-of-fold predictions for ROC, confusion, and scalar metrics.",
        )
        _section_header(
            "Step 5 · Explain",
            "Optional interpretability",
            "Refit on the entire pool, score the holdout with MDA shuffles, and explain with SHAP (background from pool, explain rows from holdout). Trigger from **Model lab → Segment 2**, or enable during training.",
            rail="pink",
        )
        with st.container(border=True):
            st.markdown(
                "Defaults mirror **`ExperimentConfig`** in `model_run_utils.py` until you override them in the sidebar."
            )


if __name__ == "__main__":
    main()
