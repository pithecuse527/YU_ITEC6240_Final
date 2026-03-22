"""Shared helpers for simple_net-style training scripts: MDA, SHAP, ROC figures, CSV + HTML exports."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve

MODELS_DIR = Path(__file__).resolve().parent
OUTPUTS_ROOT = MODELS_DIR / "outputs"
METRICS_ROOT = MODELS_DIR / "performance_metrics"
DATA_COMBINED_DEFAULT = MODELS_DIR.parent / "data" / "combined.csv"


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """One shared setup for every `simple_net_*.py`: splits, CV, imputation, MDA/SHAP, RNG seeds."""

    random_state: int = 42
    test_size: float = 0.2
    cv_n_splits: int = 5
    cv_shuffle: bool = True
    imputer_strategy: str = "median"
    mda_n_repeats: int = 50
    shap_max_background: int = 100
    shap_max_explain: int = 150
    shap_class_index: int = 1


EXPERIMENT = ExperimentConfig()


def split_pool_holdout(X, y, cfg: ExperimentConfig = EXPERIMENT):
    from sklearn.model_selection import train_test_split

    return train_test_split(
        X, y, test_size=cfg.test_size, stratify=y, random_state=cfg.random_state
    )


def make_stratified_kfold(cfg: ExperimentConfig = EXPERIMENT):
    from sklearn.model_selection import StratifiedKFold

    return StratifiedKFold(
        n_splits=cfg.cv_n_splits,
        shuffle=cfg.cv_shuffle,
        random_state=cfg.random_state,
    )


def make_imputer(cfg: ExperimentConfig = EXPERIMENT):
    from sklearn.impute import SimpleImputer

    return SimpleImputer(strategy=cfg.imputer_strategy)


def mda_rng(cfg: ExperimentConfig = EXPERIMENT):
    return np.random.default_rng(cfg.random_state)


def load_combined_xy(csv_path: Path | str | None = None):
    """Load `combined.csv`-style table; return feature column names, X (float), y."""
    path = Path(csv_path) if csv_path is not None else DATA_COMBINED_DEFAULT
    df = pd.read_csv(path)
    feature_names = df.drop(columns=["target"]).columns.tolist()
    X = df.drop(columns=["target"]).to_numpy(dtype=float)
    y = df["target"].to_numpy()
    return feature_names, X, y


def run_permutation_mda(
    clf, X_test, y_test, feature_names, *, rng=None, n_repeats: int | None = None, cfg: ExperimentConfig = EXPERIMENT
):
    """MDA via repeated single-feature shuffles on a fixed test set. Returns long DataFrame (feature, mda)."""
    if rng is None:
        rng = np.random.default_rng(cfg.random_state)
    if n_repeats is None:
        n_repeats = cfg.mda_n_repeats
    X_test = np.asarray(X_test, dtype=float)
    base = accuracy_score(y_test, clf.predict(X_test))
    rows = []
    for _ in range(n_repeats):
        Xp = X_test.copy()
        for j, name in enumerate(feature_names):
            col = Xp[:, j].copy()
            rng.shuffle(Xp[:, j])
            rows.append(
                {"feature": name, "mda": base - accuracy_score(y_test, clf.predict(Xp))}
            )
            Xp[:, j] = col
    return pd.DataFrame(rows)


def run_shap_proba(
    clf,
    X_background,
    X_explain,
    feature_names,
    *,
    class_index: int | None = None,
    max_background: int | None = None,
    max_explain: int | None = None,
    random_state: int | None = None,
    cfg: ExperimentConfig = EXPERIMENT,
):
    """SHAP for sklearn `predict_proba` (tabular). Returns long DataFrame (sample, feature, shap)."""
    import shap

    if class_index is None:
        class_index = cfg.shap_class_index
    if max_background is None:
        max_background = cfg.shap_max_background
    if max_explain is None:
        max_explain = cfg.shap_max_explain
    if random_state is None:
        random_state = cfg.random_state

    X_bg = np.asarray(X_background, dtype=float)
    X_ex = np.asarray(X_explain, dtype=float)
    rng = np.random.default_rng(random_state)
    if max_explain is not None and X_ex.shape[0] > max_explain:
        X_ex = X_ex[rng.choice(X_ex.shape[0], max_explain, replace=False)]
    if len(X_bg) > max_background:
        X_bg = X_bg[rng.choice(len(X_bg), max_background, replace=False)]
    explainer = shap.Explainer(clf.predict_proba, X_bg)
    exp = explainer(X_ex)
    vals = exp.values[:, :, class_index]
    rows = []
    for i in range(vals.shape[0]):
        for j, name in enumerate(feature_names):
            rows.append({"sample": i, "feature": name, "shap": vals[i, j]})
    return pd.DataFrame(rows)


def roc_oob_figure(y_true, y_score, *, title_prefix: str):
    """ROC curve (Plotly); `y_score` is positive-class probability or score."""
    auc = float(roc_auc_score(y_true, y_score))
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig = px.line(
        pd.DataFrame({"fpr": fpr, "tpr": tpr}),
        x="fpr",
        y="tpr",
        title=f"{title_prefix} (AUC = {auc:.3f})",
        labels={"fpr": "False positive rate", "tpr": "True positive rate"},
    )
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="gray"))
    return fig, auc


def write_performance_metrics_csv(
    csv_path: Path,
    *,
    model_name: str,
    y_true,
    y_pred,
    y_score,
) -> None:
    """One row: accuracy, AUC-ROC, precision/recall/F1 (weighted average over classes)."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    w = rep["weighted avg"]
    row = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "auc_roc": float(roc_auc_score(y_true, y_score)),
        "precision": w["precision"],
        "recall": w["recall"],
        "f1": w["f1-score"],
    }
    pd.DataFrame([row]).to_csv(csv_path, index=False)


def persist_run_artifacts(
    script_stem: str,
    *,
    y_true,
    y_pred,
    y_score,
    roc_fig,
    fig_mda,
    fig_shap,
    mda_df: pd.DataFrame,
    shap_df: pd.DataFrame,
    show_plots: bool | None = None,
) -> Path:
    """
    Writes:
      - performance_metrics/{script_stem}.csv
      - outputs/{script_stem}/mda_long.csv, shap_long.csv, *.html plots
    If show_plots is True, or None and running under ipykernel, also fig.show().
    Returns the per-script output directory.
    """
    out_dir = OUTPUTS_ROOT / script_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    METRICS_ROOT.mkdir(parents=True, exist_ok=True)

    write_performance_metrics_csv(
        METRICS_ROOT / f"{script_stem}.csv",
        model_name=script_stem,
        y_true=y_true,
        y_pred=y_pred,
        y_score=y_score,
    )
    mda_df.to_csv(out_dir / "mda_long.csv", index=False)
    shap_df.to_csv(out_dir / "shap_long.csv", index=False)
    roc_fig.write_html(out_dir / "roc_oob.html")
    fig_mda.write_html(out_dir / "mda_violin.html")
    fig_shap.write_html(out_dir / "shap_violin.html")

    do_show = (show_plots if show_plots is not None else ("ipykernel" in sys.modules))
    if do_show:
        roc_fig.show()
        fig_mda.show()
        fig_shap.show()
    else:
        print(f"Wrote metrics: {METRICS_ROOT / (script_stem + '.csv')}")
        print(f"Wrote artifacts under: {out_dir}")

    return out_dir
