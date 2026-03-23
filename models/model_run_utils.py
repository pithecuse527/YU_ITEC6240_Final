"""Shared helpers for simple_net-style training scripts: MDA, SHAP, ROC figures, CSV + HTML exports.

Also hosts the **single** implementation of stratified OOF CV and pool→holdout refit used by
``simple_net_*.py`` and ``streamlit_dashboard.py`` (``pool_cv_oof_predictions``,
``fit_simple_net_on_pool_for_holdout``, ``make_simple_net_classifier``).
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
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
    # Optuna: nested CV on train pool only (scripts must not pass holdout). See optuna_hpo.py.
    optuna_n_trials: int = 24
    optuna_show_progress: bool = True
    optuna_outer_splits: int = 4
    optuna_inner_splits: int = 3


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


# --- Shared by ``simple_net_*.py`` and ``streamlit_dashboard.py`` (one implementation) ---

SIMPLE_NET_MODEL_KINDS = frozenset(
    {"mlp", "knn", "logistic", "rf", "catboost", "svm", "xgboost"}
)


def simple_net_scale_after_impute(model_kind: str) -> bool:
    """KNN and SVM use ``StandardScaler`` after imputation, matching the scripts."""
    return model_kind in {"knn", "svm"}


def make_simple_net_classifier(model_kind: str, est_kw: dict, cfg: ExperimentConfig):
    """Build the fitted estimator class used in ``simple_net_<kind>.py`` (same kwargs + RNG)."""
    rs = cfg.random_state
    if model_kind == "mlp":
        from sklearn.neural_network import MLPClassifier

        return MLPClassifier(**est_kw, random_state=rs)
    if model_kind == "knn":
        from sklearn.neighbors import KNeighborsClassifier

        return KNeighborsClassifier(**est_kw)
    if model_kind == "logistic":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(**est_kw, random_state=rs)
    if model_kind == "rf":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(**est_kw, random_state=rs)
    if model_kind == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(**est_kw, random_seed=rs)
    if model_kind == "svm":
        from sklearn.svm import SVC

        return SVC(**est_kw, random_state=rs)
    if model_kind == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(**est_kw, random_state=rs)
    raise ValueError(f"Unknown simple_net model_kind={model_kind!r}")


def pool_cv_oof_predictions(
    model_kind: str,
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    cfg: ExperimentConfig,
    est_kw: dict,
) -> tuple[list, list, list]:
    """Stratified OOF CV on the train pool: imputer per fold, optional scaler, then ``est_kw`` classifier.

    Matches the manual loops in ``simple_net_*.py`` (including CatBoost ``predict`` dtype).
    """
    skf = make_stratified_kfold(cfg)
    y_true: list = []
    y_pred: list = []
    y_score: list = []
    scale = simple_net_scale_after_impute(model_kind)
    for tr_idx, va_idx in skf.split(X_pool, y_pool):
        imp = make_imputer(cfg)
        X_tr = imp.fit_transform(X_pool[tr_idx])
        X_va = imp.transform(X_pool[va_idx])
        y_tr, y_va = y_pool[tr_idx], y_pool[va_idx]
        if scale:
            from sklearn.preprocessing import StandardScaler

            sc = StandardScaler()
            X_tr = sc.fit_transform(X_tr)
            X_va = sc.transform(X_va)
        clf = make_simple_net_classifier(model_kind, est_kw, cfg)
        clf.fit(X_tr, y_tr)
        y_true.extend(y_va)
        pred = clf.predict(X_va)
        if model_kind == "catboost":
            pred = pred.astype(int).ravel()
        y_pred.extend(pred)
        y_score.extend(clf.predict_proba(X_va)[:, 1])
    return y_true, y_pred, y_score


def fit_simple_net_on_pool_for_holdout(
    model_kind: str,
    X_pool: np.ndarray,
    X_holdout: np.ndarray,
    y_pool: np.ndarray,
    cfg: ExperimentConfig,
    est_kw: dict,
):
    """Impute (+ scale if needed), fit on full pool, return classifier and transformed matrices.

    Same preprocessing as the ``imp_final`` / ``clf_final.fit`` blocks in ``simple_net_*.py``.
    """
    imp = make_imputer(cfg)
    X_pi = imp.fit_transform(X_pool)
    X_hi = imp.transform(X_holdout)
    if simple_net_scale_after_impute(model_kind):
        from sklearn.preprocessing import StandardScaler

        sc = StandardScaler()
        X_pi = sc.fit_transform(X_pi)
        X_hi = sc.transform(X_hi)
    clf = make_simple_net_classifier(model_kind, est_kw, cfg)
    clf.fit(X_pi, y_pool)
    return clf, X_pi, X_hi


def oof_scalar_metrics(y_true, y_pred, y_score) -> dict[str, float]:
    """Accuracy, AUC-ROC, weighted precision/recall/F1 — same construction as ``write_performance_metrics_csv``."""
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    w = rep["weighted avg"]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auc_roc": float(roc_auc_score(y_true, y_score)),
        "precision": float(w["precision"]),
        "recall": float(w["recall"]),
        "f1": float(w["f1-score"]),
    }


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


def build_run_config_json(
    *,
    cfg: ExperimentConfig | None = None,
    estimator_kw: dict | None = None,
    optuna_best_value: float | None = None,
    extra: dict | None = None,
) -> str:
    """JSON blob for the `run_config_json` metrics column (experiment + HPO + estimator kwargs)."""
    payload: dict = {}
    if cfg is not None:
        payload["experiment"] = asdict(cfg)
    if estimator_kw is not None:
        payload["estimator_kw"] = estimator_kw
    if optuna_best_value is not None:
        payload["hpo_best_value"] = optuna_best_value
    if extra:
        payload["extra"] = extra
    return json.dumps(payload, sort_keys=True, default=str)


def write_performance_metrics_csv(
    csv_path: Path,
    *,
    model_name: str,
    y_true,
    y_pred,
    y_score,
    run_config_json: str | None = None,
) -> None:
    """One row: accuracy, AUC-ROC, precision/recall/F1 (weighted average over classes).

    If ``run_config_json`` is set, it is stored in the last column for reproducibility.
    """
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
    if run_config_json is not None:
        row["run_config_json"] = run_config_json
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
    run_config_json: str | None = None,
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
        run_config_json=run_config_json,
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
