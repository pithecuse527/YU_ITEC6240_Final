# %%
# CatBoost classifier — same protocol as simple_net.py (holdout, CV, median impute, MDA, SHAP).
# Optuna tunes learning_rate / depth / l2_leaf_reg; categorical features declared explicitly.

# %%
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from model_run_utils import (
    build_run_config_json,
    persist_run_artifacts,
    roc_oob_figure,
    run_permutation_mda,
    run_shap_proba,
)

SCRIPT_STEM = Path(__file__).stem

CATEGORICAL_FEATURES = [
    "sex",
    "chest pain type",
    "fasting blood sugar",
    "resting ecg",
    "exercise angina",
    "ST slope",
]

# %%
df = pd.read_csv("/home/syntheticdemon/ml/data/combined.csv")
feature_names = df.drop(columns=["target"]).columns.tolist()
cat_indices = [feature_names.index(c) for c in CATEGORICAL_FEATURES if c in feature_names]
X = df.drop(columns=["target"]).to_numpy(dtype=float)
y = df["target"].to_numpy()

X_pool, X_holdout, y_pool, y_holdout = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# %%  — Optuna hyperparameter search on a single train/val split inside the pool
import optuna
from catboost import CatBoostClassifier

_imp_tune = SimpleImputer(strategy="median")
_X_tune_tr, _X_tune_va, _y_tune_tr, _y_tune_va = train_test_split(
    X_pool, y_pool, test_size=0.2, stratify=y_pool, random_state=42
)
_X_tune_tr = _imp_tune.fit_transform(_X_tune_tr)
_X_tune_va = _imp_tune.transform(_X_tune_va)


def _objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.1, log=True)
    depth = trial.suggest_int("depth", 4, 10)
    l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 1e-2, 10, log=True)
    clf = CatBoostClassifier(
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        random_state=42,
        verbose=0,
        iterations=1000,
        early_stopping_rounds=50,
        cat_features=cat_indices,
    )
    clf.fit(_X_tune_tr, _y_tune_tr, eval_set=(_X_tune_va, _y_tune_va))
    return clf.score(_X_tune_va, _y_tune_va)


study = optuna.create_study(direction="maximize", study_name="CatBoost_Hyperparameter_Optimization")
study.optimize(_objective, n_trials=50, show_progress_bar=False)
best = study.best_params
print(f"Optuna best params: {best}  acc={study.best_value:.4f}")

# %%  — Stratified 5-fold CV with best params
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_true, y_pred, y_score = [], [], []

for tr_idx, va_idx in skf.split(X_pool, y_pool):
    imp = SimpleImputer(strategy="median")
    X_tr = imp.fit_transform(X_pool[tr_idx])
    X_va = imp.transform(X_pool[va_idx])
    y_tr, y_va = y_pool[tr_idx], y_pool[va_idx]

    clf = CatBoostClassifier(
        learning_rate=best["learning_rate"],
        depth=best["depth"],
        l2_leaf_reg=best["l2_leaf_reg"],
        random_state=42,
        verbose=0,
        iterations=1000,
        early_stopping_rounds=50,
        cat_features=cat_indices,
    )
    clf.fit(X_tr, y_tr, eval_set=(X_va, y_va))
    y_true.extend(y_va)
    y_pred.extend(clf.predict(X_va))
    y_score.extend(clf.predict_proba(X_va)[:, 1])

# %%
print("Metrics: stratified CV on train pool (holdout test unused here)")
print(classification_report(y_true, y_pred))
print("AUC-ROC (OOF on train pool):", round(roc_auc_score(y_true, y_score), 4))

# %%
roc_fig, _auc = roc_oob_figure(y_true, y_score, title_prefix="ROC — CV on train pool, CatBoost")

# %%
imp_final = SimpleImputer(strategy="median")
X_pool_i = imp_final.fit_transform(X_pool)
X_hold_i = imp_final.transform(X_holdout)
clf_final = CatBoostClassifier(
    learning_rate=best["learning_rate"],
    depth=best["depth"],
    l2_leaf_reg=best["l2_leaf_reg"],
    random_state=42,
    verbose=0,
    iterations=1000,
    cat_features=cat_indices,
)
clf_final.fit(X_pool_i, y_pool)

mda_df = run_permutation_mda(
    clf_final, X_hold_i, y_holdout, feature_names, rng=np.random.default_rng(42), n_repeats=50
)
fig_mda = px.violin(
    mda_df,
    x="feature",
    y="mda",
    box=True,
    points="all",
    title="MDA on held-out test — CatBoost (50 shuffle reps / feature)",
)
fig_mda.update_xaxes(tickangle=-45)

# %%
shap_df = run_shap_proba(
    clf_final, X_pool_i, X_hold_i, feature_names, class_index=1, max_background=100, random_state=42
)
fig_shap = px.violin(
    shap_df,
    x="feature",
    y="shap",
    box=True,
    points=False,
    title="SHAP for P(class=1) on holdout — CatBoost (background from train pool)",
)
fig_shap.update_xaxes(tickangle=-45)

# %%
persist_run_artifacts(
    SCRIPT_STEM,
    y_true=y_true,
    y_pred=y_pred,
    y_score=y_score,
    roc_fig=roc_fig,
    fig_mda=fig_mda,
    fig_shap=fig_shap,
    mda_df=mda_df,
    shap_df=shap_df,
    run_config_json=build_run_config_json(
        cfg=None,
        estimator_kw={
            "learning_rate": float(best["learning_rate"]),
            "depth": int(best["depth"]),
            "l2_leaf_reg": float(best["l2_leaf_reg"]),
            "iterations": 1000,
            "verbose": 0,
            "early_stopping_rounds": 50,
            "cat_features": cat_indices,
        },
        optuna_best_value=float(study.best_value),
        extra={
            "optuna_trials": 50,
            "tuning_objective": "accuracy_on_holdout_within_pool",
            "final_pool_fit": "same_lr_depth_l2_without_early_stopping",
        },
    ),
)

# %%
