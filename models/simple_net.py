# %%
from pathlib import Path

import pandas as pd
import plotly.express as px
from sklearn.metrics import classification_report, roc_auc_score

from model_run_utils import (
    EXPERIMENT,
    build_run_config_json,
    fit_simple_net_on_pool_for_holdout,
    load_combined_xy,
    persist_run_artifacts,
    pool_cv_oof_predictions,
    roc_oob_figure,
    run_permutation_mda,
    run_shap_proba,
    split_pool_holdout,
)
from optuna_hpo import get_simple_net_tune_fn

SCRIPT_STEM = Path(__file__).stem
MODEL_KIND = "mlp"

# %%
feature_names, X, y = load_combined_xy()
X_pool, X_holdout, y_pool, y_holdout = split_pool_holdout(X, y)

study, est_kw = get_simple_net_tune_fn(MODEL_KIND)(X_pool, y_pool, EXPERIMENT)
print("Optuna — mean CV ROC-AUC (train pool):", round(study.best_value, 4))
print("Best hyperparameters:", est_kw)

y_true, y_pred, y_score = pool_cv_oof_predictions(MODEL_KIND, X_pool, y_pool, EXPERIMENT, est_kw)

# %%
print("Metrics: stratified CV on train pool (holdout test unused here)")
print(classification_report(y_true, y_pred))
print("AUC-ROC (OOF on train pool):", round(roc_auc_score(y_true, y_score), 4))

# %%
roc_fig, _ = roc_oob_figure(y_true, y_score, title_prefix="ROC — CV on train pool, MLP")

# %%
clf_final, X_pool_i, X_hold_i = fit_simple_net_on_pool_for_holdout(
    MODEL_KIND, X_pool, X_holdout, y_pool, EXPERIMENT, est_kw
)

mda_df = run_permutation_mda(clf_final, X_hold_i, y_holdout, feature_names)
fig_mda = px.violin(
    mda_df,
    x="feature",
    y="mda",
    box=True,
    points="all",
    title=f"MDA on held-out test ({EXPERIMENT.mda_n_repeats} shuffle reps / feature)",
)
fig_mda.update_xaxes(tickangle=-45)

# %%
shap_df = run_shap_proba(clf_final, X_pool_i, X_hold_i, feature_names)
fig_shap = px.violin(
    shap_df,
    x="feature",
    y="shap",
    box=True,
    points=False,
    title=(
        f"SHAP for P(class=1) on holdout (up to {EXPERIMENT.shap_max_explain} rows; "
        "background from train pool)"
    ),
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
        cfg=EXPERIMENT,
        estimator_kw=est_kw,
        optuna_best_value=float(study.best_value),
    ),
)

# %%
