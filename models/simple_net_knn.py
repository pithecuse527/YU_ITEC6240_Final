# %%
# KNN: same experiment config as other simple_net_* scripts; StandardScaler after imputation (distance-based).
# %%
from pathlib import Path

import pandas as pd
import plotly.express as px
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from model_run_utils import (
    EXPERIMENT,
    load_combined_xy,
    make_imputer,
    make_stratified_kfold,
    persist_run_artifacts,
    roc_oob_figure,
    run_permutation_mda,
    run_shap_proba,
    split_pool_holdout,
)

SCRIPT_STEM = Path(__file__).stem

# %%
feature_names, X, y = load_combined_xy()
X_pool, X_holdout, y_pool, y_holdout = split_pool_holdout(X, y)
skf = make_stratified_kfold()
y_true, y_pred, y_score = [], [], []

for tr_idx, va_idx in skf.split(X_pool, y_pool):
    imp = make_imputer()
    X_tr = imp.fit_transform(X_pool[tr_idx])
    X_va = imp.transform(X_pool[va_idx])
    y_tr, y_va = y_pool[tr_idx], y_pool[va_idx]
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_va = scaler.transform(X_va)

    clf = KNeighborsClassifier(n_neighbors=11, weights="distance", n_jobs=-1)
    clf.fit(X_tr, y_tr)
    y_true.extend(y_va)
    y_pred.extend(clf.predict(X_va))
    y_score.extend(clf.predict_proba(X_va)[:, 1])

# %%
print("Metrics: stratified CV on train pool (holdout test unused here)")
print(classification_report(y_true, y_pred))
print("AUC-ROC (OOF on train pool):", round(roc_auc_score(y_true, y_score), 4))

# %%
roc_fig, _ = roc_oob_figure(y_true, y_score, title_prefix="ROC — CV on train pool, KNN")

# %%
imp_final = make_imputer()
X_pool_i = imp_final.fit_transform(X_pool)
X_hold_i = imp_final.transform(X_holdout)
scaler_final = StandardScaler()
X_pool_i = scaler_final.fit_transform(X_pool_i)
X_hold_i = scaler_final.transform(X_hold_i)
clf_final = KNeighborsClassifier(n_neighbors=11, weights="distance", n_jobs=-1)
clf_final.fit(X_pool_i, y_pool)

mda_df = run_permutation_mda(clf_final, X_hold_i, y_holdout, feature_names)
fig_mda = px.violin(
    mda_df,
    x="feature",
    y="mda",
    box=True,
    points="all",
    title=f"MDA on held-out test — KNN ({EXPERIMENT.mda_n_repeats} shuffle reps / feature)",
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
        f"SHAP for P(class=1) — KNN (up to {EXPERIMENT.shap_max_explain} holdout rows; "
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
)

# %%
