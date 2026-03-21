# %%
# KNN classifier — same protocol as simple_net.py (holdout, CV, median impute, MDA, SHAP).
# Uses KNeighborsClassifier. After imputation, StandardScaler is fit on train only (needed for distances).

# %%
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from model_run_utils import (
    persist_run_artifacts,
    roc_oob_figure,
    run_permutation_mda,
    run_shap_proba,
)

SCRIPT_STEM = Path(__file__).stem

# %%
df = pd.read_csv("/home/syntheticdemon/ml/data/combined.csv")
feature_names = df.drop(columns=["target"]).columns.tolist()
X = df.drop(columns=["target"]).to_numpy(dtype=float)
y = df["target"].to_numpy()

X_pool, X_holdout, y_pool, y_holdout = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_true, y_pred, y_score = [], [], []

for tr_idx, va_idx in skf.split(X_pool, y_pool):
    imp = SimpleImputer(strategy="median")
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
roc_fig, _auc = roc_oob_figure(y_true, y_score, title_prefix="ROC — CV on train pool, KNN")

# %%
imp_final = SimpleImputer(strategy="median")
X_pool_i = imp_final.fit_transform(X_pool)
X_hold_i = imp_final.transform(X_holdout)
scaler_final = StandardScaler()
X_pool_i = scaler_final.fit_transform(X_pool_i)
X_hold_i = scaler_final.transform(X_hold_i)
clf_final = KNeighborsClassifier(n_neighbors=11, weights="distance", n_jobs=-1)
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
    title="MDA on held-out test — KNN (50 shuffle reps / feature)",
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
    title="SHAP for P(class=1) on holdout — KNN (background from train pool)",
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
