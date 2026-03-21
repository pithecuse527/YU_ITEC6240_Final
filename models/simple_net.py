# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier


def run_permutation_mda(clf, X_test, y_test, feature_names, *, rng=None, n_repeats=50):
    """MDA via repeated single-feature shuffles on a fixed test set. Returns long DataFrame (feature, mda)."""
    if rng is None:
        rng = np.random.default_rng()
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
    clf, X_background, X_explain, feature_names, *, class_index=1, max_background=100, random_state=42
):
    """SHAP for sklearn `predict_proba` (tabular). Returns long DataFrame (sample, feature, shap)."""
    import shap

    X_bg = np.asarray(X_background, dtype=float)
    X_ex = np.asarray(X_explain, dtype=float)
    rng = np.random.default_rng(random_state)
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

    clf = MLPClassifier(
        hidden_layer_sizes=(256,),
        activation="logistic",
        max_iter=1500,
        random_state=42,
    )
    clf.fit(X_tr, y_tr)
    y_true.extend(y_va)
    y_pred.extend(clf.predict(X_va))
    y_score.extend(clf.predict_proba(X_va)[:, 1])

# %%
print("Metrics: stratified CV on train pool (holdout test unused here)")
print(classification_report(y_true, y_pred))
print("AUC-ROC (OOF on train pool):", round(roc_auc_score(y_true, y_score), 4))

# %%
fpr, tpr, _ = roc_curve(y_true, y_score)
auc = roc_auc_score(y_true, y_score)
roc_fig = px.line(
    pd.DataFrame({"fpr": fpr, "tpr": tpr}),
    x="fpr",
    y="tpr",
    title=f"ROC — CV on train pool (AUC = {auc:.3f})",
    labels={"fpr": "False positive rate", "tpr": "True positive rate"},
)
roc_fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="gray"))

# %%
imp_final = SimpleImputer(strategy="median")
X_pool_i = imp_final.fit_transform(X_pool)
X_hold_i = imp_final.transform(X_holdout)
clf_final = MLPClassifier(
    hidden_layer_sizes=(256,),
    activation="logistic",
    max_iter=1500,
    random_state=42,
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
    title="MDA on held-out test (50 shuffle reps / feature)",
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
    title="SHAP values for P(class=1) on holdout (shap.Explainer, background from train pool)",
)
fig_shap.update_xaxes(tickangle=-45)

# %%
_out = Path(__file__).resolve().parent
for fig, stem in (roc_fig, "roc_oob"), (fig_mda, "mda_violin"), (fig_shap, "shap_violin"):
    if "ipykernel" in sys.modules:
        fig.show()
    else:
        p = _out / f"{stem}.html"
        fig.write_html(p)
        print(f"Wrote {p}")

# %%
