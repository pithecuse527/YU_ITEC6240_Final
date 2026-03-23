"""Optuna hyperparameter search on the train pool only — holdout must never be passed in.

Leakage controls:
- Nested CV: for each outer fold, inner CV runs only on outer-train rows. Imputer and scaler
  (when used) are fit strictly on each inner training fold, then applied to that fold's validation.
- Holdout is reserved for MDA/SHAP in the driver scripts and is not used here.
- Final OOF metrics in scripts still use EXPERIMENT.cv_n_splits on the full pool with the chosen
  hyperparameters (honest OOF with per-fold imputation).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from model_run_utils import EXPERIMENT, ExperimentConfig, make_imputer

logger = logging.getLogger(__name__)


def _hpo_logger() -> logging.Logger:
    """Ensure HPO messages appear when the app has not configured logging."""
    if not logger.handlers and not logging.root.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(levelname)s [optuna_hpo] %(message)s"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def _nested_cv_mean_roc_auc(
    X: np.ndarray,
    y: np.ndarray,
    cfg: ExperimentConfig,
    make_classifier,
    *,
    scale_after_impute: bool = False,
) -> tuple[float, list[float]]:
    """Nested CV score: mean over outer folds of (mean inner-fold val ROC-AUC).

    Returns (overall_mean, per_outer_fold_means) for logging.
    """
    y = np.asarray(y)
    outer = StratifiedKFold(
        n_splits=cfg.optuna_outer_splits,
        shuffle=True,
        random_state=cfg.random_state,
    )
    inner = StratifiedKFold(
        n_splits=cfg.optuna_inner_splits,
        shuffle=True,
        random_state=cfg.random_state + 1000,
    )
    outer_scores: list[float] = []
    for tr_o, _va_o in outer.split(X, y):
        X_o, y_o = X[tr_o], y[tr_o]
        inner_scores: list[float] = []
        for tr_i, va_i in inner.split(X_o, y_o):
            imp = make_imputer(cfg)
            X_tr = imp.fit_transform(X_o[tr_i])
            X_va = imp.transform(X_o[va_i])
            if scale_after_impute:
                from sklearn.preprocessing import StandardScaler

                sc = StandardScaler()
                X_tr = sc.fit_transform(X_tr)
                X_va = sc.transform(X_va)
            clf = make_classifier()
            clf.fit(X_tr, y_o[tr_i])
            inner_scores.append(
                roc_auc_score(y_o[va_i], clf.predict_proba(X_va)[:, 1])
            )
        outer_scores.append(float(np.mean(inner_scores)))
    return float(np.mean(outer_scores)), outer_scores


def _log_trial_result(
    log: logging.Logger,
    study_name: str,
    trial: optuna.Trial,
    score: float,
    outer_means: list[float],
    extra: str | None = None,
) -> None:
    outer_str = ", ".join(f"{m:.4f}" for m in outer_means)
    msg = (
        "[%s] trial %d finished: nested_cv_mean_roc_auc=%.6f | outer_fold_means=[%s] | params=%s"
        % (study_name, trial.number, score, outer_str, trial.params)
    )
    if extra:
        msg += f" | {extra}"
    log.info(msg)


def _run_study(
    objective,
    study_name: str,
    cfg: ExperimentConfig,
    n_trials: int | None,
    *,
    pool_n_samples: int,
    optuna_callbacks: Sequence | None = None,
    show_progress_bar: bool | None = None,
) -> optuna.Study:
    log = _hpo_logger()
    n_trials = cfg.optuna_n_trials if n_trials is None else n_trials
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        sampler=TPESampler(seed=cfg.random_state),
    )
    log.info(
        "[%s] starting hyperparameter search: n_trials=%d outer_splits=%d inner_splits=%d pool_samples=%d",
        study_name,
        n_trials,
        cfg.optuna_outer_splits,
        cfg.optuna_inner_splits,
        pool_n_samples,
    )
    spb = cfg.optuna_show_progress if show_progress_bar is None else show_progress_bar
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=spb,
        callbacks=list(optuna_callbacks) if optuna_callbacks else [],
    )
    best = study.best_trial
    log.info(
        "[%s] best trial %d: nested_cv_mean_roc_auc=%.6f | params=%s | estimator_kw=%s",
        study_name,
        best.number,
        float(best.value),
        best.params,
        best.user_attrs.get("estimator_kw"),
    )
    return study


def tune_mlp(
    X_pool,
    y_pool,
    cfg: ExperimentConfig = EXPERIMENT,
    n_trials: int | None = None,
    *,
    optuna_callbacks: Sequence | None = None,
    show_progress_bar: bool | None = None,
):
    from sklearn.neural_network import MLPClassifier

    log = _hpo_logger()
    study_name = "simple_net_mlp"
    n_pool = int(np.asarray(y_pool).shape[0])

    def objective(trial: optuna.Trial) -> float:
        hidden_layer_sizes = trial.suggest_categorical(
            "hidden_layer_sizes",
            [(128,), (256,), (128, 64), (256, 128)],
        )
        activation = trial.suggest_categorical("activation", ["relu", "tanh"])
        alpha = trial.suggest_float("alpha", 1e-5, 5e-2, log=True)
        learning_rate_init = trial.suggest_float("learning_rate_init", 1e-3, 1e-1, log=True)
        kw = dict(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_iter=2500,
        )
        trial.set_user_attr("estimator_kw", kw)

        def make_clf():
            return MLPClassifier(**kw, random_state=cfg.random_state)

        score, outer_means = _nested_cv_mean_roc_auc(X_pool, y_pool, cfg, make_clf)
        _log_trial_result(log, study_name, trial, score, outer_means)
        return score

    study = _run_study(
        objective,
        study_name,
        cfg,
        n_trials,
        pool_n_samples=n_pool,
        optuna_callbacks=optuna_callbacks,
        show_progress_bar=show_progress_bar,
    )
    return study, study.best_trial.user_attrs["estimator_kw"]


def tune_knn(
    X_pool,
    y_pool,
    cfg: ExperimentConfig = EXPERIMENT,
    n_trials: int | None = None,
    *,
    optuna_callbacks: Sequence | None = None,
    show_progress_bar: bool | None = None,
):
    from sklearn.neighbors import KNeighborsClassifier

    log = _hpo_logger()
    study_name = "simple_net_knn"
    n_pool = int(np.asarray(y_pool).shape[0])

    def objective(trial: optuna.Trial) -> float:
        n_neighbors = trial.suggest_int("n_neighbors", 3, 21, step=2)
        weights = trial.suggest_categorical("weights", ["uniform", "distance"])
        p = trial.suggest_int("p", 1, 2)
        kw = dict(n_neighbors=n_neighbors, weights=weights, metric="minkowski", p=p, n_jobs=-1)
        trial.set_user_attr("estimator_kw", kw)

        def make_clf():
            return KNeighborsClassifier(**kw)

        score, outer_means = _nested_cv_mean_roc_auc(
            X_pool, y_pool, cfg, make_clf, scale_after_impute=True
        )
        _log_trial_result(log, study_name, trial, score, outer_means)
        return score

    study = _run_study(
        objective,
        study_name,
        cfg,
        n_trials,
        pool_n_samples=n_pool,
        optuna_callbacks=optuna_callbacks,
        show_progress_bar=show_progress_bar,
    )
    return study, study.best_trial.user_attrs["estimator_kw"]


def tune_logistic(
    X_pool,
    y_pool,
    cfg: ExperimentConfig = EXPERIMENT,
    n_trials: int | None = None,
    *,
    optuna_callbacks: Sequence | None = None,
    show_progress_bar: bool | None = None,
):
    """lbfgs + L2 only (stable with sklearn 1.8+; avoids saga/L1 path)."""
    from sklearn.linear_model import LogisticRegression

    log = _hpo_logger()
    study_name = "simple_net_logistic"
    n_pool = int(np.asarray(y_pool).shape[0])

    def objective(trial: optuna.Trial) -> float:
        C = trial.suggest_float("C", 0.02, 20.0, log=True)
        # l2 implied for lbfgs; omit penalty for sklearn>=1.8 deprecation cleanliness
        kw = dict(C=C, solver="lbfgs", max_iter=5000)
        trial.set_user_attr("estimator_kw", kw)

        def make_clf():
            return LogisticRegression(**kw, random_state=cfg.random_state)

        score, outer_means = _nested_cv_mean_roc_auc(X_pool, y_pool, cfg, make_clf)
        _log_trial_result(log, study_name, trial, score, outer_means)
        return score

    study = _run_study(
        objective,
        study_name,
        cfg,
        n_trials,
        pool_n_samples=n_pool,
        optuna_callbacks=optuna_callbacks,
        show_progress_bar=show_progress_bar,
    )
    return study, study.best_trial.user_attrs["estimator_kw"]


def tune_random_forest(
    X_pool,
    y_pool,
    cfg: ExperimentConfig = EXPERIMENT,
    n_trials: int | None = None,
    *,
    optuna_callbacks: Sequence | None = None,
    show_progress_bar: bool | None = None,
):
    from sklearn.ensemble import RandomForestClassifier

    log = _hpo_logger()
    study_name = "simple_net_rf"
    n_pool = int(np.asarray(y_pool).shape[0])

    def objective(trial: optuna.Trial) -> float:
        n_estimators = trial.suggest_categorical("n_estimators", [150, 200, 250, 300, 350])
        max_depth = trial.suggest_categorical("max_depth", [None, 8, 10, 12, 16, 20])
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 4)
        max_features = trial.suggest_categorical("max_features", ["sqrt", "log2"])
        kw = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight="balanced",
            n_jobs=-1,
        )
        trial.set_user_attr("estimator_kw", kw)

        def make_clf():
            return RandomForestClassifier(**kw, random_state=cfg.random_state)

        score, outer_means = _nested_cv_mean_roc_auc(X_pool, y_pool, cfg, make_clf)
        _log_trial_result(log, study_name, trial, score, outer_means)
        return score

    study = _run_study(
        objective,
        study_name,
        cfg,
        n_trials,
        pool_n_samples=n_pool,
        optuna_callbacks=optuna_callbacks,
        show_progress_bar=show_progress_bar,
    )
    return study, study.best_trial.user_attrs["estimator_kw"]


def tune_catboost(
    X_pool,
    y_pool,
    cfg: ExperimentConfig = EXPERIMENT,
    n_trials: int | None = None,
    *,
    optuna_callbacks: Sequence | None = None,
    show_progress_bar: bool | None = None,
):
    from catboost import CatBoostClassifier

    log = _hpo_logger()
    study_name = "simple_net_catboost"
    n_pool = int(np.asarray(y_pool).shape[0])

    def objective(trial: optuna.Trial) -> float:
        iterations = trial.suggest_categorical("iterations", [300, 400, 500, 600])
        depth = trial.suggest_int("depth", 4, 8)
        learning_rate = trial.suggest_float("learning_rate", 0.03, 0.2, log=True)
        l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 2.0, 10.0)
        kw = dict(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            l2_leaf_reg=l2_leaf_reg,
            verbose=False,
            allow_writing_files=False,
        )
        trial.set_user_attr("estimator_kw", kw)

        def make_clf():
            return CatBoostClassifier(**kw, random_seed=cfg.random_state)

        score, outer_means = _nested_cv_mean_roc_auc(X_pool, y_pool, cfg, make_clf)
        _log_trial_result(log, study_name, trial, score, outer_means)
        return score

    study = _run_study(
        objective,
        study_name,
        cfg,
        n_trials,
        pool_n_samples=n_pool,
        optuna_callbacks=optuna_callbacks,
        show_progress_bar=show_progress_bar,
    )
    return study, study.best_trial.user_attrs["estimator_kw"]


def tune_svm(
    X_pool,
    y_pool,
    cfg: ExperimentConfig = EXPERIMENT,
    n_trials: int | None = None,
    *,
    optuna_callbacks: Sequence | None = None,
    show_progress_bar: bool | None = None,
):
    from sklearn.svm import SVC

    log = _hpo_logger()
    study_name = "simple_net_svm"
    n_pool = int(np.asarray(y_pool).shape[0])

    def objective(trial: optuna.Trial) -> float:
        C = trial.suggest_float("C", 0.1, 100.0, log=True)
        gamma_kind = trial.suggest_categorical("gamma_kind", ["scale", "auto", "value"])
        if gamma_kind == "value":
            gamma = trial.suggest_float("gamma_value", 1e-3, 0.5, log=True)
        else:
            gamma = gamma_kind
        kw = dict(kernel="rbf", C=C, gamma=gamma, probability=True)
        trial.set_user_attr("estimator_kw", kw)

        def make_clf():
            return SVC(**kw, random_state=cfg.random_state)

        score, outer_means = _nested_cv_mean_roc_auc(
            X_pool, y_pool, cfg, make_clf, scale_after_impute=True
        )
        _log_trial_result(log, study_name, trial, score, outer_means)
        return score

    study = _run_study(
        objective,
        study_name,
        cfg,
        n_trials,
        pool_n_samples=n_pool,
        optuna_callbacks=optuna_callbacks,
        show_progress_bar=show_progress_bar,
    )
    return study, study.best_trial.user_attrs["estimator_kw"]


def tune_xgboost(
    X_pool,
    y_pool,
    cfg: ExperimentConfig = EXPERIMENT,
    n_trials: int | None = None,
    *,
    optuna_callbacks: Sequence | None = None,
    show_progress_bar: bool | None = None,
):
    from xgboost import XGBClassifier

    log = _hpo_logger()
    study_name = "simple_net_xgboost"
    n_pool = int(np.asarray(y_pool).shape[0])

    def objective(trial: optuna.Trial) -> float:
        n_estimators = trial.suggest_categorical("n_estimators", [200, 300, 400, 500])
        max_depth = trial.suggest_int("max_depth", 4, 9)
        learning_rate = trial.suggest_float("learning_rate", 0.02, 0.15, log=True)
        subsample = trial.suggest_float("subsample", 0.7, 1.0)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.7, 1.0)
        min_child_weight = trial.suggest_int("min_child_weight", 1, 6)
        kw = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            eval_metric="logloss",
            verbosity=0,
            n_jobs=-1,
        )
        trial.set_user_attr("estimator_kw", kw)

        def make_clf():
            return XGBClassifier(**kw, random_state=cfg.random_state)

        score, outer_means = _nested_cv_mean_roc_auc(X_pool, y_pool, cfg, make_clf)
        _log_trial_result(log, study_name, trial, score, outer_means)
        return score

    study = _run_study(
        objective,
        study_name,
        cfg,
        n_trials,
        pool_n_samples=n_pool,
        optuna_callbacks=optuna_callbacks,
        show_progress_bar=show_progress_bar,
    )
    return study, study.best_trial.user_attrs["estimator_kw"]


# Registry: ``model_kind`` keys match ``simple_net_<name>.py`` stems / ``streamlit_dashboard`` ids.
_SIMPLE_NET_TUNERS = {
    "mlp": tune_mlp,
    "knn": tune_knn,
    "logistic": tune_logistic,
    "rf": tune_random_forest,
    "catboost": tune_catboost,
    "svm": tune_svm,
    "xgboost": tune_xgboost,
}


def get_simple_net_tune_fn(model_kind: str):
    """Return ``tune_*`` callable for a ``simple_net`` family model (same as the driver scripts)."""
    try:
        return _SIMPLE_NET_TUNERS[model_kind]
    except KeyError as e:
        raise KeyError(f"Unknown simple_net model_kind {model_kind!r}") from e
