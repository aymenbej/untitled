"""Validation « walk-forward » et métriques d'évaluation.

On entraîne sur le passé, on teste sur l'année suivante, puis on avance d'un an.
Jamais de mélange aléatoire des dates. Une zone tampon (« purge ») retire de
l'entraînement les étiquettes dont la sortie tombe dans la période de test.

Option : entraînement sur TOUTES les actions (plus de données), mais test et
évaluation UNIQUEMENT sur les actions halal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .dataset import ALL_FEATURES
from .models import MODEL_NAMES, make_model

KEEP = ["date", "symbol", "label", "outcome_ret", "tp_ret", "sl_ret", "exit_date"]


@dataclass(frozen=True)
class Fold:
    year: int
    train: np.ndarray
    test: np.ndarray


def walk_forward_folds(panel: pd.DataFrame, first_test_year: int, train_on_all: bool = False) -> list[Fold]:
    labeled = panel["label"].notna().to_numpy()
    halal = panel["halal"].to_numpy(bool) if "halal" in panel else np.ones(len(panel), bool)
    years = panel["date"].dt.year.to_numpy()
    folds = []
    for year in range(first_test_year, int(years.max()) + 1):
        start = pd.Timestamp(year=year, month=1, day=1, tz="UTC")
        train = labeled & (panel["exit_date"] < start).to_numpy()
        if not train_on_all:
            train &= halal
        test = labeled & (years == year) & halal
        if train.sum() > 1000 and test.sum() > 0:
            folds.append(Fold(year, np.flatnonzero(train), np.flatnonzero(test)))
    return folds


def daily_ic(df: pd.DataFrame, pred_col: str) -> pd.Series:
    """Corrélation de rang, jour par jour, entre prévision et rendement réel."""
    out = {}
    for day, g in df.groupby("date"):
        if len(g) >= 5 and g[pred_col].nunique() > 1:
            out[day] = spearmanr(g[pred_col], g["outcome_ret"]).statistic
    return pd.Series(out, dtype=float).sort_index()


def evaluate(preds: pd.DataFrame, pred_col: str, horizon: int) -> dict[str, float]:
    """Métriques de sélection hors échantillon.

    - ic_moyen, ic_ecart_type, icir (= moyenne / écart-type)
    - t_stat : calculé sur des dates espacées de `horizon` jours, car les
      rendements de dates proches se chevauchent (sinon t-stat gonflé)
    - jours_ic_positif : part des jours où le classement est meilleur que le hasard
    - ecart_decile_haut_bas : rendement moyen décile 10 − décile 1, par date
    - r2_hors_echantillon : 1 − MSE(modèle) / MSE(moyenne historique)
    """
    ic = daily_ic(preds, pred_col)
    if ic.empty:
        return {}
    sparse = ic.iloc[::horizon]
    t_stat = float(sparse.mean() / sparse.std() * np.sqrt(len(sparse))) if sparse.std() > 0 else np.nan

    def spread(g: pd.DataFrame) -> float:
        if len(g) < 5:
            return np.nan
        cut = 0.1 if len(g) >= 20 else 0.2  # quintiles quand l'univers du jour est petit
        q = g[pred_col].rank(pct=True)
        return g.loc[q > 1 - cut, "outcome_ret"].mean() - g.loc[q <= cut, "outcome_ret"].mean()

    spreads = preds.groupby("date")[[pred_col, "outcome_ret"]].apply(spread)
    mse_model = float(((preds[pred_col] - preds["outcome_ret"]) ** 2).mean())
    mse_ref = float(((preds["pred_reference"] - preds["outcome_ret"]) ** 2).mean())
    return {
        "ic_moyen": float(ic.mean()),
        "ic_ecart_type": float(ic.std()),
        "icir": float(ic.mean() / ic.std()) if ic.std() > 0 else np.nan,
        "t_stat": t_stat,
        "jours_ic_positif": float((ic > 0).mean()),
        "ecart_decile_haut_bas": float(spreads.mean()),
        "r2_hors_echantillon": 1 - mse_model / mse_ref if mse_ref > 0 else np.nan,
    }


def run_walk_forward(
    panel: pd.DataFrame, first_test_year: int, train_on_all: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Renvoie (prédictions hors échantillon, IC par année et par modèle, derniers modèles entraînés)."""
    preds, rows = [], []
    X = panel[ALL_FEATURES].to_numpy(float)
    y = panel["outcome_ret"].to_numpy(float)
    last_models: dict = {}
    for fold in walk_forward_folds(panel, first_test_year, train_on_all):
        out = panel.iloc[fold.test][KEEP].copy()
        for name in MODEL_NAMES:
            model = make_model(name).fit(X[fold.train], y[fold.train])
            out[f"pred_{name}"] = model.predict(X[fold.test])
            last_models[name] = model
        for name in MODEL_NAMES[1:]:
            ic = daily_ic(out, f"pred_{name}")
            rows.append({"annee": fold.year, "modele": name, "ic": float(ic.mean()) if len(ic) else np.nan,
                         "n_test": len(fold.test), "n_train": len(fold.train)})
        preds.append(out)
    return pd.concat(preds, ignore_index=True), pd.DataFrame(rows), last_models


def permutation_importance(model, panel_test: pd.DataFrame, n_repeats: int = 3, seed: int = 0) -> pd.Series:
    """Baisse d'IC quand on mélange une variable (au sein de chaque date) : plus la
    baisse est forte, plus le modèle s'appuie sur cette variable."""
    rng = np.random.default_rng(seed)
    base_df = panel_test[["date", "outcome_ret"]].copy()
    X = panel_test[ALL_FEATURES].to_numpy(float)
    base_df["p"] = model.predict(X)
    base = daily_ic(base_df, "p").mean()
    drops = {}
    groups = list(panel_test.reset_index(drop=True).groupby("date").indices.values())
    for j, feature in enumerate(ALL_FEATURES):
        scores = []
        for _ in range(n_repeats):
            Xp = X.copy()
            for idx in groups:
                Xp[idx, j] = Xp[rng.permutation(idx), j]
            base_df["p"] = model.predict(Xp)
            scores.append(daily_ic(base_df, "p").mean())
        drops[feature] = base - float(np.mean(scores))
    return pd.Series(drops).sort_values(ascending=False)
