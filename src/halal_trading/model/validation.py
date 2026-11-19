"""Validation « walk-forward » : on entraîne sur le passé, on teste sur l'année
suivante, puis on avance d'un an. Jamais de mélange aléatoire des dates.

Une zone tampon (« purge ») est retirée entre entraînement et test : une
étiquette du jour J dépend des prix jusqu'à J+horizon, elle ne doit pas
chevaucher la période de test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .dataset import ALL_FEATURES
from .models import MODEL_NAMES, make_model


@dataclass(frozen=True)
class Fold:
    year: int
    train: np.ndarray
    test: np.ndarray


def walk_forward_folds(panel: pd.DataFrame, first_test_year: int) -> list[Fold]:
    labeled = panel["label"].notna().to_numpy()
    years = panel["date"].dt.year.to_numpy()
    folds = []
    for year in range(first_test_year, int(years.max()) + 1):
        start = pd.Timestamp(year=year, month=1, day=1, tz="UTC")
        # purge : l'étiquette doit être connue (sortie) avant le début du test
        train = labeled & (panel["exit_date"] < start).to_numpy()
        test = labeled & (years == year)
        if train.sum() > 1000 and test.sum() > 0:
            folds.append(Fold(year, np.flatnonzero(train), np.flatnonzero(test)))
    return folds


def information_coefficient(df: pd.DataFrame, pred_col: str) -> float:
    """Corrélation de rang moyenne, jour par jour, entre prévision et rendement réel
    (« IC »). 0 = aucun pouvoir de sélection ; 0,03 à 0,05 est déjà considéré utile."""
    ics = []
    for _, day in df.groupby("date"):
        if len(day) >= 3 and day[pred_col].nunique() > 1:
            ics.append(spearmanr(day[pred_col], day["outcome_ret"]).statistic)
    return float(np.nanmean(ics)) if ics else np.nan


def run_walk_forward(panel: pd.DataFrame, first_test_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Renvoie (prédictions hors échantillon, métriques par année et par modèle)."""
    preds = []
    metrics = []
    X = panel[ALL_FEATURES].to_numpy(float)
    y = panel["outcome_ret"].to_numpy(float)
    for fold in walk_forward_folds(panel, first_test_year):
        out = panel.iloc[fold.test][
            ["date", "symbol", "label", "outcome_ret", "tp_ret", "sl_ret", "exit_date"]
        ].copy()
        for name in MODEL_NAMES:
            model = make_model(name).fit(X[fold.train], y[fold.train])
            out[f"pred_{name}"] = model.predict(X[fold.test])
            top = out[f"pred_{name}"] >= out[f"pred_{name}"].quantile(0.9)
            metrics.append(
                {
                    "annee": fold.year,
                    "modele": name,
                    "n_test": len(fold.test),
                    "ic": information_coefficient(out, f"pred_{name}") if name != "reference" else 0.0,
                    "rendement_moyen_top10pct": float(out.loc[top, "outcome_ret"].mean()),
                    "rendement_moyen_tous": float(out["outcome_ret"].mean()),
                }
            )
        preds.append(out)
    return pd.concat(preds, ignore_index=True), pd.DataFrame(metrics)
