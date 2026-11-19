"""Modèles comparés. Cible : rendement réel du trade (sortie à l'objectif, au
stop ou à l'échéance), avant frais.

Un modèle n'est utile que s'il bat la référence naïve sur des années qu'il n'a
jamais vues.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class MeanReturn:
    """Référence naïve : prédit toujours le rendement moyen historique."""

    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


def make_model(name: str):
    if name == "reference":
        return MeanReturn()
    if name == "lineaire":
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    if name == "gradient_boosting":
        return HistGradientBoostingRegressor(
            loss="absolute_error",  # robuste aux rendements extrêmes
            max_iter=300,
            learning_rate=0.03,
            max_leaf_nodes=15,
            min_samples_leaf=500,
            l2_regularization=1.0,
            random_state=0,
        )
    raise ValueError(f"modèle inconnu : {name}")


MODEL_NAMES = ["reference", "lineaire", "gradient_boosting"]
