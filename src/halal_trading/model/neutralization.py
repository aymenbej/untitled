"""Neutralisation de la volatilité : la prévision garde-t-elle du pouvoir une fois
retiré son penchant pour les actions volatiles ?

Trois mesures, toutes calculées date par date (en coupe transversale) :

1. IC de la prévision NEUTRALISÉE : on retire de la prévision (en rangs) la part
   expliquée linéairement par le rang de volatilité, puis on calcule l'IC du résidu.
2. IC À VOLATILITÉ COMPARABLE : IC calculé séparément dans chaque quintile de
   volatilité du jour, puis moyenné. On ne compare que des actions de risque proche.
3. IC sur le rendement NORMALISÉ : rendement du trade divisé par σ·√horizon, pour
   que +10 % sur une action calme compte plus que +10 % sur une action agitée.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _pct_rank(df: pd.DataFrame, col: str, keys: list[str]) -> pd.Series:
    return df.groupby(keys)[col].rank(pct=True)


def rank_ic_by_group(df: pd.DataFrame, a: str, b: str, keys: list[str], min_size: int = 5) -> pd.Series:
    """Corrélation de Spearman entre `a` et `b` dans chaque groupe `keys`
    (vectorisée : corrélation de Pearson des rangs)."""
    work = pd.DataFrame({
        "ra": _pct_rank(df, a, keys), "rb": _pct_rank(df, b, keys), **{k: df[k] for k in keys},
    })
    g = work.groupby(keys)
    n = g["ra"].transform("size")
    work = work[n >= min_size].copy()
    g = work.groupby(keys)
    work["da"] = work["ra"] - g["ra"].transform("mean")
    work["db"] = work["rb"] - g["rb"].transform("mean")
    work["ab"], work["aa"], work["bb"] = work["da"] * work["db"], work["da"] ** 2, work["db"] ** 2
    sums = work.groupby(keys)[["ab", "aa", "bb"]].sum()
    denom = np.sqrt(sums["aa"] * sums["bb"])
    return (sums["ab"] / denom.replace(0, np.nan)).dropna()


def neutralize(df: pd.DataFrame, col: str, by: str = "vol_20") -> pd.Series:
    """Résidu, date par date, de la régression du rang de `col` sur le rang de `by`."""
    ra = _pct_rank(df, col, ["date"])
    rb = _pct_rank(df, by, ["date"])
    work = pd.DataFrame({"date": df["date"], "ra": ra, "rb": rb})
    g = work.groupby("date")
    da = work["ra"] - g["ra"].transform("mean")
    db = work["rb"] - g["rb"].transform("mean")
    work["ab"], work["bb"] = da * db, db ** 2
    sums = work.groupby("date")[["ab", "bb"]].transform("sum")
    beta = sums["ab"] / sums["bb"].replace(0, np.nan)
    resid = da - beta.fillna(0) * db
    # score quasi identique à la volatilité : le résidu n'est que du bruit d'arrondi
    spread = resid.groupby(work["date"]).transform("std")
    return resid.where(spread > 1e-9).rename(f"{col}_neutre")


def _t_stat(daily: pd.Series, horizon: int) -> float:
    sparse = daily.sort_index().iloc[::horizon]
    return float(sparse.mean() / sparse.std() * np.sqrt(len(sparse))) if len(sparse) > 2 and sparse.std() > 0 else np.nan


def neutralization_report(df: pd.DataFrame, score: str, horizon: int, vol: str = "vol_20") -> dict[str, float]:
    """IC brut et IC neutralisés pour un score. `df` doit contenir date, score,
    outcome_ret et la volatilité."""
    work = df[["date", score, "outcome_ret", vol]].dropna().copy()
    work["score_neutre"] = neutralize(work, score, vol)
    work["rendement_normalise"] = work["outcome_ret"] / (work[vol] * np.sqrt(horizon))
    work["quintile_vol"] = work.groupby("date")[vol].transform(
        lambda v: pd.qcut(v.rank(method="first"), 5, labels=False) if len(v) >= 10 else np.nan
    )

    raw = rank_ic_by_group(work, score, "outcome_ret", ["date"])
    neutral = rank_ic_by_group(work, "score_neutre", "outcome_ret", ["date"])
    bucketed = rank_ic_by_group(work.dropna(subset=["quintile_vol"]), score, "outcome_ret",
                                ["date", "quintile_vol"]).groupby(level="date").mean()
    normalized = rank_ic_by_group(work, score, "rendement_normalise", ["date"])
    vol_corr = rank_ic_by_group(work, score, vol, ["date"])

    return {
        "correlation_avec_volatilite": float(vol_corr.mean()),
        "ic_brut": float(raw.mean()), "t_brut": _t_stat(raw, horizon),
        "ic_prevision_neutralisee": float(neutral.mean()), "t_prevision_neutralisee": _t_stat(neutral, horizon),
        "ic_volatilite_comparable": float(bucketed.mean()), "t_volatilite_comparable": _t_stat(bucketed, horizon),
        "ic_rendement_normalise": float(normalized.mean()), "t_rendement_normalise": _t_stat(normalized, horizon),
    }
