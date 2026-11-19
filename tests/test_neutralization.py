"""Tests de la neutralisation : un score qui n'est que de la volatilité perd son
IC, un vrai signal indépendant de la volatilité le garde."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from halal_trading.model.neutralization import neutralization_report, neutralize, rank_ic_by_group


def panel(n_days=300, n_assets=60, seed=0):
    rng = np.random.default_rng(seed)
    dates = np.repeat(pd.bdate_range("2020-01-01", periods=n_days, tz="UTC"), n_assets)
    vol = rng.uniform(0.01, 0.05, n_days * n_assets)
    signal = rng.normal(0, 1, n_days * n_assets)
    # le rendement monte avec la volatilité ET avec le signal
    outcome = 2 * vol + 0.004 * signal + rng.normal(0, 0.01, n_days * n_assets)
    return pd.DataFrame({"date": dates, "vol_20": vol, "signal": signal, "outcome_ret": outcome,
                         "vol_score": vol + rng.normal(0, 0.006, len(vol))})


def test_rank_ic_matches_scipy():
    df = panel(n_days=3)
    ours = rank_ic_by_group(df, "signal", "outcome_ret", ["date"])
    for day, g in df.groupby("date"):
        assert ours[day] == pytest.approx(spearmanr(g["signal"], g["outcome_ret"]).statistic, abs=1e-9)


def test_neutralized_score_is_uncorrelated_with_volatility():
    df = panel(n_days=20)
    df["mix"] = df["signal"] + 50 * df["vol_20"]
    df["mix_neutre"] = neutralize(df, "mix")
    corr = rank_ic_by_group(df, "mix_neutre", "vol_20", ["date"]).mean()
    assert abs(corr) < 0.1


def test_pure_volatility_score_loses_its_ic():
    rep = neutralization_report(panel(), "vol_score", horizon=20)
    assert rep["correlation_avec_volatilite"] > 0.7
    assert rep["ic_brut"] > 0.3  # semble très prédictif...
    assert abs(rep["ic_prevision_neutralisee"]) < 0.05  # ... mais ce n'est que de la volatilité
    # les quintiles ne neutralisent qu'en partie (la volatilité varie encore dans chaque quintile)
    assert rep["ic_volatilite_comparable"] < rep["ic_brut"] / 2
    assert abs(rep["ic_rendement_normalise"]) < 0.05


def test_score_identical_to_volatility_has_no_neutral_ic():
    df = panel(n_days=20)
    df["copie"] = df["vol_20"]
    assert neutralize(df, "copie").isna().all()


def test_real_signal_survives_neutralization():
    rep = neutralization_report(panel(), "signal", horizon=20)
    assert rep["ic_prevision_neutralisee"] > 0.1
    assert rep["ic_volatilite_comparable"] > 0.1
    assert rep["t_prevision_neutralisee"] > 3
