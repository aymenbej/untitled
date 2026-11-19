"""Tests des règles techniques sur des bougies construites à la main."""

import numpy as np
import pandas as pd
import pytest

from halal_trading.model.rules import SIGNALS, event_study, technical_signals, trend_following


def base(n=260, start=100.0, step=0.0):
    close = start + step * np.arange(n)
    return pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=n, tz="UTC"),
                         "open": close, "high": close + 0.5, "low": close - 0.5, "close": close.copy()})


def set_candle(df, i, o, h, l, c):
    df.loc[i, ["open", "high", "low", "close"]] = [o, h, l, c]


def with_downtrend(df, end):
    """Baisse régulière sur les 6 séances qui précèdent `end`."""
    for k, i in enumerate(range(end - 6, end)):
        level = 100 - 2 * k
        set_candle(df, i, level + 0.5, level + 1, level - 0.5, level)
    return df


def test_hammer_detected_after_decline():
    df = with_downtrend(base(), 255)
    set_candle(df, 255, 89.6, 90.0, 85.0, 89.9)  # petit corps en haut, longue mèche basse
    assert technical_signals(df)["marteau"].iloc[255]


def test_hammer_needs_a_prior_decline():
    df = base()
    set_candle(df, 255, 99.6, 100.0, 95.0, 99.9)
    assert not technical_signals(df)["marteau"].iloc[255]


def test_bullish_engulfing():
    df = with_downtrend(base(), 254)
    set_candle(df, 254, 91.0, 91.2, 89.8, 90.0)  # baissière
    set_candle(df, 255, 89.8, 91.6, 89.5, 91.5)  # haussière qui englobe
    assert technical_signals(df)["avalement_haussier"].iloc[255]


def test_morning_star():
    df = base()
    set_candle(df, 253, 100, 100.2, 94.8, 95)  # grande baissière
    set_candle(df, 254, 94.0, 94.5, 93.5, 94.2)  # petite bougie plus bas
    set_candle(df, 255, 94.5, 98.5, 94.4, 98.0)  # haussière au-dessus du milieu (97,5)
    assert technical_signals(df)["etoile_du_matin"].iloc[255]


def test_three_white_soldiers():
    df = base()
    set_candle(df, 253, 100, 101.2, 99.9, 101)
    set_candle(df, 254, 100.5, 102.2, 100.4, 102)
    set_candle(df, 255, 101.5, 103.2, 101.4, 103)
    assert technical_signals(df)["trois_soldats_blancs"].iloc[255]


def test_golden_cross_fires_once():
    close = np.r_[np.linspace(150, 100, 250), np.linspace(100, 200, 150)]
    df = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=len(close), tz="UTC"),
                       "open": close, "high": close + 1, "low": close - 1, "close": close})
    s = technical_signals(df)
    assert s["croisement_50_200"].sum() == 1


def test_breakout_52_weeks():
    df = base(step=-0.01)
    set_candle(df, 259, 100, 103, 99, 102)
    s = technical_signals(df)
    assert s["cassure_52_semaines"].iloc[259] and not s["cassure_52_semaines"].iloc[:259].any()


def test_signals_never_use_future_data():
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 500)))
    df = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=500, tz="UTC"),
                       "open": close * (1 + rng.normal(0, 0.005, 500)), "high": close * 1.02,
                       "low": close * 0.98, "close": close})
    full = technical_signals(df)
    cut = technical_signals(df.iloc[:400])
    pd.testing.assert_frame_equal(full.iloc[:400].reset_index(drop=True), cut.reset_index(drop=True))
    assert set(SIGNALS) <= set(full.columns)


def test_event_study_detects_real_edge():
    rng = np.random.default_rng(1)
    n = 40_000
    df = pd.DataFrame({"date": np.repeat(pd.bdate_range("2015-01-01", periods=n // 40, tz="UTC"), 40),
                       "symbol": np.tile([f"S{i}" for i in range(40)], n // 40),
                       "vol_20": rng.uniform(0.01, 0.03, n), "sig": rng.random(n) < 0.05})
    df["outcome_ret"] = rng.normal(0.005, 0.05, n) + 0.02 * df["sig"]
    res = event_study(df, "sig", fee=0.001)
    assert res["excedent"] == pytest.approx(0.02, abs=0.005)
    assert res["t_excedent"] > 5


def test_trend_following_goes_to_cash_below_ma200():
    dates = pd.bdate_range("2020-01-01", periods=4, tz="UTC")
    panel = pd.DataFrame({"date": np.tile(dates, 2), "symbol": np.repeat(["A", "B"], 4),
                          "ret_1": np.log([1, 1.1, 1.1, 1.1, 1, 0.9, 0.9, 0.9]),
                          "au_dessus_ma200": [True] * 4 + [False] * 4})
    trend, always, exposure = trend_following(panel, fee=0.0)
    assert exposure.iloc[0] == pytest.approx(0.5)  # B est hors tendance : moitié en liquidités
    assert trend.iloc[-1] == pytest.approx(1.05 ** 3)  # moitié dans A (+10 %/jour), moitié en cash
    assert always.iloc[-1] == pytest.approx(1.0)  # +10 % et −10 % se compensent chaque jour
