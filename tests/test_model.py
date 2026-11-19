"""Tests du modèle : étiquettes, absence de fuite du futur, validation, simulation."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from halal_trading.model.backtest import performance, simulate
from halal_trading.model.dataset import ModelConfig
from halal_trading.model.features import adjusted_ohlc, compute_features
from halal_trading.model.labels import triple_barrier
from halal_trading.model.validation import evaluate, run_walk_forward, walk_forward_folds

CFG = ModelConfig("stocks", horizon_days=3, tp_mult=1.0, sl_mult=1.0, fee_per_side=0.001,
                  first_test_year=2021, max_positions=2, min_expected_return=0.0)


def ohlc(opens, highs, lows, closes):
    n = len(opens)
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n, tz="UTC"),
        "open": opens, "high": highs, "low": lows, "close": closes, "volume": 1.0,
    })


def label_first(df, vol=0.05, horizon=3):
    """Barrières à ±vol×√horizon autour de l'ouverture du 2e jour."""
    return triple_barrier(df, pd.Series([vol / np.sqrt(horizon)] * len(df)), horizon, 1.0, 1.0).iloc[0]


# ------------------------------------------------------------------ étiquettes
def test_take_profit_hit():
    row = label_first(ohlc([100, 100, 101, 102, 103], [100, 101, 106, 103, 104], [100, 99, 100, 101, 102],
                           [100, 100, 104, 102, 103]))
    assert row.label == 1 and row.outcome_ret == pytest.approx(0.05)


def test_stop_hit():
    row = label_first(ohlc([100, 100, 99, 98, 97], [100, 101, 100, 99, 98], [100, 99, 94, 97, 96],
                           [100, 100, 96, 98, 97]))
    assert row.label == 0 and row.outcome_ret == pytest.approx(-0.05)


def test_both_barriers_same_day_counts_as_stop():
    row = label_first(ohlc([100, 100, 100, 100, 100], [100, 101, 110, 101, 101], [100, 99, 90, 99, 99],
                           [100, 100, 100, 100, 100]))
    assert row.label == 0 and row.outcome_ret == pytest.approx(-0.05)


def test_gap_down_exits_at_open_with_bigger_loss():
    row = label_first(ohlc([100, 100, 90, 90, 90], [100, 101, 91, 91, 91], [100, 99, 89, 89, 89],
                           [100, 100, 90, 90, 90]))
    assert row.outcome_ret == pytest.approx(-0.10)


def test_gap_up_limit_order_fills_at_open():
    row = label_first(ohlc([100, 100, 110, 110, 110], [100, 101, 111, 111, 111], [100, 99, 109, 109, 109],
                           [100, 100, 110, 110, 110]))
    assert row.label == 1 and row.outcome_ret == pytest.approx(0.10)


def test_timeout_sells_at_last_close():
    row = label_first(ohlc([100, 100, 100, 100, 100], [100, 101, 101, 103, 101], [100, 99, 99, 99, 99],
                           [100, 100, 100, 102, 100]))
    assert row.label == 0 and row.outcome_ret == pytest.approx(0.02)
    assert row.exit_date == pd.Timestamp("2024-01-04", tz="UTC")


def test_last_rows_have_no_label():
    df = ohlc([100] * 6, [101] * 6, [99] * 6, [100] * 6)
    labels = triple_barrier(df, pd.Series([0.01] * 6), 3, 1.0, 1.0)
    assert labels["label"].iloc[-4:].isna().all()


# ------------------------------------------------------------------ fuite du futur
def random_prices(n=400, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    return pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=n, tz="UTC"),
        "open": close * (1 + rng.normal(0, 0.003, n)), "high": close * 1.01, "low": close * 0.99,
        "close": close, "adj_close": close, "volume": rng.uniform(1e5, 2e5, n),
    })


def test_features_do_not_use_future_data():
    prices = random_prices()
    full = compute_features(adjusted_ohlc(prices))
    truncated = compute_features(adjusted_ohlc(prices.iloc[:300]))
    pd.testing.assert_frame_equal(full.iloc[:300].reset_index(drop=True), truncated.reset_index(drop=True))


def test_adjusted_ohlc_fixes_incoherent_candles():
    prices = random_prices(10)
    prices.loc[3, "close"] = prices.loc[3, "high"] * 1.5
    adj = adjusted_ohlc(prices)
    assert (adj["high"] >= adj[["open", "close", "low"]].max(axis=1)).all()


# ------------------------------------------------------------------ validation
def synthetic_panel(n_days=1500, symbols=("A", "B", "C", "D", "E", "F"), seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n_days, tz="UTC")
    from halal_trading.model.dataset import ALL_FEATURES

    rows = []
    for s in symbols:
        signal = rng.normal(0, 1, n_days)
        frame = pd.DataFrame({f: rng.normal(0, 1, n_days) for f in ALL_FEATURES})
        frame["ret_20"] = signal  # une feature réellement prédictive
        frame["date"], frame["symbol"] = dates, s
        frame["outcome_ret"] = 0.01 * signal + rng.normal(0, 0.02, n_days)
        frame["label"] = (frame["outcome_ret"] > 0.02).astype(float)
        frame["tp_ret"], frame["sl_ret"] = 0.05, 0.05
        frame["exit_date"] = dates + pd.Timedelta(days=28)
        frame["ret_1"] = rng.normal(0.0005, 0.01, n_days)
        rows.append(frame)
    return pd.concat(rows, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)


def test_walk_forward_never_trains_on_test_period():
    panel = synthetic_panel()
    folds = walk_forward_folds(panel, 2021)
    assert [f.year for f in folds] == [2021, 2022, 2023]
    for f in folds:
        start = pd.Timestamp(year=f.year, month=1, day=1, tz="UTC")
        assert (panel.iloc[f.train]["exit_date"] < start).all()  # purge : étiquettes connues avant le test
        assert (panel.iloc[f.test]["date"].dt.year == f.year).all()


def test_models_find_a_real_signal_and_beat_reference():
    preds, metrics, _ = run_walk_forward(synthetic_panel(), 2021)
    ic = metrics.groupby("modele")["ic"].mean()
    assert ic["lineaire"] > 0.2 and ic["gradient_boosting"] > 0.2
    scores = evaluate(preds, "pred_lineaire", horizon=20)
    assert scores["t_stat"] > 3 and scores["r2_hors_echantillon"] > 0
    assert scores["ecart_decile_haut_bas"] > 0


def test_train_on_all_but_test_only_on_halal():
    panel = synthetic_panel(symbols=("A", "B", "C", "D", "E", "F", "BANK"))
    panel["halal"] = panel["symbol"] != "BANK"
    for train_on_all in (False, True):
        for f in walk_forward_folds(panel, 2021, train_on_all):
            assert panel.iloc[f.test]["halal"].all()  # jamais d'évaluation sur un actif non halal
            assert panel.iloc[f.train]["halal"].all() != train_on_all
    preds, _, _ = run_walk_forward(panel, 2021, train_on_all=True)
    assert "BANK" not in set(preds["symbol"])


# ------------------------------------------------------------------ simulation
def test_simulation_respects_position_limit_and_fees():
    dates = pd.date_range("2024-01-01", periods=3, tz="UTC")
    preds = pd.DataFrame({
        "date": [dates[0]] * 3 + [dates[2]],
        "symbol": ["A", "B", "C", "A"],
        "pred_x": [0.05, 0.04, 0.03, 0.05],
        "outcome_ret": [0.10, -0.05, 0.20, 0.0],
        "exit_date": [dates[2]] * 3 + [dates[2] + pd.Timedelta(days=5)],
        "label": [1, 0, 1, 0], "tp_ret": 0.1, "sl_ret": 0.05,
    })
    result = simulate(preds, "pred_x", CFG)
    first = result.trades[result.trades["decision_date"] == dates[0]]
    assert sorted(first["symbol"]) == ["A", "B"]  # 2 positions max, les meilleures prévisions
    # 0,5 × (1 + 0,10 − 0,002) + 0,5 × (1 − 0,05 − 0,002) = 1,023
    assert result.equity.loc[dates[2]] == pytest.approx(1.023)


def test_threshold_filters_trades():
    preds = pd.DataFrame({
        "date": [pd.Timestamp("2024-01-01", tz="UTC")], "symbol": ["A"], "pred_x": [0.001],
        "outcome_ret": [0.1], "exit_date": [pd.Timestamp("2024-01-05", tz="UTC")],
        "label": [1], "tp_ret": 0.1, "sl_ret": 0.05,
    })
    assert simulate(preds, "pred_x", replace(CFG, min_expected_return=0.01)).trades.empty


def test_performance_metrics():
    equity = pd.Series([1.0, 1.1, 0.88, 1.21], index=pd.date_range("2024-01-01", periods=4, tz="UTC"))
    perf = performance(equity)
    assert perf["rendement_total"] == pytest.approx(0.21)
    assert perf["perte_max"] == pytest.approx(-0.2)
