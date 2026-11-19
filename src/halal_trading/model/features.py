"""Caractéristiques (features) calculées à partir des prix journaliers.

Règle absolue : la valeur au jour J n'utilise que des données connues à la
clôture de J (aucune information du futur).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    "ret_1", "ret_5", "ret_10", "ret_20", "ret_60", "ret_120",
    "vol_20", "vol_60", "vol_ratio", "range_14", "rsi_14",
    "ma_dist_20", "ma_dist_50", "ma_dist_200", "drawdown_252",
    "volume_z_20", "gap_1",
]
CROSS_SECTIONAL = ["rank_ret_20", "rank_ret_60", "rank_vol_20", "market_ret_20", "market_vol_20"]


def adjusted_ohlc(prices: pd.DataFrame) -> pd.DataFrame:
    """Prix ajustés des dividendes et divisions, bougies incohérentes corrigées."""
    df = prices.sort_values("date").reset_index(drop=True).copy()
    factor = (df["adj_close"] / df["close"]).where(df["close"] > 0)
    out = pd.DataFrame({"date": df["date"], "volume": df["volume"].astype(float)})
    for col in ("open", "high", "low", "close"):
        out[col] = df[col] * factor
    # Erreurs de la source (ex. clôture au-dessus du plus haut) : on élargit la bougie.
    out["high"] = out[["open", "high", "low", "close"]].max(axis=1)
    out["low"] = out[["open", "high", "low", "close"]].min(axis=1)
    out = out[(out[["open", "high", "low", "close"]] > 0).all(axis=1)]
    return out.reset_index(drop=True)


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def compute_features(ohlc: pd.DataFrame) -> pd.DataFrame:
    """Features par date pour un actif (entrée : sortie de ``adjusted_ohlc``)."""
    close, high, low, open_ = ohlc["close"], ohlc["high"], ohlc["low"], ohlc["open"]
    logret = np.log(close).diff()
    f = pd.DataFrame({"date": ohlc["date"]})

    for n in (1, 5, 10, 20, 60, 120):
        f[f"ret_{n}"] = np.log(close / close.shift(n))
    f["vol_20"] = logret.rolling(20).std()
    f["vol_60"] = logret.rolling(60).std()
    f["vol_ratio"] = f["vol_20"] / f["vol_60"]
    f["range_14"] = ((high - low) / close).rolling(14).mean()
    f["rsi_14"] = _rsi(close)
    for n in (20, 50, 200):
        f[f"ma_dist_{n}"] = close / close.rolling(n).mean() - 1
    f["drawdown_252"] = close / close.rolling(252, min_periods=60).max() - 1
    logvol = np.log1p(ohlc["volume"])
    f["volume_z_20"] = (logvol - logvol.rolling(20).mean()) / logvol.rolling(20).std().replace(0, np.nan)
    f["gap_1"] = open_ / close.shift(1) - 1
    return f


def add_cross_sectional(panel: pd.DataFrame) -> pd.DataFrame:
    """Position de chaque actif par rapport aux autres le même jour, et contexte de marché."""
    by_date = panel.groupby("date")
    panel["rank_ret_20"] = by_date["ret_20"].rank(pct=True)
    panel["rank_ret_60"] = by_date["ret_60"].rank(pct=True)
    panel["rank_vol_20"] = by_date["vol_20"].rank(pct=True)
    panel["market_ret_20"] = by_date["ret_20"].transform("mean")
    panel["market_vol_20"] = by_date["vol_20"].transform("mean")
    return panel
