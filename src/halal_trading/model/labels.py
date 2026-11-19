"""Étiquettes « triple barrière » : l'objectif est-il atteint avant le stop ?

Hypothèses prudentes :
- décision à la clôture de J, achat à l'OUVERTURE de J+1 ;
- si objectif et stop sont touchés le même jour, on considère le stop touché d'abord ;
- si le cours ouvre sous le stop (trou de cotation), la sortie se fait à
  l'ouverture, donc avec une perte plus grande que le stop ;
- si le cours ouvre au-dessus de l'objectif, l'ordre de vente à cours limite
  s'exécute à l'ouverture (gain supérieur à l'objectif).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def triple_barrier(
    ohlc: pd.DataFrame,
    daily_vol: pd.Series,
    horizon: int,
    tp_mult: float,
    sl_mult: float,
) -> pd.DataFrame:
    o, h, l, c = (ohlc[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    vol = daily_vol.to_numpy(float)
    dates = pd.to_datetime(ohlc["date"], utc=True).dt.tz_convert(None).to_numpy("datetime64[ns]")
    n = len(ohlc)

    label = np.full(n, np.nan)
    outcome = np.full(n, np.nan)
    tp_ret = np.full(n, np.nan)
    sl_ret = np.full(n, np.nan)
    exit_date = np.full(n, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")

    scale = np.sqrt(horizon)
    for i in range(n - horizon - 1):
        if not np.isfinite(vol[i]) or vol[i] <= 0:
            continue
        entry = o[i + 1]
        up, down = tp_mult * vol[i] * scale, sl_mult * vol[i] * scale
        tp_price, sl_price = entry * (1 + up), entry * (1 - down)
        result, hit, exit_j = 0.0, None, i + horizon
        for j in range(i + 1, i + horizon + 1):
            if j > i + 1 and o[j] <= sl_price:  # ouverture sous le stop
                result, hit, exit_j = o[j] / entry - 1, 0, j
                break
            if l[j] <= sl_price:
                result, hit, exit_j = -down, 0, j
                break
            if j > i + 1 and o[j] >= tp_price:  # ouverture au-dessus de l'objectif : l'ordre limite s'exécute à l'ouverture
                result, hit, exit_j = o[j] / entry - 1, 1, j
                break
            if h[j] >= tp_price:
                result, hit, exit_j = up, 1, j
                break
        if hit is None:  # échéance : on vend à la clôture du dernier jour
            result, hit = c[i + horizon] / entry - 1, 0
        label[i], outcome[i], tp_ret[i], sl_ret[i] = hit, result, up, down
        exit_date[i] = dates[exit_j]

    return pd.DataFrame(
        {"date": ohlc["date"], "label": label, "outcome_ret": outcome, "tp_ret": tp_ret, "sl_ret": sl_ret,
         "exit_date": pd.to_datetime(exit_date).tz_localize("UTC")}
    )
