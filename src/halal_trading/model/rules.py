"""Règles classiques d'analyse technique, sans modèle.

Chaque signal vaut True à la CLÔTURE du jour J s'il est déclenché avec les seules
données connues à cette clôture. L'achat a lieu à l'ouverture de J+1, comme
pour le modèle, ce qui permet de comparer les deux sur les mêmes trades.

Figures de chandeliers (haussières), définitions usuelles :
- marteau : petit corps en haut de la bougie, longue mèche basse (≥ 2 × corps),
  après une baisse ;
- avalement haussier : bougie haussière dont le corps englobe celui de la bougie
  baissière précédente, après une baisse ;
- étoile du matin : grande bougie baissière, petite bougie plus bas, puis bougie
  haussière qui clôture au-dessus du milieu de la première ;
- trois soldats blancs : trois bougies haussières à clôtures croissantes, chacune
  ouvrant dans le corps de la précédente.
Indicateurs : croisement moyennes 50/200, croisement MACD (12, 26, 9), retour du
RSI 14 au-dessus de 30, cassure du plus haut des 252 séances précédentes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SIGNALS = {
    "marteau": "Marteau",
    "avalement_haussier": "Avalement haussier",
    "etoile_du_matin": "Étoile du matin",
    "trois_soldats_blancs": "Trois soldats blancs",
    "croisement_50_200": "Croisement des moyennes 50/200",
    "croisement_macd": "Croisement MACD",
    "rebond_rsi_30": "Retour du RSI au-dessus de 30",
    "cassure_52_semaines": "Cassure du plus haut 52 semaines",
}


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def technical_signals(ohlc: pd.DataFrame) -> pd.DataFrame:
    """Signaux par date pour un actif (entrée : prix ajustés, colonnes open/high/low/close)."""
    o, h, l, c = ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"]
    body = (c - o).abs()
    rng = (h - l).replace(0, np.nan)
    lower_shadow = np.minimum(o, c) - l
    upper_shadow = h - np.maximum(o, c)
    downtrend = c < c.shift(5)  # baisse sur la semaine précédente
    bull, bear = c > o, c < o

    s = pd.DataFrame({"date": ohlc["date"]})
    s["marteau"] = (body <= 0.3 * rng) & (lower_shadow >= 2 * body) & (upper_shadow <= 0.15 * rng) & downtrend.shift(1)
    s["avalement_haussier"] = bear.shift(1) & bull & (o <= c.shift(1)) & (c >= o.shift(1)) & downtrend.shift(1)
    long_bear_2 = bear.shift(2) & (body.shift(2) >= 0.6 * rng.shift(2))
    small_1 = body.shift(1) <= 0.3 * rng.shift(1)
    gap_down_1 = np.maximum(o.shift(1), c.shift(1)) < c.shift(2)
    s["etoile_du_matin"] = long_bear_2 & small_1 & gap_down_1 & bull & (c > (o.shift(2) + c.shift(2)) / 2)
    opens_in_prev_body = lambda k: (o.shift(k) >= o.shift(k + 1)) & (o.shift(k) <= c.shift(k + 1))  # noqa: E731
    s["trois_soldats_blancs"] = (bull & bull.shift(1) & bull.shift(2) & (c > c.shift(1)) & (c.shift(1) > c.shift(2))
                                 & opens_in_prev_body(0) & opens_in_prev_body(1))

    ma50, ma200 = c.rolling(50).mean(), c.rolling(200).mean()
    s["croisement_50_200"] = (ma50 > ma200) & (ma50.shift(1) <= ma200.shift(1))
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    signal = macd.ewm(span=9, adjust=False).mean()
    s["croisement_macd"] = (macd > signal) & (macd.shift(1) <= signal.shift(1))
    rsi = _rsi(c)
    s["rebond_rsi_30"] = (rsi > 30) & (rsi.shift(1) <= 30)
    s["cassure_52_semaines"] = c > c.shift(1).rolling(252).max()
    s["au_dessus_ma200"] = c > ma200  # état (pas un événement) : sert au suivi de tendance
    # les premiers jours n'ont pas assez d'historique : aucun signal
    for col in list(SIGNALS) + ["au_dessus_ma200"]:
        s[col] = s[col].fillna(False).astype(bool)
        s.loc[ma200.isna(), col] = False
    return s


def event_study(df: pd.DataFrame, signal: str, fee: float) -> dict[str, float]:
    """Gain des trades déclenchés par `signal`, comparé à un trade pris n'importe quel jour.

    - excedent : rendement net moyen des trades signalés − rendement moyen de tous les trades
      du même jour (on compare à ce qui était possible ce jour-là) ;
    - excedent_vol_comparable : même chose, mais comparé aux actions du même quintile de
      volatilité le même jour ;
    - t-stats calculés sur les moyennes mensuelles (les trades d'un même mois se chevauchent).
    `df` : date, outcome_ret, vol_20, colonnes de signaux.
    """
    work = df[["date", "outcome_ret", "vol_20", signal]].dropna(subset=["outcome_ret"]).copy()
    work["base_jour"] = work.groupby("date")["outcome_ret"].transform("mean")
    work["quintile"] = work.groupby("date")["vol_20"].transform(
        lambda v: pd.qcut(v.rank(method="first"), 5, labels=False) if len(v) >= 10 else 0)
    work["base_vol"] = work.groupby(["date", "quintile"])["outcome_ret"].transform("mean")
    hits = work[work[signal]]
    if hits.empty:
        return {"evenements": 0}
    month = hits["date"].dt.tz_localize(None).dt.to_period("M") if hits["date"].dt.tz is not None else hits["date"].dt.to_period("M")

    def t_stat(x: pd.Series) -> float:
        m = x.groupby(month).mean()
        return float(m.mean() / m.std() * np.sqrt(len(m))) if len(m) > 2 and m.std() > 0 else np.nan

    excess = hits["outcome_ret"] - hits["base_jour"]
    excess_vol = hits["outcome_ret"] - hits["base_vol"]
    return {
        "evenements": int(len(hits)),
        "par_action_et_par_an": float(len(hits) / work["date"].dt.year.nunique() / max(1, df["symbol"].nunique()))
        if "symbol" in df else np.nan,
        "gain_net_moyen": float(hits["outcome_ret"].mean() - 2 * fee),
        "gain_net_tous_trades": float(work["outcome_ret"].mean() - 2 * fee),
        "taux_gagnants": float((hits["outcome_ret"] - 2 * fee > 0).mean()),
        "excedent": float(excess.mean()), "t_excedent": t_stat(excess),
        "excedent_vol_comparable": float(excess_vol.mean()), "t_excedent_vol_comparable": t_stat(excess_vol),
    }


def trend_following(panel: pd.DataFrame, fee: float) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Portefeuille à poids égaux des actions dont le cours est au-dessus de leur moyenne
    200 jours (sinon liquidités, sans intérêt), comparé au même portefeuille toujours investi.

    Décision à la clôture de J, rendement de la clôture J à la clôture J+1.
    Renvoie (capital suivi de tendance, capital toujours investi, part investie).
    """
    p = panel.sort_values(["symbol", "date"])[["date", "symbol", "ret_1", "au_dessus_ma200"]].copy()
    p["next_ret"] = np.expm1(p.groupby("symbol")["ret_1"].shift(-1))
    p = p.dropna(subset=["next_ret"])
    p["n"] = p.groupby("date")["symbol"].transform("size")
    p["w_trend"] = p["au_dessus_ma200"] / p["n"]
    by_date = p.groupby("date")
    gross_trend = (p["w_trend"] * p["next_ret"]).groupby(p["date"]).sum()
    always = by_date["next_ret"].mean()
    exposure = by_date["w_trend"].sum()
    # coût : chaque entrée ou sortie d'une action paie les frais sur son poids
    weights = p.pivot(index="date", columns="symbol", values="w_trend").fillna(0)
    turnover = weights.diff().abs().sum(axis=1).fillna(0)
    net_trend = gross_trend - turnover.reindex(gross_trend.index).fillna(0) * fee
    return (1 + net_trend).cumprod(), (1 + always).cumprod(), exposure
