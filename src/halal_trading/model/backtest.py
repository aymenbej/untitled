"""Simulation de la stratégie sur les prédictions hors échantillon.

Chaque jour : pour chaque actif, espérance de gain = rendement prévu du trade
moins les frais (achat + vente). On achète les meilleures espérances au-dessus
du seuil, dans la limite de
`max_positions` positions de taille égale, une seule position par actif.
Chaque position sort à l'objectif, au stop ou à l'échéance.

Simplification : entre l'achat et la vente, une position est comptée à son prix
d'achat. La perte maximale (drawdown) affichée est donc un peu sous-estimée.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .dataset import ModelConfig


def expected_return(predicted: pd.Series, fee: float) -> pd.Series:
    return predicted - 2 * fee


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity: pd.Series
    exposure: pd.Series


def simulate(
    preds: pd.DataFrame, pred_col: str, cfg: ModelConfig, min_expected_return: float | None = None
) -> BacktestResult:
    """``min_expected_return=-inf`` : toujours investi dans les meilleures prévisions."""
    threshold = cfg.min_expected_return if min_expected_return is None else min_expected_return
    df = preds.copy()
    df["ev"] = expected_return(df[pred_col], cfg.fee_per_side)
    df = df[df["ev"] > threshold].sort_values(["date", "ev"], ascending=[True, False])
    candidates = {d: g for d, g in df.groupby("date")}

    cash, open_pos, trades = 1.0, [], []
    equity, exposure = {}, {}
    for day in sorted(preds["date"].unique()):
        # 1. sorties
        still_open = []
        for pos in open_pos:
            if pos["exit_date"] <= day:
                net = pos["outcome_ret"] - 2 * cfg.fee_per_side
                cash += pos["alloc"] * (1 + net)
                trades.append({**pos, "net_ret": net})
            else:
                still_open.append(pos)
        open_pos = still_open
        invested = sum(p["alloc"] for p in open_pos)
        total = cash + invested

        # 2. entrées
        if day in candidates:
            held = {p["symbol"] for p in open_pos}
            for row in candidates[day].itertuples():
                if len(open_pos) >= cfg.max_positions:
                    break
                if row.symbol in held:
                    continue
                alloc = total / cfg.max_positions
                if alloc > cash + 1e-12:
                    break
                cash -= alloc
                held.add(row.symbol)
                open_pos.append(
                    {"symbol": row.symbol, "decision_date": day, "exit_date": row.exit_date, "alloc": alloc,
                     "outcome_ret": row.outcome_ret, "ev": row.ev}
                )
        invested = sum(p["alloc"] for p in open_pos)
        equity[day] = cash + invested
        exposure[day] = invested / (cash + invested)

    for pos in open_pos:  # positions encore ouvertes en fin de période
        net = pos["outcome_ret"] - 2 * cfg.fee_per_side
        trades.append({**pos, "net_ret": net})

    return BacktestResult(pd.DataFrame(trades), pd.Series(equity).sort_index(), pd.Series(exposure).sort_index())


def performance(equity: pd.Series) -> dict[str, float]:
    if equity.empty:
        return {}
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    total = equity.iloc[-1] / equity.iloc[0] - 1
    daily = equity.pct_change().dropna()
    sharpe = float(daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else 0.0
    drawdown = float((equity / equity.cummax() - 1).min())
    return {
        "rendement_total": float(total),
        "rendement_annuel": float((1 + total) ** (1 / years) - 1),
        "sharpe": sharpe,
        "perte_max": drawdown,
    }


def buy_and_hold(panel: pd.DataFrame, start, end) -> pd.Series:
    """Référence passive : tous les actifs de l'univers à poids égaux."""
    window = panel[(panel["date"] >= start) & (panel["date"] <= end)]
    daily = window.groupby("date")["ret_1"].mean()
    return np.exp(daily.cumsum())


def decile_table(preds: pd.DataFrame, pred_col: str, fee: float, bins: int = 10) -> pd.DataFrame:
    """Rendement réel moyen selon le rendement prévu (par décile) : un bon modèle
    a des déciles croissants."""
    df = preds.copy()
    df["decile"] = pd.qcut(df[pred_col].rank(method="first"), bins, labels=False) + 1
    return df.groupby("decile").agg(
        prevu=(pred_col, "mean"),
        reel_net=("outcome_ret", lambda r: r.mean() - 2 * fee),
        objectif_atteint=("label", "mean"),
        n=("label", "size"),
    )
