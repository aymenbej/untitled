"""Diagnostic : le rendement du modèle vient-il de son pouvoir de prévision ou
d'un simple penchant pour certaines actions (volatiles, en tendance) ?

Compare, sur les actions halal et la même période de test, le modèle linéaire à
des classements naïfs sans modèle. Écrit un JSON {rows, meta} pour le tableau de bord.

Usage : python scripts/diagnostics.py <sortie.json>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from halal_trading.data.pipeline import load_data_config, stock_universe
from halal_trading.data.storage import ParquetStore
from halal_trading.model.backtest import buy_and_hold, performance, simulate
from halal_trading.model.dataset import build_dataset, compliant_stocks, load_model_config
from halal_trading.model.validation import run_walk_forward

ROOT = Path(__file__).resolve().parents[1]


def main(out_path: str) -> None:
    store = ParquetStore(ROOT / "data")
    cfg = load_model_config(ROOT / "config" / "model.yaml", "stocks")
    data_cfg = load_data_config(ROOT / "config" / "data_sources.yaml")
    universe = set(stock_universe(data_cfg, ROOT / "data"))
    symbols = [s for s in store.list_symbols("stocks") if s in universe]
    halal = set(compliant_stocks(ROOT / "data")) & universe
    panel = build_dataset(store, cfg, symbols, halal)

    preds, _, _ = run_walk_forward(panel, cfg.first_test_year, train_on_all=False)
    test = panel[panel["halal"] & panel["label"].notna() & (panel["date"].dt.year >= cfg.first_test_year)]
    test = test.merge(preds[["date", "symbol", "pred_lineaire"]], on=["date", "symbol"])
    corr_vol = float(test.groupby("date").apply(
        lambda g: spearmanr(g["pred_lineaire"], g["vol_20"]).statistic).mean())

    rng = np.random.default_rng(0)
    variants = [
        ("Modèle linéaire", test["pred_lineaire"]),
        ("Les plus volatiles (20 j)", test["vol_20"]),
        ("Les moins volatiles (20 j)", -test["vol_20"]),
        ("Momentum 120 j", test["ret_120"]),
        ("Hasard", rng.random(len(test))),
    ]
    rows = []
    for name, score in variants:
        frame = test.assign(score=np.asarray(score))
        result = simulate(frame, "score", cfg, -np.inf)
        perf = performance(result.equity)
        rows.append({"strategie": name, "rendement_annuel": perf["rendement_annuel"], "sharpe": perf["sharpe"],
                     "perte_max": perf["perte_max"], "gain_moyen_par_trade": float(result.trades["net_ret"].mean())})
    bench = performance(buy_and_hold(panel[panel["halal"]], test["date"].min(), test["date"].max()))
    rows.append({"strategie": "Acheter et garder (halal)", "rendement_annuel": bench["rendement_annuel"],
                 "sharpe": bench["sharpe"], "perte_max": bench["perte_max"], "gain_moyen_par_trade": None})

    meta = [
        {"label": "diagnostics.py", "value": Path(__file__).read_text(encoding="utf-8"), "type": "code",
         "lang": "python"},
        {"label": "Corrélation prévision ~ volatilité", "value": f"{corr_vol:+.2f} (rang, moyenne par jour)"},
        {"label": "Période", "value": f"{test['date'].min():%Y-%m-%d} → {test['date'].max():%Y-%m-%d}"},
    ]
    Path(out_path).write_text(json.dumps({"rows": rows, "meta": meta, "correlation_volatilite": corr_vol},
                                         ensure_ascii=False), encoding="utf-8")
    print(f"corrélation prévision ~ volatilité : {corr_vol:+.2f}")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main(sys.argv[1])
