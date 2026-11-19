"""Évaluation des règles d'analyse technique, sans modèle, sur les actions halal.

1. Étude d'événements : pour chaque signal (chandeliers, croisements, RSI, cassure),
   gain des trades déclenchés comparé à un trade pris n'importe quel jour, avec et
   sans contrôle de la volatilité, t-stats sur moyennes mensuelles.
2. Portefeuille : 5 positions choisies parmi les actions signalées du jour (au
   hasard si plus de 5), mêmes sorties que le modèle (objectif, stop, 20 jours).
3. Suivi de tendance : actions détenues seulement au-dessus de leur moyenne 200 jours.

Usage : python scripts/regles_techniques.py <dossier_sortie>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from halal_trading.data.pipeline import load_data_config, stock_universe
from halal_trading.data.storage import ParquetStore
from halal_trading.model.backtest import buy_and_hold, performance, simulate
from halal_trading.model.dataset import build_dataset, compliant_stocks, load_model_config
from halal_trading.model.features import adjusted_ohlc
from halal_trading.model.rules import SIGNALS, event_study, technical_signals, trend_following

ROOT = Path(__file__).resolve().parents[1]


def clean(v):
    return None if isinstance(v, float) and not np.isfinite(v) else v


def main(out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    store = ParquetStore(ROOT / "data")
    cfg = load_model_config(ROOT / "config" / "model.yaml", "stocks")
    data_cfg = load_data_config(ROOT / "config" / "data_sources.yaml")
    universe = set(stock_universe(data_cfg, ROOT / "data"))
    halal = sorted(set(compliant_stocks(ROOT / "data")) & universe & set(store.list_symbols("stocks")))

    panel = build_dataset(store, cfg, halal)
    signals = []
    for symbol in halal:
        prices = store.read_prices("stocks", symbol)
        if prices is None or len(prices) < 300:
            continue
        s = technical_signals(adjusted_ohlc(prices))
        s["symbol"] = symbol
        signals.append(s)
    panel = panel.merge(pd.concat(signals, ignore_index=True), on=["date", "symbol"], how="left")
    panel[list(SIGNALS) + ["au_dessus_ma200"]] = panel[list(SIGNALS) + ["au_dessus_ma200"]].fillna(False)
    test = panel[(panel["date"].dt.year >= cfg.first_test_year) & panel["label"].notna()].copy()
    start, end = test["date"].min(), test["date"].max()
    rng = np.random.default_rng(0)
    test["hasard"] = rng.random(len(test))

    # 1 + 2. Signaux d'entrée
    rows = []
    for key, name in SIGNALS.items():
        stats = event_study(test, key, cfg.fee_per_side)
        picked = test[test[key]].copy()
        perf = performance(simulate(picked, "hasard", cfg, -np.inf).equity) if len(picked) else {}
        rows.append({"regle": name, **stats, **{f"portefeuille_{k}": v for k, v in perf.items()}})
        print(name, {k: round(v, 4) if isinstance(v, float) else v for k, v in rows[-1].items() if k != "regle"})
    for name, col in (("Hasard (toutes les actions)", "hasard"), ("Momentum 120 j", "ret_120")):
        perf = performance(simulate(test, col, cfg, -np.inf).equity)
        rows.append({"regle": name, **{f"portefeuille_{k}": v for k, v in perf.items()}})
    bench = performance(buy_and_hold(panel, start, end))
    rows.append({"regle": "Acheter et garder (halal)", **{f"portefeuille_{k}": v for k, v in bench.items()}})

    # 3. Suivi de tendance
    period = panel[(panel["date"] >= start) & (panel["date"] <= end)]
    trend, always, exposure = trend_following(period, cfg.fee_per_side)
    trend_rows = [
        {"strategie": "Au-dessus de la moyenne 200 jours", **performance(trend),
         "exposition_moyenne": float(exposure.mean())},
        {"strategie": "Toujours investi (poids égaux)", **performance(always), "exposition_moyenne": 1.0},
    ]
    for r in trend_rows:
        print(r)
    curves = []
    for name, series in (("Au-dessus de la moyenne 200 jours", trend), ("Toujours investi (poids égaux)", always)):
        weekly = series.resample("W").last().dropna()
        curves += [{"strategie": name, "date": str(d.date()), "valeur": round(float(v), 5)} for d, v in weekly.items()]

    meta = [
        {"label": "regles_techniques.py", "value": Path(__file__).read_text(encoding="utf-8"), "type": "code",
         "lang": "python"},
        {"label": "Règles", "value": (ROOT / "src/halal_trading/model/rules.py").read_text(encoding="utf-8"),
         "type": "code", "lang": "python"},
        {"label": "Univers", "value": f"{len(halal)} actions halal, {start:%Y-%m-%d} → {end:%Y-%m-%d}, "
                                      f"frais {cfg.fee_per_side:.2%} par ordre"},
    ]
    for name, data in (("regles", rows), ("tendance", trend_rows), ("tendance_courbes", curves)):
        payload = [{k: clean(v) for k, v in r.items()} for r in data]
        (out / f"{name}.json").write_text(json.dumps({"rows": payload, "meta": meta}, ensure_ascii=False),
                                          encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
