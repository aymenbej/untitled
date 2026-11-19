"""Neutralisation de la volatilité appliquée au vrai modèle et au momentum.

Pour chaque score (modèle linéaire, gradient boosting, momentum 120 j) :
IC brut, IC de la prévision neutralisée, IC à volatilité comparable (quintiles),
IC sur rendement normalisé par la volatilité, avec leurs t-stats, puis simulation
de la stratégie top 5 avec le score brut et avec le score neutralisé.
Écrit un JSON {rows, meta} pour le tableau de bord.

Usage : python scripts/neutralisation.py <sortie.json>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from halal_trading.data.pipeline import load_data_config, stock_universe
from halal_trading.data.storage import ParquetStore
from halal_trading.model.backtest import performance, simulate
from halal_trading.model.dataset import build_dataset, compliant_stocks, load_model_config
from halal_trading.model.neutralization import neutralization_report, neutralize
from halal_trading.model.validation import run_walk_forward

ROOT = Path(__file__).resolve().parents[1]
MODES = {False: "Entraîné sur les actions halal", True: "Entraîné sur toutes les actions"}
COLS = ["date", "symbol", "outcome_ret", "label", "tp_ret", "sl_ret", "exit_date", "vol_20", "ret_120"]


def main(out_path: str) -> None:
    store = ParquetStore(ROOT / "data")
    cfg = load_model_config(ROOT / "config" / "model.yaml", "stocks")
    data_cfg = load_data_config(ROOT / "config" / "data_sources.yaml")
    universe = set(stock_universe(data_cfg, ROOT / "data"))
    symbols = [s for s in store.list_symbols("stocks") if s in universe]
    halal = set(compliant_stocks(ROOT / "data")) & universe
    panel = build_dataset(store, cfg, symbols, halal)
    test = panel[panel["halal"] & panel["label"].notna() & (panel["date"].dt.year >= cfg.first_test_year)][COLS]

    rows = []
    for train_on_all, label in MODES.items():
        preds, _, _ = run_walk_forward(panel, cfg.first_test_year, train_on_all)
        df = test.merge(preds[["date", "symbol", "pred_lineaire", "pred_gradient_boosting"]], on=["date", "symbol"])
        scores = {"Modèle linéaire": "pred_lineaire", "Gradient boosting": "pred_gradient_boosting"}
        if not train_on_all:
            scores["Momentum 120 j (sans modèle)"] = "ret_120"
        for name, col in scores.items():
            report = neutralization_report(df, col, cfg.horizon_days)
            df["neutre"] = neutralize(df, col)
            brut = performance(simulate(df.dropna(subset=[col]), col, cfg, -np.inf).equity)
            neutre = performance(simulate(df.dropna(subset=["neutre"]), "neutre", cfg, -np.inf).equity)
            rows.append({
                "entrainement": label if col.startswith("pred_") else "Sans entraînement",
                "score": name,
                **report,
                "rendement_annuel_brut": brut["rendement_annuel"], "sharpe_brut": brut["sharpe"],
                "perte_max_brut": brut["perte_max"],
                "rendement_annuel_neutre": neutre["rendement_annuel"], "sharpe_neutre": neutre["sharpe"],
                "perte_max_neutre": neutre["perte_max"],
            })
            print(rows[-1])

    meta = [
        {"label": "neutralisation.py", "value": Path(__file__).read_text(encoding="utf-8"), "type": "code",
         "lang": "python"},
        {"label": "Méthode", "value": (ROOT / "src/halal_trading/model/neutralization.py").read_text(encoding="utf-8"),
         "type": "code", "lang": "python"},
        {"label": "Univers", "value": f"{len(halal)} actions halal, test {cfg.first_test_year} → "
                                      f"{test['date'].max():%Y-%m-%d}"},
    ]
    clean = [{k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in r.items()} for r in rows]
    Path(out_path).write_text(json.dumps({"rows": clean, "meta": meta}, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
