"""Momentum sur un univers point-in-time : les membres du S&P 500 à chaque date.

Compare la même stratégie sur deux univers :
- « membres d'aujourd'hui » : les membres actuels sur toute la période (biais du survivant) ;
- « membres à la date » : seulement les sociétés réellement dans l'indice ce jour-là.
Mesure aussi la couverture : part des (membre, jour) pour lesquels on a des prix.

Pas de filtre halal ici (il faut les comptes historiques EDGAR) : le but est de
mesurer l'effet du biais du survivant sur la stratégie elle-même.

Usage : python scripts/sp500_pit.py <sortie.json>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from halal_trading.data import sp500_history
from halal_trading.data.storage import ParquetStore
from halal_trading.model.backtest import performance, simulate
from halal_trading.model.dataset import build_dataset, load_model_config

ROOT = Path(__file__).resolve().parents[1]


def coverage(intervals: pd.DataFrame, store: ParquetStore, start: str) -> dict[str, float]:
    days = pd.bdate_range(start, pd.Timestamp.today().normalize(), freq="W-FRI")
    expected = covered = 0
    first_dates = {}
    for s in intervals["symbol"].unique():
        df = store.read_prices("stocks", s)
        first_dates[s] = (df["date"].dt.tz_localize(None).min(), df["date"].dt.tz_localize(None).max()) \
            if df is not None and len(df) else None
    for row in intervals.itertuples():
        mine = days[(days >= row.start) & (days < row.end)]
        expected += len(mine)
        rng = first_dates.get(row.symbol)
        if rng:
            covered += int(((mine >= rng[0]) & (mine <= rng[1])).sum())
    return {"membres_jours_attendus": expected, "membres_jours_couverts": covered,
            "couverture": covered / expected if expected else np.nan}


def main(out_path: str) -> None:
    store = ParquetStore(ROOT / "data")
    cfg = load_model_config(ROOT / "config" / "model.yaml", "stocks")
    intervals = sp500_history.load(ROOT / "data")
    start = f"{cfg.first_test_year}-01-01"
    cov = coverage(intervals[intervals["end"] > pd.Timestamp(start)], store, start)
    print("couverture", cov)

    symbols = sorted(set(intervals["symbol"]) & set(store.list_symbols("stocks")))
    panel = build_dataset(store, cfg, symbols)
    panel = panel[(panel["date"].dt.year >= cfg.first_test_year) & panel["label"].notna()].copy()
    panel["membre_a_la_date"] = sp500_history.membership_mask(panel, intervals).to_numpy()
    current = sp500_history.members_on(intervals, pd.Timestamp.today().normalize() - pd.Timedelta(days=1))
    panel["membre_aujourdhui"] = panel["symbol"].isin(current)
    rng = np.random.default_rng(0)
    panel["hasard"] = rng.random(len(panel))

    rows = []
    for universe, col in (("Membres d'aujourd'hui (biaisé)", "membre_aujourdhui"),
                          ("Membres à la date (point-in-time)", "membre_a_la_date")):
        sub = panel[panel[col]]
        daily = sub.groupby("date")["ret_1"].mean()
        bench = performance(np.exp(daily.cumsum()))
        for name, score in (("Momentum 120 j", "ret_120"), ("Hasard", "hasard")):
            perf = performance(simulate(sub, score, cfg, -np.inf).equity)
            rows.append({"univers": universe, "strategie": name, **perf})
        rows.append({"univers": universe, "strategie": "Acheter et garder", **bench})
        rows[-1]["actions_par_jour"] = float(sub.groupby("date").size().mean())
        print(universe, rows[-3:])

    meta = [
        {"label": "sp500_pit.py", "value": Path(__file__).read_text(encoding="utf-8"), "type": "code", "lang": "python"},
        {"label": "Couverture des prix", "value": f"{cov['couverture']:.1%} des (membre, semaine) depuis {start}"},
    ]
    clean = [{k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in r.items()} for r in rows]
    Path(out_path).write_text(json.dumps({"rows": clean, "meta": meta, "couverture": cov}, ensure_ascii=False),
                              encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
