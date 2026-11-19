"""Convertit le rapport de backtest (data/reports/backtest_stocks_*.json) en
jeux de données JSON {rows, meta} pour le tableau de bord d'évaluation.

Usage : python scripts/report_to_dashboard.py <rapport.json> <dossier_sortie>
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

MODES = {"halal": "Entraîné sur les actions halal", "tout": "Entraîné sur toutes les actions"}


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main(report_path: str, out_dir: str) -> None:
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    code = Path(__file__).read_text(encoding="utf-8")
    cfg = report["config"]
    meta = [
        {"label": "report_to_dashboard.py", "value": code, "type": "code", "lang": "python"},
        {"label": "Entrée", "value": Path(report_path).name},
        {"label": "Commande", "value": "halal-model backtest stocks"},
        {"label": "Réglage", "value": f"horizon {cfg['horizon_days']} j, objectif {cfg['tp_mult']}σ, "
                                       f"stop {cfg['sl_mult']}σ, frais {cfg['fee_per_side']:.2%} par ordre, "
                                       f"{cfg['max_positions']} positions"},
        {"label": "Univers", "value": f"{report['n_actifs']} actions dont {report['n_halal']} halal"},
    ]
    tables: dict[str, list[dict]] = {k: [] for k in
                                     ("evaluation", "strategies", "ic_annee", "deciles", "importance", "courbes")}
    for mode, block in report["modes"].items():
        label = MODES[mode]
        for model, metrics in block["evaluation"].items():
            tables["evaluation"].append({"entrainement": label, "modele": model,
                                         **{k: clean(v) for k, v in metrics.items()}})
        for row in block["strategies"]:
            tables["strategies"].append({"entrainement": label, **{k: clean(v) for k, v in row.items()}})
        for row in block["ic_par_annee"]:
            year = row["annee"]
            for model, ic in row.items():
                if model != "annee":
                    tables["ic_annee"].append({"entrainement": label, "annee": int(year), "modele": model,
                                               "ic": clean(ic)})
        for model, rows in block["deciles"].items():
            for row in rows:
                tables["deciles"].append({"entrainement": label, "modele": model,
                                          **{k: clean(v) for k, v in row.items()}})
        for model, values in block["importance"].items():
            for feature, drop in values.items():
                tables["importance"].append({"entrainement": label, "modele": model, "variable": feature,
                                             "baisse_ic": clean(drop)})
        for strategy, points in block["courbes"].items():
            for date, value in points.items():
                tables["courbes"].append({"entrainement": label, "strategie": strategy, "date": date,
                                          "valeur": clean(round(value, 5))})
    for name, rows in tables.items():
        (out / f"{name}.json").write_text(json.dumps({"rows": rows, "meta": meta}, ensure_ascii=False),
                                          encoding="utf-8")
        print(name, len(rows))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
