"""Ligne de commande du modèle.

    halal-model backtest stocks    # validation walk-forward + simulation
    halal-model backtest crypto
    halal-model predict stocks     # recommandations du jour
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from halal_trading.compliance import load_policy
from halal_trading.data.pipeline import load_data_config, stock_universe
from halal_trading.data.storage import ParquetStore

from .backtest import buy_and_hold, decile_table, expected_return, performance, simulate
from .dataset import (
    ALL_FEATURES,
    build_dataset,
    compliant_stocks,
    load_model_config,
    load_model_name,
    load_training_universe,
)
from .models import MODEL_NAMES, make_model
from .validation import evaluate, permutation_importance, run_walk_forward

ROOT = Path(__file__).resolve().parents[3]


def _universe(asset_class: str, storage_dir: Path, data_cfg, store: ParquetStore) -> tuple[list[str], set[str]]:
    """(actifs disponibles pour l'entraînement, actifs halal autorisés au trading)."""
    if asset_class == "stocks":
        universe = set(stock_universe(data_cfg, storage_dir))
        available = [s for s in store.list_symbols("stocks") if s in universe]
        halal = set(compliant_stocks(storage_dir)) & universe
        return available, halal
    policy = load_policy(ROOT / "config" / "compliance.yaml")
    approved = {f"{a.symbol}-EUR" for a in policy.crypto_whitelist}
    return list(data_cfg.crypto_products), approved


def _pct(x: float) -> str:
    return f"{x:+.1%}" if np.isfinite(x) else "n/a"


def _strategies(preds, panel_halal, cfg) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    start, end = preds["date"].min(), preds["date"].max()
    bench_equity = buy_and_hold(panel_halal, start, end)
    rows, curves = [], {"acheter et garder (halal)": bench_equity}
    for name in MODEL_NAMES:
        for label, threshold in ((f"top {cfg.max_positions}, investi", -np.inf), ("seuil", None)):
            if name == "reference" and threshold is None:
                continue
            result = simulate(preds, f"pred_{name}", cfg, threshold)
            t = result.trades
            key = f"{name} ({label})"
            curves[key] = result.equity
            rows.append({"strategie": key, **performance(result.equity), "trades": len(t),
                         "trades_gagnants": float((t["net_ret"] > 0).mean()) if len(t) else np.nan,
                         "gain_moyen_par_trade": float(t["net_ret"].mean()) if len(t) else np.nan,
                         "exposition_moyenne": float(result.exposure.mean())})
    rows.append({"strategie": "acheter et garder (halal)", **performance(bench_equity)})
    return pd.DataFrame(rows).set_index("strategie"), curves


def cmd_backtest(asset_class: str, store: ParquetStore, storage_dir: Path, data_cfg) -> int:
    cfg = load_model_config(ROOT / "config" / "model.yaml", asset_class)
    symbols, halal = _universe(asset_class, storage_dir, data_cfg, store)
    if asset_class == "crypto":
        halal = set(symbols)  # recherche uniquement : la liste blanche crypto est vide
    panel = build_dataset(store, cfg, symbols, halal)
    n_halal = panel.loc[panel["halal"], "symbol"].nunique()
    print(f"Jeu de données {asset_class} : {len(panel):,} lignes, {panel['symbol'].nunique()} actifs "
          f"dont {n_halal} halal, {panel['date'].min():%Y-%m-%d} → {panel['date'].max():%Y-%m-%d}")
    print(f"Cible : rendement du trade, objectif {cfg.tp_mult}σ√h / stop {cfg.sl_mult}σ√h, h = {cfg.horizon_days} "
          f"jours ; frais {cfg.fee_per_side:.2%} par ordre. Évaluation sur les actions halal uniquement.\n")

    report: dict = {"config": cfg.__dict__, "n_lignes": len(panel), "n_actifs": int(panel["symbol"].nunique()),
                    "n_halal": int(n_halal), "modes": {}}
    modes = [("halal", False), ("tout", True)] if n_halal < panel["symbol"].nunique() else [("halal", False)]
    for mode, train_on_all in modes:
        preds, yearly, models = run_walk_forward(panel, cfg.first_test_year, train_on_all)
        print(f"=== Entraînement sur : {'toutes les actions' if train_on_all else 'actions halal seulement'} ===")
        evals = pd.DataFrame({name: evaluate(preds, f"pred_{name}", cfg.horizon_days)
                              for name in MODEL_NAMES[1:]}).T
        print(evals.round(4).to_string(), "\n")
        table, curves = _strategies(preds, panel[panel["halal"]], cfg)
        display = table.copy()
        for col in ("rendement_total", "rendement_annuel", "perte_max", "trades_gagnants",
                    "gain_moyen_par_trade", "exposition_moyenne"):
            display[col] = display[col].map(lambda v: _pct(v) if pd.notna(v) else "")
        display["sharpe"] = display["sharpe"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "")
        display["trades"] = display["trades"].map(lambda v: f"{int(v)}" if pd.notna(v) else "")
        print(display.to_string(), "\n")
        ic_year = yearly.pivot(index="annee", columns="modele", values="ic")
        print("IC par année :")
        print(ic_year.round(3).T.to_string(), "\n")

        deciles = {name: decile_table(preds, f"pred_{name}", cfg.fee_per_side) for name in MODEL_NAMES[1:]}
        last_year = preds["date"].dt.year.max()
        test_last = panel[(panel["date"].dt.year == last_year) & panel["halal"] & panel["label"].notna()]
        importance = {name: permutation_importance(models[name], test_last) for name in MODEL_NAMES[1:]}
        report["modes"][mode] = {
            "evaluation": evals.to_dict(orient="index"),
            "strategies": table.reset_index().to_dict(orient="records"),
            "ic_par_annee": ic_year.reset_index().to_dict(orient="records"),
            "deciles": {k: v.reset_index().to_dict(orient="records") for k, v in deciles.items()},
            "importance": {k: v.to_dict() for k, v in importance.items()},
            "courbes": {k: {str(d.date()): float(v) for d, v in c.resample("W").last().dropna().items()}
                        for k, c in curves.items()},
        }

    out = storage_dir / "reports" / f"backtest_{asset_class}_{datetime.now(timezone.utc):%Y-%m-%d}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"Rapport détaillé : {out}")
    return 0


def cmd_predict(asset_class: str, store: ParquetStore, storage_dir: Path, data_cfg) -> int:
    cfg = load_model_config(ROOT / "config" / "model.yaml", asset_class)
    symbols, tradable = _universe(asset_class, storage_dir, data_cfg, store)
    panel = build_dataset(store, cfg, symbols, tradable)
    train = panel[panel["label"].notna()]
    if load_training_universe(ROOT / "config" / "model.yaml") == "halal":
        train = train[train["halal"]]
    model_name = load_model_name(ROOT / "config" / "model.yaml")
    model = make_model(model_name).fit(
        train[ALL_FEATURES].to_numpy(float), train["outcome_ret"].to_numpy(float)
    )

    # Dernière cotation de chaque actif (les places n'ont pas toutes le même calendrier),
    # en ignorant les actifs dont les données ont plus de 5 jours.
    last_day = panel["date"].max()
    today = panel.sort_values("date").groupby("symbol").tail(1)
    today = today[(today["date"] >= last_day - pd.Timedelta(days=5)) & today["halal"]].copy()
    today["prevu"] = model.predict(today[ALL_FEATURES].to_numpy(float))
    sigma = today["vol_20"] * np.sqrt(cfg.horizon_days)
    today["objectif"], today["stop"] = cfg.tp_mult * sigma, cfg.sl_mult * sigma
    today["esperance"] = expected_return(today["prevu"], cfg.fee_per_side)
    today = today.sort_values("esperance", ascending=False)

    print(f"Prévisions {asset_class} au {last_day:%Y-%m-%d} (modèle {model_name}, horizon {cfg.horizon_days} jours) :\n")
    print(f"  {'actif':<11}{'objectif':>10}{'stop':>9}{'espérance nette':>17}  décision")
    for r in today.itertuples():
        if r.symbol not in tradable:
            decision = "non tradable (non validé halal)"
        elif r.esperance > cfg.min_expected_return:
            decision = "ACHAT candidat"
        else:
            decision = "-"
        print(f"  {r.symbol:<11}{r.objectif:>+10.1%}{-r.stop:>+9.1%}{r.esperance:>+17.2%}  {decision}")
    print("\nAucun ordre n'est passé : ces signaux sont indicatifs (phase de recherche).")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="halal-model", description="Modèle de prévision")
    parser.add_argument("command", choices=["backtest", "predict"])
    parser.add_argument("asset_class", choices=["stocks", "crypto"])
    args = parser.parse_args(argv)

    data_cfg = load_data_config(ROOT / "config" / "data_sources.yaml")
    storage_dir = data_cfg.storage_dir if data_cfg.storage_dir.is_absolute() else ROOT / data_cfg.storage_dir
    store = ParquetStore(storage_dir)
    if args.command == "backtest":
        return cmd_backtest(args.asset_class, store, storage_dir, data_cfg)
    return cmd_predict(args.asset_class, store, storage_dir, data_cfg)


if __name__ == "__main__":
    sys.exit(main())
