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
from halal_trading.data.pipeline import load_data_config
from halal_trading.data.storage import ParquetStore

from .backtest import buy_and_hold, decile_table, expected_return, performance, simulate
from .dataset import ALL_FEATURES, build_dataset, compliant_stocks, load_model_config, load_model_name
from .models import MODEL_NAMES, make_model
from .validation import run_walk_forward

ROOT = Path(__file__).resolve().parents[3]


def _universe(asset_class: str, storage_dir: Path, data_cfg) -> tuple[list[str], set[str]]:
    """(actifs utilisés pour la recherche, actifs autorisés au trading)."""
    if asset_class == "stocks":
        symbols = compliant_stocks(storage_dir)
        return symbols, set(symbols)
    policy = load_policy(ROOT / "config" / "compliance.yaml")
    approved = {f"{a.symbol}-EUR" for a in policy.crypto_whitelist}
    return list(data_cfg.crypto_products), approved


def _pct(x: float) -> str:
    return f"{x:+.1%}" if np.isfinite(x) else "n/a"


def cmd_backtest(asset_class: str, store: ParquetStore, storage_dir: Path, data_cfg) -> int:
    cfg = load_model_config(ROOT / "config" / "model.yaml", asset_class)
    symbols, _ = _universe(asset_class, storage_dir, data_cfg)
    panel = build_dataset(store, cfg, symbols)
    print(f"Jeu de données {asset_class} : {len(panel):,} lignes, {panel['symbol'].nunique()} actifs, "
          f"{panel['date'].min():%Y-%m-%d} → {panel['date'].max():%Y-%m-%d}")
    print(f"Cible : objectif ±{cfg.tp_mult}σ√{cfg.horizon_days} en {cfg.horizon_days} jours, "
          f"frais {cfg.fee_per_side:.2%} par ordre\n")

    preds, metrics = run_walk_forward(panel, cfg.first_test_year)

    print("Pouvoir de sélection hors échantillon (moyenne des années de test) :")
    summary = metrics.groupby("modele")[["ic", "rendement_moyen_top10pct", "rendement_moyen_tous"]].mean()
    summary = summary.reindex(MODEL_NAMES)
    print(summary.assign(
        ic=summary["ic"].map("{:+.3f}".format),
        rendement_moyen_top10pct=summary["rendement_moyen_top10pct"].map("{:+.2%}".format),
        rendement_moyen_tous=summary["rendement_moyen_tous"].map("{:+.2%}".format),
    ).to_string())
    print("  (IC : corrélation entre prévision et rendement réel ; 0 = hasard, 0,03–0,05 = déjà utile)\n")

    start, end = preds["date"].min(), preds["date"].max()
    bench = performance(buy_and_hold(panel, start, end))
    rows = []
    variants = [(name, None) for name in MODEL_NAMES[1:]]
    variants += [(name, -np.inf) for name in MODEL_NAMES]  # toujours investi dans le top N
    for name, threshold in variants:
        result = simulate(preds, f"pred_{name}", cfg, threshold)
        perf = performance(result.equity)
        t = result.trades
        label = f"{name} (seuil)" if threshold is None else f"{name} (top {cfg.max_positions}, investi)"
        rows.append({
            "strategie": label,
            **perf,
            "trades": len(t),
            "trades_gagnants": float((t["net_ret"] > 0).mean()) if len(t) else np.nan,
            "gain_moyen_par_trade": float(t["net_ret"].mean()) if len(t) else np.nan,
            "exposition_moyenne": float(result.exposure.mean()),
        })
    rows.append({"strategie": "acheter et garder (univers)", **bench})
    table = pd.DataFrame(rows).set_index("strategie")

    print(f"Simulation {start:%Y-%m-%d} → {end:%Y-%m-%d}, frais déduits :")
    display = table.copy()
    for col in ("rendement_total", "rendement_annuel", "perte_max", "trades_gagnants",
                "gain_moyen_par_trade", "exposition_moyenne"):
        display[col] = display[col].map(lambda v: _pct(v) if pd.notna(v) else "")
    display["sharpe"] = display["sharpe"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "")
    display["trades"] = display["trades"].map(lambda v: f"{int(v)}" if pd.notna(v) else "")
    print(display.to_string(), "\n")

    for name in MODEL_NAMES[1:]:
        print(f"{name} — rendement réel par décile de rendement prévu (1 = prévu le plus bas) :")
        dec = decile_table(preds, f"pred_{name}", cfg.fee_per_side)
        print(dec.assign(
            prevu=dec["prevu"].map("{:+.2%}".format),
            reel_net=dec["reel_net"].map("{:+.2%}".format),
            objectif_atteint=dec["objectif_atteint"].map("{:.0%}".format),
        ).T.to_string(), "\n")

    print("IC par année :")
    yearly = metrics.pivot(index="annee", columns="modele", values="ic")[MODEL_NAMES[1:]]
    print(yearly.round(3).T.to_string())

    out = storage_dir / "reports" / f"backtest_{asset_class}_{datetime.now(timezone.utc):%Y-%m-%d}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "config": cfg.__dict__,
        "summary": table.reset_index().to_dict(orient="records"),
        "metrics": metrics.to_dict(orient="records"),
    }, indent=2, default=str), encoding="utf-8")
    print(f"\nRapport : {out}")
    return 0


def cmd_predict(asset_class: str, store: ParquetStore, storage_dir: Path, data_cfg) -> int:
    cfg = load_model_config(ROOT / "config" / "model.yaml", asset_class)
    symbols, tradable = _universe(asset_class, storage_dir, data_cfg)
    panel = build_dataset(store, cfg, symbols)
    train = panel[panel["label"].notna()]
    model_name = load_model_name(ROOT / "config" / "model.yaml")
    model = make_model(model_name).fit(
        train[ALL_FEATURES].to_numpy(float), train["outcome_ret"].to_numpy(float)
    )

    # Dernière cotation de chaque actif (les places n'ont pas toutes le même calendrier),
    # en ignorant les actifs dont les données ont plus de 5 jours.
    last_day = panel["date"].max()
    today = panel.sort_values("date").groupby("symbol").tail(1)
    today = today[today["date"] >= last_day - pd.Timedelta(days=5)].copy()
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
