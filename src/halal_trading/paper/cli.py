"""Commande quotidienne du portefeuille virtuel.

    halal-paper univers   # fige la liste des actions halal (à refaire chaque trimestre)
    halal-paper run       # traite les séances depuis le dernier passage
    halal-paper bilan     # résumé : valeur, trades, comparaison à la référence

L'état est versionné dans le dépôt (dossier paper/), pour qu'une tâche planifiée
puisse le reprendre chaque jour depuis une session neuve.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from halal_trading.compliance import ApprovedUniverse, AuditLog, OrderGuard, load_policy
from halal_trading.compliance.screening import ScreeningResult
from halal_trading.domain import InstrumentType
from halal_trading.model.features import adjusted_ohlc, compute_features

from .engine import PaperConfig, PaperEngine, PaperState

ROOT = Path(__file__).resolve().parents[3]
PAPER = ROOT / "paper"
STATE = PAPER / "etat.json"
UNIVERSE = PAPER / "univers_halal.json"
FX_TICKERS = {"USD": "EURUSD=X", "GBP": "EURGBP=X", "CHF": "EURCHF=X", "DKK": "EURDKK=X", "SEK": "EURSEK=X",
              "NOK": "EURNOK=X"}


def cmd_univers() -> int:
    screening = sorted((ROOT / "data" / "screening").glob("*.json"))
    discovered = json.loads((ROOT / "data" / "universe" / "discovered.json").read_text(encoding="utf-8"))
    info = {c["symbol"]: c for c in discovered}
    rows = json.loads(screening[-1].read_text(encoding="utf-8"))
    out = [{"symbol": r["symbol"], "currency": info[r["symbol"]]["currency"],
            "pence": info[r["symbol"]]["exchange"] == "LSE",  # Londres cote en pence
            "screened_at": r["screened_at"], "purification": r.get("dividend_purification_ratio")}
           for r in rows if r["status"] == "CONFORME" and r["symbol"] in info]
    PAPER.mkdir(exist_ok=True)
    UNIVERSE.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(out)} actions halal figées dans {UNIVERSE.relative_to(ROOT)}")
    return 0


def _history(symbol: str, start: str) -> pd.DataFrame | None:
    import yfinance as yf

    hist = yf.Ticker(symbol).history(start=start, auto_adjust=False, actions=True)
    if hist is None or hist.empty:
        return None
    hist = hist.reset_index()
    dates = pd.to_datetime(hist["Date"])
    hist["date"] = dates.dt.tz_localize(None).dt.normalize() if dates.dt.tz is not None else dates.dt.normalize()
    return hist.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close",
                                "Adj Close": "adj_close", "Volume": "volume"})[
        ["date", "open", "high", "low", "close", "adj_close", "volume"]]


def _guard(universe: list[dict]) -> OrderGuard:
    policy = load_policy(ROOT / "config" / "compliance.yaml")
    approved = ApprovedUniverse(max_age_days=policy.max_screening_age_days)
    for u in universe:
        approved.add(ScreeningResult(u["symbol"], True, (), datetime.fromisoformat(u["screened_at"]),
                                     InstrumentType.COMMON_STOCK))
    return OrderGuard(policy, approved, AuditLog(PAPER / "audit_conformite.jsonl"))


def cmd_run(cfg: PaperConfig) -> int:
    universe = json.loads(UNIVERSE.read_text(encoding="utf-8"))
    state = (PaperState.from_dict(json.loads(STATE.read_text(encoding="utf-8"))) if STATE.exists()
             else PaperState.new(datetime.now(timezone.utc).date().isoformat(), cfg.capital_eur))
    since = (pd.Timestamp(state.last_date or datetime.now(timezone.utc).date()) - timedelta(days=400)).date().isoformat()

    with ThreadPoolExecutor(max_workers=4) as pool:
        prices = dict(zip([u["symbol"] for u in universe],
                          pool.map(lambda u: _history(u["symbol"], since), universe)))
        fx_hist = dict(zip(FX_TICKERS, pool.map(lambda t: _history(t, since), FX_TICKERS.values())))
    fx_close = pd.DataFrame({cur: h.set_index("date")["close"] for cur, h in fx_hist.items() if h is not None}).ffill()

    meta = {u["symbol"]: u for u in universe}
    frames, features = {}, []
    for symbol, df in prices.items():
        if df is None or len(df) < 130:
            continue
        if meta[symbol]["pence"]:
            df[["open", "high", "low", "close", "adj_close"]] /= 100.0
        frames[symbol] = df.set_index("date")
        f = compute_features(adjusted_ohlc(df.assign(date=pd.to_datetime(df["date"]))))
        f["symbol"] = symbol
        features.append(f[["date", "symbol", "ret_120", "vol_20"]])
    panel = pd.concat(features, ignore_index=True)

    # Seules les séances TERMINÉES comptent : avant 22 h UTC (clôture de New York passée),
    # la séance du jour est encore en cours et ses prix ne sont pas définitifs.
    now = datetime.now(timezone.utc)
    cutoff = pd.Timestamp(now.date()) if now.hour >= 22 else pd.Timestamp(now.date()) - timedelta(days=1)
    all_days = sorted(d for d in set().union(*[set(f.index) for f in frames.values()]) if d <= cutoff)
    last = pd.Timestamp(state.last_date) if state.last_date else None
    todo = [d for d in all_days if last is None or d > last]
    if last is None:
        todo = todo[-1:]  # premier passage : on décide à la dernière clôture, sans rejouer le passé
    engine = PaperEngine(cfg, _guard(universe))
    for day in todo:
        bars = {s: {**f.loc[day, ["open", "high", "low", "close"]].astype(float).to_dict(),
                    "currency": meta[s]["currency"]}
                for s, f in frames.items() if day in f.index}
        fx_row = fx_close[fx_close.index <= day].iloc[-1].to_dict() if len(fx_close) else {}
        today = panel[panel["date"] == day].dropna().sort_values("ret_120", ascending=False)
        ranking = [(r.symbol, meta[r.symbol]["currency"], float(r.vol_20), float(r.ret_120))
                   for r in today.itertuples()]
        engine.process_day(state, day.date().isoformat(), bars, fx_row, ranking)

    STATE.write_text(json.dumps(state.to_dict(), indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    _write_csv(PAPER / "journal.csv", state.journal)
    _write_csv(PAPER / "trades.csv", state.trades)
    print(f"{len(todo)} séance(s) traitée(s), jusqu'au {state.last_date}")
    return cmd_bilan()


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def cmd_bilan() -> int:
    if not STATE.exists():
        print("Aucun portefeuille virtuel : lancer `halal-paper run`.")
        return 1
    st = PaperState.from_dict(json.loads(STATE.read_text(encoding="utf-8")))
    last = st.journal[-1] if st.journal else {}
    first = st.journal[0] if st.journal else {}
    print(f"Démarré le {st.start_date}, dernière séance {st.last_date}")
    if last:
        print(f"Valeur : {last['valeur_eur']:.2f} € (départ {first['valeur_eur']:.2f} €) | "
              f"référence achat-conservation halal : {last['reference_eur']} €")
    print(f"Positions : {[p.symbol for p in st.positions]} | ordres en attente : {[o.symbol for o in st.pending]}")
    if st.trades:
        res = [t["rendement"] for t in st.trades]
        print(f"{len(res)} trades clôturés, rendement moyen {np.mean(res):+.2%}, gagnants {np.mean([r > 0 for r in res]):.0%}")
    for e in st.events[-5:]:
        print(f"  ! {e['date']} {e['type']} {e.get('symbol', '')} : {e.get('raison', '')}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="halal-paper", description="Portefeuille virtuel quotidien")
    parser.add_argument("command", choices=["univers", "run", "bilan"])
    args = parser.parse_args(argv)
    if args.command == "univers":
        return cmd_univers()
    if args.command == "run":
        return cmd_run(PaperConfig())
    return cmd_bilan()


if __name__ == "__main__":
    sys.exit(main())
