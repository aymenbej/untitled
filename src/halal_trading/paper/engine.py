"""Portefeuille virtuel (« paper trading ») : la stratégie tourne chaque jour sur
les vrais prix, sans argent réel. C'est le seul test sans biais du survivant :
personne ne connaît le futur.

Règles identiques aux tests historiques : décision à la clôture du jour J, achat
à l'ouverture du premier jour de cotation suivant, sortie à l'objectif, au stop
(prioritaire si les deux sont touchés le même jour) ou après `horizon` séances.

Conformité, comme en réel :
- compte « cash » en euros ; avant d'acheter une action en devise étrangère, on
  CONVERTIT d'abord les euros au comptant (sinon le solde en devise deviendrait
  négatif, c'est-à-dire un emprunt à intérêt) ;
- chaque ordre passe par le contrôle de conformité (OrderGuard) ;
- le produit d'une vente est reconverti en euros.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from decimal import Decimal

import pandas as pd

from halal_trading.compliance import ComplianceViolation, OrderGuard
from halal_trading.domain import AccountState, Instrument, InstrumentType, OrderRequest, OrderType, Side


@dataclass
class FeeModel:
    min_fee: float = 1.0  # € par ordre (ordre de grandeur d'un courtier à bas coût)
    rate: float = 0.0005  # 0,05 % de commission
    slippage: float = 0.0005  # 0,05 % d'écart d'exécution
    fx_rate: float = 0.0002  # 0,02 % sur une conversion de devise

    def order_cost(self, amount_eur: float) -> float:
        return max(self.min_fee, self.rate * amount_eur) + self.slippage * amount_eur


@dataclass
class PaperConfig:
    capital_eur: float = 500.0
    max_positions: int = 2
    horizon_days: int = 20
    tp_mult: float = 3.0
    sl_mult: float = 1.5
    fees: FeeModel = field(default_factory=FeeModel)


@dataclass
class Position:
    symbol: str
    currency: str
    quantity: float
    entry_date: str
    entry_price: float  # devise locale
    tp_price: float
    sl_price: float
    bars_held: int = 0
    cost_eur: float = 0.0  # montant investi, frais compris


@dataclass
class PendingOrder:
    symbol: str
    currency: str
    budget_eur: float
    decided_on: str
    tp_ret: float
    sl_ret: float
    score: float


@dataclass
class PaperState:
    start_date: str
    cash_eur: float
    last_date: str | None = None
    positions: list[Position] = field(default_factory=list)
    pending: list[PendingOrder] = field(default_factory=list)
    trades: list[dict] = field(default_factory=list)
    journal: list[dict] = field(default_factory=list)
    benchmark_start: dict[str, float] = field(default_factory=dict)  # prix d'origine en €
    events: list[dict] = field(default_factory=list)  # refus de conformité, etc.

    @staticmethod
    def new(start_date: str, capital_eur: float) -> "PaperState":
        return PaperState(start_date=start_date, cash_eur=capital_eur)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(raw: dict) -> "PaperState":
        return PaperState(
            start_date=raw["start_date"], cash_eur=raw["cash_eur"], last_date=raw.get("last_date"),
            positions=[Position(**p) for p in raw.get("positions", [])],
            pending=[PendingOrder(**p) for p in raw.get("pending", [])],
            trades=raw.get("trades", []), journal=raw.get("journal", []),
            benchmark_start=raw.get("benchmark_start", {}), events=raw.get("events", []),
        )


def _eur(amount_local: float, currency: str, fx: dict[str, float]) -> float:
    """fx[cur] = unités de devise pour 1 € (ex. 1,08 USD). L'euro vaut 1."""
    return amount_local if currency == "EUR" else amount_local / fx[currency]


def _local(amount_eur: float, currency: str, fx: dict[str, float]) -> float:
    return amount_eur if currency == "EUR" else amount_eur * fx[currency]


class PaperEngine:
    def __init__(self, cfg: PaperConfig, guard: OrderGuard):
        self.cfg = cfg
        self.guard = guard

    # ------------------------------------------------------------------ une séance
    def process_day(self, state: PaperState, day: str, bars: dict[str, dict], fx: dict[str, float],
                    ranking: list[tuple[str, str, float, float]] | None) -> None:
        """Traite une séance.

        bars : symbole -> {"open", "high", "low", "close"} du jour (devise locale).
        ranking : à la clôture, candidats classés du meilleur au moins bon :
                  (symbole, devise, volatilité journalière, score). None : pas de décision.
        """
        self._fill_pending(state, day, bars, fx)
        self._check_exits(state, day, bars, fx)
        if ranking is not None:
            self._decide(state, day, fx, ranking, bars)
        self._journal(state, day, bars, fx)
        state.last_date = day

    def _account(self, state: PaperState, currency: str, local_cash: float) -> AccountState:
        held = {p.symbol: Decimal(str(p.quantity)) for p in state.positions}
        return AccountState(cash={currency: Decimal(str(local_cash))}, positions=held)

    def _fill_pending(self, state: PaperState, day: str, bars: dict[str, dict], fx: dict[str, float]) -> None:
        remaining = []
        for order in state.pending:
            bar = bars.get(order.symbol)
            if bar is None or not math.isfinite(bar.get("open", math.nan)) or bar["open"] <= 0:
                remaining.append(order)  # pas de cotation aujourd'hui : on réessaie demain
                continue
            fees = self.cfg.fees
            budget = min(order.budget_eur, state.cash_eur)
            fee_eur = fees.order_cost(budget)
            fx_fee = 0.0 if order.currency == "EUR" else fees.fx_rate * budget
            invest_eur = budget - fee_eur - fx_fee
            if invest_eur <= 0:
                state.events.append({"date": day, "type": "ordre annulé", "symbol": order.symbol,
                                     "raison": "budget insuffisant pour couvrir les frais"})
                continue
            local_cash = _local(invest_eur + fee_eur, order.currency, fx)  # conversion au comptant AVANT l'achat
            price = bar["open"]
            qty = _local(invest_eur, order.currency, fx) / price
            request = OrderRequest(
                instrument=Instrument(order.symbol, InstrumentType.COMMON_STOCK, order.currency),
                side=Side.BUY, quantity=Decimal(str(round(qty, 6))), estimated_price=Decimal(str(price)),
                estimated_fee=Decimal(str(round(_local(fee_eur, order.currency, fx), 6))),
                order_type=OrderType.LIMIT, limit_price=Decimal(str(price)),
            )
            try:
                self.guard.check(request, self._account(state, order.currency, local_cash))
            except ComplianceViolation as exc:
                state.events.append({"date": day, "type": "ordre bloqué (conformité)", "symbol": order.symbol,
                                     "raison": " ; ".join(exc.reasons)})
                continue
            state.cash_eur -= budget
            state.positions.append(Position(
                symbol=order.symbol, currency=order.currency, quantity=qty, entry_date=day, entry_price=price,
                tp_price=price * (1 + order.tp_ret), sl_price=price * (1 - order.sl_ret), cost_eur=budget,
            ))
        state.pending = remaining

    def _check_exits(self, state: PaperState, day: str, bars: dict[str, dict], fx: dict[str, float]) -> None:
        kept = []
        for pos in state.positions:
            bar = bars.get(pos.symbol)
            if bar is None:
                kept.append(pos)
                continue
            pos.bars_held += 1
            first_day = pos.entry_date == day
            exit_price, reason = None, None
            if not first_day and bar["open"] <= pos.sl_price:
                exit_price, reason = bar["open"], "stop (ouverture sous le stop)"
            elif bar["low"] <= pos.sl_price:
                exit_price, reason = pos.sl_price, "stop"
            elif not first_day and bar["open"] >= pos.tp_price:
                exit_price, reason = bar["open"], "objectif (ouverture au-dessus)"
            elif bar["high"] >= pos.tp_price:
                exit_price, reason = pos.tp_price, "objectif"
            elif pos.bars_held >= self.cfg.horizon_days:
                exit_price, reason = bar["close"], "échéance"
            if exit_price is None:
                kept.append(pos)
                continue
            gross_eur = _eur(pos.quantity * exit_price, pos.currency, fx)
            fees = self.cfg.fees
            fee = fees.order_cost(gross_eur) + (0.0 if pos.currency == "EUR" else fees.fx_rate * gross_eur)
            proceeds = gross_eur - fee  # reconverti en euros
            state.cash_eur += proceeds
            state.trades.append({
                "symbol": pos.symbol, "entree": pos.entry_date, "sortie": day, "motif": reason,
                "prix_entree": pos.entry_price, "prix_sortie": exit_price, "investi_eur": round(pos.cost_eur, 2),
                "recupere_eur": round(proceeds, 2), "resultat_eur": round(proceeds - pos.cost_eur, 2),
                "rendement": round(proceeds / pos.cost_eur - 1, 5),
            })
        state.positions = kept

    def _decide(self, state: PaperState, day: str, fx: dict[str, float],
                ranking: list[tuple[str, str, float, float]], bars: dict[str, dict]) -> None:
        busy = {p.symbol for p in state.positions} | {o.symbol for o in state.pending}
        slots = self.cfg.max_positions - len(state.positions) - len(state.pending)
        if slots <= 0:
            return
        equity = self.equity(state, bars, fx)
        budget = equity / self.cfg.max_positions
        reserved = sum(o.budget_eur for o in state.pending)
        scale = math.sqrt(self.cfg.horizon_days)
        for symbol, currency, vol, score in ranking:
            if slots <= 0:
                break
            if symbol in busy or not (vol > 0):
                continue
            if state.cash_eur - reserved < budget * 0.98:
                break
            state.pending.append(PendingOrder(symbol, currency, budget, day, self.cfg.tp_mult * vol * scale,
                                              self.cfg.sl_mult * vol * scale, score))
            reserved += budget
            busy.add(symbol)
            slots -= 1

    def equity(self, state: PaperState, bars: dict[str, dict], fx: dict[str, float]) -> float:
        value = state.cash_eur
        for pos in state.positions:
            bar = bars.get(pos.symbol)
            price = bar["close"] if bar else pos.entry_price
            value += _eur(pos.quantity * price, pos.currency, fx)
        return value

    def _journal(self, state: PaperState, day: str, bars: dict[str, dict], fx: dict[str, float]) -> None:
        # référence : panier halal à poids égaux acheté le premier jour (en euros)
        if not state.benchmark_start:
            state.benchmark_start = {s: _eur(b["close"], b.get("currency", "EUR"), fx)
                                     for s, b in bars.items() if b.get("close", 0) > 0}
        ratios = [(_eur(bars[s]["close"], bars[s].get("currency", "EUR"), fx) / p0)
                  for s, p0 in state.benchmark_start.items() if s in bars and bars[s].get("close", 0) > 0]
        bench = sum(ratios) / len(ratios) if ratios else float("nan")
        state.journal.append({
            "date": day, "valeur_eur": round(self.equity(state, bars, fx), 2), "liquidites_eur": round(state.cash_eur, 2),
            "positions": len(state.positions), "ordres_en_attente": len(state.pending),
            "reference_eur": round(bench * self.cfg.capital_eur, 2) if math.isfinite(bench) else None,
        })


def momentum_ranking(panel: pd.DataFrame, day: pd.Timestamp, currencies: dict[str, str]) -> list[tuple]:
    """Classement momentum 120 jours des actions halal à la clôture de `day`."""
    today = panel[panel["date"] == day].dropna(subset=["ret_120", "vol_20"])
    today = today.sort_values("ret_120", ascending=False)
    return [(r.symbol, currencies.get(r.symbol, "USD"), float(r.vol_20), float(r.ret_120)) for r in today.itertuples()]
