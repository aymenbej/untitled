"""Contrôle de chaque ordre AVANT envoi au broker.

Aucun module (modèle ML, moteur de décision, agent IA) ne peut envoyer un ordre
sans passer par ``OrderGuard.check``. Le connecteur broker l'appelle lui-même.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from halal_trading.domain import AccountState, OrderRequest, OrderType, Side

from .audit import AuditLog
from .errors import ComplianceViolation
from .policy import STABLECOIN_CODES, CompliancePolicy
from .universe import ApprovedUniverse

ZERO = Decimal("0")


class OrderGuard:
    def __init__(
        self,
        policy: CompliancePolicy,
        universe: ApprovedUniverse,
        audit: AuditLog,
        price_buffer: Decimal = Decimal("0.02"),
    ):
        # price_buffer : marge de sécurité sur le prix d'un achat (glissement),
        # pour que le solde ne puisse jamais devenir négatif (= intérêts débiteurs).
        self.policy = policy
        self.universe = universe
        self.audit = audit
        self.price_buffer = price_buffer

    def violations(self, order: OrderRequest, account: AccountState, now: datetime | None = None) -> list[str]:
        reasons: list[str] = []
        inst = order.instrument

        # Type d'instrument et actif
        if inst.type not in self.policy.allowed_instrument_types:
            reasons.append(f"type d'instrument interdit : {inst.type.value}")
        reasons.extend(self.universe.check(inst.symbol, now))

        # Devise : fiat uniquement
        quote = inst.quote_currency.upper()
        if quote in STABLECOIN_CODES:
            reasons.append(f"cotation en stablecoin interdite : {quote}")
        elif quote not in self.policy.allowed_quote_currencies:
            reasons.append(f"devise de cotation non autorisée : {quote}")

        # Aucun emprunt, aucun levier
        if order.margin:
            reasons.append("ordre sur marge interdit (emprunt à intérêt)")
        if order.leverage != 1:
            reasons.append(f"levier interdit : x{order.leverage}")

        # Cohérence de l'ordre
        invalid = []
        if order.quantity <= ZERO:
            invalid.append("quantité nulle ou négative")
        if order.estimated_price <= ZERO:
            invalid.append("prix estimé nul ou négatif")
        if order.estimated_fee < ZERO:
            invalid.append("frais estimés négatifs")
        if order.order_type == OrderType.LIMIT and (order.limit_price is None or order.limit_price <= ZERO):
            invalid.append("ordre limite sans prix limite valide")
        if invalid:
            return reasons + invalid

        price = order.limit_price if order.order_type == OrderType.LIMIT else order.estimated_price

        if order.side == Side.BUY:
            # Achat comptant intégral : le solde ne doit jamais passer sous zéro.
            buffer = ZERO if order.order_type == OrderType.LIMIT else self.price_buffer
            cost = order.quantity * price * (1 + buffer) + order.estimated_fee
            available = account.available_cash(quote)
            if cost > available:
                reasons.append(
                    f"liquidités insuffisantes en {quote} : besoin {cost:.2f}, disponible {available:.2f} "
                    "(un découvert génèrerait des intérêts)"
                )
        else:
            # Pas de vente à découvert : on ne vend que ce qu'on détient.
            held = account.available_quantity(inst.symbol)
            if order.quantity > held:
                reasons.append(
                    f"vente à découvert interdite : vente de {order.quantity} {inst.symbol}, détenu {held}"
                )
            proceeds = order.quantity * price
            if order.estimated_fee > proceeds + account.available_cash(quote):
                reasons.append("frais supérieurs au produit de la vente et aux liquidités : solde négatif")

        return reasons

    def check(self, order: OrderRequest, account: AccountState, now: datetime | None = None) -> None:
        reasons = self.violations(order, account, now)
        details = {
            "symbol": order.instrument.symbol,
            "side": order.side.value,
            "quantity": str(order.quantity),
            "price": str(order.limit_price or order.estimated_price),
        }
        self.audit.record("order_check", allowed=not reasons, reasons=reasons, **details)
        if reasons:
            raise ComplianceViolation(f"Ordre {order.side.value} {order.instrument.symbol}", reasons)
