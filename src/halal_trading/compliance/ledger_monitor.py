"""Surveillance de l'historique du compte : détecte tout flux lié à l'intérêt.

- Intérêt / récompense / prêt de titres reçu  -> purification à 100 % + arrêt du bot
- Intérêt payé (découvert)                   -> arrêt du bot, cause à corriger
- Ligne ambiguë (mot-clé suspect)             -> arrêt du bot jusqu'à revue humaine
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal

from halal_trading.domain import Transaction, TransactionType

from .audit import AuditLog
from .purification import PurificationEntry, PurificationRegister, dividend_purification_amount

INCOME_TYPES = {
    TransactionType.INTEREST_CREDIT: "intérêt reçu",
    TransactionType.STAKING_REWARD: "récompense de staking reçue",
    TransactionType.LENDING_INCOME: "revenu de prêt de titres reçu",
    TransactionType.YIELD_REWARD: "récompense de rendement reçue",
}

SUSPICIOUS = re.compile(
    r"interest|int[ée]r[êe]t|reward|r[ée]compense|staking|stake|lending|loan|pr[êe]t|yield|"
    r"rendement|syep|borrow|emprunt|margin|marge|rebate|apy|apr",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class LedgerAlert:
    transaction_id: str
    severity: str  # "critique" ou "à_revoir"
    message: str


@dataclass
class LedgerReport:
    alerts: list[LedgerAlert] = field(default_factory=list)

    @property
    def halt_trading(self) -> bool:
        return bool(self.alerts)


class LedgerMonitor:
    def __init__(self, purification: PurificationRegister, audit: AuditLog):
        self.purification = purification
        self.audit = audit

    def scan(
        self,
        transactions: list[Transaction],
        dividend_impure_ratios: dict[str, Decimal] | None = None,
    ) -> LedgerReport:
        ratios = dividend_impure_ratios or {}
        report = LedgerReport()

        for tx in transactions:
            if tx.type in INCOME_TYPES:
                label = INCOME_TYPES[tx.type]
                self.purification.add(
                    PurificationEntry(tx.id, tx.timestamp, abs(tx.amount), tx.currency, label)
                )
                report.alerts.append(
                    LedgerAlert(tx.id, "critique", f"{label} : {abs(tx.amount)} {tx.currency}, purifié à 100 %")
                )

            elif tx.type == TransactionType.INTEREST_DEBIT:
                report.alerts.append(
                    LedgerAlert(
                        tx.id,
                        "critique",
                        f"intérêt payé : {abs(tx.amount)} {tx.currency} (découvert ou emprunt), cause à corriger",
                    )
                )

            elif tx.type == TransactionType.DIVIDEND:
                ratio = ratios.get(tx.symbol or "")
                if ratio is None:
                    report.alerts.append(
                        LedgerAlert(tx.id, "à_revoir", f"dividende de {tx.symbol} sans ratio de purification connu")
                    )
                elif ratio > 0:
                    amount = dividend_purification_amount(tx.amount, ratio)
                    self.purification.add(
                        PurificationEntry(
                            tx.id, tx.timestamp, amount, tx.currency, f"part non conforme du dividende {tx.symbol}"
                        )
                    )
                    report.alerts.append(
                        LedgerAlert(tx.id, "critique", f"{tx.symbol} a des revenus non conformes ({ratio:.2%})")
                    )

            elif SUSPICIOUS.search(tx.description or ""):
                report.alerts.append(
                    LedgerAlert(tx.id, "à_revoir", f"libellé suspect « {tx.description} » ({tx.type.value})")
                )

        self.audit.record(
            "ledger_scan",
            allowed=not report.halt_trading,
            reasons=[f"[{a.severity}] {a.transaction_id} : {a.message}" for a in report.alerts],
            transactions_scanned=len(transactions),
        )
        return report
