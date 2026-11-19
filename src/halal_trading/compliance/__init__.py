"""Module de conformité halal, profil tolérance zéro sur l'intérêt.

Ce module est le seul point de passage obligatoire avant tout ordre. Il ne peut
être contourné ni par le modèle de prévision, ni par le moteur de décision, ni par
un agent IA : les règles sont dans le code, pas dans un prompt.
"""

from .account_guard import account_violations, check_account
from .audit import AuditLog
from .errors import ComplianceError, ComplianceViolation
from .ledger_monitor import LedgerAlert, LedgerMonitor, LedgerReport
from .order_guard import OrderGuard
from .policy import CompliancePolicy, CryptoApproval, load_policy
from .purification import PurificationEntry, PurificationRegister, dividend_purification_amount
from .screening import (
    CompanyFinancials,
    CompanyProfile,
    CryptoAsset,
    ScreeningResult,
    screen_company,
    screen_crypto,
)
from .universe import ApprovedUniverse

__all__ = [
    "ApprovedUniverse",
    "AuditLog",
    "CompanyFinancials",
    "CompanyProfile",
    "ComplianceError",
    "CompliancePolicy",
    "ComplianceViolation",
    "CryptoApproval",
    "CryptoAsset",
    "LedgerAlert",
    "LedgerMonitor",
    "LedgerReport",
    "OrderGuard",
    "PurificationEntry",
    "PurificationRegister",
    "ScreeningResult",
    "account_violations",
    "check_account",
    "dividend_purification_amount",
    "load_policy",
    "screen_company",
    "screen_crypto",
]
