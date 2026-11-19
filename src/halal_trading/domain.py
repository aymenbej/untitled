"""Objets métier partagés par tous les modules (données, modèles, conformité, exécution).

Les montants sont toujours des ``Decimal`` pour éviter les erreurs d'arrondi des flottants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum


class InstrumentType(str, Enum):
    """Type d'instrument financier. Seuls COMMON_STOCK et CRYPTO_SPOT sont autorisés."""

    COMMON_STOCK = "common_stock"
    CRYPTO_SPOT = "crypto_spot"
    # Tous les types ci-dessous sont interdits par la politique de conformité.
    PREFERRED_STOCK = "preferred_stock"  # dividende fixe, assimilable à un intérêt
    BOND = "bond"
    MONEY_MARKET_FUND = "money_market_fund"
    ETF = "etf"  # contenu non filtré ligne par ligne
    CFD = "cfd"  # financement quotidien = intérêt
    FUTURE = "future"
    OPTION = "option"
    PERPETUAL = "perpetual"  # taux de financement
    LEVERAGED_PRODUCT = "leveraged_product"
    STABLECOIN = "stablecoin"  # réserves placées à intérêt
    OTHER = "other"


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


@dataclass(frozen=True)
class Instrument:
    symbol: str
    type: InstrumentType
    quote_currency: str  # devise dans laquelle l'actif est payé (EUR, USD...)
    exchange: str | None = None
    name: str | None = None


@dataclass(frozen=True)
class OrderRequest:
    """Ordre proposé par le moteur de décision, avant tout contrôle."""

    instrument: Instrument
    side: Side
    quantity: Decimal
    estimated_price: Decimal  # prix attendu (dernier prix ou limite)
    estimated_fee: Decimal  # frais estimés, dans la devise de cotation
    order_type: OrderType = OrderType.LIMIT
    limit_price: Decimal | None = None
    margin: bool = False
    leverage: Decimal = Decimal("1")


@dataclass
class AccountState:
    """État du compte au moment du contrôle d'un ordre."""

    cash: dict[str, Decimal] = field(default_factory=dict)  # devise -> solde
    reserved_cash: dict[str, Decimal] = field(default_factory=dict)  # bloqué par des ordres ouverts
    positions: dict[str, Decimal] = field(default_factory=dict)  # symbole -> quantité détenue
    reserved_positions: dict[str, Decimal] = field(default_factory=dict)  # en vente dans des ordres ouverts

    def available_cash(self, currency: str) -> Decimal:
        return self.cash.get(currency, Decimal("0")) - self.reserved_cash.get(currency, Decimal("0"))

    def available_quantity(self, symbol: str) -> Decimal:
        return self.positions.get(symbol, Decimal("0")) - self.reserved_positions.get(symbol, Decimal("0"))


class AccountType(str, Enum):
    CASH = "cash"
    MARGIN = "margin"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class AccountSettings:
    """Paramètres du compte chez le broker, lus par API ou confirmés manuellement.

    ``None`` signifie « inconnu » : la conformité le traite comme un refus.
    """

    broker: str
    account_type: AccountType
    margin_enabled: bool | None
    securities_lending_enabled: bool | None
    cash_interest_enabled: bool | None
    staking_enabled: bool | None
    yield_rewards_enabled: bool | None
    source: str  # "api" ou "manual"
    confirmed_at: datetime
    confirmed_by: str | None = None


class TransactionType(str, Enum):
    TRADE = "trade"
    FEE = "fee"
    DIVIDEND = "dividend"
    DEPOSIT = "deposit"
    WITHDRAWAL = "withdrawal"
    FX_CONVERSION = "fx_conversion"
    INTEREST_CREDIT = "interest_credit"
    INTEREST_DEBIT = "interest_debit"
    STAKING_REWARD = "staking_reward"
    LENDING_INCOME = "lending_income"
    YIELD_REWARD = "yield_reward"
    OTHER = "other"


@dataclass(frozen=True)
class Transaction:
    """Ligne de l'historique du compte, telle que renvoyée par le broker."""

    id: str
    timestamp: datetime
    type: TransactionType
    amount: Decimal
    currency: str
    description: str = ""
    symbol: str | None = None
