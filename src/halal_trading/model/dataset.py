"""Assemblage du jeu de données « panel » (une ligne par actif et par jour)."""

from __future__ import annotations

import glob
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from halal_trading.data.storage import ParquetStore

from .features import CROSS_SECTIONAL, FEATURES, add_cross_sectional, adjusted_ohlc, compute_features
from .labels import triple_barrier

ALL_FEATURES = FEATURES + CROSS_SECTIONAL


@dataclass(frozen=True)
class ModelConfig:
    asset_class: str
    horizon_days: int
    tp_mult: float
    sl_mult: float
    fee_per_side: float
    first_test_year: int
    max_positions: int
    min_expected_return: float


def load_model_config(path: str | Path, asset_class: str) -> ModelConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))[asset_class]
    return ModelConfig(asset_class=asset_class, **raw)


def load_model_name(path: str | Path) -> str:
    return str(yaml.safe_load(Path(path).read_text(encoding="utf-8")).get("model", "lineaire"))


def compliant_stocks(storage_dir: str | Path) -> list[str]:
    """Actions conformes au DERNIER filtrage halal enregistré."""
    files = sorted(glob.glob(str(Path(storage_dir) / "screening" / "*.json")))
    if not files:
        raise FileNotFoundError("aucun filtrage halal enregistré : lancer `halal-data screen`")
    rows = json.loads(Path(files[-1]).read_text(encoding="utf-8"))
    return sorted(r["symbol"] for r in rows if r["status"] == "CONFORME")


def build_dataset(store: ParquetStore, cfg: ModelConfig, symbols: list[str]) -> pd.DataFrame:
    frames = []
    for symbol in symbols:
        prices = store.read_prices(cfg.asset_class, symbol)
        if prices is None or len(prices) < 300:
            continue
        ohlc = adjusted_ohlc(prices)
        feats = compute_features(ohlc)
        labels = triple_barrier(ohlc, feats["vol_20"], cfg.horizon_days, cfg.tp_mult, cfg.sl_mult)
        frame = feats.merge(labels, on="date")
        frame["symbol"] = symbol
        frames.append(frame)
    if not frames:
        raise ValueError(f"aucune donnée de prix exploitable pour {cfg.asset_class}")
    panel = add_cross_sectional(pd.concat(frames, ignore_index=True))
    panel = panel.dropna(subset=ALL_FEATURES).sort_values(["date", "symbol"]).reset_index(drop=True)
    return panel
