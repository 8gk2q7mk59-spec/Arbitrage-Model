#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Arbitrage / Mean-Reversion Backtester (v6.0-stable, full-grid adaptive)

Basis: v5_89 (stabil) + integrierte Features aus v6_2 + Full Grid Scan:
- yfinance Caching (--cache)
- Risk/Reward & Kelly Ratio aus realisierten Trades
- Neue Defaults: MAX_HOLD_DAYS=25, MIN_CORR=0.30, Z_TRAIL_STEP=0.2, Z_TRAIL_START=-1.0
- PDF-Optionen: --pdf-per-strategy, --save-composite-pdf
- Meta JSON: --save-meta
- Parallelisierung: --parallel N (ProcessPoolExecutor)
- Full Grid Scan / Adaptive: --adaptive (+ Ranges steuerbar), Heatmap optional

Run-Beispiele:
    # Standard (ein fester Parametersatz)
    python3 arbitrage_model_v6_0_stable.py --assets etf --cache --parallel 4 --top 10 --pdf-per-strategy --save-composite-pdf --save-meta

    # Full Grid Scan (macht mehrere Parameter-Kombinationen durch und filtert danach)
    python3 arbitrage_model_v6_0_stable.py --adaptive --cache --parallel 4 \
        --z-entry-range 1.0,2.4,0.1 --min-corr-range 0.10,0.90,0.05 --lookbacks 20,30,60 \
        --top 20 --pdf-per-strategy --save-composite-pdf --save-meta --heatmap
"""

from __future__ import annotations
import json


def convert_keys_to_str(obj):
    """Recursively convert mapping keys to strings; handles pandas/numpy datetimes."""
    try:
        import numpy as np
        import pandas as pd
    except Exception:
        np = None
        pd = None
    if isinstance(obj, dict):
        return {str(k): convert_keys_to_str(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_keys_to_str(v) for v in obj]
    if pd is not None and isinstance(obj, (pd.Timestamp,)):
        return str(obj)
    if np is not None and isinstance(obj, (np.datetime64,)):
        return str(obj)
    return obj
# ----------------------------
# Imports & Setup
# ----------------------------
import os
# removed inner import json
import pickle
import argparse
import itertools
import math
import re
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:
    yf = None  # wir versuchen ggf. aus Cache zu laden

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    from sklearn.linear_model import LinearRegression
except Exception:
    LinearRegression = None

try:
    from statsmodels.tsa.stattools import adfuller, coint
except Exception:
    adfuller = None
    coint = None

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        Image as RLImage,
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib import colors
except Exception:
    letter = None  # PDF wird dann übersprungen

from concurrent.futures import ProcessPoolExecutor, as_completed

# ----------------------------
# Config / Defaults
# ----------------------------
DEFAULT_ETFS = [
    "SPY", "QQQ", "IWM", "DIA",
    "EFA", "EEM", "XLF", "XLK",
    "XLE", "XLV", "XLY", "XLP",
    "VNQ", "TLT", "GLD", "SLV",
]

BASE_PROJECT = Path.cwd()
OUT_ROOT = BASE_PROJECT / "backtests"
BENCHMARK = "SPY"

# Strategie-Defaults
Z_ENTRY_DEFAULT = 2.0
Z_EXIT_DEFAULT = 0.5
MIN_LOOKBACK_DEFAULT = 60
MAX_HOLD_DAYS = 25
Z_TRAIL_FACTOR_DEFAULT = 0.5
Z_TRAIL_STEP_DEFAULT = 0.2
Z_TRAIL_START_DEFAULT = -1.0  # <0 ⇒ keine Trailing-Phase vor Entry
MIN_CORR_DEFAULT = 0.30

ADAPTIVE_TARGET_TRADES_DEFAULT = 150  # Ziel bei adaptivem Modus (informativ)

if plt is not None:
    try:
        plt.rcParams.update({"figure.max_open_warning": 0})
    except Exception:
        pass

# ----------------------------
# Utils
# ----------------------------
def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def next_run_dir(root: Path, prefix: str = "run") -> Path:
    """Erstellt Ordner im Stil backtests/run_### oder backtests/run_fullscan_###."""
    safe_mkdir(root)
    pat = re.compile(rf"{re.escape(prefix)}_\d{{3}}$")
    existing = [p for p in root.iterdir() if p.is_dir() and pat.match(p.name)]
    if not existing:
        rd = root / f"{prefix}_001"
        safe_mkdir(rd)
        return rd
    nums = sorted(int(p.name.split("_")[-1]) for p in existing)
    rd = root / f"{prefix}_{nums[-1] + 1:03d}"
    safe_mkdir(rd)
    return rd

def trading_days_between(start: pd.Timestamp, end: pd.Timestamp) -> int:
    bdays = pd.bdate_range(start=start, end=end)
    return max(1, len(bdays))

def frange(start: float, stop: float, step: float) -> List[float]:
    """float-Range inkl. Toleranz auf Rundungsfehler, stop exklusiv (wie np.arange)."""
    vals = []
    cur = float(start)
    # kleine epsilon, um float drift zu vermeiden
    eps = abs(step) * 1e-9 + 1e-12
    while (step > 0 and cur <= stop - eps) or (step < 0 and cur >= stop + eps):
        vals.append(round(cur, 10))
        cur += step
    return vals

def parse_range_str(s: str, default_triplet: Tuple[float, float, float]) -> Tuple[float, float, float]:
    try:
        parts = [float(x.strip()) for x in s.split(",")]
        if len(parts) == 3:
            return parts[0], parts[1], parts[2]
    except Exception:
        pass
    return default_triplet

def parse_int_list(s: str, default_list: List[int]) -> List[int]:
    try:
        items = [int(x.strip()) for x in s.split(",") if x.strip()]
        if items:
            return items
    except Exception:
        pass
    return default_list

# ----------------------------
# Data: yfinance + Cache
# ----------------------------
CACHE_ROOT = BASE_PROJECT / ".cache" / "yf"
safe_mkdir(CACHE_ROOT)

def _cache_key(symbol: str, start: Optional[str], end: Optional[str], interval: str) -> Path:
    s = start or "None"
    e = end or "None"
    key = f"{symbol}_{s}_{e}_{interval}.pkl".replace(":", "-")
    return CACHE_ROOT / key
def get_data_cached_yf(
    symbol: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    interval: str = "1d",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Lädt OHLCV via yfinance mit optionalem lokalen Cache.
    Falls keine Start-/End-Daten angegeben sind, wird automatisch
    die Historie ab 2005 bis heute geladen.
    Stabilisiert Spaltennamen:
      - Falls MultiIndex (z.B. ('Close','SPY')), reduziere auf 'Close'
      - Danach Title-Case ('Close', 'Open', ...)
    """
    import yfinance as yf
    import pickle
    import pandas as pd
    from datetime import datetime

    # 🔧 Standard-Zeitraum erzwingen, falls kein Start/End angegeben
    if start is None:
        start = "2005-01-01"
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    cache_path = _cache_key(symbol, start, end, interval)

    # 🧠 Cache verwenden, falls verfügbar
    if use_cache and cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                df = pickle.load(f)
            if isinstance(df, pd.DataFrame) and isinstance(df.index, pd.DatetimeIndex) and not df.empty:
                return df
        except Exception:
            pass  # Fallback: Neu laden

    # 📥 Neue Daten laden
    try:
        df = yf.download(symbol, start=start, end=end, interval=interval, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].title() for c in df.columns]
        else:
            df.columns = [c.title() for c in df.columns]
        df = df.dropna(how="all")

        # 🔒 Cache speichern
        if use_cache:
            with open(cache_path, "wb") as f:
                pickle.dump(df, f)

        return df

    except Exception as e:
        print(f"⚠️ Fehler beim Laden von {symbol}: {e}")
        return pd.DataFrame()

    df = yf.download(
        symbol,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )

    if not isinstance(df, pd.DataFrame) or df.empty:
        raise RuntimeError(f"Keine Daten für {symbol} erhalten.")

    # ---------- Spalten normalisieren (fix für MultiIndex/tuple columns) ----------
    if isinstance(df.columns, pd.MultiIndex):
        # nimm die erste Ebene (z.B. 'Close' aus ('Close','SPY'))
        df.columns = [str(t[0]).title() for t in df.columns.to_list()]
    else:
        df.columns = [str(c).title() for c in df.columns.to_list()]
    # ------------------------------------------------------------------------------

    if use_cache:
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(df, f)
        except Exception:
            pass

    return df

def download_price_data(
    symbols: Iterable[str], start=None, end=None, interval="1d", cache=True
) -> Dict[str, pd.Series]:
    if isinstance(symbols, str):
        symbols = [symbols]
    out: Dict[str, pd.Series] = {}
    for s in symbols:
        df = get_data_cached_yf(s, start=start, end=end, interval=interval, use_cache=cache)
        px = df["Close"].copy()
        px.name = s
        out[s] = px
    return out

# ----------------------------
# Pair-Bausteine & Stats
# ----------------------------
def make_pairs(symbols: List[str]) -> List[Tuple[str, str]]:
    return list(itertools.combinations(sorted(set(symbols)), 2))


def rolling_corr(a: pd.Series, b: pd.Series, lookback: int) -> pd.Series:
    return a.rolling(lookback).corr(b)


def _beta_lr_fallback(y: pd.Series, x: pd.Series) -> float:
    """Falls sklearn fehlt: Beta = Cov(y,x)/Var(x)."""
    x_ = x.dropna()
    y_ = y.reindex_like(x_).dropna()
    aligned = pd.concat([y_, x_.reindex_like(y_)], axis=1).dropna()
    if len(aligned) < 5:
        return 1.0
    cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1], ddof=0)[0, 1]
    var = np.var(aligned.iloc[:, 1], ddof=0)
    return float(cov / var) if var > 0 else 1.0


def estimate_half_life(spread: pd.Series) -> float:
    """
    Schätzt die Half-Life der Mean-Reversion über eine einfache OU-Approximation.
    ΔS_t ~ k * S_{t-1} + ε ⇒ half_life = -ln(2) / k (falls k < 0)
    """
    if spread is None:
        return float("nan")
    s = spread.dropna()
    if len(s) < 20:
        return float("nan")

    y_lag = s.shift(1).dropna()
    delta = s.diff().dropna()
    y_lag = y_lag.loc[delta.index]
    if len(y_lag) < 5:
        return float("nan")

    x = y_lag.values
    y = delta.values
    denom = np.sum(x * x)
    if denom <= 0:
        return float("nan")
    b = float(np.sum(x * y) / denom)
    if b >= 0:
        return float("nan")
    half_life = -math.log(2) / b
    return float(half_life)


def compute_cointegration_pvalue(y: pd.Series, x: pd.Series) -> float:
    """
    Engle-Granger Cointegration p-value (falls statsmodels verfügbar).
    Kleinere Werte => stärkere Cointegration.
    """
    if coint is None:
        return float("nan")
    try:
        _, pvalue, _ = coint(y.dropna(), x.dropna())
        return float(pvalue)
    except Exception:
        return float("nan")


def compute_time_metrics_oos(eq: pd.Series, split: float = 0.7) -> Dict[str, float]:
    """
    Berechnet Time-Metriken nur auf dem Out-of-Sample-Teil (letzte (1-split) des Verlaufs).
    """
    if eq is None or eq.empty:
        return {"cagr": 0.0, "sharpe": 0.0, "max_dd": 0.0, "ret": 0.0}
    n = len(eq)
    if n < 10:
        return {"cagr": 0.0, "sharpe": 0.0, "max_dd": 0.0, "ret": 0.0}
    start_idx = int(n * split)
    eq_oos = eq.iloc[start_idx:].copy()
    if eq_oos.empty:
        return {"cagr": 0.0, "sharpe": 0.0, "max_dd": 0.0, "ret": 0.0}
    return compute_time_metrics(eq_oos)


def build_portfolio_equity(
    all_results: List[Dict],
    start_equity: float = 100_000.0,
    risk_per_trade: float = 0.005,
) -> pd.Series:
    """
    Aggregiert alle Trades aller Strategien zu einer Portfolio-Equity-Kurve.
    Nutzt die gleiche Risikosizing-Logik wie equity_from_trades.
    """
    pnl_by_date: Dict[pd.Timestamp, float] = {}
    for res in all_results:
        trades = res.get("trades", [])
        for t in trades:
            exit_date = pd.to_datetime(t.get("exit_date"))
            ret = float(t.get("return_pct", 0.0))
            pnl_by_date.setdefault(exit_date, 0.0)
            pnl_by_date[exit_date] += ret

    if not pnl_by_date:
        return pd.Series(dtype=float)

    dates = sorted(pnl_by_date)
    eq = pd.Series(index=pd.DatetimeIndex(dates), dtype=float)
    equity = float(start_equity)
    for d in dates:
        position_size = equity * float(risk_per_trade)
        equity += position_size * float(pnl_by_date[d])
        eq.loc[d] = equity
    eq.name = "Portfolio"
    return eq


def compute_spread_zscore(y: pd.Series, x: pd.Series, lookback: int = MIN_LOOKBACK_DEFAULT) -> pd.DataFrame:
    """
    Schätzt Hedge-Ratio (y ~ beta * x) und Z-Score des Spreads.
    Beta wird auf einem Fenster von ca. 2*lookback geschätzt, um Regimewechsel besser abzubilden.
    Return: DataFrame mit Spalten: [spread, z, beta]
    """
    df = pd.concat([y, x], axis=1).dropna()
    df.columns = ["y", "x"]
    if len(df) < lookback + 5:
        raise RuntimeError("Zu wenig überlappende Daten für Regression/Rolling-Stats.")

    win = max(lookback * 2, lookback + 5)
    df_reg = df.tail(win)

    if LinearRegression is not None:
        lr = LinearRegression(fit_intercept=False)
        lr.fit(df_reg[["x"]].values, df_reg["y"].values)
        beta = float(lr.coef_[0])
    else:
        beta = _beta_lr_fallback(df_reg["y"], df_reg["x"])

    spread = df["y"] - beta * df["x"]
    mu = spread.rolling(lookback).mean()
    sd = spread.rolling(lookback).std(ddof=0)
    z = (spread - mu) / (sd + 1e-12)

    out = pd.DataFrame({"spread": spread, "z": z, "beta": beta})
    out.dropna(inplace=True)
    return out

# ----------------------------
# Trading Simulation
# ----------------------------
@dataclass
class Trade:
    pair: str
    direction: str  # "long_spread" oder "short_spread"
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_z: float
    exit_z: float
    return_pct: float  # PnL-Proxy in %


def simulate_pair_z(
    y: pd.Series,
    x: pd.Series,
    z_entry: float = Z_ENTRY_DEFAULT,
    z_exit: float = Z_EXIT_DEFAULT,
    lookback: int = MIN_LOOKBACK_DEFAULT,
    max_hold_days: int = MAX_HOLD_DAYS,
    z_trail_factor: float = Z_TRAIL_FACTOR_DEFAULT,
    z_trail_step: float = Z_TRAIL_STEP_DEFAULT,
    z_trail_start: float = Z_TRAIL_START_DEFAULT,
    min_corr: float = MIN_CORR_DEFAULT,
    pair_name: Optional[str] = None,
) -> Tuple[List[Trade], pd.Series, float, pd.Series]:
    """
    Kern-Strategie:
    - Entry: |z| >= z_entry (z< -z_entry ⇒ long_spread, z>z_entry ⇒ short_spread)
    - Exit: |z| <= z_exit oder Haltedauer erreicht
    - Trailing: exit-threshold kann sich Richtung 0 bewegen (Factor/Step/Start)
    Rückgabe: (Trades, EquityCurve-% (kumulativ), beta, Spread-Serie)
    """
    stats = compute_spread_zscore(y, x, lookback=lookback)
    corr = rolling_corr(y.reindex(stats.index), x.reindex(stats.index), lookback).iloc[-1]
    if pd.isna(corr) or corr < min_corr:
        return [], pd.Series(dtype=float), float("nan"), stats["spread"]

    z = stats["z"].copy()
    spread = stats["spread"].copy()
    dates = z.index
    trades: List[Trade] = []

    in_trade = False
    trade_dir: Optional[str] = None
    entry_idx: Optional[int] = None
    entry_z_val: Optional[float] = None
    current_exit_thresh = z_exit

    for i, d in enumerate(dates):
        zt = float(z.iloc[i])
        if not in_trade:
            if zt >= z_entry:
                in_trade = True
                trade_dir = "short_spread"
                entry_idx = i
                entry_z_val = zt
                current_exit_thresh = z_exit if z_trail_start < 0 else z_trail_start
            elif zt <= -z_entry:
                in_trade = True
                trade_dir = "long_spread"
                entry_idx = i
                entry_z_val = zt
                current_exit_thresh = z_exit if z_trail_start < 0 else z_trail_start
        else:
            held_days = trading_days_between(dates[entry_idx], d)
            exit_signal = (abs(zt) <= current_exit_thresh) or (held_days >= max_hold_days)

            if not pd.isna(zt) and not pd.isna(entry_z_val):
                move = abs(zt)
                target_thresh = max(z_exit, move * z_trail_factor)
                current_exit_thresh = max(z_exit, min(current_exit_thresh, target_thresh + z_trail_step))

            if exit_signal:
                exit_idx = i
                exit_z_val = zt

                entry_spread = float(spread.iloc[entry_idx])
                exit_spread = float(spread.iloc[exit_idx])
                if trade_dir == "long_spread":
                    pnl_raw = exit_spread - entry_spread
                else:
                    pnl_raw = entry_spread - exit_spread
                denom = max(1e-8, abs(entry_spread))
                ret = pnl_raw / denom

                trades.append(
                    Trade(
                        pair=pair_name or f"{y.name}-{x.name}",
                        direction=trade_dir,
                        entry_date=pd.to_datetime(dates[entry_idx]),
                        exit_date=pd.to_datetime(d),
                        entry_z=float(entry_z_val),
                        exit_z=float(exit_z_val),
                        return_pct=float(ret),
                    )
                )
                in_trade = False
                trade_dir = None
                entry_idx = None
                entry_z_val = None
                current_exit_thresh = z_exit

    if in_trade and entry_idx is not None:
        exit_idx = len(dates) - 1
        exit_z_val = float(z.iloc[exit_idx])
        entry_spread = float(spread.iloc[entry_idx])
        exit_spread = float(spread.iloc[exit_idx])
        if trade_dir == "long_spread":
            pnl_raw = exit_spread - entry_spread
        else:
            pnl_raw = entry_spread - exit_spread
        denom = max(1e-8, abs(entry_spread))
        ret = pnl_raw / denom

        trades.append(
            Trade(
                pair=pair_name or f"{y.name}-{x.name}",
                direction=trade_dir or "long_spread",
                entry_date=pd.to_datetime(dates[entry_idx]),
                exit_date=pd.to_datetime(dates[exit_idx]),
                entry_z=float(entry_z_val or 0.0),
                exit_z=float(exit_z_val),
                return_pct=float(ret),
            )
        )

    if not trades:
        return trades, pd.Series(dtype=float), float(stats["beta"].iloc[0]), spread

    pnl_by_date: Dict[pd.Timestamp, float] = {}
    for t in trades:
        pnl_by_date.setdefault(pd.to_datetime(t.exit_date), 0.0)
        pnl_by_date[pd.to_datetime(t.exit_date)] += t.return_pct

    idx = pd.DatetimeIndex(sorted(pnl_by_date))
    eq = pd.Series(index=idx, data=np.cumsum([pnl_by_date[d] for d in idx]))
    eq.name = pair_name or f"{y.name}-{x.name}"
    return trades, eq, float(stats["beta"].iloc[0]), spread

# ----------------------------
# Metriken & Scoring
# ----------------------------
def _max_drawdown(series: pd.Series) -> float:
    if series is None or series.empty:
        return 0.0
    cummax = series.cummax()
    dd = (series - cummax)
    return float(abs(dd.min()))

def compute_rr_and_kelly_from_trades(trades_df: pd.DataFrame) -> Tuple[float, float, float, float]:
    """
    Return: avg_win, avg_loss(negativ), R=avg_win/|avg_loss|, Kelly
    Kelly = p - (1-p)/R, p = win_rate
    """
    if trades_df is None or trades_df.empty:
        return (float("nan"), float("nan"), float("nan"), float("nan"))

    pnl = trades_df["return_pct"].astype(float).values
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    avg_win = float(np.mean(wins)) if wins.size else float("nan")
    avg_loss_abs = float(abs(np.mean(losses))) if losses.size else float("nan")
    R = float(avg_win / avg_loss_abs) if (not math.isnan(avg_win) and not math.isnan(avg_loss_abs) and avg_loss_abs > 0) else float("nan")
    p = float((pnl > 0).mean()) if pnl.size else float("nan")
    if not math.isnan(R) and R > 0 and not math.isnan(p):
        kelly = p - (1 - p) / R
    else:
        kelly = float("nan")
    return avg_win, -avg_loss_abs, R, float(kelly)

def equity_from_trades(trades: List[Trade], start_equity: float = 100_000.0, risk_per_trade: float = 0.005) -> pd.Series:
    """Wandelt Trade-%-Returns in Equity-Kurve mit Risikosizing (PnL bei Exit)."""
    if not trades:
        return pd.Series(dtype=float)
    pnl_by_date: Dict[pd.Timestamp, float] = {}
    for t in sorted(trades, key=lambda x: x.exit_date):
        pnl_by_date.setdefault(pd.to_datetime(t.exit_date), 0.0)
        pnl_by_date[pd.to_datetime(t.exit_date)] += t.return_pct

    dates = sorted(pnl_by_date)
    eq = pd.Series(index=pd.DatetimeIndex(dates), dtype=float)
    equity = start_equity
    for d in dates:
        position_size = equity * float(risk_per_trade)
        equity += position_size * float(pnl_by_date[d])
        eq.loc[d] = equity
    return eq

def compute_time_metrics(eq: pd.Series) -> Dict[str, float]:
    if eq is None or eq.empty:
        return {"cagr": 0.0, "sharpe": 0.0, "max_dd": 0.0, "ret": 0.0}
    rets = eq.pct_change().dropna()
    if rets.empty:
        return {"cagr": 0.0, "sharpe": 0.0, "max_dd": 0.0, "ret": 0.0}
    ann_sharpe = float(np.sqrt(252) * (rets.mean() / (rets.std(ddof=0) + 1e-12)))
    total_ret = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    days = max(1, (eq.index[-1] - eq.index[0]).days)
    years = days / 365.25
    cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (1 / max(1e-9, years)) - 1) if years > 0 else 0.0
    mdd = _max_drawdown(eq / eq.iloc[0])
    return {"cagr": cagr, "sharpe": ann_sharpe, "max_dd": mdd, "ret": total_ret}

def compute_score(metrics: Dict[str, float], kelly: float) -> float:
    """Score: hohe Sharpe/CAGR gut, Drawdown schlecht, Kelly als Bonus."""
    s = 0.0
    s += 2.0 * metrics.get("sharpe", 0.0)
    s += 1.0 * metrics.get("cagr", 0.0)
    s += 0.5 * metrics.get("ret", 0.0)
    s -= 1.0 * metrics.get("max_dd", 0.0)
    if not math.isnan(kelly):
        s += 0.5 * kelly
    return float(s)

# ----------------------------
# Single-Pair Processing
# ----------------------------
def _process_single_pair(args: Tuple) -> Dict:
    (
        pair,
        prices_map,
        z_entry,
        z_exit,
        lookback,
        min_corr,
        z_trail_factor,
        z_trail_step,
        z_trail_start,
        account,
        risk_per_trade,
    ) = args

    a, b = pair
    y = prices_map[a]
    x = prices_map[b]

    try:
        trades, eq_pct, beta = simulate_pair_z(
            y, x,
            z_entry=z_entry,
            z_exit=z_exit,
            lookback=lookback,
            max_hold_days=MAX_HOLD_DAYS,
            z_trail_factor=z_trail_factor,
            z_trail_step=z_trail_step,
            z_trail_start=z_trail_start,
            min_corr=min_corr,
            pair_name=f"{a}-{b}",
        )
    except Exception as e:
        return {
            "pair": f"{a}-{b}",
            "error": str(e),
            "trades": [],
            "equity_curve": [],
            "metrics": {},
            "beta": float("nan"),
            "score": -1e9,
        }

    if not trades:
        return {
            "pair": f"{a}-{b}",
            "trades": [],
            "equity_curve": [],
            "metrics": {},
            "beta": float("nan"),
            "score": -1e9,
        }

    tdf = pd.DataFrame([t.__dict__ for t in trades])
    eq_cash = equity_from_trades(trades, start_equity=float(account), risk_per_trade=float(risk_per_trade))
    tm = compute_time_metrics(eq_cash)
    avg_win, avg_loss, R, kelly = compute_rr_and_kelly_from_trades(tdf)
    score = compute_score(tm, kelly)

    return {
        "pair": f"{a}-{b}",
        "trades": tdf.to_dict(orient="records"),
        "equity_curve": eq_cash.to_dict(),
        "metrics": {**tm, "avg_win": avg_win, "avg_loss": avg_loss, "R": R, "kelly": kelly},
        "beta": beta,
        "score": score,
    }

# ----------------------------
# Portfolio-Run (seq/parallel)
# ----------------------------
def _process_single_pair(args: Tuple) -> Dict:
    (
        pair,
        prices_map,
        z_entry,
        z_exit,
        lookback,
        min_corr,
        z_trail_factor,
        z_trail_step,
        z_trail_start,
        account,
        risk_per_trade,
    ) = args

    a, b = pair
    y = prices_map[a]
    x = prices_map[b]

    try:
        trades, eq_pct, beta, spread = simulate_pair_z(
            y,
            x,
            z_entry=z_entry,
            z_exit=z_exit,
            lookback=lookback,
            max_hold_days=MAX_HOLD_DAYS,
            z_trail_factor=z_trail_factor,
            z_trail_step=z_trail_step,
            z_trail_start=z_trail_start,
            min_corr=min_corr,
            pair_name=f"{a}-{b}",
        )
    except Exception as e:
        return {
            "pair": f"{a}-{b}",
            "error": str(e),
            "trades": [],
            "equity_curve": [],
            "metrics": {},
            "beta": float("nan"),
            "score": -1e9,
        }

    if not trades:
        return {
            "pair": f"{a}-{b}",
            "trades": [],
            "equity_curve": [],
            "metrics": {},
            "beta": float("nan"),
            "score": -1e9,
        }

    tdf = pd.DataFrame([t.__dict__ for t in trades])
    eq_cash = equity_from_trades(trades, start_equity=float(account), risk_per_trade=float(risk_per_trade))

    tm = compute_time_metrics(eq_cash)
    tm_oos = compute_time_metrics_oos(eq_cash, split=0.7)

    avg_win, avg_loss, R, kelly = compute_rr_and_kelly_from_trades(tdf)
    score = compute_score(tm, kelly)

    hl = estimate_half_life(spread)
    coint_p = compute_cointegration_pvalue(y, x)

    metrics = {
        **tm,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "R": R,
        "kelly": kelly,
        "cagr_oos": tm_oos.get("cagr", 0.0),
        "sharpe_oos": tm_oos.get("sharpe", 0.0),
        "max_dd_oos": tm_oos.get("max_dd", 0.0),
        "ret_oos": tm_oos.get("ret", 0.0),
        "half_life": hl,
        "coint_pvalue": coint_p,
    }

    return {
        "pair": f"{a}-{b}",
        "trades": tdf.to_dict(orient="records"),
        "equity_curve": eq_cash.to_dict(),
        "metrics": metrics,
        "beta": beta,
        "score": score,
    }


def run_full_backtest(
    symbols: List[str],
    *,
    z_entry: float,
    z_exit: float,
    min_corr: float,
    lookback: int,
    z_trail_factor: float,
    z_trail_step: float,
    z_trail_start: float,
    account: float,
    risk_per_trade: float,
    cache: bool,
    start: Optional[str] = None,
    end: Optional[str] = None,
    prices: Optional[Dict[str, pd.Series]] = None,
) -> Tuple[pd.DataFrame, List[Dict]]:
    if prices is None:
        prices = download_price_data(symbols, start=start, end=end, interval="1d", cache=cache)
    pairs = make_pairs(symbols)

    results: List[Dict] = []
    for pair in pairs:
        res = _process_single_pair(
            (
                pair,
                prices,
                z_entry,
                z_exit,
                lookback,
                min_corr,
                z_trail_factor,
                z_trail_step,
                z_trail_start,
                account,
                risk_per_trade,
            )
        )
        if res.get("trades"):
            results.append(res)

    df = pd.DataFrame(results)
    if not df.empty:
        df.sort_values("score", ascending=False, inplace=True)
    return df, results


def run_full_backtest_parallel(
    symbols: List[str],
    *,
    z_entry: float,
    z_exit: float,
    min_corr: float,
    lookback: int,
    z_trail_factor: float,
    z_trail_step: float,
    z_trail_start: float,
    account: float,
    risk_per_trade: float,
    cache: bool,
    start: Optional[str] = None,
    end: Optional[str] = None,
    max_workers: int = 4,
    prices: Optional[Dict[str, pd.Series]] = None,
) -> Tuple[pd.DataFrame, List[Dict]]:
    if prices is None:
        prices = download_price_data(symbols, start=start, end=end, interval="1d", cache=cache)
    pairs = make_pairs(symbols)

    tasks = []
    for pair in pairs:
        tasks.append(
            (
                pair,
                prices,
                z_entry,
                z_exit,
                lookback,
                min_corr,
                z_trail_factor,
                z_trail_step,
                z_trail_start,
                account,
                risk_per_trade,
            )
        )

    results: List[Dict] = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_process_single_pair, t) for t in tasks]
        for f in as_completed(futs):
            try:
                r = f.result()
                if r.get("trades"):
                    results.append(r)
            except Exception:
                continue

    df = pd.DataFrame(results)
    if not df.empty:
        df.sort_values("score", ascending=False, inplace=True)
    return df, results

# ----------------------------
# Grid Scan / Adaptive
# ----------------------------
def run_full_grid_scan(
    universe: List[str],
    *,
    z_entry_vals: List[float],
    min_corr_vals: List[float],
    lookbacks: List[int],
    z_exit: float,
    z_trail_factor: float,
    z_trail_step: float,
    z_trail_start: float,
    account: float,
    risk_per_trade: float,
    cache: bool,
    start: Optional[str],
    end: Optional[str],
    parallel_workers: int,
    run_dir: Path,
) -> pd.DataFrame:
    """
    Durchläuft ALLE Kombinationen (z_entry, min_corr, lookback), sammelt alle Pair-Results
    und gibt eine große Ergebnistabelle zurück (eine Zeile pro Pair & Param-Kombi).
    Nutzt ein einmal geladenes Price-Dict für Speed.
    """
    all_rows: List[pd.DataFrame] = []

    total_combos = len(z_entry_vals) * len(min_corr_vals) * len(lookbacks)
    print(f"[grid] combinations = {total_combos} | z_entry={z_entry_vals} | min_corr={min_corr_vals} | lookback={lookbacks}")

    base_prices = download_price_data(universe, start=start, end=end, interval="1d", cache=cache)

    combo_idx = 0
    for z_entry in z_entry_vals:
        for min_corr in min_corr_vals:
            for lookback in lookbacks:
                combo_idx += 1
                print(f"[grid] {combo_idx}/{total_combos}: z_entry={z_entry:.2f}, min_corr={min_corr:.2f}, lookback={lookback}")

                run_kwargs = dict(
                    z_entry=z_entry,
                    z_exit=z_exit,
                    min_corr=min_corr,
                    lookback=lookback,
                    z_trail_factor=z_trail_factor,
                    z_trail_step=z_trail_step,
                    z_trail_start=z_trail_start,
                    account=account,
                    risk_per_trade=risk_per_trade,
                    cache=cache,
                    start=start,
                    end=end,
                )

                if parallel_workers and parallel_workers > 1:
                    df_combo, _ = run_full_backtest_parallel(universe, max_workers=parallel_workers, prices=base_prices, **run_kwargs)
                else:
                    df_combo, _ = run_full_backtest(universe, prices=base_prices, **run_kwargs)

                if df_combo is None or df_combo.empty:
                    continue

                df_combo = df_combo.copy()
                df_combo["z_entry"] = float(z_entry)
                df_combo["min_corr"] = float(min_corr)
                df_combo["lookback"] = int(lookback)
                df_combo["z_exit"] = float(z_exit)
                df_combo["z_trail_factor"] = float(z_trail_factor)
                df_combo["z_trail_step"] = float(z_trail_step)
                df_combo["z_trail_start"] = float(z_trail_start)

                all_rows.append(df_combo)

    if not all_rows:
        return pd.DataFrame()

    df_all = pd.concat(all_rows, ignore_index=True)
    df_all.sort_values("score", ascending=False, inplace=True)
    df_all.to_csv(run_dir / "results_fullscan.csv", index=False)
    return df_all

# ----------------------------
# CLI
# ----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pairs/Arbitrage Backtester v6.0-stable (with full-grid adaptive)")

    # Universe & data
    p.add_argument("--assets", choices=["etf", "stocks", "both"], default="etf")
    p.add_argument("--symbols", type=str, default="", help="Comma-separated symbols override (e.g. 'SPY,QQQ,IWM')")
    p.add_argument("--start", type=str, default=None)
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--cache", action="store_true", help="Enable yfinance disk cache")

    # Strategy params (fixed mode)
    p.add_argument("--z-entry", type=float, default=Z_ENTRY_DEFAULT)
    p.add_argument("--z-exit", type=float, default=Z_EXIT_DEFAULT)
    p.add_argument("--lookback", type=int, default=MIN_LOOKBACK_DEFAULT)
    p.add_argument("--min-corr", type=float, default=MIN_CORR_DEFAULT)
    p.add_argument("--z-trail-factor", type=float, default=Z_TRAIL_FACTOR_DEFAULT)
    p.add_argument("--z-trail-step", type=float, default=Z_TRAIL_STEP_DEFAULT)
    p.add_argument("--z-trail-start", type=float, default=Z_TRAIL_START_DEFAULT)

    # Sizing / account
    p.add_argument("--account", type=float, default=100_000.0)
    p.add_argument("--risk-per-trade", type=float, default=0.005)

    # Execution
    p.add_argument("--parallel", type=int, default=0, help="#worker processes (0/1 = sequential)")

    # Reporting
    p.add_argument("--top", type=int, default=10, help="How many top pairs to plot")
    p.add_argument("--pdf-per-strategy", action="store_true")
    p.add_argument("--save-composite-pdf", action="store_true")
    p.add_argument("--save-meta", action="store_true")
    p.add_argument("--heatmap", action="store_true", help="Create heatmap for grid scan")

    # Adaptive / Full Grid Scan
    p.add_argument("--adaptive", action="store_true", help="Run full grid scan over parameter ranges")
    p.add_argument("--z-entry-range", type=str, default="1.0,2.5,0.1", help="start,end,step for z-entry (exclusive end, like arange)")
    p.add_argument("--min-corr-range", type=str, default="0.10,0.90,0.05", help="start,end,step for min-corr (exclusive end)")
    p.add_argument("--lookbacks", type=str, default="20,30,60", help="comma-separated list of lookbacks (e.g., '20,30,60')")

    return p.parse_args()

def load_universe(args: argparse.Namespace) -> List[str]:
    if args.symbols:
        syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        if len(syms) >= 2:
            return syms
    return DEFAULT_ETFS

# ----------------------------
# Main
# ----------------------------
def main() -> None:
    import pandas as pd

    args = parse_args()

    universe = load_universe(args)
    mode = "FULL-GRID" if args.adaptive else "FIXED"
    print(f"Mode={mode} | Universe size = {len(universe)} | cache={args.cache} | parallel={args.parallel}")

    # Run-Ordner
    prefix = "run_fullscan" if args.adaptive else "run"
    safe_mkdir(OUT_ROOT)
    run_dir = next_run_dir(OUT_ROOT, prefix=prefix)
    safe_mkdir(run_dir)

    run_title = f"Pairs Backtest v6.0-stable | {mode} | {datetime.now().strftime('%Y-%m-%d %H:%M')} | N={len(universe)}"

    if not args.adaptive:
        # Fester Parametersatz
        run_kwargs = dict(
            z_entry=args.z_entry,
            z_exit=args.z_exit,
            min_corr=args.min_corr,
            lookback=args.lookback,
            z_trail_factor=args.z_trail_factor,
            z_trail_step=args.z_trail_step,
            z_trail_start=args.z_trail_start,
            account=args.account,
            risk_per_trade=args.risk_per_trade,
            cache=args.cache,
            start=args.start,
            end=args.end,
        )

        if args.parallel and args.parallel > 1:
            results_df, raw_data = run_full_backtest_parallel(universe, max_workers=args.parallel, **run_kwargs)
        else:
            results_df, raw_data = run_full_backtest(universe, **run_kwargs)

        if results_df is None or results_df.empty:
            print("No results. Exiting.")
            return

        # Save CSV
        results_csv = run_dir / "results.csv"
        results_df.to_csv(results_csv, index=False)

        # Benchmark price series (für Overlay)
        bench_eq = None
        try:
            bench = download_price_data([BENCHMARK], start=args.start, end=args.end, cache=args.cache)[BENCHMARK]
            bench_eq = bench
        except Exception:
            bench_eq = None

        # Top strategies: Plots
        per_pngs: List[Tuple[str, Path]] = []
        equity_list: List[pd.Series] = []

        top = results_df.head(max(1, int(args.top)))
        for _, row in top.iterrows():
            pair = row["pair"]
            eq_dict = row["equity_curve"]
            eq = pd.Series(eq_dict)
            equity_list.append(eq)

            png_path = run_dir / f"{pair}_equity.png"
            sharpe = row["metrics"].get("sharpe", 0)
            plot_pair_equity(eq, bench_eq, png_path, title=f"{pair} | Sharpe={sharpe:.2f}")
            if args.pdf_per_strategy:
                per_pngs.append((pair, png_path))

        # Portfolio equity über alle Strategien
        portfolio_eq = build_portfolio_equity(raw_data, start_equity=float(args.account), risk_per_trade=float(args.risk_per_trade))
        if portfolio_eq is not None and not portfolio_eq.empty:
            port_png = run_dir / "portfolio_equity.png"
            plot_pair_equity(portfolio_eq, bench_eq, port_png, title="Portfolio (alle Strategien)")

        # Composite
        comp_png = run_dir / "composite_equity.png"
        if equity_list:
            plot_composite_equity(equity_list, bench_eq, comp_png, title="Composite of Top Strategies")

        # PDF
        out_pdf = run_dir / "summary.pdf"
        generate_summary_pdf(
            out_pdf,
            top,
            run_title,
            composite_png=(comp_png if args.save_composite_pdf else None),
            per_strategy_pngs=(per_pngs if args.pdf_per_strategy else None),
        )

        # Meta
        if args.save_meta:
            meta = {
                "title": run_title,
                "args": vars(args),
                "top_pairs": top.to_dict(orient="records"),
            }
            with open(run_dir / "run_meta.json", "w", encoding="utf-8") as f:
                meta = convert_keys_to_str(meta)
                json.dump(meta, f, indent=2, default=str)

        print(f"Done. Output saved to: {run_dir}")
        return

    # --------- FULL GRID SCAN / ADAPTIVE ---------
    # Ranges parsen
    z_entry_start, z_entry_end, z_entry_step = parse_range_str(args.z_entry_range, (1.0, 2.5, 0.1))
    min_corr_start, min_corr_end, min_corr_step = parse_range_str(args.min_corr_range, (0.10, 0.90, 0.05))
    lookbacks = parse_int_list(args.lookbacks, [20, 30, 60])

    z_entry_vals = frange(z_entry_start, z_entry_end, z_entry_step)
    min_corr_vals = frange(min_corr_start, min_corr_end, min_corr_step)

    df_all = run_full_grid_scan(
        universe,
        z_entry_vals=z_entry_vals,
        min_corr_vals=min_corr_vals,
        lookbacks=lookbacks,
        z_exit=args.z_exit,
        z_trail_factor=args.z_trail_factor,
        z_trail_step=args.z_trail_step,
        z_trail_start=args.z_trail_start,
        account=args.account,
        risk_per_trade=args.risk_per_trade,
        cache=args.cache,
        start=args.start,
        end=args.end,
        parallel_workers=args.parallel,
        run_dir=run_dir,
    )

    if df_all is None or df_all.empty:
        print("No results from full grid scan. Exiting.")
        return

    # Speichere globale CSV (bereits in run_full_grid_scan getan)
    results_csv = run_dir / "results_fullscan_sorted.csv"
    df_all.to_csv(results_csv, index=False)

    # Benchmark price series (für Overlay)
    bench_eq = None
    try:
        bench = download_price_data([BENCHMARK], start=args.start, end=args.end, cache=args.cache)[BENCHMARK]
        bench_eq = bench
    except Exception:
        bench_eq = None

    # Top N der global besten Strategien (über alle Parameter-Kombis)
    per_pngs: List[Tuple[str, Path]] = []
    equity_list: List[pd.Series] = []

    top = df_all.head(max(1, int(args.top)))
    for _, row in top.iterrows():
        pair = row["pair"]
        eq_dict = row["equity_curve"]
        eq = pd.Series(eq_dict)
        equity_list.append(eq)

        png_path = run_dir / f"{pair}_equity.png"
        sharpe = row["metrics"].get("sharpe", 0)
        plot_pair_equity(eq, bench_eq, png_path, title=f"{pair} | Sharpe={sharpe:.2f} | z={row['z_entry']:.2f} corr>={row['min_corr']:.2f} lb={int(row['lookback'])}")
        if args.pdf_per_strategy:
            per_pngs.append((pair, png_path))

    # Composite
    comp_png = run_dir / "composite_equity.png"
    if equity_list:
        plot_composite_equity(equity_list, bench_eq, comp_png, title="Composite of Top Strategies (Full Grid)")

    # Heatmap (optional)
    if args.heatmap:
        heatmap_png = run_dir / "grid_heatmap_score.png"
        try:
            plot_grid_heatmap(df_all.assign(score=df_all["score"].astype(float)), heatmap_png, value_col="score")
        except Exception:
            pass

    # PDF
    out_pdf = run_dir / "summary_fullscan.pdf"
    generate_summary_pdf(
        out_pdf,
        top,
        run_title + " (Full Grid Scan)",
        composite_png=(comp_png if args.save_composite_pdf else None),
        per_strategy_pngs=(per_pngs if args.pdf_per_strategy else None),
    )

    # Meta
    if args.save_meta:
        import pandas as pd
        import numpy as np

        # --- FULL SANITIZER: recursively convert any Timestamp keys to strings ---
        def ensure_str_keys(x):
            """Recursively convert all dict keys and nested structures to JSON-safe strings."""
            if isinstance(x, dict):
                clean = {}
                for k, v in x.items():
                    clean[str(k)] = ensure_str_keys(v)
                return clean
            elif isinstance(x, (list, tuple, set, pd.Series, np.ndarray)):
                return [ensure_str_keys(i) for i in list(x)]
            elif isinstance(x, (pd.Timestamp, pd.Timedelta, np.datetime64)):
                return str(x)
            elif isinstance(x, (np.int64, np.int32)):
                return int(x)
            elif isinstance(x, (np.float64, np.float32)):
                return float(x)
            else:
                return x

        # --- Clean equity_curve and bench_equity columns directly in top DataFrame ---
        if "equity_curve" in top.columns:
            top["equity_curve"] = top["equity_curve"].apply(ensure_str_keys)
        if "bench_equity" in top.columns:
            top["bench_equity"] = top["bench_equity"].apply(ensure_str_keys)

        # --- Build meta object (safe for JSON) ---
        meta = {
            "title": run_title + " (Full Grid Scan)",
            "args": vars(args),
            "top_pairs": top.to_dict(orient="records"),
            "grid": {
                "z_entry": [float(x) for x in z_entry_vals],
                "min_corr": [float(x) for x in min_corr_vals],
                "lookbacks": [int(x) for x in lookbacks],
            },
        }

        # --- Save meta to JSON (guaranteed serializable) ---
        with open(run_dir / "run_meta.json", "w", encoding="utf-8") as f:
            safe_meta = ensure_str_keys(meta)
            json.dump(safe_meta, f, indent=2)

        print(f"✅ Done (Full Grid). Output saved to: {run_dir}")

if __name__ == "__main__":
    main()