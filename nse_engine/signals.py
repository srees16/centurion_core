"""
Causal forecast panels: fast_trend, slow_trend and low_vol.

Every function returns date x symbol DataFrames whose row ``t`` depends only
on rows ``<= t`` of the inputs.

Pipeline
--------
1. Raw rules
   * EWMAC(f, s) = (EWMA_f(close) - EWMA_s(close)) / (close * daily_vol), with
     daily_vol the EWM std (span ``vol_span``) of daily returns.  Dividing by
     ``close * daily_vol`` converts the price difference into units of daily
     price volatility, so the forecast is invariant to the price level.
   * 12-1 momentum: close[t-skip] / close[t-lookback] - 1, turned into a
     centred rank score (-0.5, 0.5) within the universe of the day.
   * low vol: centred rank of -(realised vol over ``low_vol_lookback``).
2. Each rule is normalised with a point-in-time scalar
   ``target_abs_forecast / expanding mean |raw|`` pooled over universe members
   on rows strictly before ``t`` (rules within a group are normalised
   separately so that EWMAC units and rank units are comparable before they
   are averaged), then the group average is normalised the same way and
   capped at ``+-forecast_cap``.
3. Groups are combined with the fixed ``SignalConfig.weights()`` and a
   forecast diversification multiplier FDM = 1 / sqrt(w' C w), where C is the
   pooled correlation of group forecasts (negative correlations floored at 0)
   over the previous ``fdm_lookback_days`` rows, refreshed every
   ``fdm_refresh_every_n_days`` rows and capped at ``fdm_cap``.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from nse_engine.config import SignalConfig

logger = logging.getLogger(__name__)

GROUPS: Tuple[str, ...] = ("fast_trend", "slow_trend", "low_vol", "delivery")
_EWM_CHUNK = 64   # columns per pass of the finite-memory EWM (see _truncated_ewm)
FDM_MIN_POOLED_OBS = 100  # pooled (date, symbol) observations needed before FDM != 1


# ----------------------------------------------------------------------------
# raw rules
# ----------------------------------------------------------------------------


def daily_returns(close: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns (NaN where either close is missing)."""
    return close.astype("float64").pct_change(fill_method=None)


def _truncated_ewm(
    frame: pd.DataFrame, span: int, memory: int, min_periods: int, want_std: bool = False
) -> pd.DataFrame:
    """Causal EWM mean (or debiased std) whose kernel stops after ``memory`` rows.

    pandas' EWM remembers every row since the first loaded one, so the value on
    a date depends on where the data was loaded from (2% of the slow EWMA's
    weight sits beyond two years for span 256). Here weight ``(1-a)^k`` is
    applied to lag ``k < memory`` only, NaNs are skipped and the weights
    renormalised over the rows present (as pandas does with ``adjust=True``),
    and the sum is accumulated lag by lag in a fixed order — so the result is
    the same number whatever row the panel starts at, provided ``memory`` rows
    precede the date. ``min_periods`` counts observations inside the window.
    """
    x = frame.to_numpy(dtype="float64")
    n, m = x.shape
    alpha = 2.0 / (span + 1.0)
    L = int(min(max(memory, 1), n))
    w = (1.0 - alpha) ** np.arange(L)
    present = np.isfinite(x)
    x0 = np.where(present, x, 0.0)
    # observations inside the window (integer cumulative sums are exact)
    cum = np.cumsum(present, axis=0)
    cnt = cum.copy()
    cnt[L:] -= cum[:-L]
    out = np.empty((n, m))
    # Column chunks keep each working set in cache: the lag loop then streams
    # from L2 instead of RAM (the panel-wide version spent ~85 s on 2,768 names).
    for c0 in range(0, m, _EWM_CHUNK):
        c1 = min(c0 + _EWM_CHUNK, m)
        pr = present[:, c0:c1].astype("float64")
        xx = x0[:, c0:c1]
        v1 = np.zeros((n, c1 - c0)); s1 = np.zeros((n, c1 - c0))
        v2 = np.zeros((n, c1 - c0)) if want_std else None
        s2 = np.zeros((n, c1 - c0)) if want_std else None
        xsq = xx * xx if want_std else None
        for k in range(L):                   # fixed order: exact for any load start
            wk = w[k]
            src = slice(0, n - k); dst = slice(k, n)
            v1[dst] += wk * pr[src]
            s1[dst] += wk * xx[src]
            if want_std:
                v2[dst] += (wk * wk) * pr[src]
                s2[dst] += wk * xsq[src]
        with np.errstate(divide="ignore", invalid="ignore"):
            safe_v1 = np.where(v1 > 0, v1, np.nan)
            if want_std:
                num = s2 - s1 * s1 / safe_v1                          # sum w (x - mean)^2
                den = v1 - v2 / safe_v1                               # pandas bias=False
                out[:, c0:c1] = np.sqrt(np.where(den > 0, np.maximum(num, 0.0) / den, np.nan))
            else:
                out[:, c0:c1] = s1 / safe_v1
    out = np.where(cnt >= max(int(min_periods), 1), out, np.nan)
    return pd.DataFrame(out, index=frame.index, columns=frame.columns)


def ewm_daily_vol(returns: pd.DataFrame, span: int, memory_spans: int = 0) -> pd.DataFrame:
    """EWM standard deviation of daily returns (finite memory when ``memory_spans`` > 0)."""
    min_periods = max(span // 2, 2)
    if memory_spans > 0:
        return _truncated_ewm(returns, span, memory_spans * span, min_periods, want_std=True)
    return returns.ewm(span=span, min_periods=min_periods).std()


def ewmac_raw(
    close: pd.DataFrame, fast: int, slow: int, vol_span: int, daily_vol: Optional[pd.DataFrame] = None,
    memory_spans: int = 0,
) -> pd.DataFrame:
    """(EWMA_fast - EWMA_slow) / (close * daily_vol): price-level invariant."""
    close = close.astype("float64")
    if daily_vol is None:
        daily_vol = ewm_daily_vol(daily_returns(close), vol_span, memory_spans)
    if memory_spans > 0:
        ew_f = _truncated_ewm(close, fast, memory_spans * fast, fast)
        ew_s = _truncated_ewm(close, slow, memory_spans * slow, slow)
    else:
        ew_f = close.ewm(span=fast, min_periods=fast).mean()
        ew_s = close.ewm(span=slow, min_periods=slow).mean()
    denom = close * daily_vol
    denom = denom.where(denom > 0)
    return (ew_f - ew_s) / denom


def momentum_raw(close: pd.DataFrame, lookback: int, skip: int) -> pd.DataFrame:
    """Return from t-lookback to t-skip (12-1 momentum by default)."""
    close = close.astype("float64")
    return close.shift(skip) / close.shift(lookback) - 1.0


def realised_vol(returns: pd.DataFrame, lookback: int, annualise: bool = True) -> pd.DataFrame:
    """Rolling standard deviation of daily returns."""
    vol = returns.rolling(lookback, min_periods=max(lookback // 2, 2)).std()
    return vol * np.sqrt(252.0) if annualise else vol


def centred_rank(raw: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional rank within ``mask`` mapped to (-0.5, 0.5), mean 0.

    score = (rank - 0.5) / n - 0.5 with rank 1 = smallest raw value.
    """
    x = raw.where(mask.to_numpy() & np.isfinite(raw.to_numpy()))
    ranks = x.rank(axis=1, method="average")
    n = x.notna().sum(axis=1).replace(0, np.nan)
    return (ranks.sub(0.5)).div(n, axis=0) - 0.5


# ----------------------------------------------------------------------------
# normalisation, FDM, combination
# ----------------------------------------------------------------------------


def _past_window_sums(values: np.ndarray, window: int) -> np.ndarray:
    """sum(values[t-window : t]) for each t — the strict past, fixed-order sums."""
    n = len(values)
    out = np.zeros(n, dtype="float64")
    for k in range(1, min(window, n - 1) + 1):   # lag k contributes to t >= k
        out[k:] += values[: n - k]
    return out


def pit_forecast_scalar(
    raw: pd.DataFrame, mask: pd.DataFrame, target_abs: float, min_obs: int, window: int = 0
) -> Tuple[pd.Series, pd.Series]:
    """Point-in-time forecast scalar and warm-up flag per date.

    scalar[t] = target_abs / mean(|raw|) pooled over (date < t, symbol in
    mask) observations — over every earlier loaded date when ``window`` is 0
    (legacy; the value then depends on where the data was loaded from), or
    over the ``window`` dates before ``t``.  ``warmup[t]`` is True while fewer
    than ``min_obs`` dates with observations precede ``t`` (the scalar is
    still the causal estimate; it is NaN only when no past observation exists).
    """
    a = np.abs(raw.to_numpy(dtype="float64"))
    valid = mask.to_numpy() & np.isfinite(a)
    sums = np.where(valid, a, 0.0).sum(axis=1)
    counts = valid.sum(axis=1)
    if window > 0:
        cum_sum = _past_window_sums(sums, window)
        cum_cnt = _past_window_sums(counts.astype("float64"), window)
        cum_dates = _past_window_sums((counts > 0).astype("float64"), window)
    else:
        cum_sum = np.concatenate([[0.0], np.cumsum(sums)[:-1]])
        cum_cnt = np.concatenate([[0], np.cumsum(counts)[:-1]])
        cum_dates = np.concatenate([[0], np.cumsum(counts > 0)[:-1]])
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_abs = np.where(cum_cnt > 0, cum_sum / np.maximum(cum_cnt, 1), np.nan)
        scalar = np.where(mean_abs > 0, target_abs / mean_abs, np.nan)
    idx = raw.index
    return pd.Series(scalar, index=idx), pd.Series(cum_dates < min_obs, index=idx)


def normalise_forecast(raw: pd.DataFrame, mask: pd.DataFrame, cfg: SignalConfig) -> Tuple[pd.DataFrame, pd.Series]:
    """Scale raw forecasts to ``target_abs_forecast`` average and cap."""
    scalar, warm = pit_forecast_scalar(raw, mask, cfg.target_abs_forecast, cfg.normalizer_min_obs,
                                       cfg.normalizer_window_days)
    out = raw.mul(scalar, axis=0).clip(-cfg.forecast_cap, cfg.forecast_cap)
    return out, warm


def _nanmean_frames(frames: List[pd.DataFrame]) -> pd.DataFrame:
    stack = np.stack([f.to_numpy(dtype="float64") for f in frames])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        m = np.nanmean(stack, axis=0)
    return pd.DataFrame(m, index=frames[0].index, columns=frames[0].columns)


def fdm_series(groups: Dict[str, pd.DataFrame], weights: Dict[str, float], mask: pd.DataFrame, cfg: SignalConfig) -> pd.Series:
    """Forecast diversification multiplier per date (causal, stepwise)."""
    names = [g for g in weights if g in groups]
    idx = mask.index
    n = len(idx)
    if len(names) < 2:
        return pd.Series(1.0, index=idx)
    k = len(names)
    F = np.stack([groups[g].to_numpy(dtype="float64") for g in names], axis=-1)  # n x m x k
    valid = mask.to_numpy() & np.isfinite(F).all(axis=-1)
    Fz = np.where(valid[..., None], F, 0.0)
    cnt = valid.sum(axis=1).astype("float64")  # n
    s1 = Fz.sum(axis=1)  # n x k
    s2 = np.einsum("tmi,tmj->tij", Fz, Fz)  # n x k x k
    c_cnt = np.concatenate([[0.0], np.cumsum(cnt)])
    c_s1 = np.concatenate([np.zeros((1, k)), np.cumsum(s1, axis=0)])
    c_s2 = np.concatenate([np.zeros((1, k, k)), np.cumsum(s2, axis=0)])
    date_has = np.concatenate([[0], np.cumsum(cnt > 0)])
    w = np.array([weights[g] for g in names], dtype="float64")
    w = w / w.sum()
    out = np.ones(n)
    every = max(int(cfg.fdm_refresh_every_n_days), 1)
    if cfg.calendar_schedule:
        from nse_engine.calendar import period_start_mask
        refresh_here = period_start_mask(idx, every)
    else:
        refresh_here = (np.arange(n) % every) == 0
    current = 1.0
    for pos in range(n):
        if refresh_here[pos]:
            lo = max(0, pos - cfg.fdm_lookback_days)
            if cfg.calendar_schedule:
                # direct sums over the window: the same numbers for any load start
                N = float(cnt[lo:pos].sum())
                n_dates = int((cnt[lo:pos] > 0).sum())
                win_s1 = s1[lo:pos].sum(axis=0)
                win_s2 = s2[lo:pos].sum(axis=0)
            else:
                N = c_cnt[pos] - c_cnt[lo]
                n_dates = date_has[pos] - date_has[lo]
                win_s1 = c_s1[pos] - c_s1[lo]
                win_s2 = c_s2[pos] - c_s2[lo]
            current = 1.0
            if N >= FDM_MIN_POOLED_OBS and n_dates >= cfg.normalizer_min_obs:
                m1 = win_s1 / N
                cov = win_s2 / N - np.outer(m1, m1)
                sd = np.sqrt(np.clip(np.diag(cov), 0, None))
                with np.errstate(divide="ignore", invalid="ignore"):
                    corr = cov / np.outer(sd, sd)
                corr = np.nan_to_num(corr, nan=0.0)
                corr = np.clip(corr, 0.0, 1.0)
                np.fill_diagonal(corr, 1.0)
                var = float(w @ corr @ w)
                if var > 0:
                    current = float(min(1.0 / np.sqrt(var), cfg.fdm_cap))
        out[pos] = current
    return pd.Series(out, index=idx)


def combine_forecasts(
    groups: Dict[str, pd.DataFrame], weights: Dict[str, float], fdm: pd.Series, cap: float
) -> pd.DataFrame:
    """Weighted group average (weights renormalised over available groups) x FDM, capped."""
    names = [g for g in weights if g in groups]
    first = groups[names[0]]
    num = np.zeros(first.shape)
    den = np.zeros(first.shape)
    for g in names:
        f = groups[g].to_numpy(dtype="float64")
        ok = np.isfinite(f)
        num += np.where(ok, f, 0.0) * weights[g]
        den += ok * weights[g]
    with np.errstate(divide="ignore", invalid="ignore"):
        comb = np.where(den > 0, num / den, np.nan)
    comb = comb * fdm.to_numpy()[:, None]
    return pd.DataFrame(np.clip(comb, -cap, cap), index=first.index, columns=first.columns)


# ----------------------------------------------------------------------------
# group panels
# ----------------------------------------------------------------------------


@dataclass
class SignalPanels:
    groups: Dict[str, pd.DataFrame]
    combined: pd.DataFrame
    fdm: pd.Series
    warmup: pd.Series  # True where any normaliser is still warming up
    diagnostics: Dict[str, pd.Series] = field(default_factory=dict)


def fast_trend_forecast(
    close: pd.DataFrame, mask: pd.DataFrame, cfg: SignalConfig, daily_vol: Optional[pd.DataFrame] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    if daily_vol is None:
        daily_vol = ewm_daily_vol(daily_returns(close), cfg.vol_span, cfg.ewm_memory_spans)
    parts, warm = [], pd.Series(False, index=close.index)
    for f, s in cfg.fast_ewmac:
        n, w = normalise_forecast(ewmac_raw(close, f, s, cfg.vol_span, daily_vol, cfg.ewm_memory_spans), mask, cfg)
        parts.append(n)
        warm |= w
    out, w = normalise_forecast(_nanmean_frames(parts), mask, cfg)
    return out, warm | w


def slow_trend_forecast(
    close: pd.DataFrame, mask: pd.DataFrame, cfg: SignalConfig, daily_vol: Optional[pd.DataFrame] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    if daily_vol is None:
        daily_vol = ewm_daily_vol(daily_returns(close), cfg.vol_span, cfg.ewm_memory_spans)
    parts, warm = [], pd.Series(False, index=close.index)
    for f, s in cfg.slow_ewmac:
        n, w = normalise_forecast(ewmac_raw(close, f, s, cfg.vol_span, daily_vol, cfg.ewm_memory_spans), mask, cfg)
        parts.append(n)
        warm |= w
    mom = centred_rank(momentum_raw(close, cfg.momentum_lookback, cfg.momentum_skip), mask)
    n, w = normalise_forecast(mom, mask, cfg)
    parts.append(n)
    warm |= w
    out, w = normalise_forecast(_nanmean_frames(parts), mask, cfg)
    return out, warm | w


def low_vol_forecast(
    close: pd.DataFrame, mask: pd.DataFrame, cfg: SignalConfig, returns: Optional[pd.DataFrame] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    if returns is None:
        returns = daily_returns(close)
    raw = centred_rank(-realised_vol(returns, cfg.low_vol_lookback), mask)
    return normalise_forecast(raw, mask, cfg)


def delivery_forecast(
    delivery_pct: pd.DataFrame, mask: pd.DataFrame, cfg: SignalConfig
) -> Tuple[pd.DataFrame, pd.Series]:
    """Rank of trailing mean delivery %: names whose volume is taken to demat.

    The India evidence (BacktestIndia, 2006-2025) puts the momentum premium in
    low-turnover, patiently held names; delivery % is the NSE-published proxy
    for that. Rows before ``delivery_lookback`` sessions of data, and names
    without MTO coverage, are NaN and drop out of the rank.
    """
    d = delivery_pct.astype("float64")
    mean = d.rolling(cfg.delivery_lookback, min_periods=max(cfg.delivery_lookback // 2, 5)).mean()
    raw = centred_rank(mean, mask)
    return normalise_forecast(raw, mask, cfg)


def compute_signal_panels(
    close: pd.DataFrame, universe_mask: pd.DataFrame, cfg: SignalConfig, returns: Optional[pd.DataFrame] = None,
    delivery_pct: Optional[pd.DataFrame] = None,
) -> SignalPanels:
    """All group forecasts, FDM and the combined forecast."""
    close = close.astype("float64")
    if returns is None:
        returns = daily_returns(close)
    daily_vol = ewm_daily_vol(returns, cfg.vol_span, cfg.ewm_memory_spans)
    weights = cfg.weights()

    def _delivery():
        if delivery_pct is None:
            raise ValueError("signal group 'delivery' needs MarketData.delivery_pct (NSE MTO files in the store)")
        return delivery_forecast(delivery_pct.reindex(index=close.index, columns=close.columns), universe_mask, cfg)

    builders = {
        "fast_trend": lambda: fast_trend_forecast(close, universe_mask, cfg, daily_vol),
        "slow_trend": lambda: slow_trend_forecast(close, universe_mask, cfg, daily_vol),
        "low_vol": lambda: low_vol_forecast(close, universe_mask, cfg, returns),
        "delivery": _delivery,
    }
    unknown = set(weights) - set(builders)
    if unknown:
        raise ValueError(f"unknown signal groups in config: {sorted(unknown)}")
    groups: Dict[str, pd.DataFrame] = {}
    warm = pd.Series(False, index=close.index)
    for name, w in weights.items():
        if w == 0:
            continue
        groups[name], gw = builders[name]()
        warm |= gw
    active = {g: w for g, w in weights.items() if g in groups}
    fdm = fdm_series(groups, active, universe_mask, cfg)
    combined = combine_forecasts(groups, active, fdm, cfg.forecast_cap)
    return SignalPanels(groups=groups, combined=combined, fdm=fdm, warmup=warm)
