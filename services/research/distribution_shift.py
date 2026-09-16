"""
Distribution Shift Detector — measures the reality gap between backtest and live returns.

Compares live/paper daily returns with the backtest's daily returns using:

1. Wasserstein-1 distance (``scipy.stats.wasserstein_distance``) on decimal
   daily returns.
2. KL divergence KL(live || backtest) from histograms (``scipy.special.rel_entr``).
   Bin edges are quantiles of the backtest returns and the bin count grows
   with the live sample (sqrt(n), clipped to [5, 50]); Jeffreys smoothing
   (+0.5 per bin) avoids log(0).  Fixed wide bins make the estimate mostly
   a function of sample size — with 50 bins and 30 live days KL exceeds 0.3
   even when nothing has changed.
3. Sinkhorn divergence (``geomloss``) on returns standardised by the backtest
   standard deviation; skipped when geomloss/torch are not installed.

Two verdicts are returned:

* ``verdict`` — the fixed thresholds: "stable" if Wasserstein < 0.05 and
  KL < 0.3; "drifting" if Wasserstein is in [0.05, 0.15] or KL in [0.3, 1.0];
  "regime_break" above that.  On decimal daily returns (daily sd ~0.5-1.5%)
  a Wasserstein distance of 0.05 is a 5%-per-day gap, so in practice this
  verdict is driven by KL and only flags large shape changes.
* ``calibrated_verdict`` — bootstrap p-values: the backtest is resampled at
  the live sample size to get the null distribution of each statistic.  With
  a Bonferroni correction over the two statistics, min p <= 0.05/2 is
  "drifting" and min p <= 0.01/2 is "regime_break".  This detects doubled
  volatility or a sustained -20 bp/day change in mean return within 30-120
  live days while keeping false alarms near the nominal rate.

``detect_distribution_shift_rolling`` applies the same tests to trailing
windows (default 60 days) and ``find_drift_onset`` reports when drift began.
``load_backtest_reference`` / ``compare_live_to_backtest`` pick the backtest
returns that correspond to the live period.
"""
from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.special import rel_entr
from scipy.stats import wasserstein_distance

logger = logging.getLogger(__name__)

# ── Thresholds (decimal daily returns) ─────────────────────────────────
WASSERSTEIN_THRESHOLDS = (0.05, 0.15)   # (drifting from, regime_break above)
KL_THRESHOLDS = (0.3, 1.0)
P_VALUE_THRESHOLDS = (0.05, 0.01)       # family-wise (drifting, regime_break)
MIN_SAMPLES = 30
DEFAULT_WINDOW = 60

_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REFERENCE_CSV = _ROOT / "data" / "shift_reference_returns.csv"
DEFAULT_RUNS_DIR = _ROOT / "data" / "nse_engine" / "runs"
_LEGACY_PICKLES = ("r21a_optimization_results.pkl", "r21a_oos_evaluation.pkl")
_SEVERITY = {"insufficient_data": -1, "stable": 0, "drifting": 1, "regime_break": 2}

ArrayLike = Union[np.ndarray, Sequence[float], pd.Series]


# ── Statistics ──────────────────────────────────────────────────────────

def _clean(x: ArrayLike) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64).ravel()
    return a[np.isfinite(a)]


def kl_bins_for(n_live: int) -> int:
    """Histogram bin count used for a live sample of size ``n_live``."""
    return int(np.clip(int(np.sqrt(max(n_live, 1))), 5, 50))


def _quantile_edges(backtest: np.ndarray, n_bins: int) -> np.ndarray:
    edges = np.unique(np.quantile(backtest, np.linspace(0.0, 1.0, n_bins + 1)))
    if len(edges) < 3:  # degenerate backtest (e.g. constant returns)
        lo, hi = float(backtest.min()), float(backtest.max())
        edges = np.linspace(lo - 1e-9, hi + 1e-9, n_bins + 1)
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def histogram_kl(backtest: np.ndarray, live: np.ndarray, n_bins: Optional[int] = None) -> float:
    """KL(live || backtest) over backtest-quantile bins with Jeffreys smoothing."""
    n_bins = n_bins or kl_bins_for(len(live))
    edges = _quantile_edges(backtest, n_bins)
    return _kl_with_edges(backtest, live, edges)


def _kl_with_edges(backtest: np.ndarray, live: np.ndarray, edges: np.ndarray) -> float:
    k = len(edges) - 1
    bt_hist, _ = np.histogram(backtest, bins=edges)
    lv_hist, _ = np.histogram(live, bins=edges)
    p_bt = (bt_hist + 0.5) / (bt_hist.sum() + 0.5 * k)
    q_lv = (lv_hist + 0.5) / (lv_hist.sum() + 0.5 * k)
    return float(np.sum(rel_entr(q_lv, p_bt)))


def sinkhorn_divergence(backtest: np.ndarray, live: np.ndarray, blur: float = 0.05) -> Tuple[Optional[float], str]:
    """Debiased Sinkhorn divergence (geomloss) on backtest-standardised returns.

    Returns ``(value, status)``; value is None and status explains why when
    geomloss or torch is unavailable or the computation fails.
    """
    try:
        import torch
        from geomloss import SamplesLoss
    except ImportError:
        return None, "skipped: geomloss not installed"
    try:
        scale = float(np.std(backtest)) or 1.0
        loss = SamplesLoss(loss="sinkhorn", p=2, blur=blur)
        bt = torch.tensor((backtest / scale).reshape(-1, 1), dtype=torch.float32)
        lv = torch.tensor((live / scale).reshape(-1, 1), dtype=torch.float32)
        return float(loss(bt, lv).item()), "ok"
    except Exception as exc:  # pragma: no cover - depends on optional deps
        logger.debug("Sinkhorn computation failed: %s", exc)
        return None, f"skipped: {type(exc).__name__}"


def classify(wasserstein: float, kl: float,
             wasserstein_thresholds: Tuple[float, float] = WASSERSTEIN_THRESHOLDS,
             kl_thresholds: Tuple[float, float] = KL_THRESHOLDS) -> str:
    """Fixed-threshold verdict."""
    w_lo, w_hi = wasserstein_thresholds
    k_lo, k_hi = kl_thresholds
    if wasserstein > w_hi or kl > k_hi:
        return "regime_break"
    if wasserstein >= w_lo or kl >= k_lo:
        return "drifting"
    return "stable"


def classify_p_values(p_wasserstein: float, p_kl: float,
                      p_thresholds: Tuple[float, float] = P_VALUE_THRESHOLDS) -> str:
    """Calibrated verdict from bootstrap p-values (Bonferroni over 2 tests)."""
    p_min = min(p_wasserstein, p_kl)
    drift, brk = p_thresholds
    if p_min <= brk / 2:
        return "regime_break"
    if p_min <= drift / 2:
        return "drifting"
    return "stable"


def more_severe(a: str, b: Optional[str]) -> str:
    if b is None:
        return a
    return a if _SEVERITY.get(a, 0) >= _SEVERITY.get(b, 0) else b


class _NullDistribution:
    """Bootstrap null of (Wasserstein, KL) for live samples of size n."""

    def __init__(self, backtest: np.ndarray, n: int, n_bootstrap: int, rng: np.random.Generator):
        self.edges = _quantile_edges(backtest, kl_bins_for(n))
        idx = rng.integers(0, len(backtest), size=(n_bootstrap, n))
        samples = backtest[idx]
        self.wass = np.array([wasserstein_distance(backtest, s) for s in samples])
        self.kl = np.array([_kl_with_edges(backtest, s, self.edges) for s in samples])

    def p_values(self, wass: float, kl: float) -> Tuple[float, float]:
        b = len(self.wass)
        return (float((1 + np.sum(self.wass >= wass)) / (b + 1)),
                float((1 + np.sum(self.kl >= kl)) / (b + 1)))


def detect_distribution_shift(
    backtest_returns: ArrayLike,
    live_returns: ArrayLike,
    *,
    n_bootstrap: int = 500,
    random_state: int = 0,
    wasserstein_thresholds: Tuple[float, float] = WASSERSTEIN_THRESHOLDS,
    kl_thresholds: Tuple[float, float] = KL_THRESHOLDS,
) -> Dict:
    """Compute distribution shift between backtest and live daily returns.

    Parameters
    ----------
    backtest_returns, live_returns : decimal daily returns (NaN/inf dropped).
    n_bootstrap : resamples for calibrated p-values (0 disables them).
    random_state : seed for the bootstrap.

    Returns
    -------
    dict with wasserstein, kl_divergence, sinkhorn, sinkhorn_status, verdict,
    calibrated_verdict, p_value_wasserstein, p_value_kl, kl_bins,
    n_backtest, n_live, live_mean, backtest_mean, live_std, backtest_std.
    """
    bt = _clean(backtest_returns)
    lv = _clean(live_returns)
    base = {"n_backtest": int(len(bt)), "n_live": int(len(lv))}
    if len(bt) < MIN_SAMPLES or len(lv) < MIN_SAMPLES:
        return {**base, "wasserstein": None, "kl_divergence": None, "sinkhorn": None,
                "sinkhorn_status": "skipped: insufficient data", "verdict": "insufficient_data",
                "calibrated_verdict": None, "p_value_wasserstein": None, "p_value_kl": None,
                "kl_bins": None}

    n_bins = kl_bins_for(len(lv))
    wass = float(wasserstein_distance(bt, lv))
    kl = histogram_kl(bt, lv, n_bins)
    sinkhorn, sinkhorn_status = sinkhorn_divergence(bt, lv)

    p_w = p_k = None
    calibrated = None
    if n_bootstrap > 0:
        null = _NullDistribution(bt, len(lv), n_bootstrap, np.random.default_rng(random_state))
        p_w, p_k = null.p_values(wass, kl)
        calibrated = classify_p_values(p_w, p_k)

    return {
        **base,
        "wasserstein": round(wass, 6),
        "kl_divergence": round(kl, 6),
        "sinkhorn": round(sinkhorn, 6) if sinkhorn is not None else None,
        "sinkhorn_status": sinkhorn_status,
        "verdict": classify(wass, kl, wasserstein_thresholds, kl_thresholds),
        "calibrated_verdict": calibrated,
        "p_value_wasserstein": round(p_w, 4) if p_w is not None else None,
        "p_value_kl": round(p_k, 4) if p_k is not None else None,
        "kl_bins": n_bins,
        "live_mean": float(lv.mean()), "backtest_mean": float(bt.mean()),
        "live_std": float(lv.std(ddof=1)), "backtest_std": float(bt.std(ddof=1)),
    }


def detect_distribution_shift_rolling(
    backtest_returns: ArrayLike,
    live_returns: ArrayLike,
    window: int = DEFAULT_WINDOW,
    step: int = 1,
    *,
    dates: Optional[Sequence] = None,
    n_bootstrap: int = 300,
    random_state: int = 0,
) -> List[Dict]:
    """Trailing-window shift detection over chronologically ordered live returns.

    Each window of ``window`` live days is compared with the full backtest
    distribution.  ``dates`` (same length as ``live_returns``, or taken from a
    Series index) labels each window so drift onset can be dated.
    """
    if dates is None and isinstance(live_returns, pd.Series):
        dates = live_returns.index
    lv_raw = np.asarray(live_returns, dtype=np.float64).ravel()
    keep = np.isfinite(lv_raw)
    lv = lv_raw[keep]
    lv_dates = None if dates is None else pd.DatetimeIndex(pd.to_datetime(list(dates)))[keep]
    bt = _clean(backtest_returns)
    if len(lv) < window or len(bt) < MIN_SAMPLES:
        return []

    null = (_NullDistribution(bt, window, n_bootstrap, np.random.default_rng(random_state))
            if n_bootstrap > 0 else None)
    edges = null.edges if null is not None else _quantile_edges(bt, kl_bins_for(window))

    results = []
    for start in range(0, len(lv) - window + 1, step):
        end = start + window
        seg = lv[start:end]
        wass = float(wasserstein_distance(bt, seg))
        kl = _kl_with_edges(bt, seg, edges)
        row = {
            "window_start": start,
            "window_end": end,
            "wasserstein": round(wass, 6),
            "kl_divergence": round(kl, 6),
            "verdict": classify(wass, kl),
        }
        if lv_dates is not None:
            row["start_date"] = lv_dates[start].date().isoformat()
            row["end_date"] = lv_dates[end - 1].date().isoformat()
        if null is not None:
            p_w, p_k = null.p_values(wass, kl)
            row.update(p_value_wasserstein=round(p_w, 4), p_value_kl=round(p_k, 4),
                       calibrated_verdict=classify_p_values(p_w, p_k))
        results.append(row)
    return results


def find_drift_onset(rolling: List[Dict], persist: int = 3,
                     key: str = "effective") -> Optional[Dict]:
    """First window of the first run of ``persist`` consecutive non-stable windows.

    ``key`` selects the verdict: "verdict", "calibrated_verdict" or
    "effective" (the more severe of the two).
    """
    def verdict_of(row):
        if key == "effective":
            return more_severe(row["verdict"], row.get("calibrated_verdict"))
        return row.get(key) or row["verdict"]

    run = 0
    for i, row in enumerate(rolling):
        run = run + 1 if verdict_of(row) != "stable" else 0
        if run >= persist:
            first = rolling[i - persist + 1]
            return {"window_start": first["window_start"], "start_date": first.get("start_date"),
                    "end_date": first.get("end_date"), "verdict": verdict_of(first),
                    "persisted_windows": persist}
    return None


# ── Reference selection ─────────────────────────────────────────────────

def _read_returns_csv(path: Path) -> Optional[pd.Series]:
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        logger.warning("Could not read reference returns %s: %s", path, exc)
        return None
    date_col = "date" if "date" in df.columns else df.columns[0]
    ret_col = "return" if "return" in df.columns else df.columns[-1]
    s = pd.Series(df[ret_col].to_numpy(dtype="float64"),
                  index=pd.DatetimeIndex(pd.to_datetime(df[date_col])))
    return s[~s.index.duplicated(keep="last")].sort_index().dropna()


def _legacy_pickle_returns(data_dir: Path) -> Optional[np.ndarray]:
    for name in _LEGACY_PICKLES:
        path = data_dir / name
        if not path.exists():
            continue
        with open(path, "rb") as f:
            data = pickle.load(f)
        for key in ("best_test", "best_full", "r21a_test", "r21a_full"):
            res = data.get(key, {}) if isinstance(data, dict) else {}
            if isinstance(res, dict) and "daily_returns" in res:
                return np.asarray(res["daily_returns"], dtype=np.float64)
    return None


def load_backtest_reference(
    live_dates: Sequence,
    *,
    reference_csv: Optional[Union[str, Path]] = None,
    runs_dir: Optional[Union[str, Path]] = None,
    run_id: Optional[str] = None,
    trailing_days: int = 504,
    min_overlap: int = MIN_SAMPLES,
    data_dir: Optional[Union[str, Path]] = None,
) -> Tuple[Optional[pd.Series], str, str]:
    """Pick backtest daily returns to compare with live returns.

    Sources, in order: ``reference_csv`` (or env CENTURION_SHIFT_REFERENCE_CSV,
    default data/shift_reference_returns.csv — e.g. a backtest re-run over
    the live dates via ``runners/run_nse_engine.py shift-reference``); an NSE
    engine run (``run_id`` or env CENTURION_SHIFT_REFERENCE_RUN) under
    ``runs_dir``; legacy optimizer pickles.

    Returns ``(returns, mode, source)``.  ``mode`` is "same_period" when the
    reference covers at least ``min_overlap`` live dates (returns restricted
    to those dates), "trailing_history" (last ``trailing_days`` backtest days)
    otherwise, "legacy_undated" for pickles without dates, or "unavailable".
    """
    live_idx = pd.DatetimeIndex(pd.to_datetime(list(live_dates))).normalize()
    candidates: List[Tuple[str, Optional[pd.Series]]] = []

    csv_path = Path(reference_csv or os.environ.get("CENTURION_SHIFT_REFERENCE_CSV") or DEFAULT_REFERENCE_CSV)
    if csv_path.exists():
        candidates.append((str(csv_path), _read_returns_csv(csv_path)))

    run_id = run_id or os.environ.get("CENTURION_SHIFT_REFERENCE_RUN")
    if run_id:
        run_path = Path(runs_dir or DEFAULT_RUNS_DIR) / run_id / "returns.csv"
        if run_path.exists():
            candidates.append((f"nse_engine run {run_id}", _read_returns_csv(run_path)))
        else:
            logger.warning("Shift reference run %s not found at %s", run_id, run_path)

    for source, series in candidates:
        if series is None or len(series) < MIN_SAMPLES:
            continue
        overlap = series.index.normalize().intersection(live_idx)
        if len(overlap) >= min_overlap:
            ref = series[series.index.normalize().isin(overlap)]
            return ref, "same_period", source
    for source, series in candidates:
        if series is not None and len(series) >= MIN_SAMPLES:
            return series.iloc[-trailing_days:], "trailing_history", source

    legacy = _legacy_pickle_returns(Path(data_dir or _ROOT / "data"))
    if legacy is not None and len(legacy) >= MIN_SAMPLES:
        return pd.Series(legacy), "legacy_undated", "legacy optimizer pickle"
    return None, "unavailable", ""


def compare_live_to_backtest(
    live_returns: pd.Series,
    *,
    window: int = DEFAULT_WINDOW,
    rolling_step: int = 1,
    n_bootstrap: int = 500,
    **reference_kwargs,
) -> Dict:
    """Full reality-gap report for dated live returns.

    When the reference covers the live dates (``same_period``) both series
    are restricted to the common dates, so the comparison is day-for-day
    against the backtest's returns for the same period.
    """
    live = live_returns.dropna().sort_index()
    live.index = pd.DatetimeIndex(live.index).normalize()
    ref, mode, source = load_backtest_reference(live.index, **reference_kwargs)
    report: Dict = {"reference_mode": mode, "reference_source": source,
                    "live_start": live.index[0].date().isoformat() if len(live) else None,
                    "live_end": live.index[-1].date().isoformat() if len(live) else None}
    if ref is None:
        report.update(verdict="insufficient_data", calibrated_verdict=None, n_live=int(len(live)),
                      n_backtest=0, note="no backtest reference returns available")
        return report
    if mode == "same_period":
        ref = ref.copy()
        ref.index = pd.DatetimeIndex(ref.index).normalize()
        ref = ref[~ref.index.duplicated(keep="last")]
        aligned = pd.concat({"live": live, "ref": ref}, axis=1, join="inner").dropna()
        live, ref = aligned["live"], aligned["ref"]
        report["mean_daily_gap"] = float((live - ref).mean())
        report["tracking_error_annual"] = (float((live - ref).std(ddof=1) * np.sqrt(252))
                                           if len(aligned) > 1 else None)

    report.update(detect_distribution_shift(ref.to_numpy(), live.to_numpy(), n_bootstrap=n_bootstrap))
    report["effective_verdict"] = more_severe(report["verdict"], report.get("calibrated_verdict"))
    if len(live) >= window:
        rolling = detect_distribution_shift_rolling(ref.to_numpy(), live, window=window, step=rolling_step,
                                                    n_bootstrap=min(n_bootstrap, 300))
        report["rolling"] = rolling
        report["drift_onset"] = find_drift_onset(rolling)
    return report
