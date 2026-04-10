#!/usr/bin/env python3
"""Hanalei road closure predictor (v3: multi-rain-gauge).

Predicts the probability that the Hanalei River gauge (USGS 16103000) will
exceed the 5.0 ft road-closure threshold at any point in the next 1-3 hours,
using in-watershed USGS precipitation gauges (Mt Waialeale + 5 others) as the
main signal rather than gridded ERA5 reanalysis.

Subcommands:
  train      Fetch data, build features, train model, save artifacts.
  predict    Load model, fetch recent data, emit JSON prediction.
  backtest   Load model + cached data, evaluate on a custom date range.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

# --- constants -------------------------------------------------------------

USGS_IV_URL = "https://waterservices.usgs.gov/nwis/iv/"
STREAM_SITE = "16103000"
PARAM_GAUGE = "00065"   # gauge height, ft
PARAM_Q = "00060"       # discharge, cfs
PARAM_RAIN = "00045"    # precipitation, inches (per-interval totals)

# NOAA CO-OPS tide station (closest to Hanalei on Kauai).
TIDE_STATION = "1611400"  # Nawiliwili Harbor
NOAA_COOPS_URL = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"

# In-watershed and nearby USGS rain gauges (all have 15-min IV data through present).
RAIN_GAUGES: list[tuple[str, str]] = [
    ("waiakoali",   "220739159373001"),
    ("mohihi_crsg", "220713159361201"),
    ("waialae",     "220523159341201"),
    ("kilohana",    "220927159355001"),
    ("waialeale",   "220427159300201"),  # Mt Waialeale — highest annual rainfall
    ("n_wailua",    "220356159281401"),
]

# Earliest date for which BOTH the stream gauge and all rain gauges have data.
DATA_START = datetime(2007, 10, 1, tzinfo=timezone.utc)

CLOSURE_FT = 5.0
HORIZON_H = 3

# Rolling rain windows — includes long antecedent windows for soil-saturation proxy.
RAIN_WINDOWS_PER_GAUGE = (1, 3, 6, 24)
RAIN_WINDOWS_XGAUGE = (1, 3, 6, 12, 24, 48, 72, 168)  # 168h = 7 days


def _rain_col(name: str) -> str:
    return f"rain_{name}"


def build_feature_columns() -> list[str]:
    cols = [
        # --- stream gauge ---
        "gauge_ft",
        "gauge_lag_1", "gauge_lag_2", "gauge_lag_3",
        "gauge_lag_6", "gauge_lag_12", "gauge_lag_24",
        "gauge_delta_1", "gauge_delta_3", "gauge_delta_6",
        # --- discharge ---
        "q_cfs", "q_lag_1", "q_lag_3", "q_lag_6",
        "q_delta_1", "q_delta_3",
    ]
    # --- per-gauge rain rolling sums ---
    for name, _ in RAIN_GAUGES:
        for w in RAIN_WINDOWS_PER_GAUGE:
            cols.append(f"{_rain_col(name)}_sum_{w}")
    # --- cross-gauge aggregates (spatial max + total, including long antecedent) ---
    for w in RAIN_WINDOWS_XGAUGE:
        cols.append(f"rain_max_sum_{w}")
        cols.append(f"rain_total_sum_{w}")
    # --- tide ---
    cols += [
        "tide_ft",               # observed water level (MLLW)
        "tide_pred_ft",          # harmonic prediction (astronomical tide)
        "storm_surge_ft",        # observed - predicted (weather-driven anomaly)
        "tide_delta_3",          # tide change over 3h (rising vs falling)
    ]
    # --- time / indicator ---
    cols += [
        "hour_sin", "hour_cos", "month_sin", "month_cos",
        "above_3ft", "above_4ft",
    ]
    return cols


FEATURE_COLUMNS = build_feature_columns()


# --- data fetch ------------------------------------------------------------

def _fetch_usgs_iv(
    site: str,
    parm_cd: str,
    col_name: str,
    start: datetime,
    end: datetime,
    label: str,
    chunk_days: int = 120,
) -> pd.DataFrame:
    """Generic USGS IV fetcher in chunks. Returns DataFrame indexed by UTC ts."""
    frames: list[pd.DataFrame] = []
    chunk = timedelta(days=chunk_days)
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + chunk, end)
        print(f"  {label:<14} {cursor.date()} -> {chunk_end.date()}", file=sys.stderr)
        params = {
            "sites": site,
            "parameterCd": parm_cd,
            "startDT": cursor.strftime("%Y-%m-%d"),
            "endDT": chunk_end.strftime("%Y-%m-%d"),
            "format": "json",
            "siteStatus": "all",
        }
        r = requests.get(USGS_IV_URL, params=params, timeout=180)
        r.raise_for_status()
        series = r.json().get("value", {}).get("timeSeries", [])
        if series:
            values = series[0].get("values", [])
            if values and values[0].get("value"):
                df = pd.DataFrame(values[0]["value"])
                df["dateTime"] = pd.to_datetime(df["dateTime"], utc=True, format="ISO8601")
                df["value"] = pd.to_numeric(df["value"], errors="coerce")
                df.loc[df["value"] < -100, "value"] = np.nan
                df = df.rename(columns={"dateTime": "ts", "value": col_name})[["ts", col_name]]
                frames.append(df)
        cursor = chunk_end + timedelta(days=1)
        time.sleep(0.2)

    if not frames:
        return pd.DataFrame(columns=[col_name]).set_index(
            pd.DatetimeIndex([], tz="UTC", name="ts")
        )
    return (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates("ts")
        .sort_values("ts")
        .set_index("ts")
    )


def fetch_gauge(start: datetime, end: datetime) -> pd.DataFrame:
    return _fetch_usgs_iv(STREAM_SITE, PARAM_GAUGE, "gauge_ft", start, end, "gauge")


def fetch_discharge(start: datetime, end: datetime) -> pd.DataFrame:
    return _fetch_usgs_iv(STREAM_SITE, PARAM_Q, "q_cfs", start, end, "discharge")


def fetch_rain_gauge(name: str, site: str, start: datetime, end: datetime) -> pd.DataFrame:
    return _fetch_usgs_iv(site, PARAM_RAIN, _rain_col(name), start, end, f"rain/{name}")


# --- NOAA CO-OPS tide data --------------------------------------------------

def _fetch_noaa_tide(
    start: datetime, end: datetime, product: str, col_name: str, label: str
) -> pd.DataFrame:
    """Fetch NOAA CO-OPS data in 30-day chunks (API limit ~31 days for 6-min data)."""
    frames: list[pd.DataFrame] = []
    chunk = timedelta(days=30)
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + chunk, end)
        print(f"  {label:<14} {cursor.date()} -> {chunk_end.date()}", file=sys.stderr)
        params = {
            "station": TIDE_STATION,
            "begin_date": cursor.strftime("%Y%m%d"),
            "end_date": chunk_end.strftime("%Y%m%d"),
            "product": product,
            "datum": "MLLW",
            "units": "english",
            "time_zone": "gmt",
            "format": "json",
            "application": "hanalei_research",
        }
        if product == "predictions":
            params["interval"] = "6"  # 6-minute predictions to match observed cadence
        r = requests.get(NOAA_COOPS_URL, params=params, timeout=120)
        r.raise_for_status()
        j = r.json()
        key = "predictions" if product == "predictions" else "data"
        records = j.get(key, [])
        if records:
            df = pd.DataFrame(records)
            df["ts"] = pd.to_datetime(df["t"], utc=True)
            df[col_name] = pd.to_numeric(df["v"], errors="coerce")
            frames.append(df[["ts", col_name]])
        cursor = chunk_end + timedelta(days=1)
        time.sleep(0.15)

    if not frames:
        return pd.DataFrame(columns=[col_name]).set_index(
            pd.DatetimeIndex([], tz="UTC", name="ts")
        )
    return (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates("ts")
        .sort_values("ts")
        .set_index("ts")
    )


def fetch_tide_observed(start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch observed water level at Nawiliwili (ft above MLLW)."""
    return _fetch_noaa_tide(start, end, "water_level", "tide_ft", "tide/obs")


def fetch_tide_predicted(start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch harmonic tide prediction at Nawiliwili (ft above MLLW)."""
    return _fetch_noaa_tide(start, end, "predictions", "tide_pred_ft", "tide/pred")


def fetch_all_rain(start: datetime, end: datetime) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for name, site in RAIN_GAUGES:
        out[name] = fetch_rain_gauge(name, site, start, end)
        print(f"  fetched {len(out[name]):,} rows from {name}", file=sys.stderr)
    return out


def fetch_gauge_recent(hours: int = 200) -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    return fetch_gauge(now - timedelta(hours=hours), now + timedelta(days=1))


def fetch_discharge_recent(hours: int = 200) -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    return fetch_discharge(now - timedelta(hours=hours), now + timedelta(days=1))


def fetch_all_rain_recent(hours: int = 200) -> dict[str, pd.DataFrame]:
    now = datetime.now(timezone.utc)
    return fetch_all_rain(now - timedelta(hours=hours), now + timedelta(days=1))


def fetch_tide_recent(hours: int = 200) -> tuple[pd.DataFrame, pd.DataFrame]:
    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=hours)
    end = now + timedelta(days=1)
    return fetch_tide_observed(start, end), fetch_tide_predicted(start, end)


# --- merging / resampling --------------------------------------------------

def to_hourly(
    gauge_raw: pd.DataFrame,
    rain_raw: dict[str, pd.DataFrame],
    q_raw: pd.DataFrame | None = None,
    tide_obs: pd.DataFrame | None = None,
    tide_pred: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Resample all sources to aligned hourly records.

    - gauge_ft: hourly MAX of 15-min gauge height
    - q_cfs:    hourly MAX of 15-min discharge
    - rain:     hourly SUM of 15-min interval totals (inches)
    - tide_ft:  hourly MEAN of 6-min observed water level
    - tide_pred_ft: hourly MEAN of 6-min harmonic prediction
    """
    gauge_hourly = gauge_raw["gauge_ft"].resample("1h").max()
    frames: list[pd.Series] = [gauge_hourly]

    # Discharge
    if q_raw is not None and not q_raw.empty:
        frames.append(q_raw["q_cfs"].resample("1h").max())

    # Rain gauges
    for name, _ in RAIN_GAUGES:
        col = _rain_col(name)
        if name in rain_raw and not rain_raw[name].empty:
            hourly = rain_raw[name][col].resample("1h").sum(min_count=1)
        else:
            hourly = pd.Series(dtype=float, name=col)
        frames.append(hourly.rename(col))

    # Tide
    if tide_obs is not None and not tide_obs.empty:
        frames.append(tide_obs["tide_ft"].resample("1h").mean())
    if tide_pred is not None and not tide_pred.empty:
        frames.append(tide_pred["tide_pred_ft"].resample("1h").mean())

    df = pd.concat(frames, axis=1, sort=True)

    # Gentle gap filling
    df["gauge_ft"] = df["gauge_ft"].interpolate(method="time", limit=2)
    if "q_cfs" in df.columns:
        df["q_cfs"] = df["q_cfs"].interpolate(method="time", limit=2)
    for name, _ in RAIN_GAUGES:
        df[_rain_col(name)] = df[_rain_col(name)].fillna(0.0)
    if "tide_ft" in df.columns:
        df["tide_ft"] = df["tide_ft"].interpolate(method="time", limit=6)
    if "tide_pred_ft" in df.columns:
        df["tide_pred_ft"] = df["tide_pred_ft"].interpolate(method="time", limit=6)

    df = df.dropna(subset=["gauge_ft"])
    return df


# --- features and target ---------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features to an hourly df."""
    out = df.copy()
    g = out["gauge_ft"]

    # --- gauge features ---
    for lag in (1, 2, 3, 6, 12, 24):
        out[f"gauge_lag_{lag}"] = g.shift(lag)
    for d in (1, 3, 6):
        out[f"gauge_delta_{d}"] = g - g.shift(d)

    # --- discharge features ---
    if "q_cfs" in out.columns:
        q = out["q_cfs"]
        for lag in (1, 3, 6):
            out[f"q_lag_{lag}"] = q.shift(lag)
        for d in (1, 3):
            out[f"q_delta_{d}"] = q - q.shift(d)
    else:
        for col in ["q_cfs", "q_lag_1", "q_lag_3", "q_lag_6", "q_delta_1", "q_delta_3"]:
            out[col] = np.nan

    # --- per-gauge rain rolling sums ---
    for name, _ in RAIN_GAUGES:
        col = _rain_col(name)
        for w in RAIN_WINDOWS_PER_GAUGE:
            out[f"{col}_sum_{w}"] = out[col].rolling(w, min_periods=1).sum()

    # --- cross-gauge aggregates (including long antecedent windows) ---
    for w in RAIN_WINDOWS_XGAUGE:
        per_gauge_sums: list[pd.Series] = []
        for name, _ in RAIN_GAUGES:
            col = _rain_col(name)
            per_gauge_sums.append(out[col].rolling(w, min_periods=1).sum())
        stacked = pd.concat(per_gauge_sums, axis=1)
        out[f"rain_max_sum_{w}"] = stacked.max(axis=1)
        out[f"rain_total_sum_{w}"] = stacked.sum(axis=1)

    # --- tide features ---
    if "tide_ft" in out.columns:
        out["storm_surge_ft"] = out.get("tide_ft", 0) - out.get("tide_pred_ft", 0)
        out["tide_delta_3"] = out["tide_ft"] - out["tide_ft"].shift(3)
    else:
        for col in ["tide_ft", "tide_pred_ft", "storm_surge_ft", "tide_delta_3"]:
            out[col] = np.nan

    # --- time / indicator ---
    hour = out.index.hour
    month = out.index.month
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    out["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)

    out["above_3ft"] = (g > 3.0).astype(int)
    out["above_4ft"] = (g > 4.0).astype(int)
    return out


def build_target(df: pd.DataFrame) -> pd.Series:
    """closure_in_next_H[t] = 1 iff max(gauge[t+1..t+H]) > CLOSURE_FT."""
    g = df["gauge_ft"]
    shifts = pd.concat([g.shift(-i) for i in range(1, HORIZON_H + 1)], axis=1)
    future_max = shifts.max(axis=1, skipna=True)
    y = pd.Series(pd.NA, index=df.index, dtype="Int8")
    mask = future_max.notna()
    y[mask] = (future_max[mask] > CLOSURE_FT).astype(int)
    return y


# --- training --------------------------------------------------------------

@dataclass
class TrainBundle:
    model: HistGradientBoostingClassifier
    threshold: float
    features: list
    closure_ft: float
    horizon_h: int
    rain_gauge_names: list


def date_split(df: pd.DataFrame, test_years: float, val_years: float):
    """Chronological: last test_years = test, prior val_years = val, rest = train."""
    end = df.index.max()
    test_start = end - pd.Timedelta(days=int(test_years * 365.25))
    val_start = test_start - pd.Timedelta(days=int(val_years * 365.25))
    train = df.loc[df.index < val_start]
    val = df.loc[(df.index >= val_start) & (df.index < test_start)]
    test = df.loc[df.index >= test_start]
    return train, val, test


def tune_threshold_f1(y_true: np.ndarray, y_prob: np.ndarray, min_precision: float = 0.35) -> float:
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    prec_t, rec_t = prec[:-1], rec[:-1]
    f1 = 2 * prec_t * rec_t / np.clip(prec_t + rec_t, 1e-9, None)
    eligible = prec_t >= min_precision
    if eligible.any():
        i = int(np.argmax(np.where(eligible, f1, -1)))
    else:
        i = int(np.argmax(f1))
    return float(thr[i])


def tune_threshold_event_recall(
    val_df: pd.DataFrame,
    y_prob: np.ndarray,
    min_precision: float = 0.15,
    n_candidates: int = 400,
) -> float:
    """Max validation event recall s.t. hour-level precision >= min_precision."""
    y = val_df["y"].values
    g = val_df["gauge_ft"].values
    above = g > CLOSURE_FT
    prior = np.concatenate([[False], above[:-1]])
    event_pos = np.where(above & ~prior)[0]
    if len(event_pos) == 0:
        return tune_threshold_f1(y, y_prob, min_precision=min_precision)

    candidates = np.unique(np.clip(
        np.linspace(0.005, 0.99, n_candidates),
        1e-4, 0.9999,
    ))

    best = None
    for t in candidates:
        alerts = y_prob >= t
        tp = int((alerts & (y == 1)).sum())
        fp = int((alerts & (y == 0)).sum())
        prec = tp / max(tp + fp, 1)
        if prec < min_precision:
            continue
        hits = 0
        for pos in event_pos:
            lo = max(0, pos - HORIZON_H)
            if alerts[lo:pos].any():
                hits += 1
        key = (hits, -int(alerts.sum()), -float(t))
        if best is None or key > best[0]:
            best = (key, float(t))

    if best is None:
        return tune_threshold_f1(y, y_prob, min_precision=min_precision)
    return best[1]


def event_metrics(test_df: pd.DataFrame, y_prob: np.ndarray, threshold: float) -> dict:
    g = test_df["gauge_ft"].values
    above = g > CLOSURE_FT
    prior = np.concatenate([[False], above[:-1]])
    event_positions = np.where(above & ~prior)[0]
    alerts = y_prob >= threshold

    hits = 0
    lead_times = []
    for pos in event_positions:
        lo = max(0, pos - HORIZON_H)
        window = alerts[lo:pos]
        if window.size and window.any():
            hits += 1
            first_alert_rel = int(np.argmax(window))
            lead_h = pos - (lo + first_alert_rel)
            lead_times.append(lead_h * 60)

    return {
        "n_events": int(len(event_positions)),
        "events_caught": int(hits),
        "event_recall": float(hits / len(event_positions)) if len(event_positions) else None,
        "mean_lead_time_min": float(np.mean(lead_times)) if lead_times else None,
    }


def fetch_all_training_data(end: datetime) -> pd.DataFrame:
    start = DATA_START
    print(f"Fetching stream gauge {start.date()} -> {end.date()}", file=sys.stderr)
    gauge_raw = fetch_gauge(start, end)
    print(f"  stream gauge rows: {len(gauge_raw):,}", file=sys.stderr)
    if gauge_raw.empty:
        raise RuntimeError("No stream gauge data.")

    print(f"Fetching discharge {start.date()} -> {end.date()}", file=sys.stderr)
    q_raw = fetch_discharge(start, end)
    print(f"  discharge rows: {len(q_raw):,}", file=sys.stderr)

    print(f"Fetching {len(RAIN_GAUGES)} rain gauges {start.date()} -> {end.date()}", file=sys.stderr)
    rain_raw = fetch_all_rain(start, end)

    print(f"Fetching tide data {start.date()} -> {end.date()}", file=sys.stderr)
    tide_obs = fetch_tide_observed(start, end)
    print(f"  tide observed rows: {len(tide_obs):,}", file=sys.stderr)
    tide_pred = fetch_tide_predicted(start, end)
    print(f"  tide predicted rows: {len(tide_pred):,}", file=sys.stderr)

    hourly = to_hourly(gauge_raw, rain_raw, q_raw=q_raw, tide_obs=tide_obs, tide_pred=tide_pred)
    print(f"Merged hourly rows: {len(hourly):,}", file=sys.stderr)
    return hourly


def train_cmd(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_parquet = out_dir / "hourly.parquet"
    cache_csv = out_dir / "hourly.csv"

    if args.use_cache and (cache_parquet.exists() or cache_csv.exists()):
        src = cache_parquet if cache_parquet.exists() else cache_csv
        print(f"Loading cached data from {src}", file=sys.stderr)
        if src.suffix == ".parquet":
            hourly = pd.read_parquet(src)
        else:
            hourly = pd.read_csv(src, index_col=0, parse_dates=True)
        hourly.index = pd.to_datetime(hourly.index, utc=True)
    else:
        end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        hourly = fetch_all_training_data(end)
        try:
            hourly.to_parquet(cache_parquet)
        except Exception as exc:
            print(f"Parquet cache failed ({exc}); writing CSV instead.", file=sys.stderr)
            hourly.to_csv(cache_csv)

    feats = build_features(hourly)
    feats["y"] = build_target(feats)
    feats = feats.dropna(subset=FEATURE_COLUMNS + ["y"])
    feats["y"] = feats["y"].astype(int)

    train_df, val_df, test_df = date_split(
        feats, test_years=args.test_years, val_years=args.val_years
    )
    print(
        f"split  train {train_df.index.min().date()}..{train_df.index.max().date()} "
        f"({len(train_df):,})  "
        f"val {val_df.index.min().date()}..{val_df.index.max().date()} "
        f"({len(val_df):,})  "
        f"test {test_df.index.min().date()}..{test_df.index.max().date()} "
        f"({len(test_df):,})",
        file=sys.stderr,
    )
    X_train = train_df[FEATURE_COLUMNS].values
    y_train = train_df["y"].values
    X_val = val_df[FEATURE_COLUMNS].values
    y_val = val_df["y"].values
    X_test = test_df[FEATURE_COLUMNS].values
    y_test = test_df["y"].values

    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    print(f"train rows {len(y_train):,}  positives {pos}  neg/pos ratio {neg / max(pos, 1):.1f}", file=sys.stderr)
    if pos == 0:
        raise RuntimeError("No positive examples in training split.")
    sw = np.where(y_train == 1, neg / pos, 1.0)

    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=600,
        max_leaf_nodes=63,
        min_samples_leaf=40,
        l2_regularization=0.5,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=25,
        random_state=0,
    )
    model.fit(X_train, y_train, sample_weight=sw)

    val_prob = model.predict_proba(X_val)[:, 1]
    if y_val.sum() == 0:
        print("Validation split has no positives; threshold=0.5", file=sys.stderr)
        threshold = 0.5
    elif args.objective == "event_recall":
        threshold = tune_threshold_event_recall(
            val_df, val_prob, min_precision=args.min_precision
        )
    else:
        threshold = tune_threshold_f1(y_val, val_prob, min_precision=args.min_precision)
    print(f"tuned threshold = {threshold:.4f}  (objective={args.objective})", file=sys.stderr)

    test_prob = model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= threshold).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, test_prob)) if y_test.sum() > 0 else None,
        "pr_auc": float(average_precision_score(y_test, test_prob)) if y_test.sum() > 0 else None,
        "threshold": threshold,
        "confusion_matrix": confusion_matrix(y_test, test_pred).tolist(),
        "test_positive_rate": float(y_test.mean()),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "train_positive_rate": float(y_train.mean()),
        "val_positive_rate": float(y_val.mean()),
    }
    metrics.update({f"event_{k}": v for k, v in event_metrics(test_df, test_prob, threshold).items()})
    metrics["horizon_h"] = HORIZON_H
    metrics["closure_ft"] = CLOSURE_FT
    metrics["rain_gauges"] = [name for name, _ in RAIN_GAUGES]
    metrics["n_features"] = len(FEATURE_COLUMNS)
    metrics["train_start"] = train_df.index.min().isoformat()
    metrics["train_end"] = train_df.index.max().isoformat()
    metrics["val_start"] = val_df.index.min().isoformat() if len(val_df) else None
    metrics["val_end"] = val_df.index.max().isoformat() if len(val_df) else None
    metrics["test_start"] = test_df.index.min().isoformat()
    metrics["test_end"] = test_df.index.max().isoformat()

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(json.dumps(metrics, indent=2))

    # Feature importance via permutation on a small val subsample would be ideal
    # but HistGBT has no built-in. Store the feature list only.
    bundle = TrainBundle(
        model=model,
        threshold=threshold,
        features=list(FEATURE_COLUMNS),
        closure_ft=CLOSURE_FT,
        horizon_h=HORIZON_H,
        rain_gauge_names=[name for name, _ in RAIN_GAUGES],
    )
    joblib.dump(bundle, out_dir / "model.joblib")

    # --- plots ---
    if y_test.sum() > 0:
        fpr, tpr, _ = roc_curve(y_test, test_prob)
        plt.figure(figsize=(5, 5))
        plt.plot(fpr, tpr, label=f"AUC={metrics['roc_auc']:.3f}")
        plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC (test)"); plt.legend()
        plt.tight_layout(); plt.savefig(out_dir / "roc.png", dpi=120); plt.close()

        prec, rec, _ = precision_recall_curve(y_test, test_prob)
        plt.figure(figsize=(5, 5))
        plt.plot(rec, prec, label=f"AP={metrics['pr_auc']:.3f}")
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR (test)"); plt.legend()
        plt.tight_layout(); plt.savefig(out_dir / "pr.png", dpi=120); plt.close()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(test_df.index, test_df["gauge_ft"], lw=0.8, label="gauge_ft")
    ax.axhline(CLOSURE_FT, color="red", ls="--", alpha=0.5, label="5 ft closure")
    alert_idx = test_df.index[test_pred.astype(bool)]
    if len(alert_idx):
        ax.scatter(alert_idx, np.full(len(alert_idx), CLOSURE_FT + 0.1),
                   s=6, color="orange", label="alert")
    ax.set_title("Test window"); ax.legend(loc="upper right")
    plt.tight_layout(); plt.savefig(out_dir / "timeline.png", dpi=120); plt.close()

    print(f"Wrote model and artifacts to {out_dir}", file=sys.stderr)


# --- predict ---------------------------------------------------------------

def predict_cmd(args):
    bundle: TrainBundle = joblib.load(args.model)
    gauge_raw = fetch_gauge_recent(hours=200)
    q_raw = fetch_discharge_recent(hours=200)
    rain_raw = fetch_all_rain_recent(hours=200)
    tide_obs, tide_pred = fetch_tide_recent(hours=200)
    hourly = to_hourly(gauge_raw, rain_raw, q_raw=q_raw, tide_obs=tide_obs, tide_pred=tide_pred)
    feats = build_features(hourly).dropna(subset=bundle.features)
    if feats.empty:
        print(json.dumps({"error": "insufficient recent data"}))
        return
    latest = feats.iloc[[-1]]
    x = latest[bundle.features].values
    prob = float(bundle.model.predict_proba(x)[0, 1])
    alert = bool(prob >= bundle.threshold)
    rain_totals_6h = {
        name: float(latest[f"{_rain_col(name)}_sum_6"].iloc[0])
        for name, _ in RAIN_GAUGES
    }
    out = {
        "timestamp_utc": latest.index[-1].isoformat(),
        "horizon_h": bundle.horizon_h,
        "gauge_ft_now": float(latest["gauge_ft"].iloc[0]),
        f"prob_closure_in_next_{bundle.horizon_h}h": round(prob, 4),
        "threshold": round(bundle.threshold, 4),
        "alert": alert,
        "already_above_closure": bool(float(latest["gauge_ft"].iloc[0]) > bundle.closure_ft),
        "rain_last_6h_in": {k: round(v, 2) for k, v in rain_totals_6h.items()},
    }
    print(json.dumps(out, indent=2))


# --- backtest --------------------------------------------------------------

def backtest_cmd(args):
    bundle: TrainBundle = joblib.load(args.model)
    out_dir = Path(args.model).parent
    cache_parquet = out_dir / "hourly.parquet"
    cache_csv = out_dir / "hourly.csv"
    if cache_parquet.exists():
        hourly = pd.read_parquet(cache_parquet)
    elif cache_csv.exists():
        hourly = pd.read_csv(cache_csv, index_col=0, parse_dates=True)
    else:
        raise FileNotFoundError("No cached hourly dataset next to model.joblib; rerun train.")
    hourly.index = pd.to_datetime(hourly.index, utc=True)

    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    window = hourly.loc[(hourly.index >= start) & (hourly.index < end)]
    feats = build_features(window)
    feats["y"] = build_target(feats)
    feats = feats.dropna(subset=bundle.features + ["y"])
    feats["y"] = feats["y"].astype(int)
    probs = bundle.model.predict_proba(feats[bundle.features].values)[:, 1]
    feats = feats.assign(
        prob=probs,
        alert=(probs >= bundle.threshold).astype(int),
    )
    evt = event_metrics(feats, probs, bundle.threshold)
    summary = {
        "rows": int(len(feats)),
        "alert_rate": float(feats["alert"].mean()),
        "actual_positive_rate": float(feats["y"].mean()),
        **evt,
    }
    print(json.dumps(summary, indent=2))
    feats[["gauge_ft", "prob", "alert", "y"]].to_csv(args.out)
    print(f"Wrote {args.out}", file=sys.stderr)


# --- cli -------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("train")
    pt.add_argument("--out-dir", default="./hanalei_out")
    pt.add_argument("--use-cache", action="store_true")
    pt.add_argument("--test-years", type=float, default=5.0)
    pt.add_argument("--val-years", type=float, default=2.0)
    pt.add_argument("--min-precision", type=float, default=0.15)
    pt.add_argument("--objective", choices=["event_recall", "f1"], default="event_recall")
    pt.set_defaults(func=train_cmd)

    pp = sub.add_parser("predict")
    pp.add_argument("--model", required=True)
    pp.set_defaults(func=predict_cmd)

    pb = sub.add_parser("backtest")
    pb.add_argument("--model", required=True)
    pb.add_argument("--start", required=True)
    pb.add_argument("--end", required=True)
    pb.add_argument("--out", default="backtest.csv")
    pb.set_defaults(func=backtest_cmd)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
