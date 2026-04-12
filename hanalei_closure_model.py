#!/usr/bin/env python3
"""Hanalei road closure predictor (v7: TFT + 15-min + Google Weather API).

Predicts the probability that the Hanalei River gauge (USGS 16103000) will
exceed the 5.0 ft road-closure threshold at any point in the next 3 hours,
using a Temporal Fusion Transformer (TFT) that learns directly from raw 15-min
time series data from USGS gauges, NOAA tides, ERA5 reanalysis, NWS forecasts,
NASA satellite precipitation, and Google Weather API forecasts.

The TFT predicts quantiles of future gauge_ft, then post-processes them into
P(closure) via quantile interpolation + isotonic calibration.

Subcommands:
  train      Fetch data, build features, train TFT model, save artifacts.
  predict    Load model, fetch recent data, emit JSON prediction.
  backtest   Load model + cached data, evaluate on a custom date range.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

import torch
import lightning.pytorch as pl
from pytorch_forecasting import (
    TemporalFusionTransformer,
    TimeSeriesDataSet,
    QuantileLoss,
)
from pytorch_forecasting.data import NaNLabelEncoder
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor

# --- constants -------------------------------------------------------------

USGS_IV_URL = "https://waterservices.usgs.gov/nwis/iv/"
STREAM_SITE = "16103000"
BRIDGE_SITE = "16104200"  # Hanalei River at Hwy 56 Bridge (downstream, at actual closure point)
BRIDGE_FLOOD_FT = 7.3     # NWS minor flood stage at bridge
PARAM_GAUGE = "00065"   # gauge height, ft
PARAM_Q = "00060"       # discharge, cfs
PARAM_RAIN = "00045"    # precipitation, inches (per-interval totals)

# NOAA CO-OPS tide station (closest to Hanalei on Kauai).
TIDE_STATION = "1611400"  # Nawiliwili Harbor
NOAA_COOPS_URL = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"

# NASA POWER API — daily satellite-derived (IMERG) precipitation, free, no auth.
NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

# In-watershed and nearby USGS rain gauges (all have 15-min IV data through present).
RAIN_GAUGES: list[tuple[str, str]] = [
    ("waiakoali",   "220739159373001"),
    ("mohihi_crsg", "220713159361201"),
    ("waialae",     "220523159341201"),
    ("kilohana",    "220927159355001"),
    ("waialeale",   "220427159300201"),  # Mt Waialeale — highest annual rainfall
    ("n_wailua",    "220356159281401"),
]

# NWS API forecast gridpoint for Hanalei, Kauai
NWS_GRIDPOINT = "HFO/87,185"
NWS_API_URL = f"https://api.weather.gov/gridpoints/{NWS_GRIDPOINT}"
NWS_HEADERS = {"User-Agent": "hanalei-flood-model (research)", "Accept": "application/geo+json"}

# Google Weather API (WeatherNext 2)
GOOGLE_WEATHER_URL = "https://weather.googleapis.com/v1/forecast/hours:lookup"
GOOGLE_WEATHER_API_KEY = os.environ.get("GOOGLE_WEATHER_API_KEY", "")

# --- TFT hyperparameters ---
TFT_ENCODER_LENGTH = 96       # 24 hours of lookback at 15-min
TFT_PREDICTION_LENGTH = 12    # 3 hours of forecast at 15-min
TFT_HIDDEN_SIZE = 64
TFT_ATTENTION_HEADS = 4
TFT_DROPOUT = 0.1
TFT_QUANTILES = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
TFT_MAX_EPOCHS = 100
TFT_BATCH_SIZE = 512
TFT_LEARNING_RATE = 1e-3

# Earliest date for which BOTH the stream gauge and all rain gauges have data.
DATA_START = datetime(2007, 10, 1, tzinfo=timezone.utc)

# Open-Meteo Historical Weather API (ERA5 reanalysis, free, no key needed).
OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
HANALEI_LAT = 22.198
HANALEI_LON = -159.467

CLOSURE_FT = 5.0
HORIZON_H = 3
TIMESTEP_MIN = 15           # 15-minute resolution (v6)
STEPS_PER_HOUR = 60 // TIMESTEP_MIN  # 4
HORIZON_STEPS = HORIZON_H * STEPS_PER_HOUR  # 12

# Rolling rain windows in HOURS — will be converted to steps internally.
RAIN_WINDOWS_PER_GAUGE = (1, 3, 6, 24)
RAIN_WINDOWS_XGAUGE = (1, 3, 6, 12, 24, 48, 72, 168)  # 168h = 7 days


def _rain_col(name: str) -> str:
    return f"rain_{name}"


# --- TFT covariate definitions -----------------------------------------------
# Time-varying OBSERVED: only available in encoder (lookback) window
TFT_OBSERVED_COLS = [
    "gauge_ft",        # also the prediction target
    "bridge_ft",
    "q_cfs",
    *[f"rain_{name}" for name, _ in RAIN_GAUGES],
    "tide_ft",
    "storm_surge_ft",  # derived: tide_ft - tide_pred_ft
    "wx_humidity_pct",
    "wx_wind_kmh",
    "wx_wind_gust_kmh",
    "wx_pressure_hpa",
    "wx_precip_era5_mm",
    "soil_moisture_shallow",
    "soil_moisture_mid",
    "soil_moisture_deep",
    "sat_precip_mm",
    "rain_ema_168h",   # 7-day antecedent rain EMA (captures saturation beyond lookback)
]

# Time-varying KNOWN: available for future/decoder window
TFT_KNOWN_COLS = [
    "tide_pred_ft",
    "hour_sin", "hour_cos",
    "month_sin", "month_cos",
    "nws_qpf_3h", "nws_qpf_6h",
    "gw_precip_rate", "gw_temperature", "gw_humidity",
    "gw_wind_speed", "gw_wind_gust", "gw_pressure",
    "has_nws_forecast", "has_google_weather",
]

# Google Weather API columns
GW_COLS = ["gw_precip_rate", "gw_temperature", "gw_humidity",
           "gw_wind_speed", "gw_wind_gust", "gw_pressure"]

# Legacy alias for backward compat (web module may reference)
FEATURE_COLUMNS = TFT_OBSERVED_COLS + TFT_KNOWN_COLS


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


def fetch_bridge_gauge(start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch gauge height at Hwy 56 Bridge (USGS 16104200, downstream)."""
    return _fetch_usgs_iv(BRIDGE_SITE, PARAM_GAUGE, "bridge_ft", start, end, "bridge")


def fetch_bridge_gauge_recent(hours: int = 200) -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    return fetch_bridge_gauge(now - timedelta(hours=hours), now + timedelta(days=1))


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
        for attempt in range(3):
            try:
                r = requests.get(NOAA_COOPS_URL, params=params, timeout=120)
                r.raise_for_status()
                break
            except (requests.exceptions.RequestException,) as exc:
                if attempt < 2:
                    print(f"    retry {attempt+1}/2 after {exc}", file=sys.stderr)
                    time.sleep(5 * (attempt + 1))
                else:
                    raise
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


# --- Open-Meteo / ERA5 historical weather -----------------------------------

def fetch_weather_era5(start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch ERA5 reanalysis weather data from Open-Meteo (free, no key).

    Returns hourly DataFrame with columns:
      wx_humidity_pct, wx_wind_kmh, wx_wind_gust_kmh, wx_pressure_hpa, wx_precip_era5_mm
    """
    print(f"  era5/weather   {start.date()} -> {end.date()}", file=sys.stderr)
    params = {
        "latitude": HANALEI_LAT,
        "longitude": HANALEI_LON,
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "hourly": "precipitation,relative_humidity_2m,wind_speed_10m,pressure_msl,wind_gusts_10m",
        "timezone": "UTC",
    }
    r = requests.get(OPEN_METEO_URL, params=params, timeout=120)
    r.raise_for_status()
    hourly = r.json().get("hourly", {})
    df = pd.DataFrame({
        "ts": pd.to_datetime(hourly["time"], utc=True),
        "wx_humidity_pct": hourly.get("relative_humidity_2m"),
        "wx_wind_kmh": hourly.get("wind_speed_10m"),
        "wx_wind_gust_kmh": hourly.get("wind_gusts_10m"),
        "wx_pressure_hpa": hourly.get("pressure_msl"),
        "wx_precip_era5_mm": hourly.get("precipitation"),
    })
    df = df.set_index("ts")
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    print(f"  era5/weather   {len(df):,} hourly rows", file=sys.stderr)
    return df


def fetch_weather_recent(hours: int = 200) -> pd.DataFrame:
    """Fetch recent weather from Open-Meteo forecast API (includes past days + forecast)."""
    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=hours)
    print(f"  weather/recent {start.date()} -> {now.date()}", file=sys.stderr)
    params = {
        "latitude": HANALEI_LAT,
        "longitude": HANALEI_LON,
        "hourly": "precipitation,relative_humidity_2m,wind_speed_10m,pressure_msl,wind_gusts_10m",
        "past_days": max(1, (now - start).days + 1),
        "timezone": "UTC",
    }
    try:
        r = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=60)
        r.raise_for_status()
        hourly = r.json().get("hourly", {})
        df = pd.DataFrame({
            "ts": pd.to_datetime(hourly["time"], utc=True),
            "wx_humidity_pct": hourly.get("relative_humidity_2m"),
            "wx_wind_kmh": hourly.get("wind_speed_10m"),
            "wx_wind_gust_kmh": hourly.get("wind_gusts_10m"),
            "wx_pressure_hpa": hourly.get("pressure_msl"),
            "wx_precip_era5_mm": hourly.get("precipitation"),
        })
        df = df.set_index("ts")
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        # Trim to requested window
        df = df.loc[df.index >= start]
        print(f"  weather/recent {len(df):,} hourly rows", file=sys.stderr)
        return df
    except Exception as e:
        print(f"  weather/recent FAILED: {e}", file=sys.stderr)
        return pd.DataFrame()


# --- ERA5 soil moisture (Open-Meteo, free, no key) -------------------------

def fetch_soil_moisture_era5(start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch ERA5-Land soil moisture from Open-Meteo archive API.

    Returns hourly DataFrame with columns:
      soil_moisture_shallow (0-7cm), soil_moisture_mid (7-28cm),
      soil_moisture_deep (28-100cm)  — all in m³/m³.
    """
    print(f"  era5/soil      {start.date()} -> {end.date()}", file=sys.stderr)
    params = {
        "latitude": HANALEI_LAT,
        "longitude": HANALEI_LON,
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "hourly": "soil_moisture_0_to_7cm,soil_moisture_7_to_28cm,soil_moisture_28_to_100cm",
        "timezone": "UTC",
    }
    r = requests.get(OPEN_METEO_URL, params=params, timeout=120)
    r.raise_for_status()
    hourly = r.json().get("hourly", {})
    df = pd.DataFrame({
        "ts": pd.to_datetime(hourly["time"], utc=True),
        "soil_moisture_shallow": hourly.get("soil_moisture_0_to_7cm"),
        "soil_moisture_mid": hourly.get("soil_moisture_7_to_28cm"),
        "soil_moisture_deep": hourly.get("soil_moisture_28_to_100cm"),
    })
    df = df.set_index("ts")
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    print(f"  era5/soil      {len(df):,} hourly rows", file=sys.stderr)
    return df


def fetch_soil_moisture_recent(hours: int = 200) -> pd.DataFrame:
    """Fetch recent soil moisture from Open-Meteo forecast API."""
    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=hours)
    print(f"  soil/recent    {start.date()} -> {now.date()}", file=sys.stderr)
    params = {
        "latitude": HANALEI_LAT,
        "longitude": HANALEI_LON,
        "hourly": "soil_moisture_0_to_7cm,soil_moisture_7_to_28cm,soil_moisture_28_to_100cm",
        "past_days": max(1, (now - start).days + 1),
        "timezone": "UTC",
    }
    try:
        r = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=60)
        r.raise_for_status()
        hourly = r.json().get("hourly", {})
        df = pd.DataFrame({
            "ts": pd.to_datetime(hourly["time"], utc=True),
            "soil_moisture_shallow": hourly.get("soil_moisture_0_to_7cm"),
            "soil_moisture_mid": hourly.get("soil_moisture_7_to_28cm"),
            "soil_moisture_deep": hourly.get("soil_moisture_28_to_100cm"),
        })
        df = df.set_index("ts")
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.loc[df.index >= start]
        print(f"  soil/recent    {len(df):,} hourly rows", file=sys.stderr)
        return df
    except Exception as e:
        print(f"  soil/recent    FAILED: {e}", file=sys.stderr)
        return pd.DataFrame()


# --- NASA POWER satellite precipitation (IMERG, daily) ---------------------

def fetch_nasa_power_precip(start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch daily satellite-derived (IMERG) precipitation from NASA POWER API.

    Returns daily DataFrame with column 'sat_precip_mm' (mm/day).
    Free API, no authentication needed.  Chunked into 365-day requests.
    """
    print(f"  power/precip   {start.date()} -> {end.date()}", file=sys.stderr)
    frames: list[pd.DataFrame] = []
    chunk = timedelta(days=365)
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + chunk, end)
        params = {
            "parameters": "PRECTOTCORR",
            "community": "RE",
            "longitude": HANALEI_LON,
            "latitude": HANALEI_LAT,
            "start": cursor.strftime("%Y%m%d"),
            "end": chunk_end.strftime("%Y%m%d"),
            "format": "JSON",
        }
        for attempt in range(3):
            try:
                r = requests.get(NASA_POWER_URL, params=params, timeout=120)
                r.raise_for_status()
                break
            except requests.exceptions.RequestException as exc:
                if attempt < 2:
                    print(f"    power retry {attempt+1}: {exc}", file=sys.stderr)
                    time.sleep(5 * (attempt + 1))
                else:
                    raise
        data = r.json()
        daily = data.get("properties", {}).get("parameter", {}).get("PRECTOTCORR", {})
        for date_str, val in daily.items():
            if val is not None and val > -900:  # -999 = missing
                ts = pd.Timestamp(date_str, tz="UTC")
                frames.append(pd.DataFrame({"sat_precip_mm": [float(val)]}, index=[ts]))
        cursor = chunk_end + timedelta(days=1)
        time.sleep(0.3)

    if not frames:
        return pd.DataFrame(columns=["sat_precip_mm"]).set_index(
            pd.DatetimeIndex([], tz="UTC", name="ts")
        )
    df = pd.concat(frames).sort_index()
    df.index.name = "ts"
    print(f"  power/precip   {len(df):,} daily rows", file=sys.stderr)
    return df


def fetch_nasa_power_precip_recent(days: int = 14) -> pd.DataFrame:
    """Fetch recent satellite precip (NASA POWER has ~days latency so may be partial)."""
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=days)
    try:
        return fetch_nasa_power_precip(start, now)
    except Exception as e:
        print(f"  power/recent   FAILED: {e}", file=sys.stderr)
        return pd.DataFrame()


# --- NWS forecast data -----------------------------------------------------

def _parse_nws_timeseries(values: list[dict], col_name: str) -> pd.DataFrame:
    """Parse NWS ISO 8601 duration-based time series into a regular hourly DataFrame."""
    records = []
    for entry in values:
        valid_time = entry.get("validTime", "")
        value = entry.get("value")
        if value is None or "/" not in valid_time:
            continue
        ts_str, duration = valid_time.split("/")
        start = pd.Timestamp(ts_str)
        # Parse ISO 8601 duration (e.g., PT1H, PT6H, PT4H)
        hours = 0
        dur = duration.replace("PT", "")
        if "H" in dur:
            hours = int(dur.replace("H", ""))
        elif "D" in dur:
            hours = int(dur.replace("D", "")) * 24
        if hours == 0:
            hours = 1
        # Spread the value evenly across the duration (for QPF, value is total mm)
        for h in range(hours):
            records.append({"ts": start + timedelta(hours=h), col_name: float(value) / hours})
    if not records:
        return pd.DataFrame(columns=[col_name]).set_index(
            pd.DatetimeIndex([], tz="UTC", name="ts")
        )
    df = pd.DataFrame(records)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.drop_duplicates("ts").sort_values("ts").set_index("ts")


def fetch_nws_forecast() -> dict[str, pd.DataFrame]:
    """Fetch current NWS gridpoint forecast for Hanalei.

    Returns dict with keys: 'qpf_mm', 'wind_kmh', 'humidity_pct', 'pressure_pa'.
    Each is a DataFrame indexed by UTC timestamp at hourly resolution.
    Returns empty DataFrames on failure (graceful degradation).
    """
    result = {}
    try:
        print("  nws/forecast  fetching...", file=sys.stderr)
        r = requests.get(NWS_API_URL, headers=NWS_HEADERS, timeout=30)
        r.raise_for_status()
        props = r.json().get("properties", {})

        # QPF — mm of precipitation per period
        qpf = props.get("quantitativePrecipitation", {}).get("values", [])
        result["qpf_mm"] = _parse_nws_timeseries(qpf, "qpf_mm")

        # Wind speed — km/h
        wind = props.get("windSpeed", {}).get("values", [])
        result["wind_kmh"] = _parse_nws_timeseries(wind, "wind_kmh")

        # Relative humidity — %
        rh = props.get("relativeHumidity", {}).get("values", [])
        result["humidity_pct"] = _parse_nws_timeseries(rh, "humidity_pct")

        # Barometric pressure — Pa
        pressure = props.get("pressure", {}).get("values", [])
        result["pressure_pa"] = _parse_nws_timeseries(pressure, "pressure_pa")

        for k, df in result.items():
            print(f"  nws/{k:<14s} {len(df)} hourly values", file=sys.stderr)

    except Exception as e:
        print(f"  nws/forecast  FAILED: {e}", file=sys.stderr)
        for col in ("qpf_mm", "wind_kmh", "humidity_pct", "pressure_pa"):
            result[col] = pd.DataFrame(columns=[col]).set_index(
                pd.DatetimeIndex([], tz="UTC", name="ts")
            )

    return result


# --- Google Weather API (WeatherNext 2) ------------------------------------

def fetch_google_weather() -> pd.DataFrame | None:
    """Fetch hourly forecast from Google Weather API for Hanalei.

    Returns DataFrame with columns: gw_precip_rate, gw_temperature,
    gw_humidity, gw_wind_speed, gw_wind_gust, gw_pressure.
    Returns None on failure (graceful degradation).
    """
    if not GOOGLE_WEATHER_API_KEY:
        print("  google/weather SKIPPED (no API key)", file=sys.stderr)
        return None
    try:
        print("  google/weather fetching...", file=sys.stderr)
        params = {
            "key": GOOGLE_WEATHER_API_KEY,
            "location.latitude": HANALEI_LAT,
            "location.longitude": HANALEI_LON,
        }
        r = requests.get(GOOGLE_WEATHER_URL, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        hours = data.get("forecastHours", data.get("hours", []))
        if not hours:
            print("  google/weather no forecast hours returned", file=sys.stderr)
            return None

        records = []
        for h in hours:
            ts_str = h.get("interval", {}).get("startTime") or h.get("startTime")
            if not ts_str:
                continue
            ts = pd.Timestamp(ts_str, tz="UTC")
            precip = h.get("precipitation", {})
            wind = h.get("wind", {})
            records.append({
                "ts": ts,
                "gw_precip_rate": precip.get("probability", {}).get("value",
                    precip.get("qpf", {}).get("quantity", {}).get("value", 0)),
                "gw_temperature": h.get("temperature", {}).get("value", np.nan),
                "gw_humidity": h.get("relativeHumidity", {}).get("value", np.nan),
                "gw_wind_speed": wind.get("speed", {}).get("value", np.nan),
                "gw_wind_gust": wind.get("gust", {}).get("value", np.nan),
                "gw_pressure": h.get("pressure", {}).get("meanSeaLevel", {}).get("value", np.nan),
            })

        if not records:
            return None
        df = pd.DataFrame(records)
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df = df.drop_duplicates("ts").sort_values("ts").set_index("ts")
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        print(f"  google/weather {len(df)} hourly forecast rows", file=sys.stderr)
        return df
    except Exception as e:
        print(f"  google/weather FAILED: {e}", file=sys.stderr)
        return None


# --- merging / resampling --------------------------------------------------

def to_hourly(
    gauge_raw: pd.DataFrame,
    rain_raw: dict[str, pd.DataFrame],
    q_raw: pd.DataFrame | None = None,
    tide_obs: pd.DataFrame | None = None,
    tide_pred: pd.DataFrame | None = None,
    nws: dict[str, pd.DataFrame] | None = None,
    weather: pd.DataFrame | None = None,
    bridge_raw: pd.DataFrame | None = None,
    soil_moisture: pd.DataFrame | None = None,
    sat_precip: pd.DataFrame | None = None,
    google_weather: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Resample all sources to aligned 15-minute records (v7).

    USGS data is natively 15-min, so gauge/discharge/rain keep full resolution.
    Hourly sources (ERA5, NWS, soil moisture, Google Weather) are forward-filled to 15-min.
    """
    freq = f"{TIMESTEP_MIN}min"
    gauge_15 = gauge_raw["gauge_ft"].resample(freq).max()
    frames: list[pd.Series] = [gauge_15]

    # Bridge gauge (v6) — natively 15-min
    if bridge_raw is not None and not bridge_raw.empty:
        frames.append(bridge_raw["bridge_ft"].resample(freq).max())

    # Discharge — natively 15-min
    if q_raw is not None and not q_raw.empty:
        frames.append(q_raw["q_cfs"].resample(freq).max())

    # Rain gauges — natively 15-min, SUM per interval
    for name, _ in RAIN_GAUGES:
        col = _rain_col(name)
        if name in rain_raw and not rain_raw[name].empty:
            r15 = rain_raw[name][col].resample(freq).sum(min_count=1)
        else:
            r15 = pd.Series(dtype=float, name=col)
        frames.append(r15.rename(col))

    # Tide — natively 6-min, resample to 15-min mean
    if tide_obs is not None and not tide_obs.empty:
        frames.append(tide_obs["tide_ft"].resample(freq).mean())
    if tide_pred is not None and not tide_pred.empty:
        frames.append(tide_pred["tide_pred_ft"].resample(freq).mean())

    # ERA5 weather (hourly) — forward-fill to 15-min
    if weather is not None and not weather.empty:
        wx_15 = weather.resample(freq).ffill()
        for col in wx_15.columns:
            frames.append(wx_15[col])

    # Soil moisture (hourly) — forward-fill to 15-min (v6)
    if soil_moisture is not None and not soil_moisture.empty:
        sm_15 = soil_moisture.resample(freq).ffill()
        for col in sm_15.columns:
            frames.append(sm_15[col])

    # NWS forecast (hourly) — forward-fill to 15-min
    if nws is not None:
        for col_name, nws_df in nws.items():
            if not nws_df.empty:
                nws_15 = nws_df[col_name].resample(freq).ffill()
                frames.append(nws_15)

    # Google Weather forecast (hourly) — forward-fill to 15-min (v7)
    if google_weather is not None and not google_weather.empty:
        gw_15 = google_weather.resample(freq).ffill()
        for col in gw_15.columns:
            frames.append(gw_15[col])

    df = pd.concat(frames, axis=1, sort=True)

    # Satellite precip: daily → spread to 15-min by forward-filling (v6)
    if sat_precip is not None and not sat_precip.empty:
        # Convert daily mm to per-15-min mm (divide by 96 = 24*4)
        daily_15 = sat_precip["sat_precip_mm"].resample(freq).ffill() / (24 * STEPS_PER_HOUR)
        df = df.join(daily_15, how="left")

    # Gentle gap filling
    gap_limit = 2 * STEPS_PER_HOUR  # 2 hours
    df["gauge_ft"] = df["gauge_ft"].interpolate(method="time", limit=gap_limit)
    if "bridge_ft" in df.columns:
        df["bridge_ft"] = df["bridge_ft"].interpolate(method="time", limit=gap_limit)
    if "q_cfs" in df.columns:
        df["q_cfs"] = df["q_cfs"].interpolate(method="time", limit=gap_limit)
    for name, _ in RAIN_GAUGES:
        df[_rain_col(name)] = df[_rain_col(name)].fillna(0.0)
    if "tide_ft" in df.columns:
        df["tide_ft"] = df["tide_ft"].interpolate(method="time", limit=6 * STEPS_PER_HOUR)
    if "tide_pred_ft" in df.columns:
        df["tide_pred_ft"] = df["tide_pred_ft"].interpolate(method="time", limit=6 * STEPS_PER_HOUR)
    # Soil moisture: forward-fill (changes slowly, gaps are OK)
    for sm_col in ["soil_moisture_shallow", "soil_moisture_mid", "soil_moisture_deep"]:
        if sm_col in df.columns:
            df[sm_col] = df[sm_col].interpolate(method="time", limit=12 * STEPS_PER_HOUR)

    df = df.dropna(subset=["gauge_ft"])
    return df


# --- features and target ---------------------------------------------------

def build_tft_features(df: pd.DataFrame) -> pd.DataFrame:
    """Minimal feature engineering for TFT input (v7).

    TFT learns temporal patterns (lags, deltas, rolling windows) internally
    from raw time series. We only compute:
    - storm_surge_ft (derived observed)
    - rain_ema_168h (7-day antecedent rain EMA — captures soil saturation beyond lookback)
    - Time encodings (hour_sin/cos, month_sin/cos)
    - NWS QPF forward sums (known future)
    - has_nws_forecast / has_google_weather indicators
    - Ensure all expected columns exist (fill NaN if missing)
    """
    S = STEPS_PER_HOUR
    out = df.copy()

    # --- storm surge (derived observed) ---
    if "tide_ft" in out.columns and "tide_pred_ft" in out.columns:
        out["storm_surge_ft"] = out["tide_ft"] - out["tide_pred_ft"]
    else:
        out["storm_surge_ft"] = np.nan

    # --- 7-day antecedent rain EMA (captures soil saturation beyond 24h lookback) ---
    total_rain = sum(
        out[_rain_col(name)].fillna(0) for name, _ in RAIN_GAUGES
    )
    out["rain_ema_168h"] = total_rain.ewm(halflife=168 * S, min_periods=1).mean()

    # --- NWS forecast QPF forward sums ---
    if "qpf_mm" in out.columns:
        qpf_rev = out["qpf_mm"].iloc[::-1]
        out["nws_qpf_3h"] = qpf_rev.rolling(3 * S, min_periods=1).sum().iloc[::-1].values
        out["nws_qpf_6h"] = qpf_rev.rolling(6 * S, min_periods=1).sum().iloc[::-1].values
    else:
        out["nws_qpf_3h"] = 0.0
        out["nws_qpf_6h"] = 0.0

    # --- time encodings ---
    hour = out.index.hour + out.index.minute / 60.0
    month = out.index.month
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    out["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)

    # --- forecast availability indicators ---
    has_nws = "qpf_mm" in out.columns and out["qpf_mm"].notna().any()
    out["has_nws_forecast"] = 1.0 if has_nws else 0.0

    has_gw = any(c in out.columns and out[c].notna().any() for c in GW_COLS)
    out["has_google_weather"] = 1.0 if has_gw else 0.0

    # --- ensure all Google Weather columns exist ---
    for col in GW_COLS:
        if col not in out.columns:
            out[col] = 0.0

    # --- ensure all expected observed columns exist ---
    for col in TFT_OBSERVED_COLS:
        if col not in out.columns:
            out[col] = np.nan

    return out


# Legacy alias for backward compat
build_features = build_tft_features


def build_target(df: pd.DataFrame) -> pd.Series:
    """closure_in_next_H[t] = 1 iff max(gauge[t..t+H]) > CLOSURE_FT.

    Includes the current step (shift 0) so that "gauge is already at flood
    stage" is a positive label.
    """
    g = df["gauge_ft"]
    shifts = pd.concat([g.shift(-i) for i in range(0, HORIZON_STEPS + 1)], axis=1)
    future_max = shifts.max(axis=1, skipna=True)
    y = pd.Series(pd.NA, index=df.index, dtype="Int8")
    mask = future_max.notna()
    y[mask] = (future_max[mask] > CLOSURE_FT).astype(int)
    return y


# --- TFT model infrastructure -----------------------------------------------

@dataclass
class TFTBundle:
    """Serializable metadata for the trained TFT model."""
    model_path: str              # path to .ckpt file
    threshold: float
    target_col: str              # "gauge_ft"
    observed_cols: list = field(default_factory=lambda: list(TFT_OBSERVED_COLS))
    known_cols: list = field(default_factory=lambda: list(TFT_KNOWN_COLS))
    closure_ft: float = CLOSURE_FT
    horizon_h: int = HORIZON_H
    horizon_steps: int = HORIZON_STEPS
    encoder_length: int = TFT_ENCODER_LENGTH
    prediction_length: int = TFT_PREDICTION_LENGTH
    quantiles: list = field(default_factory=lambda: list(TFT_QUANTILES))
    rain_gauge_names: list = field(default_factory=lambda: [n for n, _ in RAIN_GAUGES])
    fill_medians: dict = field(default_factory=dict)


def save_tft_bundle(bundle: TFTBundle, out_dir: Path) -> None:
    (out_dir / "bundle.json").write_text(json.dumps(asdict(bundle), indent=2, default=str))


def load_tft_bundle(path: Path) -> TFTBundle:
    if path.is_dir():
        path = path / "bundle.json"
    data = json.loads(path.read_text())
    return TFTBundle(**data)


# Legacy alias so web module can load either
TrainBundle = TFTBundle


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
            lo = max(0, pos - HORIZON_STEPS)
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
            lead_steps = pos - (lo + first_alert_rel)
            lead_times.append(lead_steps * TIMESTEP_MIN)

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

    print(f"Fetching bridge gauge {start.date()} -> {end.date()}", file=sys.stderr)
    try:
        bridge_raw = fetch_bridge_gauge(start, end)
        print(f"  bridge gauge rows: {len(bridge_raw):,}", file=sys.stderr)
    except Exception as e:
        print(f"  bridge gauge FAILED: {e}", file=sys.stderr)
        bridge_raw = None

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

    print(f"Fetching ERA5 weather {start.date()} -> {end.date()}", file=sys.stderr)
    wx = fetch_weather_era5(start, end)

    print(f"Fetching ERA5 soil moisture {start.date()} -> {end.date()}", file=sys.stderr)
    try:
        soil = fetch_soil_moisture_era5(start, end)
    except Exception as e:
        print(f"  soil moisture FAILED: {e}", file=sys.stderr)
        soil = None

    print(f"Fetching NASA POWER satellite precip {start.date()} -> {end.date()}", file=sys.stderr)
    try:
        sat_precip = fetch_nasa_power_precip(start, end)
    except Exception as e:
        print(f"  satellite precip FAILED: {e}", file=sys.stderr)
        sat_precip = None

    hourly = to_hourly(
        gauge_raw, rain_raw, q_raw=q_raw, tide_obs=tide_obs, tide_pred=tide_pred,
        weather=wx, bridge_raw=bridge_raw, soil_moisture=soil, sat_precip=sat_precip,
    )
    print(f"Merged hourly rows: {len(hourly):,}", file=sys.stderr)
    return hourly


def prepare_tft_dataframe(df: pd.DataFrame, fill_medians: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Prepare a DataFrame for TFT TimeSeriesDataSet construction.

    - Adds time_idx (sequential integer) and group columns
    - Imputes NaN using provided medians (or computes them from df)
    - Returns (prepared_df, fill_medians)
    """
    out = df.copy()

    # Sequential time index (required by pytorch-forecasting)
    out = out.sort_index()
    out["time_idx"] = np.arange(len(out))
    out["group"] = "hanalei"  # single time series

    # Compute fill medians from this data if not provided
    all_cols = TFT_OBSERVED_COLS + TFT_KNOWN_COLS
    if fill_medians is None:
        fill_medians = {}
        for col in all_cols:
            if col in out.columns:
                med = out[col].median()
                fill_medians[col] = float(med) if pd.notna(med) else 0.0

    # Impute NaN: forward-fill first, then fill with medians
    for col in all_cols:
        if col in out.columns:
            out[col] = out[col].ffill(limit=8)  # 2 hours at 15-min
            med = fill_medians.get(col, 0.0)
            out[col] = out[col].fillna(med)

    # Target must not be NaN
    out = out.dropna(subset=["gauge_ft"])

    return out, fill_medians


def build_tft_dataset(
    df: pd.DataFrame,
    encoder_length: int = TFT_ENCODER_LENGTH,
    prediction_length: int = TFT_PREDICTION_LENGTH,
    training: bool = True,
    dataset_params: dict | None = None,
) -> TimeSeriesDataSet:
    """Construct a pytorch-forecasting TimeSeriesDataSet for TFT."""
    # Filter to columns that actually exist in df
    observed_cols = [c for c in TFT_OBSERVED_COLS if c in df.columns and c != "gauge_ft"]
    known_cols = [c for c in TFT_KNOWN_COLS if c in df.columns]

    if dataset_params is not None and not training:
        # Reconstruct from saved params for prediction
        return TimeSeriesDataSet.from_parameters(
            dataset_params, df, predict=True,
        )

    dataset = TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target="gauge_ft",
        group_ids=["group"],
        min_encoder_length=encoder_length // 2,
        max_encoder_length=encoder_length,
        min_prediction_length=1,
        max_prediction_length=prediction_length,
        time_varying_known_reals=known_cols,
        time_varying_unknown_reals=["gauge_ft"] + observed_cols,
        target_normalizer=None,  # we handle scaling ourselves if needed
        categorical_encoders={"group": NaNLabelEncoder(add_nan=True)},
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )
    return dataset


def quantiles_to_closure_prob(
    quantile_predictions: np.ndarray,
    quantile_levels: list[float],
    closure_ft: float = CLOSURE_FT,
) -> np.ndarray:
    """Convert TFT quantile predictions to P(closure) for each sample.

    Args:
        quantile_predictions: shape (n_samples, prediction_length, n_quantiles)
        quantile_levels: e.g. [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
        closure_ft: threshold (5.0 ft)

    Returns:
        probs: shape (n_samples,) — estimated P(max gauge > closure_ft in horizon)
    """
    n_samples = quantile_predictions.shape[0]
    probs = np.zeros(n_samples)
    q_levels = np.array(quantile_levels)

    for i in range(n_samples):
        # Take max across prediction horizon at each quantile level
        max_per_quantile = quantile_predictions[i].max(axis=0)  # (n_quantiles,)

        # If all quantiles are below threshold → very low probability
        if max_per_quantile[-1] <= closure_ft:
            # Extrapolate: use the highest quantile to estimate tail
            probs[i] = max(0.0, (1.0 - q_levels[-1]) *
                          max(0, max_per_quantile[-1] - closure_ft + 0.5) / 0.5)
            continue

        # If all quantiles are above threshold → very high probability
        if max_per_quantile[0] >= closure_ft:
            probs[i] = 1.0
            continue

        # Interpolate: find where closure_ft falls in the quantile distribution
        # P(gauge > closure_ft) = 1 - P(gauge <= closure_ft)
        # P(gauge <= closure_ft) is interpolated from the quantile function
        p_below = np.interp(closure_ft, max_per_quantile, q_levels)
        probs[i] = 1.0 - p_below

    return probs


class CachedBatchDataLoader:
    """Pre-caches all batches from a DataLoader into RAM.

    Eliminates per-batch TimeSeriesDataSet overhead by running the full
    DataLoader once (with its custom collate), caching every batch as
    tensors, then replaying them on subsequent epochs with zero Python
    processing overhead.
    """

    def __init__(self, dataloader, desc: str = "", shuffle: bool = False):
        print(f"  Caching {len(dataloader)} batches ({desc})...",
              file=sys.stderr, end=" ", flush=True)
        t0 = time.time()
        self._batches = []
        for batch in dataloader:
            # Move to CPU and detach to avoid holding graph
            x, y = batch
            x_cpu = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                     for k, v in x.items()}
            y_cpu = tuple(v.detach().cpu() if isinstance(v, torch.Tensor) else v
                          for v in y)
            self._batches.append((x_cpu, y_cpu))
        self._shuffle = shuffle
        elapsed = time.time() - t0
        print(f"done in {elapsed:.0f}s ({len(self._batches)} batches)", file=sys.stderr)

    def __len__(self):
        return len(self._batches)

    def __iter__(self):
        indices = list(range(len(self._batches)))
        if self._shuffle:
            import random
            random.shuffle(indices)
        for i in indices:
            yield self._batches[i]


def train_tft(
    train_dataset: TimeSeriesDataSet,
    val_dataset: TimeSeriesDataSet,
    out_dir: Path,
    hidden_size: int = TFT_HIDDEN_SIZE,
    attention_head_size: int = TFT_ATTENTION_HEADS,
    dropout: float = TFT_DROPOUT,
    max_epochs: int = TFT_MAX_EPOCHS,
    batch_size: int = TFT_BATCH_SIZE,
    learning_rate: float = TFT_LEARNING_RATE,
    quantiles: list[float] | None = None,
    num_workers: int = 0,
    precision: str = "32-true",
    devices: int = 1,
    pretensorize: bool = False,
) -> TemporalFusionTransformer:
    """Train a TFT model and return the best checkpoint."""
    if quantiles is None:
        quantiles = TFT_QUANTILES

    use_cuda = torch.cuda.is_available()

    if pretensorize:
        print("Pre-caching batches (one-time cost)...", file=sys.stderr)
        torch.multiprocessing.set_sharing_strategy('file_system')
        # Use fewer workers for the cache pass to avoid too-many-open-files
        cache_workers = min(num_workers, 4)
        _dl_kwargs = dict(num_workers=cache_workers, pin_memory=False,
                          persistent_workers=cache_workers > 0)
        _train_dl = train_dataset.to_dataloader(train=True, batch_size=batch_size, **_dl_kwargs)
        _val_dl = val_dataset.to_dataloader(train=False, batch_size=batch_size, **_dl_kwargs)
        # Cache all batches into RAM — subsequent epochs are instant
        train_dl = CachedBatchDataLoader(_train_dl, desc="train", shuffle=True)
        val_dl = CachedBatchDataLoader(_val_dl, desc="val", shuffle=False)
        del _train_dl, _val_dl  # free original DataLoader workers
    else:
        dl_kwargs = dict(
            num_workers=num_workers,
            pin_memory=use_cuda,
            persistent_workers=num_workers > 0,
        )
        train_dl = train_dataset.to_dataloader(
            train=True, batch_size=batch_size, **dl_kwargs,
        )
        val_dl = val_dataset.to_dataloader(
            train=False, batch_size=batch_size, **dl_kwargs,
        )

    tft = TemporalFusionTransformer.from_dataset(
        train_dataset,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        hidden_continuous_size=hidden_size // 2,
        loss=QuantileLoss(quantiles=quantiles),
        learning_rate=learning_rate,
        reduce_on_plateau_patience=5,
        log_interval=10,
        optimizer="adam",
    )

    early_stop = EarlyStopping(
        monitor="val_loss", patience=10, mode="min", verbose=True,
    )
    lr_monitor = LearningRateMonitor()

    # Multi-GPU uses DDP strategy
    strategy = "ddp" if devices > 1 else "auto"

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=devices,
        strategy=strategy,
        precision=precision,
        gradient_clip_val=0.1,
        callbacks=[early_stop, lr_monitor],
        default_root_dir=str(out_dir),
        enable_progress_bar=True,
        log_every_n_steps=10,
    )

    n_params = sum(p.numel() for p in tft.parameters())
    print(f"Training TFT: {n_params:,} params | batch={batch_size} | "
          f"workers={num_workers} | precision={precision} | "
          f"devices={devices} | strategy={strategy}", file=sys.stderr)
    trainer.fit(tft, train_dataloaders=train_dl, val_dataloaders=val_dl)

    # Load best checkpoint
    best_path = trainer.checkpoint_callback.best_model_path
    best_score = trainer.checkpoint_callback.best_model_score
    score_str = f"{best_score:.4f}" if best_score is not None else "N/A"
    print(f"Best checkpoint: {best_path} (val_loss={score_str})", file=sys.stderr)
    if best_path:
        best_tft = TemporalFusionTransformer.load_from_checkpoint(best_path)
    else:
        # Fallback: find checkpoint in lightning_logs
        import glob
        ckpts = sorted(glob.glob(str(out_dir / "lightning_logs" / "**" / "*.ckpt"), recursive=True))
        if ckpts:
            best_path = ckpts[-1]
            print(f"  Fallback checkpoint: {best_path}", file=sys.stderr)
            best_tft = TemporalFusionTransformer.load_from_checkpoint(best_path)
        else:
            print("WARNING: No checkpoint found, returning last model state", file=sys.stderr)
            best_tft = tft

    return best_tft


def tft_predict_quantiles(
    model: TemporalFusionTransformer,
    dataset: TimeSeriesDataSet,
    batch_size: int = 256,
) -> np.ndarray:
    """Run TFT inference and return quantile predictions.

    Returns: shape (n_samples, prediction_length, n_quantiles)
    """
    dl = dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
    model.eval()
    device = next(model.parameters()).device
    all_preds = []
    with torch.no_grad():
        for batch in dl:
            x, _ = batch
            # Move all tensor inputs to model's device
            x = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in x.items()}
            out = model(x)
            # out["prediction"] shape: (batch, prediction_length, n_quantiles)
            all_preds.append(out["prediction"].cpu().numpy())
    return np.concatenate(all_preds, axis=0)


# --- training command -------------------------------------------------------

def train_cmd(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_parquet = out_dir / "raw_15min.parquet"
    cache_csv = out_dir / "raw_15min.csv"

    if args.use_cache and (cache_parquet.exists() or cache_csv.exists()):
        src = cache_parquet if cache_parquet.exists() else cache_csv
        print(f"Loading cached data from {src}", file=sys.stderr)
        if src.suffix == ".parquet":
            raw_df = pd.read_parquet(src)
        else:
            raw_df = pd.read_csv(src, index_col=0, parse_dates=True)
        raw_df.index = pd.to_datetime(raw_df.index, utc=True)
    else:
        end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        raw_df = fetch_all_training_data(end)
        try:
            raw_df.to_parquet(cache_parquet)
        except Exception as exc:
            print(f"Parquet cache failed ({exc}); writing CSV instead.", file=sys.stderr)
            raw_df.to_csv(cache_csv)

    # Build TFT features (minimal — just derived columns + time encodings)
    feats = build_tft_features(raw_df)

    # Drop rows where gauge_ft is NaN (required target)
    feats = feats.dropna(subset=["gauge_ft"])
    print(f"Feature rows after dropna: {len(feats):,}", file=sys.stderr)

    # Chronological split
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

    # Prepare for TFT: add time_idx, group, impute NaN
    train_prep, fill_medians = prepare_tft_dataframe(train_df)
    val_prep, _ = prepare_tft_dataframe(val_df, fill_medians=fill_medians)
    test_prep, _ = prepare_tft_dataframe(test_df, fill_medians=fill_medians)

    # Build TimeSeriesDataSets
    encoder_length = args.encoder_length if hasattr(args, 'encoder_length') else TFT_ENCODER_LENGTH
    prediction_length = TFT_PREDICTION_LENGTH

    train_dataset = build_tft_dataset(
        train_prep, encoder_length=encoder_length,
        prediction_length=prediction_length, training=True,
    )
    val_dataset = build_tft_dataset(
        val_prep, encoder_length=encoder_length,
        prediction_length=prediction_length, training=True,
    )

    # Save dataset params for reconstruction at prediction time
    ds_params = train_dataset.get_parameters()
    (out_dir / "dataset_params.json").write_text(
        json.dumps(ds_params, indent=2, default=str)
    )

    # Train TFT
    batch_size = getattr(args, 'batch_size', TFT_BATCH_SIZE)
    hidden_size = getattr(args, 'hidden_size', TFT_HIDDEN_SIZE)
    tft_model = train_tft(
        train_dataset, val_dataset, out_dir,
        hidden_size=hidden_size,
        max_epochs=getattr(args, 'max_epochs', TFT_MAX_EPOCHS),
        batch_size=batch_size,
        num_workers=getattr(args, 'num_workers', 0),
        precision=getattr(args, 'precision', '32-true'),
        devices=getattr(args, 'devices', 1),
        pretensorize=getattr(args, 'pretensorize', False),
    )

    # Save TFT checkpoint to known location
    ckpt_path = out_dir / "tft_model.ckpt"
    trainer_ckpt = tft_model.trainer.checkpoint_callback.best_model_path if hasattr(tft_model, 'trainer') else None
    if trainer_ckpt and Path(trainer_ckpt).exists():
        import shutil
        shutil.copy2(trainer_ckpt, ckpt_path)
    else:
        torch.save(tft_model.state_dict(), ckpt_path)
    print(f"Saved TFT checkpoint to {ckpt_path}", file=sys.stderr)

    # --- Generate closure probabilities on val & test for calibration/eval ---
    quantiles = TFT_QUANTILES

    # Validation predictions
    val_preds = tft_predict_quantiles(tft_model, val_dataset, batch_size=batch_size)
    val_probs = quantiles_to_closure_prob(val_preds, quantiles)

    # Build target for val (need to align with dataset output indices)
    val_y = build_target(val_df)
    # The dataset drops early rows (need encoder_length history), align indices
    n_val_preds = len(val_probs)
    val_y_aligned = val_y.iloc[-n_val_preds:].values
    valid_mask = ~pd.isna(val_y_aligned)
    val_y_clean = val_y_aligned[valid_mask].astype(int)
    val_probs_clean = val_probs[valid_mask]

    # Isotonic calibration
    print("Fitting isotonic calibrator on validation set...", file=sys.stderr)
    calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    if val_y_clean.sum() > 0:
        calibrator.fit(val_probs_clean, val_y_clean)
    else:
        print("  WARNING: no positive examples in val set, using identity calibration", file=sys.stderr)

    # Threshold tuning
    if val_y_clean.sum() == 0:
        threshold = 0.5
    elif args.objective == "event_recall":
        # Create a mini DataFrame for event_recall tuning
        val_for_tune = val_df.iloc[-n_val_preds:][valid_mask].copy()
        val_for_tune["y"] = val_y_clean
        threshold = tune_threshold_event_recall(
            val_for_tune, val_probs_clean, min_precision=args.min_precision,
        )
    else:
        threshold = tune_threshold_f1(val_y_clean, val_probs_clean, min_precision=args.min_precision)
    print(f"tuned threshold = {threshold:.4f}  (objective={args.objective})", file=sys.stderr)

    # Test predictions
    test_dataset = build_tft_dataset(
        test_prep, encoder_length=encoder_length,
        prediction_length=prediction_length, training=True,
    )
    test_preds = tft_predict_quantiles(tft_model, test_dataset, batch_size=batch_size)
    test_probs = quantiles_to_closure_prob(test_preds, quantiles)

    test_y = build_target(test_df)
    n_test_preds = len(test_probs)
    test_y_aligned = test_y.iloc[-n_test_preds:].values
    test_valid = ~pd.isna(test_y_aligned)
    y_test = test_y_aligned[test_valid].astype(int)
    test_probs_clean = test_probs[test_valid]

    # Calibrate test probabilities
    test_probs_cal = calibrator.predict(test_probs_clean)
    test_pred = (test_probs_cal >= threshold).astype(int)

    # Metrics
    metrics = {
        "model": "TFT_v7",
        "roc_auc": float(roc_auc_score(y_test, test_probs_cal)) if y_test.sum() > 0 else None,
        "pr_auc": float(average_precision_score(y_test, test_probs_cal)) if y_test.sum() > 0 else None,
        "threshold": threshold,
        "confusion_matrix": confusion_matrix(y_test, test_pred).tolist(),
        "test_positive_rate": float(y_test.mean()),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
    }
    test_df_for_metrics = test_df.iloc[-n_test_preds:][test_valid].copy()
    test_df_for_metrics["y"] = y_test
    metrics.update({f"event_{k}": v for k, v in event_metrics(
        test_df_for_metrics, test_probs_cal, threshold).items()})
    metrics["horizon_h"] = HORIZON_H
    metrics["closure_ft"] = CLOSURE_FT
    metrics["rain_gauges"] = [name for name, _ in RAIN_GAUGES]
    metrics["encoder_length"] = encoder_length
    metrics["prediction_length"] = prediction_length
    metrics["quantiles"] = quantiles
    metrics["n_parameters"] = sum(p.numel() for p in tft_model.parameters())
    metrics["train_start"] = train_df.index.min().isoformat()
    metrics["train_end"] = train_df.index.max().isoformat()
    metrics["val_start"] = val_df.index.min().isoformat() if len(val_df) else None
    metrics["val_end"] = val_df.index.max().isoformat() if len(val_df) else None
    metrics["test_start"] = test_df.index.min().isoformat()
    metrics["test_end"] = test_df.index.max().isoformat()

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(json.dumps(metrics, indent=2))

    # Save calibrator
    joblib.dump(calibrator, out_dir / "calibrator.joblib")

    # Save TFTBundle
    bundle = TFTBundle(
        model_path=str(ckpt_path),
        threshold=threshold,
        target_col="gauge_ft",
        fill_medians=fill_medians,
    )
    save_tft_bundle(bundle, out_dir)

    # --- Plots ---
    if y_test.sum() > 0:
        fpr, tpr, _ = roc_curve(y_test, test_probs_cal)
        plt.figure(figsize=(5, 5))
        plt.plot(fpr, tpr, label=f"AUC={metrics['roc_auc']:.3f}")
        plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC (test) — TFT v7"); plt.legend()
        plt.tight_layout(); plt.savefig(out_dir / "roc.png", dpi=120); plt.close()

        prec, rec, _ = precision_recall_curve(y_test, test_probs_cal)
        plt.figure(figsize=(5, 5))
        plt.plot(rec, prec, label=f"AP={metrics['pr_auc']:.3f}")
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR (test) — TFT v7"); plt.legend()
        plt.tight_layout(); plt.savefig(out_dir / "pr.png", dpi=120); plt.close()

    test_df_plot = test_df.iloc[-n_test_preds:][test_valid]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(test_df_plot.index, test_df_plot["gauge_ft"], lw=0.8, label="gauge_ft")
    ax.axhline(CLOSURE_FT, color="red", ls="--", alpha=0.5, label="5 ft closure")
    alert_mask = test_pred.astype(bool)
    alert_idx = test_df_plot.index[alert_mask]
    if len(alert_idx):
        ax.scatter(alert_idx, np.full(len(alert_idx), CLOSURE_FT + 0.1),
                   s=6, color="orange", label="alert")
    ax.set_title("Test window — TFT v7"); ax.legend(loc="upper right")
    plt.tight_layout(); plt.savefig(out_dir / "timeline.png", dpi=120); plt.close()

    print(f"Wrote TFT model and artifacts to {out_dir}", file=sys.stderr)


# --- predict ---------------------------------------------------------------

def _load_tft_for_prediction(model_dir: Path):
    """Load TFTBundle, TFT model, calibrator, and dataset params for prediction."""
    bundle = load_tft_bundle(model_dir / "bundle.json")
    ckpt_path = model_dir / "tft_model.ckpt"

    # Load dataset params
    ds_params_path = model_dir / "dataset_params.json"
    ds_params = json.loads(ds_params_path.read_text()) if ds_params_path.exists() else None

    # Load model from checkpoint
    tft_model = TemporalFusionTransformer.load_from_checkpoint(str(ckpt_path))
    tft_model.eval()

    # Load calibrator
    cal_path = model_dir / "calibrator.joblib"
    calibrator = joblib.load(cal_path) if cal_path.exists() else None

    return bundle, tft_model, calibrator, ds_params


def _fetch_recent_and_build_features(bundle: TFTBundle, hours: int = 200):
    """Fetch recent data, merge, and build TFT features. Returns (feats_df, rain_raw)."""
    gauge_raw = fetch_gauge_recent(hours=hours)
    q_raw = fetch_discharge_recent(hours=hours)
    rain_raw = fetch_all_rain_recent(hours=hours)
    tide_obs, tide_pred = fetch_tide_recent(hours=hours)
    nws = fetch_nws_forecast()
    wx = fetch_weather_recent(hours=hours)
    try:
        bridge_raw = fetch_bridge_gauge_recent(hours=hours)
    except Exception as e:
        print(f"Bridge gauge failed: {e}", file=sys.stderr)
        bridge_raw = None
    try:
        soil = fetch_soil_moisture_recent(hours=hours)
    except Exception as e:
        print(f"Soil moisture failed: {e}", file=sys.stderr)
        soil = None
    try:
        sat_precip = fetch_nasa_power_precip_recent(days=14)
    except Exception as e:
        print(f"Satellite precip failed: {e}", file=sys.stderr)
        sat_precip = None
    try:
        gw = fetch_google_weather()
    except Exception as e:
        print(f"Google Weather failed: {e}", file=sys.stderr)
        gw = None

    merged = to_hourly(
        gauge_raw, rain_raw, q_raw=q_raw, tide_obs=tide_obs, tide_pred=tide_pred,
        nws=nws, weather=wx, bridge_raw=bridge_raw, soil_moisture=soil,
        sat_precip=sat_precip, google_weather=gw,
    )

    # Trim to last actual rain observation
    last_rain_ts = None
    for name in rain_raw:
        if not rain_raw[name].empty:
            ts = rain_raw[name].index[-1]
            if last_rain_ts is None or ts > last_rain_ts:
                last_rain_ts = ts
    if last_rain_ts is not None and not merged.empty:
        freq_str = f"{TIMESTEP_MIN}min"
        cutoff = last_rain_ts.floor(freq_str)
        merged = merged.loc[merged.index <= cutoff]

    feats = build_tft_features(merged)
    return feats, rain_raw


def predict_cmd(args):
    model_dir = Path(args.model)
    if model_dir.is_file():
        model_dir = model_dir.parent
    bundle, tft_model, calibrator, ds_params = _load_tft_for_prediction(model_dir)

    feats, rain_raw = _fetch_recent_and_build_features(bundle)
    if feats.empty or len(feats) < bundle.encoder_length + bundle.prediction_length:
        print(json.dumps({"error": "insufficient recent data"}))
        return

    # Prepare for TFT
    prep, _ = prepare_tft_dataframe(feats, fill_medians=bundle.fill_medians)

    # Build prediction dataset (use last encoder_length + prediction_length rows)
    min_rows = bundle.encoder_length + bundle.prediction_length
    if len(prep) > min_rows * 2:
        prep = prep.iloc[-min_rows * 2:]
        prep["time_idx"] = np.arange(len(prep))

    pred_dataset = build_tft_dataset(
        prep, encoder_length=bundle.encoder_length,
        prediction_length=bundle.prediction_length, training=True,
    )

    # Predict
    preds = tft_predict_quantiles(tft_model, pred_dataset)
    if len(preds) == 0:
        print(json.dumps({"error": "TFT produced no predictions"}))
        return

    # Take the last prediction (most recent)
    last_pred = preds[-1:]  # (1, prediction_length, n_quantiles)
    raw_prob = float(quantiles_to_closure_prob(last_pred, bundle.quantiles)[0])
    prob = float(calibrator.predict([raw_prob])[0]) if calibrator else raw_prob
    alert = bool(prob >= bundle.threshold)

    latest = feats.iloc[-1]
    gauge_now = float(latest["gauge_ft"])

    # Compute 6h rain totals from raw rain columns
    S = STEPS_PER_HOUR
    rain_totals_6h = {}
    for name, _ in RAIN_GAUGES:
        col = _rain_col(name)
        if col in feats.columns:
            rain_totals_6h[name] = float(feats[col].iloc[-6*S:].sum())
        else:
            rain_totals_6h[name] = 0.0

    out = {
        "timestamp_utc": feats.index[-1].isoformat(),
        "horizon_h": bundle.horizon_h,
        "gauge_ft_now": gauge_now,
        f"prob_closure_in_next_{bundle.horizon_h}h": round(prob, 4),
        "threshold": round(bundle.threshold, 4),
        "alert": alert,
        "already_above_closure": bool(gauge_now > bundle.closure_ft),
        "rain_last_6h_in": {k: round(v, 2) for k, v in rain_totals_6h.items()},
    }
    print(json.dumps(out, indent=2))


# --- backtest --------------------------------------------------------------

def backtest_cmd(args):
    model_dir = Path(args.model)
    if model_dir.is_file():
        model_dir = model_dir.parent
    bundle, tft_model, calibrator, ds_params = _load_tft_for_prediction(model_dir)

    cache_parquet = model_dir / "raw_15min.parquet"
    cache_csv = model_dir / "raw_15min.csv"
    if cache_parquet.exists():
        raw_df = pd.read_parquet(cache_parquet)
    elif cache_csv.exists():
        raw_df = pd.read_csv(cache_csv, index_col=0, parse_dates=True)
    else:
        raise FileNotFoundError("No cached dataset next to bundle.json; rerun train.")
    raw_df.index = pd.to_datetime(raw_df.index, utc=True)

    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")

    # Need encoder_length of history before start
    lookback = pd.Timedelta(minutes=TIMESTEP_MIN * bundle.encoder_length * 2)
    window = raw_df.loc[(raw_df.index >= start - lookback) & (raw_df.index < end)]

    feats = build_tft_features(window)
    prep, _ = prepare_tft_dataframe(feats, fill_medians=bundle.fill_medians)

    bt_dataset = build_tft_dataset(
        prep, encoder_length=bundle.encoder_length,
        prediction_length=bundle.prediction_length, training=True,
    )

    preds = tft_predict_quantiles(tft_model, bt_dataset)
    probs = quantiles_to_closure_prob(preds, bundle.quantiles)
    if calibrator:
        probs = calibrator.predict(probs)

    # Build target for the backtest window
    y = build_target(feats)
    # Align: predictions correspond to the tail of the feature df
    n_preds = len(probs)
    bt_feats = feats.iloc[-n_preds:].copy()
    bt_y = y.iloc[-n_preds:].values
    valid = ~pd.isna(bt_y)
    bt_feats = bt_feats[valid]
    probs_valid = probs[valid]
    y_valid = bt_y[valid].astype(int)

    # Filter to requested date range
    in_range = (bt_feats.index >= start) & (bt_feats.index < end)
    bt_feats = bt_feats[in_range]
    probs_valid = probs_valid[in_range.values[valid][:len(probs_valid)]] if len(probs_valid) > 0 else probs_valid
    y_valid = y_valid[in_range.values[valid][:len(y_valid)]] if len(y_valid) > 0 else y_valid

    bt_feats = bt_feats.assign(
        prob=probs_valid[:len(bt_feats)],
        alert=(probs_valid[:len(bt_feats)] >= bundle.threshold).astype(int),
        y=y_valid[:len(bt_feats)],
    )

    evt = event_metrics(bt_feats, bt_feats["prob"].values, bundle.threshold)
    summary = {
        "rows": int(len(bt_feats)),
        "alert_rate": float(bt_feats["alert"].mean()) if len(bt_feats) else 0,
        "actual_positive_rate": float(bt_feats["y"].mean()) if len(bt_feats) else 0,
        **evt,
    }
    print(json.dumps(summary, indent=2))
    bt_feats[["gauge_ft", "prob", "alert", "y"]].to_csv(args.out)
    print(f"Wrote {args.out}", file=sys.stderr)


# --- cli -------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Hanalei flood predictor v7 (TFT)")
    sub = p.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("train")
    pt.add_argument("--out-dir", default="./hanalei_v7_out")
    pt.add_argument("--use-cache", action="store_true")
    pt.add_argument("--test-years", type=float, default=5.0)
    pt.add_argument("--val-years", type=float, default=2.0)
    pt.add_argument("--min-precision", type=float, default=0.15)
    pt.add_argument("--objective", choices=["event_recall", "f1"], default="event_recall")
    pt.add_argument("--max-epochs", type=int, default=TFT_MAX_EPOCHS)
    pt.add_argument("--batch-size", type=int, default=TFT_BATCH_SIZE)
    pt.add_argument("--encoder-length", type=int, default=TFT_ENCODER_LENGTH)
    pt.add_argument("--num-workers", type=int, default=0,
                    help="DataLoader workers (0=main process, 8+ for cloud GPU)")
    pt.add_argument("--precision", type=str, default="32-true",
                    choices=["32-true", "16-mixed", "bf16-mixed"],
                    help="Training precision (bf16-mixed for H200/A100)")
    pt.add_argument("--hidden-size", type=int, default=TFT_HIDDEN_SIZE)
    pt.add_argument("--devices", type=int, default=1,
                    help="Number of GPUs to use (>1 enables DDP)")
    pt.add_argument("--pretensorize", action="store_true",
                    help="Pre-compute all dataset samples into RAM tensors (faster training)")
    pt.set_defaults(func=train_cmd)

    pp = sub.add_parser("predict")
    pp.add_argument("--model", required=True, help="Path to model dir or bundle.json")
    pp.set_defaults(func=predict_cmd)

    pb = sub.add_parser("backtest")
    pb.add_argument("--model", required=True, help="Path to model dir or bundle.json")
    pb.add_argument("--start", required=True)
    pb.add_argument("--end", required=True)
    pb.add_argument("--out", default="backtest.csv")
    pb.set_defaults(func=backtest_cmd)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
