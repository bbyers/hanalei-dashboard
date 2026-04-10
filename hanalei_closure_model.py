#!/usr/bin/env python3
"""Hanalei road closure predictor (v6: bridge gauge + soil moisture + satellite precip).

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
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # plotting not available (web deployment)
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
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


def build_feature_columns() -> list[str]:
    cols = [
        # --- stream gauge (upstream, USGS 16103000) ---
        "gauge_ft",
        "gauge_lag_1", "gauge_lag_2", "gauge_lag_3",
        "gauge_lag_6", "gauge_lag_12", "gauge_lag_24",
        "gauge_delta_1", "gauge_delta_3", "gauge_delta_6",
        # --- gauge dynamics (v5) ---
        "gauge_accel_1",       # acceleration: delta_1 - prev delta_1 (rate of rise change)
        "gauge_accel_3",       # acceleration over 3h
        "gauge_max_rise_6h",   # max hourly rise in last 6h (rapid-rise indicator)
        # --- bridge gauge (v6, downstream, USGS 16104200 at Hwy 56) ---
        "bridge_ft",           # water level at actual bridge
        "bridge_lag_1", "bridge_lag_3", "bridge_lag_6",
        "bridge_delta_1", "bridge_delta_3",
        "bridge_above_7ft",    # NWS minor flood stage at bridge
        "bridge_gauge_diff",   # upstream - bridge difference (backwater indicator)
        # --- discharge ---
        "q_cfs", "q_lag_1", "q_lag_3", "q_lag_6",
        "q_delta_1", "q_delta_3",
    ]
    # --- per-gauge rain rolling sums ---
    for name, _ in RAIN_GAUGES:
        for w in RAIN_WINDOWS_PER_GAUGE:
            cols.append(f"{_rain_col(name)}_sum_{w}")
    # --- per-gauge rain intensity (v5): max hourly rate in window ---
    for name, _ in RAIN_GAUGES:
        for w in (3, 6):
            cols.append(f"{_rain_col(name)}_max_{w}")
    # --- cross-gauge aggregates (spatial max + total, including long antecedent) ---
    for w in RAIN_WINDOWS_XGAUGE:
        cols.append(f"rain_max_sum_{w}")
        cols.append(f"rain_total_sum_{w}")
    # --- cross-gauge intensity (v5) ---
    for w in (3, 6):
        cols.append(f"rain_max_rate_{w}")      # max of per-gauge max-rate
    # --- soil saturation (v6): ERA5 soil moisture from Open-Meteo ---
    cols += [
        "soil_moisture_shallow",  # 0-7 cm, m³/m³ (surface response to recent rain)
        "soil_moisture_mid",      # 7-28 cm, m³/m³ (shallow root zone)
        "soil_moisture_deep",     # 28-100 cm, m³/m³ (deep saturation)
        "soil_moisture_delta_24", # change in shallow moisture over 24h
    ]
    # --- soil saturation proxy (v5): exponentially-decayed rain accumulation ---
    cols += [
        "rain_ema_48h",        # half-life 48h exponential moving average
        "rain_ema_168h",       # half-life 168h (7-day)
    ]
    # --- satellite precip (v6): NASA POWER / IMERG daily gridded precip ---
    cols += [
        "sat_precip_mm",       # daily satellite precip (spread to hourly)
        "sat_precip_sum_3d",   # 3-day rolling sum of satellite precip
        "sat_precip_sum_7d",   # 7-day rolling sum
    ]
    # --- interaction features (v5) ---
    cols += [
        "rain_x_saturation",   # current rain × soil saturation → nonlinear flood risk
        "rain_x_tide",         # current rain × tide level → compound flooding
        "surge_x_rain",        # storm surge × rain → coastal + fluvial compound
        "rain_x_soil_moisture", # (v6) rain × actual soil moisture
    ]
    # --- tide ---
    cols += [
        "tide_ft",               # observed water level (MLLW)
        "tide_pred_ft",          # harmonic prediction (astronomical tide)
        "storm_surge_ft",        # observed - predicted (weather-driven anomaly)
        "tide_delta_3",          # tide change over 3h (rising vs falling)
    ]
    # --- atmospheric / weather (ERA5 historical for training, Open-Meteo + NWS for prediction) ---
    cols += [
        "wx_humidity_pct",       # relative humidity (%)
        "wx_wind_kmh",           # wind speed (km/h)
        "wx_wind_gust_kmh",     # wind gusts (km/h)
        "wx_pressure_hpa",       # mean sea level pressure (hPa)
        "wx_pressure_delta_3",   # pressure change over 3h (falling = storm)
        "wx_pressure_delta_6",   # pressure change over 6h
        "wx_precip_era5_mm",     # ERA5 gridded precip (mm) — coarse but available historically
        "wx_precip_era5_sum_3",  # 3h rolling sum of ERA5 precip
        "wx_precip_era5_sum_6",  # 6h rolling sum of ERA5 precip
    ]
    # --- NWS forecast-only (available at prediction time; NaN during training) ---
    cols += [
        "nws_qpf_3h",      # NWS forecast QPF next 3h (mm)
        "nws_qpf_6h",      # NWS forecast QPF next 6h (mm)
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
) -> pd.DataFrame:
    """Resample all sources to aligned 15-minute records (v6).

    USGS data is natively 15-min, so gauge/discharge/rain keep full resolution.
    Hourly sources (ERA5, NWS, soil moisture) are forward-filled to 15-min.
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

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features to a 15-min df (v6)."""
    S = STEPS_PER_HOUR  # 4 steps per hour
    out = df.copy()
    g = out["gauge_ft"]

    # --- gauge features (lags in hours, converted to steps) ---
    for lag_h in (1, 2, 3, 6, 12, 24):
        out[f"gauge_lag_{lag_h}"] = g.shift(lag_h * S)
    for d_h in (1, 3, 6):
        out[f"gauge_delta_{d_h}"] = g - g.shift(d_h * S)

    # --- gauge dynamics (v5) ---
    d1 = g - g.shift(1 * S)
    out["gauge_accel_1"] = d1 - d1.shift(1 * S)         # acceleration (2nd derivative)
    d3 = g - g.shift(3 * S)
    out["gauge_accel_3"] = d3 - d3.shift(3 * S)
    out["gauge_max_rise_6h"] = d1.rolling(6 * S, min_periods=1).max()  # max hourly rise in 6h

    # --- bridge gauge features (v6) ---
    if "bridge_ft" in out.columns:
        b = out["bridge_ft"]
        for lag_h in (1, 3, 6):
            out[f"bridge_lag_{lag_h}"] = b.shift(lag_h * S)
        for d_h in (1, 3):
            out[f"bridge_delta_{d_h}"] = b - b.shift(d_h * S)
        out["bridge_above_7ft"] = (b > BRIDGE_FLOOD_FT).astype(int)
        out["bridge_gauge_diff"] = g - b  # upstream vs bridge (backwater indicator)
    else:
        for col in ["bridge_ft", "bridge_lag_1", "bridge_lag_3", "bridge_lag_6",
                     "bridge_delta_1", "bridge_delta_3", "bridge_above_7ft",
                     "bridge_gauge_diff"]:
            out[col] = np.nan

    # --- discharge features ---
    if "q_cfs" in out.columns:
        q = out["q_cfs"]
        for lag_h in (1, 3, 6):
            out[f"q_lag_{lag_h}"] = q.shift(lag_h * S)
        for d_h in (1, 3):
            out[f"q_delta_{d_h}"] = q - q.shift(d_h * S)
    else:
        for col in ["q_cfs", "q_lag_1", "q_lag_3", "q_lag_6", "q_delta_1", "q_delta_3"]:
            out[col] = np.nan

    # --- per-gauge rain rolling sums (window in hours → steps) ---
    for name, _ in RAIN_GAUGES:
        col = _rain_col(name)
        for w_h in RAIN_WINDOWS_PER_GAUGE:
            out[f"{col}_sum_{w_h}"] = out[col].rolling(w_h * S, min_periods=1).sum()

    # --- per-gauge rain intensity (v5): max per-step rate in window ---
    for name, _ in RAIN_GAUGES:
        col = _rain_col(name)
        for w_h in (3, 6):
            out[f"{col}_max_{w_h}"] = out[col].rolling(w_h * S, min_periods=1).max()

    # --- cross-gauge aggregates (including long antecedent windows) ---
    for w_h in RAIN_WINDOWS_XGAUGE:
        per_gauge_sums: list[pd.Series] = []
        for name, _ in RAIN_GAUGES:
            col = _rain_col(name)
            per_gauge_sums.append(out[col].rolling(w_h * S, min_periods=1).sum())
        stacked = pd.concat(per_gauge_sums, axis=1)
        out[f"rain_max_sum_{w_h}"] = stacked.max(axis=1)
        out[f"rain_total_sum_{w_h}"] = stacked.sum(axis=1)

    # --- cross-gauge intensity (v5): max of per-gauge max-rate ---
    for w_h in (3, 6):
        max_rates = []
        for name, _ in RAIN_GAUGES:
            col = _rain_col(name)
            max_rates.append(out[col].rolling(w_h * S, min_periods=1).max())
        out[f"rain_max_rate_{w_h}"] = pd.concat(max_rates, axis=1).max(axis=1)

    # --- soil saturation proxy (v5): exponentially-decayed rain accumulation ---
    # Use total cross-gauge rain, apply EMA with specified half-life
    total_rain = sum(
        out[_rain_col(name)].fillna(0) for name, _ in RAIN_GAUGES
    )
    out["rain_ema_48h"] = total_rain.ewm(halflife=48 * S, min_periods=1).mean()
    out["rain_ema_168h"] = total_rain.ewm(halflife=168 * S, min_periods=1).mean()

    # --- soil moisture features (v6, ERA5 via Open-Meteo) ---
    if "soil_moisture_shallow" in out.columns:
        sm = out["soil_moisture_shallow"]
        out["soil_moisture_delta_24"] = sm - sm.shift(24 * S)
    else:
        for col in ["soil_moisture_shallow", "soil_moisture_mid",
                     "soil_moisture_deep", "soil_moisture_delta_24"]:
            out[col] = np.nan

    # --- satellite precip features (v6, NASA POWER / IMERG) ---
    if "sat_precip_mm" in out.columns:
        sp = out["sat_precip_mm"].fillna(0)
        out["sat_precip_sum_3d"] = sp.rolling(72 * S, min_periods=1).sum()   # 3 days
        out["sat_precip_sum_7d"] = sp.rolling(168 * S, min_periods=1).sum()  # 7 days
    else:
        out["sat_precip_mm"] = np.nan
        out["sat_precip_sum_3d"] = np.nan
        out["sat_precip_sum_7d"] = np.nan

    # --- tide features ---
    if "tide_ft" in out.columns:
        out["storm_surge_ft"] = out.get("tide_ft", 0) - out.get("tide_pred_ft", 0)
        out["tide_delta_3"] = out["tide_ft"] - out["tide_ft"].shift(3 * S)
    else:
        for col in ["tide_ft", "tide_pred_ft", "storm_surge_ft", "tide_delta_3"]:
            out[col] = np.nan

    # --- atmospheric / weather features ---
    # These columns come from ERA5 (training) or Open-Meteo + NWS (prediction).
    # Pressure deltas are strong indicators of approaching storms.
    if "wx_pressure_hpa" in out.columns:
        p = out["wx_pressure_hpa"]
        out["wx_pressure_delta_3"] = p - p.shift(3 * S)
        out["wx_pressure_delta_6"] = p - p.shift(6 * S)
    else:
        for col in ["wx_humidity_pct", "wx_wind_kmh", "wx_wind_gust_kmh",
                     "wx_pressure_hpa", "wx_pressure_delta_3", "wx_pressure_delta_6"]:
            if col not in out.columns:
                out[col] = np.nan

    # ERA5 precip rolling sums
    if "wx_precip_era5_mm" in out.columns:
        ep = out["wx_precip_era5_mm"]
        out["wx_precip_era5_sum_3"] = ep.rolling(3 * S, min_periods=1).sum()
        out["wx_precip_era5_sum_6"] = ep.rolling(6 * S, min_periods=1).sum()
    else:
        out["wx_precip_era5_mm"] = np.nan
        out["wx_precip_era5_sum_3"] = np.nan
        out["wx_precip_era5_sum_6"] = np.nan

    # --- NWS forecast-only features (forward-looking QPF) ---
    if "qpf_mm" in out.columns:
        qpf_rev = out["qpf_mm"].iloc[::-1]
        out["nws_qpf_3h"] = qpf_rev.rolling(3 * S, min_periods=1).sum().iloc[::-1].values
        out["nws_qpf_6h"] = qpf_rev.rolling(6 * S, min_periods=1).sum().iloc[::-1].values
    else:
        out["nws_qpf_3h"] = np.nan
        out["nws_qpf_6h"] = np.nan

    # --- time / indicator ---
    hour = out.index.hour
    month = out.index.month
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    out["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)

    out["above_3ft"] = (g > 3.0).astype(int)
    out["above_4ft"] = (g > 4.0).astype(int)

    # --- interaction features (v5) ---
    # Rain × soil saturation: saturated soil + heavy rain → much higher flood risk
    rain_now = out["rain_max_sum_3"].fillna(0)
    out["rain_x_saturation"] = rain_now * out["rain_ema_48h"].fillna(0)
    # Rain × tide: high tide impedes drainage, amplifying flood from rain
    out["rain_x_tide"] = rain_now * out["tide_ft"].fillna(0)
    # Storm surge × rain: compound coastal + fluvial flooding
    out["surge_x_rain"] = out["storm_surge_ft"].fillna(0) * rain_now
    # Rain × actual soil moisture (v6)
    out["rain_x_soil_moisture"] = rain_now * out.get("soil_moisture_shallow", pd.Series(0, index=out.index)).fillna(0)

    return out


def build_target(df: pd.DataFrame) -> pd.Series:
    """closure_in_next_H[t] = 1 iff max(gauge[t..t+H]) > CLOSURE_FT.

    Includes the current hour (shift 0) so that "gauge is already at flood
    stage" is a positive label — the model learns to output ~1.0 when the
    river is already above 5 ft.
    """
    g = df["gauge_ft"]
    shifts = pd.concat([g.shift(-i) for i in range(0, HORIZON_STEPS + 1)], axis=1)
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
    # NWS forecast columns are NaN during training (no historical data) — that's OK,
    # HistGradientBoosting handles NaN natively. Exclude them from dropna.
    # These features may legitimately be NaN (no historical NWS, bridge gaps,
    # satellite precip latency, soil moisture gaps). HistGBT handles NaN natively.
    NAN_OK = {"nws_qpf_3h", "nws_qpf_6h",
              "bridge_ft", "bridge_lag_1", "bridge_lag_3", "bridge_lag_6",
              "bridge_delta_1", "bridge_delta_3", "bridge_above_7ft", "bridge_gauge_diff",
              "sat_precip_mm", "sat_precip_sum_3d", "sat_precip_sum_7d",
              "soil_moisture_shallow", "soil_moisture_mid", "soil_moisture_deep",
              "soil_moisture_delta_24", "rain_x_soil_moisture"}
    dropna_cols = [c for c in FEATURE_COLUMNS if c not in NAN_OK] + ["y"]
    feats = feats.dropna(subset=dropna_cols)
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

    # --- isotonic calibration on validation set ---
    # Maps raw model scores → actual closure probabilities so the dashboard
    # displays percentages that match empirical rates.
    print("Fitting isotonic calibrator on validation set...", file=sys.stderr)
    calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    calibrator.fit(val_prob, y_val)
    # Show a few calibration examples
    sample_raws = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95]
    for r in sample_raws:
        print(f"  raw {r:.2f} → calibrated {calibrator.predict([r])[0]:.4f}", file=sys.stderr)

    # Also calibrate test metrics for reporting
    test_prob_cal = calibrator.predict(test_prob)
    metrics["calibration_examples"] = {
        f"raw_{r:.2f}": round(float(calibrator.predict([r])[0]), 4) for r in sample_raws
    }

    bundle = TrainBundle(
        model=model,
        threshold=threshold,
        features=list(FEATURE_COLUMNS),
        closure_ft=CLOSURE_FT,
        horizon_h=HORIZON_H,
        rain_gauge_names=[name for name, _ in RAIN_GAUGES],
    )
    bundle.calibrator = calibrator  # attach calibrator to bundle
    joblib.dump(bundle, out_dir / "model.joblib")
    joblib.dump(calibrator, out_dir / "calibrator.joblib")  # standalone backup

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
    nws = fetch_nws_forecast()
    wx = fetch_weather_recent(hours=200)
    try:
        bridge_raw = fetch_bridge_gauge_recent(hours=200)
    except Exception as e:
        print(f"Bridge gauge failed: {e}", file=sys.stderr)
        bridge_raw = None
    try:
        soil = fetch_soil_moisture_recent(hours=200)
    except Exception as e:
        print(f"Soil moisture failed: {e}", file=sys.stderr)
        soil = None
    try:
        sat_precip = fetch_nasa_power_precip_recent(days=14)
    except Exception as e:
        print(f"Satellite precip failed: {e}", file=sys.stderr)
        sat_precip = None
    hourly = to_hourly(
        gauge_raw, rain_raw, q_raw=q_raw, tide_obs=tide_obs, tide_pred=tide_pred,
        nws=nws, weather=wx, bridge_raw=bridge_raw, soil_moisture=soil, sat_precip=sat_precip,
    )
    # Trim trailing hours beyond the last actual rain observation (USGS lags 1-2h)
    last_rain_ts = None
    for name in rain_raw:
        if not rain_raw[name].empty:
            ts = rain_raw[name].index[-1]
            if last_rain_ts is None or ts > last_rain_ts:
                last_rain_ts = ts
    if last_rain_ts is not None and not hourly.empty:
        cutoff = last_rain_ts.floor("h")
        hourly = hourly.loc[hourly.index <= cutoff]
    # These features may be NaN at prediction time — model handles NaN natively
    NAN_OK = {"nws_qpf_3h", "nws_qpf_6h",
              "bridge_ft", "bridge_lag_1", "bridge_lag_3", "bridge_lag_6",
              "bridge_delta_1", "bridge_delta_3", "bridge_above_7ft", "bridge_gauge_diff",
              "sat_precip_mm", "sat_precip_sum_3d", "sat_precip_sum_7d",
              "soil_moisture_shallow", "soil_moisture_mid", "soil_moisture_deep",
              "soil_moisture_delta_24", "rain_x_soil_moisture"}
    dropna_cols = [c for c in bundle.features if c not in NAN_OK]
    feats = build_features(hourly).dropna(subset=dropna_cols)
    if feats.empty:
        print(json.dumps({"error": "insufficient recent data"}))
        return
    latest = feats.iloc[[-1]]
    x = latest[bundle.features].values
    raw_prob = float(bundle.model.predict_proba(x)[0, 1])
    calibrator = getattr(bundle, "calibrator", None)
    prob = float(calibrator.predict([raw_prob])[0]) if calibrator else raw_prob
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
