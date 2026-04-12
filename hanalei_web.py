#!/usr/bin/env python3
"""Hanalei Bridge Closure Probability — Live Dashboard (v7: TFT).

A Flask web app that displays a running probability of Hanalei River bridge
closure (gauge > 5.0 ft) within the next 3 hours, powered by the v7 TFT model.

Usage:
    python hanalei_web.py [--model ./hanalei_v7_out] [--port 5000]
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from flask import Flask, jsonify, render_template_string

print("[startup] importing hanalei_closure_model...", file=sys.stderr, flush=True)
try:
    from hanalei_closure_model import (
        RAIN_GAUGES,
        STEPS_PER_HOUR,
        TIMESTEP_MIN,
        TFTBundle,
        TFT_QUANTILES,
        build_tft_features,
        fetch_all_rain_recent,
        fetch_bridge_gauge_recent,
        fetch_discharge_recent,
        fetch_gauge_recent,
        fetch_google_weather,
        fetch_nasa_power_precip_recent,
        fetch_nws_forecast,
        fetch_soil_moisture_recent,
        fetch_tide_recent,
        fetch_weather_recent,
        load_tft_bundle,
        prepare_tft_dataframe,
        build_tft_dataset,
        quantiles_to_closure_prob,
        tft_predict_quantiles,
        to_hourly,
        _rain_col,
    )
    from pytorch_forecasting import TemporalFusionTransformer
    # Legacy alias
    TrainBundle = TFTBundle
    build_features = build_tft_features
    print("[startup] import OK", file=sys.stderr, flush=True)
except Exception as e:
    print(f"[startup] IMPORT FAILED: {e}", file=sys.stderr, flush=True)
    traceback.print_exc(file=sys.stderr)
    raise

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

app = Flask(__name__)

_bundle: TFTBundle | None = None
_tft_model: TemporalFusionTransformer | None = None
_calibrator = None
_latest: dict = {"status": "loading", "message": "Fetching initial data..."}
_history: list[dict] = []          # last 72 hours of predictions
_lock = threading.Lock()
_refresh_event = threading.Event()  # signal background thread to re-fetch now
_REFRESH_SECONDS = 300              # auto-refresh every 5 minutes
_fetching = False                   # True while a fetch is in progress


# ---------------------------------------------------------------------------
# Background prediction loop
# ---------------------------------------------------------------------------

def _log(msg):
    print(msg, file=sys.stderr, flush=True)


def _run_prediction() -> dict:
    """Run one prediction cycle using TFT, return result dict."""
    bundle = _bundle
    model = _tft_model
    calibrator = _calibrator

    _log("[predict] fetching gauge...")
    gauge_raw = fetch_gauge_recent(hours=300)
    _log(f"[predict] gauge: {len(gauge_raw)} rows")
    _log("[predict] fetching discharge...")
    q_raw = fetch_discharge_recent(hours=300)
    _log(f"[predict] discharge: {len(q_raw)} rows")
    _log("[predict] fetching rain...")
    rain_raw = fetch_all_rain_recent(hours=300)
    _log(f"[predict] rain: {sum(len(v) for v in rain_raw.values())} total rows")
    _log("[predict] fetching tide...")
    tide_obs, tide_pred = fetch_tide_recent(hours=300)
    _log(f"[predict] tide obs: {len(tide_obs)}, pred: {len(tide_pred)}")
    try:
        _log("[predict] fetching NWS...")
        nws = fetch_nws_forecast()
    except Exception as e:
        _log(f"[predict] NWS failed: {e}")
        nws = None
    try:
        _log("[predict] fetching weather...")
        wx = fetch_weather_recent(hours=300)
    except Exception as e:
        _log(f"[predict] weather failed: {e}")
        wx = pd.DataFrame()
    bridge_raw = None
    try:
        _log("[predict] fetching bridge gauge...")
        bridge_raw = fetch_bridge_gauge_recent(hours=300)
        _log(f"[predict] bridge gauge: {len(bridge_raw)} rows")
    except Exception as e:
        _log(f"[predict] bridge gauge failed: {e}")
    soil = None
    try:
        _log("[predict] fetching soil moisture...")
        soil = fetch_soil_moisture_recent(hours=300)
        _log(f"[predict] soil moisture: {len(soil)} rows")
    except Exception as e:
        _log(f"[predict] soil moisture failed: {e}")
    sat_precip = None
    try:
        _log("[predict] fetching satellite precip...")
        sat_precip = fetch_nasa_power_precip_recent(days=14)
        _log(f"[predict] sat precip: {len(sat_precip) if sat_precip is not None else 0} rows")
    except Exception as e:
        _log(f"[predict] satellite precip failed: {e}")
    gw = None
    try:
        _log("[predict] fetching Google Weather...")
        gw = fetch_google_weather()
        _log(f"[predict] google weather: {len(gw) if gw is not None else 0} rows")
    except Exception as e:
        _log(f"[predict] Google Weather failed: {e}")

    _log("[predict] building 15-min merged data...")
    merged = to_hourly(
        gauge_raw, rain_raw, q_raw=q_raw, tide_obs=tide_obs, tide_pred=tide_pred,
        nws=nws, weather=wx, bridge_raw=bridge_raw, soil_moisture=soil,
        sat_precip=sat_precip, google_weather=gw,
    )
    # Trim trailing rows beyond the last actual rain observation (USGS lags 1-2h)
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

    _log(f"[predict] merged rows: {len(merged)}, cols: {list(merged.columns)[:5]}...")

    feats = build_tft_features(merged)
    feats = feats.dropna(subset=["gauge_ft"])
    _log(f"[predict] feats rows: {len(feats)}")

    min_rows = bundle.encoder_length + bundle.prediction_length
    if len(feats) < min_rows:
        _log(f"[predict] ERROR: need {min_rows} rows, only have {len(feats)}")
        return {"status": "error", "message": "Insufficient recent data from USGS/NOAA"}

    # Prepare for TFT
    prep, _ = prepare_tft_dataframe(feats, fill_medians=bundle.fill_medians)

    # Use last N rows for prediction (keep enough for encoder + decoder)
    max_keep = 5 * 24 * STEPS_PER_HOUR + bundle.encoder_length
    if len(prep) > max_keep:
        prep = prep.iloc[-max_keep:]
        prep["time_idx"] = np.arange(len(prep))

    pred_dataset = build_tft_dataset(
        prep, encoder_length=bundle.encoder_length,
        prediction_length=bundle.prediction_length, training=True,
    )

    _log("[predict] running TFT inference...")
    preds = tft_predict_quantiles(model, pred_dataset)
    if len(preds) == 0:
        return {"status": "error", "message": "TFT produced no predictions"}

    # Latest prediction
    last_pred = preds[-1:]
    raw_prob = float(quantiles_to_closure_prob(last_pred, bundle.quantiles)[0])
    prob = float(calibrator.predict([raw_prob])[0]) if calibrator else raw_prob

    gauge_now = float(feats["gauge_ft"].iloc[-1])
    already_above = bool(gauge_now >= bundle.closure_ft)
    if already_above:
        prob = 1.0
    _log(f"[predict] DONE — raw={raw_prob:.4f}, calibrated={prob:.4f}, above={already_above}")
    alert = bool(prob >= bundle.threshold)

    # Rain totals (compute from raw 15-min data)
    S = STEPS_PER_HOUR
    rain_6h = {}
    rain_1h = {}
    for name, _ in RAIN_GAUGES:
        col = _rain_col(name)
        if col in feats.columns:
            rain_6h[name] = round(float(feats[col].iloc[-6*S:].sum()), 2)
            rain_1h[name] = round(float(feats[col].iloc[-1*S:].sum()), 2)
        else:
            rain_6h[name] = 0.0
            rain_1h[name] = 0.0

    _log("[predict] building history arrays...")
    # Gauge history for sparkline (last 5 days)
    gauge_hist = []
    last_48 = feats.tail(5 * 24 * S)
    for ts, row in last_48.iterrows():
        gauge_hist.append({
            "ts": ts.isoformat(),
            "gauge_ft": round(float(row["gauge_ft"]), 2),
        })

    # Tide history (last 5 days)
    tide_hist = []
    for ts, row in last_48.iterrows():
        entry = {"ts": ts.isoformat()}
        if "tide_ft" in row.index and not pd.isna(row["tide_ft"]):
            entry["tide_ft"] = round(float(row["tide_ft"]), 2)
        if "tide_pred_ft" in row.index and not pd.isna(row["tide_pred_ft"]):
            entry["tide_pred_ft"] = round(float(row["tide_pred_ft"]), 2)
        if "storm_surge_ft" in row.index and not pd.isna(row["storm_surge_ft"]):
            entry["surge_ft"] = round(float(row["storm_surge_ft"]), 2)
        tide_hist.append(entry)

    _log("[predict] computing prob history...")
    # Probability history: use all TFT predictions from the batch
    prob_hist = []
    if len(preds) > 1:
        all_probs = quantiles_to_closure_prob(preds, bundle.quantiles)
        if calibrator:
            all_probs = calibrator.predict(all_probs)
        # Align with feats timestamps (predictions correspond to tail of dataset)
        n_preds = len(all_probs)
        pred_feats = feats.iloc[-n_preds:]
        for ts, p in zip(pred_feats.index, all_probs):
            # If gauge is already above closure threshold, prob = 100%
            gauge_at_ts = float(pred_feats.loc[ts, "gauge_ft"])
            if gauge_at_ts >= bundle.closure_ft:
                p = 1.0
            prob_hist.append({
                "ts": ts.isoformat(),
                "prob": round(float(p), 4),
            })

    # Discharge
    latest_row = feats.iloc[-1]
    q_now = float(latest_row["q_cfs"]) if "q_cfs" in feats.columns and not pd.isna(latest_row["q_cfs"]) else None

    # Tide
    tide_now = float(latest_row["tide_ft"]) if "tide_ft" in feats.columns and not pd.isna(latest_row["tide_ft"]) else None
    surge_now = float(latest_row["storm_surge_ft"]) if "storm_surge_ft" in feats.columns and not pd.isna(latest_row["storm_surge_ft"]) else None

    ts_utc = feats.index[-1]
    ts_hst = ts_utc - timedelta(hours=10)

    _log(f"[predict] returning result — status=ok, gauge={gauge_now:.2f}, prob={prob:.4f}")
    return {
        "status": "ok",
        "timestamp_utc": ts_utc.isoformat(),
        "timestamp_hst": ts_hst.strftime("%Y-%m-%d %I:%M %p HST"),
        "prob": round(prob, 4),
        "prob_pct": round(prob * 100, 1),
        "threshold": round(bundle.threshold, 4),
        "alert": alert,
        "gauge_ft": round(gauge_now, 2),
        "closure_ft": bundle.closure_ft,
        "already_above": already_above,
        "q_cfs": round(q_now, 0) if q_now else None,
        "tide_ft": round(tide_now, 2) if tide_now else None,
        "storm_surge_ft": round(surge_now, 2) if surge_now else None,
        "rain_1h": rain_1h,
        "rain_6h": rain_6h,
        "gauge_history": gauge_hist,
        "tide_history": tide_hist,
        "prob_history": prob_hist,
        "horizon_h": bundle.horizon_h,
        "model_threshold_pct": round(bundle.threshold * 100, 1),
    }


def _do_fetch():
    """Run one fetch cycle, update globals."""
    global _latest, _history, _fetching
    with _lock:
        _fetching = True
    try:
        result = _run_prediction()
        _log(f"[fetch] prediction returned status={result.get('status')}")
        with _lock:
            _latest = result
            _fetching = False
            if result.get("status") == "ok":
                _history.append({
                    "ts": result["timestamp_utc"],
                    "prob": result["prob"],
                    "gauge_ft": result["gauge_ft"],
                })
                _history = _history[-864:]
    except Exception as e:
        print(f"PREDICTION ERROR: {e}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        with _lock:
            _latest = {"status": "error", "message": str(e), "trace": traceback.format_exc()}
            _fetching = False


def _prediction_loop():
    """Background thread: fetch on startup, then every N seconds or on signal."""
    while True:
        _do_fetch()
        # Wait for either the timer or a manual refresh signal
        _refresh_event.wait(timeout=_REFRESH_SECONDS)
        _refresh_event.clear()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/api/predict")
def api_predict():
    with _lock:
        return jsonify({**_latest, "fetching": _fetching})


@app.route("/api/debug")
def api_debug():
    with _lock:
        return jsonify({
            "status": _latest.get("status"),
            "fetching": _fetching,
            "message": _latest.get("message"),
            "trace": _latest.get("trace"),
            "has_data": "prob" in _latest,
            "initialized": _initialized,
            "init_error": _init_error,
            "bundle_loaded": _bundle is not None,
            "tft_model_loaded": _tft_model is not None,
            "cwd": str(Path.cwd()),
            "model_exists": Path("bundle.json").exists() or Path("model.joblib").exists(),
        })


@app.route("/api/refresh", methods=["POST"])
def api_refresh():
    """Force an immediate re-fetch from USGS/NOAA."""
    with _lock:
        if _fetching:
            return jsonify({"status": "already_fetching"})
    _refresh_event.set()
    return jsonify({"status": "refresh_triggered"})


@app.route("/api/history")
def api_history():
    with _lock:
        return jsonify(_history)


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


# ---------------------------------------------------------------------------
# HTML / CSS / JS  (single-page dashboard)
# ---------------------------------------------------------------------------

HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hanalei Bridge Closure Predictor</title>
<style>
  :root {
    --bg: #0b1120;
    --card: #131c31;
    --card-border: #1e2d4a;
    --text: #e2e8f0;
    --text-dim: #8892a8;
    --accent: #38bdf8;
    --green: #22c55e;
    --yellow: #eab308;
    --orange: #f97316;
    --red: #ef4444;
    --red-glow: rgba(239,68,68,0.3);
    --green-glow: rgba(34,197,94,0.2);
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
  }
  .header {
    text-align: center;
    padding: 24px 16px 8px;
  }
  .header h1 {
    font-size: 1.5rem;
    font-weight: 700;
    letter-spacing: -0.02em;
  }
  .header h1 span { color: var(--accent); }
  .header .subtitle {
    color: var(--text-dim);
    font-size: 0.85rem;
    margin-top: 4px;
  }
  .timestamp {
    text-align: center;
    color: var(--text-dim);
    font-size: 0.8rem;
    padding: 4px 0 12px;
  }
  .timestamp .live-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--green);
    margin-right: 6px;
    animation: pulse 2s infinite;
  }
  .timestamp .fetching-dot {
    background: var(--yellow);
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
  }

  /* Refresh button */
  .refresh-row {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 12px;
    padding: 0 16px 12px;
  }
  .refresh-btn {
    background: var(--card);
    border: 1px solid var(--card-border);
    color: var(--accent);
    font-size: 0.82rem;
    font-weight: 600;
    padding: 8px 18px;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.2s;
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .refresh-btn:hover {
    background: var(--card-border);
    border-color: var(--accent);
  }
  .refresh-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  .refresh-btn .spin-icon {
    display: inline-block;
    transition: transform 0.3s;
  }
  .refresh-btn.spinning .spin-icon {
    animation: spin 1s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  .next-refresh {
    color: var(--text-dim);
    font-size: 0.75rem;
  }

  /* Main probability display */
  .prob-section {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 8px 16px 20px;
  }
  .gauge-ring {
    position: relative;
    width: 220px; height: 220px;
  }
  .gauge-ring svg { width: 100%; height: 100%; transform: rotate(-90deg); }
  .gauge-ring .bg-ring {
    fill: none;
    stroke: var(--card-border);
    stroke-width: 12;
  }
  .gauge-ring .fg-ring {
    fill: none;
    stroke-width: 12;
    stroke-linecap: round;
    transition: stroke-dashoffset 1s ease, stroke 0.5s ease;
  }
  .gauge-center {
    position: absolute;
    top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    text-align: center;
  }
  .gauge-center .prob-value {
    font-size: 3rem;
    font-weight: 800;
    line-height: 1;
    letter-spacing: -0.03em;
  }
  .gauge-center .prob-unit {
    font-size: 1.2rem;
    font-weight: 600;
    color: var(--text-dim);
  }
  .gauge-center .prob-label {
    font-size: 0.75rem;
    color: var(--text-dim);
    margin-top: 2px;
  }

  /* Alert banner */
  .alert-banner {
    margin: 0 16px 16px;
    padding: 12px 16px;
    border-radius: 10px;
    text-align: center;
    font-weight: 600;
    font-size: 0.95rem;
    transition: all 0.5s ease;
  }
  .alert-banner.alert-high {
    background: rgba(239,68,68,0.15);
    border: 1px solid var(--red);
    color: var(--red);
    box-shadow: 0 0 20px var(--red-glow);
  }
  .alert-banner.alert-low {
    background: rgba(34,197,94,0.1);
    border: 1px solid rgba(34,197,94,0.3);
    color: var(--green);
  }
  .alert-banner.alert-elevated {
    background: rgba(249,115,22,0.12);
    border: 1px solid var(--orange);
    color: var(--orange);
  }
  .alert-banner.alert-above {
    background: rgba(239,68,68,0.25);
    border: 2px solid var(--red);
    color: #fff;
    box-shadow: 0 0 30px var(--red-glow);
  }

  /* Cards grid */
  .cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 10px;
    padding: 0 16px 16px;
  }
  .card {
    background: var(--card);
    border: 1px solid var(--card-border);
    border-radius: 10px;
    padding: 14px;
  }
  .card .label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-dim);
    margin-bottom: 6px;
  }
  .card .value {
    font-size: 1.4rem;
    font-weight: 700;
  }
  .card .unit {
    font-size: 0.75rem;
    color: var(--text-dim);
    margin-left: 2px;
  }

  /* Charts */
  .chart-section {
    padding: 0 16px 16px;
  }
  .chart-card {
    background: var(--card);
    border: 1px solid var(--card-border);
    border-radius: 10px;
    padding: 14px;
    margin-bottom: 12px;
  }
  .chart-card .chart-title {
    font-size: 0.8rem;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 10px;
  }
  canvas { width: 100% !important; }

  /* Rain table */
  .rain-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82rem;
  }
  .rain-table th {
    text-align: left;
    font-weight: 600;
    color: var(--text-dim);
    padding: 4px 8px;
    border-bottom: 1px solid var(--card-border);
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  .rain-table td {
    padding: 6px 8px;
    border-bottom: 1px solid rgba(30,45,74,0.5);
  }
  .rain-bar {
    height: 6px;
    background: var(--accent);
    border-radius: 3px;
    transition: width 0.8s ease;
    min-width: 2px;
  }
  .rain-bar-bg {
    height: 6px;
    background: var(--card-border);
    border-radius: 3px;
    width: 100%;
  }

  /* Loading / error */
  .loading {
    text-align: center;
    padding: 60px 20px;
    color: var(--text-dim);
  }
  .loading .spinner {
    width: 40px; height: 40px;
    border: 3px solid var(--card-border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 1s linear infinite;
    margin: 0 auto 16px;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  .footer {
    text-align: center;
    color: var(--text-dim);
    font-size: 0.7rem;
    padding: 16px;
    line-height: 1.6;
  }
  .footer a { color: var(--accent); text-decoration: none; }

  @media (max-width: 480px) {
    .cards { grid-template-columns: repeat(2, 1fr); }
    .gauge-ring { width: 180px; height: 180px; }
    .gauge-center .prob-value { font-size: 2.4rem; }
  }
</style>
</head>
<body>

<div class="header">
  <h1><span>Hanalei</span> Bridge Closure Predictor</h1>
  <div class="subtitle">USGS Gauge 16103000 &middot; ML model with 6 rain gauges + tide + discharge</div>
</div>

<div class="timestamp" id="timestamp">
  <span class="live-dot"></span>Connecting...
</div>

<div class="refresh-row">
  <button class="refresh-btn" id="refresh-btn" onclick="manualRefresh()">
    <span class="spin-icon">&#x21bb;</span> Refresh Now
  </button>
  <span class="next-refresh" id="next-refresh"></span>
</div>

<div id="main-content">
  <div class="loading">
    <div class="spinner"></div>
    Fetching live data from USGS & NOAA...
  </div>
</div>

<div class="footer">
  Model: HistGradientBoosting &middot; 115 features &middot; 18.5 yr training set &middot; 15-min resolution<br>
  Data: <a href="https://waterdata.usgs.gov/nwis/uv?site_no=16103000" target="_blank">USGS 16103000</a> &middot;
  <a href="https://tidesandcurrents.noaa.gov/stationhome.html?id=1611400" target="_blank">NOAA Nawiliwili</a><br>
  Not an official warning system
</div>

<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns@3/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
<script>
const AUTO_REFRESH_MS = 5 * 60 * 1000;  // auto-refresh every 5 minutes
let gaugeChart = null;
let probChart = null;
let tideChart = null;
let nextRefreshTimer = null;
let countdownInterval = null;
let nextRefreshAt = null;

function probColor(p) {
  if (p < 0.10) return '#22c55e';
  if (p < 0.24) return '#eab308';
  if (p < 0.50) return '#f97316';
  return '#ef4444';
}

function renderDashboard(d) {
  const mc = document.getElementById('main-content');
  mc.innerHTML = `
    <div class="prob-section">
      <div class="gauge-ring">
        <svg viewBox="0 0 220 220">
          <circle class="bg-ring" cx="110" cy="110" r="95"/>
          <circle class="fg-ring" id="prob-ring" cx="110" cy="110" r="95"/>
        </svg>
        <div class="gauge-center">
          <div class="prob-value" id="prob-value">${d.prob_pct}</div>
          <div class="prob-unit">%</div>
          <div class="prob-label">closure probability</div>
        </div>
      </div>
    </div>

    <div class="alert-banner ${d.already_above ? 'alert-above' : d.prob >= 0.5 ? 'alert-high' : d.prob >= 0.15 ? 'alert-elevated' : 'alert-low'}" id="alert-banner">
      ${d.already_above
        ? '🚨 GAUGE IS ABOVE 5.0 FT — ROAD LIKELY CLOSED'
        : d.prob >= 0.5
          ? '⚠️ HIGH PROBABILITY of closure within 3 hours'
          : d.prob >= 0.15
            ? '⚠️ ELEVATED PROBABILITY — monitor closely'
            : '✅ Low probability of closure in next 3 hours'}
    </div>

    <div class="cards">
      <div class="card">
        <div class="label">Gauge Height</div>
        <div class="value" style="color: ${d.gauge_ft >= 5 ? '#ef4444' : d.gauge_ft >= 4 ? '#f97316' : '#22c55e'}">${d.gauge_ft}<span class="unit">ft</span></div>
      </div>
      <div class="card">
        <div class="label">Discharge</div>
        <div class="value">${d.q_cfs ? Math.round(d.q_cfs).toLocaleString() : '—'}<span class="unit">cfs</span></div>
      </div>
      <div class="card">
        <div class="label">Tide Level</div>
        <div class="value">${d.tide_ft ?? '—'}<span class="unit">ft</span></div>
      </div>
      <div class="card">
        <div class="label">Storm Surge</div>
        <div class="value" style="color: ${d.storm_surge_ft > 0.3 ? '#f97316' : '#e2e8f0'}">${d.storm_surge_ft ?? '—'}<span class="unit">ft</span></div>
      </div>
    </div>

    <div class="chart-section">
      <div class="chart-card">
        <div class="chart-title">Closure Probability — Last 5 Days</div>
        <div style="position:relative; height:200px;"><canvas id="probChart"></canvas></div>
      </div>
      <div class="chart-card">
        <div class="chart-title">Gauge Height — Last 5 Days</div>
        <div style="position:relative; height:200px;"><canvas id="gaugeChart"></canvas></div>
      </div>
      <div class="chart-card">
        <div class="chart-title">Tide Level — Last 5 Days</div>
        <div style="position:relative; height:200px;"><canvas id="tideChart"></canvas></div>
      </div>
      <div class="chart-card">
        <div class="chart-title">Rainfall by Gauge</div>
        <table class="rain-table">
          <thead><tr><th>Gauge</th><th>1h</th><th>6h</th><th></th></tr></thead>
          <tbody id="rain-body"></tbody>
        </table>
      </div>
    </div>
  `;

  // Animate ring
  const ring = document.getElementById('prob-ring');
  const circ = 2 * Math.PI * 95;
  ring.style.strokeDasharray = circ;
  ring.style.strokeDashoffset = circ * (1 - d.prob);
  ring.style.stroke = probColor(d.prob);

  // Prob value color
  document.getElementById('prob-value').style.color = probColor(d.prob);

  // Rain table
  const rainBody = document.getElementById('rain-body');
  const gaugeNames = {
    waiakoali: 'Waiakoali',
    mohihi_crsg: 'Mohihi CRSG',
    waialae: 'Waialae',
    kilohana: 'Kilohana',
    waialeale: 'Mt Waialeale',
    n_wailua: 'N. Wailua',
  };
  const maxRain6 = Math.max(0.1, ...Object.values(d.rain_6h || {}));
  for (const [key, label] of Object.entries(gaugeNames)) {
    const r1 = d.rain_1h?.[key] ?? 0;
    const r6 = d.rain_6h?.[key] ?? 0;
    const pct = Math.min(100, (r6 / maxRain6) * 100);
    const barColor = r6 > 1 ? '#f97316' : r6 > 0.5 ? '#eab308' : '#38bdf8';
    rainBody.innerHTML += `
      <tr>
        <td>${label}</td>
        <td>${r1.toFixed(2)}"</td>
        <td>${r6.toFixed(2)}"</td>
        <td style="width:40%">
          <div class="rain-bar-bg">
            <div class="rain-bar" style="width:${pct}%; background:${barColor}"></div>
          </div>
        </td>
      </tr>`;
  }

  // Charts
  buildProbChart(d.prob_history || [], d.model_threshold_pct);
  buildGaugeChart(d.gauge_history || [], d.closure_ft);
  buildTideChart(d.tide_history || []);
}

function buildProbChart(data, threshPct) {
  const ctx = document.getElementById('probChart');
  if (!ctx) return;
  if (probChart) probChart.destroy();
  probChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: data.map(d => new Date(d.ts)),
      datasets: [{
        data: data.map(d => d.prob * 100),
        borderColor: '#38bdf8',
        backgroundColor: 'rgba(56,189,248,0.1)',
        fill: true,
        tension: 0.3,
        pointRadius: 0,
        borderWidth: 2,
      }, {
        data: data.map(() => threshPct),
        borderColor: 'rgba(239,68,68,0.5)',
        borderDash: [6, 4],
        pointRadius: 0,
        borderWidth: 1,
        fill: false,
      }]
    },
    options: {
      plugins: { legend: { display: false } },
      scales: {
        x: {
          type: 'time',
          time: { unit: 'hour', displayFormats: { hour: 'ha' } },
          grid: { color: 'rgba(30,45,74,0.5)' },
          ticks: { color: '#8892a8', maxTicksLimit: 8 },
        },
        y: {
          beginAtZero: true,
          min: 0,
          max: 100,
          grid: { color: 'rgba(30,45,74,0.5)' },
          ticks: { color: '#8892a8', stepSize: 25, callback: v => v + '%', autoSkip: false },
        }
      },
      interaction: { intersect: false, mode: 'index' },
      responsive: true,
      maintainAspectRatio: false,
    }
  });
}

function buildGaugeChart(data, closureFt) {
  const ctx = document.getElementById('gaugeChart');
  if (!ctx) return;
  if (gaugeChart) gaugeChart.destroy();
  const maxG = 10;  // static y-axis: 0–10 ft covers all historical floods
  gaugeChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: data.map(d => new Date(d.ts)),
      datasets: [{
        data: data.map(d => d.gauge_ft),
        borderColor: '#22c55e',
        backgroundColor: 'rgba(34,197,94,0.1)',
        fill: true,
        tension: 0.3,
        pointRadius: 0,
        borderWidth: 2,
      }, {
        data: data.map(() => closureFt),
        borderColor: 'rgba(239,68,68,0.6)',
        borderDash: [6, 4],
        pointRadius: 0,
        borderWidth: 1.5,
        fill: false,
      }]
    },
    options: {
      plugins: { legend: { display: false } },
      scales: {
        x: {
          type: 'time',
          time: { unit: 'hour', displayFormats: { hour: 'ha' } },
          grid: { color: 'rgba(30,45,74,0.5)' },
          ticks: { color: '#8892a8', maxTicksLimit: 8 },
        },
        y: {
          beginAtZero: true,
          min: 0,
          max: maxG,
          grid: { color: 'rgba(30,45,74,0.5)' },
          ticks: { color: '#8892a8', stepSize: 2, callback: v => v + ' ft', autoSkip: false },
        }
      },
      interaction: { intersect: false, mode: 'index' },
      responsive: true,
      maintainAspectRatio: false,
    }
  });
}

function buildTideChart(data) {
  const ctx = document.getElementById('tideChart');
  if (!ctx) return;
  if (tideChart) tideChart.destroy();
  tideChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: data.map(d => new Date(d.ts)),
      datasets: [{
        label: 'Observed',
        data: data.map(d => d.tide_ft ?? null),
        borderColor: '#38bdf8',
        backgroundColor: 'rgba(56,189,248,0.1)',
        fill: true,
        tension: 0.3,
        pointRadius: 0,
        borderWidth: 2,
      }, {
        label: 'Predicted (astronomical)',
        data: data.map(d => d.tide_pred_ft ?? null),
        borderColor: 'rgba(148,163,184,0.6)',
        borderDash: [4, 3],
        pointRadius: 0,
        borderWidth: 1.5,
        fill: false,
      }, {
        label: 'Storm surge',
        data: data.map(d => d.surge_ft ?? null),
        borderColor: '#f97316',
        backgroundColor: 'rgba(249,115,22,0.08)',
        fill: true,
        tension: 0.3,
        pointRadius: 0,
        borderWidth: 1.5,
      }]
    },
    options: {
      plugins: { legend: { display: true, labels: { color: '#8892a8', boxWidth: 12, padding: 8 } } },
      scales: {
        x: {
          type: 'time',
          time: { unit: 'hour', displayFormats: { hour: 'ha' } },
          grid: { color: 'rgba(30,45,74,0.5)' },
          ticks: { color: '#8892a8', maxTicksLimit: 8 },
        },
        y: {
          min: -1,
          max: 3,
          grid: { color: 'rgba(30,45,74,0.5)' },
          ticks: { color: '#8892a8', stepSize: 1, callback: v => v + ' ft' },
        }
      },
      interaction: { intersect: false, mode: 'index' },
      responsive: true,
      maintainAspectRatio: false,
    }
  });
}

function updateTimestamp(d) {
  const el = document.getElementById('timestamp');
  const dotClass = d.fetching ? 'live-dot fetching-dot' : 'live-dot';
  const fetchLabel = d.fetching ? 'Fetching new data...' : (d.timestamp_hst || 'Connecting...');
  el.innerHTML = `<span class="${dotClass}"></span>${fetchLabel}`;
}

function startCountdown() {
  nextRefreshAt = Date.now() + AUTO_REFRESH_MS;
  if (countdownInterval) clearInterval(countdownInterval);
  countdownInterval = setInterval(() => {
    const el = document.getElementById('next-refresh');
    if (!el) return;
    const secs = Math.max(0, Math.round((nextRefreshAt - Date.now()) / 1000));
    const m = Math.floor(secs / 60);
    const s = secs % 60;
    el.textContent = `Next refresh in ${m}:${s.toString().padStart(2, '0')}`;
  }, 1000);
}

async function fetchAndRender() {
  try {
    const resp = await fetch('/api/predict');
    const d = await resp.json();
    if (d.status === 'ok') {
      renderDashboard(d);
      updateTimestamp(d);
    } else if (d.status === 'loading' || d.fetching) {
      updateTimestamp(d);
    } else {
      document.getElementById('main-content').innerHTML =
        `<div class="loading" style="color:#ef4444">Error: ${d.message || 'Unknown'}</div>`;
    }
  } catch (e) {
    console.error('Fetch error:', e);
  }
}

async function manualRefresh() {
  const btn = document.getElementById('refresh-btn');
  btn.disabled = true;
  btn.classList.add('spinning');
  btn.querySelector('.spin-icon').textContent = '\u21bb';

  // Tell backend to re-fetch from USGS/NOAA
  await fetch('/api/refresh', { method: 'POST' });

  // Poll until the backend finishes fetching (new timestamp)
  const oldTs = document.getElementById('timestamp').textContent;
  let attempts = 0;
  const poll = setInterval(async () => {
    attempts++;
    const resp = await fetch('/api/predict');
    const d = await resp.json();
    if (d.status === 'ok' && !d.fetching) {
      clearInterval(poll);
      renderDashboard(d);
      updateTimestamp(d);
      btn.disabled = false;
      btn.classList.remove('spinning');
      // Reset auto-refresh countdown
      if (nextRefreshTimer) clearTimeout(nextRefreshTimer);
      scheduleAutoRefresh();
    } else if (attempts > 60) {  // timeout after ~60s
      clearInterval(poll);
      btn.disabled = false;
      btn.classList.remove('spinning');
    }
  }, 1000);
}

function scheduleAutoRefresh() {
  startCountdown();
  if (nextRefreshTimer) clearTimeout(nextRefreshTimer);
  nextRefreshTimer = setTimeout(async () => {
    await manualRefresh();
  }, AUTO_REFRESH_MS);
}

// Initial load: poll every 3s until data is ready, then switch to 5min auto-refresh
async function initialPoll() {
  const resp = await fetch('/api/predict');
  const d = await resp.json();
  if (d.status === 'ok') {
    renderDashboard(d);
    updateTimestamp(d);
    scheduleAutoRefresh();
  } else {
    updateTimestamp(d);
    setTimeout(initialPoll, 3000);
  }
}
initialPoll();
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _init_model_and_thread(model_path: str):
    """Load the TFT model and start the background prediction thread."""
    global _bundle, _tft_model, _calibrator
    model_dir = Path(model_path)
    if model_dir.is_file():
        model_dir = model_dir.parent

    print(f"Loading TFT model from {model_dir}...", file=sys.stderr, flush=True)

    # Load TFTBundle
    bundle_path = model_dir / "bundle.json"
    if bundle_path.exists():
        _bundle = load_tft_bundle(bundle_path)
    else:
        raise FileNotFoundError(f"bundle.json not found in {model_dir}")

    # Load TFT checkpoint (CPU-only: patch torch.zeros for torchmetrics CUDA device refs)
    ckpt_path = model_dir / "tft_model_cpu.ckpt"
    if not ckpt_path.exists():
        ckpt_path = model_dir / "tft_model.ckpt"
    if ckpt_path.exists():
        _orig_zeros = torch.zeros
        def _patched_zeros(*args, **kwargs):
            if 'device' in kwargs and kwargs['device'] is not None:
                dev = kwargs['device']
                if (isinstance(dev, torch.device) and dev.type == 'cuda') or \
                   (isinstance(dev, str) and 'cuda' in dev):
                    kwargs['device'] = torch.device('cpu')
            return _orig_zeros(*args, **kwargs)
        torch.zeros = _patched_zeros
        _tft_model = TemporalFusionTransformer.load_from_checkpoint(
            str(ckpt_path), map_location="cpu",
        )
        torch.zeros = _orig_zeros
        _tft_model.eval()
    else:
        raise FileNotFoundError(f"tft_model_cpu.ckpt not found in {model_dir}")

    # Load calibrator
    cal_path = model_dir / "calibrator.joblib"
    if cal_path.exists():
        _calibrator = joblib.load(cal_path)

    print(f"  horizon: {_bundle.horizon_h}h, threshold: {_bundle.threshold:.4f}, "
          f"encoder: {_bundle.encoder_length}, params: {sum(p.numel() for p in _tft_model.parameters()):,}",
          file=sys.stderr, flush=True)

    t = threading.Thread(target=_prediction_loop, daemon=True)
    t.start()
    print(f"Background predictor started (refreshes every {_REFRESH_SECONDS}s)",
          file=sys.stderr, flush=True)


def main():
    parser = argparse.ArgumentParser(description="Hanalei Bridge Closure Dashboard (v7 TFT)")
    parser.add_argument("--model", default="./hanalei_v7_out",
                        help="Path to model directory or bundle.json")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    _init_model_and_thread(args.model)
    print(f"\n  Dashboard: http://{args.host}:{args.port}\n")
    app.run(host=args.host, port=args.port, debug=False)


# --- Lazy init for gunicorn: threads don't survive fork, so start on first request ---
_initialized = False
_init_error = None

@app.before_request
def _lazy_init():
    global _initialized, _init_error, _latest
    if not _initialized:
        # Search for TFT model (bundle.json) or legacy model (model.joblib)
        search_paths = [
            "bundle.json", "./bundle.json",
            "hanalei_v7_out/bundle.json", "./hanalei_v7_out/bundle.json",
            "hanalei_v7_out", "./hanalei_v7_out",
        ]
        for _p in search_paths:
            if Path(_p).exists():
                try:
                    _log(f"[init] lazy init from {_p} (worker pid={__import__('os').getpid()})")
                    _init_model_and_thread(_p)
                    _initialized = True
                    _init_error = None
                    _log("[init] SUCCESS — TFT model loaded, background thread started")
                except Exception as e:
                    _init_error = traceback.format_exc()
                    _log(f"[init] FAILED: {e}")
                    _log(_init_error)
                    _latest = {"status": "error", "message": f"Model init failed: {e}",
                               "trace": _init_error}
                    _initialized = True  # don't retry every request — surface the error
                return
        _log("[init] ERROR: bundle.json not found!")
        _init_error = "bundle.json not found"
        _latest = {"status": "error", "message": "TFT model (bundle.json) not found"}
        _initialized = True

if __name__ == "__main__":
    main()
