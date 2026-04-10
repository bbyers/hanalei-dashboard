#!/usr/bin/env python3
"""Hanalei Bridge Closure Probability — Live Dashboard.

A Flask web app that displays a running probability of Hanalei River bridge
closure (gauge > 5.0 ft) within the next 3 hours, powered by the v3 ML model.

Usage:
    python hanalei_web.py [--model ./hanalei_v3_out/model.joblib] [--port 5000]
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template_string

# Import model components
from hanalei_closure_model import (
    RAIN_GAUGES,
    TrainBundle,
    build_features,
    fetch_all_rain_recent,
    fetch_discharge_recent,
    fetch_gauge_recent,
    fetch_tide_recent,
    to_hourly,
    _rain_col,
)

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

app = Flask(__name__)

_bundle: TrainBundle | None = None
_latest: dict = {"status": "loading", "message": "Fetching initial data..."}
_history: list[dict] = []          # last 72 hours of predictions
_lock = threading.Lock()
_refresh_event = threading.Event()  # signal background thread to re-fetch now
_REFRESH_SECONDS = 300              # auto-refresh every 5 minutes
_fetching = False                   # True while a fetch is in progress


# ---------------------------------------------------------------------------
# Background prediction loop
# ---------------------------------------------------------------------------

def _run_prediction() -> dict:
    """Run one prediction cycle, return result dict."""
    bundle = _bundle
    gauge_raw = fetch_gauge_recent(hours=200)
    q_raw = fetch_discharge_recent(hours=200)
    rain_raw = fetch_all_rain_recent(hours=200)
    tide_obs, tide_pred = fetch_tide_recent(hours=200)
    hourly = to_hourly(gauge_raw, rain_raw, q_raw=q_raw, tide_obs=tide_obs, tide_pred=tide_pred)
    feats = build_features(hourly).dropna(subset=bundle.features)

    if feats.empty:
        return {"status": "error", "message": "Insufficient recent data from USGS/NOAA"}

    latest = feats.iloc[[-1]]
    x = latest[bundle.features].values
    prob = float(bundle.model.predict_proba(x)[0, 1])
    alert = bool(prob >= bundle.threshold)
    gauge_now = float(latest["gauge_ft"].iloc[0])
    already_above = bool(gauge_now >= bundle.closure_ft)

    rain_6h = {}
    for name, _ in RAIN_GAUGES:
        col = f"{_rain_col(name)}_sum_6"
        if col in latest.columns:
            rain_6h[name] = round(float(latest[col].iloc[0]), 2)

    rain_1h = {}
    for name, _ in RAIN_GAUGES:
        col = f"{_rain_col(name)}_sum_1"
        if col in latest.columns:
            rain_1h[name] = round(float(latest[col].iloc[0]), 2)

    # Gauge history for sparkline (last 48h)
    gauge_hist = []
    last_48 = feats.tail(48)
    for ts, row in last_48.iterrows():
        gauge_hist.append({
            "ts": ts.isoformat(),
            "gauge_ft": round(float(row["gauge_ft"]), 2),
        })

    # Probability history from feats (last 48h)
    prob_hist = []
    if len(feats) > 1:
        recent_feats = feats.tail(48)
        x_recent = recent_feats[bundle.features].values
        probs_recent = bundle.model.predict_proba(x_recent)[:, 1]
        for ts, p in zip(recent_feats.index, probs_recent):
            prob_hist.append({
                "ts": ts.isoformat(),
                "prob": round(float(p), 4),
            })

    # Discharge
    q_now = float(latest["q_cfs"].iloc[0]) if "q_cfs" in latest.columns and not pd.isna(latest["q_cfs"].iloc[0]) else None

    # Tide
    tide_now = float(latest["tide_ft"].iloc[0]) if "tide_ft" in latest.columns and not pd.isna(latest["tide_ft"].iloc[0]) else None
    surge_now = float(latest["storm_surge_ft"].iloc[0]) if "storm_surge_ft" in latest.columns and not pd.isna(latest["storm_surge_ft"].iloc[0]) else None

    # Tide history (last 48h) — observed, predicted, and storm surge
    tide_hist = []
    for ts, row in last_48.iterrows():
        entry = {"ts": ts.isoformat()}
        if "tide_ft" in row.index and not pd.isna(row.get("tide_ft")):
            entry["obs"] = round(float(row["tide_ft"]), 2)
        if "tide_pred_ft" in row.index and not pd.isna(row.get("tide_pred_ft")):
            entry["pred"] = round(float(row["tide_pred_ft"]), 2)
        if "storm_surge_ft" in row.index and not pd.isna(row.get("storm_surge_ft")):
            entry["surge"] = round(float(row["storm_surge_ft"]), 2)
        tide_hist.append(entry)

    ts_utc = latest.index[-1]
    ts_hst = ts_utc - timedelta(hours=10)

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
        "prob_history": prob_hist,
        "tide_history": tide_hist,
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
  Model: HistGradientBoosting &middot; 66 features &middot; 18.5 yr training set<br>
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
let surgeChart = null;
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

    <div class="alert-banner ${d.already_above ? 'alert-above' : d.alert ? (d.prob >= 0.5 ? 'alert-high' : 'alert-elevated') : 'alert-low'}" id="alert-banner">
      ${d.already_above
        ? '🚨 GAUGE IS ABOVE 5.0 FT — ROAD LIKELY CLOSED'
        : d.alert
          ? (d.prob >= 0.5
            ? '⚠️ HIGH PROBABILITY of closure within 3 hours'
            : '⚠️ ELEVATED PROBABILITY — monitor closely')
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
        <div class="chart-title">Closure Probability — Last 48 Hours</div>
        <div style="position:relative; height:200px;"><canvas id="probChart"></canvas></div>
      </div>
      <div class="chart-card">
        <div class="chart-title">Gauge Height — Last 48 Hours</div>
        <div style="position:relative; height:200px;"><canvas id="gaugeChart"></canvas></div>
      </div>
      <div class="chart-card">
        <div class="chart-title">Tide Level — Last 48 Hours</div>
        <div style="position:relative; height:200px;"><canvas id="tideChart"></canvas></div>
      </div>
      <div class="chart-card">
        <div class="chart-title">Storm Surge — Last 48 Hours</div>
        <div style="position:relative; height:200px;"><canvas id="surgeChart"></canvas></div>
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
  buildSurgeChart(d.tide_history || []);
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
        data: data.map(d => d.obs ?? null),
        borderColor: '#38bdf8',
        backgroundColor: 'rgba(56,189,248,0.08)',
        fill: true,
        tension: 0.3,
        pointRadius: 0,
        borderWidth: 2,
      }, {
        label: 'Predicted (astronomical)',
        data: data.map(d => d.pred ?? null),
        borderColor: 'rgba(139,92,246,0.6)',
        borderDash: [5, 3],
        pointRadius: 0,
        borderWidth: 1.5,
        fill: false,
      }]
    },
    options: {
      plugins: {
        legend: {
          display: true,
          position: 'top',
          labels: { color: '#8892a8', boxWidth: 12, font: { size: 11 } }
        }
      },
      scales: {
        x: {
          type: 'time',
          time: { unit: 'hour', displayFormats: { hour: 'ha' } },
          grid: { color: 'rgba(30,45,74,0.5)' },
          ticks: { color: '#8892a8', maxTicksLimit: 8 },
        },
        y: {
          beginAtZero: true,
          min: -0.5,
          max: 3.5,
          grid: { color: 'rgba(30,45,74,0.5)' },
          ticks: { color: '#8892a8', stepSize: 0.5, callback: v => v + ' ft' },
        }
      },
      interaction: { intersect: false, mode: 'index' },
      responsive: true,
      maintainAspectRatio: false,
    }
  });
}

function buildSurgeChart(data) {
  const ctx = document.getElementById('surgeChart');
  if (!ctx) return;
  if (surgeChart) surgeChart.destroy();
  surgeChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: data.map(d => new Date(d.ts)),
      datasets: [{
        data: data.map(d => d.surge ?? null),
        borderColor: '#f97316',
        backgroundColor: 'rgba(249,115,22,0.1)',
        fill: true,
        tension: 0.3,
        pointRadius: 0,
        borderWidth: 2,
      }, {
        data: data.map(() => 0),
        borderColor: 'rgba(255,255,255,0.15)',
        borderDash: [4, 4],
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
          min: -1.0,
          max: 1.5,
          grid: { color: 'rgba(30,45,74,0.5)' },
          ticks: { color: '#8892a8', stepSize: 0.5, callback: v => v.toFixed(1) + ' ft' },
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

// Initial load: fetch cached data, then start auto-refresh cycle
fetchAndRender();
scheduleAutoRefresh();
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Hanalei Bridge Closure Dashboard")
    parser.add_argument("--model", default="./model.joblib",
                        help="Path to trained model.joblib")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 5000)))
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    global _bundle
    print(f"Loading model from {args.model}...")
    _bundle = joblib.load(args.model)
    print(f"  horizon: {_bundle.horizon_h}h, threshold: {_bundle.threshold:.4f}, "
          f"features: {len(_bundle.features)}")

    # Start background prediction thread
    t = threading.Thread(target=_prediction_loop, daemon=True)
    t.start()
    print(f"Background predictor started (refreshes every {_REFRESH_SECONDS}s)")

    print(f"\n  Dashboard: http://{args.host}:{args.port}\n")
    app.run(host=args.host, port=args.port, debug=False)


_init_done = False

def _init_app():
    """Initialize model and background thread. Safe to call multiple times."""
    global _bundle, _init_done
    if _init_done:
        return
    _init_done = True
    model_path = os.environ.get("MODEL_PATH", "./model.joblib")
    if Path(model_path).exists():
        _bundle = joblib.load(model_path)
        print(f"Loaded model: horizon={_bundle.horizon_h}h, threshold={_bundle.threshold:.4f}, "
              f"features={len(_bundle.features)}")
        t = threading.Thread(target=_prediction_loop, daemon=True)
        t.start()
        print(f"Background predictor started (refreshes every {_REFRESH_SECONDS}s)")
    else:
        print(f"WARNING: model not found at {model_path}")


# Auto-init when imported by gunicorn
_init_app()


if __name__ == "__main__":
    main()
