"""
dashboard_minimal.py — Minimalist Concrete Curing AI Dashboard
───────────────────────────────────────────────────────────────
Install deps: pip install flask pandas numpy
Run:          python dashboard_minimal.py [--demo]
Open:         http://localhost:5050

Works alongside concrete_monitor_refined.py
Reads logs/curing_log.csv in real time.
"""

import argparse, csv, json, math, os, random, threading, time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, render_template_string

# ── Config ────────────────────────────────────────────────────────────────────
LOG_PATH    = Path("logs/curing_log.csv")
IDEAL_WATER = 50       # soil moisture % midpoint (0-100 scale)
DATUM_TEMP  = -10.0
IDEAL_TEMP  = 23.0
IDEAL_HUM   = 80.0

app = Flask(__name__)

# ── Demo data generator ───────────────────────────────────────────────────────
_demo_state = dict(temp=22.0, hum=72.0, soil=55.0, elapsed=0.0, count=0, running=False)

def _maturity(t, h, w):
    tf = max(0, (t - DATUM_TEMP) / (IDEAL_TEMP - DATUM_TEMP))
    hf = min(1.2, h / IDEAL_HUM)
    wf = max(0.3, min(1.0, 1 - abs(w - IDEAL_WATER) / 50))
    return min(1.3, max(0, tf * hf * wf))

def _curing_rate(t, h, w, hrs, mf):
    base = 100 * mf * (1 - math.exp(-hrs / (15 + (1 - mf) * 30)))
    if t < 2:  base *= 0.1
    if t > 40: base *= max(0, 1 - (t - 40) * 0.05)
    return min(100, max(0, base + random.gauss(0, 1.5)))

def _grade(rate, hrs):
    exp = (hrs / 72) * 50 if hrs <= 72 else \
          50 + ((hrs - 72) / 96) * 15 if hrs <= 168 else \
          65 + min(35, (hrs - 168) / 504 * 35)
    r = rate / max(exp, 1)
    return "A" if r >= 0.95 else "B" if r >= 0.85 else "C" if r >= 0.70 else "D" if r >= 0.55 else "F"

def _health(t, h, w, rate, hrs):
    i = 0
    if t < 5 or t > 38:       i += 2
    if h < 40:                 i += 2
    elif h < 60:               i += 1
    if w < 15 or w > 90:      i += 1
    if rate < 30 and hrs > 24: i += 2
    return "Healthy" if i == 0 else "At Risk" if i <= 2 else "Critical"

def _demo_loop():
    LOG_PATH.parent.mkdir(exist_ok=True)
    write_header = not LOG_PATH.exists() or LOG_PATH.stat().st_size == 0
    f = open(LOG_PATH, "a", newline="")
    w = csv.DictWriter(f, fieldnames=[
        "timestamp","elapsed_h","temperature_c","humidity_pct","soil_pct",
        "curing_rate","grade","health","buzzer"
    ])
    if write_header:
        w.writeheader()
    s = _demo_state
    s["running"] = True
    while s["running"]:
        s["temp"] = float(np.clip(s["temp"] + random.gauss(0, 0.5),  -5,  50))
        s["hum"]  = float(np.clip(s["hum"]  + random.gauss(0, 1.2),  10, 100))
        s["soil"] = float(np.clip(s["soil"] + random.gauss(0, 2.5),   0, 100))
        s["elapsed"] += 2 / 3600
        s["count"]   += 1
        mf     = _maturity(s["temp"], s["hum"], s["soil"])
        rate   = _curing_rate(s["temp"], s["hum"], s["soil"], s["elapsed"], mf)
        grade  = _grade(rate, s["elapsed"])
        health = _health(s["temp"], s["hum"], s["soil"], rate, s["elapsed"])
        bad    = health == "Critical" or s["temp"] < 5 or s["temp"] > 38
        w.writerow({
            "timestamp"    : datetime.now().isoformat(timespec="seconds"),
            "elapsed_h"    : round(s["elapsed"], 4),
            "temperature_c": round(s["temp"], 2),
            "humidity_pct" : round(s["hum"],  2),
            "soil_pct"     : round(s["soil"], 1),
            "curing_rate"  : round(rate, 2),
            "grade"        : grade,
            "health"       : health,
            "buzzer"       : "1" if bad else "0",
        })
        f.flush()
        time.sleep(2)

# ── API endpoint ──────────────────────────────────────────────────────────────
@app.route("/api/data")
def api_data():
    if not LOG_PATH.exists() or LOG_PATH.stat().st_size == 0:
        return jsonify({"empty": True})
    try:
        df = pd.read_csv(LOG_PATH)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    except Exception:
        return jsonify({"empty": True})
    if df.empty:
        return jsonify({"empty": True})

    # Support both column names for backward compatibility
    soil_col = "soil_pct" if "soil_pct" in df.columns else "soil_analog"

    latest      = df.iloc[-1]
    history     = df.tail(60)
    grade_counts = df["grade"].value_counts().to_dict()
    soil_val    = float(latest[soil_col])

    mf       = _maturity(float(latest["temperature_c"]), float(latest["humidity_pct"]), soil_val)
    rate_now = float(latest["curing_rate"])
    forecast = [
        round(min(100, max(0, rate_now + i * 1.8 * (mf - 0.5) + random.gauss(0, 1.2))), 1)
        for i in range(1, 7)
    ]

    return jsonify({
        "empty"       : False,
        "count"       : len(df),
        "timestamp"   : latest["timestamp"].strftime("%d %b · %H:%M:%S"),
        "elapsed"     : round(float(latest["elapsed_h"]), 2),
        "rate"        : round(float(latest["curing_rate"]), 1),
        "grade"       : str(latest["grade"]),
        "health"      : str(latest["health"]),
        "temp"        : round(float(latest["temperature_c"]), 1),
        "hum"         : round(float(latest["humidity_pct"]), 1),
        "soil"        : round(soil_val, 1),
        "buzzer"      : str(latest.get("buzzer", "0")),
        "grade_counts": grade_counts,
        "forecast"    : forecast,
        "rate_history": [round(v, 1) for v in history["curing_rate"].tolist()],
        "soil_history": [round(v, 1) for v in history[soil_col].tolist()],
    })

# ── SSE stream ────────────────────────────────────────────────────────────────
@app.route("/stream")
def stream():
    def event_stream():
        while True:
            time.sleep(2)
            yield "data: ping\n\n"
    return Response(event_stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

# ── HTML ──────────────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Concrete Curing Monitor</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',system-ui,sans-serif;background:#0d1117;color:#c9d1d9;min-height:100vh}
header{background:#161b22;border-bottom:1px solid #30363d;padding:.8rem 1.5rem;
       display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:10}
header h1{font-size:1rem;font-weight:600;color:#58a6ff;letter-spacing:.04em}
#ts{font-size:.72rem;color:#8b949e}
.kpis{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:.75rem;padding:1.25rem 1.5rem}
.kpi{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:.9rem 1rem;transition:border-color .2s}
.kpi:hover{border-color:#58a6ff}
.kpi-label{font-size:.65rem;text-transform:uppercase;letter-spacing:.07em;color:#8b949e;margin-bottom:.35rem}
.kpi-val{font-size:1.65rem;font-weight:700;line-height:1.1}
.kpi-sub{font-size:.7rem;color:#8b949e;margin-top:.25rem}
.green{color:#3fb950}.yellow{color:#d29922}.red{color:#f85149}
.blue{color:#58a6ff}.purple{color:#bc8cff}.orange{color:#f0883e}
.charts{display:grid;grid-template-columns:1fr 1fr;gap:.75rem;padding:0 1.5rem 1.5rem}
.chart-box{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:1rem}
.chart-box h3{font-size:.7rem;text-transform:uppercase;letter-spacing:.07em;color:#8b949e;margin-bottom:.7rem}
canvas{max-height:180px!important}
#splash{display:flex;flex-direction:column;align-items:center;justify-content:center;
        min-height:70vh;gap:.75rem;color:#8b949e;text-align:center}
#splash h2{font-size:1rem;color:#c9d1d9}
#splash code{background:#161b22;border:1px solid #30363d;padding:.3rem .8rem;
             border-radius:6px;font-size:.82rem;color:#58a6ff}
@media(max-width:640px){.charts{grid-template-columns:1fr}.kpis{grid-template-columns:repeat(2,1fr)}}
</style>
</head>
<body>

<header>
  <h1>⚙ Concrete Curing AI Monitor</h1>
  <span id="ts">— connecting —</span>
</header>

<div id="splash">
  <h2>Waiting for sensor data…</h2>
  <p>Start the monitor in another terminal:</p>
  <code>python "concrete_monitor_refined (2).py" --demo</code>
</div>

<div id="dashboard" style="display:none">
  <div class="kpis">
    <div class="kpi"><div class="kpi-label">🌡 Temperature</div>
      <div class="kpi-val" id="v-temp">—</div><div class="kpi-sub" id="s-temp"></div></div>
    <div class="kpi"><div class="kpi-label">💧 Humidity</div>
      <div class="kpi-val" id="v-hum">—</div><div class="kpi-sub" id="s-hum"></div></div>
    <div class="kpi"><div class="kpi-label">🪨 Soil Moisture</div>
      <div class="kpi-val" id="v-soil">—</div><div class="kpi-sub" id="s-soil"></div></div>
    <div class="kpi"><div class="kpi-label">📊 Curing Rate</div>
      <div class="kpi-val" id="v-rate">—</div><div class="kpi-sub" id="s-rate"></div></div>
    <div class="kpi"><div class="kpi-label">🏆 Grade</div>
      <div class="kpi-val" id="v-grade">—</div><div class="kpi-sub"></div></div>
    <div class="kpi"><div class="kpi-label">❤️ Health</div>
      <div class="kpi-val" id="v-health">—</div><div class="kpi-sub" id="s-health"></div></div>
    <div class="kpi"><div class="kpi-label">⏱ Elapsed</div>
      <div class="kpi-val blue" id="v-elapsed">—</div><div class="kpi-sub">hours</div></div>
    <div class="kpi"><div class="kpi-label">📦 Readings</div>
      <div class="kpi-val blue" id="v-count">—</div><div class="kpi-sub"></div></div>
    <div class="kpi"><div class="kpi-label">🔔 Buzzer D9</div>
      <div class="kpi-val" id="v-buzzer">—</div><div class="kpi-sub"></div></div>
  </div>

  <div class="charts">
    <div class="chart-box"><h3>Curing Rate History</h3><canvas id="c-rate"></canvas></div>
    <div class="chart-box"><h3>Soil Moisture History (%)</h3><canvas id="c-soil"></canvas></div>
    <div class="chart-box"><h3>MLP Forecast — Next 12 s</h3><canvas id="c-forecast"></canvas></div>
    <div class="chart-box"><h3>Grade Distribution</h3><canvas id="c-grade"></canvas></div>
  </div>
</div>

<script>
// ── Chart helpers ─────────────────────────────────────────────────────────
function lineChart(id, label, color) {
  return new Chart(document.getElementById(id), {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label, data: [],
        borderColor: color,
        backgroundColor: color + '22',
        borderWidth: 2, fill: true, tension: 0.4, pointRadius: 0
      }]
    },
    options: {
      responsive: true, animation: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { display: false },
        y: { ticks: { color: '#8b949e', font: { size: 11 } },
             grid: { color: '#21262d' } }
      }
    }
  });
}

const rateChart = lineChart('c-rate',  'Rate %',  '#58a6ff');
const soilChart = lineChart('c-soil',  'Soil %',  '#3fb950');

const fcChart = new Chart(document.getElementById('c-forecast'), {
  type: 'bar',
  data: { labels: [], datasets: [{ label: 'Forecast %', data: [],
    backgroundColor: '#bc8cff', borderRadius: 4 }] },
  options: {
    responsive: true, animation: false,
    plugins: { legend: { display: false } },
    scales: {
      y: { min: 0, max: 100, ticks: { color: '#8b949e' }, grid: { color: '#21262d' } },
      x: { ticks: { color: '#8b949e' } }
    }
  }
});

const gradeChart = new Chart(document.getElementById('c-grade'), {
  type: 'doughnut',
  data: { labels: [], datasets: [{ data: [],
    backgroundColor: ['#3fb950','#58a6ff','#d29922','#f0883e','#f85149'],
    borderWidth: 0 }] },
  options: {
    responsive: true, animation: false,
    plugins: { legend: { labels: { color: '#c9d1d9', font: { size: 11 } } } }
  }
});

// ── Update helpers ─────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

function tempColor(t)  { return t<5||t>38 ? 'red' : t>=15&&t<=30 ? 'green' : 'yellow'; }
function humColor(h)   { return h<40 ? 'red' : h<60 ? 'yellow' : 'green'; }
function soilColor(s)  { return s<20 ? 'red' : s>85 ? 'yellow' : 'green'; }
function rateColor(r)  { return r>70 ? 'green' : r>40 ? 'yellow' : 'red'; }
function gradeColor(g) { return {A:'green',B:'blue',C:'yellow',D:'orange',F:'red'}[g]||''; }
function healthColor(h){ return {Healthy:'green','At Risk':'yellow',Critical:'red'}[h]||''; }

function setKpi(id, val, cls, sub, subId) {
  const el = $(id);
  el.textContent = val;
  el.className   = 'kpi-val ' + (cls||'');
  if (subId) $(subId).textContent = sub||'';
}

function pushLine(chart, data) {
  const labels = data.map((_, i) => i);
  chart.data.labels              = labels;
  chart.data.datasets[0].data   = data;
  chart.update('none');   // 'none' = no animation = instant refresh
}

// ── Main update ────────────────────────────────────────────────────────────
function update(d) {
  $('splash').style.display    = 'none';
  $('dashboard').style.display = 'block';
  $('ts').textContent          = d.timestamp;

  setKpi('v-temp',   d.temp + ' °C',   tempColor(d.temp),
         d.temp<5?'Too Cold':d.temp>38?'Too Hot':d.temp>=15&&d.temp<=30?'Optimal':'Marginal', 's-temp');

  setKpi('v-hum',    d.hum  + ' %',    humColor(d.hum),
         d.hum<40?'Too Dry':d.hum<60?'Low':'Good', 's-hum');

  setKpi('v-soil',   d.soil + ' %',    soilColor(d.soil),
         d.soil<20?'Very Dry':d.soil>85?'Over-wet':'Good', 's-soil');

  setKpi('v-rate',   d.rate + '%',     rateColor(d.rate),
         d.rate>70?'Good':d.rate>40?'Progressing':'Slow', 's-rate');

  setKpi('v-grade',  d.grade,          gradeColor(d.grade));
  setKpi('v-health', d.health,         healthColor(d.health),
         d.health==='Healthy'?'✓ All good':d.health==='At Risk'?'⚠ Monitor':'🚨 Critical', 's-health');

  setKpi('v-elapsed', d.elapsed + ' h', 'blue');
  setKpi('v-count',   d.count,          'blue');

  const bz = d.buzzer === '1';
  setKpi('v-buzzer', bz ? 'ON' : 'off', bz ? 'red' : 'green');

  // Charts — replace data entirely then call update('none') for instant repaint
  pushLine(rateChart, d.rate_history);
  pushLine(soilChart, d.soil_history);

  fcChart.data.labels             = d.forecast.map((_,i) => '+' + (i+1)*2 + 's');
  fcChart.data.datasets[0].data  = d.forecast;
  fcChart.update('none');

  const gKeys = Object.keys(d.grade_counts);
  gradeChart.data.labels            = gKeys;
  gradeChart.data.datasets[0].data  = gKeys.map(k => d.grade_counts[k]);
  gradeChart.update('none');
}

// ── Polling ────────────────────────────────────────────────────────────────
async function poll() {
  try {
    const r = await fetch('/api/data');
    if (!r.ok) return;
    const d = await r.json();
    if (!d.empty) update(d);
  } catch(e) {}
}

// SSE push — triggers poll immediately on each ping
const es = new EventSource('/stream');
es.onmessage = () => poll();
es.onerror   = () => {};   // suppress console noise on reconnect

// Also poll on a 3-second fallback timer in case SSE hiccups
poll();
setInterval(poll, 3000);
</script>
</body>
</html>
"""

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(HTML)

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concrete Curing Dashboard")
    parser.add_argument("--demo", action="store_true", help="Generate simulated data")
    parser.add_argument("--port", type=int, default=5050, help="HTTP port (default 5050)")
    args = parser.parse_args()

    if args.demo:
        threading.Thread(target=_demo_loop, daemon=True).start()

    print(f"\n  ┌──────────────────────────────────────────┐")
    print(f"  │  Dashboard  →  http://localhost:{args.port}     │")
    print(f"  └──────────────────────────────────────────┘\n")

    app.run(host="0.0.0.0", port=args.port, debug=False, use_reloader=False)
