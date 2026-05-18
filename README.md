# 🏗️ Concrete Curing AI Monitor

An end-to-end IoT + Machine Learning system for real-time monitoring, prediction, and health assessment of concrete during the critical curing phase. Built with Arduino, Python 3.12, and four complementary AI models — **no TensorFlow required**.

> **Standards compliance:** ACI 308 · ASTM C1074 · Nurse-Saul Maturity Method

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [System Architecture](#system-architecture)
- [Hardware Setup](#hardware-setup)
- [AI Models](#ai-models)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Dashboard](#dashboard)
- [Training Data](#training-data)
- [Configuration](#configuration)
- [Known Issues & Fixes](#known-issues--fixes)
- [Tech Stack](#tech-stack)

---

## Overview

The Concrete Curing AI Monitor pairs a low-cost **Arduino Uno** sensor board with a **Python AI inference engine**. Every 2 seconds, live temperature, humidity, and soil moisture readings are processed by four machine learning models that predict curing rate, assign a grade (A–F), assess structural health, and forecast the next 12 seconds of curing progress.

Feedback is sent back to the Arduino to trigger a buzzer alert when conditions become critical. All readings are logged to CSV and visualised in a live **Plotly Dash** web dashboard.

**Demo mode** lets you run and test the full AI pipeline without any hardware.

---

## Key Results

| Model | Metric | Score |
|---|---|---|
| XGBoost — Curing Rate | R² | **0.987** |
| XGBoost — Curing Rate | MAE | ~2.1% |
| Random Forest — Grade (A–F) | Accuracy | ~94% |
| XGBoost — Health | Accuracy | ~96% |
| MLP — 6-step Forecast | MAE | ~3.4% |

Full inference cycle: **< 50 ms** · No GPU required

---

## System Architecture

```
[Arduino Uno]
    DHT22 (Temp + Humidity)  →  D2
    Capacitive Soil Sensor   →  A1
    Passive Buzzer           →  D3 (PWM)
         │
         │  Serial @ 9600 baud
         │  Arduino → Python : "22.5,68.3,452"
         │  Python  → Arduino: '0' (clear) | '1' (alert)
         ▼
[Python Inference Engine]
    Feature Engineering (7 features)
         │
         ├── XGBoost     → Curing Rate (0–100%)
         ├── Random Forest → Curing Grade (A–F)
         ├── XGBoost     → Health (Healthy / At Risk / Critical)
         └── MLP         → 6-step Rate Forecast
         │
         ├── Rich Terminal Dashboard (live)
         ├── Plotly Dash Web Dashboard → localhost:5050
         └── CSV Logger → logs/curing_log.csv
```

---

## Hardware Setup

| Component | Pin | Notes |
|---|---|---|
| DHT22 | D2 | 10kΩ pull-up resistor required. ±0.5°C accuracy |
| Capacitive Soil Sensor | A1 | Reads 0–1023. Ideal reading ~450 |
| Passive Buzzer | D3 (PWM) | `analogWrite(3, 5)` = soft alert hum |
| Arduino Uno | USB | Serial bidirectional, 9600 baud |

Upload `concrete_curing.ino` to the Arduino before running the Python monitor.

---

## AI Models

### Feature Engineering

Seven features are derived from just three physical sensors:

| Feature | Source | Scientific Basis |
|---|---|---|
| `temperature_c` | DHT22 | Primary driver of cement hydration (Arrhenius kinetics) |
| `humidity_pct` | DHT22 | ACI 308: min 60% RH; below 40% causes desiccation cracking |
| `soil_analog` | A1 | Water-to-cement ratio proxy; ideal ~450 |
| `elapsed_hours` | System clock | Curing follows S-curve; milestones at 7 and 28 days |
| `maturity_factor` | Computed | Nurse-Saul index (ASTM C1074): combines all 3 sensors into 0–1.3 score |
| `temp_humidity_ix` | Computed | `temp × humidity / 100` — combined environmental stress term |
| `water_deviation` | Computed | `abs(soil - 450)` — symmetric penalty for deviation from ideal moisture |

### Curing Grade Scale

| Grade | Condition | Recommended Action |
|---|---|---|
| A | ≥ 95% of expected rate | No action — optimal curing |
| B | 85–94% | Monitor; minor adjustment may help |
| C | 70–84% | Increase humidity (misting) or adjust insulation |
| D | 55–69% | Immediate review — humidity or temperature marginal |
| F | < 55% | Critical failure — freezing, heat damage, or dehydration |

### Buzzer Logic

| Signal | Condition |
|---|---|
| `'0'` — Silent | Health = Healthy AND temperature within 5–38°C |
| `'1'` — Alert hum | Health = Critical OR temperature outside 5–38°C |

---

## Project Structure

```
concrete-curing-ai/
│
├── concrete_monitor_refined.py   # Main real-time monitor (run this)
├── dashboard_minimal.py          # Plotly Dash web dashboard
├── generate_training_data.py     # Synthetic dataset generator
├── train_models.py               # Train all 4 AI models
├── Requirements.txt              # Python dependencies
│
├── concrete_curing.ino           # Arduino firmware
│
├── data/
│   └── training_data.csv         # Generated training data (gitignored)
│
├── models/                       # Saved model files (gitignored)
│   ├── xgb_curing_rate.pkl
│   ├── rf_curing_grade.pkl
│   ├── xgb_concrete_health.pkl
│   ├── mlp_forecast.pkl
│   ├── forecast_config.npy
│   ├── feature_scaler.pkl
│   ├── label_encoder_grade.pkl
│   └── label_encoder_health.pkl
│
└── logs/
    └── curing_log.csv            # Live session log
```

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/concrete-curing-ai.git
cd concrete-curing-ai

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r Requirements.txt
```

### 2. Generate Training Data

```bash
# Default: 2 million records (~240 MB, ~4 GB RAM)
python generate_training_data.py

# Low-RAM option (500K records, ~1 GB RAM)
python generate_training_data.py --records 500000

# High-accuracy option (5 million records, ~8 GB RAM)
python generate_training_data.py --records 5000000
```

### 3. Train the Models

```bash
python train_models.py
```

This saves all 8 model/encoder files to the `models/` directory.

### 4. Run the Monitor

```bash
# Demo mode (no Arduino needed — uses simulated data)
python concrete_monitor_refined.py --demo

# With Arduino (Windows)
python concrete_monitor_refined.py --port COM3

# With Arduino (Linux/Mac)
python concrete_monitor_refined.py --port /dev/ttyUSB0
```

The dashboard launches automatically at **http://localhost:5050**

---

## Usage

```
usage: concrete_monitor_refined.py [-h] [--port PORT] [--demo]

options:
  --port PORT   Serial port for Arduino (default: COM3)
  --demo        Run with simulated data — no Arduino required
```

The terminal displays a live Rich dashboard with all sensor readings, AI predictions, grade probabilities, and the MLP forecast panel. The dashboard URL is always visible inside the live display.

---

## Dashboard

The Plotly Dash web dashboard at **http://localhost:5050** auto-refreshes every 3 seconds and shows:

- **KPI cards** — current curing rate, grade, health status, and total reading count
- **Curing Rate Progress** — time-series chart with health overlaid as background colour
- **Grade Distribution** — pie chart of A–F grades across the session
- **Temperature & Humidity** — dual-axis time-series
- **Soil / Water Proxy** — trend chart with ideal moisture reference line

The dashboard reads `logs/curing_log.csv` and can be opened independently of the monitor.

---

## Training Data

Training data is fully synthetic, generated using the **Nurse-Saul maturity method** with realistic Gaussian drift per session. Grade and health labels are derived deterministically from computed rates and ACI 308 / ASTM C1074 thresholds.

| Records | CSV Size | RAM (Training) | Recommended For |
|---|---|---|---|
| 500,000 | ~60 MB | ~1 GB | Low-RAM / quick test |
| 2,000,000 | ~240 MB | ~4 GB | Production baseline |
| 5,000,000 | ~600 MB | ~8 GB | High accuracy |
| 10,000,000 | ~1.2 GB | ~16 GB | Maximum accuracy |

---

## Configuration

Key constants in `concrete_monitor_refined.py`:

```python
BAUD_RATE      = 9600          # Must match Arduino sketch
DASHBOARD_PORT = 5050          # Web dashboard port
DATUM_TEMP     = -10.0         # Nurse-Saul datum temperature (°C)
IDEAL_TEMP     = 23.0          # Optimal curing temperature (°C)
IDEAL_HUM      = 80.0          # Optimal relative humidity (%)
IDEAL_WATER    = 50            # Ideal soil moisture midpoint (%)
```

---

## Known Issues & Fixes

| Issue | Root Cause | Fix Applied |
|---|---|---|
| TensorFlow DLL failure on Windows | TF 2.16 requires Python ≤ 3.11 | Replaced LSTM with `sklearn` MLPRegressor |
| `ModuleNotFoundError: keras` | Old import in previous file | Replaced with TF-free version |
| `OSError: directory 'data' not found` | Directories not pre-created | Added `os.makedirs(..., exist_ok=True)` to all scripts |
| Progress bar garbled in PowerShell | `\r` not handled correctly | Switched to newline-per-update printing |
| Panel shown as memory address | `f-string` called `str()` on Rich objects | Used `rich.console.Group(fc_panel, prob_panel)` |
| Only 81K records generated | Old script lacked `--records` flag | Replaced with script supporting `--records N` |

---

## Tech Stack

| Category | Package | Version |
|---|---|---|
| Hardware | Arduino Uno | Sensor acquisition & buzzer |
| Language | Python | 3.12 |
| ML — Regression | xgboost | 2.0.3 |
| ML — Classification & Forecast | scikit-learn | 1.5.0 |
| Serial I/O | pyserial | 3.5 |
| Terminal UI | rich | 13.7.1 |
| Web Dashboard | dash + dash-bootstrap-components | 2.17.1 + 1.6.0 |
| Data | pandas / numpy | 2.2.2 / 1.26.4 |
| Visualisation | plotly | 5.22.0 |
| Model Persistence | joblib | 1.4.2 |

---

## Author

**Yashvardhan Chavan** — March 2026

*Built in accordance with ACI 308 and ASTM C1074 standards.*
