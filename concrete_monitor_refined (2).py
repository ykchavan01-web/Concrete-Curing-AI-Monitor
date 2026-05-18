"""
concrete_monitor_refined.py - Real-time Concrete Curing AI Monitor
-------------------------------------------------------------------
Usage:
  python concrete_monitor_refined.py --port COM3          (Windows)
  python concrete_monitor_refined.py --port /dev/ttyUSB0  (Linux/Mac)
  python concrete_monitor_refined.py --demo               (no Arduino, simulated data)

  Auto-launches dashboard_minimal.py and keeps URL visible in terminal.

Arduino:
  DHT22 -> D2 | Soil Sensor -> A0 (0-100%) | Buzzer -> D9 | LEDs -> D10-D13
  Serial out: "temp,humidity,soil_pct"   e.g. "28.4,65.2,73"
  Serial in:  '1' = alert  |  '0' = all clear
"""

import os, sys, time, argparse, warnings, subprocess, socket
warnings.filterwarnings("ignore")

import numpy as np
import joblib
import csv
from datetime import datetime
from collections import deque
from pathlib import Path

try:
    import serial
    SERIAL_OK = True
except ImportError:
    SERIAL_OK = False

from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.columns import Columns
from rich.console import Group
from rich.text import Text
from rich import box

# -- CONFIG -------------------------------------------------------------------
BAUD_RATE      = 9600
LOG_PATH       = Path("logs/curing_log.csv")
MODEL_DIR      = Path("models")
DASHBOARD_FILE = Path("dashboard_minimal.py")
DASHBOARD_PORT = 5050

DATUM_TEMP  = -10.0
IDEAL_TEMP  = 23.0
IDEAL_HUM   = 80.0
IDEAL_WATER = 50   # % moisture midpoint

console = Console()

# -- DASHBOARD LAUNCHER -------------------------------------------------------
def _is_port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0

def launch_dashboard(demo: bool):
    if not DASHBOARD_FILE.exists():
        console.print(f"[yellow]Warning: {DASHBOARD_FILE} not found — skipping.[/yellow]")
        return False

    if _is_port_open(DASHBOARD_PORT):
        console.print(f"[cyan]Dashboard already running.[/cyan]")
        return True

    cmd = [sys.executable, str(DASHBOARD_FILE)]
    if demo:
        cmd.append("--demo")

    subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=(subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0)
    )

    # Wait up to 12 s for Flask to come up
    for _ in range(24):
        time.sleep(0.5)
        if _is_port_open(DASHBOARD_PORT):
            return True
    return False

# -- HELPERS ------------------------------------------------------------------
def maturity_factor(t, h, w):
    tf_ = max(0, (t - DATUM_TEMP) / (IDEAL_TEMP - DATUM_TEMP))
    hf  = np.clip(h / IDEAL_HUM, 0, 1.2)
    wf  = np.clip(1 - abs(w - IDEAL_WATER) / 50, 0.3, 1.0)
    return float(np.clip(tf_ * hf * wf, 0, 1.3))

def build_feature_row(temp, humidity, soil_pct, elapsed_h):
    mf   = maturity_factor(temp, humidity, soil_pct)
    thi  = temp * humidity / 100
    wdev = abs(soil_pct - IDEAL_WATER)
    return [temp, humidity, soil_pct, elapsed_h, mf, thi, wdev]

def grade_color(grade):
    return {"A":"green","B":"cyan","C":"yellow","D":"orange1","F":"red"}.get(grade,"white")

def health_color(health):
    return {"Healthy":"green","At Risk":"yellow","Critical":"red"}.get(health,"white")

# -- LOAD MODELS --------------------------------------------------------------
def load_models():
    console.log("[bold cyan]Loading AI models...[/bold cyan]")
    m = {}
    try:
        m["scaler"]     = joblib.load(MODEL_DIR / "feature_scaler.pkl")
        m["xgb_rate"]   = joblib.load(MODEL_DIR / "xgb_curing_rate.pkl")
        m["rf_grade"]   = joblib.load(MODEL_DIR / "rf_curing_grade.pkl")
        m["xgb_health"] = joblib.load(MODEL_DIR / "xgb_concrete_health.pkl")
        m["le_grade"]   = joblib.load(MODEL_DIR / "label_encoder_grade.pkl")
        m["le_health"]  = joblib.load(MODEL_DIR / "label_encoder_health.pkl")
        m["mlp"]        = joblib.load(MODEL_DIR / "mlp_forecast.pkl")
        cfg = np.load(MODEL_DIR / "forecast_config.npy")
        m["SEQ_LEN"]    = int(cfg[0])
        m["PRED_STEPS"] = int(cfg[1])
        m["N_FEAT"]     = int(cfg[2])
        console.log("[green]All 4 models loaded.[/green]")
    except FileNotFoundError as e:
        console.print(f"[red]Model not found: {e}[/red]")
        console.print("[yellow]Run: python train_models.py first![/yellow]")
        sys.exit(1)
    return m

# -- INFERENCE ----------------------------------------------------------------
def predict(models, feature_row, seq_buffer):
    scaler      = models["scaler"]
    row_sc      = scaler.transform([feature_row])
    rate        = float(np.clip(models["xgb_rate"].predict(row_sc)[0], 0, 100))
    g_enc       = models["rf_grade"].predict(row_sc)[0]
    grade       = models["le_grade"].inverse_transform([g_enc])[0]
    grade_proba = models["rf_grade"].predict_proba(row_sc)[0]
    h_enc       = models["xgb_health"].predict(row_sc)[0]
    health      = models["le_health"].inverse_transform([h_enc])[0]
    forecast    = None
    SEQ_LEN     = models["SEQ_LEN"]
    if len(seq_buffer) == SEQ_LEN:
        seq_arr  = np.array(list(seq_buffer), dtype=np.float32)
        seq_sc   = np.hstack([scaler.transform(seq_arr[i:i+1]) for i in range(SEQ_LEN)])
        forecast = np.clip(models["mlp"].predict(seq_sc)[0], 0, 100)
    return rate, grade, grade_proba, health, forecast

# -- DEMO GENERATOR -----------------------------------------------------------
def demo_generator():
    temp, hum, soil_pct = 22.0, 72.0, 55.0
    while True:
        temp     += np.random.normal(0, 0.3);  temp     = float(np.clip(temp,     5, 42))
        hum      += np.random.normal(0, 0.8);  hum      = float(np.clip(hum,     20, 99))
        soil_pct += np.random.normal(0, 2.5);  soil_pct = float(np.clip(soil_pct, 0,100))
        yield round(temp,1), round(hum,1), round(soil_pct,1)
        time.sleep(2)

# -- SERIAL PARSING -----------------------------------------------------------
def parse_serial_line(line: str):
    """
    Accepts: "28.4,65.2,73"
    soil_pct must be 0-100 (Arduino does map() before sending).
    Silently skips boot messages and malformed lines.
    """
    line = line.strip()
    if not line or line.startswith("ERROR") or line.startswith("Concrete"):
        if line:
            console.log(f"[yellow]Arduino: {line}[/yellow]")
        return None
    parts = line.split(",")
    if len(parts) != 3:
        return None
    try:
        temp     = float(parts[0])
        humidity = float(parts[1])
        soil_pct = float(parts[2])
        # Reject obviously wrong values
        if not (-10 <= temp     <=  80): return None
        if not (  0 <= humidity <= 100): return None
        if not (  0 <= soil_pct <= 100): return None
        return temp, humidity, soil_pct
    except ValueError:
        return None

# -- STATUS -------------------------------------------------------------------
def temp_status(t):
    if t < 5:         return "[red]Too Cold[/red]"
    if t > 38:        return "[red]Too Hot[/red]"
    if 15 <= t <= 30: return "[green]Optimal[/green]"
    return "[yellow]Marginal[/yellow]"

def hum_status(h):
    if h < 40: return "[red]Too Dry[/red]"
    if h < 60: return "[yellow]Low[/yellow]"
    return "[green]Good[/green]"

def soil_status(s):
    if s < 20:  return "[red]Very Dry[/red]"
    if s > 85:  return "[yellow]Over-wet[/yellow]"
    if s >= 40: return "[green]Good[/green]"
    return "[yellow]Low[/yellow]"

# -- MAIN ---------------------------------------------------------------------
def run(port=None, demo=False):
    models     = load_models()
    SEQ_LEN    = models["SEQ_LEN"]
    PRED_STEPS = models["PRED_STEPS"]
    seq_buffer = deque(maxlen=SEQ_LEN)
    start_time = time.time()

    LOG_PATH.parent.mkdir(exist_ok=True)
    log_file   = open(LOG_PATH, "a", newline="")
    log_writer = csv.DictWriter(log_file, fieldnames=[
        "timestamp","elapsed_h","temperature_c","humidity_pct","soil_pct",
        "curing_rate","grade","health","buzzer"
    ])
    if LOG_PATH.stat().st_size == 0:
        log_writer.writeheader()

    # -- Connect serial -------------------------------------------------------
    ser = None
    if not demo:
        if not SERIAL_OK:
            console.print("[red]pyserial missing. Run: pip install pyserial[/red]")
            sys.exit(1)
        try:
            ser = serial.Serial(port, BAUD_RATE, timeout=3)
            console.log(f"[green]Connected to {port} @ {BAUD_RATE} baud[/green]")
            time.sleep(2)
        except serial.SerialException as e:
            console.print(f"[red]Cannot open {port}: {e}[/red]")
            sys.exit(1)

    # -- Launch dashboard AFTER serial is confirmed ---------------------------
    console.print()
    console.rule("[bold cyan]Starting Dashboard")
    dash_ok = launch_dashboard(demo)

    # Print URL clearly BEFORE Live starts so it is never overwritten
    url = f"http://localhost:{DASHBOARD_PORT}"
    if dash_ok:
        console.print()
        console.print(f"  [bold white on green]  DASHBOARD LIVE: {url}  [/bold white on green]")
    else:
        console.print(f"  [yellow]Dashboard not responding. Try {url} manually.[/yellow]")
    console.print()
    console.rule()
    time.sleep(1)   # Give user 1 s to read the URL before Live takes over

    # -- Helpers --------------------------------------------------------------
    demo_gen = demo_generator() if demo else None

    def get_reading():
        if demo:
            return next(demo_gen)
        try:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            return parse_serial_line(line)
        except Exception:
            return None

    def send_feedback(good: bool):
        if ser:
            try: ser.write(b'0' if good else b'1')
            except Exception: pass

    reading_count = 0

    # -- Live display ---------------------------------------------------------
    with Live(console=console, refresh_per_second=1) as live:
        while True:
            reading = get_reading()
            if reading is None:
                continue

            temp, hum, soil_pct = reading
            elapsed_h = (time.time() - start_time) / 3600
            feat_row  = build_feature_row(temp, hum, soil_pct, elapsed_h)
            seq_buffer.append(feat_row)

            rate, grade, grade_proba, health, forecast = predict(models, feat_row, seq_buffer)
            reading_count += 1

            is_bad = (health == "Critical") or (temp < 5 or temp > 38)
            send_feedback(not is_bad)
            buzzer_str = "[red]ON[/red]" if is_bad else "[green]off[/green]"

            log_writer.writerow({
                "timestamp"    : datetime.now().isoformat(timespec="seconds"),
                "elapsed_h"    : round(elapsed_h, 4),
                "temperature_c": temp,
                "humidity_pct" : hum,
                "soil_pct"     : soil_pct,   # 0-100 %
                "curing_rate"  : round(rate, 2),
                "grade"        : grade,
                "health"       : health,
                "buzzer"       : "1" if is_bad else "0"
            })
            log_file.flush()

            # Sensor table
            tbl = Table(box=box.ROUNDED, show_header=True,
                        header_style="bold magenta", min_width=60)
            tbl.add_column("Parameter", style="bold", width=22)
            tbl.add_column("Value",     width=16)
            tbl.add_column("Status",    width=20)

            tbl.add_row("Temperature",   f"{temp:.1f} C",      temp_status(temp))
            tbl.add_row("Humidity",      f"{hum:.1f} %RH",     hum_status(hum))
            tbl.add_row("Soil Moisture", f"{soil_pct:.1f} %",  soil_status(soil_pct))
            tbl.add_row("", "", "")
            tbl.add_row("Elapsed",       f"{elapsed_h:.2f} h", "")
            tbl.add_row(
                "Curing Rate",
                f"[bold]{rate:.1f}%[/bold]",
                f"[{'green' if rate>70 else 'yellow' if rate>40 else 'red'}]"
                f"{'Good' if rate>70 else 'Progressing' if rate>40 else 'Slow'}[/]"
            )
            tbl.add_row("Grade",
                f"[bold {grade_color(grade)}]{grade}[/bold {grade_color(grade)}]", "")
            tbl.add_row("Health",
                f"[bold {health_color(health)}]{health}[/bold {health_color(health)}]", "")
            tbl.add_row("Buzzer (D9)", buzzer_str, "")
            # URL always visible inside the live table
            tbl.add_row(
                "[bold cyan]Dashboard URL[/bold cyan]",
                f"[bold cyan underline]{url}[/bold cyan underline]",
                "[green]LIVE[/green]" if dash_ok else "[yellow]CHECK[/yellow]"
            )

            # Forecast panel
            if forecast is not None:
                fc_str = " ".join(
                    f"[{'green' if v>70 else 'yellow' if v>40 else 'red'}]{v:.0f}%[/]"
                    for v in forecast
                )
                fc_panel = Panel(f"Next {PRED_STEPS*2}s -> {fc_str}",
                                 title="[bold cyan]MLP Forecast[/bold cyan]",
                                 border_style="cyan")
            else:
                remaining = SEQ_LEN - len(seq_buffer)
                fc_panel  = Panel(
                    f"Collecting {remaining} more reading{'s' if remaining!=1 else ''}...",
                    title="[bold cyan]MLP Forecast[/bold cyan]",
                    border_style="dim")

            classes  = models["le_grade"].classes_
            prob_str = " ".join(
                f"[{grade_color(c)}]{c}[/]:{p*100:.0f}%"
                for c, p in zip(classes, grade_proba)
            )
            prob_panel = Panel(prob_str, title="Grade Probabilities", border_style="magenta")

            main_panel = Panel(
                tbl,
                title=(f"[bold white]CONCRETE CURING MONITOR[/bold white] "
                       f"[dim]#{reading_count} · {datetime.now().strftime('%H:%M:%S')}[/dim]"),
                border_style="blue"
            )

            live.update(Columns([main_panel, Group(fc_panel, prob_panel)]))

# -- ENTRY POINT --------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concrete Curing AI Monitor")
    parser.add_argument("--port", default="COM3",
                        help="Serial port (e.g. COM3 or /dev/ttyUSB0)")
    parser.add_argument("--demo", action="store_true",
                        help="Run without Arduino using simulated data")
    args = parser.parse_args()
    try:
        run(port=args.port, demo=args.demo)
    except KeyboardInterrupt:
        console.print("\n[yellow]Monitor stopped. Logs saved.[/yellow]")
