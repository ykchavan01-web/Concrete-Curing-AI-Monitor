"""
generate_training_data.py
Generates synthetic concrete curing data for model training.

Usage:
    python generate_training_data.py                     -> 2,000,000 records
    python generate_training_data.py --records 5000000   -> 5 million
    python generate_training_data.py --records 500000    -> 500K (low RAM)
"""

import argparse, csv, os, time
import numpy as np
from pathlib import Path

os.makedirs("data",   exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("logs",   exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--records", type=int, default=2_000_000,
                    help="Target number of records (default: 2,000,000)")
parser.add_argument("--chunk",   type=int, default=50_000,
                    help="Rows per disk-write chunk (default: 50,000)")
args = parser.parse_args()

TARGET  = args.records
CHUNK   = args.chunk

# ─── CONSTANTS ───────────────────────────────────────────────────────────────
DATUM_TEMP  = -10.0
IDEAL_TEMP  = 23.0
IDEAL_HUM   = 80.0
IDEAL_WATER = 450
np.random.seed(42)

# ─── FUNCTIONS ───────────────────────────────────────────────────────────────
def maturity_factor(temp, hum, water):
    tf_ = max(0.0, (temp - DATUM_TEMP) / (IDEAL_TEMP - DATUM_TEMP))
    hf  = float(np.clip(hum / IDEAL_HUM, 0, 1.2))
    wf  = float(np.clip(1 - abs(water - IDEAL_WATER) / 400, 0.3, 1.0))
    return float(np.clip(tf_ * hf * wf, 0, 1.3))

def curing_rate(temp, hum, water, hours, mf):
    base = 100.0 * mf * (1.0 - np.exp(-hours / (15.0 + (1.0 - mf) * 30.0)))
    if temp < 2.0:  base *= 0.1
    if temp > 40.0: base *= max(0.0, 1.0 - (temp - 40.0) * 0.05)
    return float(np.clip(base + np.random.normal(0, 1.5), 0, 100))

def grade(rate, hours):
    if   hours <= 72:  exp = (hours / 72.0) * 50.0
    elif hours <= 168: exp = 50.0 + ((hours - 72.0) / 96.0) * 15.0
    else:              exp = 65.0 + min(35.0, (hours - 168.0) / 504.0 * 35.0)
    r = rate / max(exp, 1.0)
    if   r >= 0.95: return "A"
    elif r >= 0.85: return "B"
    elif r >= 0.70: return "C"
    elif r >= 0.55: return "D"
    else:           return "F"

def health(temp, hum, water, rate, hours):
    i = 0
    if temp < 5 or temp > 38:      i += 2
    if hum < 40:                   i += 2
    elif hum < 60:                 i += 1
    if water < 100 or water > 800: i += 1
    if rate < 30 and hours > 24:   i += 2
    return "Healthy" if i == 0 else "At Risk" if i <= 2 else "Critical"

FIELDS = [
    "session_id","elapsed_hours","temperature_c","humidity_pct","soil_analog",
    "curing_rate_pct","curing_grade","concrete_health",
    "maturity_factor","temp_humidity_ix","water_deviation"
]

# ─── MAIN ────────────────────────────────────────────────────────────────────
OUT = Path("data/training_data.csv")

print("=" * 60)
print(f"  GENERATING {TARGET:,} RECORDS")
print(f"  Output : {OUT}")
print(f"  Chunk  : {CHUNK:,} rows per flush")
print("=" * 60)

# Print every this many records
PRINT_EVERY = max(CHUNK, TARGET // 50)   # ~50 progress updates total

t0         = time.time()
written    = 0
session_id = 0
buf        = []
last_print = 0

with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS)
    w.writeheader()

    while written < TARGET:
        bt  = np.random.uniform(5, 40)
        bh  = np.random.uniform(30, 95)
        bwa = np.random.uniform(100, 850)
        hrs = np.random.randint(2, 673)

        for hour in range(0, hrs, 2):
            if written >= TARGET:
                break

            temp  = float(np.clip(bt  + np.random.normal(0, 1.5), -5, 50))
            hum   = float(np.clip(bh  + np.random.normal(0, 2.0), 10, 100))
            water = float(np.clip(bwa + np.random.normal(0, 20),    0, 1023))
            eh    = hour + 1
            mf    = maturity_factor(temp, hum, water)
            cr    = curing_rate(temp, hum, water, eh, mf)

            buf.append({
                "session_id"      : session_id,
                "elapsed_hours"   : eh,
                "temperature_c"   : round(temp,  2),
                "humidity_pct"    : round(hum,   2),
                "soil_analog"     : round(water, 1),
                "curing_rate_pct" : round(cr,    2),
                "curing_grade"    : grade(cr, eh),
                "concrete_health" : health(temp, hum, water, cr, eh),
                "maturity_factor" : round(mf, 4),
                "temp_humidity_ix": round(temp * hum / 100, 3),
                "water_deviation" : round(abs(water - IDEAL_WATER), 1),
            })
            written += 1

            # Flush chunk to disk
            if len(buf) >= CHUNK:
                w.writerows(buf)
                f.flush()
                buf.clear()

            # Print progress on a NEW LINE each time (works in all terminals)
            if written - last_print >= PRINT_EVERY:
                last_print = written
                elapsed    = time.time() - t0
                speed      = written / elapsed          # rows/sec
                eta_s      = (TARGET - written) / max(speed, 1)
                pct        = written / TARGET * 100
                bar_done   = int(pct / 2)
                bar        = "#" * bar_done + "-" * (50 - bar_done)
                print(
                    f"  [{bar}] {pct:5.1f}%"
                    f"  {written:>10,} / {TARGET:,}"
                    f"  {speed:,.0f} rows/s"
                    f"  ETA {eta_s/60:.1f} min"
                )

        session_id += 1

    # Final flush
    if buf:
        w.writerows(buf)

elapsed = time.time() - t0
size_mb = OUT.stat().st_size / 1e6
print()
print("=" * 60)
print(f"  DONE!")
print(f"  Records   : {written:,}")
print(f"  Sessions  : {session_id:,}")
print(f"  Time      : {elapsed/60:.1f} min  ({elapsed:.0f}s)")
print(f"  CSV size  : {size_mb:.1f} MB")
print("=" * 60)
print(f"\nNext step: python train_models.py")
