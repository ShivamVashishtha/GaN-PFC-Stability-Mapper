
from __future__ import annotations

import difflib
import math
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta

try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

# ==============================
# USER CONFIGURATION
# ==============================

LTSPICE_EXE = r"C:\Users\vashi\AppData\Local\Programs\ADI\LTspice\LTspice.exe"

PROJECT_DIR = Path(
    r"C:\Users\vashi\Downloads\a-010_ac-dc_totem-pole_bridgeless_pfc_vin=200v_iin=100a_synchronous_fets"
    r"\a-010_ac-dc_totem-pole_bridgeless_pfc_vin=200v_iin=100a_synchronous_fets"
)

OUTPUT_DIR = Path(r"C:\Users\vashi\bmw_gan")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==============================
# OUTPUT STRUCTURE
# ==============================

RUN_ID = time.strftime("%Y%m%d_%H%M%S")

DATA_DIR = OUTPUT_DIR / "data"
MASTER_DATA_DIR = DATA_DIR / "master"
RUNS_DIR = DATA_DIR / "runs"
REPORTS_DIR = OUTPUT_DIR / "reports"
PLOTS_DIR = OUTPUT_DIR / "plots"
TEMPLATES_DIR = OUTPUT_DIR / "templates"
ARCHIVE_DIR = OUTPUT_DIR / "archive"

RUN_DIR = RUNS_DIR / RUN_ID
RUN_CASES_DIR = RUN_DIR / "cases"
RUN_LOGS_DIR = RUN_DIR / "logs"
RUN_NETLISTS_DIR = RUN_DIR / "netlists"
RUN_RAW_DIR = RUN_DIR / "raw"
RUN_DIFFS_DIR = RUN_DIR / "diffs"
RUN_REPORTS_DIR = RUN_DIR / "reports"
RUN_PLOTS_DIR = RUN_DIR / "plots"

for d in [
    DATA_DIR, MASTER_DATA_DIR, RUNS_DIR, REPORTS_DIR, PLOTS_DIR,
    TEMPLATES_DIR, ARCHIVE_DIR, RUN_DIR, RUN_CASES_DIR, RUN_LOGS_DIR,
    RUN_NETLISTS_DIR, RUN_RAW_DIR, RUN_DIFFS_DIR, RUN_REPORTS_DIR, RUN_PLOTS_DIR
]:
    d.mkdir(parents=True, exist_ok=True)

MASTER_DATASET_XLSX = MASTER_DATA_DIR / "gan_pfc_master_dataset.xlsx"
MASTER_DATASET_CSV = MASTER_DATA_DIR / "gan_pfc_master_dataset.csv"

TEMPLATE_ASC = PROJECT_DIR / "gan_pfc_template_real_load.asc"
KNOWN_GOOD_TEMPLATE_NET = PROJECT_DIR / "gan_pfc_template.net"
TEMPLATE_MODE = "real_load"

ARCHIVE_LEVEL = "debug"
# options:
# "minimal"      -> only datasets, plots, reports
# "debug"        -> save logs/netlists/asc for failed cases and top cases
# "full_archive" -> save everything including raw files

MANUAL_CASES: List[Tuple[float, float, float]] = []

# Run modes:
#   "single_known_good" : only Rg_on=2, Rg_off=1. Use first.
#   "rg_sweep"          : wider Rg_on/Rg_off sweep.
#   "full_sweep"        : Rg_on/Rg_off/f_sw/dead-time sweep. Only works if f_sw and dead-time params exist.
RUN_MODE = "single_known_good"

# Baseline first. After it works, change RUN_MODE to "rg_sweep".
if RUN_MODE == "single_known_good":
    rg_on_values = [2]
    rg_off_values = [1]
    fsw_values = [None]
    deadtime_values = [None]
elif RUN_MODE == "rg_sweep":
    rg_on_values = [0.5, 1, 2, 3, 5, 7.5, 10]
    rg_off_values = [0.5, 1, 2, 3, 5, 7.5, 10]
    fsw_values = [None]
    deadtime_values = [None]
elif RUN_MODE == "full_sweep":
    rg_on_values = [0.5, 1, 2, 3, 5, 7.5]
    rg_off_values = [0.5, 1, 2, 3, 5, 7.5]
    fsw_values = [65_000, 100_000, 150_000, 200_000]
    deadtime_values = [20e-9, 50e-9, 100e-9, 150e-9]
else:
    raise ValueError(f"Unknown RUN_MODE: {RUN_MODE}")

# Parameter name akas. The script basically patches whichever alias is present in the .asc .param line.
PARAM_ALIASES = {
    "Rg_on": ["Rg_on", "Rgon", "RGON", "RG_ON"],
    "Rg_off": ["Rg_off", "Rgoff", "RGOFF", "RG_OFF"],
    "f_sw": ["f_sw", "fsw", "Fsw", "FSW", "Freq", "freq", "SwitchFreq"],
    "deadtime": ["deadtime", "DeadTime", "DT", "dt", "Tdead", "tdead"],
    "Rload_val": ["Rload_val", "Rload", "RL", "R_load"],
    "Cbus_val": ["Cbus_val", "Cbus", "C_bus", "C_BUS"],
    "Lboost_val": ["Lboost_val", "Lboost", "L_boost", "LBOOST"],
}

# Rload sweep values (prioritized outer loop)
rload_values = [20, 22, 24, 25, 26, 28, 30, 35, 40]

# Additional circuit design knobs (for future component sizing sweeps)
cbus_values = [2200e-6, 3300e-6, 4700e-6, 6800e-6]
lboost_values = [50e-6, 75e-6, 100e-6, 150e-6]

# Design / scoring profiles.
DESIGN_MODE = "balanced"
SCORING_PROFILES = {
    "balanced": {
        "eff": 0.35,
        "pf": 0.20,
        "regulation": 0.20,
        "ripple": 0.15,
        "ipeak": 0.10,
    },
    "high_efficiency": {
        "eff": 0.55,
        "pf": 0.15,
        "regulation": 0.15,
        "ripple": 0.10,
        "ipeak": 0.05,
    },
    "best_regulation": {
        "eff": 0.20,
        "pf": 0.20,
        "regulation": 0.40,
        "ripple": 0.10,
        "ipeak": 0.10,
    },
    "low_ripple": {
        "eff": 0.25,
        "pf": 0.15,
        "regulation": 0.20,
        "ripple": 0.30,
        "ipeak": 0.10,
    },
    "low_current_stress": {
        "eff": 0.20,
        "pf": 0.15,
        "regulation": 0.20,
        "ripple": 0.10,
        "ipeak": 0.35,
    },
}

DEVICE_VOLTAGE_RATING = 650.0
VOLTAGE_DERATING = 0.80
INDUCTOR_SATURATION_LIMIT_A = 120.0
THERMAL_RTH_EQUIV_C_PER_W = 0.03
AMBIENT_C = 85.0

# Existing template logs already contain these measures.
REQUIRED_MEASURES = ["pin", "iin_rms", "iin_pk", "vout_avg", "vout_pp"]

# Add low-risk LTspice measurements to the generated case schematic only.
# PF is calculated in Python from pin/(vin_rms*iin_rms), so only vin_rms must be injected.
INJECT_EXTRA_MEASURES = False
EXTRA_MEASURE_DIRECTIVES = []

# Physics-health thresholds. Can be tuned later once more real cases are obtained.
MIN_GOOD_ELAPSED_S = 10.0
MIN_PIN_W = 1000.0
MIN_IIN_RMS_A = 1.0
MIN_IIN_PK_A = 5.0
MAX_IIN_RMS_A = 120.0
VOUT_MIN_V = 100.0
VOUT_MAX_V = 600.0
VOUT_TARGET_V = 400.0
PF_TARGET_MIN = 0.90
EFF_TARGET_MIN = 90.0

# If the netlist contains an output resistor connected to OUT and ground, the script estimates Pout = Vout_avg^2/R.
# This is an approximation unless OUT is clean DC and the load is purely resistive.
ENABLE_POUT_FROM_LOAD_RESISTOR = True

# ==============================
# UTILITY FUNCTIONS
# ==============================


def read_bytes(path: Path) -> bytes:
    return path.read_bytes()


def decode_asc(data: bytes) -> str:
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            pass
    return data.decode("latin-1", errors="ignore")


def encode_asc(text: str) -> bytes:
    return text.encode("cp1252", errors="replace")


def ltspice_number(value: float | int | None) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and abs(value) < 1e-6 and value != 0:
        return f"{value:.12g}"
    return f"{value:g}"


def safe_case_value(value: float | int | None) -> str:
    if value is None:
        return "base"
    s = f"{value:g}"
    return s.replace(".", "p").replace("-", "m").replace("+", "")


# ==============================
# TERMINAL DASHBOARD HELPERS
# ==============================

USE_COLOR = True

class Term:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    GRAY = "\033[90m"

def c(text: str, color: str) -> str:
    if not USE_COLOR:
        return text
    return f"{color}{text}{Term.RESET}"

def fmt_num(x, unit="", digits=3, width=10):
    try:
        if x is None or np.isnan(float(x)):
            return f"{'—':>{width}}"
        return f"{float(x):>{width}.{digits}f}{unit}"
    except Exception:
        return f"{str(x):>{width}}{unit}"

def fmt_pct(x, width=8):
    try:
        if x is None or np.isnan(float(x)):
            return f"{'—':>{width}}"
        return f"{float(x):>{width}.2f}%"
    except Exception:
        return f"{'—':>{width}}"

def fmt_status(stable: int, bad_physics: bool, reason: str) -> str:
    if stable == 1 and not bad_physics:
        return c("PASS", Term.GREEN + Term.BOLD)
    if "timeout" in str(reason).lower() or "failed" in str(reason).lower():
        return c("FAIL", Term.RED + Term.BOLD)
    return c("WARN", Term.YELLOW + Term.BOLD)

def fmt_eta(seconds: float) -> str:
    if seconds is None or np.isnan(seconds) or seconds < 0:
        return "—"
    return str(timedelta(seconds=int(seconds)))

def print_banner(title: str) -> None:
    line = "=" * 86
    print(c("\n" + line, Term.CYAN))
    print(c(title.center(86), Term.CYAN + Term.BOLD))
    print(c(line, Term.CYAN))


def ask_user_mode() -> str:
    print_banner("GAN PFC WORKFLOW SELECTOR")
    print("1) Generate reports from existing data")
    print("2) Run new LTspice simulation")
    print("3) Run new simulation and append to master dataset")
    print("4) Exit")

    choice = input("\nSelect option [1/2/3/4] (default 3): ").strip() or "3"

    if choice == "1":
        return "report_only"
    if choice == "2":
        return "simulate_only"
    if choice == "3":
        return "simulate_append"
    if choice == "4":
        return "exit"

    print("Invalid choice. Defaulting to report-only mode.")
    return "report_only"


def parse_float_list(text: str, default: List[float]) -> List[float]:
    text = text.strip()
    if not text:
        return default

    try:
        values = []
        for item in text.split(","):
            item = item.strip()
            if item:
                values.append(float(item))
        return values if values else default
    except ValueError:
        print("Invalid numeric list. Using default.")
        return default


def configure_test_run() -> Tuple[List[float], List[float], List[float], List[Optional[float]], List[Optional[float]]]:
    print_banner("MANUAL TEST-BENCH SETUP")
    print("Select test type:")
    print("1) Known-good sanity test")
    print("2) Single custom case")
    print("3) Small Rload sweep")
    print("4) Small gate-resistance sweep")
    print("5) Custom N-case manual list")

    choice = input("\nSelect test type [1/2/3/4/5] (default 1): ").strip() or "1"

    if choice == "1":
        return [22], [2], [1], [None], [None]

    if choice == "2":
        rload = float(input("Rload Ω [default 22]: ").strip() or "22")
        rg_on = float(input("Rg_on Ω [default 2]: ").strip() or "2")
        rg_off = float(input("Rg_off Ω [default 1]: ").strip() or "1")
        return [rload], [rg_on], [rg_off], [None], [None]

    if choice == "3":
        rloads = parse_float_list(
            input("Rload values comma-separated [default 20,22,25]: "),
            [20, 22, 25],
        )
        rg_on = float(input("Fixed Rg_on Ω [default 2]: ").strip() or "2")
        rg_off = float(input("Fixed Rg_off Ω [default 1]: ").strip() or "1")
        return rloads, [rg_on], [rg_off], [None], [None]

    if choice == "4":
        rload = float(input("Fixed Rload Ω [default 22]: ").strip() or "22")
        rg_ons = parse_float_list(
            input("Rg_on values comma-separated [default 1,2,3]: "),
            [1, 2, 3],
        )
        rg_offs = parse_float_list(
            input("Rg_off values comma-separated [default 0.5,1,2]: "),
            [0.5, 1, 2],
        )
        return [rload], rg_ons, rg_offs, [None], [None]

    if choice == "5":
        print("\nManual N-case mode will ask each case one by one.")
        n = int(input("How many cases? [default 2]: ").strip() or "2")

        manual_cases = []
        for i in range(n):
            print(f"\nCase {i + 1}/{n}")
            rload = float(input("  Rload Ω [default 22]: ").strip() or "22")
            rg_on = float(input("  Rg_on Ω [default 2]: ").strip() or "2")
            rg_off = float(input("  Rg_off Ω [default 1]: ").strip() or "1")
            manual_cases.append((rload, rg_on, rg_off))

        # Store globally for manual-case execution.
        global MANUAL_CASES
        MANUAL_CASES = manual_cases

        # These lists are placeholders. The main loop will use MANUAL_CASES directly.
        return [], [], [], [None], [None]

    print("Invalid selection. Using known-good sanity test.")
    return [22], [2], [1], [None], [None]


def append_to_master_dataset(run_df: pd.DataFrame) -> pd.DataFrame:
    if run_df.empty:
        return run_df

    run_df = run_df.copy()
    run_df["Run_ID"] = RUN_ID
    run_df["Run_Timestamp"] = RUN_ID
    run_df["Template_Mode"] = TEMPLATE_MODE
    run_df["Run_Mode"] = RUN_MODE

    if MASTER_DATASET_XLSX.exists():
        master_df = pd.read_excel(MASTER_DATASET_XLSX)
        combined = pd.concat([master_df, run_df], ignore_index=True)
    else:
        combined = run_df

    key_cols = [
        "Run_ID", "Rload_val", "Rg_on", "Rg_off",
        "f_sw_Hz", "Deadtime_s", "Pin_W", "Pout_W",
        "Vout_avg_V", "PF", "Score"
    ]
    existing_keys = [c for c in key_cols if c in combined.columns]
    if existing_keys:
        combined = combined.drop_duplicates(subset=existing_keys, keep="last")

    combined.to_excel(MASTER_DATASET_XLSX, index=False)
    combined.to_csv(MASTER_DATASET_CSV, index=False)

    return combined


def generate_reports_from_existing_data() -> None:
    dataset_candidates = [
        MASTER_DATASET_XLSX,
        RUN_DIR / "run_dataset.xlsx",
        OUTPUT_DIR / "gan_pfc_innovation_dataset.xlsx",
        DATA_DIR / "gan_pfc_innovation_dataset.xlsx",
    ]

    dataset_path = None
    for p in dataset_candidates:
        if p.exists():
            dataset_path = p
            break

    if dataset_path is None:
        raise FileNotFoundError("No existing dataset found. Run a simulation first.")

    print(f"Loading existing dataset: {dataset_path}")
    df = pd.read_excel(dataset_path)

    if "Run_ID" in df.columns and df["Run_ID"].notna().any():
        latest_run = df["Run_ID"].dropna().astype(str).sort_values().iloc[-1]
        print(f"Using latest run: {latest_run}")
        df_run = df[df["Run_ID"].astype(str) == latest_run].copy()
    else:
        df_run = df.copy()

    save_design_plots(df_run)
    generate_executive_brief(df_run)
    generate_run_summary(df_run)
    multi_objective_selector(df_run)
    best_by_category(df_run)
    risk_register(df_run)
    ai_next_simulation_recommendations(df_run)
    application_mode_scores(df_run)
    generate_templates()
    generate_html_dashboard(df_run)
    print_final_leaderboard(df_run, n=10)

    print("\nReport generation complete.")
    print(f"Dashboard: {RUN_REPORTS_DIR / 'dashboard.html'}")

def print_case_header(case_idx: int, total_cases: int, case_tag: str, rload, rg_on, rg_off, f_sw, deadtime, elapsed_global_s: float, avg_case_s: float) -> None:
    remaining = max(total_cases - case_idx + 1, 0)
    eta = remaining * avg_case_s if avg_case_s > 0 else np.nan

    print(c("\n" + "-" * 86, Term.GRAY))
    print(
        c(f"CASE {case_idx}/{total_cases}", Term.BLUE + Term.BOLD)
        + f" | {case_tag}"
    )
    print(
        f"  Rload={rload} Ω | Rg_on={rg_on} Ω | Rg_off={rg_off} Ω | "
        f"fsw={f_sw if f_sw is not None else 'base'} | "
        f"deadtime={deadtime if deadtime is not None else 'base'}"
    )
    print(
        f"  Global elapsed: {fmt_eta(elapsed_global_s)} | "
        f"Avg/case: {fmt_eta(avg_case_s)} | "
        f"ETA: {fmt_eta(eta)}"
    )

def print_case_result(row: dict) -> None:
    status = fmt_status(row.get("Stable", 0), row.get("Bad_Physics", True), row.get("Reason", ""))

    print(f"\n  Status: {status} | Score: {fmt_num(row.get('Score'), digits=3, width=8)} | Reason: {row.get('Reason', '')}")

    print(c("  Electrical Metrics", Term.BOLD))
    print(
        f"    Pin      {fmt_num(row.get('Pin_W'), ' W')}   "
        f"Pout     {fmt_num(row.get('Pout_W'), ' W')}   "
        f"Loss     {fmt_num(row.get('Loss_W'), ' W')}   "
        f"Eff {fmt_pct(row.get('Efficiency_pct'))}"
    )
    print(
        f"    PF       {fmt_num(row.get('PF'), '', 4)}   "
        f"Vin_rms  {fmt_num(row.get('Vin_rms_V'), ' V')}   "
        f"Iin_rms  {fmt_num(row.get('Iin_rms_A'), ' A')}   "
        f"Iin_pk {fmt_num(row.get('Iin_pk_A'), ' A')}"
    )
    print(
        f"    Vout_avg {fmt_num(row.get('Vout_avg_V'), ' V')}   "
        f"Vout_pp  {fmt_num(row.get('Vout_pp_V'), ' V')}   "
        f"Elapsed  {fmt_num(row.get('Elapsed_s'), ' s')}"
    )

    pin_pos = row.get("Pin_pos_W", np.nan)
    pin_neg = row.get("Pin_neg_W", np.nan)
    if not pd.isna(pin_pos) or not pd.isna(pin_neg):
        print(
            f"    Pin_pos  {fmt_num(pin_pos, ' W')}   "
            f"Pin_neg  {fmt_num(pin_neg, ' W')}"
        )

def interpret_case(row: dict) -> str:
    if row.get("Stable", 0) != 1:
        return f"Rejected: {row.get('Reason', 'Unknown issue')}"

    notes = []

    eff = row.get("Efficiency_pct", np.nan)
    pf = row.get("PF", np.nan)
    vout = row.get("Vout_avg_V", np.nan)
    ripple = row.get("Vout_pp_V", np.nan)
    ipeak = row.get("Iin_pk_A", np.nan)

    if not pd.isna(pf):
        if pf >= 0.98:
            notes.append("excellent PF")
        elif pf >= 0.90:
            notes.append("acceptable PF")
        else:
            notes.append("weak PF")

    if not pd.isna(vout):
        if 390 <= vout <= 410:
            notes.append("well-regulated bus")
        elif vout > 410:
            notes.append("bus high")
        elif vout < 390:
            notes.append("bus low")

    if not pd.isna(ripple):
        if ripple <= 50:
            notes.append("low ripple")
        elif ripple <= 100:
            notes.append("moderate ripple")
        else:
            notes.append("high ripple")

    if not pd.isna(eff):
        if eff >= 90:
            notes.append("high efficiency")
        elif eff >= 70:
            notes.append("moderate efficiency")
        else:
            notes.append("low efficiency")

    if not pd.isna(ipeak):
        if ipeak <= 85:
            notes.append("reasonable peak current")
        else:
            notes.append("high peak current")

    return "; ".join(notes) if notes else "Valid case"

def print_best_so_far(results: List[Dict[str, object]]) -> None:
    if not results:
        return

    df_tmp = pd.DataFrame(results)
    if "Stable" not in df_tmp.columns or "Score" not in df_tmp.columns:
        return

    stable_df = df_tmp[df_tmp["Stable"] == 1].copy()
    if stable_df.empty:
        print(c("  Best so far: no stable case yet", Term.YELLOW))
        return

    best = stable_df.sort_values("Score", ascending=False).iloc[0]
    print(
        c("  Best so far:", Term.MAGENTA + Term.BOLD)
        + f" Rload={best.get('Rload_val', '—')} Ω | "
        f"Rg_on={best.get('Rg_on', '—')} Ω | "
        f"Rg_off={best.get('Rg_off', '—')} Ω | "
        f"Score={best.get('Score', np.nan):.3f} | "
        f"Eff={best.get('Efficiency_pct', np.nan):.2f}% | "
        f"PF={best.get('PF', np.nan):.4f} | "
        f"Vout={best.get('Vout_avg_V', np.nan):.2f} V"
    )

def print_final_leaderboard(df: pd.DataFrame, n: int = 10) -> None:
    print_banner("FINAL DESIGN LEADERBOARD")

    if df.empty:
        print("No results.")
        return

    stable_df = df[df["Stable"] == 1].copy()
    if stable_df.empty:
        print(c("No stable designs found.", Term.RED + Term.BOLD))
        return

    cols = [
        "Rload_val", "Rg_on", "Rg_off", "Score", "Efficiency_pct", "PF",
        "Vout_avg_V", "Vout_pp_V", "Iin_pk_A", "Pout_W"
    ]
    cols = [col for col in cols if col in stable_df.columns]

    top = stable_df.sort_values("Score", ascending=False).head(n)

    header = (
        f"{'Rank':>4} | {'Rload':>7} | {'Rg_on':>6} | {'Rg_off':>7} | "
        f"{'Score':>8} | {'Eff %':>8} | {'PF':>7} | {'Vout':>9} | "
        f"{'Ripple':>9} | {'Ipk':>9} | {'Pout':>10}"
    )
    print(c(header, Term.BOLD))
    print("-" * len(header))

    for idx, (_, row) in enumerate(top.iterrows(), start=1):
        print(
            f"{idx:>4} | "
            f"{row.get('Rload_val', np.nan):>7.2f} | "
            f"{row.get('Rg_on', np.nan):>6.2f} | "
            f"{row.get('Rg_off', np.nan):>7.2f} | "
            f"{row.get('Score', np.nan):>8.3f} | "
            f"{row.get('Efficiency_pct', np.nan):>8.2f} | "
            f"{row.get('PF', np.nan):>7.4f} | "
            f"{row.get('Vout_avg_V', np.nan):>9.2f} | "
            f"{row.get('Vout_pp_V', np.nan):>9.2f} | "
            f"{row.get('Iin_pk_A', np.nan):>9.2f} | "
            f"{row.get('Pout_W', np.nan):>10.2f}"
        )

    best = top.iloc[0]
    print(c("\nBest interpretation:", Term.GREEN + Term.BOLD))
    print("  " + interpret_case(best.to_dict()))



def parse_ltspice_value(token: str) -> float:
    """Parse LTspice numeric strings such as 10k, 4.7meg, 20n."""
    token = token.strip().strip("{}").strip()
    token = token.split(";")[0]
    m = re.match(r"^([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)([a-zA-Z]*)$", token)
    if not m:
        return math.nan
    base = float(m.group(1))
    suffix = m.group(2).lower()
    mult = {
        "f": 1e-15,
        "p": 1e-12,
        "n": 1e-9,
        "u": 1e-6,
        "µ": 1e-6,
        "m": 1e-3,
        "": 1.0,
        "k": 1e3,
        "meg": 1e6,
        "g": 1e9,
        "t": 1e12,
    }.get(suffix)
    if mult is None:
        return math.nan
    return base * mult


def param_regex(alias_list: List[str]) -> re.Pattern:
    body = "|".join(re.escape(a) for a in alias_list)
    return re.compile(rf"\b({body})\s*=\s*([^\s]+)", re.I)


def patch_one_param(line: str, canonical_name: str, value: float | int | None) -> Tuple[str, bool]:
    if value is None:
        return line, False
    aliases = PARAM_ALIASES[canonical_name]
    rx = param_regex(aliases)
    if rx.search(line):
        # Preserve the actual alias found in the schematic line.
        def repl(match: re.Match) -> str:
            return f"{match.group(1)}={ltspice_number(value)}"
        return rx.sub(repl, line), True
    return line, False


def patch_param_line_preserve_schematic(
    original_bytes: bytes,
    rg_on: float,
    rg_off: float,
    f_sw: float | int | None,
    deadtime: float | int | None,
    rload: float | int | None,
    cbus: float | int | None = None,
    lboost: float | int | None = None,
) -> bytes:
    """
    LTspice .asc-safe patcher.

    LTspice stores multi-line schematic text directives on one ASC line using
    literal backslash-n sequences, for example:

        TEXT ... !.param \\n+Vin=200\\n+Rg_on=2\\n+Rg_off=1

    This function edits each physical ASC line and appends missing parameters
    with "\\n+param=value" so LTspice netlists them correctly.
    """
    text = decode_asc(original_bytes)
    newline = "\r\n" if "\r\n" in text else "\n"
    lines = text.splitlines()

    rg_on_rx = re.compile(r"\b(?:Rg_on|Rgon|RGON|RG_ON)\s*=\s*[^\\\s]+", re.I)
    rg_off_rx = re.compile(r"\b(?:Rg_off|Rgoff|RGOFF|RG_OFF)\s*=\s*[^\\\s]+", re.I)

    patched_lines: List[str] = []
    patched_param_line = False
    f_sw_found = f_sw is None
    deadtime_found = deadtime is None

    for line in lines:
        is_param_text = line.lstrip().startswith("TEXT") and "!.param" in line

        if is_param_text and (
            rg_on_rx.search(line)
            or rg_off_rx.search(line)
            or "Rg_on" in line
            or "Rg_off" in line
            or "Rgon" in line
            or "Rgoff" in line
        ):
            patched_param_line = True

            if rg_on_rx.search(line):
                line = rg_on_rx.sub(f"Rg_on={ltspice_number(rg_on)}", line)
            else:
                line += f"\\n+Rg_on={ltspice_number(rg_on)}"

            if rg_off_rx.search(line):
                line = rg_off_rx.sub(f"Rg_off={ltspice_number(rg_off)}", line)
            else:
                line += f"\\n+Rg_off={ltspice_number(rg_off)}"

            if f_sw is not None:
                new_line, ok = patch_one_param(line, "f_sw", f_sw)
                line = new_line
                f_sw_found = f_sw_found or ok

            if deadtime is not None:
                new_line, ok = patch_one_param(line, "deadtime", deadtime)
                line = new_line
                deadtime_found = deadtime_found or ok

            if rload is not None:
                new_line, ok = patch_one_param(line, "Rload_val", rload)
                line = new_line
                if not ok:
                    line += f"\\n+Rload_val={ltspice_number(rload)}"

            if cbus is not None:
                new_line, ok = patch_one_param(line, "Cbus_val", cbus)
                line = new_line
                if not ok:
                    line += f"\\n+Cbus_val={ltspice_number(cbus)}"

            if lboost is not None:
                new_line, ok = patch_one_param(line, "Lboost_val", lboost)
                line = new_line
                if not ok:
                    line += f"\\n+Lboost_val={ltspice_number(lboost)}"

        patched_lines.append(line)

    if not patched_param_line:
        raise RuntimeError(
            "Could not find a usable LTspice TEXT directive with .param Rg_on/Rg_off or Rgon/Rgoff. "
            "Open the schematic and confirm the gate-resistance parameters exist in the .param directive."
        )

    if f_sw is not None and not f_sw_found:
        raise RuntimeError(
            "f_sw/fsw sweep was requested, but no matching f_sw/fsw/Fsw/FSW/Freq parameter exists in the schematic. "
            "Use RUN_MODE='rg_sweep' until that parameter is added."
        )

    if deadtime is not None and not deadtime_found:
        raise RuntimeError(
            "deadtime sweep was requested, but no matching deadtime/DeadTime/DT/tdead parameter exists in the schematic. "
            "Use RUN_MODE='rg_sweep' until that parameter is added."
        )

    if INJECT_EXTRA_MEASURES:
        existing_lower = "\n".join(patched_lines).lower()
        y = 96
        for directive in EXTRA_MEASURE_DIRECTIVES:
            parts = directive.split()
            measure_name = parts[2].lower() if len(parts) >= 3 else ""
            has_measure = bool(re.search(rf"\.meas\s+tran\s+{re.escape(measure_name)}\b", existing_lower, re.I))
            if measure_name and not has_measure:
                patched_lines.append(f"TEXT 64 {y} Left 2 !{directive}")
                y += 32

    patched_text = newline.join(patched_lines) + newline

    param_debug_lines = [
        ln for ln in patched_text.splitlines()
        if ".param" in ln.lower() and re.search(r"Rg[_]?on|Rg[_]?off|Rgon|Rgoff|fsw|f_sw|dead|dt", ln, re.I)
    ]
    print("PARAM DEBUG:")
    for ln in param_debug_lines:
        print("  " + ln.replace("\\n", " | "))

    if INJECT_EXTRA_MEASURES:
        print("EXTRA MEASURE DEBUG:")
        for directive in EXTRA_MEASURE_DIRECTIVES:
            print("  " + directive)

    return encode_asc(patched_text)

def delete_case_outputs(case_stem: str) -> None:
    for ext in (".log", ".net", ".raw", ".op.raw"):
        p = PROJECT_DIR / f"{case_stem}{ext}"
        if p.exists():
            for _ in range(8):
                try:
                    p.unlink()
                    break
                except PermissionError:
                    time.sleep(0.5)


def run_ltspice_case(case_asc: Path, timeout_s: int = 900) -> subprocess.CompletedProcess:
    return subprocess.run(
        [LTSPICE_EXE, "-b", str(case_asc)],
        cwd=str(PROJECT_DIR),
        timeout=timeout_s,
        capture_output=True,
        text=True,
    )


def wait_for_complete_log(log_path: Path, start_time: float, timeout_s: int = 45) -> str:
    deadline = time.time() + timeout_s
    last_text = ""

    while time.time() < deadline:
        if log_path.exists() and log_path.stat().st_mtime >= start_time:
            text = log_path.read_text(encoding="utf-8", errors="ignore")
            last_text = text
            has_elapsed = "Total elapsed time:" in text
            has_required = all(re.search(rf"^{m}\s*:", text, re.I | re.M) for m in REQUIRED_MEASURES)
            if has_elapsed and has_required:
                return text
        time.sleep(0.5)

    if last_text:
        return last_text
    raise FileNotFoundError(f"LTspice did not produce a new complete log: {log_path}")


def parse_measure(log_text: str, name: str) -> float:
    patterns = [
        rf"^{re.escape(name)}\s*:.*?=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        rf"^{re.escape(name)}\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
    ]

    for pat in patterns:
        m = re.search(pat, log_text, re.I | re.M)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return np.nan

    return np.nan


def parse_elapsed(log_text: str) -> float:
    m = re.search(r"Total elapsed time:\s*([-+]?\d*\.?\d+)", log_text, re.I)
    return float(m.group(1)) if m else np.nan


def detect_failure(log_text: str, returncode: int) -> Tuple[bool, str]:
    failure_markers = [
        "Simulation Failed",
        "Iteration limit reached",
        "Fatal Error",
        "No such parameter defined",
        "Unknown subcircuit",
        "Can't find definition of model",
        "Questionable use of curly braces",
    ]
    for marker in failure_markers:
        if marker.lower() in log_text.lower():
            return True, marker
    if returncode not in (0, None):
        return True, f"LTspice return code {returncode}"
    return False, ""


def evaluate_physics(
    failed: bool,
    fail_reason: str,
    pin: float,
    iin_rms: float,
    iin_pk: float,
    vout_avg: float,
    vout_pp: float,
    elapsed_s: float,
    pf: float,
    efficiency_pct: float,
) -> Tuple[int, bool, str]:
    if failed:
        return 0, True, f"LTspice failed: {fail_reason}"

    if np.isnan(pin) or np.isnan(iin_rms) or np.isnan(iin_pk) or np.isnan(vout_avg):
        return 0, True, "Missing required measurement"

    if not np.isnan(elapsed_s) and elapsed_s < MIN_GOOD_ELAPSED_S:
        return 0, True, "Too fast: likely not switching"

    if pin < MIN_PIN_W:
        return 0, True, "Input power too low"

    if iin_rms < MIN_IIN_RMS_A:
        return 0, True, "Input RMS current too low"

    if iin_pk < MIN_IIN_PK_A:
        return 0, True, "Input peak current too low"

    if iin_rms > MAX_IIN_RMS_A:
        return 0, True, "Input RMS current too high"

    if not (VOUT_MIN_V <= vout_avg <= VOUT_MAX_V):
        return 0, True, "Output voltage out of valid range"

    if not np.isnan(vout_pp) and vout_pp > 100:
        return 0, True, "Output ripple too high / unstable real-load bus"

    if not np.isnan(pf) and pf > 1.15:
        return 0, True, "PF calculation physically impossible; check measurement signs/nodes"

    if TEMPLATE_MODE == "real_load":
        if np.isnan(efficiency_pct):
            return 0, True, "Missing real-load efficiency measurement"

        if efficiency_pct > 105:
            return 0, True, "Efficiency physically impossible in real-load mode"

    elif TEMPLATE_MODE == "ideal_bus":
        if not np.isnan(efficiency_pct) and efficiency_pct > 105:
            return 1, False, "OK - efficiency ignored because output is clamped by ideal V2 source"

    return 1, False, "OK"


def write_netlist_diff(reference_net: Path, case_net: Path, diff_path: Path) -> None:
    if not reference_net.exists() or not case_net.exists():
        return
    good_lines = reference_net.read_text(encoding="utf-8", errors="ignore").splitlines()
    case_lines = case_net.read_text(encoding="utf-8", errors="ignore").splitlines()
    diff = difflib.unified_diff(
        good_lines,
        case_lines,
        fromfile=str(reference_net.name),
        tofile=str(case_net.name),
        lineterm="",
    )
    diff_text = "\n".join(diff)
    if diff_text.strip():
        diff_path.write_text(diff_text, encoding="utf-8")


def copy_if_exists(src: Path, dst: Path) -> str:
    if src.exists():
        shutil.copy2(src, dst)
        return str(dst)
    return ""


def find_output_load_resistor_ohms(net_path: Path) -> Tuple[float, str]:
    """
    Look for simple resistor load candidates in the generated netlist.
    Examples:
        Rload OUT 0 20
        R3 out 0 20
        Rload N001 OUT 20
    Returns (ohms, description). This is intentionally conservative.
    """
    if not net_path.exists():
        return np.nan, "No netlist"

    text = net_path.read_text(encoding="utf-8", errors="ignore")
    candidates: List[Tuple[str, float, str]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("*") or not re.match(r"^[Rr]\S+\s+", line):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        name, n1, n2, val = parts[0], parts[1], parts[2], parts[3]
        nodes = {n1.upper(), n2.upper()}
        if "OUT" in nodes and ("0" in nodes or "GND" in nodes):
            ohms = parse_ltspice_value(val)
            if not np.isnan(ohms) and ohms > 0:
                candidates.append((name, ohms, line))

    if not candidates:
        return np.nan, "No OUT-to-ground resistor detected"

    # Prefer explicit load names, else first valid candidate.
    candidates.sort(key=lambda x: (0 if "load" in x[0].lower() else 1, x[0]))
    name, ohms, line = candidates[0]
    return ohms, line


def calculate_pf(pin: float, vin_rms: float, iin_rms: float) -> float:
    if np.isnan(pin) or np.isnan(vin_rms) or np.isnan(iin_rms):
        return np.nan
    if vin_rms <= 0 or iin_rms <= 0:
        return np.nan
    return abs(pin) / (vin_rms * iin_rms)


def calculate_score(row: pd.Series) -> float:
    if row.get("Stable", 0) != 1:
        return -1e9

    eff = row.get("Efficiency_pct", np.nan)
    pf = row.get("PF", np.nan)
    iin_pk = row.get("Iin_pk_A", np.nan)
    vout_pp = row.get("Vout_pp_V", np.nan)
    vout_avg = row.get("Vout_avg_V", np.nan)

    eff_norm = 0 if np.isnan(eff) else np.clip(eff / 100.0, 0, 1.05)
    pf_norm = 0 if np.isnan(pf) else np.clip(pf, 0, 1.0)
    peak_penalty = 0 if np.isnan(iin_pk) else np.clip((iin_pk - 75.0) / 50.0, 0, 1)
    ripple_penalty = 0 if np.isnan(vout_pp) else np.clip(vout_pp / 20.0, 0, 1)
    regulation_penalty = 0 if np.isnan(vout_avg) else np.clip(abs(vout_avg - VOUT_TARGET_V) / 50.0, 0, 1)

    return round(100 * (0.45 * eff_norm + 0.25 * pf_norm + 0.15 * (1 - peak_penalty) + 0.10 * (1 - ripple_penalty) + 0.05 * (1 - regulation_penalty)), 3)


def save_plot_scatter(df: pd.DataFrame, y_col: str, filename: str, title: str) -> None:
    if y_col not in df.columns or df[y_col].dropna().empty:
        return
    plt.figure(figsize=(8, 5))
    scatter = plt.scatter(df["Rg_on"], df["Rg_off"], c=df[y_col], s=90, edgecolors="black")
    plt.colorbar(scatter, label=y_col)
    plt.xlabel("Rg_on (ohm)")
    plt.ylabel("Rg_off (ohm)")
    plt.title(title)
    plt.grid(True)
    plt.savefig(RUN_PLOTS_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def save_design_plots(df: pd.DataFrame) -> None:
    if df.empty:
        return

    if "Rload_val" in df.columns and df["Rload_val"].nunique() > 1:
        save_rload_sweep_plots(df)
    elif df["Rg_on"].nunique() > 1 or df["Rg_off"].nunique() > 1:
        save_gate_drive_maps(df)
    else:
        save_single_case_summary_plot(df)

    save_top_designs_bar(df)


def save_rload_sweep_plots(df: pd.DataFrame) -> None:
    dfp = df.sort_values("Rload_val").copy()

    stable = dfp[dfp["Stable"] == 1]
    unstable = dfp[dfp["Stable"] != 1]

    def plot_metric(y_col: str, filename: str, title: str, ylabel: str, target_y=None):
        if y_col not in dfp.columns or dfp[y_col].dropna().empty:
            return

        plt.figure(figsize=(9, 5))

        if not stable.empty:
            plt.plot(
                stable["Rload_val"],
                stable[y_col],
                marker="o",
                linewidth=2,
                label="Stable"
            )

        if not unstable.empty:
            plt.scatter(
                unstable["Rload_val"],
                unstable[y_col],
                marker="x",
                s=90,
                label="Rejected"
            )

        if target_y is not None:
            plt.axhline(target_y, linestyle="--", linewidth=1.5, label=f"Target {target_y}")

        plt.xlabel("Load Resistance Rload (Ω)")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(RUN_PLOTS_DIR / filename, dpi=300, bbox_inches="tight")
        plt.close()

    plot_metric("Efficiency_pct", "efficiency_vs_rload.png", "Efficiency vs Load Resistance", "Efficiency (%)")
    plot_metric("PF", "pf_vs_rload.png", "Power Factor vs Load Resistance", "Power Factor", target_y=0.95)
    plot_metric("Vout_avg_V", "vout_vs_rload.png", "Output Voltage vs Load Resistance", "Vout Avg (V)", target_y=400)
    plot_metric("Vout_pp_V", "ripple_vs_rload.png", "Output Ripple vs Load Resistance", "Vout Ripple (Vpp)")
    plot_metric("Pout_W", "pout_vs_rload.png", "Output Power vs Load Resistance", "Pout (W)")
    plot_metric("Iin_pk_A", "current_vs_rload.png", "Peak Input Current vs Load Resistance", "Iin Peak (A)")

    plot_score_vs_rload(dfp)
    save_tradeoff_plot(dfp)
    save_load_sweep_dashboard(dfp)


def plot_score_vs_rload(df: pd.DataFrame) -> None:
    dfp = df.sort_values("Rload_val").copy()
    stable = dfp[dfp["Stable"] == 1]
    rejected = dfp[dfp["Stable"] != 1]

    plt.figure(figsize=(9, 5))

    if not stable.empty:
        plt.plot(
            stable["Rload_val"],
            stable["Score"],
            marker="o",
            linewidth=2,
            label="Stable score"
        )

    if not rejected.empty:
        ymin = stable["Score"].min() - 5 if not stable.empty else 0
        plt.scatter(
            rejected["Rload_val"],
            [ymin] * len(rejected),
            marker="x",
            s=90,
            label="Rejected case"
        )

    plt.xlabel("Load Resistance Rload (Ω)")
    plt.ylabel("Optimization Score")
    plt.title("Optimization Score vs Load Resistance")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RUN_PLOTS_DIR / "score_vs_rload.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_top_designs_bar(df: pd.DataFrame) -> None:
    if df.empty or "Stable" not in df.columns or "Score" not in df.columns:
        return

    top = df[df["Stable"] == 1].sort_values("Score", ascending=False).head(10)
    if top.empty:
        return

    labels = []
    for r in top.itertuples():
        rload = getattr(r, "Rload_val", np.nan)
        rg_on = getattr(r, "Rg_on", np.nan)
        rg_off = getattr(r, "Rg_off", np.nan)

        if not pd.isna(rload):
            labels.append(f"Rload={rload:g}Ω\nRg_on={rg_on:g}, Rg_off={rg_off:g}")
        else:
            labels.append(f"Rg_on={rg_on:g}\nRg_off={rg_off:g}")

    plt.figure(figsize=(11, 5))
    plt.bar(range(len(top)), top["Score"])
    plt.xticks(range(len(top)), labels, rotation=45, ha="right")
    plt.ylabel("Optimization Score")
    plt.title("Top Ranked GaN PFC Real-Load Designs")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(RUN_PLOTS_DIR / "top_10_designs_score.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_load_sweep_dashboard(df: pd.DataFrame) -> None:
    cols_needed = ["Rload_val", "Score", "Efficiency_pct", "PF", "Vout_avg_V", "Vout_pp_V", "Pout_W", "Iin_pk_A"]
    if any(col not in df.columns for col in cols_needed):
        return

    dfp = df.sort_values("Rload_val").copy()
    stable = dfp[dfp["Stable"] == 1]
    unstable = dfp[dfp["Stable"] != 1]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.ravel()

    metrics = [
        ("Score", "Score"),
        ("Efficiency_pct", "Efficiency (%)"),
        ("PF", "Power Factor"),
        ("Vout_avg_V", "Vout Avg (V)"),
        ("Vout_pp_V", "Ripple (Vpp)"),
        ("Pout_W", "Output Power (W)"),
    ]

    for ax, (col, ylabel) in zip(axes, metrics):
        if not stable.empty:
            ax.plot(stable["Rload_val"], stable[col], marker="o", linewidth=2, label="Stable")

        if not unstable.empty:
            ax.scatter(unstable["Rload_val"], unstable[col], marker="x", s=80, label="Rejected")

        if col == "Vout_avg_V":
            ax.axhline(400, linestyle="--", linewidth=1)
        if col == "PF":
            ax.axhline(0.95, linestyle="--", linewidth=1)

        ax.set_xlabel("Rload (Ω)")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(True)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2)

    fig.suptitle("GaN PFC Real-Load Sweep Dashboard", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(RUN_PLOTS_DIR / "load_sweep_dashboard.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_gate_drive_maps(df: pd.DataFrame) -> None:
    colors = df["Stable"].map({1: "green", 0: "red"}).fillna("gray")

    plt.figure(figsize=(8, 5))
    plt.scatter(df["Rg_on"], df["Rg_off"], c=colors, s=90, edgecolors="black")
    plt.xlabel("Rg_on (Ω)")
    plt.ylabel("Rg_off (Ω)")
    plt.title("GaN PFC Stability Map")
    plt.grid(True)
    plt.savefig(RUN_PLOTS_DIR / "stability_map.png", dpi=300, bbox_inches="tight")
    plt.close()

    save_plot_scatter(df, "Score", "optimization_score_map.png", "Optimization Score Map")
    save_plot_scatter(df, "PF", "pf_map.png", "Power Factor Map")
    save_plot_scatter(df, "Efficiency_pct", "efficiency_map.png", "Efficiency Map")
    save_plot_scatter(df, "Iin_pk_A", "peak_current_map.png", "Peak Current Map")
    save_plot_scatter(df, "Vout_pp_V", "vout_ripple_map.png", "Output Ripple Map")


def save_single_case_summary_plot(df: pd.DataFrame) -> None:
    row = df.iloc[0]

    metrics = {
        "Score": row.get("Score", np.nan),
        "Efficiency %": row.get("Efficiency_pct", np.nan),
        "PF × 100": row.get("PF", np.nan) * 100 if not pd.isna(row.get("PF", np.nan)) else np.nan,
        "Vout/4": row.get("Vout_avg_V", np.nan) / 4 if not pd.isna(row.get("Vout_avg_V", np.nan)) else np.nan,
        "Ripple": row.get("Vout_pp_V", np.nan),
    }

    metrics = {k: v for k, v in metrics.items() if not pd.isna(v)}

    if not metrics:
        return

    plt.figure(figsize=(9, 5))
    plt.bar(metrics.keys(), metrics.values())
    plt.ylabel("Scaled Metric Value")
    plt.title("Single Case Performance Summary")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(RUN_PLOTS_DIR / "single_case_summary.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_tradeoff_plot(df: pd.DataFrame) -> None:
    if df.empty or "Efficiency_pct" not in df.columns or "Vout_avg_V" not in df.columns:
        return

    dfp = df.copy()
    dfp["Regulation_Error_V"] = (dfp["Vout_avg_V"] - VOUT_TARGET_V).abs()

    plt.figure(figsize=(9, 6))

    stable = dfp[dfp["Stable"] == 1]
    unstable = dfp[dfp["Stable"] != 1]

    if not stable.empty:
        plt.scatter(
            stable["Regulation_Error_V"],
            stable["Efficiency_pct"],
            s=90,
            label="Stable"
        )

        for _, row in stable.iterrows():
            plt.annotate(
                f"{row['Rload_val']:g}Ω",
                (row["Regulation_Error_V"], row["Efficiency_pct"]),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=8
            )

    if not unstable.empty:
        plt.scatter(
            unstable["Regulation_Error_V"],
            unstable["Efficiency_pct"],
            marker="x",
            s=90,
            label="Rejected"
        )

    plt.xlabel("|Vout - 400 V| (V)")
    plt.ylabel("Efficiency (%)")
    plt.title("Efficiency vs Voltage Regulation Tradeoff")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RUN_PLOTS_DIR / "efficiency_regulation_tradeoff.png", dpi=300, bbox_inches="tight")
    plt.close()


def estimate_losses(row: pd.Series, rds_on_mohm: float = 35, qg_nc: float = 40, vg: float = 18, fsw: float = 50_000) -> Dict[str, float]:
    i_rms = row.get("Iin_rms_A", np.nan)
    pin = row.get("Pin_W", np.nan)
    pout = row.get("Pout_W", np.nan)

    rds = rds_on_mohm / 1000
    p_cond = np.nan if np.isnan(i_rms) else i_rms ** 2 * rds
    p_gate = qg_nc * 1e-9 * vg * fsw * 4
    p_total_measured = np.nan if np.isnan(pin) or np.isnan(pout) else pin - pout

    return {
        "Pcond_est_W": p_cond,
        "Pgate_est_W": p_gate,
        "Punexplained_loss_W": np.nan if np.isnan(p_total_measured) or np.isnan(p_cond) else p_total_measured - p_cond - p_gate,
    }


def estimate_thermal(ploss_w: float, rth_ja: float = THERMAL_RTH_EQUIV_C_PER_W, tamb: float = AMBIENT_C) -> Tuple[float, str]:
    if np.isnan(ploss_w):
        return np.nan, "Unknown"
    tj = tamb + ploss_w * rth_ja
    if tj < 125:
        risk = "Green"
    elif tj < 150:
        risk = "Yellow"
    else:
        risk = "Red"
    return tj, risk


def validate_generated_netlist(net_path: Path, expected_rload: float) -> List[str]:
    warnings: List[str] = []
    if not net_path.exists():
        return ["Netlist missing"]

    text = net_path.read_text(encoding="utf-8", errors="ignore")
    if "AC_IN" not in text:
        warnings.append("AC_IN node missing")
    if "AC_RET" not in text:
        warnings.append("AC_RET node missing")
    if "Cbus" not in text and "C_bus" not in text and "C_BUS" not in text:
        warnings.append("Cbus missing")
    if "Rload" not in text and "RLOAD" not in text:
        warnings.append("Rload missing")
    if f"{expected_rload:g}" not in text:
        warnings.append(f"Expected Rload {expected_rload:g} may not be patched")
    return warnings


def recommend_next_cases(df: pd.DataFrame) -> List[float]:
    if df.empty or "Stable" not in df.columns:
        return []

    stable = df[df["Stable"] == 1].copy()
    if stable.empty:
        return []

    best = stable.sort_values("Score", ascending=False).iloc[0]
    rload = float(best.get("Rload_val", np.nan))
    if np.isnan(rload):
        return []

    candidates = [rload - 1, rload - 0.5, rload + 0.5, rload + 1]
    return [x for x in candidates if x > 0]


def score_with_profile(row: pd.Series, profile_name: str = DESIGN_MODE) -> float:
    if profile_name not in SCORING_PROFILES:
        profile_name = "balanced"

    weights = SCORING_PROFILES[profile_name]

    score = row.get("Efficiency_pct", np.nan)
    pf = row.get("PF", np.nan)
    vout_avg = row.get("Vout_avg_V", np.nan)
    vout_pp = row.get("Vout_pp_V", np.nan)
    iin_pk = row.get("Iin_pk_A", np.nan)

    eff_norm = 0 if np.isnan(score) else np.clip(score / 100.0, 0, 1.05)
    pf_norm = 0 if np.isnan(pf) else np.clip(pf, 0, 1.0)
    regulation_norm = 0 if np.isnan(vout_avg) else np.clip(1 - abs(vout_avg - VOUT_TARGET_V) / 50.0, 0, 1)
    ripple_norm = 0 if np.isnan(vout_pp) else np.clip(1 - vout_pp / 100.0, 0, 1)
    ipeak_norm = 0 if np.isnan(iin_pk) else np.clip(1 - iin_pk / 100.0, 0, 1)

    return round(
        100 * (
            weights["eff"] * eff_norm
            + weights["pf"] * pf_norm
            + weights["regulation"] * regulation_norm
            + weights["ripple"] * ripple_norm
            + weights["ipeak"] * ipeak_norm
        ),
        3,
    )


def generate_design_mode_leaderboards(df: pd.DataFrame) -> None:
    if df.empty:
        return

    stable = df[df["Stable"] == 1].copy()
    if stable.empty:
        return

    rows: List[Dict[str, object]] = []
    for mode in SCORING_PROFILES:
        scored = stable.copy()
        scored["ModeScore"] = scored.apply(lambda row, mode=mode: score_with_profile(row, mode), axis=1)
        best = scored.sort_values("ModeScore", ascending=False).head(1)
        if not best.empty:
            r = best.iloc[0].to_dict()
            r["Design_Mode"] = mode
            rows.append(r)

    if rows:
        out = pd.DataFrame(rows)
        out.to_csv(RUN_REPORTS_DIR / "design_mode_leaderboards.csv", index=False)
        out.to_excel(RUN_REPORTS_DIR / "design_mode_leaderboards.xlsx", index=False)


# ==============================
# LEADERSHIP FEATURE PACK
# ==============================

def generate_executive_brief(df: pd.DataFrame) -> None:
    if df.empty:
        return
    stable = df[df["Stable"] == 1]
    best = stable.sort_values("Score", ascending=False).head(1)
    top_count = int(stable.shape[0])
    overall_best = best.iloc[0] if not best.empty else None

    lines = []
    lines.append("GAN PFC Real-Load Optimization — Executive Brief")
    lines.append("")
    lines.append(f"Total cases run: {len(df)}")
    lines.append(f"Stable designs: {top_count}")
    if overall_best is not None:
        lines.append("")
        lines.append("Top design summary:")
        lines.append(f"  Rload: {overall_best.get('Rload_val', '—')} Ω")
        lines.append(f"  Score: {overall_best.get('Score', np.nan):.3f}")
        lines.append(f"  Efficiency: {overall_best.get('Efficiency_pct', np.nan):.2f}%")
        lines.append(f"  PF: {overall_best.get('PF', np.nan):.4f}")
        lines.append(f"  Vout: {overall_best.get('Vout_avg_V', np.nan):.2f} V")

    path_txt = RUN_REPORTS_DIR / "executive_brief.txt"
    path_html = RUN_REPORTS_DIR / "executive_brief.html"
    path_txt.write_text("\n".join(lines), encoding="utf-8")

    # simple HTML
    html = ["<html><body>", f"<h2>{lines[0]}</h2>", "<pre>"]
    html.extend(lines[1:])
    html += ["</pre>", "</body></html>"]
    path_html.write_text("\n".join(html), encoding="utf-8")


def multi_objective_selector(df: pd.DataFrame, weights: Optional[Dict[str, float]] = None) -> None:
    if df.empty:
        return

    if len(df) < 2:
        df_single = df.copy()
        df_single["MO_Score"] = df_single.get("Score", np.nan)
        df_single.to_csv(RUN_REPORTS_DIR / "multi_objective_selector_top20.csv", index=False)
        return

    if weights is None:
        weights = {
            "Score": 0.4,
            "Efficiency_pct": 0.3,
            "PF": 0.2,
            "Vout_avg_V": 0.1,
        }

    dfm = df.copy()

    for k in weights:
        if k not in dfm.columns:
            dfm[k] = np.nan

    norm_cols = {}

    for k in weights:
        col = pd.to_numeric(dfm[k], errors="coerce")
        mn = col.min()
        mx = col.max()

        if pd.isna(mn) or pd.isna(mx):
            norm = pd.Series(0.0, index=dfm.index)

        elif mn == mx:
            # Single-case or no variation case.
            # Treat valid non-null values as fully normalized.
            norm = pd.Series(
                np.where(col.notna(), 1.0, 0.0),
                index=dfm.index
            )

        else:
            if k == "Vout_avg_V":
                # For voltage, closeness to 400 V is better, not larger voltage.
                error = (col - VOUT_TARGET_V).abs()
                err_min = error.min()
                err_max = error.max()

                if err_min == err_max:
                    norm = pd.Series(1.0, index=dfm.index)
                else:
                    norm = 1.0 - ((error - err_min) / (err_max - err_min))
            else:
                norm = (col - mn) / (mx - mn)

            norm = norm.fillna(0.0)

        norm_cols[k] = norm

    mo_score = pd.Series(0.0, index=dfm.index)
    for k, w in weights.items():
        mo_score += norm_cols[k] * w

    dfm["MO_Score"] = mo_score

    out = dfm.sort_values("MO_Score", ascending=False).head(20)
    out.to_csv(RUN_REPORTS_DIR / "multi_objective_selector_top20.csv", index=False)


def best_by_category(df: pd.DataFrame) -> None:
    if df.empty:
        return
    rows = []
    stable = df[df["Stable"] == 1].copy()
    if stable.empty:
        return

    def add_category(name: str, selected: pd.DataFrame):
        if selected.empty:
            return
        r = selected.iloc[0].to_dict()
        r["Category"] = name
        rows.append(r)

    add_category("Best Overall Score", stable.sort_values("Score", ascending=False).head(1))
    add_category("Highest Efficiency", stable.sort_values("Efficiency_pct", ascending=False).head(1))
    add_category("Highest PF", stable.sort_values("PF", ascending=False).head(1))
    add_category("Best Voltage Regulation", stable.iloc[(stable["Vout_avg_V"] - VOUT_TARGET_V).abs().argsort().values[:1]])
    add_category("Lowest Ripple", stable.sort_values("Vout_pp_V", ascending=True).head(1))
    add_category("Lowest Peak Current", stable.sort_values("Iin_pk_A", ascending=True).head(1))
    add_category("Highest Output Power", stable.sort_values("Pout_W", ascending=False).head(1))

    if rows:
        out = pd.DataFrame(rows)
        out.to_csv(RUN_REPORTS_DIR / "best_by_category.csv", index=False)
        out.to_excel(RUN_REPORTS_DIR / "best_by_category.xlsx", index=False)


def risk_register(df: pd.DataFrame) -> None:
    if df.empty:
        return
    reasons = df["Reason"].fillna("Unknown").value_counts().reset_index()
    reasons.columns = ["Reason", "Count"]
    reasons["RiskScore"] = reasons["Count"] / len(df)
    reasons.to_csv(RUN_REPORTS_DIR / "risk_register.csv", index=False)


def ai_next_simulation_recommendations(df: pd.DataFrame) -> None:
    # Lightweight heuristic-based recommendations (stand-in for AI)
    recs: List[str] = []
    if df.empty:
        recs.append("No data to analyze. Run at least one case.")
    else:
        stable = df[df["Stable"] == 1]
        if stable.empty:
            recs.append("No stable cases: try widening Rg_on/Rg_off or increase Rload range.")
            recs.append("Consider reducing gate resistances or increasing simulation time.")
        else:
            best = stable.sort_values("Score", ascending=False).head(1).iloc[0]
            rload = best.get("Rload_val", None)
            recs.append(f"Explore Rload around {rload} (±10%) with finer steps.")
            recs.append("If PF low, consider adding PF measurement probes or checking input wiring.")

    (RUN_REPORTS_DIR / "ai_next_sim_recommendations.txt").write_text("\n".join(recs), encoding="utf-8")


def application_mode_scores(df: pd.DataFrame) -> None:
    if df.empty:
        return

    modes = {
        "Industrial": {
            "Efficiency_pct": 0.5,
            "PF": 0.3,
            "Vout_avg_V": 0.2,
        },
        "Datacenter": {
            "PF": 0.4,
            "Efficiency_pct": 0.4,
            "Iin_pk_A": 0.2,
        },
        "EV_Onboard_Charger": {
            "Efficiency_pct": 0.35,
            "PF": 0.25,
            "Vout_avg_V": 0.25,
            "Vout_pp_V": 0.15,
        },
    }

    results = []

    for mode_name, weights in modes.items():
        dfm = df.copy()
        score = pd.Series(0.0, index=dfm.index)

        for col_name, weight in weights.items():
            if col_name not in dfm.columns:
                continue

            col = pd.to_numeric(dfm[col_name], errors="coerce")

            if col_name == "Vout_avg_V":
                metric = 1.0 / (1.0 + (col - VOUT_TARGET_V).abs())
            elif col_name in ["Iin_pk_A", "Vout_pp_V"]:
                metric = 1.0 / (1.0 + col.abs())
            else:
                max_val = col.max()
                if pd.isna(max_val) or max_val == 0:
                    metric = pd.Series(0.0, index=dfm.index)
                else:
                    metric = col / max_val

            metric = metric.fillna(0.0)
            score += weight * metric

        dfm["Application_Mode"] = mode_name
        dfm["Application_Score"] = score

        best = dfm.sort_values("Application_Score", ascending=False).head(1)
        if not best.empty:
            results.append(best.iloc[0].to_dict())

    if results:
        pd.DataFrame(results).to_csv(RUN_REPORTS_DIR / "application_mode_scores.csv", index=False)
        pd.DataFrame(results).to_excel(RUN_REPORTS_DIR / "application_mode_scores.xlsx", index=False)


def generate_templates() -> None:
    # GaN device FOM template
    fom = pd.DataFrame(
        columns=["Device", "Rds_on_mOhm", "Qg_nC", "Vds_V", "Package", "Comments"]
    )
    fom.to_csv(TEMPLATES_DIR / "gan_device_fom_template.csv", index=False)

    # Supplier scouting matrix
    suppliers = pd.DataFrame(columns=["Supplier", "PartNumber", "LeadTime_days", "Price_USD", "Notes"])
    suppliers.to_csv(TEMPLATES_DIR / "supplier_scouting_matrix.csv", index=False)

    # Wide-bandgap material comparison
    materials = pd.DataFrame(columns=["Material", "Bandgap_eV", "ThermalCond_WmK", "DielectricStrength_MVcm", "Notes"])
    materials.to_csv(TEMPLATES_DIR / "wide_bandgap_materials.csv", index=False)


def generate_html_dashboard(df: pd.DataFrame) -> None:
    top = df[df["Stable"] == 1].sort_values("Score", ascending=False).head(10)

    if not top.empty:
        best = top.iloc[0]
        executive_conclusion = f"""
<div class='summary'>
<h2>Executive Conclusion</h2>
<p>
The strongest balanced real-load operating point is <b>Rload = {best['Rload_val']:.0f} Ω</b>,
with <b>PF = {best['PF']:.4f}</b>, <b>Vout = {best['Vout_avg_V']:.2f} V</b>,
<b>ripple = {best['Vout_pp_V']:.2f} Vpp</b>, <b>Pout = {best['Pout_W']:.2f} W</b>,
and <b>efficiency = {best['Efficiency_pct']:.2f}%</b>.
</p>
<p>
This result demonstrates an automated LTspice-to-dataset workflow for physics-aware GaN PFC design ranking.
</p>
</div>
"""
    else:
        executive_conclusion = """
<div class='summary'>
<h2>Executive Conclusion</h2>
<p>No stable design was available in the current dataset.</p>
</div>
"""

    html_parts = ["""
<html>
<head>
<meta charset='utf-8'>
<title>GaN PFC Real-Load Optimization Dashboard</title>
<style>
body {
    font-family: Arial, Helvetica, sans-serif;
    margin: 36px;
    color: #111827;
    background: #f8fafc;
}
h1 {
    font-size: 30px;
    margin-bottom: 4px;
}
h2 {
    font-size: 20px;
    margin-top: 0;
}
.summary {
    background: #111827;
    color: white;
    padding: 22px;
    border-radius: 14px;
    margin-bottom: 24px;
}
.card {
    background: white;
    border-radius: 14px;
    padding: 18px;
    box-shadow: 0 6px 18px rgba(15, 23, 42, 0.08);
    margin-bottom: 22px;
}
img {
    width: 100%;
    height: auto;
    border-radius: 8px;
}
table {
    border-collapse: collapse;
    width: 100%;
    background: white;
    font-size: 12px;
}
th, td {
    border: 1px solid #d1d5db;
    padding: 6px 8px;
    text-align: right;
}
th {
    background: #e5e7eb;
}
</style>
</head>
<body>
"""]
    html_parts.append("<h1>GaN PFC Real-Load Optimization Dashboard</h1>")
    html_parts.append(executive_conclusion)

    if (RUN_REPORTS_DIR / "executive_brief.html").exists():
        html_parts.append("<div class='card'><h2>Executive Brief</h2>")
        html_parts.append((RUN_REPORTS_DIR / "executive_brief.html").read_text(encoding="utf-8"))
        html_parts.append("</div>")

    imgs = [
        ("load_sweep_dashboard.png", "Real-Load Sweep Dashboard"),
        ("score_vs_rload.png", "Optimization Score vs Load Resistance"),
        ("efficiency_vs_rload.png", "Efficiency vs Load Resistance"),
        ("vout_vs_rload.png", "Output Voltage Regulation"),
        ("pf_vs_rload.png", "Power Factor Performance"),
        ("ripple_vs_rload.png", "Output Ripple"),
        ("pout_vs_rload.png", "Output Power"),
        ("current_vs_rload.png", "Peak Input Current"),
        ("efficiency_regulation_tradeoff.png", "Efficiency vs Regulation Tradeoff"),
        ("top_10_designs_score.png", "Top Ranked Designs"),
    ]
    for im, title in imgs:
        p = RUN_PLOTS_DIR / im
        if p.exists():
            rel_src = os.path.relpath(p, RUN_REPORTS_DIR).replace("\\", "/")
            html_parts.append(f"<section class='card'><h2>{title}</h2>")
            html_parts.append(f"<img src='{rel_src}'></section>")

    try:
        display_cols = [
            "Rload_val", "Rg_on", "Rg_off", "Score", "Efficiency_pct",
            "PF", "Vout_avg_V", "Vout_pp_V", "Pout_W", "Iin_pk_A", "Reason"
        ]
        top_display = top[[c for c in display_cols if c in top.columns]].copy()
        html_parts.append("<div class='card'><h2>Top 10 Designs</h2>")
        html_parts.append(top_display.to_html(index=False, float_format=lambda x: f"{x:.3f}"))
        html_parts.append("</div>")
    except Exception:
        pass

    html_parts.append("</body></html>")
    (RUN_REPORTS_DIR / "dashboard.html").write_text("\n".join(html_parts), encoding="utf-8")


def generate_run_summary(df: pd.DataFrame) -> None:
    if df.empty:
        return

    stable = df[df["Stable"] == 1].copy()
    rejected = df[df["Stable"] != 1].copy()

    lines = []
    lines.append("# GaN PFC Optimization Run Summary")
    lines.append("")
    lines.append(f"- Total cases: {len(df)}")
    lines.append(f"- Stable cases: {len(stable)}")
    lines.append(f"- Rejected cases: {len(rejected)}")
    lines.append("")

    if not stable.empty:
        best = stable.sort_values("Score", ascending=False).iloc[0]
        lines.append("## Best Overall Design")
        lines.append("")
        lines.append(f"- Rload: {best.get('Rload_val', np.nan):.2f} Ω")
        lines.append(f"- Rg_on: {best.get('Rg_on', np.nan):.2f} Ω")
        lines.append(f"- Rg_off: {best.get('Rg_off', np.nan):.2f} Ω")
        lines.append(f"- Score: {best.get('Score', np.nan):.3f}")
        lines.append(f"- Efficiency: {best.get('Efficiency_pct', np.nan):.2f}%")
        lines.append(f"- PF: {best.get('PF', np.nan):.4f}")
        lines.append(f"- Vout_avg: {best.get('Vout_avg_V', np.nan):.2f} V")
        lines.append(f"- Vout_pp: {best.get('Vout_pp_V', np.nan):.2f} V")
        lines.append(f"- Pout: {best.get('Pout_W', np.nan):.2f} W")
        lines.append("")

    if not rejected.empty:
        lines.append("## Rejected Design Reasons")
        lines.append("")
        counts = rejected["Reason"].value_counts()
        for reason, count in counts.items():
            lines.append(f"- {reason}: {count}")

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "The sweep identifies the strongest operating region by combining power factor, "
        "output voltage regulation, ripple, current stress, and efficiency into a single "
        "physics-aware score. Rejected cases are not discarded silently; they are preserved "
        "as boundary information for future surrogate-model training."
    )

    (RUN_REPORTS_DIR / "run_summary.md").write_text("\n".join(lines), encoding="utf-8")


def archive_top_cases(df: pd.DataFrame, n: int = 3) -> None:
    if df.empty or "Stable" not in df.columns or "Score" not in df.columns:
        return

    top = df[df["Stable"] == 1].sort_values("Score", ascending=False).head(n)
    if top.empty:
        return

    top_dir = RUN_DIR / "top_cases"
    top_dir.mkdir(parents=True, exist_ok=True)

    for _, row in top.iterrows():
        for col in ["ASC_File", "Log_File", "Net_File"]:
            src = row.get(col, "")
            if isinstance(src, str) and src and Path(src).exists():
                shutil.copy2(src, top_dir / Path(src).name)




def train_surrogates(df: pd.DataFrame) -> Dict[str, Dict[str, float | str]]:
    report: Dict[str, Dict[str, float | str]] = {}
    if not SKLEARN_OK:
        return {"sklearn": {"status": "skipped", "reason": "scikit-learn not available"}}

    feature_cols = []

    if "Rload_val" in df.columns and df["Rload_val"].notna().any():
        feature_cols.append("Rload_val")

    if "Rg_on" in df.columns and df["Rg_on"].notna().any():
        feature_cols.append("Rg_on")

    if "Rg_off" in df.columns and df["Rg_off"].notna().any():
        feature_cols.append("Rg_off")

    if "f_sw_Hz" in df.columns and df["f_sw_Hz"].notna().any():
        feature_cols.append("f_sw_Hz")
    if "Deadtime_s" in df.columns and df["Deadtime_s"].notna().any():
        feature_cols.append("Deadtime_s")

    if not feature_cols:
        return {"features": {"status": "skipped", "reason": "No usable feature columns"}}

    clean_features = df[feature_cols].copy()
    for col in feature_cols:
        clean_features[col] = clean_features[col].fillna(-1)

    # Classifier: stable vs bad physics.
    if df["Stable"].nunique() >= 2 and len(df) >= 8:
        X_train, X_test, y_train, y_test = train_test_split(
            clean_features, df["Stable"], test_size=0.25, random_state=42, stratify=df["Stable"]
        )
        clf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        report["Stable_classifier"] = {
            "status": "trained",
            "accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "features": ", ".join(feature_cols),
        }
    else:
        report["Stable_classifier"] = {"status": "skipped", "reason": "Need at least 2 classes and >=8 rows"}

    # Regressors: useful continuous design targets.
    targets = ["Score", "PF", "Efficiency_pct", "Iin_pk_A", "Pin_W", "Vout_pp_V"]
    valid_rows = df[df["Stable"] == 1].copy()
    for target in targets:
        if target not in valid_rows.columns:
            continue
        subset = valid_rows.dropna(subset=[target])
        if len(subset) < 8 or subset[target].nunique() < 2:
            report[f"{target}_regressor"] = {"status": "skipped", "reason": "Need >=8 valid rows with target variation"}
            continue
        X = subset[feature_cols].fillna(-1)
        y = subset[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        reg = RandomForestRegressor(n_estimators=300, random_state=42)
        reg.fit(X_train, y_train)
        pred = reg.predict(X_test)
        report[f"{target}_regressor"] = {
            "status": "trained",
            "r2": round(float(r2_score(y_test, pred)), 4) if len(y_test) > 1 else np.nan,
            "mae": round(float(mean_absolute_error(y_test, pred)), 4),
            "features": ", ".join(feature_cols),
        }

    return report


@dataclass
class CaseResult:
    data: Dict[str, object]


def run_one_case(
    original_template: bytes,
    rload: float,
    rg_on: float,
    rg_off: float,
    f_sw: Optional[float],
    deadtime: Optional[float],
    case_idx: int,
    total_cases: int,
    global_start_time: float,
    completed_case_times: List[float],
) -> CaseResult:
    case_tag = (
        f"Rload_{safe_case_value(rload)}"
        f"_Rgon_{safe_case_value(rg_on)}_Rgoff_{safe_case_value(rg_off)}"
        f"_fsw_{safe_case_value(f_sw)}_dt_{safe_case_value(deadtime)}"
    )
    case_stem = f"case_{case_tag}"
    case_asc = PROJECT_DIR / f"{case_stem}.asc"
    case_log = PROJECT_DIR / f"{case_stem}.log"
    case_net = PROJECT_DIR / f"{case_stem}.net"
    case_raw = PROJECT_DIR / f"{case_stem}.raw"

    elapsed_global_s = time.time() - global_start_time
    avg_case_s = float(np.mean(completed_case_times)) if completed_case_times else 0.0

    print_case_header(
        case_idx=case_idx,
        total_cases=total_cases,
        case_tag=case_tag,
        rload=rload,
        rg_on=rg_on,
        rg_off=rg_off,
        f_sw=f_sw,
        deadtime=deadtime,
        elapsed_global_s=elapsed_global_s,
        avg_case_s=avg_case_s,
    )

    patched_bytes = patch_param_line_preserve_schematic(
        original_template,
        rg_on,
        rg_off,
        f_sw,
        deadtime,
        rload,
    )
    case_asc.write_bytes(patched_bytes)

    delete_case_outputs(case_stem)
    start_time = time.time()
    completed = run_ltspice_case(case_asc)
    log_text = wait_for_complete_log(case_log, start_time)

    out_log = RUN_LOGS_DIR / f"{case_stem}.log"
    out_net = RUN_NETLISTS_DIR / f"{case_stem}.net"
    out_raw = RUN_RAW_DIR / f"{case_stem}.raw"
    out_asc = RUN_CASES_DIR / f"{case_stem}.asc"
    out_diff = RUN_DIFFS_DIR / f"{case_stem}_netlist_diff.txt"

    copy_if_exists(case_asc, out_asc)
    net_file = ""
    raw_file = ""

    if ARCHIVE_LEVEL in ("debug", "full_archive"):
        copy_if_exists(case_log, out_log)
        net_file = copy_if_exists(case_net, out_net)

    if ARCHIVE_LEVEL == "full_archive":
        raw_file = copy_if_exists(case_raw, out_raw)

    failed, fail_reason = detect_failure(log_text, completed.returncode)

    pin = parse_measure(log_text, "pin")
    pin_pos = parse_measure(log_text, "pin_pos")
    pin_neg = parse_measure(log_text, "pin_neg")
    iin_rms = parse_measure(log_text, "iin_rms")
    iin_pk = parse_measure(log_text, "iin_pk")
    vout_avg = parse_measure(log_text, "vout_avg")
    vout_pp = parse_measure(log_text, "vout_pp")
    icbus_rms = parse_measure(log_text, "icbus_rms")
    icbus_pk = parse_measure(log_text, "icbus_pk")
    il_avg = parse_measure(log_text, "il_avg")
    il_rms = parse_measure(log_text, "il_rms")
    il_pk = parse_measure(log_text, "il_pk")
    il_min = parse_measure(log_text, "il_min")
    vsw_pk = parse_measure(log_text, "vsw_pk")
    vout_pk = parse_measure(log_text, "vout_pk")
    vout_min = parse_measure(log_text, "vout_min")

    il_ripple = np.nan
    if not np.isnan(il_pk) and not np.isnan(il_min):
        il_ripple = il_pk - abs(il_min)

    # Template 1: ideal-bus source-side PF
    vin_rms = parse_measure(log_text, "vin_src_rms")

    # Optional direct LTspice PF, only used if it exists
    pf_src = parse_measure(log_text, "pf_src")

    if not np.isnan(pf_src):
        pf = pf_src
    else:
        pf = calculate_pf(pin, vin_rms, iin_rms)

    if TEMPLATE_MODE == "ideal_bus":
        pout_w = np.nan
        efficiency_pct = np.nan
        loss_w = np.nan
        pout_method = "Ignored for ideal V2 bus template"

    elif TEMPLATE_MODE == "real_load":
        pout_w = parse_measure(log_text, "pout")
        efficiency_pct = parse_measure(log_text, "eff")
        loss_w = parse_measure(log_text, "ploss")
        pout_method = "Direct real-load LTspice measurement"

    elapsed_s = parse_elapsed(log_text)
    if TEMPLATE_MODE == "real_load":
        load_ohms, load_detect_line = find_output_load_resistor_ohms(case_net)
    else:
        load_ohms, load_detect_line = (np.nan, "Ignored for ideal V2 bus template")

    stable, bad_physics, reason = evaluate_physics(
        failed=failed,
        fail_reason=fail_reason,
        pin=pin,
        iin_rms=iin_rms,
        iin_pk=iin_pk,
        vout_avg=vout_avg,
        vout_pp=vout_pp,
        elapsed_s=elapsed_s,
        pf=pf,
        efficiency_pct=efficiency_pct,
    )

    if TEMPLATE_MODE == "ideal_bus" and "efficiency ignored" in reason.lower():
        efficiency_pct = np.nan
        pout_w = np.nan
        loss_w = np.nan

    if bad_physics:
        write_netlist_diff(KNOWN_GOOD_TEMPLATE_NET, case_net, out_diff)

    if bad_physics and ARCHIVE_LEVEL in ("debug", "full_archive"):
        copy_if_exists(case_log, out_log)
        net_file = copy_if_exists(case_net, out_net)
        copy_if_exists(case_asc, out_asc)

    netlist_warnings = validate_generated_netlist(case_net, rload)

    if not bad_physics and not np.isnan(iin_pk) and iin_pk > INDUCTOR_SATURATION_LIMIT_A:
        stable, bad_physics, reason = 0, True, "Inductor saturation risk"

    safe_v_limit = DEVICE_VOLTAGE_RATING * VOLTAGE_DERATING
    if not bad_physics:
        for stress_v in (vout_pk, vsw_pk):
            if not np.isnan(stress_v) and stress_v > safe_v_limit:
                stable, bad_physics, reason = 0, True, "Device voltage derating exceeded"
                break

    row = {
        "Rload_val": rload,
        "Rg_on": rg_on,
        "Rg_off": rg_off,
        "f_sw_Hz": f_sw,
        "Deadtime_s": deadtime,
        "Pin_W": pin,
        "Pin_pos_W": pin_pos,
        "Pin_neg_W": pin_neg,
        "Pout_W": pout_w,
        "Loss_W": loss_w,
        "Efficiency_pct": efficiency_pct,
        "Vin_rms_V": vin_rms,
        "PF": pf,
        "Iin_rms_A": iin_rms,
        "Iin_pk_A": iin_pk,
        "Vout_avg_V": vout_avg,
        "Vout_pp_V": vout_pp,
        "Vout_pk_V": vout_pk,
        "Vout_min_V": vout_min,
        "Vsw_pk_V": vsw_pk,
        "Icbus_rms_A": icbus_rms,
        "Icbus_pk_A": icbus_pk,
        "IL_avg_A": il_avg,
        "IL_rms_A": il_rms,
        "IL_pk_A": il_pk,
        "IL_min_A": il_min,
        "IL_ripple_A": il_ripple,
        "Elapsed_s": elapsed_s,
        "Stable": stable,
        "Bad_Physics": bad_physics,
        "Reason": reason,
        "Failed": failed,
        "Fail_Reason": fail_reason,
        "LTspice_returncode": completed.returncode,
        "Load_Ohms_Detected": load_ohms,
        "Load_Detect_Line": load_detect_line,
        "Pout_Method": pout_method,
        "ASC_File": str(out_asc),
        "Log_File": str(out_log),
        "Net_File": net_file,
        "Raw_File": raw_file,
        "Diff_File": str(out_diff) if out_diff.exists() else "",
        "Netlist_Warnings": "; ".join(netlist_warnings),
    }
    row["Score"] = calculate_score(pd.Series(row))

    loss_est = estimate_losses(pd.Series(row), fsw=f_sw if f_sw is not None else 50_000)
    row.update(loss_est)
    tj_est, thermal_risk = estimate_thermal(row.get("Loss_W", np.nan))
    row["Tj_est_C"] = tj_est
    row["Thermal_Risk"] = thermal_risk

    print_case_result(row)
    print(c("  Interpretation: ", Term.BOLD) + interpret_case(row))

    return CaseResult(row)


def main() -> None:
    user_mode = ask_user_mode()

    global rload_values, rg_on_values, rg_off_values, fsw_values, deadtime_values

    if user_mode == "simulate_only":
        rload_values, rg_on_values, rg_off_values, fsw_values, deadtime_values = configure_test_run()

    if user_mode == "exit":
        print("Exiting.")
        return

    if user_mode == "report_only":
        generate_reports_from_existing_data()
        return

    if not TEMPLATE_ASC.exists():
        raise FileNotFoundError(f"Template schematic not found: {TEMPLATE_ASC}")
    if not Path(LTSPICE_EXE).exists():
        raise FileNotFoundError(f"LTspice executable not found: {LTSPICE_EXE}")

    original_template = read_bytes(TEMPLATE_ASC)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    permanent_backup = ARCHIVE_DIR / f"gan_pfc_template_backup_{timestamp}.asc"
    permanent_backup.write_bytes(original_template)

    append_to_master = user_mode == "simulate_append"

    if MANUAL_CASES:
        total_cases = len(MANUAL_CASES)
    else:
        total_cases = len(rload_values) * len(rg_on_values) * len(rg_off_values) * len(fsw_values) * len(deadtime_values)
    print_banner("GAN PFC REAL-LOAD OPTIMIZATION RUN")
    print(f"Run mode: {RUN_MODE}")
    print(f"Total cases: {total_cases}")
    print(f"Template: {TEMPLATE_ASC.name}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Template backup saved: {permanent_backup}")

    results: List[Dict[str, object]] = []
    global_start_time = time.time()
    completed_case_times: List[float] = []
    case_idx = 0

    if MANUAL_CASES:
        case_plan = [
            (rload, rg_on, rg_off, None, None)
            for rload, rg_on, rg_off in MANUAL_CASES
        ]
    else:
        case_plan = [
            (rload, rg_on, rg_off, f_sw, deadtime)
            for rload in rload_values
            for rg_on in rg_on_values
            for rg_off in rg_off_values
            for f_sw in fsw_values
            for deadtime in deadtime_values
        ]

    for rload, rg_on, rg_off, f_sw, deadtime in case_plan:
        case_idx += 1
        case_start = time.time()
        try:
            case_result = run_one_case(
                original_template=original_template,
                rload=rload,
                rg_on=rg_on,
                rg_off=rg_off,
                f_sw=f_sw,
                deadtime=deadtime,
                case_idx=case_idx,
                total_cases=total_cases,
                global_start_time=global_start_time,
                completed_case_times=completed_case_times,
            )
            case_elapsed = time.time() - case_start
            completed_case_times.append(case_elapsed)
            results.append(case_result.data)
            print_best_so_far(results)

        except subprocess.TimeoutExpired:
            completed_case_times.append(time.time() - case_start)
            print(c("ERROR: LTspice timeout. Case skipped.", Term.RED + Term.BOLD))
            results.append({
                "Rload_val": rload,
                "Rg_on": rg_on,
                "Rg_off": rg_off,
                "f_sw_Hz": f_sw,
                "Deadtime_s": deadtime,
                "Stable": 0,
                "Bad_Physics": True,
                "Reason": "LTspice timeout",
                "Failed": True,
                "Fail_Reason": "Timeout",
                "Score": -1e9,
            })

        except Exception as exc:
            completed_case_times.append(time.time() - case_start)
            print(c(f"ERROR: {exc}", Term.RED + Term.BOLD))
            results.append({
                "Rload_val": rload,
                "Rg_on": rg_on,
                "Rg_off": rg_off,
                "f_sw_Hz": f_sw,
                "Deadtime_s": deadtime,
                "Stable": 0,
                "Bad_Physics": True,
                "Reason": str(exc),
                "Failed": True,
                "Fail_Reason": type(exc).__name__,
                "Score": -1e9,
            })

    df = pd.DataFrame(results)

    # Guarantee output schema even if every case errors before LTspice produces measures.
    expected_columns = [
        "Rload_val", "Rg_on", "Rg_off", "f_sw_Hz", "Deadtime_s", "Pin_W", "Pout_W", "Loss_W",
        "Efficiency_pct", "Vin_rms_V", "PF", "Iin_rms_A", "Iin_pk_A", "Vout_avg_V",
        "Vout_pp_V", "Vout_pk_V", "Vout_min_V", "Vsw_pk_V", "Icbus_rms_A", "Icbus_pk_A",
        "IL_avg_A", "IL_rms_A", "IL_pk_A", "IL_min_A", "IL_ripple_A", "Tj_est_C", "Thermal_Risk",
        "Pcond_est_W", "Pgate_est_W", "Punexplained_loss_W", "Elapsed_s", "Stable", "Bad_Physics", "Reason", "Failed",
        "Fail_Reason", "LTspice_returncode", "Load_Ohms_Detected", "Load_Detect_Line",
        "Pout_Method", "Netlist_Warnings", "ASC_File", "Log_File", "Net_File", "Raw_File", "Diff_File", "Score"
    ]
    for col in expected_columns:
        if col not in df.columns:
            df[col] = np.nan

    df = df.sort_values(["Stable", "Score"], ascending=[False, False]).reset_index(drop=True)

    dataset_xlsx = RUN_DIR / "run_dataset.xlsx"
    dataset_csv = RUN_DIR / "run_dataset.csv"
    top10_xlsx = RUN_REPORTS_DIR / "top_10_designs.xlsx"
    diagnostics_xlsx = RUN_REPORTS_DIR / "bad_physics_diagnostics.xlsx"
    ml_report_xlsx = RUN_REPORTS_DIR / "ml_surrogate_report.xlsx"

    df.to_excel(dataset_xlsx, index=False)
    df.to_csv(dataset_csv, index=False)

    if append_to_master:
        master_df = append_to_master_dataset(df)
        print(f"Master dataset updated: {MASTER_DATASET_XLSX} ({len(master_df)} rows)")
    else:
        master_df = df
        print(f"Master dataset not updated (simulate-only mode, {len(master_df)} rows in run dataset)")

    top10 = df[df["Stable"] == 1].sort_values("Score", ascending=False).head(10)
    top10.to_excel(top10_xlsx, index=False)

    diagnostics = df[df["Bad_Physics"] == True].copy()
    diagnostics.to_excel(diagnostics_xlsx, index=False)

    save_design_plots(df)

    ml_report = train_surrogates(df)
    ml_rows = []
    for model_name, fields in ml_report.items():
        row = {"Model": model_name}
        row.update(fields)
        ml_rows.append(row)
    pd.DataFrame(ml_rows).to_excel(ml_report_xlsx, index=False)

    # Leadership feature pack: generate executive brief, reports, templates, and dashboard
    try:
        generate_run_summary(df)
        generate_executive_brief(df)
        multi_objective_selector(df)
        best_by_category(df)
        risk_register(df)
        ai_next_simulation_recommendations(df)
        generate_design_mode_leaderboards(df)
        application_mode_scores(df)
        generate_templates()
        generate_html_dashboard(df)
        next_cases = recommend_next_cases(df)
        if next_cases:
            (RUN_REPORTS_DIR / "next_cases_recommendations.txt").write_text("\n".join(f"{x:g} Ω" for x in next_cases), encoding="utf-8")
    except Exception as exc:
        print(c(f"WARNING: leadership feature pack generation failed: {exc}", Term.YELLOW))

    print("\n================ FINAL SUMMARY ================")
    print(df[["Rload_val", "Rg_on", "Rg_off", "f_sw_Hz", "Deadtime_s", "Pin_W", "Pout_W", "Vout_avg_V", "Vout_pp_V", "PF", "Efficiency_pct", "Stable", "Score", "Reason"]])
    print("\nStable count:", int(df["Stable"].sum()) if "Stable" in df else 0)
    print("Bad physics count:", int(df["Bad_Physics"].sum()) if "Bad_Physics" in df else 0)
    archive_top_cases(df, n=3)

    print(f"Saved run dataset: {dataset_xlsx}")
    print(f"Saved run CSV: {dataset_csv}")
    print(f"Saved master dataset: {MASTER_DATASET_XLSX if append_to_master else 'not updated'}")
    print(f"Saved top designs: {top10_xlsx}")
    print(f"Saved diagnostics: {diagnostics_xlsx}")
    print(f"Saved ML report: {ml_report_xlsx}")
    print_final_leaderboard(df, n=10)
    print("Saved plots:")
    for name in [
        "load_sweep_dashboard.png",
        "score_vs_rload.png",
        "efficiency_vs_rload.png",
        "vout_vs_rload.png",
        "pf_vs_rload.png",
        "ripple_vs_rload.png",
        "pout_vs_rload.png",
        "current_vs_rload.png",
        "efficiency_regulation_tradeoff.png",
        "top_10_designs_score.png",
        "single_case_summary.png",
        "stability_map.png",
        "optimization_score_map.png",
        "pf_map.png",
        "efficiency_map.png",
        "peak_current_map.png",
        "vout_ripple_map.png",
    ]:
        p = RUN_PLOTS_DIR / name
        if p.exists():
            print(f"  {p}")

    if not top10.empty:
        best = top10.iloc[0]
        print("\nBest design:")
        print(best[["Rload_val", "Rg_on", "Rg_off", "f_sw_Hz", "Deadtime_s", "Score", "PF", "Efficiency_pct", "Pin_W", "Pout_W", "Vout_avg_V", "Vout_pp_V", "Iin_pk_A", "Reason"]])


if __name__ == "__main__":
    main()
