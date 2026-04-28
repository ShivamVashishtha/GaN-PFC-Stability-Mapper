
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

TEMPLATE_ASC = PROJECT_DIR / "gan_pfc_template_real_load.asc"
KNOWN_GOOD_TEMPLATE_NET = PROJECT_DIR / "gan_pfc_template.net"

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

# Parameter name aliases. The script patches whichever alias is present in the .asc .param line.
PARAM_ALIASES = {
    "Rg_on": ["Rg_on", "Rgon", "RGON", "RG_ON"],
    "Rg_off": ["Rg_off", "Rgoff", "RGOFF", "RG_OFF"],
    "f_sw": ["f_sw", "fsw", "Fsw", "FSW", "Freq", "freq", "SwitchFreq"],
    "deadtime": ["deadtime", "DeadTime", "DT", "dt", "Tdead", "tdead"],
}

# Existing template logs already contain these measures.
REQUIRED_MEASURES = ["pin", "iin_rms", "iin_pk", "vout_avg", "vout_pp"]

# Add low-risk LTspice measurements to the generated case schematic only.
# PF is calculated in Python from pin/(vin_rms*iin_rms), so only vin_rms must be injected.
INJECT_EXTRA_MEASURES = False
EXTRA_MEASURE_DIRECTIVES = []

# Physics-health thresholds. Tune later after you get more real cases.
MIN_GOOD_ELAPSED_S = 10.0
MIN_PIN_W = 1000.0
MIN_IIN_RMS_A = 1.0
MIN_IIN_PK_A = 5.0
MAX_IIN_RMS_A = 120.0
VOUT_MIN_V = 350.0
VOUT_MAX_V = 450.0
VOUT_TARGET_V = 400.0
PF_TARGET_MIN = 0.90
EFF_TARGET_MIN = 90.0

# If your netlist contains an output resistor connected to OUT and ground, the script estimates Pout = Vout_avg^2/R.
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
    m = re.search(rf"^{re.escape(name)}\s*:.*?=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", log_text, re.I | re.M)
    if not m:
        return np.nan
    try:
        return float(m.group(1))
    except ValueError:
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
    if not np.isnan(pf) and pf > 1.15:
        return 0, True, "PF calculation physically impossible; check measurement signs/nodes"

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
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def save_design_plots(df: pd.DataFrame) -> None:
    if df.empty:
        return

    colors = df["Stable"].map({1: "green", 0: "red"}).fillna("gray")
    plt.figure(figsize=(8, 5))
    plt.scatter(df["Rg_on"], df["Rg_off"], c=colors, s=90, edgecolors="black")
    plt.xlabel("Rg_on (ohm)")
    plt.ylabel("Rg_off (ohm)")
    plt.title("GaN PFC Stability Map")
    plt.grid(True)
    plt.savefig(OUTPUT_DIR / "stability_map.png", dpi=300, bbox_inches="tight")
    plt.close()

    save_plot_scatter(df, "Score", "optimization_score_map.png", "Optimization Score Map")
    save_plot_scatter(df, "PF", "pf_map.png", "Power Factor Map")
    save_plot_scatter(df, "Efficiency_pct", "efficiency_map.png", "Efficiency Map")
    save_plot_scatter(df, "Iin_pk_A", "peak_current_map.png", "Peak Current Map")
    save_plot_scatter(df, "Vout_pp_V", "vout_ripple_map.png", "Output Ripple Map")

    top = df[df["Stable"] == 1].sort_values("Score", ascending=False).head(10)
    if not top.empty:
        labels = [f"Rg_on={r.Rg_on:g}\nRg_off={r.Rg_off:g}" for r in top.itertuples()]
        plt.figure(figsize=(10, 5))
        plt.bar(range(len(top)), top["Score"])
        plt.xticks(range(len(top)), labels, rotation=45, ha="right")
        plt.ylabel("Optimization Score")
        plt.title("Top Ranked GaN PFC Gate-Drive Designs")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "top_10_designs_score.png", dpi=300, bbox_inches="tight")
        plt.close()


def train_surrogates(df: pd.DataFrame) -> Dict[str, Dict[str, float | str]]:
    report: Dict[str, Dict[str, float | str]] = {}
    if not SKLEARN_OK:
        return {"sklearn": {"status": "skipped", "reason": "scikit-learn not available"}}

    feature_cols = ["Rg_on", "Rg_off"]
    if "f_sw_Hz" in df.columns and df["f_sw_Hz"].notna().any():
        feature_cols.append("f_sw_Hz")
    if "Deadtime_s" in df.columns and df["Deadtime_s"].notna().any():
        feature_cols.append("Deadtime_s")

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


def run_one_case(original_template: bytes, rg_on: float, rg_off: float, f_sw: Optional[float], deadtime: Optional[float]) -> CaseResult:
    case_tag = (
        f"Rgon_{safe_case_value(rg_on)}_Rgoff_{safe_case_value(rg_off)}"
        f"_fsw_{safe_case_value(f_sw)}_dt_{safe_case_value(deadtime)}"
    )
    case_stem = f"case_{case_tag}"
    case_asc = PROJECT_DIR / f"{case_stem}.asc"
    case_log = PROJECT_DIR / f"{case_stem}.log"
    case_net = PROJECT_DIR / f"{case_stem}.net"
    case_raw = PROJECT_DIR / f"{case_stem}.raw"

    print(f"\nRunning {case_tag}")

    patched_bytes = patch_param_line_preserve_schematic(original_template, rg_on, rg_off, f_sw, deadtime)
    case_asc.write_bytes(patched_bytes)

    delete_case_outputs(case_stem)
    start_time = time.time()
    completed = run_ltspice_case(case_asc)
    log_text = wait_for_complete_log(case_log, start_time)

    out_log = OUTPUT_DIR / f"{case_stem}.log"
    out_net = OUTPUT_DIR / f"{case_stem}.net"
    out_raw = OUTPUT_DIR / f"{case_stem}.raw"
    out_asc = OUTPUT_DIR / f"{case_stem}.asc"
    out_diff = OUTPUT_DIR / f"{case_stem}_netlist_diff.txt"

    copy_if_exists(case_asc, out_asc)
    copy_if_exists(case_log, out_log)
    net_file = copy_if_exists(case_net, out_net)
    raw_file = copy_if_exists(case_raw, out_raw)

    failed, fail_reason = detect_failure(log_text, completed.returncode)

    pin = parse_measure(log_text, "pin")
    iin_rms = parse_measure(log_text, "iin_rms")
    iin_pk = parse_measure(log_text, "iin_pk")
    vout_avg = parse_measure(log_text, "vout_avg")
    vout_pp = parse_measure(log_text, "vout_pp")

    # Template 1: ideal-bus source-side PF
    vin_rms = parse_measure(log_text, "vin_src_rms")

    # Optional direct LTspice PF, only used if it exists
    pf_src = parse_measure(log_text, "pf_src")

    if not np.isnan(pf_src):
        pf = pf_src
    else:
        pf = calculate_pf(pin, vin_rms, iin_rms)

    # Template 1 ignores output power/efficiency
    pout_w = np.nan
    efficiency_pct = np.nan
    loss_w = np.nan
    pout_method = "Ignored for ideal V2 bus template"

    elapsed_s = parse_elapsed(log_text)
    load_ohms, load_detect_line = (np.nan, "Ignored for ideal V2 bus template")

    stable, bad_physics, reason = evaluate_physics(
        failed=failed,
        fail_reason=fail_reason,
        pin=pin,
        iin_rms=iin_rms,
        iin_pk=iin_pk,
        vout_avg=vout_avg,
        elapsed_s=elapsed_s,
        pf=pf,
        efficiency_pct=efficiency_pct,
    )

    if "efficiency ignored" in reason.lower():
        efficiency_pct = np.nan
        pout_w = np.nan
        loss_w = np.nan

    if bad_physics:
        write_netlist_diff(KNOWN_GOOD_TEMPLATE_NET, case_net, out_diff)

    row = {
        "Rg_on": rg_on,
        "Rg_off": rg_off,
        "f_sw_Hz": f_sw,
        "Deadtime_s": deadtime,
        "Pin_W": pin,
        "Pout_W": pout_w,
        "Loss_W": loss_w,
        "Efficiency_pct": efficiency_pct,
        "Vin_rms_V": vin_rms,
        "PF": pf,
        "Iin_rms_A": iin_rms,
        "Iin_pk_A": iin_pk,
        "Vout_avg_V": vout_avg,
        "Vout_pp_V": vout_pp,
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
    }
    row["Score"] = calculate_score(pd.Series(row))

    print(
        f"Elapsed={elapsed_s:.3f}s | Pin={pin:.3f} W | Iin_rms={iin_rms:.3f} A | "
        f"PF={pf if not np.isnan(pf) else np.nan:.4f} | Eff={efficiency_pct if not np.isnan(efficiency_pct) else np.nan:.3f}% | "
        f"Stable={stable} | Score={row['Score']} | {reason}"
    )

    return CaseResult(row)


def main() -> None:
    if not TEMPLATE_ASC.exists():
        raise FileNotFoundError(f"Template schematic not found: {TEMPLATE_ASC}")
    if not Path(LTSPICE_EXE).exists():
        raise FileNotFoundError(f"LTspice executable not found: {LTSPICE_EXE}")

    original_template = read_bytes(TEMPLATE_ASC)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    permanent_backup = OUTPUT_DIR / f"gan_pfc_template_backup_{timestamp}.asc"
    permanent_backup.write_bytes(original_template)

    total_cases = len(rg_on_values) * len(rg_off_values) * len(fsw_values) * len(deadtime_values)
    print(f"Run mode: {RUN_MODE}")
    print(f"Total cases: {total_cases}")
    print(f"Template backup saved: {permanent_backup}")

    results: List[Dict[str, object]] = []

    for rg_on in rg_on_values:
        for rg_off in rg_off_values:
            for f_sw in fsw_values:
                for deadtime in deadtime_values:
                    try:
                        case_result = run_one_case(original_template, rg_on, rg_off, f_sw, deadtime)
                        results.append(case_result.data)
                    except subprocess.TimeoutExpired:
                        print("ERROR: LTspice timeout. Case skipped.")
                        results.append({
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
                        print(f"ERROR: {exc}")
                        results.append({
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
        "Rg_on", "Rg_off", "f_sw_Hz", "Deadtime_s", "Pin_W", "Pout_W", "Loss_W",
        "Efficiency_pct", "Vin_rms_V", "PF", "Iin_rms_A", "Iin_pk_A", "Vout_avg_V",
        "Vout_pp_V", "Elapsed_s", "Stable", "Bad_Physics", "Reason", "Failed",
        "Fail_Reason", "LTspice_returncode", "Load_Ohms_Detected", "Load_Detect_Line",
        "Pout_Method", "ASC_File", "Log_File", "Net_File", "Raw_File", "Diff_File", "Score"
    ]
    for col in expected_columns:
        if col not in df.columns:
            df[col] = np.nan

    df = df.sort_values(["Stable", "Score"], ascending=[False, False]).reset_index(drop=True)

    dataset_xlsx = OUTPUT_DIR / "gan_pfc_innovation_dataset.xlsx"
    dataset_csv = OUTPUT_DIR / "gan_pfc_innovation_dataset.csv"
    top10_xlsx = OUTPUT_DIR / "top_10_designs.xlsx"
    diagnostics_xlsx = OUTPUT_DIR / "bad_physics_diagnostics.xlsx"
    ml_report_xlsx = OUTPUT_DIR / "ml_surrogate_report.xlsx"

    df.to_excel(dataset_xlsx, index=False)
    df.to_csv(dataset_csv, index=False)

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

    print("\n================ FINAL SUMMARY ================")
    print(df[["Rg_on", "Rg_off", "f_sw_Hz", "Deadtime_s", "Pin_W", "PF", "Efficiency_pct", "Stable", "Score", "Reason"]])
    print("\nStable count:", int(df["Stable"].sum()) if "Stable" in df else 0)
    print("Bad physics count:", int(df["Bad_Physics"].sum()) if "Bad_Physics" in df else 0)
    print(f"Saved dataset: {dataset_xlsx}")
    print(f"Saved CSV: {dataset_csv}")
    print(f"Saved top designs: {top10_xlsx}")
    print(f"Saved diagnostics: {diagnostics_xlsx}")
    print(f"Saved ML report: {ml_report_xlsx}")
    print("Saved plots:")
    for name in [
        "stability_map.png",
        "optimization_score_map.png",
        "pf_map.png",
        "efficiency_map.png",
        "peak_current_map.png",
        "vout_ripple_map.png",
        "top_10_designs_score.png",
    ]:
        p = OUTPUT_DIR / name
        if p.exists():
            print(f"  {p}")

    if not top10.empty:
        best = top10.iloc[0]
        print("\nBest design:")
        print(best[["Rg_on", "Rg_off", "f_sw_Hz", "Deadtime_s", "Score", "PF", "Efficiency_pct", "Pin_W", "Iin_pk_A", "Reason"]])


if __name__ == "__main__":
    main()
