import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

LTSPICE_EXE = r"C:\Users\vashi\AppData\Local\Programs\ADI\LTspice\LTspice.exe"

PROJECT_DIR = Path(r"C:\Users\vashi\Downloads\a-010_ac-dc_totem-pole_bridgeless_pfc_vin=200v_iin=100a_synchronous_fets\a-010_ac-dc_totem-pole_bridgeless_pfc_vin=200v_iin=100a_synchronous_fets")
OUTPUT_DIR = Path(r"C:\Users\vashi\bmw_gan")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEMPLATE_ASC = PROJECT_DIR / "gan_pfc_template.asc"

# Expand this later after the single known-good case works.
rg_on_values = [1, 2, 3]
rg_off_values = [0.5, 1, 2]

MEASURE_NAMES = ["pin", "iin_rms", "iin_pk", "vout_avg", "vout_pp"]


def read_bytes(path: Path) -> bytes:
    return path.read_bytes()


def decode_asc(data: bytes) -> str:
    # LTspice .asc is normally ANSI/ASCII-compatible. Keep fallback safe.
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            pass
    return data.decode("latin-1", errors="ignore")


def encode_asc(text: str) -> bytes:
    # Use cp1252 to avoid changing typical Windows LTspice schematic encoding.
    return text.encode("cp1252", errors="replace")


def patch_param_line_preserve_schematic(original_bytes: bytes, rg_on: float, rg_off: float) -> bytes:
    """
    Patch only the LTspice TEXT directive containing .param Rg_on/Rg_off.
    Do not rewrite the original template file. Do not append duplicate .param lines.
    """
    text = decode_asc(original_bytes)
    newline = "\r\n" if "\r\n" in text else "\n"
    lines = text.splitlines()

    patched_lines = []
    patched = False

    for line in lines:
        original_line = line

        # Only edit a visible LTspice SPICE directive, not comments or random text.
        if line.lstrip().startswith("TEXT") and "!.param" in line and re.search(r"\b(?:Rg_on|Rgon)\b|\b(?:Rg_off|Rgoff)\b", line, re.I):
            # Support either naming style if the schematic uses Rg_on/Rg_off or Rgon/Rgoff.
            line = re.sub(r"\b(?:Rg_on|Rgon)\s*=\s*[^\s]+", f"Rg_on={rg_on}", line, flags=re.I)
            line = re.sub(r"\b(?:Rg_off|Rgoff)\s*=\s*[^\s]+", f"Rg_off={rg_off}", line, flags=re.I)
            if not re.search(r"\bRg_on\s*=", line, re.I):
                line += f" Rg_on={rg_on}"
            if not re.search(r"\bRg_off\s*=", line, re.I):
                line += f" Rg_off={rg_off}"
            patched = True

        patched_lines.append(line)

    if not patched:
        raise RuntimeError(
            "Could not find an existing LTspice TEXT directive with .param Rg_on/Rg_off. "
            "Do not append a second .param line; fix the template parameter names first."
        )

    patched_text = newline.join(patched_lines) + newline

    # Do not use a strict word-boundary value check here. LTspice parameter lines can
    # contain braces, suffixes, escaped directive text, or continuation formatting.
    # Instead, print the edited parameter directive and let the generated netlist/log
    # confirm the electrical result.
    param_debug_lines = [ln for ln in patched_text.splitlines() if ".param" in ln.lower() and re.search(r"Rg[_]?on|Rg[_]?off|Rgon|Rgoff", ln, re.I)]
    if not param_debug_lines:
        raise RuntimeError("Parameter line was patched, but no Rg_on/Rg_off parameter directive remains.")

    print("PARAM DEBUG:")
    for ln in param_debug_lines:
        print("  " + ln)

    return encode_asc(patched_text)


def delete_case_outputs(case_stem: str) -> None:
    for ext in (".log", ".net", ".raw", ".op.raw"):
        p = PROJECT_DIR / f"{case_stem}{ext}"
        if p.exists():
            for _ in range(5):
                try:
                    p.unlink()
                    break
                except PermissionError:
                    time.sleep(0.5)


def run_ltspice_case(case_asc: Path, timeout_s: int = 420) -> subprocess.CompletedProcess:
    return subprocess.run(
        [LTSPICE_EXE, "-b", str(case_asc)],
        cwd=str(PROJECT_DIR),
        timeout=timeout_s,
        capture_output=True,
        text=True,
    )


def wait_for_complete_log(log_path: Path, start_time: float, timeout_s: int = 30) -> str:
    deadline = time.time() + timeout_s
    last_text = ""

    while time.time() < deadline:
        if log_path.exists() and log_path.stat().st_mtime >= start_time:
            text = log_path.read_text(encoding="utf-8", errors="ignore")
            last_text = text
            has_elapsed = "Total elapsed time:" in text
            has_measures = all(re.search(rf"^{m}\s*:", text, re.I | re.M) for m in MEASURE_NAMES)
            if has_elapsed and has_measures:
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


def classify_stability(pin: float, iin_rms: float, failed: bool) -> int:
    if failed or np.isnan(pin) or np.isnan(iin_rms):
        return 0
    if pin <= 1000:
        return 0
    if iin_rms <= 1:
        return 0
    if iin_rms > 100:
        return 0
    return 1


def main() -> None:
    original_template = read_bytes(TEMPLATE_ASC)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    permanent_backup = OUTPUT_DIR / f"gan_pfc_template_backup_{timestamp}.asc"
    permanent_backup.write_bytes(original_template)

    results = []

    for rg_on in rg_on_values:
        for rg_off in rg_off_values:
            case_tag = f"Rgon_{rg_on}_Rgoff_{rg_off}".replace(".", "p")
            case_stem = f"case_{case_tag}"
            case_asc = PROJECT_DIR / f"{case_stem}.asc"
            case_log = PROJECT_DIR / f"{case_stem}.log"
            case_net = PROJECT_DIR / f"{case_stem}.net"
            case_raw = PROJECT_DIR / f"{case_stem}.raw"

            print(f"\nRunning {case_tag}")

            patched_bytes = patch_param_line_preserve_schematic(original_template, rg_on, rg_off)
            case_asc.write_bytes(patched_bytes)

            delete_case_outputs(case_stem)

            start_time = time.time()
            completed = run_ltspice_case(case_asc)
            log_text = wait_for_complete_log(case_log, start_time)

            out_log = OUTPUT_DIR / f"{case_stem}.log"
            out_net = OUTPUT_DIR / f"{case_stem}.net"
            out_raw = OUTPUT_DIR / f"{case_stem}.raw"
            shutil.copy2(case_log, out_log)
            if case_net.exists():
                shutil.copy2(case_net, out_net)
            if case_raw.exists():
                shutil.copy2(case_raw, out_raw)

            failed = any(s in log_text for s in [
                "Simulation Failed",
                "Iteration limit reached",
                "Fatal Error",
                "No such parameter defined",
                "Unknown subcircuit",
                "Can't find definition of model",
            ])

            pin = parse_measure(log_text, "pin")
            iin_rms = parse_measure(log_text, "iin_rms")
            iin_pk = parse_measure(log_text, "iin_pk")
            vout_avg = parse_measure(log_text, "vout_avg")
            vout_pp = parse_measure(log_text, "vout_pp")
            elapsed_s = parse_elapsed(log_text)
            stable = classify_stability(pin, iin_rms, failed)

            print(f"Elapsed={elapsed_s:.3f}s, Pin={pin:.3f} W, Iin_rms={iin_rms:.3f} A, Iin_pk={iin_pk:.3f} A, Stable={stable}")

            if not failed and pin < 1000 and iin_rms < 1:
                print("WARNING: Batch run completed but converter is not switching / not drawing real input current.")
                print(f"Diagnostic files copied to: {out_log} and {out_net}")

            results.append({
                "Rg_on": rg_on,
                "Rg_off": rg_off,
                "Pin_W": pin,
                "Iin_rms_A": iin_rms,
                "Iin_pk_A": iin_pk,
                "Vout_avg_V": vout_avg,
                "Vout_pp_V": vout_pp,
                "Elapsed_s": elapsed_s,
                "Stable": stable,
                "Failed": failed,
                "LTspice_returncode": completed.returncode,
                "Log_File": str(out_log),
                "Net_File": str(out_net) if out_net.exists() else "",
            })

    df = pd.DataFrame(results)
    xlsx_path = OUTPUT_DIR / "gan_pfc_dataset.xlsx"
    csv_path = OUTPUT_DIR / "gan_pfc_dataset.csv"
    df.to_excel(xlsx_path, index=False)
    df.to_csv(csv_path, index=False)

    print("\nFinal dataset:")
    print(df)
    print("\nStable count:", int(df["Stable"].sum()))
    print("Failed count:", int(df["Failed"].sum()))
    print(f"Saved: {xlsx_path}")
    print(f"Saved: {csv_path}")

    plt.figure(figsize=(7, 5))
    colors = df["Stable"].map({1: "green", 0: "red"})
    plt.scatter(df["Rg_on"], df["Rg_off"], c=colors, s=90, edgecolors="black")
    plt.xlabel("Rg_on (ohm)")
    plt.ylabel("Rg_off (ohm)")
    plt.title("LTspice GaN PFC Stability Map")
    plt.grid(True)
    plt.savefig(OUTPUT_DIR / "stability_map.png", dpi=300, bbox_inches="tight")
    plt.close()

    if SKLEARN_OK and df["Stable"].nunique() >= 2:
        X = df[["Rg_on", "Rg_off"]]
        y = df["Stable"]
        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X, y)

        x_grid = np.linspace(df["Rg_on"].min(), df["Rg_on"].max(), 150)
        y_grid = np.linspace(df["Rg_off"].min(), df["Rg_off"].max(), 150)
        xx, yy = np.meshgrid(x_grid, y_grid)
        grid = pd.DataFrame({"Rg_on": xx.ravel(), "Rg_off": yy.ravel()})
        zz = model.predict(grid).reshape(xx.shape)

        plt.figure(figsize=(7, 5))
        plt.contourf(xx, yy, zz, alpha=0.35)
        plt.scatter(df["Rg_on"], df["Rg_off"], c=colors, s=90, edgecolors="black")
        plt.xlabel("Rg_on (ohm)")
        plt.ylabel("Rg_off (ohm)")
        plt.title("AI-Predicted Stable Gate-Drive Region")
        plt.grid(True)
        plt.savefig(OUTPUT_DIR / "ai_stability_boundary.png", dpi=300, bbox_inches="tight")
        plt.close()
    else:
        print("Skipping ML boundary plot: need sklearn and at least two stability classes.")

    print("\nSaved plots.")


if __name__ == "__main__":
    main()
