import os
import re
import shutil
import subprocess
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

LTSPICE_EXE = r"C:\Users\vashi\AppData\Local\Programs\ADI\LTspice\LTspice.exe"

PROJECT_DIR = r"C:\Users\vashi\Downloads\a-010_ac-dc_totem-pole_bridgeless_pfc_vin=200v_iin=100a_synchronous_fets\a-010_ac-dc_totem-pole_bridgeless_pfc_vin=200v_iin=100a_synchronous_fets"

OUTPUT_DIR = r"C:\Users\vashi\bmw_gan"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TEMPLATE_ASC = os.path.join(PROJECT_DIR, "gan_pfc_template.asc")
BACKUP_ASC = os.path.join(PROJECT_DIR, "gan_pfc_template_BACKUP.asc")

TEMPLATE_LOG = os.path.join(PROJECT_DIR, "gan_pfc_template.log")
TEMPLATE_NET = os.path.join(PROJECT_DIR, "gan_pfc_template.net")
TEMPLATE_RAW = os.path.join(PROJECT_DIR, "gan_pfc_template.raw")

rg_on_values = [2]
rg_off_values = [1]

def backup_template():
    if not os.path.exists(BACKUP_ASC):
        shutil.copy2(TEMPLATE_ASC, BACKUP_ASC)

def restore_template():
    shutil.copy2(BACKUP_ASC, TEMPLATE_ASC)

def read_text(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def write_text(path, text):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)

def patch_param_line(text, rg_on, rg_off):
    lines = text.splitlines()
    new_lines = []
    patched = False

    for line in lines:
        # LTspice schematic directive line usually looks like:
        # TEXT x y Left 2 !.param Vin=200 Vo=400 ... Rg_on=2 Rg_off=1
        if "!.param" in line and ("Rg_on" in line or "Rg_off" in line):
            line = re.sub(r"Rg_on\s*=\s*[^\s]+", f"Rg_on={rg_on}", line, flags=re.IGNORECASE)
            line = re.sub(r"Rg_off\s*=\s*[^\s]+", f"Rg_off={rg_off}", line, flags=re.IGNORECASE)

            if "Rg_on" not in line:
                line += f" Rg_on={rg_on}"
            if "Rg_off" not in line:
                line += f" Rg_off={rg_off}"

            patched = True

        new_lines.append(line)

    if not patched:
        # Add a new LTspice SPICE directive safely
        new_lines.append(f"TEXT 64 64 Left 2 !.param Rg_on={rg_on} Rg_off={rg_off}")
        patched = True

    text2 = "\n".join(new_lines) + "\n"

    # Hard check
    if f"Rg_on={rg_on}" not in text2 or f"Rg_off={rg_off}" not in text2:
        raise RuntimeError("Param replacement failed after patching.")

    return text2

def delete_old_outputs():
    for p in [TEMPLATE_LOG, TEMPLATE_NET, TEMPLATE_RAW]:
        if os.path.exists(p):
            try:
                os.remove(p)
            except PermissionError:
                time.sleep(1)
                os.remove(p)

def run_ltspice():
    completed = subprocess.run(
        [LTSPICE_EXE, "-b", TEMPLATE_ASC],
        cwd=PROJECT_DIR,
        timeout=300,
        capture_output=True,
        text=True
    )
    return completed

def parse_measure(log_text, name):
    for line in log_text.splitlines():
        line_clean = line.strip()
        if line_clean.lower().startswith(name.lower() + ":"):
            try:
                return float(line_clean.split("=")[1].split()[0])
            except Exception:
                return np.nan
    return np.nan

def classify_stability(pin, iin_rms, failed):
    if failed:
        return 0
    if np.isnan(pin) or np.isnan(iin_rms):
        return 0
    if pin <= 1000:
        return 0
    if iin_rms <= 1:
        return 0
    if iin_rms > 100:
        return 0
    return 1

backup_template()
base_text = read_text(BACKUP_ASC)

results = []

try:
    for rg_on in rg_on_values:
        for rg_off in rg_off_values:
            print(f"\nRunning Rg_on={rg_on}, Rg_off={rg_off}")

            patched = patch_param_line(base_text, rg_on, rg_off)
            for line in patched.splitlines():
                if "Rg_on" in line or "Rg_off" in line:
                    print("PARAM DEBUG:", line)
            write_text(TEMPLATE_ASC, patched)

            delete_old_outputs()
            completed = run_ltspice()

            if os.path.exists(TEMPLATE_LOG):
                log_text = read_text(TEMPLATE_LOG)
            else:
                log_text = completed.stdout + "\n" + completed.stderr

            case_tag = f"Rgon_{rg_on}_Rgoff_{rg_off}".replace(".", "p")
            shutil.copy2(
                TEMPLATE_LOG,
                os.path.join(OUTPUT_DIR, f"case_{case_tag}.log")
            )

            failed = (
                "Simulation Failed" in log_text
                or "Iteration limit reached" in log_text
                or "Fatal Error" in log_text
                or "No such parameter defined" in log_text
            )

            pin = parse_measure(log_text, "pin")
            iin_rms = parse_measure(log_text, "iin_rms")
            iin_pk = parse_measure(log_text, "iin_pk")
            vout_avg = parse_measure(log_text, "vout_avg")
            vout_pp = parse_measure(log_text, "vout_pp")

            stable = classify_stability(pin, iin_rms, failed)

            print(f"Pin={pin:.3f}, Iin_rms={iin_rms:.3f}, Iin_pk={iin_pk:.3f}, Stable={stable}")

            results.append({
                "Rg_on": rg_on,
                "Rg_off": rg_off,
                "Pin_W": pin,
                "Iin_rms_A": iin_rms,
                "Iin_pk_A": iin_pk,
                "Vout_avg_V": vout_avg,
                "Vout_pp_V": vout_pp,
                "Stable": stable,
                "Failed": failed
            })

finally:
    restore_template()

df = pd.DataFrame(results)

xlsx_path = os.path.join(OUTPUT_DIR, "gan_pfc_dataset.xlsx")
csv_path = os.path.join(OUTPUT_DIR, "gan_pfc_dataset.csv")

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
plt.savefig(os.path.join(OUTPUT_DIR, "stability_map.png"), dpi=300, bbox_inches="tight")
plt.close()

if df["Stable"].nunique() >= 2:
    X = df[["Rg_on", "Rg_off"]]
    y = df["Stable"]

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)

    x_grid = np.linspace(df["Rg_on"].min(), df["Rg_on"].max(), 150)
    y_grid = np.linspace(df["Rg_off"].min(), df["Rg_off"].max(), 150)
    xx, yy = np.meshgrid(x_grid, y_grid)

    grid = pd.DataFrame({
        "Rg_on": xx.ravel(),
        "Rg_off": yy.ravel()
    })

    zz = model.predict(grid).reshape(xx.shape)

    plt.figure(figsize=(7, 5))
    plt.contourf(xx, yy, zz, alpha=0.35)
    plt.scatter(df["Rg_on"], df["Rg_off"], c=colors, s=90, edgecolors="black")
    plt.xlabel("Rg_on (ohm)")
    plt.ylabel("Rg_off (ohm)")
    plt.title("AI-Predicted Stable Gate-Drive Region")
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "ai_stability_boundary.png"), dpi=300, bbox_inches="tight")
    plt.close()
else:
    print("Only one class found. Skipping ML boundary plot.")

print("\nSaved plots.")