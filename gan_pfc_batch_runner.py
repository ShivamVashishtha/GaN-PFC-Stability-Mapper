import os
import re
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

LTSPICE_EXE = r"C:\Users\vashi\AppData\Local\Programs\ADI\LTspice\LTspice.exe"

PROJECT_DIR = r"C:\Users\vashi\Downloads\a-010_ac-dc_totem-pole_bridgeless_pfc_vin=200v_iin=100a_synchronous_fets\a-010_ac-dc_totem-pole_bridgeless_pfc_vin=200v_iin=100a_synchronous_fets"

TEMPLATE_ASC = PROJECT_DIR + r"\gan_pfc_template.asc"

OUTPUT_DIR = r"C:\Users\vashi\bmw_gan"
os.makedirs(OUTPUT_DIR, exist_ok=True)

rg_on_values = [2]
rg_off_values = [1]

results = []

def replace_param(text, name, value):
    pattern = rf"({name}\s*=\s*)[^\s]+"
    replacement = rf"\g<1>{value}"
    return re.sub(pattern, replacement, text, flags=re.IGNORECASE)

def parse_measure(log_text, name):
    # LTspice batch mode outputs measurements as: <name>: <description>=<value> FROM...
    # Example: pin: AVG(V(IN)*I(V1))=-5.88568780387 FROM 0.01 TO 0.02
    pattern = rf"{name}:\s+.*?=([\-0-9.eE+]+)\s+FROM"
    match = re.search(pattern, log_text, flags=re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return np.nan
    return np.nan

with open(TEMPLATE_ASC, "r", encoding="utf-8", errors="ignore") as f:
    template = f.read()

for rg_on in rg_on_values:
    for rg_off in rg_off_values:
        case_name = f"case_Rgon_{rg_on}_Rgoff_{rg_off}".replace(".", "p")
        asc_path = os.path.join(PROJECT_DIR, case_name + ".asc")
        log_path = os.path.join(PROJECT_DIR, case_name + ".log")

        text = template
        text = replace_param(text, "Rg_on", rg_on)
        text = replace_param(text, "Rg_off", rg_off)

        with open(asc_path, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"Running Rg_on={rg_on}, Rg_off={rg_off}")

        try:
            completed = subprocess.run(
                [LTSPICE_EXE, "-b", asc_path],
                cwd=PROJECT_DIR,
                timeout=240,
                capture_output=True,
                text=True
            )

            if os.path.exists(log_path):
                with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                    log_text = f.read()
            else:
                log_text = completed.stdout + completed.stderr

            # Debug: print first 500 chars of log to inspect format
            print(f"  Log preview: {log_text[:500]}")

            failed = "Simulation Failed" in log_text or "Iteration limit reached" in log_text

            pin = parse_measure(log_text, "pin")
            iin_rms = parse_measure(log_text, "iin_rms")
            iin_pk = parse_measure(log_text, "iin_pk")
            vout_avg = parse_measure(log_text, "vout_avg")
            vout_pp = parse_measure(log_text, "vout_pp")
            
            print(f"  Parsed: pin={pin}, iin_rms={iin_rms}, vout_avg={vout_avg}")

            stable = int(
                (not failed)
                and not np.isnan(pin)
                and not np.isnan(iin_rms)
                and pin > 0
                and iin_rms < 100
            )

        except subprocess.TimeoutExpired:
            pin = np.nan
            iin_rms = np.nan
            iin_pk = np.nan
            vout_avg = np.nan
            vout_pp = np.nan
            stable = 0
            failed = True

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

df = pd.DataFrame(results)
df.to_excel(os.path.join(OUTPUT_DIR, "gan_pfc_dataset.xlsx"), index=False)
df.to_csv(os.path.join(OUTPUT_DIR, "gan_pfc_dataset.csv"), index=False)
print(df[["Rg_on", "Rg_off", "Pin_W", "Iin_rms_A", "Stable", "Failed"]])
print("Stable count:", df["Stable"].sum())
print("Failed count:", df["Failed"].sum())

print("\nSaved gan_pfc_dataset.xlsx")
print(df)

# Plot measured stability map
plt.figure(figsize=(7, 5))
colors = df["Stable"].map({1: "green", 0: "red"})
plt.scatter(df["Rg_on"], df["Rg_off"], c=colors, s=90, edgecolors="black")
plt.xlabel("Rg_on (ohm)")
plt.ylabel("Rg_off (ohm)")
plt.title("LTspice GaN PFC Stability Map")
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "stability_map.png"), dpi=300, bbox_inches="tight")

# Train ML classifier
X = df[["Rg_on", "Rg_off"]]
y = df["Stable"]

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y)

x_grid = np.linspace(min(rg_on_values), max(rg_on_values), 150)
y_grid = np.linspace(min(rg_off_values), max(rg_off_values), 150)
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

print("\nSaved plots:")
print("stability_map.png")
print("ai_stability_boundary.png")