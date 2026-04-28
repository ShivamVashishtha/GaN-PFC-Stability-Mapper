import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# =========================
# 1. PARSE LTSPICE LOG
# =========================

log_file = "ltspice_log.txt"

with open(log_file, "r") as f:
    text = f.read()

def extract_measure(name):
    pattern = rf"Measurement: {name}.*?\n((?:.*\n)+?)\n"
    match = re.search(pattern, text)
    if not match:
        return None
    lines = match.group(1).strip().split("\n")[1:]
    values = []
    for line in lines:
        parts = line.split()
        try:
            values.append(float(parts[1]))
        except:
            values.append(np.nan)
    return values

pin = extract_measure("pin")
iin_rms = extract_measure("iin_rms")
iin_pk = extract_measure("iin_pk")
vout = extract_measure("vout_avg")

# =========================
# 2. EXTRACT PARAMS
# =========================

param_lines = re.findall(r"\.step rg_off=([0-9\.]+) rg_on=([0-9\.]+)", text)

rg_off = [float(x[0]) for x in param_lines]
rg_on  = [float(x[1]) for x in param_lines]

# Align lengths
n = min(len(pin), len(rg_on), len(rg_off))

df = pd.DataFrame({
    "Rg_on": rg_on[:n],
    "Rg_off": rg_off[:n],
    "Pin": pin[:n],
    "Iin_rms": iin_rms[:n],
    "Iin_pk": iin_pk[:n],
    "Vout": vout[:n]
})

# =========================
# 3. STABILITY LABEL
# =========================

def label(row):
    if pd.isna(row["Iin_rms"]) or pd.isna(row["Pin"]):
        return 0
    if row["Iin_rms"] > 100:
        return 0
    if row["Pin"] < 0:
        return 0
    return 1

df["Stable"] = df.apply(label, axis=1)

# =========================
# 4. SAVE TO EXCEL
# =========================

df.to_excel("gan_pfc_dataset.xlsx", index=False)

print("Dataset saved: gan_pfc_dataset.xlsx")

# =========================
# 5. TRAIN ML MODEL
# =========================

X = df[["Rg_on", "Rg_off"]]
y = df["Stable"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("\nModel Performance:\n")
print(classification_report(y_test, pred))

# =========================
# 6. STABILITY MAP PLOT
# =========================

plt.figure(figsize=(6,5))

colors = df["Stable"].map({1:"green", 0:"red"})

plt.scatter(df["Rg_on"], df["Rg_off"], c=colors)

plt.xlabel("Rg_on (Ω)")
plt.ylabel("Rg_off (Ω)")
plt.title("GaN PFC Stability Map")
plt.grid(True)

plt.savefig("stability_map.png", dpi=300)
plt.show()

# =========================
# 7. DECISION BOUNDARY (OPTIONAL)
# =========================

x_range = np.linspace(df["Rg_on"].min(), df["Rg_on"].max(), 100)
y_range = np.linspace(df["Rg_off"].min(), df["Rg_off"].max(), 100)

xx, yy = np.meshgrid(x_range, y_range)

grid = np.c_[xx.ravel(), yy.ravel()]
zz = model.predict(grid).reshape(xx.shape)

plt.figure(figsize=(6,5))
plt.contourf(xx, yy, zz, alpha=0.3)

plt.scatter(df["Rg_on"], df["Rg_off"], c=colors)

plt.xlabel("Rg_on (Ω)")
plt.ylabel("Rg_off (Ω)")
plt.title("AI-Predicted Stability Region")
plt.grid(True)

plt.savefig("ai_boundary.png", dpi=300)
plt.show()