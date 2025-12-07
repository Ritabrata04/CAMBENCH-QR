import numpy as np
import matplotlib.pyplot as plt

# --- Data from your core results (ResNet-50) ---
# BL, FMR, TMR, DtS, ms/img
data = {
    # ZS
    ("ZS","LayerCAM"):      dict(BL=0.011, FMR=0.161, TMR=0.067, DtS=0.180, ms=4.1),
    ("ZS","EigenGrad-CAM"): dict(BL=0.004, FMR=0.135, TMR=0.078, DtS=0.170, ms=4.5),
    ("ZS","XGrad-CAM"):     dict(BL=0.061, FMR=0.265, TMR=0.037, DtS=0.210, ms=3.8),
    # FT-Struct
    ("FT-Struct","LayerCAM"):      dict(BL=0.019, FMR=0.313, TMR=0.036, DtS=0.150, ms=4.1),
    ("FT-Struct","EigenGrad-CAM"): dict(BL=0.012, FMR=0.349, TMR=0.037, DtS=0.140, ms=4.5),
    ("FT-Struct","XGrad-CAM"):     dict(BL=0.055, FMR=0.281, TMR=0.033, DtS=0.180, ms=3.8),
    # FT-LeakMin
    ("FT-LeakMin","LayerCAM"):      dict(BL=0.007, FMR=0.176, TMR=0.061, DtS=0.120, ms=4.1),
    ("FT-LeakMin","EigenGrad-CAM"): dict(BL=0.002, FMR=0.155, TMR=0.070, DtS=0.110, ms=4.5),
    ("FT-LeakMin","XGrad-CAM"):     dict(BL=0.039, FMR=0.229, TMR=0.047, DtS=0.160, ms=3.8),
}

# --- One-figure plot; annotate ZS, FT-Struct, and FT-LeakMin explicitly ---
method_marker = {"LayerCAM":"o", "EigenGrad-CAM":"s", "XGrad-CAM":"^"}

def proxy_score(d):
    return d["FMR"] + d["TMR"] - 3*d["BL"] - d["DtS"]

regime_label = {
    "ZS": "ZS",
    "FT-Struct": "FT-Struct",
    "FT-LeakMin": "FT-LeakMin",   # exact casing per your request
}

plt.figure(figsize=(7,5))

for (regime, method), d in data.items():
    x = d["ms"]
    y = proxy_score(d)
    plt.scatter(x, y, marker=method_marker[method])
    # annotate every point by regime name (no "FT-XGrad" etc.)
    plt.annotate(regime_label[regime], (x, y), xytext=(6, 6),
                 textcoords="offset points", fontsize=9)

plt.xlabel("Latency (ms / image) ↓")
plt.ylabel("StructureScore = FMR + TMR − 3·BL − DtS ↑")
plt.title("Structure–Speed Pareto (ResNet-50)")

# Legend: methods only (marker encodes CAM)
handles, labels = [], []
for m, mk in method_marker.items():
    handles.append(plt.Line2D([], [], marker=mk, linestyle='None'))
    labels.append(m)
plt.legend(handles, labels, title="Method", loc="best")

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("structure_speed_pareto_resnet50.png", dpi=600, bbox_inches="tight")
plt.close()
