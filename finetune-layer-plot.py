import numpy as np
import matplotlib.pyplot as plt

# From the "where to fine-tune" ablation (ResNet-50, EigenGrad-CAM, CE objective)
labels = ["Head", "L4", "L3–L4", "All"]
FMR = np.array([0.135, 0.349, 0.355, 0.360])
BL  = np.array([0.004, 0.012, 0.026, 0.045])

x = np.arange(len(labels))

fig = plt.figure(figsize=(7,5))
ax1 = plt.gca()
ax2 = ax1.twinx()

ax1.plot(x, FMR, marker='o')
ax2.plot(x, BL, marker='s', linestyle='--')

ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.set_xlabel("Trainable depth")
ax1.set_ylabel("Finder mass (FMR) ↑")
ax2.set_ylabel("Background leakage (BL) ↓")

plt.title("Fine-tuning depth: structure vs leakage (EigenGrad-CAM)")
for i in range(len(x)):
    ax1.annotate(f"{FMR[i]:.3f}", (x[i], FMR[i]), xytext=(0,6), textcoords="offset points", ha="center", fontsize=9)
    ax2.annotate(f"{BL[i]:.3f}",  (x[i], BL[i]),  xytext=(0,-14), textcoords="offset points", ha="center", fontsize=9)

fig.tight_layout()
plt.show()
fig.savefig("finetune_layer_plot.png", dpi=300, bbox_inches="tight")

# assumes you already have: labels, FMR, BL, x, fig/axes from your current plot

import numpy as np
import matplotlib.pyplot as plt

labels = ["Head", "L4", "L3–L4", "All"]
FMR = np.array([0.135, 0.349, 0.355, 0.360])
BL  = np.array([0.004, 0.012, 0.026, 0.045])
pct_params = ["<0.1%", "≈40%", "≈75%", "100%"]

x = np.arange(len(labels))
fig = plt.figure(figsize=(7,5))
ax1 = plt.gca()
ax2 = ax1.twinx()

# plot the two series
l1, = ax1.plot(x, FMR, marker='o', label="FMR")
l2, = ax2.plot(x, BL, marker='s', linestyle='--', label="BL")

# invert BL axis so "up = better"
ax2.invert_yaxis()

# sync axis label/tick colors with their lines (uses whatever default colors matplotlib chose)
ax1.set_ylabel("Finder mass (FMR) ↑", color=l1.get_color())
ax1.tick_params(axis='y', colors=l1.get_color())
ax2.set_ylabel("Background leakage (BL) ↓", color=l2.get_color())
ax2.tick_params(axis='y', colors=l2.get_color())

# x-ticks + params on a second line
ax1.set_xticks(x)
ax1.set_xticklabels([f"{lab}\n{pp}" for lab, pp in zip(labels, pct_params)])

# tighter y-ranges with small padding
ax1.set_ylim(0.12, 0.37)
ax2.set_ylim(0.00, 0.05)  # note: inverted above

# annotate points, nudged to avoid overlap
for i in range(len(x)):
    ax1.annotate(f"{FMR[i]:.3f}", (x[i], FMR[i]), xytext=(0, 8), textcoords="offset points",
                 ha="center", fontsize=9)
    ax2.annotate(f"{BL[i]:.3f}",  (x[i], BL[i]),  xytext=(0, -14), textcoords="offset points",
                 ha="center", fontsize=9)

# vertical guide at the sweet spot (L4)
ax1.axvline(1, linestyle=':', linewidth=1)
ax1.text(1, ax1.get_ylim()[1], "*", ha="center", va="bottom", fontsize=9)

# simple legend
ax1.legend(handles=[l1, l2], loc="upper left")

plt.title("Fine-tuning depth: structure vs leakage (EigenGrad-CAM)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
fig.savefig("finetune_layer_plot_with_params.png", dpi=600, bbox_inches="tight")