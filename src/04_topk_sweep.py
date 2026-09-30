"""
Sweep TOP_K using the saved per-window ranks (no retraining needed).
Also plots the LSTM's best F1 at each K against the Isolation Forest baseline.
"""
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score

with open("data/processed/lstm_ranks.pkl", "rb") as f:
    d = pickle.load(f)
block_ranks = d["block_ranks"]
test_ids = d["test_ids"]
y_test = d["y_test"]

K_VALUES = [1, 2, 3, 5, 7, 9, 12, 15, 20, 30]

rows = []
for K in K_VALUES:
    scores = np.array([
        (block_ranks[bid] >= K).mean() if len(block_ranks[bid]) > 0 else 0.0
        for bid in test_ids
    ])
    precisions, recalls, thresholds = precision_recall_curve(y_test, scores)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1s)
    rows.append({
        "K": K,
        "best_f1": f1s[best_idx],
        "precision": precisions[best_idx],
        "recall": recalls[best_idx],
    })
    print(f"K={K:>2}: best F1={f1s[best_idx]:.4f} "
          f"(precision={precisions[best_idx]:.4f}, recall={recalls[best_idx]:.4f})")

baseline_f1 = 0.6525
baseline_precision = 0.7473
baseline_recall = 0.5791

best_row = max(rows, key=lambda r: r["best_f1"])
print(f"\nBest LSTM setting: K={best_row['K']} with F1={best_row['best_f1']:.4f}")
print(f"Isolation Forest baseline: F1={baseline_f1:.4f}")

if best_row["best_f1"] > baseline_f1:
    print("-> LSTM beats the baseline at this K.")
else:
    print("-> Isolation Forest baseline is still stronger on F1, "
          "but check the precision/recall trade-off below.")

Ks = [r["K"] for r in rows]
f1s_plot = [r["best_f1"] for r in rows]
precisions_plot = [r["precision"] for r in rows]
recalls_plot = [r["recall"] for r in rows]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(Ks, f1s_plot, marker="o", label="LSTM F1")
ax.plot(Ks, precisions_plot, marker="s", linestyle="--", label="LSTM Precision")
ax.plot(Ks, recalls_plot, marker="^", linestyle="--", label="LSTM Recall")
ax.axhline(baseline_f1, color="gray", linestyle=":", label=f"Isolation Forest F1 ({baseline_f1:.3f})")
ax.set_xlabel("TOP_K (candidates considered 'expected')")
ax.set_ylabel("Score")
ax.set_title("LSTM performance vs TOP_K, compared to event-count baseline")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("reports/topk_sweep.png", dpi=150)
print("\nSaved plot to reports/topk_sweep.png")