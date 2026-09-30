"""
Build event-count feature vectors per block, label them, and train an
Isolation Forest baseline for anomaly detection.
"""
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve

# --- Load sequences and labels ---
with open("data/processed/block_sequences.json") as f:
    sequences = json.load(f)

with open("data/processed/templates.json") as f:
    templates = json.load(f)

labels = pd.read_csv("data/raw/preprocessed/anomaly_label.csv")
labels["y"] = (labels["Label"] == "Anomaly").astype(int)
label_map = dict(zip(labels["BlockId"], labels["y"]))

n_templates = len(templates)
# Drain3 template IDs are not guaranteed to start at 0, so map them to
# contiguous array indices explicitly
template_id_to_idx = {int(tid): i for i, tid in enumerate(sorted(int(t) for t in templates.keys()))}
n_features = len(template_id_to_idx)
print(f"{n_templates} templates, {n_features} feature slots, "
      f"{len(sequences):,} blocks with sequences, {len(labels):,} blocks with labels")

# --- Build event-count feature matrix ---
block_ids = []
rows = []
missing_label = 0

for bid, seq in sequences.items():
    if bid not in label_map:
        missing_label += 1
        continue
    counts = np.zeros(n_features, dtype=np.int32)
    for t in seq:
        counts[template_id_to_idx[int(t)]] += 1
    block_ids.append(bid)
    rows.append(counts)

X = np.vstack(rows)
y = np.array([label_map[b] for b in block_ids])

print(f"Feature matrix: {X.shape}")
print(f"Blocks with no label found: {missing_label}")
print(f"Anomaly rate in matched blocks: {y.mean():.4f}")

# --- Train/test split (stratified, so both sets have anomalies) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"\nTrain: {X_train.shape}, anomaly rate {y_train.mean():.4f}")
print(f"Test:  {X_test.shape}, anomaly rate {y_test.mean():.4f}")

# --- Isolation Forest ---
# contamination = expected proportion of anomalies; we use the true training rate
model = IsolationForest(
    n_estimators=200,
    contamination=y_train.mean(),
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train)

# IsolationForest: -1 = anomaly, 1 = normal -> convert to 1 = anomaly, 0 = normal
raw_pred = model.predict(X_test)
pred = (raw_pred == -1).astype(int)

print("\n=== Isolation Forest baseline ===")
print(classification_report(y_test, pred, digits=4))
print("Confusion matrix [[TN FP] [FN TP]]:")
print(confusion_matrix(y_test, pred))

# --- Also report using the anomaly score directly, at a few thresholds ---
scores = -model.score_samples(X_test)  # higher = more anomalous
precisions, recalls, thresholds = precision_recall_curve(y_test, scores)
f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
best_idx = np.argmax(f1s)
print(f"\nBest achievable F1 by threshold tuning: {f1s[best_idx]:.4f} "
      f"(precision={precisions[best_idx]:.4f}, recall={recalls[best_idx]:.4f})")

# Save everything needed for later steps (LSTM, comparison)
np.savez("data/processed/baseline_data.npz",
         X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
print("\nSaved train/test arrays to data/processed/baseline_data.npz")