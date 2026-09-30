"""
DeepLog-style LSTM anomaly detector.

Trains an LSTM to predict the next log-event template given a window of
previous templates, using ONLY normal training sequences. At test time,
a block is flagged anomalous if too many of its actual next-events fall
outside the model's top-K predicted candidates.
"""
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve

torch.manual_seed(42)
np.random.seed(42)

WINDOW = 10          # how many previous events the model sees
TOP_K = 9            # a prediction "counts" if truth is in the top-K candidates
MAX_TRAIN_WINDOWS = 400_000   # cap for speed on a laptop CPU
EPOCHS = 5
BATCH_SIZE = 512

# --- Load data ---
with open("data/processed/block_sequences.json") as f:
    sequences = json.load(f)
with open("data/processed/templates.json") as f:
    templates = json.load(f)

labels = pd.read_csv("data/raw/preprocessed/anomaly_label.csv")
labels["y"] = (labels["Label"] == "Anomaly").astype(int)
label_map = dict(zip(labels["BlockId"], labels["y"]))

template_id_to_idx = {int(tid): i for i, tid in enumerate(sorted(int(t) for t in templates.keys()))}
n_events = len(template_id_to_idx)
PAD = n_events  # padding token id
vocab_size = n_events + 1

block_ids = [b for b in sequences if b in label_map]
y_all = np.array([label_map[b] for b in block_ids])

train_ids, test_ids = train_test_split(
    block_ids, test_size=0.3, random_state=42, stratify=y_all
)
train_ids_set = set(train_ids)
print(f"Train blocks: {len(train_ids):,}  Test blocks: {len(test_ids):,}")

def encode(seq):
    return [template_id_to_idx[int(t)] for t in seq]

# --- Build training windows: ONLY from normal blocks in the train split ---
train_windows_x = []
train_windows_y = []
for bid in train_ids:
    if label_map[bid] != 0:
        continue  # skip anomalous blocks for training the "normal" model
    seq = encode(sequences[bid])
    padded = [PAD] * WINDOW + seq
    for i in range(len(seq)):
        train_windows_x.append(padded[i:i + WINDOW])
        train_windows_y.append(padded[i + WINDOW])

print(f"Total available training windows: {len(train_windows_x):,}")
if len(train_windows_x) > MAX_TRAIN_WINDOWS:
    idx = np.random.choice(len(train_windows_x), MAX_TRAIN_WINDOWS, replace=False)
    train_windows_x = [train_windows_x[i] for i in idx]
    train_windows_y = [train_windows_y[i] for i in idx]
print(f"Training on: {len(train_windows_x):,} windows")

X_train = torch.tensor(train_windows_x, dtype=torch.long)
y_train = torch.tensor(train_windows_y, dtype=torch.long)
loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)

# --- Model ---
class DeepLogLSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim=16, hidden_dim=64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        e = self.emb(x)
        _, (h, _) = self.lstm(e)
        return self.fc(h[-1])

device = "cuda" if torch.cuda.is_available() else "cpu"
model = DeepLogLSTM(vocab_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print(f"\nTraining on device: {device}")
start = time.time()
for epoch in range(EPOCHS):
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    avg_loss = total_loss / len(X_train)
    print(f"  Epoch {epoch+1}/{EPOCHS}: loss={avg_loss:.4f} ({time.time()-start:.0f}s elapsed)")

print(f"Training done in {time.time()-start:.0f}s")

# --- Evaluate: for every block in test set, count "violations" ---
model.eval()

def event_ranks(seq_idx):
    padded = [PAD] * WINDOW + seq_idx
    windows = []
    targets = []
    for i in range(len(seq_idx)):
        windows.append(padded[i:i + WINDOW])
        targets.append(padded[i + WINDOW])
    if not windows:
        return np.array([], dtype=np.int16)
    xb = torch.tensor(windows, dtype=torch.long).to(device)
    with torch.no_grad():
        logits = model(xb)
        order = torch.argsort(logits, dim=1, descending=True).cpu().numpy()
    targets = np.array(targets)
    # rank[i] = position of the true event in the model's ranked predictions (0 = top guess)
    ranks = np.array([np.where(order[i] == targets[i])[0][0] for i in range(len(targets))],
                     dtype=np.int16)
    return ranks

print("\nScoring test blocks (this may take a few minutes)...")
block_ranks = {}
y_test = []
start = time.time()
for i, bid in enumerate(test_ids):
    seq_idx = encode(sequences[bid])
    ranks = event_ranks(seq_idx)
    block_ranks[bid] = ranks
    y_test.append(label_map[bid])
    if (i + 1) % 20000 == 0:
        print(f"  {i+1:,}/{len(test_ids):,} blocks scored ({time.time()-start:.0f}s)")

y_test = np.array(y_test)
print(f"Scoring done in {time.time()-start:.0f}s")

# Default TOP_K report, same as before, for continuity
scores = np.array([
    (r >= TOP_K).mean() if len(r) > 0 else 0.0 for r in block_ranks.values()
])

# --- Pick threshold via best-F1 on the violation rate, then report ---
precisions, recalls, thresholds = precision_recall_curve(y_test, scores)
f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
best_idx = np.argmax(f1s)
best_thresh = thresholds[min(best_idx, len(thresholds) - 1)]

pred = (scores >= best_thresh).astype(int)

print(f"\n=== LSTM (DeepLog-style) ===")
print(f"Best threshold (violation rate): {best_thresh:.4f}")
print(classification_report(y_test, pred, digits=4))
print("Confusion matrix [[TN FP] [FN TP]]:")
print(confusion_matrix(y_test, pred))
print(f"\nBest achievable F1: {f1s[best_idx]:.4f} "
      f"(precision={precisions[best_idx]:.4f}, recall={recalls[best_idx]:.4f})")
np.savez("data/processed/lstm_results.npz", scores=scores, y_test=y_test)
torch.save(model.state_dict(), "data/processed/lstm_model.pt")

# Save per-block ranks for later TOP_K sweeping (as an object array)
import pickle
with open("data/processed/lstm_ranks.pkl", "wb") as f:
    pickle.dump({"block_ranks": block_ranks, "test_ids": test_ids, "y_test": y_test}, f)

print("\nSaved scores to data/processed/lstm_results.npz")
print("Saved model to data/processed/lstm_model.pt")
print("Saved per-window ranks to data/processed/lstm_ranks.pkl")
