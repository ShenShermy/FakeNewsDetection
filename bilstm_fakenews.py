"""
=============================================================
CDS525 Group Project: Fake News Detection with BiLSTM
=============================================================
Structure:
  1. Data Preprocessing
  2. Dataset & DataLoader
  3. BiLSTM Model
  4. Training & Evaluation Functions
  5. Experiments (Loss / LR / Batch Size)
  6. All Visualizations
=============================================================
"""

# ── 0. Imports ────────────────────────────────────────────
import os, re, random, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ── Reproducibility ───────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# ── Device Selection ──────────────────────────────────────
# Options: "auto" | "cuda" | "cpu"
#   "auto" → use GPU if available, otherwise fall back to CPU
#   "cuda" → force GPU (will raise error if no GPU found)
#   "cpu"  → force CPU
DEVICE_MODE = "auto"

def get_device(mode: str) -> torch.device:
    mode = mode.strip().lower()
    if mode == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA GPU detected. Set DEVICE_MODE='cpu' or 'auto'.")
        device = torch.device("cuda")
    elif mode == "cpu":
        device = torch.device("cpu")
    else:
        raise ValueError(f"Unknown DEVICE_MODE '{mode}'. Choose from: 'auto', 'cuda', 'cpu'.")

    print(f"{'='*50}")
    print(f"  Device mode : {mode.upper()}")
    print(f"  Using       : {device}")
    if device.type == "cuda":
        print(f"  GPU Name    : {torch.cuda.get_device_name(0)}")
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU Memory  : {total:.1f} GB")
    print(f"{'='*50}")
    return device

DEVICE = get_device(DEVICE_MODE)

# =============================================================
# 1. DATA PREPROCESSING
# =============================================================

def clean_text(text: str) -> str:
    """Lowercase, remove punctuation and extra spaces."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)          # remove URLs
    text = re.sub(r"[^a-z\s]", " ", text)               # keep letters only
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_vocab(texts, max_vocab=20000):
    """Build word-to-index vocabulary from training texts."""
    counter = Counter()
    for text in texts:
        counter.update(text.split())
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in counter.most_common(max_vocab - 2):
        vocab[word] = len(vocab)
    return vocab


def encode(text, vocab, max_len=200):
    """Convert text to padded integer sequence."""
    tokens = text.split()[:max_len]
    ids = [vocab.get(t, 1) for t in tokens]             # 1 = <UNK>
    ids += [0] * (max_len - len(ids))                   # 0 = <PAD>
    return ids


# ── Load & split data ─────────────────────────────────────
def load_data(csv_path="fakenews.csv"):
    df = pd.read_csv(csv_path)
    df.dropna(inplace=True)
    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].str.strip() != ""]

    # Ensure binary labels 0/1
    unique_labels = sorted(df["label"].unique())
    label_map = {v: i for i, v in enumerate(unique_labels)}
    df["label"] = df["label"].map(label_map)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"].tolist(), df["label"].tolist(),
        test_size=0.2, random_state=SEED, stratify=df["label"]
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=0.1, random_state=SEED
    )
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# =============================================================
# 2. DATASET & DATALOADER
# =============================================================

class FakeNewsDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=200):
        self.X = [encode(t, vocab, max_len) for t in texts]
        self.y = labels

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.long),
            torch.tensor(self.y[idx],  dtype=torch.float)
        )


def make_loaders(X_train, X_val, X_test, y_train, y_val, y_test,
                 vocab, batch_size=32, max_len=200):
    train_ds = FakeNewsDataset(X_train, y_train, vocab, max_len)
    val_ds   = FakeNewsDataset(X_val,   y_val,   vocab, max_len)
    test_ds  = FakeNewsDataset(X_test,  y_test,  vocab, max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size)
    return train_loader, val_loader, test_loader


# =============================================================
# 3. BiLSTM MODEL
# =============================================================

class BiLSTMClassifier(nn.Module):
    """
    Embedding → BiLSTM → Attention → FC → Sigmoid
    """
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256,
                 num_layers=2, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.attention = nn.Linear(hidden_dim * 2, 1)   # attention scoring
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        emb = self.dropout(self.embedding(x))            # (B, L, E)
        out, _ = self.lstm(emb)                          # (B, L, 2H)

        # Attention pooling
        score = torch.softmax(self.attention(out), dim=1)  # (B, L, 1)
        context = (score * out).sum(dim=1)               # (B, 2H)

        logit = self.fc(self.dropout(context))           # (B, 1)
        return logit.squeeze(1)


# =============================================================
# 4. TRAINING & EVALUATION
# =============================================================

class FocalLoss(nn.Module):
    """Focal Loss – alternative to BCEWithLogitsLoss for imbalanced data."""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, preds_all, labels_all = 0, [], []
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(X)
        loss   = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * len(y)
        preds = (torch.sigmoid(logits) > 0.5).long().cpu().tolist()
        preds_all.extend(preds)
        labels_all.extend(y.long().cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(labels_all, preds_all)
    return avg_loss, acc


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, preds_all, labels_all = 0, [], []
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        logits = model(X)
        loss   = criterion(logits, y)

        total_loss += loss.item() * len(y)
        preds = (torch.sigmoid(logits) > 0.5).long().cpu().tolist()
        preds_all.extend(preds)
        labels_all.extend(y.long().cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(labels_all, preds_all)
    return avg_loss, acc, preds_all, labels_all


def run_experiment(X_train, X_val, X_test, y_train, y_val, y_test,
                   vocab, criterion, lr=1e-3, batch_size=32,
                   num_epochs=15, tag=""):
    """Train BiLSTM and return history dict."""
    train_loader, val_loader, test_loader = make_loaders(
        X_train, X_val, X_test, y_train, y_val, y_test, vocab, batch_size
    )
    model = BiLSTMClassifier(vocab_size=len(vocab)).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )

    history = {"train_loss": [], "train_acc": [], "test_acc": []}
    best_val_acc, best_state = 0, None

    for epoch in range(1, num_epochs + 1):
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        v_loss, v_acc, _, _ = evaluate(model, val_loader, criterion)
        _, te_acc, _, _     = evaluate(model, test_loader, criterion)

        scheduler.step(v_loss)
        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["test_acc"].append(te_acc)

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}

        print(f"[{tag}] Epoch {epoch:02d} | "
              f"Loss {t_loss:.4f} | TrainAcc {t_acc:.4f} | "
              f"ValAcc {v_acc:.4f} | TestAcc {te_acc:.4f}")

    # Load best model and get final predictions
    model.load_state_dict(best_state)
    _, _, final_preds, final_labels = evaluate(model, test_loader, criterion)
    history["final_preds"]  = final_preds
    history["final_labels"] = final_labels
    return history


# =============================================================
# 5. VISUALIZATION FUNCTIONS
# =============================================================

COLORS = ["#2196F3", "#E91E63", "#4CAF50", "#FF9800", "#9C27B0"]


def plot_single(history, title, save_path):
    """Figure with train loss, train acc, test acc vs epochs (Fig 1 or 2)."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, ax1 = plt.subplots(figsize=(9, 5))

    ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss", markersize=4)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", color="blue", fontsize=12)
    ax1.tick_params(axis="y", labelcolor="blue")

    ax2 = ax1.twinx()
    ax2.plot(epochs, history["train_acc"], "g-s", label="Train Acc", markersize=4)
    ax2.plot(epochs, history["test_acc"],  "r-^", label="Test Acc",  markersize=4)
    ax2.set_ylabel("Accuracy", color="black", fontsize=12)
    ax2.set_ylim(0, 1)

    lines1, lbls1 = ax1.get_legend_handles_labels()
    lines2, lbls2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lbls1 + lbls2, loc="center right", fontsize=10)

    plt.title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_multi(histories_dict, metric, ylabel, title, save_path, log_scale=False):
    """
    Overlay multiple curves (different LR or batch sizes).
    histories_dict = {label_str: history_dict}
    """
    epochs = range(1, len(next(iter(histories_dict.values()))["train_loss"]) + 1)
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (label, hist) in enumerate(histories_dict.items()):
        ax.plot(epochs, hist[metric],
                color=COLORS[i % len(COLORS)],
                marker="o", markersize=4, label=label)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    if log_scale:
        ax.set_yscale("log")
    if "acc" in metric:
        ax.set_ylim(0, 1)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_predictions_table(preds, labels, texts, save_path, n=100):
    """Visualize first n test predictions as a color-coded table (Fig 7)."""
    n = min(n, len(preds))
    results = []
    for i in range(n):
        snippet = " ".join(texts[i].split()[:12]) + "..."
        correct = "✓" if preds[i] == labels[i] else "✗"
        results.append([i + 1, snippet,
                         "REAL" if labels[i] == 1 else "FAKE",
                         "REAL" if preds[i]  == 1 else "FAKE",
                         correct])

    df = pd.DataFrame(results,
                      columns=["#", "Text Snippet", "True Label",
                               "Predicted", "Correct?"])

    fig, ax = plt.subplots(figsize=(18, max(6, n * 0.28)))
    ax.axis("off")
    tbl = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="left",
        loc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.auto_set_column_width(col=list(range(len(df.columns))))

    # Color header
    for j in range(len(df.columns)):
        tbl[(0, j)].set_facecolor("#1565C0")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")

    # Color rows: green = correct, red = wrong
    for i in range(1, n + 1):
        color = "#E8F5E9" if results[i-1][4] == "✓" else "#FFEBEE"
        for j in range(len(df.columns)):
            tbl[(i, j)].set_facecolor(color)

    plt.title(f"Prediction Results – First {n} Test Samples",
              fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# =============================================================
# 6. MAIN – RUN ALL EXPERIMENTS
# =============================================================

if __name__ == "__main__":

    # ── Load data ─────────────────────────────────────────
    CSV_PATH = "fakenews.csv"   # ← change if needed
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(CSV_PATH)
    vocab = build_vocab(X_train, max_vocab=20000)
    print(f"Vocab size: {len(vocab)}")

    EPOCHS = 15
    os.makedirs("figures", exist_ok=True)

    # ─────────────────────────────────────────────────────
    # Fig 1: BCE Loss (baseline)
    # ─────────────────────────────────────────────────────
    print("\n=== Experiment 1: BCEWithLogitsLoss (baseline) ===")
    bce_criterion = nn.BCEWithLogitsLoss()
    hist_bce = run_experiment(
        X_train, X_val, X_test, y_train, y_val, y_test,
        vocab, criterion=bce_criterion, lr=1e-3, batch_size=32,
        num_epochs=EPOCHS, tag="BCE"
    )
    plot_single(hist_bce,
                "Fig 1 – BiLSTM with BCEWithLogitsLoss (lr=0.001, bs=32)",
                "figures/fig1_bce_baseline.png")

    # ─────────────────────────────────────────────────────
    # Fig 2: Focal Loss (different loss function)
    # ─────────────────────────────────────────────────────
    print("\n=== Experiment 2: FocalLoss ===")
    focal_criterion = FocalLoss(alpha=0.25, gamma=2.0)
    hist_focal = run_experiment(
        X_train, X_val, X_test, y_train, y_val, y_test,
        vocab, criterion=focal_criterion, lr=1e-3, batch_size=32,
        num_epochs=EPOCHS, tag="Focal"
    )
    plot_single(hist_focal,
                "Fig 2 – BiLSTM with Focal Loss (lr=0.001, bs=32)",
                "figures/fig2_focal_loss.png")

    # ─────────────────────────────────────────────────────
    # Fig 3 & 4: Different Learning Rates
    # ─────────────────────────────────────────────────────
    print("\n=== Experiment 3: Different Learning Rates ===")
    LR_LIST = [0.1, 0.01, 0.001, 0.0001]
    lr_histories = {}
    for lr in LR_LIST:
        tag = f"lr={lr}"
        print(f"\n--- LR = {lr} ---")
        h = run_experiment(
            X_train, X_val, X_test, y_train, y_val, y_test,
            vocab, criterion=nn.BCEWithLogitsLoss(),
            lr=lr, batch_size=32, num_epochs=EPOCHS, tag=tag
        )
        lr_histories[tag] = h

    plot_multi(lr_histories, "train_loss", "Train Loss",
               "Fig 3 – Train Loss: Different Learning Rates",
               "figures/fig3_lr_train_loss.png", log_scale=True)
    plot_multi(lr_histories, "test_acc", "Test Accuracy",
               "Fig 4 – Test Accuracy: Different Learning Rates",
               "figures/fig4_lr_test_acc.png")

    # ─────────────────────────────────────────────────────
    # Fig 5 & 6: Different Batch Sizes
    # ─────────────────────────────────────────────────────
    print("\n=== Experiment 4: Different Batch Sizes ===")
    BS_LIST = [8, 16, 32, 64, 128]
    bs_histories = {}
    for bs in BS_LIST:
        tag = f"bs={bs}"
        print(f"\n--- Batch Size = {bs} ---")
        h = run_experiment(
            X_train, X_val, X_test, y_train, y_val, y_test,
            vocab, criterion=nn.BCEWithLogitsLoss(),
            lr=1e-3, batch_size=bs, num_epochs=EPOCHS, tag=tag
        )
        bs_histories[tag] = h

    plot_multi(bs_histories, "train_loss", "Train Loss",
               "Fig 5 – Train Loss: Different Batch Sizes",
               "figures/fig5_bs_train_loss.png")
    plot_multi(bs_histories, "test_acc", "Test Accuracy",
               "Fig 6 – Test Accuracy: Different Batch Sizes",
               "figures/fig6_bs_test_acc.png")

    # ─────────────────────────────────────────────────────
    # Fig 7: First 100 Predictions Table
    # ─────────────────────────────────────────────────────
    print("\n=== Generating Prediction Table ===")
    plot_predictions_table(
        hist_bce["final_preds"],
        hist_bce["final_labels"],
        X_test,
        "figures/fig7_predictions_table.png",
        n=100
    )

    # ─────────────────────────────────────────────────────
    # Final Classification Report
    # ─────────────────────────────────────────────────────
    print("\n=== Final Classification Report (BCE baseline) ===")
    print(classification_report(
        hist_bce["final_labels"],
        hist_bce["final_preds"],
        target_names=["FAKE", "REAL"]
    ))

    print("\n✅ All experiments done! Figures saved in ./figures/")
