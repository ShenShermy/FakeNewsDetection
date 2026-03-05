"""
=============================================================
CDS525 Group Project — Model 2: BERT + BiLSTM
Fake News Detection
=============================================================
Run AFTER bilstm_fakenews.py, then run compare_models.py.

Install:
    pip install torch transformers scikit-learn pandas matplotlib tqdm
=============================================================
"""

import os, re, random, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import BertTokenizer, BertModel

# =============================================================
# ★ CONFIG
# =============================================================
DEVICE_MODE = "auto"    # "auto" | "cuda" | "cpu"
SEED        = 42        # Keep same as bilstm_fakenews.py!
EPOCHS      = 10        # BERT converges faster than BiLSTM
CSV_PATH    = "fakenews.csv"
SAVE_DIR    = "results"
FIG_DIR     = "figures/bert_bilstm"

# BERT memory tip:
#   freeze_bert=True  → freeze all BERT layers (fast, lower accuracy)
#   freeze_bert=False → fine-tune last 2 BERT layers (recommended)
FREEZE_BERT = False
# =============================================================

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def get_device(mode):
    mode = mode.strip().lower()
    if mode == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("No GPU found. Use DEVICE_MODE='auto' or 'cpu'.")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"\n{'='*50}")
    print(f"  [BERT+BiLSTM] Device : {device}")
    if device.type == "cuda":
        print(f"  GPU : {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  Mem : {mem:.1f} GB")
    print(f"  freeze_bert = {FREEZE_BERT}")
    print(f"{'='*50}\n")
    return device

DEVICE = get_device(DEVICE_MODE)
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(FIG_DIR,  exist_ok=True)


# =============================================================
# 1. DATA
# =============================================================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df.dropna(inplace=True)
    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].str.strip() != ""]
    labels = sorted(df["label"].unique())
    df["label"] = df["label"].map({v: i for i, v in enumerate(labels)})

    X_tr, X_te, y_tr, y_te = train_test_split(
        df["text"].tolist(), df["label"].tolist(),
        test_size=0.2, random_state=SEED, stratify=df["label"])
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tr, y_tr, test_size=0.1, random_state=SEED)
    print(f"Train {len(X_tr)} | Val {len(X_va)} | Test {len(X_te)}")
    return X_tr, X_va, X_te, y_tr, y_va, y_te


# =============================================================
# 2. DATASET
# =============================================================

class BertDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[idx], dtype=torch.float)
        }


def make_loaders(X_tr, X_va, X_te, y_tr, y_va, y_te, tokenizer, bs):
    tr = DataLoader(BertDataset(X_tr, y_tr, tokenizer), batch_size=bs, shuffle=True)
    va = DataLoader(BertDataset(X_va, y_va, tokenizer), batch_size=bs)
    te = DataLoader(BertDataset(X_te, y_te, tokenizer), batch_size=bs)
    return tr, va, te


# =============================================================
# 3. MODEL
# =============================================================

class BertBiLSTMClassifier(nn.Module):
    """
    BERT → BiLSTM → Masked Attention → FC → Sigmoid

    Flow:
        token ids  →  BERT (768-dim contextual embeddings)
                   →  BiLSTM (captures sequential patterns)
                   →  Attention pooling (focus on key tokens)
                   →  FC + Sigmoid  →  REAL/FAKE
    """
    def __init__(self, bert_model_name="bert-base-uncased",
                 hidden_dim=256, num_layers=2, dropout=0.3,
                 freeze_bert=False):
        super().__init__()

        # ── BERT ──────────────────────────────────────────
        self.bert    = BertModel.from_pretrained(bert_model_name)
        bert_dim     = self.bert.config.hidden_size   # 768

        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False
        else:
            # Fine-tune only last 2 transformer layers + pooler
            for name, p in self.bert.named_parameters():
                p.requires_grad = (
                    any(f"layer.{i}" in name for i in [10, 11])
                    or "pooler" in name
                )

        # ── BiLSTM ────────────────────────────────────────
        self.lstm = nn.LSTM(bert_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0)

        # ── Attention ─────────────────────────────────────
        self.attention = nn.Linear(hidden_dim * 2, 1)

        # ── Classifier ────────────────────────────────────
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_dim * 2, 1)

    def forward(self, input_ids, attention_mask):
        # BERT: (B, L, 768)
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq_out  = self.dropout(bert_out.last_hidden_state)

        # BiLSTM: (B, L, 2H)
        lstm_out, _ = self.lstm(seq_out)

        # Masked attention — ignore [PAD] tokens
        mask  = attention_mask.unsqueeze(-1).float()
        score = self.attention(lstm_out)
        score = score.masked_fill(mask == 0, -1e9)
        score = torch.softmax(score, dim=1)
        ctx   = (score * lstm_out).sum(dim=1)   # (B, 2H)

        return self.fc(self.dropout(ctx)).squeeze(1)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma
    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt  = torch.exp(-bce)
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()


# =============================================================
# 4. TRAIN / EVAL
# =============================================================

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, preds_all, labels_all = 0, [], []
    for batch in tqdm(loader, desc="  train", leave=False):
        ids  = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        y    = batch["label"].to(DEVICE)
        optimizer.zero_grad()
        logits = model(ids, mask)
        loss   = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(y)
        preds_all.extend((torch.sigmoid(logits) > 0.5).long().cpu().tolist())
        labels_all.extend(y.long().cpu().tolist())
    return total_loss / len(loader.dataset), accuracy_score(labels_all, preds_all)


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, preds_all, labels_all = 0, [], []
    for batch in loader:
        ids    = batch["input_ids"].to(DEVICE)
        mask   = batch["attention_mask"].to(DEVICE)
        y      = batch["label"].to(DEVICE)
        logits = model(ids, mask)
        total_loss += criterion(logits, y).item() * len(y)
        preds_all.extend((torch.sigmoid(logits) > 0.5).long().cpu().tolist())
        labels_all.extend(y.long().cpu().tolist())
    return total_loss / len(loader.dataset), accuracy_score(labels_all, preds_all), preds_all, labels_all


def run_experiment(X_tr, X_va, X_te, y_tr, y_va, y_te, tokenizer,
                   criterion, lr, batch_size, num_epochs, tag):
    tr_l, va_l, te_l = make_loaders(X_tr, X_va, X_te, y_tr, y_va, y_te, tokenizer, batch_size)
    model     = BertBiLSTMClassifier(freeze_bert=FREEZE_BERT).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    hist = {"train_loss": [], "train_acc": [], "test_acc": []}
    best_val_acc, best_state = 0, None

    print(f"\n[{tag}]  lr={lr}  bs={batch_size}  epochs={num_epochs}")
    for ep in range(1, num_epochs + 1):
        tl, ta        = train_epoch(model, tr_l, criterion, optimizer)
        vl, va, _, _  = eval_epoch(model, va_l, criterion)
        _,  tea, _, _ = eval_epoch(model, te_l, criterion)
        scheduler.step(vl)
        hist["train_loss"].append(tl)
        hist["train_acc"].append(ta)
        hist["test_acc"].append(tea)
        if va > best_val_acc:
            best_val_acc = va
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"  Ep {ep:02d}/{num_epochs} | Loss {tl:.4f} | "
              f"TrainAcc {ta:.4f} | ValAcc {va:.4f} | TestAcc {tea:.4f}")

    model.load_state_dict(best_state)
    _, _, fp, fl = eval_epoch(model, te_l, criterion)
    hist["final_preds"]  = fp
    hist["final_labels"] = fl
    return hist


# =============================================================
# 5. PLOTS (same style as bilstm_fakenews.py)
# =============================================================

COLORS = ["#2196F3", "#E91E63", "#4CAF50", "#FF9800", "#9C27B0"]

def plot_single(hist, title, path):
    eps = range(1, len(hist["train_loss"]) + 1)
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(eps, hist["train_loss"], "b-o", markersize=4, label="Train Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax2 = ax1.twinx()
    ax2.plot(eps, hist["train_acc"], "g-s", markersize=4, label="Train Acc")
    ax2.plot(eps, hist["test_acc"],  "r-^", markersize=4, label="Test Acc")
    ax2.set_ylabel("Accuracy"); ax2.set_ylim(0, 1)
    l1, n1 = ax1.get_legend_handles_labels()
    l2, n2 = ax2.get_legend_handles_labels()
    ax1.legend(l1+l2, n1+n2, loc="center right", fontsize=10)
    plt.title(title, fontsize=13, fontweight="bold")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved: {path}")

def plot_multi(hists_dict, metric, ylabel, title, path, log_scale=False):
    eps = range(1, len(next(iter(hists_dict.values()))["train_loss"]) + 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (lbl, h) in enumerate(hists_dict.items()):
        ax.plot(eps, h[metric], color=COLORS[i % len(COLORS)],
                marker="o", markersize=4, label=lbl)
    ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
    if log_scale: ax.set_yscale("log")
    if "acc" in metric: ax.set_ylim(0, 1)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    plt.title(title, fontsize=13, fontweight="bold")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved: {path}")

def plot_predictions_table(preds, labels, texts, path, n=100):
    n = min(n, len(preds))
    rows = [[i+1,
             " ".join(texts[i].split()[:12]) + "...",
             "REAL" if labels[i] else "FAKE",
             "REAL" if preds[i]  else "FAKE",
             "✓" if preds[i] == labels[i] else "✗"]
            for i in range(n)]
    df = pd.DataFrame(rows, columns=["#","Text Snippet","True Label","Predicted","Correct?"])
    fig, ax = plt.subplots(figsize=(18, max(6, n * 0.28)))
    ax.axis("off")
    tbl = ax.table(cellText=df.values, colLabels=df.columns, cellLoc="left", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(7.5)
    tbl.auto_set_column_width(list(range(len(df.columns))))
    for j in range(len(df.columns)):
        tbl[(0,j)].set_facecolor("#1A237E")
        tbl[(0,j)].set_text_props(color="white", fontweight="bold")
    for i in range(1, n+1):
        c = "#E8F5E9" if rows[i-1][4]=="✓" else "#FFEBEE"
        for j in range(len(df.columns)):
            tbl[(i,j)].set_facecolor(c)
    plt.title(f"BERT+BiLSTM Predictions – First {n} Test Samples",
              fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")


# =============================================================
# 6. MAIN
# =============================================================

if __name__ == "__main__":

    X_tr, X_va, X_te, y_tr, y_va, y_te = load_data(CSV_PATH)

    print("\nLoading BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def exp(criterion, lr=2e-5, bs=16, tag=""):
        return run_experiment(X_tr, X_va, X_te, y_tr, y_va, y_te,
                              tokenizer, criterion, lr, bs, EPOCHS, tag)

    # ── Fig 1: BCE baseline ───────────────────────────────
    print("\n" + "="*50 + "\nExp 1 — BCE Loss\n" + "="*50)
    h_bce = exp(nn.BCEWithLogitsLoss(), tag="BCE")
    plot_single(h_bce, "Fig 1 – BERT+BiLSTM | BCE Loss (lr=2e-5, bs=16)",
                f"{FIG_DIR}/fig1_bce_baseline.png")

    # ── Fig 2: Focal Loss ─────────────────────────────────
    print("\n" + "="*50 + "\nExp 2 — Focal Loss\n" + "="*50)
    h_focal = exp(FocalLoss(), tag="Focal")
    plot_single(h_focal, "Fig 2 – BERT+BiLSTM | Focal Loss (lr=2e-5, bs=16)",
                f"{FIG_DIR}/fig2_focal_loss.png")

    # ── Fig 3 & 4: Learning Rate ──────────────────────────
    print("\n" + "="*50 + "\nExp 3 — Different LR\n" + "="*50)
    lr_hists = {}
    for lr in [1e-4, 5e-5, 2e-5, 1e-5]:
        lr_hists[f"lr={lr}"] = exp(nn.BCEWithLogitsLoss(), lr=lr, tag=f"lr={lr}")
    plot_multi(lr_hists, "train_loss", "Train Loss",
               "Fig 3 – BERT+BiLSTM Train Loss: Different LR",
               f"{FIG_DIR}/fig3_lr_train_loss.png", log_scale=True)
    plot_multi(lr_hists, "test_acc",  "Test Accuracy",
               "Fig 4 – BERT+BiLSTM Test Accuracy: Different LR",
               f"{FIG_DIR}/fig4_lr_test_acc.png")

    # ── Fig 5 & 6: Batch Size ─────────────────────────────
    print("\n" + "="*50 + "\nExp 4 — Different Batch Size\n" + "="*50)
    bs_hists = {}
    for bs in [4, 8, 16, 32]:   # keep small due to BERT memory
        bs_hists[f"bs={bs}"] = exp(nn.BCEWithLogitsLoss(), bs=bs, tag=f"bs={bs}")
    plot_multi(bs_hists, "train_loss", "Train Loss",
               "Fig 5 – BERT+BiLSTM Train Loss: Different Batch Size",
               f"{FIG_DIR}/fig5_bs_train_loss.png")
    plot_multi(bs_hists, "test_acc",  "Test Accuracy",
               "Fig 6 – BERT+BiLSTM Test Accuracy: Different Batch Size",
               f"{FIG_DIR}/fig6_bs_test_acc.png")

    # ── Fig 7: Prediction Table ───────────────────────────
    print("\n" + "="*50 + "\nPrediction Table\n" + "="*50)
    plot_predictions_table(h_bce["final_preds"], h_bce["final_labels"],
                           X_te, f"{FIG_DIR}/fig7_predictions.png")

    # ── Classification Report ─────────────────────────────
    print("\n" + "="*50)
    print(classification_report(h_bce["final_labels"], h_bce["final_preds"],
                                target_names=["FAKE","REAL"]))

    # ── Save results for comparison ───────────────────────
    save_data = {
        "model": "BERT+BiLSTM",
        "bce_train_loss": h_bce["train_loss"],
        "bce_train_acc":  h_bce["train_acc"],
        "bce_test_acc":   h_bce["test_acc"],
        "focal_test_acc": h_focal["test_acc"],
        "final_test_acc": max(h_bce["test_acc"]),
        "final_preds":    h_bce["final_preds"],
        "final_labels":   h_bce["final_labels"],
    }
    with open(f"{SAVE_DIR}/bert_bilstm_results.json", "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n✅ BERT+BiLSTM done! Results saved to {SAVE_DIR}/bert_bilstm_results.json")
    print(f"   Now run: python compare_models.py")
