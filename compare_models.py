"""
=============================================================
CDS525 Group Project — Model Comparison
BiLSTM  vs  BERT + BiLSTM
=============================================================
Run this AFTER both training scripts have finished:
    python bilstm_fakenews.py
    python bert_bilstm_fakenews.py
    python compare_models.py        ← this file

Output: figures/comparison/  (4 comparison figures)
=============================================================
"""

import os, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)

# =============================================================
# CONFIG
# =============================================================
RESULT_DIR = "results"
FIG_DIR    = "figures/comparison"
os.makedirs(FIG_DIR, exist_ok=True)

BILSTM_COLOR      = "#2196F3"   # blue
BERT_BILSTM_COLOR = "#E91E63"   # pink-red


# =============================================================
# 1. LOAD SAVED RESULTS
# =============================================================

def load_results():
    bilstm_path      = f"{RESULT_DIR}/bilstm_results.json"
    bert_bilstm_path = f"{RESULT_DIR}/bert_bilstm_results.json"

    missing = [p for p in [bilstm_path, bert_bilstm_path] if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"\n❌ Missing result files: {missing}\n"
            f"   Please run both training scripts first:\n"
            f"   1) python bilstm_fakenews.py\n"
            f"   2) python bert_bilstm_fakenews.py"
        )

    with open(bilstm_path)      as f: bilstm      = json.load(f)
    with open(bert_bilstm_path) as f: bert_bilstm = json.load(f)

    print("✅ Results loaded:")
    print(f"   BiLSTM      best test acc = {bilstm['final_test_acc']:.4f}")
    print(f"   BERT+BiLSTM best test acc = {bert_bilstm['final_test_acc']:.4f}")
    return bilstm, bert_bilstm


def compute_metrics(preds, labels):
    return {
        "accuracy":  accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall":    recall_score(labels, preds, zero_division=0),
        "f1":        f1_score(labels, preds, zero_division=0),
    }


# =============================================================
# 2. COMPARISON FIGURE 1 — Learning Curves Side-by-Side
# =============================================================

def plot_learning_curves(bilstm, bert_bilstm):
    """
    Left panel  : Train Loss comparison
    Right panel : Test Accuracy comparison
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ── Train Loss ────────────────────────────────────────
    ep_b  = range(1, len(bilstm["bce_train_loss"]) + 1)
    ep_bb = range(1, len(bert_bilstm["bce_train_loss"]) + 1)

    ax1.plot(ep_b,  bilstm["bce_train_loss"],
             color=BILSTM_COLOR, marker="o", markersize=4, label="BiLSTM")
    ax1.plot(ep_bb, bert_bilstm["bce_train_loss"],
             color=BERT_BILSTM_COLOR, marker="s", markersize=4, label="BERT+BiLSTM")
    ax1.set_title("Training Loss Comparison", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.legend(fontsize=11); ax1.grid(True, alpha=0.3)

    # ── Test Accuracy ─────────────────────────────────────
    ax2.plot(ep_b,  bilstm["bce_test_acc"],
             color=BILSTM_COLOR, marker="o", markersize=4, label="BiLSTM")
    ax2.plot(ep_bb, bert_bilstm["bce_test_acc"],
             color=BERT_BILSTM_COLOR, marker="s", markersize=4, label="BERT+BiLSTM")
    ax2.set_title("Test Accuracy Comparison", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1); ax2.legend(fontsize=11); ax2.grid(True, alpha=0.3)

    plt.suptitle("BiLSTM vs BERT+BiLSTM — Training Curves",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = f"{FIG_DIR}/compare_fig1_learning_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")


# =============================================================
# 3. COMPARISON FIGURE 2 — Metrics Bar Chart
# =============================================================

def plot_metrics_bar(bilstm, bert_bilstm):
    """
    Grouped bar chart: Accuracy / Precision / Recall / F1
    for BiLSTM vs BERT+BiLSTM.
    """
    m_b  = compute_metrics(bilstm["final_preds"],      bilstm["final_labels"])
    m_bb = compute_metrics(bert_bilstm["final_preds"], bert_bilstm["final_labels"])

    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    vals_b  = [m_b["accuracy"],  m_b["precision"],  m_b["recall"],  m_b["f1"]]
    vals_bb = [m_bb["accuracy"], m_bb["precision"], m_bb["recall"], m_bb["f1"]]

    x   = np.arange(len(metrics))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))

    bars_b  = ax.bar(x - w/2, vals_b,  w, label="BiLSTM",
                     color=BILSTM_COLOR, alpha=0.85, edgecolor="white")
    bars_bb = ax.bar(x + w/2, vals_bb, w, label="BERT+BiLSTM",
                     color=BERT_BILSTM_COLOR, alpha=0.85, edgecolor="white")

    # Value labels on bars
    for bar in bars_b:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                f"{h:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    for bar in bars_bb:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                f"{h:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylim(0, 1.1); ax.set_ylabel("Score", fontsize=12)
    ax.legend(fontsize=12); ax.grid(axis="y", alpha=0.3)
    ax.set_title("BiLSTM vs BERT+BiLSTM — Performance Metrics",
                 fontsize=14, fontweight="bold")

    plt.tight_layout()
    path = f"{FIG_DIR}/compare_fig2_metrics_bar.png"
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved: {path}")

    # Print table
    print("\n" + "="*52)
    print(f"{'Metric':<12} {'BiLSTM':>12} {'BERT+BiLSTM':>14}")
    print("-"*40)
    for name, vb, vbb in zip(metrics, vals_b, vals_bb):
        diff = vbb - vb
        sign = "+" if diff >= 0 else ""
        print(f"{name:<12} {vb:>12.4f} {vbb:>14.4f}   ({sign}{diff:.4f})")
    print("="*52)


# =============================================================
# 4. COMPARISON FIGURE 3 — Confusion Matrices
# =============================================================

def plot_confusion_matrices(bilstm, bert_bilstm):
    """Side-by-side confusion matrices."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for ax, data, title, color in [
        (ax1, bilstm,      "BiLSTM",       BILSTM_COLOR),
        (ax2, bert_bilstm, "BERT+BiLSTM",  BERT_BILSTM_COLOR),
    ]:
        cm = confusion_matrix(data["final_labels"], data["final_preds"])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        im = ax.imshow(cm_norm, interpolation="nearest",
                       cmap=plt.cm.Blues, vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        classes = ["FAKE", "REAL"]
        ax.set_xticks([0,1]); ax.set_xticklabels(classes, fontsize=12)
        ax.set_yticks([0,1]); ax.set_yticklabels(classes, fontsize=12)
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)
        ax.set_title(f"{title}\nConfusion Matrix",
                     fontsize=13, fontweight="bold")

        thresh = 0.5
        for i in range(2):
            for j in range(2):
                ax.text(j, i,
                        f"{cm[i,j]}\n({cm_norm[i,j]:.1%})",
                        ha="center", va="center", fontsize=13,
                        color="white" if cm_norm[i,j] > thresh else "black",
                        fontweight="bold")

    plt.suptitle("Confusion Matrix Comparison", fontsize=15, fontweight="bold")
    plt.tight_layout()
    path = f"{FIG_DIR}/compare_fig3_confusion_matrices.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")


# =============================================================
# 5. COMPARISON FIGURE 4 — BCE vs Focal Summary
# =============================================================

def plot_loss_function_comparison(bilstm, bert_bilstm):
    """
    Compare best test accuracy under BCE vs Focal Loss
    for both models — 2×2 grouped bar chart.
    """
    bce_b   = max(bilstm["bce_test_acc"])
    focal_b = max(bilstm["focal_test_acc"])
    bce_bb  = max(bert_bilstm["bce_test_acc"])
    focal_bb= max(bert_bilstm["focal_test_acc"])

    labels = ["BiLSTM", "BERT+BiLSTM"]
    bce_vals   = [bce_b,   bce_bb]
    focal_vals = [focal_b, focal_bb]

    x = np.arange(len(labels)); w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))

    bars1 = ax.bar(x - w/2, bce_vals,   w, label="BCE Loss",
                   color="#42A5F5", edgecolor="white", alpha=0.9)
    bars2 = ax.bar(x + w/2, focal_vals, w, label="Focal Loss",
                   color="#EF5350", edgecolor="white", alpha=0.9)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.003,
                    f"{h:.3f}", ha="center", va="bottom",
                    fontsize=11, fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=13)
    ax.set_ylim(0, 1.1); ax.set_ylabel("Best Test Accuracy", fontsize=12)
    ax.legend(fontsize=12); ax.grid(axis="y", alpha=0.3)
    ax.set_title("BCE Loss vs Focal Loss — Best Test Accuracy",
                 fontsize=13, fontweight="bold")

    plt.tight_layout()
    path = f"{FIG_DIR}/compare_fig4_loss_function.png"
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved: {path}")


# =============================================================
# MAIN
# =============================================================

if __name__ == "__main__":
    print("\n" + "="*52)
    print("  Model Comparison: BiLSTM vs BERT+BiLSTM")
    print("="*52)

    bilstm, bert_bilstm = load_results()

    print("\n── Fig 1: Learning Curves ──────────────────")
    plot_learning_curves(bilstm, bert_bilstm)

    print("\n── Fig 2: Metrics Bar Chart ────────────────")
    plot_metrics_bar(bilstm, bert_bilstm)

    print("\n── Fig 3: Confusion Matrices ───────────────")
    plot_confusion_matrices(bilstm, bert_bilstm)

    print("\n── Fig 4: BCE vs Focal Loss ────────────────")
    plot_loss_function_comparison(bilstm, bert_bilstm)

    print(f"\n✅ All comparison figures saved to ./{FIG_DIR}/")
    print("\nSuggested order for report:")
    print("  compare_fig1_learning_curves.png  → Training process")
    print("  compare_fig2_metrics_bar.png      → Overall performance")
    print("  compare_fig3_confusion_matrices.png → Error analysis")
    print("  compare_fig4_loss_function.png    → Loss function impact")
