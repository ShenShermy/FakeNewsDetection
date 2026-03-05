"""
=============================================================
CDS525 Group Project — Model Comparison
BiLSTM  vs  BERT  vs  BERT + BiLSTM
=============================================================
Run AFTER all three training scripts:
    1) python bilstm_fakenews.py
    2) python bert_bilstm_fakenews.py
    3) python bert_fakenews.py
    4) python compare_models.py        ← this file

Output: figures/comparison/  (5 comparison figures)
=============================================================
"""

import os, json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)

# =============================================================
# CONFIG TEST RESULTS & FIGURE SETTINGS
# =============================================================
RESULT_DIR = "results"
FIG_DIR    = "figures/comparison"
os.makedirs(FIG_DIR, exist_ok=True)

# Colors for each model — consistent across all figures
MODEL_COLORS = {
    "BiLSTM":       "#2196F3",   # blue
    "BERT":         "#4CAF50",   # green
    "BERT+BiLSTM":  "#E91E63",   # red
}
MARKERS = {"BiLSTM": "o", "BERT": "s", "BERT+BiLSTM": "^"}


# =============================================================
# 1. LOAD RESULTS
# =============================================================

def load_results():
    paths = {
        "BiLSTM":      f"{RESULT_DIR}/bilstm_results.json",
        "BERT":        f"{RESULT_DIR}/bert_results.json",
        "BERT+BiLSTM": f"{RESULT_DIR}/bert_bilstm_results.json",
    }
    missing = [p for p in paths.values() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"\n❌ Missing result files:\n  " + "\n  ".join(missing) +
            "\n\nPlease run all training scripts first:\n"
            "  1) python bilstm_fakenews.py\n"
            "  2) python bert_fakenews.py\n"
            "  3) python bert_bilstm_fakenews.py"
        )
    results = {}
    print("\n" + "="*52)
    print("  Loaded Results")
    print("="*52)
    for name, path in paths.items():
        with open(path) as f:
            results[name] = json.load(f)
        print(f"  {name:<14} best test acc = {results[name]['final_test_acc']:.4f}")
    print("="*52)
    return results


def compute_metrics(preds, labels):
    return {
        "Accuracy":  accuracy_score(labels, preds),
        "Precision": precision_score(labels, preds, zero_division=0),
        "Recall":    recall_score(labels, preds, zero_division=0),
        "F1-Score":  f1_score(labels, preds, zero_division=0),
    }


# =============================================================
# 2. FIG 1 — Training Loss Curves (3 models)
# =============================================================

def plot_train_loss(results):
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, data in results.items():
        eps = range(1, len(data["bce_train_loss"]) + 1)
        ax.plot(eps, data["bce_train_loss"],
                color=MODEL_COLORS[name], marker=MARKERS[name],
                markersize=5, linewidth=2, label=name)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Training Loss", fontsize=12)
    ax.legend(fontsize=12); ax.grid(True, alpha=0.3)
    ax.set_title("Training Loss Comparison — All Models",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = f"{FIG_DIR}/compare_fig1_train_loss.png"
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved: {path}")


# =============================================================
# 3. FIG 2 — Test Accuracy Curves (3 models)
# =============================================================

def plot_test_accuracy(results):
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, data in results.items():
        eps = range(1, len(data["bce_test_acc"]) + 1)
        ax.plot(eps, data["bce_test_acc"],
                color=MODEL_COLORS[name], marker=MARKERS[name],
                markersize=5, linewidth=2, label=name)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=12); ax.grid(True, alpha=0.3)
    ax.set_title("Test Accuracy Comparison — All Models",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = f"{FIG_DIR}/compare_fig2_test_accuracy.png"
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved: {path}")


# =============================================================
# 4. FIG 3 — Grouped Metrics Bar Chart
# =============================================================

def plot_metrics_bar(results):
    all_metrics = {}
    for name, data in results.items():
        all_metrics[name] = compute_metrics(data["final_preds"], data["final_labels"])

    metric_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
    model_names  = list(results.keys())
    x  = np.arange(len(metric_names))
    w  = 0.25
    offsets = [-w, 0, w]

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, name in enumerate(model_names):
        vals = [all_metrics[name][m] for m in metric_names]
        bars = ax.bar(x + offsets[i], vals, w,
                      label=name, color=MODEL_COLORS[name],
                      alpha=0.88, edgecolor="white")
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.004,
                    f"{h:.3f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(metric_names, fontsize=12)
    ax.set_ylim(0, 1.12); ax.set_ylabel("Score", fontsize=12)
    ax.legend(fontsize=11); ax.grid(axis="y", alpha=0.3)
    ax.set_title("Performance Metrics — All Models",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = f"{FIG_DIR}/compare_fig3_metrics_bar.png"
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved: {path}")

    # Print comparison table
    print("\n" + "="*62)
    print(f"{'Metric':<12}", end="")
    for name in model_names:
        print(f"  {name:>14}", end="")
    print()
    print("-"*62)
    for m in metric_names:
        print(f"{m:<12}", end="")
        vals = [all_metrics[n][m] for n in model_names]
        for v in vals:
            print(f"  {v:>14.4f}", end="")
        best = max(range(len(vals)), key=lambda i: vals[i])
        print(f"   ← best: {model_names[best]}")
    print("="*62)


# =============================================================
# 5. FIG 4 — Confusion Matrices (3 models side by side)
# =============================================================

def plot_confusion_matrices(results):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, (name, data) in zip(axes, results.items()):
        cm      = confusion_matrix(data["final_labels"], data["final_preds"])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        im = ax.imshow(cm_norm, cmap=plt.cm.Blues, vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        classes = ["FAKE", "REAL"]
        ax.set_xticks([0,1]); ax.set_xticklabels(classes, fontsize=11)
        ax.set_yticks([0,1]); ax.set_yticklabels(classes, fontsize=11)
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("True Label", fontsize=11)
        ax.set_title(name, fontsize=13, fontweight="bold",
                     color=MODEL_COLORS[name])

        for i in range(2):
            for j in range(2):
                ax.text(j, i,
                        f"{cm[i,j]}\n({cm_norm[i,j]:.1%})",
                        ha="center", va="center", fontsize=12,
                        color="white" if cm_norm[i,j] > 0.5 else "black",
                        fontweight="bold")

    plt.suptitle("Confusion Matrices — All Models",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    path = f"{FIG_DIR}/compare_fig4_confusion_matrices.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")


# =============================================================
# 6. FIG 5 — BCE vs Focal Loss (all 3 models)
# =============================================================

def plot_loss_function_comparison(results):
    model_names = list(results.keys())
    bce_vals    = [max(results[n]["bce_test_acc"])   for n in model_names]
    focal_vals  = [max(results[n]["focal_test_acc"]) for n in model_names]

    x = np.arange(len(model_names)); w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))

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

    ax.set_xticks(x); ax.set_xticklabels(model_names, fontsize=12)
    ax.set_ylim(0, 1.12); ax.set_ylabel("Best Test Accuracy", fontsize=12)
    ax.legend(fontsize=12); ax.grid(axis="y", alpha=0.3)
    ax.set_title("BCE Loss vs Focal Loss — Best Test Accuracy (All Models)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = f"{FIG_DIR}/compare_fig5_loss_function.png"
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved: {path}")


# =============================================================
# MAIN
# =============================================================

if __name__ == "__main__":
    print("\n" + "="*52)
    print("  Model Comparison: BiLSTM vs BERT vs BERT+BiLSTM")
    print("="*52)

    results = load_results()

    print("\n── Fig 1: Training Loss Curves ─────────────")
    plot_train_loss(results)

    print("\n── Fig 2: Test Accuracy Curves ─────────────")
    plot_test_accuracy(results)

    print("\n── Fig 3: Metrics Bar Chart ─────────────────")
    plot_metrics_bar(results)

    print("\n── Fig 4: Confusion Matrices ────────────────")
    plot_confusion_matrices(results)

    print("\n── Fig 5: BCE vs Focal Loss ─────────────────")
    plot_loss_function_comparison(results)

    print(f"\n✅ All comparison figures saved to ./{FIG_DIR}/")
    print("\nSuggested use in report:")
    print("  compare_fig1_train_loss.png        → Training process")
    print("  compare_fig2_test_accuracy.png     → Convergence comparison")
    print("  compare_fig3_metrics_bar.png       → Overall performance")
    print("  compare_fig4_confusion_matrices.png → Error analysis")
    print("  compare_fig5_loss_function.png     → Loss function impact")
