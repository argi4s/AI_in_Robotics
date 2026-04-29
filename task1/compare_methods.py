import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from task1.augmentation import darken_dataset


# ─── Main comparison ───────────────────────────────────────────────────────────

def run_comparison(knn_model, svm_model, cnn_model,
                   X_test, y_test, class_names,
                   dark_gamma: float = 10,
                   save_dir=None,
                   knn_aug_model=None,
                   svm_aug_model=None) -> dict:
    
    from sklearn.metrics import accuracy_score

    X_dark = darken_dataset(X_test, gamma=dark_gamma)

    def _flat(X):
        return X.reshape(len(X), -1)

    results = {}

    # ── Majority-class baseline ──
    majority_class = Counter(y_test).most_common(1)[0][0]
    baseline_pred  = np.full(len(y_test), majority_class)
    baseline_acc   = accuracy_score(y_test, baseline_pred)
    results["Baseline\n(majority)"] = {
        "normal": baseline_acc, "dark": baseline_acc, "trained_dark": False
    }
    print(f"Base  — normal: {baseline_acc*100:.1f}%   dark: {baseline_acc*100:.1f}%"
          f"  (always predicts '{class_names[majority_class]}')")

    # ── KNN ──
    knn_normal = accuracy_score(y_test, knn_model.predict(_flat(X_test)))
    knn_dark   = accuracy_score(y_test, knn_model.predict(_flat(X_dark)))
    results["KNN"] = {"normal": knn_normal, "dark": knn_dark, "trained_dark": False}
    print(f"KNN   — normal: {knn_normal*100:.1f}%   dark: {knn_dark*100:.1f}%")

    # ── SVM ──
    svm_normal = accuracy_score(y_test, svm_model.predict(_flat(X_test)))
    svm_dark   = accuracy_score(y_test, svm_model.predict(_flat(X_dark)))
    results["SVM"] = {"normal": svm_normal, "dark": svm_dark, "trained_dark": False}
    print(f"SVM   — normal: {svm_normal*100:.1f}%   dark: {svm_dark*100:.1f}%")

    # ── KNN (augmented) ──
    if knn_aug_model is not None:
        knn_aug_normal = accuracy_score(y_test, knn_aug_model.predict(_flat(X_test)))
        knn_aug_dark   = accuracy_score(y_test, knn_aug_model.predict(_flat(X_dark)))
        results["KNN (aug.)"] = {
            "normal": knn_aug_normal, "dark": knn_aug_dark, "trained_dark": True
        }
        print(f"KNN+  — normal: {knn_aug_normal*100:.1f}%   dark: {knn_aug_dark*100:.1f}%")

    # ── SVM (augmented) ──
    if svm_aug_model is not None:
        svm_aug_normal = accuracy_score(y_test, svm_aug_model.predict(_flat(X_test)))
        svm_aug_dark   = accuracy_score(y_test, svm_aug_model.predict(_flat(X_dark)))
        results["SVM (aug.)"] = {
            "normal": svm_aug_normal, "dark": svm_aug_dark, "trained_dark": True
        }
        print(f"SVM+  — normal: {svm_aug_normal*100:.1f}%   dark: {svm_aug_dark*100:.1f}%")

    # ── CNN ──
    if cnn_model is not None:
        cnn_normal = accuracy_score(
            y_test, np.argmax(cnn_model.predict(X_test, verbose=0), axis=1))
        cnn_dark   = accuracy_score(
            y_test, np.argmax(cnn_model.predict(X_dark, verbose=0), axis=1))
        results["CNN\n(aug. trained)"] = {
            "normal": cnn_normal, "dark": cnn_dark, "trained_dark": True
        }
        print(f"CNN   — normal: {cnn_normal*100:.1f}%   dark: {cnn_dark*100:.1f}%")

    _plot_comparison(results, dark_gamma, save_dir)
    _print_table(results, dark_gamma)
    _save_summary_table(results, dark_gamma, save_dir)

    return results


# ─── Plotting ──────────────────────────────────────────────────────────────────

def _plot_comparison(results: dict, dark_gamma: float, save_dir=None):
    models = list(results.keys())
    normal = [results[m]["normal"] * 100 for m in models]
    dark   = [results[m]["dark"]   * 100 for m in models]

    x  = np.arange(len(models))
    w  = 0.35

    # Baseline gets a neutral grey; others use standard colours
    bar_colours_n = ["#AAAAAA" if "Baseline" in m else "#4C72B0" for m in models]
    bar_colours_d = ["#BBBBBB" if "Baseline" in m else "#DD8452" for m in models]

    fig, ax = plt.subplots(figsize=(10, 5))

    bars_n = ax.bar(x - w/2, normal, w, label="1-block (normal)",
                    color=bar_colours_n)
    bars_d = ax.bar(x + w/2, dark,   w, label=f"2-block (γ={dark_gamma})",
                    color=bar_colours_d)

    for bar in list(bars_n) + list(bars_d):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%", ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 115)
    ax.set_title("Entity Detection Accuracy: All Methods × Both Scenarios\n"
                 "Grey = majority-class baseline  |  Blue = 1-block  |  Orange = 2-block",
                 fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()

    if save_dir:
        plt.savefig(os.path.join(save_dir, "method_comparison.png"), dpi=150)
    plt.close()


def _print_table(results: dict, dark_gamma: float):
    print(f"\n{'='*70}")
    print(f"  DETECTION ACCURACY — 1-block (normal) vs 2-block (γ={dark_gamma})")
    print(f"{'='*70}")
    print(f"  {'Method':<26}  {'1-block':>9}  {'2-block':>9}  "
          f"{'Change':>8}  {'Dark train?':>11}")
    print(f"  {'─'*26}  {'─'*9}  {'─'*9}  {'─'*8}  {'─'*11}")
    for name, v in results.items():
        label    = name.replace("\n", " ")
        drop     = v["dark"] - v["normal"]   # negative = degradation
        trained  = "Yes" if v.get("trained_dark") else "No"
        print(f"  {label:<26}  {v['normal']*100:>8.1f}%  {v['dark']*100:>8.1f}%  "
              f"{drop*100:>+7.1f}%  {trained:>11}")
    print(f"{'='*70}")
    print("  Drop = accuracy lost when switching to the dark (2-block) scenario.")
    print(f"{'='*70}\n")


def _save_summary_table(results: dict, dark_gamma: float, save_dir=None):
    """
    Save a clean summary table as a PNG — ready to paste into the report.
    Columns: Model | 1-block | 2-block | Drop | Trained on dark?
    """
    rows = []
    for name, v in results.items():
        label   = name.replace("\n", " ")
        drop    = v["normal"] - v["dark"]
        trained = "Yes" if v.get("trained_dark") else "No"
        rows.append([
            label,
            f"{v['normal']*100:.1f}%",
            f"{v['dark']*100:.1f}%",
            f"{(v['dark']-v['normal'])*100:+.1f}%",
            trained,
        ])

    col_labels = ["Model", f"1-block\n(normal)", f"2-block\n(γ={dark_gamma})",
                  "Accuracy\nDrop", "Trained on\ndark data?"]

    fig, ax = plt.subplots(figsize=(10, 1.2 + 0.5 * len(rows)))
    ax.axis('off')

    tbl = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc='center',
        cellLoc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.8)

    # Style header row
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#2C3E50")
        tbl[0, j].set_text_props(color='white', fontweight='bold')

    # Style baseline row (row 1)
    for j in range(len(col_labels)):
        tbl[1, j].set_facecolor("#EEEEEE")

    # Highlight CNN row (last row if CNN exists)
    last = len(rows)
    if "Yes" in [r[4] for r in rows]:
        for j in range(len(col_labels)):
            tbl[last, j].set_facecolor("#EAF4EA")

    fig.suptitle(
        f"Task 1 — Classification Accuracy Summary\n"
        f"Test set: held-out images (normal) and same images darkened at γ={dark_gamma}",
        fontsize=11, fontweight='bold', y=0.98
    )
    plt.tight_layout()

    if save_dir:
        plt.savefig(os.path.join(save_dir, "summary_table.png"),
                    dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: summary_table.png")
