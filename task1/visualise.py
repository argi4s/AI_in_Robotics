import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix


# ─── Dataset samples ───────────────────────────────────────────────────────────

def plot_class_samples(X, y, class_names, n_per_class=5, save_dir=None):
    """Show n_per_class random images for each class in a grid."""
    n_classes = len(class_names)
    fig, axes = plt.subplots(n_classes, n_per_class,
                             figsize=(n_per_class * 2, n_classes * 2))
    fig.suptitle("Dataset samples (normal brightness)", fontsize=13, fontweight="bold")

    for row, cls in enumerate(class_names):
        idxs = np.where(y == row)[0]
        chosen = np.random.choice(idxs, size=min(n_per_class, len(idxs)), replace=False)
        for col, idx in enumerate(chosen):
            ax = axes[row, col]
            ax.imshow(X[idx])
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(cls, fontsize=10, rotation=0,
                              labelpad=50, va="center")

    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "class_samples.png"), dpi=150)
    plt.close()
    print("Saved: class_samples.png")


# ─── Augmentation examples ─────────────────────────────────────────────────────

def plot_augmentation(X, y, class_names, gammas=(10, 20),
                      n=3, save_dir=None):
    """
    Show n random images with their gamma-darkened versions.
    Columns: Original | γ=gammas[0] | γ=gammas[1] | …
    """
    n_cols = 1 + len(gammas)
    fig, axes = plt.subplots(n, n_cols, figsize=(n_cols * 2.5, n * 2.5))
    col_titles = ["Original"] + [f"γ = {g}" for g in gammas]

    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=11, fontweight="bold")

    idxs = np.random.choice(len(X), size=n, replace=False)
    for row, idx in enumerate(idxs):
        img = X[idx]
        cls = class_names[y[idx]]
        axes[row, 0].imshow(img)
        axes[row, 0].set_ylabel(cls, fontsize=9, rotation=0,
                                labelpad=45, va="center")
        axes[row, 0].axis("off")

        for col, gamma in enumerate(gammas, start=1):
            dark = np.clip(img ** gamma, 0, 1)
            axes[row, col].imshow(dark)
            axes[row, col].axis("off")

    fig.suptitle("Gamma darkening — simulating 2-block distance",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "augmentation_examples.png"), dpi=150)
    plt.close()
    print("Saved: augmentation_examples.png")


# ─── Prediction examples ───────────────────────────────────────────────────────

def plot_predictions(predict_fn, X_test, y_test, class_names,
                     title="Model", n=12, save_dir=None):
    """
    Show a grid of n test images with predicted vs true labels.
    Green border = correct, red border = wrong.

    predict_fn : callable that takes X (N, H, W, 3) → y_pred (N,) int array
    """
    idxs   = np.random.choice(len(X_test), size=n, replace=False)
    y_pred = predict_fn(X_test[idxs])

    cols = 6
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.6))
    axes = axes.flatten()

    for i, idx in enumerate(idxs):
        ax  = axes[i]
        ax.imshow(X_test[idx])
        ax.axis("off")

        correct   = y_pred[i] == y_test[idx]
        border    = "green" if correct else "red"
        pred_name = class_names[y_pred[i]]
        true_name = class_names[y_test[idx]]

        for spine in ax.spines.values():
            spine.set_edgecolor(border)
            spine.set_linewidth(3)
            spine.set_visible(True)

        label = f"P: {pred_name}" if correct else f"P: {pred_name}\nT: {true_name}"
        col   = "green" if correct else "red"
        ax.set_title(label, fontsize=7.5, color=col, pad=3)

    for ax in axes[n:]:
        ax.axis("off")

    correct_patch = mpatches.Patch(color="green", label="Correct")
    wrong_patch   = mpatches.Patch(color="red",   label="Wrong")
    fig.legend(handles=[correct_patch, wrong_patch], loc="lower right",
               fontsize=9, framealpha=0.8)

    fig.suptitle(f"{title} — Prediction Examples", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_dir:
        fname = title.lower().replace(" ", "_").replace("/", "_") + "_preds.png"
        plt.savefig(os.path.join(save_dir, fname), dpi=150)
    plt.close()
    print(f"Saved: {fname}")


def plot_predictions_dark(predict_fn, X_test, y_test, class_names,
                          gamma=10, title="Model (dark)", n=12, save_dir=None):
    """Same as plot_predictions but on gamma-darkened images."""
    X_dark = np.clip(X_test ** gamma, 0, 1)
    plot_predictions(predict_fn, X_dark, y_test, class_names,
                     title=title, n=n, save_dir=save_dir)


# ─── Dark confusion matrix ─────────────────────────────────────────────────────

def plot_confusion_dark(predict_fn, X_test, y_test, class_names,
                        gamma=10, title="Model", save_dir=None):
    """
    Confusion matrix evaluated on gamma-darkened test images.
    Reveals which species get confused specifically under low-light conditions —
    the failure mode the CNN is designed to handle.
    """
    X_dark = np.clip(X_test ** gamma, 0, 1)
    y_pred = predict_fn(X_dark)
    acc    = accuracy_score(y_test, y_pred)
    cm     = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=class_names, yticklabels=class_names,
                cmap='Oranges', ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{title} — Confusion Matrix (dark, γ={gamma})\n"
                 f"Accuracy: {acc*100:.1f}%", fontsize=11, fontweight="bold")
    plt.tight_layout()
    if save_dir:
        fname = title.lower().replace(" ", "_").replace("/", "_") + f"_dark_cm.png"
        plt.savefig(os.path.join(save_dir, fname), dpi=150)
    plt.close()
    print(f"Saved: {fname}")


# ─── Misclassification gallery ─────────────────────────────────────────────────

def plot_misclassifications(predict_fn, X_test, y_test, class_names,
                            gamma=10, n=12, title="Model", save_dir=None):
    """
    Gallery of images the model got WRONG on the dark test set.
    Each cell shows the darkened image with true label (top) and
    predicted label (bottom) in red — visually demonstrates where and
    why the classifier fails under low-light conditions.
    """
    X_dark = np.clip(X_test ** gamma, 0, 1)
    y_pred = predict_fn(X_dark)
    wrong  = np.where(y_pred != y_test)[0]

    if len(wrong) == 0:
        print(f"[{title}] No misclassifications on dark set — skipping gallery.")
        return

    chosen = wrong[:n] if len(wrong) >= n else wrong
    n_show = len(chosen)
    cols   = min(6, n_show)
    rows   = (n_show + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.8))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    axes_flat = axes.flatten()

    for i, idx in enumerate(chosen):
        ax = axes_flat[i]
        ax.imshow(X_dark[idx])
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_edgecolor("red")
            spine.set_linewidth(3)
            spine.set_visible(True)
        ax.set_title(
            f"True:  {class_names[y_test[idx]]}\nPred: {class_names[y_pred[i]]}",
            fontsize=7.5, color="red", pad=3
        )

    for ax in axes_flat[n_show:]:
        ax.axis("off")

    fig.suptitle(
        f"{title} — Misclassifications on dark images (γ={gamma})\n"
        f"{len(wrong)} errors out of {len(y_test)} samples",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    if save_dir:
        fname = title.lower().replace(" ", "_").replace("/", "_") + "_misclass.png"
        plt.savefig(os.path.join(save_dir, fname), dpi=150)
    plt.close()
    print(f"Saved: {fname}")


# ─── Gamma sensitivity curve ───────────────────────────────────────────────────

def plot_gamma_sensitivity(models_dict: dict, X_test, y_test,
                           gammas=None, save_dir=None):
    """
    Accuracy vs gamma for each model.

    models_dict : {'KNN': predict_fn, 'SVM': predict_fn, 'CNN': predict_fn}
                  Each predict_fn takes X (N,H,W,3) → y_pred (N,) int array.
    gammas      : sequence of gamma values to test (default 1..25).

    The crossing point — where CNN overtakes KNN/SVM — directly justifies
    using CNN for the 2-block (dark) scenario in the RL pipeline.
    """
    if gammas is None:
        gammas = [1, 2, 3, 5, 7, 10, 13, 15, 18, 20, 25]

    colours = {"KNN": "#4C72B0", "KNN (aug.)": "#89ABDA",
               "SVM": "#DD8452", "SVM (aug.)": "#EEB48A",
               "CNN\n(aug. trained)": "#55A868", "CNN": "#55A868"}

    fig, ax = plt.subplots(figsize=(9, 5))

    for name, pred_fn in models_dict.items():
        accs = []
        for g in gammas:
            X_dark = np.clip(X_test ** g, 0, 1)
            accs.append(accuracy_score(y_test, pred_fn(X_dark)) * 100)
        col = colours.get(name, None)
        ax.plot(gammas, accs, marker='o', label=name, color=col, linewidth=2)

    ax.axvline(x=10, color='grey', linestyle='--', alpha=0.6, label='γ=10 (2-block)')
    ax.set_xlabel("Gamma (darkness level)", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("Classifier Robustness to Gamma Darkening\n"
                 "Justifies CNN as the 2-block detector",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "gamma_sensitivity.png"), dpi=150)
    plt.close()
    print("Saved: gamma_sensitivity.png")


# ─── Per-class accuracy breakdown ─────────────────────────────────────────────

def plot_per_class_accuracy(models_dict: dict, X_test, y_test, class_names,
                            gamma=10, save_dir=None):
    """
    Grouped bar chart: for each species, show accuracy of each model under
    normal and dark conditions.

    Reveals which species are hardest to classify — e.g. orc vs lizard
    confusion under dark conditions (both are 'tank' cluster).
    """
    X_dark  = np.clip(X_test ** gamma, 0, 1)
    n_cls   = len(class_names)
    model_names = list(models_dict.keys())
    n_models    = len(model_names)

    # per_class[model][condition][class_idx] = accuracy
    normal_acc = {m: np.zeros(n_cls) for m in model_names}
    dark_acc   = {m: np.zeros(n_cls) for m in model_names}

    for name, pred_fn in models_dict.items():
        y_pred_n = pred_fn(X_test)
        y_pred_d = pred_fn(X_dark)
        for c in range(n_cls):
            mask = y_test == c
            if mask.sum() == 0:
                continue
            normal_acc[name][c] = accuracy_score(y_test[mask], y_pred_n[mask])
            dark_acc[name][c]   = accuracy_score(y_test[mask], y_pred_d[mask])

    x        = np.arange(n_cls)
    total_w  = 0.8
    bar_w    = total_w / (n_models * 2)
    colours  = ["#4C72B0", "#DD8452", "#55A868"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, acc_dict, scenario in zip(
            axes, [normal_acc, dark_acc],
            [f"1-block (normal)", f"2-block (dark, γ={gamma})"]):
        for i, (name, col) in enumerate(zip(model_names, colours)):
            offset = (i - n_models / 2 + 0.5) * bar_w
            bars = ax.bar(x + offset, acc_dict[name] * 100, bar_w,
                          label=name, color=col, alpha=0.85)
            for bar in bars:
                h = bar.get_height()
                if h > 5:
                    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                            f"{h:.0f}", ha='center', va='bottom', fontsize=6.5)

        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=20, ha='right', fontsize=9)
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 115)
        ax.set_title(scenario, fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, axis='y', alpha=0.3)

    fig.suptitle("Per-Species Classification Accuracy",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "per_class_accuracy.png"), dpi=150)
    plt.close()
    print("Saved: per_class_accuracy.png")
