import os
import re
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

_HERE      = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
IMAGES_DIR = os.path.join(_REPO_ROOT, "images")
SAVE_DIR   = os.path.join(_HERE, "results")
os.makedirs(SAVE_DIR, exist_ok=True)

CLASS_NAMES = ["human", "orc", "lizard", "wingedrat", "halfling"]
IMG_SIZE    = 32
K_VALUES    = [1, 3, 5]
TEST_SIZE   = 0.20
RANDOM_SEED = 42

SVM_C     = 10
SVM_GAMMA = "scale"


def load_with_groups(images_dir=IMAGES_DIR):
    X, y, groups, fnames = [], [], [], []
    _pattern = re.compile(r'^(.*?)(\d+)\.png$', re.IGNORECASE)

    for label, cls in enumerate(CLASS_NAMES):
        cls_dir = os.path.join(images_dir, cls)
        if not os.path.isdir(cls_dir):
            print(f"  [WARN] missing folder: {cls_dir}")
            continue
        for fname in sorted(os.listdir(cls_dir)):
            m = _pattern.match(fname)
            if not m:
                continue
            sprite_id = m.group(1).rstrip('_')
            fpath = os.path.join(cls_dir, fname)
            try:
                img = (Image.open(fpath)
                       .convert("RGB")
                       .resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))
                X.append(np.array(img, dtype=np.float32) / 255.0)
                y.append(label)
                groups.append(sprite_id)
                fnames.append(fname)
            except Exception:
                pass

    return (np.array(X),
            np.array(y),
            np.array(groups),
            np.array(fnames))


def make_splits(y, groups):
    idx_tr, idx_te = train_test_split(
        np.arange(len(y)), test_size=TEST_SIZE, random_state=RANDOM_SEED,
        stratify=y)
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE,
                            random_state=RANDOM_SEED)
    flat_dummy = np.zeros((len(y), 1))
    idx_tr_g, idx_te_g = next(gss.split(flat_dummy, y, groups=groups))
    return idx_tr, idx_te, idx_tr_g, idx_te_g


def evaluate_knn(flat, y, idx_tr, idx_te, idx_tr_g, idx_te_g):
    random_accs, group_accs = [], []
    for k in K_VALUES:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(flat[idx_tr], y[idx_tr])
        random_accs.append(accuracy_score(y[idx_te], knn.predict(flat[idx_te])))

        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(flat[idx_tr_g], y[idx_tr_g])
        group_accs.append(accuracy_score(y[idx_te_g], knn.predict(flat[idx_te_g])))

    print("\nKNN accuracy")
    print(f"  {'k':>3}  {'random':>8}  {'group':>8}  {'drop':>8}")
    print(f"  {'-'*35}")
    for k, ra, ga in zip(K_VALUES, random_accs, group_accs):
        print(f"  {k:>3}  {ra*100:>7.1f}%  {ga*100:>7.1f}%  {(ra-ga)*100:>+7.1f}pp")
    return random_accs, group_accs


def evaluate_svm(flat, y, idx_tr, idx_te, idx_tr_g, idx_te_g):
    print(f"\nSVM (RBF C={SVM_C}, gamma={SVM_GAMMA!r}) -- training on random split ...")
    t0 = time.time()
    svm_r = SVC(kernel="rbf", C=SVM_C, gamma=SVM_GAMMA)
    svm_r.fit(flat[idx_tr], y[idx_tr])
    acc_random = accuracy_score(y[idx_te], svm_r.predict(flat[idx_te]))
    print(f"  random split: {acc_random*100:.1f}%  ({time.time()-t0:.0f}s)")

    print("  training on group split ...")
    t0 = time.time()
    svm_g = SVC(kernel="rbf", C=SVM_C, gamma=SVM_GAMMA)
    svm_g.fit(flat[idx_tr_g], y[idx_tr_g])
    acc_group = accuracy_score(y[idx_te_g], svm_g.predict(flat[idx_te_g]))
    print(f"  group  split: {acc_group*100:.1f}%  ({time.time()-t0:.0f}s)")
    print(f"  drop: {(acc_random-acc_group)*100:+.1f}pp")
    return acc_random, acc_group


def plot_knn_accuracy(random_accs, group_accs):
    fig, ax = plt.subplots(figsize=(8, 5))
    x, w = np.arange(len(K_VALUES)), 0.35
    c_r, c_g = "#4C72B0", "#DD8452"

    bars_r = ax.bar(x - w/2, [v*100 for v in random_accs], w,
                    label="Random split", color=c_r, alpha=0.85, edgecolor='white')
    bars_g = ax.bar(x + w/2, [v*100 for v in group_accs],  w,
                    label="Group split",  color=c_g, alpha=0.85, edgecolor='white')

    for bar in list(bars_r) + list(bars_g):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.4,
                f"{h:.1f}%", ha='center', va='bottom', fontsize=9)

    drop = (random_accs[0] - group_accs[0]) * 100
    ax.annotate(f"k=1 drops\n{drop:.1f} pp",
                xy=(x[0] + w/2, group_accs[0]*100),
                xytext=(x[0] + w/2 + 0.55, group_accs[0]*100 + 12),
                arrowprops=dict(arrowstyle='->', color='#C44E52', lw=1.5),
                fontsize=9, color='#C44E52', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([f"k = {k}" for k in K_VALUES], fontsize=11)
    ax.set_ylabel("Test accuracy (%)", fontsize=11)
    ax.set_ylim(0, 110)
    ax.set_title("KNN: random split vs group split\n"
                 "(group split = sprite identity never crosses train/test boundary)",
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    out = os.path.join(SAVE_DIR, "knn_random_vs_group_accuracy.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def plot_knn_vs_svm(knn_random, knn_group, svm_random, svm_group):
    models     = ["KNN (k=1)", "SVM (RBF)"]
    acc_random = [knn_random * 100, svm_random * 100]
    acc_group  = [knn_group  * 100, svm_group  * 100]
    drops      = [(knn_random - knn_group) * 100,
                  (svm_random - svm_group) * 100]

    x, w = np.arange(len(models)), 0.30
    c_r, c_g = "#4C72B0", "#DD8452"

    fig, ax = plt.subplots(figsize=(7, 5))
    bars_r = ax.bar(x - w/2, acc_random, w, label="Random split",
                    color=c_r, alpha=0.85, edgecolor='white')
    bars_g = ax.bar(x + w/2, acc_group,  w, label="Group split",
                    color=c_g, alpha=0.85, edgecolor='white')

    for bar in list(bars_r) + list(bars_g):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                f"{h:.1f}%", ha='center', va='bottom', fontsize=10)

    for i, (drop, ra, ga) in enumerate(zip(drops, acc_random, acc_group)):
        brace_x = x[i] + w/2 + 0.06
        ax.annotate("",
                    xy=(brace_x, ga), xytext=(brace_x, ra),
                    arrowprops=dict(arrowstyle='<->', color='#C44E52',
                                   lw=1.4, shrinkA=0, shrinkB=0))
        ax.text(brace_x + 0.04, (ra + ga) / 2,
                f"-{drop:.1f} pp", va='center', ha='left',
                fontsize=9, color='#C44E52', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12)
    ax.set_ylabel("Test accuracy (%)", fontsize=11)
    ax.set_ylim(0, 115)
    ax.set_title("KNN vs SVM: random split vs group split\n"
                 "group split = sprite identity never crosses the boundary",
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    out = os.path.join(SAVE_DIR, "svm_knn_group_vs_random.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def plot_nn_examples(X, y, groups, fnames, idx_tr, idx_te, n_examples=10):
    flat = X.reshape(len(X), -1)
    knn1 = KNeighborsClassifier(n_neighbors=1)
    knn1.fit(flat[idx_tr], y[idx_tr])
    _, nn_indices = knn1.kneighbors(flat[idx_te])
    nn_train_idx  = idx_tr[nn_indices[:, 0]]

    same_sprite = np.array([
        groups[idx_te[i]] == groups[nn_train_idx[i]]
        for i in range(len(idx_te))
    ])
    same_sprite_rate = same_sprite.mean() * 100
    print(f"\nk=1 -- same-sprite nearest neighbour: {same_sprite_rate:.1f}% of test samples")

    rng = np.random.default_rng(RANDOM_SEED)
    same_idx  = np.where(same_sprite)[0]
    cross_idx = np.where(~same_sprite)[0]
    n_each    = n_examples // 2
    chosen    = np.concatenate([
        rng.choice(same_idx,  min(n_each, len(same_idx)),  replace=False),
        rng.choice(cross_idx, min(n_each, len(cross_idx)), replace=False),
    ])

    ncols = min(n_examples, len(chosen))
    fig, axes = plt.subplots(2, ncols, figsize=(ncols * 1.6 + 0.5, 4.5))

    for col, ci in enumerate(chosen[:ncols]):
        te_i   = idx_te[ci]
        tr_i   = nn_train_idx[ci]
        colour = "#2ca02c" if same_sprite[ci] else "#d62728"

        for row, img_idx in enumerate([te_i, tr_i]):
            ax = axes[row, col]
            ax.imshow(X[img_idx])
            ax.axis('off')
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(colour)
                spine.set_linewidth(3)
            if row == 0:
                short = fnames[img_idx].replace(CLASS_NAMES[y[img_idx]] + '_', '')
                ax.set_title(short, fontsize=6.5, pad=2)
            else:
                label = "same sprite" if same_sprite[ci] else "diff. sprite"
                ax.set_xlabel(label, fontsize=6.5, color=colour,
                              labelpad=2, fontweight='bold')

    axes[0, 0].set_ylabel("test", fontsize=8, rotation=0, labelpad=28)
    axes[1, 0].set_ylabel("NN",   fontsize=8, rotation=0, labelpad=28)

    green_patch = mpatches.Patch(color="#2ca02c",
                                 label=f"same sprite -- {same_sprite_rate:.0f}% of test set")
    red_patch   = mpatches.Patch(color="#d62728", label="different sprite")
    fig.legend(handles=[green_patch, red_patch],
               loc='lower center', ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        f"k=1 nearest neighbours under random split\n"
        f"{same_sprite_rate:.0f}% of test images have a same-sprite neighbour in training",
        fontsize=11, fontweight='bold')
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    out = os.path.join(SAVE_DIR, "knn_nn_examples.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    print("Loading dataset ...")
    X, y, groups, fnames = load_with_groups()
    print(f"  {len(X)} images, {len(np.unique(groups))} sprite identities, "
          f"{len(CLASS_NAMES)} classes")

    flat = X.reshape(len(X), -1)
    idx_tr, idx_te, idx_tr_g, idx_te_g = make_splits(y, groups)

    knn_random, knn_group = evaluate_knn(flat, y, idx_tr, idx_te, idx_tr_g, idx_te_g)
    svm_random, svm_group = evaluate_svm(flat, y, idx_tr, idx_te, idx_tr_g, idx_te_g)

    plot_knn_accuracy(knn_random, knn_group)
    plot_knn_vs_svm(knn_random[0], knn_group[0], svm_random, svm_group)
    plot_nn_examples(X, y, groups, fnames, idx_tr, idx_te, n_examples=10)

    print("\nSummary")
    print(f"  {'model':<12}  {'random':>8}  {'group':>8}  {'drop':>8}")
    print(f"  {'-'*42}")
    for k, ra, ga in zip(K_VALUES, knn_random, knn_group):
        print(f"  {'KNN k='+str(k):<12}  {ra*100:>7.1f}%  {ga*100:>7.1f}%  {(ra-ga)*100:>+7.1f}pp")
    print(f"  {'SVM (RBF)':<12}  {svm_random*100:>7.1f}%  {svm_group*100:>7.1f}%  "
          f"{(svm_random-svm_group)*100:>+7.1f}pp")

    print("\nDone -- see task1/results/")
