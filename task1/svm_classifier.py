import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def _flat(X: np.ndarray) -> np.ndarray:
    return X.reshape(len(X), -1)


# ─── Training ──────────────────────────────────────────────────────────────────

def train_svm(X_train, y_train, C: float = 1.0,
              gamma: str = "scale") -> SVC:
    print(f"Training SVM (C={C}, gamma={gamma}) ...")
    svm = SVC(kernel="rbf", C=C, gamma=gamma, probability=True)
    svm.fit(_flat(X_train), y_train)
    print("SVM training complete.")
    return svm


# ─── Hyperparameter tuning ─────────────────────────────────────────────────────

def tune_svm(X_train, y_train, subset: int = 2000,
             n_iter: int = 4, cv: int = 3) -> SVC:
    """
    RandomizedSearchCV over C and gamma on a small subset for speed.
    Returns the best fitted estimator (trained on the subset).
    """
    X_s = _flat(X_train)[:subset]
    y_s = y_train[:subset]

    param_dist = {
        "C":      [0.1, 1, 10],
        "gamma":  ["scale", 0.01],
        "kernel": ["rbf"],
    }

    search = RandomizedSearchCV(
        SVC(probability=True),
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        verbose=1,
        n_jobs=-1,
        random_state=42,
    )

    t0 = time.time()
    search.fit(X_s, y_s)
    elapsed = (time.time() - t0) / 60

    print(f"\nSVM tuning done in {elapsed:.1f} min")
    print(f"Best params: {search.best_params_}")
    return search.best_estimator_, search.best_params_


# ─── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model: SVC, X_test, y_test,
             class_names, title="SVM", save_dir=None) -> float:
    """
    Print classification report and save a confusion matrix.
    Returns accuracy as a float in [0, 1].
    """
    y_pred = model.predict(_flat(X_test))
    acc = accuracy_score(y_test, y_pred)

    print(f"\n{'─'*50}")
    print(f"{title}  Accuracy: {acc*100:.2f}%")
    print(f"{'─'*50}")
    print(classification_report(y_test, y_pred, target_names=class_names))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=class_names, yticklabels=class_names, cmap='Oranges')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{title} — Confusion Matrix")
    plt.tight_layout()
    if save_dir:
        fname = title.lower().replace(" ", "_") + "_cm.png"
        plt.savefig(os.path.join(save_dir, fname), dpi=150)
    plt.close()

    return acc
