import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def _flat(X: np.ndarray) -> np.ndarray:
    return X.reshape(len(X), -1)


# ─── Training ──────────────────────────────────────────────────────────────────

def train_knn(X_train: np.ndarray, y_train: np.ndarray,
              n_neighbors: int = 5) -> KNeighborsClassifier:
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(_flat(X_train), y_train)
    return knn


#Hyperparameter tunin

def tune_knn(X_train, y_train, X_test, y_test,
             k_values=(1, 3, 5, 7, 9, 11, 15, 21),
             save_dir=None) -> int:
    """
    Evaluate KNN for each k on the test set and return the best k.
    Saves a tuning curve plot to save_dir if provided.
    """
    accuracies = []
    for k in k_values:
        m = KNeighborsClassifier(n_neighbors=k)
        m.fit(_flat(X_train), y_train)
        acc = accuracy_score(y_test, m.predict(_flat(X_test)))
        accuracies.append(acc)
        print(f"  k={k:>2d}  acc={acc*100:.2f}%")

    best_k = k_values[int(np.argmax(accuracies))]
    print(f"Best k: {best_k}  (acc={max(accuracies)*100:.2f}%)")

    plt.figure(figsize=(7, 4))
    plt.plot(k_values, [a * 100 for a in accuracies], marker='o', color='steelblue')
    plt.xlabel("K")
    plt.ylabel("Accuracy (%)")
    plt.title("KNN — Hyperparameter Tuning")
    plt.xticks(k_values)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "knn_tuning.png"), dpi=150)
    plt.close()

    return best_k


#Evaluation 

def evaluate(model: KNeighborsClassifier, X_test, y_test,
             class_names, title="KNN", save_dir=None) -> float:
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
                xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{title} — Confusion Matrix")
    plt.tight_layout()
    if save_dir:
        fname = title.lower().replace(" ", "_") + "_cm.png"
        plt.savefig(os.path.join(save_dir, fname), dpi=150)
    plt.close()

    return acc
