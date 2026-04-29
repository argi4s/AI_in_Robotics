import os
import numpy as np
import matplotlib.pyplot as plt

try:
    import tensorflow as tf
    from tensorflow.keras import layers
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("[WARNING] TensorFlow not installed — CNN unavailable.")


# ─── Architecture ──────────────────────────────────────────────────────────────

def build_cnn(input_shape=(32, 32, 3), num_classes=5,
              learning_rate=3e-4) -> "tf.keras.Model":
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required for the CNN classifier.")

    model = tf.keras.Sequential([
        tf.keras.Input(shape=input_shape),

        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),

        layers.GlobalAveragePooling2D(),

        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),

        layers.Dense(num_classes,  activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ─── Training ──────────────────────────────────────────────────────────────────

def train_cnn(X_train, y_train, X_val, y_val,
              num_classes: int, epochs: int = 15,
              batch_size: int = 32, patience: int = 5,
              save_dir=None):

    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required for the CNN classifier.")

    model = build_cnn(input_shape=X_train.shape[1:], num_classes=num_classes)
    model.summary()

    datagen = ImageDataGenerator(zoom_range=0.1, horizontal_flip=True)

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True
    )

    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=1,
    )

    _plot_training(history, save_dir)
    return model, history


def _plot_training(history, save_dir=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"],     label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Val")
    axes[0].set_title("CNN — Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["loss"],     label="Train")
    axes[1].plot(history.history["val_loss"], label="Val")
    axes[1].set_title("CNN — Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "cnn_training.png"), dpi=150)
    plt.close()


# ─── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, X_test, y_test,
             class_names, title="CNN", save_dir=None) -> float:
    """
    Evaluate the CNN; print accuracy + per-class report; save confusion matrix.
    Returns accuracy as a float in [0, 1].
    """
    import seaborn as sns
    from sklearn.metrics import (accuracy_score, classification_report,
                                 confusion_matrix)

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n{'─'*50}")
    print(f"{title}  Accuracy: {acc*100:.2f}%")
    print(f"{'─'*50}")
    print(classification_report(y_test, y_pred, target_names=class_names))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=class_names, yticklabels=class_names, cmap='Greens')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{title} — Confusion Matrix")
    plt.tight_layout()
    if save_dir:
        fname = title.lower().replace(" ", "_") + "_cm.png"
        plt.savefig(os.path.join(save_dir, fname), dpi=150)
    plt.close()

    return acc
