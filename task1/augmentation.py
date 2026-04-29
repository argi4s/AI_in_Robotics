import numpy as np


def adjust_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    """Apply gamma correction to a single image in [0, 1]."""
    return np.clip(image ** gamma, 0.0, 1.0)


def darken_dataset(X: np.ndarray, gamma: float = 10) -> np.ndarray:
    """Return a darkened copy of an entire dataset (used for test-time simulation)."""
    return np.clip(X ** gamma, 0.0, 1.0)


def build_augmented_dataset(X_train: np.ndarray, y_train: np.ndarray,
                             gammas: tuple = (5, 10)):
    """
    Deterministic augmentation — append fixed-γ darkened copies.
    Kept for reference and comparison against stochastic augmentation.
    """
    aug_imgs, aug_labels = [], []
    for gamma in gammas:
        aug_imgs.append(np.clip(X_train ** gamma, 0.0, 1.0))
        aug_labels.append(y_train)

    X_aug = np.concatenate([X_train] + aug_imgs, axis=0)
    y_aug = np.concatenate([y_train] + aug_labels, axis=0)
    print(f"Deterministic augmentation: {X_train.shape[0]} → {X_aug.shape[0]} "
          f"(gammas={gammas})")
    return X_aug, y_aug


def build_stochastic_augmented_dataset(X_train: np.ndarray, y_train: np.ndarray,
                                        gamma_range: tuple = (1.5, 12),
                                        n_copies: int = 2,
                                        seed: int = 42):
    rng = np.random.default_rng(seed)
    aug_imgs, aug_labels = [], []

    for copy_idx in range(n_copies):
        gammas = rng.uniform(gamma_range[0], gamma_range[1], size=len(X_train))
        darkened = np.stack([
            np.clip(X_train[i] ** gammas[i], 0.0, 1.0)
            for i in range(len(X_train))
        ]).astype(np.float32)
        aug_imgs.append(darkened)
        aug_labels.append(y_train)

    X_aug = np.concatenate([X_train]  + aug_imgs, axis=0)
    y_aug = np.concatenate([y_train]  + aug_labels, axis=0)
    print(f"Stochastic augmentation: {X_train.shape[0]} → {X_aug.shape[0]} "
          f"({n_copies} copies, γ ~ Uniform{gamma_range})")
    return X_aug, y_aug
