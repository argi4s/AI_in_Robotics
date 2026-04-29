import os
import pickle
import numpy as np

ARTEFACTS_PATH = os.path.join(os.path.dirname(__file__), "results", "gmm_artefacts.pkl")


def enemy_behavior_gmm(enemy_idx: int, probs: np.ndarray, cluster_names: list) -> dict:
    """
    Returns soft cluster membership percentages for a single enemy.

    Parameters
    ----------
    enemy_idx     : row index into the dataset
    probs         : (N, n_clusters) soft probability array from GMM
    cluster_names : list of cluster label strings, length n_clusters

    Returns
    -------
    dict mapping cluster name → % membership  (values sum to 100)

    Example
    -------
    >>> enemy_behavior_gmm(0, probs, ['tank', 'smart', 'flying'])
    {'tank': 94.74, 'smart': 5.26, 'flying': 0.0}
    """
    p = probs[enemy_idx]
    return {cluster_names[j]: round(float(p[j]) * 100, 2) for j in range(len(cluster_names))}


def dominant_behavior(enemy_idx: int, probs: np.ndarray, cluster_names: list) -> str:
    """Return the single dominant cluster name for an enemy."""
    return cluster_names[int(np.argmax(probs[enemy_idx]))]


def load_artefacts(path: str = ARTEFACTS_PATH) -> dict:
    """Load the GMM artefacts produced by run_pca_clustering.py."""
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_behavior(feature_vector: np.ndarray, artefacts: dict) -> dict:
    """
    Classify a new entity at RL runtime given its raw sensor feature vector.

    Applies the same impute → scale → PCA → GMM pipeline fitted during Task 2.

    Parameters
    ----------
    feature_vector : 1-D array of raw sensor features (same columns as training)
    artefacts      : dict loaded from gmm_artefacts.pkl

    Returns
    -------
    dict mapping cluster name → % membership
    """
    x = artefacts["imputer"].transform([feature_vector])
    x = artefacts["scaler"].transform(x)
    x = artefacts["pca_2d"].transform(x)
    probs = artefacts["gmm"].predict_proba(x)[0]
    names = artefacts["cluster_names"]
    return {names[j]: round(float(probs[j]) * 100, 2) for j in range(len(names))}
