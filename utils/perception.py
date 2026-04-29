"""
perception.py — Perception pipeline integrating Task 1 classifiers into Task 3.

At 1 block (clear image) : SVM classifies entity sprite at full brightness.
At 2 blocks (dark image)  : CNN classifies gamma-darkened entity sprite.

Narrative tradeoff
------------------
  CNN at 2 blocks  → advance warning, costs 5 energy, ~77% accuracy on dark images.
                     Correct combat AFTER a CNN scan earns +5 prepared bonus.
  SVM at 1 block   → close-range confirmation, no energy cost, ~99% accuracy.
                     No bonus — the robot had its chance to prepare earlier.

Both return cluster probabilities [p_tank, p_flying, p_smart] which become
the continuous entity sensor dims of the RL state vector.

Cluster ← species mapping (from Task 2 GMM clustering):
  tank   ← orc, lizard     (high strength / stench cluster)
  flying ← wingedrat       (flight = 1.0 cluster)
  smart  ← human, halfling (high intelligence cluster)

CNN/SVM output order matches CLASS_NAMES:
  [human(0), orc(1), lizard(2), wingedrat(3), halfling(4)]

So:
  p_tank   = probs[1] + probs[2]   (orc + lizard)
  p_flying = probs[3]              (wingedrat)
  p_smart  = probs[0] + probs[4]   (human + halfling)
"""

import os
import pickle
import random
import numpy as np
from PIL import Image

# ─── Constants ─────────────────────────────────────────────────────────────────

CLASS_NAMES  = ['human', 'orc', 'lizard', 'wingedrat', 'halfling']
IMG_SIZE     = 32
GAMMA_2BLOCK = 10   # darkness applied at 2-block range

# Which species folders represent each combat cluster
ENTITY_SPECIES = {
    'tank':   ['orc', 'lizard'],
    'flying': ['wingedrat'],
    'smart':  ['human', 'halfling'],
}

# Default model paths (relative to repo root)
_DEFAULT_SVM = os.path.join("task1", "results", "svm_model.pkl")
_DEFAULT_CNN = os.path.join("task1", "results", "cnn_model.keras")
_IMAGES_DIR  = "images"
_POOL_SIZE   = 40   # images per species pre-loaded into memory


# ─── Cluster probability helpers ───────────────────────────────────────────────

def _to_cluster_probs(raw_probs: np.ndarray) -> np.ndarray:
    """
    Aggregate 5-class softmax over species into 3 cluster probabilities.
    Works for both CNN (softmax output) and SVM (predict_proba output).

    Returns float32 array [p_tank, p_flying, p_smart].
    """
    p_tank   = float(raw_probs[1]) + float(raw_probs[2])   # orc + lizard
    p_flying = float(raw_probs[3])                          # wingedrat
    p_smart  = float(raw_probs[0]) + float(raw_probs[4])   # human + halfling
    return np.array([p_tank, p_flying, p_smart], dtype=np.float32)


# ─── Perception pipeline ───────────────────────────────────────────────────────

class PerceptionPipeline:
    """
    Loads trained SVM and CNN classifiers once at init.
    Pre-loads image pools per species for fast inference.

    Usage
    -----
        pipe = PerceptionPipeline()
        p = pipe.perceive_1block('tank')   # [p_tank, p_flying, p_smart] via SVM
        p = pipe.perceive_2block('tank')   # [p_tank, p_flying, p_smart] via CNN
    """

    def __init__(self, svm_path=_DEFAULT_SVM, cnn_path=_DEFAULT_CNN,
                 images_dir=_IMAGES_DIR):
        self._svm    = self._load_svm(svm_path)
        self._cnn    = self._load_cnn(cnn_path)
        self._pools  = self._build_pools(images_dir)
        n_loaded = sum(len(v) for v in self._pools.values())
        print(f"[Perception] SVM={'OK' if self._svm else 'FAIL'}  "
              f"CNN={'OK' if self._cnn else 'FAIL'}  "
              f"Image pool: {n_loaded} sprites across {len(self._pools)} species")

    # ── Model loading ──────────────────────────────────────────────────────────

    def _load_svm(self, path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"[Perception] SVM load failed: {e}")
            return None

    def _load_cnn(self, path):
        try:
            from tensorflow.keras.models import load_model
            return load_model(path)
        except Exception as e:
            print(f"[Perception] CNN load failed: {e}")
            return None

    # ── Image pool ────────────────────────────────────────────────────────────

    def _build_pools(self, images_dir):
        pools = {}
        for species in CLASS_NAMES:
            cls_dir = os.path.join(images_dir, species)
            if not os.path.isdir(cls_dir):
                pools[species] = []
                continue
            files = [f for f in os.listdir(cls_dir)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            chosen = random.sample(files, min(_POOL_SIZE, len(files)))
            imgs = []
            for fname in chosen:
                try:
                    img = (Image.open(os.path.join(cls_dir, fname))
                           .convert("RGB")
                           .resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))
                    imgs.append(np.array(img, dtype=np.float32) / 255.0)
                except Exception:
                    pass
            pools[species] = imgs
        return pools

    def _sample_image(self, entity_type: str) -> np.ndarray:
        species_list = ENTITY_SPECIES[entity_type]
        species      = random.choice(species_list)
        pool         = self._pools.get(species, [])
        if not pool:
            return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        return random.choice(pool).copy()

    # ── Public API ────────────────────────────────────────────────────────────

    def perceive_1block(self, entity_type: str) -> np.ndarray:
        """
        SVM classification at 1-block (full brightness).
        ~99% accuracy on clear images.
        Returns [p_tank, p_flying, p_smart].
        """
        if self._svm is None:
            return self._oracle_probs(entity_type)

        img  = self._sample_image(entity_type)
        flat = img.reshape(1, -1)

        if hasattr(self._svm, 'predict_proba'):
            raw = self._svm.predict_proba(flat)[0]
        else:
            pred = int(self._svm.predict(flat)[0])
            raw  = np.zeros(len(CLASS_NAMES), dtype=np.float32)
            raw[pred] = 1.0

        return _to_cluster_probs(raw)

    def perceive_2block(self, entity_type: str) -> np.ndarray:
        """
        CNN classification at 2-block (gamma-darkened image).
        ~77% accuracy on darkened images — intentionally noisier than SVM.
        Returns [p_tank, p_flying, p_smart].
        Falls back to SVM if CNN unavailable.
        """
        if self._cnn is None:
            return self.perceive_1block(entity_type)

        img  = self._sample_image(entity_type)
        dark = np.clip(img ** GAMMA_2BLOCK, 0.0, 1.0)
        raw  = self._cnn.predict(dark[None], verbose=0)[0]
        return _to_cluster_probs(raw)

    # ── Oracle fallback ───────────────────────────────────────────────────────

    @staticmethod
    def _oracle_probs(entity_type: str) -> np.ndarray:
        """Perfect classification — used when models unavailable."""
        p = np.zeros(3, dtype=np.float32)
        p[{'tank': 0, 'flying': 1, 'smart': 2}[entity_type]] = 1.0
        return p
