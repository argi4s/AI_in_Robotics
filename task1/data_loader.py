import os
import numpy as np
from PIL import Image

IMG_SIZE    = 32
CLASS_NAMES = ["human", "orc", "lizard", "wingedrat", "halfling"]

# Repo root is one level above this file
_REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_IMAGES_DIR = os.path.join(_REPO_ROOT, "images")


def load_dataset(data_dir=_IMAGES_DIR, img_size=IMG_SIZE,
                 class_names=CLASS_NAMES, cache_dir=None):

    if cache_dir:
        x_cache = os.path.join(cache_dir, "X.npy")
        y_cache = os.path.join(cache_dir, "y.npy")
        if os.path.exists(x_cache) and os.path.exists(y_cache):
            print("Loading cached dataset...")
            return np.load(x_cache), np.load(y_cache), class_names

    print(f"Loading images from {data_dir} ...")
    X, y = [], []

    for label, cls in enumerate(class_names):
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_dir):
            print(f"  [WARNING] class folder not found: {cls_dir}")
            continue
        count = 0
        for fname in os.listdir(cls_dir):
            fpath = os.path.join(cls_dir, fname)
            try:
                img = (Image.open(fpath)
                       .convert("RGB")
                       .resize((img_size, img_size), Image.BILINEAR))
                X.append(np.array(img, dtype=np.float32) / 255.0)
                y.append(label)
                count += 1
            except Exception:
                pass
        print(f"  {cls:<12s} {count:>5d} images")

    X = np.array(X)
    y = np.array(y)
    print(f"Dataset: {X.shape}  labels: {y.shape}")

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        np.save(x_cache, X)
        np.save(y_cache, y)
        print(f"Cached to {cache_dir}")

    return X, y, class_names
