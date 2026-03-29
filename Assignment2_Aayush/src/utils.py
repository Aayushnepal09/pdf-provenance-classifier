import os
import numpy as np
from PIL import Image

# map each subfolder to a class label
SUBDIRS = {
    'word_pdfs_png': 0,
    'google_docs_pdfs_png': 1,
    'python_pdfs_png': 2
}

def load_dataset(base_dir, target_size=(200, 200)):
    """
    Loads images from the 3 class subfolders and returns flattened
    feature vectors + labels. Label is based on which folder the
    image came from.
    """
    X = []
    y = []

    for subdir, label in SUBDIRS.items():
        folder = os.path.join(base_dir, subdir)
        if not os.path.exists(folder):
            print(f"Warning: {folder} not found, skipping.")
            continue

        files = [f for f in os.listdir(folder) if f.endswith('.png')]
        print(f"  Loading {len(files)} images from {subdir}/ (label={label})...")

        for filename in files:
            try:
                img = Image.open(os.path.join(folder, filename)).convert('L')
                img = img.resize(target_size, Image.LANCZOS)
                X.append(np.array(img).flatten())
                y.append(label)
            except Exception as e:
                print(f"  Error loading {filename}: {e}")

    X = np.array(X)
    y = np.array(y)
    print(f"Loaded dataset: {X.shape[0]} samples — Word={np.sum(y==0)}, Google={np.sum(y==1)}, Python={np.sum(y==2)}")
    return X, y


def load_augmented_dataset(aug_base_dir, augmentation_type, target_size=(200, 200)):
    """
    Same as load_dataset but only grabs files ending with
    _aug_{augmentation_type}.png from each subfolder.
    """
    X = []
    y = []
    suffix = f"_aug_{augmentation_type}.png"

    for subdir, label in SUBDIRS.items():
        folder = os.path.join(aug_base_dir, subdir)
        if not os.path.exists(folder):
            print(f"Warning: {folder} not found, skipping.")
            continue

        files = [f for f in os.listdir(folder) if f.endswith(suffix)]
        for filename in files:
            try:
                img = Image.open(os.path.join(folder, filename)).convert('L')
                img = img.resize(target_size, Image.LANCZOS)
                X.append(np.array(img).flatten())
                y.append(label)
            except:
                pass  # skip broken files silently

    X = np.array(X)
    y = np.array(y)
    if len(X) > 0:
        print(f"  Loaded {len(X)} '{augmentation_type}' images — Word={np.sum(y==0)}, Google={np.sum(y==1)}, Python={np.sum(y==2)}")
    return X, y
