"""Portable defaults; paths are resolved from this file, not the working directory."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = DATA_DIR / "cell_images"
SPLIT_DIR = PROJECT_ROOT / "splits"
LOG_DIR = PROJECT_ROOT / "logs" / "fit"
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
RANDOM_SEED = 42
CLASS_MAPPING = {"Uninfected": 0, "Parasitized": 1}
SPLIT_RATIOS = (0.70, 0.15, 0.15)
IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".bmp"})


def set_random_seed(seed: int = RANDOM_SEED, deterministic: bool = True) -> None:
    """Seed Python, NumPy and TensorFlow; optionally enable deterministic TF ops."""
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    if deterministic:
        tf.config.experimental.enable_op_determinism()
