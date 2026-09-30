"""Image discovery, portable stratified manifests, and generic input pipelines."""

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    BATCH_SIZE,
    CLASS_MAPPING,
    IMAGE_EXTENSIONS,
    IMAGE_SIZE,
    PROJECT_ROOT,
    RANDOM_SEED,
    SPLIT_DIR,
    SPLIT_RATIOS,
)

if TYPE_CHECKING:
    import tensorflow as tf


def build_dataframe(
    dataset_dir: str | Path,
    path_root: str | Path = PROJECT_ROOT,
) -> pd.DataFrame:
    """Discover sorted images directly/recursively below the two class folders.

    Pass data/cell_images, not data. Paths are POSIX strings relative to path_root.
    Files outside path_root are rejected; choose a shared ancestor for external data.
    Extension filtering ignores Thumbs.db; image decoding is deferred to the pipeline.
    """
    dataset_dir, path_root = Path(dataset_dir).resolve(), Path(path_root).resolve()
    records = []
    for class_name, label in CLASS_MAPPING.items():
        folder = dataset_dir / class_name
        if not folder.is_dir():
            raise ValueError(
                f"Missing class directory: {folder}. Pass the parent of both class folders."
            )
        images = sorted(
            f
            for f in folder.rglob("*")
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not images:
            raise ValueError(f"No supported images found in {folder}.")
        for image in images:
            try:
                relative = image.resolve().relative_to(path_root).as_posix()
            except ValueError as exc:
                raise ValueError("Dataset images must be inside path_root.") from exc
            records.append({"filepath": relative, "label": label})
    result = pd.DataFrame(records)
    _validate_frame(result)
    return result


def _validate_frame(frame: pd.DataFrame) -> None:
    """Reject invalid labels, missing values and duplicate image paths."""
    if not {"filepath", "label"}.issubset(frame.columns) or frame.empty:
        raise ValueError(
            "Expected a nonempty dataframe with filepath and label columns."
        )
    if frame[["filepath", "label"]].isna().any().any():
        raise ValueError("Filepaths and labels must not be missing.")
    if not frame["label"].isin([0, 1]).all():
        raise ValueError("Labels must be 0 (Uninfected) or 1 (Parasitized).")
    if (
        not frame["filepath"]
        .map(lambda x: isinstance(x, str) and bool(x.strip()))
        .all()
    ):
        raise ValueError("Filepaths must be nonempty strings.")
    if frame["filepath"].duplicated().any():
        raise ValueError("Duplicate filepaths could leak across splits.")


def stratified_split(
    frame: pd.DataFrame,
    output_dir: str | Path = SPLIT_DIR,
    ratios: tuple[float, float, float] = SPLIT_RATIOS,
    seed: int = RANDOM_SEED,
    overwrite: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Save disjoint stratified train/val/test CSVs (70/15/15 by default).

    Integer rounding may slightly change ratios. Existing manifests are protected
    unless overwrite=True; all members must reuse the same approved manifests.
    """
    _validate_frame(frame)
    if (
        len(ratios) != 3
        or not np.isfinite(ratios).all()
        or min(ratios) <= 0
        or not np.isclose(sum(ratios), 1)
    ):
        raise ValueError("Provide three positive finite split ratios summing to 1.")
    if set(frame["label"]) != {0, 1}:
        raise ValueError("Stratification requires both classes.")
    output_dir = Path(output_dir)
    paths = [output_dir / f"{name}_files.csv" for name in ("train", "val", "test")]
    if not overwrite and any(path.exists() for path in paths):
        raise FileExistsError(
            "Split manifests already exist. Reuse them or explicitly approve overwrite=True."
        )
    # Sorting makes seed behavior independent of dataframe row order.
    frame = frame.sort_values("filepath").reset_index(drop=True)
    try:
        train, remainder = train_test_split(
            frame,
            test_size=ratios[1] + ratios[2],
            stratify=frame.label,
            random_state=seed,
        )
        val, test = train_test_split(
            remainder,
            test_size=ratios[2] / (ratios[1] + ratios[2]),
            stratify=remainder.label,
            random_state=seed,
        )
    except ValueError as exc:
        raise ValueError(
            "Insufficient samples per class for the requested stratified splits."
        ) from exc
    parts = tuple(part.reset_index(drop=True) for part in (train, val, test))
    if any(set(part.label) != {0, 1} for part in parts):
        raise ValueError("Each split must contain both classes; supply more samples.")
    output_dir.mkdir(parents=True, exist_ok=True)
    for part, path in zip(parts, paths):
        part.to_csv(path, index=False)
    return parts


def build_tf_dataset(
    frame: pd.DataFrame,
    image_size: tuple[int, int] = IMAGE_SIZE,
    batch_size: int = BATCH_SIZE,
    shuffle: bool = False,
    augmentation: Callable | None = None,
    preprocessing_fn: Callable | None = None,
    path_root: str | Path = PROJECT_ROOT,
    seed: int = RANDOM_SEED,
) -> "tf.data.Dataset":
    """Decode RGB images and return batches of (float images, float labels).

    Augmentation is an optional callable(image), applied only when supplied.
    Pass it only for training; wrap Keras augmentation with training=True yourself.
    It receives resized float32 pixels in [0,255]. preprocessing_fn runs afterward
    on that same scale; without it pixels are divided by 255. Thus transfer-model
    preprocessing replaces default normalization (no double scaling).
    Validation/test should use shuffle=False and augmentation=None.
    """
    import tensorflow as tf

    _validate_frame(frame)
    if len(image_size) != 2 or any(
        not isinstance(x, int) or x <= 0 for x in image_size
    ):
        raise ValueError("image_size must contain two positive integers.")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if augmentation is not None and not callable(augmentation):
        raise TypeError("augmentation must be a callable or None.")
    if preprocessing_fn is not None and not callable(preprocessing_fn):
        raise TypeError("preprocessing_fn must be a callable or None.")
    root = Path(path_root).resolve()
    paths = []
    for value in frame.filepath:
        relative = Path(value)
        path = (root / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(root):
            raise ValueError("Manifest paths must be relative to and inside path_root.")
        if not path.is_file():
            raise FileNotFoundError(path)
        paths.append(str(path))
    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, frame.label.to_numpy(dtype="float32"))
    )
    if shuffle:
        dataset = dataset.shuffle(len(frame), seed=seed, reshuffle_each_iteration=True)

    def decode(
        path: "tf.Tensor", label: "tf.Tensor"
    ) -> tuple["tf.Tensor", "tf.Tensor"]:
        image = tf.io.decode_image(
            tf.io.read_file(path), channels=3, expand_animations=False
        )
        image.set_shape((None, None, 3))
        image = tf.image.resize(tf.cast(image, tf.float32), image_size)
        if augmentation is not None:
            image = augmentation(image)
        image = (
            preprocessing_fn(image) if preprocessing_fn is not None else image / 255.0
        )
        return image, tf.reshape(label, (1,))

    return (
        dataset.map(decode, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
