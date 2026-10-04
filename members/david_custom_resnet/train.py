"""Experiment runner for David's Custom ResNet: one call = one logged experiment.

Each run trains on the shared train manifest, selects weights on validation
(early stopping on validation loss), logs everything to TensorBoard and saves
artifacts under ``<output_root>/<run_name>/``:

- ``config.json``            full experiment configuration and provenance
- ``best.weights.h5``        best-validation-loss weights (never committed)
- ``history.csv``            per-epoch training/validation scalars
- ``val_predictions.csv``    validation probabilities for the restored weights
- ``metrics.json``           validation metrics (threshold 0.5)

A row is appended to ``<output_root>/experiments_summary.csv`` per run. The test
set is touched only by ``evaluate_on_test`` after the final model is chosen.
"""

import dataclasses
import hashlib
import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from src.config import PROJECT_ROOT, RANDOM_SEED, SPLIT_DIR, set_random_seed
from src.data_utils import build_tf_dataset
from src.eval_utils import evaluate_binary
from src.logging_utils import create_tensorboard_callback, log_final_metrics, make_run_name

from .model import MODEL_NAME, build_model

AUGMENTATIONS = ("none", "flip_rot", "flip_rot_color")
OPTIMIZERS = ("adam", "sgd", "rmsprop")
SCHEDULES = ("constant", "reduce_on_plateau", "cosine")


@dataclass(frozen=True)
class ExperimentConfig:
    """Everything that defines a run; change one or two fields per experiment."""

    number: int
    description: str
    hypothesis: str = ""
    stage_filters: tuple[int, ...] = (32, 64, 128)
    blocks_per_stage: tuple[int, ...] = (1, 1, 1)
    use_skip: bool = True
    use_batchnorm: bool = True
    dropout: float = 0.0
    l2: float = 0.0
    augmentation: str = "none"
    optimizer: str = "adam"
    learning_rate: float = 1e-3
    lr_schedule: str = "constant"
    batch_size: int = 32
    epochs: int = 20
    early_stopping_patience: int = 5
    image_size: tuple[int, int] = (128, 128)
    debug_subset: int | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.augmentation not in AUGMENTATIONS:
            raise ValueError(f"augmentation must be one of {AUGMENTATIONS}")
        if self.optimizer not in OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {OPTIMIZERS}")
        if self.lr_schedule not in SCHEDULES:
            raise ValueError(f"lr_schedule must be one of {SCHEDULES}")

    @property
    def run_name(self) -> str:
        return make_run_name(MODEL_NAME, self.number, self.description)

    def next(self, number: int, description: str, hypothesis: str, **changes) -> "ExperimentConfig":
        """Derive the next experiment from this one, changing only ``changes``."""
        return dataclasses.replace(
            self, number=number, description=description, hypothesis=hypothesis, **changes
        )


# --- data -------------------------------------------------------------------

def _flip_rot(image: tf.Tensor) -> tf.Tensor:
    """Cells have no canonical orientation, so flips and 90-degree turns preserve labels."""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    return tf.image.rot90(image, k=tf.random.uniform([], 0, 4, dtype=tf.int32))


def _flip_rot_color(image: tf.Tensor) -> tf.Tensor:
    """Add mild brightness/contrast jitter to mimic staining and illumination variation."""
    image = _flip_rot(image)
    image = tf.image.random_brightness(image, max_delta=20.0)  # pixels are in [0, 255]
    image = tf.image.random_contrast(image, 0.85, 1.15)
    return tf.clip_by_value(image, 0.0, 255.0)


_AUGMENT_FNS = {"none": None, "flip_rot": _flip_rot, "flip_rot_color": _flip_rot_color}


def load_split(name: str, debug_subset: int | None = None) -> pd.DataFrame:
    """Read a shared manifest; ``debug_subset`` takes a small balanced sample for smoke tests."""
    frame = pd.read_csv(SPLIT_DIR / f"{name}_files.csv")
    if debug_subset:
        frame = pd.concat(
            frame[frame.label == label].sample(debug_subset // 2, random_state=RANDOM_SEED)
            for label in (0, 1)
        ).reset_index(drop=True)
    return frame


def make_datasets(config: ExperimentConfig) -> tuple[tf.data.Dataset, tf.data.Dataset, pd.DataFrame]:
    """Return (augmented shuffled train, ordered val, val frame)."""
    common = dict(image_size=config.image_size, batch_size=config.batch_size)
    train = build_tf_dataset(
        load_split("train", config.debug_subset), shuffle=True,
        augmentation=_AUGMENT_FNS[config.augmentation], **common,
    )
    val_frame = load_split("val", config.debug_subset)
    return train, build_tf_dataset(val_frame, **common), val_frame


# --- model/optimizer --------------------------------------------------------

def make_model(config: ExperimentConfig) -> tf.keras.Model:
    return build_model(
        image_size=config.image_size,
        stage_filters=config.stage_filters,
        blocks_per_stage=config.blocks_per_stage,
        use_batchnorm=config.use_batchnorm,
        use_skip=config.use_skip,
        dropout=config.dropout,
        l2=config.l2,
    )


def make_optimizer(config: ExperimentConfig, steps_per_epoch: int) -> tf.keras.optimizers.Optimizer:
    lr = config.learning_rate
    if config.lr_schedule == "cosine":
        lr = tf.keras.optimizers.schedules.CosineDecay(lr, decay_steps=steps_per_epoch * config.epochs)
    if config.optimizer == "sgd":
        return tf.keras.optimizers.SGD(lr, momentum=0.9, nesterov=True)
    if config.optimizer == "rmsprop":
        return tf.keras.optimizers.RMSprop(lr)
    return tf.keras.optimizers.Adam(lr)


def compile_model(model: tf.keras.Model, optimizer) -> None:
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )


# --- provenance -------------------------------------------------------------

def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _split_identity() -> str:
    digest = hashlib.sha256()
    for name in ("train", "val", "test"):
        digest.update((SPLIT_DIR / f"{name}_files.csv").read_bytes())
    return digest.hexdigest()[:12]


def _hparams(config: ExperimentConfig) -> dict:
    gpus = tf.config.list_physical_devices("GPU")
    return {
        "model_name": MODEL_NAME,
        "learning_rate": config.learning_rate,
        "optimizer": config.optimizer,
        "batch_size": config.batch_size,
        "epochs": config.epochs,
        "augmentation": config.augmentation,
        "dropout": config.dropout,
        "l2": config.l2,
        "frozen_layers": "none (trained from scratch)",
        "batch_normalization": config.use_batchnorm,
        "skip_connections": config.use_skip,
        "stage_filters": "-".join(map(str, config.stage_filters)),
        "blocks_per_stage": "-".join(map(str, config.blocks_per_stage)),
        "lr_schedule": config.lr_schedule,
        "early_stopping_patience": config.early_stopping_patience,
        "image_size": f"{config.image_size[0]}x{config.image_size[1]}",
        "git_commit": _git_commit(),
        "split_sha256": _split_identity(),
        "tensorflow": tf.__version__,
        "device": "GPU" if gpus else "CPU",
        "debug_subset": config.debug_subset or 0,
    }


# --- running ----------------------------------------------------------------

def run_experiment(
    config: ExperimentConfig,
    output_root: str | Path,
    log_root: str | Path,
    verbose: int = 2,
) -> dict:
    """Train, select on validation, log and save one experiment; return its summary row.

    If the run's artifacts already exist (e.g. after a Colab reconnect), the saved
    summary is returned instead of retraining. Delete the run folders to redo it.
    """
    run_dir = Path(output_root) / config.run_name
    if (run_dir / "metrics.json").is_file():
        print(f"{config.run_name}: already complete, loading saved results.")
        return json.loads((run_dir / "metrics.json").read_text())
    if (Path(log_root) / config.run_name).exists():
        raise FileExistsError(
            f"TensorBoard run {config.run_name} exists without finished results "
            "(interrupted run?). Delete that log folder and its output folder, then rerun."
        )

    tf.keras.backend.clear_session()
    # Unseeded tf.image random ops are not allowed under op determinism, so seeds
    # are fixed but GPU kernels are left non-deterministic for speed.
    set_random_seed(RANDOM_SEED, deterministic=False)
    train_ds, val_ds, val_frame = make_datasets(config)
    steps = int(train_ds.cardinality())
    model = make_model(config)
    compile_model(model, make_optimizer(config, steps))

    hparams = _hparams(config)
    tensorboard, log_dir = create_tensorboard_callback(
        MODEL_NAME, config.number, config.description, hparams, log_root=log_root
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(
        {"run_name": config.run_name, "config": dataclasses.asdict(config),
         "hparams": hparams, "parameters": model.count_params()}, indent=2,
    ))
    callbacks = [
        tensorboard,
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=config.early_stopping_patience, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(run_dir / "best.weights.h5"), monitor="val_loss", save_best_only=True,
            save_weights_only=True,
        ),
        tf.keras.callbacks.CSVLogger(str(run_dir / "history.csv")),
    ]
    if config.lr_schedule == "reduce_on_plateau":
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1
        ))

    start = time.time()
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=config.epochs, callbacks=callbacks, verbose=verbose
    )
    minutes = (time.time() - start) / 60
    model.load_weights(run_dir / "best.weights.h5")

    probabilities = model.predict(val_ds, verbose=0).reshape(-1)
    pd.DataFrame({**val_frame, "probability": probabilities}).to_csv(
        run_dir / "val_predictions.csv", index=False
    )
    metrics = evaluate_binary(val_frame.label.to_numpy(), probabilities)
    log_final_metrics(log_dir, {k: v for k, v in metrics.items() if np.isfinite(v)})

    val_loss = history.history["val_loss"]
    best_epoch = int(np.argmin(val_loss))
    summary = {
        "run_name": config.run_name,
        "hypothesis": config.hypothesis,
        **{f"val_{k}": round(v, 4) for k, v in metrics.items()},
        "best_epoch": best_epoch + 1,
        "epochs_completed": len(val_loss),
        "train_accuracy_at_best": round(history.history["accuracy"][best_epoch], 4),
        "val_accuracy_at_best": round(history.history["val_accuracy"][best_epoch], 4),
        "train_loss_at_best": round(history.history["loss"][best_epoch], 4),
        "val_loss_at_best": round(val_loss[best_epoch], 4),
        "parameters": model.count_params(),
        "train_minutes": round(minutes, 1),
    }
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    summary_csv = Path(output_root) / "experiments_summary.csv"
    pd.DataFrame([summary]).to_csv(
        summary_csv, mode="a", header=not summary_csv.exists(), index=False
    )
    return summary


def load_trained_model(run_name: str, output_root: str | Path) -> tuple[tf.keras.Model, ExperimentConfig]:
    """Rebuild a finished run's architecture and load its best weights."""
    saved = json.loads((Path(output_root) / run_name / "config.json").read_text())["config"]
    config = ExperimentConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in saved.items()})
    model = make_model(config)
    model.load_weights(Path(output_root) / run_name / "best.weights.h5")
    return model, config


def evaluate_on_test(
    run_name: str, output_root: str | Path, log_root: str | Path, threshold: float = 0.5
) -> tuple[dict, pd.DataFrame]:
    """Evaluate the FINAL selected run on the held-out test set, once.

    Saves ``test_predictions.csv``/``test_metrics.json`` and logs ``test/*`` scalars.
    """
    run_dir = Path(output_root) / run_name
    model, config = load_trained_model(run_name, output_root)
    test_frame = load_split("test", config.debug_subset)
    test_ds = build_tf_dataset(test_frame, image_size=config.image_size, batch_size=config.batch_size)
    probabilities = model.predict(test_ds, verbose=0).reshape(-1)
    predictions = test_frame.assign(probability=probabilities)
    predictions.to_csv(run_dir / "test_predictions.csv", index=False)
    metrics = evaluate_binary(test_frame.label.to_numpy(), probabilities, threshold)
    (run_dir / "test_metrics.json").write_text(json.dumps({"threshold": threshold, **metrics}, indent=2))
    log_final_metrics(Path(log_root) / run_name, {k: v for k, v in metrics.items() if np.isfinite(v)},
                      prefix="test")
    return metrics, predictions
