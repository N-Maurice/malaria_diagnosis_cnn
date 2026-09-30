"""Descriptive TensorBoard runs and HParams metadata, with no training side effects."""

import json
import math
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from .config import LOG_DIR, RANDOM_SEED

if TYPE_CHECKING:
    import tensorflow as tf

HPARAM_FIELDS = (
    "model_name",
    "learning_rate",
    "optimizer",
    "batch_size",
    "epochs",
    "augmentation",
    "dropout",
    "l2",
    "frozen_layers",
)
METRIC_NAMES = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "sensitivity",
    "specificity",
)


def make_run_name(model: str, experiment_number: int, description: str) -> str:
    """Validate and return <model>_exp_<NN>_<description>, never a generic run."""
    for value in (model, description):
        if not re.fullmatch(r"[a-z][a-z0-9_-]*", value):
            raise ValueError(
                "Use lowercase letters, numbers, underscores and hyphens; start with a letter."
            )
        if re.fullmatch(r"(?:test|run|experiment|final|final_final)\d*", value):
            raise ValueError(
                "Use a descriptive model and hypothesis/change, not a generic name."
            )
    if (
        isinstance(experiment_number, bool)
        or not isinstance(experiment_number, int)
        or experiment_number < 1
    ):
        raise ValueError("experiment_number must be a positive integer.")
    return f"{model}_exp_{experiment_number:02d}_{description}"


def create_tensorboard_callback(
    model: str,
    experiment_number: int,
    description: str,
    hparams: Mapping[str, Any],
    log_root: str | Path = LOG_DIR,
) -> tuple["tf.keras.callbacks.TensorBoard", Path]:
    """Reserve a new run, write HParams/JSON metadata and return (callback, path).

    Log each experiment's validation metrics with log_final_metrics after its run.
    Reserve test metrics for the final selected model, using prefix='test'. Add
    git commit, split identity and package snapshot path to hparams for provenance.
    Existing runs are never reused. Keras callback records train/validation scalars.
    """
    import tensorflow as tf
    from tensorboard.plugins.hparams import api as hp

    missing = set(HPARAM_FIELDS) - set(hparams)
    if missing:
        raise ValueError(f"Missing HParams fields: {sorted(missing)}")
    if hparams["model_name"] != model:
        raise ValueError("hparams model_name must match the run model.")
    values = {"seed": RANDOM_SEED, **dict(hparams)}
    values = {
        key: (
            value
            if isinstance(value, (str, bool, int, float))
            else json.dumps(value, sort_keys=True)
        )
        for key, value in values.items()
    }
    serialized = json.dumps(values, indent=2, allow_nan=False)
    run_name = make_run_name(model, experiment_number, description)
    root = Path(log_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / run_name
    path.mkdir(exist_ok=False)
    (path / "hparams.json").write_text(serialized + "\n", encoding="utf-8")
    # Per-run schemas avoid concurrent rewrites of a shared experiment schema.
    writer = tf.summary.create_file_writer(str(path))
    try:
        with writer.as_default():
            hp.hparams_config(
                hparams=[hp.HParam(key) for key in values],
                metrics=[hp.Metric(f"validation/{name}") for name in METRIC_NAMES],
            )
            hp.hparams(values, trial_id=run_name)
        writer.flush()
    finally:
        writer.close()
    return tf.keras.callbacks.TensorBoard(log_dir=str(path), histogram_freq=0), path


def log_final_metrics(
    run_dir: str | Path,
    metrics: Mapping[str, float],
    step: int = 0,
    prefix: str = "validation",
) -> None:
    """Write supplied scalar metrics into an existing run; reject nonfinite values."""
    import tensorflow as tf

    path = Path(run_dir)
    if not (path / "hparams.json").is_file():
        raise ValueError("Use a run directory created by create_tensorboard_callback.")
    if prefix not in {"train", "validation", "test"}:
        raise ValueError("prefix must be train, validation or test.")
    if step < 0 or not all(math.isfinite(float(value)) for value in metrics.values()):
        raise ValueError(
            "Use a nonnegative step and finite metrics; omit undefined metrics."
        )
    writer = tf.summary.create_file_writer(str(path))
    try:
        with writer.as_default():
            for name, value in metrics.items():
                tf.summary.scalar(f"{prefix}/{name}", float(value), step=step)
        writer.flush()
    finally:
        writer.close()
