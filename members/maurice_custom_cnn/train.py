"""Validation-only experiments and a frozen, cached held-out evaluation."""
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time
import uuid

import numpy as np
import pandas as pd
from src.config import PROJECT_ROOT, SPLIT_DIR, RANDOM_SEED, set_random_seed
from src.data_utils import build_tf_dataset
from src.eval_utils import evaluate_binary
from src.logging_utils import create_tensorboard_callback, log_final_metrics, make_run_name
from .model import build_model

MODEL_NAME = "additional_custom_cnn"


@dataclass(frozen=True)
class ExperimentConfig:
    number: int = 1
    description: str = "baseline"
    hypothesis: str = "Establish the unregularized plain CNN baseline."
    image_size: tuple = (128, 128)
    filters: tuple = (32, 64, 128)
    convolutions_per_block: int = 2
    use_batchnorm: bool = False
    dropout: float = 0.0
    l2: float = 0.0
    augmentation: bool = False
    optimizer: str = "adam"
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 25
    patience: int = 5
    lr_schedule: bool = False

    @property
    def run_name(self):
        return make_run_name(MODEL_NAME, self.number, self.description)


def planned_experiments():
    """Controlled comparisons: 2 vs 1; 3 vs 2; all later runs vs 3."""
    base = ExperimentConfig()
    bn = replace(base, number=2, description="batchnorm", use_batchnorm=True,
                 hypothesis="Batch normalization improves optimization versus experiment 01.")
    reg = replace(bn, number=3, description="dropout_l2", dropout=0.3, l2=1e-4,
                  hypothesis="Combined dropout and L2 reduce overfitting versus experiment 02.")
    return [base, bn, reg,
            replace(reg, number=4, description="augmentation", augmentation=True,
                    hypothesis="Orientation and scale variation improve generalization versus 03."),
            replace(reg, number=5, description="lr_3e-4", learning_rate=3e-4,
                    hypothesis="A smaller Adam learning rate improves convergence versus 03."),
            replace(reg, number=6, description="sgd_momentum", optimizer="sgd",
                    learning_rate=1e-2,
                    hypothesis="SGD with momentum and its specified learning rate competes with Adam in 03."),
            replace(reg, number=7, description="batch64", batch_size=64,
                    hypothesis="A larger batch changes convergence and validation performance versus 03."),
            replace(reg, number=8, description="wider_aug_plateau", filters=(32, 64, 128, 256),
                    augmentation=True, lr_schedule=True,
                    hypothesis="A combined deeper, augmented, scheduled candidate improves on 03; effects cannot be isolated.")]


def split_identity():
    return {name: hashlib.sha256((SPLIT_DIR / f"{name}_files.csv").read_bytes()).hexdigest()
            for name in ("train", "val", "test")}


def load_split(name):
    if name not in ("train", "val", "test"):
        raise ValueError("Unknown split.")
    return pd.read_csv(SPLIT_DIR / f"{name}_files.csv")


def environment_snapshot():
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=PROJECT_ROOT,
                                    capture_output=True, text=True, check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unknown", True
    return dict(python=platform.python_version(), git_commit=commit, working_tree_dirty=dirty,
                packages={d.metadata['Name']: d.version for d in importlib.metadata.distributions()})


def make_model(cfg):
    return build_model(image_size=tuple(cfg.image_size), filters=tuple(cfg.filters),
                       convolutions_per_block=cfg.convolutions_per_block,
                       use_batchnorm=cfg.use_batchnorm, dropout=cfg.dropout, l2=cfg.l2)


def compile_model(model, cfg):
    import tensorflow as tf
    if cfg.optimizer not in ("adam", "sgd"):
        raise ValueError("Supported optimizers: adam, sgd.")
    optimizer = (tf.keras.optimizers.Adam(cfg.learning_rate) if cfg.optimizer == "adam"
                 else tf.keras.optimizers.SGD(cfg.learning_rate, momentum=0.9, nesterov=True))
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=[
        tf.keras.metrics.BinaryAccuracy(name="accuracy"), tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"), tf.keras.metrics.AUC(name="auc")])


def make_datasets(cfg):
    import tensorflow as tf
    augmentation = None
    if cfg.augmentation:
        augment = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=42),
            tf.keras.layers.RandomRotation(0.1, seed=43),
            tf.keras.layers.RandomZoom(0.1, seed=44)])
        augmentation = lambda image: augment(image, training=True)
    common = dict(image_size=tuple(cfg.image_size), batch_size=cfg.batch_size)
    return (build_tf_dataset(load_split("train"), shuffle=True, augmentation=augmentation, **common),
            build_tf_dataset(load_split("val"), **common))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def archive_incomplete_run(run_name, output_root, log_root):
    """Preserve interrupted artifacts before restarting from a fresh model."""
    run = Path(output_root) / run_name
    if (run / "metrics.json").exists():
        raise ValueError("Completed experiments cannot be archived for restart.")
    suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex
    for root in dict.fromkeys((Path(output_root), Path(log_root))):
        source = root / run_name
        if source.exists():
            destination = root / "archived_incomplete" / f"{run_name}_{suffix}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.rename(destination)
            print(f"Preserved incomplete run: {destination}")


def run_experiment(cfg, output_root, log_root, verbose=2, *, restart_incomplete=False):
    """Reuse completed results; optionally archive interrupted runs and retrain."""
    import tensorflow as tf
    run = Path(output_root) / cfg.run_name
    config = json.loads(json.dumps(asdict(cfg)))
    identity = split_identity()
    if (run / "metrics.json").exists():
        saved = json.loads((run / "config.json").read_text())
        if saved["config"] != config or saved["splits"] != identity:
            raise ValueError("Saved experiment configuration or manifests changed; use a new run number.")
        return json.loads((run / "metrics.json").read_text())
    if run.exists() or (Path(log_root) / cfg.run_name).exists():
        if not restart_incomplete:
            raise FileExistsError(
                f"Incomplete run exists: {run}. Use restart_incomplete=True to archive "
                "its results and logs and restart training, or choose a new experiment number.")
        archive_incomplete_run(cfg.run_name, output_root, log_root)
    tf.keras.backend.clear_session()
    set_random_seed(RANDOM_SEED)
    train_ds, val_ds = make_datasets(cfg)
    model = make_model(cfg)
    compile_model(model, cfg)
    snapshot = environment_snapshot()
    params = dict(model_name=MODEL_NAME, learning_rate=cfg.learning_rate, optimizer=cfg.optimizer,
                  batch_size=cfg.batch_size, epochs=cfg.epochs, augmentation=cfg.augmentation,
                  dropout=cfg.dropout, l2=cfg.l2, frozen_layers=0, architecture=config,
                  manifest_sha256=identity, tensorflow=tf.__version__, git_commit=snapshot['git_commit'])
    callback, logs = create_tensorboard_callback(MODEL_NAME, cfg.number, cfg.description, params, log_root)
    run.mkdir(parents=True)
    write_json(run / "config.json", dict(config=config, splits=identity, parameters=model.count_params()))
    write_json(run / "environment.json", snapshot)
    callbacks = [callback, tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=cfg.patience,
                                                           restore_best_weights=True),
                 tf.keras.callbacks.ModelCheckpoint(str(run / "best.weights.h5"), monitor="val_loss",
                                                     save_best_only=True, save_weights_only=True),
                 tf.keras.callbacks.CSVLogger(str(run / "history.csv"))]
    if cfg.lr_schedule:
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                                            patience=2, min_lr=1e-6))
    start = time.monotonic()
    history = model.fit(train_ds, validation_data=val_ds, epochs=cfg.epochs,
                        callbacks=callbacks, verbose=verbose).history
    model.load_weights(run / "best.weights.h5")
    val = load_split("val").assign(probability=model.predict(val_ds, verbose=0).reshape(-1))
    val.to_csv(run / "val_predictions.csv", index=False)
    metrics = evaluate_binary(val.label, val.probability)
    log_final_metrics(logs, metrics)
    best = int(np.argmin(history['val_loss']))
    summary = dict(run_name=cfg.run_name, hypothesis=cfg.hypothesis,
                   **{f"val_{k}": v for k, v in metrics.items()}, best_epoch=best + 1,
                   epochs_completed=len(history['loss']), parameters=model.count_params(),
                   train_minutes=(time.monotonic() - start) / 60,
                   train_accuracy_at_best=history['accuracy'][best],
                   val_accuracy_at_best=history['val_accuracy'][best],
                   train_loss_at_best=history['loss'][best], val_loss_at_best=history['val_loss'][best])
    write_json(run / "metrics.json", summary)
    return summary


def load_trained_model(run_name, output_root):
    run = Path(output_root) / run_name
    saved = json.loads((run / "config.json").read_text())
    if saved['splits'] != split_identity():
        raise ValueError("Manifests changed since training.")
    cfg = ExperimentConfig(**saved['config'])
    model = make_model(cfg)
    model.load_weights(run / "best.weights.h5")
    return model, cfg


def freeze_selection(run_name, output_root, threshold=0.5, rationale=""):
    """Persist a decision before accessing test images; refuse changing it."""
    if not rationale.strip() or not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Supply a validation-based rationale and threshold in [0,1].")
    run = Path(output_root) / run_name
    if not (run / "metrics.json").is_file():
        raise ValueError("Selected run must have completed validation results.")
    selection = dict(run_name=run_name, threshold=float(threshold), rationale=rationale,
                     splits=split_identity(),
                     weights_sha256=hashlib.sha256((run / "best.weights.h5").read_bytes()).hexdigest())
    path = Path(output_root) / "frozen_selection.json"
    if path.exists() and json.loads(path.read_text()) != selection:
        raise ValueError("Selection already frozen. Test results must not guide a new selection.")
    write_json(path, selection)
    return selection


def evaluate_on_test(output_root, log_root):
    selection = json.loads((Path(output_root) / "frozen_selection.json").read_text())
    run = Path(output_root) / selection['run_name']
    if selection['splits'] != split_identity():
        raise ValueError("Frozen split manifests changed.")
    if hashlib.sha256((run / "best.weights.h5").read_bytes()).hexdigest() != selection['weights_sha256']:
        raise ValueError("Frozen weights changed.")
    if (run / "test_metrics.json").exists():
        return json.loads((run / "test_metrics.json").read_text()), pd.read_csv(run / "test_predictions.csv")
    model, cfg = load_trained_model(selection['run_name'], output_root)
    frame = load_split("test")
    ds = build_tf_dataset(frame, image_size=tuple(cfg.image_size), batch_size=cfg.batch_size)
    predictions = frame.assign(probability=model.predict(ds, verbose=0).reshape(-1))
    metrics = evaluate_binary(predictions.label, predictions.probability, selection['threshold'])
    predictions.to_csv(run / "test_predictions.csv", index=False)
    log_final_metrics(Path(log_root) / selection['run_name'], metrics, prefix="test")
    write_json(run / "test_metrics.json", metrics)
    return metrics, predictions
