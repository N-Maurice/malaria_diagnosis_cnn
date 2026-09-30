"""Binary evaluation and standalone plotting; positive class is Parasitized."""

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
    roc_curve,
)


def _validate(
    y_true: Any, probabilities: Any, threshold: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """Accept (N,) or (N,1) arrays of labels and positive-class probabilities."""
    y, p = np.asarray(y_true), np.asarray(probabilities, dtype=float)
    if any(a.ndim not in (1, 2) or (a.ndim == 2 and a.shape[1] != 1) for a in (y, p)):
        raise ValueError(
            "Use (N,) or (N,1) arrays; select positive-class column for two-output models."
        )
    y, p = y.reshape(-1), p.reshape(-1)
    if len(y) == 0 or len(y) != len(p):
        raise ValueError("Labels and probabilities must be nonempty and equally sized.")
    if (
        not np.isin(y, [0, 1]).all()
        or not np.isfinite(p).all()
        or ((p < 0) | (p > 1)).any()
    ):
        raise ValueError("Labels must be 0/1 and probabilities finite within [0,1].")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite within [0,1].")
    return y.astype(int), p


def evaluate_binary(
    y_true: Any, predicted_probabilities: Any, threshold: float = 0.5
) -> dict[str, float]:
    """Return seven metrics; undefined ROC-AUC/specificity/sensitivity are NaN.

    Precision/F1 with no predicted positives use zero_division=0. Probabilities
    at the threshold are classified positive. Threshold selection uses validation.
    """
    y, p = _validate(y_true, predicted_probabilities, threshold)
    predicted = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    sensitivity = float(tp / (tp + fn)) if tp + fn else float("nan")
    return {
        "accuracy": float(accuracy_score(y, predicted)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": sensitivity,
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "roc_auc": (
            float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")
        ),
        "sensitivity": sensitivity,
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
    }


def _save(figure: Any, output_path: str | Path | None) -> None:
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=200, bbox_inches="tight")


def plot_confusion_matrix(
    y_true: Any,
    predicted_probabilities: Any,
    threshold: float = 0.5,
    output_path: str | Path | None = None,
) -> tuple[Any, Any]:
    """Plot counts with fixed label order; return (figure, axes), without show()."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    y, p = _validate(y_true, predicted_probabilities, threshold)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(
        confusion_matrix(y, p >= threshold, labels=[0, 1]),
        display_labels=["Uninfected", "Parasitized"],
    ).plot(ax=ax, cmap="Blues")
    ax.set_title(f"Confusion matrix (threshold={threshold:g})")
    _save(fig, output_path)
    return fig, ax


def plot_roc_curve(
    y_true: Any,
    predicted_probabilities: Any,
    output_path: str | Path | None = None,
) -> tuple[Any, Any, float]:
    """Return (figure, axes, AUC); ROC needs both classes in y_true."""
    import matplotlib.pyplot as plt

    y, p = _validate(y_true, predicted_probabilities)
    if len(np.unique(y)) != 2:
        raise ValueError("ROC requires both classes.")
    fpr, tpr, _ = roc_curve(y, p)
    auc = float(roc_auc_score(y, p))
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set(
        xlabel="False-positive rate",
        ylabel="Sensitivity (true-positive rate)",
        title="ROC curve",
    )
    ax.legend()
    _save(fig, output_path)
    return fig, ax, auc
