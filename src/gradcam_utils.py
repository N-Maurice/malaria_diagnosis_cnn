"""Architecture-independent Grad-CAM and RGB overlays.

Functional/Sequential models may identify a connected final convolutional layer.
Subclassed models must expose feature maps: supply a feature_model callable that
returns (features, predictions) from ONE connected forward pass. Predictions must
be computed from the returned features, not from a separate model invocation.
Design that access path when implementing each model. No layer name is assumed.
"""

from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    import tensorflow as tf


def make_gradcam_heatmap(
    model: "tf.keras.Model",
    input_image: "tf.Tensor | np.ndarray",
    layer_name: str | None = None,
    feature_model: Callable | None = None,
    class_index: int = 1,
    from_logits: bool = False,
) -> np.ndarray:
    """Return a normalized 2D heatmap for one already-preprocessed image.

    Input is HWC or a single-image NHWC batch. Output must be (1,1) binary or
    (1,2) class-ordered predictions. A sigmoid output targets p for infected and
    1-p for uninfected; a binary logit targets z or -z. For two outputs the selected
    score is used. Logits generally avoid saturation. A zero positive attribution
    returns an all-zero map. feature_model must accept training=False.
    """
    import tensorflow as tf

    if class_index not in (0, 1):
        raise ValueError("class_index must be 0 (Uninfected) or 1 (Parasitized).")
    if (layer_name is None) == (feature_model is None):
        raise ValueError("Supply exactly one of layer_name or feature_model.")
    image = tf.convert_to_tensor(input_image, dtype=tf.float32)
    if image.shape.rank == 3:
        image = image[None, ...]
    if image.shape.rank != 4 or image.shape[0] != 1:
        raise ValueError("Grad-CAM expects one image, HWC or (1,H,W,C).")
    if feature_model is None:
        try:
            feature_model = tf.keras.Model(
                model.inputs, [model.get_layer(layer_name).output, model.output]
            )
        except (ValueError, AttributeError) as exc:
            raise ValueError(
                "Layer must be connected to model output; subclassed/nested models should expose feature_model."
            ) from exc
    with tf.GradientTape() as tape:
        tape.watch(image)
        features, predictions = feature_model(image, training=False)
        if (
            features.shape.rank != 4
            or predictions.shape.rank != 2
            or predictions.shape[-1] not in (1, 2)
        ):
            raise ValueError(
                "Expected NHWC feature maps and predictions of shape (1,1) or (1,2)."
            )
        if predictions.shape[-1] == 1:
            score = predictions[:, 0]
            if class_index == 0:
                score = -score if from_logits else 1 - score
        else:
            score = predictions[:, class_index]
    gradients = tape.gradient(score, features)
    if gradients is None:
        raise ValueError(
            "Predictions are disconnected from returned convolutional features."
        )
    weights = tf.reduce_mean(gradients, axis=(0, 1, 2))
    heatmap = tf.nn.relu(tf.reduce_sum(features[0] * weights, axis=-1))
    heatmap = tf.math.divide_no_nan(heatmap, tf.reduce_max(heatmap))
    return heatmap.numpy()


def overlay_heatmap(
    image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4
) -> np.ndarray:
    """Overlay a [0,1] heatmap on original uint8 RGB pixels; return uint8 RGB."""
    import matplotlib
    from PIL import Image

    image, heatmap = np.asarray(image), np.asarray(heatmap)
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
        raise ValueError(
            "image must be an original HWC uint8 RGB image, not preprocessed input."
        )
    if (
        heatmap.ndim != 2
        or not heatmap.size
        or not np.isfinite(heatmap).all()
        or heatmap.min() < 0
        or heatmap.max() > 1
    ):
        raise ValueError("heatmap must be a finite nonempty 2D array in [0,1].")
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be within [0,1].")
    resized = np.asarray(
        Image.fromarray(heatmap.astype("float32")).resize(
            (image.shape[1], image.shape[0]), Image.Resampling.BILINEAR
        )
    )
    colors = matplotlib.colormaps["jet"](resized)[..., :3] * 255
    return np.clip((1 - alpha) * image + alpha * colors, 0, 255).astype(np.uint8)
