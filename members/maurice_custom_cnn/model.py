"""Maurice's plain CNN: explicit convolution blocks, no pretrained weights or skips."""


def build_model(image_size=(128, 128), filters=(32, 64, 128),
                convolutions_per_block=2, use_batchnorm=True, dropout=0.3, l2=1e-4):
    """Build a Functional model with a connected ``last_conv`` Grad-CAM layer.

    Inputs are RGB floats in [0,1]; output is P(Parasitized). TensorFlow is
    imported lazily so project infrastructure remains importable without it.
    """
    import tensorflow as tf
    if not filters or min(filters) <= 0 or convolutions_per_block < 1:
        raise ValueError("Use positive filters and at least one convolution per block.")
    if len(image_size) != 2 or min(image_size) < 2 ** len(filters):
        raise ValueError("Image dimensions must accommodate all pooling blocks.")
    if not 0 <= dropout < 1 or l2 < 0:
        raise ValueError("Use dropout in [0,1) and nonnegative L2.")
    layers = tf.keras.layers
    regularizer = tf.keras.regularizers.l2(l2) if l2 else None
    inputs = layers.Input((*image_size, 3), name="normalized_rgb")
    x = inputs
    for block, width in enumerate(filters, 1):
        for conv in range(1, convolutions_per_block + 1):
            last = block == len(filters) and conv == convolutions_per_block
            x = layers.Conv2D(width, 3, padding="same", use_bias=not use_batchnorm,
                              kernel_initializer="he_normal", kernel_regularizer=regularizer,
                              name="last_conv" if last else f"block{block}_conv{conv}")(x)
            if use_batchnorm:
                x = layers.BatchNormalization(name=f"block{block}_bn{conv}")(x)
            x = layers.ReLU(name=f"block{block}_relu{conv}")(x)
        x = layers.MaxPooling2D(name=f"block{block}_pool")(x)
    x = layers.GlobalAveragePooling2D(name="global_average_pool")(x)
    x = layers.Dropout(dropout, name="head_dropout")(x)
    logits = layers.Dense(1, kernel_regularizer=regularizer, name="logit")(x)
    outputs = layers.Activation("sigmoid", name="parasitized_probability")(logits)
    return tf.keras.Model(inputs, outputs, name="additional_custom_cnn")


def gradcam_model(model):
    """Return features and unsaturated logits from one connected forward pass."""
    import tensorflow as tf
    return tf.keras.Model(model.inputs,
                          [model.get_layer("last_conv").output, model.get_layer("logit").output])
