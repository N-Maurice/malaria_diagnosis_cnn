"""Custom ResNet implementation owned by David.

A residual CNN built from scratch with the Keras Subclassing API, following the
basic block of He et al. (2016): two 3x3 convolutions plus an identity shortcut,
or a 1x1 projection shortcut when the resolution or channel count changes.
No pretrained weights are used.

The network returns the Parasitized probability, shape (N, 1). For Grad-CAM,
``gradcam_outputs`` returns the final convolutional feature map and the logit
from one connected forward pass, matching ``src.gradcam_utils``'s
``feature_model`` contract (use ``from_logits=True``).
"""

from collections.abc import Sequence

import tensorflow as tf

MODEL_NAME = "custom_resnet"


@tf.keras.utils.register_keras_serializable(package="david_custom_resnet")
class ResidualBlock(tf.keras.layers.Layer):
    """Basic residual block: conv-BN-ReLU-conv-BN, add shortcut, ReLU.

    ``use_skip=False`` removes the shortcut, giving a plain block with the
    same layers, for the ablation that isolates the effect of the skip connection.
    """

    def __init__(
        self,
        filters: int,
        stride: int = 1,
        use_batchnorm: bool = True,
        use_skip: bool = True,
        l2: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.filters, self.stride = filters, stride
        self.use_batchnorm, self.use_skip, self.l2 = use_batchnorm, use_skip, l2
        reg = tf.keras.regularizers.L2(l2) if l2 else None
        conv = dict(padding="same", use_bias=not use_batchnorm, kernel_regularizer=reg,
                    kernel_initializer="he_normal")
        self.conv1 = tf.keras.layers.Conv2D(filters, 3, strides=stride, **conv)
        self.conv2 = tf.keras.layers.Conv2D(filters, 3, **conv)
        self.bn1 = tf.keras.layers.BatchNormalization() if use_batchnorm else None
        self.bn2 = tf.keras.layers.BatchNormalization() if use_batchnorm else None
        self._conv_kwargs = conv
        self.projection = self.projection_bn = None

    def build(self, input_shape) -> None:
        # A 1x1 projection is needed only when the shortcut shape would not match.
        if self.use_skip and (self.stride != 1 or input_shape[-1] != self.filters):
            self.projection = tf.keras.layers.Conv2D(
                self.filters, 1, strides=self.stride, **self._conv_kwargs
            )
            if self.use_batchnorm:
                self.projection_bn = tf.keras.layers.BatchNormalization()
        super().build(input_shape)

    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        if self.bn1 is not None:
            x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        if self.bn2 is not None:
            x = self.bn2(x, training=training)
        if self.use_skip:
            shortcut = inputs
            if self.projection is not None:
                shortcut = self.projection(inputs)
                if self.projection_bn is not None:
                    shortcut = self.projection_bn(shortcut, training=training)
            x = x + shortcut
        return tf.nn.relu(x)

    def get_config(self) -> dict:
        return {
            **super().get_config(),
            "filters": self.filters,
            "stride": self.stride,
            "use_batchnorm": self.use_batchnorm,
            "use_skip": self.use_skip,
            "l2": self.l2,
        }


@tf.keras.utils.register_keras_serializable(package="david_custom_resnet")
class CustomResNet(tf.keras.Model):
    """Stem -> residual stages -> global average pooling -> dropout -> sigmoid.

    ``stage_filters[i]`` channels with ``blocks_per_stage[i]`` blocks per stage;
    every stage after the first halves the resolution in its first block.
    """

    def __init__(
        self,
        stage_filters: Sequence[int] = (32, 64, 128),
        blocks_per_stage: Sequence[int] = (1, 1, 1),
        use_batchnorm: bool = True,
        use_skip: bool = True,
        dropout: float = 0.0,
        l2: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if len(stage_filters) != len(blocks_per_stage) or min(blocks_per_stage) < 1:
            raise ValueError("Provide one positive block count per stage.")
        self.stage_filters, self.blocks_per_stage = list(stage_filters), list(blocks_per_stage)
        self.use_batchnorm, self.use_skip = use_batchnorm, use_skip
        self.dropout_rate, self.l2 = dropout, l2
        reg = tf.keras.regularizers.L2(l2) if l2 else None
        self.stem_conv = tf.keras.layers.Conv2D(
            stage_filters[0], 3, padding="same", use_bias=not use_batchnorm,
            kernel_regularizer=reg, kernel_initializer="he_normal", name="stem_conv",
        )
        self.stem_bn = tf.keras.layers.BatchNormalization(name="stem_bn") if use_batchnorm else None
        self.stem_pool = tf.keras.layers.MaxPooling2D(2, name="stem_pool")
        self.res_blocks = [
            ResidualBlock(
                filters, stride=2 if stage > 0 and block == 0 else 1,
                use_batchnorm=use_batchnorm, use_skip=use_skip, l2=l2,
                name=f"stage{stage + 1}_block{block + 1}",
            )
            for stage, (filters, count) in enumerate(zip(stage_filters, blocks_per_stage))
            for block in range(count)
        ]
        self.global_pool = tf.keras.layers.GlobalAveragePooling2D(name="global_pool")
        self.dropout_layer = tf.keras.layers.Dropout(dropout, name="dropout")
        self.classifier = tf.keras.layers.Dense(1, kernel_regularizer=reg, name="logit")

    def extract_features(self, inputs, training=None):
        """Return the final convolutional feature map (N, h, w, C) used by Grad-CAM."""
        x = self.stem_conv(inputs)
        if self.stem_bn is not None:
            x = self.stem_bn(x, training=training)
        x = self.stem_pool(tf.nn.relu(x))
        for block in self.res_blocks:
            x = block(x, training=training)
        return x

    def classify(self, features, training=None):
        """Map feature maps to the Parasitized logit (N, 1)."""
        x = self.global_pool(features)
        return self.classifier(self.dropout_layer(x, training=training))

    def call(self, inputs, training=None):
        return tf.nn.sigmoid(self.classify(self.extract_features(inputs, training), training))

    def gradcam_outputs(self, inputs, training=False):
        """Return (final conv features, logit) from one connected forward pass."""
        features = self.extract_features(inputs, training=training)
        return features, self.classify(features, training=training)

    def get_config(self) -> dict:
        return {
            **super().get_config(),
            "stage_filters": self.stage_filters,
            "blocks_per_stage": self.blocks_per_stage,
            "use_batchnorm": self.use_batchnorm,
            "use_skip": self.use_skip,
            "dropout": self.dropout_rate,
            "l2": self.l2,
        }


def build_model(image_size: Sequence[int] = (128, 128), **kwargs) -> CustomResNet:
    """Instantiate and build the model on a dummy batch so weights exist."""
    model = CustomResNet(name=MODEL_NAME, **kwargs)
    model(tf.zeros((1, *image_size, 3)), training=False)
    return model
