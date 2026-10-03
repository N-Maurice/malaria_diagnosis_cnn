"""Synthetic infrastructure verification only; no training or real dataset access."""

import importlib
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd
from PIL import Image

from src import config
from src.data_utils import build_dataframe, build_tf_dataset, stratified_split
from src.eval_utils import evaluate_binary, plot_confusion_matrix, plot_roc_curve
from src.logging_utils import make_run_name

HAS_TF = importlib.util.find_spec("tensorflow") is not None


class ScaffoldTests(unittest.TestCase):
    """Check imports, deterministic splits, metrics and supported input paths."""

    def test_imports_and_config(self) -> None:
        for module in (
            "config",
            "data_utils",
            "eval_utils",
            "logging_utils",
            "gradcam_utils",
        ):
            importlib.import_module(f"src.{module}")
        for member in (
            "david_custom_resnet",
            "sarah_transfer_model_1",
            "laura_transfer_model_2",
            "maurice_custom_cnn",
        ):
            importlib.import_module(f"members.{member}.model")
        self.assertEqual(config.RANDOM_SEED, 42)
        self.assertEqual(config.CLASS_MAPPING, {"Uninfected": 0, "Parasitized": 1})
        self.assertTrue((config.PROJECT_ROOT / "src").is_dir())

    def test_stratification_and_reproducibility(self) -> None:
        frame = pd.DataFrame(
            {
                "filepath": [f"synthetic/{i}.png" for i in range(200)],
                "label": [0] * 100 + [1] * 100,
            }
        )
        with TemporaryDirectory() as directory:
            parts = stratified_split(frame, Path(directory) / "first")
            again = stratified_split(
                frame.sample(frac=1, random_state=7), Path(directory) / "second"
            )
            self.assertEqual([len(p) for p in parts], [140, 30, 30])
            seen = set()
            for name, part, repeated in zip(("train", "val", "test"), parts, again):
                self.assertEqual(
                    part.label.value_counts().to_dict(),
                    {0: len(part) // 2, 1: len(part) // 2},
                )
                self.assertFalse(seen & set(part.filepath))
                seen.update(part.filepath)
                pd.testing.assert_frame_equal(part, repeated)
                pd.testing.assert_frame_equal(
                    part, pd.read_csv(Path(directory) / "first" / f"{name}_files.csv")
                )
            self.assertEqual(len(seen), len(frame))
            with self.assertRaises(FileExistsError):
                stratified_split(frame, Path(directory) / "first")
            with self.assertRaises(ValueError):
                stratified_split(
                    pd.concat([frame, frame.iloc[:1]]), Path(directory) / "bad"
                )

    def test_metrics(self) -> None:
        metrics = evaluate_binary([0, 0, 1, 1], [0.1, 0.7, 0.8, 0.9])
        self.assertEqual(
            set(metrics),
            {
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "sensitivity",
                "specificity",
            },
        )
        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        self.assertAlmostEqual(metrics["precision"], 2 / 3)
        self.assertAlmostEqual(metrics["f1"], 0.8)
        self.assertEqual(metrics["recall"], metrics["sensitivity"])
        self.assertEqual(metrics["specificity"], 0.5)
        self.assertEqual(metrics["roc_auc"], 1.0)
        self.assertTrue(np.isnan(evaluate_binary([1], [0.8])["specificity"]))
        self.assertTrue(np.isnan(evaluate_binary([0], [0.2])["roc_auc"]))
        self.assertEqual(evaluate_binary([1], [0.5])["recall"], 1)
        with self.assertRaises(ValueError):
            evaluate_binary([0, 1], [float("nan"), 0.5])

    def test_plots_and_names(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        with TemporaryDirectory() as directory:
            for function, filename in (
                (plot_confusion_matrix, "confusion.png"),
                (plot_roc_curve, "roc.png"),
            ):
                result = function(
                    [0, 1], [0.2, 0.9], output_path=Path(directory) / filename
                )
                self.assertTrue((Path(directory) / filename).is_file())
                plt.close(result[0])
        self.assertEqual(
            make_run_name("custom_resnet", 2, "lr_1e-3"), "custom_resnet_exp_02_lr_1e-3"
        )
        for name in ("final", "final_final", "test1", "run1", "experiment"):
            with self.assertRaises(ValueError):
                make_run_name("custom_resnet", 1, name)

    def test_image_discovery(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                build_dataframe(root, root)
            for folder in config.CLASS_MAPPING:
                (root / folder).mkdir()
                Image.new("RGB", (8, 8), color=(255, 0, 0)).save(
                    root / folder / "synthetic.png"
                )
                (root / folder / "Thumbs.db").write_text("ignored")
            frame = build_dataframe(root, root)
            self.assertEqual(len(frame), 2)
            self.assertEqual(set(frame.label), {0, 1})
            self.assertTrue(
                all(not Path(value).is_absolute() for value in frame.filepath)
            )

    @unittest.skipUnless(
        HAS_TF,
        "TensorFlow not installed; install requirements for pipeline verification",
    )
    def test_tensorflow_pipeline(self) -> None:
        import tensorflow as tf

        with TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (8, 8), color=(255, 0, 0)).save(root / "synthetic.png")
            frame = pd.DataFrame({"filepath": ["synthetic.png"], "label": [1]})
            images, labels = next(
                iter(build_tf_dataset(frame, image_size=(4, 4), path_root=root))
            )
            self.assertEqual(tuple(images.shape), (1, 4, 4, 3))
            self.assertEqual(tuple(labels.shape), (1, 1))
            self.assertEqual(float(tf.reduce_max(images)), 1.0)
            images, _ = next(
                iter(
                    build_tf_dataset(
                        frame,
                        path_root=root,
                        augmentation=lambda x: x * 0.5,
                        preprocessing_fn=lambda x: x - 10,
                    )
                )
            )
            self.assertEqual(float(tf.reduce_max(images)), 117.5)

    @unittest.skipUnless(
        HAS_TF,
        "TensorFlow not installed; install requirements for Grad-CAM verification",
    )
    def test_gradcam_connected_features(self) -> None:
        import tensorflow as tf
        from src.gradcam_utils import make_gradcam_heatmap, overlay_heatmap

        # Arithmetic-only synthetic feature contract; no architecture is built.
        def features_and_scores(image, training=False):
            features = image * 2
            score = tf.reshape(tf.reduce_mean(features), (1, 1))
            return features, score

        heatmap = make_gradcam_heatmap(
            None,
            np.ones((4, 4, 3), dtype="float32"),
            feature_model=features_and_scores,
            from_logits=True,
        )
        np.testing.assert_allclose(heatmap, 1)
        negative = make_gradcam_heatmap(
            None,
            np.ones((4, 4, 3), dtype="float32"),
            feature_model=features_and_scores,
            class_index=0,
            from_logits=True,
        )
        np.testing.assert_allclose(negative, 0)
        overlay = overlay_heatmap(np.zeros((8, 8, 3), dtype="uint8"), heatmap)
        self.assertEqual(overlay.shape, (8, 8, 3))
        self.assertEqual(overlay.dtype, np.uint8)

    @unittest.skipUnless(
        HAS_TF,
        "TensorFlow not installed; install requirements for TensorBoard verification",
    )
    def test_tensorboard_metadata(self) -> None:
        from src.logging_utils import create_tensorboard_callback, log_final_metrics
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )

        # Temporary synthetic events verify serialization; these are not experiments.
        with TemporaryDirectory() as directory:
            params = dict(
                model_name="synthetic_fixture",
                learning_rate=0.001,
                optimizer="fixture",
                batch_size=2,
                epochs=0,
                augmentation=False,
                dropout=0.0,
                l2=0.0,
                frozen_layers=0,
            )
            callback, run = create_tensorboard_callback(
                "synthetic_fixture", 1, "metadata_check", params, directory
            )
            self.assertEqual(callback.log_dir, str(run))
            log_final_metrics(run, {"accuracy": 0.5})
            events = EventAccumulator(str(run)).Reload()
            self.assertIn("validation/accuracy", events.Tags()["tensors"])
            with self.assertRaises(FileExistsError):
                create_tensorboard_callback(
                    "synthetic_fixture", 1, "metadata_check", params, directory
                )


if __name__ == "__main__":
    unittest.main()
