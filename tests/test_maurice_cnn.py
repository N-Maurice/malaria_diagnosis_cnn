"""Verify experiment identities and the held-out selection boundary without real data."""
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from members.maurice_custom_cnn import train


class MauriceWorkflowTests(unittest.TestCase):
    def test_restart_archive_preserves_results_and_logs(self):
        with TemporaryDirectory() as tmp:
            roots = [Path(tmp) / 'results', Path(tmp) / 'logs']
            for root in roots:
                run = root / 'interrupted'
                run.mkdir(parents=True)
                (run / 'evidence').write_bytes(b'original evidence')
            train.archive_incomplete_run('interrupted', *roots)
            for root in roots:
                self.assertFalse((root / 'interrupted').exists())
                archived = list((root / 'archived_incomplete').iterdir())
                self.assertEqual(len(archived), 1)
                self.assertEqual((archived[0] / 'evidence').read_bytes(), b'original evidence')
            completed = roots[0] / 'completed'
            completed.mkdir()
            (completed / 'metrics.json').write_text('{}')
            with self.assertRaises(ValueError):
                train.archive_incomplete_run('completed', *roots)
            self.assertTrue((completed / 'metrics.json').exists())

    def test_experiment_comparisons(self):
        configs = train.planned_experiments()
        self.assertGreaterEqual(len(configs), 7)
        self.assertEqual(len({c.run_name for c in configs}), len(configs))
        self.assertTrue(configs[3].augmentation)
        self.assertEqual(configs[6].batch_size, 64)
        self.assertNotEqual(configs[7].filters, configs[2].filters)

    def test_freeze_refuses_changes_and_test_uses_cache(self):
        with TemporaryDirectory() as tmp, patch.object(train, 'split_identity', return_value={'fixture': 'sha'}):
            root = Path(tmp)
            run = root / 'additional_custom_cnn_exp_01_baseline'
            run.mkdir()
            (run / 'metrics.json').write_text('{}')
            (run / 'best.weights.h5').write_bytes(b'synthetic checkpoint')
            with self.assertRaises(ValueError):
                train.freeze_selection(run.name, root)
            train.freeze_selection(run.name, root, rationale='Synthetic validation evidence')
            with self.assertRaises(ValueError):
                train.freeze_selection(run.name, root, threshold=0.6, rationale='Changed decision')
            (run / 'test_metrics.json').write_text(json.dumps({'accuracy': 1.0}))
            pd.DataFrame({'filepath': ['fixture.png'], 'label': [1], 'probability': [0.9]}).to_csv(
                run / 'test_predictions.csv', index=False)
            with patch.object(train, 'load_trained_model', side_effect=AssertionError('No new inference')):
                metrics, predictions = train.evaluate_on_test(root, root)
                self.assertEqual(metrics['accuracy'], 1)
                self.assertEqual(len(predictions), 1)
            (run / 'best.weights.h5').write_bytes(b'changed checkpoint')
            with self.assertRaises(ValueError):
                train.evaluate_on_test(root, root)

    @unittest.skipUnless(importlib.util.find_spec('tensorflow'), 'TensorFlow is not installed')
    def test_forward_checkpoint_and_gradcam(self):
        import tensorflow as tf
        from members.maurice_custom_cnn.model import build_model, gradcam_model
        from src.gradcam_utils import make_gradcam_heatmap
        tf.keras.utils.set_random_seed(42)
        model = build_model(image_size=(32, 32), filters=(4, 8), dropout=0.0)
        image = np.random.default_rng(42).random((32, 32, 3), dtype=np.float32)
        prediction = model(image[None], training=False).numpy()
        self.assertEqual(prediction.shape, (1, 1))
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'best.weights.h5'
            model.save_weights(path)
            rebuilt = build_model(image_size=(32, 32), filters=(4, 8), dropout=0.0)
            rebuilt.load_weights(path)
            np.testing.assert_allclose(prediction, rebuilt(image[None], training=False).numpy())
        heatmap = make_gradcam_heatmap(model, image, feature_model=gradcam_model(model), from_logits=True)
        self.assertTrue(np.isfinite(heatmap).all())
        self.assertEqual(heatmap.ndim, 2)
