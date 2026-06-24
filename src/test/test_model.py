import unittest
from pathlib import Path

import tensorflow as tf

from src.model.brain_tumor_model import build_brain_tumor_model
from src.utils.data_loader import prepare_image_datasets


class BrainTumorDetectionTest(unittest.TestCase):
    def setUp(self):
        self.root_dir = Path(__file__).resolve().parents[2]
        self.dataset_dir = self.root_dir / "data" / "Dataset"

    def test_model_build_and_compile(self):
        model = build_brain_tumor_model()
        self.assertEqual(model.output_shape[-1], 4)
        self.assertEqual(model.loss.__class__.__name__, "SparseCategoricalCrossentropy")

    def test_predicts_single_image(self):
        model = build_brain_tumor_model()
        dummy_input = tf.random.uniform((1, 224, 224, 3))
        output = model(dummy_input, training=False)
        self.assertEqual(output.shape, (1, 4))

    def test_dataset_loading(self):
        if not self.dataset_dir.exists():
            self.skipTest("Dataset directory is not available in the workspace.")

        train_ds, val_ds, test_ds, class_names = prepare_image_datasets(
            str(self.dataset_dir), batch_size=16
        )
        self.assertGreaterEqual(len(class_names), 2)
        self.assertTrue(any(True for _ in train_ds.take(1)))
        self.assertTrue(any(True for _ in val_ds.take(1)))
        self.assertTrue(any(True for _ in test_ds.take(1)))


if __name__ == "__main__":
    unittest.main()
