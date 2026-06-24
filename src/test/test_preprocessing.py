import numpy as np
import tensorflow as tf
import pytest

from src.utils.preprocessing import get_augmentation_layer, preprocess_datasets


def test_augmentation_layer_shape():
    img_size = (64, 64)
    layer = get_augmentation_layer(img_size)
    x = tf.random.uniform((1, img_size[0], img_size[1], 3), dtype=tf.float32)
    y = layer(x, training=True)
    assert y.shape == x.shape


def test_preprocess_datasets_resizes_and_batches():
    # create 8 random images of size 128x128
    images = np.random.randint(0, 256, size=(8, 128, 128, 3), dtype=np.uint8)
    labels = np.zeros((8,), dtype=np.int32)
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    ds = ds.batch(4)

    train_ds, val_ds, test_ds = preprocess_datasets(ds, ds, ds, image_size=(64, 64), batch_size=4, augment=True)

    # take one batch from train
    for batch_images, batch_labels in train_ds.take(1):
        assert batch_images.shape[0] == 4
        assert batch_images.shape[1:] == (64, 64, 3)
        assert batch_labels.shape[0] == 4


def test_preprocess_without_augmentation_preserves_values_shape():
    images = np.random.randint(0, 256, size=(4, 100, 120, 3), dtype=np.uint8)
    labels = np.ones((4,), dtype=np.int32)
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    ds = ds.batch(2)

    train_ds, _, _ = preprocess_datasets(ds, ds, ds, image_size=(64, 64), batch_size=2, augment=False)

    for batch_images, batch_labels in train_ds.take(1):
        assert batch_images.shape[1:] == (64, 64, 3)
        assert batch_labels.shape[0] == 2
