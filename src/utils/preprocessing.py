from typing import Tuple
import tensorflow as tf


def get_augmentation_layer(image_size: Tuple[int, int]):
    """Return a small keras.Sequential augmentation layer."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.08),
            tf.keras.layers.RandomContrast(0.08),
        ],
        name="augmentation",
    )


def preprocess_datasets(
    train_ds, val_ds, test_ds, image_size=(224, 224), batch_size=32, augment=True
):
    """Apply resizing, optional augmentation, caching and prefetching to datasets.

    Args:
        train_ds, val_ds, test_ds: tf.data.Dataset from image_dataset_from_directory.
        image_size: target image size (h, w)
        batch_size: batch size (unused if datasets already batched)
        augment: whether to apply augmentation to training set

    Returns:
        Tuple of processed (train_ds, val_ds, test_ds)
    """
    AUTOTUNE = tf.data.AUTOTUNE

    def _resize(image, label):
        image = tf.image.resize(image, image_size)
        return image, label

    train_ds = train_ds.map(_resize, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(_resize, num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.map(_resize, num_parallel_calls=AUTOTUNE)

    if augment:
        aug = get_augmentation_layer(image_size)
        train_ds = train_ds.map(
            lambda x, y: (aug(x, training=True), y), num_parallel_calls=AUTOTUNE
        )

    train_ds = train_ds.cache().prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)
    test_ds = test_ds.cache().prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds
