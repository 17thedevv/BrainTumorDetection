from pathlib import Path
import tensorflow as tf


def prepare_image_datasets(
    dataset_dir: str,
    image_size=(224, 224),
    batch_size=32,
    validation_split=0.2,
    seed=123,
):
    base_path = Path(dataset_dir)
    train_dir = base_path / "Training"
    test_dir = base_path / "Testing"

    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(
            f"Training and Testing folders not found inside {dataset_dir}."
        )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="int",
        image_size=image_size,
        batch_size=batch_size,
        validation_split=validation_split,
        subset="training",
        seed=seed,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="int",
        image_size=image_size,
        batch_size=batch_size,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="int",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
    )

    class_names = train_ds.class_names
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)
    test_ds = test_ds.cache().prefetch(autotune)

    return train_ds, val_ds, test_ds, class_names
