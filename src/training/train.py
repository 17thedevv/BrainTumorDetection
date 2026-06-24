import argparse
from pathlib import Path

import tensorflow as tf

from src.model.brain_tumor_model import build_brain_tumor_model
from src.utils.data_loader import prepare_image_datasets
from src.utils.preprocessing import preprocess_datasets


def parse_args():
    parser = argparse.ArgumentParser(description="Train brain tumor detection model.")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="data/Dataset",
        help="Root dataset directory containing Training/ and Testing/",
    )
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs.")
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for training and evaluation."
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=[224, 224],
        help="Target image size for model input.",
    )
    parser.add_argument(
        "--output-model",
        type=str,
        default="saved_model/brain_tumor_detector",
        help="Path to save the trained model.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_path = Path(args.dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory {args.dataset_dir} does not exist.")

    train_ds, val_ds, test_ds, class_names = prepare_image_datasets(
        str(dataset_path),
        image_size=tuple(args.image_size),
        batch_size=args.batch_size,
    )

    # Apply preprocessing: resize (safety), augmentation on train set, caching and prefetch
    train_ds, val_ds, test_ds = preprocess_datasets(
        train_ds, val_ds, test_ds, image_size=tuple(args.image_size), batch_size=args.batch_size, augment=True
    )

    model = build_brain_tumor_model(
        input_shape=(*args.image_size, 3), num_classes=len(class_names)
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(Path(args.output_model) / "best_model.h5"),
            save_best_only=True,
            monitor="val_loss",
        ),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    eval_results = model.evaluate(test_ds)
    print(f"Test loss: {eval_results[0]:.4f}")
    print(f"Test accuracy: {eval_results[1]:.4f}")

    output_path = Path(args.output_model)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save(output_path)
    print(f"Saved trained model to {output_path.resolve()}")


if __name__ == "__main__":
    main()
