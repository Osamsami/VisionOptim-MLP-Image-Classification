"""Standalone training script for the from-scratch NumPy MLP.

Trains on `<data-dir>/seg_train`, evaluates on a held-out split (uses
`<data-dir>/seg_test` if present, otherwise carves one out of the training
data), and saves the learned weights so they can be reused by app.py without
retraining.

Usage:
    python train.py --data-dir ./data --epochs 20 --hidden-size 128
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

from src.mlp import (
    MLP,
    calculate_accuracy,
    calculate_loss,
    classification_report,
    load_dataset,
    train_val_split,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train the VisionOptim MLP.")
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("VISIONOPTIM_DATA_PATH", "./data"),
        help="Directory containing seg_train/ (and optionally seg_test/).",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=1000,
        help="Max images loaded per class (keeps runs fast; lower this for smoke tests).",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Fraction of seg_train held out for validation when seg_test is absent.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default="models/mlp_weights.npz",
        help="Where to save the trained weights (.npz).",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    train_dir = os.path.join(args.data_dir, "seg_train")
    test_dir = os.path.join(args.data_dir, "seg_test")

    if not os.path.isdir(train_dir):
        print(f"error: training data not found at '{train_dir}'.", file=sys.stderr)
        print(
            "Set --data-dir or VISIONOPTIM_DATA_PATH to a folder containing "
            "seg_train/<class_name>/*.jpg (see README for dataset instructions).",
            file=sys.stderr,
        )
        return 1

    print(f"Loading training data from {train_dir} ...")
    X_train, y_train, class_names = load_dataset(
        train_dir, img_size=args.img_size, limit_per_class=args.limit_per_class
    )
    print(f"Loaded {X_train.shape[0]} training images across {len(class_names)} classes: {class_names}")

    if os.path.isdir(test_dir):
        print(f"Loading held-out test data from {test_dir} ...")
        X_val, y_val, _ = load_dataset(
            test_dir, img_size=args.img_size, limit_per_class=args.limit_per_class
        )
        print(f"Loaded {X_val.shape[0]} held-out test images.")
    else:
        print(f"No seg_test/ found under {args.data_dir}; splitting seg_train instead.")
        X_train, y_train, X_val, y_val = train_val_split(
            X_train, y_train, val_fraction=args.val_split, seed=args.seed
        )
        print(f"Train: {X_train.shape[0]} images, Held-out: {X_val.shape[0]} images.")

    if X_train.shape[0] == 0:
        print("error: no training images were loaded.", file=sys.stderr)
        return 1

    model = MLP(
        input_size=X_train.shape[1],
        hidden_size=args.hidden_size,
        output_size=len(class_names),
        seed=args.seed,
    )

    print(f"\nTraining for {args.epochs} epochs (lr={args.lr}, hidden_size={args.hidden_size}) ...")
    start = time.time()
    for epoch in range(args.epochs):
        model.forward(X_train)
        model.backward(X_train, y_train, args.lr)

        train_loss = calculate_loss(model, X_train, y_train)
        train_acc = calculate_accuracy(model, X_train, y_train)
        val_loss = calculate_loss(model, X_val, y_val) if len(X_val) else float("nan")
        val_acc = calculate_accuracy(model, X_val, y_val) if len(X_val) else float("nan")

        print(
            f"Epoch {epoch + 1}/{args.epochs}: "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
    print(f"Training finished in {time.time() - start:.1f}s")

    if len(X_val):
        preds, _ = model.predict(X_val)
        print("\nHeld-out classification report:")
        print(classification_report(y_val, preds, class_names))

    model.save(args.output, class_names=class_names, img_size=args.img_size)
    print(f"\nSaved model weights to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
