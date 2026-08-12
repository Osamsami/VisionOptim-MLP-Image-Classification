"""From-scratch NumPy MLP: model, image preprocessing, dataset loading, and
weight persistence, shared by train.py and app.py.

No TensorFlow/PyTorch/scikit-learn - forward/backward pass, the Adam update
rule, and evaluation metrics are all implemented with plain NumPy, matching
the approach used in VisionOptim/notebook/VisionOptim.ipynb.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import cv2
import numpy as np

IMG_SIZE = 64


class MLP:
    """A single-hidden-layer perceptron: Dense -> ReLU -> Dense -> Softmax,
    trained with the Adam optimizer.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int, seed: int | None = None):
        rng = np.random.default_rng(seed)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.W1 = rng.standard_normal((input_size, hidden_size)) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = rng.standard_normal((hidden_size, output_size)) * 0.01
        self.b2 = np.zeros((1, output_size))

        # Adam optimizer state (first/second moment estimates + timestep).
        self._m = {k: np.zeros_like(v) for k, v in self._params().items()}
        self._v = {k: np.zeros_like(v) for k, v in self._params().items()}
        self._t = 0

    def _params(self):
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        return (x > 0).astype(x.dtype)

    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / exp_x.sum(axis=1, keepdims=True)

    def forward(self, X):
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = self.relu(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = self.softmax(self.Z2)
        return self.A2

    def backward(self, X, y, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        """Backprop + an Adam parameter update."""
        m = X.shape[0]
        one_hot = np.zeros_like(self.A2)
        one_hot[np.arange(m), y] = 1

        dZ2 = self.A2 - one_hot
        dW2 = self.A1.T @ dZ2 / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * self.relu_derivative(self.Z1)
        dW1 = X.T @ dZ1 / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

        self._t += 1
        for name, grad in grads.items():
            self._m[name] = beta1 * self._m[name] + (1 - beta1) * grad
            self._v[name] = beta2 * self._v[name] + (1 - beta2) * (grad ** 2)

            m_hat = self._m[name] / (1 - beta1 ** self._t)
            v_hat = self._v[name] / (1 - beta2 ** self._t)

            update = lr * m_hat / (np.sqrt(v_hat) + eps)
            setattr(self, name, getattr(self, name) - update)

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1), probs

    def save(self, path: str, class_names: list[str], img_size: int = IMG_SIZE):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez(
            path,
            W1=self.W1,
            b1=self.b1,
            W2=self.W2,
            b2=self.b2,
            class_names=np.array(class_names),
            img_size=np.array(img_size),
        )

    @classmethod
    def load(cls, path: str) -> tuple["MLP", list[str], int]:
        data = np.load(path, allow_pickle=False)
        input_size, hidden_size = data["W1"].shape
        output_size = data["W2"].shape[1]

        model = cls(input_size, hidden_size, output_size)
        model.W1 = data["W1"]
        model.b1 = data["b1"]
        model.W2 = data["W2"]
        model.b2 = data["b2"]

        class_names = [str(name) for name in data["class_names"]]
        img_size = int(data["img_size"])
        return model, class_names, img_size


def calculate_accuracy(model: MLP, X, y) -> float:
    preds, _ = model.predict(X)
    return float(np.mean(preds == y))


def calculate_loss(model: MLP, X, y) -> float:
    probs = model.forward(X)
    return float(-np.mean(np.log(probs[np.arange(len(y)), y] + 1e-12)))


def preprocess_image(image, img_size: int = IMG_SIZE) -> np.ndarray:
    """Resize + normalize + flatten a BGR/RGB image array the same way the
    training pipeline does. Returns a flat feature vector (not batched).
    """
    resized = cv2.resize(image, (img_size, img_size))
    normalized = resized / 255.0
    return normalized.flatten()


def load_dataset(path: str, img_size: int = IMG_SIZE, limit_per_class: int = 1000):
    """Load a folder-per-class image dataset, e.g. `<path>/<class_name>/*.jpg`.

    Returns (X, y, class_names) with class_names sorted for a deterministic
    label mapping.
    """
    X, y = [], []
    class_names = sorted(
        d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))
    )

    for label, folder in enumerate(class_names):
        folder_path = os.path.join(path, folder)
        for img_name in sorted(os.listdir(folder_path))[:limit_per_class]:
            img_path = os.path.join(folder_path, img_name)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                X.append(preprocess_image(img, img_size))
                y.append(label)
            except Exception:
                continue

    return np.array(X), np.array(y), class_names


def train_val_split(X, y, val_fraction: float = 0.2, seed: int | None = 42):
    """Shuffle and split (X, y) into train/val subsets. Used as a fallback
    when a dataset does not ship a separate held-out split.
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    indices = rng.permutation(n)
    val_size = int(n * val_fraction)
    val_idx, train_idx = indices[:val_size], indices[val_size:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


@dataclass
class TrainingHistory:
    train_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)


def confusion_matrix(y_true, y_pred, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def classification_report(y_true, y_pred, class_names: list[str]) -> str:
    """Per-class precision/recall/F1, computed from scratch (no scikit-learn)."""
    num_classes = len(class_names)
    cm = confusion_matrix(y_true, y_pred, num_classes)

    lines = [f"{'Class':12s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'Support':>10s}"]
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = cm[i, :].sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        lines.append(f"{name:12s} {precision:10.4f} {recall:10.4f} {f1:10.4f} {support:10d}")

    return "\n".join(lines)
