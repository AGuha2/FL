"""Class-specific, reliability-gated SHAP aggregation utilities.

Copy this file beside server.py and client1.py ... client5.py.
"""
from __future__ import annotations

import numpy as np
import shap
from sklearn.metrics import precision_recall_fscore_support


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def class_specific_mean_abs_shap(
    model,
    background: np.ndarray,
    samples: np.ndarray,
    sample_labels: np.ndarray,
    num_classes: int,
    num_features: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return [class, feature] mean-|SHAP| and a class-availability mask."""
    raw = shap.GradientExplainer(model, background).shap_values(samples)

    if isinstance(raw, list):
        # Older SHAP: list[output] containing [sample, feature].
        by_output = np.stack([np.asarray(item) for item in raw], axis=0)
    else:
        array = np.asarray(raw)
        if array.ndim != 3:
            raise ValueError(f"Unexpected SHAP shape: {array.shape}")
        if array.shape == (len(samples), num_features, num_classes):
            by_output = np.transpose(array, (2, 0, 1))
        elif array.shape == (num_classes, len(samples), num_features):
            by_output = array
        elif array.shape == (len(samples), num_classes, num_features):
            by_output = np.transpose(array, (1, 0, 2))
        else:
            raise ValueError(f"Unexpected SHAP shape: {array.shape}")

    result = np.zeros((num_classes, num_features), dtype=np.float64)
    available = np.zeros(num_classes, dtype=np.float64)
    for class_id in range(num_classes):
        indices = np.where(sample_labels == class_id)[0]
        if len(indices) == 0:
            continue
        result[class_id] = np.mean(
            np.abs(by_output[class_id, indices, :]),
            axis=0,
        )
        available[class_id] = 1.0

    return normalize_rows(result), available


def class_f1(y_true, y_pred, num_classes: int) -> np.ndarray:
    """One-vs-rest F1 for every class, always in label-index order."""
    _, _, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=np.arange(num_classes),
        average=None,
        zero_division=0,
    )
    return np.asarray(f1, dtype=np.float64)


def positive_cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = normalize_rows(left)
    right = normalize_rows(right)
    return np.maximum(0.0, np.sum(left * right, axis=1))


def softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    scaled = np.asarray(values, dtype=np.float64) / max(temperature, 1e-6)
    scaled -= np.max(scaled)
    probabilities = np.exp(scaled)
    return probabilities / np.maximum(probabilities.sum(), 1e-12)


def bounded_simplex(
    weights: np.ndarray,
    minimum: float,
    maximum: float,
) -> np.ndarray:
    """Project positive weights onto sum=1 with per-client bounds."""
    weights = np.asarray(weights, dtype=np.float64)
    n = len(weights)
    if minimum * n > 1.0 or maximum * n < 1.0:
        raise ValueError("Client-weight bounds cannot sum to one.")
    weights = weights / np.maximum(weights.sum(), 1e-12)

    low, high = 0.0, 1e6
    for _ in range(100):
        scale = (low + high) / 2.0
        projected = np.clip(weights * scale, minimum, maximum)
        if projected.sum() > 1.0:
            high = scale
        else:
            low = scale
    projected = np.clip(weights * low, minimum, maximum)
    # Correct tiny floating-point residue without violating the bounds.
    for _ in range(20):
        residue = 1.0 - projected.sum()
        if abs(residue) < 1e-12:
            break
        if residue > 0:
            free = np.where(projected < maximum - 1e-12)[0]
        else:
            free = np.where(projected > minimum + 1e-12)[0]
        if len(free) == 0:
            break
        projected[free] += residue / len(free)
        projected = np.clip(projected, minimum, maximum)
    return projected / projected.sum()


def calculate_client_weights(
    size_weights: np.ndarray,
    client_shap: np.ndarray,
    server_shap: np.ndarray,
    available: np.ndarray,
    client_class_f1: np.ndarray,
    global_class_f1: np.ndarray,
    previous_weights: np.ndarray | None,
    *,
    shap_blend: float = 0.50,
    temperature: float = 0.25,
    minimum: float = 0.12,
    maximum: float = 0.28,
    smoothing: float = 0.70,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Calculate bounded weights for arrays shaped [client, class, feature]."""
    client_shap = np.asarray(client_shap, dtype=np.float64)
    available = np.asarray(available, dtype=np.float64)
    reliability = np.asarray(client_class_f1, dtype=np.float64)
    global_f1 = np.asarray(global_class_f1, dtype=np.float64)

    similarities = np.stack([
        positive_cosine_rows(matrix, server_shap)
        for matrix in client_shap
    ])
    similarities *= available

    # Poorly detected classes receive more attention, with a cap to prevent
    # an almost-zero F1 class from overwhelming every other class.
    importance = 1.0 / np.maximum(global_f1, 0.05)
    importance = np.minimum(importance, 10.0)
    importance /= importance.sum()

    gain = np.maximum(0.0, reliability - global_f1[None, :])
    class_score = (
        0.50 * reliability
        + 0.30 * similarities
        + 0.20 * gain
    ) * available
    score = np.sum(class_score * importance[None, :], axis=1)
    shap_weights = softmax(score, temperature)

    size_weights = np.asarray(size_weights, dtype=np.float64)
    size_weights /= size_weights.sum()
    blended = (1.0 - shap_blend) * size_weights + shap_blend * shap_weights
    bounded = bounded_simplex(blended, minimum, maximum)

    if previous_weights is not None:
        bounded = (
            smoothing * np.asarray(previous_weights, dtype=np.float64)
            + (1.0 - smoothing) * bounded
        )
        bounded = bounded_simplex(bounded, minimum, maximum)

    diagnostics = {
        "score": score,
        "shap_weight": shap_weights,
        "similarity": similarities,
        "importance": importance,
        "reliability": reliability,
        "gain": gain,
    }
    return bounded, diagnostics
