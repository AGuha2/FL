from __future__ import annotations

import numpy as np
import shap
from sklearn.metrics import precision_recall_fscore_support


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)

# Per class mean SHAP values and the class availability. The above normalisation is applied.
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

# F1 score calculated for each class
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

# Cosine similarity that is clipped to the positive range, so that a client with a negative similarity does not get rewarded for being "opposite" to the other clients.
def positive_cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = normalize_rows(left)
    right = normalize_rows(right)
    return np.maximum(0.0, np.sum(left * right, axis=1))

# Softmax function with temperature control
def softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    scaled = np.asarray(values, dtype=np.float64) / max(temperature, 1e-6)
    scaled -= np.max(scaled)
    probabilities = np.exp(scaled)
    return probabilities / np.maximum(probabilities.sum(), 1e-12)

# To ensure that the weights sums to 1
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
    available: np.ndarray,
    client_class_f1: np.ndarray,
    previous_server_class_f1: np.ndarray,
    previous_weights: np.ndarray | None,
    *,
    shap_blend: float = 0.65,
    temperature: float = 0.25,
    minimum: float = 0.12,
    maximum: float = 0.28,
    smoothing: float = 0.70,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Combine FedAvg, class utility, gain, and consensus-SHAP agreement.

    Every submitted client model is evaluated by the server on the same
    balanced validation subset. SHAP similarity is measured against an
    element-wise median made from the other clients, so a client is never
    included in the reference used to score itself.
    """
    client_shap = np.asarray(client_shap, dtype=np.float64)
    available = np.asarray(available, dtype=np.float64)
    reliability = np.clip(
        np.asarray(client_class_f1, dtype=np.float64),
        0.0,
        1.0,
    )
    previous_f1 = np.clip(
        np.asarray(previous_server_class_f1, dtype=np.float64),
        0.0,
        1.0,
    )
    if client_shap.ndim != 3:
        raise ValueError(
            "client_shap must have shape [client, class, feature]."
        )
    num_clients, num_classes, num_features = client_shap.shape
    if available.shape != (num_clients, num_classes):
        raise ValueError(
            "available must have shape [client, class]."
        )
    if reliability.shape != (num_clients, num_classes):
        raise ValueError(
            "client_class_f1 must have shape [client, class]."
        )
    if previous_f1.shape != (num_classes,):
        raise ValueError(
            "previous_server_class_f1 must have one value per class."
        )
    if (
        not np.all(np.isfinite(client_shap))
        or not np.all(np.isfinite(reliability))
    ):
        raise ValueError(
            "Client SHAP or validation F1 contains non-finite values."
        )

    normalized_client_shap = np.stack([
        normalize_rows(matrix) for matrix in client_shap
    ])
    consensus = np.zeros_like(normalized_client_shap)
    similarities = np.zeros(
        (num_clients, num_classes),
        dtype=np.float64,
    )
    peer_count = np.zeros(
        (num_clients, num_classes),
        dtype=np.float64,
    )
    comparable = np.zeros(
        (num_clients, num_classes),
        dtype=np.float64,
    )

    # A client is never included in the reference used to score itself.
    for client_id in range(num_clients):
        for class_id in range(num_classes):
            if available[client_id, class_id] <= 0.5:
                continue
            peers = [
                other_id
                for other_id in range(num_clients)
                if (
                    other_id != client_id
                    and available[other_id, class_id] > 0.5
                )
            ]
            peer_count[client_id, class_id] = len(peers)
            if not peers:
                continue
            median_vector = np.median(
                normalized_client_shap[peers, class_id, :],
                axis=0,
            )
            median_vector = normalize_rows(
                median_vector.reshape(1, num_features)
            )[0]
            consensus[client_id, class_id] = median_vector
            similarities[client_id, class_id] = max(
                0.0,
                float(np.dot(
                    normalized_client_shap[client_id, class_id],
                    median_vector,
                )),
            )
            comparable[client_id, class_id] = 1.0

    # Poorly detected classes receive more attention, with a cap to prevent
    # an almost-zero F1 class from overwhelming every other class.
    importance = 1.0 / np.maximum(previous_f1, 0.05)
    importance = np.minimum(importance, 10.0)
    importance /= importance.sum()

    # If only one client has a class, consensus similarity is unavailable.
    # A neutral value retains that client's measurable F1 and gain evidence.
    effective_similarity = np.where(
        comparable > 0.5,
        similarities,
        0.5,
    )
    gain = np.maximum(
        0.0,
        reliability - previous_f1[None, :],
    )
    class_score = (
        0.50 * reliability
        + 0.30 * effective_similarity
        + 0.20 * gain
    ) * available

    weighted_available = available * importance[None, :]
    score_denominator = weighted_available.sum(axis=1)
    score = np.divide(
        np.sum(
            class_score * importance[None, :],
            axis=1,
        ),
        score_denominator,
        out=np.full(num_clients, 0.5, dtype=np.float64),
        where=score_denominator > 1e-12,
    )
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
        "effective_similarity": effective_similarity,
        "importance": importance,
        "reliability": reliability,
        "gain": gain,
        "previous_server_class_f1": previous_f1,
        "consensus": consensus,
        "peer_count": peer_count,
        "comparable": comparable,
    }
    return bounded, diagnostics
