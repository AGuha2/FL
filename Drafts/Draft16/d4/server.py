"""Seven-class FL server with class-specific reliable SHAP aggregation."""
import json
import sys
from pathlib import Path

import flwr as fl
import joblib
import numpy as np
import pandas as pd
import shap
import tensorflow as tf
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

from Labels7 import ALL_LABELS, NUM_CLASSES, NUM_FEATURES, le
from model import build_model
from posthoc_shap_analysis import save_final_posthoc_shap_analysis


NUM_CLIENTS = 5
NUM_ROUNDS = 10
SEED = 42

SERVER_DATA_PATH = r"D:\CAPSTONE\Server_Test_7Class.csv"
SCALER_PATH = r"D:\CAPSTONE\global_scaler_7Class.pkl"
RESULT_DIR = Path(
    "results_10round_7class_equal_f1_gated_shap_v3"
)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Aggregation starts in round 1. Before round 1, every class has the same
# minimum server F1, so no class is initially preferred.
WARMUP_ROUNDS = 0
INITIAL_SERVER_CLASS_F1 = 0.05
SHAP_BLEND = 0.40
SHAP_TEMPERATURE = 0.35
MIN_CLIENT_WEIGHT = 0.12
MAX_CLIENT_WEIGHT = 0.28
WEIGHT_SMOOTHING = 0.70
F1_COEFFICIENT = 0.55
GAIN_COEFFICIENT = 0.20
SHAP_COEFFICIENT = 0.25
DIFFICULTY_POWER = 1.50

# Each client update is evaluated on the same balanced validation subset.
UTILITY_SAMPLES_PER_CLASS = 2000

# The separate final test partition is used only after the last aggregation.
POSTHOC_SHAP_SAMPLES_PER_CLASS = 50
POSTHOC_SHAP_BACKGROUND_SIZE = 50

np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)


def normalize_rows(values):
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def class_specific_mean_abs_shap(
    model,
    background,
    samples,
    sample_labels,
    num_classes,
    num_features,
):
    """Return normalized mean-|SHAP| for every class and feature."""
    raw = shap.GradientExplainer(model, background).shap_values(samples)

    if isinstance(raw, list):
        by_output = np.stack(
            [np.asarray(item) for item in raw],
            axis=0,
        )
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

    result = np.zeros(
        (num_classes, num_features),
        dtype=np.float64,
    )
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


def class_f1(y_true, y_pred, num_classes):
    """Return F1 for every class in label-index order."""
    _, _, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=np.arange(num_classes),
        average=None,
        zero_division=0,
    )
    return np.asarray(f1, dtype=np.float64)


def softmax(values, temperature):
    scaled = np.asarray(values, dtype=np.float64)
    scaled = scaled / max(float(temperature), 1e-6)
    scaled -= np.max(scaled)
    probabilities = np.exp(scaled)
    return probabilities / np.maximum(
        probabilities.sum(),
        1e-12,
    )


def bounded_simplex(weights, minimum, maximum):
    """Normalize weights to sum to one while respecting client bounds."""
    weights = np.asarray(weights, dtype=np.float64)
    num_clients = len(weights)
    if (
        minimum * num_clients > 1.0
        or maximum * num_clients < 1.0
    ):
        raise ValueError("Client-weight bounds cannot sum to one.")
    weights /= np.maximum(weights.sum(), 1e-12)

    low, high = 0.0, 1e6
    for _ in range(100):
        scale = (low + high) / 2.0
        projected = np.clip(weights * scale, minimum, maximum)
        if projected.sum() > 1.0:
            high = scale
        else:
            low = scale

    projected = np.clip(weights * low, minimum, maximum)
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
    size_weights,
    client_shap,
    available,
    client_class_f1,
    previous_server_class_f1,
    previous_weights,
    *,
    shap_blend=0.40,
    temperature=0.35,
    minimum=0.12,
    maximum=0.28,
    smoothing=0.70,
    f1_coefficient=0.55,
    gain_coefficient=0.20,
    shap_coefficient=0.25,
    difficulty_power=1.50,
):
    """Calculate F1-gated class-specific SHAP aggregation weights."""
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

    coefficients = np.asarray(
        [f1_coefficient, gain_coefficient, shap_coefficient],
        dtype=np.float64,
    )
    if np.any(coefficients < 0.0) or not np.isclose(
        coefficients.sum(),
        1.0,
        atol=1e-8,
    ):
        raise ValueError(
            "F1, gain, and SHAP coefficients must be non-negative "
            "and sum to one."
        )
    if difficulty_power <= 0.0:
        raise ValueError("difficulty_power must be positive.")

    normalized_client_shap = np.stack([
        normalize_rows(matrix) for matrix in client_shap
    ])
    consensus = np.zeros_like(normalized_client_shap)
    similarities = np.zeros(
        (num_clients, num_classes),
        dtype=np.float64,
    )
    peer_count = np.zeros_like(similarities)
    comparable = np.zeros_like(similarities)

    # Leave-one-client-out SHAP median: a client is never part of the
    # reference against which its own SHAP vector is compared.
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

    # Difficult classes receive more attention without the instability of
    # inverse-F1 weighting.
    importance = np.power(
        np.maximum(0.0, 1.0 - previous_f1),
        difficulty_power,
    )
    if importance.sum() <= 1e-12:
        importance = np.full(num_classes, 1.0 / num_classes)
    else:
        importance /= importance.sum()

    # When no peer has the class, use neutral similarity. F1 remains the
    # primary evidence for that client and class.
    effective_similarity = np.where(
        comparable > 0.5,
        similarities,
        0.5,
    )
    gain = np.maximum(
        0.0,
        reliability - previous_f1[None, :],
    ) * available

    # Normalize F1 and gain separately within each class.
    best_reliability = np.max(reliability * available, axis=0)
    normalized_reliability = np.divide(
        reliability,
        best_reliability[None, :],
        out=np.zeros_like(reliability),
        where=best_reliability[None, :] > 1e-12,
    ) * available
    best_gain = np.max(gain, axis=0)
    normalized_gain = np.divide(
        gain,
        best_gain[None, :],
        out=np.zeros_like(gain),
        where=best_gain[None, :] > 1e-12,
    ) * available

    # This is the F1 gate. SHAP similarity contributes strongly only when
    # that client's F1 is also strong for the same class.
    f1_gated_shap = (
        normalized_reliability * effective_similarity
    ) * available
    class_score = (
        f1_coefficient * normalized_reliability
        + gain_coefficient * normalized_gain
        + shap_coefficient * f1_gated_shap
    ) * available

    weighted_available = available * importance[None, :]
    score_denominator = weighted_available.sum(axis=1)
    score = np.divide(
        np.sum(class_score * importance[None, :], axis=1),
        score_denominator,
        out=np.full(num_clients, 0.5, dtype=np.float64),
        where=score_denominator > 1e-12,
    )
    shap_weights = softmax(score, temperature)

    size_weights = np.asarray(size_weights, dtype=np.float64)
    size_weights /= np.maximum(size_weights.sum(), 1e-12)
    blended = (
        (1.0 - shap_blend) * size_weights
        + shap_blend * shap_weights
    )
    bounded = bounded_simplex(blended, minimum, maximum)

    if previous_weights is not None:
        bounded = (
            smoothing * np.asarray(
                previous_weights,
                dtype=np.float64,
            )
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
        "normalized_reliability": normalized_reliability,
        "normalized_gain": normalized_gain,
        "f1_gated_shap": f1_gated_shap,
        "class_score": class_score,
        "previous_server_class_f1": previous_f1,
        "consensus": consensus,
        "peer_count": peer_count,
        "comparable": comparable,
    }
    return bounded, diagnostics


def calculate_metrics(y_true, y_pred):
    precision_by_class, recall_by_class, f1_by_class, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=np.arange(NUM_CLASSES),
            average=None,
            zero_division=0,
        )
    )
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(NUM_CLASSES),
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "precision": float(np.mean(precision_by_class)),
        "recall": float(np.mean(recall_by_class)),
        "f1": float(np.mean(f1_by_class)),
        "class_f1": np.asarray(f1_by_class, dtype=np.float64),
        "confusion_matrix": matrix,
    }


def save_matrix(matrix, path):
    pd.DataFrame(
        matrix,
        index=[f"true_{name}" for name in ALL_LABELS],
        columns=[f"pred_{name}" for name in ALL_LABELS],
    ).to_csv(path)


def balanced_indices(targets, samples_per_class, seed):
    rng = np.random.RandomState(seed)
    selected = []
    for class_id in range(NUM_CLASSES):
        indices = np.where(targets == class_id)[0]
        if len(indices) == 0:
            raise ValueError(
                f"Server data has no {ALL_LABELS[class_id]}."
            )
        chosen = rng.choice(
            indices,
            size=min(samples_per_class, len(indices)),
            replace=False,
        )
        selected.extend(chosen.tolist())
    selected = np.asarray(selected, dtype=np.int64)
    rng.shuffle(selected)
    return selected


# ---------------------------------------------------------------------------
# Load the separate server-side dataset.
# ---------------------------------------------------------------------------
server_df = pd.read_csv(SERVER_DATA_PATH).replace(
    [np.inf, -np.inf],
    np.nan,
).dropna()
server_df.drop(
    columns=[
        column
        for column in server_df.columns
        if column.lower().startswith("unnamed:")
    ],
    inplace=True,
    errors="ignore",
)

feature_names = [
    column for column in server_df.columns
    if column != "Label"
]
if len(feature_names) != NUM_FEATURES:
    raise ValueError(
        f"Expected {NUM_FEATURES} server features, "
        f"found {len(feature_names)}."
    )

labels = server_df["Label"].astype(str).str.strip().str.upper()
unknown = sorted(set(labels) - set(ALL_LABELS))
if unknown:
    raise ValueError(f"Unexpected server labels: {unknown}")

scaler = joblib.load(SCALER_PATH)
server_x = scaler.transform(
    server_df[feature_names]
).astype(np.float32)
server_y = le.transform(labels).astype(np.int32)

# Half of the held-out server data is validation; half is final testing.
x_val, x_test, y_val, y_test = train_test_split(
    server_x,
    server_y,
    test_size=0.5,
    random_state=SEED,
    stratify=server_y,
)

# Common balanced validation subset for evaluating every submitted client
# model. The final server test partition is never used for this calculation.
utility_indices = balanced_indices(
    y_val,
    UTILITY_SAMPLES_PER_CLASS,
    SEED + 1000,
)
x_utility = x_val[utility_indices]
y_utility = y_val[utility_indices]

# A larger, final-round-only SHAP sample is drawn from the untouched test
# partition. It is not used for aggregation, model selection, or training.
posthoc_server_indices = balanced_indices(
    y_test,
    POSTHOC_SHAP_SAMPLES_PER_CLASS,
    SEED + 4000,
)
posthoc_server_samples = x_test[posthoc_server_indices]
posthoc_server_labels = y_test[posthoc_server_indices]

posthoc_background_rng = np.random.RandomState(SEED + 5000)
posthoc_background_pool = np.setdiff1d(
    np.arange(len(x_test)),
    posthoc_server_indices,
)
if len(posthoc_background_pool) == 0:
    posthoc_background_pool = np.arange(len(x_test))
posthoc_server_background_indices = posthoc_background_rng.choice(
    posthoc_background_pool,
    size=min(
        POSTHOC_SHAP_BACKGROUND_SIZE,
        len(posthoc_background_pool),
    ),
    replace=False,
)
posthoc_server_background = x_test[
    posthoc_server_background_indices
]


# ---------------------------------------------------------------------------
# Model and centralized evaluation.
# ---------------------------------------------------------------------------
evaluation_model = build_model(NUM_FEATURES, NUM_CLASSES)
evaluation_model.compile(loss="sparse_categorical_crossentropy")
initial_parameters = ndarrays_to_parameters(
    evaluation_model.get_weights()
)

server_history = []
client_history = []
best_val_f1 = -1.0
best_round = 0
server_f1_state = {
    "round": 0,
    "values": np.full(
        NUM_CLASSES,
        INITIAL_SERVER_CLASS_F1,
        dtype=np.float64,
    ),
}


def fit_config(server_round):
    return {
        "server_round": int(server_round),
        "run_posthoc_shap": bool(server_round == NUM_ROUNDS),
    }


def evaluate_config(server_round):
    return {"server_round": int(server_round)}


def centralized_evaluate(server_round, parameters, config):
    global best_val_f1, best_round

    # Flower can evaluate the random initial model before round 1. It is
    # intentionally skipped so round 1 uses INITIAL_SERVER_CLASS_F1.
    if server_round == 0:
        return None

    evaluation_model.set_weights(parameters)

    val_probabilities = evaluation_model.predict(
        x_val,
        batch_size=2048,
        verbose=0,
    )
    val_result = calculate_metrics(
        y_val,
        val_probabilities.argmax(axis=1),
    )
    val_loss = float(np.mean(
        tf.keras.losses.sparse_categorical_crossentropy(
            y_val,
            val_probabilities,
        )
    ))

    row = {
        "round": int(server_round),
        "val_loss": val_loss,
    }
    for name in (
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
    ):
        row[f"val_{name}"] = val_result[name]
    for class_id, class_name in enumerate(ALL_LABELS):
        row[f"val_f1::{class_name}"] = float(
            val_result["class_f1"][class_id]
        )

    save_matrix(
        val_result["confusion_matrix"],
        RESULT_DIR
        / f"server_validation_confusion_round_{server_round:03d}.csv",
    )

    # These values are read by aggregate_fit in the NEXT round.
    server_f1_state["round"] = int(server_round)
    server_f1_state["values"] = val_result["class_f1"].copy()

    if val_result["f1"] > best_val_f1:
        best_val_f1 = val_result["f1"]
        best_round = int(server_round)
        evaluation_model.save_weights(
            RESULT_DIR / "best_global.weights.h5"
        )

    # The final test partition is touched only after round 10 aggregation.
    if server_round == NUM_ROUNDS:
        test_probabilities = evaluation_model.predict(
            x_test,
            batch_size=2048,
            verbose=0,
        )
        test_result = calculate_metrics(
            y_test,
            test_probabilities.argmax(axis=1),
        )
        test_loss = float(np.mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                y_test,
                test_probabilities,
            )
        ))
        row["test_loss"] = test_loss
        for name in (
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
        ):
            row[f"test_{name}"] = test_result[name]
        for class_id, class_name in enumerate(ALL_LABELS):
            row[f"test_f1::{class_name}"] = float(
                test_result["class_f1"][class_id]
            )
        save_matrix(
            test_result["confusion_matrix"],
            RESULT_DIR / "final_server_test_confusion.csv",
        )
        print(
            "[F1-Gated SHAP Server] Final test: "
            f"accuracy={test_result['accuracy']:.4f}, "
            f"macro_F1={test_result['f1']:.4f}"
        )

    server_history.append(row)
    pd.DataFrame(server_history).to_csv(
        RESULT_DIR / "server_metrics.csv",
        index=False,
    )

    print(
        f"[F1-Gated SHAP Server] R{server_round}: "
        f"validation_accuracy={val_result['accuracy']:.4f}, "
        f"validation_F1={val_result['f1']:.4f}"
    )
    return val_loss, {
        name: val_result[name]
        for name in (
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
        )
    }


class ClassSpecificReliableShap(fl.server.strategy.FedAvg):
    """FedAvg blended with validation F1-gated consensus SHAP."""

    def __init__(
        self,
        *,
        evaluation_model,
        server_f1_state,
        x_utility,
        y_utility,
        posthoc_shap_background,
        posthoc_shap_samples,
        posthoc_shap_labels,
        feature_names,
        class_names,
        result_dir,
        final_round=10,
        warmup_rounds=0,
        shap_blend=0.40,
        temperature=0.35,
        minimum_weight=0.12,
        maximum_weight=0.28,
        smoothing=0.70,
        f1_coefficient=0.55,
        gain_coefficient=0.20,
        shap_coefficient=0.25,
        difficulty_power=1.50,
        **fedavg_arguments,
    ):
        super().__init__(**fedavg_arguments)
        self.model = evaluation_model
        self.server_f1_state = server_f1_state
        self.x_utility = x_utility
        self.y_utility = y_utility
        self.posthoc_shap_background = posthoc_shap_background
        self.posthoc_shap_samples = posthoc_shap_samples
        self.posthoc_shap_labels = posthoc_shap_labels
        self.feature_names = list(feature_names)
        self.class_names = list(class_names)
        self.num_classes = len(self.class_names)
        self.result_dir = Path(result_dir)
        self.final_round = int(final_round)
        self.warmup_rounds = int(warmup_rounds)
        self.shap_blend = float(shap_blend)
        self.temperature = float(temperature)
        self.minimum_weight = float(minimum_weight)
        self.maximum_weight = float(maximum_weight)
        self.smoothing = float(smoothing)
        self.f1_coefficient = float(f1_coefficient)
        self.gain_coefficient = float(gain_coefficient)
        self.shap_coefficient = float(shap_coefficient)
        self.difficulty_power = float(difficulty_power)
        self.previous_weights = None
        self.global_parameters = parameters_to_ndarrays(
            fedavg_arguments["initial_parameters"]
        )
        self.aggregation_history = []
        self.posthoc_complete = False

    def _class_f1(self, parameters):
        self.model.set_weights(parameters)
        probabilities = self.model.predict(
            self.x_utility,
            batch_size=2048,
            verbose=0,
        )
        return class_f1(
            self.y_utility,
            probabilities.argmax(axis=1),
            self.num_classes,
        )

    def _posthoc_server_shap(self):
        self.model.set_weights(self.global_parameters)
        matrix, _ = class_specific_mean_abs_shap(
            self.model,
            self.posthoc_shap_background,
            self.posthoc_shap_samples,
            self.posthoc_shap_labels,
            self.num_classes,
            len(self.feature_names),
        )
        return matrix

    def _read_client_shap(self, metrics, namespace="shap"):
        matrix = np.zeros(
            (self.num_classes, len(self.feature_names)),
            dtype=np.float64,
        )
        available = np.zeros(self.num_classes, dtype=np.float64)
        for class_id, class_name in enumerate(self.class_names):
            available[class_id] = float(
                metrics.get(
                    f"{namespace}_available::{class_name}",
                    0.0,
                )
            )
            for feature_id, feature_name in enumerate(
                self.feature_names
            ):
                key = f"{namespace}::{class_name}::{feature_name}"
                matrix[class_id, feature_id] = float(
                    metrics.get(key, 0.0)
                )
        if not np.all(np.isfinite(matrix)):
            raise ValueError(
                "A client returned non-finite SHAP values."
            )
        return matrix, available

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        records = []
        seen = set()
        for _, fit_result in results:
            metrics = dict(fit_result.metrics)
            client_id = int(metrics["client_id"])
            if client_id in seen:
                raise ValueError(f"Duplicate client ID {client_id}")
            seen.add(client_id)

            shap_matrix, available = self._read_client_shap(metrics)
            posthoc_shap = shap_matrix
            posthoc_available = available
            if (
                server_round == self.final_round
                and any(
                    key.startswith("posthoc_shap::")
                    for key in metrics
                )
            ):
                posthoc_shap, posthoc_available = (
                    self._read_client_shap(
                        metrics,
                        namespace="posthoc_shap",
                    )
                )

            records.append({
                "client_id": client_id,
                "num_examples": int(fit_result.num_examples),
                "parameters": parameters_to_ndarrays(
                    fit_result.parameters
                ),
                "shap": shap_matrix,
                "available": available,
                "posthoc_shap": posthoc_shap,
                "posthoc_available": posthoc_available,
            })

        records.sort(key=lambda item: item["client_id"])

        size_weights = np.asarray(
            [item["num_examples"] for item in records],
            dtype=np.float64,
        )
        size_weights /= size_weights.sum()

        if server_round <= self.warmup_rounds:
            final_weights = size_weights
            diagnostic = None
        else:
            # All submitted client models are evaluated on one common,
            # balanced server-validation subset. This makes their per-class
            # F1 scores directly comparable without receiving client data.
            client_class_f1 = np.stack([
                self._class_f1(item["parameters"])
                for item in records
            ])
            final_weights, diagnostic = calculate_client_weights(
                size_weights=size_weights,
                client_shap=np.stack([
                    item["shap"] for item in records
                ]),
                available=np.stack([
                    item["available"] for item in records
                ]),
                client_class_f1=client_class_f1,
                previous_server_class_f1=np.asarray(
                    self.server_f1_state["values"],
                    dtype=np.float64,
                ),
                previous_weights=self.previous_weights,
                shap_blend=self.shap_blend,
                temperature=self.temperature,
                minimum=self.minimum_weight,
                maximum=self.maximum_weight,
                smoothing=self.smoothing,
                f1_coefficient=self.f1_coefficient,
                gain_coefficient=self.gain_coefficient,
                shap_coefficient=self.shap_coefficient,
                difficulty_power=self.difficulty_power,
            )

        self.previous_weights = final_weights.copy()
        aggregated = [
            np.zeros_like(layer)
            for layer in records[0]["parameters"]
        ]
        for item, weight in zip(records, final_weights):
            for layer_id, layer in enumerate(item["parameters"]):
                aggregated[layer_id] += layer * weight

        self.global_parameters = aggregated

        # Final post-hoc analysis runs only after the last global model
        # has been aggregated. It cannot influence aggregation weights.
        if (
            server_round == self.final_round
            and not self.posthoc_complete
        ):
            self.model.set_weights(self.global_parameters)
            final_server_shap = self._posthoc_server_shap()
            final_client_records = [
                {
                    "client_id": item["client_id"],
                    "shap": item["posthoc_shap"],
                    "available": item["posthoc_available"],
                }
                for item in records
            ]
            output_paths = save_final_posthoc_shap_analysis(
                self.result_dir,
                server_round=server_round,
                feature_names=self.feature_names,
                class_names=self.class_names,
                server_shap=final_server_shap,
                client_records=final_client_records,
            )
            self.model.set_weights(self.global_parameters)
            self.model.save_weights(
                self.result_dir / "final_global.weights.h5"
            )
            self.posthoc_complete = True
            print(
                "[Final post-hoc SHAP] Saved server/client analysis: "
                + ", ".join(
                    path.name for path in output_paths.values()
                )
            )

        for index, item in enumerate(records):
            row = {
                "round": int(server_round),
                "client_id": item["client_id"],
                "num_examples": item["num_examples"],
                "fedavg_weight": float(size_weights[index]),
                "final_weight": float(final_weights[index]),
                "warmup": int(
                    server_round <= self.warmup_rounds
                ),
                "server_f1_source_round": int(
                    self.server_f1_state["round"]
                ),
            }
            if diagnostic is not None:
                row["client_score"] = float(
                    diagnostic["score"][index]
                )
                row["raw_shap_weight"] = float(
                    diagnostic["shap_weight"][index]
                )
                for class_id, class_name in enumerate(
                    self.class_names
                ):
                    row[f"importance::{class_name}"] = float(
                        diagnostic["importance"][class_id]
                    )
                    row[f"previous_server_f1::{class_name}"] = float(
                        diagnostic[
                            "previous_server_class_f1"
                        ][class_id]
                    )
                    row[f"similarity::{class_name}"] = float(
                        diagnostic["similarity"][index, class_id]
                    )
                    row[f"client_validation_f1::{class_name}"] = float(
                        diagnostic["reliability"][index, class_id]
                    )
                    row[f"normalized_f1::{class_name}"] = float(
                        diagnostic[
                            "normalized_reliability"
                        ][index, class_id]
                    )
                    row[f"gain::{class_name}"] = float(
                        diagnostic["gain"][index, class_id]
                    )
                    row[f"normalized_gain::{class_name}"] = float(
                        diagnostic["normalized_gain"][index, class_id]
                    )
                    row[f"f1_gated_shap::{class_name}"] = float(
                        diagnostic["f1_gated_shap"][index, class_id]
                    )
                    row[f"class_score::{class_name}"] = float(
                        diagnostic["class_score"][index, class_id]
                    )
                    row[f"peer_count::{class_name}"] = int(
                        diagnostic["peer_count"][index, class_id]
                    )
                    row[f"comparable::{class_name}"] = int(
                        diagnostic["comparable"][index, class_id]
                    )
            self.aggregation_history.append(row)

        pd.DataFrame(self.aggregation_history).to_csv(
            self.result_dir / "aggregation_weights.csv",
            index=False,
        )
        print(
            f"[F1-gated consensus SHAP] R{server_round}: "
            + ", ".join(
                f"C{item['client_id']}={weight:.4f}"
                for item, weight in zip(records, final_weights)
            )
        )
        return ndarrays_to_parameters(aggregated), {}


class ExperimentStrategy(ClassSpecificReliableShap):
    """Adds per-client CSV metrics to class-specific SHAP aggregation."""

    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            return None, {}

        seen_ids = set()
        for _, evaluate_result in results:
            returned = dict(evaluate_result.metrics)
            client_id = int(returned["client_id"])
            if client_id in seen_ids:
                raise ValueError(
                    f"Duplicate evaluation client ID: {client_id}"
                )
            seen_ids.add(client_id)

            matrix = np.asarray(
                json.loads(returned["confusion_matrix"]),
                dtype=int,
            )
            if matrix.shape != (NUM_CLASSES, NUM_CLASSES):
                raise ValueError(
                    f"Client {client_id}: invalid confusion matrix "
                    f"shape {matrix.shape}."
                )
            save_matrix(
                matrix,
                RESULT_DIR
                / (
                    f"client_{client_id}_confusion_"
                    f"round_{server_round:03d}.csv"
                ),
            )

            client_history.append({
                "round": int(server_round),
                "client_id": client_id,
                "num_examples": int(evaluate_result.num_examples),
                "loss": float(evaluate_result.loss),
                **{
                    name: float(returned.get(name, 0.0))
                    for name in (
                        "accuracy",
                        "balanced_accuracy",
                        "precision",
                        "recall",
                        "f1",
                    )
                },
            })

        client_history.sort(
            key=lambda row: (row["round"], row["client_id"])
        )
        pd.DataFrame(client_history).to_csv(
            RESULT_DIR / "client_metrics.csv",
            index=False,
        )
        return super().aggregate_evaluate(
            server_round,
            results,
            failures,
        )


strategy = ExperimentStrategy(
    evaluation_model=evaluation_model,
    server_f1_state=server_f1_state,
    x_utility=x_utility,
    y_utility=y_utility,
    posthoc_shap_background=posthoc_server_background,
    posthoc_shap_samples=posthoc_server_samples,
    posthoc_shap_labels=posthoc_server_labels,
    feature_names=feature_names,
    class_names=ALL_LABELS,
    result_dir=RESULT_DIR,
    final_round=NUM_ROUNDS,
    warmup_rounds=WARMUP_ROUNDS,
    shap_blend=SHAP_BLEND,
    temperature=SHAP_TEMPERATURE,
    minimum_weight=MIN_CLIENT_WEIGHT,
    maximum_weight=MAX_CLIENT_WEIGHT,
    smoothing=WEIGHT_SMOOTHING,
    f1_coefficient=F1_COEFFICIENT,
    gain_coefficient=GAIN_COEFFICIENT,
    shap_coefficient=SHAP_COEFFICIENT,
    difficulty_power=DIFFICULTY_POWER,
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=NUM_CLIENTS,
    min_evaluate_clients=NUM_CLIENTS,
    min_available_clients=NUM_CLIENTS,
    initial_parameters=initial_parameters,
    on_fit_config_fn=fit_config,
    on_evaluate_config_fn=evaluate_config,
    evaluate_fn=centralized_evaluate,
)


if len(sys.argv) < 2:
    raise ValueError("Usage: python server.py <port>")

fl.server.start_server(
    server_address=f"localhost:{sys.argv[1]}",
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    strategy=strategy,
    grpc_max_message_length=1024 * 1024 * 1024,
)

print(
    "[Class-specific SHAP Server] "
    f"Best validation macro-F1={best_val_f1:.4f} "
    f"at round {best_round}"
)
