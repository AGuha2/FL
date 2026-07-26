import json
import random
import sys
from pathlib import Path

import flwr as fl
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
)
from tensorflow import keras

from Labels import le, NUM_CLASSES, NUM_FEATURES
from experiment_config import CFG
from model import build_model


NUM_CLIENTS = 5

SERVER_TEST_PATH = r"D:\CAPSTONE\Server_Test.csv"
SCALER_PATH = r"D:\CAPSTONE\global_scaler.pkl"

COMMON_SHAP_REFERENCE_PATH = (
    r"D:\CAPSTONE\Common_SHAP_Reference.npz"
)

SEED = 42


random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)


RESULT_DIR = Path(
    f"results_{CFG['experiment']}"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


with open(
    RESULT_DIR / "config.json",
    "w",
    encoding="utf-8",
) as config_file:
    json.dump(
        {
            key: sorted(value)
            if isinstance(value, set)
            else value
            for key, value in CFG.items()
        },
        config_file,
        indent=2,
    )


print(
    f"[Server] Experiment "
    f"{CFG['experiment']}: {CFG}"
)

print(
    "[Server] Loading centralized dataset..."
)


server_df = pd.read_csv(
    SERVER_TEST_PATH
)

server_df.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True,
)

server_df.dropna(
    inplace=True
)


unnamed_columns = [
    column
    for column in server_df.columns
    if column.lower().startswith(
        "unnamed:"
    )
]

if unnamed_columns:
    print(
        f"[Server] Removing index columns: "
        f"{unnamed_columns}"
    )

    server_df.drop(
        columns=unnamed_columns,
        inplace=True,
        errors="ignore",
    )


if "Label" not in server_df.columns:
    raise ValueError(
        f"'Label' column not found in "
        f"{SERVER_TEST_PATH}"
    )


feature_names = [
    column
    for column in server_df.columns
    if column != "Label"
]


if len(feature_names) != NUM_FEATURES:
    raise ValueError(
        f"Server dataset has "
        f"{len(feature_names)} features; "
        f"expected {NUM_FEATURES}.\n"
        f"Features: {feature_names}"
    )


x_server_df = server_df[
    feature_names
]


y_server = le.transform(
    server_df["Label"]
    .astype(str)
    .str.strip()
    .str.upper()
)


scaler = joblib.load(
    SCALER_PATH
)


x_server = scaler.transform(
    x_server_df
).astype(np.float32)


def stratified_indices(
    labels,
    target_size,
    random_state,
    excluded_indices=None,
):
    """
    Select approximately class-stratified indices.

    excluded_indices prevents overlap between:
    - SHAP reference set
    - validation set
    - test set
    """

    excluded = (
        set()
        if excluded_indices is None
        else set(
            int(index)
            for index in excluded_indices
        )
    )

    selected_parts = []

    classes, class_counts = np.unique(
        labels,
        return_counts=True,
    )


    for class_id, class_count in zip(
        classes,
        class_counts,
    ):
        class_indices = np.where(
            labels == class_id
        )[0]

        available_indices = np.asarray([
            index
            for index in class_indices
            if int(index) not in excluded
        ])


        if len(available_indices) == 0:
            continue


        target_for_class = max(
            1,
            round(
                target_size
                * class_count
                / len(labels)
            ),
        )


        target_for_class = min(
            target_for_class,
            len(available_indices),
        )


        chosen = random_state.choice(
            available_indices,
            size=target_for_class,
            replace=False,
        )


        selected_parts.append(
            chosen
        )


    if not selected_parts:
        raise ValueError(
            "Could not create a stratified subset."
        )


    selected = np.unique(
        np.concatenate(
            selected_parts
        )
    )


    if len(selected) > target_size:
        selected = random_state.choice(
            selected,
            size=target_size,
            replace=False,
        )


    return np.asarray(
        selected,
        dtype=np.int64,
    )


rng = np.random.RandomState(
    SEED
)


# ============================================================
# Common SHAP reference set
# ============================================================

shap_reference_indices = stratified_indices(
    labels=y_server,
    target_size=CFG[
        "shap_reference_size"
    ],
    random_state=rng,
)


x_shap_reference = x_server[
    shap_reference_indices
]

y_shap_reference = y_server[
    shap_reference_indices
]


np.savez_compressed(
    COMMON_SHAP_REFERENCE_PATH,

    x_reference=x_shap_reference.astype(
        np.float32
    ),

    y_reference=y_shap_reference.astype(
        np.int32
    ),

    feature_names=np.asarray(
        feature_names,
        dtype=str,
    ),
)


print(
    f"[Server] Common SHAP reference saved: "
    f"{COMMON_SHAP_REFERENCE_PATH}"
)


# ============================================================
# Validation set
# ============================================================

validation_indices = stratified_indices(
    labels=y_server,

    target_size=CFG[
        "validation_sample_size"
    ],

    random_state=rng,

    excluded_indices=(
        shap_reference_indices
    ),
)


# ============================================================
# Final test set
# ============================================================

excluded_indices = np.concatenate([
    shap_reference_indices,
    validation_indices,
])


test_mask = np.ones(
    len(y_server),
    dtype=bool,
)

test_mask[
    excluded_indices
] = False


x_val = x_server[
    validation_indices
]

y_val = y_server[
    validation_indices
]


x_test = x_server[
    test_mask
]

y_test = y_server[
    test_mask
]


print(
    f"[Server] SHAP reference rows: "
    f"{len(x_shap_reference):,}"
)

print(
    f"[Server] Validation rows: "
    f"{len(x_val):,}"
)

print(
    f"[Server] Final test rows: "
    f"{len(x_test):,}"
)


eval_model = build_model(
    NUM_FEATURES,
    NUM_CLASSES,
    model_type=CFG[
        "model_type"
    ],
)


eval_model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=CFG[
            "initial_lr"
        ]
    ),

    loss=(
        "sparse_categorical_crossentropy"
    ),

    metrics=["accuracy"],
)


server_history = []
client_history = []
aggregation_history = []
shap_history = []


previous_shap_by_client = {}


best_val_f1 = -1.0
best_round = -1
best_weights = None


def all_metrics(
    y_true,
    y_pred,
):
    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_true,
            y_pred,
        )
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
    )

    return (
        float(accuracy),
        float(balanced_accuracy),
        float(precision),
        float(recall),
        float(f1),
    )


def normalized(values):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    values = np.nan_to_num(
        values,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    values = np.clip(
        values,
        0.0,
        None,
    )

    total = values.sum()

    if total > 0:
        return values / total

    return np.ones(
        len(values),
        dtype=np.float64,
    ) / len(values)


def cosine_similarity_vector(
    first,
    second,
):
    first = np.asarray(
        first,
        dtype=np.float64,
    )

    second = np.asarray(
        second,
        dtype=np.float64,
    )

    first_norm = np.linalg.norm(
        first
    )

    second_norm = np.linalg.norm(
        second
    )

    if (
        first_norm == 0
        or second_norm == 0
    ):
        return 0.0

    similarity = np.dot(
        first,
        second,
    ) / (
        first_norm
        * second_norm
    )

    return float(
        np.clip(
            similarity,
            0.0,
            1.0,
        )
    )


def validation_f1_for_weights(
    weights,
):
    eval_model.set_weights(
        weights
    )

    predictions = np.argmax(
        eval_model.predict(
            x_val,
            batch_size=2048,
            verbose=0,
        ),
        axis=1,
    )

    _, _, _, _, f1 = all_metrics(
        y_val,
        predictions,
    )

    return float(f1)


def calculate_shap_quality(
    records,
):
    """
    SHAP quality =
        70% consensus agreement
        + 30% temporal stability
    """

    shap_matrix = np.asarray([
        [
            record["shap"][
                feature_name
            ]
            for feature_name
            in feature_names
        ]
        for record in records
    ], dtype=np.float64)


    row_norms = np.linalg.norm(
        shap_matrix,
        axis=1,
        keepdims=True,
    )

    row_norms[
        row_norms == 0
    ] = 1.0


    normalized_matrix = (
        shap_matrix
        / row_norms
    )


    consensus = np.median(
        normalized_matrix,
        axis=0,
    )


    consensus_norm = np.linalg.norm(
        consensus
    )


    if consensus_norm > 0:
        consensus = (
            consensus
            / consensus_norm
        )


    quality_rows = []


    for record, current_vector in zip(
        records,
        normalized_matrix,
    ):
        client_id = int(
            record["client_id"]
        )


        agreement = cosine_similarity_vector(
            current_vector,
            consensus,
        )


        previous_vector = (
            previous_shap_by_client.get(
                client_id
            )
        )


        if previous_vector is None:
            stability = 1.0

        elif record[
            "shap_calculated"
        ] == 1:
            stability = cosine_similarity_vector(
                current_vector,
                previous_vector,
            )

        else:
            # The SHAP vector was reused.
            stability = 1.0


        shap_quality = (
            CFG[
                "consensus_influence"
            ]
            * agreement

            + CFG[
                "stability_influence"
            ]
            * stability
        )


        quality_rows.append({
            "client_id": client_id,

            "consensus_alignment":
                float(agreement),

            "shap_stability":
                float(stability),

            "shap_quality":
                float(shap_quality),
        })


        if record[
            "shap_calculated"
        ] == 1:
            previous_shap_by_client[
                client_id
            ] = current_vector.copy()


    return quality_rows


def fit_config(
    server_round,
):
    return {
        "server_round": int(
            server_round
        ),

        "proximal_mu": float(
            CFG[
                "proximal_mu"
            ]
        ),
    }


def eval_config(
    server_round,
):
    return {
        "server_round": int(
            server_round
        ),
    }


def evaluate_fn(
    server_round,
    parameters,
    config,
):
    global best_val_f1
    global best_round
    global best_weights


    eval_model.set_weights(
        parameters
    )


    validation_predictions = np.argmax(
        eval_model.predict(
            x_val,
            batch_size=2048,
            verbose=0,
        ),
        axis=1,
    )


    (
        val_accuracy,
        val_balanced_accuracy,
        val_precision,
        val_recall,
        val_f1,
    ) = all_metrics(
        y_val,
        validation_predictions,
    )


    test_loss, _ = eval_model.evaluate(
        x_test,
        y_test,
        batch_size=2048,
        verbose=0,
    )


    test_predictions = np.argmax(
        eval_model.predict(
            x_test,
            batch_size=2048,
            verbose=0,
        ),
        axis=1,
    )


    (
        test_accuracy,
        test_balanced_accuracy,
        test_precision,
        test_recall,
        test_f1,
    ) = all_metrics(
        y_test,
        test_predictions,
    )


    server_history.append({
        "round": int(server_round),

        "val_accuracy":
            val_accuracy,

        "val_balanced_accuracy":
            val_balanced_accuracy,

        "val_precision":
            val_precision,

        "val_recall":
            val_recall,

        "val_f1":
            val_f1,

        "test_loss":
            float(test_loss),

        "test_accuracy":
            test_accuracy,

        "test_balanced_accuracy":
            test_balanced_accuracy,

        "test_precision":
            test_precision,

        "test_recall":
            test_recall,

        "test_f1":
            test_f1,
    })


    pd.DataFrame(
        server_history
    ).to_csv(
        RESULT_DIR
        / "server_metrics.csv",
        index=False,
    )


    if (
        server_round > 0
        and val_f1 > best_val_f1
    ):
        best_val_f1 = val_f1

        best_round = int(
            server_round
        )

        best_weights = [
            weight.copy()
            for weight in parameters
        ]

        eval_model.save_weights(
            RESULT_DIR
            / "best_global.weights.h5"
        )


    print(
        f"[Server] R{server_round}: "
        f"val_f1={val_f1:.4f}, "
        f"test_acc={test_accuracy:.4f}, "
        f"test_bal_acc="
        f"{test_balanced_accuracy:.4f}, "
        f"test_precision="
        f"{test_precision:.4f}, "
        f"test_recall="
        f"{test_recall:.4f}, "
        f"test_f1={test_f1:.4f}"
    )


    return float(test_loss), {
        "accuracy":
            test_accuracy,

        "balanced_accuracy":
            test_balanced_accuracy,

        "precision":
            test_precision,

        "recall":
            test_recall,

        "f1":
            test_f1,

        "val_f1":
            val_f1,
    }


class ExperimentStrategy(
    fl.server.strategy.FedAvg
):
    def aggregate_fit(
        self,
        server_round,
        results,
        failures,
    ):
        if not results:
            return None, {}


        records = []


        for fallback_index, (
            _,
            fit_res,
        ) in enumerate(results):

            fit_metrics = dict(
                fit_res.metrics
            )


            client_id = int(
                fit_metrics.get(
                    "client_id",
                    fallback_index + 1,
                )
            )


            client_shap = {
                key.split(
                    "shap::",
                    1,
                )[1]: float(value)

                for key, value
                in fit_metrics.items()

                if key.startswith(
                    "shap::"
                )
            }


            if CFG[
                "aggregation"
            ] == "performance_shap_utility":

                expected = set(
                    feature_names
                )

                returned = set(
                    client_shap
                )

                missing = (
                    expected
                    - returned
                )

                extra = (
                    returned
                    - expected
                )

                if missing or extra:
                    raise ValueError(
                        f"Client {client_id} "
                        f"SHAP mismatch.\n"
                        f"Missing: "
                        f"{sorted(missing)}\n"
                        f"Extra: "
                        f"{sorted(extra)}"
                    )


            records.append({
                "client_id":
                    client_id,

                "num_examples":
                    int(
                        fit_res.num_examples
                    ),

                "weights":
                    parameters_to_ndarrays(
                        fit_res.parameters
                    ),

                "shap":
                    client_shap,

                "local_loss":
                    float(
                        fit_metrics.get(
                            "local_loss",
                            0.0,
                        )
                    ),

                "local_accuracy":
                    float(
                        fit_metrics.get(
                            "local_accuracy",
                            0.0,
                        )
                    ),

                "local_balanced_accuracy":
                    float(
                        fit_metrics.get(
                            "local_balanced_accuracy",
                            0.0,
                        )
                    ),

                "local_precision":
                    float(
                        fit_metrics.get(
                            "local_precision",
                            0.0,
                        )
                    ),

                "local_recall":
                    float(
                        fit_metrics.get(
                            "local_recall",
                            0.0,
                        )
                    ),

                "local_f1":
                    float(
                        fit_metrics.get(
                            "local_f1",
                            0.0,
                        )
                    ),

                "shap_source_round":
                    int(
                        fit_metrics.get(
                            "shap_source_round",
                            0,
                        )
                    ),

                "shap_calculated":
                    int(
                        fit_metrics.get(
                            "shap_calculated",
                            0,
                        )
                    ),
            })


        records.sort(
            key=lambda record:
                record["client_id"]
        )


        client_ids = [
            record["client_id"]
            for record in records
        ]


        for record in records:
            for feature_name in feature_names:
                shap_history.append({
                    "round":
                        int(server_round),

                    "client_id":
                        int(
                            record["client_id"]
                        ),

                    "feature":
                        feature_name,

                    "shap_value":
                        float(
                            record["shap"][
                                feature_name
                            ]
                        ),

                    "shap_source_round":
                        int(
                            record[
                                "shap_source_round"
                            ]
                        ),

                    "shap_calculated_this_round":
                        int(
                            record[
                                "shap_calculated"
                            ]
                        ),
                })


        pd.DataFrame(
            shap_history
        ).to_csv(
            RESULT_DIR
            / "shap_values.csv",
            index=False,
        )


        size_weights = normalized([
            record["num_examples"]
            for record in records
        ])


        quality_rows = (
            calculate_shap_quality(
                records
            )
        )


        quality_by_client = {
            row["client_id"]: row
            for row in quality_rows
        }


        validation_f1_scores = np.asarray([
            validation_f1_for_weights(
                record["weights"]
            )
            for record in records
        ], dtype=np.float64)


        shap_quality_values = np.asarray([
            quality_by_client[
                record["client_id"]
            ]["shap_quality"]

            for record in records
        ], dtype=np.float64)


        utility_scores = (
            validation_f1_scores
            * shap_quality_values
        )


        utility_weights = normalized(
            utility_scores
        )


        final_weights = (
            CFG["size_influence"]
            * size_weights

            + CFG["utility_influence"]
            * utility_weights
        )


        final_weights = normalized(
            final_weights
        )


        aggregated_weights = None


        for record, client_weight in zip(
            records,
            final_weights,
        ):
            if aggregated_weights is None:
                aggregated_weights = [
                    layer
                    * client_weight

                    for layer
                    in record["weights"]
                ]

            else:
                for layer_index, layer in enumerate(
                    record["weights"]
                ):
                    aggregated_weights[
                        layer_index
                    ] += (
                        layer
                        * client_weight
                    )


        for (
            record,
            size_weight,
            validation_f1,
            utility_score,
            utility_weight,
            final_weight,
        ) in zip(
            records,
            size_weights,
            validation_f1_scores,
            utility_scores,
            utility_weights,
            final_weights,
        ):
            quality = quality_by_client[
                record["client_id"]
            ]


            aggregation_history.append({
                "round":
                    int(server_round),

                "client_id":
                    int(
                        record["client_id"]
                    ),

                "num_examples":
                    int(
                        record["num_examples"]
                    ),

                "size_weight":
                    float(size_weight),

                "consensus_alignment":
                    float(
                        quality[
                            "consensus_alignment"
                        ]
                    ),

                "shap_stability":
                    float(
                        quality[
                            "shap_stability"
                        ]
                    ),

                "shap_quality":
                    float(
                        quality[
                            "shap_quality"
                        ]
                    ),

                "validation_f1":
                    float(
                        validation_f1
                    ),

                "utility_score":
                    float(
                        utility_score
                    ),

                "utility_weight":
                    float(
                        utility_weight
                    ),

                "final_weight":
                    float(
                        final_weight
                    ),

                "local_loss":
                    float(
                        record["local_loss"]
                    ),

                "local_accuracy":
                    float(
                        record[
                            "local_accuracy"
                        ]
                    ),

                "local_balanced_accuracy":
                    float(
                        record[
                            "local_balanced_accuracy"
                        ]
                    ),

                "local_precision":
                    float(
                        record[
                            "local_precision"
                        ]
                    ),

                "local_recall":
                    float(
                        record[
                            "local_recall"
                        ]
                    ),

                "local_f1":
                    float(
                        record[
                            "local_f1"
                        ]
                    ),

                "shap_source_round":
                    int(
                        record[
                            "shap_source_round"
                        ]
                    ),

                "shap_calculated":
                    int(
                        record[
                            "shap_calculated"
                        ]
                    ),
            })


        pd.DataFrame(
            aggregation_history
        ).to_csv(
            RESULT_DIR
            / "aggregation_weights.csv",
            index=False,
        )


        print(
            f"[Server] R{server_round} "
            f"final weights: "
            + ", ".join(
                f"C{client_id}="
                f"{weight:.3f}"

                for client_id, weight
                in zip(
                    client_ids,
                    final_weights,
                )
            )
        )


        return (
            ndarrays_to_parameters(
                aggregated_weights
            ),
            {},
        )


    def aggregate_evaluate(
        self,
        server_round,
        results,
        failures,
    ):
        if not results:
            return None, {}


        current_rows = []


        for fallback_index, (
            _,
            evaluate_res,
        ) in enumerate(results):

            metrics = dict(
                evaluate_res.metrics
            )


            client_id = int(
                metrics.get(
                    "client_id",
                    fallback_index + 1,
                )
            )


            row = {
                "round":
                    int(server_round),

                "client_id":
                    client_id,

                "num_test_examples":
                    int(
                        evaluate_res.num_examples
                    ),

                "loss":
                    float(
                        evaluate_res.loss
                    ),

                "accuracy":
                    float(
                        metrics.get(
                            "accuracy",
                            0.0,
                        )
                    ),

                "balanced_accuracy":
                    float(
                        metrics.get(
                            "balanced_accuracy",
                            0.0,
                        )
                    ),

                "precision":
                    float(
                        metrics.get(
                            "precision",
                            0.0,
                        )
                    ),

                "recall":
                    float(
                        metrics.get(
                            "recall",
                            0.0,
                        )
                    ),

                "f1":
                    float(
                        metrics.get(
                            "f1",
                            0.0,
                        )
                    ),
            }


            client_history.append(
                row
            )

            current_rows.append(
                row
            )


        client_history.sort(
            key=lambda row: (
                row["round"],
                row["client_id"],
            )
        )


        pd.DataFrame(
            client_history
        ).to_csv(
            RESULT_DIR
            / "client_metrics.csv",
            index=False,
        )


        current_rows.sort(
            key=lambda row:
                row["client_id"]
        )


        print(
            f"[Server] R{server_round} "
            f"client metrics:"
        )


        for row in current_rows:
            print(
                f"  C{row['client_id']}: "
                f"acc={row['accuracy']:.4f}, "
                f"bal_acc="
                f"{row['balanced_accuracy']:.4f}, "
                f"precision="
                f"{row['precision']:.4f}, "
                f"recall="
                f"{row['recall']:.4f}, "
                f"f1={row['f1']:.4f}"
            )


        return super().aggregate_evaluate(
            server_round,
            results,
            failures,
        )


strategy = ExperimentStrategy(
    fraction_fit=1.0,
    fraction_evaluate=1.0,

    min_fit_clients=NUM_CLIENTS,
    min_evaluate_clients=NUM_CLIENTS,
    min_available_clients=NUM_CLIENTS,

    on_fit_config_fn=fit_config,
    on_evaluate_config_fn=eval_config,

    evaluate_fn=evaluate_fn,
)


if len(sys.argv) < 2:
    raise ValueError(
        "Provide the server port.\n"
        "Example: python server.py 8080"
    )


fl.server.start_server(
    server_address=(
        "localhost:"
        + str(sys.argv[1])
    ),

    config=fl.server.ServerConfig(
        num_rounds=CFG[
            "num_rounds"
        ]
    ),

    grpc_max_message_length=(
        1024
        * 1024
        * 1024
    ),

    strategy=strategy,
)


print(
    f"[Server] Best validation "
    f"macro F1={best_val_f1:.4f} "
    f"at round {best_round}"
)