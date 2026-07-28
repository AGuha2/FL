import random
import sys
from typing import Dict

import flwr as fl
import joblib
import numpy as np
import pandas as pd
import shap
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras

from Labels import le, NUM_CLASSES, NUM_FEATURES
from experiment_config import CFG
from model import build_model


DATA_TEMPLATE = (
    r"D:\CAPSTONE\Dir_Client{client_id}.csv"
)

SCALER_PATH = (
    r"D:\CAPSTONE\global_scaler.pkl"
)

COMMON_SHAP_REFERENCE_PATH = (
    r"D:\CAPSTONE\Common_SHAP_Reference.npz"
)

SEED = 42


def set_seeds(
    seed: int,
):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(
        seed
    )


def calculate_metrics(
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


def build_class_weights(
    y_train: np.ndarray,
) -> Dict[int, float]:

    classes = np.unique(
        y_train
    )


    if not CFG[
        "use_class_weights"
    ]:
        return {
            int(class_id): 1.0
            for class_id in classes
        }


    raw_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )


    raw_weights = np.clip(
        raw_weights,
        0.0,
        CFG[
            "max_class_weight"
        ],
    )


    counts = (
        pd.Series(y_train)
        .value_counts()
    )


    for index, class_id in enumerate(
        classes
    ):
        if counts.get(
            class_id,
            0,
        ) < CFG[
            "min_samples_for_weighting"
        ]:
            raw_weights[index] = 1.0


    return {
        int(class_id): float(weight)
        for class_id, weight
        in zip(
            classes,
            raw_weights,
        )
    }


def extract_mean_abs_shap(
    model,
    background,
    sample,
    feature_names,
):
    explainer = shap.GradientExplainer(
        model,
        background,
    )


    shap_values = explainer.shap_values(
        sample
    )


    if isinstance(
        shap_values,
        list,
    ):
        shap_array = np.stack(
            [
                np.asarray(value)
                for value in shap_values
            ],
            axis=0,
        )

    else:
        shap_array = np.asarray(
            shap_values
        )


    number_of_features = len(
        feature_names
    )


    print(
        f"[Client] SHAP shape: "
        f"{shap_array.shape}"
    )


    if (
        shap_array.ndim == 3
        and shap_array.shape[-1]
        == number_of_features
    ):
        mean_absolute_shap = np.mean(
            np.abs(shap_array),
            axis=(0, 1),
        )


    elif (
        shap_array.ndim == 3
        and shap_array.shape[1]
        == number_of_features
    ):
        mean_absolute_shap = np.mean(
            np.abs(shap_array),
            axis=(0, 2),
        )


    elif (
        shap_array.ndim == 2
        and shap_array.shape[1]
        == number_of_features
    ):
        mean_absolute_shap = np.mean(
            np.abs(shap_array),
            axis=0,
        )


    else:
        raise ValueError(
            f"Unexpected SHAP shape "
            f"{shap_array.shape}; "
            f"expected "
            f"{number_of_features} "
            f"features."
        )


    if (
        len(mean_absolute_shap)
        != number_of_features
    ):
        raise ValueError(
            f"SHAP produced "
            f"{len(mean_absolute_shap)} "
            f"values for "
            f"{number_of_features} "
            f"features."
        )


    return {
        feature_name: float(
            shap_value
        )

        for feature_name, shap_value
        in zip(
            feature_names,
            mean_absolute_shap,
        )
    }


def run_client(
    client_id: int,
):
    set_seeds(
        SEED + client_id
    )


    data_path = DATA_TEMPLATE.format(
        client_id=client_id
    )


    print(
        f"[Client {client_id}] "
        f"Loading {data_path}"
    )


    dataframe = pd.read_csv(
        data_path
    )


    dataframe.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True,
    )


    dataframe.dropna(
        inplace=True
    )


    unnamed_columns = [
        column
        for column in dataframe.columns
        if column.lower().startswith(
            "unnamed:"
        )
    ]


    if unnamed_columns:
        dataframe.drop(
            columns=unnamed_columns,
            inplace=True,
            errors="ignore",
        )


    if "Label" not in dataframe.columns:
        raise ValueError(
            f"Client {client_id}: "
            f"Label column is missing."
        )


    feature_names = [
        column
        for column in dataframe.columns
        if column != "Label"
    ]


    if len(feature_names) != NUM_FEATURES:
        raise ValueError(
            f"Client {client_id}: "
            f"found {len(feature_names)} "
            f"features, expected "
            f"{NUM_FEATURES}.\n"
            f"Features: {feature_names}"
        )


    x_dataframe = dataframe[
        feature_names
    ]


    y = le.transform(
        dataframe["Label"]
        .astype(str)
        .str.strip()
        .str.upper()
    )


    scaler = joblib.load(
        SCALER_PATH
    )


    x = scaler.transform(
        x_dataframe
    ).astype(np.float32)


    try:
        (
            x_train,
            x_test,
            y_train,
            y_test,
        ) = train_test_split(
            x,
            y,
            test_size=0.2,
            random_state=SEED,
            stratify=y,
        )

    except ValueError:
        (
            x_train,
            x_test,
            y_train,
            y_test,
        ) = train_test_split(
            x,
            y,
            test_size=0.2,
            random_state=SEED,
        )


    print(
        f"[Client {client_id}] "
        f"Train rows: {len(x_train):,}; "
        f"test rows: {len(x_test):,}"
    )


    class_weights = build_class_weights(
        y_train
    )


    model = build_model(
        NUM_FEATURES,
        NUM_CLASSES,
        model_type=CFG[
            "model_type"
        ],
    )


    optimizer = keras.optimizers.Adam(
        learning_rate=CFG[
            "initial_lr"
        ],

        clipnorm=1.0,
    )


    model.compile(
        optimizer=optimizer,

        loss=(
            "sparse_categorical_crossentropy"
        ),

        metrics=["accuracy"],
    )


    loss_function = (
        keras.losses
        .SparseCategoricalCrossentropy(
            reduction=(
                keras.losses
                .Reduction.NONE
            )
        )
    )


    # ========================================================
    # Load common server-derived SHAP reference
    # ========================================================

    reference_data = np.load(
        COMMON_SHAP_REFERENCE_PATH,
        allow_pickle=False,
    )


    common_reference = reference_data[
        "x_reference"
    ].astype(np.float32)


    reference_feature_names = (
        reference_data[
            "feature_names"
        ]
        .astype(str)
        .tolist()
    )


    if reference_feature_names != feature_names:
        missing = (
            set(feature_names)
            - set(
                reference_feature_names
            )
        )

        extra = (
            set(
                reference_feature_names
            )
            - set(feature_names)
        )

        raise ValueError(
            f"Client {client_id}: "
            f"common SHAP reference "
            f"feature mismatch.\n"
            f"Missing: {sorted(missing)}\n"
            f"Extra: {sorted(extra)}"
        )


    reference_rng = np.random.RandomState(
        SEED
    )


    background_size = min(
        CFG[
            "shap_background_size"
        ],
        len(common_reference),
    )


    shap_sample_size = min(
        CFG[
            "shap_sample_size"
        ],
        len(common_reference),
    )


    background_indices = (
        reference_rng.choice(
            len(common_reference),
            size=background_size,
            replace=False,
        )
    )


    available_sample_indices = np.setdiff1d(
        np.arange(
            len(common_reference)
        ),
        background_indices,
    )


    if (
        len(available_sample_indices)
        >= shap_sample_size
    ):
        shap_sample_indices = (
            reference_rng.choice(
                available_sample_indices,
                size=shap_sample_size,
                replace=False,
            )
        )

    else:
        shap_sample_indices = (
            reference_rng.choice(
                len(common_reference),
                size=shap_sample_size,
                replace=False,
            )
        )


    shap_background = common_reference[
        background_indices
    ]


    shap_sample = common_reference[
        shap_sample_indices
    ]


    print(
        f"[Client {client_id}] "
        f"Common SHAP background="
        f"{len(shap_background)}, "
        f"sample={len(shap_sample)}"
    )


    last_shap = {
        feature_name: 0.0
        for feature_name
        in feature_names
    }


    last_shap_round = 0


    def batches():
        shuffled_indices = (
            np.random.permutation(
                len(x_train)
            )
        )


        sample_weights = np.asarray([
            class_weights.get(
                int(label),
                1.0,
            )
            for label in y_train
        ], dtype=np.float32)


        batch_size = CFG[
            "batch_size"
        ]


        for start_index in range(
            0,
            len(shuffled_indices),
            batch_size,
        ):
            batch_indices = (
                shuffled_indices[
                    start_index:
                    start_index
                    + batch_size
                ]
            )


            yield (
                x_train[
                    batch_indices
                ],

                y_train[
                    batch_indices
                ],

                sample_weights[
                    batch_indices
                ],
            )


    def train_one_epoch(
        global_trainable,
        proximal_mu,
    ):
        global_tensors = [
            tf.constant(weight)
            for weight in global_trainable
        ]


        for (
            x_batch,
            y_batch,
            sample_weight_batch,
        ) in batches():

            x_batch = tf.convert_to_tensor(
                x_batch,
                dtype=tf.float32,
            )

            y_batch = tf.convert_to_tensor(
                y_batch,
                dtype=tf.int32,
            )

            sample_weight_batch = (
                tf.convert_to_tensor(
                    sample_weight_batch,
                    dtype=tf.float32,
                )
            )


            with tf.GradientTape() as tape:
                predictions = model(
                    x_batch,
                    training=True,
                )


                per_sample_loss = (
                    loss_function(
                        y_batch,
                        predictions,
                    )
                )


                cross_entropy_loss = (
                    tf.reduce_sum(
                        per_sample_loss
                        * sample_weight_batch
                    )
                    / (
                        tf.reduce_sum(
                            sample_weight_batch
                        )
                        + 1e-8
                    )
                )


                if proximal_mu > 0:
                    proximal_loss = tf.add_n([
                        tf.reduce_sum(
                            tf.square(
                                local_variable
                                - global_weight
                            )
                        )

                        for (
                            local_variable,
                            global_weight,
                        ) in zip(
                            model.trainable_variables,
                            global_tensors,
                        )
                    ])


                    total_loss = (
                        cross_entropy_loss
                        + (
                            proximal_mu
                            / 2.0
                        )
                        * proximal_loss
                    )


                else:
                    total_loss = (
                        cross_entropy_loss
                    )


            gradients = tape.gradient(
                total_loss,
                model.trainable_variables,
            )


            gradients, _ = (
                tf.clip_by_global_norm(
                    gradients,
                    1.0,
                )
            )


            optimizer.apply_gradients(
                zip(
                    gradients,
                    model.trainable_variables,
                )
            )


    class Client(
        fl.client.NumPyClient
    ):
        def get_parameters(
            self,
            config,
        ):
            return model.get_weights()


        def fit(
            self,
            parameters,
            config,
        ):
            nonlocal last_shap
            nonlocal last_shap_round


            round_number = int(
                config.get(
                    "server_round",
                    0,
                )
            )


            proximal_mu = float(
                config.get(
                    "proximal_mu",
                    0.0,
                )
            )


            model.set_weights(
                parameters
            )


            learning_rate = (
                CFG[
                    "initial_lr"
                ]
                * (
                    CFG[
                        "lr_decay"
                    ]
                    ** max(
                        round_number - 1,
                        0,
                    )
                )
            )


            optimizer.learning_rate.assign(
                learning_rate
            )


            global_trainable = [
                variable.numpy().copy()

                for variable
                in model.trainable_variables
            ]


            for _ in range(
                CFG[
                    "local_epochs"
                ]
            ):
                train_one_epoch(
                    global_trainable,
                    proximal_mu,
                )


            shap_calculated = 0


            if (
                round_number
                in CFG[
                    "shap_rounds"
                ]
            ):
                print(
                    f"[Client {client_id}] "
                    f"Calculating common-reference "
                    f"SHAP for round "
                    f"{round_number}"
                )


                last_shap = (
                    extract_mean_abs_shap(
                        model,
                        shap_background,
                        shap_sample,
                        feature_names,
                    )
                )


                last_shap_round = (
                    round_number
                )


                shap_calculated = 1


            local_loss, _ = model.evaluate(
                x_test,
                y_test,
                batch_size=2048,
                verbose=0,
            )


            local_predictions = np.argmax(
                model.predict(
                    x_test,
                    batch_size=2048,
                    verbose=0,
                ),
                axis=1,
            )


            (
                local_accuracy,
                local_balanced_accuracy,
                local_precision,
                local_recall,
                local_f1,
            ) = calculate_metrics(
                y_test,
                local_predictions,
            )


            output_metrics = {
                "client_id":
                    int(client_id),

                "local_loss":
                    float(local_loss),

                "local_accuracy":
                    float(
                        local_accuracy
                    ),

                "local_balanced_accuracy":
                    float(
                        local_balanced_accuracy
                    ),

                "local_precision":
                    float(
                        local_precision
                    ),

                "local_recall":
                    float(
                        local_recall
                    ),

                "local_f1":
                    float(local_f1),

                "shap_source_round":
                    int(
                        last_shap_round
                    ),

                "shap_calculated":
                    int(
                        shap_calculated
                    ),
            }


            output_metrics.update({
                f"shap::{feature_name}":
                    float(shap_value)

                for (
                    feature_name,
                    shap_value,
                ) in last_shap.items()
            })


            return (
                model.get_weights(),
                len(x_train),
                output_metrics,
            )


        def evaluate(
            self,
            parameters,
            config,
        ):
            model.set_weights(
                parameters
            )


            round_number = int(
                config.get(
                    "server_round",
                    0,
                )
            )


            loss, _ = model.evaluate(
                x_test,
                y_test,
                batch_size=2048,
                verbose=0,
            )


            predictions = np.argmax(
                model.predict(
                    x_test,
                    batch_size=2048,
                    verbose=0,
                ),
                axis=1,
            )


            (
                accuracy,
                balanced_accuracy,
                precision,
                recall,
                f1,
            ) = calculate_metrics(
                y_test,
                predictions,
            )


            return (
                float(loss),
                len(x_test),

                {
                    "client_id":
                        int(client_id),

                    "round":
                        int(round_number),

                    "accuracy":
                        float(accuracy),

                    "balanced_accuracy":
                        float(
                            balanced_accuracy
                        ),

                    "precision":
                        float(precision),

                    "recall":
                        float(recall),

                    "f1":
                        float(f1),
                },
            )


    if len(sys.argv) < 2:
        raise ValueError(
            "Provide the server port.\n"
            "Example: python client1.py 8080"
        )


    fl.client.start_numpy_client(
        server_address=(
            "localhost:"
            + str(sys.argv[1])
        ),

        client=Client(),

        grpc_max_message_length=(
            1024
            * 1024
            * 1024
        ),
    )