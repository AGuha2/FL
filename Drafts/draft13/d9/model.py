from tensorflow import keras
from tensorflow.keras import layers


def build_model(
    input_features,
    num_classes,
    output_activation="softmax",
    model_type="mlp",
):
    model_type = str(model_type).strip().lower()

    if model_type == "cnn":
        inputs = keras.Input(
            shape=(input_features,),
            name="input_features",
        )

        x = layers.Reshape(
            (input_features, 1),
            name="reshape_for_cnn",
        )(inputs)

        x = layers.Conv1D(
            64,
            kernel_size=3,
            padding="same",
            activation="relu",
        )(x)

        x = layers.LayerNormalization()(x)

        x = layers.Conv1D(
            64,
            kernel_size=3,
            padding="same",
            activation="relu",
        )(x)

        x = layers.GlobalAveragePooling1D()(x)

        x = layers.Dense(
            128,
            activation="relu",
        )(x)

        x = layers.Dropout(0.3)(x)

        x = layers.Dense(
            64,
            activation="relu",
        )(x)

        x = layers.Dropout(0.2)(x)

        outputs = layers.Dense(
            num_classes,
            activation=output_activation,
        )(x)

        return keras.Model(
            inputs=inputs,
            outputs=outputs,
            name="cnn_classifier",
        )
    elif model_type == "mlp":
        model = keras.Sequential(
            [
                layers.Input(shape=(input_features,)),

                layers.Dense(256, activation="relu"),
                layers.LayerNormalization(),
                layers.Dropout(0.30),

                layers.Dense(128, activation="relu"),
                layers.LayerNormalization(),
                layers.Dropout(0.25),

                layers.Dense(64, activation="relu"),
                layers.Dropout(0.15),

                layers.Dense(
                    num_classes,
                    activation=output_activation,
                ),
            ],
            name="mlp_classifier",
        )

        return model

    else:
        raise ValueError(
            f"Unsupported model_type: {model_type!r}. "
            "Use 'cnn' or 'mlp'."
        )