"""CNN model shared by the federated server and all five clients.

This is an ablation of the CNN-LSTM architecture. The convolutional front
end and dense classifier are unchanged; only the two LSTM layers are removed
and replaced by Flatten.
"""

from tensorflow import keras
from tensorflow.keras import layers


def build_model(
    num_features: int,
    num_classes: int,
    output_activation: str = "softmax",
) -> keras.Model:
    """Build the CNN ablation of the CNN-LSTM model."""

    inputs = keras.Input(shape=(num_features,), name="network_features")
    x = layers.Reshape((num_features, 1), name="feature_sequence")(inputs)

    # Identical convolutional front end to the CNN-LSTM model.
    x = layers.Conv1D(
        128,
        kernel_size=3,
        padding="same",
        activation="relu",
        name="conv1",
    )(x)
    x = layers.LayerNormalization(name="layer_norm")(x)
    x = layers.MaxPooling1D(pool_size=2, name="max_pool1")(x)

    # The CNN-LSTM has two 128-unit LSTM layers here. Removing them is the
    # only architectural ablation; Flatten connects the CNN output to the
    # unchanged dense classifier.
    x = layers.Flatten(name="flatten")(x)

    x = layers.Dense(50, activation="tanh", name="dense50")(x)
    x = layers.Dense(100, activation="tanh", name="dense100")(x)
    x = layers.Dropout(0.2, name="dropout")(x)

    outputs = layers.Dense(
        num_classes,
        activation=output_activation,
        name="class_probabilities",
    )(x)

    return keras.Model(inputs, outputs, name="federated_cnn")
