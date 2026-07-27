"""Model definition shared by the server and every client."""
from tensorflow import keras
from tensorflow.keras import layers


def build_model(num_features: int, num_classes: int) -> keras.Model:
    inputs = keras.Input(shape=(num_features,), name="network_features")

    x = layers.Dense(256, kernel_initializer="he_normal")(inputs)
    x = layers.LayerNormalization()(x)
    x = layers.Activation("swish")(x)
    x = layers.Dropout(0.20)(x)

    residual = layers.Dense(128)(x)
    x = layers.Dense(128, activation="swish", kernel_initializer="he_normal")(x)
    x = layers.Dropout(0.15)(x)
    x = layers.Dense(128, activation="swish")(x)
    x = layers.Add()([x, residual])
    x = layers.LayerNormalization()(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="federated_ids_8class")
