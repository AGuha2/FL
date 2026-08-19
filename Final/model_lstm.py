
"""CNN-LSTM model shared by the federated server and all five clients."""
from tensorflow import keras
from tensorflow.keras import layers

def build_model(
    num_features: int,
    num_classes: int,
    output_activation: str = "softmax",
) -> keras.Model:
    
    inputs = keras.Input(shape=(num_features,), name="network_features")
    x = layers.Reshape((num_features, 1), name="feature_sequence")(inputs)
    # Single CNN block 
    x = layers.Conv1D(
        128,
        kernel_size=3,
        padding="same",
        activation="relu",
        name="conv1",
    )(x)
    x = layers.LayerNormalization(name="layer_norm")(x)
    x = layers.MaxPooling1D(pool_size=2, name="max_pool1")(x)
    
    x = layers.LSTM(
        128,
        activation="tanh",
        return_sequences=True,
        name="lstm1",
    )(x)
    x = layers.LSTM(
        128,
        activation="tanh",
        return_sequences=False,
        name="lstm2",
    )(x)
    x = layers.Dense(50, activation="tanh", name="dense50")(x)
    x = layers.Dense(100, activation="tanh", name="dense100")(x)
    x = layers.Dropout(0.2, name="dropout")(x)
    outputs = layers.Dense(
        num_classes,
        activation=output_activation,
        name="class_probabilities",
    )(x)
    return keras.Model(inputs, outputs, name="federated_cnn_lstm")
