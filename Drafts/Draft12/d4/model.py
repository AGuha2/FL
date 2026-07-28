from tensorflow import keras
from tensorflow.keras import layers


def build_model(input_features, num_classes, output_activation='softmax'):
    """
    Plain feed-forward Dense architecture — the original working model
    before the CNN/CNN-LSTM experiments.

        Input               Input(shape=(input_features,))
        Dense               Dense(256, activation='relu')
        BatchNormalization  BatchNormalization()
        Dropout             Dropout(rate=0.3)
        Dense               Dense(256, activation='relu')
        BatchNormalization  BatchNormalization()
        Dropout             Dropout(rate=0.3)
        Dense               Dense(128, activation='relu')
        Dense               Dense(num_classes, activation='softmax')

    Why this is the appropriate architecture for this data: it's tabular
    network-flow features in arbitrary column order — there's no spatial or
    sequential locality between adjacent columns for a Conv1D kernel or an
    LSTM to exploit. A Dense stack has no such assumption built in, trains
    faster per round, and is exactly what shap.DeepExplainer's op-handler
    rules were designed for (Dense/BatchNorm/Dropout/ReLU), so there's no
    SHAP-compatibility risk at all.

    Takes flat 2D input (samples, input_features) — same calling convention
    as every other model in this project (see model_cnn.py), so swapping
    which architecture is active is just replacing the contents of
    model.py; nothing in client.py/server.py needs to change.
    """
    inputs = keras.Input(shape=(input_features,))

    x = layers.Dense(256, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(128, activation='relu')(x)

    outputs = layers.Dense(num_classes, activation=output_activation)(x)

    return keras.Model(inputs=inputs, outputs=outputs)
