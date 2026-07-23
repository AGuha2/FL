from tensorflow import keras
from tensorflow.keras import layers


def build_model(input_features, num_classes, num_timesteps=1, output_activation='softmax'):
    """
    CNN-LSTM architecture, per the provided model diagram:

        Input               Input(shape=(None, shape, 1))
        Conv1D              TimeDistributed(Conv1D(128, 3, activation='relu'))
        BatchNormalization  TimeDistributed(BatchNormalization())
        MaxPool1D           TimeDistributed(MaxPool1D(2, 2))
        Conv1D              TimeDistributed(Conv1D(64, 3, activation='relu'))
        BatchNormalization  TimeDistributed(BatchNormalization())
        MaxPool1D           TimeDistributed(MaxPool1D(2, 2))
        Flatten             TimeDistributed(Flatten())
        LSTM                LSTM(128, activation='tanh', return_sequences=True)
        LSTM                LSTM(128, activation='tanh')
        Dense               Dense(50, activation='tanh')
        Dense               Dense(100, activation='tanh')
        Dropout             Dropout(rate=0.2)
        Dense               Dense(1, activation='sigmoid')

    EXPECTED INPUT SHAPE
    ---------------------
    The TimeDistributed/Conv1D/LSTM stack expects a SEQUENCE per sample:
        X.shape == (num_samples, num_timesteps, input_features, 1)

    NOTE ON num_timesteps:
    The original diagram uses `None` (a symbolic, variable-length timesteps
    dimension) so the model could accept sequences of different lengths.
    This builder instead takes a concrete `num_timesteps` (default 1) for
    two reasons:
      1. Every caller in this codebase currently uses a fixed, single value
         (NUM_TIMESTEPS = 1 — each row reshaped as a 1-step "sequence" since
         real multi-timestep windowed data isn't set up yet).
      2. A symbolic None dimension breaks shap.DeepExplainer, which inspects
         the model's static input shape and tries to reshape samples against
         it internally — a None in that shape tuple crashes with
         "'NoneType' object cannot be interpreted as an integer".
    If you later build genuine variable-length sequence data, you can still
    pass num_timesteps=None here — just know SHAP explanation will need a
    different explainer (e.g. shap.GradientExplainer) or a fixed-length
    sample workaround at that point.

    OUTPUT LAYER
    ------------
    The diagram specifies Dense(1, activation='sigmoid') — binary
    classification. This builder defaults to Dense(num_classes,
    activation='softmax') instead, to stay compatible with an existing
    multi-class pipeline (sparse_categorical_crossentropy, label-encoded
    classes, per-class precision/recall/F1, etc.). Pass
    num_classes=1, output_activation='sigmoid' to reproduce the diagram
    exactly for a binary setup (and switch the training loss to
    'binary_crossentropy' accordingly).
    """
    inputs = keras.Input(shape=(num_timesteps, input_features, 1))

    x = layers.TimeDistributed(layers.Conv1D(128, 3, activation='relu'))(inputs)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed(layers.MaxPooling1D(2, 2))(x)

    x = layers.TimeDistributed(layers.Conv1D(64, 3, activation='relu'))(x)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed(layers.MaxPooling1D(2, 2))(x)

    x = layers.TimeDistributed(layers.Flatten())(x)

    x = layers.LSTM(128, activation='tanh', return_sequences=True)(x)
    x = layers.LSTM(128, activation='tanh')(x)

    x = layers.Dense(50, activation='tanh')(x)
    x = layers.Dense(100, activation='tanh')(x)
    x = layers.Dropout(rate=0.2)(x)

    outputs = layers.Dense(num_classes, activation=output_activation)(x)

    return keras.Model(inputs=inputs, outputs=outputs)