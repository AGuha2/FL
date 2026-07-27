from tensorflow import keras
from tensorflow.keras import layers


def build_model(input_features, num_classes, output_activation='softmax'):
    """
    CNN architecture, adapted from the provided model diagram with the LSTM
    layers removed:

        Input               Input(shape=(input_features, 1))
        Conv1D              Conv1D(128, 3, activation='relu')
        BatchNormalization  BatchNormalization()
        MaxPool1D           MaxPool1D(2, 2)
        Conv1D              Conv1D(64, 3, activation='relu')
        BatchNormalization  BatchNormalization()
        MaxPool1D           MaxPool1D(2, 2)
        Flatten             Flatten()
        Dense               Dense(50, activation='tanh')
        Dense               Dense(100, activation='tanh')
        Dropout             Dropout(rate=0.2)
        Dense               Dense(num_classes, activation='softmax')

    WHY THE LSTM LAYERS AND TimeDistributed/sequence WRAPPING WERE DROPPED
    ------------------------------------------------------------------
    1. shap.DeepExplainer (the fast, single-backward-pass SHAP method) does
       not reliably support LSTM/GRU/any recurrent layer — its DeepLIFT-style
       op-override mechanism has no rule for LSTM's fused recurrent ops or
       its variable x variable gate multiplications, and crashes with
       "'TFDeep' object has no attribute 'between_tensors'". Removing the
       recurrent layers makes DeepExplainer usable again — this is a
       feed-forward CNN, which is exactly what DeepLIFT/DeepExplainer was
       designed for.
    2. The pipeline never had genuine multi-timestep sequence data — each
       row was being reshaped as a 1-step "sequence" purely so the LSTM
       layers had valid input. Without LSTM, that reshape and the
       TimeDistributed wrapping around every Conv/BatchNorm/MaxPool/Flatten
       layer serve no purpose (TimeDistributed over a sequence of length 1
       is mathematically identical to just applying the layer directly), so
       both were removed. Input is now plain 2D: (input_features, 1) instead
       of (num_timesteps, input_features, 1).

    OUTPUT LAYER
    ------------
    The diagram specifies Dense(1, activation='sigmoid') — binary
    classification. This builder defaults to Dense(num_classes,
    activation='softmax') instead, to stay compatible with the existing
    multi-class pipeline (sparse_categorical_crossentropy, label-encoded
    classes, per-class precision/recall/F1, etc.). Pass
    num_classes=1, output_activation='sigmoid' to reproduce the diagram's
    binary output instead (and switch the training loss to
    'binary_crossentropy' accordingly).
    """
    inputs = keras.Input(shape=(input_features, 1))

    x = layers.Conv1D(128, 3, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2, 2)(x)

    x = layers.Conv1D(64, 3, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2, 2)(x)

    x = layers.Flatten()(x)

    x = layers.Dense(50, activation='tanh')(x)
    x = layers.Dense(100, activation='tanh')(x)
    x = layers.Dropout(rate=0.2)(x)

    outputs = layers.Dense(num_classes, activation=output_activation)(x)

    return keras.Model(inputs=inputs, outputs=outputs)
