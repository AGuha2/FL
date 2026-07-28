from tensorflow import keras
from tensorflow.keras import layers


def build_model(input_features, num_classes, output_activation='softmax', model_type='mlp'):
    if model_type == 'cnn':
        inputs = keras.Input(shape=(input_features,))
        x = layers.Reshape((input_features, 1))(inputs)
        x = layers.Conv1D(64, 3, padding='same', activation='relu')(x)
        x = layers.LayerNormalization()(x)
        x = layers.Conv1D(64, 3, padding='same', activation='relu')(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(num_classes, activation=output_activation)(x)
        return keras.Model(inputs, outputs)

    inputs = keras.Input(shape=(input_features,))
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation=output_activation)(x)
    return keras.Model(inputs, outputs)
