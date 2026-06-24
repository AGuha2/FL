import flwr as fl
import tensorflow as tf
from tensorflow import keras
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from Labels import le, NUM_CLASSES, NUM_FEATURES

# Load dataset

df = pd.read_csv('D:\CAPSTONE\Merged02.csv')

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df = df.dropna()

print(df.head())

X = df.drop(columns=['Label'])
y = df['Label'].str.strip().str.upper()  

y = le.transform(y)

scaler = RobustScaler()
X = scaler.fit_transform(X)

scaler = RobustScaler()
X = scaler.fit_transform(X)

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {len(x_train)} | Test: {len(x_test)}")

model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(NUM_CLASSES, activation='softmax')
])
model.compile(
    optimizer = keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self,client_id):
        self.client_id = client_id

    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        r = model.fit(x_train, y_train, epochs=1, validation_data=(x_test, y_test), verbose=0)
        val_acc = np.mean(r.history['val_accuracy'])
        val_loss = np.mean(r.history['val_loss'])
        print(f"[Client {self.client_id}] LOCAL VAL   → loss: {val_loss:.4f} | accuracy: {val_acc:.4f}")
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        print(f"[Client {self.client_id}] GLOBAL → loss: {loss:.4f} | accuracy: {accuracy:.4f}")
        return loss, len(x_test), {"accuracy": accuracy}

# Start Flower client
fl.client.start_numpy_client(
        server_address="localhost:"+str(sys.argv[1]), 
        client=FlowerClient(client_id=2), 
        grpc_max_message_length = 1024*1024*1024
)
