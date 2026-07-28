# simulation.py

import flwr as fl
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────
# SECTION 1: data loading
# One function that loads ANY of your CSVs
# by client ID
# ─────────────────────────────────────────
DATASETS = {
    "0": r"D:\CAPSTONE\Merged01.csv",
    "1": r"D:\CAPSTONE\Merged02.csv",
    # add more clients just by adding lines:
    # "2": r"D:\CAPSTONE\Merged03.csv",
}

def load_data(cid):
    path = DATASETS[cid]
    df = pd.read_csv(path)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna()

    X = df.drop(columns=["Label"])
    y = LabelEncoder().fit_transform(df["Label"])
    X = RobustScaler().fit_transform(X)

    return train_test_split(X, y, test_size=0.2,
                            random_state=42, stratify=y)


# ─────────────────────────────────────────
# SECTION 2: model definition
# Same architecture as before
# ─────────────────────────────────────────
def build_model(input_dim, num_classes):
    model = keras.Sequential([
        keras.layers.Dense(128, activation="relu",
                           input_shape=(input_dim,)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


# ─────────────────────────────────────────
# SECTION 3: FlowerClient
# Exactly the same as your old client1.py
# and client2.py — just combined
# ─────────────────────────────────────────
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid):
        self.cid = cid
        x_train, x_test, y_train, y_test = load_data(cid)
        self.x_train = x_train
        self.x_test  = x_test
        self.y_train = y_train
        self.y_test  = y_test
        num_classes  = len(set(y_train))
        self.model   = build_model(x_train.shape[1], num_classes)

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.x_train, self.y_train,
                       epochs=1, verbose=0)
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, acc = self.model.evaluate(
            self.x_test, self.y_test, verbose=0)
        print(f"Client {self.cid} accuracy: {acc:.4f}")
        return loss, len(self.x_test), {"accuracy": acc}


# ─────────────────────────────────────────
# SECTION 4: client_fn
# Flower calls this to CREATE each client.
# cid is just a string: "0", "1", "2" ...
# ─────────────────────────────────────────
def client_fn(cid):
    return FlowerClient(cid)


# ─────────────────────────────────────────
# SECTION 5: run simulation
# This replaces ALL THREE of your terminals
# ─────────────────────────────────────────
fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=2,           # matches how many datasets you have
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=fl.server.strategy.FedAvg(),
)