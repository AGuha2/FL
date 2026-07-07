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
import shap

# Load dataset

df = pd.read_csv(r'D:\CAPSTONE\NonIID_Client1.csv')

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df = df.dropna()

print(df.head())

X = df.drop(columns=['Label'])
y = df['Label'].str.strip().str.upper()  

# changing the labels to numbers 
y = le.transform(y)


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

#SHAP
bg_size = min(100, x_train.shape[0])
bg_idx = np.random.choice(x_train.shape[0], bg_size, replace=False)
background = x_train[bg_idx]

explainer = shap.DeepExplainer(model, background)

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self,client_id):
        self.client_id = client_id

    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        round_num = config.get("server_round", 0)

        print(f"[Client {self.client_id}] --- ROUND {round_num} (FIT) ---")

        model.set_weights(parameters)
        r = model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test), verbose=0)
        
        val_acc = np.mean(r.history['val_accuracy'])
        val_loss = np.mean(r.history['val_loss'])

        print(f"[Client {self.client_id}] LOCAL VAL   → loss: {val_loss:.4f} | accuracy: {val_acc:.4f}")

               
        sample_size = min(200, x_train.shape[0])
        _, x_sample, _, _ = train_test_split(
            x_train, y_train,
            test_size=sample_size,
            stratify=y_train,
            random_state=42
        )
        
        # SHAP
        shap_values = explainer.shap_values(x_sample, check_additivity=False)

        shap_array = np.array(shap_values)                         # (classes, samples, features)
        mean_abs_shap = np.mean(np.abs(shap_array), axis=(0, 1))  # (features,)

        feature_names = df.drop(columns=['Label']).columns.tolist()

        num_features = mean_abs_shap.shape[0]
        feature_names = feature_names[:num_features]
        
        #print(f"\n[Client {self.client_id}] SHAP Feature Importances (Round {round_num}):")
        print(f"{'Feature':<30} {'Mean |SHAP|':>12}")
        print("-" * 44)

        sorted_idx = np.argsort(mean_abs_shap)[::-1]
        #for i in sorted_idx:
            #print(f"{df.drop(columns=['Label']).columns[i]:<30} {mean_abs_shap[i]:>12.6f}") 

        shap_metrics = {feat: float(mean_abs_shap[i])
                        for i, feat in enumerate(feature_names)}

        return model.get_weights(), len(x_train), shap_metrics

    def evaluate(self, parameters, config):
        round_num = config.get("server_round", 0)
        print(f"[Client {self.client_id}] --- ROUND {round_num} (EVALUATE) ---")
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        print(f"[Client {self.client_id}] GLOBAL → loss: {loss:.4f} | accuracy: {accuracy:.4f}")
        
        return loss, len(x_test), {"accuracy": accuracy}

# Start Flower client
fl.client.start_numpy_client(
        server_address="localhost:"+str(sys.argv[1]), 
        client=FlowerClient(client_id=1) , 
        grpc_max_message_length = 1024*1024*1024
)