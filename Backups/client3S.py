import flwr as fl
import tensorflow as tf
from tensorflow import keras
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from Labels import le, NUM_CLASSES, NUM_FEATURES
import shap

# ── Load dataset ──────────────────────────────────────────────────────────────
df = pd.read_csv(r'D:\CAPSTONE\NonIID_Client3.csv')   # ← change per client
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df = df.dropna()

X = df.drop(columns=['Label'])
y = df['Label'].str.strip().str.upper()

unseen = set(y) - set(le.classes_)
if unseen:
    raise ValueError(f"Unseen labels: {unseen}")
y = le.transform(y)

scaler = RobustScaler()
X = scaler.fit_transform(X)

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(x_train)} | Test: {len(x_test)}")

# ── Class weights ─────────────────────────────────────────────────────────────
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))

# ── Model ─────────────────────────────────────────────────────────────────────
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu',
                       input_shape=(NUM_FEATURES,)),
    keras.layers.LayerNormalization(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.LayerNormalization(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(NUM_CLASSES, activation='softmax')
])
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ── Fixed background for SHAP (computed once, reused every round) ─────────────
# 200 rows sampled once from training set — never changes, never sent to server
SHAP_BACKGROUND = x_train[
    np.random.choice(len(x_train), 200, replace=False)
]


def compute_shap_vector(current_model, background, explain_data):
    """
    Returns a normalised 40-value SHAP importance vector.
    Mean absolute SHAP across all classes and all explain rows.
    """
    explainer   = shap.DeepExplainer(current_model, background)
    shap_values = explainer.shap_values(explain_data)
    # shap_values: list of (n_samples, n_features) — one per class
    mean_shap = np.mean(
        [np.abs(sv) for sv in shap_values], axis=(0, 1)
    )
    # Normalise to unit length
    norm = np.linalg.norm(mean_shap)
    if norm > 0:
        mean_shap = mean_shap / norm
    return mean_shap.astype(np.float32)


# ── Flower client ──────────────────────────────────────────────────────────────
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.client_id = client_id

    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        r = model.fit(
            x_train, y_train,
            epochs=1,
            validation_data=(x_test, y_test),
            class_weight=class_weight_dict,
            verbose=0
        )
        val_acc  = float(np.mean(r.history['val_accuracy']))
        val_loss = float(np.mean(r.history['val_loss']))
        print(f"[Client {self.client_id}] LOCAL → "
              f"loss: {val_loss:.4f} | acc: {val_acc:.4f}")

        # ── Compute SHAP vector ───────────────────────────────────────────
        # Use 200 explain rows from test set
        explain_rows = x_test[:200]
        shap_vec     = compute_shap_vector(
            model, SHAP_BACKGROUND, explain_rows
        )

        # Pack SHAP as metrics dict — shap_f0 ... shap_f39
        shap_metrics = {
            f"shap_f{j}": float(shap_vec[j])
            for j in range(len(shap_vec))
        }
        shap_metrics["val_accuracy"] = val_acc
        shap_metrics["val_loss"]     = val_loss

        print(f"[Client {self.client_id}] SHAP computed. "
              f"Top feature: f{np.argmax(shap_vec)} "
              f"= {float(shap_vec[np.argmax(shap_vec)]):.4f}")

        return model.get_weights(), len(x_train), shap_metrics

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        print(f"[Client {self.client_id}] GLOBAL → "
              f"loss: {loss:.4f} | acc: {accuracy:.4f}")

        metrics = {"accuracy": float(accuracy)}

        # ── Full classification report on final round ─────────────────────
        if config.get("run_report", False):
            y_pred = np.argmax(
                model.predict(x_test, verbose=0), axis=1
            )
            report = classification_report(
                y_test, y_pred,
                target_names=le.classes_,
                output_dict=True,
                zero_division=0
            )
            df_report = pd.DataFrame(report).T
            df_report.to_csv(
                f"client{self.client_id}_final_report.csv"
            )
            # Print worst 5 classes
            class_f1 = {
                k: v['f1-score']
                for k, v in report.items()
                if k in le.classes_
            }
            worst5 = sorted(
                class_f1.items(), key=lambda x: x[1]
            )[:5]
            print(f"[Client {self.client_id}] "
                  f"Worst 5 F1: {worst5}")

        return loss, len(x_test), metrics


fl.client.start_numpy_client(
    server_address="localhost:" + str(sys.argv[1]),
    client=FlowerClient(client_id=3),   # ← change per client
    grpc_max_message_length=1024 * 1024 * 1024
)